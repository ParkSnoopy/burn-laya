use std::path::Path;

use anyhow::{
    Context,
    Result,
    ensure,
};
use indexmap::IndexMap;
use serde::{
    Deserialize,
    Serialize,
};
use tokenizers::Tokenizer;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub state: String,
    pub questions: IndexMap<String, Question>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase", deny_unknown_fields)]
pub enum Question {
    Choice {
        instructions: String,
        criteria: IndexMap<String, Option<String>>,
    },
    Score {
        instructions: String,
        criteria: Vec<String>,
    },
    Noul {
        instructions: String,
        #[serde(default)]
        criteria: IndexMap<String, String>,
    },
}

impl Question {
    pub(crate) fn kind(&self) -> (&'static str, i64) {
        match self {
            Self::Choice { .. } => ("choice", 0),
            Self::Score { .. } => ("score", 1),
            Self::Noul { .. } => ("noul", 2),
        }
    }

    fn instructions(&self) -> &str {
        match self {
            Self::Choice { instructions, .. }
            | Self::Score { instructions, .. }
            | Self::Noul { instructions, .. } => instructions,
        }
    }

    pub(crate) fn options(&self) -> Result<Vec<(String, String)>> {
        let options: Vec<_> = match self {
            Self::Choice { criteria, .. } => {
                criteria
                    .iter()
                    .map(|(label, description)| {
                        let text = match description {
                            Some(description) if !description.is_empty() => {
                                format!("{label}: {description}")
                            }
                            _ => label.clone(),
                        };
                        (label.clone(), text)
                    })
                    .collect()
            }
            Self::Score { criteria, .. } => {
                criteria
                    .iter()
                    .enumerate()
                    .map(|(index, text)| (index.to_string(), format!("level {index}: {text}")))
                    .collect()
            }
            Self::Noul { criteria, .. } => {
                ensure!(
                    criteria.keys().all(|key| key == "false" || key == "true"),
                    "noul criteria must use false or true keys"
                );
                [
                    ("false", "no, the statement does not hold"),
                    ("true", "yes, the statement holds"),
                ]
                .into_iter()
                .map(|(label, default)| {
                    let description = criteria
                        .get(label)
                        .filter(|value| !value.is_empty())
                        .map_or(default, String::as_str);
                    (label.to_owned(), format!("{label}: {description}"))
                })
                .collect()
            }
        };
        ensure!(!options.is_empty(), "question needs at least one option");
        ensure!(
            options.len() <= 256,
            "question exceeds the supported 256 options"
        );
        Ok(options)
    }
}

#[derive(Debug, Deserialize, Serialize, PartialEq)]
pub struct OptionStats {
    pub options: usize,
    pub options_distinct: usize,
    pub tokens_per_option: Option<usize>,
}

#[derive(Debug, Deserialize, Serialize, PartialEq)]
pub struct StateStats {
    pub state_tokens: usize,
    pub state_tokens_used: usize,
    pub state_tokens_dropped: usize,
    pub truncated: bool,
}

#[derive(Debug, Deserialize, Serialize, PartialEq)]
pub struct Row {
    pub ids: Vec<i64>,
    pub markers: Vec<i64>,
    pub qtype: i64,
    pub options: OptionStats,
    pub state_stats: StateStats,
}

#[derive(Debug, Deserialize, Serialize, PartialEq)]
pub struct Batch {
    pub input_ids: Vec<Vec<i64>>,
    pub attention_mask: Vec<Vec<i64>>,
    pub marker_pos: Vec<Vec<i64>>,
    pub marker_mask: Vec<Vec<bool>>,
    pub qtype: Vec<i64>,
}

pub struct InputBuilder {
    tokenizer: Tokenizer,
    cls: i64,
    sep: i64,
    pad: i64,
    mask: i64,
    mask_token: String,
    max_len: usize,
    head_max_len: usize,
}

#[derive(Deserialize)]
struct TokenConfig {
    cls_token: String,
    sep_token: String,
    pad_token: String,
    mask_token: String,
}

impl InputBuilder {
    pub fn load(directory: &Path, max_len: usize, head_max_len: usize) -> Result<Self> {
        ensure!(
            (16..=512).contains(&max_len),
            "max_len must be between 16 and 512 for this graph"
        );
        ensure!(
            (16..=max_len).contains(&head_max_len),
            "head_max_len must be between 16 and max_len"
        );
        let mut tokenizer = Tokenizer::from_file(directory.join("tokenizer/tokenizer.json"))
            .map_err(anyhow::Error::msg)?;
        tokenizer
            .with_truncation(None)
            .map_err(anyhow::Error::msg)?;
        tokenizer.with_padding(None);
        let special: TokenConfig = serde_json::from_slice(&std::fs::read(
            directory.join("tokenizer/tokenizer_config.json"),
        )?)?;
        let token_id = |text: &str| -> Result<i64> {
            let id = tokenizer
                .token_to_id(text)
                .with_context(|| format!("tokenizer does not contain {text}"))?;
            ensure!(id < 50368, "special token exceeds graph vocabulary");
            Ok(i64::from(id))
        };
        Ok(Self {
            cls: token_id(&special.cls_token)?,
            sep: token_id(&special.sep_token)?,
            pad: token_id(&special.pad_token)?,
            mask: token_id(&special.mask_token)?,
            mask_token: special.mask_token,
            tokenizer,
            max_len,
            head_max_len,
        })
    }

    fn tokens(&self, text: &str) -> Result<Vec<i64>> {
        let sanitized = text.replace(&self.mask_token, " ");
        let encoding = self
            .tokenizer
            .encode(sanitized, false)
            .map_err(anyhow::Error::msg)?;
        ensure!(
            encoding.get_ids().iter().all(|id| *id < 50368),
            "token exceeds graph vocabulary"
        );
        Ok(encoding.get_ids().iter().map(|id| i64::from(*id)).collect())
    }

    pub fn encode(&self, request: &Request) -> Result<(Vec<Row>, Batch)> {
        ensure!(
            !request.questions.is_empty(),
            "request needs at least one question"
        );
        ensure!(
            request.questions.len() <= 64,
            "request exceeds the supported 64 question rows"
        );
        let state = self.tokens(&request.state)?;
        let mut rows = Vec::with_capacity(request.questions.len());
        for (id, question) in &request.questions {
            ensure!(!id.trim().is_empty(), "question id must not be empty");
            ensure!(
                !question.instructions().trim().is_empty(),
                "question {id}: instructions must not be empty"
            );
            let options = question
                .options()
                .with_context(|| format!("question {id}"))?;
            let (kind, qtype) = question.kind();
            let mut head = self.tokens(&format!("{kind} question: {}", question.instructions()))?;
            let mut spans = Vec::with_capacity(options.len());
            for (_, text) in &options {
                let mut tokens = self.tokens(&format!(" {text}"))?;
                tokens.truncate(48);
                let mut span = vec![self.mask];
                span.extend(tokens);
                spans.push(span);
            }
            let mut budget =
                self.head_max_len as i64 - spans.iter().map(Vec::len).sum::<usize>() as i64;
            let mut tokens_per_option = None;
            if budget < 16 {
                let per = 4.max((self.head_max_len - 16) / spans.len());
                for span in &mut spans {
                    span.truncate(per);
                }
                tokens_per_option = Some(per);
                budget =
                    self.head_max_len as i64 - spans.iter().map(Vec::len).sum::<usize>() as i64;
            }
            head.truncate(budget.max(8) as usize);
            let distinct = spans.iter().collect::<std::collections::HashSet<_>>().len();
            let mut ids = vec![self.cls];
            ids.extend(head);
            ids.push(self.sep);
            let mut markers = Vec::with_capacity(spans.len());
            for span in spans {
                markers.push(ids.len() as i64);
                ids.extend(span);
            }
            ids.push(self.sep);
            let used = state.len().min(self.max_len.saturating_sub(ids.len() + 1));
            ids.extend_from_slice(&state[..used]);
            ids.push(self.sep);
            ids.truncate(self.max_len);
            ensure!(
                markers
                    .iter()
                    .all(|position| *position < self.max_len as i64),
                "question {id}: option markers exceed max_len"
            );
            rows.push(Row {
                ids,
                markers,
                qtype,
                options: OptionStats {
                    options: options.len(),
                    options_distinct: distinct,
                    tokens_per_option,
                },
                state_stats: StateStats {
                    state_tokens: state.len(),
                    state_tokens_used: used,
                    state_tokens_dropped: state.len() - used,
                    truncated: used < state.len(),
                },
            });
        }
        let length = rows
            .iter()
            .map(|row| row.ids.len())
            .max()
            .context("missing rows")?;
        let width = rows
            .iter()
            .map(|row| row.markers.len())
            .max()
            .context("missing markers")?
            .max(2);
        let mut batch = Batch {
            input_ids: vec![vec![self.pad; length]; rows.len()],
            attention_mask: vec![vec![0; length]; rows.len()],
            marker_pos: vec![vec![0; width]; rows.len()],
            marker_mask: vec![vec![false; width]; rows.len()],
            qtype: rows.iter().map(|row| row.qtype).collect(),
        };
        for (index, row) in rows.iter().enumerate() {
            batch.input_ids[index][..row.ids.len()].copy_from_slice(&row.ids);
            batch.attention_mask[index][..row.ids.len()].fill(1);
            batch.marker_pos[index][..row.markers.len()].copy_from_slice(&row.markers);
            batch.marker_mask[index][..row.markers.len()].fill(true);
        }
        Ok((rows, batch))
    }
}
