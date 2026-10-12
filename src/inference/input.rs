//! Typed questions and checkpoint-compatible token assembly.

use std::path::Path;

use anyhow::{Context, Result, ensure};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
/// A text state and ordered questions to evaluate together.
pub struct Request {
    /// Text to evaluate.
    pub state: String,
    /// Questions keyed by caller-defined identifiers.
    pub questions: IndexMap<String, Question>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase", deny_unknown_fields)]
/// A supported typed decision question.
pub enum Question {
    /// Select one labeled option.
    Choice {
        /// Task instructions rendered into the question head.
        instructions: String,
        /// Ordered labels and optional descriptions.
        criteria: IndexMap<String, Option<String>>,
    },
    /// Report a probability-weighted level on an ordinal scale.
    Score {
        /// Task instructions rendered into the question head.
        instructions: String,
        /// Ordered level descriptions.
        criteria: Vec<String>,
    },
    /// Estimate the probability that a statement holds.
    Noul {
        /// Statement or task instructions.
        instructions: String,
        /// Optional descriptions for the false and true options.
        #[serde(default)]
        criteria: IndexMap<String, String>,
    },
}

impl Question {
    pub(crate) const fn kind(&self) -> (&'static str, i64) {
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

/// Option-budget diagnostics for one question.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
pub struct OptionStats {
    /// Requested option count.
    pub options: usize,
    /// Distinct tokenized options after budget cuts.
    pub options_distinct: usize,
    /// Applied per-option budget when descriptions needed truncation.
    pub tokens_per_option: Option<usize>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
/// State-token accounting for one encoded row.
pub struct StateStats {
    /// Original state-token count.
    pub state_tokens: usize,
    /// State tokens retained in the assembled sequence.
    pub state_tokens_used: usize,
    /// State tokens removed to satisfy the sequence budget.
    pub state_tokens_dropped: usize,
    /// Whether any state tokens were removed.
    pub truncated: bool,
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Eq)]
/// One encoded question row and its diagnostics.
pub struct Row {
    /// Complete sequence token identifiers.
    pub ids: Vec<i64>,
    /// Positions of option marker tokens.
    pub markers: Vec<i64>,
    /// Checkpoint question-type identifier.
    pub qtype: i64,
    /// Option-budget diagnostics.
    pub options: OptionStats,
    /// State-token diagnostics.
    pub state_stats: StateStats,
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Eq)]
/// Padded host-side inputs for a model forward pass.
pub struct Batch {
    /// Padded token rows.
    pub input_ids: Vec<Vec<i64>>,
    /// Binary validity masks for token rows.
    pub attention_mask: Vec<Vec<i64>>,
    /// Padded marker-position rows.
    pub marker_pos: Vec<Vec<i64>>,
    /// Validity masks for marker slots.
    pub marker_mask: Vec<Vec<bool>>,
    /// One checkpoint question-type identifier per row.
    pub qtype: Vec<i64>,
}

/// A local tokenizer and token budgets matching the loaded checkpoint.
#[derive(Debug)]
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
    #[serde(rename = "cls_token")]
    cls: String,
    #[serde(rename = "sep_token")]
    sep: String,
    #[serde(rename = "pad_token")]
    pad: String,
    #[serde(rename = "mask_token")]
    mask: String,
}

impl InputBuilder {
    /// Load tokenizer assets and validate sequence and question budgets.
    ///
    /// # Errors
    /// Returns an error for unsupported budgets, missing assets, or invalid special tokens.
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
            cls: token_id(&special.cls)?,
            sep: token_id(&special.sep)?,
            pad: token_id(&special.pad)?,
            mask: token_id(&special.mask)?,
            mask_token: special.mask,
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

    /// Encode every question and assemble a padded model batch.
    ///
    /// # Errors
    /// Returns an error for invalid questions or unsupported token and option bounds.
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
        let batch = self.collate(&rows)?;
        Ok((rows, batch))
    }

    fn collate(&self, rows: &[Row]) -> Result<Batch> {
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
        Ok(batch)
    }
}
