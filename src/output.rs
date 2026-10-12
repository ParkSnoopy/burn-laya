use anyhow::{
    Result,
    ensure,
};
use indexmap::IndexMap;
use serde::{
    Deserialize,
    Serialize,
};
use serde_json::Value;

use crate::input::{
    Question,
    Request,
};

#[derive(Debug, Deserialize)]
pub struct Config {
    pub encoder: String,
    pub head_layers: usize,
    pub max_len: usize,
    pub head_max_len: usize,
    pub act_costs: IndexMap<String, f64>,
    #[serde(default)]
    pub option_layout: Option<String>,
    #[serde(default)]
    pub temperature: [Value; 3],
    #[serde(default)]
    pub temperature_by_options: IndexMap<String, Value>,
}

impl Config {
    pub(crate) fn check(&self) -> Result<()> {
        ensure!(
            self.encoder == "answerdotai/ModernBERT-large",
            "this graph requires ModernBERT-large"
        );
        ensure!(
            self.head_layers == 2 && self.act_costs.len() == 1,
            "checkpoint head configuration does not match the graph"
        );
        ensure!(
            self.option_layout.as_deref().unwrap_or("sequential") == "sequential",
            "parallel option layout is not supported"
        );
        Ok(())
    }

    pub fn adjusted_temperatures(&self) -> bool {
        self.temperature
            .iter()
            .chain(self.temperature_by_options.values())
            .any(|value| value.as_f64().is_none_or(|raw| raw != temperature(value)))
    }

    fn scale(&self, question: &Question, count: usize) -> f32 {
        let (name, index) = question.kind();
        let size = match count {
            0..=2 => "2",
            3..=5 => "3-5",
            6..=10 => "6-10",
            _ => "11+",
        };
        temperature(
            self.temperature_by_options
                .get(&format!("{name}:{size}"))
                .unwrap_or(&self.temperature[index as usize]),
        ) as f32
    }
}

fn temperature(value: &Value) -> f64 {
    value
        .as_f64()
        .or_else(|| value.as_str()?.parse().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(1.0)
        .clamp(0.5, 5.0)
}

#[derive(Debug, Serialize)]
pub struct Answer {
    #[serde(flatten)]
    pub value: AnswerValue,
    pub confidence: f64,
    pub answer_confidence: f64,
    pub action: Action,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum AnswerValue {
    Choice {
        choice: String,
        probabilities: IndexMap<String, f64>,
    },
    Score {
        score: f64,
        legend: IndexMap<String, String>,
        probabilities: IndexMap<String, f64>,
    },
    Noul {
        noul: f64,
    },
}

#[derive(Debug, Serialize)]
pub struct Action {
    pub act_probability: f64,
}

fn rounded(value: f64) -> f64 {
    (value * 10_000.0).round_ties_even() / 10_000.0
}

fn softmax(logits: &[f32], scale: f32) -> Result<Vec<f32>> {
    ensure!(
        !logits.is_empty() && logits.iter().all(|value| value.is_finite()),
        "model outputs must be finite and nonempty"
    );
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) / scale;
    let mut probabilities: Vec<_> = logits
        .iter()
        .map(|value| (value / scale - maximum).exp())
        .collect();
    let sum: f32 = probabilities.iter().sum();
    for value in &mut probabilities {
        *value /= sum;
    }
    Ok(probabilities)
}

pub fn decode(
    config: &Config,
    request: &Request,
    logits: &[Vec<f32>],
    act_logits: &[Vec<f32>],
) -> Result<IndexMap<String, Answer>> {
    ensure!(
        logits.len() == request.questions.len() && act_logits.len() == logits.len(),
        "model output rows do not match questions"
    );
    let mut answers = IndexMap::new();
    for (row, (id, question)) in request.questions.iter().enumerate() {
        let options = question.options()?;
        let count = options.len();
        ensure!(
            logits[row].len() >= count && act_logits[row].len() == 2,
            "question {id}: model output shape mismatch"
        );
        let probabilities = softmax(&logits[row][..count], config.scale(question, count))?;
        let action = softmax(&act_logits[row], 1.0)?;
        let mut top = 0;
        for index in 1..count {
            if probabilities[index] > probabilities[top] {
                top = index;
            }
        }
        let confidence = if count < 2 {
            1.0
        } else {
            let entropy: f32 = probabilities
                .iter()
                .map(|p| -p * p.clamp(1e-12, 1.0).ln())
                .sum();
            (1.0 - f64::from(entropy) / (count as f64).ln()).clamp(0.0, 1.0)
        };
        let distribution = || {
            options
                .iter()
                .zip(&probabilities)
                .map(|((label, _), value)| (label.clone(), rounded(f64::from(*value))))
                .collect()
        };
        let value = match question {
            Question::Choice { .. } => {
                AnswerValue::Choice {
                    choice: options[top].0.clone(),
                    probabilities: distribution(),
                }
            }
            Question::Score { criteria, .. } => {
                AnswerValue::Score {
                    score: rounded(
                        probabilities
                            .iter()
                            .enumerate()
                            .map(|(index, value)| index as f64 * f64::from(*value))
                            .sum(),
                    ),
                    legend: criteria
                        .iter()
                        .enumerate()
                        .map(|(index, text)| (index.to_string(), text.clone()))
                        .collect(),
                    probabilities: distribution(),
                }
            }
            Question::Noul { .. } => {
                AnswerValue::Noul {
                    noul: rounded(f64::from(probabilities[1])),
                }
            }
        };
        answers.insert(
            id.clone(),
            Answer {
                value,
                confidence: rounded(if matches!(question, Question::Noul { .. }) {
                    f64::from(probabilities[1]).max(1.0 - f64::from(probabilities[1]))
                } else {
                    confidence
                }),
                answer_confidence: rounded(f64::from(probabilities[top]).clamp(0.0, 1.0)),
                action: Action {
                    act_probability: rounded(f64::from(action[0])),
                },
            },
        );
    }
    Ok(answers)
}
