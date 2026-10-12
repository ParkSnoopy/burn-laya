use std::path::Path;

use anyhow::{
    Context,
    Result,
    ensure,
};
use burn::tensor::Device;
use burn_store::{
    BurnpackStore,
    ModuleSnapshot,
};
use indexmap::IndexMap;
use serde::Serialize;

use crate::{
    input::{
        InputBuilder,
        OptionStats,
        Request,
    },
    model::{
        self,
        Model,
    },
    output::{
        Answer,
        Config,
        decode,
    },
};

pub struct Agent {
    config: Config,
    inputs: InputBuilder,
    model: Model,
    device: Device,
}

#[derive(Debug, Serialize)]
pub struct Prediction {
    pub answers: IndexMap<String, Answer>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub state_tokens: usize,
    pub state_tokens_dropped: usize,
    pub truncated: bool,
    pub truncated_questions: Vec<String>,
    #[serde(skip_serializing_if = "IndexMap::is_empty")]
    pub options: IndexMap<String, OptionStats>,
}

impl Agent {
    pub fn load(directory: &Path, device: &Device) -> Result<Self> {
        let config_path = directory.join("rl_agent_config.json");
        let config: Config = serde_json::from_slice(
            &std::fs::read(&config_path)
                .with_context(|| format!("cannot read {}", config_path.display()))?,
        )
        .with_context(|| format!("invalid configuration: {}", config_path.display()))?;
        config.check()?;
        let inputs = InputBuilder::load(directory, config.max_len, config.head_max_len)?;
        let weights = directory.join("laya.bpk");
        ensure!(
            weights.is_file(),
            "missing model weights: {}",
            weights.display()
        );
        let mut model = Model::new(device);
        let report = model
            .load_from(&mut BurnpackStore::from_file(&weights))
            .context("cannot load model weights")?;
        ensure!(
            report.is_success()
                && report.missing.is_empty()
                && report.unused.is_empty()
                && report.skipped.is_empty(),
            "incomplete or incompatible model weights: {report}"
        );
        if config.adjusted_temperatures() {
            eprintln!(
                "Checkpoint temperatures require adjustment to [0.5, 5.0]. Treat affected confidence values as uncalibrated."
            );
        }
        Ok(Self {
            config,
            inputs,
            model,
            device: device.clone(),
        })
    }

    pub fn predict(&self, request: &Request) -> Result<Prediction> {
        let (rows, batch) = self.inputs.encode(request)?;
        let (logits, actions) = model::infer(&self.model, &batch, &self.device)?;
        let answers = decode(&self.config, request, &logits, &actions)?;
        let dropped = rows
            .iter()
            .map(|row| row.state_stats.state_tokens_dropped)
            .max()
            .context("missing question rows")?;
        let usage = Usage {
            input_tokens: rows.iter().map(|row| row.ids.len()).sum(),
            output_tokens: 0,
            state_tokens: rows[0].state_stats.state_tokens,
            state_tokens_dropped: dropped,
            truncated: dropped > 0,
            truncated_questions: request
                .questions
                .keys()
                .zip(&rows)
                .filter(|(_, row)| row.state_stats.truncated)
                .map(|(id, _)| id.clone())
                .collect(),
            options: request
                .questions
                .keys()
                .zip(rows)
                .filter(|(_, row)| row.options.options_distinct < row.options.options)
                .map(|(id, row)| (id.clone(), row.options))
                .collect(),
        };
        Ok(Prediction { answers, usage })
    }
}
