//! JSON command-line boundary for local Laya inference.

use std::{env, io::Write, path::PathBuf};

use anyhow::{Context, Result, ensure};
use burn::tensor::Device;
use laya::{Agent, Request};

fn main() -> Result<()> {
    let arguments: Vec<_> = env::args_os().skip(1).collect();
    ensure!(
        (2..=3).contains(&arguments.len()),
        "usage: laya MODEL_DIRECTORY REQUEST_JSON [cpu|vulkan-integrated|vulkan-discrete]"
    );
    let directory = PathBuf::from(&arguments[0]);
    let request: Request =
        serde_json::from_slice(&std::fs::read(&arguments[1]).context("cannot read request")?)
            .context("invalid request")?;
    let backend = arguments
        .get(2)
        .map(|value| value.to_str().context("backend must be UTF-8"))
        .transpose()?
        .unwrap_or("cpu");
    let device = match backend {
        "cpu" => Device::flex(),
        #[cfg(feature = "vulkan")]
        "vulkan-integrated" => Device::vulkan(burn::tensor::DeviceKind::IntegratedGpu(0)),
        #[cfg(feature = "vulkan")]
        "vulkan-discrete" => Device::vulkan(burn::tensor::DeviceKind::DiscreteGpu(0)),
        backend => {
            anyhow::bail!("backend {backend:?} is not available in this build")
        }
    };
    let agent = Agent::load(&directory, &device)?;
    writeln!(
        std::io::stdout().lock(),
        "{}",
        serde_json::to_string_pretty(&agent.predict(&request)?)?
    )?;
    Ok(())
}
