//! Deploy and validate the weights generated for the selected checkpoint.

use std::{io::Write, path::Path};

use anyhow::Result;
use burn::tensor::Device;
use laya_rs::Agent;

fn main() -> Result<()> {
    let directory = Path::new(env!("LAYA_MODEL_DIR"));
    std::fs::copy(
        concat!(env!("OUT_DIR"), "/model/laya.bpk"),
        directory.join("laya.bpk"),
    )?;
    Agent::load(directory, &Device::flex())?;
    writeln!(
        std::io::stdout().lock(),
        "Prepared complete Burn weights in {}",
        directory.display()
    )?;
    Ok(())
}
