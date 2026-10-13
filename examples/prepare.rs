//! Convert the selected checkpoint and validate its complete Burn weights.

use std::{io::Write, path::Path};

use anyhow::Result;
use burn::tensor::Device;
use burn_laya::{Agent, inference::prepare};

fn main() -> Result<()> {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(std::env::var_os("LAYA_MODEL_DIR").unwrap_or_else(|| "models/laya".into()))
        .canonicalize()?;
    prepare::generate(&directory)?;
    Agent::load(&directory, &Device::flex())?;
    writeln!(
        std::io::stdout().lock(),
        "Prepared complete Burn weights in {}",
        directory.display()
    )?;
    Ok(())
}
