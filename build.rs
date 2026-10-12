//! Convert the selected local ONNX checkpoint to Burn source and weights.

use std::io::Write;

use burn_onnx::{LoadStrategy, ModelGen};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut output = std::io::stdout().lock();
    writeln!(output, "cargo:rerun-if-env-changed=LAYA_MODEL_DIR")?;
    let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(std::env::var_os("LAYA_MODEL_DIR").unwrap_or_else(|| "models/laya".into()))
        .canonicalize()?;
    let graph = directory.join("laya.onnx");
    writeln!(
        output,
        "cargo:rustc-env=LAYA_MODEL_DIR={}",
        directory.display()
    )?;
    writeln!(output, "cargo:rerun-if-changed={}", graph.display())?;
    writeln!(
        output,
        "cargo:rerun-if-changed={}",
        directory.join("laya.onnx.data").display()
    )?;
    writeln!(output, "cargo:rerun-if-changed=build.rs")?;
    if !graph.is_file() {
        return Err(
            "Missing laya.onnx. Prepare the verified checkpoint with tools/reference.py --export."
                .into(),
        );
    }
    ModelGen::new()
        .input(graph.to_str().ok_or("Model path must be UTF-8")?)
        .out_dir("model")
        .load_strategy(LoadStrategy::File)
        .run_from_script();
    Ok(())
}
