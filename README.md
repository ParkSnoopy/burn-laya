# Laya on Burn

Typed local inference for the English Laya checkpoint, with a Python fine-tuning and ONNX export pipeline.
The Rust runtime does not require Python.

## Development contracts

- [Architecture and checkpoint lifecycle](docs/ARCHITECTURE.md) defines supported requests, devices, artifact preparation, and validation.
- [AI-agent guide](docs/LLM_WIKI.md) maps code ownership, invariants, and verification workflows.
- [Training workflow](tools/train.py) uses published `laya==0.4.6` and exports each trained checkpoint separately.
- [Reference workflow](tools/reference.py) validates ONNX output against the selected checkpoint's PyTorch model.
- [Burn preparation](examples/prepare.rs) deploys the complete weights generated for the selected model directory.
- [Python checks](docs/LLM_WIKI.md#python-linting) run Ruff and Flake8 with wemake-python-styleguide in isolated uvx environments.
- The `laya` submodule is a passive parity reference, never a build, training, runtime, or validation dependency.

## Licensing

This crate is licensed under [GPL-3.0-only](LICENSE).
Upstream Laya and its model artifacts retain their Apache-2.0 license and applicable notices.