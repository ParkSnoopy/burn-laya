# Laya on Burn

## Runtime contract

- The initial runtime supports the English ModernBERT-large checkpoint with two decision-head layers and two action outputs.
- A request contains a text state and an ordered map of questions.
- Question variants are choice, score, and noul. Criteria descriptions and score levels are strings.
- Choice criteria are an ordered map of labels to descriptions. A missing description uses the label alone.
- Noul reports the probability of true. Its option order remains false, then true.
- Unsupported fields, structured states, parallel layouts, and incompatible configurations fail explicitly.
- Limits are 512 tokens per sequence, 64 question rows, and 256 marker slots.
- Input diagnostics report state truncation and option descriptions that collapse after token-budget cuts.
- Confidence is normalized entropy for choice and score. Answer confidence is the largest temperature-scaled option probability.
- Shipped temperature values do not establish calibration on a user's data.

## Modules

- [src/inference/input.rs](../src/inference/input.rs) defines requests, token assembly, marker positions, batching, and input diagnostics.
- [src/inference/model.rs](../src/inference/model.rs) includes generated Burn code and checks raw batch shapes and outputs.
- [src/inference/output.rs](../src/inference/output.rs) reads decision configuration and decodes option and action logits.
- [src/inference/agent.rs](../src/inference/agent.rs) loads complete local artifacts and connects preprocessing, inference, and decoding.
- [src/inference/main.rs](../src/inference/main.rs) provides the JSON command-line boundary.
- [src/lib.rs](../src/lib.rs) preserves the crate's original public imports through re-exports.
- build.rs converts the complete ONNX graph into Burn operations and separate Burnpack weights.
- The laya submodule remains a passive parity reference. No build, runtime, preparation, or validation tool may load its files or depend on it.

## Core data flow

1. The command-line boundary parses a typed Request and selects an explicit Burn device.
2. Agent loads configuration, tokenizer data, and complete Burnpack weights from the selected directory.
3. InputBuilder renders each question and assembles one padded row per question.
4. The generated model computes option logits and action logits through Burn operations.
5. The decoder applies checkpoint temperatures and returns typed answers with input diagnostics.

## Devices

- Flex is the default CPU backend. The library accepts an explicit Burn Device.
- The optional vulkan Cargo feature enables WGPU execution through the Vulkan API.
- The CLI selects CPU, integrated Vulkan GPU zero, or discrete Vulkan GPU zero explicitly.
- GPU selection never substitutes a CPU adapter when the selected GPU is absent.
- tests/gpu.rs checks the integrated GPU against the same real checkpoint fixtures and raw-output tolerances.
- GPU outputs return to host memory before comparisons or timing complete.

## Artifact lifecycle

- The checkpoint comes from revision 1c5edc17a7acd8701df6fc341c0d179f1c62c982 of convaiinnovations/laya.
- tests/fixtures/checkpoint.json owns the checkpoint revision, expected weight digest, and published reference version.
- [tools/reference.py](../tools/reference.py) checks the checkpoint digest and exports the complete model through PyTorch when requested.
- Preparation and Python parity checks use published laya==0.4.4, never an editable installation from the submodule.
- The reference harness rejects imports that resolve inside the submodule before loading Python model code.
- The harness compares exported outputs against eager FP32 PyTorch before writing fixtures.
- Reference fixture metadata records package versions and checkpoint provenance. Fixture outputs are precomputed from real weights.
- Cargo requires laya.onnx and its external tensor data from the directory selected by LAYA_MODEL_DIR, defaulting to models/laya.
- Generated source and weights remain under OUT_DIR. Models and Python environments remain ignored.
- The deployment directory contains laya.bpk, rl_agent_config.json, and tokenizer files.
- Run cargo run --example prepare to deploy and validate the generated Burnpack in the selected model directory.
- The deployed runtime does not need Python, ONNX Runtime, ONNX files, the submodule, or network access.
- Every weight load must report no missing, unused, skipped, or failed tensor mappings.

## Verification boundaries

- tests/model_parity.rs compares both output heads against real PyTorch fixtures.
- Raw FP32 comparisons use absolute and relative tolerances of 1e-4.
- tests/input_parity.rs requires exact equality of token IDs, markers, masks, and diagnostics.
- tests/output_parity.rs requires exact decoded answers when supplied the same reference logits.
- tests/inference.rs checks complete predictions against the reference cases.
- End-to-end numerical fields permit one unit in the fourth decimal place. FP32 drift can cross a rounding boundary.
- Labels, question types, legends, and output structure must match exactly.
- Required artifact-dependent tests fail when artifacts are absent. They never substitute initialized weights or skip silently.
- The optimized test profile makes Flex tensor kernels practical without release-mode test commands.
- Generated code has a separate lint boundary with reasoned, checked expectations for generator-owned patterns.
- Handwritten production code follows the maximum lint preset without broad exceptions.
- Integration targets expect fixture unwraps and captured measurement output; invalid required artifacts still fail tests immediately.
- Vulkan enables the existing branches/std feature because CubeCL disables its defaults and the no-std path uses an obsolete nightly abort intrinsic.

## Reference workflow

- Install the published reference into an isolated environment with uv pip install --python .venv/bin/python 'laya[onnx]==0.4.4'.
- Run tools/reference.py --export to prepare the ONNX graph and regenerate real checkpoint fixtures.
- Run tools/reference.py without --export to compare the existing graph against the published reference and regenerate fixtures.
- Run tools/reference.py --binary target/release/laya-rs to compare the CPU and integrated Vulkan CLI against frozen answers without importing the Python model.
- Pass --model-directory for a trained checkpoint. Its checkpoint.json and reference.json stay beside its weights, separate from the base fixtures.
- Cargo parity tests consume frozen fixtures and local model artifacts. They do not invoke the Python reference or require an initialized submodule.

## Training workflow

[tools/train.py](../tools/train.py) delegates model training, AdamW updates, calibration, and evaluation to published laya==0.4.4.
Burn's ONNX importer emits an inference graph and ignores training dropout; native source-equivalent training requires a separate trainable model port.
No src/train directory is created for the Python fallback.
The trainer accepts only local English checkpoints with the supported architecture, sequential layout, and token budgets.
The output must be a new directory separate from the base; the base weight digest is checked before and after training.
Successful training writes checkpoint.json with the actual new weight digest and base digest, then exports and validates the new ONNX graph.
An export failure leaves the trained checkpoint available but does not report preparation complete.

Run these commands from the repository root with the existing isolated reference environment and prepared base checkpoint.
Provide real labeled train.jsonl and a separate eval.jsonl; JSONL records use state, questions, and expected or gold fields.

```bash
.venv/bin/python tools/train.py --data train.jsonl --dry-run
.venv/bin/python tools/train.py \
  --data train.jsonl --eval eval.jsonl --out models/laya-finetuned \
  --freeze-encoder --loss soft-ce --micro-batch 1 --grad-accum 8 --device cpu
LAYA_MODEL_DIR=models/laya-finetuned cargo run --example prepare
LAYA_MODEL_DIR=models/laya-finetuned cargo test --all-features
LAYA_MODEL_DIR=models/laya-finetuned cargo build --release --features vulkan
.venv/bin/python tools/reference.py \
  --model-directory models/laya-finetuned --binary target/release/laya-rs
```

Remove --freeze-encoder for a full fine-tune; --loss rlcd selects the upstream default objective.
Select cuda or mps only with a compatible PyTorch build and device; Rust Vulkan availability does not establish PyTorch training support.
The inherited token budgets stay within the current Rust runtime's 512-token limit.
The shared parity tests select the same checkpoint directory as code generation; they never compare trained weights against the base model's answers.
The [synthetic training](../tests/fixtures/train-smoke.jsonl) and [evaluation](../tests/fixtures/eval-smoke.jsonl) fixtures exercise training and artifact conversion only.
A tiny smoke dataset is not evidence of useful task accuracy or fitted calibration.

## Deferred scope

- Other architectures, native Burn training, automatic language routing, quantization, and long-document scanning require separate validation.
- This crate is a Burn implementation, not an upstream official Rust SDK.