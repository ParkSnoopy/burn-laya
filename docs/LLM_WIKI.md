# Laya: AI-agent guide

## Start here

This repository runs the English Laya decision model locally through Burn and delegates fine-tuning to the published Python package.
Read the [root index](../README.md), [architecture contract](ARCHITECTURE.md), and [Cargo manifest](../Cargo.toml) before changing code.
Check the worktree and preserve unrelated edits, including an already selected application version or submodule revision.
Only README belongs at the project root; maintained project Markdown belongs under docs/.

The [checkpoint manifest](../tests/fixtures/checkpoint.json) owns the base checkpoint revision, weight digest, and required Python reference version.
The current published reference is laya==0.4.6; a package release is not a new model checkpoint.
The laya/ submodule is passive: do not import, execute, modify, or use its files for build, training, preparation, or validation.
Inspect published package sources for compatibility work, never replace the isolated installation with an editable submodule installation.

## Ownership map

| Boundary | Owner | Responsibility |
| --- | --- | --- |
| Public library | [src/lib.rs](../src/lib.rs), [inference module](../src/inference/mod.rs) | Preserve public re-exports while keeping implementation under src/inference/. |
| JSON CLI | [main.rs](../src/inference/main.rs) | Parse a request file, select a device, emit a Prediction, and fail on invalid input or an unavailable backend. |
| Input encoding | [input.rs](../src/inference/input.rs) | Typed questions, tokenizer assets, rendering, token budgets, option markers, padded batches, and truncation diagnostics. |
| Tensor boundary | [model.rs](../src/inference/model.rs) | Include generated code, validate input tensors, execute both output heads, and reject nonfinite outputs. |
| Answer decoding | [output.rs](../src/inference/output.rs) | Checkpoint compatibility, temperatures, ordered option probabilities, typed answers, and action probability. |
| Runtime assembly | [agent.rs](../src/inference/agent.rs) | Load complete weights and tokenizer/configuration assets, then connect encoding, inference, and decoding. |
| Graph conversion | [build.rs](../build.rs), [prepare example](../examples/prepare.rs) | Generate matching source and Burnpack under OUT_DIR, then deploy and validate the selected Burnpack. |
| Python reference | [reference.py](../tools/reference.py) | Verify checkpoint identity, export ONNX, compare real FP32 outputs, and regenerate frozen fixtures. |
| Python training | [train.py](../tools/train.py) | Validate the local base/output boundary, delegate training, record provenance, and export the trained checkpoint. |

```text
local PyTorch checkpoint + tokenizer/configuration
          | tools/reference.py --export
          v
ONNX graph + external weights + frozen reference fixtures
          | build.rs / burn-onnx
          v
generated Burn source + Burnpack in OUT_DIR
          | cargo run --example prepare
          v
deployment directory: laya.bpk + configuration + tokenizer
          | Agent: encode -> forward -> decode
          v
typed answers + input diagnostics
```

The build-time graph and runtime weights must describe the same architecture.
Do not copy a Burnpack from an unrelated Cargo build directory or reuse generated source after replacing a graph without rebuilding.
Deployment needs laya.bpk, rl_agent_config.json, and tokenizer/; it does not need Python, ONNX Runtime, ONNX files, encoder/, network access, or the submodule.
Cargo conversion and Python training still need their respective source artifacts.

## Request and result contract

Requests have a string state and a nonempty ordered questions map.
Unknown request/question fields and non-string states, including null, are rejected rather than coerced.
Each question has a nonempty identifier, a type, and nonblank instructions.

| Type | Criteria | Answer |
| --- | --- | --- |
| choice | Nonempty ordered label-to-description map; null or empty descriptions use the label alone. | Selected label and probabilities keyed by label; equal probabilities select the first option. |
| score | Nonempty ordered list of level descriptions. | Probability-weighted zero-based level, legend, and probabilities keyed by level index. |
| noul | Optional descriptions keyed only by false and true. | Probability that the statement holds; internal order is always false, then true. |

This is a complete minimal request:

```json
{
  "state": "The invoice was charged twice.",
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which department should handle this?",
      "criteria": {"billing": "payments and refunds", "technical": "software failures"}
    }
  }
}
```

The [mixed request fixture](../tests/fixtures/requests.json) exercises all three question types.
CLI output contains answers keyed by the original question identifiers and a usage object with token/truncation diagnostics.
Each answer includes confidence, answer_confidence, and action.act_probability; the action-head estimate is separate from answer confidence.
Choice/score confidence uses normalized entropy, while noul confidence is the larger binary probability.
For a single option, both confidence values are 1.0 by definition, not evidence of correctness or useful calibration.
Numerical answer fields are rounded to four decimal places; usage.output_tokens is zero because this model does not generate text.

The graph supports ModernBERT-large, two decision-head layers, sequential options, and two action outputs.
Fixed runtime limits are 512 tokens per sequence, 64 question rows, 256 marker slots, and token IDs below 50368.
Checkpoint sequence and head budgets are validated before use; individual option descriptions are cut before assembling the bounded sequence.
State truncation is reported, and usage.options identifies questions whose option descriptions collapse to identical token spans.
Missing assets, incompatible configuration, incomplete tensor mappings, out-of-bound markers, and nonfinite outputs are errors.
Do not add structured states, parallel layouts, router/shortlist wrappers, other architectures, or native training by assuming upstream support implies Rust support.

## Prepare and run a base checkpoint

Run the following commands from the repository root using the explicit Rust binary path, not the published Python CLI also named laya.
Cargo uses the active PATH toolchain; do not add +stable or another override or install components into a different toolchain as a workaround.
For a fresh checkout, create the isolated Python environment; reuse an existing .venv instead of recreating it.
The following setup uses a POSIX shell and CPU-only PyTorch; use a compatible PyTorch build instead when training on CUDA or MPS.

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python 'laya[onnx]==0.4.6'
.venv/bin/hf download convaiinnovations/laya \
  model.safetensors rl_agent_config.json encoder/config.json \
  tokenizer/tokenizer.json tokenizer/tokenizer_config.json \
  --revision 1c5edc17a7acd8701df6fc341c0d179f1c62c982 --local-dir models/laya
.venv/bin/python tools/reference.py --export
cargo run --example prepare
cargo build --release
target/release/laya models/laya tests/fixtures/requests.json cpu
```

The download revision must agree with the checkpoint manifest; reference preparation rejects a weight digest mismatch before loading the model.
Export compares both ONNX heads against eager FP32 PyTorch and writes actual tokenizer inputs, logits, answers, package versions, and graph/weight digests into the fixture.
Never invent fixtures, update a digest merely to accept unexpected weights, or label old measurements as fresh evidence.
The fixture is precomputed; running the reference harness performs live verification and replaces it only after all cases succeed.
Models, .venv, Cargo output, and transient evidence remain ignored according to [.gitignore](../.gitignore).

For Vulkan, build with the optional feature and select the actual adapter kind:

```bash
cargo build --release --features vulkan
target/release/laya models/laya tests/fixtures/requests.json vulkan-integrated
```

The other GPU selector is vulkan-discrete; neither GPU selector falls back to CPU when its requested adapter is missing.
The maintained GPU parity test requires an integrated Vulkan GPU; its success does not establish discrete-GPU compatibility.
Rust Vulkan availability says nothing about PyTorch CUDA or MPS training support.

## Fine-tune and deploy

Training is Python-only in this implementation; there is no src/train/.
Burn's imported ONNX inference graph drops training-mode dropout, so adding autodiff alone does not reproduce the source trainer.
The wrapper uses published Laya's model, AdamW updates, loss, evaluation, and calibration rather than a partial optimizer/model port.
JSONL records contain state, questions, and expected or gold answers; choice labels must name existing criteria.
Use separate real training and evaluation files; the [training smoke fixture](../tests/fixtures/train-smoke.jsonl) and [evaluation smoke fixture](../tests/fixtures/eval-smoke.jsonl) are synthetic integration inputs, not task-quality evidence.

```bash
.venv/bin/python tools/train.py --data train.jsonl --dry-run
.venv/bin/python tools/train.py \
  --data train.jsonl --eval eval.jsonl --out models/laya-finetuned \
  --freeze-encoder --loss soft-ce --micro-batch 1 --grad-accum 8 --device cpu
LAYA_MODEL_DIR=models/laya-finetuned cargo run --example prepare
LAYA_MODEL_DIR=models/laya-finetuned cargo test
LAYA_MODEL_DIR=models/laya-finetuned cargo build --release --features vulkan
.venv/bin/python tools/reference.py \
  --model-directory models/laya-finetuned --binary target/release/laya
```

The final binary comparison needs an available integrated Vulkan GPU; on CPU-only systems, build without vulkan and use the CPU parity tests and CLI.
--base selects another local supported checkpoint; remote base IDs are rejected by the wrapper.
--out must not already exist or contain/be contained by the base directory; failed training/export may leave a partial output directory, so inspect it and choose a new output path for a retry.
The base weight digest is checked before and after training, and the new checkpoint records its own weight digest, reference version, and base digest.
Each trained checkpoint owns checkpoint.json and reference.json beside its weights; changing LAYA_MODEL_DIR selects its graph and its fixtures together.
Export failure preserves trained weights but is not preparation success; Rust deployment begins only after export/parity succeeds.
Remove --freeze-encoder for a full fine-tune, or select --loss rlcd for the upstream default objective.
Inherited or overridden budgets must still fit the Rust graph; unsupported architecture/layout/action shapes fail instead of silently widening support.

Laya 0.4.6 preserves the model and training CLI interfaces, but fixes calibration fitting by grouping rows by actual option count and bounding log-temperature search.
These fixes are inherited through the published trainer; do not duplicate its calibration optimizer in Rust.
Too few labeled examples leave neutral temperatures rather than demonstrating fitted calibration, and runtime clamping of shipped temperatures is not calibration on a user's data.
Previously trained checkpoints retain their recorded reference version; install that version when reproducing their Python reference, or explicitly regenerate under the selected new reference before claiming new parity.

## Change and verification workflow

### Python linting

Run these commands from the repository root; uvx keeps lint tools separate from the model reference environment.

```bash
uvx ruff check tools/
uvx ruff format --check tools/
uvx --with wemake-python-styleguide flake8 --select=WPS tools/
```

Ruff uses its defaults, and WPS is selected on the command line.
The checks target tools/ rather than the passive submodule, model artifacts, build outputs, or local environments.
The uvx --with option loads wemake-python-styleguide alongside Flake8.
Use uvx ruff check --fix tools/ and uvx ruff format tools/ for safe automated edits; do not disable rule families or inflate complexity limits merely to make a check pass.

### Implementation checks

1. Identify the owning boundary and its callers using the ownership map; preserve ordered question/option semantics and public imports.
2. Make the smallest shared change, including directly contradicted documentation and fixture coverage; do not relax production lints or numeric tolerances to hide incompatibility.
3. After Rust edits, run cargo fmt with [rustfmt.toml](../rustfmt.toml); preserve [Clippy thresholds](../clippy.toml) and the manifest's strict lint policy.
4. Regenerate reference fixtures with the selected published version when Python/model semantics change; keep real checkpoint identity and generated source/weights paired.
5. Exercise the affected integration seam, then the relevant wider gates below; missing required artifacts or devices are blockers, not silently skipped success.
6. Keep this guide and the architecture contract aligned with supported behavior, not temporary task logs.

```bash
cargo fmt --check
cargo clippy --all-targets --all-features
cargo test
cargo test --all-features
cargo doc --no-deps --all-features
cargo build --release --features vulkan
.venv/bin/python tools/reference.py --binary target/release/laya
```

Run hardware-dependent commands only where an integrated Vulkan GPU is available.
Test mode is deliberately optimized for Flex kernels; release builds are deployment artifacts, not a substitute for tests.
The release CLI comparison runs from outside the repository to exercise explicit artifact paths; it checks the mixed request, while Rust parity tests cover all frozen cases.
Reasoned lint expectations are confined to generated code and required test behavior; handwritten production code keeps the full configured policy.
Do not edit Cargo registry sources or generated graph source to resolve an upstream import/toolchain failure.
The optional branches/std feature selects the supported dependency implementation for nightly Vulkan builds; it is not a toolchain override.

| Verification owner | Required property |
| --- | --- |
| [input_parity.rs](../tests/input_parity.rs) | Exact token IDs, marker positions, masks, and diagnostics; reject invalid request shapes, including null state. |
| [model_parity.rs](../tests/model_parity.rs) | Finite outputs from both heads match real FP32 fixtures with absolute and relative tolerances of 1e-4. |
| [output_parity.rs](../tests/output_parity.rs) | Exact typed decoding for the same logits; ties, temperature handling, and invalid-output errors. |
| [inference.rs](../tests/inference.rs) | Complete CPU predictions preserve labels/structure and numerical fields within the rounding boundary. |
| [gpu.rs](../tests/gpu.rs) | Integrated Vulkan predictions match the same fixtures; synchronized host results separate encoding, forward, and decoding timings. |
| [shared fixture access/checks](../tests/common/mod.rs) | Select the checkpoint's own fixture and apply the declared tolerances without substituting base-model answers. |

For a reference-version update, compare the installed published sources, update the checkpoint manifest and maintained version examples, regenerate base fixtures, and repeat training/export/Rust parity on a separate output checkpoint.
Keep the base weight revision unchanged unless a checkpoint replacement is explicitly intended.
The [license](../LICENSE) covers this crate; upstream Laya and model artifacts retain their own Apache-2.0 terms and notices.