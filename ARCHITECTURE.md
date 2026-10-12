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

- src/input.rs defines requests, token assembly, marker positions, batching, and input diagnostics.
- src/model.rs includes generated Burn code and checks raw batch shapes and outputs.
- src/output.rs reads decision configuration and decodes option and action logits.
- src/agent.rs loads complete local artifacts and connects preprocessing, inference, and decoding.
- src/main.rs provides the JSON command-line boundary.
- build.rs converts the complete ONNX graph into Burn operations and separate Burnpack weights.
- The laya submodule remains the unchanged Python reference.

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
- Its expected weight digest comes from laya/verify/checkpoints.json.
- tests/reference.py checks the checkpoint digest and invokes the upstream full-model exporter when requested.
- The harness compares exported outputs against eager FP32 PyTorch before writing fixtures.
- Reference fixture metadata records package versions and checkpoint provenance. Fixture outputs are precomputed from real weights.
- Cargo requires models/laya/laya.onnx and its external tensor data before code generation.
- Generated source and weights remain under OUT_DIR. Models and Python environments remain ignored.
- The deployment directory contains laya.bpk, rl_agent_config.json, and tokenizer files.
- Copy the generated Burnpack artifact into the deployment directory after conversion.
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
- Generated code has a separate lint boundary. Handwritten code retains normal lint checks.

## Deferred scope

- Other checkpoints, automatic language routing, training, quantization, and long-document scanning require separate validation.
- This crate is a Burn implementation, not an upstream official Rust SDK.