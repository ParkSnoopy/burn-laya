"""Fine-tune with published Laya, then export and validate the new checkpoint."""

import json
from pathlib import Path
import sys

from reference import ROOT, digest, main as prepare_reference, require_reference


def main():
    provenance = json.loads((ROOT / "tests/fixtures/checkpoint.json").read_text())
    require_reference(provenance["reference_version"])
    from laya.train_cli import build_parser, main as train_reference

    parser = build_parser()
    parser.description += " The result is exported to ONNX and checked against PyTorch."
    parser.set_defaults(model_dir=str(ROOT / "models/laya"))
    args = parser.parse_args()
    base = Path(args.model_dir).resolve()
    if not (base / "model.safetensors").is_file():
        parser.error("--base must be a local checkpoint directory")
    trainer_args = [*sys.argv[1:], "--base", str(base)]
    if args.dry_run:
        return train_reference(trainer_args)
    if not args.output_dir:
        parser.error("--out is required unless --dry-run is set")
    output = Path(args.output_dir).resolve()
    if output.exists() or output.is_relative_to(base) or base.is_relative_to(output):
        parser.error("--out must be a new directory separate from the base checkpoint")
    config = json.loads((base / "rl_agent_config.json").read_text())
    if config["encoder"] != "answerdotai/ModernBERT-large" or config["head_layers"] != 2:
        parser.error("training export supports the English ModernBERT-large architecture")
    if config.get("option_layout", "sequential") != "sequential" or len(config["act_costs"]) != 1:
        parser.error("training export requires sequential options and two action outputs")
    max_len = args.max_len or config["max_len"]
    head_max_len = args.head_max_len or config["head_max_len"]
    if not 16 <= head_max_len <= max_len <= 512:
        parser.error("training token budgets must fit the Rust runtime's 512-token limit")
    original = digest(base / "model.safetensors")
    status = train_reference(trainer_args)
    if status:
        return status
    if digest(base / "model.safetensors") != original:
        raise ValueError("training modified the base checkpoint")
    manifest = {
        "revision": None,
        "sha256": digest(output / "model.safetensors"),
        "reference_version": provenance["reference_version"],
        "base_sha256": original,
    }
    (output / "checkpoint.json").write_text(json.dumps(manifest, indent=2) + "\n")
    prepare_reference(["--model-directory", str(output), "--export"])
    print(f"Training and ONNX parity complete: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())