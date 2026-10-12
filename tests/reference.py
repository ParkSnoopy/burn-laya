"""Prepare real checkpoint fixtures and check the upstream ONNX export."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import runpy
import subprocess
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/laya"
INPUTS = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def cases():
    request = json.loads((ROOT / "tests/fixtures/requests.json").read_text())
    yield "mixed", request
    for name, state in (
        ("unicode", "退款重复收费。 Bitte erstatten Sie die doppelte Zahlung. café 👩‍💻"),
        ("empty", ""),
        ("mask", "The literal [MASK] token must not create an option marker."),
        ("long", "We were charged twice and need a refund. " * 120),
    ):
        yield name, {**request, "state": state}
    for count in (1, 2, 5, 10, 11):
        yield f"choice-{count}", {
            "state": request["state"],
            "questions": {
                "department": {
                    "type": "choice",
                    "instructions": "Which department should handle this?",
                    "criteria": {f"team-{i}": f"department {i}" for i in range(count)},
                }
            },
        }
    yield "score-1", {
        "state": "Nothing is urgent.",
        "questions": {"urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["not urgent"]}},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--export", action="store_true")
    mode.add_argument("--binary", type=Path, help="Check and measure a CPU/Vulkan release binary on Linux")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "laya/verify/checkpoints.json").read_text())
    expected = next(entry for entry in manifest["checkpoints"] if entry["name"] == "english")["sha256"]
    if digest(MODEL / "model.safetensors") != expected:
        raise ValueError("checkpoint digest mismatch")
    graph = MODEL / "laya.onnx"
    if args.binary:
        fixture = json.loads((ROOT / "tests/fixtures/reference.json").read_text())
        if fixture["weight_sha256"] != expected or fixture["onnx_sha256"] != digest(graph):
            raise ValueError("reference fixture does not match the prepared model")
        if fixture["cases"][0]["request"] != json.loads((ROOT / "tests/fixtures/requests.json").read_text()):
            raise ValueError("reference fixture does not match the CLI request")
        expected = fixture["cases"][0]["answers"]
        for backend in ("cpu", "vulkan-integrated"):
            output = ROOT / "target" / f"{backend}-output.json"
            errors = ROOT / "target" / f"{backend}-stderr.log"
            command = [str(args.binary.resolve()), str(MODEL), str(ROOT / "tests/fixtures/requests.json"), backend]
            start = time.perf_counter()
            with output.open("w") as stdout, errors.open("w") as stderr:
                process = subprocess.Popen(command, stdout=stdout, stderr=stderr, cwd=ROOT.parent)
                _, status, usage = os.wait4(process.pid, 0)
                process.returncode = os.waitstatus_to_exitcode(status)
            if process.returncode:
                raise RuntimeError(f"{backend} exited with {process.returncode}: {errors.read_text()}")
            actual = json.loads(output.read_text())["answers"]
            def compare(actual, expected):
                if isinstance(expected, dict):
                    if actual.keys() != expected.keys():
                        raise AssertionError("answer keys differ")
                    for key in expected:
                        compare(actual[key], expected[key])
                elif isinstance(expected, (int, float)):
                    np.testing.assert_allclose(actual, expected, atol=1e-4 + 1e-12, rtol=0)
                elif actual != expected:
                    raise AssertionError(f"expected {expected!r}, got {actual!r}")
            compare(actual, expected)
            print(f"{backend}/release CLI: elapsed={time.perf_counter() - start:.6f}s peak_process_memory={usage.ru_maxrss}KiB. Answers match.", flush=True)
        return
    import onnxruntime as ort
    import torch
    from laya.agent import Agent
    from laya.common import collate_items

    if args.export:
        exporter = runpy.run_path(str(ROOT / "laya/scripts/export_onnx.py"))
        exporter["export_to_onnx"](str(MODEL), str(graph))
    agent = Agent(str(MODEL), device="cpu", compile=False)
    agent.model.eval().float()
    if agent.amp_enabled:
        raise ValueError("reference requires FP32 without autocast")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(graph), sess_options=options, providers=["CPUExecutionProvider"])
    fixture = {
        "revision": manifest["revision"],
        "weight_sha256": expected,
        "onnx_sha256": digest(graph),
        "packages": {name: importlib.metadata.version(name) for name in ("laya", "torch", "transformers", "tokenizers", "onnx", "onnxruntime")},
        "cases": [],
    }
    for name, request in cases():
        questions = request["questions"]
        for qid, definition in questions.items():
            agent._check_question(qid, definition)
        internal = {qid: agent._to_internal(definition) for qid, definition in questions.items()}
        start = time.perf_counter()
        items = agent._encode_state(request["state"], list(questions), internal)
        batch = collate_items([items], agent.tok.pad_token_id)
        # Match ONNXAgent: the traced TopK requires two slots even for a single option.
        if batch["marker_pos"].shape[1] < 2:
            batch["marker_pos"] = torch.nn.functional.pad(batch["marker_pos"], (0, 1))
            batch["marker_mask"] = torch.nn.functional.pad(batch["marker_mask"], (0, 1), value=False)
        encode_seconds = time.perf_counter() - start
        start = time.perf_counter()
        with torch.inference_mode():
            outputs = agent.model(*(batch[key] for key in INPUTS))
        inference_seconds = time.perf_counter() - start
        tensors = {key: batch[key].numpy() for key in INPUTS}
        actual = session.run(None, tensors)
        for label, reference, exported in zip(("logits", "act_logits"), outputs, actual):
            if not np.isfinite(reference.numpy()).all() or not np.isfinite(exported).all():
                raise ValueError(f"{name}/{label}: nonfinite model output")
            np.testing.assert_allclose(exported, reference.numpy(), atol=1e-4, rtol=1e-4, err_msg=f"{name}/{label}")
            delta = np.abs(exported - reference.numpy())
            print(f"{name}/{label}: max={delta.max():.8g} mean={delta.mean():.8g}", flush=True)
        start = time.perf_counter()
        answers = agent._decode_answers(outputs[0].numpy(), torch.softmax(outputs[1], -1).numpy(), items, list(questions), internal, 0)
        print(f"{name}/PyTorch timings: encode={encode_seconds:.6f}s forward={inference_seconds:.6f}s decode={time.perf_counter() - start:.6f}s", flush=True)
        fixture["cases"].append({
            "name": name,
            "request": request,
            "items": items,
            "inputs": {key: value.tolist() for key, value in tensors.items()},
            "logits": outputs[0].tolist(),
            "act_logits": outputs[1].tolist(),
            "answers": answers,
        })
    (ROOT / "tests/fixtures/reference.json").write_text(json.dumps(fixture, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    print(f"Checked {len(fixture['cases'])} real checkpoint cases.", flush=True)



if __name__ == "__main__":
    main()