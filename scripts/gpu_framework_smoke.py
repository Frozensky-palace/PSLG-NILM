"""Minimal GPU forward/backward smoke for TensorFlow and PyTorch (guide §5.2).

Prints the detected device name, framework version and (for torch) memory
totals, runs one tiny forward/backward per framework, and reports PASS/FAIL
per framework as JSON. Exits non-zero if a required framework fails.

On machines without a GPU (for example this Windows CPU box) pass
``--require-gpu=false``: results are still reported honestly as
``device=cuda:False`` so the record cannot be mistaken for a GPU pass.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def smoke_tensorflow() -> dict:
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    with tf.device("/GPU:0" if gpus else "/CPU:0"):
        x = tf.Variable(tf.ones((8, 16)))
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(x * x)
        gradient = tape.gradient(loss, [x])
    if gradient[0] is None:
        raise RuntimeError("TensorFlow backward produced no gradient")
    return {"framework": "tensorflow", "version": tf.__version__,
            "gpu_detected": bool(gpus),
            "gpu_name": (gpus[0].name if gpus else "cpu"),
            "forward_backward": "PASS"}


def smoke_torch() -> dict:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(8, 16, device=device, requires_grad=True)
    (x.square().mean()).backward()
    if x.grad is None:
        raise RuntimeError("PyTorch backward produced no gradient")
    result = {"framework": "torch", "version": torch.__version__,
              "cuda_build": torch.version.cuda,
              "gpu_detected": torch.cuda.is_available(),
              "gpu_name": (torch.cuda.get_device_name(0)
                           if torch.cuda.is_available() else "cpu"),
              "forward_backward": "PASS"}
    if torch.cuda.is_available():
        result["memory_total_mb"] = int(
            torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))
    return result


def _str2bool(value: str) -> bool:
    if value.lower() in ("0", "false", "no"):
        return False
    if value.lower() in ("1", "true", "yes"):
        return True
    raise argparse.ArgumentTypeError(f"invalid boolean: {value}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--framework", choices=("tensorflow", "torch", "both"),
                    default="both")
    ap.add_argument("--require-gpu", default=True, type=_str2bool,
                    help="fail when a framework cannot see a GPU "
                         "(use --require-gpu false on CPU-only machines)")
    ap.add_argument("--output", default=None, help="optional JSON report path")
    args = ap.parse_args()

    smokes = {"tensorflow": smoke_tensorflow, "torch": smoke_torch}
    names = sorted(smokes if args.framework == "both" else [args.framework])
    results = []
    failed = False
    for name in names:
        try:
            result = smokes[name]()
            if args.require_gpu and not result["gpu_detected"]:
                raise RuntimeError(
                    f"{name} did not detect a GPU (require_gpu=true)")
            result["pass"] = True
        except Exception as error:  # noqa: BLE001 - report then fail
            failed = True
            result = {"framework": name, "pass": False, "error": str(error)}
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    report = {"require_gpu": bool(args.require_gpu),
              "all_passed": not failed, "results": results}
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                          encoding="utf-8")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
