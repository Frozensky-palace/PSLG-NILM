"""Run Seq2Point inference and save predictions as an NPZ (Phase B1).

The output NPZ carries y_true / y_pred plus the window indices and is directly
readable by scripts/evaluate_nilm_predictions.py. Validation is allowed freely;
the test partition stays locked behind an explicit freeze acknowledgement.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.nilm.checkpoint import load_checkpoint  # noqa: E402
from src.nilm.seq2point import Seq2PointCNN  # noqa: E402
from src.nilm.trainer import evaluate_validation_mae  # noqa: E402
from src.nilm.window_dataset import ShardedWindowDataset  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--arm", required=True,
                    help="arm label, e.g. B0/B1/B2 or a D/E route label; "
                         "'test' is forbidden")
    ap.add_argument("--partition", choices=("validation", "test"),
                    default="validation")
    ap.add_argument("--sample-source", choices=("monitor", "first_n"),
                    default="monitor",
                    help="monitor = the frozen validation_monitor_indices "
                         "used by the trainer, so numbers are directly "
                         "comparable; first_n = the first N windows")
    ap.add_argument("--limit", type=int, default=None,
                    help="predict only the first N windows of the selected "
                         "sample set")
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"),
                    default="auto")
    ap.add_argument("--i-confirm-test-protocol-frozen", action="store_true",
                    help="required unlock flag when --partition test")
    args = ap.parse_args()

    if args.partition == "test" and not args.i_confirm_test_protocol_frozen:
        raise SystemExit(
            "test partition is locked: pass --i-confirm-test-protocol-frozen "
            "only after protocol_freeze_before_test.md is signed")
    if args.partition == "test":
        access_log = Path(args.output).parent / "test_access_log.json"
        access_log.parent.mkdir(parents=True, exist_ok=True)
        entry = {"checkpoint": str(args.checkpoint), "arm": args.arm,
                 "output": str(args.output)}
        access_log.write_text(json.dumps(entry, indent=2), encoding="utf-8")

    dataset = ShardedWindowDataset(
        Path(args.experiment_dir), arm=args.arm, partition=args.partition)
    app_norm = dataset.normalization["appliance_w"]
    if args.sample_source == "monitor":
        if args.partition != "validation":
            raise SystemExit("--sample-source monitor requires --partition "
                             "validation")
        monitor_path = Path(args.experiment_dir) / "validation_monitor_indices.npy"
        if not monitor_path.exists():
            raise SystemExit(f"missing monitor index file: {monitor_path}")
        indices = np.load(monitor_path).astype(np.int64)
        print(f"[predict] using frozen validation monitor indices "
              f"({len(indices):,} windows, identical to training selection)")
    else:
        indices = np.arange(len(dataset), dtype=np.int64)
    count = len(indices) if args.limit is None else min(args.limit, len(indices))
    indices = indices[:count]
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")
    model = Seq2PointCNN(dataset.window_length).to(device)
    load_checkpoint(Path(args.checkpoint), model=model, map_location=device)
    _, predictions, targets = evaluate_validation_mae(
        dataset, indices, model, app_norm)
    y_true = targets * app_norm["std"] + app_norm["mean"]
    y_pred = np.maximum(
        predictions * app_norm["std"] + app_norm["mean"], 0.0)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        y_true=y_true.astype(np.float32),
        y_pred=y_pred.astype(np.float32),
        indices=indices,
        arm=np.asarray(args.arm),
        partition=np.asarray(args.partition),
        checkpoint=np.asarray(str(args.checkpoint)),
    )
    print(f"[predict] wrote {count:,} windows -> {output}")
    dataset.close()


if __name__ == "__main__":
    main()
