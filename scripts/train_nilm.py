"""Train one Seq2Point arm on frozen NILM inputs (Phase B1)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.nilm.trainer import train_seq2point  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment-dir", required=True,
                    help="directory produced by prepare_nilm_b0_b2_inputs.py")
    ap.add_argument("--arm", required=True, choices=("B0", "B1", "B2"))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--steps-per-epoch", type=int, default=200)
    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--validation-count", type=int, default=20000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"),
                    default="auto",
                    help="auto uses CUDA when available; formal server jobs "
                         "should pass --device cuda")
    args = ap.parse_args()
    summary = train_seq2point(
        Path(args.experiment_dir), Path(args.output_dir), arm=args.arm,
        seed=args.seed, batch_size=args.batch_size,
        steps_per_epoch=args.steps_per_epoch, max_epochs=args.max_epochs,
        patience=args.patience, learning_rate=args.learning_rate,
        validation_count=args.validation_count, resume=args.resume,
        device_name=args.device)
    print(f"[train] best epoch={summary['best_epoch']} "
          f"val_mae={summary['best_val_mae_w']:.3f}W "
          f"params={summary['parameter_count']:,}")


if __name__ == "__main__":
    main()
