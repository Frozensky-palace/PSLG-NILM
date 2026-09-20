"""Evaluate aligned NILM predictions saved as an NPZ or two-column CSV."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.nilm.metrics import nilm_metrics  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--threshold-w", type=float, default=20.0)
    args = ap.parse_args()
    path = Path(args.predictions)
    if path.suffix.lower() == ".npz":
        with np.load(path) as data:
            true, pred = data["y_true"], data["y_pred"]
    else:
        frame = pd.read_csv(path)
        true, pred = frame["y_true"].to_numpy(), frame["y_pred"].to_numpy()
    result = nilm_metrics(true, pred, threshold_w=args.threshold_w)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
