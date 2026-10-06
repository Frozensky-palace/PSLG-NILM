"""Compare freshly trained 6s/12s models on the same 12s target bins."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.nilm.metrics import nilm_metrics
from scripts.downsample_aligned_pairs import sha256, write_json


def compare(native_path, coarse_path, output_dir):
    with np.load(native_path, allow_pickle=False) as d:
        native = {k: d[k] for k in d.files}
    with np.load(coarse_path, allow_pickle=False) as d:
        coarse = {k: d[k] for k in d.files}
    for data, dt in ((native, 6), (coarse, 12)):
        timestamps = data["timestamp"]
        if (timestamps.ndim != 1 or not np.issubdtype(timestamps.dtype, np.integer)
                or np.any(timestamps % dt)):
            raise ValueError("prediction timestamps must be integer epoch-grid vectors")
        if data["sample_seconds"].item() != dt or np.any(np.diff(timestamps) <= 0):
            raise ValueError("wrong rate or unordered/duplicate prediction timestamps")
        if any(data[k].ndim != 1 or len(data[k]) != len(timestamps) for k in ("y_true", "y_pred")):
            raise ValueError("unaligned prediction arrays")
        if not np.isfinite(data["y_true"]).all() or not np.isfinite(data["y_pred"]).all():
            raise ValueError("nonfinite predictions")
    for key in ("arm", "partition", "seed"):
        if native[key].item() != coarse[key].item():
            raise ValueError(f"comparison must have the same {key}")
    if int(native["seed"]) < 0:
        raise ValueError("training seed is unknown; verify the checkpoint configuration first")
    ts = native["timestamp"]
    t = coarse["timestamp"]
    left, right = np.searchsorted(ts, t), np.searchsorted(ts, t + 6)
    valid = (left < len(ts)) & (right < len(ts))
    candidates = np.flatnonzero(valid)
    valid[candidates] &= ((ts[left[candidates]] == t[candidates])
                          & (ts[right[candidates]] == t[candidates] + 6))
    if not valid.any():
        raise ValueError("no complete common 12-second bins")
    left, right = left[valid], right[valid]
    truth = (native["y_true"][left].astype(float) + native["y_true"][right]) / 2
    native_pred = (native["y_pred"][left].astype(float) + native["y_pred"][right]) / 2
    coarse_truth, coarse_pred = coarse["y_true"][valid], coarse["y_pred"][valid]
    if not np.allclose(truth, coarse_truth, rtol=1e-5, atol=0.01):
        raise ValueError("targets disagree: datasets/bins are not a paired rate experiment")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "protocol": "common_12s_prediction_bins_v1", "common_bins": int(valid.sum()),
        "coarse_bins_without_native_pair": int((~valid).sum()),
        "arm": coarse["arm"].item(), "partition": coarse["partition"].item(),
        "seed": int(coarse["seed"]), "threshold_w": 20.,
        "native_prediction_sha256": sha256(native_path), "coarse_prediction_sha256": sha256(coarse_path),
        "native_6s_model_on_common_12s": nilm_metrics(coarse_truth, native_pred),
        "retrained_12s_model_on_common_12s": nilm_metrics(coarse_truth, coarse_pred),
        "scope": "Same timestamps and targets; model/context/parameter changes still require separate reporting.",
    }
    np.savez_compressed(output / "common_predictions.npz", timestamp=t[valid],
                        y_true=coarse_truth, native_pair_mean=native_pred, coarse_pred=coarse_pred)
    write_json(output / "comparison.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-predictions", required=True)
    parser.add_argument("--coarse-predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.native_predictions, args.coarse_predictions, args.output_dir), indent=2))
