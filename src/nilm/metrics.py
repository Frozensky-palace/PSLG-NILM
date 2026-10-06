"""Shared NILM regression and on/off metrics."""
from __future__ import annotations

import numpy as np


def nilm_metrics(y_true, y_pred, *, threshold_w: float = 20.0) -> dict:
    true = np.maximum(np.asarray(y_true, dtype=np.float64).reshape(-1), 0.0)
    pred = np.maximum(np.asarray(y_pred, dtype=np.float64).reshape(-1), 0.0)
    if len(true) != len(pred) or not len(true):
        raise ValueError("true and predicted arrays must be non-empty and aligned")
    true_on = true >= threshold_w
    pred_on = pred >= threshold_w
    tp = int(np.sum(true_on & pred_on))
    fp = int(np.sum(~true_on & pred_on))
    fn = int(np.sum(true_on & ~pred_on))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    true_sum = float(true.sum())
    return {
        "samples": int(len(true)),
        "mae_w": float(np.mean(np.abs(true - pred))),
        "rmse_w": float(np.sqrt(np.mean((true - pred) ** 2))),
        "sae": float(abs(pred.sum() - true_sum) / true_sum) if true_sum else None,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "threshold_w": float(threshold_w),
    }
