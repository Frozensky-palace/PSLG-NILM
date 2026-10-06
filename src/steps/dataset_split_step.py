"""DatasetSplit step: build train/test sets with event-level masking.

Algorithm preserved from the legacy step: split few-shot / non-few-shot
activity records by ratio (seeded), then for each dataset (train, test_a,
test_b) knock out the dropped events from the appliance branch series and
subtract the same energy from the mains series (optionally clipped at 0).

Inputs resolve from the manifest (primitive_activity_mapping artifacts) and
config (``paths.raw_series``, ``dataset_split.mains_series``) — no per-step
path arguments to wire by hand anymore.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from src.framework.step import Step


class DatasetSplitStep(Step):
    step_type = "dataset_split"

    def __init__(self, raw_series_path: str = None, mains_series_path: str = None,
                 few_train_ratio: float = 0.5, non_few_train_ratio: float = 0.8,
                 random_seed: int = 42, timestamp_tolerance_seconds: float = 0.0,
                 clip_negative_mains_to_zero: bool = True):
        super().__init__(variant="")
        self.raw_series_path = raw_series_path
        self.mains_series_path = mains_series_path
        self.few_train_ratio = float(few_train_ratio)
        self.non_few_train_ratio = float(non_few_train_ratio)
        self.random_seed = int(random_seed)
        self.timestamp_tolerance_seconds = max(0.0, float(timestamp_tolerance_seconds))
        self.clip_negative_mains_to_zero = bool(clip_negative_mains_to_zero)

    def log_subdir(self) -> str:
        return "DatasetSplit"

    # ── input ────────────────────────────────────────────────────

    def _resolve_inputs(self, context: dict) -> dict:
        raw = self.raw_series_path or (context.get("config", {}).get("paths", {}) or {}).get("raw_series")
        mains = self.mains_series_path or \
            (context.get("config", {}).get("dataset_split", {}) or {}).get("mains_series")
        if not raw:
            raise ValueError("[dataset_split] paths.raw_series is missing (config or constructor).")
        if not mains:
            raise ValueError("[dataset_split] dataset_split.mains_series is missing.")

        pam = {}
        for key in ("few_shot_activities", "non_few_shot_activities"):
            p = self.resolve(context, "primitive_activity_mapping", key)
            if not (p and os.path.exists(p)):
                raise FileNotFoundError(
                    f"[dataset_split] primitive_activity_mapping.{key} not resolvable — "
                    "run --steps pam first (same --run-id).")
            pam[key] = p
        return {"raw_series_path": raw, "mains_series_path": mains, **pam}

    @staticmethod
    def _load_series_2col(file_path: str, name: str) -> np.ndarray:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"[dataset_split] {name} not found: {file_path}")
        ext = path.suffix.lower()
        if ext == ".npy":
            data = np.load(path)
        elif ext == ".dat":
            data = np.loadtxt(path)
        elif ext == ".csv":
            # Data preparation and extract_active_data both use the canonical
            # timestamp,power CSV schema, so the final split must accept it as
            # well.  Named columns are preferred; otherwise use the first two.
            import pandas as pd

            frame = pd.read_csv(path)
            if {"timestamp", "power"}.issubset(frame.columns):
                data = frame[["timestamp", "power"]].to_numpy()
            elif {"datetime", "power"}.issubset(frame.columns):
                data = frame[["datetime", "power"]].to_numpy()
            else:
                data = frame.iloc[:, :2].to_numpy()
        else:
            raise ValueError(
                f"[dataset_split] unsupported {name} format: {ext} "
                "(.csv/.dat/.npy only)")
        data = np.asarray(data)
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError(f"[dataset_split] {name} must be (len, >=2) [timestamp, power], "
                             f"got {data.shape}")
        return data[:, :2].astype(np.float64, copy=False)

    @staticmethod
    def _load_activity_records(json_path: str, tag: str) -> list:
        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, list):
            raise ValueError(f"[dataset_split] {tag} activity json must be a list")
        valid = []
        for rec in payload:
            if not isinstance(rec, dict):
                continue
            try:
                start, end = float(rec["start_timestamp"]), float(rec["end_timestamp"])
            except (KeyError, TypeError, ValueError):
                print(f"[dataset_split] skip {tag} record with invalid timestamps: {rec}")
                continue
            if end < start:
                start, end = end, start
            valid.append({**rec, "start_timestamp": start, "end_timestamp": end})
        return valid

    # ── algorithm (preserved from legacy) ────────────────────────

    @staticmethod
    def _ratio_to_count(total: int, ratio: float) -> int:
        return int(round(total * min(max(float(ratio), 0.0), 1.0)))

    def _split_records(self, records, train_ratio, rng):
        if not records:
            return [], []
        idx = np.arange(len(records))
        rng.shuffle(idx)
        train_idx = set(int(i) for i in idx[: self._ratio_to_count(len(records), train_ratio)])
        return ([records[i] for i in range(len(records)) if i in train_idx],
                [records[i] for i in range(len(records)) if i not in train_idx])

    def _build_drop_mask(self, timestamps, drop_records):
        mask = np.zeros(timestamps.shape[0], dtype=bool)
        tol = self.timestamp_tolerance_seconds
        for rec in drop_records:
            start, end = float(rec["start_timestamp"]) - tol, float(rec["end_timestamp"]) + tol
            if end < start:
                start, end = end, start
            mask |= (timestamps >= start) & (timestamps <= end)
        return mask

    def _apply_knockout(self, raw_branch, raw_mains, drop_mask):
        branch = raw_branch.copy()
        mains = raw_mains.copy()
        # A long recording gap may be represented as NaN in the branch.  NaN
        # multiplied by a false mask is still NaN, which would otherwise
        # contaminate a perfectly valid mains sample outside a dropped event.
        delta = np.where(drop_mask, np.nan_to_num(raw_branch[:, 1], nan=0.0), 0.0)
        branch[drop_mask, 1] = 0.0
        mains[:, 1] -= delta
        quality = {
            "drop_points": int(np.sum(drop_mask)),
            "drop_ratio": float(np.mean(drop_mask)) if drop_mask.size else 0.0,
            "mains_negative_points_before_clip": int(np.sum(mains[:, 1] < 0)),
            "mains_negative_total_before_clip": float(np.sum(np.abs(np.minimum(mains[:, 1], 0.0)))),
            "clip_negative_mains_to_zero": bool(self.clip_negative_mains_to_zero),
        }
        if self.clip_negative_mains_to_zero:
            mains[:, 1] = np.maximum(mains[:, 1], 0.0)
        return branch, mains, quality

    @staticmethod
    def _events_total_duration(records) -> float:
        return float(sum(max(0.0, float(r["end_timestamp"]) - float(r["start_timestamp"]))
                         for r in records)) if records else 0.0

    @staticmethod
    def _mask_duration(timestamps, mask) -> float:
        if timestamps.size <= 1 or mask.size == 0:
            return 0.0
        dt = np.diff(timestamps.astype(np.float64))
        dt = np.append(dt, dt[-1])
        return float(np.sum(dt[mask]))

    def _composition_summary(self, name, keep_few, keep_non, timestamps, drop_mask, quality):
        few_count, non_count = len(keep_few), len(keep_non)
        total_count = few_count + non_count
        few_dur = self._events_total_duration(keep_few)
        non_dur = self._events_total_duration(keep_non)
        total_dur = few_dur + non_dur
        return {
            "dataset_name": name,
            "event_count": {"few_shot": few_count, "non_few_shot": non_count, "total": total_count},
            "event_duration_seconds": {"few_shot": few_dur, "non_few_shot": non_dur,
                                       "total": total_dur},
            "event_ratio": {"few_shot": few_count / total_count if total_count else 0.0,
                            "non_few_shot": non_count / total_count if total_count else 0.0},
            "duration_ratio": {"few_shot": few_dur / total_dur if total_dur else 0.0,
                               "non_few_shot": non_dur / total_dur if total_dur else 0.0},
            "mask_duration_seconds": {"dropped": self._mask_duration(timestamps, drop_mask),
                                      "kept": self._mask_duration(timestamps, ~drop_mask)},
            "quality": quality,
        }

    # ── main ─────────────────────────────────────────────────────

    def run(self, context: dict) -> dict:
        if not (0.0 <= self.few_train_ratio <= 1.0 and 0.0 <= self.non_few_train_ratio <= 1.0):
            raise ValueError("[dataset_split] train ratios must be in [0, 1]")
        log_dir = self.log_dir(context)
        paths = self._resolve_inputs(context)

        raw_branch = self._load_series_2col(paths["raw_series_path"], "raw branch series")
        raw_mains = self._load_series_2col(paths["mains_series_path"], "raw mains series")
        if raw_branch.shape[0] != raw_mains.shape[0]:
            raise ValueError(f"[dataset_split] branch/mains length mismatch: "
                             f"{raw_branch.shape[0]} vs {raw_mains.shape[0]}")
        if not np.allclose(raw_branch[:, 0], raw_mains[:, 0],
                           atol=max(1e-9, self.timestamp_tolerance_seconds)):
            raise ValueError("[dataset_split] branch and mains timestamps are not aligned.")

        few_records = self._load_activity_records(paths["few_shot_activities"], "few-shot")
        non_records = self._load_activity_records(paths["non_few_shot_activities"], "non-few-shot")

        rng = np.random.default_rng(self.random_seed)
        few_train, few_test = self._split_records(few_records, self.few_train_ratio, rng)
        non_train, non_test = self._split_records(non_records, self.non_few_train_ratio, rng)

        datasets = {
            "train":  {"keep_few": few_train, "keep_non": non_train,
                       "drop": few_test + non_test},
            "test_a": {"keep_few": few_test, "keep_non": non_test,
                       "drop": few_train + non_train},
            "test_b": {"keep_few": few_test, "keep_non": [],
                       "drop": few_train + non_train + non_test},
        }

        timestamps = raw_branch[:, 0]
        composition, output_paths = {}, {}
        for name, cfg in datasets.items():
            drop_mask = self._build_drop_mask(timestamps, cfg["drop"])
            branch_ds, mains_ds, quality = self._apply_knockout(raw_branch, raw_mains, drop_mask)
            summary = self._composition_summary(name, cfg["keep_few"], cfg["keep_non"],
                                                timestamps, drop_mask, quality)
            prefix_paths = {}
            for suffix, obj, kind in (
                    ("branch.npy", branch_ds, "npy"), ("mains.npy", mains_ds, "npy"),
                    ("drop_mask.npy", drop_mask.astype(np.uint8), "npy"),
                    ("keep_few_shot_events.json", cfg["keep_few"], "json"),
                    ("keep_non_few_shot_events.json", cfg["keep_non"], "json"),
                    ("composition_summary.json", summary, "json")):
                p = os.path.join(log_dir, f"{name}_{suffix}")
                if kind == "npy":
                    np.save(p, obj)
                else:
                    with open(p, "w", encoding="utf-8") as f:
                        json.dump(obj, f, ensure_ascii=False, indent=2)
                prefix_paths[f"{name}_{suffix.replace('.', '_')}"] = p
            output_paths[name] = prefix_paths
            composition[name] = summary
            print(f"[dataset_split] {name}: drop {quality['drop_points']} points "
                  f"({quality['drop_ratio']:.1%}), keep few/non = "
                  f"{len(cfg['keep_few'])}/{len(cfg['keep_non'])}")

        global_summary = {
            "input_paths": paths,
            "raw_branch_shape": list(raw_branch.shape),
            "raw_mains_shape": list(raw_mains.shape),
            "hyper_parameters": {
                "few_train_ratio": self.few_train_ratio,
                "non_few_train_ratio": self.non_few_train_ratio,
                "random_seed": self.random_seed,
                "timestamp_tolerance_seconds": self.timestamp_tolerance_seconds,
                "clip_negative_mains_to_zero": self.clip_negative_mains_to_zero,
            },
            "split_counts": {"few_total": len(few_records), "few_train": len(few_train),
                             "few_test": len(few_test), "non_few_total": len(non_records),
                             "non_few_train": len(non_train), "non_few_test": len(non_test)},
            "datasets": composition,
        }
        summary_path = os.path.join(log_dir, "dataset_split_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(global_summary, f, ensure_ascii=False, indent=2)

        self.record(context, artifacts={
            "summary": self.rel(context, summary_path),
            "train_branch": self.rel(context, output_paths["train"]["train_branch_npy"]),
            "train_mains": self.rel(context, output_paths["train"]["train_mains_npy"]),
            "test_a_branch": self.rel(context, output_paths["test_a"]["test_a_branch_npy"]),
            "test_a_mains": self.rel(context, output_paths["test_a"]["test_a_mains_npy"]),
            "test_b_branch": self.rel(context, output_paths["test_b"]["test_b_branch_npy"]),
            "test_b_mains": self.rel(context, output_paths["test_b"]["test_b_mains_npy"]),
        }, extra=global_summary["split_counts"])

        print(f"[dataset_split] done -> {log_dir}")
        return context
