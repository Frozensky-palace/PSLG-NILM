"""TimeSegmentation step: segment each active-data CSV into sub-event regimes.

Behavior is preserved from the legacy project (wavelet separation + a
changepoint algorithm per file → padded 4-channel tensor). Changes:
- adapts to the new ``Step`` base + run manifest (registers X/lengths/indices);
- resolves its input directory from ``context['input_root']`` or, when run in
  isolation, from the manifest's ``extract_active_data.segments_dir`` — no more
  guessing ``log_root/DataLoader``;
- drops the dead matplotlib import and the unused ``dataset_split_step`` import;
- drops the per-N checkpoint .npy files (redundant intermediate artifacts).

Backend libraries (claspy / fluss / espresso) are imported lazily inside their
branch so importing this module never pulls in numba/stumpy/TF.
"""
from __future__ import annotations

import gc
import os

import numpy as np
import pandas as pd
import pywt
from scipy.signal import medfilt

from src.framework.step import Step


class TimeSegmentationStep(Step):
    step_type = "time_segmentation"

    def __init__(self, segment_method: str = "clasp", appliance_name: str = "",
                 window_size: int = 100, n_regimes: int = 3, excl_factor: int = 5,
                 clasp_n_jobs: int = 1, clasp_n_segments: str = "learn",
                 max_seg_len: int = 0, input_dir: str = ""):
        super().__init__(variant=segment_method)
        self.segment_method = segment_method
        self.appliance_name = appliance_name
        self.window_size = window_size
        self.n_regimes = n_regimes
        self.excl_factor = excl_factor
        self.clasp_n_jobs = int(clasp_n_jobs)
        self.clasp_n_segments = str(clasp_n_segments)
        self.max_seg_len = int(max_seg_len or 0)
        self.input_dir = str(input_dir or "")

    # ── input resolution ─────────────────────────────────────────

    def _resolve_input_dir(self, context: dict):
        candidates = []
        if self.input_dir:
            candidates.append(self.input_dir)
        if context.get("input_root"):
            candidates.append(context["input_root"])
        seg_dir = self.resolve(context, "extract_active_data", "segments_dir")
        if seg_dir:
            candidates.append(seg_dir)
        candidates.append(os.path.join(context["log_root"], "DataLoader"))  # legacy fallback
        for c in candidates:
            if c and os.path.isdir(c):
                return c
        return None

    # ── signal processing (unchanged logic) ─────────────────────

    def medfilt_outlier_removal(self, series):
        ts = np.asarray(series)
        cleaned_series = medfilt(ts, kernel_size=5)
        outlier_mask = np.zeros_like(ts, dtype=bool)
        return cleaned_series, outlier_mask

    def get_segmentation_points(self, time_series, distance="znormed_euclidean_distance"):
        if time_series is None or len(time_series) == 0:
            print("[time_segmentation] empty signal, skipping.")
            return []
        if np.any(np.isnan(time_series)) or np.any(np.isinf(time_series)):
            time_series = np.nan_to_num(time_series, nan=0.0, posinf=0.0, neginf=0.0)

        method = self.segment_method
        if method == "none":
            return []

        if method == "fluss":
            os.environ.setdefault("NUMBA_NUM_THREADS", "1")
            os.environ.setdefault("MKL_NUM_THREADS", "1")
            os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
            from fluss import fluss
            try:
                ts_1d = time_series.flatten().astype(np.float64)
                min_len = max(self.window_size * 2, self.window_size + self.n_regimes)
                if len(ts_1d) < min_len:
                    print(f"[time_segmentation] FLUSS: signal too short ({len(ts_1d)}<{min_len})")
                    return []
                _, change_points = fluss(
                    ts_1d, window_size=self.window_size, n_regimes=self.n_regimes,
                    excl_factor=self.excl_factor, visualize=False)
                return change_points
            except Exception as e:
                print(f"[time_segmentation] FLUSS error: {e}")
                return []

        if method == "espresso":
            from models.time_segmentation.espresso import EspressoModel
            try:
                model = EspressoModel(config={"window_size": self.window_size})
                return model.train(time_series)
            except Exception as e:
                print(f"[time_segmentation] ESPRESSO error: {e}")
                return []

        if method == "clasp-origin":
            from models.time_segmentation.clasp_origin import ClaspOriginModel
            try:
                model = ClaspOriginModel(config={"distance": "euclidean_distance"})
                return model.train(time_series)
            except Exception as e:
                print(f"[time_segmentation] clasp-origin error: {e}")
                return []

        if method == "prim-glr":
            from models.time_segmentation.prim_glr import PrimGLRModel
            try:
                model = PrimGLRModel()
                return model.train(time_series)
            except Exception as e:
                print(f"[time_segmentation] prim-glr error: {e}")
                return []

        # default: clasp
        from claspy.segmentation import BinaryClaSPSegmentation
        try:
            clasp = BinaryClaSPSegmentation(
                n_segments=self.clasp_n_segments, window_size="suss",
                validation="score_threshold", threshold=0.001,
                distance=distance, n_jobs=self.clasp_n_jobs)
            clasp.fit_predict(time_series)
            return clasp.change_points
        except Exception as e:
            print(f"[time_segmentation] Clasp error: {e}")
            return []

    def synthesize_changepoints(self, orig_cp, low_cp, high_cp):
        if len(low_cp) == 0 and len(high_cp) == 0:
            return [], "None"
        if len(low_cp) >= len(high_cp):
            ref_cp, others, ref_name = np.sort(low_cp), [np.sort(high_cp)], "Low-Freq"
        else:
            ref_cp, others, ref_name = np.sort(high_cp), [np.sort(low_cp)], "High-Freq"
        if len(ref_cp) == 0:
            return [], "None"
        groups = {i: [ref_val] for i, ref_val in enumerate(ref_cp)}
        for other_list in others:
            for p in other_list:
                groups[int(np.argmin(np.abs(ref_cp - p)))].append(p)
        synthesized_cp = [float(np.mean(groups[i])) for i in sorted(groups.keys())]
        return sorted(synthesized_cp), ref_name

    def run_wavelet_analysis(self, signal, wavelet, orig_cp):
        level = 2
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        cA2, cD2, cD1 = coeffs
        low_freq_signal = pywt.waverec([cA2, np.zeros_like(cD2), np.zeros_like(cD1)], wavelet)[:len(signal)]
        high_freq_combined = pywt.waverec([np.zeros_like(cA2), cD2, cD1], wavelet)[:len(signal)]
        low_cp = self.get_segmentation_points(low_freq_signal)
        high_cp = self.get_segmentation_points(high_freq_combined)
        synthesized_cp, ref_name = self.synthesize_changepoints(orig_cp, low_cp, high_cp)
        return {
            "low_freq_signal": low_freq_signal,
            "high_freq_combined": high_freq_combined,
            "synthesized_cp": synthesized_cp,
            "ref_name": ref_name,
        }

    # ── main ─────────────────────────────────────────────────────

    def run(self, context: dict) -> dict:
        # Sliding context release: free ExtractActiveData's payload once we have read it.
        if "data" in context and "extract_active_data" in context["data"]:
            del context["data"]["extract_active_data"]
            gc.collect()

        input_dir = self._resolve_input_dir(context)
        log_dir = self.log_dir(context)
        if not input_dir:
            print(f"[time_segmentation] no input dir found for segments; skipping.")
            return context

        candidate_files = sorted(
            f for f in os.listdir(input_dir) if f.lower().endswith(".csv"))
        # A cycle directory may also contain provenance CSVs.  Exclude them
        # before assigning csv_idx so indices remain aligned with the source map.
        target_files = []
        for file_name in candidate_files:
            columns = pd.read_csv(os.path.join(input_dir, file_name), nrows=0).columns
            if "power" in columns:
                target_files.append(file_name)
        print(f"[time_segmentation] {len(target_files)} CSV files, method={self.segment_method}")

        all_samples, all_lengths, all_indices = [], [], []

        for i, file_name in enumerate(target_files):
            print(f"  [{i + 1}/{len(target_files)}] {file_name}", flush=True)
            df = pd.read_csv(os.path.join(input_dir, file_name))
            if "power" not in df.columns:
                print(f"  skip {file_name}: no 'power' column")
                continue
            signal = df["power"].values

            if self.segment_method == "none":
                signal_cleaned = signal
                low_freq, high_freq = signal, np.zeros_like(signal)
                synth_cp = []
            elif self.segment_method == "clasp":
                signal_cleaned, _ = self.medfilt_outlier_removal(signal)
                orig_cp = self.get_segmentation_points(signal_cleaned)
                res = self.run_wavelet_analysis(signal_cleaned, "db4", orig_cp)
                low_freq, high_freq = res["low_freq_signal"], res["high_freq_combined"]
                synth_cp = res["synthesized_cp"]
            else:  # fluss / espresso / clasp-origin: direct segmentation, no wavelet
                signal_cleaned = signal
                low_freq, high_freq = signal, np.zeros_like(signal)
                synth_cp = self.get_segmentation_points(signal_cleaned)

            cps = sorted({int(round(cp)) for cp in synth_cp})
            boundaries = [0] + cps + [len(signal_cleaned)]
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                if end <= start:
                    continue
                sample = np.stack([
                    signal[start:end], signal_cleaned[start:end],
                    low_freq[start:end], high_freq[start:end]], axis=1)
                all_samples.append(sample)
                all_lengths.append(len(sample))
                all_indices.append([i, start])

            del df, signal, signal_cleaned
            gc.collect()

        if not all_samples:
            print("[time_segmentation] no segments produced.")
            return context

        # Optional hard cap on primitive length (max_seg_len>0). Guard against
        # pathological longest-primitive tensors: very long BiGRU sequences can
        # hang TF/CuDNN training (observed at 1765 steps on 4090 while <=1536
        # trains). Truncating the tail keeps indices/coverage intact; only the
        # longest few primitives are affected.
        if self.max_seg_len > 0:
            for idx, sample in enumerate(all_samples):
                if len(sample) > self.max_seg_len:
                    all_samples[idx] = sample[: self.max_seg_len]
                    all_lengths[idx] = self.max_seg_len

        max_len = max(all_lengths)
        n_samples = len(all_samples)
        X = np.zeros((n_samples, max_len, 4), dtype=np.float32)
        for idx, sample in enumerate(all_samples):
            X[idx, : all_lengths[idx], :] = sample
        lengths = np.array(all_lengths, dtype=np.int32).reshape(-1, 1)
        indices = np.array(all_indices, dtype=np.int32)

        x_path = os.path.join(log_dir, "X.npy")
        l_path = os.path.join(log_dir, "lengths.npy")
        i_path = os.path.join(log_dir, "indices.npy")
        np.save(x_path, X)
        np.save(l_path, lengths)
        np.save(i_path, indices)

        context.setdefault("data", {}).update({"X": X, "lengths": lengths, "indices": indices})
        context["segment_method"] = self.segment_method

        self.record(context, artifacts={
            "X": self.rel(context, x_path),
            "lengths": self.rel(context, l_path),
            "indices": self.rel(context, i_path),
        }, extra={"n_samples": n_samples, "max_len": int(max_len)})

        print(f"[time_segmentation] tensor: {n_samples} samples, max_len={max_len} -> {log_dir}")

        del all_samples, all_lengths, all_indices
        gc.collect()
        return context
