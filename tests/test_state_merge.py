"""Tests for the temporal state-merge step (functional restoration).

Exercises the pure merge primitives (RLE -> short-run absorption -> similar
merge) and a full in-memory step run on a tiny synthetic manifest (no TF, no
GPU; sklearn is required, as for clustering).
"""
import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.framework.step import Step
from src.framework.workflow import Workflow
from src.steps.temporal_state_merge_step import (TemporalStateMergeStep,
                                                 absorb_short_blocks,
                                                 merge_blocks, merge_activity,
                                                 rle_blocks)


class SegmentStub(Step):
    """Emits fake segmentation tensors into context data."""
    step_type = "time_segmentation"

    def __init__(self, indices, lengths, X):
        super().__init__()
        self._i, self._l, self._X = indices, lengths, X

    def run(self, context):
        context["data"]["X"] = self._X
        context["data"]["lengths"] = self._l
        context["data"]["indices"] = self._i
        return context


class ClusterStub(Step):
    """Registers one fake tagged kmeans result in the manifest."""
    step_type = "time_clustering"

    def __init__(self, tag, labels, indices, seq_len, feats, kept_rows):
        super().__init__()
        self._tag = tag
        self._labels, self._i, self._l, self._f, self._k = labels, indices, seq_len, feats, kept_rows

    def run(self, context):
        log_dir = self.log_dir(context)
        p = os.path.join(log_dir, self._tag)
        os.makedirs(p, exist_ok=True)
        paths = {
            "labels": os.path.join(p, "cluster_labels.npy"),
            "indices": os.path.join(p, "indices.npy"),
            "seq_len": os.path.join(p, "seq_len.npy"),
            "feature_matrix": os.path.join(p, "feature_matrix.npy"),
            "kept_rows": os.path.join(p, "kept_rows.npy"),
        }
        np.save(paths["labels"], self._labels)
        np.save(paths["indices"], np.column_stack((self._i, self._labels)))
        np.save(paths["seq_len"], self._l)
        np.save(paths["feature_matrix"], self._f)
        np.save(paths["kept_rows"], self._k)
        context["manifest"].add_cluster_result(
            self._tag, os.path.join(self.log_subdir(), self._tag).replace(os.sep, "/"),
            {k: self.rel(context, v) for k, v in paths.items()})
        return context


def synthetic_activity_data():
    """Two activities.

    Activity 0: labels  A B B C   (the canonical A,B,B,C -> A,B,C case),
                segments 30 samples each -> no short-run absorption.
    Activity 1: labels  A C A C A (short alternating spurious C between A's),
                segments 10 samples each -> all blocks are short and get
                absorbed into one state.
    Features are far apart per label so the similar-merge never fires at
    tol=0.5.
    """
    rng = np.random.RandomState(0)
    n0 = 4
    starts0 = np.arange(n0) * 30
    labels0 = np.array([0, 1, 1, 2])
    len0 = 30
    n1 = 5
    starts1 = np.arange(n1) * 10
    labels1 = np.array([0, 2, 0, 2, 0])
    len1 = 10
    indices = np.concatenate([
        np.stack([np.zeros(n0, dtype=np.int64), starts0], axis=1),
        np.stack([np.ones(n1, dtype=np.int64), starts1], axis=1)])
    labels = np.concatenate([labels0, labels1])
    seq_len = np.concatenate([np.full(n0, len0, dtype=np.int64),
                              np.full(n1, len1, dtype=np.int64)])
    centers = {0: np.zeros(4), 1: np.full(4, 8.0), 2: np.full(4, -8.0)}
    feats = np.vstack([centers[l] + rng.randn(4) * 0.01 for l in labels])
    return labels, indices, seq_len, feats


class MergePrimitivesTest(unittest.TestCase):
    def setUp(self):
        self.labels, self.indices, self.seq_len, self.feats = synthetic_activity_data()
        self.starts = self.indices[:, 1]
        self.csv = self.indices[:, 0]
        # z-score space (same as the step)
        mu = self.feats.mean(0)
        sd = self.feats.std(0) + 1e-12
        self.feat_norm = (self.feats - mu) / sd

    def test_rle_merges_adjacent_same_label(self):
        rows = [0, 1, 2, 3]  # labels 0,1,1,2
        blocks = rle_blocks(rows, self.labels, self.starts, self.seq_len,
                            self.feat_norm, self.feats)
        self.assertEqual([b["label"] for b in blocks], [0, 1, 2])
        self.assertEqual([b["n_segments"] for b in blocks], [1, 2, 1])
        self.assertEqual(blocks[1]["rows"], [1, 2])

    def test_short_absorption_removes_spurious_state(self):
        rows = [4, 5, 6, 7, 8]  # labels 0,2,0,2,0 (single 10-sample blocks)
        blocks = rle_blocks(rows, self.labels, self.starts, self.seq_len,
                            self.feat_norm, self.feats)
        self.assertEqual(len(blocks), 5)
        merged = absorb_short_blocks(blocks, min_len=15)
        # every 10-sample block is short; the interior ones get absorbed by
        # their longer/equal neighbours until the whole activity is one state
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["n_segments"], 5)
        self.assertGreater(merged[0]["n_absorbed_segments"], 0)

    def test_merge_activity_reconstructs_state_sequence(self):
        rows = [0, 1, 2, 3]  # A,B,B,C
        blocks = merge_activity(rows, self.labels, self.starts, self.seq_len,
                                self.feat_norm, self.feats,
                                min_len=15, enable_similar=True, similar_tol=0.5)
        self.assertEqual([b["label"] for b in blocks], [0, 1, 2])

    def test_merge_blocks_concatenates_in_order(self):
        a = rle_blocks([0], self.labels, self.starts, self.seq_len,
                       self.feat_norm, self.feats)[0]
        b = rle_blocks([1, 2], self.labels, self.starts, self.seq_len,
                       self.feat_norm, self.feats)[0]
        merged = merge_blocks(a, b, label=1, reason="short")
        self.assertEqual(merged["rows"], [0, 1, 2])
        self.assertEqual(merged["length"], 90)
        self.assertEqual(merged["n_segments"], 3)
        self.assertEqual(merged["absorbed_labels"], [0])
        self.assertEqual(merged["n_absorbed_segments"], 1)


class StateMergeStepTest(unittest.TestCase):
    def run_step(self):
        labels, indices, seq_len, feats = synthetic_activity_data()
        kept = np.arange(len(labels), dtype=np.int64)
        # segments look like 4-channel rows; only lengths matter here
        X = np.zeros((len(labels), 10, 4), dtype=np.float32)
        lengths = seq_len.reshape(-1, 1)

        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                wf = Workflow("test_state_merge", "wm", {"paths": {"cache_dir": ".cache"}})
                wf.add(ClusterStub("kmeans_k2", labels, indices, seq_len, feats, kept))
                wf.add(TemporalStateMergeStep(
                    cluster_method="kmeans", feature_model="detsec",
                    segment_method="clasp", min_block_seconds=90.0, fs=0.1666667,
                    enable_similar_merge=True, similar_feature_tol=0.5))
                wf.run()

                m = wf.manifest
                tags = m.cluster_tags()
                self.assertIn("kmeans_k2_merged", tags)
                self.assertNotIn("kmeans_k2_merged_merged", tags)

                res = m.data["steps"]["time_clustering"]["results"]["kmeans_k2_merged"]
                base = os.path.join(tmp, "log", "test_state_merge", res["subdir"])
                seg_labels = np.load(os.path.join(base, "cluster_labels.npy"))
                btb = np.load(os.path.join(base, "segment_to_block.npy"))
                with open(os.path.join(base, "blocks.json"),
                          encoding="utf-8") as stream:
                    blocks = json.load(stream)
                with open(os.path.join(base, "state_sequences.json"),
                          encoding="utf-8") as stream:
                    ss = json.load(stream)
                with open(os.path.join(base, "metrics.json"),
                          encoding="utf-8") as stream:
                    metrics = json.load(stream)

                # activity 0: A,B,B,C -> blocks A,B,C (labels 0,1,2)
                seq0 = ss["0"]
                self.assertEqual([s["state_label"] for s in seq0], [0, 1, 2])
                # activity 1: A,C,A,C,A all short -> absorbed into one state
                self.assertEqual(len(ss["1"]), 1)

                self.assertEqual(len(blocks), metrics["n_blocks"])
                self.assertLess(metrics["n_blocks"], metrics["n_segments"])
                # every segment maps to a block
                self.assertTrue((btb[:, 1] >= 0).all())
                self.assertEqual(len(btb), len(seg_labels))
            finally:
                os.chdir(cwd)

    def test_step_emits_merged_tag(self):
        self.run_step()


if __name__ == "__main__":
    unittest.main()
