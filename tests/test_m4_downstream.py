"""M4 tests: PrimitiveActivityMapping + DatasetSplit on a synthetic scenario.

Scenario: 4 activity CSVs (100 rows each, ts = fid*100 + 0..99), 8 segments
(2 per activity, length 50). Cluster labels [0,0,1,1,2,2,2,2]; cluster 2 is the
true few-shot cluster => activities 2 and 3 are few-shot.
"""
import json
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.framework.step import Step
from src.framework.workflow import Workflow
from src.steps.primitive_activity_mapping_step import PrimitiveActivityMappingStep
from src.steps.dataset_split_step import DatasetSplitStep

CFG = {"paths": {"cache_dir": ".cache", "raw_series": "raw_branch.npy"},
       "dataset_split": {"mains_series": "raw_mains.npy"}}

N_FILES, ROWS, SEG_LEN = 4, 100, 50
LABELS = np.array([0, 0, 1, 1, 2, 2, 2, 2])


class StubUpstream(Step):
    """Writes the scenario files and records manifest entries like the real
    extract/segment/cluster/fewshot steps would."""
    step_type = "stub_upstream"

    def run(self, context):
        log_root = context["log_root"]

        # activities
        seg_dir = os.path.join(log_root, "segments")
        os.makedirs(seg_dir)
        for fid in range(N_FILES):
            pd.DataFrame({"ts": fid * ROWS + np.arange(ROWS, dtype=np.float64),
                          "power": np.full(ROWS, 10.0 * (fid + 1))}).to_csv(
                os.path.join(seg_dir, f"act{fid}.csv"), index=False)
        context["manifest"].add_step(
            "extract_active_data", "stub", "segments",
            {"segments_dir": self.rel(context, seg_dir)})

        # segmentation: 2 segments per activity
        indices = np.array([[f, s * SEG_LEN] for f in range(N_FILES)
                            for s in range(2)], dtype=np.int64)
        lengths = np.full((8, 1), SEG_LEN, dtype=np.int64)
        i_path = os.path.join(log_root, "seg_indices.npy")
        l_path = os.path.join(log_root, "seg_lengths.npy")
        np.save(i_path, indices)
        np.save(l_path, lengths)
        context["manifest"].add_step(
            "time_segmentation", "stub", "seg",
            {"indices": self.rel(context, i_path), "lengths": self.rel(context, l_path)})

        # clustering result (tag kmeans_k3), cluster 2 = few-shot
        cdir = os.path.join(log_root, "TimeClustering_stub", "kmeans_k3")
        os.makedirs(cdir)
        np.save(os.path.join(cdir, "cluster_labels.npy"), LABELS)
        np.save(os.path.join(cdir, "kept_rows.npy"), np.arange(8))
        np.save(os.path.join(cdir, "seq_len.npy"), lengths.reshape(-1))
        context["manifest"].add_cluster_result(
            "kmeans_k3", "TimeClustering_stub/kmeans_k3", {
                "labels": self.rel(context, os.path.join(cdir, "cluster_labels.npy")),
                "kept_rows": self.rel(context, os.path.join(cdir, "kept_rows.npy")),
                "seq_len": self.rel(context, os.path.join(cdir, "seq_len.npy")),
            })

        # fewshot summary
        fdir = os.path.join(log_root, "FewShot_stub")
        os.makedirs(fdir)
        summary_path = os.path.join(fdir, "few_shot_cluster_summary.json")
        with open(summary_path, "w") as f:
            json.dump({"cluster_tag": "kmeans_k3", "true_few_shot_clusters": [2]}, f)
        context["manifest"].add_step(
            "few_shot_cluster_extract", "stub", "FewShot_stub",
            {"summary": self.rel(context, summary_path)})

        # raw series for dataset_split (branch energy = 100 everywhere)
        ts = np.arange(N_FILES * ROWS, dtype=np.float64)
        np.save("raw_branch.npy", np.column_stack([ts, np.full_like(ts, 100.0)]))
        np.save("raw_mains.npy", np.column_stack([ts, np.full_like(ts, 500.0)]))
        return context


class ChdirCase(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self._cwd = os.getcwd()
        os.chdir(self._td.name)

    def tearDown(self):
        os.chdir(self._cwd)
        self._td.cleanup()


class TestPam(ChdirCase):
    def test_index_matching_and_few_shot_split(self):
        wf = Workflow("rpam", "fridge", CFG)
        wf.add(StubUpstream())
        wf.add(PrimitiveActivityMappingStep(cluster_tag="kmeans_k3"))
        wf.run()
        m = wf.manifest

        with open(m.artifact_path("primitive_activity_mapping", "mapping")) as f:
            mapping = json.load(f)
        self.assertEqual(len(mapping), 8)
        self.assertTrue(all(r["match_type"] == "index_match" for r in mapping))
        # timestamps come from the activity CSV, e.g. segment (fid=1, start=50)
        r = next(r for r in mapping
                 if r["activity_csv_idx"] == 1 and r["start_index_in_csv"] == 50)
        self.assertEqual(r["primitive_start_timestamp"], 150.0)
        self.assertEqual(r["primitive_end_timestamp"], 199.0)

        with open(m.artifact_path("primitive_activity_mapping", "few_shot_activities")) as f:
            few = json.load(f)
        with open(m.artifact_path("primitive_activity_mapping", "non_few_shot_activities")) as f:
            non = json.load(f)
        self.assertEqual({a["file_name"] for a in few}, {"act2.csv", "act3.csv"})
        self.assertEqual({a["file_name"] for a in non}, {"act0.csv", "act1.csv"})

        few_tensor = np.load(m.artifact_path("primitive_activity_mapping", "few_shot_tensor"))
        self.assertEqual(few_tensor.shape, (2, ROWS, 2))
        self.assertEqual(m.data["steps"]["primitive_activity_mapping"]["extra"]["n_primitives"], 8)


class TestDatasetSplit(ChdirCase):
    def test_load_series_accepts_canonical_csv(self):
        pd.DataFrame({"timestamp": [1, 2], "power": [10, 20]}).to_csv(
            "series.csv", index=False)
        got = DatasetSplitStep._load_series_2col("series.csv", "series")
        np.testing.assert_array_equal(
            got, np.array([[1.0, 10.0], [2.0, 20.0]]))

    def test_knockout_and_masks(self):
        wf = Workflow("rsplit", "fridge", CFG)
        wf.add(StubUpstream())
        wf.add(PrimitiveActivityMappingStep(cluster_tag="kmeans_k3"))
        wf.add(DatasetSplitStep(raw_series_path="raw_branch.npy",
                                mains_series_path="raw_mains.npy",
                                few_train_ratio=0.5, non_few_train_ratio=0.8))
        wf.run()
        m = wf.manifest

        with open(m.artifact_path("dataset_split", "summary")) as f:
            summary = json.load(f)
        sc = summary["split_counts"]
        self.assertEqual((sc["few_total"], sc["few_train"], sc["few_test"]), (2, 1, 1))
        self.assertEqual((sc["non_few_total"], sc["non_few_train"], sc["non_few_test"]), (2, 2, 0))

        # train drops the few_test activity (100 points) and nothing else
        train_branch = np.load(m.artifact_path("dataset_split", "train_branch"))
        train_mains = np.load(m.artifact_path("dataset_split", "train_mains"))
        zeroed = train_branch[:, 1] == 0.0
        self.assertEqual(int(zeroed.sum()), ROWS)
        np.testing.assert_array_equal(train_mains[zeroed, 1], 400.0)   # 500 - 100
        np.testing.assert_array_equal(train_mains[~zeroed, 1], 500.0)

        # test_b drops everything except few_test events
        tb = np.load(m.artifact_path("dataset_split", "test_b_branch"))
        self.assertEqual(int((tb[:, 1] == 0.0).sum()), 3 * ROWS)

        # masks are consistent with event timestamps
        train_ds = summary["datasets"]["train"]
        self.assertEqual(train_ds["quality"]["drop_points"], ROWS)
        self.assertEqual(train_ds["event_count"],
                         {"few_shot": 1, "non_few_shot": 2, "total": 3})

    def test_timestamp_mismatch_rejected(self):
        # first run: upstream + PAM, so the manifest holds the PAM artifacts
        wf = Workflow("rbad", "fridge", CFG)
        wf.add(StubUpstream())
        wf.add(PrimitiveActivityMappingStep(cluster_tag="kmeans_k3"))
        wf.run()
        # break alignment: mains timestamps shifted
        ts = np.arange(N_FILES * ROWS, dtype=np.float64)
        np.save("raw_mains.npy", np.column_stack([ts + 1, np.full_like(ts, 500.0)]))
        # narrow re-run of just dataset_split reusing the same run-id
        wf2 = Workflow("rbad", "fridge", CFG)
        wf2.add(DatasetSplitStep(raw_series_path="raw_branch.npy",
                                 mains_series_path="raw_mains.npy"))
        with self.assertRaises(ValueError):
            wf2.run()


if __name__ == "__main__":
    unittest.main(verbosity=2)
