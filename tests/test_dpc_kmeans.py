"""Tests for the density-peak-initialized K-Means (dpc-kmeans) clustering.

Covers the new ``cluster_method="dpc-kmeans"`` (all-candidate-k tagged results)
and ``"dpc-kmeans-scan"`` (diagnostic, best-k by rank-sum) in
TimeClusteringStep, plus the module-level functions. sklearn/scipy only — no TF.
"""
import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.framework.workflow import Workflow
from src.steps.time_clustering_step import TimeClusteringStep

try:
    from tests.test_m3_clustering import FeatureStub, make_blobs
except ImportError:  # discovery without package context inserts tests/ itself
    from test_m3_clustering import FeatureStub, make_blobs

CFG = {"paths": {"cache_dir": ".cache"}}


class TestDpcKmeansBlobs(unittest.TestCase):
    def test_all_candidate_k_produces_tagged_results(self):
        with tempfile.TemporaryDirectory() as td:
            cwd = os.getcwd(); os.chdir(td)
            try:
                wf = Workflow("rdpc", "fridge", CFG)
                wf.add(FeatureStub(make_blobs()))
                wf.add(TimeClusteringStep(cluster_method="dpc-kmeans",
                                          n_clusters=[2, 3], dpc_random_state=0))
                wf.run()
                m = wf.manifest
                self.assertEqual(m.cluster_tags(), ["dpc_kmeans_k2", "dpc_kmeans_k3"])
                for tag, k in (("dpc_kmeans_k2", 2), ("dpc_kmeans_k3", 3)):
                    labels = np.load(m.cluster_artifact_path(tag, "labels"))
                    self.assertEqual(len(set(labels.tolist())), k)
                    with open(m.cluster_artifact_path(tag, "metrics")) as f:
                        metrics = json.load(f)
                    self.assertEqual(metrics["cluster_method"], "dpc-kmeans")
                    self.assertEqual(metrics["n_clusters_requested"], k)
                    self.assertGreaterEqual(len(metrics["dpc_init_indices"]), k)
                with open(m.cluster_artifact_path("dpc_kmeans_k3", "metrics")) as f:
                    self.assertGreater(json.load(f)["silhouette_score"], 0.8)
            finally:
                os.chdir(cwd)

    def test_scan_is_diagnostic_only(self):
        with tempfile.TemporaryDirectory() as td:
            cwd = os.getcwd(); os.chdir(td)
            try:
                wf = Workflow("rdsc", "fridge", CFG)
                wf.add(FeatureStub(make_blobs()))
                wf.add(TimeClusteringStep(cluster_method="dpc-kmeans-scan",
                                          n_clusters=[2, 3, 4], dpc_random_state=0))
                wf.run()
                m = wf.manifest
                self.assertEqual(m.cluster_tags(), [])  # no results registered
                scan_path = os.path.join(
                    "log", "rdsc",
                    "TimeClustering_dpc-kmeans-scan_on_detsec_on_clasp",
                    "dpc_scan.json")
                with open(scan_path, encoding="utf-8") as f:
                    scan = json.load(f)
                self.assertEqual(scan["scan_method"], "dpc-kmeans-scan")
                self.assertEqual([r["n_clusters"] for r in scan["records"]], [2, 3, 4])
                self.assertEqual(scan["recommended_n_clusters"], 3)  # clean 3 blobs
            finally:
                os.chdir(cwd)


class TestModuleFunctions(unittest.TestCase):
    def test_dpc_kmeans_clusters_clean_blobs(self):
        from models.clustering.dpc_kmeans import dpc_kmeans
        rng = np.random.RandomState(0)
        Z = np.vstack([c + rng.randn(20, 8) * 0.05
                       for c in (np.zeros(8), np.ones(8) * 5, np.ones(8) * 10)])
        labels, centers, init_idx = dpc_kmeans(Z, K=3, percent=2.0,
                                               min_dist_tau=1.0, random_state=0)
        self.assertEqual(sorted(set(labels.tolist())), [0, 1, 2])
        self.assertEqual(len(init_idx), 3)

    def test_sweep_k_rank_sum(self):
        from models.clustering.dpc_kmeans import sweep_k
        rng = np.random.RandomState(1)
        Z = np.vstack([c + rng.randn(20, 8) * 0.05
                       for c in (np.zeros(8), np.ones(8) * 5, np.ones(8) * 10)])
        best, table = sweep_k(Z, k_range=range(2, 6), random_state=0)
        self.assertEqual(best["K"], 3)
        self.assertEqual(len(table), 4)
        self.assertEqual(len(table[0]["labels"]), 60)

    def test_detsec_pc_registered_for_cache_fingerprint(self):
        """detsec_pc must be in _MODEL_SOURCE so cache invalidates on edits."""
        from src.steps.feature_extract_step import FeatureExtractStep
        srcs = FeatureExtractStep._MODEL_SOURCE["detsec_pc"]
        self.assertTrue(any("detsec_pc" in s for s in srcs))
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.assertTrue(os.path.exists(os.path.join(root, srcs[0])))


if __name__ == "__main__":
    unittest.main(verbosity=2)
