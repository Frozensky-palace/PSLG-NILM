"""FeatureExtract step: train an autoencoder on the segmentation tensor.

This is the slow step (GPU training, ``epochs`` loops) and is therefore the
ONLY step with a cache — a content-addressed one (see
``src/framework/feature_cache.py``): the key is a SHA256 over the input tensor
content + model name + every hyperparameter, so a re-run with identical inputs
reuses features no matter the ``--run-id``, while any real change re-trains.

Behavior is preserved from the legacy project for the model dispatch and the
feature contract. Changes:
- resolves input from ``context['data']['X']`` or, when run in isolation, from
  the manifest's ``time_segmentation.X`` — no more suffix-string path guessing;
- NO inline figure generation. The training history is saved as JSON only;
  regenerating the loss curve is the visualize script's job (M4);
- records its outputs in the manifest, with ``extra.cache_hit`` / ``cache_key``.

Model backends (TF / sklearn) are imported lazily inside ``_compute_features``
so importing this module is cheap and safe for tests.
"""
from __future__ import annotations

import datetime
import json
import os

import numpy as np

from src.framework import feature_cache
from src.framework.step import Step


def _jsonable(obj):
    """Convert numpy scalars/arrays in a training-history dict to JSON-safe types."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


class FeatureExtractStep(Step):
    step_type = "feature_extract"

    #: model name -> source files whose content fingerprints go into the cache
    #: key (editing the architecture must invalidate old cache entries).
    #: All models also share ``models/base_model.py``.
    _MODEL_SOURCE = {
        "lstm_ae": ["models/feature_extract/lstm_ae.py"],
        "bilstm_ae": ["models/feature_extract/bilstm_ae.py"],
        "cnn_ae": ["models/feature_extract/cnn_ae.py"],
        "bilstm_ae_attention": ["models/feature_extract/bilstm_ae_attention.py"],
        "detsec": ["models/feature_extract/detsec_model.py"],
        "detsec_pc": ["models/feature_extract/detsec_pc.py"],
        "autoencoder": ["models/feature_extract/autoencoder.py"],
        "dtw": ["models/feature_extract/dtw.py"],
        "physical_stats": ["models/feature_extract/physical_stats.py"],
    }
    _SHARED_SOURCE = ["models/base_model.py"]

    def __init__(self, model_name: str = "detsec", segment_method: str = "clasp",
                 latent_dim: int = 16, epochs: int = 50, batch_size: int = 32,
                 learning_rate: float = 0.0001, patience: int = 5,
                 attention_size: int = 32, cache_enabled: bool = True,
                 embed_dim: int = 32, lambda_phy: float = 0.1,
                 nonneg_channels=None, norm_mode: str = "znorm",
                 embed_proj: str = "none", nonneg_activation: str = "softplus",
                 tf_ratio: float = 1.0, tf_schedule: str = "constant"):
        super().__init__(variant=f"{model_name}_on_{segment_method}")
        self.model_name = model_name
        self.segment_method = segment_method
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        self.attention_size = attention_size
        self.cache_enabled = bool(cache_enabled)
        self.embed_dim = int(embed_dim)
        self.lambda_phy = float(lambda_phy)
        self.nonneg_channels = sorted(
            int(c) for c in (nonneg_channels if nonneg_channels is not None
                             else [0, 1, 2, 3]))
        self.norm_mode = str(norm_mode).lower()
        self.embed_proj = str(embed_proj).lower()
        self.nonneg_activation = str(nonneg_activation).lower()
        self.tf_ratio = float(tf_ratio)
        self.tf_schedule = str(tf_schedule).lower()

    def log_subdir(self) -> str:
        return f"FeatureExtract_{self.model_name}_on_{self.segment_method}"

    # ── input ────────────────────────────────────────────────────

    def _load_input(self, context: dict):
        data = context.get("data", {}) or {}
        X, lengths = data.get("X"), data.get("lengths")
        if X is None:
            x_path = self.resolve(context, "time_segmentation", "X")
            if x_path and os.path.exists(x_path):
                X = np.load(x_path)
                l_path = self.resolve(context, "time_segmentation", "lengths")
                lengths = np.load(l_path) if (l_path and os.path.exists(l_path)) else None
        if X is None:
            raise ValueError(
                "[feature_extract] no input tensor. Run time_segmentation first, or run with a "
                "--run-id whose manifest has time_segmentation.X.")
        if not isinstance(X, np.ndarray) or X.ndim != 3:
            raise ValueError(
                f"[feature_extract] invalid tensor shape {getattr(X, 'shape', None)}; "
                "expected (num, len, dim).")
        return X, lengths

    # ── compute (wrapped by the content-addressed cache) ─────────

    def _model_config(self) -> dict:
        """Every hyperparameter that affects the features — used BOTH for the
        cache key and for the training call, so they can never drift apart."""
        cfg = {
            "latent_dim": self.latent_dim,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "patience": self.patience,
        }
        if self.model_name == "bilstm_ae_attention":
            cfg["attention_size"] = self.attention_size
        if self.model_name == "detsec_pc":
            cfg["embed_dim"] = self.embed_dim
            cfg["lambda_phy"] = self.lambda_phy
            cfg["nonneg_channels"] = self.nonneg_channels
            cfg["norm_mode"] = self.norm_mode
            cfg["embed_proj"] = self.embed_proj
            cfg["nonneg_activation"] = self.nonneg_activation
            cfg["tf_ratio"] = self.tf_ratio
            cfg["tf_schedule"] = self.tf_schedule
        return cfg

    def _code_id(self) -> str:
        """Fingerprint of the model implementation, for cache invalidation on
        code edits. Cheap: reads two small .py files, no TF import."""
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        rels = self._MODEL_SOURCE.get(self.model_name, []) + self._SHARED_SOURCE
        return ":".join(feature_cache.file_fingerprint(os.path.join(root, rel))
                        for rel in rels)

    def _compute_features(self, np_data: np.ndarray, lengths):
        model_config = self._model_config()
        if lengths is not None:
            model_config["lengths"] = lengths

        name = self.model_name
        if name == "lstm_ae":
            from models.feature_extract.lstm_ae import lstm_ae as fn
        elif name == "bilstm_ae":
            from models.feature_extract.bilstm_ae import bilstm_ae as fn
        elif name == "cnn_ae":
            from models.feature_extract.cnn_ae import cnn_ae as fn
        elif name == "bilstm_ae_attention":
            from models.feature_extract.bilstm_ae_attention import bilstm_ae_attention as fn
        elif name == "detsec":
            from models.feature_extract.detsec_model import detsec_ae as fn
        elif name == "detsec_pc":
            from models.feature_extract.detsec_pc import detsec_pc as fn
        elif name == "autoencoder":
            from models.feature_extract.autoencoder import autoencoder as fn
        elif name == "dtw":
            from models.feature_extract.dtw import dtw_feature_extract as fn
        elif name == "physical_stats":
            from models.feature_extract.physical_stats import physical_stats as fn
        else:
            raise ValueError(f"[feature_extract] unknown model: {name}")

        print(f"[feature_extract] training {name} (latent_dim={self.latent_dim}, epochs={self.epochs})")
        features, training_history = fn(np_data, model_config)
        return features, training_history

    # ── main ─────────────────────────────────────────────────────

    def run(self, context: dict) -> dict:
        log_dir = self.log_dir(context)
        X, lengths = self._load_input(context)
        print(f"[feature_extract] model={self.model_name}  input={X.shape}")

        cache_dir = context.get("cache_dir", ".cache")
        cache_hit = False
        cache_key = None
        code_id = self._code_id() if self.cache_enabled else ""
        if self.cache_enabled:
            cache_key = feature_cache.compute_key(
                X, lengths, self.model_name, self._model_config(), code_id=code_id)
            hit = feature_cache.load(cache_dir, cache_key)
            if hit is not None:
                features, training_history = hit
                cache_hit = True
                print(f"[feature_extract] cache HIT ({cache_key[:12]}...) — skip training")
        if not cache_hit:
            features, training_history = self._compute_features(X, lengths)
            if self.cache_enabled and cache_key is not None:
                feature_cache.store(cache_dir, cache_key, features,
                                    _jsonable(training_history or {}), meta={
                                        # provenance for humans — deliberately
                                        # NOT part of the cache key
                                        "appliance": context.get("appliance"),
                                        "model": self.model_name,
                                        "segment_method": self.segment_method,
                                        "model_config": self._model_config(),
                                        "code_id": code_id,
                                        "input_shape": list(X.shape),
                                        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
                                    })
        print(f"[feature_extract] features shape: {features.shape}")

        feature_path = os.path.join(log_dir, "features.npy")
        np.save(feature_path, features)
        history_path = os.path.join(log_dir, "training_history.json")
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(_jsonable(training_history or {}), f, indent=2, ensure_ascii=False)

        context.setdefault("data", {})["features"] = features

        self.record(context, artifacts={
            "features": self.rel(context, feature_path),
            "training_history": self.rel(context, history_path),
        }, extra={
            "model": self.model_name,
            "feature_shape": list(features.shape),
            "cache_hit": cache_hit,
            "cache_key": cache_key,
        })

        # Sliding release: the big segmentation tensor is no longer needed in
        # memory (downstream reads it from the manifest if it must).
        context["data"].pop("X", None)

        print(f"[feature_extract] done -> {log_dir}")
        return context
