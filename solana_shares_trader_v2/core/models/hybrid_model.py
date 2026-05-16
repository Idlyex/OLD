"""Hybrid Model — Transformer/Mamba encoder + LightGBM/CatBoost stacking.

Architecture:
  1. Feature Extractor: Transformer Encoder (or Mamba) on 1m/5m sequences
  2. Tabular Head: LightGBM + CatBoost ensemble (stacking)
  3. Multi-task outputs: direction, expected return, reversal prob, hold time

Separate models:
  - Exit Model: predicts when to exit (even at loss)
  - Confidence Model: predicts signal reliability
  - Regime Classifier: HMM-based regime detection
"""

import os
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any

from core.models.transformer import FeatureTransformerEncoder, FeatureMambaEncoder
from core.utils.logger import log
from config import config

_model_cfg = config.get("models", {}).get("primary", {})
_data_cfg = config.get("infrastructure", {}).get("data", {})
MODELS_DIR = Path(_data_cfg.get("models_dir", "data/models"))


class MultiTaskHead(nn.Module):
    """Multi-task prediction heads on top of transformer embeddings."""

    def __init__(self, d_model: int = 128):
        super().__init__()

        # Direction classifier (5m)
        self.direction_5m = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2),  # UP / DOWN
        )

        # Direction classifier (15m)
        self.direction_15m = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2),
        )

        # Expected return (regression)
        self.expected_return = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

        # Reversal probability
        self.reversal_prob = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Optimal hold time (in seconds)
        self.hold_time = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.ReLU(),
        )

    def forward(self, embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "direction_5m": self.direction_5m(embedding),
            "direction_15m": self.direction_15m(embedding),
            "expected_return": self.expected_return(embedding).squeeze(-1),
            "reversal_prob": self.reversal_prob(embedding).squeeze(-1),
            "hold_time": self.hold_time(embedding).squeeze(-1),
        }


class ExitHead(nn.Module):
    """Separate exit model — predicts whether to exit now."""

    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class HybridModel:
    """Master model combining deep learning and gradient boosting.

    Pipeline:
    1. Sequence → Transformer/Mamba → embedding
    2. Embedding + tabular features → LightGBM + CatBoost
    3. Stacking: meta-learner combines all predictions
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Config
        use_mamba = _model_cfg.get("transformer", {}).get("use_mamba", False)
        d_model = _model_cfg.get("transformer", {}).get("d_model", 128)
        n_heads = _model_cfg.get("transformer", {}).get("n_heads", 4)
        n_layers = _model_cfg.get("transformer", {}).get("n_layers", 3)
        dropout = _model_cfg.get("transformer", {}).get("dropout", 0.1)
        seq_len = _model_cfg.get("sequence_length", 60)
        n_features = 82

        # Transformer encoder
        if use_mamba:
            self.encoder = FeatureMambaEncoder(
                n_features=n_features, d_model=d_model,
                n_layers=n_layers, dropout=dropout,
            ).to(self.device)
            log.info("HybridModel: using Mamba encoder")
        else:
            self.encoder = FeatureTransformerEncoder(
                n_features=n_features, d_model=d_model,
                n_heads=n_heads, n_layers=n_layers,
                dropout=dropout, seq_len=seq_len,
            ).to(self.device)
            log.info("HybridModel: using Transformer encoder")

        # Multi-task heads
        self.task_heads = MultiTaskHead(d_model).to(self.device)

        # Exit model
        self.exit_model = ExitHead(n_features + 5).to(self.device)  # features + position state

        # Gradient boosting models (lazy-loaded)
        self._lgbm_classifier = None
        self._lgbm_regressor = None
        self._catboost_classifier = None
        self._catboost_regressor = None
        self._meta_model = None  # stacking meta-learner

        self._fitted = False

        MODELS_DIR.mkdir(parents=True, exist_ok=True)

    def predict(self, features: np.ndarray, sequence: np.ndarray = None) -> Dict[str, Any]:
        """Full prediction pipeline.

        Args:
            features: (n_features,) current feature vector
            sequence: (seq_len, n_features) historical sequence for transformer

        Returns:
            Dict with direction, confidence, expected_return, reversal_prob, hold_time
        """
        result = {
            "direction": "NEUTRAL",
            "direction_5m": 0.5,
            "direction_15m": 0.5,
            "confidence": 0.0,
            "expected_return": 0.0,
            "reversal_prob": 0.5,
            "hold_time": 60.0,
            "should_take": False,
            "lgbm_score": 0.5,
            "catboost_score": 0.5,
            "transformer_score": 0.5,
        }

        # ── Deep learning path (if sequence available) ──
        transformer_embedding = None
        if sequence is not None and len(sequence) >= 10:
            try:
                self.encoder.eval()
                self.task_heads.eval()
                with torch.no_grad():
                    x = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
                    embedding = self.encoder(x)
                    tasks = self.task_heads(embedding)

                    dir_5m_probs = torch.softmax(tasks["direction_5m"], dim=-1)
                    dir_15m_probs = torch.softmax(tasks["direction_15m"], dim=-1)

                    result["direction_5m"] = float(dir_5m_probs[0, 1])  # P(UP)
                    result["direction_15m"] = float(dir_15m_probs[0, 1])
                    result["expected_return"] = float(tasks["expected_return"][0])
                    result["reversal_prob"] = float(tasks["reversal_prob"][0])
                    result["hold_time"] = float(tasks["hold_time"][0])
                    result["transformer_score"] = float(dir_5m_probs[0, 1])

                    transformer_embedding = embedding.cpu().numpy()[0]
            except Exception as e:
                log.warning(f"Transformer predict error: {e}")

        # ── Gradient boosting path ──
        features_2d = features.reshape(1, -1)

        if self._lgbm_classifier is not None:
            try:
                lgbm_prob = self._lgbm_classifier.predict_proba(features_2d)[0, 1]
                result["lgbm_score"] = float(lgbm_prob)
            except Exception as e:
                log.debug(f"LightGBM predict error: {e}")

        if self._catboost_classifier is not None:
            try:
                cb_prob = self._catboost_classifier.predict_proba(features_2d)[0, 1]
                result["catboost_score"] = float(cb_prob)
            except Exception as e:
                log.debug(f"CatBoost predict error: {e}")

        # ── Stacking: combine all predictions ──
        scores = [result["lgbm_score"], result["catboost_score"], result["transformer_score"]]
        avg_score = np.mean(scores)

        if self._meta_model is not None:
            try:
                meta_features = np.array(scores + [result["expected_return"]]).reshape(1, -1)
                avg_score = float(self._meta_model.predict_proba(meta_features)[0, 1])
            except Exception:
                pass

        result["confidence"] = float(avg_score)

        # Direction decision
        if avg_score > 0.55:
            result["direction"] = "UP"
            result["should_take"] = True
        elif avg_score < 0.45:
            result["direction"] = "DOWN"
            result["should_take"] = True
        else:
            result["direction"] = "NEUTRAL"
            result["should_take"] = False

        return result

    def predict_exit(self, features: np.ndarray, position_state: np.ndarray) -> float:
        """Predict exit probability. Returns P(should_exit)."""
        try:
            self.exit_model.eval()
            with torch.no_grad():
                x = np.concatenate([features, position_state])
                x_t = torch.FloatTensor(x).unsqueeze(0).to(self.device)
                prob = float(self.exit_model(x_t)[0])
                return prob
        except Exception as e:
            log.debug(f"Exit model predict error: {e}")
            return 0.0

    def train_tabular(self, X: np.ndarray, y: np.ndarray, y_reg: np.ndarray = None):
        """Train LightGBM + CatBoost on tabular features."""
        log.info(f"Training tabular models on {len(X)} samples...")

        lgbm_cfg = _model_cfg.get("lgbm", {})
        cb_cfg = _model_cfg.get("catboost", {})

        # ── LightGBM ──
        try:
            import lightgbm as lgb

            self._lgbm_classifier = lgb.LGBMClassifier(
                n_estimators=lgbm_cfg.get("n_estimators", 500),
                learning_rate=lgbm_cfg.get("learning_rate", 0.05),
                max_depth=lgbm_cfg.get("max_depth", 7),
                num_leaves=lgbm_cfg.get("num_leaves", 63),
                subsample=lgbm_cfg.get("subsample", 0.8),
                colsample_bytree=lgbm_cfg.get("colsample_bytree", 0.8),
                min_child_samples=lgbm_cfg.get("min_child_samples", 20),
                verbose=-1,
                random_state=42,
            )
            self._lgbm_classifier.fit(X, y)
            log.info(f"LightGBM classifier trained ✅")

            if y_reg is not None:
                self._lgbm_regressor = lgb.LGBMRegressor(
                    n_estimators=lgbm_cfg.get("n_estimators", 500),
                    learning_rate=lgbm_cfg.get("learning_rate", 0.05),
                    max_depth=lgbm_cfg.get("max_depth", 7),
                    verbose=-1,
                )
                self._lgbm_regressor.fit(X, y_reg)
                log.info("LightGBM regressor trained ✅")
        except Exception as e:
            log.error(f"LightGBM training error: {e}")

        # ── CatBoost ──
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor

            self._catboost_classifier = CatBoostClassifier(
                iterations=cb_cfg.get("iterations", 500),
                learning_rate=cb_cfg.get("learning_rate", 0.05),
                depth=cb_cfg.get("depth", 7),
                l2_leaf_reg=cb_cfg.get("l2_leaf_reg", 3.0),
                verbose=0,
                random_seed=42,
            )
            self._catboost_classifier.fit(X, y)
            log.info("CatBoost classifier trained ✅")

            if y_reg is not None:
                self._catboost_regressor = CatBoostRegressor(
                    iterations=cb_cfg.get("iterations", 500),
                    learning_rate=cb_cfg.get("learning_rate", 0.05),
                    depth=cb_cfg.get("depth", 7),
                    verbose=0,
                )
                self._catboost_regressor.fit(X, y_reg)
                log.info("CatBoost regressor trained ✅")
        except Exception as e:
            log.error(f"CatBoost training error: {e}")

        # ── Meta-learner (stacking) ──
        try:
            from sklearn.linear_model import LogisticRegression

            lgbm_preds = self._lgbm_classifier.predict_proba(X)[:, 1] if self._lgbm_classifier else np.full(len(X), 0.5)
            cb_preds = self._catboost_classifier.predict_proba(X)[:, 1] if self._catboost_classifier else np.full(len(X), 0.5)
            meta_X = np.column_stack([lgbm_preds, cb_preds, np.full(len(X), 0.5)])  # transformer placeholder
            if y_reg is not None:
                meta_X = np.column_stack([meta_X, y_reg])

            self._meta_model = LogisticRegression(random_state=42)
            self._meta_model.fit(meta_X, y)
            log.info("Meta-learner (stacking) trained ✅")
        except Exception as e:
            log.error(f"Meta-learner training error: {e}")

        self._fitted = True

    def save(self, tag: str = "latest"):
        """Save all models to disk."""
        save_dir = MODELS_DIR / tag
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save PyTorch models
        torch.save(self.encoder.state_dict(), save_dir / "encoder.pt")
        torch.save(self.task_heads.state_dict(), save_dir / "task_heads.pt")
        torch.save(self.exit_model.state_dict(), save_dir / "exit_model.pt")

        # Save sklearn/gbm models
        for name, model in [
            ("lgbm_cls", self._lgbm_classifier),
            ("lgbm_reg", self._lgbm_regressor),
            ("catboost_cls", self._catboost_classifier),
            ("catboost_reg", self._catboost_regressor),
            ("meta", self._meta_model),
        ]:
            if model is not None:
                with open(save_dir / f"{name}.pkl", "wb") as f:
                    pickle.dump(model, f)

        log.info(f"Models saved → {save_dir}")

    def load(self, tag: str = "latest") -> bool:
        """Load all models from disk."""
        load_dir = MODELS_DIR / tag
        if not load_dir.exists():
            log.warning(f"Model directory not found: {load_dir}")
            return False

        try:
            # Load PyTorch
            enc_path = load_dir / "encoder.pt"
            if enc_path.exists():
                self.encoder.load_state_dict(torch.load(enc_path, map_location=self.device))
                self.encoder.eval()

            heads_path = load_dir / "task_heads.pt"
            if heads_path.exists():
                self.task_heads.load_state_dict(torch.load(heads_path, map_location=self.device))
                self.task_heads.eval()

            exit_path = load_dir / "exit_model.pt"
            if exit_path.exists():
                self.exit_model.load_state_dict(torch.load(exit_path, map_location=self.device))
                self.exit_model.eval()

            # Load sklearn/gbm
            for name, attr in [
                ("lgbm_cls", "_lgbm_classifier"),
                ("lgbm_reg", "_lgbm_regressor"),
                ("catboost_cls", "_catboost_classifier"),
                ("catboost_reg", "_catboost_regressor"),
                ("meta", "_meta_model"),
            ]:
                pkl_path = load_dir / f"{name}.pkl"
                if pkl_path.exists():
                    with open(pkl_path, "rb") as f:
                        setattr(self, attr, pickle.load(f))

            self._fitted = True
            log.info(f"Models loaded ← {load_dir}")
            return True

        except Exception as e:
            log.error(f"Model load error: {e}")
            return False

    @property
    def is_fitted(self) -> bool:
        return self._fitted
