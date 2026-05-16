"""Hyperparameter Tuning — Optuna-based optimization for LightGBM + CatBoost.

Uses Purged K-Fold with embargo to prevent data leakage.
Optimizes both individual models and ensemble weights.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from pathlib import Path

from training.dataset import TrainingDataset, PurgedKFold
from core.utils.logger import log
from config import config


class HyperTuner:
    """Optuna-based hyperparameter optimization."""

    def __init__(self, n_trials: int = 100, n_folds: int = 5, forward_minutes: int = 5):
        self.n_trials = n_trials
        self.n_folds = n_folds
        self.forward_minutes = forward_minutes

    def tune(self, data: pd.DataFrame) -> Dict:
        """Run hyperparameter optimization.

        Args:
            data: processed OHLCV DataFrame

        Returns:
            Best parameters dict
        """
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            log.error("Optuna not installed. Run: pip install optuna")
            return {}

        log.info("╔══════════════════════════════════════════════════╗")
        log.info(f"║  Hyperparameter Tuning — {self.n_trials} trials              ║")
        log.info("╚══════════════════════════════════════════════════╝")

        # Build dataset
        ds = TrainingDataset()
        dataset = ds.build_from_dataframe(data, self.forward_minutes)
        X = dataset["X"]
        y = dataset["y_direction"]

        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)

        # Purged K-Fold
        pkf = PurgedKFold(n_splits=self.n_folds, embargo_pct=0.02)
        splits = pkf.split(X_scaled)

        # ── LightGBM Tuning ──
        log.info("\n--- LightGBM Tuning ---")

        def lgbm_objective(trial):
            import lightgbm as lgb

            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "verbose": -1,
                "random_state": 42,
                "class_weight": "balanced",
            }

            scores = []
            for train_idx, test_idx in splits:
                model = lgb.LGBMClassifier(**params)
                model.fit(X_scaled[train_idx], y[train_idx])
                preds = model.predict(X_scaled[test_idx])
                acc = float(np.mean(preds == y[test_idx]))
                scores.append(acc)

            return float(np.mean(scores))

        lgbm_study = optuna.create_study(direction="maximize", study_name="lgbm")
        lgbm_study.optimize(lgbm_objective, n_trials=self.n_trials // 2, show_progress_bar=True)

        best_lgbm = lgbm_study.best_params
        log.info(f"  Best LightGBM: acc={lgbm_study.best_value:.4f}")
        log.info(f"  Params: {best_lgbm}")

        # ── CatBoost Tuning ──
        log.info("\n--- CatBoost Tuning ---")

        def catboost_objective(trial):
            from catboost import CatBoostClassifier

            params = {
                "iterations": trial.suggest_int("iterations", 100, 800),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "depth": trial.suggest_int("depth", 3, 10),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 10.0, log=True),
                "border_count": trial.suggest_int("border_count", 32, 255),
                "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 10.0),
                "verbose": 0,
                "random_seed": 42,
                "auto_class_weights": "Balanced",
            }

            scores = []
            for train_idx, test_idx in splits:
                model = CatBoostClassifier(**params)
                model.fit(X_scaled[train_idx], y[train_idx])
                preds = model.predict(X_scaled[test_idx]).flatten().astype(int)
                acc = float(np.mean(preds == y[test_idx]))
                scores.append(acc)

            return float(np.mean(scores))

        cb_study = optuna.create_study(direction="maximize", study_name="catboost")
        cb_study.optimize(catboost_objective, n_trials=self.n_trials // 2, show_progress_bar=True)

        best_cb = cb_study.best_params
        log.info(f"  Best CatBoost: acc={cb_study.best_value:.4f}")
        log.info(f"  Params: {best_cb}")

        result = {
            "lgbm": {
                "best_params": best_lgbm,
                "best_accuracy": float(lgbm_study.best_value),
            },
            "catboost": {
                "best_params": best_cb,
                "best_accuracy": float(cb_study.best_value),
            },
        }

        # Save best params
        import json
        out_path = Path("training/model_registry/best_params.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        log.info(f"  💾 Best params saved → {out_path}")

        return result
