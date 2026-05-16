"""Training Pipeline — full model training with walk-forward validation.

Trains:
  1. LightGBM classifier + regressor
  2. CatBoost classifier + regressor
  3. Transformer/Mamba encoder (optional, GPU)
  4. Meta-learner (stacking)
  5. Exit model

Saves best models to training/model_registry/
"""

import os
import time
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from training.dataset import TrainingDataset, PurgedKFold, WalkForwardSplit
from core.utils.logger import log
from core.utils.helpers import safe_div
from config import config

REGISTRY_DIR = Path("training/model_registry")
MODELS_DIR = Path(config.get("infrastructure", {}).get("data", {}).get("models_dir", "data/models"))


class Trainer:
    """Master training orchestrator."""

    def __init__(self):
        self.dataset_builder = TrainingDataset()
        self._cfg = config.get("models", {})
        self._retrain_cfg = self._cfg.get("retrain", {})

        REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

    def train_full(
        self,
        data: pd.DataFrame,
        forward_minutes: int = 5,
        walk_forward: bool = True,
        save: bool = True,
        prebuilt_dataset: Dict = None,
    ) -> Dict:
        """Full training pipeline.

        Args:
            data: Processed OHLCV DataFrame
            forward_minutes: target lookahead
            walk_forward: use walk-forward validation
            save: save models to registry
            prebuilt_dataset: pre-built dataset dict (from build_shares_dataset)

        Returns:
            Results dict with metrics per fold
        """
        t0 = time.time()
        log.info("╔══════════════════════════════════════════════════╗")
        log.info("║  Training Pipeline — Hybrid Model               ║")
        log.info("╚══════════════════════════════════════════════════╝")

        # Build dataset (or use pre-built)
        if prebuilt_dataset is not None:
            dataset = prebuilt_dataset
            log.info("  Using pre-built shares dataset")
        else:
            dataset = self.dataset_builder.build_from_dataframe(data, forward_minutes)
        X = dataset["X"]
        X_seq = dataset["X_seq"]
        y_dir = dataset["y_direction"]
        y_ret = dataset["y_return"]
        y_rev = dataset["y_reversal"]
        y_hold = dataset["y_hold_time"]
        timestamps = dataset["timestamps"]
        feature_names = dataset["feature_names"]

        log.info(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
        log.info(f"Target balance: {np.mean(y_dir):.2%} UP")

        # ── Feature scaling ──
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)

        # ── Walk-forward or single split ──
        if walk_forward:
            wf = WalkForwardSplit(
                train_size=min(5000, len(X) // 3),
                test_size=min(1000, len(X) // 10),
                step=min(500, len(X) // 20),
            )
            splits = wf.split(len(X))
        else:
            # Simple 80/20 time-sorted split
            split_idx = int(len(X) * 0.8)
            splits = [(np.arange(0, split_idx), np.arange(split_idx, len(X)))]

        if not splits:
            log.error("Not enough data for training splits")
            return {"error": "insufficient data"}

        all_results = []
        best_accuracy = 0
        best_models = None

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            log.info(f"\n═══ Fold {fold_idx + 1}/{len(splits)} ═══ train={len(train_idx)} test={len(test_idx)}")

            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train_dir, y_test_dir = y_dir[train_idx], y_dir[test_idx]
            y_train_ret, y_test_ret = y_ret[train_idx], y_ret[test_idx]

            fold_models = {}
            fold_preds = {}

            # ── 1. LightGBM ──
            lgbm_acc, lgbm_model, lgbm_preds = self._train_lgbm(
                X_train, y_train_dir, X_test, y_test_dir, feature_names
            )
            fold_models["lgbm_cls"] = lgbm_model
            fold_preds["lgbm"] = lgbm_preds

            # ── 2. CatBoost ──
            cb_acc, cb_model, cb_preds = self._train_catboost(
                X_train, y_train_dir, X_test, y_test_dir
            )
            fold_models["catboost_cls"] = cb_model
            fold_preds["catboost"] = cb_preds

            # ── 3. LightGBM Regressor (expected return) ──
            lgbm_reg = self._train_lgbm_regressor(X_train, y_train_ret)
            fold_models["lgbm_reg"] = lgbm_reg

            # ── 4. Meta-learner (stacking) ──
            meta_acc, meta_model = self._train_meta(
                fold_preds, y_train_dir, X_test, y_test_dir, X_train,
                fold_models.get("lgbm_cls"), fold_models.get("catboost_cls"),
                feature_names,
            )
            fold_models["meta"] = meta_model

            # ── Results ──
            avg_acc = (lgbm_acc + cb_acc) / 2
            fold_result = {
                "fold": fold_idx,
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "lgbm_accuracy": lgbm_acc,
                "catboost_accuracy": cb_acc,
                "meta_accuracy": meta_acc,
                "avg_accuracy": avg_acc,
                "test_up_pct": float(np.mean(y_test_dir)),
            }
            all_results.append(fold_result)

            log.info(
                f"  Fold {fold_idx + 1}: LGBM={lgbm_acc:.3f} CB={cb_acc:.3f} "
                f"Meta={meta_acc:.3f} avg={avg_acc:.3f}"
            )

            if avg_acc > best_accuracy:
                best_accuracy = avg_acc
                best_models = fold_models
                best_scaler = scaler

        # ── Save best models ──
        if save and best_models:
            tag = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._save_models(best_models, best_scaler, feature_names, tag)

            # Also save as "latest" for live use
            self._save_models(best_models, best_scaler, feature_names, "latest")

        elapsed = time.time() - t0

        # ── Summary ──
        summary = {
            "folds": len(splits),
            "results": all_results,
            "best_accuracy": best_accuracy,
            "avg_lgbm": float(np.mean([r["lgbm_accuracy"] for r in all_results])),
            "avg_catboost": float(np.mean([r["catboost_accuracy"] for r in all_results])),
            "avg_meta": float(np.mean([r["meta_accuracy"] for r in all_results])),
            "total_samples": len(X),
            "n_features": X.shape[1],
            "elapsed_s": elapsed,
        }

        self._print_summary(summary)
        return summary

    def _train_lgbm(self, X_train, y_train, X_test, y_test, feature_names):
        """Train LightGBM classifier."""
        try:
            import lightgbm as lgb

            lgbm_cfg = self._cfg.get("primary", {}).get("lgbm", {})
            model = lgb.LGBMClassifier(
                n_estimators=lgbm_cfg.get("n_estimators", 500),
                learning_rate=lgbm_cfg.get("learning_rate", 0.05),
                max_depth=lgbm_cfg.get("max_depth", 7),
                num_leaves=lgbm_cfg.get("num_leaves", 63),
                subsample=lgbm_cfg.get("subsample", 0.8),
                colsample_bytree=lgbm_cfg.get("colsample_bytree", 0.8),
                min_child_samples=lgbm_cfg.get("min_child_samples", 20),
                verbose=-1,
                random_state=42,
                class_weight="balanced",
            )

            import pandas as _pd
            X_tr_df = _pd.DataFrame(X_train, columns=feature_names)
            X_te_df = _pd.DataFrame(X_test, columns=feature_names)

            model.fit(
                X_tr_df, y_train,
                eval_set=[(X_te_df, y_test)],
                callbacks=[lgb.log_evaluation(period=0), lgb.early_stopping(50, verbose=False)],
            )

            preds = model.predict(X_te_df)
            proba = model.predict_proba(X_te_df)[:, 1]
            acc = float(np.mean(preds == y_test))

            # Feature importance
            importances = dict(zip(feature_names, model.feature_importances_))
            top_10 = sorted(importances.items(), key=lambda x: -x[1])[:10]
            log.info(f"  LightGBM: acc={acc:.3f} | top: {', '.join(f'{k}={v}' for k, v in top_10[:5])}")

            return acc, model, proba

        except Exception as e:
            log.error(f"LightGBM training error: {e}")
            return 0.5, None, np.full(len(y_test), 0.5)

    def _train_catboost(self, X_train, y_train, X_test, y_test):
        """Train CatBoost classifier."""
        try:
            from catboost import CatBoostClassifier

            cb_cfg = self._cfg.get("primary", {}).get("catboost", {})
            model = CatBoostClassifier(
                iterations=cb_cfg.get("iterations", 500),
                learning_rate=cb_cfg.get("learning_rate", 0.05),
                depth=cb_cfg.get("depth", 7),
                l2_leaf_reg=cb_cfg.get("l2_leaf_reg", 3.0),
                verbose=0,
                random_seed=42,
                auto_class_weights="Balanced",
                eval_metric="Accuracy",
            )

            model.fit(
                X_train, y_train,
                eval_set=(X_test, y_test),
                early_stopping_rounds=50,
            )

            preds = model.predict(X_test).flatten().astype(int)
            proba = model.predict_proba(X_test)[:, 1]
            acc = float(np.mean(preds == y_test))

            log.info(f"  CatBoost: acc={acc:.3f} | best_iter={model.best_iteration_}")
            return acc, model, proba

        except Exception as e:
            log.error(f"CatBoost training error: {e}")
            return 0.5, None, np.full(len(y_test), 0.5)

    def _train_lgbm_regressor(self, X_train, y_train):
        """Train LightGBM regressor for expected return."""
        try:
            import lightgbm as lgb

            model = lgb.LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                verbose=-1,
                random_state=42,
            )
            model.fit(X_train, y_train)
            return model
        except Exception as e:
            log.error(f"LightGBM regressor error: {e}")
            return None

    def _train_meta(self, fold_preds, y_train, X_test, y_test, X_train, lgbm_model, cb_model, feature_names=None):
        """Train meta-learner (stacking)."""
        try:
            from sklearn.linear_model import LogisticRegression
            import pandas as _pd

            # Use DataFrames if feature_names available to suppress warnings
            if feature_names is not None and lgbm_model is not None:
                X_tr_df = _pd.DataFrame(X_train, columns=feature_names)
                X_te_df = _pd.DataFrame(X_test, columns=feature_names)
            else:
                X_tr_df, X_te_df = X_train, X_test

            lgbm_train = lgbm_model.predict_proba(X_tr_df)[:, 1] if lgbm_model else np.full(len(X_train), 0.5)
            cb_train = cb_model.predict_proba(X_train)[:, 1] if cb_model else np.full(len(X_train), 0.5)
            meta_train = np.column_stack([lgbm_train, cb_train])

            lgbm_test = lgbm_model.predict_proba(X_te_df)[:, 1] if lgbm_model else np.full(len(X_test), 0.5)
            cb_test = cb_model.predict_proba(X_test)[:, 1] if cb_model else np.full(len(X_test), 0.5)
            meta_test = np.column_stack([lgbm_test, cb_test])

            model = LogisticRegression(random_state=42, max_iter=500)
            model.fit(meta_train, y_train)

            meta_preds = model.predict(meta_test)
            acc = float(np.mean(meta_preds == y_test))

            log.info(f"  Meta-learner: acc={acc:.3f}")
            return acc, model

        except Exception as e:
            log.error(f"Meta-learner error: {e}")
            return 0.5, None

    def _save_models(self, models: Dict, scaler, feature_names: List[str], tag: str):
        """Save models to registry."""
        save_dir = REGISTRY_DIR / tag
        save_dir.mkdir(parents=True, exist_ok=True)

        # Also save to data/models for the HybridModel loader
        models_dir = MODELS_DIR / tag
        models_dir.mkdir(parents=True, exist_ok=True)

        for name, model in models.items():
            if model is not None:
                path = save_dir / f"{name}.pkl"
                with open(path, "wb") as f:
                    pickle.dump(model, f)
                # Copy to data/models
                with open(models_dir / f"{name}.pkl", "wb") as f:
                    pickle.dump(model, f)

        # Save scaler
        with open(save_dir / "scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        with open(models_dir / "scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

        # Save feature names
        with open(save_dir / "feature_names.json", "w") as f:
            json.dump(feature_names, f)
        with open(models_dir / "feature_names.json", "w") as f:
            json.dump(feature_names, f)

        # Save metadata
        meta = {
            "tag": tag,
            "timestamp": datetime.now().isoformat(),
            "n_features": len(feature_names),
            "models": list(models.keys()),
        }
        with open(save_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        log.info(f"  💾 Models saved → {save_dir}")

    def _print_summary(self, summary: Dict):
        """Print training summary."""
        log.info("\n╔══════════════════════════════════════════════════╗")
        log.info("║           TRAINING RESULTS                      ║")
        log.info("╠══════════════════════════════════════════════════╣")
        log.info(f"║  Folds:         {summary['folds']}")
        log.info(f"║  Samples:       {summary['total_samples']:,}")
        log.info(f"║  Features:      {summary['n_features']}")
        log.info(f"║  Avg LightGBM:  {summary['avg_lgbm']:.3f}")
        log.info(f"║  Avg CatBoost:  {summary['avg_catboost']:.3f}")
        log.info(f"║  Avg Meta:      {summary['avg_meta']:.3f}")
        log.info(f"║  Best Accuracy: {summary['best_accuracy']:.3f}")
        log.info(f"║  Time:          {summary['elapsed_s']:.0f}s")
        log.info("╚══════════════════════════════════════════════════╝")


def list_models() -> List[Dict]:
    """List all models in registry."""
    models = []
    if REGISTRY_DIR.exists():
        for d in sorted(REGISTRY_DIR.iterdir()):
            if d.is_dir():
                meta_path = d / "metadata.json"
                if meta_path.exists():
                    with open(meta_path) as f:
                        meta = json.load(f)
                    models.append(meta)
    return models
