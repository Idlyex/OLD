"""Walk-Forward Optimization — rolling window model evaluation.

Simulates real deployment conditions by training on past data
and testing on unseen future data in a rolling fashion.

Outputs per-fold metrics + aggregated statistics.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path

from training.dataset import TrainingDataset, WalkForwardSplit
from training.train import Trainer
from core.utils.logger import log
from config import config


class WalkForwardOptimizer:
    """Rolling-window walk-forward model validation."""

    def __init__(
        self,
        train_size: int = 5000,
        test_size: int = 1000,
        step: int = 500,
        forward_minutes: int = 5,
    ):
        self.train_size = train_size
        self.test_size = test_size
        self.step = step
        self.forward_minutes = forward_minutes
        self.results: List[Dict] = []

    def run(self, data: pd.DataFrame) -> Dict:
        """Run walk-forward optimization on processed data.

        Args:
            data: processed OHLCV DataFrame

        Returns:
            Aggregated results dict
        """
        t0 = time.time()
        log.info("╔══════════════════════════════════════════════════╗")
        log.info("║  Walk-Forward Optimization                      ║")
        log.info(f"║  train={self.train_size} test={self.test_size} step={self.step}  ║")
        log.info("╚══════════════════════════════════════════════════╝")

        # Build full dataset first
        ds = TrainingDataset()
        dataset = ds.build_from_dataframe(data, self.forward_minutes)
        X = dataset["X"]
        y_dir = dataset["y_direction"]
        y_ret = dataset["y_return"]
        feature_names = dataset["feature_names"]

        # Feature scaling per-fold
        from sklearn.preprocessing import RobustScaler

        # Generate splits
        wf = WalkForwardSplit(
            train_size=self.train_size,
            test_size=self.test_size,
            step=self.step,
        )
        splits = wf.split(len(X))

        if not splits:
            log.error("Not enough data for walk-forward splits")
            return {"error": "insufficient data"}

        fold_results = []
        cumulative_pnl = []

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            log.info(f"\n─── WF Fold {fold_idx + 1}/{len(splits)} ───")

            # Scale per-fold (only fit on train)
            scaler = RobustScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])
            y_train = y_dir[train_idx]
            y_test = y_dir[test_idx]
            y_train_ret = y_ret[train_idx]
            y_test_ret = y_ret[test_idx]

            # Train models
            lgbm_model, cb_model = None, None

            try:
                import lightgbm as lgb

                lgbm_cfg = config.get("models", {}).get("primary", {}).get("lgbm", {})
                lgbm_model = lgb.LGBMClassifier(
                    n_estimators=lgbm_cfg.get("n_estimators", 300),
                    learning_rate=lgbm_cfg.get("learning_rate", 0.05),
                    max_depth=lgbm_cfg.get("max_depth", 7),
                    num_leaves=lgbm_cfg.get("num_leaves", 63),
                    subsample=lgbm_cfg.get("subsample", 0.8),
                    colsample_bytree=lgbm_cfg.get("colsample_bytree", 0.8),
                    verbose=-1,
                    random_state=42,
                    class_weight="balanced",
                )
                lgbm_model.fit(X_train, y_train)
            except Exception as e:
                log.warning(f"  LightGBM failed: {e}")

            try:
                from catboost import CatBoostClassifier

                cb_cfg = config.get("models", {}).get("primary", {}).get("catboost", {})
                cb_model = CatBoostClassifier(
                    iterations=cb_cfg.get("iterations", 300),
                    learning_rate=cb_cfg.get("learning_rate", 0.05),
                    depth=cb_cfg.get("depth", 7),
                    verbose=0,
                    random_seed=42,
                    auto_class_weights="Balanced",
                )
                cb_model.fit(X_train, y_train)
            except Exception as e:
                log.warning(f"  CatBoost failed: {e}")

            # Evaluate
            lgbm_proba = lgbm_model.predict_proba(X_test)[:, 1] if lgbm_model else np.full(len(X_test), 0.5)
            cb_proba = cb_model.predict_proba(X_test)[:, 1] if cb_model else np.full(len(X_test), 0.5)

            # Ensemble average
            ensemble_proba = (lgbm_proba + cb_proba) / 2
            ensemble_preds = (ensemble_proba > 0.5).astype(int)

            accuracy = float(np.mean(ensemble_preds == y_test))

            # Simulate PnL: trade only when confidence > threshold
            threshold = 0.55
            high_conf_mask = np.abs(ensemble_proba - 0.5) > (threshold - 0.5)
            if np.sum(high_conf_mask) > 0:
                traded_preds = ensemble_preds[high_conf_mask]
                traded_actual_dir = y_test[high_conf_mask]
                traded_returns = y_test_ret[high_conf_mask]
                trade_accuracy = float(np.mean(traded_preds == traded_actual_dir))

                # PnL: correct direction → capture return, wrong → lose return
                pnl_per_trade = np.where(
                    traded_preds == traded_actual_dir,
                    np.abs(traded_returns),
                    -np.abs(traded_returns),
                )
                total_pnl = float(np.sum(pnl_per_trade))
                n_trades = int(np.sum(high_conf_mask))
            else:
                trade_accuracy = 0
                total_pnl = 0
                n_trades = 0

            fold_result = {
                "fold": fold_idx,
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "accuracy": accuracy,
                "trade_accuracy": trade_accuracy,
                "n_trades": n_trades,
                "pnl": total_pnl,
                "lgbm_acc": float(np.mean((lgbm_proba > 0.5).astype(int) == y_test)) if lgbm_model else 0,
                "cb_acc": float(np.mean((cb_proba > 0.5).astype(int) == y_test)) if cb_model else 0,
            }
            fold_results.append(fold_result)
            cumulative_pnl.append(total_pnl)

            log.info(
                f"  acc={accuracy:.3f} trade_acc={trade_accuracy:.3f} "
                f"trades={n_trades} pnl={total_pnl:+.5f} "
                f"cum_pnl={sum(cumulative_pnl):+.5f}"
            )

        # Aggregate
        elapsed = time.time() - t0
        summary = {
            "folds": len(splits),
            "fold_results": fold_results,
            "avg_accuracy": float(np.mean([r["accuracy"] for r in fold_results])),
            "avg_trade_accuracy": float(np.mean([r["trade_accuracy"] for r in fold_results if r["n_trades"] > 0])) if any(r["n_trades"] > 0 for r in fold_results) else 0,
            "total_trades": sum(r["n_trades"] for r in fold_results),
            "total_pnl": sum(r["pnl"] for r in fold_results),
            "avg_pnl_per_fold": float(np.mean([r["pnl"] for r in fold_results])),
            "pnl_sharpe": float(np.mean(cumulative_pnl) / (np.std(cumulative_pnl) + 1e-10)),
            "elapsed_s": elapsed,
        }

        self._print_summary(summary)
        return summary

    def _print_summary(self, s: Dict):
        log.info("\n╔══════════════════════════════════════════════════╗")
        log.info("║       WALK-FORWARD RESULTS                      ║")
        log.info("╠══════════════════════════════════════════════════╣")
        log.info(f"║  Folds:           {s['folds']}")
        log.info(f"║  Avg Accuracy:    {s['avg_accuracy']:.3f}")
        log.info(f"║  Avg Trade Acc:   {s['avg_trade_accuracy']:.3f}")
        log.info(f"║  Total Trades:    {s['total_trades']}")
        log.info(f"║  Total PnL:       {s['total_pnl']:+.5f}")
        log.info(f"║  PnL Sharpe:      {s['pnl_sharpe']:.2f}")
        log.info(f"║  Time:            {s['elapsed_s']:.0f}s")
        log.info("╚══════════════════════════════════════════════════╝")
