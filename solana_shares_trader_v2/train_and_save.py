"""Train LGBM model and save for live trading.

Trains on ALL available data (no split — for live use, we want max data).
Saves model, scaler, and feature names to training/model_registry/latest/

Usage:
  python train_and_save.py
  python train_and_save.py --duration 5
  python train_and_save.py --duration 15
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.preprocessing import RobustScaler

from config import config
from core.utils.logger import log
from training.dataset import TrainingDataset


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=int, default=5, help="Market duration (5 or 15)")
    p.add_argument("--data", type=str, default=None)
    args = p.parse_args()

    # Load data
    data_path = args.data or "data/processed/SOLUSDT_processed.parquet"
    if not Path(data_path).exists():
        log.error(f"No data at {data_path}. Run: python main.py --mode download --days 10")
        sys.exit(1)

    sol_data = pd.read_parquet(data_path)
    log.info(f"Loaded {len(sol_data):,} bars from {data_path}")

    # Build dataset
    ds = TrainingDataset()
    dataset = ds.build_shares_dataset(sol_data, duration_minutes=args.duration)

    X = dataset["X"]
    y = dataset["y_direction"]
    feature_names = dataset["feature_names"]

    if X.size == 0:
        log.error("Empty dataset")
        sys.exit(1)

    log.info(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    log.info(f"Direction balance: {np.mean(y)*100:.1f}% UP")

    # Scale
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # Train
    lgbm_ds = lgb.Dataset(
        pd.DataFrame(X_scaled, columns=feature_names),
        label=y,
    )
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "n_jobs": -1,
    }
    model = lgb.train(
        params, lgbm_ds,
        num_boost_round=300,
        callbacks=[lgb.log_evaluation(0)],
    )

    # Accuracy
    preds = model.predict(X_scaled)
    acc = np.mean((preds > 0.5) == y)
    log.info(f"Train accuracy: {acc:.3f}")

    # Save
    out_dir = Path("training/model_registry/latest")
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, out_dir / "lgbm_cls.pkl")
    joblib.dump(scaler, out_dir / "scaler.pkl")
    with open(out_dir / "meta.json", "w") as f:
        json.dump({
            "feature_names": feature_names,
            "duration_minutes": args.duration,
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "train_accuracy": float(acc),
            "direction_balance": float(np.mean(y)),
        }, f, indent=2)

    log.info(f"\n✅ Model saved to {out_dir}/")
    log.info(f"   lgbm_cls.pkl  — LightGBM classifier")
    log.info(f"   scaler.pkl    — RobustScaler")
    log.info(f"   meta.json     — {len(feature_names)} feature names + metadata")

    # Feature importance
    importance = model.feature_importance(importance_type="gain")
    top = sorted(zip(feature_names, importance), key=lambda x: -x[1])[:10]
    log.info(f"\n  Top features:")
    for name, val in top:
        log.info(f"    {name:<35} {val:.0f}")


if __name__ == "__main__":
    main()
