"""Optuna Hyperparameter Tuner — Full Pipeline.

Tunes LightGBM, CatBoost, XGBoost via Optuna on the training split,
then retrains with the best configs on full 70/30 split and reports.

Usage:
    # Called from run_pipeline.py --tune
    # Or standalone:
    python -m training.optuna_tuner
"""

import sys
import os
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.utils.logger import log


RESULTS_DIR = Path("results/pipeline")
MODELS_DIR = Path("training/model_registry")


def _tune_lgbm(X_train, y_train, n_trials: int = 80) -> Dict:
    """Tune LightGBM with Optuna."""
    import optuna
    import lightgbm as lgb
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Use TimeSeriesSplit for CV inside training data (no test data leakage)
    tscv = TimeSeriesSplit(n_splits=3)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
            "verbose": -1, "random_state": 42, "class_weight": "balanced", "n_jobs": -1,
        }

        scores = []
        for train_idx, val_idx in tscv.split(X_train):
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train[train_idx], y_train[train_idx])
            preds = model.predict(X_train[val_idx])
            scores.append(accuracy_score(y_train[val_idx], preds))

        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", study_name="lgbm_tune")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, n_jobs=1)

    log.info(f"  LGBM best CV: {study.best_value:.4f}")
    return study.best_params


def _tune_catboost(X_train, y_train, n_trials: int = 60) -> Dict:
    """Tune CatBoost with Optuna."""
    import optuna
    from catboost import CatBoostClassifier
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    tscv = TimeSeriesSplit(n_splits=3)

    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 200, 1200),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "depth": trial.suggest_int("depth", 3, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 10.0, log=True),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 10.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
            "verbose": 0, "random_seed": 42, "auto_class_weights": "Balanced",
            "thread_count": -1,
        }

        scores = []
        for train_idx, val_idx in tscv.split(X_train):
            model = CatBoostClassifier(**params)
            model.fit(X_train[train_idx], y_train[train_idx])
            preds = model.predict(X_train[val_idx]).flatten().astype(int)
            scores.append(accuracy_score(y_train[val_idx], preds))

        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", study_name="catboost_tune")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, n_jobs=1)

    log.info(f"  CatBoost best CV: {study.best_value:.4f}")
    return study.best_params


def _tune_xgboost(X_train, y_train, n_trials: int = 80) -> Dict:
    """Tune XGBoost with Optuna."""
    import optuna
    import xgboost as xgb
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    tscv = TimeSeriesSplit(n_splits=3)
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    spw = n_neg / max(n_pos, 1)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 100),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight": spw,
            "verbosity": 0, "random_state": 42, "n_jobs": -1,
            "eval_metric": "logloss", "tree_method": "hist",
        }

        scores = []
        for train_idx, val_idx in tscv.split(X_train):
            model = xgb.XGBClassifier(**params)
            model.fit(X_train[train_idx], y_train[train_idx], verbose=False)
            preds = model.predict(X_train[val_idx])
            scores.append(accuracy_score(y_train[val_idx], preds))

        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", study_name="xgboost_tune")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, n_jobs=1)

    log.info(f"  XGBoost best CV: {study.best_value:.4f}")
    return study.best_params


def retrain_with_best_params(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
    best_params: Dict[str, Dict],
    ds: Dict = None,
) -> Dict:
    """Retrain all models with tuned params, evaluate honestly on test set."""
    from run_pipeline import evaluate_model_honest

    results = {}

    for model_name, params in best_params.items():
        log.info(f"  Retraining {model_name} with tuned params...")
        try:
            if model_name == "lgbm":
                import lightgbm as lgb
                p = {**params, "verbose": -1, "random_state": 42, "class_weight": "balanced", "n_jobs": -1}
                model = lgb.LGBMClassifier(**p)
                X_tr_df = pd.DataFrame(X_train, columns=feature_names)
                X_te_df = pd.DataFrame(X_test, columns=feature_names)
                model.fit(
                    X_tr_df, y_train,
                    eval_set=[(X_te_df, y_test)],
                    callbacks=[lgb.log_evaluation(0), lgb.early_stopping(50, verbose=False)],
                )
                proba = model.predict_proba(X_te_df)[:, 1]

            elif model_name == "catboost":
                from catboost import CatBoostClassifier
                p = {**params, "verbose": 0, "random_seed": 42, "auto_class_weights": "Balanced", "thread_count": -1}
                model = CatBoostClassifier(**p)
                model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)
                proba = model.predict_proba(X_test)[:, 1]

            elif model_name == "xgboost":
                import xgboost as xgb
                n_pos = max(np.sum(y_train == 1), 1)
                n_neg = max(np.sum(y_train == 0), 1)
                p = {
                    **params, "verbosity": 0, "random_state": 42, "n_jobs": -1,
                    "eval_metric": "logloss", "tree_method": "hist",
                    "scale_pos_weight": n_neg / n_pos,
                }
                model = xgb.XGBClassifier(**p)
                model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
                proba = model.predict_proba(X_test)[:, 1]
            else:
                continue

            # Honest evaluation
            eval_ds = ds if ds is not None else {"y_test": y_test}
            honest = evaluate_model_honest(proba, y_test, eval_ds, model_name)
            honest["params"] = params
            honest["model"] = model

            results[model_name] = honest

            log.info(f"    {model_name}: overall_acc={honest['overall_accuracy']:.4f} logloss={honest['log_loss']:.4f}")
            ea = honest.get("entry_bar_accuracy", {})
            if ea:
                log.info(f"      entry-bar: {ea['accuracy']:.4f} ({ea['n_markets']} markets)")
            pa = honest.get("phase_accuracy", {})
            for pn, pv in pa.items():
                log.info(f"      {pn}: {pv['accuracy']:.4f} ({pv['n_samples']} samples)")

        except Exception as e:
            log.error(f"    {model_name} retrain failed: {e}")
            import traceback
            traceback.print_exc()

    # Ensemble
    probas = []
    for mn in ["lgbm", "catboost", "xgboost"]:
        if mn in results and "model" in results[mn]:
            m = results[mn]["model"]
            if mn == "lgbm":
                p = m.predict_proba(pd.DataFrame(X_test, columns=feature_names))[:, 1]
            else:
                p = m.predict_proba(X_test)[:, 1]
            probas.append(p)

    if probas:
        ens_proba = np.mean(probas, axis=0)
        eval_ds = ds if ds is not None else {"y_test": y_test}
        ens_result = evaluate_model_honest(ens_proba, y_test, eval_ds, "ensemble_tuned")
        results["ensemble_tuned"] = ens_result
        log.info(f"    ENSEMBLE_TUNED: overall_acc={ens_result['overall_accuracy']:.4f} logloss={ens_result['log_loss']:.4f}")
        ea = ens_result.get("entry_bar_accuracy", {})
        if ea:
            log.info(f"      entry-bar: {ea['accuracy']:.4f} ({ea['n_markets']} markets)")

    return results


def run_optuna_pipeline(datasets: Dict, base_tag: str):
    """Full Optuna pipeline: tune → retrain → save → report."""
    import joblib

    t0 = time.perf_counter()

    log.info("\n╔══════════════════════════════════════════════════════════╗")
    log.info("║  OPTUNA HYPERPARAMETER TUNING PIPELINE                 ║")
    log.info("╚══════════════════════════════════════════════════════════╝")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for tf_key, ds in datasets.items():
        if isinstance(ds.get("X_train"), np.ndarray) and ds["X_train"].size == 0:
            continue

        log.info(f"\n{'═'*60}")
        log.info(f"  TUNING {tf_key.upper()} MARKETS")
        log.info(f"{'═'*60}")

        feature_names = ds["feature_names"]
        X_train = ds["X_train"]
        X_test = ds["X_test"]
        y_train = ds["y_train"]
        y_test = ds["y_test"]

        scaler = RobustScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Phase 1: Tune
        log.info("\n  Phase 1: Tuning hyperparameters...")
        t_tune = time.perf_counter()

        best_params = {}

        log.info("  ── Tuning LightGBM (80 trials) ──")
        best_params["lgbm"] = _tune_lgbm(X_train_s, y_train, n_trials=80)

        log.info("  ── Tuning CatBoost (60 trials) ──")
        best_params["catboost"] = _tune_catboost(X_train_s, y_train, n_trials=60)

        try:
            log.info("  ── Tuning XGBoost (80 trials) ──")
            best_params["xgboost"] = _tune_xgboost(X_train_s, y_train, n_trials=80)
        except ImportError:
            log.warning("  XGBoost not available, skipping")

        tune_elapsed = time.perf_counter() - t_tune
        log.info(f"\n  Tuning done in {tune_elapsed:.0f}s")

        # Save best params
        params_path = RESULTS_DIR / f"best_params_{base_tag}_{tf_key}.json"
        with open(params_path, "w") as f:
            json.dump(best_params, f, indent=2)
        log.info(f"  📊 Best params → {params_path}")

        # Phase 2: Retrain with best params on 70/30 split
        log.info("\n  Phase 2: Retraining with tuned params on 70/30...")
        tuned_results = retrain_with_best_params(
            X_train_s, y_train, X_test_s, y_test, feature_names, best_params, ds=ds,
        )

        # Phase 3: Save tuned models
        model_dir = MODELS_DIR / f"tuned_{base_tag}_{tf_key}"
        model_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(scaler, model_dir / "scaler.pkl")
        with open(model_dir / "meta.json", "w") as f:
            json.dump({
                "feature_names": feature_names,
                "best_params": best_params,
                "tag": f"tuned_{base_tag}_{tf_key}",
            }, f, indent=2)

        for mn, mr in tuned_results.items():
            if "model" in mr:
                joblib.dump(mr["model"], model_dir / f"{mn}_cls.pkl")

        # Also save as "latest" for live use
        latest_dir = MODELS_DIR / "latest"
        latest_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, latest_dir / "scaler.pkl")

        best_model_name = max(
            [(mn, mr.get("overall_accuracy", mr.get("accuracy", 0))) for mn, mr in tuned_results.items() if "overall_accuracy" in mr or "accuracy" in mr],
            key=lambda x: x[1],
        )[0]
        if "model" in tuned_results[best_model_name]:
            joblib.dump(tuned_results[best_model_name]["model"], latest_dir / f"{best_model_name}_cls.pkl")

        with open(latest_dir / "meta.json", "w") as f:
            json.dump({
                "feature_names": feature_names,
                "models": [best_model_name],
                "best_params": best_params.get(best_model_name, {}),
                "source": f"tuned_{base_tag}_{tf_key}",
            }, f, indent=2)

        log.info(f"  💾 Tuned models → {model_dir}")
        log.info(f"  💾 Best model ({best_model_name}) → {latest_dir}")

        # Clean results for JSON
        clean_results = {}
        for mn, mr in tuned_results.items():
            clean_results[mn] = {k: v for k, v in mr.items() if k != "model"}
        all_results[tf_key] = {
            "best_params": best_params,
            "tuned_results": clean_results,
            "tune_time_s": round(tune_elapsed, 1),
        }

    # Save full Optuna report
    report_path = RESULTS_DIR / f"optuna_report_{base_tag}.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary
    elapsed = time.perf_counter() - t0
    log.info("\n╔══════════════════════════════════════════════════════════╗")
    log.info("║          OPTUNA TUNING REPORT                          ║")
    log.info("╚══════════════════════════════════════════════════════════╝")

    for tf_key, tf_data in all_results.items():
        log.info(f"\n  ═══ {tf_key.upper()} ═══")
        log.info(f"  Tune time: {tf_data['tune_time_s']:.0f}s")
        for mn, mr in tf_data.get("tuned_results", {}).items():
            acc = mr.get("overall_accuracy", mr.get("accuracy", 0))
            ll = mr.get("log_loss", 0)
            log.info(f"  {mn:>15}: overall_acc={acc:.4f} logloss={ll:.4f}")
            ea = mr.get("entry_bar_accuracy", {})
            if ea:
                log.info(f"    entry-bar: {ea['accuracy']:.4f} ({ea['n_markets']} markets)")
            pa = mr.get("phase_accuracy", {})
            for pn, pv in pa.items():
                log.info(f"    {pn}: {pv['accuracy']:.4f} ({pv['n_samples']} samples)")

    log.info(f"\n  📊 Full report → {report_path}")
    log.info(f"  ⏱ Total Optuna time: {elapsed:.0f}s")

    return all_results


# ═══════════════════════════════════════════════════════════
#  STANDALONE
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    from training.fast_dataset import FastDatasetBuilder

    log.info("Running Optuna tuner standalone...")

    data_path = Path("data/processed/SOLUSDT_processed.parquet")
    if not data_path.exists():
        log.error(f"No data at {data_path}. Run pipeline first.")
        sys.exit(1)

    df = pd.read_parquet(data_path)
    log.info(f"Loaded {len(df):,} rows")

    builder = FastDatasetBuilder()
    datasets = builder.build_both_timeframes(df, train_ratio=0.7)

    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_optuna_pipeline(datasets, tag)
