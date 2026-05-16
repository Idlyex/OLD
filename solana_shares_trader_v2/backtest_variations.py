"""Backtest v4 -- bar-level, pre-market lookback, per-bar calibration, save models.

Usage:  python backtest_variations.py --duration 5
"""
import sys, os, argparse, gc, time, warnings, json, joblib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")
from config import config
P = print

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=int, default=5)
    p.add_argument("--split", type=float, default=0.7)
    p.add_argument("--capital", type=float, default=100.0)
    p.add_argument("--bet-size", type=float, default=2.0, dest="bet_size")
    p.add_argument("--data", type=str, default=None)
    return p.parse_args()

def load_data(args):
    if args.data and os.path.exists(args.data):
        return pd.read_parquet(args.data)
    for p in ["data/processed/SOLUSDT_processed.parquet", "data/processed/SOLUSDT_1m.parquet"]:
        if os.path.exists(p):
            return pd.read_parquet(p)
    P("ERROR: No data found."); sys.exit(1)

def build_markets(sol_data, dur):
    sol_data = sol_data.sort_values("ts").reset_index(drop=True)
    ts = sol_data["ts"].values; interval_ms = dur * 60_000
    s, e = int(ts[0]), int(ts[-1])
    first = (s // interval_ms + 1) * interval_ms
    markets = []; cur = first
    while cur + interval_ms <= e:
        mask = (ts >= cur) & (ts < cur + interval_ms)
        idx = np.where(mask)[0]
        if len(idx) >= 3:
            ptb = float(sol_data.iloc[idx[0]]["close"])
            fin = float(sol_data.iloc[idx[-1]]["close"])
            markets.append({"start_ts": cur, "end_ts": cur + interval_ms,
                "ptb": ptb, "final_price": fin,
                "outcome": "UP" if fin >= ptb else "DOWN",
                "bar_indices": idx, "n_bars": len(idx)})
        cur += interval_ms
    return markets

def pick_by_bar(markets, timestamps, bar_idx):
    samples = []
    for mi, m in enumerate(markets):
        mask = (timestamps >= m["start_ts"]) & (timestamps < m["end_ts"])
        si = np.where(mask)[0]
        if len(si) == 0: continue
        samples.append((mi, si[min(bar_idx, len(si) - 1)]))
    return samples

def simulate(test_samples, markets, probs, conf_thr, ep, capital, bet):
    init = capital; wins = losses = 0; pnl_list = []; equity = [capital]
    up_w = up_t = dn_w = dn_t = ultra_w = ultra_t = 0
    for (mi, _), prob in zip(test_samples, probs):
        m = markets[mi]; dp = max(prob, 1 - prob)
        if dp < conf_thr or capital < bet: continue
        is_up = prob > 0.5
        won = (is_up and m["outcome"] == "UP") or (not is_up and m["outcome"] == "DOWN")
        b = min(bet, capital); capital -= b
        if won: capital += b / ep; pnl = b / ep - b; wins += 1
        else: pnl = -b; losses += 1
        pnl_list.append(pnl); equity.append(capital)
        if is_up: up_t += 1; up_w += int(won)
        else: dn_t += 1; dn_w += int(won)
        if dp >= 0.80: ultra_t += 1; ultra_w += int(won)
    n = wins + losses
    if n == 0: return None
    pa = np.array(pnl_list); eq = np.array(equity)
    pk = np.maximum.accumulate(eq)
    dd = ((eq - pk) / np.where(pk > 0, pk, 1) * 100).min()
    gp = pa[pa > 0].sum() if (pa > 0).any() else 0
    gl = abs(pa[pa < 0].sum()) if (pa < 0).any() else 0.01
    return {"n": n, "w": wins, "wr": wins/n*100, "pnl": pa.sum(),
        "ret": (capital/init-1)*100, "dd": dd,
        "sharpe": float(np.mean(pa)/max(np.std(pa),1e-9)*np.sqrt(252)),
        "pf": gp/max(gl,0.01),
        "up_wr": up_w/max(up_t,1)*100, "dn_wr": dn_w/max(dn_t,1)*100,
        "ultra_n": ultra_t, "ultra_wr": ultra_w/max(ultra_t,1)*100}

def calibration_row(probs, yt, bins):
    """Compute actual WR per confidence bin."""
    dir_probs = np.maximum(probs, 1 - probs)
    correct = ((probs > 0.5) == yt).astype(float)
    cells = []
    for lo, hi in bins:
        mask = (dir_probs >= lo) & (dir_probs < hi)
        cnt = mask.sum()
        if cnt >= 5:
            cells.append(f" {correct[mask].mean():>5.0%}({cnt:>3})")
        else:
            cells.append(f"     -     ")
    return "".join(cells)

def main():
    args = parse_args(); t0 = time.time()
    P("\n" + "=" * 90)
    P(f"  BACKTEST v4 -- {args.duration}m | bar-level | pre-market lookback | PER-BAR calibration")
    P("=" * 90)

    sol_data = load_data(args)
    P(f"  Data: {len(sol_data):,} bars")
    markets = build_markets(sol_data, args.duration)
    up_count = sum(1 for m in markets if m["outcome"] == "UP")
    P(f"  Markets: {len(markets)} ({up_count} UP = {up_count/len(markets)*100:.1f}%)")

    # Build features (now with pre-market lookback 2/5/10m)
    P(f"\n  Building features + pre-market lookback (2/5/10 min)...")
    from training.dataset import TrainingDataset
    ds = TrainingDataset()
    dataset = ds.build_shares_dataset(sol_data, duration_minutes=args.duration)
    X_all = dataset["X"]; y_all = dataset["y_direction"]
    ts_all = dataset["timestamps"]; feat_names = dataset["feature_names"]
    P(f"  Dataset: {X_all.shape[0]:,} x {X_all.shape[1]} features")

    # Check pre-market features are present
    pm_feats = [f for f in feat_names if f.startswith("pre_mkt_")]
    P(f"  Pre-market features: {pm_feats}")

    n_bars = args.duration
    bar_indices = list(range(n_bars))
    bar_sets = {}
    for bi in bar_indices:
        bar_sets[bi] = pick_by_bar(markets, ts_all, bi)
        P(f"  bar {bi} ({bi*100//n_bars:>2}%): {len(bar_sets[bi])} markets")

    # Split
    ref = bar_sets[1]
    split_pt = int(len(ref) * args.split)
    train_mkt_set = set(mi for mi, _ in ref[:split_pt])
    n_test = len(ref) - split_pt
    P(f"  Train: {len(train_mkt_set)} | Test: {n_test}")

    # Use ALL bars from train markets for training (more data!)
    train_idx = []
    for bi in bar_indices:
        for mi, si in bar_sets[bi]:
            if mi in train_mkt_set:
                train_idx.append(si)
    train_idx = sorted(set(train_idx))
    X_train = X_all[train_idx]; y_train = y_all[train_idx]
    P(f"  Training samples: {len(train_idx)} (all bars from train markets)")

    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)

    def get_test(bar):
        s = bar_sets[bar]
        test = [(mi, si) for mi, si in s if mi not in train_mkt_set]
        idx = [si for _, si in test]
        return test, scaler.transform(X_all[idx]), y_all[idx]

    test_data = {bi: get_test(bi) for bi in bar_indices}

    # =============================================
    # TRAIN MODELS (faster: reduced rounds)
    # =============================================
    P(f"\n  Training models...")
    import lightgbm as lgb
    models_by_bar = {bi: {} for bi in bar_indices}
    train_accs = {}
    saved_models = {}  # for saving to disk

    # 1) LGBM
    P("  [1/6] LGBM...")
    lgb_ds = lgb.Dataset(pd.DataFrame(X_train_s, columns=feat_names), label=y_train)
    lgb_m = lgb.train(
        {"objective": "binary", "metric": "binary_logloss", "num_leaves": 31,
         "learning_rate": 0.05, "feature_fraction": 0.8, "bagging_fraction": 0.8,
         "bagging_freq": 5, "verbose": -1, "n_jobs": -1},
        lgb_ds, num_boost_round=200, valid_sets=[lgb_ds], callbacks=[lgb.log_evaluation(0)])
    importance = lgb_m.feature_importance(importance_type="gain")
    top_feats = sorted(zip(feat_names, importance), key=lambda x: -x[1])[:15]
    tp = lgb_m.predict(X_train_s); train_accs["LGBM"] = np.mean((tp > 0.5) == y_train)
    saved_models["lgbm"] = lgb_m
    for bi in bar_indices:
        t, Xt, yt = test_data[bi]; models_by_bar[bi]["LGBM"] = (lgb_m.predict(Xt), yt)
    P(f"    train={train_accs['LGBM']:.3f} test@b1={np.mean((lgb_m.predict(test_data[1][1]) > 0.5) == test_data[1][2]):.3f}")
    del lgb_ds; gc.collect()

    # 2) CatBoost
    P("  [2/6] CatBoost...")
    try:
        from catboost import CatBoostClassifier
        cb = CatBoostClassifier(iterations=200, learning_rate=0.05, depth=6,
                                verbose=0, thread_count=-1, random_seed=42)
        cb.fit(X_train_s, y_train)
        train_accs["CatBoost"] = np.mean(cb.predict(X_train_s) == y_train)
        saved_models["catboost"] = cb
        for bi in bar_indices:
            t, Xt, yt = test_data[bi]; models_by_bar[bi]["CatBoost"] = (cb.predict_proba(Xt)[:, 1], yt)
        P(f"    train={train_accs['CatBoost']:.3f} test@b1={np.mean((cb.predict_proba(test_data[1][1])[:, 1] > 0.5) == test_data[1][2]):.3f}")
    except ImportError:
        P("    not installed")

    # 3) RF
    P("  [3/6] RandomForest...")
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=200, max_depth=12, n_jobs=-1, random_state=42)
    rf.fit(X_train_s, y_train)
    train_accs["RF"] = np.mean(rf.predict(X_train_s) == y_train)
    saved_models["rf"] = rf
    for bi in bar_indices:
        t, Xt, yt = test_data[bi]; models_by_bar[bi]["RF"] = (rf.predict_proba(Xt)[:, 1], yt)
    P(f"    train={train_accs['RF']:.3f} test@b1={np.mean((rf.predict_proba(test_data[1][1])[:, 1] > 0.5) == test_data[1][2]):.3f}")

    # 4) XGBoost
    P("  [4/6] XGBoost...")
    try:
        import xgboost as xgb
        xgb_m = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            verbosity=0, n_jobs=-1, random_state=42)
        xgb_m.fit(X_train_s, y_train)
        train_accs["XGBoost"] = np.mean(xgb_m.predict(X_train_s) == y_train)
        saved_models["xgboost"] = xgb_m
        for bi in bar_indices:
            t, Xt, yt = test_data[bi]; models_by_bar[bi]["XGBoost"] = (xgb_m.predict_proba(Xt)[:, 1], yt)
        P(f"    train={train_accs['XGBoost']:.3f} test@b1={np.mean((xgb_m.predict_proba(test_data[1][1])[:, 1] > 0.5) == test_data[1][2]):.3f}")
    except ImportError:
        P("    not installed")

    # 5) Ensembles
    P("  [5/6] Ensembles...")
    for bi in bar_indices:
        pd_map = {k: v[0] for k, v in models_by_bar[bi].items()}
        _, _, yt = test_data[bi]
        if "LGBM" in pd_map and "CatBoost" in pd_map:
            models_by_bar[bi]["LGBM+CB"] = ((pd_map["LGBM"] + pd_map["CatBoost"]) / 2, yt)
        if all(k in pd_map for k in ["LGBM", "CatBoost", "RF"]):
            models_by_bar[bi]["LGBM+CB+RF"] = ((pd_map["LGBM"] + pd_map["CatBoost"] + pd_map["RF"]) / 3, yt)
        if "LGBM" in pd_map and "XGBoost" in pd_map:
            models_by_bar[bi]["LGBM+XGB"] = ((pd_map["LGBM"] + pd_map["XGBoost"]) / 2, yt)
        all_p = list(pd_map.values())
        if len(all_p) >= 3:
            models_by_bar[bi]["ALL-Ens"] = (np.mean(all_p, axis=0), yt)

    # 6) SAVE ALL MODELS for live use
    P("  [6/6] Saving models for live trading...")
    model_dir = Path("training/model_registry/latest")
    model_dir.mkdir(parents=True, exist_ok=True)
    for name, model in saved_models.items():
        joblib.dump(model, model_dir / f"{name}_cls.pkl")
    joblib.dump(scaler, model_dir / "scaler.pkl")
    meta = {"feature_names": feat_names, "models": list(saved_models.keys()),
            "trained_on": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_train": len(train_idx), "n_features": len(feat_names)}
    with open(model_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    P(f"    Saved {len(saved_models)} models + scaler + meta to {model_dir}")

    del X_all, X_train, X_train_s; gc.collect()

    all_model_names = sorted(set().union(*[set(models_by_bar[bi].keys()) for bi in bar_indices]))

    # =============================================
    # SECTION 1: OVERFITTING CHECK
    # =============================================
    P("\n" + "=" * 70)
    P("  SECTION 1: OVERFITTING CHECK")
    P("=" * 70)
    P(f"  {'Model':<16} {'Train':>8} {'Test@b1':>8} {'Gap':>8} {'Status':>10}")
    P("  " + "-" * 55)
    for mn in all_model_names:
        if mn not in train_accs: continue
        ta = train_accs[mn]
        pr, yt = models_by_bar[1][mn] if mn in models_by_bar[1] else (np.array([]), np.array([]))
        te = np.mean((pr > 0.5) == yt) if len(pr) > 0 else 0
        gap = ta - te
        flag = "YES!!!" if gap > 0.15 else ("maybe" if gap > 0.08 else "OK")
        P(f"  {mn:<16} {ta:>7.1%} {te:>7.1%} {gap:>+7.1%} {flag:>10}")

    # =============================================
    # SECTION 2: ACCURACY BY BAR
    # =============================================
    P("\n" + "=" * 80)
    P(f"  SECTION 2: ACCURACY BY BAR (each bar = 1 min)")
    P("=" * 80)
    hdr = f"  {'Model':<16}" + "".join(f"{'b'+str(b)+'('+str(b)+'m)':>10}" for b in bar_indices)
    P(hdr); P("  " + "-" * (16 + 10 * len(bar_indices)))
    for mn in all_model_names:
        row = f"  {mn:<16}"
        for bi in bar_indices:
            if mn in models_by_bar[bi]:
                pr, yt = models_by_bar[bi][mn]
                row += f"{np.mean((pr > 0.5) == yt):>9.1%} "
            else:
                row += "        - "
        P(row)

    # =============================================
    # SECTION 3: PER-BAR CALIBRATION (THE KEY TABLE)
    # =============================================
    bins = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70),
            (0.70, 0.75), (0.75, 0.80), (0.80, 0.85), (0.85, 1.01)]
    bin_labels = "0.50-55  0.55-60  0.60-65  0.65-70  0.70-75  0.75-80  0.80-85   0.85+"

    for bi in bar_indices:
        P(f"\n" + "=" * 110)
        P(f"  CALIBRATION @ BAR {bi} ({bi} min into market, {bi*100//n_bars}% elapsed)")
        P("=" * 110)
        P(f"  {'Model':<16} {bin_labels}")
        P("  " + "-" * 100)
        for mn in all_model_names:
            if mn not in models_by_bar[bi]: continue
            pr, yt = models_by_bar[bi][mn]
            P(f"  {mn:<16}{calibration_row(pr, yt, bins)}")

    # =============================================
    # RUN VARIATIONS (fewer: skip bar 3,4 for simulations — too late)
    # =============================================
    conf_thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    share_prices = [0.45, 0.50, 0.55, 0.60]
    all_results = []

    for bi in bar_indices:
        for mn in all_model_names:
            if mn not in models_by_bar[bi]: continue
            probs, _ = models_by_bar[bi][mn]
            test_samples = [(mi, si) for mi, si in bar_sets[bi] if mi not in train_mkt_set]
            for conf in conf_thresholds:
                for sp in share_prices:
                    r = simulate(test_samples, markets, probs, conf, sp, args.capital, args.bet_size)
                    if r is None: continue
                    r["model"] = mn; r["bar"] = bi; r["conf"] = conf; r["sp"] = sp
                    all_results.append(r)

    P(f"\n  Total variations: {len(all_results)}")

    # =============================================
    # SECTION 4: TOP 20 BY SHARPE (bar 0-2 only = realistic)
    # =============================================
    P("\n" + "=" * 115)
    P("  SECTION 4: TOP 20 BY SHARPE (bar 0-2, realistic entry)")
    P("=" * 115)
    by_s = sorted([r for r in all_results if r["bar"] <= 2], key=lambda x: -x["sharpe"])[:20]
    P(f"  {'#':>3} {'Model':<16} {'Bar':>4} {'Conf':>5} {'SP':>5} {'Trd':>5} {'WR':>6} {'PnL':>9} {'DD':>6} {'Shrp':>6} {'PF':>5}")
    P("  " + "-" * 100)
    for i, r in enumerate(by_s):
        P(f"  {i+1:>3} {r['model']:<16} b{r['bar']}   {r['conf']:>5.2f} ${r['sp']:.2f} "
          f"{r['n']:>5} {r['wr']:>5.1f}% ${r['pnl']:>+7.0f} {r['dd']:>5.1f}% {r['sharpe']:>5.1f} {r['pf']:>4.1f}")

    # =============================================
    # SECTION 5: HEAD-TO-HEAD per bar (conf=0.70, SP=$0.50)
    # =============================================
    for bi in [0, 1, 2]:
        P(f"\n  --- HEAD-TO-HEAD @ bar {bi} ({bi}m, conf=0.70, SP=$0.50) ---")
        P(f"  {'Model':<16} {'Trd':>5} {'WR':>6} {'PnL':>9} {'DD':>6} {'Shrp':>6} {'PF':>5}")
        P("  " + "-" * 55)
        h2h = sorted([r for r in all_results if r["bar"]==bi and r["conf"]==0.70 and r["sp"]==0.50],
                     key=lambda x: -x["sharpe"])
        for r in h2h:
            P(f"  {r['model']:<16} {r['n']:>5} {r['wr']:>5.1f}% ${r['pnl']:>+7.0f} {r['dd']:>5.1f}% {r['sharpe']:>5.1f} {r['pf']:>4.1f}")

    # =============================================
    # SECTION 6: FEATURES
    # =============================================
    P("\n" + "=" * 60)
    P("  SECTION 6: TOP 15 FEATURES (LightGBM gain)")
    P("=" * 60)
    for i, (fn, fv) in enumerate(top_feats):
        bar = "#" * int(fv / max(top_feats[0][1], 1) * 30)
        P(f"  {i+1:>2}. {fn:<35} {fv:>6.0f} {bar}")

    # =============================================
    # FINAL SUMMARY
    # =============================================
    P("\n" + "=" * 90)
    P("  FINAL SUMMARY")
    P("=" * 90)
    real = [r for r in all_results if r["bar"] <= 2 and r["sp"] >= 0.50 and r["conf"] >= 0.65 and r["n"] >= 20]
    best_s = max(all_results, key=lambda x: x["sharpe"])
    best_p = max(all_results, key=lambda x: x["pnl"])
    wr50 = [r for r in all_results if r["n"] >= 50]
    best_w = max(wr50, key=lambda x: x["wr"]) if wr50 else None
    best_r = max(real, key=lambda x: x["sharpe"]) if real else None

    P(f"\n  Best Sharpe:    {best_s['model']} b{best_s['bar']} conf={best_s['conf']} SP=${best_s['sp']:.2f} "
      f"-> {best_s['n']}t {best_s['wr']:.1f}% Sharpe={best_s['sharpe']:.1f}")
    P(f"  Best PnL:       {best_p['model']} b{best_p['bar']} conf={best_p['conf']} SP=${best_p['sp']:.2f} "
      f"-> {best_p['n']}t ${best_p['pnl']:+.0f}")
    if best_w:
        P(f"  Best WR (50+):  {best_w['model']} b{best_w['bar']} conf={best_w['conf']} SP=${best_w['sp']:.2f} "
          f"-> {best_w['n']}t {best_w['wr']:.1f}%")
    if best_r:
        P(f"  Best Realistic: {best_r['model']} b{best_r['bar']} conf={best_r['conf']} SP=${best_r['sp']:.2f} "
          f"-> {best_r['n']}t {best_r['wr']:.1f}% ${best_r['pnl']:+.0f} Sharpe={best_r['sharpe']:.1f}")

    P(f"\n  Models saved to: {model_dir}")
    P(f"  Saved models: {list(saved_models.keys())}")

    # Save JSON
    Path("results").mkdir(exist_ok=True)
    report = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
              "duration_m": args.duration, "n_markets": len(markets), "n_test": n_test,
              "n_features": len(feat_names), "pre_market_features": pm_feats,
              "train_accs": {k: round(v, 4) for k, v in train_accs.items()},
              "saved_models": list(saved_models.keys()),
              "features_top15": [(f, round(v, 1)) for f, v in top_feats]}
    Path("results/backtest_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    P(f"  Report: results/backtest_report.json")
    P(f"  Time: {time.time()-t0:.1f}s")
    P("=" * 90)

if __name__ == "__main__":
    main()
