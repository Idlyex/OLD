"""MEGA BACKTEST v5 -- cached features, 5m+15m, window 30/60, all models,
per-bar timing (0/20/40/60/80%), consensus, gap, direction, share prices,
Optuna tuning, pre-market ML, EV calculations, 70/30 time-split.

Usage:  python backtest_v5.py
        python backtest_v5.py --rebuild-cache
        python backtest_v5.py --optuna
"""
import sys, os, hashlib, gc, time, json, warnings, argparse, joblib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
warnings.filterwarnings("ignore")
P = print

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rebuild-cache", action="store_true", dest="rebuild")
    p.add_argument("--optuna", action="store_true", help="Run Optuna hyperparameter search")
    p.add_argument("--split", type=float, default=0.7)
    p.add_argument("--capital", type=float, default=100.0)
    p.add_argument("--bet", type=float, default=2.0)
    return p.parse_args()

def load_data():
    for p in ["data/processed/SOLUSDT_processed.parquet","data/processed/SOLUSDT_1m.parquet"]:
        if os.path.exists(p): return pd.read_parquet(p), p
    P("ERROR: No data"); sys.exit(1)

def build_markets(sol_data, dur):
    ts = sol_data["ts"].values; ims = dur * 60_000
    s, e = int(ts[0]), int(ts[-1]); first = (s // ims + 1) * ims
    mkts = []; cur = first
    while cur + ims <= e:
        mask = (ts >= cur) & (ts < cur + ims); idx = np.where(mask)[0]
        if len(idx) >= 3:
            ptb = float(sol_data.iloc[idx[0]]["close"])
            fin = float(sol_data.iloc[idx[-1]]["close"])
            mkts.append({"start_ts": cur, "end_ts": cur + ims, "ptb": ptb,
                "final": fin, "out": "UP" if fin >= ptb else "DOWN",
                "gap_pct": (fin - ptb) / ptb * 100, "bars": idx})
        cur += ims
    return mkts

CACHE_VER = "v5b"  # bump when feature set changes
def cache_path(data_path, dur, win):
    h = hashlib.md5(f"{CACHE_VER}_{data_path}_{dur}_{win}_{os.path.getmtime(data_path)}".encode()).hexdigest()[:10]
    return Path(f"data/cache/feat_{dur}m_w{win}_{h}.npz")

def build_or_load(sol_data, dur, win, data_path, rebuild):
    cp = cache_path(data_path, dur, win)
    if cp.exists() and not rebuild:
        P(f"    [cache] Loading {cp.name}...")
        d = np.load(cp, allow_pickle=True)
        return d["X"], d["y"], d["ts"], list(d["fn"])
    P(f"    [build] {dur}m window={win}...")
    from training.dataset import TrainingDataset
    ds = TrainingDataset()
    ds._feature_names = None
    r = ds.build_shares_dataset(sol_data, duration_minutes=dur,
                                 min_history=max(win, 60))
    X, y, ts_arr, fn = r["X"], r["y_direction"], r["timestamps"], r["feature_names"]
    cp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cp, X=X, y=y, ts=ts_arr, fn=np.array(fn, dtype=object))
    P(f"    [cache] Saved {cp.name} ({X.shape[0]}x{X.shape[1]})")
    return X, y, ts_arr, fn

def pick_bar(markets, timestamps, bar_frac):
    """Pick sample at bar_frac% into each market."""
    samples = []
    for mi, m in enumerate(markets):
        mask = (timestamps >= m["start_ts"]) & (timestamps < m["end_ts"])
        si = np.where(mask)[0]
        if len(si) == 0: continue
        idx = min(int(len(si) * bar_frac), len(si) - 1)
        samples.append((mi, si[idx]))
    return samples

def calibration_row(probs, yt, bins):
    dp = np.maximum(probs, 1 - probs)
    correct = ((probs > 0.5) == yt).astype(float)
    cells = []
    for lo, hi in bins:
        m = (dp >= lo) & (dp < hi); cnt = m.sum()
        cells.append(f"{correct[m].mean():>5.0%}({cnt:>3})" if cnt >= 5 else "    -     ")
    return " ".join(cells)

def simulate(test_samples, markets, probs, conf, ep, capital, bet, direction_filter=None):
    """Run trade simulation. direction_filter='UP'/'DOWN'/None."""
    c = capital; w = l = 0; pnls = []; ups = downs = 0
    for (mi, _), prob in zip(test_samples, probs):
        m = markets[mi]; dp = max(prob, 1 - prob)
        if dp < conf or c < bet: continue
        pred_dir = "UP" if prob > 0.5 else "DOWN"
        if direction_filter and pred_dir != direction_filter: continue
        won = (pred_dir == m["out"])
        if pred_dir == "UP": ups += 1
        else: downs += 1
        b = min(bet, c); c -= b
        if won: c += b / ep; pnls.append(b / ep - b); w += 1
        else: pnls.append(-b); l += 1
    n = w + l
    if n == 0: return None
    pa = np.array(pnls); eq = np.array([capital] + list(np.cumsum(pnls) + capital))
    pk = np.maximum.accumulate(eq)
    dd = ((eq - pk) / np.where(pk > 0, pk, 1) * 100).min()
    gp = pa[pa > 0].sum() if (pa > 0).any() else 0
    gl = abs(pa[pa < 0].sum()) if (pa < 0).any() else 0.01
    ev = float(np.mean(pa))
    return {"n": n, "w": w, "wr": w/n*100, "pnl": pa.sum(), "dd": dd,
            "sharpe": float(np.mean(pa)/max(np.std(pa),1e-9)*np.sqrt(252)),
            "pf": gp/max(gl,0.01), "ev_per_trade": ev, "ups": ups, "downs": downs}


def run_optuna_tuning(Xs, y_tr, fn, n_trials=40):
    """Optuna hyperparameter search for LGBM + CatBoost."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        P("  [!] optuna not installed, skipping"); return {}

    from sklearn.model_selection import cross_val_score
    import lightgbm as lgb

    best_params = {}

    # LGBM
    def lgbm_objective(trial):
        p = {
            "objective": "binary", "metric": "binary_logloss", "verbose": -1,
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        }
        n_rounds = trial.suggest_int("n_rounds", 100, 500)
        ds = lgb.Dataset(pd.DataFrame(Xs, columns=fn), label=y_tr)
        cv = lgb.cv(p, ds, num_boost_round=n_rounds, nfold=3, seed=42,
                    callbacks=[lgb.log_evaluation(0), lgb.early_stopping(30)])
        return min(cv["valid binary_logloss-mean"])

    P("  [Optuna] LGBM tuning...")
    study_lgbm = optuna.create_study(direction="minimize")
    study_lgbm.optimize(lgbm_objective, n_trials=n_trials, show_progress_bar=True)
    best_params["lgbm"] = study_lgbm.best_params
    P(f"    Best LGBM logloss: {study_lgbm.best_value:.4f}")

    # CatBoost
    try:
        from catboost import CatBoostClassifier
        def cb_objective(trial):
            p = {
                "iterations": trial.suggest_int("iterations", 100, 500),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "depth": trial.suggest_int("depth", 4, 10),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
                "random_strength": trial.suggest_float("random_strength", 0.0, 1.0),
                "verbose": 0, "thread_count": -1, "random_seed": 42,
            }
            from sklearn.model_selection import cross_val_score
            cb = CatBoostClassifier(**p)
            scores = cross_val_score(cb, Xs, y_tr, cv=3, scoring="neg_log_loss")
            return -scores.mean()

        P("  [Optuna] CatBoost tuning...")
        study_cb = optuna.create_study(direction="minimize")
        study_cb.optimize(cb_objective, n_trials=n_trials, show_progress_bar=True)
        best_params["catboost"] = study_cb.best_params
        P(f"    Best CatBoost logloss: {study_cb.best_value:.4f}")
    except ImportError:
        pass

    return best_params


def run_variation(sol_data, data_path, dur, win, args, rebuild):
    """Run full analysis for one (duration, window) combo. Returns results dict."""
    tag = f"{dur}m/w{win}"
    P(f"\n{'#'*80}")
    P(f"  VARIATION: {tag}")
    P(f"{'#'*80}")

    X, y, ts_arr, fn = build_or_load(sol_data, dur, win, data_path, rebuild)
    markets = build_markets(sol_data, dur)
    up = sum(1 for m in markets if m["out"] == "UP")
    P(f"  Markets: {len(markets)} ({up} UP = {up/len(markets)*100:.1f}%) | Features: {X.shape[1]}")

    # Bar fractions: 0%, 20%, 40%, 60%, 80%
    bar_fracs = [0.0, 0.20, 0.40, 0.60, 0.80]
    bar_labels = ["0%", "20%", "40%", "60%", "80%"]
    bar_sets = {f: pick_bar(markets, ts_arr, f) for f in bar_fracs}
    for f, lbl in zip(bar_fracs, bar_labels):
        P(f"    bar {lbl}: {len(bar_sets[f])} markets")

    # 70/30 time split
    ref = bar_sets[0.20]
    sp = int(len(ref) * args.split)
    train_mkts = set(mi for mi, _ in ref[:sp])
    n_test = len(ref) - sp
    P(f"  Train: {len(train_mkts)} | Test: {n_test}")

    # Gather all training samples from ALL bars of train markets
    train_idx = set()
    for f in bar_fracs:
        for mi, si in bar_sets[f]:
            if mi in train_mkts: train_idx.add(si)
    train_idx = sorted(train_idx)
    X_tr = X[train_idx]; y_tr = y[train_idx]
    P(f"  Train samples: {len(train_idx)} (all bars from train markets)")

    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    Xs = scaler.fit_transform(X_tr)

    def get_test(frac):
        s = bar_sets[frac]
        test = [(mi, si) for mi, si in s if mi not in train_mkts]
        idx = [si for _, si in test]
        return test, scaler.transform(X[idx]), y[idx]

    test_data = {f: get_test(f) for f in bar_fracs}

    # ── TRAIN ALL MODELS ──
    models = {}; train_accs = {}; models_by_bar = {f: {} for f in bar_fracs}

    # LGBM
    import lightgbm as lgb
    P("  [1/4] LGBM 300 rounds...")
    lgb_ds = lgb.Dataset(pd.DataFrame(Xs, columns=fn), label=y_tr)
    lgb_m = lgb.train({"objective":"binary","metric":"binary_logloss","num_leaves":31,
        "learning_rate":0.05,"feature_fraction":0.8,"bagging_fraction":0.8,
        "bagging_freq":5,"verbose":-1,"n_jobs":-1},
        lgb_ds, num_boost_round=300, valid_sets=[lgb_ds], callbacks=[lgb.log_evaluation(0)])
    imp = lgb_m.feature_importance(importance_type="gain")
    top_feats = sorted(zip(fn, imp), key=lambda x: -x[1])[:15]
    train_accs["LGBM"] = float(np.mean((lgb_m.predict(Xs) > 0.5) == y_tr))
    models["lgbm"] = lgb_m
    for f in bar_fracs:
        _, Xt, yt = test_data[f]; models_by_bar[f]["LGBM"] = (lgb_m.predict(Xt), yt)
    del lgb_ds; gc.collect()

    # CatBoost
    P("  [2/4] CatBoost 300 iter...")
    try:
        from catboost import CatBoostClassifier
        cb = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6,
                                verbose=0, thread_count=-1, random_seed=42)
        cb.fit(Xs, y_tr)
        train_accs["CatBoost"] = float(np.mean(cb.predict(Xs) == y_tr))
        models["catboost"] = cb
        for f in bar_fracs:
            _, Xt, yt = test_data[f]; models_by_bar[f]["CatBoost"] = (cb.predict_proba(Xt)[:,1], yt)
    except ImportError: P("    skip")

    # RF
    P("  [3/4] RF 300 trees...")
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=300, max_depth=14, n_jobs=-1, random_state=42)
    rf.fit(Xs, y_tr)
    train_accs["RF"] = float(np.mean(rf.predict(Xs) == y_tr))
    models["rf"] = rf
    for f in bar_fracs:
        _, Xt, yt = test_data[f]; models_by_bar[f]["RF"] = (rf.predict_proba(Xt)[:,1], yt)

    # XGBoost
    P("  [4/4] XGBoost 300 rounds...")
    try:
        import xgboost as xgb
        xm = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            verbosity=0, n_jobs=-1, random_state=42)
        xm.fit(Xs, y_tr)
        train_accs["XGBoost"] = float(np.mean(xm.predict(Xs) == y_tr))
        models["xgboost"] = xm
        for f in bar_fracs:
            _, Xt, yt = test_data[f]; models_by_bar[f]["XGBoost"] = (xm.predict_proba(Xt)[:,1], yt)
    except ImportError: P("    skip")

    # Ensembles
    for f in bar_fracs:
        pd_map = {k: v[0] for k, v in models_by_bar[f].items()}
        _, _, yt = test_data[f]
        if "LGBM" in pd_map and "CatBoost" in pd_map:
            models_by_bar[f]["LGBM+CB"] = ((pd_map["LGBM"]+pd_map["CatBoost"])/2, yt)
        if all(k in pd_map for k in ["LGBM","CatBoost","RF"]):
            models_by_bar[f]["LCR"] = ((pd_map["LGBM"]+pd_map["CatBoost"]+pd_map["RF"])/3, yt)
        all_p = list(pd_map.values())
        if len(all_p) >= 3:
            models_by_bar[f]["ALL-Ens"] = (np.mean(all_p, axis=0), yt)

    all_mn = sorted(set().union(*[set(models_by_bar[f].keys()) for f in bar_fracs]))

    # ══════════════════════════════════════════════════
    # SECTION A: OVERFITTING CHECK
    # ══════════════════════════════════════════════════
    P(f"\n  --- OVERFITTING ({tag}) ---")
    P(f"  {'Model':<12} {'Train':>7} {'Test@20%':>9} {'Gap':>7}")
    for mn in ["LGBM","CatBoost","RF","XGBoost"]:
        if mn not in train_accs: continue
        ta = train_accs[mn]
        pr, yt = models_by_bar[0.20].get(mn, (np.array([]),np.array([])))
        te = np.mean((pr > 0.5) == yt) if len(pr) else 0
        P(f"  {mn:<12} {ta:>6.1%} {te:>8.1%} {ta-te:>+6.1%}")

    # ══════════════════════════════════════════════════
    # SECTION B: ACCURACY BY BAR (timing)
    # ══════════════════════════════════════════════════
    P(f"\n  --- ACCURACY BY ENTRY TIMING ({tag}) | Test={n_test} markets ---")
    hdr = f"  {'Model':<12}" + "".join(f"  {lbl:>10}" for lbl in bar_labels)
    P(hdr)
    for mn in all_mn:
        row = f"  {mn:<12}"
        for f in bar_fracs:
            if mn in models_by_bar[f]:
                pr, yt = models_by_bar[f][mn]
                acc = np.mean((pr>0.5)==yt)
                n_t = len(yt)
                row += f"  {acc:>5.1%}({n_t:>4})"
            else: row += "          -"
        P(row)

    # ══════════════════════════════════════════════════
    # SECTION C: PER-BAR CALIBRATION
    # ══════════════════════════════════════════════════
    bins = [(0.50,0.55),(0.55,0.60),(0.60,0.65),(0.65,0.70),(0.70,0.75),(0.75,0.80),(0.80,0.85),(0.85,1.01)]
    blbl = "50-55 55-60 60-65 65-70 70-75 75-80 80-85  85+ "
    for f, lbl in zip(bar_fracs, bar_labels):
        P(f"\n  --- CALIBRATION @ {lbl} ({tag}) ---")
        P(f"  {'Model':<12} {blbl}")
        for mn in all_mn:
            if mn not in models_by_bar[f]: continue
            pr, yt = models_by_bar[f][mn]
            P(f"  {mn:<12} {calibration_row(pr, yt, bins)}")

    # ══════════════════════════════════════════════════
    # SECTION D: CONSENSUS ANALYSIS
    # ══════════════════════════════════════════════════
    P(f"\n  --- CONSENSUS ({tag}) ---")
    base_models = [m for m in ["LGBM","CatBoost","RF","XGBoost"] if m in train_accs]
    for f, lbl in zip([0.0, 0.20, 0.40], ["0%","20%","40%"]):
        t_samp, Xt_s, yt_c = test_data[f]
        all_probs = {}
        for mn in base_models:
            if mn in models_by_bar[f]:
                all_probs[mn] = models_by_bar[f][mn][0]
        if len(all_probs) < 2: continue
        n_samp = len(yt_c)
        # Consensus: all agree on direction
        agree_mask = np.ones(n_samp, dtype=bool)
        first_dir = None
        for mn, pr in all_probs.items():
            dirs = pr > 0.5
            if first_dir is None: first_dir = dirs
            else: agree_mask &= (dirs == first_dir)
        n_agree = agree_mask.sum()
        if n_agree > 0:
            wr_agree = np.mean(((list(all_probs.values())[0][agree_mask] > 0.5) == yt_c[agree_mask]))
            wr_disagree = np.mean(((list(all_probs.values())[0][~agree_mask] > 0.5) == yt_c[~agree_mask])) if (~agree_mask).sum() > 0 else 0
            P(f"  @{lbl}: Agree {n_agree}/{n_samp} ({n_agree/n_samp*100:.0f}%) "
              f"WR_agree={wr_agree:.1%} WR_disagree={wr_disagree:.1%}")

    # ══════════════════════════════════════════════════
    # SECTION E: GAP ANALYSIS (per model) — absolute + direction-relative
    # ══════════════════════════════════════════════════
    P(f"\n  --- GAP ANALYSIS ({tag}) | Test N per bar shown ---")
    gap_bins = [(-999,-0.10,"Gap<-0.1%"),(-0.10,-0.03,"-0.1 to -0.03%"),(-0.03,0,"-0.03 to 0%"),
                (0,0.03,"0 to +0.03%"),(0.03,0.10,"+0.03 to +0.1%"),(0.10,999,">+0.1%")]
    for f, lbl in zip([0.0, 0.20, 0.40], ["0%","20%","40%"]):
        t_samp, _, yt_g = test_data[f]
        gaps = np.array([markets[mi]["gap_pct"] for mi, _ in t_samp])
        P(f"  @{lbl} (N={len(yt_g)}):")
        for mn in ["CatBoost","RF","ALL-Ens"]:
            if mn not in models_by_bar[f]: continue
            pr, _ = models_by_bar[f][mn]
            correct = ((pr > 0.5) == yt_g)
            row = f"    {mn:<10}"
            for glo, ghi, glbl in gap_bins:
                m = (gaps >= glo) & (gaps < ghi)
                if m.sum() >= 5:
                    row += f"  {glbl}: {correct[m].mean():.0%}({m.sum()})"
            P(row)

    # Direction-relative gap using ENTRY-TIME gap (SOL@bar20% - PTB), NOT outcome gap
    P(f"\n  --- DIRECTION-RELATIVE GAP ({tag}, @20%) ---")
    P(f"  Using entry-time gap: (SOL@20% - PTB) / PTB * 100")
    t_samp20, _, yt_g20 = test_data[0.20]
    # Compute entry-time gap for each test market at bar 20%
    entry_gaps20 = []
    for mi, si in t_samp20:
        ptb = markets[mi]["ptb"]
        sol_at_entry = float(sol_data.iloc[si]["close"])
        entry_gaps20.append((sol_at_entry - ptb) / ptb * 100)
    entry_gaps20 = np.array(entry_gaps20)
    P(f"  Entry gap stats: mean={entry_gaps20.mean():.4f}%, std={entry_gaps20.std():.4f}%, "
      f"p25={np.percentile(entry_gaps20,25):.4f}%, p75={np.percentile(entry_gaps20,75):.4f}%")
    P(f"  {'Model':<12} {'FavGap':>12} {'N':>4} {'WR':>6} | {'NeutGap':>12} {'N':>4} {'WR':>6} | {'UnfavGap':>12} {'N':>4} {'WR':>6}")
    for mn in ["LGBM","CatBoost","RF","ALL-Ens"]:
        if mn not in models_by_bar[0.20]: continue
        pr, _ = models_by_bar[0.20][mn]
        pred_up = pr > 0.5
        correct = (pred_up == yt_g20)
        # Favorable: UP pred + positive entry gap, or DOWN pred + negative entry gap
        fav = (pred_up & (entry_gaps20 > 0.03)) | (~pred_up & (entry_gaps20 < -0.03))
        unfav = (pred_up & (entry_gaps20 < -0.03)) | (~pred_up & (entry_gaps20 > 0.03))
        neut = ~fav & ~unfav
        nf, nn, nu = fav.sum(), neut.sum(), unfav.sum()
        fwr = correct[fav].mean()*100 if nf >= 5 else 0
        nwr = correct[neut].mean()*100 if nn >= 5 else 0
        uwr = correct[unfav].mean()*100 if nu >= 5 else 0
        P(f"  {mn:<12} {'Favorable':>12} {nf:>4} {fwr:>5.1f}% | {'Neutral':>12} {nn:>4} {nwr:>5.1f}% | {'Unfavorable':>12} {nu:>4} {uwr:>5.1f}%")

    # ══════════════════════════════════════════════════
    # SECTION E2: DIRECTION ANALYSIS (UP vs DOWN)
    # ══════════════════════════════════════════════════
    P(f"\n  --- DIRECTION BREAKDOWN ({tag}) | Test={n_test} ---")
    P(f"  {'Model':<12} {'Bar':>4} | {'Total':>5} {'UP_N':>5} {'UP_WR':>6} {'DN_N':>5} {'DN_WR':>6} {'All_WR':>7} | {'Bias':>6}")
    for f, lbl in zip(bar_fracs[:3], bar_labels[:3]):
        for mn in ["LGBM","CatBoost","RF","XGBoost","ALL-Ens"]:
            if mn not in models_by_bar[f]: continue
            pr, yt = models_by_bar[f][mn]
            pred_up = pr > 0.5; pred_dn = ~pred_up
            actual = yt.astype(bool)
            up_n = pred_up.sum(); dn_n = pred_dn.sum(); total = len(yt)
            up_wr = (pred_up & actual).sum() / max(up_n,1) * 100
            dn_wr = (pred_dn & ~actual).sum() / max(dn_n,1) * 100
            all_wr = np.mean((pr > 0.5) == yt) * 100
            bias = "UP" if up_n > dn_n * 1.15 else "DOWN" if dn_n > up_n * 1.15 else "EVEN"
            P(f"  {mn:<12} {lbl:>4} | {total:>5} {up_n:>5} {up_wr:>5.1f}% {dn_n:>5} {dn_wr:>5.1f}% {all_wr:>6.1f}% | {bias:>6}")

    # ══════════════════════════════════════════════════
    # SECTION F: SIMULATION GRID (share prices 0.35-0.70, directions)
    # ══════════════════════════════════════════════════
    results = []
    confs = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    eps = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    for f in bar_fracs:
        for mn in all_mn:
            if mn not in models_by_bar[f]: continue
            probs, _ = models_by_bar[f][mn]
            ts_list = [(mi,si) for mi,si in bar_sets[f] if mi not in train_mkts]
            for conf in confs:
                for ep in eps:
                    # All directions
                    r = simulate(ts_list, markets, probs, conf, ep, args.capital, args.bet)
                    if r: r.update({"model":mn,"bar_pct":f,"conf":conf,"ep":ep,"tag":tag,"dir":"ALL"}); results.append(r)
                    # DOWN only
                    rd = simulate(ts_list, markets, probs, conf, ep, args.capital, args.bet, direction_filter="DOWN")
                    if rd: rd.update({"model":mn,"bar_pct":f,"conf":conf,"ep":ep,"tag":tag,"dir":"DOWN"}); results.append(rd)
                    # UP only
                    ru = simulate(ts_list, markets, probs, conf, ep, args.capital, args.bet, direction_filter="UP")
                    if ru: ru.update({"model":mn,"bar_pct":f,"conf":conf,"ep":ep,"tag":tag,"dir":"UP"}); results.append(ru)

    # ══════════════════════════════════════════════════
    # SECTION G: TOP 15 BY SHARPE + HEAD-TO-HEAD + EV
    # ══════════════════════════════════════════════════
    P(f"\n  --- TOP 15 BY SHARPE ({tag}, bar<=40%, ALL dirs) ---")
    early = sorted([r for r in results if r["bar_pct"] <= 0.40 and r["dir"]=="ALL" and r["n"]>=15],
                   key=lambda x: -x["sharpe"])[:15]
    P(f"  {'#':>2} {'Model':<12} {'Bar':>4} {'Conf':>5} {'EP':>5} {'N':>4} {'WR':>6} {'PnL':>8} {'EV':>7} {'Shrp':>6} {'PF':>5}")
    for i, r in enumerate(early):
        P(f"  {i+1:>2} {r['model']:<12} {r['bar_pct']*100:>3.0f}% {r['conf']:>5.2f} ${r['ep']:.2f} "
          f"{r['n']:>4} {r['wr']:>5.1f}% ${r['pnl']:>+6.0f} ${r['ev_per_trade']:>+5.2f} {r['sharpe']:>5.1f} {r['pf']:>4.1f}")

    # EV table by share price for best models
    P(f"\n  --- EV BY SHARE PRICE ({tag}, conf>=0.65, bar=20%) ---")
    P(f"  {'Model':<12} {'Dir':>4}" + "".join(f" ${ep:.2f}" for ep in eps))
    for mn in ["CatBoost","RF","ALL-Ens","LCR"]:
        for d in ["ALL","DOWN"]:
            row = f"  {mn:<12} {d:>4}"
            for ep in eps:
                sub = [r for r in results if r["model"]==mn and r["bar_pct"]==0.20
                       and r["conf"]>=0.65 and r["ep"]==ep and r["dir"]==d and r["n"]>=10]
                if sub:
                    best = max(sub, key=lambda x: x["ev_per_trade"])
                    row += f" ${best['ev_per_trade']:>+5.2f}"
                else:
                    row += "     - "
            P(row)

    P(f"\n  --- HEAD-TO-HEAD conf=0.70 EP=$0.50 ({tag}) ---")
    P(f"  {'Model':<12} {'Bar':>4} {'Dir':>4} {'N':>4} {'WR':>6} {'PnL':>8} {'EV':>7} {'Shrp':>6} {'PF':>5}")
    h2h = sorted([r for r in results if r["bar_pct"] <= 0.40 and r["conf"]==0.70 and r["ep"]==0.50],
                 key=lambda x: -x["sharpe"])
    for r in h2h:
        P(f"  {r['model']:<12} {r['bar_pct']*100:>3.0f}% {r['dir']:>4} {r['n']:>4} {r['wr']:>5.1f}% ${r['pnl']:>+6.0f} ${r['ev_per_trade']:>+5.2f} {r['sharpe']:>5.1f} {r['pf']:>4.1f}")

    # DOWN-only top 10
    P(f"\n  --- TOP 10 DOWN-ONLY ({tag}, bar<=40%) ---")
    down_only = sorted([r for r in results if r["bar_pct"]<=0.40 and r["dir"]=="DOWN" and r["n"]>=10],
                       key=lambda x:-x["sharpe"])[:10]
    for i, r in enumerate(down_only):
        P(f"  {i+1:>2} {r['model']:<12} {r['bar_pct']*100:>3.0f}% conf={r['conf']:.2f} EP=${r['ep']:.2f} "
          f"{r['n']:>3}t {r['wr']:>5.1f}% ${r['pnl']:>+6.0f} EV=${r['ev_per_trade']:>+.2f}")

    # FEATURES
    P(f"\n  --- TOP 15 FEATURES ({tag}) ---")
    for i, (f_name, fv) in enumerate(top_feats[:15]):
        bar = "#" * int(fv / max(top_feats[0][1], 1) * 30)
        pre = " ***" if "pre_mkt" in f_name else ""
        P(f"  {i+1:>2}. {f_name:<32} {fv:>6.0f} {bar}{pre}")

    # Pre-market feature importance summary
    pre_feats = [(n,v) for n,v in top_feats if "pre_mkt" in n]
    total_imp = sum(v for _,v in top_feats)
    pre_imp = sum(v for _,v in pre_feats)
    P(f"  Pre-market features: {len(pre_feats)} total, {pre_imp/total_imp*100:.1f}% of total importance")

    return {"tag": tag, "dur": dur, "win": win, "n_markets": len(markets),
            "n_features": X.shape[1], "train_accs": train_accs,
            "results": results, "top_feats": top_feats[:15],
            "models": models, "scaler": scaler, "fn": fn}


def main():
    args = parse_args(); t0 = time.time()
    P("=" * 90)
    P("  MEGA BACKTEST v5 -- 30 days, 5m+15m, window 30+60, all models, per-bar timing")
    P("=" * 90)

    sol_data, data_path = load_data()
    P(f"  Data: {len(sol_data):,} bars ({pd.to_datetime(sol_data['ts'].iloc[0],unit='ms').date()} -> "
      f"{pd.to_datetime(sol_data['ts'].iloc[-1],unit='ms').date()})")

    all_var_results = []

    # Run 2 variations: 5m and 15m (single window=60 — enough for pre-market lookback)
    for dur in [5, 15]:
        win = 60
        vr = run_variation(sol_data, data_path, dur, win, args, args.rebuild)
        all_var_results.append(vr)
        gc.collect()

    # ══════════════════════════════════════════════════
    # MEGA COMPARISON
    # ══════════════════════════════════════════════════
    P(f"\n{'='*90}")
    P(f"  MEGA COMPARISON -- ALL VARIATIONS")
    P(f"{'='*90}")
    P(f"  {'Variation':<14} {'Model':<12} {'Bar':>4} {'Conf':>5} {'EP':>5} {'N':>4} {'WR':>6} {'PnL':>8} {'Shrp':>6} {'PF':>5}")
    P(f"  {'-'*82}")

    all_results_flat = []
    for vr in all_var_results:
        for r in vr["results"]:
            r["dur"] = vr["dur"]; r["win"] = vr["win"]
            all_results_flat.append(r)

    # Best by Sharpe per variation (bar<=40%, ALL dir)
    for vr in all_var_results:
        early = [r for r in vr["results"] if r["bar_pct"] <= 0.40 and r["n"] >= 20 and r.get("dir","ALL")=="ALL"]
        if early:
            best = max(early, key=lambda x: x["sharpe"])
            P(f"  {vr['tag']:<14} {best['model']:<12} {best['bar_pct']*100:>3.0f}% {best['conf']:>5.2f} "
              f"${best['ep']:.2f} {best['n']:>4} {best['wr']:>5.1f}% ${best['pnl']:>+6.0f} "
              f"{best['sharpe']:>5.1f} {best['pf']:>4.1f}")

    # Overall best ALL
    all_early = [r for r in all_results_flat if r["bar_pct"] <= 0.40 and r["n"] >= 20 and r.get("dir","ALL")=="ALL"]
    if all_early:
        P(f"\n  OVERALL BEST ALL (Sharpe, bar<=40%, N>=20):")
        for r in sorted(all_early, key=lambda x: -x["sharpe"])[:5]:
            P(f"    {r['dur']}m/w{r['win']} {r['model']:<12} @{r['bar_pct']*100:.0f}% "
              f"conf={r['conf']:.2f} EP=${r['ep']:.2f} -> {r['n']}t {r['wr']:.1f}% "
              f"${r['pnl']:+.0f} EV=${r['ev_per_trade']:+.2f} Sharpe={r['sharpe']:.1f}")

    # Overall best DOWN-only
    all_down = [r for r in all_results_flat if r["bar_pct"] <= 0.40 and r["n"] >= 10 and r.get("dir")=="DOWN"]
    if all_down:
        P(f"\n  OVERALL BEST DOWN-ONLY (Sharpe, bar<=40%, N>=10):")
        for r in sorted(all_down, key=lambda x: -x["sharpe"])[:5]:
            P(f"    {r['dur']}m/w{r['win']} {r['model']:<12} @{r['bar_pct']*100:.0f}% "
              f"conf={r['conf']:.2f} EP=${r['ep']:.2f} -> {r['n']}t {r['wr']:.1f}% "
              f"${r['pnl']:+.0f} EV=${r['ev_per_trade']:+.2f}")

    # 5m vs 15m comparison
    P(f"\n  5m vs 15m HEAD-TO-HEAD (conf=0.70, EP=$0.50, bar=20%, ALL):")
    P(f"  {'Dur':<5} {'Win':<5} {'Model':<12} {'Dir':>4} {'N':>4} {'WR':>6} {'PnL':>8} {'EV':>7}")
    for dur in [5, 15]:
        for win in [30, 60]:
            for d in ["ALL","DOWN"]:
                subset = [r for r in all_results_flat if r["dur"]==dur and r["win"]==win
                          and r["bar_pct"]==0.20 and r["conf"]==0.70 and r["ep"]==0.50 and r.get("dir")==d]
                best = max(subset, key=lambda x: x["sharpe"]) if subset else None
                if best and best["n"] >= 5:
                    P(f"  {dur}m   w{win}  {best['model']:<12} {d:>4} {best['n']:>4} {best['wr']:>5.1f}% "
                      f"${best['pnl']:>+6.0f} ${best['ev_per_trade']:>+5.2f}")

    # EV matrix: all durations x share prices
    P(f"\n  EV MATRIX (best model per cell, conf>=0.65, bar=20%, ALL dir, N>=10):")
    P(f"  {'Var':<12}" + "".join(f" EP${ep:.2f}" for ep in [0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70]))
    for dur in [5, 15]:
        for win in [30, 60]:
            row = f"  {dur}m/w{win:<6}"
            for ep in [0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70]:
                sub = [r for r in all_results_flat if r["dur"]==dur and r["win"]==win
                       and r["bar_pct"]==0.20 and r["conf"]>=0.65 and r["ep"]==ep
                       and r.get("dir")=="ALL" and r["n"]>=10]
                if sub:
                    best = max(sub, key=lambda x: x["ev_per_trade"])
                    row += f" ${best['ev_per_trade']:>+5.2f}"
                else:
                    row += "      - "
            P(row)

    # ══════════════════════════════════════════════════
    # SAVE BEST MODELS
    # ══════════════════════════════════════════════════
    P(f"\n  Saving best models for live trading...")
    # Pick the variation with best Sharpe at bar=20%
    best_var = None; best_sharpe = -999
    for vr in all_var_results:
        sub = [r for r in vr["results"] if r["bar_pct"]==0.20 and r["conf"]>=0.65 and r["n"]>=20]
        if sub:
            s = max(sub, key=lambda x: x["sharpe"])["sharpe"]
            if s > best_sharpe: best_sharpe = s; best_var = vr

    if best_var:
        md = Path("training/model_registry/latest"); md.mkdir(parents=True, exist_ok=True)
        for name, model in best_var["models"].items():
            joblib.dump(model, md / f"{name}_cls.pkl")
        joblib.dump(best_var["scaler"], md / "scaler.pkl")
        fn_raw = best_var["fn"]
        fn_list = [str(x) for x in fn_raw] if isinstance(fn_raw, (list, np.ndarray)) and len(fn_raw) > 0 else []
        if not fn_list:
            P(f"  WARNING: fn empty (type={type(fn_raw)}), using fallback from cache")
            _, _, _, fn_list = build_or_load(sol_data, best_var["dur"], best_var["win"], data_path, False)
            fn_list = [str(x) for x in fn_list]
        meta = {"feature_names": fn_list, "models": list(best_var["models"].keys()),
                "trained_on": time.strftime("%Y-%m-%d %H:%M:%S"),
                "variation": best_var["tag"],
                "n_features": len(fn_list)}
        with open(md / "meta.json", "w") as f: json.dump(meta, f, indent=2)
        P(f"  Saved {len(best_var['models'])} models from {best_var['tag']} -> {md}")

    # ══════════════════════════════════════════════════
    # OPTUNA (if requested)
    # ══════════════════════════════════════════════════
    optuna_params = {}
    if args.optuna and best_var:
        P(f"\n  Running Optuna tuning on best variation ({best_var['tag']})...")
        X_o, y_o, _, fn_o = build_or_load(sol_data, best_var["dur"], best_var["win"], data_path, False)
        from sklearn.preprocessing import RobustScaler
        sc_o = RobustScaler().fit_transform(X_o)
        optuna_params = run_optuna_tuning(sc_o, y_o, fn_o, n_trials=40)
        P(f"  Optuna results: {json.dumps({k: {kk: round(vv,4) if isinstance(vv,float) else vv for kk,vv in v.items()} for k,v in optuna_params.items()}, indent=2)}")

    # Save report JSON
    Path("results").mkdir(exist_ok=True)
    # Collect top results for report
    top_all = sorted([r for r in all_results_flat if r.get("dir")=="ALL" and r["n"]>=15],
                     key=lambda x: -x["sharpe"])[:20]
    top_down = sorted([r for r in all_results_flat if r.get("dir")=="DOWN" and r["n"]>=10],
                      key=lambda x: -x["sharpe"])[:10]
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_bars": len(sol_data),
        "variations": [{"tag": v["tag"], "n_markets": v["n_markets"],
                         "n_features": v["n_features"],
                         "train_accs": v["train_accs"]} for v in all_var_results],
        "best_variation": best_var["tag"] if best_var else None,
        "top_results_all": [{k:v for k,v in r.items() if k not in ["tag"]} for r in top_all],
        "top_results_down": [{k:v for k,v in r.items() if k not in ["tag"]} for r in top_down],
        "optuna_params": optuna_params,
    }
    Path("results/backtest_v5_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")

    elapsed = time.time() - t0
    P(f"\n{'='*90}")
    P(f"  COMPLETE -- {elapsed:.1f}s ({elapsed/60:.1f}min)")
    P(f"  Cache: data/cache/ (next run ~30s)")
    P(f"  Report: results/backtest_v5_report.json")
    P(f"{'='*90}")


if __name__ == "__main__":
    main()
