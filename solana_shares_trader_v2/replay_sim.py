"""Replay Simulation — test ML models on RECORDED real Polymarket data.

Loads recorded snapshots from data/recorded/shares/, builds features identical
to live trader, runs all models at various thresholds/combos, reports results.

Usage:
    python replay_sim.py                          # latest day
    python replay_sim.py --date 2026-05-04        # specific day
    python replay_sim.py --date 2026-05-04 --dur 15
"""
import sys, os, json, time, argparse, joblib, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
from datetime import datetime
warnings.filterwarnings("ignore")
P = print


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--date", type=str, default=None, help="YYYY-MM-DD (default: latest)")
    p.add_argument("--dur", type=int, default=None, help="5 or 15 (default: all)")
    p.add_argument("--bet", type=float, default=2.0)
    p.add_argument("--capital", type=float, default=100.0)
    return p.parse_args()


def find_recorded_data(date_str=None):
    """Find recorded parquet files."""
    base = Path("data/recorded/shares")
    if not base.exists():
        P("ERROR: No recorded data in data/recorded/shares/")
        sys.exit(1)

    if date_str:
        day_dirs = [base / date_str]
    else:
        day_dirs = sorted(base.iterdir())
        if not day_dirs:
            P("ERROR: No recorded data found"); sys.exit(1)
        day_dirs = [day_dirs[-1]]  # latest

    results = []
    for dd in day_dirs:
        if not dd.is_dir():
            continue
        for dur_dir in sorted(dd.iterdir()):
            if not dur_dir.is_dir():
                continue
            pq = dur_dir / "snapshots.parquet"
            if pq.exists():
                dur = int(dur_dir.name.replace("m", ""))
                results.append({"path": pq, "date": dd.name, "dur": dur})
    return results


def load_models():
    """Load trained models from registry."""
    md = Path("training/model_registry/latest")
    meta_path = md / "meta.json"
    if not meta_path.exists():
        P("ERROR: No models in training/model_registry/latest/")
        sys.exit(1)

    meta = json.loads(meta_path.read_text())
    fn = meta["feature_names"]

    models = {}
    for name in ["lgbm", "catboost", "rf", "xgboost"]:
        mp = md / f"{name}_cls.pkl"
        if mp.exists():
            models[name] = joblib.load(mp)

    scaler = joblib.load(md / "scaler.pkl")
    P(f"  Models: {list(models.keys())} | Features: {len(fn)}")
    return models, scaler, fn


def extract_markets(df, dur):
    """Extract individual markets from recorded snapshots.
    
    Uses slug timestamp to find market window, determines outcome from
    SOL price at start vs end of market.
    """
    markets = []
    for slug, grp_all in df.groupby("slug"):
        grp_all = grp_all.sort_values("ts")
        
        # Parse market start/end from slug: sol-updown-5m-{start_ts}
        parts = slug.split("-")
        if len(parts) < 4:
            continue
        try:
            mkt_start_ts = int(parts[-1])
            mkt_dur = int(parts[2].replace("m", ""))
        except ValueError:
            continue
        mkt_end_ts = mkt_start_ts + mkt_dur * 60
        
        # Filter snapshots to within market window
        grp = grp_all[(grp_all["ts"] >= mkt_start_ts) & (grp_all["ts"] <= mkt_end_ts)]
        if len(grp) < 3:
            continue
        
        # Coverage check
        coverage = (grp["ts"].max() - mkt_start_ts) / (mkt_dur * 60)
        if coverage < 0.80:
            P(f"    [skip] {slug}: only {coverage:.0%} coverage")
            continue
        
        first = grp.iloc[0]
        last = grp.iloc[-1]

        # Get PTB — from recorded data, or SOL price at market start
        ptb = first.get("price_to_beat", 0)
        if pd.isna(ptb) or ptb == 0:
            # Use SOL price at market start (closest snapshot to start_ts)
            near_start = grp_all[(grp_all["ts"] >= mkt_start_ts) & (grp_all["ts"] <= mkt_start_ts + 10)]
            ptb = float(near_start.iloc[0]["sol_price"]) if len(near_start) > 0 else float(first["sol_price"])

        # Determine outcome from SOL price at end of market
        sol_at_end = float(last["sol_price"])
        outcome = "UP" if sol_at_end >= ptb else "DOWN"

        markets.append({
            "market_id": first.get("market_id", slug),
            "slug": slug,
            "dur": mkt_dur,
            "ptb": ptb,
            "sol_at_start": float(first["sol_price"]),
            "sol_at_end": sol_at_end,
            "outcome": outcome,
            "snapshots": grp.reset_index(drop=True),
            "start_ts": float(mkt_start_ts),
            "end_ts": float(mkt_end_ts),
            "n_snaps": len(grp),
            "coverage": coverage,
        })

    return markets


def build_features_from_snaps(market, snap_idx, sol_history):
    """Build feature vector from recorded snapshot — matching live trader."""
    snaps = market["snapshots"]
    row = snaps.iloc[snap_idx]

    features = {}

    # SOL price features
    sol_price = float(row.get("sol_price", 0))
    ptb = market["ptb"]

    if sol_price > 0 and ptb > 0:
        features["distance_from_ptb_pct"] = (sol_price - ptb) / ptb * 100
        features["distance_from_ptb_norm"] = (sol_price - ptb) / ptb

    # Shares features from recorded CLOB data
    for col in ["up_best_bid", "up_best_ask", "up_mid_price", "up_spread", "up_spread_pct",
                 "dn_best_bid", "dn_best_ask", "dn_mid_price", "dn_spread", "dn_spread_pct",
                 "up_bid_volume", "up_ask_volume", "dn_bid_volume", "dn_ask_volume",
                 "up_ob_imbalance", "dn_ob_imbalance",
                 "shares_momentum_30s", "shares_momentum_2m", "shares_acceleration",
                 "liquidity_score", "volume_spike", "volume_imbalance"]:
        if col in snaps.columns:
            features[col] = float(row.get(col, 0) or 0)

    # Time features
    total_dur_s = market["dur"] * 60
    elapsed_s = float(row["ts"]) - market["start_ts"]
    features["time_elapsed_pct"] = min(elapsed_s / total_dur_s, 1.0) if total_dur_s > 0 else 0
    features["time_remaining_pct"] = 1.0 - features["time_elapsed_pct"]
    features["life_phase"] = features["time_elapsed_pct"]

    # Implied prob
    up_mid = features.get("up_mid_price", 0.5)
    features["up_implied_prob"] = up_mid if 0 < up_mid < 1 else 0.5

    return features


def simulate_replay(markets, models, scaler, feature_names, confs, eps,
                    capital, bet, entry_pcts=[0.20]):
    """Run simulation on recorded markets with various settings."""
    results = []

    for entry_pct in entry_pcts:
        for mkt in markets:
            if mkt["outcome"] is None:
                continue

            snaps = mkt["snapshots"]
            n = len(snaps)
            snap_idx = min(int(n * entry_pct), n - 1)

            # Build features
            feat_dict = build_features_from_snaps(mkt, snap_idx, None)

            # Create feature vector matching model's expected features
            fv = np.array([feat_dict.get(fn, 0.0) for fn in feature_names], dtype=np.float64)
            fv_scaled = scaler.transform(fv.reshape(1, -1))

            # Get predictions from all models
            preds = {}
            for name, model in models.items():
                try:
                    if name == "lgbm":
                        prob = float(model.predict(fv_scaled)[0])
                    else:
                        prob = float(model.predict_proba(fv_scaled)[0, 1])
                    preds[name] = prob
                except Exception:
                    pass

            if not preds:
                continue

            # Ensemble
            if len(preds) >= 2:
                preds["ensemble"] = np.mean(list(preds.values()))

            # Store for grid search
            actual_up = mkt["outcome"] == "UP"

            for model_name, prob in preds.items():
                dp = max(prob, 1 - prob)
                pred_dir = "UP" if prob > 0.5 else "DOWN"
                won = (pred_dir == "UP" and actual_up) or (pred_dir == "DOWN" and not actual_up)

                for conf in confs:
                    if dp < conf:
                        continue

                    # Share price at entry
                    row = snaps.iloc[snap_idx]
                    if pred_dir == "UP":
                        share_price = float(row.get("up_best_ask", 0.50) or 0.50)
                    else:
                        share_price = float(row.get("dn_best_ask", 0.50) or 0.50)

                    for max_ep in eps:
                        if share_price > max_ep:
                            continue

                        b = min(bet, capital)
                        shares = b / share_price
                        pnl = shares * (1.0 - share_price) if won else -b

                        results.append({
                            "model": model_name,
                            "entry_pct": entry_pct,
                            "conf": conf,
                            "max_ep": max_ep,
                            "slug": mkt["slug"],
                            "dir": pred_dir,
                            "won": won,
                            "pnl": round(pnl, 4),
                            "share_price": round(share_price, 4),
                            "prob": round(prob, 4),
                            "outcome": mkt["outcome"],
                        })

    return results


def report(results, markets, all_market_preds):
    """Print comprehensive report with per-market detail."""
    P(f"\n{'='*90}")
    P(f"  REPLAY SIMULATION REPORT")
    P(f"{'='*90}")
    with_outcome = [m for m in markets if m.get("outcome")]
    P(f"  Markets total: {len(markets)} | With outcomes: {len(with_outcome)}")
    P(f"  Total result rows: {len(results)}")

    # ── PER-MARKET DETAIL (like live) ──
    P(f"\n  === PER-MARKET ML PREDICTIONS (like live) ===")
    for mp in all_market_preds:
        mkt = mp["market"]
        P(f"\n  ┌─ {mkt['slug']}")
        P(f"  │  PTB: ${mkt['ptb']:.4f}  |  SOL@start: ${mkt['sol_at_start']:.3f}  |  SOL@end: ${mkt['sol_at_end']:.3f}")
        P(f"  │  Outcome: {mkt['outcome']}  |  Coverage: {mkt['coverage']:.0%}  |  Snaps: {mkt['n_snaps']}")
        for entry_lbl, entry_info in mp["entries"].items():
            snap_row = entry_info["snap_row"]
            sol_at = snap_row.get("sol_price", 0)
            gap = (sol_at - mkt["ptb"]) / mkt["ptb"] * 100 if mkt["ptb"] else 0
            up_ask = snap_row.get("up_best_ask", "?")
            dn_ask = snap_row.get("dn_best_ask", "?")
            P(f"  │  @{entry_lbl}: SOL=${sol_at:.3f}  gap={gap:+.4f}%  UP_ask=${up_ask}  DN_ask=${dn_ask}")
            P(f"  │    {'Model':<12} {'Prob_UP':>7} {'Dir':>4} {'Conf':>5} | {'Would trade?':>12} {'Won?':>4}")
            for mn, pred in sorted(entry_info["preds"].items()):
                dp = max(pred, 1-pred)
                d = "UP" if pred > 0.5 else "DN"
                # Would trade at conf=0.65, EP=$0.50?
                sp = float(up_ask if pred > 0.5 else dn_ask) if up_ask != "?" else 0.5
                would = "YES" if dp >= 0.65 and sp <= 0.50 else "no"
                won = (d == "UP" and mkt["outcome"] == "UP") or (d == "DN" and mkt["outcome"] == "DOWN")
                P(f"  │    {mn:<12} {pred:>7.3f} {d:>4} {dp:>4.0%} | {would:>12} {'✅' if won else '❌':>4}")
            # Ensemble consensus
            all_p = list(entry_info["preds"].values())
            if all_p:
                ens = np.mean(all_p)
                n_up = sum(1 for p in all_p if p > 0.5)
                P(f"  │    {'CONSENSUS':<12} {ens:>7.3f} {'UP' if ens>0.5 else 'DN':>4} {max(ens,1-ens):>4.0%} | {n_up}/{len(all_p)} say UP")
        P(f"  └─")

    if not results:
        P("  No grid results to show."); return

    df = pd.DataFrame(results)

    # Best combos by WR (min N)
    min_n = max(3, len(with_outcome) // 2)
    P(f"\n  --- TOP COMBOS BY WIN RATE (N>={min_n}) ---")
    P(f"  {'#':>2} {'Model':<12} {'Entry':>5} {'Conf':>5} {'MaxEP':>6} {'N':>4} {'WR':>6} {'PnL':>8} {'EV':>7} {'Dir':>5}")
    grouped = df.groupby(["model", "entry_pct", "conf", "max_ep"])
    combo_stats = []
    for (model, ep, conf, mep), grp in grouped:
        n = len(grp)
        if n < min_n:
            continue
        w = grp["won"].sum()
        wr = w / n * 100
        pnl = grp["pnl"].sum()
        ev = grp["pnl"].mean()
        bias = "UP" if (grp["dir"] == "UP").sum() > n * 0.6 else "DOWN" if (grp["dir"] == "DOWN").sum() > n * 0.6 else "MIX"
        combo_stats.append({
            "model": model, "entry_pct": ep, "conf": conf, "max_ep": mep,
            "n": n, "wr": wr, "pnl": pnl, "ev": ev, "bias": bias,
        })

    combo_stats.sort(key=lambda x: -x["wr"])
    for i, c in enumerate(combo_stats[:20]):
        P(f"  {i+1:>2} {c['model']:<12} {c['entry_pct']*100:>4.0f}% {c['conf']:>5.2f} ${c['max_ep']:.2f} "
          f"{c['n']:>4} {c['wr']:>5.1f}% ${c['pnl']:>+6.0f} ${c['ev']:>+5.2f} {c['bias']:>5}")

    # Model head-to-head
    P(f"\n  --- MODEL HEAD-TO-HEAD (conf=0.65, EP=$0.50, entry=20%) ---")
    P(f"  {'Model':<12} {'N':>4} {'WR':>6} {'PnL':>8} {'EV':>7}")
    for model in sorted(df["model"].unique()):
        sub = df[(df["model"]==model) & (df["conf"]==0.65) & (df["max_ep"]==0.50) & (df["entry_pct"]==0.20)]
        if len(sub) >= 1:
            n = len(sub); wr = sub["won"].sum()/max(n,1)*100
            P(f"  {model:<12} {n:>4} {wr:>5.1f}% ${sub['pnl'].sum():>+6.2f} ${sub['pnl'].mean():>+5.2f}")

    P(f"\n{'='*90}")
    P(f"  REPLAY COMPLETE")
    P(f"{'='*90}")


def main():
    args = parse_args()
    P("=" * 90)
    P("  REPLAY SIMULATION — ML on RECORDED real Polymarket data")
    P("=" * 90)

    # Find data
    recorded = find_recorded_data(args.date)
    if not recorded:
        P("  No recorded data found."); return

    # Filter by duration
    if args.dur:
        recorded = [r for r in recorded if r["dur"] == args.dur]

    P(f"  Found {len(recorded)} recorded datasets:")
    for r in recorded:
        df = pd.read_parquet(r["path"])
        n_mkts = df["market_id"].nunique() if "market_id" in df.columns else "?"
        P(f"    {r['date']} {r['dur']}m: {len(df)} snapshots, {n_mkts} markets")

    # Load models
    models, scaler, fn = load_models()

    # Process each dataset
    all_markets = []
    for r in recorded:
        df = pd.read_parquet(r["path"])
        mkts = extract_markets(df, r["dur"])
        all_markets.extend(mkts)
        P(f"  {r['date']} {r['dur']}m: {len(mkts)} markets ({sum(1 for m in mkts if m['outcome'])} with outcomes)")

    P(f"  Total markets: {len(all_markets)}")

    # Collect per-market predictions (for detailed report)
    all_market_preds = []
    entry_labels = {"0%": 0.0, "10%": 0.10, "20%": 0.20, "40%": 0.40}
    for mkt in all_markets:
        if not mkt.get("outcome"):
            continue
        mp = {"market": mkt, "entries": {}}
        snaps = mkt["snapshots"]
        n = len(snaps)
        for lbl, frac in entry_labels.items():
            snap_idx = min(int(n * frac), n - 1)
            feat_dict = build_features_from_snaps(mkt, snap_idx, None)
            fv = np.array([feat_dict.get(f, 0.0) for f in fn], dtype=np.float64)
            fv_scaled = scaler.transform(fv.reshape(1, -1))
            preds = {}
            for name, model in models.items():
                try:
                    if name == "lgbm":
                        prob = float(model.predict(fv_scaled)[0])
                    else:
                        prob = float(model.predict_proba(fv_scaled)[0, 1])
                    preds[name] = prob
                except Exception:
                    pass
            snap_row = snaps.iloc[snap_idx].to_dict()
            mp["entries"][lbl] = {"preds": preds, "snap_row": snap_row}
        all_market_preds.append(mp)

    # Run simulation grid
    confs = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    eps = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    entry_pcts = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]

    results = simulate_replay(all_markets, models, scaler, fn, confs, eps,
                               args.capital, args.bet, entry_pcts)
    report(results, all_markets, all_market_preds)

    # Save to JSON
    Path("results").mkdir(exist_ok=True)
    out = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_markets": len(all_markets),
        "n_results": len(results),
        "datasets": [{"date": r["date"], "dur": r["dur"]} for r in recorded],
    }
    Path("results/replay_sim_report.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    P(f"  Saved: results/replay_sim_report.json")


if __name__ == "__main__":
    main()
