#!/usr/bin/env python3
"""
MULTI-DIMENSIONAL EDGE ANALYSIS SYSTEM

Conditional analysis across 5 dimensions:
  1. Model (catboost, rf, xgboost, ensemble, lgbm)
  2. Confidence threshold (55%-95%)
  3. Timing (entry_pct bins within market lifecycle)
  4. Share price (determines payout/EV)
  5. Market duration (5m vs 15m — completely separate regimes)

EV is calculated PER 1 SHARE (not per bet). This shows the true edge:
  EV/share = P(win) * (1 - price) - P(loss) * price
  If you buy 1 share at $0.50: win→$0.50 profit, lose→$0.50 loss
  EV/share > 0 means profitable per share regardless of bet size.

Usage:
    python analyze_live.py                        # today's data
    python analyze_live.py --date 2026-05-06      # specific date
    python analyze_live.py --all                   # all available days
    python analyze_live.py --bet 5.0              # custom bet size
"""

import argparse
import sys
import io
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

_OUT_FILE = None

def P(*args, **kwargs):
    """Print to console AND file simultaneously."""
    print(*args, **kwargs)
    if _OUT_FILE:
        print(*args, **kwargs, file=_OUT_FILE)


def load_ticks(date_str: str = None, all_days: bool = False) -> pd.DataFrame:
    """Load JSONL tick files → single DataFrame."""
    results_dir = Path("results")
    if not results_dir.exists():
        P("  ERROR: results/ directory not found"); sys.exit(1)

    if all_days:
        files = sorted(results_dir.glob("live_ticks_*.jsonl"))
    elif date_str:
        files = [results_dir / f"live_ticks_{date_str}.jsonl"]
    else:
        files = [results_dir / f"live_ticks_{datetime.now().strftime('%Y-%m-%d')}.jsonl"]

    frames = []
    for f in files:
        if not f.exists():
            P(f"  WARNING: {f.name} not found"); continue
        try:
            df = pd.read_json(f, lines=True)
            if len(df) > 0:
                frames.append(df)
                P(f"  Loaded {f.name}: {len(df)} ticks, {df.slug.nunique()} markets")
        except Exception as e:
            P(f"  ERROR loading {f.name}: {e}")

    if not frames:
        P("  No tick data found."); sys.exit(1)
    return pd.concat(frames, ignore_index=True)


def load_outcomes(date_str: str = None, all_days: bool = False) -> pd.DataFrame:
    """Load JSONL outcome files → DataFrame."""
    results_dir = Path("results")
    if all_days:
        files = sorted(results_dir.glob("live_outcomes_*.jsonl"))
    elif date_str:
        files = [results_dir / f"live_outcomes_{date_str}.jsonl"]
    else:
        files = [results_dir / f"live_outcomes_{datetime.now().strftime('%Y-%m-%d')}.jsonl"]

    frames = []
    for f in files:
        if not f.exists(): continue
        try:
            df = pd.read_json(f, lines=True)
            if len(df) > 0:
                # Deduplicate: keep last outcome per slug
                df = df.drop_duplicates("slug", keep="last")
                frames.append(df)
        except Exception:
            pass

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates("slug", keep="last")


def merge_ticks_outcomes(ticks: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    """Merge ticks with outcomes on slug. Vectorized."""
    if outcomes.empty:
        P("  WARNING: No outcomes found — cannot compute WR")
        return ticks

    # Keep only outcome column from outcomes
    out = outcomes[["slug", "outcome"]].copy()
    merged = ticks.merge(out, on="slug", how="inner")
    P(f"  Merged: {len(merged)} ticks with outcomes ({merged.slug.nunique()} markets)")
    return merged


def compute_all_ticks(df: pd.DataFrame, model: str, conf_thresh: float,
                      max_sp: float, min_sp: float, bet: float,
                      t_lo: float = 0.0, t_hi: float = 1.0) -> pd.DataFrame:
    """Vectorized: first qualifying tick per market = 1 realistic trade.

    Filters all ticks by (conf, SP, timing), then takes the FIRST matching
    tick per market (sorted by entry_pct). Once "entered", that market is
    done — no double-counting.

    This gives realistic trade counts: N trades = N markets.
    """
    if model not in df.columns or "outcome" not in df.columns:
        return pd.DataFrame()

    prob = df[model].values
    conf = np.maximum(prob, 1 - prob)
    model_up = prob > 0.5

    # Direction-aware share price: UP → yes_price, DOWN → no_price
    sp = np.where(model_up, df["yes"].values, df["no"].values)

    # ALL filters vectorized
    mask = (
        (conf >= conf_thresh) &
        (sp <= max_sp) & (sp >= min_sp) &
        (df["entry_pct"].values >= t_lo) & (df["entry_pct"].values < t_hi)
    )
    valid = df[mask].copy()
    if valid.empty:
        return pd.DataFrame()

    valid["_conf"] = conf[mask]
    valid["_sp"] = sp[mask]
    valid["_dir"] = np.where(model_up[mask], "UP", "DOWN")

    # DEDUP: first qualifying tick per market → 1 trade per market
    valid = valid.sort_values("entry_pct").drop_duplicates("slug", keep="first").copy()

    # Win/loss: prediction matches actual outcome
    valid["won"] = valid["_dir"] == valid["outcome"]
    shares = bet / valid["_sp"]
    valid["pnl"] = np.where(valid["won"], shares * (1.0 - valid["_sp"]), -bet)

    # Liquidity check: can we actually fill at this price?
    if "yes_depth" in valid.columns and "no_depth" in valid.columns:
        depth = np.where(valid["_dir"] == "UP", valid["yes_depth"].values, valid["no_depth"].values)
        valid["_depth"] = depth
        valid["_can_fill"] = depth >= shares.values
    else:
        valid["_depth"] = 0.0
        valid["_can_fill"] = True  # No data — assume fillable (legacy ticks)

    return valid


def report_threshold_table(df: pd.DataFrame, bet: float):
    """Section 1: ALL TICKS as entries — Model × Threshold (SP $0.01-$0.60)."""
    P(f"\n{'='*90}")
    P(f"  THRESHOLD TABLE — 1 TRADE/MARKET (first qualifying tick, SP $0.01-$0.60)")
    P(f"{'='*90}")
    P(f"  {'Model':<12} {'Conf>=':>6} {'Trades':>6} {'W':>5} {'L':>5} {'WR':>6} {'PnL':>9} {'EV/trade':>9} {'AvgSP':>7}")
    P(f"  {'-'*78}")

    models = [c for c in ["catboost", "lgbm", "rf", "xgboost", "ensemble"] if c in df.columns]
    thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    for model in models:
        for thresh in thresholds:
            t = compute_all_ticks(df, model, thresh, max_sp=0.60, min_sp=0.01, bet=bet)
            if t.empty or len(t) < 1:
                continue
            w = t.won.sum(); lo = len(t) - w
            wr = w / len(t) * 100
            P(f"  {model:<12} {thresh:>5.0%} {len(t):>6} {w:>5} {lo:>5} {wr:>5.1f}% "
              f"${t.pnl.sum():>+7.2f}  ${t.pnl.mean():>+6.3f} ${t._sp.mean():>.3f}")
        P("")


def report_fair_price_table(df: pd.DataFrame, bet: float):
    """Section 2: Fair prices only — SP $0.40-$0.55, all ticks."""
    P(f"\n{'='*90}")
    P(f"  FAIR PRICE TABLE — 1 TRADE/MARKET (SP $0.40-$0.55)")
    P(f"{'='*90}")
    P(f"  {'Model':<12} {'Conf>=':>6} {'Trades':>6} {'W':>5} {'L':>5} {'WR':>6} {'PnL':>9} {'EV/trade':>9} {'AvgSP':>7}")
    P(f"  {'-'*78}")

    models = [c for c in ["catboost", "lgbm", "rf", "xgboost", "ensemble"] if c in df.columns]
    thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    for model in models:
        for thresh in thresholds:
            t = compute_all_ticks(df, model, thresh, max_sp=0.55, min_sp=0.40, bet=bet)
            if t.empty or len(t) < 1:
                continue
            w = t.won.sum(); lo = len(t) - w
            wr = w / len(t) * 100
            P(f"  {model:<12} {thresh:>5.0%} {len(t):>6} {w:>5} {lo:>5} {wr:>5.1f}% "
              f"${t.pnl.sum():>+7.2f}  ${t.pnl.mean():>+6.3f} ${t._sp.mean():>.3f}")
        P("")


def report_entry_timing(df: pd.DataFrame, bet: float):
    """Section 3: Entry timing × confidence — ALL ticks, catboost, SP $0.40-$0.55."""
    P(f"\n{'='*90}")
    P(f"  ENTRY TIMING × CONFIDENCE — 1 TRADE/MARKET (catboost, SP $0.40-$0.55)")
    P(f"{'='*90}")

    timing_bins = [(0.0, 0.10), (0.10, 0.20), (0.20, 0.40), (0.40, 0.60), (0.60, 0.80), (0.80, 1.0)]
    thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

    P(f"  {'Timing':<10} {'Conf>=':>6} {'Trades':>6} {'W':>5} {'L':>5} {'WR':>6} {'EV/trade':>9} {'AvgSP':>7}")
    P(f"  {'-'*65}")
    for t_lo, t_hi in timing_bins:
        for thresh in thresholds:
            t = compute_all_ticks(df, "catboost", thresh, max_sp=0.55, min_sp=0.40,
                                  bet=bet, t_lo=t_lo, t_hi=t_hi)
            if t.empty or len(t) < 1:
                continue
            w = t.won.sum(); lo = len(t) - w
            wr = w / len(t) * 100
            P(f"  {t_lo*100:.0f}-{t_hi*100:.0f}%     {thresh:>5.0%} {len(t):>6} {w:>5} {lo:>5} {wr:>5.1f}% "
              f" ${t.pnl.mean():>+6.3f} ${t._sp.mean():>.3f}")
        P("")


def report_share_price_bins(df: pd.DataFrame, bet: float):
    """Section 4: Share price × confidence — ALL ticks, catboost."""
    P(f"\n{'='*90}")
    P(f"  SHARE PRICE × CONFIDENCE — 1 TRADE/MARKET (catboost)")
    P(f"{'='*90}")

    price_bins = [(0.01, 0.20), (0.20, 0.35), (0.35, 0.45), (0.45, 0.50), (0.50, 0.55), (0.55, 0.60)]
    thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

    P(f"  {'Price':<14} {'Conf>=':>6} {'Trades':>6} {'W':>5} {'L':>5} {'WR':>6} {'EV/trade':>9}")
    P(f"  {'-'*65}")
    for p_lo, p_hi in price_bins:
        for thresh in thresholds:
            t = compute_all_ticks(df, "catboost", thresh, max_sp=p_hi, min_sp=p_lo, bet=bet)
            if t.empty or len(t) < 1:
                continue
            w = t.won.sum(); lo = len(t) - w
            wr = w / len(t) * 100
            P(f"  ${p_lo:.2f}-${p_hi:.2f}   {thresh:>5.0%} {len(t):>6} {w:>5} {lo:>5} {wr:>5.1f}%  ${t.pnl.mean():>+6.3f}")
        P("")


def report_direction(df: pd.DataFrame, bet: float):
    """Section 5: Direction UP vs DOWN — all ticks."""
    P(f"\n{'='*90}")
    P(f"  DIRECTION — 1 TRADE/MARKET, catboost, SP $0.40-$0.55")
    P(f"{'='*90}")

    t = compute_all_ticks(df, "catboost", 0.60, max_sp=0.55, min_sp=0.40, bet=bet)
    if t.empty:
        P("  No ticks"); return

    P(f"  {'Dir':<6} {'Conf>=':>6} {'Trades':>6} {'W':>5} {'L':>5} {'WR':>6} {'EV/trade':>9}")
    P(f"  {'-'*58}")
    for d in ["UP", "DOWN"]:
        for thresh in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
            t2 = compute_all_ticks(df, "catboost", thresh, max_sp=0.55, min_sp=0.40, bet=bet)
            if t2.empty: continue
            sub = t2[t2._dir == d]
            if len(sub) == 0: continue
            w = sub.won.sum(); lo = len(sub) - w
            wr = w / len(sub) * 100
            P(f"  {d:<6} {thresh:>5.0%} {len(sub):>6} {w:>5} {lo:>5} {wr:>5.1f}%  ${sub.pnl.mean():>+6.3f}")
        P("")


def report_consensus(df: pd.DataFrame, bet: float):
    """Section 6: Model consensus — ALL ticks, how many models agree."""
    P(f"\n{'='*90}")
    P(f"  MODEL CONSENSUS — 1 TRADE/MARKET (SP $0.40-$0.55)")
    P(f"{'='*90}")

    models = [c for c in ["catboost", "lgbm", "rf", "xgboost"] if c in df.columns]
    if len(models) < 2 or "catboost" not in df.columns or "outcome" not in df.columns:
        P("  Not enough data"); return

    cb_up = df["catboost"] > 0.5
    cb_conf = np.maximum(df["catboost"], 1 - df["catboost"])

    agree_count = np.zeros(len(df))
    for m in models:
        m_up = df[m] > 0.5
        m_conf = np.maximum(df[m], 1 - df[m])
        agree_count += ((m_up == cb_up) & (m_conf >= 0.55)).astype(float)

    df2 = df.copy()
    df2["_agree"] = agree_count
    sp = np.where(cb_up, df2["yes"].values, df2["no"].values)
    df2["_sp"] = sp
    df2["_dir"] = np.where(cb_up, "UP", "DOWN")
    df2["_conf"] = cb_conf.values

    # Filter: SP $0.40-$0.55
    mask = (df2["_sp"] >= 0.40) & (df2["_sp"] <= 0.55) & (df2["_conf"] >= 0.55)
    valid = df2[mask].copy()

    if valid.empty:
        P("  No ticks"); return

    # DEDUP: first qualifying tick per market
    valid = valid.sort_values("entry_pct").drop_duplicates("slug", keep="first").copy()

    valid["won"] = valid["_dir"] == valid["outcome"]
    valid["pnl"] = np.where(valid["won"], bet / valid["_sp"] * (1 - valid["_sp"]), -bet)

    P(f"  {'Agree':<8} {'Trades':>6} {'W':>5} {'L':>5} {'WR':>6} {'EV/trade':>9} {'AvgConf':>8}")
    P(f"  {'-'*55}")
    for n_agree in sorted(valid._agree.unique()):
        sub = valid[valid._agree == n_agree]
        if len(sub) == 0: continue
        w = sub.won.sum(); lo = len(sub) - w
        wr = w / len(sub) * 100
        P(f"  {int(n_agree)}/{len(models)}     {len(sub):>6} {w:>5} {lo:>5} {wr:>5.1f}% "
          f"${sub.pnl.mean():>+6.3f} {sub._conf.mean():>7.1%}")


def report_best_combos(df: pd.DataFrame, bet: float, dur_filter: int = None):
    """Section 7: replay_v2-style grid — TOP combos by WR / EV / PnL."""
    dur_label = f" — {dur_filter}min ONLY" if dur_filter else ""
    work_df = df[df.dur_min == dur_filter].copy() if dur_filter and "dur_min" in df.columns else df
    if work_df.empty or "outcome" not in work_df.columns:
        P("  No outcomes"); return
    n_mkts = work_df.slug.nunique()
    if n_mkts < 3:
        return

    P(f"\n{'='*90}")
    P(f"  BEST COMBOS GRID{dur_label} ({n_mkts} mkts, model × conf × timing × maxSP)")
    P(f"{'='*90}")

    models = [c for c in ["catboost", "lgbm", "rf", "xgboost", "ensemble"] if c in work_df.columns]

    timing_bins = [(0.0, 1.0), (0.0, 0.20), (0.0, 0.40), (0.20, 0.60), (0.40, 0.80), (0.60, 1.0)]
    max_sps     = [0.30, 0.40, 0.45, 0.50, 0.55, 0.60]
    thresholds  = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

    combos = []
    for model in models:
        for thresh in thresholds:
            for t_lo, t_hi in timing_bins:
                for max_sp in max_sps:
                    t = compute_all_ticks(work_df, model, thresh, max_sp=max_sp, min_sp=0.01,
                                          bet=bet, t_lo=t_lo, t_hi=t_hi)
                    if t.empty or len(t) < 3:
                        continue
                    w = int(t.won.sum())
                    combos.append({
                        "model": model, "conf": thresh,
                        "timing": f"{t_lo*100:.0f}-{t_hi*100:.0f}%",
                        "maxSP": max_sp,
                        "n": len(t), "w": w, "wr": w / len(t) * 100,
                        "ev": t.pnl.mean(), "pnl": t.pnl.sum(),
                        "avg_sp": t._sp.mean(),
                    })

    if not combos:
        P("  No combos found (need more data)"); return

    cdf = pd.DataFrame(combos)
    filt = cdf[cdf.n >= 3]
    hdr = f"  {'Model':<12} {'Conf':>5} {'Timing':<10} {'MaxSP':>6} {'N':>4} {'WR':>6} {'PnL':>9} {'EV':>7} {'AvgSP':>7}"

    P(f"\n  --- TOP 30 BY WIN RATE ---")
    P(hdr)
    for _, r in filt.sort_values("wr", ascending=False).head(30).iterrows():
        P(f"  {r['model']:<12} {r['conf']:>4.0%} {r['timing']:<10} ${r['maxSP']:.2f} "
          f"{r['n']:>4} {r['wr']:>5.1f}% ${r['pnl']:>+7.2f} ${r['ev']:>+5.2f} ${r['avg_sp']:>.3f}")

    P(f"\n  --- TOP 30 BY EV PER TRADE ---")
    P(hdr)
    for _, r in filt.sort_values("ev", ascending=False).head(30).iterrows():
        P(f"  {r['model']:<12} {r['conf']:>4.0%} {r['timing']:<10} ${r['maxSP']:.2f} "
          f"{r['n']:>4} {r['wr']:>5.1f}% ${r['pnl']:>+7.2f} ${r['ev']:>+5.2f} ${r['avg_sp']:>.3f}")

    P(f"\n  --- TOP 30 BY TOTAL PNL ---")
    P(hdr)
    for _, r in filt.sort_values("pnl", ascending=False).head(30).iterrows():
        P(f"  {r['model']:<12} {r['conf']:>4.0%} {r['timing']:<10} ${r['maxSP']:.2f} "
          f"{r['n']:>4} {r['wr']:>5.1f}% ${r['pnl']:>+7.2f} ${r['ev']:>+5.2f} ${r['avg_sp']:>.3f}")

    P(f"\n  --- WORST 15 BY EV ---")
    P(hdr)
    for _, r in filt.sort_values("ev", ascending=True).head(15).iterrows():
        P(f"  {r['model']:<12} {r['conf']:>4.0%} {r['timing']:<10} ${r['maxSP']:.2f} "
          f"{r['n']:>4} {r['wr']:>5.1f}% ${r['pnl']:>+7.2f} ${r['ev']:>+5.2f} ${r['avg_sp']:>.3f}")


def report_consensus_combos(df: pd.DataFrame, bet: float, dur_filter: int = None):
    """Section 8: Model consensus combos — min_agree × conf × timing."""
    dur_label = f" — {dur_filter}min ONLY" if dur_filter else ""
    work_df = df[df.dur_min == dur_filter].copy() if dur_filter and "dur_min" in df.columns else df
    if work_df.empty or "outcome" not in work_df.columns:
        return
    n_mkts = work_df.slug.nunique()
    if n_mkts < 3:
        return

    P(f"\n{'='*90}")
    P(f"  MODEL CONSENSUS COMBOS{dur_label} ({n_mkts} mkts)")
    P(f"{'='*90}")

    base_models = [c for c in ["catboost", "lgbm", "rf", "xgboost"] if c in work_df.columns]
    if len(base_models) < 2:
        P("  Not enough data"); return

    confs = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    timing_bins = [(0.0, 1.0), (0.0, 0.40), (0.20, 0.60), (0.40, 0.80), (0.60, 1.0)]
    max_sps = [0.45, 0.50, 0.55]

    # Precompute per-model direction and confidence arrays
    model_ups = {m: work_df[m].values > 0.5 for m in base_models}
    model_confs = {m: np.maximum(work_df[m].values, 1 - work_df[m].values) for m in base_models}

    results = []
    for min_agree in [2, 3, 4]:
        if min_agree > len(base_models): continue
        for ct in confs:
            cb_up = model_ups["catboost"]
            agree = np.zeros(len(work_df))
            for m in base_models:
                agree += ((model_ups[m] == cb_up) & (model_confs[m] >= ct)).astype(float)

            mask_agree = agree >= min_agree
            if mask_agree.sum() == 0: continue

            cb_conf = model_confs["catboost"]
            sp = np.where(cb_up, work_df["yes"].values, work_df["no"].values)

            for t_lo, t_hi in timing_bins:
                for max_sp in max_sps:
                    full_mask = (
                        mask_agree &
                        (cb_conf >= ct) &
                        (sp >= 0.01) & (sp <= max_sp) &
                        (work_df["entry_pct"].values >= t_lo) & (work_df["entry_pct"].values < t_hi)
                    )
                    sub = work_df[full_mask].copy()
                    if sub.empty: continue

                    sub["_sp"] = sp[full_mask]
                    sub["_dir"] = np.where(cb_up[full_mask], "UP", "DOWN")
                    sub = sub.sort_values("entry_pct").drop_duplicates("slug", keep="first")
                    if len(sub) < 3: continue

                    sub["won"] = sub["_dir"] == sub["outcome"]
                    sub["pnl"] = np.where(sub["won"], bet / sub["_sp"] * (1 - sub["_sp"]), -bet)
                    w = int(sub.won.sum())
                    results.append({
                        "agree": f"{min_agree}/{len(base_models)}",
                        "conf": ct,
                        "timing": f"{t_lo*100:.0f}-{t_hi*100:.0f}%",
                        "maxSP": max_sp,
                        "n": len(sub), "w": w, "wr": w / len(sub) * 100,
                        "ev": sub.pnl.mean(), "pnl": sub.pnl.sum(),
                    })

    if not results:
        P("  No consensus combos found"); return

    rdf = pd.DataFrame(results)
    P(f"  {'Agree':<6} {'Conf>=':>6} {'Timing':<10} {'MaxSP':>6} {'N':>4} {'WR':>6} {'PnL':>9} {'EV':>7}")
    P(f"  {'-'*65}")

    show = rdf[rdf.n >= 3].sort_values(["agree", "conf", "timing"])
    for _, r in show.iterrows():
        P(f"  {r['agree']:<6} {r['conf']:>5.0%} {r['timing']:<10} ${r['maxSP']:.2f} "
          f"{r['n']:>4} {r['wr']:>5.1f}% ${r['pnl']:>+7.2f} ${r['ev']:>+5.2f}")

    P(f"\n  TOP 15 CONSENSUS BY WR (N>=3):")
    best = rdf[rdf.n >= 3].sort_values("wr", ascending=False).head(15)
    for _, r in best.iterrows():
        P(f"  {r['agree']} models, conf>={r['conf']:.0%}, {r['timing']}, SP<=${r['maxSP']:.2f}: "
          f"{r['n']}t WR={r['wr']:.1f}% PnL=${r['pnl']:+.2f} EV=${r['ev']:+.2f}")


def report_entry_x_shareprice(df: pd.DataFrame, bet: float):
    """Section 9: Entry timing × share price bucket — ensemble conf>=0.65."""
    if "outcome" not in df.columns: return
    P(f"\n{'='*90}")
    P(f"  ENTRY TIMING × SHARE PRICE (ensemble conf>=0.65)")
    P(f"{'='*90}")

    model = "ensemble" if "ensemble" in df.columns else "catboost"
    sp_bins = [(0.01, 0.30), (0.30, 0.40), (0.40, 0.50), (0.50, 0.55), (0.55, 0.60)]
    timing_bins = [(0.0, 0.20), (0.20, 0.40), (0.40, 0.60), (0.60, 0.80), (0.80, 1.0)]

    P(f"  {'Timing':<10} {'SP range':<14} {'N':>4} {'WR':>6} {'PnL':>9} {'EV':>7}")
    P(f"  {'-'*60}")
    for t_lo, t_hi in timing_bins:
        for p_lo, p_hi in sp_bins:
            t = compute_all_ticks(df, model, 0.65, max_sp=p_hi, min_sp=p_lo,
                                  bet=bet, t_lo=t_lo, t_hi=t_hi)
            if t.empty or len(t) < 2: continue
            w = t.won.sum()
            wr = w / len(t) * 100
            P(f"  {t_lo*100:.0f}-{t_hi*100:.0f}%     ${p_lo:.2f}-${p_hi:.2f}    "
              f"{len(t):>4} {wr:>5.1f}% ${t.pnl.sum():>+7.2f} ${t.pnl.mean():>+5.2f}")
        P("")


def report_sp_sweep(df: pd.DataFrame, bet: float, dur_filter: int = None):
    """Pivot table: WR%(N) for model × conf × SP cap. Optionally filter by duration."""
    dur_label = f"{dur_filter}min ONLY" if dur_filter else "ALL"
    work_df = df[df.dur_min == dur_filter].copy() if dur_filter and "dur_min" in df.columns else df
    if work_df.empty or "outcome" not in work_df.columns:
        return
    n_mkts = work_df.slug.nunique()
    if n_mkts < 3:
        return

    sp_caps = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    models = [c for c in ["catboost", "rf", "xgboost", "ensemble"] if c in work_df.columns]
    confs = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    P(f"\n{'='*110}")
    P(f"  SP CAP SWEEP — {dur_label} ({n_mkts} markets)")
    P(f"{'='*110}")

    for model in models:
        # --- WR%(N) matrix ---
        P(f"\n  {model.upper()} — WR%(N), SP $0.10 → cap:")
        hdr = f"  {'Conf':>5}" + "".join(f"  {'≤$'+f'{s:.2f}':>10}" for s in sp_caps)
        P(hdr)
        P(f"  {'-'*(7 + 12*len(sp_caps))}")
        for conf in confs:
            cells = []
            for sp_cap in sp_caps:
                t = compute_all_ticks(work_df, model, conf, max_sp=sp_cap, min_sp=0.10, bet=bet)
                if t.empty or len(t) < 2:
                    cells.append(f"{'—':>10}")
                else:
                    wr = t.won.sum() / len(t) * 100
                    cells.append(f"{wr:.0f}%({len(t)})".rjust(10))
            P(f"  {conf:>4.0%}" + "  ".join(cells))

        # --- EV matrix ---
        P(f"\n  {model.upper()} — EV$/trade:")
        P(hdr)
        P(f"  {'-'*(7 + 12*len(sp_caps))}")
        for conf in confs:
            cells = []
            for sp_cap in sp_caps:
                t = compute_all_ticks(work_df, model, conf, max_sp=sp_cap, min_sp=0.10, bet=bet)
                if t.empty or len(t) < 2:
                    cells.append(f"{'—':>10}")
                else:
                    cells.append(f"${t.pnl.mean():>+.2f}".rjust(10))
            P(f"  {conf:>4.0%}" + "  ".join(cells))

        # --- Total PnL matrix ---
        P(f"\n  {model.upper()} — Total PnL:")
        P(hdr)
        P(f"  {'-'*(7 + 12*len(sp_caps))}")
        for conf in confs:
            cells = []
            for sp_cap in sp_caps:
                t = compute_all_ticks(work_df, model, conf, max_sp=sp_cap, min_sp=0.10, bet=bet)
                if t.empty or len(t) < 2:
                    cells.append(f"{'—':>10}")
                else:
                    cells.append(f"${t.pnl.sum():>+.0f}".rjust(10))
            P(f"  {conf:>4.0%}" + "  ".join(cells))
        P("")


def report_duration_breakdown(df: pd.DataFrame, bet: float):
    """WR and EV broken down by market duration (5min vs 15min)."""
    if "outcome" not in df.columns or "dur_min" not in df.columns:
        return

    P(f"\n{'='*90}")
    P(f"  DURATION BREAKDOWN — Performance by market duration")
    P(f"{'='*90}")

    models = [c for c in ["catboost", "rf", "xgboost", "ensemble"] if c in df.columns]
    durations = sorted(df.dur_min.unique())

    P(f"  {'Model':<12} {'Dur':>5} {'Conf>=':>6} {'N':>4} {'WR':>6} {'PnL':>9} {'EV':>7} {'AvgSP':>7}")
    P(f"  {'-'*70}")
    for model in models[:3]:
        for dur in durations:
            dur_df = df[df.dur_min == dur]
            for thresh in [0.65, 0.70, 0.75, 0.80]:
                t = compute_all_ticks(dur_df, model, thresh, max_sp=0.55, min_sp=0.01, bet=bet)
                if t.empty or len(t) < 3: continue
                w = t.won.sum()
                wr = w / len(t) * 100
                P(f"  {model:<12} {dur:>3}m {thresh:>5.0%} {len(t):>4} {wr:>5.1f}% "
                  f"${t.pnl.sum():>+7.2f} ${t.pnl.mean():>+5.2f} ${t._sp.mean():>.3f}")
        P("")


def report_hourly_performance(df: pd.DataFrame, bet: float):
    """WR and PnL broken down by hour of day."""
    if "outcome" not in df.columns:
        return

    P(f"\n{'='*90}")
    P(f"  HOURLY PERFORMANCE (ensemble conf>=70%, SP $0.40-$0.55)")
    P(f"{'='*90}")

    model = "ensemble" if "ensemble" in df.columns else "catboost"
    df_c = df.copy()
    df_c["_hour"] = pd.to_datetime(df_c["ts"], unit="s").dt.hour

    P(f"  {'Hour':>6} {'N':>4} {'WR':>6} {'PnL':>9} {'EV':>7} {'Markets':>8}")
    P(f"  {'-'*50}")
    for hour in sorted(df_c["_hour"].unique()):
        hour_df = df_c[df_c["_hour"] == hour]
        t = compute_all_ticks(hour_df, model, 0.70, max_sp=0.55, min_sp=0.40, bet=bet)
        if t.empty or len(t) < 2: continue
        w = t.won.sum()
        wr = w / len(t) * 100
        P(f"  {hour:>4}:00 {len(t):>4} {wr:>5.1f}% ${t.pnl.sum():>+7.2f} ${t.pnl.mean():>+5.2f} {t.slug.nunique():>8}")


def report_streak_analysis(df: pd.DataFrame, bet: float):
    """Win/loss streak analysis and profit curve stats."""
    if "outcome" not in df.columns:
        return

    P(f"\n{'='*90}")
    P(f"  STREAK & PROFIT CURVE ANALYSIS (ensemble conf>=70%, SP $0.40-$0.55)")
    P(f"{'='*90}")

    model = "ensemble" if "ensemble" in df.columns else "catboost"
    t = compute_all_ticks(df, model, 0.70, max_sp=0.55, min_sp=0.40, bet=bet)
    if t.empty or len(t) < 5:
        P("  Insufficient trades"); return

    t = t.sort_values("ts").reset_index(drop=True)
    wins = t["won"].values
    pnls = t["pnl"].values
    cum_pnl = np.cumsum(pnls)

    # Streaks
    max_win_streak = max_loss_streak = cur_win = cur_loss = 0
    for w in wins:
        if w:
            cur_win += 1; cur_loss = 0
            max_win_streak = max(max_win_streak, cur_win)
        else:
            cur_loss += 1; cur_win = 0
            max_loss_streak = max(max_loss_streak, cur_loss)

    # Drawdown
    peak = np.maximum.accumulate(cum_pnl)
    drawdown = peak - cum_pnl
    max_dd = drawdown.max()

    # Profit factor
    gross_profit = pnls[pnls > 0].sum()
    gross_loss = abs(pnls[pnls < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    P(f"  Total trades:       {len(t)}")
    P(f"  Win rate:           {wins.mean()*100:.1f}%")
    P(f"  Total PnL:          ${cum_pnl[-1]:+.2f}")
    P(f"  Avg PnL/trade:      ${pnls.mean():+.3f}")
    P(f"  Best trade:         ${pnls.max():+.2f}")
    P(f"  Worst trade:        ${pnls.min():+.2f}")
    P(f"  Max win streak:     {max_win_streak}")
    P(f"  Max loss streak:    {max_loss_streak}")
    P(f"  Max drawdown:       ${max_dd:.2f}")
    P(f"  Profit factor:      {profit_factor:.2f}")
    P(f"  Gross profit:       ${gross_profit:.2f}")
    P(f"  Gross loss:         ${gross_loss:.2f}")
    P(f"  Peak PnL:           ${peak[-1]:+.2f}")
    P(f"  Sharpe (approx):    {pnls.mean() / pnls.std():.2f}" if pnls.std() > 0 else "  Sharpe: N/A")

    # Show cumulative PnL at checkpoints
    P(f"\n  Cumulative PnL checkpoints:")
    checkpoints = np.linspace(0, len(t)-1, min(10, len(t)), dtype=int)
    for i in checkpoints:
        P(f"    Trade {i+1:>4}/{len(t)}: ${cum_pnl[i]:>+7.2f} (WR: {wins[:i+1].mean()*100:.1f}%)")


def report_executive_summary(df: pd.DataFrame, bet: float):
    """Final executive summary with best config recommendation."""
    if "outcome" not in df.columns:
        return

    P(f"\n{'='*90}")
    P(f"  ★ EXECUTIVE SUMMARY — RECOMMENDED CONFIG")
    P(f"{'='*90}")

    models = [c for c in ["catboost", "lgbm", "rf", "xgboost", "ensemble"] if c in df.columns]
    best_results = []

    # Find best single-model config
    for model in models:
        for thresh in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
            t = compute_all_ticks(df, model, thresh, max_sp=0.55, min_sp=0.40, bet=bet)
            if t.empty or len(t) < 5: continue
            w = t.won.sum()
            best_results.append({
                "model": model, "conf": thresh,
                "n": len(t), "wr": w/len(t)*100,
                "pnl": t.pnl.sum(), "ev": t.pnl.mean(),
                "avg_sp": t._sp.mean()
            })

    if not best_results:
        P("  Insufficient data for summary"); return

    bdf = pd.DataFrame(best_results)
    # Best by EV with decent sample
    good = bdf[bdf.n >= 10].sort_values("ev", ascending=False)
    if good.empty:
        good = bdf.sort_values("ev", ascending=False)

    P(f"\n  TOP 5 CONFIGS (by EV, N>=10, SP $0.40-$0.55):")
    P(f"  {'Model':<12} {'Conf':>5} {'N':>4} {'WR':>6} {'PnL':>9} {'EV':>7} {'AvgSP':>7}")
    P(f"  {'-'*60}")
    for _, r in good.head(5).iterrows():
        P(f"  {r['model']:<12} {r['conf']:>4.0%} {r['n']:>4} {r['wr']:>5.1f}% "
          f"${r['pnl']:>+7.2f} ${r['ev']:>+5.2f} ${r['avg_sp']:>.3f}")

    # Best overall
    top = good.iloc[0]
    P(f"\n  ★ RECOMMENDED: {top['model']} @ {top['conf']:.0%} conf, SP $0.40-$0.55")
    P(f"    → {int(top['n'])} trades, {top['wr']:.1f}% WR, ${top['pnl']:+.2f} PnL, ${top['ev']:+.3f}/trade")

    # Consensus recommendation
    base_models = [c for c in ["catboost", "lgbm", "rf", "xgboost"] if c in df.columns]
    if len(base_models) >= 3:
        model_ups = {m: df[m].values > 0.5 for m in base_models}
        model_confs = {m: np.maximum(df[m].values, 1 - df[m].values) for m in base_models}
        cb_up = model_ups["catboost"]

        # Try 3/4 agreement at 70%
        agree = np.zeros(len(df))
        for m in base_models:
            agree += ((model_ups[m] == cb_up) & (model_confs[m] >= 0.70)).astype(float)

        mask = (
            (agree >= 3) &
            (model_confs["catboost"] >= 0.70) &
            (np.where(cb_up, df["yes"].values, df["no"].values) >= 0.40) &
            (np.where(cb_up, df["yes"].values, df["no"].values) <= 0.55)
        )
        sub = df[mask].copy()
        if not sub.empty:
            sp = np.where(cb_up[mask], sub["yes"].values, sub["no"].values)
            sub["_sp"] = sp
            sub["_dir"] = np.where(cb_up[mask], "UP", "DOWN")
            sub = sub.sort_values("entry_pct").drop_duplicates("slug", keep="first")
            if len(sub) >= 5:
                sub["won"] = sub["_dir"] == sub["outcome"]
                sub["pnl"] = np.where(sub["won"], bet / sub["_sp"] * (1 - sub["_sp"]), -bet)
                wr = sub.won.mean() * 100
                P(f"\n  ★ CONSENSUS (3/4 agree @ 70%): {len(sub)} trades, {wr:.1f}% WR, "
                  f"${sub.pnl.sum():+.2f} PnL, ${sub.pnl.mean():+.3f}/trade")

    # Overall stats
    n_markets = df.slug.nunique()
    P(f"\n  Dataset: {len(df):,} ticks across {n_markets} markets")
    ts_range = pd.to_datetime(df["ts"], unit="s")
    hours = (ts_range.max() - ts_range.min()).total_seconds() / 3600
    P(f"  Duration: {hours:.1f} hours of recording")
    P(f"  Markets/hour: {n_markets / max(hours, 0.1):.1f}")


def report_optimal_config(df: pd.DataFrame, bet: float):
    """Find absolute best config per duration, per model — exhaustive scan."""
    if "outcome" not in df.columns:
        return

    P(f"\n{'='*110}")
    P(f"  OPTIMAL CONFIG FINDER — exhaustive scan per duration")
    P(f"{'='*110}")

    models = [c for c in ["catboost", "rf", "xgboost", "ensemble"] if c in df.columns]
    confs = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    sp_caps = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    timing_bins = [(0.0, 1.0), (0.0, 0.40), (0.20, 0.60), (0.40, 0.80), (0.60, 1.0)]

    durations = [None]  # None = ALL
    if "dur_min" in df.columns:
        durations += sorted(df.dur_min.unique())

    for dur in durations:
        dur_label = f"{dur}min" if dur else "ALL"
        work_df = df[df.dur_min == dur].copy() if dur else df
        if work_df.empty or work_df.slug.nunique() < 3:
            continue

        all_combos = []
        for model in models:
            for conf in confs:
                for sp_cap in sp_caps:
                    for t_lo, t_hi in timing_bins:
                        t = compute_all_ticks(work_df, model, conf, max_sp=sp_cap, min_sp=0.10,
                                              bet=bet, t_lo=t_lo, t_hi=t_hi)
                        if t.empty or len(t) < 5:
                            continue
                        w = int(t.won.sum())
                        all_combos.append({
                            "model": model, "conf": conf,
                            "sp_cap": sp_cap,
                            "timing": f"{t_lo*100:.0f}-{t_hi*100:.0f}%",
                            "n": len(t), "wr": w / len(t) * 100,
                            "ev": t.pnl.mean(), "pnl": t.pnl.sum(),
                        })

        if not all_combos:
            continue

        cdf = pd.DataFrame(all_combos)
        hdr = f"    {'Model':<11} {'Conf':>5} {'SP≤':>5} {'Timing':<10} {'N':>4} {'WR':>6} {'PnL':>9} {'EV':>7}"

        P(f"\n  --- {dur_label} ({work_df.slug.nunique()} markets) ---")

        # Best by WR (N>=10)
        good = cdf[cdf.n >= 10].sort_values("wr", ascending=False)
        if not good.empty:
            P(f"\n    TOP 5 by WR (N>=10):")
            P(hdr)
            for _, r in good.head(5).iterrows():
                P(f"    {r['model']:<11} {r['conf']:>4.0%} ${r['sp_cap']:.2f} {r['timing']:<10} "
                  f"{r['n']:>4} {r['wr']:>5.1f}% ${r['pnl']:>+7.2f} ${r['ev']:>+5.2f}")

        # Best by EV (N>=10)
        good_ev = cdf[cdf.n >= 10].sort_values("ev", ascending=False)
        if not good_ev.empty:
            P(f"\n    TOP 5 by EV (N>=10):")
            P(hdr)
            for _, r in good_ev.head(5).iterrows():
                P(f"    {r['model']:<11} {r['conf']:>4.0%} ${r['sp_cap']:.2f} {r['timing']:<10} "
                  f"{r['n']:>4} {r['wr']:>5.1f}% ${r['pnl']:>+7.2f} ${r['ev']:>+5.2f}")

        # Best by PnL (N>=10)
        good_pnl = cdf[cdf.n >= 10].sort_values("pnl", ascending=False)
        if not good_pnl.empty:
            P(f"\n    TOP 5 by PnL (N>=10):")
            P(hdr)
            for _, r in good_pnl.head(5).iterrows():
                P(f"    {r['model']:<11} {r['conf']:>4.0%} ${r['sp_cap']:.2f} {r['timing']:<10} "
                  f"{r['n']:>4} {r['wr']:>5.1f}% ${r['pnl']:>+7.2f} ${r['ev']:>+5.2f}")

        # Best balanced (WR>=80% AND N>=10)
        balanced = cdf[(cdf.wr >= 80) & (cdf.n >= 10)].sort_values("pnl", ascending=False)
        if not balanced.empty:
            P(f"\n    BEST BALANCED (WR>=80%, N>=10) — top 5:")
            P(hdr)
            for _, r in balanced.head(5).iterrows():
                P(f"    {r['model']:<11} {r['conf']:>4.0%} ${r['sp_cap']:.2f} {r['timing']:<10} "
                  f"{r['n']:>4} {r['wr']:>5.1f}% ${r['pnl']:>+7.2f} ${r['ev']:>+5.2f}")


def report_bet_projection(df: pd.DataFrame, bet: float):
    """Show PnL projections at different bet sizes using best config."""
    if "outcome" not in df.columns:
        return

    P(f"\n{'='*90}")
    P(f"  BET SIZE PROJECTION — PnL at $2, $5, $10, $20, $50")
    P(f"{'='*90}")

    models = [c for c in ["catboost", "rf", "xgboost", "ensemble"] if c in df.columns]
    best_configs = [
        ("catboost", 0.85, 0.55, 0.10),
        ("catboost", 0.80, 0.55, 0.10),
        ("xgboost", 0.80, 0.55, 0.10),
        ("rf", 0.85, 0.55, 0.10),
    ]
    bet_sizes = [2.0, 5.0, 10.0, 20.0, 50.0]

    P(f"  {'Config':<30}" + "".join(f"  {'$'+f'{b:.0f}':>8}" for b in bet_sizes))
    P(f"  {'-'*(30 + 10*len(bet_sizes))}")

    for model, conf, sp_max, sp_min in best_configs:
        if model not in df.columns:
            continue
        label = f"{model} {conf:.0%} ≤${sp_max}"
        cells = []
        for b in bet_sizes:
            t = compute_all_ticks(df, model, conf, max_sp=sp_max, min_sp=sp_min, bet=b)
            if t.empty or len(t) < 3:
                cells.append(f"{'—':>8}")
            else:
                wr = t.won.sum() / len(t) * 100
                cells.append(f"${t.pnl.sum():>+.0f}".rjust(8))
        P(f"  {label:<30}" + "  ".join(cells))

    P(f"\n  Note: PnL scales linearly with bet size (same WR, same number of trades).")
    P(f"  Actual slippage/liquidity may differ at higher bet sizes.")

    # Duration-specific projection
    if "dur_min" in df.columns:
        P(f"\n  --- 5min markets only ---")
        df5 = df[df.dur_min == 5]
        if not df5.empty and df5.slug.nunique() >= 3:
            P(f"  {'Config':<30}" + "".join(f"  {'$'+f'{b:.0f}':>8}" for b in bet_sizes))
            P(f"  {'-'*(30 + 10*len(bet_sizes))}")
            for model, conf, sp_max, sp_min in best_configs:
                if model not in df5.columns:
                    continue
                label = f"{model} {conf:.0%} ≤${sp_max}"
                cells = []
                for b in bet_sizes:
                    t = compute_all_ticks(df5, model, conf, max_sp=sp_max, min_sp=sp_min, bet=b)
                    if t.empty or len(t) < 3:
                        cells.append(f"{'—':>8}")
                    else:
                        cells.append(f"${t.pnl.sum():>+.0f}".rjust(8))
                P(f"  {label:<30}" + "  ".join(cells))


def report_liquidity(df: pd.DataFrame, bet: float):
    """Section: Liquidity check — can trades actually fill at recorded depth?"""
    has_depth = "yes_depth" in df.columns and "no_depth" in df.columns
    P(f"\n{'='*90}")
    P(f"  LIQUIDITY ANALYSIS (depth check @ ${bet:.2f} bet)")
    P(f"{'='*90}")

    if not has_depth or df["yes_depth"].sum() == 0:
        P("  ⚠ No depth data in ticks (old format). Depth recording starts next session.")
        P("  Once live trader records yes_depth/no_depth, this section will show:")
        P("    - How many trades could actually fill at orderbook depth")
        P("    - Fillable vs unfillable WR comparison")
        P("    - Liquidity-filtered best combos")
        return

    models = [c for c in ["catboost", "lgbm", "rf", "xgboost", "ensemble"] if c in df.columns]
    if "outcome" not in df.columns:
        P("  No outcomes"); return

    P(f"\n  {'Model':<12} {'Conf>=':>6} {'Trades':>6} {'Fillable':>9} {'Fill%':>6} "
      f"{'WR(all)':>8} {'WR(fill)':>9} {'EV(fill)':>9}")
    P(f"  {'-'*80}")

    for model in models[:3]:  # top 3 models
        for thresh in [0.60, 0.65, 0.70, 0.75, 0.80]:
            t = compute_all_ticks(df, model, thresh, max_sp=0.55, min_sp=0.01, bet=bet)
            if t.empty or len(t) < 2: continue
            fillable = t[t["_can_fill"]]
            fill_pct = len(fillable) / len(t) * 100
            wr_all = t.won.mean() * 100
            wr_fill = fillable.won.mean() * 100 if len(fillable) > 0 else 0
            ev_fill = fillable.pnl.mean() if len(fillable) > 0 else 0
            P(f"  {model:<12} {thresh:>5.0%} {len(t):>6} {len(fillable):>9} {fill_pct:>5.0f}% "
              f"{wr_all:>7.1f}% {wr_fill:>8.1f}% ${ev_fill:>+7.3f}")
        P("")

    # Show depth stats
    P(f"\n  Depth stats across all ticks:")
    P(f"    YES depth: min={df.yes_depth.min():.0f}  median={df.yes_depth.median():.0f}  "
      f"mean={df.yes_depth.mean():.0f}  max={df.yes_depth.max():.0f}")
    P(f"    NO depth:  min={df.no_depth.min():.0f}  median={df.no_depth.median():.0f}  "
      f"mean={df.no_depth.mean():.0f}  max={df.no_depth.max():.0f}")


def report_overview(df: pd.DataFrame):
    """Quick overview of recorded data."""
    P(f"\n{'='*90}")
    P(f"  LIVE TICK RECORDING — OVERVIEW")
    P(f"{'='*90}")
    P(f"  Total ticks:   {len(df):,}")
    P(f"  Markets:       {df.slug.nunique()}")
    if "outcome" in df.columns:
        outcomes = df.drop_duplicates("slug")
        n_up = (outcomes.outcome == "UP").sum()
        n_dn = (outcomes.outcome == "DOWN").sum()
        P(f"  Outcomes:      {n_up} UP / {n_dn} DOWN")
    P(f"  Ticks/market:  {len(df) / max(df.slug.nunique(), 1):.0f} avg")

    models = [c for c in ["catboost", "lgbm", "rf", "xgboost", "ensemble"] if c in df.columns]
    P(f"  Models:        {', '.join(models)}")

    if "entry_pct" in df.columns:
        P(f"  Entry range:   {df.entry_pct.min()*100:.0f}% - {df.entry_pct.max()*100:.0f}%")
    if "yes" in df.columns:
        P(f"  YES price:     ${df.yes.min():.3f} - ${df.yes.max():.3f} (avg ${df.yes.mean():.3f})")
    if "sol" in df.columns:
        P(f"  SOL range:     ${df.sol.min():.2f} - ${df.sol.max():.2f}")

    ts_range = pd.to_datetime(df["ts"], unit="s")
    P(f"  Time range:    {ts_range.min().strftime('%H:%M:%S')} - {ts_range.max().strftime('%H:%M:%S')}")


def report_data_coverage(df: pd.DataFrame):
    """Section 0: Data coverage — ticks per market, timing distribution."""
    P(f"\n{'='*90}")
    P(f"  DATA COVERAGE")
    P(f"{'='*90}")

    tpm = df.groupby("slug").size()
    P(f"  Ticks/market:  min={tpm.min()}, median={tpm.median():.0f}, max={tpm.max()}, total={tpm.sum()}")

    # Timing distribution
    bins = [0, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 0.90, 1.0]
    labels = ["0-5%", "5-10%", "10-20%", "20-40%", "40-60%", "60-80%", "80-90%", "90-100%"]
    df_c = df.copy()
    df_c["_tbin"] = pd.cut(df_c["entry_pct"], bins=bins, labels=labels, include_lowest=True)
    dist = df_c["_tbin"].value_counts().sort_index()
    P(f"\n  Ticks by timing:")
    for lbl in labels:
        n = dist.get(lbl, 0)
        bar = "█" * min(int(n / max(len(df), 1) * 200), 50)
        P(f"    {lbl:>8}  {n:>5}  {bar}")

    # Duration distribution
    if "dur_min" in df.columns:
        dur_counts = df.drop_duplicates("slug").dur_min.value_counts().sort_index()
        P(f"\n  Markets by duration:")
        for d, n in dur_counts.items():
            P(f"    {d:>3}min  {n:>3} markets")


def report_per_market_detail(df: pd.DataFrame, bet: float):
    """Section 8: Per-market detail — every resolved market with predictions."""
    if "outcome" not in df.columns:
        return

    P(f"\n{'='*90}")
    P(f"  PER-MARKET DETAIL")
    P(f"{'='*90}")

    markets = df.drop_duplicates("slug", keep="last").sort_values("ts")
    P(f"  {'Slug':<40} {'Dur':>4} {'Out':>4} {'CB':>5} {'LG':>5} {'RF':>5} {'XG':>5} ")
    P(f"  {'-'*80}")

    for _, row in markets.iterrows():
        cb = row.get("catboost", np.nan)
        lg = row.get("lgbm", np.nan)
        rf = row.get("rf", np.nan)
        xg = row.get("xgboost", np.nan)
        out = row.get("outcome", "?")
        slug_short = row["slug"][-35:] if len(row["slug"]) > 35 else row["slug"]
        dur = row.get("dur_min", 0)

        # Direction for catboost
        cb_dir = "UP" if cb > 0.5 else "DN"
        cb_conf = max(cb, 1 - cb) if not np.isnan(cb) else 0
        correct = (cb_dir == "UP" and out == "UP") or (cb_dir == "DN" and out == "DOWN")
        mark = "✅" if correct else "❌"

        P(f"  {slug_short:<40} {dur:>3}m {out:>4} "
          f"{cb:.2f}  {lg:.2f}  {rf:.2f}  {xg:.2f}  "
          f"{cb_dir} {cb_conf:.0%} {mark}")


def report_continuous_accuracy(df: pd.DataFrame):
    """Section 9: Model accuracy at every tick (not just first entry)."""
    if "outcome" not in df.columns:
        return

    P(f"\n{'='*90}")
    P(f"  CONTINUOUS PREDICTION ACCURACY (every tick, not just first entry)")
    P(f"{'='*90}")

    models = [c for c in ["catboost", "lgbm", "rf", "xgboost", "ensemble"] if c in df.columns]

    P(f"  {'Model':<12} {'Ticks':>6} {'Correct':>8} {'Acc':>6} {'AvgConf':>8}")
    P(f"  {'-'*50}")
    for model in models:
        prob = df[model].values
        m_dir = np.where(prob > 0.5, "UP", "DOWN")
        correct = m_dir == df["outcome"].values
        conf = np.maximum(prob, 1 - prob)
        P(f"  {model:<12} {len(df):>6} {correct.sum():>8} {correct.mean()*100:>5.1f}% {conf.mean():>7.1%}")

    # By timing bins
    P(f"\n  Catboost accuracy by timing:")
    if "catboost" in df.columns:
        prob = df["catboost"].values
        m_dir = np.where(prob > 0.5, "UP", "DOWN")
        correct = m_dir == df["outcome"].values
        conf = np.maximum(prob, 1 - prob)

        bins = [(0, 0.10), (0.10, 0.20), (0.20, 0.40), (0.40, 0.60), (0.60, 0.80), (0.80, 1.0)]
        P(f"  {'Timing':<10} {'Ticks':>6} {'Acc':>6} {'AvgConf':>8} {'HiConf(>70%)':>14}")
        for lo, hi in bins:
            mask = (df["entry_pct"].values >= lo) & (df["entry_pct"].values < hi)
            if mask.sum() == 0:
                continue
            sub_correct = correct[mask]
            sub_conf = conf[mask]
            hi_mask = sub_conf >= 0.70
            hi_acc = sub_correct[hi_mask].mean() * 100 if hi_mask.sum() > 0 else 0
            P(f"  {lo*100:.0f}-{hi*100:.0f}%     {mask.sum():>6} {sub_correct.mean()*100:>5.1f}% "
              f"{sub_conf.mean():>7.1%}    {hi_acc:>5.1f}% ({hi_mask.sum()})")


def report_model_vs_model(df: pd.DataFrame, bet: float):
    """Section 10: Head-to-head model comparison — ALL ticks at multiple thresholds."""
    if "outcome" not in df.columns:
        return

    P(f"\n{'='*90}")
    P(f"  MODEL vs MODEL — 1 TRADE/MARKET, SP $0.40-$0.55")
    P(f"{'='*90}")

    models = [c for c in ["catboost", "lgbm", "rf", "xgboost", "ensemble"] if c in df.columns]
    thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

    P(f"  {'Model':<12} {'Conf>=':>6} {'Trades':>6} {'W':>5} {'L':>5} {'WR':>6} {'EV/trade':>9} {'AvgSP':>7} {'AvgConf':>8}")
    P(f"  {'-'*78}")

    for model in models:
        for thresh in thresholds:
            t = compute_all_ticks(df, model, thresh, max_sp=0.55, min_sp=0.40, bet=bet)
            if t.empty:
                continue
            w = t.won.sum(); lo = len(t) - w
            wr = w / len(t) * 100
            P(f"  {model:<12} {thresh:>5.0%} {len(t):>6} {w:>5} {lo:>5} {wr:>5.1f}% "
              f" ${t.pnl.mean():>+6.3f} ${t._sp.mean():>.3f} {t._conf.mean():>7.1%}")
        P("")


def report_raw_tick_stats(df: pd.DataFrame):
    """Section 11: Raw tick-level stats — how much data we're capturing."""
    P(f"\n{'='*90}")
    P(f"  RAW TICK STATISTICS")
    P(f"{'='*90}")

    # Tick frequency
    if len(df) > 1:
        ts_sorted = df.sort_values("ts")["ts"].values
        diffs = np.diff(ts_sorted)
        P(f"  Tick interval:  median={np.median(diffs):.1f}s, mean={np.mean(diffs):.1f}s, min={np.min(diffs):.1f}s")

    # Price ranges
    if "yes" in df.columns:
        P(f"  YES prices:    ${df.yes.min():.3f} — ${df.yes.max():.3f} (std ${df.yes.std():.3f})")
        P(f"  NO prices:     ${df.no.min():.3f} — ${df.no.max():.3f}")
    if "spread" in df.columns:
        P(f"  Spread:        avg ${df.spread.mean():.3f}, max ${df.spread.max():.3f}")

    # Model confidence distribution
    models = [c for c in ["catboost", "lgbm", "rf", "xgboost"] if c in df.columns]
    if models:
        P(f"\n  Model confidence distributions (directional):")
        P(f"  {'Model':<12} {'<55%':>6} {'55-60%':>7} {'60-65%':>7} {'65-70%':>7} {'70-80%':>7} {'80%+':>6}")
        for m in models:
            conf = np.maximum(df[m], 1 - df[m])
            bins = [
                (conf < 0.55).sum(),
                ((conf >= 0.55) & (conf < 0.60)).sum(),
                ((conf >= 0.60) & (conf < 0.65)).sum(),
                ((conf >= 0.65) & (conf < 0.70)).sum(),
                ((conf >= 0.70) & (conf < 0.80)).sum(),
                (conf >= 0.80).sum(),
            ]
            P(f"  {m:<12} {bins[0]:>6} {bins[1]:>7} {bins[2]:>7} {bins[3]:>7} {bins[4]:>7} {bins[5]:>6}")


def print_guide():
    """Print analysis guide explaining all tables and metrics."""
    P("""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║                    MULTI-DIMENSIONAL EDGE ANALYSIS — GUIDE                              ║
╠══════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                          ║
║  METHODOLOGY:                                                                            ║
║    - 1 trade per market (first qualifying tick → deduplicated by slug)                   ║
║    - Direction: model prob > 0.5 → UP (buy YES), else DOWN (buy NO)                      ║
║    - Share price = ASK price for chosen direction (what you actually pay)                 ║
║    - Win = prediction matches actual outcome                                             ║
║                                                                                          ║
║  KEY METRICS:                                                                            ║
║    N      = number of trades (= number of unique markets that qualified)                 ║
║    WR     = win rate (% of trades that were correct)                                     ║
║    EV/sh  = expected value PER 1 SHARE = P(win)*(1-price) - P(loss)*price                ║
║             If EV/sh > 0, every share you buy has positive expectation.                   ║
║             EV/sh at $0.50 price: 60%WR → $0.10, 70%WR → $0.20, 80%WR → $0.30          ║
║    PnL    = total profit/loss for given bet size (scales linearly with bet)               ║
║    AvgSP  = average share price at entry                                                 ║
║                                                                                          ║
║  LEVELS:                                                                                 ║
║    L1 — BASE TABLES: Model × Duration × Confidence (the fundamentals)                   ║
║    L2 — TIMING TABLES: When in market lifecycle the edge exists                          ║
║    L3 — PRICE TABLES: Share price bins (cheap shares = high payout but risky)            ║
║    L4 — COMBO GRID: Best multi-dimensional intersections                                 ║
║    L5 — THRESHOLD SWEEP: Model + price range variations                                  ║
║                                                                                          ║
║  WHY SEPARATE 5m AND 15m:                                                                ║
║    5m:  faster trajectory lock-in, momentum dominant, less noise                         ║
║    15m: more randomness, mean reversion, weaker microstructure                           ║
║    Mixing them → destroyed statistics (signal dilution)                                  ║
║                                                                                          ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝
""")


def report_L1_base_tables(df: pd.DataFrame, bet: float):
    """LEVEL 1: Model × Duration × Confidence — the fundamental breakdown."""
    if "outcome" not in df.columns:
        return

    P(f"\n{'='*110}")
    P(f"  LEVEL 1 — BASE TABLES: Model × Duration × Confidence")
    P(f"  Each model evaluated separately per market duration. EV = per 1 share.")
    P(f"{'='*110}")

    models = [c for c in ["catboost", "rf", "xgboost", "ensemble", "lgbm"] if c in df.columns]
    confs = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    durations = sorted(df.dur_min.unique()) if "dur_min" in df.columns else [None]

    for dur in durations:
        dur_label = f"{dur}min" if dur else "ALL"
        work = df[df.dur_min == dur] if dur else df
        n_mkts = work.slug.nunique()
        if n_mkts < 3:
            continue

        for model in models:
            P(f"\n  ┌── {model.upper()} — {dur_label} ({n_mkts} markets) ──┐")
            P(f"  │ {'Conf':>5} {'N':>4} {'W':>3} {'L':>3} {'WR':>6} {'EV/sh':>7} {'PnL':>8} {'AvgSP':>6} │")
            P(f"  │{'-'*50}│")
            has_data = False
            for conf in confs:
                t = compute_all_ticks(work, model, conf, max_sp=0.60, min_sp=0.01, bet=bet)
                if t.empty or len(t) < 1:
                    continue
                has_data = True
                w = int(t.won.sum()); l = len(t) - w
                wr = w / len(t)
                avg_sp = t._sp.mean()
                ev_share = wr * (1 - avg_sp) - (1 - wr) * avg_sp
                P(f"  │ {conf:>4.0%} {len(t):>4} {w:>3} {l:>3} {wr*100:>5.1f}% "
                  f"${ev_share:>+.3f} ${t.pnl.sum():>+6.1f} ${avg_sp:.3f} │")
            if not has_data:
                P(f"  │  (no qualifying ticks){'':>27}│")
            P(f"  └{'─'*50}┘")


def report_L2_timing_tables(df: pd.DataFrame, bet: float):
    """LEVEL 2: Timing × Confidence per duration — WHERE the edge lives."""
    if "outcome" not in df.columns:
        return

    P(f"\n{'='*110}")
    P(f"  LEVEL 2 — TIMING TABLES: When in market lifecycle does edge exist?")
    P(f"  Timing = % elapsed of market duration. EV = per 1 share.")
    P(f"{'='*110}")

    models = [c for c in ["catboost", "rf", "xgboost", "ensemble"] if c in df.columns]
    timing_bins = [(0.0, 0.20), (0.20, 0.40), (0.40, 0.60), (0.60, 0.80), (0.80, 1.0)]
    confs = [0.60, 0.70, 0.80, 0.90]
    durations = sorted(df.dur_min.unique()) if "dur_min" in df.columns else [None]

    for dur in durations:
        dur_label = f"{dur}min" if dur else "ALL"
        work = df[df.dur_min == dur] if dur else df
        if work.slug.nunique() < 3:
            continue

        P(f"\n  === {dur_label} ===")
        for model in models[:3]:  # top 3
            P(f"\n  {model.upper()}:")
            P(f"  {'Timing':<10}" + "".join(f" {'≥'+f'{c:.0%}':>9}" for c in confs))
            P(f"  {'-'*50}")
            for t_lo, t_hi in timing_bins:
                cells = []
                for conf in confs:
                    t = compute_all_ticks(work, model, conf, max_sp=0.60, min_sp=0.01,
                                          bet=bet, t_lo=t_lo, t_hi=t_hi)
                    if t.empty or len(t) < 1:
                        cells.append(f"{'—':>9}")
                    else:
                        wr = t.won.sum() / len(t) * 100
                        cells.append(f"{wr:.0f}%({len(t)})".rjust(9))
                P(f"  {t_lo*100:.0f}-{t_hi*100:.0f}%    " + " ".join(cells))


def report_L3_price_tables(df: pd.DataFrame, bet: float):
    """LEVEL 3: Share price bins × Confidence — payout structure."""
    if "outcome" not in df.columns:
        return

    P(f"\n{'='*110}")
    P(f"  LEVEL 3 — PRICE TABLES: Share price determines payout")
    P(f"  Cheap shares ($0.20) = 4x payout but lower WR. Expensive ($0.55) = 0.8x but higher WR.")
    P(f"  Key insight: 55% WR @ $0.20 = insanely profitable. 70% WR @ $0.60 = mediocre.")
    P(f"{'='*110}")

    price_bins = [(0.01, 0.20), (0.20, 0.35), (0.35, 0.45), (0.45, 0.50), (0.50, 0.55), (0.55, 0.60)]
    models = [c for c in ["catboost", "rf", "xgboost", "ensemble"] if c in df.columns]
    confs = [0.60, 0.70, 0.80, 0.90]
    durations = sorted(df.dur_min.unique()) if "dur_min" in df.columns else [None]

    for dur in durations:
        dur_label = f"{dur}min" if dur else "ALL"
        work = df[df.dur_min == dur] if dur else df
        if work.slug.nunique() < 3:
            continue

        P(f"\n  === {dur_label} ===")
        for model in models[:3]:
            P(f"\n  {model.upper()}:")
            P(f"  {'Price':<14}" + "".join(f" {'≥'+f'{c:.0%}':>9}" for c in confs))
            P(f"  {'-'*55}")
            for p_lo, p_hi in price_bins:
                cells = []
                for conf in confs:
                    t = compute_all_ticks(work, model, conf, max_sp=p_hi, min_sp=p_lo, bet=bet)
                    if t.empty or len(t) < 1:
                        cells.append(f"{'—':>9}")
                    else:
                        wr = t.won.sum() / len(t)
                        avg_sp = t._sp.mean()
                        ev_sh = wr * (1 - avg_sp) - (1 - wr) * avg_sp
                        cells.append(f"{wr*100:.0f}%${ev_sh:+.2f}".rjust(9))
                P(f"  ${p_lo:.2f}-${p_hi:.2f}  " + " ".join(cells))
            P(f"  (format: WR% $EV/share)")


def report_L4_combo_grid(df: pd.DataFrame, bet: float):
    """LEVEL 4: Best multi-dimensional combos — the STRONGEST table."""
    if "outcome" not in df.columns:
        return

    P(f"\n{'='*110}")
    P(f"  LEVEL 4 — BEST COMBO GRID: Model × Timing × Price × Conf (top results)")
    P(f"  Exhaustive scan of all parameter combinations. EV = per 1 share.")
    P(f"{'='*110}")

    models = [c for c in ["catboost", "rf", "xgboost", "ensemble"] if c in df.columns]
    confs = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    timing_bins = [(0.0, 1.0), (0.0, 0.40), (0.20, 0.60), (0.40, 0.80), (0.60, 1.0)]
    price_ranges = [(0.01, 0.60), (0.01, 0.50), (0.30, 0.50), (0.40, 0.55), (0.01, 0.40)]
    durations = sorted(df.dur_min.unique()) if "dur_min" in df.columns else [None]

    for dur in durations:
        dur_label = f"{dur}min" if dur else "ALL"
        work = df[df.dur_min == dur] if dur else df
        if work.slug.nunique() < 3:
            continue

        combos = []
        for model in models:
            for conf in confs:
                for t_lo, t_hi in timing_bins:
                    for p_lo, p_hi in price_ranges:
                        t = compute_all_ticks(work, model, conf, max_sp=p_hi, min_sp=p_lo,
                                              bet=bet, t_lo=t_lo, t_hi=t_hi)
                        if t.empty or len(t) < 3:
                            continue
                        w = int(t.won.sum())
                        wr = w / len(t)
                        avg_sp = t._sp.mean()
                        ev_sh = wr * (1 - avg_sp) - (1 - wr) * avg_sp
                        combos.append({
                            "model": model, "conf": conf,
                            "timing": f"{t_lo*100:.0f}-{t_hi*100:.0f}%",
                            "price": f"${p_lo:.2f}-${p_hi:.2f}",
                            "n": len(t), "w": w, "wr": wr * 100,
                            "ev_sh": ev_sh, "pnl": t.pnl.sum(), "avg_sp": avg_sp,
                        })

        if not combos:
            continue

        cdf = pd.DataFrame(combos)
        hdr = f"  {'Model':<10} {'Conf':>5} {'Timing':<9} {'Price':<13} {'N':>3} {'WR':>6} {'EV/sh':>7} {'PnL':>8} {'AvgSP':>6}"

        P(f"\n  ━━━ {dur_label} ({work.slug.nunique()} markets) ━━━")

        # Top by EV/share (N>=5)
        good = cdf[cdf.n >= 5].sort_values("ev_sh", ascending=False)
        if not good.empty:
            P(f"\n  TOP 20 BY EV/SHARE (N>=5):")
            P(hdr)
            for _, r in good.head(20).iterrows():
                P(f"  {r['model']:<10} {r['conf']:>4.0%} {r['timing']:<9} {r['price']:<13} "
                  f"{r['n']:>3} {r['wr']:>5.1f}% ${r['ev_sh']:>+.3f} ${r['pnl']:>+6.1f} ${r['avg_sp']:.3f}")

        # Top by WR (N>=5)
        good_wr = cdf[cdf.n >= 5].sort_values("wr", ascending=False)
        if not good_wr.empty:
            P(f"\n  TOP 20 BY WIN RATE (N>=5):")
            P(hdr)
            for _, r in good_wr.head(20).iterrows():
                P(f"  {r['model']:<10} {r['conf']:>4.0%} {r['timing']:<9} {r['price']:<13} "
                  f"{r['n']:>3} {r['wr']:>5.1f}% ${r['ev_sh']:>+.3f} ${r['pnl']:>+6.1f} ${r['avg_sp']:.3f}")

        # Top by PnL (N>=5)
        good_pnl = cdf[cdf.n >= 5].sort_values("pnl", ascending=False)
        if not good_pnl.empty:
            P(f"\n  TOP 20 BY TOTAL PNL (N>=5):")
            P(hdr)
            for _, r in good_pnl.head(20).iterrows():
                P(f"  {r['model']:<10} {r['conf']:>4.0%} {r['timing']:<9} {r['price']:<13} "
                  f"{r['n']:>3} {r['wr']:>5.1f}% ${r['ev_sh']:>+.3f} ${r['pnl']:>+6.1f} ${r['avg_sp']:.3f}")


def report_L5_threshold_sweep(df: pd.DataFrame, bet: float):
    """LEVEL 5: Threshold table with price range variations."""
    if "outcome" not in df.columns:
        return

    P(f"\n{'='*110}")
    P(f"  LEVEL 5 — THRESHOLD SWEEP: Model × Confidence × Price Range")
    P(f"  Each row = 1 trade per market (first qualifying tick). EV = per 1 share.")
    P(f"{'='*110}")

    models = [c for c in ["catboost", "rf", "xgboost", "ensemble"] if c in df.columns]
    confs = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    price_ranges = [
        (0.01, 0.60, "all"),
        (0.10, 0.50, "$0.1-0.5"),
        (0.10, 0.40, "$0.1-0.4"),
        (0.10, 0.30, "$0.1-0.3"),
        (0.10, 0.20, "$0.1-0.2"),
        (0.40, 0.55, "$0.4-0.55"),
    ]
    durations = sorted(df.dur_min.unique()) if "dur_min" in df.columns else [None]

    for dur in durations:
        dur_label = f"{dur}min" if dur else "ALL"
        work = df[df.dur_min == dur] if dur else df
        if work.slug.nunique() < 3:
            continue

        P(f"\n  ━━━ {dur_label} ({work.slug.nunique()} markets) ━━━")

        for model in models:
            P(f"\n  {model.upper()}:")
            P(f"  {'Conf':>5} {'$all':>10} {'$0.1-0.5':>10} {'$0.1-0.4':>10} {'$0.1-0.3':>10} {'$0.1-0.2':>10} {'$0.4-0.55':>10}")
            P(f"  {'-'*70}")
            for conf in confs:
                cells = []
                for p_lo, p_hi, _ in price_ranges:
                    t = compute_all_ticks(work, model, conf, max_sp=p_hi, min_sp=p_lo, bet=bet)
                    if t.empty or len(t) < 1:
                        cells.append(f"{'—':>10}")
                    else:
                        wr = t.won.sum() / len(t) * 100
                        cells.append(f"{wr:.0f}%({len(t)})".rjust(10))
                P(f"  {conf:>4.0%} " + " ".join(cells))

            # EV row
            P(f"  {'EV/sh':>5} {'$all':>10} {'$0.1-0.5':>10} {'$0.1-0.4':>10} {'$0.1-0.3':>10} {'$0.1-0.2':>10} {'$0.4-0.55':>10}")
            P(f"  {'-'*70}")
            for conf in confs:
                cells = []
                for p_lo, p_hi, _ in price_ranges:
                    t = compute_all_ticks(work, model, conf, max_sp=p_hi, min_sp=p_lo, bet=bet)
                    if t.empty or len(t) < 1:
                        cells.append(f"{'—':>10}")
                    else:
                        wr = t.won.sum() / len(t)
                        avg_sp = t._sp.mean()
                        ev_sh = wr * (1 - avg_sp) - (1 - wr) * avg_sp
                        cells.append(f"${ev_sh:+.3f}".rjust(10))
                P(f"  {conf:>4.0%} " + " ".join(cells))
            P("")


def report_consensus_matrix(df: pd.DataFrame, bet: float):
    """Model consensus — how many models agreeing affects edge."""
    if "outcome" not in df.columns:
        return

    base_models = [c for c in ["catboost", "rf", "xgboost", "lgbm"] if c in df.columns]
    if len(base_models) < 2:
        return

    P(f"\n{'='*110}")
    P(f"  MODEL CONSENSUS — agreement count × confidence")
    P(f"{'='*110}")

    durations = sorted(df.dur_min.unique()) if "dur_min" in df.columns else [None]
    confs = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

    for dur in durations:
        dur_label = f"{dur}min" if dur else "ALL"
        work = df[df.dur_min == dur] if dur else df
        if work.slug.nunique() < 3:
            continue

        P(f"\n  === {dur_label} ===")
        # Leader model = catboost
        cb_up = work["catboost"].values > 0.5
        cb_conf = np.maximum(work["catboost"].values, 1 - work["catboost"].values)

        P(f"  {'Agree':<6} {'Conf>=':>6} {'N':>4} {'WR':>6} {'EV/sh':>7} {'PnL':>8}")
        P(f"  {'-'*45}")

        for min_agree in [2, 3, 4]:
            if min_agree > len(base_models):
                continue
            for ct in confs:
                agree = np.zeros(len(work))
                for m in base_models:
                    m_up = work[m].values > 0.5
                    m_conf = np.maximum(work[m].values, 1 - work[m].values)
                    agree += ((m_up == cb_up) & (m_conf >= ct)).astype(float)

                sp = np.where(cb_up, work["yes"].values, work["no"].values)
                mask = (agree >= min_agree) & (cb_conf >= ct) & (sp >= 0.01) & (sp <= 0.55)
                sub = work[mask].copy()
                if sub.empty:
                    continue

                sub["_sp"] = sp[mask]
                sub["_dir"] = np.where(cb_up[mask], "UP", "DOWN")
                sub = sub.sort_values("entry_pct").drop_duplicates("slug", keep="first")
                if len(sub) < 2:
                    continue

                sub["won"] = sub["_dir"] == sub["outcome"]
                sub["pnl"] = np.where(sub["won"], bet / sub["_sp"] * (1 - sub["_sp"]), -bet)
                wr = sub.won.mean()
                avg_sp = sub._sp.mean()
                ev_sh = wr * (1 - avg_sp) - (1 - wr) * avg_sp
                P(f"  {min_agree}/{len(base_models)}   {ct:>5.0%} {len(sub):>4} {wr*100:>5.1f}% "
                  f"${ev_sh:>+.3f} ${sub.pnl.sum():>+6.1f}")


def report_ultra_table(df: pd.DataFrame, bet: float):
    """ULTRA TABLE: absolute exhaustive scan. Model × Conf × Timing × PriceCap × Duration.
    Finds the single best combination across ALL dimensions with different price caps."""
    if "outcome" not in df.columns:
        return

    P(f"\n{'#'*110}")
    P(f"  ### ULTRA TABLE — EXHAUSTIVE IMBA FINDER ###")
    P(f"  Every model × confidence × timing × price_cap. Separate per duration.")
    P(f"  Price caps: ≤$0.30, ≤$0.40, ≤$0.50 (from config limits)")
    P(f"{'#'*110}")

    models = [c for c in ["catboost", "rf", "xgboost", "ensemble", "lgbm"] if c in df.columns]
    confs = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    timing_bins = [
        (0.0, 1.0), (0.0, 0.20), (0.0, 0.40), (0.0, 0.60),
        (0.20, 0.40), (0.20, 0.60), (0.20, 0.80),
        (0.40, 0.60), (0.40, 0.80), (0.40, 1.0),
        (0.60, 0.80), (0.60, 1.0), (0.80, 1.0),
    ]
    price_caps = [0.30, 0.40, 0.50]
    min_sp = 0.10  # from config
    durations = sorted(df.dur_min.unique()) if "dur_min" in df.columns else [None]

    for dur in durations:
        dur_label = f"{dur}min" if dur else "ALL"
        work = df[df.dur_min == dur] if dur else df
        n_mkts = work.slug.nunique()
        if n_mkts < 3:
            continue

        all_combos = []
        for model in models:
            for conf in confs:
                for t_lo, t_hi in timing_bins:
                    for pcap in price_caps:
                        t = compute_all_ticks(work, model, conf,
                                              max_sp=pcap, min_sp=min_sp,
                                              bet=bet, t_lo=t_lo, t_hi=t_hi)
                        if t.empty or len(t) < 2:
                            continue
                        w = int(t.won.sum())
                        wr = w / len(t)
                        avg_sp = t._sp.mean()
                        ev_sh = wr * (1 - avg_sp) - (1 - wr) * avg_sp
                        all_combos.append({
                            "model": model, "conf": conf,
                            "timing": f"{t_lo*100:.0f}-{t_hi*100:.0f}%",
                            "pcap": pcap,
                            "n": len(t), "w": w, "wr": wr * 100,
                            "ev_sh": ev_sh, "pnl": t.pnl.sum(),
                            "avg_sp": avg_sp,
                        })

        if not all_combos:
            continue

        cdf = pd.DataFrame(all_combos)
        hdr = (f"  {'Model':<10} {'Conf':>5} {'Timing':<10} {'SP≤':>5} "
               f"{'N':>3} {'WR':>6} {'EV/sh':>7} {'PnL':>8} {'AvgSP':>6}")

        P(f"\n  ━━━ {dur_label} ({n_mkts} markets, {len(all_combos)} combos scanned) ━━━")

        # === TOP 30 by EV/share ===
        good = cdf[cdf.n >= 3].sort_values("ev_sh", ascending=False)
        if not good.empty:
            P(f"\n  TOP 30 BY EV/SHARE (N>=3):")
            P(hdr)
            P(f"  {'-'*75}")
            for _, r in good.head(30).iterrows():
                P(f"  {r['model']:<10} {r['conf']:>4.0%} {r['timing']:<10} ${r['pcap']:.2f} "
                  f"{r['n']:>3} {r['wr']:>5.1f}% ${r['ev_sh']:>+.3f} ${r['pnl']:>+6.1f} ${r['avg_sp']:.3f}")

        # === TOP 30 by WR (N>=5) ===
        good_wr = cdf[cdf.n >= 5].sort_values("wr", ascending=False)
        if not good_wr.empty:
            P(f"\n  TOP 30 BY WIN RATE (N>=5):")
            P(hdr)
            P(f"  {'-'*75}")
            for _, r in good_wr.head(30).iterrows():
                P(f"  {r['model']:<10} {r['conf']:>4.0%} {r['timing']:<10} ${r['pcap']:.2f} "
                  f"{r['n']:>3} {r['wr']:>5.1f}% ${r['ev_sh']:>+.3f} ${r['pnl']:>+6.1f} ${r['avg_sp']:.3f}")

        # === TOP 30 by TOTAL PNL (N>=5) ===
        good_pnl = cdf[cdf.n >= 5].sort_values("pnl", ascending=False)
        if not good_pnl.empty:
            P(f"\n  TOP 30 BY TOTAL PNL (N>=5):")
            P(hdr)
            P(f"  {'-'*75}")
            for _, r in good_pnl.head(30).iterrows():
                P(f"  {r['model']:<10} {r['conf']:>4.0%} {r['timing']:<10} ${r['pcap']:.2f} "
                  f"{r['n']:>3} {r['wr']:>5.1f}% ${r['ev_sh']:>+.3f} ${r['pnl']:>+6.1f} ${r['avg_sp']:.3f}")

        # === IMBA FINDER: WR>=75% AND N>=5 sorted by EV ===
        imba = cdf[(cdf.wr >= 75) & (cdf.n >= 5)].sort_values("ev_sh", ascending=False)
        if not imba.empty:
            P(f"\n  ★ IMBA CONFIGS (WR>=75%, N>=5, sorted by EV/share):")
            P(hdr)
            P(f"  {'-'*75}")
            for _, r in imba.head(30).iterrows():
                P(f"  {r['model']:<10} {r['conf']:>4.0%} {r['timing']:<10} ${r['pcap']:.2f} "
                  f"{r['n']:>3} {r['wr']:>5.1f}% ${r['ev_sh']:>+.3f} ${r['pnl']:>+6.1f} ${r['avg_sp']:.3f}")

        # === PER PRICE CAP: best 5 per cap ===
        P(f"\n  --- BEST PER PRICE CAP ---")
        for pcap in price_caps:
            cap_df = cdf[(cdf.pcap == pcap) & (cdf.n >= 3)].sort_values("ev_sh", ascending=False)
            if cap_df.empty:
                continue
            P(f"\n  SP ≤ ${pcap:.2f}:")
            P(hdr)
            for _, r in cap_df.head(10).iterrows():
                P(f"  {r['model']:<10} {r['conf']:>4.0%} {r['timing']:<10} ${r['pcap']:.2f} "
                  f"{r['n']:>3} {r['wr']:>5.1f}% ${r['ev_sh']:>+.3f} ${r['pnl']:>+6.1f} ${r['avg_sp']:.3f}")


def report_config_filter_table(df: pd.DataFrame, bet: float):
    """THRESHOLD/FAIR PRICE using config limits: SP $0.10-$0.50 (from trading.yaml)."""
    if "outcome" not in df.columns:
        return

    P(f"\n{'='*90}")
    P(f"  CONFIG-FILTERED TABLE — SP $0.10-$0.50 (from config limits)")
    P(f"{'='*90}")
    P(f"  {'Model':<12} {'Conf>=':>6} {'N':>4} {'W':>3} {'L':>3} {'WR':>6} {'EV/sh':>7} {'PnL':>8} {'AvgSP':>6}")
    P(f"  {'-'*65}")

    models = [c for c in ["catboost", "rf", "xgboost", "ensemble", "lgbm"] if c in df.columns]
    confs = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    for model in models:
        for conf in confs:
            t = compute_all_ticks(df, model, conf, max_sp=0.50, min_sp=0.10, bet=bet)
            if t.empty or len(t) < 1:
                continue
            w = int(t.won.sum()); l = len(t) - w
            wr = w / len(t)
            avg_sp = t._sp.mean()
            ev_sh = wr * (1 - avg_sp) - (1 - wr) * avg_sp
            P(f"  {model:<12} {conf:>4.0%} {len(t):>4} {w:>3} {l:>3} {wr*100:>5.1f}% "
              f"${ev_sh:>+.3f} ${t.pnl.sum():>+6.1f} ${avg_sp:.3f}")
        P("")


def report_final_summary(df: pd.DataFrame, bet: float):
    """Compact final recommendation."""
    if "outcome" not in df.columns:
        return

    P(f"\n{'='*110}")
    P(f"  ★ FINAL VERDICT")
    P(f"{'='*110}")

    models = [c for c in ["catboost", "rf", "xgboost", "ensemble"] if c in df.columns]
    durations = sorted(df.dur_min.unique()) if "dur_min" in df.columns else [None]

    for dur in durations:
        dur_label = f"{dur}min" if dur else "ALL"
        work = df[df.dur_min == dur] if dur else df
        if work.slug.nunique() < 5:
            continue

        best = []
        for model in models:
            for conf in [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
                t = compute_all_ticks(work, model, conf, max_sp=0.55, min_sp=0.01, bet=bet)
                if t.empty or len(t) < 5:
                    continue
                wr = t.won.sum() / len(t)
                avg_sp = t._sp.mean()
                ev_sh = wr * (1 - avg_sp) - (1 - wr) * avg_sp
                best.append({
                    "model": model, "conf": conf, "n": len(t),
                    "wr": wr * 100, "ev_sh": ev_sh, "pnl": t.pnl.sum(),
                })

        if not best:
            continue

        bdf = pd.DataFrame(best)
        top = bdf.sort_values("ev_sh", ascending=False).head(5)
        P(f"\n  {dur_label} — TOP 5 BY EV/SHARE (N>=5):")
        P(f"  {'Model':<12} {'Conf':>5} {'N':>4} {'WR':>6} {'EV/sh':>7} {'PnL':>8}")
        P(f"  {'-'*50}")
        for _, r in top.iterrows():
            P(f"  {r['model']:<12} {r['conf']:>4.0%} {r['n']:>4} {r['wr']:>5.1f}% "
              f"${r['ev_sh']:>+.3f} ${r['pnl']:>+6.1f}")

    # Overall stats
    n_markets = df.slug.nunique()
    ts_range = pd.to_datetime(df["ts"], unit="s")
    hours = (ts_range.max() - ts_range.min()).total_seconds() / 3600
    P(f"\n  Dataset: {len(df):,} ticks, {n_markets} markets, {hours:.1f}h recording")


def main():
    global _OUT_FILE
    parser = argparse.ArgumentParser(description="Multi-dimensional edge analysis")
    parser.add_argument("--date", type=str, default=None, help="Date YYYY-MM-DD")
    parser.add_argument("--all", action="store_true", help="Load all available days")
    parser.add_argument("--bet", type=float, default=2.0, help="Bet size for PnL calc")
    parser.add_argument("--out", type=str, default=None, help="Output file path")
    args = parser.parse_args()

    date_label = args.date or datetime.now().strftime("%Y-%m-%d")
    out_path = args.out or f"results/analysis_{date_label}.txt"
    Path(out_path).parent.mkdir(exist_ok=True)
    _OUT_FILE = open(out_path, "w", encoding="utf-8")

    P(f"\n{'='*110}")
    P(f"  MULTI-DIMENSIONAL EDGE ANALYSIS")
    P(f"  Date: {date_label} | Bet: ${args.bet:.2f} | EV calculated per 1 share")
    P(f"{'='*110}")

    # Load data
    ticks = load_ticks(args.date, args.all)
    outcomes = load_outcomes(args.date, args.all)
    df = merge_ticks_outcomes(ticks, outcomes)

    if df.empty or "outcome" not in df.columns:
        P("  ERROR: No data with outcomes. Need more recording time.")
        _OUT_FILE.close()
        return

    # Guide
    print_guide()

    # Overview
    report_overview(df)
    report_data_coverage(df)

    # LEVEL 1: Base tables
    report_L1_base_tables(df, args.bet)

    # LEVEL 2: Timing tables
    report_L2_timing_tables(df, args.bet)

    # LEVEL 3: Price tables
    report_L3_price_tables(df, args.bet)

    # LEVEL 4: Combo grid
    report_L4_combo_grid(df, args.bet)

    # LEVEL 5: Threshold sweep with price ranges
    report_L5_threshold_sweep(df, args.bet)

    # Config-filtered table (SP from config: $0.10-$0.50)
    report_config_filter_table(df, args.bet)

    # Old-style tables (kept for reference)
    report_threshold_table(df, args.bet)
    report_fair_price_table(df, args.bet)

    # Model consensus
    report_consensus_matrix(df, args.bet)

    # ULTRA TABLE — exhaustive imba finder
    report_ultra_table(df, args.bet)

    # Existing useful reports
    report_streak_analysis(df, args.bet)
    report_liquidity(df, args.bet)

    # Final verdict
    report_final_summary(df, args.bet)

    P(f"\n{'='*110}")
    P(f"  DONE — {len(df):,} ticks, {df.slug.nunique()} markets")
    P(f"  Saved to: {out_path}")
    P(f"{'='*110}\n")

    _OUT_FILE.close()
    _OUT_FILE = None


if __name__ == "__main__":
    main()
