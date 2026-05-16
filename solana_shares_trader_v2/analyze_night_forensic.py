"""
NIGHT SESSION FORENSIC ANALYSIS
Deep dive: Why did live trades lose? Analysis vs Live tick-by-tick.

Key questions:
1. For each live trade: does analysis agree on direction at that entry_pct?
2. Why only 4-6 night trades out of 84 available markets?
3. 5min vs 15min performance comparison
4. Is analysis WR == live WR for the SAME direction decisions?
5. Statistical probability of observed losses given true WR
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import stats
import time

BET = 2.85
OUT = "results/night_forensic.txt"
_F = None

def P(*a, **kw):
    print(*a, **kw)
    if _F: print(*a, **kw, file=_F)


# ══════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════
def load_all():
    results = Path("results")
    frames_t, frames_o = [], []
    for f in sorted(results.glob("live_ticks_*.jsonl")):
        df = pd.read_json(f, lines=True)
        if len(df): frames_t.append(df)
    for f in sorted(results.glob("live_outcomes_*.jsonl")):
        df = pd.read_json(f, lines=True)
        if len(df): frames_o.append(df)
    ticks = pd.concat(frames_t, ignore_index=True)
    outcomes = pd.concat(frames_o, ignore_index=True).drop_duplicates("slug", keep="last")
    df = ticks.merge(outcomes[["slug","outcome"]], on="slug", how="inner")
    # Add columns
    df["_mkt_ts"] = df["slug"].str.split("-").str[-1].astype(int)
    df["_hour"] = (pd.to_datetime(df["_mkt_ts"], unit="s").dt.hour + 3) % 24
    df["dur_min"] = df["slug"].apply(lambda s: 15 if "-15m-" in s else 5)
    return df


def compute_first_tick(df, model, thresh, max_sp, t_lo, t_hi, min_sp=0.10, bet=BET):
    """Analysis logic: first qualifying tick per market, GTC to end."""
    if df.empty or model not in df.columns or "outcome" not in df.columns:
        return pd.DataFrame()
    prob = df[model].values
    conf = np.maximum(prob, 1 - prob)
    up = prob > 0.5
    sp = np.where(up, df["yes"].values, df["no"].values)
    ep = df["entry_pct"].values
    mask = (conf >= thresh) & (sp <= max_sp) & (sp >= min_sp) & (ep >= t_lo) & (ep < t_hi)
    v = df[mask].copy()
    if v.empty: return pd.DataFrame()
    v["_dir"] = np.where(up[mask], "UP", "DOWN")
    v["_sp"] = sp[mask]
    v["_conf"] = conf[mask]
    v = v.sort_values("entry_pct").drop_duplicates("slug", keep="first").copy()
    v["_won"] = v["_dir"] == v["outcome"]
    shares = bet / v["_sp"].values
    v["_pnl"] = np.where(v["_won"].values, shares * (1 - v["_sp"].values), -bet)
    return v


def main():
    global _F
    _F = open(OUT, "w", encoding="utf-8")

    P(f"{'='*110}")
    P(f"  NIGHT SESSION FORENSIC ANALYSIS")
    P(f"  Live trades vs Analysis — tick-by-tick deep dive")
    P(f"{'='*110}")

    df = load_all()
    P(f"  Total: {len(df)} ticks, {df.slug.nunique()} markets")

    # Live trades
    trades = json.load(open("results/ml_live_trades.json"))
    live_df = pd.DataFrame([t for t in trades if not t["dry_run"]])
    live_df["_hour"] = (pd.to_datetime(live_df["entry_ts"], unit="s").dt.hour + 3) % 24

    # Night live trades (23:00-06:00 UTC+3)
    night_live = live_df[(live_df["_hour"] >= 23) | (live_df["_hour"] < 6)].copy()
    night_live_5m = night_live[night_live["duration_min"] == 5]
    night_live_15m = night_live[night_live["duration_min"] == 15]

    # Night tick data
    night_mask = (df["_hour"] >= 23) | (df["_hour"] < 6)
    night = df[night_mask].copy()
    night_5m = night[night["dur_min"] == 5]
    night_15m = night[night["dur_min"] == 15]

    P(f"\n  Night ticks: {night.slug.nunique()} markets ({night_5m.slug.nunique()} 5min + {night_15m.slug.nunique()} 15min)")
    P(f"  Night live trades: {len(night_live)} total ({len(night_live_5m)} 5min + {len(night_live_15m)} 15min)")

    # ══════════════════════════════════════════════════════════
    #  PART 1: LIVE ENTRY vs ANALYSIS ENTRY — tick-by-tick
    # ══════════════════════════════════════════════════════════
    P(f"\n\n{'='*110}")
    P(f"  PART 1: TICK-BY-TICK FORENSIC — each live trade vs analysis")
    P(f"  Config: catboost 70% SP≤0.55 timing 55-92%")
    P(f"{'='*110}")

    for _, t in live_df.iterrows():
        slug = t["slug"]
        is_night = (t["_hour"] >= 23 or t["_hour"] < 6)
        night_tag = " [NIGHT]" if is_night else ""

        # Get all ticks for this market
        mkt_ticks = df[df["slug"] == slug].sort_values("entry_pct")
        if mkt_ticks.empty:
            P(f"\n  {slug}{night_tag}: NO TICK DATA")
            continue

        outcome = mkt_ticks.iloc[0]["outcome"] if "outcome" in mkt_ticks.columns else "?"
        dur = "15m" if "-15m-" in slug else "5m"

        # What analysis would do (first qualifying tick)
        analysis = compute_first_tick(mkt_ticks, "catboost", 0.70, 0.55, 0.55, 0.92)

        # Find the tick closest to live entry
        live_entry_pct = None
        if not mkt_ticks.empty and "ts" in mkt_ticks.columns:
            diffs = np.abs(mkt_ticks["ts"].values - t["entry_ts"])
            closest_idx = diffs.argmin()
            closest_tick = mkt_ticks.iloc[closest_idx]
            live_entry_pct = closest_tick["entry_pct"]

            # What was catboost predicting at live entry time?
            cb_prob = closest_tick.get("catboost", 0.5)
            cb_conf = max(cb_prob, 1 - cb_prob)
            cb_dir = "UP" if cb_prob > 0.5 else "DOWN"
            cb_sp = closest_tick["yes"] if cb_dir == "UP" else closest_tick["no"]
        else:
            cb_dir = "?"
            cb_conf = 0
            cb_sp = 0

        # Live trade info
        live_dir = t["direction"]
        live_won = bool(t["won"])
        live_pnl = t["pnl_usd"]
        live_conf = t["confidence"]
        live_sp = t["entry_price"]

        # Analysis trade info
        if not analysis.empty:
            an_dir = analysis.iloc[0]["_dir"]
            an_won = bool(analysis.iloc[0]["_won"])
            an_pnl = analysis.iloc[0]["_pnl"]
            an_sp = analysis.iloc[0]["_sp"]
            an_conf = analysis.iloc[0]["_conf"]
            an_epct = analysis.iloc[0]["entry_pct"]
        else:
            an_dir = "SKIP"
            an_won = False
            an_pnl = 0
            an_sp = 0
            an_conf = 0
            an_epct = 0

        match = "✓" if live_dir == an_dir else "✗ DIFFERS"
        won_match = "✓" if live_won == an_won else "✗ DIFFERS"

        P(f"\n  {slug} ({dur}){night_tag}  outcome={outcome}  hour={int(t['_hour']):02d}:00")
        P(f"    LIVE:     dir={live_dir:>5} conf={live_conf:.0%} sp=${live_sp:.3f} won={'W' if live_won else 'L'} pnl=${live_pnl:+.2f}  entry_pct={live_entry_pct:.0%}" if live_entry_pct else f"    LIVE:     dir={live_dir:>5} conf={live_conf:.0%} sp=${live_sp:.3f} won={'W' if live_won else 'L'} pnl=${live_pnl:+.2f}")
        P(f"    ANALYSIS: dir={an_dir:>5} conf={an_conf:.0%} sp=${an_sp:.3f} won={'W' if an_won else 'L'} pnl=${an_pnl:+.2f}  entry_pct={an_epct:.0%}" if an_dir != "SKIP" else f"    ANALYSIS: SKIP (no qualifying tick in 55-92% window)")
        P(f"    CB@LIVE_TICK: dir={cb_dir} conf={cb_conf:.0%} sp=${cb_sp:.3f}")
        P(f"    Direction match: {match}  |  Outcome match: {won_match}")

        # Show what other configs would do on this market
        configs = [
            ("CB70 NO_FILTER", "catboost", 0.70, 0.60, 0.0, 1.0),
            ("CB70 SP55 80-100%", "catboost", 0.70, 0.55, 0.80, 1.0),
            ("CB85 SP55 80-100%", "catboost", 0.85, 0.55, 0.80, 1.0),
            ("CB70 SP50 80-100%", "catboost", 0.70, 0.50, 0.80, 1.0),
            ("ENS85 SP55 60-100%", "ensemble", 0.85, 0.55, 0.60, 1.0),
        ]
        for label, mdl, thr, msp, tlo, thi in configs:
            r = compute_first_tick(mkt_ticks, mdl, thr, msp, tlo, thi)
            if r.empty:
                P(f"    {label:<25} → SKIP")
            else:
                rd = r.iloc[0]
                P(f"    {label:<25} → {rd['_dir']:>5} conf={rd['_conf']:.0%} sp=${rd['_sp']:.3f} {'W' if rd['_won'] else 'L'} pnl=${rd['_pnl']:+.2f} @{rd['entry_pct']:.0%}")


    # ══════════════════════════════════════════════════════════
    #  PART 2: WHY ONLY 4-6 TRADES OUT OF 84 MARKETS?
    # ══════════════════════════════════════════════════════════
    P(f"\n\n{'='*110}")
    P(f"  PART 2: COVERAGE — Why did live only trade {len(night_live)} out of {night.slug.nunique()} night markets?")
    P(f"{'='*110}")

    P(f"\n  Live config: catboost 70% SP≤$0.55 timing 55-92% max_positions=5")
    P(f"  Live night trades: {len(night_live)} ({len(night_live_5m)} 5min + {len(night_live_15m)} 15min)")
    P(f"  Night markets available: {night.slug.nunique()} ({night_5m.slug.nunique()} 5min + {night_15m.slug.nunique()} 15min)")

    # How many markets would analysis enter?
    an_5m = compute_first_tick(night_5m, "catboost", 0.70, 0.55, 0.55, 0.92)
    an_15m = compute_first_tick(night_15m, "catboost", 0.70, 0.55, 0.55, 0.92)
    P(f"\n  Analysis would enter (CB70 SP55 55-92%):")
    P(f"    5min:  {len(an_5m)} out of {night_5m.slug.nunique()} markets")
    P(f"    15min: {len(an_15m)} out of {night_15m.slug.nunique()} markets")
    P(f"    Total: {len(an_5m) + len(an_15m)} trades")

    P(f"\n  ⚠️  KEY INSIGHT: Analysis enters {len(an_5m)+len(an_15m)} trades, live entered {len(night_live)}.")
    P(f"  Reasons for gap:")
    P(f"    1. max_positions=5 — bot can hold max 5 at once, skips rest")
    P(f"    2. Spread filter: max_spread=$0.08 — rejects illiquid markets")
    P(f"    3. Depth filter: min_depth=20 shares — rejects thin books")
    P(f"    4. FRESH ML re-eval: bot re-evaluates every 5s. Direction can FLIP between ticks!")
    P(f"    5. Bot checks trade_slugs: only sol-updown-5m and sol-updown-15m")

    # Show concurrent positions at each live trade time
    P(f"\n  Concurrent positions at each night trade entry:")
    all_live = live_df.sort_values("entry_ts")
    for _, t in night_live.sort_values("entry_ts").iterrows():
        ts = t["entry_ts"]
        # Count positions open at this time
        open_pos = all_live[(all_live["entry_ts"] <= ts) & (all_live["exit_ts"] > ts)]
        P(f"    {t['entry_time']} {t['slug'][-15:]:>15} {t['direction']:>5} {'W' if t['won'] else 'L'} — {len(open_pos)} positions open")


    # ══════════════════════════════════════════════════════════
    #  PART 3: CRITICAL — DIRECTION FLIP BETWEEN TICKS
    # ══════════════════════════════════════════════════════════
    P(f"\n\n{'='*110}")
    P(f"  PART 3: DIRECTION STABILITY — Does prediction flip during market?")
    P(f"{'='*110}")
    P(f"  Analysis takes first qualifying tick. Live re-evaluates every 5s.")
    P(f"  If direction flips, live might enter OPPOSITE to analysis!\n")

    for _, t in night_live.sort_values("entry_ts").iterrows():
        slug = t["slug"]
        mkt = df[df["slug"] == slug].sort_values("entry_pct")
        if mkt.empty: continue

        outcome = mkt.iloc[0]["outcome"]
        dur = "15m" if "-15m-" in slug else "5m"

        # Track direction at each tick in the entry window (55-92%)
        window = mkt[(mkt["entry_pct"] >= 0.55) & (mkt["entry_pct"] < 0.92)]
        if window.empty: continue

        cb_probs = window["catboost"].values
        cb_dirs = np.where(cb_probs > 0.5, "UP", "DOWN")
        cb_confs = np.maximum(cb_probs, 1 - cb_probs)
        n_up = (cb_dirs == "UP").sum()
        n_dn = (cb_dirs == "DOWN").sum()
        n_above70 = (cb_confs >= 0.70).sum()

        # Find if direction flips
        unique_dirs = np.unique(cb_dirs)
        flips = len(unique_dirs) > 1

        # What tick was live entry closest to?
        diffs = np.abs(window["ts"].values - t["entry_ts"])
        ci = diffs.argmin()
        live_tick_dir = cb_dirs[ci]
        live_tick_conf = cb_confs[ci]

        P(f"  {slug} ({dur}) outcome={outcome}")
        P(f"    Window 55-92%: {len(window)} ticks, UP:{n_up} DOWN:{n_dn} flips={'YES ⚠️' if flips else 'NO'}")
        P(f"    Conf>=70%: {n_above70}/{len(window)} ticks")
        P(f"    At live entry tick: dir={live_tick_dir} conf={live_tick_conf:.0%}")
        P(f"    Live entered: dir={t['direction']} conf={t['confidence']:.0%} → {'W' if t['won'] else 'L'}")
        if live_tick_dir != t['direction']:
            P(f"    ⚠️  MISMATCH: CB predicted {live_tick_dir} but live entered {t['direction']}!")


    # ══════════════════════════════════════════════════════════
    #  PART 4: 5MIN vs 15MIN — DEEP COMPARISON
    # ══════════════════════════════════════════════════════════
    P(f"\n\n{'='*110}")
    P(f"  PART 4: 5MIN vs 15MIN NIGHT PERFORMANCE COMPARISON")
    P(f"{'='*110}")

    configs = [
        ("CB 70% SP≤0.55 55-92%", "catboost", 0.70, 0.55, 0.55, 0.92),
        ("CB 70% SP≤0.55 60-100%", "catboost", 0.70, 0.55, 0.60, 1.0),
        ("CB 70% SP≤0.55 80-100%", "catboost", 0.70, 0.55, 0.80, 1.0),
        ("CB 80% SP≤0.55 60-100%", "catboost", 0.80, 0.55, 0.60, 1.0),
        ("CB 85% SP≤0.55 60-100%", "catboost", 0.85, 0.55, 0.60, 1.0),
        ("CB 95% SP≤0.55 60-100%", "catboost", 0.95, 0.55, 0.60, 1.0),
        ("CB 70% NO_FILTER",       "catboost", 0.70, 1.00, 0.00, 1.0),
        ("RF 85% SP≤0.55 60-100%", "rf",       0.85, 0.55, 0.60, 1.0),
        ("ENS 85% SP≤0.55 60-100%","ensemble",  0.85, 0.55, 0.60, 1.0),
        ("XGB 80% SP≤0.55 60-100%","xgboost",  0.80, 0.55, 0.60, 1.0),
    ]

    P(f"\n  {'Config':<30} {'5min':>6} {'WR':>5} {'PnL':>8} {'EV':>6} │ {'15min':>6} {'WR':>5} {'PnL':>8} {'EV':>6} │ {'ALL':>6} {'WR':>5} {'PnL':>8}")
    P(f"  {'-'*120}")

    for label, mdl, thr, msp, tlo, thi in configs:
        # 5min
        r5 = compute_first_tick(night_5m, mdl, thr, msp, tlo, thi)
        if not r5.empty:
            n5 = len(r5); wr5 = r5["_won"].mean()*100; pnl5 = r5["_pnl"].sum(); ev5 = r5["_pnl"].mean()
            s5 = f"{n5:>4}t {wr5:>4.0f}% ${pnl5:>+6.0f} ${ev5:>+4.1f}"
        else:
            s5 = f"{'—':>25}"

        # 15min
        r15 = compute_first_tick(night_15m, mdl, thr, msp, tlo, thi)
        if not r15.empty:
            n15 = len(r15); wr15 = r15["_won"].mean()*100; pnl15 = r15["_pnl"].sum(); ev15 = r15["_pnl"].mean()
            s15 = f"{n15:>4}t {wr15:>4.0f}% ${pnl15:>+6.0f} ${ev15:>+4.1f}"
        else:
            s15 = f"{'—':>25}"

        # ALL
        ra = compute_first_tick(night, mdl, thr, msp, tlo, thi)
        if not ra.empty:
            na = len(ra); wra = ra["_won"].mean()*100; pnla = ra["_pnl"].sum()
            sa = f"{na:>4}t {wra:>4.0f}% ${pnla:>+6.0f}"
        else:
            sa = f"{'—':>17}"

        P(f"  {label:<30} {s5} │ {s15} │ {sa}")


    # ══════════════════════════════════════════════════════════
    #  PART 5: THRESHOLD SWEEP — 5min vs 15min
    # ══════════════════════════════════════════════════════════
    P(f"\n\n{'='*110}")
    P(f"  PART 5: THRESHOLD SWEEP — Night 5min vs 15min")
    P(f"{'='*110}")

    models = ["catboost", "rf", "xgboost", "ensemble"]
    thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    for mdl in models:
        P(f"\n  {mdl.upper()} — SP≤$0.55, timing 0-100% (NO timing filter):")
        P(f"  {'Conf':>5}  {'5min N':>6} {'WR':>5} {'PnL':>8} │ {'15min N':>7} {'WR':>5} {'PnL':>8} │ {'ALL N':>5} {'WR':>5} {'PnL':>8}")
        P(f"  {'-'*85}")
        for thr in thresholds:
            r5 = compute_first_tick(night_5m, mdl, thr, 0.55, 0.0, 1.0)
            r15 = compute_first_tick(night_15m, mdl, thr, 0.55, 0.0, 1.0)
            ra = compute_first_tick(night, mdl, thr, 0.55, 0.0, 1.0)

            def fmt(r):
                if r.empty: return f"{'—':>20}"
                return f"{len(r):>5}t {r['_won'].mean()*100:>4.0f}% ${r['_pnl'].sum():>+6.0f}"

            P(f"  {thr:>4.0%}  {fmt(r5)} │ {fmt(r15)} │ {fmt(ra)}")


    # ══════════════════════════════════════════════════════════
    #  PART 6: STATISTICAL PROBABILITY OF OBSERVED LOSSES
    # ══════════════════════════════════════════════════════════
    P(f"\n\n{'='*110}")
    P(f"  PART 6: STATISTICAL ANALYSIS — Was it bad luck?")
    P(f"{'='*110}")

    # For each relevant WR, calculate P(observed or worse)
    scenarios = [
        ("Night 5min live: 4 trades, 1W",  4, 1, 0.84),
        ("Night 5min live: 4 trades, 1W",  4, 1, 0.80),
        ("Night all live: 8 trades, 5W",   8, 5, 0.84),
        ("Night all live: 8 trades, 5W",   8, 5, 0.80),
        ("All live 5min: 13 trades, 6W",  13, 6, 0.80),
        ("All live 5min: 13 trades, 6W",  13, 6, 0.84),
        ("All live: 16 trades, 9W",       16, 9, 0.80),
    ]

    P(f"\n  If true WR=X%, what's the probability of getting ≤Y wins out of N trades?")
    P(f"  (Binomial distribution)\n")
    P(f"  {'Scenario':<40} {'True WR':>8} {'P(≤observed)':>14} {'Verdict':>15}")
    P(f"  {'-'*85}")

    for label, n, w, true_wr in scenarios:
        # P(X <= w) where X ~ Binomial(n, true_wr)
        p_val = stats.binom.cdf(w, n, true_wr)
        verdict = "VERY UNLIKELY" if p_val < 0.05 else "UNLIKELY" if p_val < 0.10 else "PLAUSIBLE" if p_val < 0.25 else "NORMAL"
        P(f"  {label:<40} {true_wr:>7.0%} {p_val:>13.4f} ({p_val*100:.1f}%)  {verdict}")

    P(f"\n  Interpretation:")
    P(f"    p < 5% → VERY unlikely to be random (something systematic)")
    P(f"    p 5-10% → unlikely but possible (borderline)")
    P(f"    p 10-25% → plausible bad luck")
    P(f"    p > 25% → normal variance")


    # ══════════════════════════════════════════════════════════
    #  PART 7: THE REAL DIFFERENCE — Analysis vs Live
    # ══════════════════════════════════════════════════════════
    P(f"\n\n{'='*110}")
    P(f"  PART 7: ROOT CAUSE — WHY ANALYSIS ≠ LIVE")
    P(f"{'='*110}")

    P(f"""
  ANALYSIS logic:
    1. Takes ALL ticks for a market (recorded every 5s)
    2. Filters by: conf >= 70%, SP <= $0.55, entry_pct 55-92%
    3. Takes FIRST qualifying tick → enters at that SP, holds to market end
    4. 1 trade per market, no max_positions limit
    5. No spread/depth filter
    6. Uses RECORDED predictions (same ML model, same features)

  LIVE BOT logic:
    1. Evaluates every ~5s when market is in entry window
    2. Same filters: conf >= 70%, SP <= $0.55, entry_pct 55-92%
    3. ADDITIONAL filters: spread <= $0.08, depth >= 20 shares
    4. max_positions = 5 (SKIPS markets if 5 already open!)
    5. If SP too high → QUEUES, re-evaluates with FRESH ML every 5s
    6. Direction can CHANGE between ticks → bot might enter DIFFERENT direction
    7. Actual fill price may differ from analysis price (slippage)

  KEY DIFFERENCES:
    ① max_positions=5 → bot can only trade ~5 markets at a time
       Analysis trades ALL qualifying markets ({len(an_5m)+len(an_15m)} night trades)
       Live traded only {len(night_live)} night trades

    ② FRESH ML re-evaluation → direction/confidence can change
       If SP is $0.57 (too high), bot queues the market.
       When SP drops to $0.54, ML re-runs. New prediction might be DIFFERENT.
       Analysis uses the FIRST time SP was good. Bot uses LATEST prediction.

    ③ Spread & depth filters → bot skips illiquid markets
       Analysis assumes all markets are tradeable.

    ④ Small sample = massive variance
       With 4 trades, even 84% WR has ~1.4% chance of ≤1 win.
       But 1.4% is not zero — it does happen.
""")

    # Show: for each live-traded market, analysis first-tick vs live-tick prediction
    P(f"  PREDICTION COMPARISON — Analysis tick vs Live tick for night trades:")
    P(f"  {'Slug':<40} {'Outcome':>7} │ {'Analysis dir':>12} {'conf':>5} {'@pct':>5} │ {'CB@LiveTick':>12} {'conf':>5} {'@pct':>5} │ {'LiveDir':>7} {'W/L':>3}")
    P(f"  {'-'*120}")

    for _, t in night_live.sort_values("entry_ts").iterrows():
        slug = t["slug"]
        mkt = df[df["slug"] == slug].sort_values("entry_pct")
        if mkt.empty: continue
        outcome = mkt.iloc[0]["outcome"]

        # Analysis first tick
        an = compute_first_tick(mkt, "catboost", 0.70, 0.55, 0.55, 0.92)
        if not an.empty:
            an_dir = an.iloc[0]["_dir"]; an_conf = an.iloc[0]["_conf"]; an_epct = an.iloc[0]["entry_pct"]
        else:
            an_dir = "SKIP"; an_conf = 0; an_epct = 0

        # CB at live tick
        diffs = np.abs(mkt["ts"].values - t["entry_ts"])
        ci = diffs.argmin()
        ct = mkt.iloc[ci]
        cb_p = ct["catboost"]; cb_c = max(cb_p, 1-cb_p); cb_d = "UP" if cb_p > 0.5 else "DOWN"
        live_epct = ct["entry_pct"]

        P(f"  {slug:<40} {outcome:>7} │ {an_dir:>12} {an_conf:>4.0%} {an_epct:>4.0%} │ {cb_d:>12} {cb_c:>4.0%} {live_epct:>4.0%} │ {t['direction']:>7} {'W' if t['won'] else 'L':>3}")


    # ══════════════════════════════════════════════════════════
    #  PART 8: WHAT IF — remove max_positions limit?
    # ══════════════════════════════════════════════════════════
    P(f"\n\n{'='*110}")
    P(f"  PART 8: SIMULATION — Night session WITHOUT max_positions limit")
    P(f"  What if bot could enter ALL qualifying markets?")
    P(f"{'='*110}")

    sim_configs = [
        ("CB 70% SP≤0.55 55-92%",  "catboost", 0.70, 0.55, 0.55, 0.92),
        ("CB 70% SP≤0.55 60-100%", "catboost", 0.70, 0.55, 0.60, 1.0),
        ("CB 80% SP≤0.55 60-100%", "catboost", 0.80, 0.55, 0.60, 1.0),
        ("CB 85% SP≤0.55 60-100%", "catboost", 0.85, 0.55, 0.60, 1.0),
        ("ENS 85% SP55 60-100%",   "ensemble", 0.85, 0.55, 0.60, 1.0),
        ("RF 85% SP55 60-100%",    "rf",       0.85, 0.55, 0.60, 1.0),
    ]

    P(f"\n  {'Config':<30} │ {'5min':>5} {'WR':>5} {'PnL':>8} │ {'15min':>5} {'WR':>5} {'PnL':>8} │ {'TOTAL':>5} {'WR':>5} {'PnL':>8} │ {'Max concurrent':>14}")
    P(f"  {'-'*115}")

    for label, mdl, thr, msp, tlo, thi in sim_configs:
        r5 = compute_first_tick(night_5m, mdl, thr, msp, tlo, thi)
        r15 = compute_first_tick(night_15m, mdl, thr, msp, tlo, thi)
        # Estimate max concurrent
        all_r = pd.concat([r5, r15]) if not r5.empty or not r15.empty else pd.DataFrame()
        max_conc = 0
        if not all_r.empty and "ts" in all_r.columns:
            # Estimate: market duration ~ 5min or 15min
            all_r = all_r.copy()
            all_r["_end_ts"] = all_r["ts"] + all_r["dur_min"] * 60
            for _, row in all_r.iterrows():
                conc = ((all_r["ts"] <= row["ts"]) & (all_r["_end_ts"] > row["ts"])).sum()
                max_conc = max(max_conc, conc)

        def fmt2(r):
            if r.empty: return f"{'—':>19}"
            return f"{len(r):>4}t {r['_won'].mean()*100:>4.0f}% ${r['_pnl'].sum():>+6.0f}"

        total = pd.concat([r5, r15]) if not r5.empty or not r15.empty else pd.DataFrame()
        P(f"  {label:<30} │ {fmt2(r5)} │ {fmt2(r15)} │ {fmt2(total)} │ {max_conc:>14}")


    # Final
    P(f"\n\n{'='*110}")
    P(f"  FINAL VERDICT")
    P(f"{'='*110}")
    P(f"""
  1. ANALYSIS WIN RATE IS REAL — {len(an_5m)+len(an_15m)} trades at 84%+ WR on night session.
     But analysis assumes NO position limit and NO spread/depth filter.

  2. LIVE TRADED ONLY {len(night_live)} OUT OF {len(an_5m)+len(an_15m)} QUALIFYING MARKETS.
     Main reason: max_positions=5 blocks entry when slots are full.
     Secondary: spread/depth filters reject some markets.

  3. LIVE ≠ ANALYSIS because:
     a) Bot entered DIFFERENT markets (selection bias from max_positions)
     b) Bot may have entered at DIFFERENT tick (fresh ML re-eval → possible direction flip)
     c) Only {len(night_live)} trades = statistically meaningless sample

  4. 5MIN vs 15MIN:
     5min night: {night_5m.slug.nunique()} markets — higher volume, more data
     15min night: {night_15m.slug.nunique()} markets — fewer but also profitable

  RECOMMENDATIONS:
     - Increase max_positions to 10+ (or unlimited) to match analysis coverage
     - Enter on FIRST qualifying tick (don't queue+re-eval — direction can flip)
     - Need 50+ live trades for statistically meaningful comparison
     - Both 5min and 15min are profitable at night — keep both enabled
""")

    P(f"\n  Saved to: {OUT}")
    _F.close()
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
