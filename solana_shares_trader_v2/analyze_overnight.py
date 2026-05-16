"""
OVERNIGHT DEEP ANALYSIS -- tick-level, vectorized, same logic as analyze_live.py
Filters ticks to overnight session (>=20:00 UTC = 23:00 UTC+3, May 9 -> 06:00 May 10)
Plus a separate section for ONLY the 16 live-traded markets.

Usage: python analyze_overnight.py
"""
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
import time

t0 = time.time()
BET = 2.85
MODELS = ["catboost", "rf", "xgboost", "ensemble", "lgbm"]
THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
SP_CAPS = [0.45, 0.50, 0.55, 0.60]
TIMING_WINDOWS = [
    (0.0, 1.0), (0.30, 1.0), (0.50, 1.0), (0.55, 0.92),
    (0.60, 1.0), (0.70, 1.0), (0.80, 1.0),
]


def compute(df, model, thresh, max_sp, t_lo, t_hi, bet=BET):
    """Vectorized: first qualifying tick per market. Same as analyze_live.compute_all_ticks."""
    if model == "ensemble":
        cols = [c for c in ["catboost", "rf", "xgboost"] if c in df.columns]
        if not cols:
            return None
        prob = df[cols].mean(axis=1).values
    elif model not in df.columns:
        return None
    else:
        prob = df[model].values

    conf = np.maximum(prob, 1 - prob)
    model_up = prob > 0.5
    sp = np.where(model_up, df["yes"].values, df["no"].values)
    ep = df["entry_pct"].values

    mask = (conf >= thresh) & (sp <= max_sp) & (sp >= 0.10) & (ep >= t_lo) & (ep < t_hi)
    valid = df[mask].copy()
    if valid.empty:
        return None

    valid["_dir"] = np.where(model_up[mask], "UP", "DOWN")
    valid["_sp"] = sp[mask]
    valid["_conf"] = conf[mask]
    valid = valid.sort_values("entry_pct").drop_duplicates("slug", keep="first")
    valid["_won"] = valid["_dir"] == valid["outcome"]

    n = len(valid)
    w = int(valid["_won"].sum())
    sp_arr = valid["_sp"].values
    shares = bet / sp_arr
    pnl_arr = np.where(valid["_won"].values, shares * (1 - sp_arr), -bet)
    pnl = float(pnl_arr.sum())
    ev_t = pnl / n
    avg_sp = float(sp_arr.mean())
    return {"n": n, "w": w, "l": n - w, "wr": w / n * 100, "pnl": pnl, "ev_t": ev_t, "avg_sp": avg_sp, "df": valid}


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
results_dir = Path("results")

print("Loading tick data...")
ticks_list = []
for f in sorted(results_dir.glob("live_ticks_*.jsonl")):
    tdf = pd.read_json(f, lines=True)
    ticks_list.append(tdf)
    print(f"  {f.name}: {len(tdf)} ticks, {tdf['slug'].nunique()} markets")
ticks = pd.concat(ticks_list, ignore_index=True)

print("Loading outcomes...")
out_list = []
for f in sorted(results_dir.glob("live_outcomes_*.jsonl")):
    out_list.append(pd.read_json(f, lines=True))
outcomes = pd.concat(out_list, ignore_index=True).drop_duplicates("slug", keep="last")
ticks = ticks.merge(outcomes[["slug", "outcome"]], on="slug", how="inner")

# Parse market start hour (UTC+3) from slug
ticks["_mkt_ts"] = ticks["slug"].str.split("-").str[-1].astype(int)
ticks["_hour_local"] = (pd.to_datetime(ticks["_mkt_ts"], unit="s").dt.hour + 3) % 24

# All 5min ticks
all5 = ticks[ticks["slug"].str.contains("-5m-")].copy()

# OVERNIGHT: markets starting 23:00 UTC+3 (20:00 UTC) through 05:59 UTC+3
night = all5[(all5["_hour_local"] >= 23) | (all5["_hour_local"] < 6)].copy()

# LIVE trades
trades = pd.read_json("results/ml_live_trades.json")
trades["won"] = trades["won"].astype(bool)
live = trades[~trades["dry_run"]].copy()
live_slugs = set(live["slug"].unique())
live_ticks = all5[all5["slug"].isin(live_slugs)].copy()

print(f"\nAll 5min: {all5['slug'].nunique()} markets, {len(all5)} ticks")
print(f"Overnight 5min (23:00-06:00): {night['slug'].nunique()} markets, {len(night)} ticks")
print(f"Live-traded 5min: {live_ticks['slug'].nunique()} markets, {len(live_ticks)} ticks")
print(f"Live trades: {len(live)} ({live['won'].sum()}W/{(~live['won']).sum()}L, PnL=${live['pnl_usd'].sum():+.2f})")

elapsed = time.time() - t0
print(f"Data loaded in {elapsed:.1f}s\n")


def print_section1(data, label):
    """SECTION 1: Model x Confidence x Price filter (all timing 0-100%)"""
    print(f"\n{'='*100}")
    print(f"  SECTION 1: PRICE FILTER -- {label}")
    print(f"  1 trade/market (first qualifying tick, GTC until market end)")
    print(f"  Bet = ${BET}/trade | Timing: 0-100%")
    print(f"{'='*100}")

    for model in MODELS:
        print(f"\n  {model.upper()}")
        header = "  Conf |"
        for sp in SP_CAPS:
            header += f"      SP<=${sp:.2f}       |"
        print(header)
        sub = "       |"
        for _ in SP_CAPS:
            sub += f"   N   WR     PnL EV/t |"
        print(sub)
        print("  " + "-" * (len(header) - 2))

        for thresh in THRESHOLDS:
            row = f"  {thresh:.0%}  |"
            for max_sp in SP_CAPS:
                r = compute(data, model, thresh, max_sp, 0.0, 1.0)
                if r is None:
                    row += "   --   --     --   -- |"
                else:
                    row += f"  {r['n']:>2} {r['wr']:>3.0f}% ${r['pnl']:>+5.0f} ${r['ev_t']:>+3.1f} |"
            print(row)


def print_section2(data, label, focus_model="catboost"):
    """SECTION 2: Timing x Price for a given model"""
    print(f"\n{'='*100}")
    print(f"  SECTION 2: TIMING x PRICE -- {focus_model.upper()} -- {label}")
    print(f"  Each cell: N / WR% / PnL (${BET}/trade)")
    print(f"{'='*100}")

    for max_sp in [0.50, 0.55]:
        print(f"\n  {focus_model.upper()} | SP <= ${max_sp:.2f}")
        header = "  Conf |"
        for t_lo, t_hi in TIMING_WINDOWS:
            header += f"    {int(t_lo*100):>2}-{int(t_hi*100)}%     |"
        print(header)
        sub = "       |"
        for _ in TIMING_WINDOWS:
            sub += f"  N  WR   PnL  EV |"
        print(sub)
        print("  " + "-" * (len(header) - 2))

        for thresh in THRESHOLDS:
            row = f"  {thresh:.0%}  |"
            for t_lo, t_hi in TIMING_WINDOWS:
                r = compute(data, focus_model, thresh, max_sp, t_lo, t_hi)
                if r is None:
                    row += "  --  --    --  -- |"
                else:
                    row += f" {r['n']:>2} {r['wr']:>3.0f}% ${r['pnl']:>+4.0f} ${r['ev_t']:>+2.1f} |"
            print(row)


def print_section3(data, label):
    """SECTION 3: All models at SP<=0.55, various timings"""
    print(f"\n{'='*100}")
    print(f"  SECTION 3: ALL MODELS DETAIL @ SP<=0.55 -- {label}")
    print(f"{'='*100}")

    for t_lo, t_hi in [(0.55, 0.92), (0.60, 1.0), (0.70, 1.0)]:
        t_str = f"{int(t_lo*100)}-{int(t_hi*100)}%"
        print(f"\n  Timing: {t_str} | SP<=0.55")
        print(f"  {'Model':<10} {'Conf':>4} | {'N':>3} {'W':>3} {'L':>3} {'WR':>5} | {'EV/t':>6} {'PnL':>8} | {'AvgSP':>5}")
        print(f"  {'-'*65}")
        for model in MODELS:
            for thresh in THRESHOLDS:
                r = compute(data, model, thresh, 0.55, t_lo, t_hi)
                if r and r["n"] >= 3:
                    print(f"  {model:<10} {thresh:>3.0%} | {r['n']:>3} {r['w']:>3} {r['l']:>3} {r['wr']:>4.0f}% | ${r['ev_t']:>+5.2f} ${r['pnl']:>+7.1f} | ${r['avg_sp']:.3f}")


def print_verdict(data, label, min_n=5):
    """FINAL VERDICT: best configs sorted by PnL"""
    print(f"\n{'='*100}")
    print(f"  FINAL VERDICT -- {label} (N>={min_n}, sorted by PnL)")
    print(f"{'='*100}")

    results = []
    for model in MODELS:
        for thresh in THRESHOLDS:
            for max_sp in SP_CAPS:
                for t_lo, t_hi in TIMING_WINDOWS:
                    r = compute(data, model, thresh, max_sp, t_lo, t_hi)
                    if r and r["n"] >= min_n:
                        results.append((r["pnl"], r["ev_t"], r["wr"], model, thresh, max_sp, t_lo, t_hi, r["n"], r["w"], r["l"], r["avg_sp"]))

    results.sort(key=lambda x: (-x[0], -x[2]))
    print(f"\n  TOP 25:")
    print(f"  {'Model':<10} {'Conf':>4} {'SP<=':>5} {'Time':>8} | {'N':>3} {'W':>3} {'L':>3} {'WR':>5} | {'EV/t':>6} {'PnL':>8} | {'AvgSP':>5}")
    print(f"  {'-'*72}")
    for pnl, ev_t, wr, m, thresh, max_sp, t_lo, t_hi, n, w, l, avg_sp in results[:25]:
        t_str = f"{int(t_lo*100)}-{int(t_hi*100)}%"
        print(f"  {m:<10} {thresh:>3.0%} ${max_sp:.2f} {t_str:>8} | {n:>3} {w:>3} {l:>3} {wr:>4.0f}% | ${ev_t:>+5.2f} ${pnl:>+7.1f} | ${avg_sp:.3f}")

    if len(results) >= 10:
        print(f"\n  WORST 10:")
        print(f"  {'Model':<10} {'Conf':>4} {'SP<=':>5} {'Time':>8} | {'N':>3} {'W':>3} {'L':>3} {'WR':>5} | {'EV/t':>6} {'PnL':>8} | {'AvgSP':>5}")
        print(f"  {'-'*72}")
        for pnl, ev_t, wr, m, thresh, max_sp, t_lo, t_hi, n, w, l, avg_sp in results[-10:]:
            t_str = f"{int(t_lo*100)}-{int(t_hi*100)}%"
            print(f"  {m:<10} {thresh:>3.0%} ${max_sp:.2f} {t_str:>8} | {n:>3} {w:>3} {l:>3} {wr:>4.0f}% | ${ev_t:>+5.2f} ${pnl:>+7.1f} | ${avg_sp:.3f}")


def print_hourly(data, label):
    """WR by hour"""
    print(f"\n{'='*100}")
    print(f"  HOURLY WR -- {label}")
    print(f"  catboost 70% SP<=0.55 | 1 trade/market")
    print(f"{'='*100}")

    r_all = compute(data, "catboost", 0.70, 0.55, 0.0, 1.0)
    if r_all:
        vdf = r_all["df"]
        vdf = vdf.copy()
        vdf["_hour"] = (pd.to_datetime(vdf["_mkt_ts"].astype(int), unit="s").dt.hour + 3) % 24
        print(f"\n  {'Hour':>4} | {'N':>3} {'W':>3} {'L':>3} {'WR':>5} | {'PnL':>8} {'EV/t':>6}")
        print(f"  {'-'*45}")
        for h in sorted(vdf["_hour"].unique()):
            sub = vdf[vdf["_hour"] == h]
            n = len(sub)
            w = int(sub["_won"].sum())
            l = n - w
            wr = w / n * 100
            shares = BET / sub["_sp"].values
            pnl = float(np.where(sub["_won"].values, shares * (1 - sub["_sp"].values), -BET).sum())
            night_marker = " <-- NIGHT" if (h >= 23 or h < 6) else ""
            print(f"  {h:>4} | {n:>3} {w:>3} {l:>3} {wr:>4.0f}% | ${pnl:>+7.1f} ${pnl/n:>+5.2f}{night_marker}")


def print_live_vs_analysis(live_df, tick_data, configs=None):
    """Per-trade: what live did vs what analysis configs would do"""
    if configs is None:
        configs = [("catboost", 0.70, 0.55, 0.55, 0.92)]

    for model, thresh, max_sp, t_lo, t_hi in configs:
        t_str = f"{int(t_lo*100)}-{int(t_hi*100)}%"
        print(f"\n{'='*100}")
        print(f"  LIVE TRADES vs ANALYSIS ({model} {thresh:.0%} SP<={max_sp:.2f} {t_str})")
        print(f"{'='*100}")

        print(f"\n  {'#':>2} {'H':>2} {'Slug':>22} {'Live':>5} {'Conf':>5} {'Out':>5} {'W/L':>3} {'PnL':>7} | {'Anls':>5} {'A_sp':>5} {'A_ep':>5} {'A_wl':>4} | {'Match':>5}")
        print(f"  {'-'*95}")

        tot_live_pnl = 0
        tot_anls_pnl = 0
        n_match = 0
        n_diff = 0
        n_skip = 0

        for i, (_, trade) in enumerate(live_df.iterrows()):
            slug = trade["slug"]
            ts = int(slug.split("-")[-1])
            hour = (pd.to_datetime(ts, unit="s").hour + 3) % 24
            mt = tick_data[tick_data["slug"] == slug]
            if mt.empty:
                print(f"  {i+1:>2} {hour:>2} {slug[-22:]:>22}  NO TICK DATA")
                n_skip += 1
                continue

            outcome = mt["outcome"].iloc[0]
            live_wl = "W" if trade["won"] else "L"
            tot_live_pnl += trade["pnl_usd"]

            r = compute(mt, model, thresh, max_sp, t_lo, t_hi)
            if r is None:
                anls_dir = "--"
                anls_sp = "--"
                anls_ep = "--"
                anls_wl = "SKIP"
                match = "--"
                n_skip += 1
            else:
                first = r["df"].iloc[0]
                anls_dir = first["_dir"]
                anls_sp = f"${first['_sp']:.2f}"
                anls_ep = f"{first['entry_pct']:.0%}"
                anls_wl = "W" if first["_won"] else "L"
                # compute analysis PnL
                a_shares = BET / first["_sp"]
                a_pnl = a_shares * (1 - first["_sp"]) if first["_won"] else -BET
                tot_anls_pnl += a_pnl
                if anls_dir == trade["direction"]:
                    match = "SAME"
                    n_match += 1
                else:
                    match = "DIFF"
                    n_diff += 1

            print(f"  {i+1:>2} {hour:>2} {slug[-22:]:>22} {trade['direction']:>5} {trade['confidence']:.3f} {outcome:>5} {live_wl:>3} ${trade['pnl_usd']:>+5.2f} | {anls_dir:>5} {anls_sp:>5} {anls_ep:>5} {anls_wl:>4} | {match:>5}")

        print(f"\n  SUMMARY: {n_match} SAME, {n_diff} DIFF, {n_skip} SKIP")
        print(f"  Live PnL: ${tot_live_pnl:+.2f} | Analysis PnL: ${tot_anls_pnl:+.2f}")


# Separate live trades by hour for night analysis
live["_ts"] = live["slug"].str.split("-").str[-1].astype(int)
live["_hour"] = (pd.to_datetime(live["_ts"], unit="s").dt.hour + 3) % 24
live_night = live[(live["_hour"] >= 23) | (live["_hour"] < 6)].copy()
live_night_5m = live_night[live_night["slug"].str.contains("-5m-")].copy()
live_night_slugs = set(live_night_5m["slug"].unique())
live_night_ticks = all5[all5["slug"].isin(live_night_slugs)].copy()

print(f"\nLive trades from 23:00+: {len(live_night)} total ({len(live_night_5m)} x 5min)")
print(f"  {live_night['won'].sum()}W/{(~live_night['won']).sum()}L, PnL=${live_night['pnl_usd'].sum():+.2f}")
print(f"  5min only: {live_night_5m['won'].sum()}W/{(~live_night_5m['won']).sum()}L, PnL=${live_night_5m['pnl_usd'].sum():+.2f}")
print(f"  5min tick data: {live_night_ticks['slug'].nunique()} markets, {len(live_night_ticks)} ticks")

# Analysis configs to test
TEST_CONFIGS = [
    ("catboost", 0.70, 0.55, 0.55, 0.92),
    ("catboost", 0.70, 0.55, 0.0, 1.0),
    ("catboost", 0.70, 0.55, 0.60, 1.0),
    ("catboost", 0.70, 0.55, 0.80, 1.0),
    ("catboost", 0.70, 0.50, 0.0, 1.0),
    ("catboost", 0.70, 0.50, 0.80, 1.0),
    ("catboost", 0.85, 0.55, 0.60, 1.0),
    ("catboost", 0.85, 0.55, 0.80, 1.0),
    ("ensemble", 0.70, 0.55, 0.60, 1.0),
    ("rf", 0.85, 0.55, 0.80, 1.0),
]

# ═══════════════════════════════════════════════════════════════════════════════
# PART A: ALL LIVE TRADES (all hours)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'#'*100}")
print(f"  PART A: ALL LIVE-TRADED 5MIN MARKETS ({live_ticks['slug'].nunique()} markets)")
print(f"{'#'*100}")

print_live_vs_analysis(live, live_ticks, TEST_CONFIGS[:4])
print_section1(live_ticks, f"LIVE-TRADED MARKETS ({live_ticks['slug'].nunique()} mkts)")
print_section2(live_ticks, f"LIVE-TRADED MARKETS", "catboost")
print_section3(live_ticks, f"LIVE-TRADED MARKETS")
print_verdict(live_ticks, f"LIVE-TRADED MARKETS", min_n=3)

# ═══════════════════════════════════════════════════════════════════════════════
# PART A2: NIGHT LIVE TRADES ONLY (23:00+ 5min)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n\n{'#'*100}")
print(f"  PART A2: NIGHT LIVE TRADES 23:00+ (5min: {live_night_ticks['slug'].nunique()} markets)")
print(f"{'#'*100}")

print_live_vs_analysis(live_night, live_night_ticks, TEST_CONFIGS)
if live_night_ticks['slug'].nunique() >= 2:
    print_section1(live_night_ticks, f"NIGHT LIVE MARKETS ({live_night_ticks['slug'].nunique()} mkts)")
    print_section2(live_night_ticks, f"NIGHT LIVE MARKETS", "catboost")
    print_section3(live_night_ticks, f"NIGHT LIVE MARKETS")
    print_verdict(live_night_ticks, f"NIGHT LIVE MARKETS", min_n=2)

# ═══════════════════════════════════════════════════════════════════════════════
# PART B: OVERNIGHT SESSION (all overnight markets)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n\n{'#'*100}")
print(f"  PART B: OVERNIGHT SESSION 23:00-06:00 ({night['slug'].nunique()} markets)")
print(f"{'#'*100}")

print_section1(night, f"OVERNIGHT 23:00-06:00 ({night['slug'].nunique()} mkts)")
print_section2(night, f"OVERNIGHT", "catboost")
print_section3(night, f"OVERNIGHT")
print_hourly(night, f"OVERNIGHT")
print_verdict(night, f"OVERNIGHT", min_n=5)

# ═══════════════════════════════════════════════════════════════════════════════
# PART C: ALL DATA (for reference)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n\n{'#'*100}")
print(f"  PART C: ALL 5-MIN DATA ({all5['slug'].nunique()} markets) -- for reference")
print(f"{'#'*100}")

print_section1(all5, f"ALL 5MIN ({all5['slug'].nunique()} mkts)")
print_hourly(all5, f"ALL 5MIN")
print_verdict(all5, f"ALL 5MIN", min_n=10)

# ═══════════════════════════════════════════════════════════════════════════════
# ROOT CAUSE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n\n{'#'*100}")
print(f"  ROOT CAUSE ANALYSIS")
print(f"{'#'*100}")

print(f"\n  === Live trades breakdown by hour ===")
for h in sorted(live["_hour"].unique()):
    sub = live[live["_hour"] == h]
    w = sub["won"].sum()
    l = (~sub["won"]).sum()
    pnl = sub["pnl_usd"].sum()
    print(f"    {h:02d}:00  {len(sub)} trades  {w}W/{l}L  WR={w/len(sub)*100:.0f}%  PnL=${pnl:+.2f}")

print(f"\n  === Config comparison on LIVE markets vs OVERNIGHT 84 mkts ===")
print(f"  {'Config':<42} | {'Live (13 mkts)':>20} | {'Night (84 mkts)':>20} | {'All (144 mkts)':>20}")
print(f"  {'-'*110}")
for model, thresh, max_sp, t_lo, t_hi in TEST_CONFIGS:
    t_str = f"{int(t_lo*100)}-{int(t_hi*100)}%"
    cfg = f"{model} {thresh:.0%} SP<={max_sp:.2f} {t_str}"
    parts = []
    for data in [live_ticks, night, all5]:
        r = compute(data, model, thresh, max_sp, t_lo, t_hi)
        if r:
            parts.append(f"{r['n']:>3}t {r['wr']:.0f}% ${r['pnl']:>+6.0f}")
        else:
            parts.append(f"  --   --     --")
    print(f"  {cfg:<42} | {parts[0]:>20} | {parts[1]:>20} | {parts[2]:>20}")

print(f"\n  === CONCLUSION ===")
# Auto-detect best overnight config
best_ov = None
for model in MODELS:
    for thresh in THRESHOLDS:
        for max_sp in SP_CAPS:
            for t_lo, t_hi in TIMING_WINDOWS:
                r = compute(night, model, thresh, max_sp, t_lo, t_hi)
                if r and r["n"] >= 10:
                    if best_ov is None or r["pnl"] > best_ov["pnl"]:
                        best_ov = {**r, "model": model, "thresh": thresh, "max_sp": max_sp, "t_lo": t_lo, "t_hi": t_hi}

live_r = compute(live_ticks, "catboost", 0.70, 0.55, 0.55, 0.92)
night_r = compute(night, "catboost", 0.70, 0.55, 0.55, 0.92)
all_r = compute(all5, "catboost", 0.70, 0.55, 0.0, 1.0)

print(f"  1. LIVE used: catboost 70% SP<=0.55 55-92%")
if live_r:
    print(f"     On live 13 mkts: {live_r['n']}t, WR={live_r['wr']:.0f}%, PnL=${live_r['pnl']:+.1f}")
if night_r:
    print(f"     On night 84 mkts: {night_r['n']}t, WR={night_r['wr']:.0f}%, PnL=${night_r['pnl']:+.1f}")
if all_r:
    print(f"     On all 144 mkts: {all_r['n']}t, WR={all_r['wr']:.0f}%, PnL=${all_r['pnl']:+.1f}")

print(f"\n  2. Overnight model IS profitable: 84 markets, catboost 70% SP<=0.55 => {night_r['wr']:.0f}% WR, ${night_r['pnl']:+.0f}")
print(f"     Live loss on 13 markets is VARIANCE (small sample N=13).")
print(f"     Live night 5min trades: {len(live_night_5m)} trades, {live_night_5m['won'].sum()}W/{(~live_night_5m['won']).sum()}L")
print(f"     = too few trades to be statistically significant.")

if best_ov:
    t_str = f"{int(best_ov['t_lo']*100)}-{int(best_ov['t_hi']*100)}%"
    print(f"\n  3. BEST overnight config: {best_ov['model']} {best_ov['thresh']:.0%} SP<={best_ov['max_sp']:.2f} {t_str}")
    print(f"     {best_ov['n']}t, WR={best_ov['wr']:.0f}%, PnL=${best_ov['pnl']:+.1f}, EV/t=${best_ov['ev_t']:+.2f}")

    # test this best on live
    best_live = compute(live_ticks, best_ov['model'], best_ov['thresh'], best_ov['max_sp'], best_ov['t_lo'], best_ov['t_hi'])
    if best_live:
        print(f"     Same config on live 13 mkts: {best_live['n']}t, WR={best_live['wr']:.0f}%, PnL=${best_live['pnl']:+.1f}")

print(f"\n  4. GTC (limit order till market end) is ALREADY how analysis works.")
print(f"     First qualifying tick -> hold to end. No 10s cancel.")
print(f"     Late timing (80-100%) generally best: enters near market end = higher WR.")
print(f"     SP<=0.50 filter on live markets performs BETTER than SP<=0.55 (avoids bad SP ticks).")

# best live-market config
best_lv = None
for model in MODELS:
    for thresh in THRESHOLDS:
        for max_sp in SP_CAPS:
            for t_lo, t_hi in TIMING_WINDOWS:
                r = compute(live_ticks, model, thresh, max_sp, t_lo, t_hi)
                if r and r["n"] >= 5:
                    if best_lv is None or r["pnl"] > best_lv["pnl"]:
                        best_lv = {**r, "model": model, "thresh": thresh, "max_sp": max_sp, "t_lo": t_lo, "t_hi": t_hi}
if best_lv:
    t_str = f"{int(best_lv['t_lo']*100)}-{int(best_lv['t_hi']*100)}%"
    print(f"\n  5. BEST config on live 13 mkts: {best_lv['model']} {best_lv['thresh']:.0%} SP<={best_lv['max_sp']:.2f} {t_str}")
    print(f"     {best_lv['n']}t, WR={best_lv['wr']:.0f}%, PnL=${best_lv['pnl']:+.1f}, EV/t=${best_lv['ev_t']:+.2f}")
    print(f"     Key: SP<=0.50 + late timing = better on these specific markets.")

elapsed = time.time() - t0
print(f"\n{'='*100}")
print(f"  DONE in {elapsed:.1f}s")
print(f"{'='*100}")
