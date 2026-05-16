"""
DEEP FULL ANALYSIS -- ML quality vs Filter impact
Answers: is the loss due to bad filters or bad ML?

Subsets:
 - Live-traded markets (all)
 - From 15:00 (afternoon + night session)
 - From 23:00 (night only)
 - ALL 5min markets

For each: raw ML accuracy (no filter), filter impact tables, trade-by-trade.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import time, sys

t0 = time.time()
BET = 2.85
MODELS = ["catboost", "rf", "xgboost", "ensemble", "lgbm"]
THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
SP_CAPS = [0.45, 0.50, 0.55, 0.60, 1.00]  # 1.00 = NO CAP
TIMING_WINDOWS = [
    (0.0, 1.0), (0.30, 1.0), (0.55, 0.92),
    (0.60, 1.0), (0.70, 1.0), (0.80, 1.0),
]

P = print


def compute(df, model, thresh, max_sp, t_lo, t_hi, bet=BET):
    """Vectorized: first qualifying tick per market -> GTC to market end."""
    if df.empty:
        return None
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


# ====================== LOAD DATA ======================
results_dir = Path("results")

P("Loading tick data...")
ticks_list = []
for f in sorted(results_dir.glob("live_ticks_*.jsonl")):
    tdf = pd.read_json(f, lines=True)
    ticks_list.append(tdf)
    P(f"  {f.name}: {len(tdf)} ticks, {tdf['slug'].nunique()} markets")
ticks = pd.concat(ticks_list, ignore_index=True)

P("Loading outcomes...")
out_list = []
for f in sorted(results_dir.glob("live_outcomes_*.jsonl")):
    out_list.append(pd.read_json(f, lines=True))
outcomes = pd.concat(out_list, ignore_index=True).drop_duplicates("slug", keep="last")
ticks = ticks.merge(outcomes[["slug", "outcome"]], on="slug", how="inner")

ticks["_mkt_ts"] = ticks["slug"].str.split("-").str[-1].astype(int)
ticks["_hour"] = (pd.to_datetime(ticks["_mkt_ts"], unit="s").dt.hour + 3) % 24

all5 = ticks[ticks["slug"].str.contains("-5m-")].copy()
from15 = all5[all5["_hour"] >= 15].copy()  # 15:00+ (afternoon + evening + night)
night = all5[(all5["_hour"] >= 23) | (all5["_hour"] < 6)].copy()

# Live trades
trades = pd.read_json("results/ml_live_trades.json")
trades["won"] = trades["won"].astype(bool)
live = trades[~trades["dry_run"]].copy()
live["_ts"] = live["slug"].str.split("-").str[-1].astype(int)
live["_hour"] = (pd.to_datetime(live["_ts"], unit="s").dt.hour + 3) % 24
live_slugs = set(live["slug"].unique())
live_5m = live[live["slug"].str.contains("-5m-")].copy()
live_ticks = all5[all5["slug"].isin(live_slugs)].copy()

live_night = live[(live["_hour"] >= 23) | (live["_hour"] < 6)].copy()
live_night_5m = live_night[live_night["slug"].str.contains("-5m-")].copy()

P(f"\nAll 5min: {all5['slug'].nunique()} markets, {len(all5)} ticks")
P(f"From 15:00: {from15['slug'].nunique()} markets, {len(from15)} ticks")
P(f"Night 23:00-06:00: {night['slug'].nunique()} markets, {len(night)} ticks")
P(f"Live trades: {len(live)} ({live['won'].sum()}W/{(~live['won']).sum()}L, PnL=${live['pnl_usd'].sum():+.2f})")
P(f"  5min live: {len(live_5m)} ({live_5m['won'].sum()}W/{(~live_5m['won']).sum()}L, PnL=${live_5m['pnl_usd'].sum():+.2f})")
P(f"  night live: {len(live_night)} ({live_night['won'].sum()}W/{(~live_night['won']).sum()}L)")
P(f"  night 5m live: {len(live_night_5m)} ({live_night_5m['won'].sum()}W/{(~live_night_5m['won']).sum()}L)")
P(f"Live 5min tick data: {live_ticks['slug'].nunique()} mkts, {len(live_ticks)} ticks")
P(f"Loaded in {time.time()-t0:.1f}s")

# ========================================================================
#  PART 1: TRADE-BY-TRADE VERIFICATION
# ========================================================================
P(f"\n\n{'#'*100}")
P(f"  PART 1: TRADE-BY-TRADE -- ALL 16 LIVE TRADES")
P(f"  What LIVE did vs what ANALYSIS would do with different configs")
P(f"{'#'*100}")

configs_to_test = [
    ("catboost", 0.70, 0.55, 0.55, 0.92, "CB70 SP55 55-92% (LIVE CFG)"),
    ("catboost", 0.70, 1.00, 0.00, 1.00, "CB70 NO_FILTER"),
    ("catboost", 0.70, 0.55, 0.00, 1.00, "CB70 SP55 0-100%"),
    ("catboost", 0.70, 0.55, 0.80, 1.00, "CB70 SP55 80-100%"),
    ("catboost", 0.70, 0.50, 0.80, 1.00, "CB70 SP50 80-100%"),
    ("catboost", 0.85, 0.55, 0.80, 1.00, "CB85 SP55 80-100%"),
    ("ensemble", 0.70, 0.55, 0.60, 1.00, "ENS70 SP55 60-100%"),
]

P(f"\n  {'#':>2} {'H':>2} {'Slug':>22} {'Out':>4} | {'LIVE':>6} {'W/L':>3} {'$PnL':>7} |", end="")
for _, _, _, _, _, label in configs_to_test:
    P(f" {label:>22} |", end="")
P()
P(f"  {'-'*180}")

for i, (_, trade) in enumerate(live.iterrows()):
    slug = trade["slug"]
    ts = int(slug.split("-")[-1])
    hour = (pd.to_datetime(ts, unit="s").hour + 3) % 24
    mt = all5[all5["slug"] == slug]
    outcome = mt["outcome"].iloc[0] if not mt.empty else "?"

    P(f"  {i+1:>2} {hour:>2} {slug[-22:]:>22} {outcome:>4} | {trade['direction']:>6} {('W' if trade['won'] else 'L'):>3} ${trade['pnl_usd']:>+5.2f} |", end="")

    for model, thresh, max_sp, t_lo, t_hi, label in configs_to_test:
        if mt.empty:
            P(f" {'NO DATA':>22} |", end="")
        else:
            r = compute(mt, model, thresh, max_sp, t_lo, t_hi)
            if r is None:
                P(f" {'SKIP':>22} |", end="")
            else:
                first = r["df"].iloc[0]
                wl = "W" if first["_won"] else "L"
                a_shares = BET / first["_sp"]
                a_pnl = a_shares * (1 - first["_sp"]) if first["_won"] else -BET
                P(f" {first['_dir']:>4} {wl} ep={first['entry_pct']:.0%} ${a_pnl:>+5.1f} |", end="")
    P()

# Totals row
P(f"  {'TOTAL':>28} {'':>4} | {live['won'].sum()}W/{(~live['won']).sum()}L   ${live['pnl_usd'].sum():>+5.2f} |", end="")
for model, thresh, max_sp, t_lo, t_hi, label in configs_to_test:
    r = compute(live_ticks, model, thresh, max_sp, t_lo, t_hi)
    if r:
        P(f"  {r['w']}W/{r['l']}L WR={r['wr']:.0f}% ${r['pnl']:>+5.0f} |", end="")
    else:
        P(f" {'--':>22} |", end="")
P()


# ========================================================================
#  PART 2: RAW ML QUALITY -- NO FILTERS (first tick with conf >= thresh)
#  SP up to 1.0, timing 0-100%
#  This shows: how good is the MODEL itself, without any filters?
# ========================================================================
P(f"\n\n{'#'*100}")
P(f"  PART 2: RAW ML QUALITY -- NO SP/TIMING FILTER")
P(f"  First tick where conf >= threshold, GTC to market end. SP up to $1.00.")
P(f"  Shows: is the MODEL right or wrong? Filters removed.")
P(f"{'#'*100}")

datasets = [
    (live_ticks, f"LIVE MARKETS ({live_ticks['slug'].nunique()} mkts)"),
    (from15, f"FROM 15:00 ({from15['slug'].nunique()} mkts)"),
    (night, f"NIGHT 23-06 ({night['slug'].nunique()} mkts)"),
    (all5, f"ALL 5MIN ({all5['slug'].nunique()} mkts)"),
]

for data, label in datasets:
    P(f"\n{'='*100}")
    P(f"  {label} -- NO FILTER (SP<=1.00, timing 0-100%)")
    P(f"{'='*100}")
    P(f"\n  {'Model':<10} {'Conf':>4} | {'N':>3} {'W':>3} {'L':>3} {'WR':>5} | {'EV/t':>6} {'PnL':>9} | {'AvgSP':>5}")
    P(f"  {'-'*60}")
    for model in MODELS:
        for thresh in THRESHOLDS:
            r = compute(data, model, thresh, 1.00, 0.0, 1.0)
            if r and r["n"] >= 2:
                P(f"  {model:<10} {thresh:>3.0%} | {r['n']:>3} {r['w']:>3} {r['l']:>3} {r['wr']:>4.0f}% | ${r['ev_t']:>+5.2f} ${r['pnl']:>+8.1f} | ${r['avg_sp']:.3f}")


# ========================================================================
#  PART 3: FILTER IMPACT -- catboost 70%
#  Compare: NO_CAP vs SP<=0.55 vs SP<=0.50, different timings
#  Shows: does the FILTER help or hurt?
# ========================================================================
P(f"\n\n{'#'*100}")
P(f"  PART 3: FILTER IMPACT -- CATBOOST 70%")
P(f"  Comparing SP caps and timing windows. Same model, different filters.")
P(f"  Question: does the filter HELP or HURT?")
P(f"{'#'*100}")

for data, label in datasets:
    P(f"\n{'='*100}")
    P(f"  {label}")
    P(f"{'='*100}")

    header = f"  {'SP cap':>6} {'Timing':>8} |"
    header += f" {'N':>3} {'W':>3} {'L':>3} {'WR':>5} | {'EV/t':>6} {'PnL':>9} | {'AvgSP':>5}"
    P(f"\n{header}")
    P(f"  {'-'*65}")

    for max_sp in SP_CAPS:
        sp_label = "NO_CAP" if max_sp >= 1.0 else f"<={max_sp:.2f}"
        for t_lo, t_hi in TIMING_WINDOWS:
            t_str = f"{int(t_lo*100)}-{int(t_hi*100)}%"
            r = compute(data, "catboost", 0.70, max_sp, t_lo, t_hi)
            if r and r["n"] >= 2:
                P(f"  {sp_label:>6} {t_str:>8} | {r['n']:>3} {r['w']:>3} {r['l']:>3} {r['wr']:>4.0f}% | ${r['ev_t']:>+5.2f} ${r['pnl']:>+8.1f} | ${r['avg_sp']:.3f}")
            elif r is None:
                P(f"  {sp_label:>6} {t_str:>8} |  -- --  --   -- |    --        -- |    --")
        P(f"  {'-'*65}")


# ========================================================================
#  PART 4: FULL MODEL x THRESHOLD x SP TABLE
#  For live markets and overnight, all models, with NO_CAP and SP<=0.55
# ========================================================================
P(f"\n\n{'#'*100}")
P(f"  PART 4: ALL MODELS -- LIVE MARKETS + NIGHT")
P(f"  Each: conf threshold, no SP cap vs SP<=0.55, timing 0-100% (GTC)")
P(f"{'#'*100}")

for data, label in [(live_ticks, f"LIVE ({live_ticks['slug'].nunique()} mkts)"),
                     (night, f"NIGHT ({night['slug'].nunique()} mkts)")]:
    P(f"\n{'='*100}")
    P(f"  {label}")
    P(f"{'='*100}")

    for model in MODELS:
        P(f"\n  {model.upper()}")
        P(f"  {'Conf':>4} |  {'NO SP CAP (<=1.00)':^25} | {'SP <= 0.55':^25} | {'SP <= 0.50':^25} |")
        P(f"       |  {'N':>3} {'WR':>4} {'PnL':>8} {'EV/t':>6} | {'N':>3} {'WR':>4} {'PnL':>8} {'EV/t':>6} | {'N':>3} {'WR':>4} {'PnL':>8} {'EV/t':>6} |")
        P(f"  {'-'*90}")
        for thresh in THRESHOLDS:
            row = f"  {thresh:>3.0%} |"
            for max_sp in [1.0, 0.55, 0.50]:
                r = compute(data, model, thresh, max_sp, 0.0, 1.0)
                if r and r["n"] >= 1:
                    row += f"  {r['n']:>3} {r['wr']:>3.0f}% ${r['pnl']:>+7.0f} ${r['ev_t']:>+4.1f} |"
                else:
                    row += f"   -- --% $     -- $  -- |"
            P(row)


# ========================================================================
#  PART 5: BEST CONFIG FINDER -- for each subset
# ========================================================================
P(f"\n\n{'#'*100}")
P(f"  PART 5: BEST CONFIGS -- ALL SUBSETS")
P(f"{'#'*100}")

for data, label, min_n in [
    (live_ticks, f"LIVE ({live_ticks['slug'].nunique()} mkts)", 3),
    (from15, f"FROM 15:00 ({from15['slug'].nunique()} mkts)", 10),
    (night, f"NIGHT ({night['slug'].nunique()} mkts)", 10),
    (all5, f"ALL ({all5['slug'].nunique()} mkts)", 10),
]:
    P(f"\n{'='*100}")
    P(f"  TOP CONFIGS -- {label} (N>={min_n})")
    P(f"{'='*100}")

    results = []
    for model in MODELS:
        for thresh in THRESHOLDS:
            for max_sp in SP_CAPS:
                for t_lo, t_hi in TIMING_WINDOWS:
                    r = compute(data, model, thresh, max_sp, t_lo, t_hi)
                    if r and r["n"] >= min_n:
                        results.append((r["pnl"], r["ev_t"], r["wr"], model, thresh, max_sp, t_lo, t_hi, r["n"], r["w"], r["l"], r["avg_sp"]))

    results.sort(key=lambda x: (-x[0], -x[2]))
    P(f"\n  TOP 20:")
    P(f"  {'Model':<10} {'Conf':>4} {'SP':>6} {'Time':>8} | {'N':>3} {'W':>3} {'L':>3} {'WR':>5} | {'EV/t':>6} {'PnL':>9} | {'AvgSP':>5}")
    P(f"  {'-'*75}")
    for pnl, ev_t, wr, m, thresh, max_sp, t_lo, t_hi, n, w, l, avg_sp in results[:20]:
        sp_label = "NOCAP" if max_sp >= 1.0 else f"<={max_sp:.2f}"
        t_str = f"{int(t_lo*100)}-{int(t_hi*100)}%"
        P(f"  {m:<10} {thresh:>3.0%} {sp_label:>6} {t_str:>8} | {n:>3} {w:>3} {l:>3} {wr:>4.0f}% | ${ev_t:>+5.2f} ${pnl:>+8.1f} | ${avg_sp:.3f}")

    if len(results) >= 10:
        P(f"\n  WORST 5:")
        P(f"  {'Model':<10} {'Conf':>4} {'SP':>6} {'Time':>8} | {'N':>3} {'W':>3} {'L':>3} {'WR':>5} | {'EV/t':>6} {'PnL':>9} | {'AvgSP':>5}")
        P(f"  {'-'*75}")
        for pnl, ev_t, wr, m, thresh, max_sp, t_lo, t_hi, n, w, l, avg_sp in results[-5:]:
            sp_label = "NOCAP" if max_sp >= 1.0 else f"<={max_sp:.2f}"
            t_str = f"{int(t_lo*100)}-{int(t_hi*100)}%"
            P(f"  {m:<10} {thresh:>3.0%} {sp_label:>6} {t_str:>8} | {n:>3} {w:>3} {l:>3} {wr:>4.0f}% | ${ev_t:>+5.2f} ${pnl:>+8.1f} | ${avg_sp:.3f}")


# ========================================================================
#  PART 6: HOURLY BREAKDOWN
# ========================================================================
P(f"\n\n{'#'*100}")
P(f"  PART 6: HOURLY BREAKDOWN")
P(f"{'#'*100}")

P(f"\n  A) Live trades actual results by hour:")
P(f"  {'Hour':>4} | {'N':>3} {'W':>3} {'L':>3} {'WR':>5} | {'PnL':>8}")
P(f"  {'-'*40}")
for h in sorted(live["_hour"].unique()):
    sub = live[live["_hour"] == h]
    w = sub["won"].sum()
    l = (~sub["won"]).sum()
    pnl = sub["pnl_usd"].sum()
    P(f"  {h:>4} | {len(sub):>3} {w:>3} {l:>3} {w/len(sub)*100:>4.0f}% | ${pnl:>+7.2f}")

P(f"\n  B) Analysis on ALL 5min by hour -- catboost 70% NO FILTER:")
r_all = compute(all5, "catboost", 0.70, 1.0, 0.0, 1.0)
if r_all:
    vdf = r_all["df"].copy()
    vdf["_h"] = (pd.to_datetime(vdf["_mkt_ts"].astype(int), unit="s").dt.hour + 3) % 24
    P(f"  {'Hour':>4} | {'N':>3} {'W':>3} {'L':>3} {'WR':>5} | {'PnL':>8} {'EV/t':>6}")
    P(f"  {'-'*48}")
    for h in sorted(vdf["_h"].unique()):
        sub = vdf[vdf["_h"] == h]
        n = len(sub)
        w = int(sub["_won"].sum())
        shares = BET / sub["_sp"].values
        pnl = float(np.where(sub["_won"].values, shares * (1 - sub["_sp"].values), -BET).sum())
        night_marker = " <-- NIGHT" if (h >= 23 or h < 6) else ""
        P(f"  {h:>4} | {n:>3} {w:>3} {n-w:>3} {w/n*100:>4.0f}% | ${pnl:>+7.1f} ${pnl/n:>+5.2f}{night_marker}")

P(f"\n  C) Analysis on ALL 5min by hour -- catboost 70% SP<=0.55:")
r_all2 = compute(all5, "catboost", 0.70, 0.55, 0.0, 1.0)
if r_all2:
    vdf = r_all2["df"].copy()
    vdf["_h"] = (pd.to_datetime(vdf["_mkt_ts"].astype(int), unit="s").dt.hour + 3) % 24
    P(f"  {'Hour':>4} | {'N':>3} {'W':>3} {'L':>3} {'WR':>5} | {'PnL':>8} {'EV/t':>6}")
    P(f"  {'-'*48}")
    for h in sorted(vdf["_h"].unique()):
        sub = vdf[vdf["_h"] == h]
        n = len(sub)
        w = int(sub["_won"].sum())
        shares = BET / sub["_sp"].values
        pnl = float(np.where(sub["_won"].values, shares * (1 - sub["_sp"].values), -BET).sum())
        night_marker = " <-- NIGHT" if (h >= 23 or h < 6) else ""
        P(f"  {h:>4} | {n:>3} {w:>3} {n-w:>3} {w/n*100:>4.0f}% | ${pnl:>+7.1f} ${pnl/n:>+5.2f}{night_marker}")


# ========================================================================
#  PART 7: ROOT CAUSE VERDICT
# ========================================================================
P(f"\n\n{'#'*100}")
P(f"  PART 7: ROOT CAUSE VERDICT")
P(f"{'#'*100}")

# ML accuracy on live markets -- no filters at all
P(f"\n  === A) ML ACCURACY on LIVE 13 markets (no filter, first tick) ===")
for model in MODELS:
    r = compute(live_ticks, model, 0.55, 1.0, 0.0, 1.0)
    if r:
        P(f"    {model:<10}: {r['n']}t, {r['w']}W/{r['l']}L, WR={r['wr']:.0f}%, PnL=${r['pnl']:+.1f}")

# Same on night
P(f"\n  === B) ML ACCURACY on NIGHT 84 markets (no filter, first tick) ===")
for model in MODELS:
    r = compute(night, model, 0.55, 1.0, 0.0, 1.0)
    if r:
        P(f"    {model:<10}: {r['n']}t, {r['w']}W/{r['l']}L, WR={r['wr']:.0f}%, PnL=${r['pnl']:+.1f}")

# Filter comparison on live
P(f"\n  === C) FILTER COMPARISON on LIVE 13 markets -- CATBOOST 70% ===")
filter_tests = [
    (1.00, 0.0, 1.0, "NO FILTER"),
    (0.55, 0.0, 1.0, "SP<=0.55 only"),
    (0.50, 0.0, 1.0, "SP<=0.50 only"),
    (1.00, 0.55, 0.92, "timing 55-92% only"),
    (0.55, 0.55, 0.92, "SP<=0.55 + 55-92% (LIVE)"),
    (1.00, 0.60, 1.0, "timing 60-100% only"),
    (0.55, 0.60, 1.0, "SP<=0.55 + 60-100%"),
    (1.00, 0.80, 1.0, "timing 80-100% only"),
    (0.55, 0.80, 1.0, "SP<=0.55 + 80-100%"),
    (0.50, 0.80, 1.0, "SP<=0.50 + 80-100%"),
]
P(f"    {'Filter':<30} | {'N':>3} {'W':>3} {'L':>3} {'WR':>5} | {'PnL':>8} {'EV/t':>6}")
P(f"    {'-'*65}")
for max_sp, t_lo, t_hi, label in filter_tests:
    r = compute(live_ticks, "catboost", 0.70, max_sp, t_lo, t_hi)
    if r:
        P(f"    {label:<30} | {r['n']:>3} {r['w']:>3} {r['l']:>3} {r['wr']:>4.0f}% | ${r['pnl']:>+7.1f} ${r['ev_t']:>+5.2f}")
    else:
        P(f"    {label:<30} |  -- --  --   -- |       --     --")

# Same on night
P(f"\n  === D) FILTER COMPARISON on NIGHT 84 markets -- CATBOOST 70% ===")
P(f"    {'Filter':<30} | {'N':>3} {'W':>3} {'L':>3} {'WR':>5} | {'PnL':>8} {'EV/t':>6}")
P(f"    {'-'*65}")
for max_sp, t_lo, t_hi, label in filter_tests:
    r = compute(night, "catboost", 0.70, max_sp, t_lo, t_hi)
    if r:
        P(f"    {label:<30} | {r['n']:>3} {r['w']:>3} {r['l']:>3} {r['wr']:>4.0f}% | ${r['pnl']:>+7.1f} ${r['ev_t']:>+5.2f}")
    else:
        P(f"    {label:<30} |  -- --  --   -- |       --     --")

P(f"\n  === E) FINAL ANSWER ===")
# compute key metrics
r_live_nofilt = compute(live_ticks, "catboost", 0.70, 1.0, 0.0, 1.0)
r_live_filt = compute(live_ticks, "catboost", 0.70, 0.55, 0.55, 0.92)
r_night_nofilt = compute(night, "catboost", 0.70, 1.0, 0.0, 1.0)
r_night_filt = compute(night, "catboost", 0.70, 0.55, 0.55, 0.92)
r_all_nofilt = compute(all5, "catboost", 0.70, 1.0, 0.0, 1.0)
r_all_filt = compute(all5, "catboost", 0.70, 0.55, 0.55, 0.92)

P(f"\n  catboost 70% -- NO FILTER vs SP<=0.55+55-92%:")
P(f"  {'Subset':<18} | {'NO FILTER':^25} | {'SP55 + 55-92%':^25} | Filter effect")
P(f"  {'-'*90}")
for label, rn, rf in [("Live 13 mkts", r_live_nofilt, r_live_filt),
                       ("Night 84 mkts", r_night_nofilt, r_night_filt),
                       ("All 144 mkts", r_all_nofilt, r_all_filt)]:
    if rn and rf:
        delta_wr = rf["wr"] - rn["wr"]
        P(f"  {label:<18} | {rn['n']:>3}t WR={rn['wr']:>4.0f}% ${rn['pnl']:>+7.0f} | {rf['n']:>3}t WR={rf['wr']:>4.0f}% ${rf['pnl']:>+7.0f} | WR {delta_wr:>+5.1f}pp")

P(f"""
  CONCLUSION:
  -----------
  1. ML on live 13 markets (NO filter): {r_live_nofilt['wr']:.0f}% WR = MODEL ITSELF is {'OK' if r_live_nofilt['wr'] >= 55 else 'WEAK'} on these markets
  2. ML on night 84 markets (NO filter): {r_night_nofilt['wr']:.0f}% WR = MODEL is {'STRONG' if r_night_nofilt['wr'] >= 70 else 'OK'}
  3. Filter (SP<=0.55 + 55-92%) on live: {r_live_filt['wr']:.0f}% WR = filter {'HURTS' if r_live_filt['wr'] < r_live_nofilt['wr'] else 'HELPS'} ({r_live_filt['wr']-r_live_nofilt['wr']:+.0f}pp)
  4. Filter on night 84 mkts: {r_night_filt['wr']:.0f}% WR = filter {'HELPS' if r_night_filt['wr'] > r_night_nofilt['wr'] else 'NEUTRAL'} ({r_night_filt['wr']-r_night_nofilt['wr']:+.0f}pp)
  5. Live trades = {len(live_5m)} 5min trades out of {all5['slug'].nunique()} available markets = {len(live_5m)/all5['slug'].nunique()*100:.0f}% coverage
  6. Night live 5min = {len(live_night_5m)} trades = too few for statistics
  7. GTC (limit to market end) IS how analysis works. No difference.
""")

elapsed = time.time() - t0
P(f"{'='*100}")
P(f"  DONE in {elapsed:.1f}s")
P(f"{'='*100}")
