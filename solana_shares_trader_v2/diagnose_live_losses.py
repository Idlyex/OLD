"""
DIAGNOSE LIVE LOSSES — Why did live trades lose while analysis says 85% WR?

Cross-references actual live trades with tick data to find the root cause.
Uses same vectorized logic as analyze_live.py (compute_all_ticks).
Focuses on overnight session (22:00+ May 9 → May 10).
"""
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import time

t0 = time.time()

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
results_dir = Path("results")

# Load live trades
trades = pd.read_json("results/ml_live_trades.json")
trades["won"] = trades["won"].astype(bool)
live = trades[~trades["dry_run"]].copy()

# Load ALL tick data (May 9 + May 10)
ticks_list = []
for f in sorted(results_dir.glob("live_ticks_*.jsonl")):
    tdf = pd.read_json(f, lines=True)
    ticks_list.append(tdf)
    print(f"  Loaded {f.name}: {len(tdf)} ticks, {tdf['slug'].nunique()} markets")
ticks = pd.concat(ticks_list, ignore_index=True)

# Load ALL outcomes
out_list = []
for f in sorted(results_dir.glob("live_outcomes_*.jsonl")):
    odf = pd.read_json(f, lines=True)
    out_list.append(odf)
outcomes = pd.concat(out_list, ignore_index=True).drop_duplicates("slug", keep="last")

# Merge
ticks = ticks.merge(outcomes[["slug", "outcome"]], on="slug", how="inner")
print(f"  Total: {len(ticks)} ticks, {ticks['slug'].nunique()} markets with outcomes")

# Parse market timing for live trades
for idx, r in live.iterrows():
    slug_parts = r["slug"].split("-")
    market_start_ts = int(slug_parts[-1])
    dur_min = r["duration_min"]
    market_end_ts = market_start_ts + dur_min * 60
    elapsed = r["entry_ts"] - market_start_ts
    live.loc[idx, "entry_pct"] = elapsed / (dur_min * 60)
    live.loc[idx, "market_start_ts"] = market_start_ts

BET = 2.85
MODELS = ["catboost", "rf", "xgboost", "ensemble", "lgbm"]

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: LIVE TRADE-BY-TRADE FORENSICS
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*100}")
print(f"  SECTION 1: FORENSICS — Each live trade vs tick data predictions")
print(f"  What did ALL models say at the EXACT moment of entry + at peak confidence?")
print(f"{'='*100}\n")

for idx, trade in live.iterrows():
    slug = trade["slug"]
    # Get all ticks for this market
    mt = ticks[ticks["slug"] == slug].copy()
    if mt.empty:
        print(f"  [{slug[-25:]}] NO TICK DATA\n")
        continue

    outcome = mt["outcome"].iloc[0]
    wl = "WIN" if trade["won"] else "LOSS"
    pnl = trade["pnl_usd"]

    print(f"  [{slug[-25:]}] {str(trade['entry_time'])[:19]}")
    print(f"    Live: dir={trade['direction']} conf={trade['confidence']:.3f} price=${trade['entry_price']:.4f} → {wl} ${pnl:+.2f}")
    print(f"    Outcome: {outcome} | Entry pct: {trade['entry_pct']:.1%} | Ticks: {len(mt)}")

    # What did each model say at the tick closest to entry time?
    entry_ts = trade["entry_ts"]
    mt["_ts_diff"] = (mt["ts"] - entry_ts).abs()
    entry_tick = mt.loc[mt["_ts_diff"].idxmin()]

    print(f"    Models @ entry tick (entry_pct={entry_tick['entry_pct']:.1%}):")
    for model in MODELS:
        if model not in mt.columns:
            continue
        prob = entry_tick[model]
        conf = max(prob, 1 - prob)
        direction = "UP" if prob > 0.5 else "DOWN"
        correct = direction == outcome
        print(f"      {model:<10}: prob={prob:.4f} conf={conf:.4f} dir={direction} {'✓' if correct else '✗ WRONG'}")

    # What did catboost say across the market lifecycle?
    if "catboost" in mt.columns:
        cb = mt["catboost"].values
        cb_conf = np.maximum(cb, 1 - cb)
        cb_dir = np.where(cb > 0.5, "UP", "DOWN")
        cb_correct = cb_dir == outcome

        # How many ticks predicted correctly vs wrong?
        n_correct = cb_correct.sum()
        n_wrong = len(cb_correct) - n_correct
        pct_correct = n_correct / len(cb_correct) * 100

        # SP at entry tick
        if trade["direction"] == "UP":
            sp_entry = entry_tick.get("yes", entry_tick.get("yes_ask", 0.5))
        else:
            sp_entry = entry_tick.get("no", entry_tick.get("no_ask", 0.5))

        print(f"    Catboost across all {len(mt)} ticks: {n_correct} correct ({pct_correct:.0f}%), {n_wrong} wrong")
        print(f"    Peak CB conf: {cb_conf.max():.4f} at entry_pct={mt['entry_pct'].iloc[cb_conf.argmax()]:.1%}")
        
        # Did direction FLIP during market?
        dirs_unique = np.unique(cb_dir)
        if len(dirs_unique) > 1:
            flip_pcts = []
            for i in range(1, len(cb_dir)):
                if cb_dir[i] != cb_dir[i-1]:
                    flip_pcts.append(mt["entry_pct"].iloc[i])
            print(f"    ⚠ DIRECTION FLIP at entry_pct: {[f'{p:.1%}' for p in flip_pcts]}")
        else:
            print(f"    Direction stable: always {dirs_unique[0]}")
    print()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: OVERNIGHT TRADES (22:00+) — the losing session
# ═══════════════════════════════════════════════════════════════════════════════
print(f"{'='*100}")
print(f"  SECTION 2: OVERNIGHT SESSION (22:00+ May 9)")
print(f"{'='*100}")

overnight = live[pd.to_datetime(live["entry_time"]).dt.hour >= 22].copy()
# Also include 00:xx trades
midnight = live[pd.to_datetime(live["entry_time"]).dt.hour < 6].copy()
overnight = pd.concat([overnight, midnight]).drop_duplicates(subset=["slug"])

if len(overnight) > 0:
    w = overnight["won"].sum()
    l = len(overnight) - w
    pnl = overnight["pnl_usd"].sum()
    print(f"  Overnight: {len(overnight)} trades, {w}W/{l}L, WR={w/len(overnight)*100:.0f}%, PnL=${pnl:+.2f}")
    print(f"  Losses:")
    for _, r in overnight[~overnight["won"]].iterrows():
        print(f"    {str(r['entry_time'])[:19]} {r['slug'][-22:]} {r['direction']} conf={r['confidence']:.3f} ${r['pnl_usd']:+.2f}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: ANALYSIS WR for the EXACT same markets as live trades
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*100}")
print(f"  SECTION 3: Analysis WR on the EXACT same {len(live)} markets (not all markets)")
print(f"  If GTC until market end → what would have happened?")
print(f"{'='*100}")

live_slugs = set(live["slug"].unique())
live_ticks = ticks[ticks["slug"].isin(live_slugs)].copy()
print(f"  Ticks for live-traded markets: {len(live_ticks)} ticks, {live_ticks['slug'].nunique()} markets")

# For each config, compute analysis result on ONLY the markets that were live-traded
configs = [
    ("catboost", 0.70, 0.55, 0.55, 0.92),
    ("catboost", 0.70, 0.55, 0.60, 1.00),
    ("catboost", 0.70, 0.55, 0.70, 1.00),
    ("catboost", 0.80, 0.55, 0.55, 0.92),
    ("catboost", 0.85, 0.55, 0.55, 0.92),
    ("catboost", 0.90, 0.55, 0.55, 0.92),
    ("catboost", 0.95, 0.55, 0.55, 0.92),
    ("rf",       0.70, 0.55, 0.55, 0.92),
    ("rf",       0.85, 0.55, 0.55, 0.92),
    ("rf",       0.95, 0.55, 0.55, 0.92),
    ("xgboost",  0.70, 0.55, 0.55, 0.92),
    ("xgboost",  0.85, 0.55, 0.55, 0.92),
    ("ensemble", 0.70, 0.55, 0.55, 0.92),
    ("ensemble", 0.85, 0.55, 0.55, 0.92),
    ("ensemble", 0.70, 0.55, 0.70, 1.00),
    ("ensemble", 0.85, 0.55, 0.70, 1.00),
]

print(f"\n  {'Model':<10} {'Conf':>4} {'SP<=':>5} {'Timing':>8} | {'N':>3} {'W':>3} {'L':>3} {'WR':>5} | {'PnL':>8} {'EV/t':>6}")
print(f"  {'-'*70}")

for model, thresh, max_sp, t_lo, t_hi in configs:
    if model not in live_ticks.columns:
        continue
    prob = live_ticks[model].values
    conf = np.maximum(prob, 1 - prob)
    model_up = prob > 0.5
    sp = np.where(model_up, live_ticks["yes"].values, live_ticks["no"].values)
    entry_pct = live_ticks["entry_pct"].values

    mask = (conf >= thresh) & (sp <= max_sp) & (sp >= 0.10) & (entry_pct >= t_lo) & (entry_pct < t_hi)
    valid = live_ticks[mask].copy()
    if valid.empty:
        continue
    valid["_dir"] = np.where(model_up[mask], "UP", "DOWN")
    valid["_sp"] = sp[mask]
    valid = valid.sort_values("entry_pct").drop_duplicates("slug", keep="first")
    valid["_won"] = valid["_dir"] == valid["outcome"]

    n = len(valid)
    w = int(valid["_won"].sum())
    l = n - w
    wr = w / n * 100
    sp_arr = valid["_sp"].values
    shares = BET / sp_arr
    pnl_arr = np.where(valid["_won"].values, shares * (1 - sp_arr), -BET)
    pnl = pnl_arr.sum()
    ev_t = pnl / n

    t_str = f"{int(t_lo*100)}-{int(t_hi*100)}%"
    print(f"  {model:<10} {thresh:>3.0%} ${max_sp:.2f} {t_str:>8} | {n:>3} {w:>3} {l:>3} {wr:>4.0f}% | ${pnl:>+7.1f} ${ev_t:>+5.2f}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: PER-TRADE — Would analysis have taken the same trade?
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*100}")
print(f"  SECTION 4: Per-trade comparison — Live direction vs Analysis direction")
print(f"  Live config: catboost 70%, SP<=0.55, timing 55-92%")
print(f"{'='*100}\n")

print(f"  {'#':>2} {'Slug':>22} {'Live_dir':>8} {'Live_conf':>9} {'Outcome':>7} {'Live':>5} |"
      f" {'CB_dir':>6} {'CB_sp':>6} {'CB_ePct':>7} {'Anls':>5} | {'Match':>5}")
print(f"  {'-'*100}")

model = "catboost"
for i, (idx, trade) in enumerate(live.iterrows()):
    slug = trade["slug"]
    mt = live_ticks[live_ticks["slug"] == slug].copy()
    if mt.empty:
        print(f"  {i+1:>2} {slug[-22:]:>22} NO TICK DATA")
        continue

    outcome = mt["outcome"].iloc[0]

    # Apply analysis filters: catboost, conf>=70%, sp<=0.55, entry_pct 55-92%
    prob = mt[model].values
    conf = np.maximum(prob, 1 - prob)
    model_up = prob > 0.5
    sp = np.where(model_up, mt["yes"].values, mt["no"].values)
    ep = mt["entry_pct"].values

    mask = (conf >= 0.70) & (sp <= 0.55) & (sp >= 0.10) & (ep >= 0.55) & (ep < 0.92)
    valid = mt[mask]

    if valid.empty:
        anls_dir = "—"
        anls_sp = "—"
        anls_ep = "—"
        anls_won = "SKIP"
        match = "—"
    else:
        # First qualifying tick
        first = valid.sort_values("entry_pct").iloc[0]
        first_prob = first[model]
        anls_dir = "UP" if first_prob > 0.5 else "DOWN"
        anls_sp_val = first["yes"] if first_prob > 0.5 else first["no"]
        anls_ep = f"{first['entry_pct']:.0%}"
        anls_sp = f"${anls_sp_val:.2f}"
        anls_won = "W" if anls_dir == outcome else "L"
        match = "SAME" if anls_dir == trade["direction"] else "DIFF!"

    live_wl = "W" if trade["won"] else "L"
    print(f"  {i+1:>2} {slug[-22:]:>22} {trade['direction']:>8} {trade['confidence']:>9.3f} {outcome:>7} {live_wl:>5} |"
          f" {anls_dir:>6} {anls_sp:>6} {anls_ep:>7} {anls_won:>5} | {match:>5}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: WHAT IF — Alternative filters on ALL overnight markets
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*100}")
print(f"  SECTION 5: WHAT-IF on overnight markets (22:00+ slugs)")
print(f"  Testing different configs on ONLY the markets that existed overnight")
print(f"{'='*100}")

# Get all slugs that were active during 22:00-06:00
# Parse ts range from ticks
overnight_slugs = set()
for slug in ticks["slug"].unique():
    slug_parts = slug.split("-")
    try:
        market_start = int(slug_parts[-1])
    except:
        continue
    # Convert to hour (UTC)
    from datetime import timezone
    dt = datetime.fromtimestamp(market_start, tz=timezone.utc)
    # We want markets starting >= 19:00 UTC (22:00 UTC+3) or < 03:00 UTC (06:00 UTC+3)
    if dt.hour >= 19 or dt.hour < 3:
        overnight_slugs.add(slug)

overnight_ticks = ticks[ticks["slug"].isin(overnight_slugs)].copy()
print(f"  Overnight markets: {len(overnight_slugs)} | Ticks: {len(overnight_ticks)}")

# Only 5min
ov5 = overnight_ticks[overnight_ticks["slug"].str.contains("-5m-")].copy()
print(f"  Overnight 5min: {ov5['slug'].nunique()} markets, {len(ov5)} ticks")

all_configs = []
models_test = ["catboost", "rf", "xgboost", "ensemble"]
thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
sp_caps = [0.50, 0.55]
timing_windows = [(0.55, 0.92), (0.60, 1.0), (0.70, 1.0), (0.0, 1.0)]

for m in models_test:
    if m not in ov5.columns:
        continue
    prob = ov5[m].values
    conf_arr = np.maximum(prob, 1 - prob)
    model_up_arr = prob > 0.5
    sp_arr = np.where(model_up_arr, ov5["yes"].values, ov5["no"].values)
    ep_arr = ov5["entry_pct"].values
    outcome_arr = ov5["outcome"].values
    dir_arr = np.where(model_up_arr, "UP", "DOWN")

    for thresh in thresholds:
        for max_sp in sp_caps:
            for t_lo, t_hi in timing_windows:
                mask = (conf_arr >= thresh) & (sp_arr <= max_sp) & (sp_arr >= 0.10) & (ep_arr >= t_lo) & (ep_arr < t_hi)
                valid = ov5[mask].copy()
                if valid.empty:
                    continue
                valid["_dir"] = dir_arr[mask]
                valid["_sp"] = sp_arr[mask]
                valid = valid.sort_values("entry_pct").drop_duplicates("slug", keep="first")
                valid["_won"] = valid["_dir"] == valid["outcome"]

                n = len(valid)
                if n < 3:
                    continue
                w = int(valid["_won"].sum())
                l = n - w
                wr = w / n * 100
                shares = BET / valid["_sp"].values
                pnl = np.where(valid["_won"].values, shares * (1 - valid["_sp"].values), -BET).sum()
                ev_t = pnl / n
                all_configs.append((pnl, ev_t, wr, m, thresh, max_sp, t_lo, t_hi, n, w, l))

all_configs.sort(key=lambda x: (-x[0], -x[2]))

print(f"\n  TOP 20 overnight configs (by PnL):")
print(f"  {'Model':<10} {'Conf':>4} {'SP<=':>5} {'Time':>8} | {'N':>3} {'W':>3} {'L':>3} {'WR':>5} | {'PnL':>8} {'EV/t':>6}")
print(f"  {'-'*65}")
for pnl, ev_t, wr, m, thresh, max_sp, t_lo, t_hi, n, w, l in all_configs[:20]:
    t_str = f"{int(t_lo*100)}-{int(t_hi*100)}%"
    print(f"  {m:<10} {thresh:>3.0%} ${max_sp:.2f} {t_str:>8} | {n:>3} {w:>3} {l:>3} {wr:>4.0f}% | ${pnl:>+7.1f} ${ev_t:>+5.2f}")

# WORST configs
print(f"\n  WORST 10 overnight configs (by PnL):")
print(f"  {'Model':<10} {'Conf':>4} {'SP<=':>5} {'Time':>8} | {'N':>3} {'W':>3} {'L':>3} {'WR':>5} | {'PnL':>8} {'EV/t':>6}")
print(f"  {'-'*65}")
for pnl, ev_t, wr, m, thresh, max_sp, t_lo, t_hi, n, w, l in all_configs[-10:]:
    t_str = f"{int(t_lo*100)}-{int(t_hi*100)}%"
    print(f"  {m:<10} {thresh:>3.0%} ${max_sp:.2f} {t_str:>8} | {n:>3} {w:>3} {l:>3} {wr:>4.0f}% | ${pnl:>+7.1f} ${ev_t:>+5.2f}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: CATBOOST DIRECTION ACCURACY OVER TIME
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*100}")
print(f"  SECTION 6: Model accuracy by hour (ALL markets, not just live-traded)")
print(f"{'='*100}")

ticks5 = ticks[ticks["slug"].str.contains("-5m-")].copy()
# Parse hour from slug timestamp
ticks5["_market_ts"] = ticks5["slug"].str.split("-").str[-1].astype(int)
ticks5["_hour_utc"] = pd.to_datetime(ticks5["_market_ts"], unit="s").dt.hour
ticks5["_hour_local"] = (ticks5["_hour_utc"] + 3) % 24  # UTC+3

for model in ["catboost"]:
    prob = ticks5[model].values
    conf = np.maximum(prob, 1 - prob)
    model_dir = np.where(prob > 0.5, "UP", "DOWN")
    sp = np.where(prob > 0.5, ticks5["yes"].values, ticks5["no"].values)

    # Filter: conf>=70%, sp<=0.55
    mask = (conf >= 0.70) & (sp <= 0.55) & (sp >= 0.10)
    valid = ticks5[mask].copy()
    valid["_dir"] = model_dir[mask]
    valid["_sp"] = sp[mask]
    valid = valid.sort_values("entry_pct").drop_duplicates("slug", keep="first")
    valid["_won"] = valid["_dir"] == valid["outcome"]
    valid["_hour"] = (pd.to_datetime(valid["_market_ts"].astype(int), unit="s").dt.hour + 3) % 24

    print(f"\n  {model.upper()} conf>=70% SP<=0.55 — WR by hour (UTC+3):")
    print(f"  {'Hour':>4} | {'N':>3} {'W':>3} {'L':>3} {'WR':>5} | {'PnL':>8}")
    print(f"  {'-'*40}")
    for h in sorted(valid["_hour"].unique()):
        sub = valid[valid["_hour"] == h]
        n = len(sub)
        w = int(sub["_won"].sum())
        l = n - w
        wr = w / n * 100 if n > 0 else 0
        shares = BET / sub["_sp"].values
        pnl = np.where(sub["_won"].values, shares * (1 - sub["_sp"].values), -BET).sum()
        marker = " ← NIGHT" if (h >= 22 or h < 6) else ""
        print(f"  {h:>4} | {n:>3} {w:>3} {l:>3} {wr:>4.0f}% | ${pnl:>+7.1f}{marker}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: ROOT CAUSE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*100}")
print(f"  SECTION 7: ROOT CAUSE SUMMARY")
print(f"{'='*100}")

# Check: did live enter DIFFERENT direction than analysis?
mismatches = 0
analysis_would_skip = 0
for _, trade in live.iterrows():
    slug = trade["slug"]
    mt = live_ticks[live_ticks["slug"] == slug]
    if mt.empty:
        continue
    prob = mt["catboost"].values
    conf = np.maximum(prob, 1 - prob)
    model_up = prob > 0.5
    sp = np.where(model_up, mt["yes"].values, mt["no"].values)
    ep = mt["entry_pct"].values
    mask = (conf >= 0.70) & (sp <= 0.55) & (sp >= 0.10) & (ep >= 0.55) & (ep < 0.92)
    valid = mt[mask]
    if valid.empty:
        analysis_would_skip += 1
        continue
    first = valid.sort_values("entry_pct").iloc[0]
    anls_dir = "UP" if first["catboost"] > 0.5 else "DOWN"
    if anls_dir != trade["direction"]:
        mismatches += 1

print(f"""
  Key findings:
  1. Live trades: {len(live)} | Won: {live['won'].sum()} | Lost: {(~live['won']).sum()} | WR: {live['won'].mean()*100:.0f}%
  2. Analysis would skip: {analysis_would_skip} trades (filters don't match)
  3. Direction mismatches: {mismatches} (live picked different direction than analysis)
  4. DOWN bias: {len(live[live['direction']=='DOWN'])}/{len(live)} trades are DOWN
     - DOWN WR: {live[live['direction']=='DOWN']['won'].mean()*100:.0f}%
     - UP WR: {live[live['direction']=='UP']['won'].mean()*100:.0f}%
""")

elapsed = time.time() - t0
print(f"  Done in {elapsed:.2f}s")
