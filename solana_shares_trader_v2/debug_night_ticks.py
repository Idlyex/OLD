"""Show all model predictions at each tick for night live-traded markets."""
import json, pandas as pd, numpy as np
from pathlib import Path

# Load ticks
frames = []
for f in sorted(Path("results").glob("live_ticks_*.jsonl")):
    frames.append(pd.read_json(f, lines=True))
ticks = pd.concat(frames, ignore_index=True)

# Load outcomes
oframes = []
for f in sorted(Path("results").glob("live_outcomes_*.jsonl")):
    oframes.append(pd.read_json(f, lines=True))
outcomes = pd.concat(oframes, ignore_index=True).drop_duplicates("slug", keep="last")
ticks = ticks.merge(outcomes[["slug","outcome"]], on="slug", how="left")

# Load live trades
trades = json.load(open("results/ml_live_trades.json"))
live = pd.DataFrame([t for t in trades if not t["dry_run"]])
live["_hour"] = (pd.to_datetime(live["entry_ts"], unit="s").dt.hour + 3) % 24
night = live[(live["_hour"] >= 23) | (live["_hour"] < 6)]

models = ["catboost", "rf", "xgboost", "lgbm", "ensemble"]

for _, t in night.sort_values("entry_ts").iterrows():
    slug = t["slug"]
    mkt = ticks[ticks["slug"] == slug].sort_values("entry_pct")
    if mkt.empty:
        continue

    outcome = mkt.iloc[0].get("outcome", "?") if "outcome" in mkt.columns else "?"

    # Find closest tick to live entry
    diffs = np.abs(mkt["ts"].values - t["entry_ts"])
    ci = diffs.argmin()
    ct = mkt.iloc[ci]

    print(f"\n{'='*120}")
    print(f"  {slug}  outcome={outcome}")
    print(f"  LIVE: dir={t['direction']} conf={t['confidence']:.0%} sp=${t['entry_price']:.3f} {'WIN' if t['won'] else 'LOSS'}")
    print(f"  Entry time: {t['entry_time']}  entry_pct~{ct['entry_pct']:.0%}  ts_diff={diffs[ci]:.1f}s")

    print(f"\n  Models at closest recorded tick to live entry (entry_pct={ct['entry_pct']:.0%}):")
    for m in models:
        if m in ct.index:
            p = ct[m]
            d = "UP" if p > 0.5 else "DOWN"
            c = max(p, 1-p)
            marker = " ← PRIMARY" if m == "catboost" else ""
            print(f"    {m:>10}: prob_up={p:.4f} → {d:>5} {c:.0%}{marker}")

    # ALL ticks in entry window
    window = mkt[(mkt["entry_pct"] >= 0.50) & (mkt["entry_pct"] <= 0.95)]
    print(f"\n  All ticks in 50-95% window ({len(window)} ticks):")
    header = f"  {'%':>5}  {'catboost':>10} {'rf':>10} {'xgboost':>10} {'lgbm':>10} {'ensemble':>10}  {'yes$':>6} {'no$':>6}"
    print(header)
    print(f"  {'-'*85}")

    for _, row in window.iterrows():
        ep = f"{row['entry_pct']:.0%}"
        parts = []
        for m in models:
            if m in row.index and pd.notna(row[m]):
                p = row[m]
                d = "U" if p > 0.5 else "D"
                c = max(p, 1 - p)
                parts.append(f"{d}{c:.0%}".rjust(10))
            else:
                parts.append("?".rjust(10))
        yes_p = f"${row['yes']:.3f}"
        no_p = f"${row['no']:.3f}"

        # Mark the tick closest to live entry
        is_live = (abs(row["ts"] - t["entry_ts"]) < 3)
        marker = " ◄◄ LIVE ENTRY" if is_live else ""
        print(f"  {ep:>5}  {' '.join(parts)}  {yes_p:>6} {no_p:>6}{marker}")

    # Summary: how stable is catboost direction in this window?
    if "catboost" in window.columns:
        cb_ups = (window["catboost"] > 0.5).sum()
        cb_dns = (window["catboost"] <= 0.5).sum()
        print(f"\n  CB stability in window: UP={cb_ups} DOWN={cb_dns} flips={'YES' if cb_ups > 0 and cb_dns > 0 else 'NO'}")
        cb_conf = window["catboost"].apply(lambda x: max(x, 1-x))
        print(f"  CB conf range: {cb_conf.min():.0%} - {cb_conf.max():.0%}")
