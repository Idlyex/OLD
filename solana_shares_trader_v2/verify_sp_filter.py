#!/usr/bin/env python3
"""Verify: threshold + share price filter (0.55 and 0.45) → hold to end."""
import numpy as np
import pandas as pd
from pathlib import Path

results_dir = Path("results")
tick_file = sorted(results_dir.glob("live_ticks_*.jsonl"))[-1]
df = pd.read_json(tick_file, lines=True)
out_file = results_dir / tick_file.name.replace("live_ticks_", "live_outcomes_")
outcomes = pd.read_json(out_file, lines=True).drop_duplicates("slug", keep="last")
df = df.merge(outcomes[["slug", "outcome"]], on="slug", how="inner")
df = df[df["slug"].str.contains("-5m-")].copy()
n_markets = df.slug.nunique()
print(f"Data: {len(df)} ticks, {n_markets} markets (5min)")

models = ["catboost", "rf", "xgboost"]

for max_sp in [0.55, 0.45]:
    for model in models:
        print(f"\n{'='*75}")
        print(f"  {model.upper()} 5min | SP <= ${max_sp:.2f} | first tick conf>=X AND price<={max_sp}")
        print(f"  Logic: place GTC limit, hold until market resolves")
        print(f"{'='*75}")
        print(f"  {'Conf':<6} {'N':>4} {'W':>4} {'L':>4} {'WR':>7} {'EV/sh':>8} {'PnL':>8} {'AvgSP':>6}")
        print(f"  {'-'*58}")

        for thresh in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
            prob = df[model].values
            conf = np.maximum(prob, 1 - prob)
            model_up = prob > 0.5
            sp = np.where(model_up, df["yes"].values, df["no"].values)

            mask = (conf >= thresh) & (sp <= max_sp) & (sp >= 0.01)
            valid = df[mask].copy()
            if valid.empty:
                print(f"  {thresh:.0%}     —")
                continue

            valid["_dir"] = np.where(model_up[mask], "UP", "DOWN")
            valid["_sp"] = sp[mask]
            valid = valid.sort_values("entry_pct").drop_duplicates("slug", keep="first")
            valid["won"] = valid["_dir"] == valid["outcome"]

            n = len(valid)
            w = int(valid["won"].sum())
            l = n - w
            wr = w / n * 100

            bet = 2.0
            sp_arr = valid["_sp"].values
            shares = bet / sp_arr
            pnl_arr = np.where(valid["won"].values, shares * (1 - sp_arr), -bet)
            pnl = pnl_arr.sum()
            avg_sp = sp_arr.mean()
            ev = (wr / 100) * (1 - avg_sp) - (1 - wr / 100) * avg_sp

            print(f"  {thresh:.0%}   {n:>4} {w:>4} {l:>4} {wr:>6.1f}% ${ev:>+.3f} ${pnl:>+6.1f} ${avg_sp:.3f}")
