#!/usr/bin/env python3
"""
VERIFY: Pure confidence threshold → win rate.

Logic:
  - For each market, check if ANY tick had model confidence >= threshold
  - If yes → "trade entered" (direction = model's call on that FIRST qualifying tick)
  - Win = direction matches actual market outcome
  - Entry price irrelevant for WR — assume $0.50 for PnL calc (binary market fair price)
  - This proves: "if I enter whenever conf crosses threshold, does WR match analysis?"

No price filters. No timing filters. Just: first tick with conf >= X → enter → hold to resolution.
"""
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


def load_data():
    results_dir = Path("results")
    
    # Load ticks
    tick_file = results_dir / f"live_ticks_{datetime.now().strftime('%Y-%m-%d')}.jsonl"
    if not tick_file.exists():
        # Try all files
        files = sorted(results_dir.glob("live_ticks_*.jsonl"))
        if not files:
            print("No tick data found"); sys.exit(1)
        tick_file = files[-1]  # latest
    
    df = pd.read_json(tick_file, lines=True)
    print(f"Loaded: {tick_file.name} → {len(df)} ticks, {df.slug.nunique()} markets")
    
    # Load outcomes
    date_str = tick_file.stem.replace("live_ticks_", "")
    out_file = results_dir / f"live_outcomes_{date_str}.jsonl"
    if not out_file.exists():
        print(f"No outcomes file: {out_file}"); sys.exit(1)
    
    outcomes = pd.read_json(out_file, lines=True).drop_duplicates("slug", keep="last")
    print(f"Outcomes: {len(outcomes)} markets")
    
    # Merge
    merged = df.merge(outcomes[["slug", "outcome"]], on="slug", how="inner")
    print(f"Merged: {len(merged)} ticks, {merged.slug.nunique()} markets with outcomes")
    return merged


def analyze(df, model, duration=None):
    """Pure threshold analysis — no price filter, no timing filter."""
    
    if duration:
        # Filter by duration (5min or 15min based on slug)
        if duration == 5:
            df = df[df["slug"].str.contains("-5m-")].copy()
        elif duration == 15:
            df = df[df["slug"].str.contains("-15m-")].copy()
    
    n_markets = df.slug.nunique()
    if model not in df.columns:
        print(f"  Model '{model}' not in data"); return
    
    print(f"\n{'='*80}")
    print(f"  PURE THRESHOLD VERIFICATION — {model.upper()} {'('+str(duration)+'min)' if duration else '(all)'}")
    print(f"  {n_markets} markets total | NO price filter | NO timing filter")
    print(f"  Entry = first tick with conf >= threshold → hold to resolution")
    print(f"  WR = direction matches outcome | PnL assumes $0.50/share entry")
    print(f"{'='*80}")
    print(f"  {'Conf≥':<7} {'N':>4} {'W':>4} {'L':>4} {'WR':>7} {'EV/sh':>9} {'PnL($2)':>9}")
    print(f"  {'-'*55}")
    
    for thresh in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        prob = df[model].values
        conf = np.maximum(prob, 1 - prob)
        model_up = prob > 0.5
        
        # Filter: ONLY by confidence
        mask = conf >= thresh
        valid = df[mask].copy()
        if valid.empty:
            print(f"  {thresh:.0%}    {'—':>4}"); continue
        
        # Direction from model
        valid["_dir"] = np.where(model_up[mask], "UP", "DOWN")
        
        # First qualifying tick per market (earliest entry_pct)
        if "entry_pct" in valid.columns:
            valid = valid.sort_values("entry_pct")
        valid = valid.drop_duplicates("slug", keep="first")
        
        # Win/loss
        valid["won"] = valid["_dir"] == valid["outcome"]
        n = len(valid)
        w = valid["won"].sum()
        l = n - w
        wr = w / n * 100 if n > 0 else 0
        
        # PnL at assumed $0.50 entry (fair binary price)
        # Win → +$0.50/share, Loss → -$0.50/share
        # With $2 bet → 4 shares → Win=$2, Loss=-$2
        bet = 2.0
        assumed_price = 0.50
        shares = bet / assumed_price
        pnl = w * (shares * (1 - assumed_price)) - l * bet  # win: shares*(1-SP), loss: -bet
        ev_per_share = (wr/100) * (1 - assumed_price) - (1 - wr/100) * assumed_price
        
        print(f"  {thresh:.0%}    {n:>4} {w:>4} {l:>4} {wr:>6.1f}% ${ev_per_share:>+7.3f} ${pnl:>+7.1f}")
    
    print()


def analyze_with_real_prices(df, model, duration=None):
    """Same but using actual ASK prices from ticks for more accurate PnL."""
    
    if duration:
        if duration == 5:
            df = df[df["slug"].str.contains("-5m-")].copy()
        elif duration == 15:
            df = df[df["slug"].str.contains("-15m-")].copy()
    
    n_markets = df.slug.nunique()
    if model not in df.columns:
        return
    
    print(f"\n{'='*80}")
    print(f"  SAME BUT WITH REAL ASK PRICES — {model.upper()} {'('+str(duration)+'min)' if duration else ''}")
    print(f"  {n_markets} markets | Entry price = ASK from first qualifying tick")
    print(f"{'='*80}")
    print(f"  {'Conf≥':<7} {'N':>4} {'W':>4} {'L':>4} {'WR':>7} {'EV/sh':>9} {'PnL($2)':>9} {'AvgSP':>7}")
    print(f"  {'-'*65}")
    
    for thresh in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        prob = df[model].values
        conf = np.maximum(prob, 1 - prob)
        model_up = prob > 0.5
        
        mask = conf >= thresh
        valid = df[mask].copy()
        if valid.empty:
            continue
        
        valid["_dir"] = np.where(model_up[mask], "UP", "DOWN")
        
        # Share price = actual ASK for chosen direction
        if "yes" in valid.columns and "no" in valid.columns:
            valid["_sp"] = np.where(valid["_dir"] == "UP", valid["yes"], valid["no"])
        else:
            valid["_sp"] = 0.50
        
        if "entry_pct" in valid.columns:
            valid = valid.sort_values("entry_pct")
        valid = valid.drop_duplicates("slug", keep="first")
        
        valid["won"] = valid["_dir"] == valid["outcome"]
        n = len(valid)
        w = valid["won"].sum()
        l = n - w
        wr = w / n * 100 if n > 0 else 0
        
        bet = 2.0
        sp = valid["_sp"].values
        shares = bet / sp
        pnl_arr = np.where(valid["won"].values, shares * (1 - sp), -bet)
        pnl = pnl_arr.sum()
        avg_sp = sp.mean()
        ev_per_share = (wr/100) * (1 - avg_sp) - (1 - wr/100) * avg_sp
        
        print(f"  {thresh:.0%}    {n:>4} {w:>4} {l:>4} {wr:>6.1f}% ${ev_per_share:>+7.3f} ${pnl:>+7.1f}  ${avg_sp:.3f}")
    
    print()


if __name__ == "__main__":
    df = load_data()
    
    print("\n" + "█"*80)
    print("  VERIFICATION: Does WR depend on entry price? NO — only on direction call")
    print("  If first tick gives conf >= X → enter at ANY price → WR is the same")
    print("█"*80)
    
    # Pure threshold (no price, $0.50 assumed)
    for model in ["catboost", "rf", "xgboost"]:
        analyze(df, model, duration=5)
    
    # Same with real prices (should show same WR, different PnL)
    print("\n\n" + "█"*80)
    print("  COMPARISON: Same WR but with REAL ask prices from first tick")
    print("█"*80)
    
    for model in ["catboost", "rf", "xgboost"]:
        analyze_with_real_prices(df, model, duration=5)
    
    # 15min
    print("\n\n" + "█"*80)
    print("  15min MARKETS")
    print("█"*80)
    for model in ["catboost"]:
        analyze(df, model, duration=15)
        analyze_with_real_prices(df, model, duration=15)
