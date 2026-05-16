"""
LIVE TRADES DEEP ANALYSIS
Analyzes actual live trades vs what analysis predicted.
Identifies WHY live is losing while analysis shows 85% WR.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import time

t0 = time.time()
print("=" * 90)
print("  LIVE TRADES DEEP ANALYSIS")
print("=" * 90)

# Load trades
df = pd.read_json("results/ml_live_trades.json")
live = df[~df["dry_run"]].copy()
dry = df[df["dry_run"]].copy()

print(f"\n  Total trades logged: {len(df)} (DRY: {len(dry)}, LIVE: {len(live)})")
print(f"  Time range: {df['entry_time'].min()} → {df['exit_time'].max()}")

# ═══════════════════════════════════════════════════════════
# SECTION 1: LIVE SUMMARY
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*90}")
print(f"  SECTION 1: LIVE TRADES SUMMARY (from ~15:00 May 9)")
print(f"{'='*90}")

live["won"] = live["won"].astype(bool)
won = live["won"].sum()
lost = len(live) - won
wr = won / len(live) * 100
total_pnl = live["pnl_usd"].sum()
avg_pnl = live["pnl_usd"].mean()
avg_conf = live["confidence"].mean()
avg_ep = live["entry_price"].mean()

print(f"  Trades: {len(live)} | Won: {won} | Lost: {lost} | WR: {wr:.1f}%")
print(f"  Total PnL: ${total_pnl:.2f}")
print(f"  Avg PnL/trade: ${avg_pnl:.2f}")
print(f"  Avg confidence: {avg_conf:.4f}")
print(f"  Avg entry price: ${avg_ep:.4f}")
print(f"  Avg shares: {live['shares'].mean():.2f}")

# ═══════════════════════════════════════════════════════════
# SECTION 2: EACH LIVE TRADE
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*90}")
print(f"  SECTION 2: ALL LIVE TRADES DETAIL")
print(f"{'='*90}")
print(f"  {'#':>2} {'Time':>11} {'Slug':>20} {'Dir':>5} {'Conf':>5} {'Price':>6} {'PnL$':>7} {'W/L':>3} {'Duration':>8}")
print(f"  {'-'*80}")

cum_pnl = 0
for i, (_, r) in enumerate(live.iterrows()):
    cum_pnl += r["pnl_usd"]
    slug_short = r["slug"].split("-")[-1][:8]  # just the timestamp part
    dur = r.get("duration_min", 5)
    entry_t = str(r["entry_time"])[11:19]
    wl = "W" if r["won"] else "L"
    print(f"  {i+1:>2} {entry_t} {r['slug'][-20:]:>20} {r['direction']:>5} {r['confidence']:.3f} ${r['entry_price']:.4f} ${r['pnl_usd']:>+6.2f} {wl:>3}  {dur}min  cum=${cum_pnl:+.2f}")

# ═══════════════════════════════════════════════════════════
# SECTION 3: WIN vs LOSS COMPARISON
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*90}")
print(f"  SECTION 3: WIN vs LOSS COMPARISON")
print(f"{'='*90}")

wins = live[live["won"]]
losses = live[~live["won"]]

print(f"\n  WINS ({len(wins)} trades):")
print(f"    Avg confidence: {wins['confidence'].mean():.4f}")
print(f"    Avg entry price: ${wins['entry_price'].mean():.4f}")
print(f"    Avg PnL: ${wins['pnl_usd'].mean():.2f}")
print(f"    Directions: UP={len(wins[wins['direction']=='UP'])}, DOWN={len(wins[wins['direction']=='DOWN'])}")

print(f"\n  LOSSES ({len(losses)} trades):")
print(f"    Avg confidence: {losses['confidence'].mean():.4f}")
print(f"    Avg entry price: ${losses['entry_price'].mean():.4f}")
print(f"    Avg PnL: ${losses['pnl_usd'].mean():.2f}")
print(f"    Directions: UP={len(losses[losses['direction']=='UP'])}, DOWN={len(losses[losses['direction']=='DOWN'])}")

# ═══════════════════════════════════════════════════════════
# SECTION 4: TIMING ANALYSIS
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*90}")
print(f"  SECTION 4: ENTRY TIMING & MARKET LIFECYCLE")
print(f"{'='*90}")

# Parse entry timing from slug (slug contains market start timestamp)
for _, r in live.iterrows():
    slug_parts = r["slug"].split("-")
    market_start_ts = int(slug_parts[-1])
    dur_min = r["duration_min"]
    market_end_ts = market_start_ts + dur_min * 60
    entry_ts = r["entry_ts"]
    market_duration = market_end_ts - market_start_ts
    elapsed = entry_ts - market_start_ts
    entry_pct = elapsed / market_duration if market_duration > 0 else 0
    live.loc[r.name, "entry_pct"] = entry_pct
    live.loc[r.name, "time_remaining_s"] = market_end_ts - entry_ts

print(f"\n  Entry timing distribution:")
print(f"    Min entry_pct: {live['entry_pct'].min():.2%}")
print(f"    Max entry_pct: {live['entry_pct'].max():.2%}")
print(f"    Mean entry_pct: {live['entry_pct'].mean():.2%}")
print(f"    Median entry_pct: {live['entry_pct'].median():.2%}")

print(f"\n  Entry timing by outcome:")
if len(wins) > 0:
    print(f"    Wins: avg entry_pct = {live.loc[wins.index, 'entry_pct'].mean():.2%}")
if len(losses) > 0:
    print(f"    Losses: avg entry_pct = {live.loc[losses.index, 'entry_pct'].mean():.2%}")

# ═══════════════════════════════════════════════════════════
# SECTION 5: DIRECTION BIAS
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*90}")
print(f"  SECTION 5: DIRECTION BIAS")
print(f"{'='*90}")

for d in ["UP", "DOWN"]:
    subset = live[live["direction"] == d]
    if len(subset) == 0:
        continue
    w = subset["won"].sum()
    l = len(subset) - w
    wr_d = w / len(subset) * 100
    pnl_d = subset["pnl_usd"].sum()
    print(f"  {d:>5}: {len(subset)} trades, {w}W/{l}L, WR={wr_d:.1f}%, PnL=${pnl_d:+.2f}")

# ═══════════════════════════════════════════════════════════
# SECTION 6: COMPARE WITH ANALYSIS EXPECTATIONS
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*90}")
print(f"  SECTION 6: LIVE vs ANALYSIS COMPARISON")
print(f"{'='*90}")

# Load today's tick data to check what analysis would say about same markets
results_dir = Path("results")
tick_files = sorted(results_dir.glob("live_ticks_*.jsonl"))

# We need ticks from the time period when live was running
# Load all available tick files
all_ticks = []
for tf in tick_files:
    try:
        tdf = pd.read_json(tf, lines=True)
        all_ticks.append(tdf)
    except:
        pass

if all_ticks:
    ticks = pd.concat(all_ticks, ignore_index=True)
    
    # For each live trade, find if analysis would have taken the same trade
    print(f"\n  Checking analysis predictions for same markets as live trades...")
    print(f"  Tick data: {len(ticks)} ticks, {ticks['slug'].nunique()} markets")
    
    # Check market outcomes for the live trade slugs
    out_files = sorted(results_dir.glob("live_outcomes_*.jsonl"))
    all_outcomes = []
    for of in out_files:
        try:
            odf = pd.read_json(of, lines=True)
            all_outcomes.append(odf)
        except:
            pass
    
    if all_outcomes:
        outcomes = pd.concat(all_outcomes, ignore_index=True).drop_duplicates("slug", keep="last")
        
        # Check each live trade slug against ticks
        live_slugs = live["slug"].unique()
        matched = ticks[ticks["slug"].isin(live_slugs)]
        matched = matched.merge(outcomes[["slug", "outcome"]], on="slug", how="inner")
        
        print(f"  Found ticks for {matched['slug'].nunique()}/{len(live_slugs)} live trade markets")
        
        if len(matched) > 0 and "catboost" in matched.columns:
            # For each live trade market, what did catboost predict?
            print(f"\n  Analysis prediction vs live trade outcome:")
            print(f"  {'Slug':<25} {'Live':>5} {'Outcome':>7} {'CB_conf':>7} {'CB_dir':>6} {'Match':>5}")
            print(f"  {'-'*65}")
            
            for _, trade in live.iterrows():
                slug = trade["slug"]
                market_ticks = matched[matched["slug"] == slug]
                if len(market_ticks) == 0:
                    continue
                
                # Get catboost prediction at peak confidence for this market
                cb_probs = market_ticks["catboost"].values
                cb_conf = np.maximum(cb_probs, 1 - cb_probs)
                best_idx = cb_conf.argmax()
                best_conf = cb_conf[best_idx]
                best_dir = "UP" if cb_probs[best_idx] > 0.5 else "DOWN"
                outcome = market_ticks["outcome"].iloc[0]
                
                # Did analysis direction match?
                analysis_correct = (best_dir == outcome)
                live_correct = trade["won"]
                
                slug_short = slug[-22:]
                print(f"  {slug_short:<25} {trade['direction']:>5} {outcome:>7} {best_conf:.4f} {best_dir:>6} {'OK' if analysis_correct else 'WRONG':>5}")

# ═══════════════════════════════════════════════════════════
# SECTION 7: KEY METRICS & ROOT CAUSE
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*90}")
print(f"  SECTION 7: ROOT CAUSE ANALYSIS")
print(f"{'='*90}")

# 1. Entry price vs expected
print(f"\n  1. ENTRY PRICES:")
print(f"     Analysis assumes: ~$0.50 (fair value)")
print(f"     Live avg entry:   ${live['entry_price'].mean():.4f}")
print(f"     Losses avg entry: ${losses['entry_price'].mean():.4f}")
print(f"     Wins avg entry:   ${wins['entry_price'].mean():.4f}")

# 2. Hold time
print(f"\n  2. HOLD TIME:")
print(f"     Avg hold: {live['hold_time_s'].mean():.0f}s")
print(f"     Wins hold: {wins['hold_time_s'].mean():.0f}s")  
print(f"     Losses hold: {losses['hold_time_s'].mean():.0f}s")

# 3. Confidence at entry
print(f"\n  3. CONFIDENCE:")
high_conf = live[live["confidence"] >= 0.90]
med_conf = live[(live["confidence"] >= 0.70) & (live["confidence"] < 0.90)]
low_conf = live[live["confidence"] < 0.70]
if len(high_conf) > 0:
    print(f"     High (>=90%): {len(high_conf)} trades, WR={high_conf['won'].mean()*100:.0f}%, PnL=${high_conf['pnl_usd'].sum():+.2f}")
if len(med_conf) > 0:
    print(f"     Med (70-90%): {len(med_conf)} trades, WR={med_conf['won'].mean()*100:.0f}%, PnL=${med_conf['pnl_usd'].sum():+.2f}")
if len(low_conf) > 0:
    print(f"     Low (<70%):   {len(low_conf)} trades, WR={low_conf['won'].mean()*100:.0f}%, PnL=${low_conf['pnl_usd'].sum():+.2f}")

# 4. Asymmetry check
print(f"\n  4. WIN/LOSS ASYMMETRY:")
avg_win = wins["pnl_usd"].mean() if len(wins) > 0 else 0
avg_loss = losses["pnl_usd"].mean() if len(losses) > 0 else 0
print(f"     Avg win:  ${avg_win:+.2f}")
print(f"     Avg loss: ${avg_loss:+.2f}")
print(f"     Risk/Reward: {abs(avg_loss/avg_win):.2f}x" if avg_win != 0 else "     Risk/Reward: N/A")

# 5. Time of day
print(f"\n  5. TIME OF DAY:")
live["hour"] = pd.to_datetime(live["entry_time"]).dt.hour
for h in sorted(live["hour"].unique()):
    subset = live[live["hour"] == h]
    w = subset["won"].sum()
    pnl = subset["pnl_usd"].sum()
    print(f"     {h:02d}:00 - {len(subset)} trades, {w}W/{len(subset)-w}L, PnL=${pnl:+.2f}")

elapsed = time.time() - t0
print(f"\n{'='*90}")
print(f"  Analysis complete in {elapsed:.2f}s")
print(f"{'='*90}")
