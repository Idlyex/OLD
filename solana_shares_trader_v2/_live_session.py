"""Quick live session summary + full analysis."""
import json, pandas as pd, numpy as np
from pathlib import Path
from datetime import datetime

# Current data
lines = Path("results/live_ticks_2026-05-10.jsonl").read_text().strip().split("\n")
slugs = set(json.loads(l)["slug"] for l in lines)
first = json.loads(lines[0]); last = json.loads(lines[-1])
print(f"Ticks: {len(lines)}, Markets: {len(slugs)}")
print(f"From: {datetime.fromtimestamp(first['ts'])}")
print(f"To:   {datetime.fromtimestamp(last['ts'])}")

oc = Path("results/live_outcomes_2026-05-10.jsonl").read_text().strip().split("\n")
print(f"Outcomes: {len(oc)}")

# Live trades
p = Path("results/ml_live_trades.json")
if p.exists():
    trades = json.load(open(p))
    dry = [t for t in trades if t.get("dry_run")]
    live = [t for t in trades if not t.get("dry_run")]
    print(f"\nTrades: {len(trades)} total (dry={len(dry)}, live={len(live)})")
    
    if trades:
        df = pd.DataFrame(trades)
        for mode, sub in [("DRY", df[df.dry_run == True]), ("LIVE", df[df.dry_run == False])]:
            if sub.empty:
                continue
            w = int(sub.won.sum()); n = len(sub); pnl = sub.pnl_usd.sum()
            print(f"\n  === {mode} TRADES: {n} trades, {w}W/{n-w}L, WR={w/n:.0%}, PnL=${pnl:+.2f} ===")
            cols = ["slug", "direction", "confidence", "entry_price", "outcome", "won", "pnl_usd", "entry_time"]
            avail = [c for c in cols if c in sub.columns]
            print(sub[avail].to_string(index=False))
else:
    print("\nNo ml_live_trades.json yet")
