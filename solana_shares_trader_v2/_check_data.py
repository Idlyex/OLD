import json
from datetime import datetime
from pathlib import Path

for f in sorted(Path("results").glob("live_ticks_*.jsonl")):
    lines = f.read_text().strip().split("\n")
    slugs = set(json.loads(l)["slug"] for l in lines)
    first = json.loads(lines[0])
    last = json.loads(lines[-1])
    print(f"{f.name}: {len(lines)} ticks, {len(slugs)} markets")
    print(f"  From: {datetime.fromtimestamp(first['ts'])}")
    print(f"  To:   {datetime.fromtimestamp(last['ts'])}")

print()
for f in sorted(Path("results").glob("live_outcomes_*.jsonl")):
    lines = f.read_text().strip().split("\n")
    print(f"{f.name}: {len(lines)} outcomes")

# Check ml_live_trades.json
p = Path("results/ml_live_trades.json")
if p.exists():
    trades = json.load(open(p))
    dry = [t for t in trades if t.get("dry_run")]
    live = [t for t in trades if not t.get("dry_run")]
    print(f"\nml_live_trades.json: {len(trades)} total (dry={len(dry)}, live={len(live)})")
else:
    print("\nNo ml_live_trades.json yet")
