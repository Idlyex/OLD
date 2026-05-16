"""Streak analysis: require N consecutive ticks with conf >= threshold in same direction."""
import json, math, sys, io
from collections import defaultdict
from pathlib import Path
from datetime import datetime

# ── output tee ──
out_path = Path("results/analysis") / f"streak_analysis_{datetime.now():%Y-%m-%d}.txt"
out_path.parent.mkdir(parents=True, exist_ok=True)
_buf = io.StringIO()
_stdout = sys.stdout

class Tee:
    def write(self, s):
        _stdout.write(s); _buf.write(s)
    def flush(self):
        _stdout.flush()

sys.stdout = Tee()

# ── detect date ──
_date = None
if "--date" in sys.argv:
    _date = sys.argv[sys.argv.index("--date") + 1]
if _date is None:
    _files = sorted(Path("results/ticks").glob("ticks_*.jsonl"))
    _files = [f for f in _files if "binance" not in f.name and "hermes" not in f.name]
    if _files: _date = _files[-1].stem.replace("ticks_", "")
if not _date:
    print("No ticks files found!"); sys.exit(1)
print(f"Date: {_date}")

# ── load data ──
ticks = []
for line in open(f"results/ticks/ticks_{_date}.jsonl", encoding="utf-8"):
    try: ticks.append(json.loads(line))
    except: pass
outcomes = []
for line in open(f"results/ticks/outcomes_{_date}.jsonl", encoding="utf-8"):
    try: outcomes.append(json.loads(line))
    except: pass

om = {o["slug"]: o["outcome"] for o in outcomes}
st = defaultdict(list)
for t in ticks:
    if t["slug"] in om:
        st[t["slug"]].append(t)

models = [
    "hermes_catboost", "hermes_lgbm", "hermes_xgboost", "hermes_ensemble",
    "binance_catboost", "binance_lgbm", "binance_xgboost", "binance_ensemble",
]
bet = 2.85

def short(m):
    return m.replace("hermes_", "H_").replace("binance_", "B_")

def hdr(extra=""):
    print(f"  {'Model':<20} Str Conf    N    W    L     WR       PnL      EV{extra}")
    print("  " + "-" * (76 + len(extra)))

def row(r):
    m, s, th, n, w, l, wr, pnl, ev = r[:9]
    print(f"  {short(m):<20} {s}x >={int(th*100):2d}%  {n:3d}  {w:3d}  {l:3d}  {wr:5.1f}%  ${pnl:>+7.1f}  ${ev:>+5.2f}")

def run_sim(sp_lo, sp_hi, max_spread=0.08, elo=0.0, ehi=0.9):
    """Run streak sim with given SP range. Returns list of result tuples."""
    res = []
    for model in models:
        for streak_req in [1, 2, 3]:
            for th in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
                wins = 0; total = 0; pnl_sum = 0
                for slug, actual in om.items():
                    sticks = st.get(slug, [])
                    streak = 0; streak_dir = None
                    for t in sticks:
                        ep = t.get("entry_pct", -1)
                        if ep < elo or ep > ehi: continue
                        v = t.get(model)
                        if v is None: continue
                        d = "UP" if v > 0.5 else "DOWN"
                        c = max(v, 1 - v)
                        sp = t.get("yes_ask" if d == "UP" else "no_ask",
                                   t.get("yes" if d == "UP" else "no", 0.5))
                        if sp <= 0 or sp > 1 or sp < sp_lo or sp >= sp_hi:
                            streak = 0; streak_dir = None
                            continue
                        spread = t.get("yes_spread" if d == "UP" else "no_spread", 0)
                        if spread > 0 and spread > max_spread:
                            streak = 0; streak_dir = None
                            continue
                        if c >= th:
                            if d == streak_dir:
                                streak += 1
                            else:
                                streak = 1; streak_dir = d
                            if streak >= streak_req:
                                total += 1
                                won = streak_dir == actual
                                if won: wins += 1
                                shares = bet / sp
                                p = shares * (1.0 - sp) if won else -bet
                                pnl_sum += p
                                break
                        else:
                            streak = 0; streak_dir = None
                if total >= 3:
                    ev = pnl_sum / total
                    wr = wins / total * 100
                    res.append((model, streak_req, th, total, wins, total - wins, wr, pnl_sum, ev))
    return res

def print_tables(results, label, min_n=5):
    print()
    print("=" * 82)
    print(f"  {label}")
    print("=" * 82)

    # BY WIN RATE
    filtered = [r for r in results if r[3] >= min_n]
    print(f"\n  --- TOP 25 BY WIN RATE (N >= {min_n}) ---")
    hdr()
    for r in sorted(filtered, key=lambda x: -x[6])[:25]:
        row(r)

    # BY PNL
    print(f"\n  --- TOP 25 BY PNL ---")
    hdr()
    for r in sorted(filtered, key=lambda x: -x[7])[:25]:
        row(r)

    # BY EV
    print(f"\n  --- TOP 25 BY EV ($/trade) ---")
    hdr()
    for r in sorted(filtered, key=lambda x: -x[8])[:25]:
        row(r)

    # BY TRADE COUNT WR >= 65%
    wr65 = [r for r in filtered if r[6] >= 65]
    if wr65:
        print(f"\n  --- TOP 25 BY TRADE COUNT (WR >= 65%) ---")
        hdr()
        for r in sorted(wr65, key=lambda x: -x[3])[:25]:
            row(r)

    # SWEET SPOT
    scored = [(r, r[8] * math.sqrt(r[3]) * r[6] / 100) for r in filtered if r[8] > 0]
    scored.sort(key=lambda x: -x[1])
    if scored:
        print(f"\n  --- TOP 25 SWEET SPOT (EV * sqrt(N) * WR) ---")
        print(f"  {'Model':<20} Str Conf    N    W    L     WR       PnL      EV  Score")
        print("  " + "-" * 82)
        for r, sc in scored[:25]:
            m, s, th, n, w, l, wr, pnl, ev = r
            print(f"  {short(m):<20} {s}x >={int(th*100):2d}%  {n:3d}  {w:3d}  {l:3d}  {wr:5.1f}%  ${pnl:>+7.1f}  ${ev:>+5.2f}  {sc:5.1f}")

# ══════════════════════════════════════════════════════════════
#  MAIN ANALYSIS
# ══════════════════════════════════════════════════════════════

# 1) Full range SP $0.10 - $0.55
results_full = run_sim(0.10, 0.55)
print_tables(results_full, "STREAK ANALYSIS -- SP $0.10-$0.55 (full range)")

# 2) SP $0.20 - $0.40
results_low = run_sim(0.20, 0.40)
print_tables(results_low, "STREAK ANALYSIS -- SP $0.20-$0.40 (low price bin)", min_n=3)

# 3) SP $0.40 - $0.60
results_mid = run_sim(0.40, 0.60)
print_tables(results_mid, "STREAK ANALYSIS -- SP $0.40-$0.60 (mid price bin)", min_n=3)

# ══════════════════════════════════════════════════════════════
#  LIVE TRADES STREAK CHECK
# ══════════════════════════════════════════════════════════════
print()
print("=" * 82)
print("  LIVE TRADES STREAK CHECK")
print("=" * 82)

live_path = Path("results/ml_live_trades.json")
if live_path.exists():
    live = json.loads(live_path.read_text(encoding="utf-8"))
    trades = live if isinstance(live, list) else live.get("trades", [])
    print(f"  {len(trades)} live trades\n")

    for streak_req in [1, 2, 3]:
        passed_trades = []
        filtered_trades = []
        for tr in trades:
            slug = tr.get("slug", tr.get("market_slug", ""))
            sticks = st.get(slug, [])
            model_name = tr.get("model", "hermes_catboost")
            th_live = tr.get("confidence", 0.85)
            streak = 0; streak_dir = None; would_enter = False
            for t in sticks:
                ep = t.get("entry_pct", -1)
                if ep < 0 or ep > 0.9: continue
                v = t.get(model_name)
                if v is None: continue
                d = "UP" if v > 0.5 else "DOWN"
                c = max(v, 1 - v)
                sp = t.get("yes_ask" if d == "UP" else "no_ask", 0.5)
                if sp <= 0 or sp > 1 or sp < 0.10 or sp >= 0.55:
                    streak = 0; streak_dir = None
                    continue
                spread = t.get("yes_spread" if d == "UP" else "no_spread", 0)
                if spread > 0 and spread > 0.08:
                    streak = 0; streak_dir = None
                    continue
                if c < th_live:
                    streak = 0; streak_dir = None
                    continue
                if d == streak_dir:
                    streak += 1
                else:
                    streak = 1; streak_dir = d
                if streak >= streak_req:
                    would_enter = True
                    break
            won = tr.get("won", False)
            pnl_val = tr.get("pnl", tr.get("profit", 0)) or 0
            tag = "W" if won else "L"
            if would_enter:
                passed_trades.append((slug, tag, pnl_val))
            else:
                filtered_trades.append((slug, tag, pnl_val))

        p = len(passed_trades)
        f = len(filtered_trades)
        pw = sum(1 for _, t, _ in passed_trades if t == "W")
        fw = sum(1 for _, t, _ in filtered_trades if t == "W")
        pp = sum(pnl for _, _, pnl in passed_trades)
        fp = sum(pnl for _, _, pnl in filtered_trades)

        print(f"  Streak >= {streak_req}:")
        print(f"    Passed:   {p}/{p+f} ({p/(p+f)*100:.0f}%)  WR: {pw}/{p} = {pw/p*100:.1f}%  PnL: ${pp:+.2f}" if p else f"    Passed: 0")
        print(f"    Filtered: {f}/{p+f} ({f/(p+f)*100:.0f}%)  WR: {fw}/{f} = {fw/f*100:.1f}%  PnL: ${fp:+.2f}" if f else f"    Filtered: 0")
        if filtered_trades:
            print(f"    Filtered trades:")
            for slug, tag, pnl_val in filtered_trades:
                print(f"      {slug[-40:]} {tag} ${pnl_val:+.2f}")
        print()
else:
    print("  No ml_live_trades.json found")

# ── save ──
sys.stdout = _stdout
out_path.write_text(_buf.getvalue(), encoding="utf-8")
print(f"\nSaved to {out_path}")
