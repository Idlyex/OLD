"""Final streak analysis with two modes + live trades breakdown."""
import json, math, sys, io
from collections import defaultdict
from pathlib import Path
from datetime import datetime

out_path = Path("results/analysis") / f"streak_final_{datetime.now():%Y-%m-%d}.txt"
out_path.parent.mkdir(parents=True, exist_ok=True)
_buf = io.StringIO()
_stdout = sys.stdout
class Tee:
    def write(self, s): _stdout.write(s); _buf.write(s)
    def flush(self): _stdout.flush()
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

# ── load ──
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

W = 82
SEP = "=" * W

def hdr(extra=""):
    print(f"  {'Model':<16} Str Conf    N    W    L     WR       PnL      EV{extra}")
    print("  " + "-" * (72 + len(extra)))

def row(r):
    m, s, th, n, w, l, wr, pnl, ev = r[:9]
    print(f"  {short(m):<16} {s}x >={int(th*100):2d}%  {n:3d}  {w:3d}  {l:3d}  {wr:5.1f}%  ${pnl:>+7.1f}  ${ev:>+5.2f}")

# ══════════════════════════════════════════════════════════════════
#  MODE A: streak counted WITH price gate (original)
# ══════════════════════════════════════════════════════════════════
def sim_mode_a(sp_lo, sp_hi):
    res = []
    for model in models:
        for streak_req in [1, 2, 3]:
            for th in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
                wins = 0; total = 0; pnl_sum = 0
                for slug, actual in om.items():
                    stk = st.get(slug, [])
                    streak = 0; streak_dir = None
                    for t in stk:
                        ep = t.get("entry_pct", -1)
                        if ep < 0 or ep > 0.9: continue
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
                        if spread > 0 and spread > 0.08:
                            streak = 0; streak_dir = None
                            continue
                        if c >= th:
                            if d == streak_dir: streak += 1
                            else: streak = 1; streak_dir = d
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
                    res.append((model, streak_req, th, total, wins, total-wins,
                                wins/total*100, pnl_sum, pnl_sum/total))
    return res

# ══════════════════════════════════════════════════════════════════
#  MODE B: streak by CONF ONLY, enter when price <= limit
# ══════════════════════════════════════════════════════════════════
def sim_mode_b(price_limit):
    """Streak counts on conf only (no price check). Entry when SP <= price_limit."""
    res = []
    for model in models:
        for streak_req in [2, 3]:
            for th in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
                wins = 0; total = 0; pnl_sum = 0
                for slug, actual in om.items():
                    stk = st.get(slug, [])
                    streak = 0; streak_dir = None; streak_met = False; sdir = None
                    for t in stk:
                        ep = t.get("entry_pct", -1)
                        if ep < 0 or ep > 0.9: continue
                        v = t.get(model)
                        if v is None: continue
                        d = "UP" if v > 0.5 else "DOWN"
                        c = max(v, 1 - v)
                        # streak by conf only
                        if c >= th:
                            if d == streak_dir: streak += 1
                            else: streak = 1; streak_dir = d
                            if streak >= streak_req:
                                streak_met = True; sdir = streak_dir
                        else:
                            streak = 0; streak_dir = None
                        # if streak met, look for price <= limit
                        if streak_met:
                            sp = t.get("yes_ask" if sdir == "UP" else "no_ask",
                                       t.get("yes" if sdir == "UP" else "no", 0.5))
                            if sp > 0 and sp <= price_limit and sp >= 0.03:
                                spread = t.get("yes_spread" if sdir == "UP" else "no_spread", 0)
                                if spread > 0 and spread > 0.08: continue
                                total += 1
                                won = sdir == actual
                                if won: wins += 1
                                shares = bet / sp
                                p = shares * (1.0 - sp) if won else -bet
                                pnl_sum += p
                                break
                if total >= 3:
                    res.append((model, streak_req, th, total, wins, total-wins,
                                wins/total*100, pnl_sum, pnl_sum/total))
    return res

def print_top(results, min_n=5, limit=20):
    filtered = [r for r in results if r[3] >= min_n]
    if not filtered:
        filtered = [r for r in results if r[3] >= 3]
    if not filtered:
        print("    No results")
        return

    print(f"\n  ── TOP {limit} BY WIN RATE (N >= {min_n}) ──")
    hdr()
    for r in sorted(filtered, key=lambda x: (-x[6], -x[3]))[:limit]: row(r)

    print(f"\n  ── TOP {limit} BY PNL ──")
    hdr()
    for r in sorted(filtered, key=lambda x: -x[7])[:limit]: row(r)

    print(f"\n  ── TOP {limit} BY EV ──")
    hdr()
    for r in sorted(filtered, key=lambda x: -x[8])[:limit]: row(r)

    scored = [(r, r[8] * math.sqrt(r[3]) * r[6] / 100) for r in filtered if r[8] > 0]
    scored.sort(key=lambda x: -x[1])
    if scored:
        print(f"\n  ── TOP {limit} SWEET SPOT ──")
        print(f"  {'Model':<16} Str Conf    N    W    L     WR       PnL      EV  Score")
        print("  " + "-" * 78)
        for r, sc in scored[:limit]:
            m, s, th, n, w, l, wr, pnl, ev = r
            print(f"  {short(m):<16} {s}x >={int(th*100):2d}%  {n:3d}  {w:3d}  {l:3d}  {wr:5.1f}%  ${pnl:>+7.1f}  ${ev:>+5.2f}  {sc:5.1f}")

# ══════════════════════════════════════════════════════════════════
#  RUN ALL
# ══════════════════════════════════════════════════════════════════

# ── Mode A: many price ranges ──
mode_a_ranges = [
    (0.10, 0.55, 5),   # full range
    (0.10, 0.50, 5),
    (0.10, 0.45, 5),
    (0.10, 0.40, 5),
    (0.10, 0.35, 3),
    (0.10, 0.30, 3),   # cheap
    (0.10, 0.25, 3),
    (0.10, 0.20, 3),
    (0.10, 0.15, 3),
    (0.10, 0.12, 3),
    (0.20, 0.55, 5),   # mid+high only
    (0.30, 0.55, 5),
    (0.15, 0.30, 3),   # mid-cheap
    (0.20, 0.40, 3),
    (0.25, 0.55, 5),
]

for sp_lo, sp_hi, mn in mode_a_ranges:
    print(f"\n{SEP}")
    print(f"  MODE A: STREAK + PRICE GATE  (SP ${sp_lo:.2f}-${sp_hi:.2f})")
    print(SEP)
    r = sim_mode_a(sp_lo, sp_hi)
    print_top(r, min_n=mn)

# ── Head-to-head: 80% vs 85% at key ranges ──
print(f"\n{'#'*W}")
print(f"  COMPARISON: 80% vs 85% CONFIDENCE THRESHOLD")
print(f"{'#'*W}")

for sp_lo, sp_hi in [(0.10, 0.55), (0.10, 0.30), (0.10, 0.20)]:
    print(f"\n  === SP ${sp_lo:.2f}-${sp_hi:.2f} ===")
    r = sim_mode_a(sp_lo, sp_hi)
    # filter to only catboost/xgboost/ensemble with 3x streak, 80% or 85%
    for th_val in [0.80, 0.85]:
        print(f"\n  ── Conf >= {int(th_val*100)}% ──")
        hdr()
        subset = [x for x in r if x[2] == th_val and x[1] == 3]
        subset.sort(key=lambda x: (-x[6], -x[8]))
        for rr in subset[:12]: row(rr)
        # also show 2x
        subset2 = [x for x in r if x[2] == th_val and x[1] == 2]
        subset2.sort(key=lambda x: (-x[6], -x[8]))
        if subset2:
            print(f"  -- 2x streak --")
            for rr in subset2[:8]: row(rr)

# ── Mode B kept but smaller ──
for pl in [0.30, 0.20, 0.15]:
    print(f"\n{SEP}")
    print(f"  MODE B: STREAK BY CONF ONLY → ENTER WHEN SP <= ${pl:.2f}")
    print(SEP)
    r = sim_mode_b(pl)
    print_top(r, min_n=3)

# ══════════════════════════════════════════════════════════════════
#  LIVE TRADES ANALYSIS
# ══════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  LIVE TRADES ANALYSIS")
print(SEP)

live_path = Path("results/ml_live_trades.json")
if live_path.exists():
    raw = json.loads(live_path.read_text(encoding="utf-8"))
    trades = raw if isinstance(raw, list) else raw.get("trades", [])

    dry = [t for t in trades if t.get("dry_run")]
    real = [t for t in trades if not t.get("dry_run")]
    all_t = trades

    def calc_pnl(tr):
        sp = tr.get("entry_price", 0)
        if sp <= 0 or sp >= 1: return 0
        shares = bet / sp
        if tr.get("won"): return shares * (1.0 - sp)
        else: return -bet

    def show_trades(label, tlist):
        if not tlist:
            print(f"\n  {label}: 0 trades\n")
            return
        w = sum(1 for t in tlist if t.get("won"))
        l = len(tlist) - w
        pnl_tot = sum(calc_pnl(t) for t in tlist)
        wr = w / len(tlist) * 100
        print(f"\n  {label}: {len(tlist)} trades | W:{w} L:{l} | WR:{wr:.1f}% | PnL:${pnl_tot:+.1f}")
        print(f"  {'slug':<40} {'dir':>5} {'conf':>5} {'sp':>5} {'W/L':>3} {'pnl':>8} out")
        print("  " + "-" * 75)
        for tr in tlist:
            s = tr.get("slug", "?")[-40:]
            d = tr.get("direction", "?")
            c = tr.get("confidence", 0)
            sp = tr.get("entry_price", 0)
            tag = "W" if tr.get("won") else "L"
            p = calc_pnl(tr)
            o = tr.get("outcome", "?")
            print(f"  {s:<40} {d:>5} {c:>.2f} {sp:>.2f} {tag:>3} ${p:>+6.2f}  {o}")

    show_trades("ALL TRADES", all_t)
    show_trades("DRY RUN ONLY", dry)
    show_trades("REAL ONLY", real)

    # ── streak filter on live trades ──
    print(f"\n  {'─'*W}")
    print(f"  LIVE TRADES STREAK FILTER CHECK")
    print(f"  {'─'*W}")

    for streak_req in [1, 2, 3]:
        passed = []; filtered = []
        for tr in all_t:
            slug = tr.get("slug", "")
            stk = st.get(slug, [])
            model_name = tr.get("model", "hermes_catboost")
            th_live = tr.get("confidence", 0.85)
            # Mode A: streak with price
            streak = 0; streak_dir = None; ok = False
            for t in stk:
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
                if c >= th_live:
                    if d == streak_dir: streak += 1
                    else: streak = 1; streak_dir = d
                    if streak >= streak_req: ok = True; break
                else:
                    streak = 0; streak_dir = None
            if ok: passed.append(tr)
            else: filtered.append(tr)

        pw = sum(1 for t in passed if t.get("won"))
        fw = sum(1 for t in filtered if t.get("won"))
        pp = sum(calc_pnl(t) for t in passed)
        fp = sum(calc_pnl(t) for t in filtered)
        p = len(passed); f = len(filtered)
        print(f"\n  Streak >= {streak_req} (Mode A):")
        if p: print(f"    PASSED:   {p:2d}/{p+f}  WR: {pw}/{p} = {pw/p*100:.1f}%  PnL: ${pp:+.1f}")
        else: print(f"    PASSED:   0/{p+f}")
        if f: print(f"    FILTERED: {f:2d}/{p+f}  WR: {fw}/{f} = {fw/f*100:.1f}%  PnL: ${fp:+.1f}")

    # Mode B streak check on live trades
    print(f"\n  {'─'*W}")
    print(f"  LIVE TRADES MODE B (conf-only streak → price <= limit)")
    print(f"  {'─'*W}")
    for price_lim in [0.30, 0.20, 0.15]:
        for streak_req in [2, 3]:
            passed = []; filtered = []
            for tr in all_t:
                slug = tr.get("slug", "")
                stk = st.get(slug, [])
                model_name = tr.get("model", "hermes_catboost")
                th_live = tr.get("confidence", 0.85)
                streak = 0; streak_dir = None; streak_met = False; sdir = None; ok = False
                for t in stk:
                    ep = t.get("entry_pct", -1)
                    if ep < 0 or ep > 0.9: continue
                    v = t.get(model_name)
                    if v is None: continue
                    d = "UP" if v > 0.5 else "DOWN"
                    c = max(v, 1 - v)
                    if c >= th_live:
                        if d == streak_dir: streak += 1
                        else: streak = 1; streak_dir = d
                        if streak >= streak_req: streak_met = True; sdir = streak_dir
                    else:
                        streak = 0; streak_dir = None
                    if streak_met:
                        sp = t.get("yes_ask" if sdir == "UP" else "no_ask", 0.5)
                        if sp > 0 and sp <= price_lim and sp >= 0.03:
                            ok = True; break
                if ok: passed.append(tr)
                else: filtered.append(tr)
            pw = sum(1 for t in passed if t.get("won"))
            fw = sum(1 for t in filtered if t.get("won"))
            pp = sum(calc_pnl(t) for t in passed)
            fp = sum(calc_pnl(t) for t in filtered)
            p = len(passed); f = len(filtered)
            print(f"\n  Streak>={streak_req} conf-only, SP<=${price_lim}:")
            if p: print(f"    PASSED:   {p:2d}/{p+f}  WR: {pw}/{p} = {pw/p*100:.1f}%  PnL: ${pp:+.1f}")
            else: print(f"    PASSED:    0/{p+f}")
            if f: print(f"    FILTERED: {f:2d}/{p+f}  WR: {fw}/{f} = {fw/f*100:.1f}%  PnL: ${fp:+.1f}")

    # ── verify outcomes vs outcomes file ──
    print(f"\n  {'─'*W}")
    print(f"  OUTCOME VERIFICATION (live trades vs outcomes file)")
    print(f"  {'─'*W}")
    mismatches = 0
    for tr in all_t:
        slug = tr.get("slug", "")
        live_out = tr.get("outcome", "")
        file_out = om.get(slug, "")
        match = "OK" if live_out == file_out else "MISMATCH"
        if live_out != file_out:
            mismatches += 1
            print(f"  {slug[-40:]}  live={live_out}  file={file_out}  {match}")
    if mismatches == 0:
        print(f"  All {len(all_t)} outcomes match.")
    else:
        print(f"  {mismatches} mismatches found!")

else:
    print("  No ml_live_trades.json found")

# ── save ──
sys.stdout = _stdout
out_path.write_text(_buf.getvalue(), encoding="utf-8")
print(f"\nSaved to {out_path}")
