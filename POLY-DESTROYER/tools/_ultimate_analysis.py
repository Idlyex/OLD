"""ULTIMATE streak/config analysis — BATCHED + CACHED for instant execution.

Key optimisations vs naive nested loops:
  1. Precompute per-slug data as plain Python lists (faster elem access than numpy)
  2. Batch ALL (streak × threshold) combos in ONE tick-pass per (slug, model)
  3. Cache sim results by param tuple — Section 10 reuses Section 1, etc.
Result: ~50-100× faster than original.
"""
import json, math, sys, io, time
from collections import defaultdict
from pathlib import Path
from datetime import datetime

out_path = Path("results/analysis") / f"ultimate_{datetime.now():%Y-%m-%d_%H%M}.txt"
out_path.parent.mkdir(parents=True, exist_ok=True)
_buf = io.StringIO()
_stdout = sys.stdout
class Tee:
    def write(self, s): _stdout.write(s); _buf.write(s)
    def flush(self): _stdout.flush()
sys.stdout = Tee()

t0 = time.time()

# ── Detect date ──
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

# ── Load data ──
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

MODELS = [
    "hermes_catboost", "hermes_lgbm", "hermes_xgboost", "hermes_ensemble",
    "binance_catboost", "binance_lgbm", "binance_xgboost", "binance_ensemble",
]
BET = 2.85
W = 90
SEP = "=" * W

def short(m):
    return m.replace("hermes_", "H_").replace("binance_", "B_")

# ══════════════════════════════════════════════════════════════════
#  PRECOMPUTE: build PLAIN PYTHON LISTS per slug × model (ONE TIME)
#  (numpy element access in Python loops is 3-5× slower than list[i])
# ══════════════════════════════════════════════════════════════════

print(f"Data: {len(ticks)} ticks, {len(om)} markets with outcomes, {len(st)} with tick data")

import numpy as np
slugs = sorted(st.keys())
outcome_list = [1 if om[s] == "UP" else 0 for s in slugs]
N_SLUGS = len(slugs)

slug_ep_L = []          # [si] -> list[float]
slug_model_L = []       # [si] -> {model: (dir_up_L, conf_L, sp_L, spread_L)}

_t_pre = time.perf_counter()

for si, slug in enumerate(slugs):
    tick_list = st[slug]
    ep = [t.get("entry_pct", -1.0) for t in tick_list]
    slug_ep_L.append(ep)

    mdata = {}
    for model in MODELS:
        probs = np.array([t.get(model, np.nan) for t in tick_list], dtype=np.float64)
        dir_up = probs > 0.5
        conf = np.where(dir_up, probs, 1.0 - probs)
        conf = np.nan_to_num(conf, nan=0.0)

        yes_ask = np.array([t.get("yes_ask", t.get("yes", 0.5)) for t in tick_list], dtype=np.float64)
        no_ask  = np.array([t.get("no_ask",  t.get("no",  0.5)) for t in tick_list], dtype=np.float64)
        sp = np.where(dir_up, yes_ask, no_ask)

        yes_spread = np.array([t.get("yes_spread", 0) for t in tick_list], dtype=np.float64)
        no_spread  = np.array([t.get("no_spread",  0) for t in tick_list], dtype=np.float64)
        spread = np.where(dir_up, yes_spread, no_spread)

        # Convert to plain lists for fast element access in tight loop
        mdata[model] = (dir_up.tolist(), conf.tolist(), sp.tolist(), spread.tolist())
    slug_model_L.append(mdata)

del np  # not needed anymore
print(f"Precompute: {time.perf_counter() - _t_pre:.3f}s")


# ══════════════════════════════════════════════════════════════════
#  BATCHED SIMULATOR: one tick-pass per (slug, model) for ALL combos
# ══════════════════════════════════════════════════════════════════

_cache = {}

def sim_fast(sp_lo, sp_hi, entry_lo=0.0, entry_hi=0.95, max_spread=0.08,
             streaks=None, thresholds=None, models_list=None, streak_mode="strict"):
    """Batch sim: iterates each slug's ticks ONCE per model, evaluating
    ALL (streak × threshold) combos simultaneously.  ~45× fewer iterations.

    streak_mode="strict": bad price/spread RESETS streak to 0 (matches live _update_streak).
    streak_mode="skip":   bad price/spread SKIPS tick (streak unchanged, no build, no reset).
                          Streak builds only on ticks where ALL conditions met.
    streak_mode="soft":   streak builds on conf+dir ONLY (ignores price for streak building).
                          Entry only when streak met AND price+spread OK on same tick.
    """
    if streaks is None: streaks = [1, 2, 3, 4, 5]
    if thresholds is None: thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    if models_list is None: models_list = MODELS

    key = (sp_lo, sp_hi, entry_lo, entry_hi, max_spread,
           tuple(streaks), tuple(thresholds), tuple(models_list), streak_mode)
    if key in _cache:
        return _cache[key]

    nS = len(streaks)
    nT = len(thresholds)
    results = []

    for model in models_list:
        # Accumulators for each (streak_idx, th_idx)
        g_total   = [[0]*nT for _ in range(nS)]
        g_wins    = [[0]*nT for _ in range(nS)]
        g_pnl     = [[0.0]*nT for _ in range(nS)]
        g_details = [[[] for _ in range(nT)] for _ in range(nS)]
        g_consec  = [[0]*nT for _ in range(nS)]
        g_maxcon  = [[0]*nT for _ in range(nS)]

        for si in range(N_SLUGS):
            ep_L = slug_ep_L[si]
            du_L, co_L, sp_L, spr_L = slug_model_L[si][model]
            actual_up = outcome_list[si]
            n = len(ep_L)
            slug_name = slugs[si]

            # Per-threshold streak state for this slug
            st_count = [0] * nT
            st_dir   = [False] * nT   # True = UP
            # Track which (streak, th) combos already entered for this slug
            entered_flat = [False] * (nS * nT)
            remaining = nS * nT

            for i in range(n):
                if remaining == 0:
                    break
                e = ep_L[i]
                if e < entry_lo or e > entry_hi:
                    continue
                c = co_L[i]
                if c < 0.01:
                    continue

                d = du_L[i]       # bool: True=UP
                s = sp_L[i]       # share price
                spr = spr_L[i]    # spread

                price_ok = sp_lo <= s < sp_hi and 0 < s <= 1
                spread_ok = spr <= 0 or spr <= max_spread

                if streak_mode == "strict":
                    if not (price_ok and spread_ok):
                        for ti in range(nT):
                            st_count[ti] = 0
                        continue
                elif streak_mode == "skip":
                    if not (price_ok and spread_ok):
                        continue   # no reset, no build — streak frozen

                # Evaluate each threshold
                for ti in range(nT):
                    if c >= thresholds[ti]:
                        if d == st_dir[ti] and st_count[ti] > 0:
                            st_count[ti] += 1
                        else:
                            st_count[ti] = 1
                            st_dir[ti] = d

                        cur = st_count[ti]
                        # SOFT: check price for entry only
                        # STRICT/SKIP: price already OK (filtered above)
                        if streak_mode == "soft" and not (price_ok and spread_ok):
                            continue
                        # Check all streak requirements
                        for si2 in range(nS):
                            idx = si2 * nT + ti
                            if not entered_flat[idx] and cur >= streaks[si2]:
                                entered_flat[idx] = True
                                remaining -= 1
                                won = (st_dir[ti] == bool(actual_up))
                                shares = BET / s
                                p = shares * (1.0 - s) if won else -BET

                                g_total[si2][ti] += 1
                                g_pnl[si2][ti] += p
                                if won:
                                    g_wins[si2][ti] += 1
                                    g_consec[si2][ti] = 0
                                else:
                                    g_consec[si2][ti] += 1
                                    if g_consec[si2][ti] > g_maxcon[si2][ti]:
                                        g_maxcon[si2][ti] = g_consec[si2][ti]
                                dir_str = "UP" if st_dir[ti] else "DOWN"
                                g_details[si2][ti].append((slug_name, dir_str, s, won, p))
                    else:
                        st_count[ti] = 0

        # Build result tuples
        for si2 in range(nS):
            for ti in range(nT):
                total = g_total[si2][ti]
                if total < 3:
                    continue
                wins = g_wins[si2][ti]
                losses = total - wins
                pnl_sum = g_pnl[si2][ti]
                wr = wins / total * 100
                ev = pnl_sum / total
                details = g_details[si2][ti]
                win_pnl = sum(p for _, _, _, w, p in details if w)
                loss_pnl = abs(sum(p for _, _, _, w, p in details if not w))
                pf = win_pnl / max(loss_pnl, 0.01)
                avg_w = win_pnl / max(wins, 1)
                avg_l = sum(p for _, _, _, w, p in details if not w) / max(losses, 1)
                results.append((
                    model, streaks[si2], thresholds[ti], total, wins, losses, wr, pnl_sum, ev,
                    g_maxcon[si2][ti], pf, avg_w, avg_l, details
                ))

    _cache[key] = results
    return results


# ══════════════════════════════════════════════════════════════════
#  PRINTING HELPERS
# ══════════════════════════════════════════════════════════════════

def hdr(extra=""):
    print(f"  {'Model':<16} Str Conf    N    W    L     WR       PnL      EV{extra}")
    print("  " + "-" * (72 + len(extra)))

def row(r, extra=""):
    m, s, th, n, w, l, wr, pnl, ev = r[:9]
    print(f"  {short(m):<16} {s}x >={int(th*100):2d}%  {n:3d}  {w:3d}  {l:3d}  {wr:5.1f}%  ${pnl:>+7.1f}  ${ev:>+5.2f}{extra}")

def print_top(results, min_n=5, limit=15, show_risk=False):
    filt = [r for r in results if r[3] >= min_n]
    if not filt: filt = [r for r in results if r[3] >= 3]
    if not filt: print("    No results"); return []

    print(f"\n  ── TOP {limit} BY WIN RATE (N ≥ {min_n}) ──")
    hdr("  PF  MaxL" if show_risk else "")
    for r in sorted(filt, key=lambda x: (-x[6], -x[3]))[:limit]:
        extra = f"  {r[10]:4.1f}  {r[9]:3d}" if show_risk else ""
        row(r, extra)

    print(f"\n  ── TOP {limit} BY PNL ──")
    hdr("  PF  MaxL" if show_risk else "")
    for r in sorted(filt, key=lambda x: -x[7])[:limit]:
        extra = f"  {r[10]:4.1f}  {r[9]:3d}" if show_risk else ""
        row(r, extra)

    print(f"\n  ── TOP {limit} BY EV ──")
    hdr("  PF  MaxL" if show_risk else "")
    for r in sorted(filt, key=lambda x: -x[8])[:limit]:
        extra = f"  {r[10]:4.1f}  {r[9]:3d}" if show_risk else ""
        row(r, extra)

    scored = [(r, r[8] * math.sqrt(r[3]) * r[6] / 100) for r in filt if r[8] > 0]
    scored.sort(key=lambda x: -x[1])
    if scored:
        print(f"\n  ── TOP {limit} SWEET SPOT (EV × √N × WR%) ──")
        e = "  PF  MaxL" if show_risk else ""
        print(f"  {'Model':<16} Str Conf    N    W    L     WR       PnL      EV  Score{e}")
        print("  " + "-" * (84 + len(e)))
        for r, sc in scored[:limit]:
            m, s, th, n, w, l, wr, pnl, ev = r[:9]
            extra = f"  {r[10]:4.1f}  {r[9]:3d}" if show_risk else ""
            print(f"  {short(m):<16} {s}x >={int(th*100):2d}%  {n:3d}  {w:3d}  {l:3d}  {wr:5.1f}%  ${pnl:>+7.1f}  ${ev:>+5.2f}  {sc:5.1f}{extra}")

    risk_sc = [(r, r[8] * r[6] / 100 / max(1, r[9] + 0.5)) for r in filt if r[8] > 0]
    risk_sc.sort(key=lambda x: -x[1])
    if risk_sc:
        print(f"\n  ── TOP {limit} RISK-ADJUSTED (EV × WR / (1+MaxL)) ──")
        print(f"  {'Model':<16} Str Conf    N    W    L     WR       PnL      EV  RiskSc MaxL")
        print("  " + "-" * 86)
        for r, sc in risk_sc[:limit]:
            print(f"  {short(r[0]):<16} {r[1]}x >={int(r[2]*100):2d}%  {r[3]:3d}  {r[4]:3d}  {r[5]:3d}  {r[6]:5.1f}%  ${r[7]:>+7.1f}  ${r[8]:>+5.2f}  {sc:5.2f}  {r[9]:3d}")

    return [r for r, _ in scored[:5]] if scored else []


# ══════════════════════════════════════════════════════════════════
#  SECTION 1: EXHAUSTIVE PRICE RANGES
# ══════════════════════════════════════════════════════════════════

print(f"\n{'#'*W}")
print(f"  SECTION 1: EXHAUSTIVE PRICE RANGES (streaks 1-5, conf 55%-95%)")
print(f"{'#'*W}")

price_ranges = [
    (0.08, 0.55, 5, "FULL RANGE (current config)"),
    (0.08, 0.50, 5, ""), (0.08, 0.45, 5, ""), (0.08, 0.40, 5, ""),
    (0.08, 0.35, 3, ""), (0.08, 0.30, 3, "CHEAP SHARES"),
    (0.08, 0.25, 3, ""), (0.08, 0.20, 3, "ULTRA CHEAP"),
    (0.08, 0.15, 3, ""), (0.08, 0.12, 3, "PENNY"),
    (0.10, 0.55, 5, "prev config"), (0.10, 0.30, 3, ""),
    (0.15, 0.55, 5, ""), (0.15, 0.40, 3, ""),
    (0.20, 0.55, 5, ""), (0.20, 0.40, 5, ""),
    (0.30, 0.55, 5, "EXPENSIVE ONLY"), (0.35, 0.55, 5, ""), (0.40, 0.55, 5, ""),
]

for sp_lo, sp_hi, mn, label in price_ranges:
    tag = f"  ← {label}" if label else ""
    print(f"\n{SEP}")
    print(f"  SP ${sp_lo:.2f}–${sp_hi:.2f}{tag}")
    print(SEP)
    r = sim_fast(sp_lo, sp_hi)
    print_top(r, min_n=mn, show_risk=True)


# ══════════════════════════════════════════════════════════════════
#  SECTION 2: ENTRY TIMING WINDOWS
# ══════════════════════════════════════════════════════════════════

print(f"\n\n{'#'*W}")
print(f"  SECTION 2: ENTRY TIMING WINDOWS (SP $0.08-$0.55)")
print(f"{'#'*W}")

for e_lo, e_hi, label in [
    (0.0, 0.50, "EARLY HALF"), (0.0, 0.30, "FIRST 30%"),
    (0.30, 0.70, "MIDDLE"), (0.50, 0.95, "LATE HALF"),
    (0.70, 0.95, "LAST 25%"), (0.0, 0.95, "FULL"),
    (0.0, 0.80, "NO LATE"), (0.20, 0.80, "CORE"),
]:
    print(f"\n{SEP}")
    print(f"  ENTRY {int(e_lo*100)}%-{int(e_hi*100)}% ({label})")
    print(SEP)
    r = sim_fast(0.08, 0.55, entry_lo=e_lo, entry_hi=e_hi)
    print_top(r, min_n=5, limit=10)


# ══════════════════════════════════════════════════════════════════
#  SECTION 3: SPREAD ANALYSIS
# ══════════════════════════════════════════════════════════════════

print(f"\n\n{'#'*W}")
print(f"  SECTION 3: SPREAD TIGHTNESS (SP $0.08-$0.55)")
print(f"{'#'*W}")

BEST4 = ["hermes_catboost", "hermes_xgboost", "binance_catboost", "hermes_ensemble"]
for max_sp in [0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15]:
    print(f"\n  ── Max spread ≤ ${max_sp:.2f} ──")
    r = sim_fast(0.08, 0.55, max_spread=max_sp, models_list=BEST4,
                 streaks=[2, 3], thresholds=[0.75, 0.80, 0.85, 0.90])
    if r:
        hdr()
        for rr in sorted(r, key=lambda x: (-x[6], -x[8]))[:8]: row(rr)
    else:
        print("    No results")


# ══════════════════════════════════════════════════════════════════
#  SECTION 4: UP vs DOWN
# ══════════════════════════════════════════════════════════════════

print(f"\n\n{'#'*W}")
print(f"  SECTION 4: UP vs DOWN DIRECTION (SP $0.08-$0.55)")
print(f"{'#'*W}")

for model, streak, th, label in [
    ("hermes_catboost", 3, 0.85, "H_cat 3x≥85%"),
    ("hermes_catboost", 3, 0.80, "H_cat 3x≥80%"),
    ("hermes_xgboost", 3, 0.80, "H_xgb 3x≥80%"),
    ("hermes_xgboost", 2, 0.60, "H_xgb 2x≥60%"),
    ("hermes_ensemble", 3, 0.85, "H_ens 3x≥85%"),
    ("binance_catboost", 3, 0.80, "B_cat 3x≥80%"),
]:
    r = sim_fast(0.08, 0.55, streaks=[streak], thresholds=[th], models_list=[model])
    if not r: continue
    details = r[0][13]
    print(f"\n  ── {label} ──")
    for dn, trades in [("UP", [t for t in details if t[1]=="UP"]),
                        ("DOWN", [t for t in details if t[1]=="DOWN"])]:
        if trades:
            w = sum(1 for _,_,_,won,_ in trades if won)
            n = len(trades); pnl = sum(p for _,_,_,_,p in trades)
            avg_sp = sum(sp for _,_,sp,_,_ in trades) / n
            print(f"    {dn:>5}: N={n:3d} W={w:3d} L={n-w:3d} WR={w/n*100:5.1f}% PnL=${pnl:+.1f} EV=${pnl/n:+.2f} avg_SP=${avg_sp:.3f}")
        else:
            print(f"    {dn:>5}: 0 trades")


# ══════════════════════════════════════════════════════════════════
#  SECTION 5: CHEAP SHARES DEEP DIVE
# ══════════════════════════════════════════════════════════════════

print(f"\n\n{'#'*W}")
print(f"  SECTION 5: CHEAP SHARES DEEP DIVE (SP $0.08-$0.30)")
print(f"  1 win at SP $0.10 → {BET/0.10:.0f} shares × $0.90 = ${BET/0.10*0.90:.0f} profit")
print(f"  1 win at SP $0.20 → {BET/0.20:.0f} shares × $0.80 = ${BET/0.20*0.80:.0f} profit")
print(f"{'#'*W}")

r = sim_fast(0.08, 0.30, streaks=[1, 2, 3, 4, 5],
             thresholds=[0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
print_top(r, min_n=3, limit=20, show_risk=True)

for lo, hi in [(0.08, 0.15), (0.08, 0.20), (0.10, 0.20), (0.10, 0.25), (0.15, 0.30)]:
    print(f"\n  ── Sub-range: ${lo:.2f}-${hi:.2f} ──")
    r = sim_fast(lo, hi, streaks=[1, 2, 3, 4, 5],
                 thresholds=[0.55, 0.60, 0.65, 0.70, 0.75, 0.80])
    if r:
        hdr()
        for rr in sorted(r, key=lambda x: -x[8])[:10]: row(rr)
    else:
        print("    No results")


# ══════════════════════════════════════════════════════════════════
#  SECTION 6: STREAK LENGTH COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════

print(f"\n\n{'#'*W}")
print(f"  SECTION 6: STREAK LENGTH COMPARISON (which streak is optimal?)")
print(f"{'#'*W}")

for sp_lo, sp_hi, label in [(0.08, 0.55, "FULL"), (0.08, 0.30, "CHEAP"), (0.30, 0.55, "EXPENSIVE")]:
    for th_val in [0.80, 0.85]:
        print(f"\n  ── {label} SP ${sp_lo:.2f}-${sp_hi:.2f} @ {int(th_val*100)}% ──")
        print(f"  {'Model':<16} {'1x':>14} {'2x':>14} {'3x':>14} {'4x':>14} {'5x':>14}")
        print("  " + "-" * 86)
        for model in BEST4:
            vals = []
            for s in [1, 2, 3, 4, 5]:
                r = sim_fast(sp_lo, sp_hi, streaks=[s], thresholds=[th_val], models_list=[model])
                if r:
                    n, wr, ev = r[0][3], r[0][6], r[0][8]
                    vals.append(f"{n:3d}t {wr:3.0f}% ${ev:+.1f}")
                else:
                    vals.append("    — n/a —  ")
            print(f"  {short(model):<16} {'  '.join(vals)}")


# ══════════════════════════════════════════════════════════════════
#  SECTION 7: CONFIDENCE HEAD-TO-HEAD
# ══════════════════════════════════════════════════════════════════

print(f"\n\n{'#'*W}")
print(f"  SECTION 7: CONFIDENCE HEAD-TO-HEAD")
print(f"{'#'*W}")

for sp_lo, sp_hi in [(0.08, 0.55), (0.08, 0.30)]:
    for model in ["hermes_catboost", "hermes_xgboost"]:
        print(f"\n  ── {short(model)} | SP ${sp_lo:.2f}-${sp_hi:.2f} | 3x streak ──")
        print(f"  {'Conf':>6}   N    W    L     WR       PnL      EV   AvgWin AvgLoss  PF  MaxL")
        print("  " + "-" * 82)
        for th in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
            r = sim_fast(sp_lo, sp_hi, streaks=[3], thresholds=[th], models_list=[model])
            if r and r[0][3] >= 1:
                rr = r[0]
                print(f"  >={int(th*100):2d}%   {rr[3]:3d}  {rr[4]:3d}  {rr[5]:3d}  {rr[6]:5.1f}%  ${rr[7]:>+7.1f}  ${rr[8]:>+5.2f}  ${rr[11]:>+5.2f}  ${rr[12]:>+5.2f}  {rr[10]:4.1f}  {rr[9]:3d}")
            else:
                print(f"  >={int(th*100):2d}%    —")


# ══════════════════════════════════════════════════════════════════
#  SECTION 8: COMBINED MULTI-SLOT STRATEGIES
# ══════════════════════════════════════════════════════════════════

print(f"\n\n{'#'*W}")
print(f"  SECTION 8: COMBINED STRATEGIES (2 configs, non-overlapping markets)")
print(f"{'#'*W}")

def sim_combined(cfgs):
    all_trades = []; claimed = set()
    for model, streak, th, sp_lo, sp_hi, label in cfgs:
        r = sim_fast(sp_lo, sp_hi, streaks=[streak], thresholds=[th], models_list=[model])
        if not r: continue
        for slug, d, sp, won, pnl in r[0][13]:
            if slug not in claimed:
                claimed.add(slug)
                all_trades.append((slug, d, sp, won, pnl, label))
    if not all_trades: return None
    w = sum(1 for _,_,_,won,_,_ in all_trades if won)
    n = len(all_trades)
    pnl = sum(p for _,_,_,_,p,_ in all_trades)
    return n, w, n-w, w/n*100, pnl, pnl/n

combos = [
    ("A: H_cat 3x≥85% full",           [("hermes_catboost", 3, 0.85, 0.08, 0.55, "main")]),
    ("B: H_cat 3x≥80% full",           [("hermes_catboost", 3, 0.80, 0.08, 0.55, "main")]),
    ("C: H_cat 3x≥85% + H_xgb 2x≥60% cheap", [("hermes_catboost", 3, 0.85, 0.20, 0.55, "norm"), ("hermes_xgboost", 2, 0.60, 0.08, 0.20, "cheap")]),
    ("D: H_cat 3x≥85% + H_xgb 3x≥60% cheap", [("hermes_catboost", 3, 0.85, 0.20, 0.55, "norm"), ("hermes_xgboost", 3, 0.60, 0.08, 0.20, "cheap")]),
    ("E: H_cat 3x≥80% + H_xgb 2x≥60% cheap", [("hermes_catboost", 3, 0.80, 0.20, 0.55, "norm"), ("hermes_xgboost", 2, 0.60, 0.08, 0.20, "cheap")]),
    ("F: H_cat 3x≥85% + H_xgb 2x≥55% penny", [("hermes_catboost", 3, 0.85, 0.15, 0.55, "mid"), ("hermes_xgboost", 2, 0.55, 0.08, 0.15, "penny")]),
    ("G: B_cat 3x≥80% full",           [("binance_catboost", 3, 0.80, 0.08, 0.55, "main")]),
    ("H: H_ens 3x≥85% full",           [("hermes_ensemble", 3, 0.85, 0.08, 0.55, "main")]),
    ("I: H_cat 3x≥85% + B_cat 2x≥70% cheap",  [("hermes_catboost", 3, 0.85, 0.20, 0.55, "norm"), ("binance_catboost", 2, 0.70, 0.08, 0.20, "cheap")]),
    ("J: H_xgb 2x≥60% cheap only",     [("hermes_xgboost", 2, 0.60, 0.08, 0.30, "cheap")]),
    ("K: H_cat 3x≥85% + H_cat 2x≥80% cheap",  [("hermes_catboost", 3, 0.85, 0.20, 0.55, "norm"), ("hermes_catboost", 2, 0.80, 0.08, 0.20, "cheap")]),
]

print(f"\n  {'Strategy':<52} {'N':>4} {'W':>4} {'L':>4} {'WR':>6} {'PnL':>8} {'EV':>7}")
print("  " + "-" * 86)
for name, cfgs in combos:
    result = sim_combined(cfgs)
    if result:
        n, w, l, wr, pnl, ev = result
        print(f"  {name:<52} {n:4d} {w:4d} {l:4d} {wr:5.1f}% ${pnl:>+7.1f} ${ev:>+5.2f}")
    else:
        print(f"  {name:<52} — no trades")


# ══════════════════════════════════════════════════════════════════
#  SECTION 9: PER-TRADE DETAILS
# ══════════════════════════════════════════════════════════════════

print(f"\n\n{'#'*W}")
print(f"  SECTION 9: PER-TRADE DETAILS (top configs)")
print(f"{'#'*W}")

for model, streak, th, sp_lo, sp_hi in [
    ("hermes_catboost", 3, 0.85, 0.08, 0.55),
    ("hermes_catboost", 3, 0.80, 0.08, 0.55),
    ("hermes_xgboost", 2, 0.60, 0.08, 0.30),
    ("hermes_xgboost", 3, 0.60, 0.08, 0.30),
    ("hermes_ensemble", 3, 0.85, 0.08, 0.55),
]:
    r = sim_fast(sp_lo, sp_hi, streaks=[streak], thresholds=[th], models_list=[model])
    if not r: continue
    rr = r[0]; details = rr[13]
    print(f"\n  ── {short(model)} {streak}x ≥{int(th*100)}% SP ${sp_lo:.2f}-${sp_hi:.2f} ──")
    print(f"  N={rr[3]} W={rr[4]} L={rr[5]} WR={rr[6]:.1f}% PnL=${rr[7]:+.1f} EV=${rr[8]:+.2f}")
    print(f"  {'#':>3} {'Slug':<40} {'Dir':>4} {'SP':>6} {'W/L':>3} {'PnL':>8} {'Shrs':>5}")
    print("  " + "-" * 75)
    cum = 0
    for i, (slug, d, sp, won, pnl) in enumerate(details):
        cum += pnl
        print(f"  {i+1:3d} {slug[-38:]:<40} {d:>4} ${sp:.3f} {'W' if won else 'L':>3} ${pnl:>+7.2f} {BET/sp:4.1f}  cum=${cum:+.1f}")


# ══════════════════════════════════════════════════════════════════
#  SECTION 10: ABSOLUTE BEST — FINAL RANKING
# ══════════════════════════════════════════════════════════════════

print(f"\n\n{'#'*W}")
print(f"  SECTION 10: ABSOLUTE BEST CONFIGS — FINAL RANKING")
print(f"{'#'*W}")

mega = []
for sp_lo, sp_hi, mn, label in price_ranges:
    for rr in sim_fast(sp_lo, sp_hi):
        mega.append((sp_lo, sp_hi, rr))

def mega_row(lo, hi, r):
    return f"  ${lo:.2f}-${hi:.2f}   {short(r[0]):<16} {r[1]}x >={int(r[2]*100):2d}%  {r[3]:3d}  {r[4]:3d}  {r[5]:3d}  {r[6]:5.1f}%  ${r[7]:>+7.1f}  ${r[8]:>+5.2f}"

valid5 = [(lo, hi, r) for lo, hi, r in mega if r[3] >= 5]

print(f"\n  ── #1 HIGHEST EV (N ≥ 5) ──")
print(f"  {'Range':<14} {'Model':<16} Str Conf    N    W    L     WR       PnL      EV")
print("  " + "-" * 86)
for lo, hi, r in sorted(valid5, key=lambda x: -x[2][8])[:10]:
    print(mega_row(lo, hi, r))

print(f"\n  ── #2 HIGHEST PnL (N ≥ 5) ──")
print(f"  {'Range':<14} {'Model':<16} Str Conf    N    W    L     WR       PnL      EV")
print("  " + "-" * 86)
for lo, hi, r in sorted(valid5, key=lambda x: -x[2][7])[:10]:
    print(mega_row(lo, hi, r))

print(f"\n  ── #3 100% WIN RATE (N ≥ 5) ──")
perfect = [(lo, hi, r) for lo, hi, r in mega if r[3] >= 5 and r[5] == 0]
if perfect:
    print(f"  {'Range':<14} {'Model':<16} Str Conf    N    W    L     WR       PnL      EV")
    print("  " + "-" * 86)
    for lo, hi, r in sorted(perfect, key=lambda x: -x[2][7])[:15]:
        print(mega_row(lo, hi, r))

print(f"\n  ── #4 SWEET SPOT (top 15 across ALL ranges) ──")
scored = [(lo, hi, r, r[8]*math.sqrt(r[3])*r[6]/100) for lo, hi, r in mega if r[3] >= 5 and r[8] > 0]
scored.sort(key=lambda x: -x[3])
if scored:
    print(f"  {'Range':<14} {'Model':<16} Str Conf    N    W    L     WR       PnL      EV  Score")
    print("  " + "-" * 92)
    for lo, hi, r, sc in scored[:15]:
        print(f"  {mega_row(lo, hi, r)}  {sc:5.1f}")


# ══════════════════════════════════════════════════════════════════
#  SECTION 11: THREE STREAK MODES — HEAD-TO-HEAD
# ══════════════════════════════════════════════════════════════════

print(f"\n\n{'#'*W}")
print(f"  SECTION 11: STREAK MODE COMPARISON (STRICT / SKIP / SOFT)")
print(f"{'#'*W}")
print(f"""
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │  STRICT: Bad price/spread → RESET streak to 0.                             │
  │          Matches live trader _update_streak() behavior exactly.             │
  │          Streak builds only when ALL conditions met simultaneously.         │
  │                                                                            │
  │  SKIP:   Bad price/spread → SKIP tick (streak FROZEN, no build, no reset). │
  │          Streak builds only on ticks where conf+price+spread all OK.        │
  │          But gaps in price don't break accumulated streak.                  │
  │          → More trades than STRICT, same quality per-tick.                  │
  │                                                                            │
  │  SOFT:   Streak builds on CONFIDENCE+DIRECTION only (ignores price).       │
  │          Entry happens when streak met AND price+spread OK on that tick.    │
  │          → Most trades. Streak = pure model conviction signal.             │
  └──────────────────────────────────────────────────────────────────────────────┘
""")

COMPARE_CONFIGS = [
    ("binance_catboost",  3, 0.80, 0.08, 0.55, "B_cat 3x>=80%"),
    ("binance_catboost",  2, 0.85, 0.08, 0.55, "B_cat 2x>=85%"),
    ("binance_catboost",  2, 0.80, 0.08, 0.55, "B_cat 2x>=80%"),
    ("binance_catboost",  3, 0.85, 0.08, 0.55, "B_cat 3x>=85%"),
    ("hermes_catboost",   3, 0.80, 0.08, 0.55, "H_cat 3x>=80%"),
    ("hermes_catboost",   3, 0.85, 0.08, 0.55, "H_cat 3x>=85%"),
    ("hermes_catboost",   2, 0.80, 0.08, 0.55, "H_cat 2x>=80%"),
    ("hermes_catboost",   2, 0.85, 0.08, 0.55, "H_cat 2x>=85%"),
    ("hermes_xgboost",    2, 0.60, 0.08, 0.30, "H_xgb 2x>=60% $0.30"),
    ("hermes_ensemble",   3, 0.85, 0.08, 0.55, "H_ens 3x>=85%"),
]

def _mode_str(r):
    if not r: return "    --    --      --      --"
    return f"{r[0][3]:4d} {r[0][6]:5.1f}% ${r[0][7]:>+7.1f} ${r[0][8]:>+5.2f}"

print(f"  {'Config':<24}  {'---- STRICT ----':>27}  {'---- SKIP -----':>27}  {'---- SOFT -----':>27}")
print(f"  {'':24}  {'N':>4} {'WR':>6} {'PnL':>8} {'EV':>7}  {'N':>4} {'WR':>6} {'PnL':>8} {'EV':>7}  {'N':>4} {'WR':>6} {'PnL':>8} {'EV':>7}")
print("  " + "-" * 108)

for model, streak, th, sp_lo, sp_hi, label in COMPARE_CONFIGS:
    kw = dict(streaks=[streak], thresholds=[th], models_list=[model])
    r_strict = sim_fast(sp_lo, sp_hi, **kw, streak_mode="strict")
    r_skip   = sim_fast(sp_lo, sp_hi, **kw, streak_mode="skip")
    r_soft   = sim_fast(sp_lo, sp_hi, **kw, streak_mode="soft")
    print(f"  {label:<24}  {_mode_str(r_strict)}  {_mode_str(r_skip)}  {_mode_str(r_soft)}")

# Also compare with tighter entry limits
print(f"\n  {'--- Entry <= $0.40 ---':^108}")
print(f"  {'Config':<24}  {'---- STRICT ----':>27}  {'---- SKIP -----':>27}  {'---- SOFT -----':>27}")
print(f"  {'':24}  {'N':>4} {'WR':>6} {'PnL':>8} {'EV':>7}  {'N':>4} {'WR':>6} {'PnL':>8} {'EV':>7}  {'N':>4} {'WR':>6} {'PnL':>8} {'EV':>7}")
print("  " + "-" * 108)

for model, streak, th, _, _, label in COMPARE_CONFIGS:
    kw = dict(streaks=[streak], thresholds=[th], models_list=[model])
    r_strict = sim_fast(0.08, 0.40, **kw, streak_mode="strict")
    r_skip   = sim_fast(0.08, 0.40, **kw, streak_mode="skip")
    r_soft   = sim_fast(0.08, 0.40, **kw, streak_mode="soft")
    print(f"  {label:<24}  {_mode_str(r_strict)}  {_mode_str(r_skip)}  {_mode_str(r_soft)}")

print(f"\n  {'--- Entry <= $0.30 ---':^108}")
print(f"  {'Config':<24}  {'---- STRICT ----':>27}  {'---- SKIP -----':>27}  {'---- SOFT -----':>27}")
print(f"  {'':24}  {'N':>4} {'WR':>6} {'PnL':>8} {'EV':>7}  {'N':>4} {'WR':>6} {'PnL':>8} {'EV':>7}  {'N':>4} {'WR':>6} {'PnL':>8} {'EV':>7}")
print("  " + "-" * 108)

for model, streak, th, _, _, label in COMPARE_CONFIGS:
    kw = dict(streaks=[streak], thresholds=[th], models_list=[model])
    r_strict = sim_fast(0.08, 0.30, **kw, streak_mode="strict")
    r_skip   = sim_fast(0.08, 0.30, **kw, streak_mode="skip")
    r_soft   = sim_fast(0.08, 0.30, **kw, streak_mode="soft")
    print(f"  {label:<24}  {_mode_str(r_strict)}  {_mode_str(r_skip)}  {_mode_str(r_soft)}")


# ══════════════════════════════════════════════════════════════════
#  SECTION 12: SKIP MODE — EXHAUSTIVE SEARCH
# ══════════════════════════════════════════════════════════════════

print(f"\n\n{'#'*W}")
print(f"  SECTION 12: SKIP MODE -- EXHAUSTIVE SEARCH")
print(f"  Streak builds when conf+price+spread OK. Bad-price ticks frozen (no reset).")
print(f"{'#'*W}")

alt_ranges = [
    (0.08, 0.55, 5, "FULL"),
    (0.08, 0.40, 5, "<=0.40"),
    (0.08, 0.30, 3, "<=0.30"),
    (0.10, 0.55, 5, ""),
    (0.10, 0.40, 5, ""),
    (0.10, 0.30, 3, ""),
]

for sp_lo, sp_hi, mn, label in alt_ranges:
    tag = f"  ({label})" if label else ""
    print(f"\n{SEP}")
    print(f"  SKIP | SP ${sp_lo:.2f}-${sp_hi:.2f}{tag}")
    print(SEP)
    r = sim_fast(sp_lo, sp_hi, streak_mode="skip")
    print_top(r, min_n=mn, show_risk=True)


# ══════════════════════════════════════════════════════════════════
#  SECTION 13: SOFT MODE — EXHAUSTIVE SEARCH
# ══════════════════════════════════════════════════════════════════

print(f"\n\n{'#'*W}")
print(f"  SECTION 13: SOFT MODE -- EXHAUSTIVE SEARCH")
print(f"  Streak builds on conf+direction only. Price checked only for entry.")
print(f"{'#'*W}")

soft_ranges = [
    (0.08, 0.55, 5, "FULL"),
    (0.08, 0.40, 5, "<=0.40"),
    (0.08, 0.30, 3, "<=0.30"),
    (0.10, 0.55, 5, ""),
    (0.10, 0.40, 5, ""),
    (0.10, 0.30, 3, ""),
]

for sp_lo, sp_hi, mn, label in soft_ranges:
    tag = f"  ({label})" if label else ""
    print(f"\n{SEP}")
    print(f"  SOFT | SP ${sp_lo:.2f}-${sp_hi:.2f}{tag}")
    print(SEP)
    r = sim_fast(sp_lo, sp_hi, streak_mode="soft")
    print_top(r, min_n=mn, show_risk=True)


# ══════════════════════════════════════════════════════════════════
#  SECTION 14: STREAK LENGTH TABLE (all 3 modes)
# ══════════════════════════════════════════════════════════════════

print(f"\n\n{'#'*W}")
print(f"  SECTION 14: STREAK LENGTH TABLE (STRICT / SKIP / SOFT)")
print(f"{'#'*W}")

_KEY_MODELS = ["hermes_catboost", "binance_catboost", "hermes_xgboost", "hermes_ensemble"]

for mode in ["strict", "skip", "soft"]:
    for sp_lo, sp_hi, label in [(0.08, 0.55, "FULL"), (0.08, 0.40, "<=0.40"), (0.08, 0.30, "<=0.30")]:
        for th_val in [0.80, 0.85]:
            print(f"\n  -- {mode.upper()} {label} SP ${sp_lo:.2f}-${sp_hi:.2f} @ {int(th_val*100)}% --")
            print(f"  {'Model':<16} {'1x':>14} {'2x':>14} {'3x':>14} {'4x':>14} {'5x':>14}")
            print("  " + "-" * 86)
            for model in _KEY_MODELS:
                vals = []
                for s in [1, 2, 3, 4, 5]:
                    r = sim_fast(sp_lo, sp_hi, streaks=[s], thresholds=[th_val],
                                 models_list=[model], streak_mode=mode)
                    if r:
                        n, wr, ev = r[0][3], r[0][6], r[0][8]
                        vals.append(f"{n:3d}t {wr:3.0f}% ${ev:+.1f}")
                    else:
                        vals.append("    -- n/a -- ")
                print(f"  {short(model):<16} {'  '.join(vals)}")


# ══════════════════════════════════════════════════════════════════
#  SECTION 15: PER-TRADE DETAILS (SKIP mode, top configs)
# ══════════════════════════════════════════════════════════════════

print(f"\n\n{'#'*W}")
print(f"  SECTION 15: PER-TRADE DETAILS (SKIP + SOFT)")
print(f"{'#'*W}")

for mode in ["skip", "soft"]:
    for model, streak, th, sp_lo, sp_hi in [
        ("binance_catboost", 3, 0.80, 0.08, 0.55),
        ("binance_catboost", 2, 0.85, 0.08, 0.55),
        ("binance_catboost", 3, 0.80, 0.08, 0.40),
        ("hermes_catboost",  3, 0.80, 0.08, 0.55),
        ("hermes_catboost",  3, 0.85, 0.08, 0.55),
    ]:
        r = sim_fast(sp_lo, sp_hi, streaks=[streak], thresholds=[th],
                     models_list=[model], streak_mode=mode)
        if not r: continue
        rr = r[0]; details = rr[13]
        print(f"\n  -- {mode.upper()} {short(model)} {streak}x >={int(th*100)}% SP ${sp_lo:.2f}-${sp_hi:.2f} --")
        print(f"  N={rr[3]} W={rr[4]} L={rr[5]} WR={rr[6]:.1f}% PnL=${rr[7]:+.1f} EV=${rr[8]:+.2f}")
        print(f"  {'#':>3} {'Slug':<40} {'Dir':>4} {'SP':>6} {'W/L':>3} {'PnL':>8}")
        print("  " + "-" * 70)
        cum = 0
        for i, (slug, d, sp, won, pnl) in enumerate(details):
            cum += pnl
            print(f"  {i+1:3d} {slug[-38:]:<40} {d:>4} ${sp:.3f} {'W' if won else 'L':>3} ${pnl:>+7.2f}  cum=${cum:+.1f}")


# ══════════════════════════════════════════════════════════════════
#  SECTION 16: FINAL RANKING (all modes, all ranges)
# ══════════════════════════════════════════════════════════════════

print(f"\n\n{'#'*W}")
print(f"  SECTION 16: FINAL RANKING -- ALL MODES, ALL RANGES")
print(f"{'#'*W}")

for mode_name, mode_ranges in [("SKIP", alt_ranges), ("SOFT", soft_ranges)]:
    mega_m = []
    for sp_lo, sp_hi, mn, label in mode_ranges:
        for rr in sim_fast(sp_lo, sp_hi, streak_mode=mode_name.lower()):
            mega_m.append((sp_lo, sp_hi, rr))

    valid5 = [(lo, hi, r) for lo, hi, r in mega_m if r[3] >= 5]

    print(f"\n  == {mode_name}: HIGHEST PnL (N >= 5) ==")
    print(f"  {'Range':<14} {'Model':<16} Str Conf    N    W    L     WR       PnL      EV")
    print("  " + "-" * 86)
    for lo, hi, r in sorted(valid5, key=lambda x: -x[2][7])[:15]:
        print(mega_row(lo, hi, r))

    scored_m = [(lo, hi, r, r[8]*math.sqrt(r[3])*r[6]/100) for lo, hi, r in mega_m if r[3] >= 5 and r[8] > 0]
    scored_m.sort(key=lambda x: -x[3])
    if scored_m:
        print(f"\n  == {mode_name}: SWEET SPOT (top 15) ==")
        print(f"  {'Range':<14} {'Model':<16} Str Conf    N    W    L     WR       PnL      EV  Score")
        print("  " + "-" * 92)
        for lo, hi, r, sc in scored_m[:15]:
            print(f"  {mega_row(lo, hi, r)}  {sc:5.1f}")


# ══════════════════════════════════════════════════════════════════
#  SECTION 17: GLOSSARY & RECOMMENDATION
# ══════════════════════════════════════════════════════════════════

print(f"\n\n{'#'*W}")
print(f"  SECTION 17: GLOSSARY & RECOMMENDATION")
print(f"{'#'*W}")
print(f"""
  TERMINOLOGY:
    N      = number of trades (1 per market, first qualifying tick)
    W / L  = wins / losses
    WR     = win rate (W/N * 100%)
    PnL    = total profit/loss in USD
    EV     = expected value per trade (PnL / N)
    SP     = share price at entry ($0.01-$0.99)
    Conf   = model confidence (max(prob, 1-prob))
    Streak = consecutive confident ticks in same direction before entry
    PF     = profit factor (gross profit / gross loss)
    MaxL   = max consecutive losses

  PnL FORMULA:
    Win:  profit = (bet / SP) * (1 - SP)   e.g. SP=$0.40 bet=$2.85 -> 7.13 shares -> $4.28 profit
    Loss: loss = -bet                       e.g. -$2.85

  STREAK MODES:
    STRICT = bad price/spread RESETS streak to 0.
             Matches live trader _update_streak(). Most conservative.
    SKIP   = bad price/spread FREEZES streak (no build, no reset).
             Streak builds only on ticks where ALL conditions met.
             Middle ground: more trades than STRICT, same per-tick quality.
    SOFT   = streak builds on confidence+direction only (ignores price).
             Entry when streak met AND price is OK.
             Most permissive: streak = pure model conviction signal.

  SWEET SPOT SCORE = EV * sqrt(N) * WR%
    Balances trade quality (EV, WR) with quantity (N).
""")

elapsed = time.time() - t0
print(f"\n  Analysis completed in {elapsed:.1f}s")
print(f"  {len(om)} markets, {len(ticks)} ticks, {N_SLUGS} slugs with data")

sys.stdout = _stdout
out_path.write_text(_buf.getvalue(), encoding="utf-8")
print(f"\n✅ Saved to {out_path} ({elapsed:.1f}s)")
