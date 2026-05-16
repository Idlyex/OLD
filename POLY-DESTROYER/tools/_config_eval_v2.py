"""Config eval v2: models 1x/2x/3x + gap_pct filter + TP=$0.90.
Quick focused run for tonight's config.

gap_pct = (sol_now - sol_bar_open) / sol_bar_open * 100
  Positive = SOL went UP from bar open
  Negative = SOL went DOWN from bar open
  For UP prediction: positive gap = momentum FOR us, negative = AGAINST
  For DOWN prediction: negative gap = momentum FOR us, positive = AGAINST
"""
import json, sys, math, time, io
from pathlib import Path
from datetime import datetime
from collections import defaultdict

BET = 2.85
MAX_SPREAD = 0.08

MODELS = [f"{src}_{alg}" for src in ("binance", "hermes")
          for alg in ("catboost", "xgboost", "lgbm", "ensemble")]

SHORT = {
    "binance_catboost": "B_cat", "binance_xgboost": "B_xgb",
    "binance_lgbm": "B_lgbm", "binance_ensemble": "B_ens",
    "hermes_catboost": "H_cat", "hermes_xgboost": "H_xgb",
    "hermes_lgbm": "H_lgbm", "hermes_ensemble": "H_ens",
}

def wilson_ci(wins, total, z=1.96):
    if total == 0: return 0, 0, 0
    p = wins / total
    d = 1 + z**2 / total
    center = (p + z**2 / (2*total)) / d
    spread = z * math.sqrt((p*(1-p) + z**2/(4*total)) / total) / d
    return p, max(0, center - spread), min(1, center + spread)

def load_day(date):
    tp = Path(f"results/ticks/ticks_{date}.jsonl")
    op = Path(f"results/ticks/outcomes_{date}.jsonl")
    if not tp.exists() or not op.exists():
        return {}, {}
    ticks_by_slug = defaultdict(list)
    for line in tp.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        t = json.loads(line)
        s = t.get("slug", "")
        if s: ticks_by_slug[s].append(t)
    outcomes = {}
    for line in op.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        o = json.loads(line)
        s = o.get("slug") or o.get("condition_id", "")
        out = o.get("outcome") or o.get("result")
        if s and out: outcomes[s] = out
    return dict(ticks_by_slug), outcomes

def sim(ticks_by_slug, outcomes, model, conf_th, streak_req, sp_lo, sp_hi,
        tp=1.0, streak_mode="strict", gap_max=None, gap_min=None):
    """gap_max: max abs gap_pct AGAINST predicted direction (positive = filter more).
       gap_min: min abs gap_pct FOR predicted direction (negative = require momentum)."""
    total = wins = 0; pnl = 0.0; trades = []
    for slug, stk in ticks_by_slug.items():
        if slug not in outcomes: continue
        actual = outcomes[slug]
        streak = 0; streak_dir = None; entered = False
        for t in stk:
            if entered: break
            ep = t.get("entry_pct", -1)
            if ep < 0 or ep > 0.95: continue
            v = t.get(model)
            if v is None: continue
            d = "UP" if v > 0.5 else "DOWN"
            c = max(v, 1-v)
            sp_k = "yes_ask" if d == "UP" else "no_ask"
            sp_f = "yes" if d == "UP" else "no"
            sp = t.get(sp_k, t.get(sp_f, 0.5))
            spr = t.get("yes_spread" if d == "UP" else "no_spread", 0)
            price_ok = sp_lo <= sp < sp_hi and 0 < sp <= 1
            spread_ok = spr <= 0 or spr <= MAX_SPREAD

            # Gap: directional gap (positive = price moved FOR predicted direction)
            raw_gap = t.get("gap_pct", 0) or 0
            dir_gap = raw_gap if d == "UP" else -raw_gap

            if streak_mode == "strict":
                if not (price_ok and spread_ok):
                    streak = 0; streak_dir = None; continue
            elif streak_mode == "skip":
                if not (price_ok and spread_ok):
                    continue
            if c >= conf_th:
                if d == streak_dir and streak > 0: streak += 1
                else: streak = 1; streak_dir = d
                if streak >= streak_req:
                    # Gap filter at entry point only (doesn't affect streak)
                    if gap_max is not None and dir_gap < -gap_max:
                        continue  # too much against us, skip entry
                    if gap_min is not None and dir_gap < gap_min:
                        continue  # not enough momentum FOR us
                    entered = True; total += 1
                    won = (streak_dir == actual)
                    shares = BET / sp
                    p = shares * (tp - sp) if won else -BET
                    if won: wins += 1
                    pnl += p
                    trades.append((slug, d, sp, won, p, dir_gap))
            else:
                streak = 0; streak_dir = None
    wr = (wins/total*100) if total else 0
    ev = (pnl/total) if total else 0
    return dict(n=total, w=wins, l=total-wins, wr=wr, pnl=pnl, ev=ev, trades=trades)

def main():
    t0 = time.time()
    dates = []
    if "--dates" in sys.argv:
        idx = sys.argv.index("--dates") + 1
        while idx < len(sys.argv) and not sys.argv[idx].startswith("-"):
            dates.append(sys.argv[idx]); idx += 1
    if not dates:
        files = sorted(Path("results/ticks").glob("ticks_*.jsonl"))
        files = [f for f in files if "binance" not in f.name and "hermes" not in f.name]
        dates = [f.stem.replace("ticks_", "") for f in files[-2:]]

    days = {}
    for d in dates:
        tbs, oc = load_day(d)
        if tbs: days[d] = (tbs, oc)

    merged_tbs = defaultdict(list)
    merged_oc = {}
    for d, (tbs, oc) in days.items():
        for slug, ticks in tbs.items():
            k = f"{d}_{slug}"
            merged_tbs[k] = ticks
            if slug in oc: merged_oc[k] = oc[slug]
    M = dict(merged_tbs)

    out_dir = Path("results/analysis"); out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"config_eval_v2_{'_'.join(dates)}.txt"
    _buf = io.StringIO()
    _stdout = sys.stdout
    class Tee:
        def write(self, s): _stdout.write(s); _buf.write(s)
        def flush(self): _stdout.flush()
    sys.stdout = Tee()

    W = 90
    total_slugs = len(merged_tbs)
    total_outcomes = len(merged_oc)
    print("=" * W)
    print(f"  CONFIG EVAL V2 — Gap Filter + Streak + TP Comparison")
    print(f"  Dates: {', '.join(dates)} ({total_slugs} slugs, {total_outcomes} outcomes)")
    for d in dates:
        tbs, oc = days[d]
        print(f"    {d}: {len(tbs)} slugs, {sum(len(v) for v in tbs.values())} ticks, {len(oc)} outcomes")
    print("=" * W)

    # ═══════════════════════════════════════════════════════════
    #  SECTION 1: ALL MODELS x 1x/2x/3x (no gap filter, TP=$1.00)
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'#'*W}")
    print(f"  SECTION 1: ALL MODELS — 1x / 2x / 3x (strict, no gap filter, TP=$1.00)")
    print(f"  Combined: {', '.join(dates)}")
    print(f"{'#'*W}")

    for sp_lo, sp_hi, sp_label in [(0.08, 0.55, "FULL $0.08-$0.55"), (0.08, 0.40, "<=0.40")]:
        for conf in [0.80, 0.85, 0.90]:
            print(f"\n  --- {sp_label} | >={int(conf*100)}% ---")
            print(f"  {'Model':<10} {'---- 1x (no streak) ----':>24} {'------- 2x streak ------':>24} {'------- 3x streak ------':>24}")
            print(f"  {'':10} {'N':>3} {'WR':>5} {'PnL':>7} {'EV':>6} {'CI':>8}  {'N':>3} {'WR':>5} {'PnL':>7} {'EV':>6} {'CI':>8}  {'N':>3} {'WR':>5} {'PnL':>7} {'EV':>6} {'CI':>8}")
            print("  " + "-" * 88)
            for model in MODELS:
                parts = []
                for s in [1, 2, 3]:
                    r = sim(M, merged_oc, model, conf, s, sp_lo, sp_hi, 1.0, "strict")
                    if r['n']:
                        _, clo, chi = wilson_ci(r['w'], r['n'])
                        parts.append(f"{r['n']:3d} {r['wr']:4.0f}% ${r['pnl']:>+6.1f} ${r['ev']:>+5.2f} [{clo*100:2.0f}-{chi*100:2.0f}]")
                    else:
                        parts.append(f"  -    -       -      -       -")
                print(f"  {SHORT[model]:<10} {'  '.join(parts)}")

    # ═══════════════════════════════════════════════════════════
    #  SECTION 2: SKIP MODE — 1x/2x/3x
    # ═══════════════════════════════════════════════════════════
    print(f"\n\n{'#'*W}")
    print(f"  SECTION 2: ALL MODELS — 1x / 2x / 3x (SKIP mode, no gap filter, TP=$1.00)")
    print(f"{'#'*W}")

    for sp_lo, sp_hi, sp_label in [(0.08, 0.55, "FULL $0.08-$0.55"), (0.08, 0.40, "<=0.40")]:
        for conf in [0.80, 0.85, 0.90]:
            print(f"\n  --- {sp_label} | >={int(conf*100)}% ---")
            print(f"  {'Model':<10} {'---- 1x (no streak) ----':>24} {'------- 2x streak ------':>24} {'------- 3x streak ------':>24}")
            print(f"  {'':10} {'N':>3} {'WR':>5} {'PnL':>7} {'EV':>6} {'CI':>8}  {'N':>3} {'WR':>5} {'PnL':>7} {'EV':>6} {'CI':>8}  {'N':>3} {'WR':>5} {'PnL':>7} {'EV':>6} {'CI':>8}")
            print("  " + "-" * 88)
            for model in MODELS:
                parts = []
                for s in [1, 2, 3]:
                    r = sim(M, merged_oc, model, conf, s, sp_lo, sp_hi, 1.0, "skip")
                    if r['n']:
                        _, clo, chi = wilson_ci(r['w'], r['n'])
                        parts.append(f"{r['n']:3d} {r['wr']:4.0f}% ${r['pnl']:>+6.1f} ${r['ev']:>+5.2f} [{clo*100:2.0f}-{chi*100:2.0f}]")
                    else:
                        parts.append(f"  -    -       -      -       -")
                print(f"  {SHORT[model]:<10} {'  '.join(parts)}")

    # ═══════════════════════════════════════════════════════════
    #  SECTION 3: GAP % FILTER
    # ═══════════════════════════════════════════════════════════
    print(f"\n\n{'#'*W}")
    print(f"  SECTION 3: GAP % FILTER (directional gap = SOL movement FOR predicted direction)")
    print(f"  gap_pct = (sol_now - sol_bar_open) / sol_bar_open * 100")
    print(f"  dir_gap: positive = price moved FOR us, negative = AGAINST us")
    print(f"  gap_max = max allowed move AGAINST us (filter ticks where dir_gap < -gap_max)")
    print(f"{'#'*W}")

    KEY_MODELS = ["binance_catboost", "hermes_catboost", "binance_xgboost",
                  "hermes_xgboost", "binance_lgbm", "hermes_lgbm",
                  "binance_ensemble", "hermes_ensemble"]

    GAP_FILTERS = [
        (None, None, "no filter"),
        (0.10, None, "max 0.10% against"),
        (0.05, None, "max 0.05% against"),
        (0.03, None, "max 0.03% against"),
        (0.02, None, "max 0.02% against"),
        (0.01, None, "max 0.01% against"),
        (0.00, None, "only FOR us (>=0)"),
        (None, 0.02, "req 0.02% FOR us"),
        (None, 0.05, "req 0.05% FOR us"),
        (None, 0.10, "req 0.10% FOR us"),
    ]

    for sp_lo, sp_hi, sp_label in [(0.08, 0.55, "FULL"), (0.08, 0.40, "<=0.40")]:
        for conf in [0.85, 0.90]:
            for streak_req in [1, 2, 3]:
                print(f"\n  === {sp_label} | >={int(conf*100)}% | {streak_req}x | strict ===")
                print(f"  {'Model':<10}", end="")
                for gmax, gmin, glabel in GAP_FILTERS:
                    print(f" {'N':>3}{'WR':>4}{'EV':>6}", end="")
                print()
                print(f"  {'':10}", end="")
                for gmax, gmin, glabel in GAP_FILTERS:
                    tag = glabel[:11]
                    print(f" {tag:>13}", end="")
                print()
                print("  " + "-" * (10 + 13*len(GAP_FILTERS)))
                for model in KEY_MODELS:
                    line = f"  {SHORT[model]:<10}"
                    for gmax, gmin, glabel in GAP_FILTERS:
                        r = sim(M, merged_oc, model, conf, streak_req, sp_lo, sp_hi, 1.0, "strict",
                                gap_max=gmax, gap_min=gmin)
                        if r['n']:
                            line += f" {r['n']:3d}{r['wr']:3.0f}%${r['ev']:>+5.2f}"
                        else:
                            line += f"   -  -     -"
                    print(line)

    # ═══════════════════════════════════════════════════════════
    #  SECTION 4: GAP FILTER — DETAILED (best combos)
    # ═══════════════════════════════════════════════════════════
    print(f"\n\n{'#'*W}")
    print(f"  SECTION 4: GAP FILTER — BEST CONFIGS (combined data, TP=$1.00)")
    print(f"{'#'*W}")

    all_results = []
    for sp_lo, sp_hi, sp_label in [(0.08, 0.55, "FULL"), (0.08, 0.40, "<=0.40")]:
        for conf in [0.80, 0.85, 0.90, 0.95]:
            for streak_req in [1, 2, 3]:
                for mode in ["strict", "skip"]:
                    for gmax, gmin, glabel in GAP_FILTERS:
                        for model in MODELS:
                            r = sim(M, merged_oc, model, conf, streak_req, sp_lo, sp_hi, 1.0,
                                    mode, gap_max=gmax, gap_min=gmin)
                            if r['n'] >= 5 and r['ev'] > 0:
                                _, clo, _ = wilson_ci(r['w'], r['n'])
                                # Cross-day check
                                cross_ok = True
                                for d in dates:
                                    if d not in days: continue
                                    tbs, oc = days[d]
                                    rd = sim(tbs, oc, model, conf, streak_req, sp_lo, sp_hi, 1.0,
                                             mode, gap_max=gmax, gap_min=gmin)
                                    if rd['n'] < 1 or rd['pnl'] <= 0:
                                        cross_ok = False
                                all_results.append((r, model, conf, streak_req, mode,
                                                   sp_lo, sp_hi, gmax, gmin, glabel, clo, cross_ok))

    # Sort by EV, show cross-day consistent first
    cross_yes = [x for x in all_results if x[11]]
    cross_yes.sort(key=lambda x: -x[0]['ev'])

    print(f"\n  CROSS-DAY CONSISTENT + EV>0 + N>=5 ({len(cross_yes)} configs)")
    print(f"  {'#':>3} {'Model':<10} {'Str':>2} {'Conf':>4} {'Mode':<6} {'SP':>12} {'Gap':>15} {'N':>3} {'WR':>5} {'PnL':>7} {'EV':>6} {'CI_lo':>5}")
    print("  " + "-" * 95)
    for i, (r, model, conf, streak_req, mode, sp_lo, sp_hi, gmax, gmin, glabel, clo, _) in enumerate(cross_yes[:40]):
        gap_str = glabel[:15]
        print(f"  {i+1:3d} {SHORT[model]:<10} {streak_req:>1}x {int(conf*100):>3}% {mode:<6} ${sp_lo:.2f}-${sp_hi:.2f} {gap_str:<15} {r['n']:3d} {r['wr']:4.0f}% ${r['pnl']:>+6.1f} ${r['ev']:>+5.2f} {clo*100:4.0f}%")

    # ═══════════════════════════════════════════════════════════
    #  SECTION 5: TP=$0.90 comparison for top configs
    # ═══════════════════════════════════════════════════════════
    print(f"\n\n{'#'*W}")
    print(f"  SECTION 5: TP=$1.00 vs TP=$0.90 for top cross-day configs")
    print(f"{'#'*W}")

    print(f"\n  {'#':>3} {'Model':<10} {'Config':<25} {'Gap':>15}  {'N':>3} {'WR':>5} {'EV@1.0':>7} {'EV@0.9':>7} {'dEV':>6}")
    print("  " + "-" * 100)
    for i, (r, model, conf, streak_req, mode, sp_lo, sp_hi, gmax, gmin, glabel, clo, _) in enumerate(cross_yes[:25]):
        r90 = sim(M, merged_oc, model, conf, streak_req, sp_lo, sp_hi, 0.9,
                  mode, gap_max=gmax, gap_min=gmin)
        cfg = f"{streak_req}x >={int(conf*100)}% {mode} ${sp_lo:.2f}-${sp_hi:.2f}"
        dev = r90['ev'] - r['ev']
        print(f"  {i+1:3d} {SHORT[model]:<10} {cfg:<25} {glabel[:15]:<15}  {r['n']:3d} {r['wr']:4.0f}% ${r['ev']:>+6.2f} ${r90['ev']:>+6.02f} ${dev:>+5.2f}")

    # ═══════════════════════════════════════════════════════════
    #  SECTION 6: PER-DAY BREAKDOWN (top 15)
    # ═══════════════════════════════════════════════════════════
    print(f"\n\n{'#'*W}")
    print(f"  SECTION 6: PER-DAY BREAKDOWN (top 15 cross-day configs)")
    print(f"{'#'*W}")

    print(f"\n  {'#':>3} {'Model':<10} {'Config':<22} {'Gap':>13}", end="")
    for d in dates:
        print(f"  {d[-5:]:>14}", end="")
    print(f"  {'COMBINED':>14}")
    print("  " + "-" * (48 + 16*len(dates) + 16))

    for i, (r, model, conf, streak_req, mode, sp_lo, sp_hi, gmax, gmin, glabel, clo, _) in enumerate(cross_yes[:15]):
        cfg = f"{streak_req}x >={int(conf*100)}% {mode}"
        line = f"  {i+1:3d} {SHORT[model]:<10} {cfg:<22} {glabel[:13]:<13}"
        for d in dates:
            if d in days:
                tbs, oc = days[d]
                rd = sim(tbs, oc, model, conf, streak_req, sp_lo, sp_hi, 1.0,
                         mode, gap_max=gmax, gap_min=gmin)
                line += f"  {rd['n']:3d}t {rd['wr']:3.0f}% ${rd['pnl']:>+5.1f}"
            else:
                line += f"    -    -      -"
        line += f"  {r['n']:3d}t {r['wr']:3.0f}% ${r['pnl']:>+5.1f}"
        print(line)

    # ═══════════════════════════════════════════════════════════
    #  SECTION 7: FINAL RECOMMENDATION
    # ═══════════════════════════════════════════════════════════
    print(f"\n\n{'#'*W}")
    print(f"  SECTION 7: TONIGHT'S RECOMMENDATION")
    print(f"{'#'*W}")

    # Current config
    print(f"\n  CURRENT: B_cat 2x >=85% strict $0.08-$0.55 (no gap filter)")
    cur = sim(M, merged_oc, "binance_catboost", 0.85, 2, 0.08, 0.55, 1.0, "strict")
    cur90 = sim(M, merged_oc, "binance_catboost", 0.85, 2, 0.08, 0.55, 0.9, "strict")
    if cur['n']:
        _, clo, chi = wilson_ci(cur['w'], cur['n'])
        print(f"    N={cur['n']} WR={cur['wr']:.0f}% EV=${cur['ev']:+.2f} (TP$0.90: ${cur90['ev']:+.2f}) CI=[{clo*100:.0f}-{chi*100:.0f}%]")
        for d in dates:
            if d in days:
                tbs, oc = days[d]
                rd = sim(tbs, oc, "binance_catboost", 0.85, 2, 0.08, 0.55, 1.0, "strict")
                print(f"      {d}: N={rd['n']} WR={rd['wr']:.0f}% PnL=${rd['pnl']:+.1f}")

    # Best alternative
    if cross_yes:
        best = cross_yes[0]
        r, model, conf, streak_req, mode, sp_lo, sp_hi, gmax, gmin, glabel, clo, _ = best
        r90 = sim(M, merged_oc, model, conf, streak_req, sp_lo, sp_hi, 0.9, mode, gap_max=gmax, gap_min=gmin)
        _, clo2, chi2 = wilson_ci(r['w'], r['n'])
        print(f"\n  BEST BY EV (cross-day): {SHORT[model]} {streak_req}x >={int(conf*100)}% {mode} ${sp_lo:.2f}-${sp_hi:.2f} gap:{glabel}")
        print(f"    N={r['n']} WR={r['wr']:.0f}% EV=${r['ev']:+.2f} (TP$0.90: ${r90['ev']:+.2f}) CI=[{clo2*100:.0f}-{chi2*100:.0f}%]")
        for d in dates:
            if d in days:
                tbs, oc = days[d]
                rd = sim(tbs, oc, model, conf, streak_req, sp_lo, sp_hi, 1.0, mode, gap_max=gmax, gap_min=gmin)
                print(f"      {d}: N={rd['n']} WR={rd['wr']:.0f}% PnL=${rd['pnl']:+.1f}")

    elapsed = time.time() - t0
    print(f"\n\n  Completed in {elapsed:.1f}s")
    sys.stdout = _stdout
    out_path.write_text(_buf.getvalue(), encoding="utf-8")
    print(f"\n✅ Saved to {out_path} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
