"""Rigorous config evaluator — multi-day, streaks, TP levels.
Scientific critical thinking: Wilson CIs, cross-day consistency, multiple-comparison warnings.

Usage:
    python tools/_config_eval.py
    python tools/_config_eval.py --dates 2026-05-11 2026-05-12
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

# ── Wilson score confidence interval ──
def wilson_ci(wins, total, z=1.96):
    if total == 0: return 0, 0, 0
    p = wins / total
    d = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / d
    spread = z * math.sqrt((p * (1-p) + z**2 / (4*total)) / total) / d
    return p, max(0, center - spread), min(1, center + spread)

# ── Load data ──
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

# ── Simulate one config ──
def sim(ticks_by_slug, outcomes, model, conf_th, streak_req, sp_lo, sp_hi,
        tp=1.0, streak_mode="strict"):
    total = wins = 0
    pnl = 0.0
    trades = []
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
            if streak_mode == "strict":
                if not (price_ok and spread_ok):
                    streak = 0; streak_dir = None; continue
            elif streak_mode == "skip":
                if not (price_ok and spread_ok):
                    continue
            elif streak_mode == "soft":
                pass  # check price only for entry below
            if c >= conf_th:
                if d == streak_dir and streak > 0: streak += 1
                else: streak = 1; streak_dir = d
                if streak_mode == "soft" and not (price_ok and spread_ok):
                    continue
                if streak >= streak_req:
                    entered = True; total += 1
                    won = (streak_dir == actual)
                    shares = BET / sp
                    p = shares * (tp - sp) if won else -BET
                    if won: wins += 1
                    pnl += p
                    trades.append((slug, d, sp, won, p))
            else:
                streak = 0; streak_dir = None
    wr = (wins / total * 100) if total else 0
    ev = (pnl / total) if total else 0
    return dict(n=total, w=wins, l=total-wins, wr=wr, pnl=pnl, ev=ev, trades=trades)

# ── Main ──
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

    # Load all days
    days = {}
    for d in dates:
        tbs, oc = load_day(d)
        if tbs:
            days[d] = (tbs, oc)
            print(f"  Loaded {d}: {len(tbs)} slugs, {sum(len(v) for v in tbs.values())} ticks, {len(oc)} outcomes")

    if not days:
        print("ERROR: No data found"); sys.exit(1)

    # Merge all days
    merged_tbs = defaultdict(list)
    merged_oc = {}
    for d, (tbs, oc) in days.items():
        for slug, ticks in tbs.items():
            dated_slug = f"{d}_{slug}"
            merged_tbs[dated_slug] = ticks
            if slug in oc:
                merged_oc[dated_slug] = oc[slug]

    # Output setup
    out_dir = Path("results/analysis"); out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"config_eval_{'_'.join(dates)}.txt"
    _buf = io.StringIO()
    _stdout = sys.stdout
    class Tee:
        def write(self, s): _stdout.write(s); _buf.write(s)
        def flush(self): _stdout.flush()
    sys.stdout = Tee()

    W = 90
    print("=" * W)
    print(f"  CONFIG EVALUATOR — Scientific Critical Thinking")
    print(f"  Dates: {', '.join(dates)}")
    print(f"  Bet: ${BET}")
    print(f"  Generated: {datetime.now():%Y-%m-%d %H:%M}")
    print("=" * W)

    # ═══════════════════════════════════════════════════════════
    # SECTION 1: METHODOLOGY NOTES
    # ═══════════════════════════════════════════════════════════
    print(f"""
{"#" * W}
  SECTION 1: METHODOLOGY & WARNINGS
{"#" * W}

  SIMULATION:
    1 trade per market (first qualifying tick). Resolution-based outcome.
    TP=$1.00 = hold to resolution (shares pay $1 if correct, $0 if wrong).
    TP=$0.90 = sell at $0.90 per share (lower profit, same WR assumption).
       NOTE: TP=$0.90 is CONSERVATIVE — some losers might hit $0.90 before
       resolving wrong, but we can't model that without price trajectories.

  STATISTICAL WARNINGS:
    - Small sample sizes (N<20): Wilson 95% CI is WIDE. E.g. 10t 80% WR
      has 95% CI [49%-96%]. Don't over-trust point estimates.
    - Multiple comparisons: testing {len(MODELS)}x models * 4 conf * 3 streaks
      * 3 price ranges = {len(MODELS)*4*3*3} configs. Some will look good by CHANCE.
    - Cross-day consistency is the strongest signal against overfitting.
    - Configs that are positive on ALL days are more trustworthy.

  STREAK MODES:
    STRICT = bad price resets streak. Matches live trader.
    SKIP   = bad price freezes streak (no build, no reset). More trades.
""")

    # ═══════════════════════════════════════════════════════════
    # SECTION 2: PER-MODEL PERFORMANCE (no streak, TP=$1.00)
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'#' * W}")
    print(f"  SECTION 2: PER-MODEL BASELINE (1x = no streak, STRICT, TP=$1.00)")
    print(f"{'#' * W}")

    CONF_LEVELS = [0.80, 0.85, 0.90, 0.95]
    SP_RANGES = [(0.08, 0.55, "FULL"), (0.08, 0.40, "<=0.40"), (0.08, 0.30, "<=0.30")]

    for sp_lo, sp_hi, sp_label in SP_RANGES:
        print(f"\n  === SP ${sp_lo:.2f}-${sp_hi:.2f} ({sp_label}) ===")
        for conf in CONF_LEVELS:
            print(f"\n  -- Conf >= {int(conf*100)}% --")
            print(f"  {'Model':<10}", end="")
            for d in dates:
                print(f"  {'N':>3} {'WR':>5} {'PnL':>7} {'EV':>6}", end="")
            print(f"  {'N':>3} {'WR':>5} {'95%CI':>11} {'PnL':>7} {'EV':>6}  COMBINED")
            print("  " + "-" * (10 + len(dates)*24 + 42))
            for model in MODELS:
                line = f"  {SHORT[model]:<10}"
                per_day_ok = []
                for d in dates:
                    if d not in days: line += "  " + " "*22; continue
                    tbs, oc = days[d]
                    r = sim(tbs, oc, model, conf, 1, sp_lo, sp_hi, 1.0, "strict")
                    if r['n']:
                        line += f"  {r['n']:3d} {r['wr']:4.0f}% ${r['pnl']:>+6.1f} ${r['ev']:>+5.2f}"
                        per_day_ok.append(r['pnl'] > 0)
                    else:
                        line += f"    -    -       -      -"
                        per_day_ok.append(None)
                # Combined
                rc = sim(dict(merged_tbs), merged_oc, model, conf, 1, sp_lo, sp_hi, 1.0, "strict")
                if rc['n']:
                    _, ci_lo, ci_hi = wilson_ci(rc['w'], rc['n'])
                    consistent = all(x == True for x in per_day_ok if x is not None)
                    flag = " **" if consistent and len([x for x in per_day_ok if x is not None]) > 1 else ""
                    line += f"  {rc['n']:3d} {rc['wr']:4.0f}% [{ci_lo*100:4.0f}-{ci_hi*100:3.0f}%] ${rc['pnl']:>+6.1f} ${rc['ev']:>+5.2f}{flag}"
                else:
                    line += f"    -    -           - -"
                print(line)

    # ═══════════════════════════════════════════════════════════
    # SECTION 3: STREAK COMPARISON (STRICT + SKIP, TP=$1.00)
    # ═══════════════════════════════════════════════════════════
    print(f"\n\n{'#' * W}")
    print(f"  SECTION 3: STREAK COMPARISON (TP=$1.00)")
    print(f"  Combined data: {', '.join(dates)}")
    print(f"{'#' * W}")

    STREAKS = [1, 2, 3]
    MODES = ["strict", "skip"]

    for sp_lo, sp_hi, sp_label in SP_RANGES:
        for conf in [0.80, 0.85, 0.90]:
            print(f"\n  === SP ${sp_lo:.2f}-${sp_hi:.2f} | Conf >= {int(conf*100)}% ===")
            print(f"  {'Model':<10} {'':8}", end="")
            for s in STREAKS:
                print(f"  {'N':>3} {'WR':>5} {'PnL':>7} {'EV':>6}", end="")
            print()
            print(f"  {'':10} {'Mode':8}", end="")
            for s in STREAKS:
                print(f"  {'--- '+str(s)+'x ---':>22}", end="")
            print()
            print("  " + "-" * (18 + len(STREAKS)*24))
            for model in MODELS:
                for mode in MODES:
                    line = f"  {SHORT[model]:<10} {mode:<8}"
                    for s in STREAKS:
                        r = sim(dict(merged_tbs), merged_oc, model, conf, s, sp_lo, sp_hi, 1.0, mode)
                        if r['n']:
                            line += f"  {r['n']:3d} {r['wr']:4.0f}% ${r['pnl']:>+6.1f} ${r['ev']:>+5.2f}"
                        else:
                            line += f"    -    -       -      -"
                    print(line)

    # ═══════════════════════════════════════════════════════════
    # SECTION 4: TP=$0.90 vs TP=$1.00
    # ═══════════════════════════════════════════════════════════
    print(f"\n\n{'#' * W}")
    print(f"  SECTION 4: TP=$0.90 vs TP=$1.00 (combined data)")
    print(f"  TP=$0.90: wins pay (0.90-SP) per share. Losses unchanged (-bet).")
    print(f"  Conservative: ignores cases where losers hit $0.90 before resolving wrong.")
    print(f"{'#' * W}")

    for sp_lo, sp_hi, sp_label in SP_RANGES:
        for conf in [0.80, 0.85, 0.90]:
            print(f"\n  === SP ${sp_lo:.2f}-${sp_hi:.2f} | Conf >= {int(conf*100)}% ===")
            print(f"  {'Model':<10} {'Str':>3} {'Mode':<7}", end="")
            print(f"  {'--- TP $1.00 ---':>22}  {'--- TP $0.90 ---':>22}  {'dPnL':>6}")
            print("  " + "-" * 74)
            for model in ["binance_catboost", "hermes_catboost", "binance_xgboost",
                          "hermes_xgboost", "hermes_ensemble", "binance_lgbm"]:
                for s in [1, 2, 3]:
                    for mode in ["strict", "skip"]:
                        r100 = sim(dict(merged_tbs), merged_oc, model, conf, s, sp_lo, sp_hi, 1.0, mode)
                        r90  = sim(dict(merged_tbs), merged_oc, model, conf, s, sp_lo, sp_hi, 0.9, mode)
                        if not r100['n'] or r100['n'] < 3: continue
                        dp = r90['pnl'] - r100['pnl']
                        print(f"  {SHORT[model]:<10} {s:>1}x  {mode:<7}"
                              f"  {r100['n']:3d} {r100['wr']:4.0f}% ${r100['pnl']:>+6.1f} ${r100['ev']:>+5.2f}"
                              f"  {r90['n']:3d} {r90['wr']:4.0f}% ${r90['pnl']:>+6.1f} ${r90['ev']:>+5.2f}"
                              f"  ${dp:>+5.1f}")

    # ═══════════════════════════════════════════════════════════
    # SECTION 5: CROSS-DAY CONSISTENCY (KEY!)
    # ═══════════════════════════════════════════════════════════
    print(f"\n\n{'#' * W}")
    print(f"  SECTION 5: CROSS-DAY CONSISTENCY (most important for config selection)")
    print(f"  Only shows configs POSITIVE on EVERY day individually.")
    print(f"{'#' * W}")

    if len(days) > 1:
        consistent = []
        for sp_lo, sp_hi, sp_label in SP_RANGES:
            for conf in CONF_LEVELS:
                for s in [1, 2, 3]:
                    for mode in ["strict", "skip"]:
                        for model in MODELS:
                            day_results = []
                            all_positive = True
                            for d in dates:
                                if d not in days: continue
                                tbs, oc = days[d]
                                r = sim(tbs, oc, model, conf, s, sp_lo, sp_hi, 1.0, mode)
                                day_results.append(r)
                                if r['n'] < 1 or r['pnl'] <= 0:
                                    all_positive = False
                            if all_positive and all(r['n'] >= 3 for r in day_results):
                                rc = sim(dict(merged_tbs), merged_oc, model, conf, s, sp_lo, sp_hi, 1.0, mode)
                                _, ci_lo, ci_hi = wilson_ci(rc['w'], rc['n'])
                                consistent.append((rc, model, conf, s, mode, sp_lo, sp_hi, sp_label, ci_lo, ci_hi, day_results))

        # Sort by combined EV
        consistent.sort(key=lambda x: -x[0]['ev'])

        print(f"\n  Found {len(consistent)} configs positive on ALL {len(days)} days (N>=3 each day)")

        if consistent:
            # TP=$1.00
            print(f"\n  -- TP=$1.00 (sorted by combined EV) --")
            print(f"  {'Model':<10} {'Str':>3} {'Conf':>4} {'Mode':<6} {'SP':>12}", end="")
            for d in dates:
                print(f"  {d[-5:]:>12}", end="")
            print(f"  {'COMBINED':>12}  {'95%CI':>11}  {'EV':>6}")
            print("  " + "-" * (36 + len(dates)*14 + 36))
            shown = 0
            for rc, model, conf, s, mode, sp_lo, sp_hi, sp_label, ci_lo, ci_hi, day_results in consistent[:30]:
                line = f"  {SHORT[model]:<10} {s:>1}x  {int(conf*100):>3}% {mode:<6} ${sp_lo:.2f}-${sp_hi:.2f}"
                for dr in day_results:
                    line += f"  {dr['n']:2d}t {dr['wr']:3.0f}% ${dr['pnl']:>+5.1f}"
                line += f"  {rc['n']:3d}t {rc['wr']:3.0f}% ${rc['pnl']:>+5.1f}"
                line += f"  [{ci_lo*100:3.0f}-{ci_hi*100:3.0f}%]"
                line += f"  ${rc['ev']:>+5.2f}"
                print(line)
                shown += 1

            # Now show TP=$0.90 for same configs
            print(f"\n  -- Same configs at TP=$0.90 --")
            print(f"  {'Model':<10} {'Str':>3} {'Conf':>4} {'Mode':<6} {'SP':>12}", end="")
            print(f"  {'N':>3} {'WR':>5} {'PnL@1.0':>8} {'PnL@0.9':>8} {'EV@0.9':>7}")
            print("  " + "-" * 70)
            for rc, model, conf, s, mode, sp_lo, sp_hi, sp_label, ci_lo, ci_hi, day_results in consistent[:30]:
                r90 = sim(dict(merged_tbs), merged_oc, model, conf, s, sp_lo, sp_hi, 0.9, mode)
                print(f"  {SHORT[model]:<10} {s:>1}x  {int(conf*100):>3}% {mode:<6} ${sp_lo:.2f}-${sp_hi:.2f}"
                      f"  {rc['n']:3d} {rc['wr']:4.0f}% ${rc['pnl']:>+7.1f} ${r90['pnl']:>+7.1f} ${r90['ev']:>+6.2f}")
    else:
        print("\n  Only 1 day loaded — cross-day consistency not available.")
        print("  WARNING: single-day results are HIGH RISK for overfitting!")

    # ═══════════════════════════════════════════════════════════
    # SECTION 6: RECOMMENDATION
    # ═══════════════════════════════════════════════════════════
    print(f"\n\n{'#' * W}")
    print(f"  SECTION 6: STATISTICAL RIGOR NOTES")
    print(f"{'#' * W}")

    # Find best overall
    best_configs = []
    for sp_lo, sp_hi, sp_label in [(0.08, 0.55, "FULL")]:
        for conf in CONF_LEVELS:
            for s in [1, 2, 3]:
                for mode in ["strict", "skip"]:
                    for model in MODELS:
                        rc = sim(dict(merged_tbs), merged_oc, model, conf, s, sp_lo, sp_hi, 1.0, mode)
                        if rc['n'] >= 5:
                            _, ci_lo, ci_hi = wilson_ci(rc['w'], rc['n'])
                            best_configs.append((rc, model, conf, s, mode, sp_lo, sp_hi, ci_lo, ci_hi))

    best_configs.sort(key=lambda x: -x[0]['ev'])

    print(f"""
  SAMPLE SIZE REALITY CHECK:
    Most configs have N=5-30 trades. At these sizes:
    - 10t 80% WR: true WR could be 49%-96% (95% CI)
    - 20t 75% WR: true WR could be 53%-89%
    - 50t 70% WR: true WR could be 56%-81%
    → Need N>=50 for CI width <20%, N>=100 for <10%.

  MULTIPLE COMPARISONS:
    Testing ~{len(best_configs)} configs. At 5% significance, ~{len(best_configs)//20} will look
    good by random chance alone.
    Mitigation: require cross-day consistency (Section 5).

  OVERFITTING RISK:
    Optimizing config on 2 days of data is inherently noisy.
    The config that maximizes PnL on past data is NOT necessarily
    the best for tomorrow. Prefer conservative configs with:
    1. Positive PnL on BOTH days independently
    2. Reasonable N (not too few trades)
    3. WR 95% CI lower bound > 50%
    4. Makes theoretical sense (not random pattern)
""")

    # ═══════════════════════════════════════════════════════════
    # SECTION 7: TONIGHT'S RECOMMENDATION
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'#' * W}")
    print(f"  SECTION 7: TONIGHT'S CONFIG RECOMMENDATION")
    print(f"{'#' * W}")

    # Score: cross-day consistency + EV + N + CI
    if len(days) > 1 and consistent:
        print(f"\n  TOP 10 CROSS-DAY CONSISTENT CONFIGS (positive PnL each day, N>=3/day):")
        print(f"  {'#':>3} {'Model':<10} {'Config':<20} {'N':>3} {'WR':>5} {'EV@1.0':>7} {'EV@0.9':>7} {'CI_lo':>5}")
        print("  " + "-" * 65)
        for i, (rc, model, conf, s, mode, sp_lo, sp_hi, sp_label, ci_lo, ci_hi, day_results) in enumerate(consistent[:10]):
            r90 = sim(dict(merged_tbs), merged_oc, model, conf, s, sp_lo, sp_hi, 0.9, mode)
            cfg = f"{s}x >={int(conf*100)}% {mode} ${sp_lo:.2f}-${sp_hi:.2f}"
            print(f"  {i+1:3d} {SHORT[model]:<10} {cfg:<20} {rc['n']:3d} {rc['wr']:4.0f}% ${rc['ev']:>+6.2f} ${r90['ev']:>+6.02f} {ci_lo*100:4.0f}%")

        # Current config comparison
        print(f"\n  CURRENT CONFIG (trading.yaml):")
        print(f"    binance_catboost, 2x streak, >=85%, strict, $0.08-$0.55")
        cur = sim(dict(merged_tbs), merged_oc, "binance_catboost", 0.85, 2, 0.08, 0.55, 1.0, "strict")
        cur90 = sim(dict(merged_tbs), merged_oc, "binance_catboost", 0.85, 2, 0.08, 0.55, 0.9, "strict")
        if cur['n']:
            _, cli, chi = wilson_ci(cur['w'], cur['n'])
            print(f"    N={cur['n']} WR={cur['wr']:.0f}% PnL=${cur['pnl']:+.1f} EV=${cur['ev']:+.2f} (TP$0.90: EV=${cur90['ev']:+.2f})")
            print(f"    95% CI: [{cli*100:.0f}-{chi*100:.0f}%]")
            # Per day
            for d in dates:
                if d in days:
                    tbs, oc = days[d]
                    rd = sim(tbs, oc, "binance_catboost", 0.85, 2, 0.08, 0.55, 1.0, "strict")
                    print(f"    {d}: N={rd['n']} WR={rd['wr']:.0f}% PnL=${rd['pnl']:+.1f}")
        else:
            print(f"    N=0 trades (no qualifying entries)")

        # Current with SKIP
        cur_skip = sim(dict(merged_tbs), merged_oc, "binance_catboost", 0.85, 2, 0.08, 0.55, 1.0, "skip")
        if cur_skip['n']:
            _, cli, chi = wilson_ci(cur_skip['w'], cur_skip['n'])
            print(f"\n  SAME BUT SKIP MODE:")
            print(f"    N={cur_skip['n']} WR={cur_skip['wr']:.0f}% PnL=${cur_skip['pnl']:+.1f} EV=${cur_skip['ev']:+.2f}")
            print(f"    95% CI: [{cli*100:.0f}-{chi*100:.0f}%]")

    elapsed = time.time() - t0
    print(f"\n\n  Analysis completed in {elapsed:.1f}s")
    print(f"  Dates: {', '.join(dates)}")

    sys.stdout = _stdout
    out_path.write_text(_buf.getvalue(), encoding="utf-8")
    print(f"\n✅ Saved to {out_path} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
