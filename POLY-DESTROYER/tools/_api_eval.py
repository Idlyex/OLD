"""V2 — Rigorous config evaluation with TP levels.
API-verified outcomes. No lookahead.
Sweeps: models × confidence × share_price × streak × gap × TP
"""
import asyncio
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import httpx

# ─── CONFIG ───
BET = 2.85
GAMMA = "https://gamma-api.polymarket.com"
MAX_SPREAD = 0.08
MIN_SP = 0.40  # proven: <$0.40 is death zone
CACHE_FILE = Path("results/analysis/_api_outcomes_cache.json")

MODELS = [
    "binance_catboost", "binance_xgboost", "binance_lgbm", "binance_ensemble",
    "hermes_catboost",  "hermes_xgboost",  "hermes_lgbm",  "hermes_ensemble",
]
CONF_THS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
MAX_SPS  = [0.40, 0.45, 0.50, 0.55]
STREAKS  = [1, 2, 3]
TP_LEVELS = [0.80, 0.90, 1.00]  # sell at this price on win
GAP_FILTERS = [
    (None,  None,  "no_gap"),
    (None,  0.01,  "gm0.01"),
    (None,  0.02,  "gm0.02"),
    (None,  0.03,  "gm0.03"),
    (None,  0.05,  "gm0.05"),
    (None,  0.10,  "gm0.10"),
]

TICK_FILES = sorted(Path("results/ticks").glob("ticks_20*.jsonl"))

def wilson_lo(w, n, z=1.96):
    if n == 0: return 0
    p = w / n
    d = 1 + z*z/n
    c = p + z*z/(2*n)
    a = z * math.sqrt((p*(1-p) + z*z/(4*n))/n)
    return max(0, (c - a) / d)


def load_ticks():
    by_slug = defaultdict(list)
    total = 0
    for fp in TICK_FILES:
        for line in fp.read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            t = json.loads(line)
            by_slug[t["slug"]].append(t)
            total += 1
    for slug in by_slug:
        by_slug[slug].sort(key=lambda x: x["ts"])
    print(f"Loaded {total:,} ticks, {len(by_slug)} markets, {len(TICK_FILES)} files")
    return dict(by_slug)


async def fetch_real_outcomes(slugs):
    """Fetch from Gamma API with disk cache."""
    # Load cache
    cached = {}
    if CACHE_FILE.exists():
        cached = json.loads(CACHE_FILE.read_text(encoding="utf-8"))

    to_fetch = [s for s in slugs if s not in cached]
    print(f"  Cache: {len(cached)} hit, {len(to_fetch)} to fetch")

    if to_fetch:
        sem = asyncio.Semaphore(10)
        errs = 0

        async def fetch_one(client, slug):
            nonlocal errs
            async with sem:
                try:
                    r = await client.get(f"{GAMMA}/events", params={"slug": slug, "limit": "1"})
                    data = r.json()
                    if not data: return
                    meta = data[0].get("eventMetadata") or {}
                    fp = meta.get("finalPrice")
                    ptb = meta.get("priceToBeat")
                    if fp is not None and ptb is not None:
                        cached[slug] = {
                            "outcome": "UP" if float(fp) >= float(ptb) else "DOWN",
                            "ptb": float(ptb), "final_price": float(fp),
                        }
                except Exception as e:
                    errs += 1
                    if errs <= 3: print(f"  API error: {slug}: {e}")

        async with httpx.AsyncClient(timeout=15.0) as client:
            for i in range(0, len(to_fetch), 50):
                await asyncio.gather(*[fetch_one(client, s) for s in to_fetch[i:i+50]])
        # Save cache
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(json.dumps(cached), encoding="utf-8")

    outcomes = {s: cached[s] for s in slugs if s in cached}
    print(f"API outcomes: {len(outcomes)}/{len(slugs)} resolved")
    return outcomes


def simulate(ticks_by_slug, outcomes, model, conf_th, max_sp, streak_req, gap_max, gap_min, tp):
    """One entry per market, strict streak. Returns (w, l, pnl, trades)."""
    wins = losses = 0
    pnl = 0.0
    trades = []

    for slug, ticks in ticks_by_slug.items():
        if slug not in outcomes: continue
        actual = outcomes[slug]["outcome"]
        streak = 0; streak_dir = None; entered = False

        for t in ticks:
            if entered: break
            v = t.get(model)
            if v is None: continue
            d = "UP" if v > 0.5 else "DOWN"
            c = max(v, 1 - v)
            if d == "UP":
                sp = t.get("yes_ask", t.get("yes", 0.5))
                spr = t.get("yes_spread", 0)
            else:
                sp = t.get("no_ask", t.get("no", 0.5))
                spr = t.get("no_spread", 0)
            ep = t.get("entry_pct", -1)

            if not (MIN_SP <= sp <= max_sp and spr <= MAX_SPREAD):
                streak = 0; streak_dir = None; continue

            if c >= conf_th:
                if d == streak_dir and streak > 0: streak += 1
                else: streak = 1; streak_dir = d
                if streak >= streak_req and 0 <= ep <= 0.95:
                    raw_gap = t.get("gap_pct", 0) or 0
                    dir_gap = raw_gap if d == "UP" else -raw_gap
                    if gap_max is not None and dir_gap < -gap_max: continue
                    if gap_min is not None and dir_gap < gap_min: continue

                    entered = True
                    won = (d == actual)
                    shares = BET / sp if sp > 0 else 0
                    p = shares * (tp - sp) if won else -BET
                    if won: wins += 1
                    else: losses += 1
                    pnl += p
                    trades.append({"slug": slug, "d": d, "sp": sp, "c": c,
                                   "gap": dir_gap, "won": won, "pnl": p, "ep": ep})
            else:
                streak = 0; streak_dir = None
    return wins, losses, pnl, trades


def run_sweep(ticks_by_slug, outcomes):
    results = []
    total = len(MODELS)*len(CONF_THS)*len(MAX_SPS)*len(STREAKS)*len(GAP_FILTERS)*len(TP_LEVELS)
    done = 0; t0 = time.time()

    for model in MODELS:
        for conf_th in CONF_THS:
            for max_sp in MAX_SPS:
                for streak_req in STREAKS:
                    for gap_max, gap_min, gl in GAP_FILTERS:
                        for tp in TP_LEVELS:
                            w, l, pnl, trades = simulate(
                                ticks_by_slug, outcomes, model, conf_th,
                                max_sp, streak_req, gap_max, gap_min, tp
                            )
                            n = w + l; done += 1
                            if n >= 5:
                                wr = w/n; ev = pnl/n
                                results.append({
                                    "model": model, "conf": conf_th, "max_sp": max_sp,
                                    "streak": streak_req, "gap": gl, "tp": tp,
                                    "n": n, "w": w, "l": l, "wr": wr, "ev": ev,
                                    "pnl": pnl, "wlo": wilson_lo(w, n),
                                })
        print(f"  {model}: {done/total*100:.0f}% ({time.time()-t0:.0f}s)")

    results.sort(key=lambda r: (-r["wlo"], -r["ev"]))
    return results


def mlabel(model):
    return model.split("_")[0][0].upper() + "_" + model.split("_",1)[1]


def format_results(results, outcomes):
    L = []
    W = 120
    L.append(f"{'='*W}")
    L.append(f"  API-VERIFIED EVAL v2 | TP levels | gap momentum | SP>=$0.40")
    L.append(f"  Data: {', '.join(f.stem for f in TICK_FILES)} | {len(outcomes)} markets")
    L.append(f"  {len(results)} configs with N>=5")
    L.append(f"{'='*W}")

    # === FULL TABLE: N>=10, EV>0, sorted by Wilson ===
    L.append(f"\n{'-'*W}")
    L.append(f"  FULL TABLE (N>=10, EV>0, by Wilson lower bound)")
    L.append(f"{'-'*W}")
    hdr = (f"  {'#':>3} {'Model':<18} {'C':>4} {'SP':>5} {'S':>2} {'Gap':<7} "
           f"{'TP':>4} {'N':>4} {'W':>3}-{'L':>3} {'WR':>4} {'WiL':>4} {'EV$':>6} {'PnL$':>7}")
    L.append(hdr)
    good = [r for r in results if r["n"] >= 10 and r["ev"] > 0]
    for i, r in enumerate(good, 1):
        L.append(
            f"  {i:>3} {mlabel(r['model']):<18} {r['conf']:.2f} "
            f"${r['max_sp']:.2f} {r['streak']:>2}x {r['gap']:<7} "
            f"${r['tp']:.2f} {r['n']:>4} {r['w']:>3}-{r['l']:>3} "
            f"{r['wr']*100:>4.0f} {r['wlo']*100:>4.0f} "
            f"{r['ev']:>+6.2f} {r['pnl']:>+7.1f}"
        )
    L.append(f"  --- {len(good)} rows ---")

    # === TP COMPARISON for key configs ===
    L.append(f"\n{'-'*W}")
    L.append(f"  TP LEVEL COMPARISON ($0.80 vs $0.90 vs $1.00)")
    L.append(f"{'-'*W}")
    key_cfgs = [
        ("hermes_catboost",  0.85, 0.55, 2, "gm0.05"),
        ("hermes_catboost",  0.85, 0.55, 2, "no_gap"),
        ("hermes_catboost",  0.85, 0.55, 3, "gm0.05"),
        ("binance_catboost", 0.85, 0.55, 2, "gm0.05"),
        ("binance_catboost", 0.85, 0.55, 3, "no_gap"),
        ("binance_catboost", 0.90, 0.55, 2, "gm0.02"),
        ("binance_ensemble", 0.85, 0.55, 2, "gm0.05"),
        ("binance_ensemble", 0.85, 0.55, 2, "no_gap"),
        ("binance_ensemble", 0.85, 0.55, 3, "no_gap"),
        ("binance_lgbm",     0.85, 0.55, 2, "gm0.05"),
        ("binance_lgbm",     0.90, 0.55, 2, "gm0.05"),
        ("hermes_ensemble",  0.85, 0.55, 2, "gm0.05"),
        ("hermes_ensemble",  0.85, 0.55, 3, "gm0.05"),
        ("hermes_ensemble",  0.95, 0.55, 2, "no_gap"),
        ("hermes_lgbm",      0.85, 0.55, 2, "gm0.05"),
    ]
    for model, conf, msp, st, gl in key_cfgs:
        row_parts = []
        for tp in TP_LEVELS:
            match = [r for r in results if r["model"]==model and r["conf"]==conf
                     and r["max_sp"]==msp and r["streak"]==st and r["gap"]==gl and r["tp"]==tp]
            if match:
                r = match[0]
                row_parts.append(f"TP${tp:.2f}: {r['n']:>3}t WR={r['wr']*100:.0f}% EV${r['ev']:+.2f} PnL${r['pnl']:+.1f}")
            else:
                row_parts.append(f"TP${tp:.2f}: ---")
        L.append(f"  {mlabel(model):<16} c{conf:.2f} s{st}x {gl:<7} | {'  |  '.join(row_parts)}")

    # === GAP MOMENTUM DEEP DIVE ===
    L.append(f"\n{'-'*W}")
    L.append(f"  GAP MOMENTUM DEEP DIVE (2x streak, SP<=$0.55, TP$1.00)")
    L.append(f"{'-'*W}")
    top4 = ["hermes_catboost", "binance_catboost", "binance_ensemble", "hermes_ensemble",
            "binance_lgbm", "hermes_lgbm"]
    for model in top4:
        L.append(f"\n  {mlabel(model)}:")
        for conf in [0.80, 0.85, 0.90]:
            row = []
            for _, gmin, gl in GAP_FILTERS:
                match = [r for r in results if r["model"]==model and r["conf"]==conf
                         and r["max_sp"]==0.55 and r["streak"]==2 and r["gap"]==gl and r["tp"]==1.0]
                if match:
                    r = match[0]
                    row.append(f"{gl}: {r['n']:>3}t {r['wr']*100:.0f}% ${r['ev']:+.2f}")
                else:
                    row.append(f"{gl}: ---")
            L.append(f"    c{conf:.2f} | {'  |  '.join(row)}")

    # === STREAK x TP MATRIX ===
    L.append(f"\n{'-'*W}")
    L.append(f"  STREAK x TP MATRIX (SP<=$0.55, gm0.05)")
    L.append(f"{'-'*W}")
    for model in ["hermes_catboost", "binance_catboost", "binance_ensemble"]:
        L.append(f"\n  {mlabel(model)} c>=0.85:")
        for st in STREAKS:
            row = []
            for tp in TP_LEVELS:
                match = [r for r in results if r["model"]==model and r["conf"]==0.85
                         and r["max_sp"]==0.55 and r["streak"]==st and r["gap"]=="gm0.05" and r["tp"]==tp]
                if match:
                    r = match[0]
                    row.append(f"TP${tp:.2f}: {r['n']:>3}t {r['wr']*100:.0f}% EV${r['ev']:+.2f} P${r['pnl']:+.0f}")
                else:
                    row.append(f"TP${tp:.2f}: ---")
            L.append(f"    {st}x | {'  |  '.join(row)}")

    # === BEST PER MODEL (N>=15) ===
    L.append(f"\n{'-'*W}")
    L.append(f"  BEST PER MODEL (N>=15, TP=$1.00)")
    L.append(f"{'-'*W}")
    seen = set()
    for r in results:
        if r["n"] >= 15 and r["tp"] == 1.0 and r["model"] not in seen:
            seen.add(r["model"])
            L.append(
                f"  {mlabel(r['model']):<18} c{r['conf']:.2f} SP${r['max_sp']:.2f} "
                f"s{r['streak']}x {r['gap']:<7} "
                f"N={r['n']:>3} {r['w']}W-{r['l']}L WR={r['wr']*100:.0f}% "
                f"WiL={r['wlo']*100:.0f}% EV=${r['ev']:+.2f} PnL=${r['pnl']:+.1f}"
            )

    return "\n".join(L)


async def main():
    t0 = time.time()
    ticks_by_slug = load_ticks()

    print("\nFetching API outcomes (cached)...")
    outcomes = await fetch_real_outcomes(ticks_by_slug.keys())

    n_combos = len(MODELS)*len(CONF_THS)*len(MAX_SPS)*len(STREAKS)*len(GAP_FILTERS)*len(TP_LEVELS)
    print(f"\nSweep: {n_combos} combos...")
    results = run_sweep(ticks_by_slug, outcomes)

    report = format_results(results, outcomes)
    out_path = Path("results/analysis/api_eval_v2.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")

    print(f"\nDone in {time.time()-t0:.0f}s. {out_path}")
    print(report)


if __name__ == "__main__":
    asyncio.run(main())
