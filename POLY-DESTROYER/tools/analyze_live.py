"""
POLY-DESTROYER — Exhaustive Live Tick + Outcome Analysis.

No lookahead. Every simulation uses only FIRST qualifying tick per market.
If tick at time T has conf >= threshold and price <= max_sp, it counts as entry.
Outcome resolved later — no peeking.

Usage:
    python tools/analyze_live.py [--date 2026-05-10] [--bet 2.85]
"""
import json, statistics, sys, os, argparse, itertools, yaml, io
from pathlib import Path
from datetime import datetime
from collections import defaultdict

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np

TICKS_DIR = Path("results/ticks")
CONFIG_PATH = Path("config/trading.yaml")

# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════

def load_config():
    """Load current trading config."""
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    entry = cfg.get("entry", {})
    model = cfg.get("model", {})
    return {
        "primary_model": model.get("primary", "catboost"),
        "min_confidence": entry.get("min_confidence", 0.85),
        "max_share_price": entry.get("max_share_price", 0.55),
        "min_share_price": entry.get("min_share_price", 0.10),
        "min_entry_pct": entry.get("min_entry_pct", 0.0),
        "max_entry_pct": entry.get("max_entry_pct", 0.9),
        "max_spread": entry.get("max_spread", 0.08),
        "bet": cfg.get("execution", {}).get("order_size_usd", 2.85),
    }

# ═══════════════════════════════════════════════════════════════
#  TABLE HELPERS
# ═══════════════════════════════════════════════════════════════

def table(headers, rows, col_widths=None, align=None):
    """Print a formatted ASCII table (Windows-safe)."""
    n = len(headers)
    if not col_widths:
        col_widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=4)) + 2 for i, h in enumerate(headers)]
    if not align:
        align = ['r'] * n

    def fmt(val, w, a):
        s = str(val)
        return s.rjust(w) if a == 'r' else s.ljust(w) if a == 'l' else s.center(w)

    rule = '  +' + '+'.join('-' * w for w in col_widths) + '+'
    hdr = '  |' + '|'.join(fmt(h, w, 'c') for h, w in zip(headers, col_widths)) + '|'

    print(rule)
    print(hdr)
    print(rule)
    for row in rows:
        print('  |' + '|'.join(fmt(row[i] if i < len(row) else '', col_widths[i], align[i]) for i in range(n)) + '|')
    print(rule)

def box(title, headers, rows, col_widths=None, align=None):
    """Pretty box table with titled border."""
    n = len(headers)
    if not col_widths:
        col_widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=4)) + 1
                      for i, h in enumerate(headers)]
    if not align:
        align = ['r'] * n

    def fmt(val, w, a):
        s = str(val)
        return s.rjust(w) if a == 'r' else s.ljust(w) if a == 'l' else s.center(w)

    inner_w = sum(col_widths) + n - 1
    pad = inner_w - len(title) - 4
    lp = max(pad // 2, 1)
    rp = max(pad - lp, 1)
    print(f"  +{'-'*lp}-- {title} {'-'*rp}+")
    hdr = '  |' + ' '.join(fmt(h, col_widths[i], align[i]) for i, h in enumerate(headers)) + ' |'
    print(hdr)
    print(f"  |{'-' * inner_w}|")
    for row in rows:
        line = '  |' + ' '.join(
            fmt(row[i] if i < len(row) else '', col_widths[i], align[i])
            for i in range(n)
        ) + ' |'
        print(line)
    print(f"  +{'-' * inner_w}+")

# ═══════════════════════════════════════════════════════════════
#  LOAD
# ═══════════════════════════════════════════════════════════════

def load_ticks(date=None):
    if date:
        paths = [TICKS_DIR / f"ticks_{date}.jsonl"]
    else:
        paths = sorted(TICKS_DIR.glob("ticks_*.jsonl"))
        paths = [p for p in paths if "binance" not in p.name and "hermes" not in p.name]
    ticks = []
    for p in paths:
        if not p.exists(): continue
        for line in open(p, encoding="utf-8"):
            try: ticks.append(json.loads(line))
            except: pass
    return ticks

def load_outcomes(date=None):
    if date:
        paths = [TICKS_DIR / f"outcomes_{date}.jsonl"]
    else:
        paths = sorted(TICKS_DIR.glob("outcomes_*.jsonl"))
    out = []
    for p in paths:
        if not p.exists(): continue
        for line in open(p, encoding="utf-8"):
            try: out.append(json.loads(line))
            except: pass
    return out

def get_model_cols(ticks):
    return sorted([k for k in ticks[0].keys()
                    if k.startswith(("binance_", "hermes_"))
                    and k not in ("binance_ensemble", "hermes_ensemble")] +
                   [k for k in ticks[0].keys() if k.endswith("_ensemble")])

def get_slug_ticks(ticks):
    d = defaultdict(list)
    for t in ticks: d[t["slug"]].append(t)
    return d

def S(title):
    print(f"\n{'='*78}")
    print(f"  {title}")
    print(f"{'='*78}")

# ═══════════════════════════════════════════════════════════════
#  1) OVERVIEW
# ═══════════════════════════════════════════════════════════════

def sec_overview(ticks, outcomes):
    S("1. OVERVIEW")
    slugs = set(t["slug"] for t in ticks)
    dur_min = (ticks[-1]["ts"] - ticks[0]["ts"]) / 60
    t0 = datetime.fromtimestamp(ticks[0]["ts"]).strftime("%H:%M:%S")
    t1 = datetime.fromtimestamp(ticks[-1]["ts"]).strftime("%H:%M:%S")
    print(f"  {len(ticks):,} ticks | {len(outcomes)} outcomes | {len(slugs)} markets | {dur_min:.1f}min ({t0}->{t1})")
    by_dur = defaultdict(set)
    for t in ticks: by_dur[t.get("dur_min",0)].add(t["slug"])
    for d in sorted(by_dur): print(f"    {d}m: {len(by_dur[d])} markets")
    by_dur_o = defaultdict(list)
    for o in outcomes: by_dur_o[o.get("dur_min",0)].append(o)
    for d in sorted(by_dur_o):
        g = by_dur_o[d]
        ups = sum(1 for o in g if o["outcome"]=="UP")
        print(f"    {d}m outcomes: {len(g)} (UP {ups} / DN {len(g)-ups})")

# ═══════════════════════════════════════════════════════════════
#  2) SOL PRICES
# ═══════════════════════════════════════════════════════════════

def sec_sol(ticks):
    S("2. SOL PRICES (Pyth vs Binance)")
    pyth = [t.get("sol_pyth",0) for t in ticks if t.get("sol_pyth",0)>0]
    bn = [t.get("sol_binance",0) for t in ticks if t.get("sol_binance",0)>0]
    diffs = [t.get("sol_diff",0) for t in ticks if t.get("sol_diff") is not None]
    if not pyth: print("  No SOL data"); return
    print(f"  Pyth:  ${min(pyth):.4f}-${max(pyth):.4f}")
    print(f"  Bin:   ${min(bn):.4f}-${max(bn):.4f}")
    if diffs:
        print(f"  Offset mean={np.mean(diffs):+.4f} std={np.std(diffs):.4f} [{min(diffs):+.4f}..{max(diffs):+.4f}]")
        print(f"  Pyth>Bin: {sum(1 for d in diffs if d>0)}/{len(diffs)} ({sum(1 for d in diffs if d>0)/len(diffs)*100:.0f}%)")

# ═══════════════════════════════════════════════════════════════
#  3) ORDERBOOK / SHARE PRICES
# ═══════════════════════════════════════════════════════════════

def sec_orderbook(ticks):
    S("3. SHARE PRICES & ORDERBOOK")
    by_dur = defaultdict(list)
    for t in ticks: by_dur[t.get("dur_min",0)].append(t)
    for dur in sorted(by_dur):
        g = by_dur[dur]
        print(f"\n  -- {dur}m --")
        for label, key in [("UP Ask","yes_ask"),("DN Ask","no_ask"),("Spread","yes_spread"),("Depth","yes_depth")]:
            vals = [t[key] for t in g if t.get(key,0)>0]
            if vals:
                u = "shares" if "depth" in key.lower() else ""
                print(f"    {label:<10} mean=${np.mean(vals):.3f}  std=${np.std(vals):.3f}  [{min(vals):.3f}..{max(vals):.3f}] {u}")

# ═══════════════════════════════════════════════════════════════
#  4) MODEL PREDICTION STATS
# ═══════════════════════════════════════════════════════════════

def sec_models(ticks):
    S("4. ML MODEL PREDICTION DISTRIBUTIONS")
    cols = get_model_cols(ticks)
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    headers = ["Model"] + [f">{int(th*100)}%" for th in thresholds]
    widths = [22] + [5] * len(thresholds)
    aligns = ['l'] + ['r'] * len(thresholds)
    rows = []
    for col in cols:
        vals = [t.get(col, 0.5) for t in ticks if col in t]
        if not vals: continue
        confs = [max(v, 1-v) for v in vals]
        row = [col]
        for th in thresholds:
            pct = sum(1 for c in confs if c >= th) / len(confs) * 100
            row.append(f"{pct:.0f}%")
        rows.append(row)
    table(headers, rows, widths, aligns)

# ═══════════════════════════════════════════════════════════════
#  5) PER-MODEL ACCURACY (on resolved markets)
# ═══════════════════════════════════════════════════════════════

def _model_short_al(model):
    """hermes_catboost -> CATBOOST (H)"""
    src = "H" if model.startswith("hermes") else "B"
    name = model.split("_", 1)[1].upper()
    return f"{name} ({src})"

def _sim_entry_al(ticks_list, model, min_conf, sp_lo, sp_hi, max_spread, elo, ehi):
    """First qualifying tick with ALL gates. Returns entry dict or None."""
    for t in ticks_list:
        ep = t.get("entry_pct", -1)
        if ep < elo or ep > ehi: continue
        v = t.get(model)
        if v is None: continue
        d = "UP" if v > 0.5 else "DOWN"
        c = max(v, 1 - v)
        if c < min_conf: continue
        sp = t.get("yes_ask" if d == "UP" else "no_ask",
                   t.get("yes" if d == "UP" else "no", 0.5))
        if sp <= 0 or sp > 1: continue
        if sp < sp_lo or sp >= sp_hi: continue
        spread = t.get("yes_spread" if d == "UP" else "no_spread", 0)
        if spread > 0 and spread > max_spread: continue
        return {"dir": d, "sp": sp, "conf": c, "ep": ep}
    return None

def sec_accuracy(ticks, outcomes):
    S("5. MODEL ACCURACY ON RESOLVED MARKETS")
    print("  First qualifying tick with ALL gates (conf + SP + spread + entry window)")
    if not outcomes: print("  No outcomes"); return
    cfg = load_config()
    outcome_map = {o["slug"]: o["outcome"] for o in outcomes}
    st = get_slug_ticks(ticks)
    cols = get_model_cols(ticks)

    sp_lo = cfg["min_share_price"]
    sp_hi = cfg["max_share_price"]
    max_spread = cfg["max_spread"]
    elo = cfg["min_entry_pct"]
    ehi = cfg["max_entry_pct"]

    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    headers = ["Conf>=", "N", "W", "L", "WR", "PnL", "AvgSP"]
    widths = [6, 5, 4, 4, 7, 9, 7]
    aligns = ['r', 'r', 'r', 'r', 'r', 'r', 'r']

    print(f"  Gates: SP ${sp_lo:.2f}-${sp_hi:.2f} | spread <=${max_spread:.2f} | window {elo:.0%}-{ehi:.0%}")

    for dur in [5, 15, None]:
        dur_label = f"{dur}m" if dur else "ALL"
        dur_outcomes = {s:o for s,o in outcome_map.items()
                        if dur is None or any(t.get("dur_min")==dur for t in st.get(s,[]))}
        if not dur_outcomes: continue
        n_mkts = len(dur_outcomes)

        # Filter ticks by duration
        dur_st = defaultdict(list)
        for s in dur_outcomes:
            dur_st[s] = st.get(s, [])

        bet = cfg.get("bet", 2.85)
        for col in cols:
            rows = []
            for th in thresholds:
                trades = []
                for slug, actual in dur_outcomes.items():
                    entry = _sim_entry_al(dur_st.get(slug, []), col, th,
                                          sp_lo, sp_hi, max_spread, elo, ehi)
                    if entry is None: continue
                    won = entry["dir"] == actual
                    shares = bet / entry["sp"]
                    pnl = shares * (1.0 - entry["sp"]) if won else -bet
                    trades.append({"won": won, "pnl": pnl, "sp": entry["sp"]})
                if not trades: continue
                n = len(trades); w = sum(1 for t in trades if t["won"])
                pnl_t = sum(t["pnl"] for t in trades)
                rows.append([f"{th:.0%}", n, w, n-w, f"{w/n*100:.1f}%",
                             f"${pnl_t:+.1f}",
                             f"${np.mean([t['sp'] for t in trades]):.3f}"])
            title = f"{_model_short_al(col)} -- {dur_label} ({n_mkts} markets)"
            box(title, headers, rows, widths, aligns)

# ═══════════════════════════════════════════════════════════════
#  6) FIRST-TICK ENTRY SIMULATION (THE BIG ONE)
# ═══════════════════════════════════════════════════════════════

def _sim_entry(ticks_list, model, min_conf, max_sp, min_sp, max_spread, elo, ehi, bet):
    """Simulate EXACTLY like live _evaluate_entry:
    Scan ticks chronologically. First tick where ALL gates pass -> entry.
    Gates: entry_pct in window, conf >= threshold, min_sp <= sp <= max_sp, spread <= max_spread.
    Returns trade dict or None.
    """
    for t in ticks_list:
        ep = t.get("entry_pct", -1)
        if ep < elo or ep > ehi: continue
        v = t.get(model)
        if v is None: continue
        direction = "UP" if v > 0.5 else "DOWN"
        conf = max(v, 1 - v)
        sp = t.get("yes_ask" if direction == "UP" else "no_ask",
                   t.get("yes" if direction == "UP" else "no", 0.5))
        spread = t.get("yes_spread" if direction == "UP" else "no_spread", 0)
        if sp <= 0 or sp > 1: continue
        # GATE: confidence
        if conf < min_conf: continue
        # GATE: price range
        if sp < min_sp or sp > max_sp: continue
        # GATE: spread
        if spread > 0 and spread > max_spread: continue
        # ALL gates passed -> entry
        shares = bet / sp
        return {"dir": direction, "sp": sp, "conf": conf, "ep": ep, "spread": spread, "shares": shares}
    return None


def sec_first_tick_sim(ticks, outcomes, bet):
    S("6. FIRST-TICK ENTRY SIMULATION (1:1 with live logic)")
    if not outcomes: print("  No outcomes"); return
    print(f"  Gates (same as live _evaluate_entry):")
    print(f"    1. entry_pct in [elo, ehi]")
    print(f"    2. conf >= threshold")
    print(f"    3. min_sp($0.10) <= share_price <= max_sp")
    print(f"    4. spread <= max_spread ($0.08)")
    print(f"    5. First tick passing ALL gates -> BUY, hold to resolution")
    print(f"  Bet=${bet:.2f}")

    outcome_map = {o["slug"]: o for o in outcomes}
    st = get_slug_ticks(ticks)
    cols = get_model_cols(ticks)

    MAX_SPREAD = 0.08  # from trading.yaml
    MIN_SP = 0.10      # from trading.yaml

    models = cols
    confs = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    max_sps = [0.45, 0.50, 0.55, 0.60, 0.65]
    entry_windows = [(0.0,0.3),(0.0,0.5),(0.0,0.9),(0.1,0.5),(0.2,0.6),(0.3,0.7),(0.5,0.9),(0.6,1.0)]

    results = []
    for model in models:
        for min_conf in confs:
            for max_sp in max_sps:
                for (elo, ehi) in entry_windows:
                    trades = []
                    for slug, oinfo in outcome_map.items():
                        actual = oinfo["outcome"]
                        entry = _sim_entry(st.get(slug, []), model, min_conf,
                                           max_sp, MIN_SP, MAX_SPREAD, elo, ehi, bet)
                        if entry is None: continue
                        won = entry["dir"] == actual
                        pnl = entry["shares"] * (1.0 - entry["sp"]) if won else -bet
                        trades.append({"won": won, "pnl": pnl, "sp": entry["sp"],
                                       "conf": entry["conf"], "ep": entry["ep"]})
                    if not trades: continue
                    n = len(trades)
                    w = sum(1 for t in trades if t["won"])
                    wr = w / n * 100
                    pnl_total = sum(t["pnl"] for t in trades)
                    ev = pnl_total / n
                    results.append({
                        "model": model, "conf": min_conf, "max_sp": max_sp,
                        "elo": elo, "ehi": ehi, "n": n, "wins": w, "wr": wr,
                        "pnl": pnl_total, "ev": ev,
                        "avg_sp": np.mean([t["sp"] for t in trades]),
                        "avg_conf": np.mean([t["conf"] for t in trades]),
                    })

    if not results:
        print("  No qualifying entries found")
        return

    # ── CURRENT CONFIG vs OPTIMAL ──
    cfg = load_config()
    cfg_model_hermes = f"hermes_{cfg['primary_model']}"
    cfg_model_binance = f"binance_{cfg['primary_model']}"
    cfg_conf = cfg["min_confidence"]
    cfg_sp = cfg["max_share_price"]
    cfg_elo = cfg["min_entry_pct"]
    cfg_ehi = cfg["max_entry_pct"]

    # Find current config result for both sources
    current_hermes = [r for r in results
                      if r["model"] == cfg_model_hermes and r["conf"] == cfg_conf
                      and r["max_sp"] == cfg_sp and r["elo"] == cfg_elo and r["ehi"] == cfg_ehi]
    current_binance = [r for r in results
                       if r["model"] == cfg_model_binance and r["conf"] == cfg_conf
                       and r["max_sp"] == cfg_sp and r["elo"] == cfg_elo and r["ehi"] == cfg_ehi]

    print(f"\n  +--- CURRENT CONFIG (trading.yaml) ----------------------------------------+")
    print(f"  |  Model: {cfg_model_hermes} (primary: hermes {cfg['primary_model']})")
    print(f"  |  Conf: >={cfg_conf:.0%}  SP: ${cfg['min_share_price']:.2f}-${cfg_sp:.2f}  Window: {cfg_elo:.0%}-{cfg_ehi:.0%}  Spread: <=${cfg['max_spread']:.2f}")
    if current_hermes:
        c = current_hermes[0]
        print(f"  |  >> hermes:  {c['n']}t  {c['wins']}W  {c['wr']:.0f}%WR  ${c['pnl']:+.2f} PnL  EV ${c['ev']:+.2f}")
    else:
        print(f"  |  >> hermes:  0 trades (all blocked by gates)")
    if current_binance:
        c = current_binance[0]
        print(f"  |  >> binance: {c['n']}t  {c['wins']}W  {c['wr']:.0f}%WR  ${c['pnl']:+.2f} PnL  EV ${c['ev']:+.2f}")
    else:
        print(f"  |  >> binance: 0 trades (all blocked by gates)")
    print(f"  +--------------------------------------------------------------------------+")

    results.sort(key=lambda x: -x["pnl"])
    best = results[0]
    wr_sorted = sorted([r for r in results if r["n"] >= 3], key=lambda x: (-x["wr"], -x["pnl"]))
    ev_sorted = sorted([r for r in results if r["n"] >= 3], key=lambda x: -x["ev"])

    print(f"\n  +--- OPTIMAL CONFIGS -------------------------------------------------+")
    print(f"  |  Best PnL: {best['model']:<18} {best['conf']:.0%} <=${best['max_sp']:.2f} {best['elo']*100:.0f}-{best['ehi']*100:.0f}%  {best['n']}t {best['wr']:.0f}%WR ${best['pnl']:+.2f}")
    if wr_sorted:
        bwr = wr_sorted[0]
        print(f"  |  Best WR:  {bwr['model']:<18} {bwr['conf']:.0%} <=${bwr['max_sp']:.2f} {bwr['elo']*100:.0f}-{bwr['ehi']*100:.0f}%  {bwr['n']}t {bwr['wr']:.0f}%WR ${bwr['pnl']:+.2f}")
    if ev_sorted:
        bev = ev_sorted[0]
        print(f"  |  Best EV:  {bev['model']:<18} {bev['conf']:.0%} <=${bev['max_sp']:.2f} {bev['elo']*100:.0f}-{bev['ehi']*100:.0f}%  {bev['n']}t {bev['wr']:.0f}%WR EV${bev['ev']:+.2f}")
    print(f"  +-------------------------------------------------------------------+")

    # TOP 20 by PnL
    sim_headers = ["#", "Model", "Conf", "SP<=", "Window", "N", "W", "WR", "PnL", "EV"]
    sim_widths = [3, 22, 4, 5, 8, 4, 3, 5, 8, 6]
    sim_align = ['r', 'l', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r']
    print(f"\n  TOP 20 by PnL ({len(results):,} combos):")
    sim_rows = []
    for i, r in enumerate(results[:20]):
        wnd = f"{r['elo']*100:.0f}-{r['ehi']*100:.0f}%"
        sim_rows.append([i+1, r['model'], f"{r['conf']:.0%}", f"${r['max_sp']:.2f}", wnd,
                         r['n'], r['wins'], f"{r['wr']:.0f}%", f"${r['pnl']:+.0f}", f"${r['ev']:+.2f}"])
    table(sim_headers, sim_rows, sim_widths, sim_align)

    # TOP 10 by WR
    if wr_sorted:
        print(f"\n  TOP 10 by Win Rate (min 3 trades):")
        wr_rows = []
        for i, r in enumerate(wr_sorted[:10]):
            wnd = f"{r['elo']*100:.0f}-{r['ehi']*100:.0f}%"
            wr_rows.append([i+1, r['model'], f"{r['conf']:.0%}", f"${r['max_sp']:.2f}", wnd,
                            r['n'], r['wins'], f"{r['wr']:.0f}%", f"${r['pnl']:+.0f}", f"${r['ev']:+.2f}"])
        table(sim_headers, wr_rows, sim_widths, sim_align)

# ═══════════════════════════════════════════════════════════════
#  7) TIMING ANALYSIS (entry % → accuracy)
# ═══════════════════════════════════════════════════════════════

def sec_timing(ticks, outcomes):
    S("7. ENTRY TIMING (first tick per window)")
    if not outcomes: print("  No outcomes"); return
    outcome_map = {o["slug"]: o["outcome"] for o in outcomes}
    st = get_slug_ticks(ticks)

    windows = [(0,0.1),(0.05,0.15),(0.1,0.2),(0.15,0.25),(0.2,0.3),(0.3,0.4),
               (0.4,0.5),(0.5,0.6),(0.6,0.7),(0.7,0.8),(0.8,0.9)]

    for model in ["hermes_catboost", "hermes_ensemble", "binance_catboost"]:
        print(f"\n  {model}:")
        print(f"  {'Window':>10} {'N':>4} {'Acc':>5} {'AvgConf':>8} {'AvgSP':>7} {'UP%':>5}")
        print(f"  {'-'*10} {'-'*4} {'-'*5} {'-'*8} {'-'*7} {'-'*5}")
        for lo,hi in windows:
            ok=0; tot=0; cnfs=[]; sps=[]; ups=0
            for slug, actual in outcome_map.items():
                for t in st.get(slug,[]):
                    ep = t.get("entry_pct",-1)
                    if ep<lo or ep>hi: continue
                    v = t.get(model)
                    if v is None: continue
                    d = "UP" if v>0.5 else "DOWN"
                    c = max(v,1-v)
                    sp = t.get("yes_ask" if d=="UP" else "no_ask", 0.5)
                    tot += 1
                    if d == actual: ok += 1
                    if d == "UP": ups += 1
                    cnfs.append(c); sps.append(sp)
                    break
            if tot > 0:
                print(f"  {lo*100:>3.0f}-{hi*100:<3.0f}%  {tot:>4} {ok/tot*100:>4.0f}% {np.mean(cnfs):>7.0%} ${np.mean(sps):>5.3f} {ups/tot*100:>4.0f}%")

# ═══════════════════════════════════════════════════════════════
#  8) SHARE PRICE BINS → OUTCOME
# ═══════════════════════════════════════════════════════════════

def sec_price_bins(ticks, outcomes, bet):
    S("8. SHARE PRICE BINS -> PnL")
    if not outcomes: print("  No outcomes"); return
    outcome_map = {o["slug"]: o for o in outcomes}
    st = get_slug_ticks(ticks)
    cols = get_model_cols(ticks)

    bins = [(0.00,0.10),(0.10,0.20),(0.20,0.30),(0.30,0.40),(0.40,0.45),
            (0.45,0.50),(0.50,0.55),(0.55,0.60),(0.60,0.70),(0.70,0.80)]

    headers = ["SP Range", "N", "W", "L", "WR", "EV/sh", "PnL", "AvgSP"]
    widths = [13, 5, 4, 4, 7, 8, 9, 7]
    aligns = ['l', 'r', 'r', 'r', 'r', 'r', 'r', 'r']

    for model in cols:
        title = f"{_model_short_al(model)} -- SP Bins (conf>=60%)"
        rows = []
        for lo,hi in bins:
            trades=[]
            for slug, oinfo in outcome_map.items():
                actual = oinfo["outcome"]
                for t in st.get(slug,[]):
                    ep = t.get("entry_pct",-1)
                    if ep < 0 or ep > 0.9: continue
                    v = t.get(model)
                    if v is None: continue
                    d = "UP" if v>0.5 else "DOWN"
                    c = max(v, 1-v)
                    if c < 0.60: continue
                    sp = t.get("yes_ask" if d=="UP" else "no_ask", t.get("yes" if d=="UP" else "no", 0.5))
                    if sp <= 0 or sp > 1: continue
                    spread = t.get("yes_spread" if d=="UP" else "no_spread", 0)
                    if spread and spread > 0.08: continue
                    if lo <= sp < hi:
                        won = d == actual
                        shares = bet/sp
                        pnl = shares*(1.0-sp) if won else -bet
                        ev_sh = (1.0-sp) if won else -sp
                        trades.append({"won":won,"pnl":pnl,"sp":sp,"ev_sh":ev_sh})
                        break
            if not trades: continue
            n = len(trades)
            w = sum(1 for t in trades if t["won"])
            l = n - w
            wr = w/n*100
            pnl = sum(t["pnl"] for t in trades)
            avg_sp = np.mean([t["sp"] for t in trades])
            ev_sh = np.mean([t["ev_sh"] for t in trades])
            rows.append([f"${lo:.2f}-${hi:.2f}", n, w, l, f"{wr:.1f}%",
                         f"${ev_sh:+.3f}", f"$ {pnl:+.1f}", f"${avg_sp:.3f}"])
        if rows:
            box(title, headers, rows, widths, aligns)

# ═══════════════════════════════════════════════════════════════
#  9) CONFIDENCE BINS → OUTCOME
# ═══════════════════════════════════════════════════════════════

def sec_conf_bins(ticks, outcomes, bet):
    S("9. CONFIDENCE BINS -> PnL")
    if not outcomes: print("  No outcomes"); return
    outcome_map = {o["slug"]: o for o in outcomes}
    st = get_slug_ticks(ticks)
    cols = get_model_cols(ticks)

    bins = [(0.50,0.55),(0.55,0.60),(0.60,0.65),(0.65,0.70),(0.70,0.75),
            (0.75,0.80),(0.80,0.85),(0.85,0.90),(0.90,0.95),(0.95,1.01)]

    headers = ["Conf", "N", "W", "L", "WR", "EV/sh", "PnL"]
    widths = [8, 5, 4, 4, 7, 8, 9]
    aligns = ['l', 'r', 'r', 'r', 'r', 'r', 'r']

    for model in cols:
        title = f"{_model_short_al(model)} -- Conf Bins (SP<=$0.60)"
        rows = []
        for lo,hi in bins:
            trades=[]
            for slug, oinfo in outcome_map.items():
                actual = oinfo["outcome"]
                for t in st.get(slug,[]):
                    ep = t.get("entry_pct",-1)
                    if ep < 0 or ep > 0.9: continue
                    v = t.get(model)
                    if v is None: continue
                    d = "UP" if v>0.5 else "DOWN"
                    c = max(v,1-v)
                    sp = t.get("yes_ask" if d=="UP" else "no_ask", t.get("yes" if d=="UP" else "no", 0.5))
                    if sp <= 0 or sp > 1 or sp > 0.60: continue
                    spread = t.get("yes_spread" if d=="UP" else "no_spread", 0)
                    if spread and spread > 0.08: continue
                    if lo <= c < hi:
                        won = d == actual
                        shares = bet/sp
                        pnl = shares*(1.0-sp) if won else -bet
                        ev_sh = (1.0-sp) if won else -sp
                        trades.append({"won":won,"pnl":pnl,"ev_sh":ev_sh})
                        break
            if not trades: continue
            n = len(trades)
            w = sum(1 for t in trades if t["won"])
            l = n - w
            wr = w/n*100
            pnl = sum(t["pnl"] for t in trades)
            ev_sh = np.mean([t["ev_sh"] for t in trades])
            rows.append([f"{lo:.0%}-{hi:.0%}", n, w, l, f"{wr:.1f}%",
                         f"${ev_sh:+.3f}", f"$ {pnl:+.1f}"])
        if rows:
            box(title, headers, rows, widths, aligns)

# ═══════════════════════════════════════════════════════════════
#  10) PER-MARKET DETAIL (every market's first-tick prediction)
# ═══════════════════════════════════════════════════════════════

def sec_market_detail(ticks, outcomes):
    S("10. PER-MARKET DETAIL")
    if not outcomes: print("  No outcomes"); return
    outcome_map = {o["slug"]: o for o in outcomes}
    st = get_slug_ticks(ticks)

    cfg = load_config()
    model = f"hermes_{cfg['primary_model']}"
    print(f"  Model: {model}")

    data_rows = []
    for slug, oinfo in outcome_map.items():
        actual = oinfo["outcome"]
        sticks = st.get(slug, [])
        if not sticks: continue
        first = sticks[0]
        v = first.get(model, 0.5)
        d = "UP" if v > 0.5 else "DOWN"
        c = max(v, 1-v)
        sp = first.get("yes_ask" if d == "UP" else "no_ask", first.get("yes" if d == "UP" else "no", 0.5))
        sol_start = oinfo.get("sol_start", 0)
        sol_end = oinfo.get("sol_end", 0)
        sol_delta = sol_end - sol_start
        correct = d == actual
        data_rows.append((slug[-28:], oinfo.get("dur_min", 0), actual, d, c, sp,
                          first.get("entry_pct", 0), sol_delta, correct))

    data_rows.sort(key=lambda x: x[0])
    headers = ["Market", "Dur", "Real", "Pred", "Conf", "SP", "EP%", "SOL Δ", ""]
    widths = [28, 3, 4, 4, 4, 6, 4, 7, 1]
    aligns = ['l', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'l']
    rows = []
    for slug, dur, actual, pred, conf, sp, ep, sol_d, correct in data_rows:
        ok = "✓" if correct else "✗"
        rows.append([slug, f"{dur}", actual, pred, f"{conf:.0%}", f"${sp:.3f}",
                     f"{ep*100:.0f}%", f"{sol_d:+.3f}", ok])
    table(headers, rows, widths, aligns)

# ═══════════════════════════════════════════════════════════════
#  11) PYTH-BIN GAP → OUTCOME
# ═══════════════════════════════════════════════════════════════

def sec_gap(ticks, outcomes):
    S("11. PYTH-BINANCE GAP vs OUTCOME")
    if not outcomes: print("  No outcomes"); return
    outcome_map = {o["slug"]: o for o in outcomes}
    st = get_slug_ticks(ticks)

    for outcome in ["UP", "DOWN"]:
        diffs = []
        for slug, actual in outcome_map.items():
            if actual != outcome: continue
            entry_ticks = [t for t in st.get(slug,[]) if 0<=t.get("entry_pct",-1)<=0.3]
            if entry_ticks:
                diffs.append(np.mean([t.get("sol_diff",0) for t in entry_ticks]))
        if diffs:
            print(f"  {outcome}: mean_diff={np.mean(diffs):+.4f} std={np.std(diffs):.4f} n={len(diffs)}")

# ═══════════════════════════════════════════════════════════════
#  12) DIAGNOSIS: Why no trades?
# ═══════════════════════════════════════════════════════════════

def sec_diagnosis(ticks, outcomes):
    S("12. TRADE ENTRY DIAGNOSIS")
    st = get_slug_ticks(ticks)
    cfg = load_config()
    model = f"hermes_{cfg['primary_model']}"
    min_conf = cfg["min_confidence"]
    max_sp = cfg["max_share_price"]
    min_sp = cfg["min_share_price"]
    max_spread = cfg["max_spread"]
    elo = cfg["min_entry_pct"]
    ehi = cfg["max_entry_pct"]

    print(f"  Config: {model} conf>={min_conf:.0%} SP ${min_sp:.2f}-${max_sp:.2f} window {elo:.0%}-{ehi:.0%} spread<=${max_spread:.2f}")

    entered, blocked_conf, blocked_sp, blocked_spread, blocked_other = 0, 0, 0, 0, 0
    rows = []

    for slug in sorted(st.keys()):
        sticks = st[slug]
        dur = sticks[0].get("dur_min", 0)
        best_conf, best_sp = 0, 1.0
        entry_found = False

        for t in sticks:
            ep = t.get("entry_pct", -1)
            if ep < elo or ep > ehi: continue
            v = t.get(model)
            if v is None: continue
            d = "UP" if v > 0.5 else "DOWN"
            c = max(v, 1-v)
            sp = t.get("yes_ask" if d == "UP" else "no_ask",
                       t.get("yes" if d == "UP" else "no", 0.5))
            if sp <= 0 or sp > 1: continue
            spread = t.get("yes_spread" if d == "UP" else "no_spread", 0)

            best_conf = max(best_conf, c)
            best_sp = min(best_sp, sp)

            if c >= min_conf and min_sp <= sp <= max_sp and (spread <= 0 or spread <= max_spread):
                entry_found = True
                entered += 1
                rows.append(["✓", slug[-30:], f"{dur}m", d, f"{c:.0%}", f"${sp:.3f}", f"{ep:.0%}", ""])
                break

        if not entry_found:
            reasons = []
            if best_conf < min_conf:
                reasons.append(f"conf {best_conf:.0%}")
                blocked_conf += 1
            if best_sp > max_sp:
                reasons.append(f"sp ${best_sp:.3f}")
                blocked_sp += 1
            if not reasons:
                reasons.append("window/spread/other")
                blocked_other += 1
            rows.append(["✗", slug[-30:], f"{dur}m", "", f"{best_conf:.0%}", f"${best_sp:.3f}", "", ", ".join(reasons)])

    total = len(st)
    print(f"\n  Summary: {entered}/{total} would enter  |  Blocked: {blocked_conf} conf, {blocked_sp} sp, {blocked_other} other")
    headers = ["", "Market", "Dur", "Dir", "Conf", "SP", "EP", "Block Reason"]
    widths = [1, 30, 3, 4, 4, 6, 4, 20]
    aligns = ['l', 'l', 'r', 'l', 'r', 'r', 'r', 'l']
    table(headers, rows, widths, aligns)

# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

class _Tee:
    def __init__(self, fp):
        self.file = open(fp, 'w', encoding='utf-8')
        self.stdout = sys.stdout
    def write(self, d):
        self.stdout.write(d); self.file.write(d)
    def flush(self):
        self.stdout.flush(); self.file.flush()
    def close(self):
        self.file.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--bet", type=float, default=None)
    args = parser.parse_args()
    if args.bet is None:
        cfg = load_config()
        args.bet = cfg.get("bet", 2.85)

    date = args.date
    if not date:
        files = sorted(TICKS_DIR.glob("ticks_*.jsonl"))
        files = [f for f in files if "binance" not in f.name and "hermes" not in f.name]
        if files: date = files[-1].stem.replace("ticks_", "")

    # Auto-save to .txt
    out_dir = Path("results/analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"analyze_live_{date}.txt"
    tee = _Tee(str(out_path))
    sys.stdout = tee

    print(f"{'#'*78}")
    print(f"  POLY-DESTROYER -- Exhaustive Live Analysis (no lookahead)")
    print(f"  Date: {date} | Bet: ${args.bet:.2f} | Output: {out_path}")
    print(f"{'#'*78}")

    ticks = load_ticks(date)
    outcomes = load_outcomes(date)
    if not ticks: print("  [X] No ticks!"); return

    sec_overview(ticks, outcomes)
    sec_sol(ticks)
    sec_orderbook(ticks)
    sec_models(ticks)
    sec_accuracy(ticks, outcomes)
    sec_first_tick_sim(ticks, outcomes, args.bet)
    sec_timing(ticks, outcomes)
    sec_price_bins(ticks, outcomes, args.bet)
    sec_conf_bins(ticks, outcomes, args.bet)
    sec_market_detail(ticks, outcomes)
    sec_gap(ticks, outcomes)
    sec_diagnosis(ticks, outcomes)

    print(f"\n{'='*78}")
    print(f"  DONE -- {len(ticks):,} ticks, {len(outcomes)} outcomes")
    print(f"  Saved to: {out_path}")
    print(f"{'='*78}")

    tee.close()
    sys.stdout = tee.stdout


if __name__ == "__main__":
    main()
