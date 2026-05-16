"""
POLY-DESTROYER -- Deep Analysis (tick-level, 1:1 live logic).

Sections:
  A. MODEL x CONF  (all gates, N drops at stricter conf)
  B. SP BINS        (Pyth prices only)
  C. GAP ANALYSIS   (Pyth gap from PTB + Binance gap from PTB, %)
  D. ENTRY TIMING   (entry_pct windows)
  E. LIVE TRADES    (actual bot trades from ml_live_trades.json)
  F. EARLY EXIT SIM (what if exit at $0.90 instead of resolution)
  G. SUMMARY

Usage:
    python tools/deep_analysis.py [--date 2026-05-11]
"""
import json, sys, os, io, argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np

TICKS_DIR = Path("results/ticks")
CONFIG_PATH = Path("config/trading.yaml")
OUTPUT_DIR = Path("results/analysis")
TRADES_PATH = Path("results/ml_live_trades.json")


# ================================================================
#  HELPERS
# ================================================================

class Tee:
    def __init__(self, filepath):
        self.file = open(filepath, 'w', encoding='utf-8')
        self.stdout = sys.stdout
    def write(self, data):
        self.stdout.write(data); self.file.write(data)
    def flush(self):
        self.stdout.flush(); self.file.flush()
    def close(self):
        self.file.close()

def load_config():
    import yaml
    if not CONFIG_PATH.exists(): return {}
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

def load_ticks(date):
    path = TICKS_DIR / f"ticks_{date}.jsonl"
    ticks = []
    if path.exists():
        for line in open(path, encoding="utf-8"):
            try: ticks.append(json.loads(line))
            except: pass
    return ticks

def load_outcomes(date):
    path = TICKS_DIR / f"outcomes_{date}.jsonl"
    out = []
    if path.exists():
        for line in open(path, encoding="utf-8"):
            try: out.append(json.loads(line))
            except: pass
    return out

def load_live_trades(date=None):
    if not TRADES_PATH.exists(): return []
    trades = json.load(open(TRADES_PATH, encoding="utf-8"))
    if date:
        trades = [t for t in trades if date in str(t.get("entry_time", ""))]
    return trades

def get_model_cols(ticks):
    return sorted([k for k in ticks[0].keys()
                    if k.startswith(("binance_", "hermes_"))
                    and k not in ("binance_ensemble", "hermes_ensemble")] +
                   [k for k in ticks[0].keys() if k.endswith("_ensemble")])

def _ms(model):
    """hermes_catboost -> CATBOOST (H)"""
    src = "H" if model.startswith("hermes") else "B"
    name = model.split("_", 1)[1].upper()
    return f"{name} ({src})"

def S(title):
    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"{'='*90}")

def box(title, headers, rows, col_widths=None, align=None):
    if not rows: return
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
    lp = max(pad // 2, 1); rp = max(pad - lp, 1)
    print(f"  +{'-'*lp}-- {title} {'-'*rp}+")
    print('  |' + ' '.join(fmt(h, col_widths[i], align[i]) for i, h in enumerate(headers)) + ' |')
    print(f"  |{'-' * inner_w}|")
    for row in rows:
        print('  |' + ' '.join(fmt(row[i] if i < len(row) else '', col_widths[i], align[i])
              for i in range(n)) + ' |')
    print(f"  +{'-' * inner_w}+")


# ================================================================
#  CORE: First-qualifying-tick entry (1:1 with live bot)
# ================================================================

def _sim_entry(ticks_list, model, min_conf, sp_lo, sp_hi, max_spread, elo, ehi):
    """Scan ticks chronologically. First tick where ALL gates pass -> entry.
    Returns entry dict or None."""
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
        return {"dir": d, "sp": sp, "conf": c, "ep": ep, "spread": spread,
                "sol_pyth": t.get("sol_pyth", 0), "sol_binance": t.get("sol_binance", 0),
                "ptb": t.get("ptb", 0), "ts": t.get("ts", 0), "slug": t.get("slug", "")}
    return None


def _run_sim(slug_ticks, outcome_map, model, min_conf, sp_lo, sp_hi, bet,
             max_spread=0.08, elo=0.0, ehi=0.9):
    """Run sim across all markets. Returns list of trade dicts."""
    trades = []
    for slug, actual in outcome_map.items():
        entry = _sim_entry(slug_ticks.get(slug, []), model, min_conf,
                           sp_lo, sp_hi, max_spread, elo, ehi)
        if entry is None: continue
        won = entry["dir"] == actual
        shares = bet / entry["sp"]
        pnl = shares * (1.0 - entry["sp"]) if won else -bet
        trades.append({**entry, "won": won, "pnl": pnl, "shares": shares,
                       "actual": actual})
    return trades


# ================================================================
#  A. MODEL x CONFIDENCE (all gates applied, N drops)
# ================================================================

def sec_model_conf(ticks, outcomes, bet, cfg):
    S("A. MODEL PERFORMANCE BY CONFIDENCE THRESHOLD")
    print("  First qualifying tick with ALL gates (conf + SP + spread + entry window)")
    print(f"  Gates: SP ${cfg['min_share_price']:.2f}-${cfg['max_share_price']:.2f}"
          f" | spread <=${cfg['max_spread']:.2f}"
          f" | window {cfg['min_entry_pct']:.0%}-{cfg['max_entry_pct']:.0%}")
    if not outcomes: print("  No outcomes"); return

    outcome_map = {o["slug"]: o["outcome"] for o in outcomes}
    slug_ticks = defaultdict(list)
    for t in ticks: slug_ticks[t["slug"]].append(t)
    cols = get_model_cols(ticks)

    sp_lo = cfg["min_share_price"]
    sp_hi = cfg["max_share_price"]
    max_spread = cfg["max_spread"]
    elo = cfg["min_entry_pct"]
    ehi = cfg["max_entry_pct"]

    conf_levels = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    headers = ["Conf>=", "N", "W", "L", "WR", "PnL", "EV", "AvgSP"]
    widths  = [6, 5, 4, 4, 7, 9, 7, 7]
    aligns  = ['r', 'r', 'r', 'r', 'r', 'r', 'r', 'r']

    for model in cols:
        rows = []
        for mc in conf_levels:
            trades = _run_sim(slug_ticks, outcome_map, model, mc,
                              sp_lo, sp_hi, bet, max_spread, elo, ehi)
            if not trades: continue
            n = len(trades)
            w = sum(1 for t in trades if t["won"])
            pnl = sum(t["pnl"] for t in trades)
            rows.append([f"{mc:.0%}", n, w, n-w, f"{w/n*100:.1f}%",
                         f"${pnl:+.1f}", f"${pnl/n:+.2f}",
                         f"${np.mean([t['sp'] for t in trades]):.3f}"])
        n_mkts = len(outcome_map)
        box(f"{_ms(model)} -- {n_mkts} markets", headers, rows, widths, aligns)


# ================================================================
#  B. SHARE PRICE BINS (Pyth prices, per model)
# ================================================================

def sec_sp_bins(ticks, outcomes, bet, cfg):
    S("B. SHARE PRICE BINS (Pyth prices, conf>=60%)")
    if not outcomes: print("  No outcomes"); return

    outcome_map = {o["slug"]: o["outcome"] for o in outcomes}
    slug_ticks = defaultdict(list)
    for t in ticks: slug_ticks[t["slug"]].append(t)
    cols = get_model_cols(ticks)

    sp_bins = [(0.01, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 0.40),
               (0.40, 0.45), (0.45, 0.50), (0.50, 0.55), (0.55, 0.60),
               (0.60, 0.70), (0.70, 0.80)]

    headers = ["SP Range", "N", "W", "L", "WR", "PnL", "EV", "AvgSP"]
    widths  = [13, 5, 4, 4, 7, 9, 7, 7]
    aligns  = ['l', 'r', 'r', 'r', 'r', 'r', 'r', 'r']

    for model in cols:
        rows = []
        for lo, hi in sp_bins:
            trades = _run_sim(slug_ticks, outcome_map, model, 0.60,
                              lo, hi, bet, 0.08, 0.0, 0.9)
            if not trades: continue
            n = len(trades)
            w = sum(1 for t in trades if t["won"])
            pnl = sum(t["pnl"] for t in trades)
            rows.append([f"${lo:.2f}-${hi:.2f}", n, w, n-w, f"{w/n*100:.1f}%",
                         f"${pnl:+.1f}", f"${pnl/n:+.2f}",
                         f"${np.mean([t['sp'] for t in trades]):.3f}"])
        box(f"{_ms(model)} -- SP Bins", headers, rows, widths, aligns)


# ================================================================
#  C. GAP ANALYSIS (Pyth vs PTB, Binance vs PTB at entry)
# ================================================================

def sec_gap(ticks, outcomes, bet, cfg):
    S("C. GAP ANALYSIS AT ENTRY (price vs PTB)")
    if not outcomes: print("  No outcomes"); return

    outcome_map = {o["slug"]: o["outcome"] for o in outcomes}
    slug_ticks = defaultdict(list)
    for t in ticks: slug_ticks[t["slug"]].append(t)

    model = f"hermes_{cfg['primary_model']}"
    sp_lo, sp_hi = cfg["min_share_price"], cfg["max_share_price"]
    elo, ehi = cfg["min_entry_pct"], cfg["max_entry_pct"]

    # Get entries
    entries = []
    for slug, actual in outcome_map.items():
        entry = _sim_entry(slug_ticks.get(slug, []), model, 0.60,
                           sp_lo, sp_hi, 0.08, elo, ehi)
        if entry is None: continue
        ptb = entry["ptb"]
        if ptb <= 0: continue
        pyth_gap = (entry["sol_pyth"] - ptb) / ptb * 100 if entry["sol_pyth"] > 0 else None
        bin_gap = (entry["sol_binance"] - ptb) / ptb * 100 if entry["sol_binance"] > 0 else None
        entries.append({"actual": actual, "won": entry["dir"] == actual,
                        "pyth_gap": pyth_gap, "bin_gap": bin_gap,
                        "dir": entry["dir"], "sp": entry["sp"], "conf": entry["conf"]})

    if not entries:
        print("  No qualifying entries"); return

    gap_bins = [(-0.20, -0.10), (-0.10, -0.05), (-0.05, -0.02), (-0.02, 0.0),
                (0.0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.50)]

    headers = ["Gap %", "N", "W", "L", "WR", "UP", "DN"]
    widths = [14, 5, 4, 4, 7, 4, 4]
    aligns = ['l', 'r', 'r', 'r', 'r', 'r', 'r']

    # Table 1: Pyth gap from PTB
    print(f"\n  gap = (price - PTB) / PTB * 100%  |  model: {model}")
    rows_pyth = []
    for glo, ghi in gap_bins:
        sub = [e for e in entries if e["pyth_gap"] is not None and glo <= e["pyth_gap"] < ghi]
        if not sub: continue
        n = len(sub); w = sum(1 for e in sub if e["won"])
        ups = sum(1 for e in sub if e["actual"] == "UP")
        rows_pyth.append([f"{glo:+.2f}..{ghi:+.2f}", n, w, n-w,
                          f"{w/n*100:.1f}%", ups, n-ups])
    box("PYTH GAP from PTB at entry", headers, rows_pyth, widths, aligns)

    # Table 2: Binance gap from PTB
    rows_bin = []
    for glo, ghi in gap_bins:
        sub = [e for e in entries if e["bin_gap"] is not None and glo <= e["bin_gap"] < ghi]
        if not sub: continue
        n = len(sub); w = sum(1 for e in sub if e["won"])
        ups = sum(1 for e in sub if e["actual"] == "UP")
        rows_bin.append([f"{glo:+.2f}..{ghi:+.2f}", n, w, n-w,
                         f"{w/n*100:.1f}%", ups, n-ups])
    box("BINANCE GAP from PTB at entry", headers, rows_bin, widths, aligns)

    # Gap direction as signal
    pyth_ok = sum(1 for e in entries if e["pyth_gap"] is not None
                  and ((e["pyth_gap"] > 0 and e["actual"] == "UP")
                       or (e["pyth_gap"] <= 0 and e["actual"] == "DOWN")))
    bin_ok = sum(1 for e in entries if e["bin_gap"] is not None
                 and ((e["bin_gap"] > 0 and e["actual"] == "UP")
                      or (e["bin_gap"] <= 0 and e["actual"] == "DOWN")))
    n_pyth = sum(1 for e in entries if e["pyth_gap"] is not None)
    n_bin = sum(1 for e in entries if e["bin_gap"] is not None)
    print(f"\n  Gap direction as signal (gap>0 -> UP):")
    if n_pyth: print(f"    Pyth:    {pyth_ok}/{n_pyth} = {pyth_ok/n_pyth*100:.1f}%")
    if n_bin:  print(f"    Binance: {bin_ok}/{n_bin} = {bin_ok/n_bin*100:.1f}%")


# ================================================================
#  D. ENTRY TIMING (entry_pct windows)
# ================================================================

def sec_timing(ticks, outcomes, bet, cfg):
    S("D. ENTRY TIMING (entry_pct window)")
    if not outcomes: print("  No outcomes"); return

    outcome_map = {o["slug"]: o["outcome"] for o in outcomes}
    slug_ticks = defaultdict(list)
    for t in ticks: slug_ticks[t["slug"]].append(t)

    sp_lo, sp_hi = cfg["min_share_price"], cfg["max_share_price"]
    windows = [(0.0,0.10),(0.05,0.15),(0.10,0.20),(0.15,0.25),(0.20,0.30),
               (0.30,0.40),(0.40,0.50),(0.50,0.60),(0.60,0.70),(0.70,0.80),(0.80,0.90)]

    headers = ["Window", "N", "W", "L", "WR", "PnL", "AvgConf", "AvgSP"]
    widths  = [10, 5, 4, 4, 7, 9, 8, 7]
    aligns  = ['l', 'r', 'r', 'r', 'r', 'r', 'r', 'r']

    for model in [f"hermes_{cfg['primary_model']}", f"binance_{cfg['primary_model']}"]:
        rows = []
        for wlo, whi in windows:
            trades = _run_sim(slug_ticks, outcome_map, model, 0.60,
                              sp_lo, sp_hi, bet, 0.08, wlo, whi)
            if not trades: continue
            n = len(trades); w = sum(1 for t in trades if t["won"])
            pnl = sum(t["pnl"] for t in trades)
            rows.append([f"{wlo:.0%}-{whi:.0%}", n, w, n-w, f"{w/n*100:.1f}%",
                         f"${pnl:+.1f}",
                         f"{np.mean([t['conf'] for t in trades]):.0%}",
                         f"${np.mean([t['sp'] for t in trades]):.3f}"])
        box(f"{_ms(model)} -- Entry Timing (conf>=60%)", headers, rows, widths, aligns)


# ================================================================
#  E. LIVE TRADES (actual bot trades)
# ================================================================

def sec_live_trades(date, bet):
    S("E. LIVE BOT TRADES")
    trades = load_live_trades(date)
    if not trades:
        print("  No live trades found"); return

    dry = [t for t in trades if t.get("dry_run")]
    real = [t for t in trades if not t.get("dry_run")]
    print(f"  Total: {len(trades)} trades (dry_run={len(dry)}, real={len(real)})")

    for label, group in [("DRY-RUN", dry), ("REAL", real)]:
        if not group: continue
        n = len(group)
        w = sum(1 for t in group if t.get("won"))
        pnl = sum(t.get("pnl_usd", 0) for t in group)
        avg_conf = np.mean([t.get("confidence", 0) for t in group])
        avg_sp = np.mean([t.get("entry_price", 0) for t in group])
        avg_hold = np.mean([t.get("hold_time_s", 0) for t in group])

        print(f"\n  --- {label} ({n} trades) ---")
        print(f"  WR: {w}/{n} = {w/n*100:.1f}%  |  PnL: ${pnl:+.2f}  |  EV: ${pnl/n:+.2f}")
        print(f"  Avg conf: {avg_conf:.0%}  |  Avg SP: ${avg_sp:.3f}  |  Avg hold: {avg_hold:.0f}s")

        # Per-trade detail
        headers = ["#", "Slug", "Dir", "Conf", "SP", "Won", "PnL", "Hold"]
        widths  = [3, 34, 4, 5, 6, 3, 8, 6]
        aligns  = ['r', 'l', 'l', 'r', 'r', 'r', 'r', 'r']
        rows = []
        for i, t in enumerate(group):
            slug_short = t.get("slug", "")[-34:]
            ok = "W" if t.get("won") else "L"
            rows.append([i+1, slug_short, t.get("direction","?"),
                         f"{t.get('confidence',0):.0%}", f"${t.get('entry_price',0):.3f}",
                         ok, f"${t.get('pnl_usd',0):+.2f}", f"{t.get('hold_time_s',0):.0f}s"])
        box(f"{label} TRADES", headers, rows, widths, aligns)

    # Early exit simulation: what if exit at $0.90 share price
    print(f"\n  --- EARLY EXIT SIMULATION ---")
    print(f"  What if we sold at $0.90 share price instead of waiting for resolution?")
    for label, group in [("DRY-RUN", dry), ("REAL", real)]:
        if not group: continue
        early_pnl = 0
        normal_pnl = 0
        for t in group:
            sp = t.get("entry_price", 0.5)
            shares = t.get("shares", bet / sp if sp > 0 else 0)
            normal_pnl += t.get("pnl_usd", 0)
            if t.get("won"):
                # Would sell at 0.90 instead of ~1.0 (resolution)
                early_pnl += shares * (0.90 - sp)
            else:
                # Loss is the same (goes to 0)
                early_pnl += -bet
        n = len(group)
        w = sum(1 for t in group if t.get("won"))
        print(f"\n  {label}: {n} trades, {w}W {n-w}L")
        print(f"    Resolution exit: ${normal_pnl:+.2f} (${normal_pnl/n:+.2f}/trade)")
        print(f"    Exit at $0.90:   ${early_pnl:+.2f} (${early_pnl/n:+.2f}/trade)")
        print(f"    Difference:      ${early_pnl - normal_pnl:+.2f}")


# ================================================================
#  F. EARLY EXIT SIM (simulated trades, not just live)
# ================================================================

def sec_early_exit(ticks, outcomes, bet, cfg):
    S("F. EARLY EXIT SIMULATION (all simulated trades)")
    if not outcomes: print("  No outcomes"); return

    outcome_map = {o["slug"]: o["outcome"] for o in outcomes}
    slug_ticks = defaultdict(list)
    for t in ticks: slug_ticks[t["slug"]].append(t)

    model = f"hermes_{cfg['primary_model']}"
    sp_lo, sp_hi = cfg["min_share_price"], cfg["max_share_price"]

    print(f"  Model: {model} conf>=85% | Comparing resolution vs $0.90 exit\n")

    trades = _run_sim(slug_ticks, outcome_map, model, 0.85,
                      sp_lo, sp_hi, bet, 0.08,
                      cfg["min_entry_pct"], cfg["max_entry_pct"])
    if not trades:
        print("  No qualifying trades"); return

    n = len(trades); w = sum(1 for t in trades if t["won"])

    # Normal PnL (resolution: win -> $1.00, lose -> $0.00)
    normal_pnl = sum(t["pnl"] for t in trades)

    # Early exit PnL (win -> sell at $0.90, lose -> $0.00)
    early_pnl = sum(t["shares"] * (0.90 - t["sp"]) if t["won"] else -bet for t in trades)

    # Even earlier: $0.85
    early85_pnl = sum(t["shares"] * (0.85 - t["sp"]) if t["won"] else -bet for t in trades)

    headers = ["Exit", "PnL", "EV/trade", "EV/win", "Diff"]
    widths = [12, 10, 10, 10, 10]
    aligns = ['l', 'r', 'r', 'r', 'r']
    rows = [
        ["Resolution", f"${normal_pnl:+.1f}", f"${normal_pnl/n:+.2f}",
         f"${normal_pnl/w:+.2f}" if w else "-", "-"],
        ["Exit $0.90", f"${early_pnl:+.1f}", f"${early_pnl/n:+.2f}",
         f"${early_pnl/w:+.2f}" if w else "-", f"${early_pnl-normal_pnl:+.1f}"],
        ["Exit $0.85", f"${early85_pnl:+.1f}", f"${early85_pnl/n:+.2f}",
         f"${early85_pnl/w:+.2f}" if w else "-", f"${early85_pnl-normal_pnl:+.1f}"],
    ]
    box(f"{_ms(model)} -- {n} trades ({w}W {n-w}L)", headers, rows, widths, aligns)


# ================================================================
#  G. SUMMARY
# ================================================================

def sec_summary(ticks, outcomes, bet, cfg):
    S("G. SUMMARY")
    n_ticks = len(ticks)
    n_markets = len(set(t["slug"] for t in ticks))
    n_outcomes = len(outcomes)
    dur_min = (ticks[-1]["ts"] - ticks[0]["ts"]) / 60
    t0 = datetime.fromtimestamp(ticks[0]["ts"]).strftime("%H:%M:%S")
    t1 = datetime.fromtimestamp(ticks[-1]["ts"]).strftime("%H:%M:%S")

    print(f"  Recording: {n_ticks:,} ticks | {n_markets} markets | {n_outcomes} outcomes")
    print(f"  Duration: {dur_min:.0f}min ({t0} -> {t1})")

    primary = f"hermes_{cfg['primary_model']}"
    print(f"\n  Config: {primary} conf>={cfg['min_confidence']:.0%} "
          f"SP ${cfg['min_share_price']:.2f}-${cfg['max_share_price']:.2f}")

    if not outcomes: return
    outcome_map = {o["slug"]: o["outcome"] for o in outcomes}
    slug_ticks = defaultdict(list)
    for t in ticks: slug_ticks[t["slug"]].append(t)
    cols = get_model_cols(ticks)

    headers = ["Model", "N", "W", "WR", "PnL", "EV"]
    widths = [22, 4, 3, 6, 9, 7]
    aligns = ['l', 'r', 'r', 'r', 'r', 'r']
    rows = []

    sp_lo, sp_hi = cfg["min_share_price"], cfg["max_share_price"]
    for model in cols:
        trades = _run_sim(slug_ticks, outcome_map, model, cfg["min_confidence"],
                          sp_lo, sp_hi, bet, cfg["max_spread"],
                          cfg["min_entry_pct"], cfg["max_entry_pct"])
        if not trades: continue
        n = len(trades); w = sum(1 for t in trades if t["won"])
        pnl = sum(t["pnl"] for t in trades)
        rows.append([model, n, w, f"{w/n*100:.1f}%", f"${pnl:+.1f}", f"${pnl/n:+.2f}"])
    box(f"Config simulation (all gates)", headers, rows, widths, aligns)


# ================================================================
#  MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config()
    bet = cfg.get("bet", 2.85)

    date = args.date
    if not date:
        files = sorted(TICKS_DIR.glob("ticks_*.jsonl"))
        files = [f for f in files if "binance" not in f.name and "hermes" not in f.name]
        if files: date = files[-1].stem.replace("ticks_", "")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"deep_analysis_{date}.txt"
    tee = Tee(str(out_path))
    sys.stdout = tee

    print(f"{'#'*90}")
    print(f"  POLY-DESTROYER -- Deep Analysis (1:1 live logic)")
    print(f"  Date: {date} | Bet: ${bet:.2f}")
    print(f"  Output: {out_path}")
    print(f"{'#'*90}")

    ticks = load_ticks(date)
    outcomes = load_outcomes(date)
    if not ticks:
        print("  No ticks!"); return

    sec_model_conf(ticks, outcomes, bet, cfg)
    sec_sp_bins(ticks, outcomes, bet, cfg)
    sec_gap(ticks, outcomes, bet, cfg)
    sec_timing(ticks, outcomes, bet, cfg)
    sec_live_trades(date, bet)
    sec_early_exit(ticks, outcomes, bet, cfg)
    sec_summary(ticks, outcomes, bet, cfg)

    print(f"\n{'='*90}")
    print(f"  DONE -- saved to {out_path}")
    print(f"{'='*90}")

    tee.close()
    sys.stdout = tee.stdout


if __name__ == "__main__":
    main()
