"""Run ALL 5 analysis scripts in sequence and generate a summary .txt report.

Usage:
    python tools/run_all_analysis.py               # auto-detect latest date
    python tools/run_all_analysis.py --date 2026-05-11
"""
import subprocess, sys, time, io
from pathlib import Path
from datetime import datetime

SCRIPTS = [
    ("1/5  analyze_live",      "tools/analyze_live.py",       "analyze_live_{date}.txt"),
    ("2/5  deep_analysis",     "tools/deep_analysis.py",      "deep_analysis_{date}.txt"),
    ("3/5  streak_analysis",   "tools/_streak_analysis.py",   "streak_analysis_{date}.txt"),
    ("4/5  streak_final",      "tools/_streak_final.py",      "streak_final_{date}.txt"),
    ("5/5  ultimate_analysis", "tools/_ultimate_analysis.py",  "ultimate_{date}_*.txt"),
]

def detect_date():
    files = sorted(Path("results/ticks").glob("ticks_*.jsonl"))
    files = [f for f in files if "binance" not in f.name and "hermes" not in f.name]
    if files:
        return files[-1].stem.replace("ticks_", "")
    return None


class Tee:
    def __init__(self, fp):
        self.file = open(fp, "w", encoding="utf-8")
        self.stdout = sys.stdout
    def write(self, s):
        self.stdout.write(s); self.file.write(s)
    def flush(self):
        self.stdout.flush(); self.file.flush()
    def close(self):
        self.file.close()


def main():
    date = None
    if "--date" in sys.argv:
        date = sys.argv[sys.argv.index("--date") + 1]
    if not date:
        date = detect_date()
    if not date:
        print("ERROR: No ticks files found in results/ticks/")
        sys.exit(1)

    out_dir = Path("results/analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"run_all_{date}_{datetime.now():%H%M}.txt"
    tee = Tee(str(report_path))
    sys.stdout = tee

    W = 90
    print("=" * W)
    print(f"  POLY-DESTROYER -- Full Analysis Suite")
    print(f"  Date: {date}")
    print(f"  Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  Scripts: {len(SCRIPTS)}")
    print("=" * W)

    # ── GLOSSARY ──
    print(f"""
{"#" * W}
  GLOSSARY & TERMINOLOGY
{"#" * W}

  DATA FILES:
    results/ticks/ticks_YYYY-MM-DD.jsonl     -- recorded tick data (all models)
    results/ticks/outcomes_YYYY-MM-DD.jsonl   -- market outcomes (UP/DOWN)
    results/ml_live_trades.json               -- actual bot live trades
    config/trading.yaml                       -- trading configuration

  KEY METRICS:
    N      = number of simulated trades (1 per market, first qualifying tick)
    W / L  = wins / losses
    WR     = win rate = W / N * 100%
    PnL    = total profit/loss in USD
    EV     = expected value per trade = PnL / N
    SP     = share price at entry ($0.01 - $0.99)
    Conf   = model confidence = max(prob, 1 - prob), range 0.50 - 1.00
    Streak = N consecutive confident ticks in same direction before entry
    PF     = profit factor = gross_profit / gross_loss
    MaxL   = max consecutive losses in a row

  PnL FORMULA:
    Win:  profit = (bet / SP) * (1 - SP)
          e.g. SP=$0.40, bet=$2.85 -> buy 7.13 shares -> resolve $1 -> profit $4.28
    Loss: loss = -bet = -$2.85 (shares go to $0)

  STREAK MODES (used in streak/ultimate analysis):
    STRICT = bad price or spread -> RESET streak to 0.
             Matches live trader _update_streak() exactly. Most conservative.
             Streak only builds when ALL conditions met on every tick.
    SKIP   = bad price or spread -> FREEZE streak (no build, no reset).
             Streak builds on ticks where conf+price+spread are ALL OK.
             But temporary price gaps don't break accumulated streak.
             Middle ground: more trades than STRICT at similar quality.
    SOFT   = streak builds on confidence+direction ONLY (ignores price).
             Entry happens when streak is met AND price+spread are OK.
             Most permissive: streak = pure model conviction signal.

  SWEET SPOT SCORE = EV * sqrt(N) * WR%
    Balances trade quality (EV, WR) with quantity (N).

  SCRIPTS:
    1. analyze_live.py     -- exhaustive live tick analysis: overview, models,
                              first-tick sim, timing, price bins, conf bins,
                              per-market detail, gap analysis
    2. deep_analysis.py    -- 1:1 live logic sim with price bins, gap, timing,
                              live trades review, early exit simulation, summary
    3. _streak_analysis.py -- streak-based sim with configurable streak lengths
                              and confidence thresholds, live trades streak check
    4. _streak_final.py    -- comprehensive streak sim with Mode A (strict/skip),
                              Mode B (conf-only streak + price gate), per-trade
                              details, live trades analysis
    5. _ultimate_analysis.py -- batched fast sim across ALL combinations of models,
                              streak lengths, confidence thresholds, price ranges.
                              Sections 1-10: STRICT mode analysis.
                              Sections 11-17: STRICT/SKIP/SOFT comparison,
                              exhaustive search, streak length tables, per-trade
                              details, final ranking, glossary.
""")

    # ── RUN SCRIPTS ──
    print(f"{"#" * W}")
    print(f"  RUNNING SCRIPTS")
    print(f"{"#" * W}")

    results = []
    t0_all = time.time()

    for label, script, out_pattern in SCRIPTS:
        print(f"\n{"-" * W}")
        print(f"  [{label}]  {script}")
        print(f"  Output: results/analysis/{out_pattern.format(date=date)}")
        print(f"{"-" * W}")

        t0 = time.time()
        cmd = [sys.executable, script, "--date", date]
        try:
            proc = subprocess.run(cmd, timeout=600, capture_output=False)
            elapsed = time.time() - t0
            status = "OK" if proc.returncode == 0 else f"FAIL (rc={proc.returncode})"
        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            status = "TIMEOUT"
        except Exception as e:
            elapsed = time.time() - t0
            status = f"ERROR: {e}"

        results.append((label, script, out_pattern, status, elapsed))
        print(f"  -> {status} ({elapsed:.1f}s)")

    total = time.time() - t0_all

    # ── SUMMARY ──
    print(f"\n{"=" * W}")
    print(f"  SUMMARY")
    print(f"{"=" * W}")
    ok_count = sum(1 for r in results if r[3] == "OK")
    print(f"  Passed: {ok_count}/{len(results)}")
    print()
    for label, script, out_pattern, status, elapsed in results:
        icon = "[OK]" if status == "OK" else "[FAIL]"
        print(f"  {icon} [{label}]  {elapsed:5.1f}s  {status}")
        print(f"         -> results/analysis/{out_pattern.format(date=date)}")
    print(f"\n  Total time: {total:.1f}s")
    print(f"  Report saved: {report_path}")

    # ── OUTPUT FILES LIST ──
    print(f"\n  Generated files in results/analysis/:")
    for f in sorted(out_dir.glob(f"*{date}*")):
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name:<50} {size_kb:6.1f} KB")

    print(f"\n{"=" * W}")

    tee.close()
    sys.stdout = tee.stdout
    print(f"\nDone! Report saved to {report_path}")


if __name__ == "__main__":
    main()
