"""
Full analysis: Model × Confidence × SP filter × Timing window
Uses first qualifying tick per market → GTC until market end (no 10s cancel)
Shows N, WR, EV/trade, total PnL
"""
import numpy as np
import pandas as pd
from pathlib import Path

results_dir = Path("results")
tick_files = sorted(results_dir.glob("live_ticks_*.jsonl"))
if not tick_files:
    print("No tick files found"); exit()
tick_file = tick_files[-1]  # latest
df = pd.read_json(tick_file, lines=True)
out_file = results_dir / tick_file.name.replace("live_ticks_", "live_outcomes_")
outcomes = pd.read_json(out_file, lines=True).drop_duplicates("slug", keep="last")
df = df.merge(outcomes[["slug", "outcome"]], on="slug", how="inner")

# 5min only
df5 = df[df["slug"].str.contains("-5m-")].copy()
n_markets = df5.slug.nunique()
print(f"Data: {len(df5)} ticks, {n_markets} markets (5min) from {tick_file.name}")

BET = 2.85
models = ["catboost", "rf", "xgboost", "ensemble"]
thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
sp_caps = [0.50, 0.55, 0.60, 0.65]
timing_windows = [(0.0, 1.0), (0.30, 1.0), (0.50, 1.0), (0.55, 0.92), (0.60, 1.0), (0.70, 1.0)]


def analyze_config(data, model, thresh, max_sp, t_lo, t_hi):
    if model == "ensemble":
        prob = data[["catboost", "rf", "xgboost"]].mean(axis=1).values
    else:
        prob = data[model].values
    conf = np.maximum(prob, 1 - prob)
    model_up = prob > 0.5
    sp = np.where(model_up, data["yes"].values, data["no"].values)
    entry_pct = data["entry_pct"].values

    mask = (conf >= thresh) & (sp <= max_sp) & (sp >= 0.10) & (entry_pct >= t_lo) & (entry_pct < t_hi)
    valid = data[mask].copy()
    if valid.empty:
        return None

    valid["_dir"] = np.where(model_up[mask], "UP", "DOWN")
    valid["_sp"] = sp[mask]
    valid = valid.sort_values("entry_pct").drop_duplicates("slug", keep="first")
    valid["won"] = valid["_dir"] == valid["outcome"]

    n = len(valid)
    w = int(valid["won"].sum())
    l = n - w
    wr = w / n * 100
    sp_arr = valid["_sp"].values
    shares = BET / sp_arr
    pnl_arr = np.where(valid["won"].values, shares * (1 - sp_arr), -BET)
    pnl = pnl_arr.sum()
    avg_sp = sp_arr.mean()
    ev_trade = pnl / n
    return n, w, l, wr, ev_trade, pnl, avg_sp


# ═══════════════════════════════════════════════════════════
# SECTION 1: Model × Confidence × Max Share Price (all timing)
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*100}")
print(f"  SECTION 1: PRICE FILTER — same WR, DIFFERENT PnL!")
print(f"  Lower entry → more profit per win ($0.45 entry wins $0.55 vs $0.55 entry wins $0.45)")
print(f"  Bet = ${BET} | Timing: 0-100%")
print(f"{'='*100}")

for model in models:
    print(f"\n  {model.upper()} — all timing (0-100%)")
    header = "  Conf  |"
    for sp in sp_caps:
        header += f"       SP<=${sp:.2f}        |"
    print(header)
    sub = "        |"
    for sp in sp_caps:
        sub += f"   N    WR      PnL  EV/t |"
    print(sub)
    print("  " + "-" * (len(header) - 2))

    for thresh in thresholds:
        row = f"  {thresh:.0%}  |"
        for max_sp in sp_caps:
            r = analyze_config(df5, model, thresh, max_sp, 0.0, 1.0)
            if r is None:
                row += "   —     —       —     — |"
            else:
                n, w, l, wr, ev_t, pnl, avg_sp = r
                row += f"  {n:>2}   {wr:>2.0f}% ${pnl:>+6.1f} ${ev_t:>+4.2f} |"
        print(row)

# ═══════════════════════════════════════════════════════════
# SECTION 2: TIMING × PRICE — catboost
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*100}")
print(f"  SECTION 2: TIMING × PRICE — CATBOOST")
print(f"  Each cell: N trades / WR% / total PnL (${BET}/trade)")
print(f"{'='*100}")

for max_sp in [0.50, 0.55, 0.60]:
    print(f"\n  CATBOOST | SP <= ${max_sp:.2f}")
    header = "  Conf  |"
    for t_lo, t_hi in timing_windows:
        header += f"     {int(t_lo*100)}-{int(t_hi*100)}%      |"
    print(header)
    sub = "        |"
    for _ in timing_windows:
        sub += f"   N  WR    PnL   EV |"
    print(sub)
    print("  " + "-" * (len(header) - 2))

    for thresh in thresholds:
        row = f"  {thresh:.0%}  |"
        for t_lo, t_hi in timing_windows:
            r = analyze_config(df5, "catboost", thresh, max_sp, t_lo, t_hi)
            if r is None:
                row += "   —   —      —    — |"
            else:
                n, w, l, wr, ev_t, pnl, avg_sp = r
                row += f"  {n:>2} {wr:>2.0f}% ${pnl:>+5.0f} ${ev_t:>+3.1f} |"
        print(row)

# ═══════════════════════════════════════════════════════════
# SECTION 3: ALL MODELS @ SP<=0.55, timing 50-100%
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*100}")
print(f"  SECTION 3: ALL MODELS @ SP<=0.55, timing 50-100%")
print(f"  Bet = ${BET}/trade")
print(f"{'='*100}")

for model in models:
    print(f"\n  {model.upper():<12} |  Conf |   N |   W |   L |     WR | EV/trade |      PnL | AvgSP")
    print(f"  {'-'*80}")
    for thresh in thresholds:
        r = analyze_config(df5, model, thresh, 0.55, 0.50, 1.0)
        if r is None:
            print(f"               |  {thresh:.0%} |   — |   — |   — |      — |        — |        — |     —")
        else:
            n, w, l, wr, ev_t, pnl, avg_sp = r
            print(f"               |  {thresh:.0%} | {n:>3} | {w:>3} | {l:>3} | {wr:>5.1f}% | ${ev_t:>+5.2f} | ${pnl:>+7.1f} | ${avg_sp:.3f}")

# ═══════════════════════════════════════════════════════════
# SECTION 4: YOUR LIVE CONFIG — catboost 70%, SP<=0.55, timing 55-92%
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*100}")
print(f"  SECTION 4: YOUR LIVE CONFIG — catboost 70%, SP<=0.55, timing 55-92%")
print(f"{'='*100}")
r = analyze_config(df5, "catboost", 0.70, 0.55, 0.55, 0.92)
if r:
    n, w, l, wr, ev_t, pnl, avg_sp = r
    print(f"  Result: {n} trades, {w}W/{l}L, WR={wr:.1f}%, EV/trade=${ev_t:+.2f}, PnL=${pnl:+.1f}, AvgSP=${avg_sp:.3f}")
else:
    print(f"  No trades matched!")

# Also check what live actually traded:
print(f"\n  Comparison with other configs at 55-92% timing:")
for model in ["catboost", "rf", "xgboost", "ensemble"]:
    for thresh in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        r = analyze_config(df5, model, thresh, 0.55, 0.55, 0.92)
        if r and r[0] >= 5:
            n, w, l, wr, ev_t, pnl, avg_sp = r
            print(f"    {model:<10} {thresh:.0%} | {n:>3}t {wr:>4.0f}% ${pnl:>+6.1f} EV/t=${ev_t:>+4.2f}")

# ═══════════════════════════════════════════════════════════
# FINAL VERDICT
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*100}")
print(f"  FINAL VERDICT: Best configs (N>=10, sorted by PnL)")
print(f"  Assumes GTC order lives until market end (no 10s cancel)")
print(f"{'='*100}")

results = []
for model in models:
    for thresh in thresholds:
        for max_sp in sp_caps:
            for t_lo, t_hi in timing_windows:
                r = analyze_config(df5, model, thresh, max_sp, t_lo, t_hi)
                if r and r[0] >= 10:
                    n, w, l, wr, ev_t, pnl, avg_sp = r
                    results.append((pnl, ev_t, wr, model, thresh, max_sp, t_lo, t_hi, n, w, l, avg_sp))

results.sort(key=lambda x: (-x[0], -x[2]))
print(f"\n  {'Model':<10} {'Conf':>4} {'SP<=':>5} {'Time':>8} | {'N':>3} {'W':>3} {'L':>3} {'WR':>4} | {'EV/t':>6} {'PnL':>8} | {'AvgSP':>5}")
print(f"  {'-'*80}")
for pnl, ev_t, wr, model, thresh, max_sp, t_lo, t_hi, n, w, l, avg_sp in results[:30]:
    t_str = f"{int(t_lo*100)}-{int(t_hi*100)}%"
    print(f"  {model:<10} {thresh:>3.0%} ${max_sp:.2f} {t_str:>8} | {n:>3} {w:>3} {l:>3} {wr:>3.0f}% | ${ev_t:>+5.2f} ${pnl:>+7.1f} | ${avg_sp:.3f}")
