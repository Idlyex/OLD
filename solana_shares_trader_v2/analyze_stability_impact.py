"""
Compare analysis results: standard (first qualifying tick) vs stability-required.
Shows how direction flips affect WR and whether requiring N stable ticks helps.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from analyze_live import load_ticks, load_outcomes, merge_ticks_outcomes

def analyze_with_stability(df, model="catboost", threshold=0.70, sp_max=0.55,
                           min_pct=0.55, max_pct=0.92, stability_n=3, dur=5):
    """Compare first-tick vs stability-required entry."""
    sub = df[df["dur_min"] == dur].copy()
    if sub.empty:
        return None

    sub["dir"] = np.where(sub[model] > 0.5, "UP", "DOWN")
    sub["conf"] = np.where(sub[model] > 0.5, sub[model], 1 - sub[model])

    results = {"standard": [], "stable": []}

    for slug in sub["slug"].unique():
        mkt = sub[sub["slug"] == slug].sort_values("entry_pct")
        outcome = mkt.iloc[0]["outcome"]

        # Window filter
        window = mkt[(mkt["entry_pct"] >= min_pct) & (mkt["entry_pct"] <= max_pct)]
        if window.empty:
            continue

        # === STANDARD: first tick with conf >= threshold and SP <= sp_max ===
        for _, row in window.iterrows():
            sp = row["yes"] if row["dir"] == "UP" else row["no"]
            if row["conf"] >= threshold and sp <= sp_max:
                won = (row["dir"] == outcome)
                pnl = (1 - sp) * 2.85 / sp if won else -2.85
                results["standard"].append({
                    "slug": slug, "dir": row["dir"], "conf": row["conf"],
                    "entry_pct": row["entry_pct"], "sp": sp,
                    "outcome": outcome, "won": won, "pnl": pnl,
                })
                break

        # === STABLE: first tick with conf >= threshold AND last N ticks agree on direction ===
        dirs_history = []
        for _, row in window.iterrows():
            dirs_history.append(row["dir"])
            sp = row["yes"] if row["dir"] == "UP" else row["no"]

            if len(dirs_history) >= stability_n:
                last_n = dirs_history[-stability_n:]
                all_same = all(d == last_n[-1] for d in last_n)
                # Average confidence over last N
                # (can't easily get from here, just use current conf)
                if all_same and row["conf"] >= threshold and sp <= sp_max:
                    won = (row["dir"] == outcome)
                    pnl = (1 - sp) * 2.85 / sp if won else -2.85
                    results["stable"].append({
                        "slug": slug, "dir": row["dir"], "conf": row["conf"],
                        "entry_pct": row["entry_pct"], "sp": sp,
                        "outcome": outcome, "won": won, "pnl": pnl,
                    })
                    break

    return results


def print_comparison(results, label):
    for mode in ["standard", "stable"]:
        trades = results[mode]
        if not trades:
            print(f"  {mode:>10}: no trades")
            continue
        n = len(trades)
        w = sum(1 for t in trades if t["won"])
        pnl = sum(t["pnl"] for t in trades)
        wr = w / n * 100
        print(f"  {mode:>10}: N={n:>3}  W={w:>3}  L={n-w:>3}  WR={wr:5.1f}%  PnL=${pnl:+7.1f}")


def main():
    ticks = load_ticks(None, True)
    outcomes = load_outcomes(None, True)
    df = merge_ticks_outcomes(ticks, outcomes)

    print(f"Data: {len(df)} ticks, {df.slug.nunique()} markets")
    print(f"  5min: {df[df.dur_min==5].slug.nunique()} markets")
    print(f"  15min: {df[df.dur_min==15].slug.nunique()} markets")

    # === Per-market direction flip analysis ===
    print(f"\n{'='*80}")
    print(f"  DIRECTION FLIP ANALYSIS (catboost, 55-92% window)")
    print(f"{'='*80}")

    model = "catboost"
    for dur in [5, 15]:
        sub = df[df.dur_min == dur].copy()
        flip_counts = []
        stable_counts = []
        for slug in sub.slug.unique():
            mkt = sub[sub.slug == slug].sort_values("entry_pct")
            window = mkt[(mkt.entry_pct >= 0.55) & (mkt.entry_pct <= 0.92)]
            if len(window) < 3:
                continue
            dirs = (window[model] > 0.5).values
            flips = sum(1 for i in range(1, len(dirs)) if dirs[i] != dirs[i-1])
            flip_counts.append(flips)
            stable_counts.append(1 if flips == 0 else 0)

        total = len(flip_counts)
        stable = sum(stable_counts)
        has_flips = total - stable
        avg_flips = np.mean(flip_counts) if flip_counts else 0
        print(f"\n  {dur}min markets ({total} total):")
        print(f"    Fully stable (0 flips): {stable} ({stable/total*100:.0f}%)" if total > 0 else "")
        print(f"    Has flips:              {has_flips} ({has_flips/total*100:.0f}%)" if total > 0 else "")
        print(f"    Avg flips per market:   {avg_flips:.1f}")
        if flip_counts:
            print(f"    Max flips:              {max(flip_counts)}")

    # === Standard vs Stable comparison ===
    print(f"\n{'='*80}")
    print(f"  STANDARD vs STABILITY-REQUIRED ENTRY")
    print(f"  (catboost, SP<=0.55, 55-92%)")
    print(f"{'='*80}")

    for dur in [5, 15]:
        for threshold in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
            for stab_n in [3]:
                results = analyze_with_stability(df, model="catboost", threshold=threshold,
                                                  stability_n=stab_n, dur=dur)
                if results is None:
                    continue
                std = results["standard"]
                stb = results["stable"]
                if not std:
                    continue

                n_std = len(std)
                w_std = sum(1 for t in std if t["won"])
                wr_std = w_std / n_std * 100 if n_std else 0
                pnl_std = sum(t["pnl"] for t in std)

                n_stb = len(stb)
                w_stb = sum(1 for t in stb if t["won"])
                wr_stb = w_stb / n_stb * 100 if n_stb else 0
                pnl_stb = sum(t["pnl"] for t in stb)

                delta_wr = wr_stb - wr_std
                delta_pnl = pnl_stb - pnl_std
                lost_trades = n_std - n_stb

                marker = " <<<" if delta_wr > 3 else (" ***" if delta_wr < -3 else "")
                print(f"  {dur}min CB>={threshold:.0%}: "
                      f"std N={n_std:>3} WR={wr_std:5.1f}% PnL=${pnl_std:+7.1f} | "
                      f"stable(3) N={n_stb:>3} WR={wr_stb:5.1f}% PnL=${pnl_stb:+7.1f} | "
                      f"dWR={delta_wr:+5.1f}pp dPnL=${delta_pnl:+6.1f} lost={lost_trades}{marker}")

    # === Check: do flipped markets have worse outcomes? ===
    print(f"\n{'='*80}")
    print(f"  FLIP vs STABLE MARKET OUTCOMES (catboost 70%, 5min)")
    print(f"{'='*80}")

    sub = df[df.dur_min == 5].copy()
    model = "catboost"
    threshold = 0.70

    stable_mkts = []
    flip_mkts = []

    for slug in sub.slug.unique():
        mkt = sub[sub.slug == slug].sort_values("entry_pct")
        outcome = mkt.iloc[0]["outcome"]
        window = mkt[(mkt.entry_pct >= 0.55) & (mkt.entry_pct <= 0.92)]
        if len(window) < 3:
            continue

        dirs = (window[model] > 0.5).values
        flips = sum(1 for i in range(1, len(dirs)) if dirs[i] != dirs[i-1])

        # Find first qualifying tick (standard)
        for _, row in window.iterrows():
            d = "UP" if row[model] > 0.5 else "DOWN"
            conf = max(row[model], 1 - row[model])
            sp = row["yes"] if d == "UP" else row["no"]
            if conf >= threshold and sp <= 0.55:
                won = (d == outcome)
                entry = {"slug": slug, "won": won, "flips": flips, "conf": conf}
                if flips == 0:
                    stable_mkts.append(entry)
                else:
                    flip_mkts.append(entry)
                break

    if stable_mkts:
        w = sum(1 for m in stable_mkts if m["won"])
        print(f"  Stable markets (0 flips): N={len(stable_mkts)}, W={w}, WR={w/len(stable_mkts)*100:.1f}%")
    if flip_mkts:
        w = sum(1 for m in flip_mkts if m["won"])
        print(f"  Flip markets (1+ flips):  N={len(flip_mkts)}, W={w}, WR={w/len(flip_mkts)*100:.1f}%")
        print(f"  Avg flips in flip group:  {np.mean([m['flips'] for m in flip_mkts]):.1f}")


if __name__ == "__main__":
    main()
