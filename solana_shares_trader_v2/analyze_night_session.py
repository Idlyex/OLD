"""
Run full analyze_live.py report suite on NIGHT SESSION data (23:00+ UTC+3).
Covers both 5min and 15min markets separately.

Usage: python analyze_night_session.py
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Import all report functions from analyze_live
import analyze_live as al

BET = 2.85
OUT_PATH = "results/analysis_night_session.txt"


def load_all_data():
    """Load all tick + outcome data, add hour info."""
    al.P("Loading all tick data...")
    ticks = al.load_ticks(all_days=True)
    outcomes = al.load_outcomes(all_days=True)
    df = al.merge_ticks_outcomes(ticks, outcomes)

    # Add market hour (UTC+3)
    df["_mkt_ts"] = df["slug"].str.split("-").str[-1].astype(int)
    df["_hour_local"] = (pd.to_datetime(df["_mkt_ts"], unit="s").dt.hour + 3) % 24

    # Add duration column if not present
    if "dur_min" not in df.columns:
        df["dur_min"] = df["slug"].apply(lambda s: 15 if "-15m-" in s else 5)

    return df


def main():
    al._OUT_FILE = open(OUT_PATH, "w", encoding="utf-8")

    al.P(f"\n{'='*110}")
    al.P(f"  NIGHT SESSION ANALYSIS (23:00-06:00 UTC+3)")
    al.P(f"  Full analyze_live.py report suite on night data")
    al.P(f"  Bet: ${BET:.2f}")
    al.P(f"{'='*110}")

    df = load_all_data()

    # Night filter: markets starting 23:00-05:59 UTC+3
    night_mask = (df["_hour_local"] >= 23) | (df["_hour_local"] < 6)
    night = df[night_mask].copy()

    # Also from 15:00 for broader context
    from15_mask = (df["_hour_local"] >= 15) | (df["_hour_local"] < 6)
    from15 = df[from15_mask].copy()

    # Split by duration
    night_5m = night[night["dur_min"] == 5].copy()
    night_15m = night[night["dur_min"] == 15].copy()
    from15_5m = from15[from15["dur_min"] == 5].copy()
    from15_15m = from15[from15["dur_min"] == 15].copy()

    al.P(f"\n  Night (23:00-06:00):")
    al.P(f"    ALL:  {night['slug'].nunique()} markets, {len(night)} ticks")
    al.P(f"    5min: {night_5m['slug'].nunique()} markets, {len(night_5m)} ticks")
    al.P(f"    15min: {night_15m['slug'].nunique()} markets, {len(night_15m)} ticks")
    al.P(f"\n  From 15:00:")
    al.P(f"    ALL:  {from15['slug'].nunique()} markets, {len(from15)} ticks")
    al.P(f"    5min: {from15_5m['slug'].nunique()} markets, {len(from15_5m)} ticks")
    al.P(f"    15min: {from15_15m['slug'].nunique()} markets, {len(from15_15m)} ticks")

    # ============================
    # SECTION A: NIGHT 5MIN
    # ============================
    al.P(f"\n\n{'#'*110}")
    al.P(f"  SECTION A: NIGHT 5MIN MARKETS ({night_5m['slug'].nunique()} markets)")
    al.P(f"{'#'*110}")

    al.report_overview(night_5m)
    al.report_threshold_table(night_5m, BET)
    al.report_fair_price_table(night_5m, BET)
    al.report_entry_timing(night_5m, BET)
    al.report_share_price_bins(night_5m, BET)
    al.report_direction(night_5m, BET)
    al.report_consensus(night_5m, BET)
    al.report_sp_sweep(night_5m, BET)
    al.report_best_combos(night_5m, BET)
    al.report_consensus_combos(night_5m, BET)
    al.report_entry_x_shareprice(night_5m, BET)
    al.report_hourly_performance(night_5m, BET)
    al.report_streak_analysis(night_5m, BET)
    al.report_optimal_config(night_5m, BET)
    al.report_executive_summary(night_5m, BET)

    # ============================
    # SECTION B: NIGHT 15MIN
    # ============================
    if night_15m['slug'].nunique() >= 3:
        al.P(f"\n\n{'#'*110}")
        al.P(f"  SECTION B: NIGHT 15MIN MARKETS ({night_15m['slug'].nunique()} markets)")
        al.P(f"{'#'*110}")

        al.report_overview(night_15m)
        al.report_threshold_table(night_15m, BET)
        al.report_fair_price_table(night_15m, BET)
        al.report_entry_timing(night_15m, BET)
        al.report_share_price_bins(night_15m, BET)
        al.report_direction(night_15m, BET)
        al.report_consensus(night_15m, BET)
        al.report_sp_sweep(night_15m, BET)
        al.report_best_combos(night_15m, BET)
        al.report_consensus_combos(night_15m, BET)
        al.report_entry_x_shareprice(night_15m, BET)
        al.report_streak_analysis(night_15m, BET)
        al.report_optimal_config(night_15m, BET)
        al.report_executive_summary(night_15m, BET)

    # ============================
    # SECTION C: NIGHT ALL (5m + 15m combined)
    # ============================
    al.P(f"\n\n{'#'*110}")
    al.P(f"  SECTION C: NIGHT ALL MARKETS ({night['slug'].nunique()} markets, 5m+15m)")
    al.P(f"{'#'*110}")

    al.report_overview(night)
    al.report_sp_sweep(night, BET)
    al.report_best_combos(night, BET)
    al.report_optimal_config(night, BET)
    al.report_executive_summary(night, BET)

    # ============================
    # SECTION D: FROM 15:00 (5min)
    # ============================
    al.P(f"\n\n{'#'*110}")
    al.P(f"  SECTION D: FROM 15:00 5MIN ({from15_5m['slug'].nunique()} markets)")
    al.P(f"{'#'*110}")

    al.report_overview(from15_5m)
    al.report_sp_sweep(from15_5m, BET)
    al.report_best_combos(from15_5m, BET)
    al.report_optimal_config(from15_5m, BET)
    al.report_executive_summary(from15_5m, BET)

    # ============================
    # SECTION E: FROM 15:00 (15min)
    # ============================
    if from15_15m['slug'].nunique() >= 3:
        al.P(f"\n\n{'#'*110}")
        al.P(f"  SECTION E: FROM 15:00 15MIN ({from15_15m['slug'].nunique()} markets)")
        al.P(f"{'#'*110}")

        al.report_overview(from15_15m)
        al.report_sp_sweep(from15_15m, BET)
        al.report_best_combos(from15_15m, BET)
        al.report_optimal_config(from15_15m, BET)
        al.report_executive_summary(from15_15m, BET)

    al.P(f"\n{'='*110}")
    al.P(f"  DONE — Saved to: {OUT_PATH}")
    al.P(f"{'='*110}\n")

    al._OUT_FILE.close()
    al._OUT_FILE = None
    print(f"\nFull report saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
