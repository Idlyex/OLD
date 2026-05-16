"""
OVERNIGHT FARM ULTRA ANALYSIS — Real Polymarket Trades + Recorded Data
Analyzes 88 live trades (58W/30L, 66% WR) + 3.3h recorded market data.
Uses vectorized numpy/pandas for speed.
"""
import sys, os, json, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
warnings.filterwarnings("ignore")
P = print

# ═══════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════
def load_trades():
    """Load live trades from JSON (richest format)."""
    with open("results/ml_live_trades.json") as f:
        raw = json.load(f)
    df = pd.DataFrame(raw)
    # Parse market start ts from slug
    df["mkt_start_ts"] = df["slug"].str.split("-").str[-1].astype(float)
    df["dur_min"] = df["slug"].str.extract(r"(\d+)m").astype(float)
    df["mkt_end_ts"] = df["mkt_start_ts"] + df["dur_min"] * 60
    # Entry timing: how far into market
    df["entry_pct"] = np.where(
        df["mkt_end_ts"] > df["mkt_start_ts"],
        (df["entry_ts"] - df["mkt_start_ts"]) / (df["mkt_end_ts"] - df["mkt_start_ts"]),
        0
    )
    df["entry_pct"] = df["entry_pct"].clip(0, 1)
    # Gap magnitude
    df["gap_abs"] = np.abs(df["gap_at_entry"])
    # Favorable gap: gap supports our direction
    df["gap_favorable"] = np.where(
        df["direction"] == "UP",
        df["gap_at_entry"],
        -df["gap_at_entry"]
    )
    # EV = pnl_pct per trade
    df["ev_pct"] = df["pnl_pct"]
    # Profit factor per trade
    df["won"] = df["won"].astype(bool)
    return df


def load_recorded_snaps():
    """Load all recorded snapshots for replay."""
    dfs = {}
    for dur in ["5m", "15m"]:
        p = Path("data/recorded/shares/2026-05-05") / dur / "snapshots.parquet"
        if p.exists():
            dfs[dur] = pd.read_parquet(p)
    return dfs


def load_models():
    """Load ML models for replay."""
    import joblib
    md = Path("training/model_registry/latest")
    meta = json.loads((md / "meta.json").read_text())
    fn = meta["feature_names"]
    models = {}
    for name in ["lgbm", "catboost", "rf", "xgboost"]:
        mp = md / f"{name}_cls.pkl"
        if mp.exists():
            models[name] = joblib.load(mp)
    scaler = joblib.load(md / "scaler.pkl")
    return models, scaler, fn


# ═══════════════════════════════════════
# ANALYSIS FUNCTIONS (vectorized)
# ═══════════════════════════════════════
def section_overview(df):
    P("\n" + "=" * 90)
    P("  OVERNIGHT FARM — ULTRA ANALYSIS")
    P("=" * 90)
    t0 = datetime.fromtimestamp(df["entry_ts"].min(), tz=timezone.utc)
    t1 = datetime.fromtimestamp(df["exit_ts"].max(), tz=timezone.utc)
    P(f"  Period: {t0.strftime('%Y-%m-%d %H:%M')} -> {t1.strftime('%H:%M')} UTC ({(df.exit_ts.max()-df.entry_ts.min())/3600:.1f}h)")
    P(f"  Trades: {len(df)}  |  W/L: {df.won.sum()}/{(~df.won).sum()}  |  WR: {df.won.mean()*100:.1f}%")
    total_pnl = df["pnl_usd"].sum()
    P(f"  PnL: ${total_pnl:+.2f}  |  EV/trade: ${total_pnl/len(df):+.2f}  |  ROI: {total_pnl/100*100:.1f}%")
    P(f"  SOL range: ${df.sol_at_entry.min():.3f} - ${df.sol_at_entry.max():.3f}")
    P(f"  Directions: {(df.direction=='UP').sum()} UP / {(df.direction=='DOWN').sum()} DOWN")
    P(f"  Dry-run: {df.dry_run.sum()} / {len(df)}")
    # Streaks
    won_arr = df.sort_values("entry_ts")["won"].values
    max_w = max_l = cur_w = cur_l = 0
    for w in won_arr:
        if w:
            cur_w += 1; cur_l = 0
        else:
            cur_l += 1; cur_w = 0
        max_w = max(max_w, cur_w)
        max_l = max(max_l, cur_l)
    P(f"  Max win streak: {max_w}  |  Max loss streak: {max_l}")


def section_direction(df):
    P(f"\n{'─'*90}")
    P(f"  SECTION 1: DIRECTION BREAKDOWN")
    P(f"{'─'*90}")
    P(f"  {'Dir':<6} {'N':>4} {'W':>3} {'L':>3} {'WR':>6} {'PnL':>8} {'EV':>7} {'AvgConf':>8} {'AvgGap':>9}")
    for d in ["UP", "DOWN", "ALL"]:
        sub = df if d == "ALL" else df[df.direction == d]
        n = len(sub)
        if n == 0: continue
        w = sub.won.sum()
        wr = w / n * 100
        pnl = sub.pnl_usd.sum()
        ev = pnl / n
        avg_conf = sub.confidence.mean() * 100
        avg_gap = sub.gap_at_entry.mean()
        P(f"  {d:<6} {n:>4} {w:>3} {n-w:>3} {wr:>5.1f}% ${pnl:>+6.2f} ${ev:>+5.2f} {avg_conf:>7.1f}% {avg_gap:>+8.4f}")

    # UP wins vs DOWN wins detail
    P(f"\n  Direction performance detail:")
    for d in ["UP", "DOWN"]:
        sub = df[df.direction == d]
        wins = sub[sub.won]
        losses = sub[~sub.won]
        P(f"  {d}: avg_entry_price=${sub.entry_price.mean():.3f}, "
          f"avg_hold={sub.hold_time_s.mean():.0f}s, "
          f"win_avg_pnl=${wins.pnl_usd.mean():.2f}" if len(wins) else "",
          end="")
        if len(losses):
            P(f", loss_avg_pnl=${losses.pnl_usd.mean():.2f}")
        else:
            P()


def section_confidence(df):
    P(f"\n{'─'*90}")
    P(f"  SECTION 2: CONFIDENCE ANALYSIS")
    P(f"{'─'*90}")
    bins = [(0.55, 0.65), (0.65, 0.70), (0.70, 0.75), (0.75, 0.80), (0.80, 0.85), (0.85, 1.0)]
    P(f"  {'Conf Range':<12} {'N':>4} {'WR':>6} {'PnL':>8} {'EV':>7} {'AvgEP':>7}")
    for lo, hi in bins:
        sub = df[(df.confidence >= lo) & (df.confidence < hi)]
        if len(sub) == 0: continue
        n = len(sub)
        wr = sub.won.mean() * 100
        pnl = sub.pnl_usd.sum()
        ev = pnl / n
        avg_ep = sub.entry_price.mean()
        P(f"  {lo:.2f}-{hi:.2f}  {n:>4} {wr:>5.1f}% ${pnl:>+6.2f} ${ev:>+5.2f} ${avg_ep:>.3f}")

    P(f"\n  High-confidence (>=0.80) vs low (<0.70):")
    hi_c = df[df.confidence >= 0.80]
    lo_c = df[df.confidence < 0.70]
    if len(hi_c):
        P(f"    HIGH (>=80%): {len(hi_c)}t, WR={hi_c.won.mean()*100:.1f}%, EV=${hi_c.pnl_usd.mean():+.2f}")
    if len(lo_c):
        P(f"    LOW  (<70%):  {len(lo_c)}t, WR={lo_c.won.mean()*100:.1f}%, EV=${lo_c.pnl_usd.mean():+.2f}")


def section_entry_timing(df):
    P(f"\n{'─'*90}")
    P(f"  SECTION 3: ENTRY TIMING (% into market)")
    P(f"{'─'*90}")
    bins = [(0, 0.20), (0.20, 0.40), (0.40, 0.60), (0.60, 0.80), (0.80, 1.0)]
    P(f"  {'Timing':<12} {'N':>4} {'WR':>6} {'PnL':>8} {'EV':>7} {'AvgConf':>8} {'AvgEP':>7}")
    for lo, hi in bins:
        sub = df[(df.entry_pct >= lo) & (df.entry_pct < hi)]
        if len(sub) == 0: continue
        n = len(sub)
        wr = sub.won.mean() * 100
        pnl = sub.pnl_usd.sum()
        ev = pnl / n
        P(f"  {lo*100:.0f}-{hi*100:.0f}%     {n:>4} {wr:>5.1f}% ${pnl:>+6.2f} ${ev:>+5.2f} {sub.confidence.mean()*100:>7.1f}% ${sub.entry_price.mean():>.3f}")

    # Sweet spot analysis
    P(f"\n  SWEET SPOT: entry 60-80% + conf >= 0.75:")
    sweet = df[(df.entry_pct >= 0.60) & (df.entry_pct < 0.80) & (df.confidence >= 0.75)]
    if len(sweet):
        P(f"    {len(sweet)} trades, WR={sweet.won.mean()*100:.1f}%, EV=${sweet.pnl_usd.mean():+.2f}, "
          f"avg_EP=${sweet.entry_price.mean():.3f}")
    else:
        P(f"    No trades in this range")

    sweet2 = df[(df.entry_pct >= 0.40) & (df.entry_pct < 0.80) & (df.confidence >= 0.70)]
    if len(sweet2):
        P(f"  SWEET SPOT 2: entry 40-80% + conf >= 0.70:")
        P(f"    {len(sweet2)} trades, WR={sweet2.won.mean()*100:.1f}%, EV=${sweet2.pnl_usd.mean():+.2f}, "
          f"avg_EP=${sweet2.entry_price.mean():.3f}")


def section_share_price(df):
    P(f"\n{'─'*90}")
    P(f"  SECTION 4: SHARE PRICE ANALYSIS (entry_price = cost per share)")
    P(f"{'─'*90}")
    bins = [(0, 0.30), (0.30, 0.40), (0.40, 0.50), (0.50, 0.60), (0.60, 0.70), (0.70, 1.0)]
    P(f"  {'EP Range':<12} {'N':>4} {'WR':>6} {'PnL':>8} {'EV$':>7} {'EV%':>7} {'MaxPayout':>10}")
    for lo, hi in bins:
        sub = df[(df.entry_price >= lo) & (df.entry_price < hi)]
        if len(sub) == 0: continue
        n = len(sub)
        wr = sub.won.mean() * 100
        pnl = sub.pnl_usd.sum()
        ev = pnl / n
        ev_pct = sub.pnl_pct.mean()
        max_payout = (1.0 / sub.entry_price.mean() - 1) * 100
        P(f"  ${lo:.2f}-{hi:.2f}  {n:>4} {wr:>5.1f}% ${pnl:>+6.2f} ${ev:>+5.2f} {ev_pct:>+5.1f}% {max_payout:>+9.0f}%")

    # Low EP + High conf = best EV
    P(f"\n  LOW PRICE + HIGH CONF combos:")
    for ep_max in [0.40, 0.45, 0.50]:
        for conf_min in [0.70, 0.80]:
            sub = df[(df.entry_price <= ep_max) & (df.confidence >= conf_min)]
            if len(sub) >= 3:
                P(f"    EP<=${ep_max:.2f} + conf>={conf_min:.0%}: {len(sub)}t, "
                  f"WR={sub.won.mean()*100:.1f}%, EV=${sub.pnl_usd.mean():+.2f}, "
                  f"EV%={sub.pnl_pct.mean():+.1f}%")


def section_gap(df):
    P(f"\n{'─'*90}")
    P(f"  SECTION 5: GAP ANALYSIS (direction-relative)")
    P(f"{'─'*90}")
    P(f"  gap_favorable > 0 = gap supports our direction")
    P(f"  gap stats: mean={df.gap_favorable.mean():+.4f}, std={df.gap_favorable.std():.4f}")
    bins = [(-np.inf, -0.05), (-0.05, -0.02), (-0.02, 0.02), (0.02, 0.05), (0.05, np.inf)]
    labels = ["strong_against", "against", "neutral", "in_favor", "strong_favor"]
    P(f"  {'Gap Zone':<16} {'N':>4} {'WR':>6} {'PnL':>8} {'EV':>7}")
    for (lo, hi), lbl in zip(bins, labels):
        sub = df[(df.gap_favorable >= lo) & (df.gap_favorable < hi)]
        if len(sub) == 0: continue
        n = len(sub)
        wr = sub.won.mean() * 100
        pnl = sub.pnl_usd.sum()
        ev = pnl / n
        P(f"  {lbl:<16} {n:>4} {wr:>5.1f}% ${pnl:>+6.2f} ${ev:>+5.2f}")


def section_time_of_day(df):
    P(f"\n{'─'*90}")
    P(f"  SECTION 6: TIME-OF-DAY ANALYSIS")
    P(f"{'─'*90}")
    df_t = df.copy()
    df_t["hour"] = pd.to_datetime(df_t["entry_ts"], unit="s", utc=True).dt.hour
    P(f"  {'Hour':>4} {'N':>4} {'WR':>6} {'PnL':>8} {'EV':>7}")
    for h in sorted(df_t.hour.unique()):
        sub = df_t[df_t.hour == h]
        n = len(sub)
        wr = sub.won.mean() * 100
        pnl = sub.pnl_usd.sum()
        ev = pnl / n
        P(f"  {h:>4} {n:>4} {wr:>5.1f}% ${pnl:>+6.2f} ${ev:>+5.2f}")


def section_consecutive(df):
    P(f"\n{'─'*90}")
    P(f"  SECTION 7: STREAK & CONSECUTIVE ANALYSIS")
    P(f"{'─'*90}")
    df_s = df.sort_values("entry_ts")
    won_arr = df_s["won"].values
    # Compute streaks
    streaks_w = []
    streaks_l = []
    cur = 0
    for w in won_arr:
        if w:
            if cur < 0:
                streaks_l.append(abs(cur))
                cur = 0
            cur += 1
        else:
            if cur > 0:
                streaks_w.append(cur)
                cur = 0
            cur -= 1
    if cur > 0: streaks_w.append(cur)
    if cur < 0: streaks_l.append(abs(cur))

    P(f"  Win streaks: {sorted(streaks_w, reverse=True)[:10]}")
    P(f"  Loss streaks: {sorted(streaks_l, reverse=True)[:10]}")

    # After-loss analysis
    df_s = df_s.reset_index(drop=True)
    after_loss = []
    for i in range(1, len(df_s)):
        if not df_s.iloc[i-1]["won"]:
            after_loss.append(df_s.iloc[i]["won"])
    if after_loss:
        P(f"  After a LOSS: {sum(after_loss)}/{len(after_loss)} wins ({sum(after_loss)/len(after_loss)*100:.1f}%)")


def section_per_market_duration(df):
    P(f"\n{'─'*90}")
    P(f"  SECTION 8: 5m vs 15m MARKETS")
    P(f"{'─'*90}")
    P(f"  {'Dur':>4} {'N':>4} {'WR':>6} {'PnL':>8} {'EV':>7} {'AvgConf':>8} {'AvgEP':>7}")
    for dur in sorted(df.dur_min.unique()):
        sub = df[df.dur_min == dur]
        n = len(sub)
        if n == 0: continue
        wr = sub.won.mean() * 100
        pnl = sub.pnl_usd.sum()
        ev = pnl / n
        P(f"  {dur:>3.0f}m {n:>4} {wr:>5.1f}% ${pnl:>+6.2f} ${ev:>+5.2f} {sub.confidence.mean()*100:>7.1f}% ${sub.entry_price.mean():>.3f}")


def section_top_bottom(df):
    P(f"\n{'─'*90}")
    P(f"  SECTION 9: TOP 10 BEST & WORST TRADES")
    P(f"{'─'*90}")
    df_s = df.sort_values("pnl_usd", ascending=False)
    P(f"  TOP 10 (most profit):")
    P(f"  {'#':>2} {'Slug':<30} {'Dir':>4} {'Conf':>5} {'EP':>5} {'PnL$':>7} {'PnL%':>7} {'Entry%':>7} {'Gap':>8}")
    for i, (_, r) in enumerate(df_s.head(10).iterrows()):
        slug_short = r.slug.split("sol-updown-")[-1]
        P(f"  {i+1:>2} {slug_short:<30} {r.direction:>4} {r.confidence:>4.0%} ${r.entry_price:.2f} ${r.pnl_usd:>+5.2f} {r.pnl_pct:>+5.0f}% {r.entry_pct*100:>5.1f}% {r.gap_at_entry:>+7.4f}")

    P(f"\n  WORST 10 (biggest loss):")
    P(f"  {'#':>2} {'Slug':<30} {'Dir':>4} {'Conf':>5} {'EP':>5} {'PnL$':>7} {'PnL%':>7} {'Entry%':>7} {'Gap':>8}")
    for i, (_, r) in enumerate(df_s.tail(10).iloc[::-1].iterrows()):
        slug_short = r.slug.split("sol-updown-")[-1]
        P(f"  {i+1:>2} {slug_short:<30} {r.direction:>4} {r.confidence:>4.0%} ${r.entry_price:.2f} ${r.pnl_usd:>+5.2f} {r.pnl_pct:>+5.0f}% {r.entry_pct*100:>5.1f}% {r.gap_at_entry:>+7.4f}")


def section_hold_time(df):
    P(f"\n{'─'*90}")
    P(f"  SECTION 10: HOLD TIME ANALYSIS")
    P(f"{'─'*90}")
    bins = [(0, 30), (30, 60), (60, 120), (120, 180), (180, 300)]
    P(f"  {'Hold(s)':<12} {'N':>4} {'WR':>6} {'PnL':>8} {'EV':>7}")
    for lo, hi in bins:
        sub = df[(df.hold_time_s >= lo) & (df.hold_time_s < hi)]
        if len(sub) == 0: continue
        n = len(sub)
        wr = sub.won.mean() * 100
        P(f"  {lo}-{hi}s     {n:>4} {wr:>5.1f}% ${sub.pnl_usd.sum():>+6.2f} ${sub.pnl_usd.mean():>+5.2f}")


def section_optimal_filter(df):
    P(f"\n{'─'*90}")
    P(f"  SECTION 11: OPTIMAL FILTER SEARCH (what combos beat 66%?)")
    P(f"{'─'*90}")
    results = []
    for conf_min in [0.65, 0.70, 0.75, 0.80, 0.85]:
        for ep_max in [0.40, 0.45, 0.50, 0.55, 0.60]:
            for entry_lo in [0.0, 0.20, 0.40, 0.60]:
                for entry_hi in [0.60, 0.80, 1.0]:
                    if entry_hi <= entry_lo: continue
                    sub = df[
                        (df.confidence >= conf_min) &
                        (df.entry_price <= ep_max) &
                        (df.entry_pct >= entry_lo) &
                        (df.entry_pct < entry_hi)
                    ]
                    n = len(sub)
                    if n < 5: continue
                    wr = sub.won.mean() * 100
                    pnl = sub.pnl_usd.sum()
                    ev = pnl / n
                    results.append({
                        "conf": conf_min, "ep": ep_max, "entry_lo": entry_lo,
                        "entry_hi": entry_hi, "n": n, "wr": wr, "pnl": pnl, "ev": ev
                    })

    if not results:
        P("  No combos with N>=5"); return

    rdf = pd.DataFrame(results)

    # Top by WR
    P(f"\n  TOP 15 BY WIN RATE (N>=5):")
    P(f"  {'Conf':>5} {'MaxEP':>6} {'Entry':>10} {'N':>4} {'WR':>6} {'PnL':>8} {'EV':>7}")
    top_wr = rdf.nlargest(15, "wr")
    for _, r in top_wr.iterrows():
        P(f"  {r.conf:>4.0%} ${r.ep:.2f} {r.entry_lo*100:.0f}-{r.entry_hi*100:.0f}%    {r.n:>4} {r.wr:>5.1f}% ${r.pnl:>+6.2f} ${r.ev:>+5.2f}")

    # Top by EV
    P(f"\n  TOP 15 BY EV (N>=5):")
    P(f"  {'Conf':>5} {'MaxEP':>6} {'Entry':>10} {'N':>4} {'WR':>6} {'PnL':>8} {'EV':>7}")
    top_ev = rdf.nlargest(15, "ev")
    for _, r in top_ev.iterrows():
        P(f"  {r.conf:>4.0%} ${r.ep:.2f} {r.entry_lo*100:.0f}-{r.entry_hi*100:.0f}%    {r.n:>4} {r.wr:>5.1f}% ${r.pnl:>+6.2f} ${r.ev:>+5.2f}")

    # Top by PnL
    P(f"\n  TOP 10 BY TOTAL PNL (N>=5):")
    top_pnl = rdf.nlargest(10, "pnl")
    for _, r in top_pnl.iterrows():
        P(f"  conf>={r.conf:.0%} EP<=${r.ep:.2f} entry={r.entry_lo*100:.0f}-{r.entry_hi*100:.0f}%: "
          f"{r.n}t WR={r.wr:.1f}% PnL=${r.pnl:+.2f} EV=${r.ev:+.2f}")


def section_replay_sim(snap_dfs, models, scaler, fn):
    """Full replay simulation on recorded data."""
    P(f"\n{'─'*90}")
    P(f"  SECTION 12: FULL REPLAY SIMULATION (recorded data)")
    P(f"{'─'*90}")

    all_preds = []

    for dur_label, snap_df in snap_dfs.items():
        dur_min = int(dur_label.replace("m", ""))
        P(f"\n  --- {dur_label} markets ---")

        for slug, grp_all in snap_df.groupby("slug"):
            grp_all = grp_all.sort_values("ts")
            parts = slug.split("-")
            if len(parts) < 4: continue
            try:
                mkt_start = int(parts[-1])
                mkt_dur = int(parts[2].replace("m", ""))
            except ValueError:
                continue
            mkt_end = mkt_start + mkt_dur * 60

            grp = grp_all[(grp_all["ts"] >= mkt_start) & (grp_all["ts"] <= mkt_end)]
            if len(grp) < 5: continue
            coverage = (grp["ts"].max() - mkt_start) / (mkt_dur * 60)
            if coverage < 0.80: continue

            # PTB
            ptb_val = grp.iloc[0].get("price_to_beat", np.nan)
            if pd.isna(ptb_val) or ptb_val == 0:
                near_start = grp[(grp["ts"] >= mkt_start) & (grp["ts"] <= mkt_start + 10)]
                ptb_val = float(near_start.iloc[0]["sol_price"]) if len(near_start) > 0 else float(grp.iloc[0]["sol_price"])

            sol_end = float(grp.iloc[-1]["sol_price"])
            outcome = "UP" if sol_end >= ptb_val else "DOWN"

            # Predict at various entry points
            entry_fracs = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
            n = len(grp)

            for ef in entry_fracs:
                idx = min(int(n * ef), n - 1)
                row = grp.iloc[idx]
                sol_price = float(row.get("sol_price", 0))
                if sol_price <= 0 or ptb_val <= 0: continue

                # Build feature vector (vectorized where possible)
                feat = {}
                feat["distance_from_ptb_pct"] = (sol_price - ptb_val) / ptb_val * 100
                feat["distance_from_ptb_norm"] = (sol_price - ptb_val) / ptb_val

                for col in ["up_best_bid", "up_best_ask", "up_mid_price", "up_spread", "up_spread_pct",
                            "dn_best_bid", "dn_best_ask", "dn_mid_price", "dn_spread", "dn_spread_pct",
                            "up_bid_volume", "up_ask_volume", "dn_bid_volume", "dn_ask_volume",
                            "up_ob_imbalance", "dn_ob_imbalance",
                            "shares_momentum_30s", "shares_momentum_2m", "shares_acceleration",
                            "liquidity_score", "volume_spike", "volume_imbalance"]:
                    if col in grp.columns:
                        feat[col] = float(row.get(col, 0) or 0)

                elapsed_s = float(row["ts"]) - mkt_start
                total_s = mkt_dur * 60
                feat["time_elapsed_pct"] = min(elapsed_s / total_s, 1.0) if total_s > 0 else 0
                feat["time_remaining_pct"] = 1.0 - feat["time_elapsed_pct"]
                feat["life_phase"] = feat["time_elapsed_pct"]
                up_mid = feat.get("up_mid_price", 0.5)
                feat["up_implied_prob"] = up_mid if 0 < up_mid < 1 else 0.5

                fv = np.array([feat.get(f, 0.0) for f in fn], dtype=np.float64).reshape(1, -1)
                fv_s = scaler.transform(fv)

                preds = {}
                for name, model in models.items():
                    try:
                        if name == "lgbm":
                            preds[name] = float(model.predict(fv_s)[0])
                        else:
                            preds[name] = float(model.predict_proba(fv_s)[0, 1])
                    except Exception:
                        pass

                if not preds: continue
                ens = np.mean(list(preds.values()))
                preds["ensemble"] = ens

                for mn, prob in preds.items():
                    dp = max(prob, 1 - prob)
                    pred_dir = "UP" if prob > 0.5 else "DOWN"
                    won = pred_dir == outcome

                    up_ask = float(row.get("up_best_ask", 0.5) or 0.5)
                    dn_ask = float(row.get("dn_best_ask", 0.5) or 0.5)
                    sp = up_ask if pred_dir == "UP" else dn_ask

                    all_preds.append({
                        "slug": slug, "dur": dur_label, "entry_pct": ef,
                        "model": mn, "prob_up": prob, "conf": dp,
                        "pred_dir": pred_dir, "outcome": outcome, "won": won,
                        "share_price": sp, "sol": sol_price, "ptb": ptb_val,
                        "gap_pct": (sol_price - ptb_val) / ptb_val * 100,
                    })

    if not all_preds:
        P("  No predictions generated."); return

    pdf = pd.DataFrame(all_preds)
    pdf["won"] = pdf["won"].astype(bool)
    n_mkts = pdf["slug"].nunique()
    P(f"\n  Total: {len(pdf)} predictions across {n_mkts} markets")

    # Summary by model at entry 20%
    P(f"\n  --- MODEL ACCURACY @20% entry ---")
    P(f"  {'Model':<12} {'N':>4} {'WR':>6} {'AvgConf':>8}")
    for mn in ["lgbm", "catboost", "rf", "xgboost", "ensemble"]:
        sub = pdf[(pdf.model == mn) & (pdf.entry_pct == 0.20)]
        if len(sub) == 0: continue
        P(f"  {mn:<12} {len(sub):>4} {sub.won.mean()*100:>5.1f}% {sub.conf.mean()*100:>7.1f}%")

    # Simulate trades with filters
    P(f"\n  --- SIMULATED TRADES (conf>=0.65, EP<=$0.50) ---")
    P(f"  {'Model':<12} {'Entry':>5} {'N':>4} {'WR':>6} | {'Dir':>4} {'AvgEP':>6}")
    for mn in ["lgbm", "catboost", "rf", "xgboost", "ensemble"]:
        for ef in [0.20, 0.40, 0.60]:
            sub = pdf[(pdf.model == mn) & (pdf.entry_pct == ef) &
                      (pdf.conf >= 0.65) & (pdf.share_price <= 0.50)]
            if len(sub) < 3: continue
            n = len(sub)
            wr = sub.won.mean() * 100
            bias = "DN" if (sub.pred_dir == "DOWN").sum() > n * 0.6 else "UP" if (sub.pred_dir == "UP").sum() > n * 0.6 else "MX"
            P(f"  {mn:<12} {ef*100:>4.0f}% {n:>4} {wr:>5.1f}% | {bias:>4} ${sub.share_price.mean():>.3f}")

    # Optimal entry timing
    P(f"\n  --- OPTIMAL ENTRY TIMING (ensemble, conf>=0.70, EP<=$0.50) ---")
    P(f"  {'Entry%':>6} {'N':>4} {'WR':>6} {'AvgConf':>8} {'AvgEP':>6} {'AvgGap':>8}")
    for ef in [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        sub = pdf[(pdf.model == "ensemble") & (pdf.entry_pct == ef) &
                  (pdf.conf >= 0.70) & (pdf.share_price <= 0.50)]
        if len(sub) < 1: continue
        P(f"  {ef*100:>5.0f}% {len(sub):>4} {sub.won.mean()*100:>5.1f}% "
          f"{sub.conf.mean()*100:>7.1f}% ${sub.share_price.mean():>.3f} {sub.gap_pct.mean():>+7.4f}%")

    # Per-market detail for first 10 markets
    P(f"\n  --- PER-MARKET DETAIL (first 10 fully covered) ---")
    shown = 0
    for slug in sorted(pdf.slug.unique()):
        if shown >= 10: break
        mkt = pdf[pdf.slug == slug]
        outcome = mkt.iloc[0]["outcome"]
        ptb = mkt.iloc[0]["ptb"]
        P(f"\n  {slug}  outcome={outcome}  PTB=${ptb:.4f}")
        # Show ensemble at key timings
        for ef in [0.0, 0.20, 0.40, 0.60, 0.80]:
            ens = mkt[(mkt.model == "ensemble") & (mkt.entry_pct == ef)]
            if len(ens) == 0: continue
            r = ens.iloc[0]
            all_models_at = mkt[(mkt.entry_pct == ef) & (mkt.model != "ensemble")]
            model_str = " | ".join(
                f"{row.model}:{row.prob_up:.0%}"
                for _, row in all_models_at.iterrows()
            )
            P(f"    @{ef*100:.0f}%: ens={r.prob_up:.3f}({'UP' if r.prob_up>0.5 else 'DN'}) "
              f"conf={r.conf:.0%} EP=${r.share_price:.2f} gap={r.gap_pct:+.4f}% "
              f"{'WIN' if r.won else 'LOSS'} | {model_str}")
        shown += 1


def section_snapshot_features(snap_dfs):
    """Analyze recorded snapshot quality."""
    P(f"\n{'─'*90}")
    P(f"  SECTION 13: RECORDED DATA QUALITY")
    P(f"{'─'*90}")
    for dur, df in snap_dfs.items():
        n_mkts = df.slug.nunique()
        t0 = datetime.fromtimestamp(df.ts.min(), tz=timezone.utc)
        t1 = datetime.fromtimestamp(df.ts.max(), tz=timezone.utc)
        P(f"  {dur}: {len(df)} snaps, {n_mkts} markets, {t0.strftime('%H:%M')}-{t1.strftime('%H:%M')} UTC")
        # Data completeness
        key_cols = ["sol_price", "up_best_bid", "up_best_ask", "dn_best_bid", "dn_best_ask",
                    "up_ob_imbalance", "shares_momentum_30s"]
        for c in key_cols:
            if c in df.columns:
                pct_na = df[c].isna().mean() * 100
                pct_zero = (df[c] == 0).mean() * 100
                P(f"    {c}: {pct_na:.1f}% NA, {pct_zero:.1f}% zero")


# ═══════════════════════════════════════
# MAIN
# ═══════════════════════════════════════
def main():
    t0 = time.time()

    # Load trades
    df = load_trades()
    P(f"Loaded {len(df)} trades")

    # Run all sections
    section_overview(df)
    section_direction(df)
    section_confidence(df)
    section_entry_timing(df)
    section_share_price(df)
    section_gap(df)
    section_time_of_day(df)
    section_consecutive(df)
    section_per_market_duration(df)
    section_top_bottom(df)
    section_hold_time(df)
    section_optimal_filter(df)

    # Load recorded data + models for replay
    snap_dfs = load_recorded_snaps()
    if snap_dfs:
        section_snapshot_features(snap_dfs)
        try:
            models, scaler, fn = load_models()
            if fn:
                section_replay_sim(snap_dfs, models, scaler, fn)
        except Exception as e:
            P(f"  [!] Replay sim error: {e}")

    P(f"\n{'='*90}")
    P(f"  ANALYSIS COMPLETE — {time.time()-t0:.1f}s")
    P(f"{'='*90}")


if __name__ == "__main__":
    main()
