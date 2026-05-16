"""REPLAY V2 — Full-fidelity ML simulation on recorded Polymarket data.

Downloads matching Binance klines → computes ALL 114 features at every tick
→ batch ML predictions → vectorized trade simulation across all parameter combos.

Usage:
    python replay_v2.py                          # latest day, auto-download binance
    python replay_v2.py --date 2026-05-05
    python replay_v2.py --date 2026-05-05 --dur 5
"""
import sys, os, json, time, argparse, warnings, asyncio, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime, timezone
from scipy.stats import norm
warnings.filterwarnings("ignore")

P = print
SEP = "-" * 90

# ═══════════════════════════════════════
# CLI
# ═══════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="Replay V2 — full-fidelity ML sim")
    p.add_argument("--date", type=str, default=None)
    p.add_argument("--dur", type=int, default=None, help="5 or 15")
    p.add_argument("--bet", type=float, default=2.0)
    p.add_argument("--skip-download", action="store_true", help="use cached binance data")
    return p.parse_args()


# ═══════════════════════════════════════
# 1. LOAD RECORDED DATA
# ═══════════════════════════════════════
def load_snapshots(date_str=None, dur_filter=None):
    base = Path("data/recorded/shares")
    if date_str is None:
        days = sorted(d for d in base.iterdir() if d.is_dir())
        if not days: raise FileNotFoundError("No recorded data")
        date_str = days[-1].name
    day_dir = base / date_str
    snaps = {}
    for dur_dir in sorted(day_dir.iterdir()):
        if not dur_dir.is_dir(): continue
        pq = dur_dir / "snapshots.parquet"
        if not pq.exists(): continue
        dur = int(dur_dir.name.replace("m", ""))
        if dur_filter and dur != dur_filter: continue
        snaps[dur] = pd.read_parquet(pq)
    return snaps, date_str


def extract_markets(snap_df, dur_min):
    """Extract markets with outcome from recorded snapshots."""
    markets = []
    for slug, grp in snap_df.groupby("slug"):
        grp = grp.sort_values("ts")
        parts = slug.split("-")
        if len(parts) < 4: continue
        try:
            mkt_start = int(parts[-1])
            mkt_dur = int(parts[2].replace("m", ""))
        except ValueError:
            continue
        mkt_end = mkt_start + mkt_dur * 60
        g = grp[(grp.ts >= mkt_start) & (grp.ts <= mkt_end)]
        if len(g) < 5: continue
        cov = (g.ts.max() - mkt_start) / (mkt_dur * 60)
        if cov < 0.80: continue
        ptb_val = g.iloc[0].get("price_to_beat", np.nan)
        if pd.isna(ptb_val) or ptb_val == 0:
            near = g[(g.ts >= mkt_start) & (g.ts <= mkt_start + 10)]
            ptb_val = float(near.iloc[0].sol_price) if len(near) else float(g.iloc[0].sol_price)
        sol_end = float(g.iloc[-1].sol_price)
        outcome = "UP" if sol_end >= ptb_val else "DOWN"
        markets.append({
            "slug": slug, "dur": mkt_dur, "ptb": ptb_val,
            "sol_start": float(g.iloc[0].sol_price), "sol_end": sol_end,
            "outcome": outcome, "start_ts": float(mkt_start), "end_ts": float(mkt_end),
            "n_snaps": len(g), "coverage": cov,
            "snaps": g.reset_index(drop=True),
        })
    return markets


# ═══════════════════════════════════════
# 2. DOWNLOAD BINANCE KLINES FOR WINDOW
# ═══════════════════════════════════════
async def download_binance_window(ts_min, ts_max):
    """Download 1m klines covering [ts_min - 30min, ts_max + 5min]."""
    import httpx
    start_ms = int((ts_min - 1800) * 1000)
    end_ms = int((ts_max + 300) * 1000)
    cache_dir = Path("data/replay_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"klines_{start_ms}_{end_ms}.parquet"
    if cache_file.exists():
        P(f"  [cache] Loading {cache_file.name}")
        return pd.read_parquet(cache_file)

    P(f"  Downloading Binance 1m klines for replay window...")
    rows = []
    cur = start_ms
    async with httpx.AsyncClient(timeout=30) as client:
        while cur < end_ms:
            r = await client.get("https://fapi.binance.com/fapi/v1/klines", params={
                "symbol": "SOLUSDT", "interval": "1m",
                "startTime": cur, "endTime": end_ms, "limit": 1500
            })
            data = r.json()
            if not data or not isinstance(data, list): break
            for c in data:
                rows.append({
                    "ts_ms": int(c[0]), "open": float(c[1]), "high": float(c[2]),
                    "low": float(c[3]), "close": float(c[4]), "volume": float(c[5]),
                    "taker_buy_volume": float(c[9]),
                })
            cur = int(data[-1][0]) + 60_000
            if len(data) < 1500: break
            await asyncio.sleep(0.15)

    if not rows:
        P("  WARNING: No klines downloaded")
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates("ts_ms").sort_values("ts_ms").reset_index(drop=True)
    df["ts"] = df["ts_ms"] / 1000.0
    df.to_parquet(cache_file, compression="snappy")
    P(f"  Downloaded {len(df)} klines -> {cache_file.name}")
    return df


# ═══════════════════════════════════════
# 3. VECTORIZED FEATURE ENGINEERING
# ═══════════════════════════════════════
def build_ohlcv_at_ts(klines_df, snap_ts, lookback=61):
    """Get OHLCV arrays for klines up to snap_ts. Returns dict or None."""
    mask = klines_df.ts.values <= snap_ts
    n = mask.sum()
    if n < 20: return None
    start = max(0, n - lookback)
    sl = klines_df.iloc[start:n]
    return {
        "open": sl.open.values.astype(np.float64),
        "high": sl.high.values.astype(np.float64),
        "low": sl.low.values.astype(np.float64),
        "close": sl.close.values.astype(np.float64),
        "volume": sl.volume.values.astype(np.float64),
        "taker_buy_volume": sl.taker_buy_volume.values.astype(np.float64),
    }


def compute_all_features(ohlcv, snap_row, ptb, mkt_dur, mkt_start, sol_vol=0.003):
    """Compute all 114 features using real feature engine + snapshot data."""
    from core.features.price_volume import PriceVolumeFeatures
    from core.features.technical import TechnicalFeatures
    from core.features.microstructure import MicrostructureFeatures
    from core.features.liquidation_funding import LiquidationFundingFeatures
    from core.features.regime import RegimeFeatures
    from core.features.shares import compute_shares_features

    sol_price = float(snap_row.get("sol_price", ohlcv["close"][-1]))
    features = {}

    # Block 1: Price/Volume (18 features)
    pv = PriceVolumeFeatures()
    features.update(pv.compute(ohlcv_1m=ohlcv, current_price=sol_price))

    # Block 2: Technical (14 features)
    tech = TechnicalFeatures()
    features.update(tech.compute(ohlcv=ohlcv, current_price=sol_price))

    # Block 3: Microstructure (22 features) — defaults (no live depth)
    micro = MicrostructureFeatures()
    features.update(micro.compute(current_price=sol_price))

    # Block 4: Liquidation/Funding (8 features) — defaults
    liq = LiquidationFundingFeatures()
    features.update(liq.compute(funding_rate=0, current_price=sol_price))

    # Block 5: On-chain (10 features) — zeros
    for k in ["onchain_large_transfers_60s", "onchain_whale_activity",
              "onchain_dex_volume_spike", "onchain_jupiter_accel",
              "onchain_mev_bundles", "onchain_priority_fee_pressure",
              "onchain_token_creation_rate", "onchain_large_transfers_300s",
              "onchain_dex_volume_spike_300s", "onchain_jupiter_accel_300s"]:
        features[k] = 0.0

    # Block 6: Regime (10 features)
    regime = RegimeFeatures()
    close_arr = ohlcv["close"]
    rets = np.diff(np.log(close_arr[close_arr > 0])) if len(close_arr) > 10 else None
    features.update(regime.compute(returns=rets, close_prices=close_arr))

    # Shares features (16 features)
    elapsed_s = float(snap_row.get("ts", mkt_start)) - mkt_start
    total_s = mkt_dur * 60
    t_rem_ms = max(0, (total_s - elapsed_s)) * 1000
    t_elap_ms = elapsed_s * 1000

    dist = (sol_price - ptb) / ptb if ptb > 0 else 0
    vol_adj = max(sol_vol, 0.001) * np.sqrt(max(t_rem_ms / 60_000, 0.01))
    d = dist / vol_adj
    yes_price = float(np.clip(norm.cdf(d), 0.02, 0.98))

    shares_feats = compute_shares_features(
        sol_price=sol_price, price_to_beat=ptb,
        yes_price=yes_price, no_price=1 - yes_price,
        time_remaining_ms=int(t_rem_ms), time_elapsed_ms=int(t_elap_ms),
        duration_minutes=mkt_dur, sol_volatility=sol_vol,
    )
    features.update(shares_feats)

    # Pre-market lookback (from kline buffer)
    closes = ohlcv["close"]
    for lb in [2, 5, 10, 15, 30]:
        kr, kv = f"pre_mkt_ret_{lb}m", f"pre_mkt_vol_{lb}m"
        if len(closes) >= lb + 1:
            seg = closes[-(lb + 1):-1]
            features[kr] = (seg[-1] - seg[0]) / (seg[0] + 1e-10)
            lr = np.diff(np.log(seg + 1e-10))
            features[kv] = float(np.std(lr)) if len(lr) > 1 else 0.0
        else:
            features[kr] = 0.0; features[kv] = 0.0

    features.setdefault("oi_change", 0.0)
    features.setdefault("long_short_ratio", 1.0)
    return features


# ═══════════════════════════════════════
# 4. BATCH ML PREDICTIONS
# ═══════════════════════════════════════
def load_models():
    md = Path("training/model_registry/latest")
    meta = json.loads((md / "meta.json").read_text())
    fn = meta["feature_names"]
    models = {}
    for name in ["lgbm", "catboost", "rf", "xgboost"]:
        mp = md / f"{name}_cls.pkl"
        if mp.exists(): models[name] = joblib.load(mp)
    scaler = joblib.load(md / "scaler.pkl")
    return models, scaler, fn


def predict_batch(X, models, scaler):
    """Batch predict with all models. X: (N, n_features). Returns dict of name -> (N,) probs."""
    X_s = scaler.transform(X)
    preds = {}
    for name, model in models.items():
        if hasattr(model, 'predict_proba'):
            preds[name] = model.predict_proba(X_s)[:, 1].astype(np.float64)
        else:
            preds[name] = model.predict(X_s).astype(np.float64)
    # Ensemble = mean of all
    if len(preds) >= 2:
        preds["ensemble"] = np.mean(list(preds.values()), axis=0)
    return preds


# ═══════════════════════════════════════
# 5. VECTORIZED TRADE SIMULATION
# ═══════════════════════════════════════
def simulate_trades(tick_df, bet=2.0):
    """Vectorized trade simulation across all parameter combos.

    tick_df: DataFrame with columns per tick:
        slug, ts, entry_pct, outcome, sol_price, ptb, gap_pct,
        up_ask, dn_ask, + model prob columns (lgbm, catboost, rf, xgboost, ensemble)

    Returns: DataFrame of trade results for all combos.
    """
    model_cols = [c for c in tick_df.columns if c in ["lgbm", "catboost", "rf", "xgboost", "ensemble"]]
    if not model_cols:
        return pd.DataFrame()

    confs = np.array([0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90])
    ep_maxs = np.array([0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60])
    entry_pct_bins = np.array([0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80])

    # For each market, pick ONE tick per entry_pct bin (closest)
    # This avoids counting multiple signals in same market
    results = []

    for slug, mkt_ticks in tick_df.groupby("slug"):
        outcome = mkt_ticks.iloc[0]["outcome"]
        actual_up = outcome == "UP"
        n = len(mkt_ticks)
        ep_vals = mkt_ticks["entry_pct"].values

        for ep_target in entry_pct_bins:
            # Find closest tick to target entry_pct
            diffs = np.abs(ep_vals - ep_target)
            idx = np.argmin(diffs)
            if diffs[idx] > 0.05:  # skip if no tick within 5%
                continue
            tick = mkt_ticks.iloc[idx]

            for mc in model_cols:
                prob = tick[mc]
                if np.isnan(prob): continue
                conf = max(prob, 1 - prob)
                pred_dir = "UP" if prob > 0.5 else "DOWN"
                won = (pred_dir == "UP" and actual_up) or (pred_dir == "DOWN" and not actual_up)

                # Share price at this tick — REAL price, skip if missing
                sp_raw = tick["up_ask"] if pred_dir == "UP" else tick["dn_ask"]
                if pd.isna(sp_raw) or sp_raw <= 0 or sp_raw >= 1:
                    continue  # no valid price — skip (never fake $0.50)
                sp = float(sp_raw)

                # Liquidity check
                spread = float(tick["up_spread"] if pred_dir == "UP" else tick["dn_spread"]) if not pd.isna(tick.get("up_spread")) else 1.0
                depth5 = float(tick["up_depth5"] if pred_dir == "UP" else tick["dn_depth5"]) if "up_depth5" in tick.index else 0
                vol = float(tick["up_vol"] if pred_dir == "UP" else tick["dn_vol"]) if "up_vol" in tick.index else 0
                shares_needed = bet / sp
                can_fill = depth5 >= shares_needed

                for c_thresh in confs:
                    if conf < c_thresh: continue
                    for ep_max in ep_maxs:
                        if sp > ep_max: continue

                        shares = bet / sp
                        pnl = shares * (1.0 - sp) if won else -bet

                        results.append({
                            "slug": slug, "model": mc, "entry_pct": ep_target,
                            "conf_thresh": c_thresh, "ep_max": ep_max,
                            "prob": prob, "conf": conf, "pred_dir": pred_dir,
                            "outcome": outcome, "won": won,
                            "share_price": sp, "pnl": round(pnl, 4),
                            "gap_pct": tick["gap_pct"],
                            "sol_price": tick["sol_price"],
                            "spread": spread, "depth5": depth5,
                            "can_fill": can_fill, "ask_vol": vol,
                        })

    return pd.DataFrame(results)


def simulate_live_trader(tick_df, bet=2.0, min_conf=0.60, max_sp=0.55, max_entry_pct=0.80):
    """Simulate EXACTLY what the live trader does:
    - catboost primary (fallback lgbm)
    - ONE entry per market at FIRST valid tick (5%-80%)
    - conf >= threshold AND share_price <= max_sp
    """
    primary = "catboost" if "catboost" in tick_df.columns else "lgbm"
    fallback = "lgbm" if primary == "catboost" and "lgbm" in tick_df.columns else None
    model_cols = [c for c in tick_df.columns if c in ["lgbm", "catboost", "rf", "xgboost", "ensemble"]]
    
    results = []
    skipped_conf = 0
    skipped_price = 0
    skipped_no_data = 0
    
    for slug, mkt_ticks in tick_df.groupby("slug"):
        mkt_ticks = mkt_ticks.sort_values("entry_pct")
        outcome = mkt_ticks.iloc[0]["outcome"]
        actual_up = outcome == "UP"
        entered = False
        
        for _, tick in mkt_ticks.iterrows():
            ep = tick["entry_pct"]
            if ep < 0.05 or ep > max_entry_pct:
                continue
            
            # Get catboost probability (fallback to lgbm)
            prob = tick.get(primary, np.nan)
            if pd.isna(prob) and fallback:
                prob = tick.get(fallback, np.nan)
            if pd.isna(prob):
                skipped_no_data += 1
                continue
            
            conf = max(prob, 1 - prob)
            pred_up = prob > 0.5
            pred_dir = "UP" if pred_up else "DOWN"
            
            # Confidence filter
            if conf < min_conf:
                continue  # keep trying later ticks (like live re-eval)
            
            # Share price
            sp_raw = tick["up_ask"] if pred_up else tick["dn_ask"]
            if pd.isna(sp_raw) or sp_raw <= 0 or sp_raw >= 1:
                skipped_no_data += 1
                continue
            sp = float(sp_raw)
            
            # Price filter (live: queue and retry if too high)
            if sp > max_sp:
                skipped_price += 1
                continue  # try next tick (like live re-eval)
            
            # ENTER — first valid tick
            won = (pred_dir == "UP" and actual_up) or (pred_dir == "DOWN" and not actual_up)
            shares = bet / sp
            pnl = shares * (1.0 - sp) if won else -bet
            
            # Count model agreement
            n_agree = 0
            for mc in model_cols:
                if mc == "ensemble":
                    continue
                mc_prob = tick.get(mc, np.nan)
                if pd.isna(mc_prob):
                    continue
                mc_up = mc_prob > 0.5
                if mc_up == pred_up:
                    n_agree += 1
            
            results.append({
                "slug": slug, "entry_pct": ep, "outcome": outcome,
                "pred_dir": pred_dir, "conf": conf, "won": won,
                "share_price": sp, "pnl": round(pnl, 4),
                "sol_price": tick["sol_price"], "gap_pct": tick["gap_pct"],
                "primary_model": primary,
                "models_agree": n_agree,
                "n_models": len([c for c in model_cols if c != "ensemble"]),
                "all_probs": {mc: round(float(tick.get(mc, 0)), 3) for mc in model_cols},
            })
            entered = True
            break  # ONE entry per market
        
        if not entered:
            skipped_conf += 1
    
    rdf = pd.DataFrame(results)
    return rdf, skipped_conf, skipped_price, skipped_no_data


def report_live_simulation(rdf, skipped_conf, skipped_price, skipped_no_data, n_markets,
                           min_conf, max_sp, bet):
    """Report for live trader simulation."""
    P(f"\n{'='*90}")
    P(f"  SECTION 15: LIVE TRADER SIMULATION (mirrors exact live logic)")
    P(f"{'='*90}")
    P(f"  Config: catboost primary, conf>={min_conf:.0%}, SP<=${max_sp:.2f}, bet=${bet:.2f}")
    P(f"  Markets: {n_markets} total")
    P(f"  Entered: {len(rdf)} trades ({len(rdf)/n_markets*100:.0f}% trigger rate)")
    P(f"  Skipped: {skipped_conf} (never hit conf), {skipped_price} ticks (price too high), {skipped_no_data} (no data)")
    
    if rdf.empty:
        P("  No trades."); return
    
    w = rdf.won.sum(); l = len(rdf) - w
    wr = w / len(rdf) * 100
    pnl = rdf.pnl.sum()
    P(f"\n  RESULTS: {w}W / {l}L ({wr:.1f}% WR)")
    P(f"  Total PnL: ${pnl:+.2f}  |  EV/trade: ${rdf.pnl.mean():+.2f}")
    P(f"  Avg entry price: ${rdf.share_price.mean():.3f}")
    P(f"  Avg entry timing: {rdf.entry_pct.mean()*100:.0f}%")
    P(f"  Avg confidence: {rdf.conf.mean()*100:.0f}%")
    
    # Model agreement breakdown
    P(f"\n  MODEL AGREEMENT:")
    for n in sorted(rdf.models_agree.unique()):
        sub = rdf[rdf.models_agree == n]
        sw = sub.won.sum(); sl = len(sub) - sw
        P(f"    {n}/{sub.n_models.iloc[0]} agree: {len(sub)} trades, "
          f"{sw}W/{sl}L ({sw/len(sub)*100:.0f}% WR), PnL=${sub.pnl.sum():+.2f}")
    
    # Entry timing breakdown
    P(f"\n  BY ENTRY TIMING:")
    for lo, hi in [(0.05, 0.20), (0.20, 0.40), (0.40, 0.60), (0.60, 0.80)]:
        sub = rdf[(rdf.entry_pct >= lo) & (rdf.entry_pct < hi)]
        if len(sub) == 0: continue
        sw = sub.won.sum()
        P(f"    {lo*100:.0f}-{hi*100:.0f}%: {len(sub)} trades, "
          f"{sw/len(sub)*100:.0f}% WR, avg_SP=${sub.share_price.mean():.3f}")
    
    # Share price breakdown
    P(f"\n  BY SHARE PRICE:")
    for lo, hi in [(0.01, 0.20), (0.20, 0.35), (0.35, 0.45), (0.45, 0.55)]:
        sub = rdf[(rdf.share_price >= lo) & (rdf.share_price < hi)]
        if len(sub) == 0: continue
        sw = sub.won.sum()
        P(f"    ${lo:.2f}-${hi:.2f}: {len(sub)} trades, "
          f"{sw/len(sub)*100:.0f}% WR, EV=${sub.pnl.mean():+.2f}")
    
    # Direction breakdown
    P(f"\n  BY DIRECTION:")
    for d in ["UP", "DOWN"]:
        sub = rdf[rdf.pred_dir == d]
        if len(sub) == 0: continue
        sw = sub.won.sum()
        P(f"    {d}: {len(sub)} trades, {sw/len(sub)*100:.0f}% WR, PnL=${sub.pnl.sum():+.2f}")
    
    # Capital curve
    P(f"\n  CAPITAL CURVE (starting $100):")
    capital = 100.0
    peak = 100.0
    max_dd = 0.0
    for _, row in rdf.iterrows():
        capital += row["pnl"]
        peak = max(peak, capital)
        dd = (peak - capital) / peak * 100
        max_dd = max(max_dd, dd)
    P(f"    Final: ${capital:.2f}  |  Peak: ${peak:.2f}  |  Max DD: {max_dd:.1f}%")


def simulate_limit_orders(tick_df, bet=2.0):
    """Simulate limit orders: ML gives signal → place limit at lower price → check if filled."""
    model_cols = [c for c in tick_df.columns if c in ["lgbm", "catboost", "rf", "xgboost", "ensemble"]]
    results = []
    limit_offsets = [0.00, 0.02, 0.05, 0.10, 0.15]  # buy X below current ask

    for slug, mkt_ticks in tick_df.groupby("slug"):
        mkt_ticks = mkt_ticks.sort_values("ts")
        outcome = mkt_ticks.iloc[0]["outcome"]
        actual_up = outcome == "UP"
        n = len(mkt_ticks)

        for mc in model_cols:
            probs = mkt_ticks[mc].values
            confs = np.maximum(probs, 1 - probs)
            dirs_up = probs > 0.5

            for sig_idx in range(n):
                if confs[sig_idx] < 0.70: continue
                pred_up = dirs_up[sig_idx]
                pred_dir = "UP" if pred_up else "DOWN"
                won = (pred_dir == "UP" and actual_up) or (pred_dir == "DOWN" and not actual_up)

                tick = mkt_ticks.iloc[sig_idx]
                base_ask_raw = tick["up_ask"] if pred_up else tick["dn_ask"]
                if pd.isna(base_ask_raw) or base_ask_raw <= 0 or base_ask_raw >= 1: continue
                base_ask = float(base_ask_raw)

                for offset in limit_offsets:
                    limit_price = max(0.01, base_ask - offset)

                    # Check if limit would fill: any future tick has ask <= limit_price
                    future = mkt_ticks.iloc[sig_idx:]
                    col = "up_ask" if pred_up else "dn_ask"
                    future_vals = future[col].values
                    filled = np.nanmin(future_vals) <= limit_price + 0.005 if len(future_vals) > 0 and not np.all(np.isnan(future_vals)) else False

                    if not filled and offset > 0: continue

                    sp = limit_price if filled else base_ask
                    shares = bet / sp
                    pnl = shares * (1.0 - sp) if won else -bet

                    results.append({
                        "slug": slug, "model": mc, "offset": offset,
                        "signal_pct": tick["entry_pct"],
                        "conf": confs[sig_idx], "pred_dir": pred_dir,
                        "outcome": outcome, "won": won, "filled": filled,
                        "base_ask": base_ask, "fill_price": sp,
                        "pnl": round(pnl, 4),
                    })
                # Only first signal per market per model
                break

    return pd.DataFrame(results)


# ═══════════════════════════════════════
# 6. REPORTING — FULL DETAILED TABLES
# ═══════════════════════════════════════
def report_overview(markets):
    P(f"\n{'='*90}")
    P(f"  REPLAY V2 -- FULL-FIDELITY ML SIMULATION")
    P(f"{'='*90}")
    n_up = sum(1 for m in markets if m["outcome"] == "UP")
    n_dn = len(markets) - n_up
    P(f"  Markets: {len(markets)} ({n_up} UP / {n_dn} DOWN)")
    durs = {}
    for m in markets:
        durs.setdefault(m["dur"], [0, 0])
        if m["outcome"] == "UP": durs[m["dur"]][0] += 1
        else: durs[m["dur"]][1] += 1
    for d in sorted(durs):
        P(f"    {d}m: {sum(durs[d])} markets ({durs[d][0]} UP / {durs[d][1]} DOWN)")
    if markets:
        t0 = datetime.fromtimestamp(min(m["start_ts"] for m in markets), tz=timezone.utc)
        t1 = datetime.fromtimestamp(max(m["end_ts"] for m in markets), tz=timezone.utc)
        P(f"  Window: {t0.strftime('%Y-%m-%d %H:%M')} -> {t1.strftime('%H:%M')} UTC")


def report_model_accuracy(tick_df):
    P(f"\n{SEP}")
    P(f"  SECTION 1: MODEL ACCURACY PER TICK (all ticks)")
    P(f"{SEP}")
    model_cols = [c for c in tick_df.columns if c in ["lgbm", "catboost", "rf", "xgboost", "ensemble"]]
    bins = [(0.0, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 0.40), (0.40, 0.50),
            (0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 1.0)]

    P(f"  {'Model':<12} {'Entry%':<10} {'N':>5} {'AccUP':>7} {'AccDN':>7} {'AccAll':>7} {'AvgConf':>8}")
    for mc in model_cols:
        for lo, hi in bins:
            mask = (tick_df.entry_pct >= lo) & (tick_df.entry_pct < hi)
            sub = tick_df[mask]
            if len(sub) < 3: continue
            probs = sub[mc].values
            actual_up = (sub.outcome == "UP").values
            pred_up = probs > 0.5
            correct = pred_up == actual_up
            acc = correct.mean() * 100
            acc_up = correct[actual_up].mean() * 100 if actual_up.sum() > 0 else 0
            acc_dn = correct[~actual_up].mean() * 100 if (~actual_up).sum() > 0 else 0
            avg_conf = np.maximum(probs, 1 - probs).mean() * 100
            P(f"  {mc:<12} {lo*100:.0f}-{hi*100:.0f}%    {len(sub):>5} {acc_up:>6.1f}% {acc_dn:>6.1f}% {acc:>6.1f}% {avg_conf:>7.1f}%")
        P("")


def report_full_model_threshold_table(rdf):
    """FULL TABLE: each model x each threshold — all entries combined (EP<=$0.60)."""
    if rdf.empty: return
    P(f"\n{'='*90}")
    P(f"  SECTION 2: FULL MODEL x THRESHOLD TABLE (all entries, EP<=$0.60)")
    P(f"{'='*90}")
    P(f"  {'Model':<12} {'Conf>=':>6} {'N':>5} {'W':>4} {'L':>4} {'WR':>6} {'PnL':>9} {'EV':>7} {'AvgEP':>7}")
    P(f"  {'-'*70}")
    for mc in ["lgbm", "catboost", "rf", "xgboost", "ensemble"]:
        for ct in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
            sub = rdf[(rdf.model == mc) & (rdf.conf_thresh == ct) & (rdf.ep_max == 0.60)]
            if len(sub) == 0: continue
            w = sub.won.sum(); l = len(sub) - w
            wr = w / len(sub) * 100
            P(f"  {mc:<12} {ct:>5.0%} {len(sub):>5} {w:>4} {l:>4} {wr:>5.1f}% ${sub.pnl.sum():>+7.2f} ${sub.pnl.mean():>+5.2f} ${sub.share_price.mean():>.3f}")
        P("")


def report_fair_price_threshold_table(rdf):
    """HONEST TABLE: only fair-price entries (SP >= $0.40), FIRST entry per market only."""
    if rdf.empty: return
    P(f"\n{'='*90}")
    P(f"  SECTION 2B: HONEST THRESHOLD TABLE (SP>=$0.40, first entry/market)")
    P(f"{'='*90}")
    P(f"  {'Model':<12} {'Conf>=':>6} {'N':>5} {'W':>4} {'L':>4} {'WR':>6} {'PnL':>9} {'EV':>7} {'AvgEP':>7}")
    P(f"  {'-'*70}")
    for mc in ["lgbm", "catboost", "rf", "xgboost", "ensemble"]:
        for ct in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
            sub = rdf[(rdf.model == mc) & (rdf.conf_thresh == ct) & (rdf.ep_max == 0.60)]
            # Filter fair prices only
            sub = sub[sub.share_price >= 0.40]
            if len(sub) == 0: continue
            # Deduplicate: first entry per market (lowest entry_pct)
            sub = sub.sort_values("entry_pct").drop_duplicates("slug", keep="first")
            w = sub.won.sum(); l = len(sub) - w
            wr = w / len(sub) * 100
            P(f"  {mc:<12} {ct:>5.0%} {len(sub):>5} {w:>4} {l:>4} {wr:>5.1f}% ${sub.pnl.sum():>+7.2f} ${sub.pnl.mean():>+5.2f} ${sub.share_price.mean():>.3f}")
        P("")


def report_full_model_entry_table(rdf):
    """FULL TABLE: each model x each entry timing — fixed conf>=0.65, EP<=$0.55."""
    if rdf.empty: return
    P(f"\n{'='*90}")
    P(f"  SECTION 3: FULL MODEL x ENTRY TIMING (conf>=0.65, EP<=$0.55)")
    P(f"{'='*90}")
    P(f"  {'Model':<12} {'Entry':>6} {'N':>5} {'W':>4} {'L':>4} {'WR':>6} {'PnL':>9} {'EV':>7} {'AvgEP':>7}")
    P(f"  {'-'*70}")
    for mc in ["lgbm", "catboost", "rf", "xgboost", "ensemble"]:
        for ep in sorted(rdf.entry_pct.unique()):
            sub = rdf[(rdf.model == mc) & (rdf.conf_thresh == 0.65) & (rdf.ep_max == 0.55) & (rdf.entry_pct == ep)]
            if len(sub) == 0: continue
            w = sub.won.sum(); l = len(sub) - w
            wr = w / len(sub) * 100
            P(f"  {mc:<12} {ep*100:>5.0f}% {len(sub):>5} {w:>4} {l:>4} {wr:>5.1f}% ${sub.pnl.sum():>+7.2f} ${sub.pnl.mean():>+5.2f} ${sub.share_price.mean():>.3f}")
        P("")


def report_full_entry_x_threshold(rdf):
    """FULL TABLE: entry timing x threshold for EACH model."""
    if rdf.empty: return
    P(f"\n{'='*90}")
    P(f"  SECTION 4: ENTRY x THRESHOLD MATRIX PER MODEL (EP<=$0.55)")
    P(f"{'='*90}")
    confs = sorted(rdf.conf_thresh.unique())
    entries = sorted(rdf.entry_pct.unique())
    for mc in ["lgbm", "catboost", "rf", "xgboost", "ensemble"]:
        P(f"\n  --- {mc.upper()} ---")
        hdr = f"  {'Entry':>6} |"
        for ct in confs:
            hdr += f" {ct:.0%}".rjust(13) + " |"
        P(hdr)
        P(f"  {'-'*len(hdr)}")
        for ep in entries:
            row = f"  {ep*100:>5.0f}% |"
            for ct in confs:
                sub = rdf[(rdf.model == mc) & (rdf.entry_pct == ep) & (rdf.conf_thresh == ct) & (rdf.ep_max == 0.55)]
                if len(sub) == 0:
                    row += "           -- |"
                else:
                    wr = sub.won.mean() * 100
                    row += f" {len(sub):>3}t {wr:>5.1f}% |"
            P(row)
        P("")


def report_share_price_breakdown(rdf):
    """FULL TABLE: share price (entry price) breakdown per model."""
    if rdf.empty: return
    P(f"\n{'='*90}")
    P(f"  SECTION 5: SHARE PRICE (ENTRY PRICE) BREAKDOWN (conf>=0.65)")
    P(f"{'='*90}")
    sp_bins = [(0.0, 0.20), (0.20, 0.30), (0.30, 0.40), (0.40, 0.50),
               (0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 1.0)]
    rdf2 = rdf[rdf.conf_thresh == 0.65].copy()
    P(f"  {'Model':<12} {'EP range':<12} {'N':>5} {'WR':>6} {'PnL':>9} {'EV':>7}")
    P(f"  {'-'*60}")
    for mc in ["lgbm", "catboost", "rf", "xgboost", "ensemble"]:
        for lo, hi in sp_bins:
            sub = rdf2[(rdf2.model == mc) & (rdf2.share_price >= lo) & (rdf2.share_price < hi)]
            if len(sub) < 2: continue
            wr = sub.won.mean() * 100
            P(f"  {mc:<12} ${lo:.2f}-{hi:.2f} {len(sub):>5} {wr:>5.1f}% ${sub.pnl.sum():>+7.2f} ${sub.pnl.mean():>+5.2f}")
        P("")


def report_direction_per_model(rdf):
    """FULL TABLE: UP vs DOWN accuracy per model per threshold."""
    if rdf.empty: return
    P(f"\n{'='*90}")
    P(f"  SECTION 6: DIRECTION BREAKDOWN PER MODEL x THRESHOLD (EP<=$0.60)")
    P(f"{'='*90}")
    P(f"  {'Model':<12} {'Conf>=':>6} {'Dir':>5} {'N':>5} {'WR':>6} {'PnL':>9} {'EV':>7}")
    P(f"  {'-'*60}")
    for mc in ["lgbm", "catboost", "rf", "xgboost", "ensemble"]:
        for ct in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
            for d in ["UP", "DOWN"]:
                sub = rdf[(rdf.model == mc) & (rdf.conf_thresh == ct) & (rdf.ep_max == 0.60) & (rdf.pred_dir == d)]
                if len(sub) < 2: continue
                wr = sub.won.mean() * 100
                P(f"  {mc:<12} {ct:>5.0%} {d:>5} {len(sub):>5} {wr:>5.1f}% ${sub.pnl.sum():>+7.2f} ${sub.pnl.mean():>+5.2f}")
            sub_all = rdf[(rdf.model == mc) & (rdf.conf_thresh == ct) & (rdf.ep_max == 0.60)]
            if len(sub_all) >= 3:
                P(f"  {mc:<12} {ct:>5.0%} {'ALL':>5} {len(sub_all):>5} {sub_all.won.mean()*100:>5.1f}% ${sub_all.pnl.sum():>+7.2f} ${sub_all.pnl.mean():>+5.2f}")
            P("")


def report_gap_per_model(rdf):
    """GAP analysis per model per threshold."""
    if rdf.empty: return
    P(f"\n{'='*90}")
    P(f"  SECTION 7: GAP ANALYSIS PER MODEL (EP<=$0.55)")
    P(f"{'='*90}")
    gap_bins = [(-np.inf, -0.05), (-0.05, -0.02), (-0.02, 0.0), (0.0, 0.02), (0.02, 0.05), (0.05, np.inf)]
    gap_labels = ["<-0.05%", "-0.05/-0.02", "-0.02/0", "0/+0.02", "+0.02/+0.05", ">+0.05%"]
    rdf2 = rdf[rdf.ep_max == 0.55].copy()
    rdf2["gap_fav"] = np.where(rdf2.pred_dir == "UP", rdf2.gap_pct, -rdf2.gap_pct)
    P(f"  Direction-relative gap (positive = gap IN FAVOR of prediction)")
    P(f"  {'Model':<12} {'Conf>=':>6} {'Gap zone':<14} {'N':>5} {'WR':>6} {'EV':>7}")
    P(f"  {'-'*60}")
    for mc in ["lgbm", "catboost", "rf", "xgboost", "ensemble"]:
        for ct in [0.60, 0.70, 0.80]:
            for (lo, hi), lbl in zip(gap_bins, gap_labels):
                sub = rdf2[(rdf2.model == mc) & (rdf2.conf_thresh == ct) & (rdf2.gap_fav >= lo) & (rdf2.gap_fav < hi)]
                if len(sub) < 2: continue
                wr = sub.won.mean() * 100
                P(f"  {mc:<12} {ct:>5.0%} {lbl:<14} {len(sub):>5} {wr:>5.1f}% ${sub.pnl.mean():>+5.2f}")
            P("")


def report_model_combos(tick_df, bet=2.0):
    """Model consensus combos: what if 2/3/4 models agree?"""
    P(f"\n{'='*90}")
    P(f"  SECTION 8: MODEL CONSENSUS COMBOS")
    P(f"{'='*90}")
    base_models = [c for c in ["lgbm", "catboost", "rf", "xgboost"] if c in tick_df.columns]
    if len(base_models) < 2: return

    entries = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    confs = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

    # For each market, pick one tick per entry bin
    P(f"  Consensus = N models predict same direction with conf >= threshold")
    P(f"  {'MinAgree':>8} {'Conf>=':>6} {'Entry':>6} {'N':>5} {'WR':>6} {'PnL':>9} {'EV':>7} {'AvgEP':>7}")
    P(f"  {'-'*70}")

    results = []
    for slug, grp in tick_df.groupby("slug"):
        outcome = grp.iloc[0]["outcome"]
        actual_up = outcome == "UP"
        ep_vals = grp.entry_pct.values
        for ep_target in entries:
            diffs = np.abs(ep_vals - ep_target)
            idx = np.argmin(diffs)
            if diffs[idx] > 0.05: continue
            tick = grp.iloc[idx]
            probs = {m: tick[m] for m in base_models if not np.isnan(tick[m])}
            if len(probs) < 2: continue
            for min_agree in [2, 3, 4]:
                if min_agree > len(probs): continue
                for ct in confs:
                    # Count models that agree on UP with conf >= ct
                    up_agree = sum(1 for p in probs.values() if p > 0.5 and max(p, 1-p) >= ct)
                    dn_agree = sum(1 for p in probs.values() if p <= 0.5 and max(p, 1-p) >= ct)
                    if up_agree >= min_agree:
                        pred_dir = "UP"; won = actual_up
                    elif dn_agree >= min_agree:
                        pred_dir = "DOWN"; won = not actual_up
                    else:
                        continue
                    sp_raw = tick["up_ask"] if pred_dir == "UP" else tick["dn_ask"]
                    if pd.isna(sp_raw) or sp_raw <= 0 or sp_raw >= 1: continue
                    sp = float(sp_raw)
                    if sp > 0.55: continue
                    shares = bet / sp
                    pnl = shares * (1.0 - sp) if won else -bet
                    results.append({
                        "min_agree": min_agree, "conf": ct, "entry": ep_target,
                        "won": won, "pnl": pnl, "sp": sp,
                    })

    if not results:
        P("  No consensus trades found."); return
    cdf = pd.DataFrame(results)
    grp = cdf.groupby(["min_agree", "conf", "entry"]).agg(
        n=("won", "size"), w=("won", "sum"), pnl=("pnl", "sum"),
        ev=("pnl", "mean"), sp=("sp", "mean")
    ).reset_index()
    grp["wr"] = grp.w / grp.n * 100

    # Show all combos with N>=3
    show = grp[grp.n >= 3].sort_values(["min_agree", "conf", "entry"])
    for _, r in show.iterrows():
        P(f"  {r.min_agree:>8} {r.conf:>5.0%} {r.entry*100:>5.0f}% {r.n:>5} {r.wr:>5.1f}% ${r.pnl:>+7.2f} ${r.ev:>+5.2f} ${r.sp:>.3f}")

    # Best consensus combos
    P(f"\n  TOP 15 CONSENSUS COMBOS BY WR (N>=5):")
    best = grp[grp.n >= 5].sort_values("wr", ascending=False).head(15)
    for _, r in best.iterrows():
        P(f"  {r.min_agree:.0f}+ models agree, conf>={r.conf:.0%}, entry={r.entry*100:.0f}%: "
          f"{r.n:.0f}t WR={r.wr:.1f}% PnL=${r.pnl:+.2f} EV=${r.ev:+.2f}")


def report_full_entry_x_shareprice(rdf):
    """Entry timing x share price bucket."""
    if rdf.empty: return
    P(f"\n{'='*90}")
    P(f"  SECTION 9: ENTRY TIMING x SHARE PRICE (conf>=0.65, ensemble)")
    P(f"{'='*90}")
    rdf2 = rdf[(rdf.conf_thresh == 0.65) & (rdf.model == "ensemble")].copy()
    if rdf2.empty:
        rdf2 = rdf[(rdf.conf_thresh == 0.65)].copy()
    if rdf2.empty: return

    sp_bins = [(0.0, 0.30), (0.30, 0.40), (0.40, 0.50), (0.50, 0.60), (0.60, 0.70), (0.70, 1.0)]
    P(f"  {'Entry':>6} {'EP range':<12} {'N':>5} {'WR':>6} {'PnL':>9} {'EV':>7}")
    P(f"  {'-'*55}")
    for ep in sorted(rdf2.entry_pct.unique()):
        for lo, hi in sp_bins:
            sub = rdf2[(rdf2.entry_pct == ep) & (rdf2.share_price >= lo) & (rdf2.share_price < hi)]
            if len(sub) < 2: continue
            wr = sub.won.mean() * 100
            P(f"  {ep*100:>5.0f}% ${lo:.2f}-{hi:.2f} {len(sub):>5} {wr:>5.1f}% ${sub.pnl.sum():>+7.2f} ${sub.pnl.mean():>+5.2f}")
        P("")


def report_best_combos_grid(rdf):
    """TOP combos sorted by WR, EV, PnL — extensive."""
    if rdf.empty: return
    P(f"\n{'='*90}")
    P(f"  SECTION 10: TOP 30 BEST COMBOS BY WR / EV / PNL (N>=5)")
    P(f"{'='*90}")
    grp = rdf.groupby(["model", "entry_pct", "conf_thresh", "ep_max"]).agg(
        n=("won", "size"), wins=("won", "sum"),
        pnl=("pnl", "sum"), ev=("pnl", "mean"),
        avg_sp=("share_price", "mean"),
    ).reset_index()
    grp["wr"] = grp.wins / grp.n * 100

    filt = grp[grp.n >= 5]
    hdr = f"  {'Model':<12} {'Entry':>5} {'Conf':>5} {'MaxEP':>6} {'N':>4} {'WR':>6} {'PnL':>9} {'EV':>7} {'AvgEP':>7}"

    P(f"\n  --- BY WIN RATE ---")
    P(hdr)
    for _, r in filt.sort_values("wr", ascending=False).head(30).iterrows():
        P(f"  {r.model:<12} {r.entry_pct*100:>4.0f}% {r.conf_thresh:>4.0%} ${r.ep_max:.2f} "
          f"{r.n:>4} {r.wr:>5.1f}% ${r.pnl:>+7.2f} ${r.ev:>+5.2f} ${r.avg_sp:>.3f}")

    P(f"\n  --- BY EV PER TRADE ---")
    P(hdr)
    for _, r in filt.sort_values("ev", ascending=False).head(30).iterrows():
        P(f"  {r.model:<12} {r.entry_pct*100:>4.0f}% {r.conf_thresh:>4.0%} ${r.ep_max:.2f} "
          f"{r.n:>4} {r.wr:>5.1f}% ${r.pnl:>+7.2f} ${r.ev:>+5.2f} ${r.avg_sp:>.3f}")

    P(f"\n  --- BY TOTAL PNL ---")
    P(hdr)
    for _, r in filt.sort_values("pnl", ascending=False).head(30).iterrows():
        P(f"  {r.model:<12} {r.entry_pct*100:>4.0f}% {r.conf_thresh:>4.0%} ${r.ep_max:.2f} "
          f"{r.n:>4} {r.wr:>5.1f}% ${r.pnl:>+7.2f} ${r.ev:>+5.2f} ${r.avg_sp:>.3f}")


def report_head_to_head_matrix(rdf):
    """H2H for ALL entry timings x ALL models at various thresholds."""
    if rdf.empty: return
    P(f"\n{'='*90}")
    P(f"  SECTION 11: HEAD-TO-HEAD — ALL MODELS x ALL ENTRIES")
    P(f"{'='*90}")
    models_list = ["lgbm", "catboost", "rf", "xgboost", "ensemble"]
    for ct in [0.60, 0.65, 0.70, 0.75, 0.80]:
        for ep_max_v in [0.50, 0.55, 0.60]:
            P(f"\n  conf>={ct:.0%}, EP<=${ep_max_v:.2f}:")
            P(f"  {'Model':<12} {'Entry':>6} {'N':>5} {'W':>4} {'L':>4} {'WR':>6} {'PnL':>9} {'EV':>7}")
            found = False
            for mc in models_list:
                for ep in sorted(rdf.entry_pct.unique()):
                    sub = rdf[(rdf.model == mc) & (rdf.conf_thresh == ct) & (rdf.ep_max == ep_max_v) & (rdf.entry_pct == ep)]
                    if len(sub) == 0: continue
                    found = True
                    w = sub.won.sum(); l = len(sub) - w
                    wr = w / len(sub) * 100
                    P(f"  {mc:<12} {ep*100:>5.0f}% {len(sub):>5} {w:>4} {l:>4} {wr:>5.1f}% ${sub.pnl.sum():>+7.2f} ${sub.pnl.mean():>+5.2f}")
            if not found:
                P(f"  (no trades)")


def report_limit_orders(ldf):
    if ldf.empty: return
    P(f"\n{'='*90}")
    P(f"  SECTION 12: LIMIT ORDER SIMULATION")
    P(f"{'='*90}")
    P(f"  Signal: conf>=0.70, then place limit at (ask - offset)")
    P(f"  {'Model':<12} {'Offset':>6} {'N':>5} {'Filled':>6} {'WR':>6} {'AvgFill':>8} {'EV':>7}")
    for mc in ["lgbm", "catboost", "rf", "xgboost", "ensemble"]:
        for off in sorted(ldf.offset.unique()):
            sub = ldf[(ldf.model == mc) & (ldf.offset == off) & ldf.filled]
            if len(sub) < 2: continue
            wr = sub.won.mean() * 100
            P(f"  {mc:<12} ${off:.2f} {len(sub):>5}  {len(sub):>5} {wr:>5.1f}% ${sub.fill_price.mean():>.3f} ${sub.pnl.mean():>+5.2f}")
        P("")


def report_per_market(tick_df, markets):
    P(f"\n{'='*90}")
    P(f"  SECTION 13: PER-MARKET ML DETAIL (all markets)")
    P(f"{'='*90}")
    model_cols = [c for c in tick_df.columns if c in ["lgbm", "catboost", "rf", "xgboost", "ensemble"]]
    for mkt in sorted(markets, key=lambda m: m["start_ts"]):
        mt = tick_df[tick_df.slug == mkt["slug"]]
        if len(mt) == 0: continue
        P(f"\n  {mkt['slug']}  outcome={mkt['outcome']}  PTB=${mkt['ptb']:.4f}  SOL: ${mkt['sol_start']:.3f}->${mkt['sol_end']:.3f}  snaps={mkt['n_snaps']}")
        for ep_target in [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
            diffs = np.abs(mt.entry_pct.values - ep_target)
            if diffs.min() > 0.05: continue
            idx = diffs.argmin()
            tick = mt.iloc[idx]
            parts = []
            for mc in model_cols:
                p = tick[mc]
                if np.isnan(p): continue
                d = "UP" if p > 0.5 else "DN"
                c = max(p, 1 - p)
                won = (d == "UP" and mkt["outcome"] == "UP") or (d == "DN" and mkt["outcome"] == "DOWN")
                parts.append(f"{mc}:{c:.0%}{d}{'W' if won else 'L'}")
            gap_s = f"gap={tick['gap_pct']:+.4f}%"
            P(f"    @{ep_target*100:.0f}%: SOL=${tick['sol_price']:.3f} {gap_s} UP=${tick['up_ask']:.3f} DN=${tick['dn_ask']:.3f} | {' '.join(parts)}")


# ═══════════════════════════════════════
# MAIN
# ═══════════════════════════════════════
async def async_main():
    args = parse_args()
    t0_total = time.time()

    # 1. Load snapshots
    snaps_dict, date_str = load_snapshots(args.date, args.dur)
    P(f"  Date: {date_str}")
    for dur, df in snaps_dict.items():
        P(f"  {dur}m: {len(df)} snaps, {df.slug.nunique()} slugs")

    # 2. Extract markets
    all_markets = []
    for dur, df in snaps_dict.items():
        mkts = extract_markets(df, dur)
        all_markets.extend(mkts)
        P(f"  {dur}m: {len(mkts)} usable markets")
    if not all_markets:
        P("  No usable markets."); return
    report_overview(all_markets)

    # 3. Download Binance klines for the window
    ts_min = min(m["start_ts"] for m in all_markets)
    ts_max = max(m["end_ts"] for m in all_markets)
    if not args.skip_download:
        klines = await download_binance_window(ts_min, ts_max)
    else:
        # Try cache
        cache_dir = Path("data/replay_cache")
        cached = sorted(cache_dir.glob("klines_*.parquet")) if cache_dir.exists() else []
        if cached:
            klines = pd.read_parquet(cached[-1])
            P(f"  [cache] {cached[-1].name}: {len(klines)} klines")
        else:
            klines = await download_binance_window(ts_min, ts_max)

    if klines.empty:
        P("  No klines available. Running with degraded features.")

    # 4. Load models
    models, scaler, fn = load_models()
    P(f"  Models: {list(models.keys())} | Features: {len(fn)}")

    # 5. Build features at sampled ticks and predict
    P(f"\n  Computing features at sampled ticks...")
    tick_rows = []
    t0_feat = time.time()

    # Sample ticks: every ~1% of market duration (~85 ticks per market)
    # Live trader re-evaluates every 5s = ~60 checks per 5m market
    # Using 1% spacing gives comparable density
    sample_fracs = np.arange(0.0, 0.86, 0.01)

    n_markets = len(all_markets)
    for mi, mkt in enumerate(all_markets):
        snaps_m = mkt["snaps"]
        n = len(snaps_m)
        if n < 5: continue

        # Sol volatility from klines near this market
        if not klines.empty:
            pre_k = klines[klines.ts <= mkt["start_ts"]]
            if len(pre_k) > 10:
                closes = pre_k.close.values[-60:]
                sol_vol = float(np.std(np.diff(np.log(closes + 1e-10))))
            else:
                sol_vol = 0.003
        else:
            sol_vol = 0.003

        for frac in sample_fracs:
            idx = min(int(n * frac), n - 1)
            snap_row = snaps_m.iloc[idx]
            snap_ts = float(snap_row["ts"])

            # Build OHLCV from klines
            if not klines.empty:
                ohlcv = build_ohlcv_at_ts(klines, snap_ts, 61)
            else:
                ohlcv = None

            if ohlcv is None:
                # Minimal features from snapshot only
                ohlcv = {
                    "open": np.array([float(snap_row.sol_price)]),
                    "high": np.array([float(snap_row.sol_price)]),
                    "low": np.array([float(snap_row.sol_price)]),
                    "close": np.array([float(snap_row.sol_price)]),
                    "volume": np.array([1.0]),
                    "taker_buy_volume": np.array([0.5]),
                }

            feats = compute_all_features(ohlcv, snap_row, mkt["ptb"], mkt["dur"], mkt["start_ts"], sol_vol)
            fv = np.array([feats.get(f, 0.0) for f in fn], dtype=np.float64)
            fv = np.nan_to_num(fv, nan=0.0, posinf=0.0, neginf=0.0)

            sol_price = float(snap_row.get("sol_price", 0))
            gap_pct = (sol_price - mkt["ptb"]) / mkt["ptb"] * 100 if mkt["ptb"] > 0 else 0

            # Orderbook data — use NaN for missing (never fake $0.50)
            up_ask_raw = snap_row.get("up_best_ask", np.nan)
            dn_ask_raw = snap_row.get("dn_best_ask", np.nan)
            up_ask = float(up_ask_raw) if pd.notna(up_ask_raw) and float(up_ask_raw) > 0 else np.nan
            dn_ask = float(dn_ask_raw) if pd.notna(dn_ask_raw) and float(dn_ask_raw) > 0 else np.nan
            up_spread = float(snap_row.get("up_spread", np.nan) or np.nan)
            dn_spread = float(snap_row.get("dn_spread", np.nan) or np.nan)
            up_depth5 = float(snap_row.get("up_ask_depth_5", 0))
            dn_depth5 = float(snap_row.get("dn_ask_depth_5", 0))
            up_vol = float(snap_row.get("up_ask_volume", 0))
            dn_vol = float(snap_row.get("dn_ask_volume", 0))

            row = {
                "slug": mkt["slug"], "ts": snap_ts, "entry_pct": frac,
                "outcome": mkt["outcome"], "sol_price": sol_price,
                "ptb": mkt["ptb"], "gap_pct": gap_pct,
                "up_ask": up_ask, "dn_ask": dn_ask,
                "up_spread": up_spread, "dn_spread": dn_spread,
                "up_depth5": up_depth5, "dn_depth5": dn_depth5,
                "up_vol": up_vol, "dn_vol": dn_vol,
            }
            # Store feature vector for batch prediction later
            row["_fv"] = fv
            tick_rows.append(row)

        if (mi + 1) % 10 == 0:
            P(f"    {mi+1}/{n_markets} markets processed...")

    feat_time = time.time() - t0_feat
    P(f"  Features computed: {len(tick_rows)} ticks in {feat_time:.1f}s ({len(tick_rows)/feat_time:.0f} ticks/s)")

    # 6. Batch predict
    t0_pred = time.time()
    X = np.vstack([r["_fv"] for r in tick_rows])
    preds = predict_batch(X, models, scaler)
    pred_time = time.time() - t0_pred
    P(f"  Batch predictions: {X.shape} in {pred_time:.1f}s")

    # Build tick DataFrame
    tick_df = pd.DataFrame([{k: v for k, v in r.items() if k != "_fv"} for r in tick_rows])
    for mc, p_arr in preds.items():
        tick_df[mc] = p_arr

    # 7. ORDERBOOK QUALITY REPORT
    P(f"\n{'='*90}")
    P(f"  SECTION 0: ORDERBOOK DATA QUALITY & LIQUIDITY")
    P(f"{'='*90}")
    n_ticks = len(tick_df)
    up_valid = tick_df["up_ask"].notna() & (tick_df["up_ask"] > 0) & (tick_df["up_ask"] < 1)
    dn_valid = tick_df["dn_ask"].notna() & (tick_df["dn_ask"] > 0) & (tick_df["dn_ask"] < 1)
    P(f"  Total ticks: {n_ticks}")
    P(f"  UP ask valid (0<p<1): {up_valid.sum()} ({up_valid.mean()*100:.0f}%)")
    P(f"  DN ask valid (0<p<1): {dn_valid.sum()} ({dn_valid.mean()*100:.0f}%)")
    if "up_depth5" in tick_df.columns:
        d5 = tick_df["up_depth5"]
        P(f"  UP depth5: mean={d5.mean():.0f} median={d5.median():.0f} p10={d5.quantile(0.1):.0f}")
    if "up_spread" in tick_df.columns:
        sp = tick_df["up_spread"].dropna()
        P(f"  UP spread: mean={sp.mean():.4f} median={sp.median():.4f} p90={sp.quantile(0.9):.4f}")
    if "up_vol" in tick_df.columns:
        vol = tick_df["up_vol"]
        P(f"  UP ask volume: mean={vol.mean():.0f} p10={vol.quantile(0.1):.0f}")

    # Fillability at $2.00 bet
    P(f"\n  FILLABILITY at bet=${args.bet:.2f}:")
    for ep in [0.10, 0.20, 0.30, 0.50, 0.70]:
        sub = tick_df[(tick_df.entry_pct >= ep - 0.05) & (tick_df.entry_pct <= ep + 0.05)]
        if len(sub) < 5: continue
        up_ok = sub[up_valid.loc[sub.index]]
        if len(up_ok) == 0: continue
        up_prices = up_ok["up_ask"]
        fair = (up_prices >= 0.30) & (up_prices <= 0.60)
        depth_col = "up_depth5" if "up_depth5" in sub.columns else None
        if depth_col:
            needed = args.bet / up_ok["up_ask"]
            can_fill = (up_ok[depth_col] >= needed).mean() * 100
        else:
            can_fill = float("nan")
        P(f"    entry@{ep*100:.0f}%: {len(up_ok)} valid | fair-priced: {fair.mean()*100:.0f}% "
          f"| can_fill_$2: {can_fill:.0f}% | avg_spread={up_ok['up_spread'].mean():.4f}" if "up_spread" in sub.columns else "")

    # 7b. Model accuracy
    report_model_accuracy(tick_df)

    # 8. Simulate trades (vectorized grid search)
    t0_sim = time.time()
    rdf = simulate_trades(tick_df, args.bet)
    sim_time = time.time() - t0_sim
    P(f"\n  Trade simulation: {len(rdf)} result rows in {sim_time:.1f}s")

    # LIQUIDITY FILTER NOTE
    if "can_fill" in rdf.columns and len(rdf) > 0:
        fillable = rdf["can_fill"].sum()
        P(f"  Fillable trades (depth5 >= shares needed): {fillable}/{len(rdf)} ({fillable/len(rdf)*100:.0f}%)")

    # Full tables (ALL trades, then filtered)
    report_full_model_threshold_table(rdf)   # S2: model x threshold
    report_fair_price_threshold_table(rdf)   # S2B: honest fair-price only
    report_full_model_entry_table(rdf)       # S3: model x entry timing
    report_full_entry_x_threshold(rdf)       # S4: entry x threshold matrix
    report_share_price_breakdown(rdf)        # S5: share price breakdown
    report_direction_per_model(rdf)          # S6: direction per model
    report_gap_per_model(rdf)                # S7: gap per model
    report_model_combos(tick_df, args.bet)   # S8: model consensus combos
    report_full_entry_x_shareprice(rdf)      # S9: entry x share price
    report_best_combos_grid(rdf)             # S10: top 30 combos
    report_head_to_head_matrix(rdf)          # S11: H2H all models x entries

    # S14: LIQUIDITY-FILTERED RESULTS (only fillable trades)
    if "can_fill" in rdf.columns and len(rdf) > 0:
        rdf_liq = rdf[rdf["can_fill"]].copy()
        P(f"\n{'='*90}")
        P(f"  SECTION 14: LIQUIDITY-FILTERED RESULTS (only where depth5 >= shares needed)")
        P(f"{'='*90}")
        P(f"  Total fillable: {len(rdf_liq)} / {len(rdf)} ({len(rdf_liq)/len(rdf)*100:.0f}%)")
        if len(rdf_liq) > 0:
            report_full_model_threshold_table(rdf_liq)
            report_best_combos_grid(rdf_liq)

    # 9. LIVE TRADER SIMULATION (most important — matches real trading)
    n_markets = len(all_markets)
    # Old config (what user was running)
    live_rdf_old, sk_c, sk_p, sk_d = simulate_live_trader(tick_df, args.bet, min_conf=0.60, max_sp=0.55)
    report_live_simulation(live_rdf_old, sk_c, sk_p, sk_d, n_markets, 0.60, 0.55, args.bet)
    # New config
    live_rdf_new, sk_c2, sk_p2, sk_d2 = simulate_live_trader(tick_df, args.bet, min_conf=0.65, max_sp=0.50)
    report_live_simulation(live_rdf_new, sk_c2, sk_p2, sk_d2, n_markets, 0.65, 0.50, args.bet)

    # 10. Limit order sim
    t0_lim = time.time()
    ldf = simulate_limit_orders(tick_df, args.bet)
    lim_time = time.time() - t0_lim
    if not ldf.empty:
        P(f"\n  Limit order sim: {len(ldf)} rows in {lim_time:.1f}s")
        report_limit_orders(ldf)             # S12: limit orders

    # 11. Per-market detail
    report_per_market(tick_df, all_markets)   # S13: per-market ML detail

    total_time = time.time() - t0_total
    P(f"\n{'='*90}")
    P(f"  REPLAY V2 COMPLETE -- {total_time:.1f}s ({len(tick_rows)} ticks, {len(rdf)} sims)")
    P(f"{'='*90}")

    # Save
    Path("results").mkdir(exist_ok=True)
    summary = {
        "date": date_str, "n_markets": len(all_markets),
        "n_ticks": len(tick_rows), "n_sims": len(rdf),
        "total_time_s": round(total_time, 1),
        "feat_time_s": round(feat_time, 1),
        "pred_time_s": round(pred_time, 1),
    }
    Path("results/replay_v2_report.json").write_text(json.dumps(summary, indent=2))
    P(f"  Saved: results/replay_v2_report.json")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
