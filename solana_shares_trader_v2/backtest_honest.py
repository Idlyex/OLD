"""Honest Train/Test Split Backtest — hold to expiry, no stops needed.

Strategy: 
  1. At each market open, model predicts: will SOL be above PTB at expiry?
  2. If model says UP → buy UP shares at theoretical price
  3. If model says DOWN → buy DOWN shares at theoretical price  
  4. Hold to expiry: win = $1, lose = $0
  
No Polymarket data needed. Only CEX SOL/USDT data.

Train on first 70% of markets, test on remaining 30% (model never sees test data).

Usage:
  python backtest_honest.py
  python backtest_honest.py --duration 15 --split 0.7 --entry-price 0.50
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import numpy as np
import pandas as pd
import time
from typing import Dict, List

from config import config
from core.utils.logger import log
from training.dataset import TrainingDataset


def parse_args():
    p = argparse.ArgumentParser(description="Honest train/test backtest")
    p.add_argument("--duration", type=int, default=15, help="Market duration in minutes")
    p.add_argument("--split", type=float, default=0.7, help="Train fraction (0.7 = 70% train, 30% test)")
    p.add_argument("--entry-price", type=float, default=0.50, dest="entry_price",
                    help="Fixed entry price for shares (0.50 = buy at fair value)")
    p.add_argument("--capital", type=float, default=100.0, help="Starting capital")
    p.add_argument("--bet-size", type=float, default=2.0, dest="bet_size", help="USD per trade")
    p.add_argument("--data", type=str, default=None, help="Path to data parquet")
    p.add_argument("--confidence-threshold", type=float, default=0.55, dest="conf_threshold",
                    help="Min model confidence to enter (0.5=enter everything, 0.6=selective)")
    return p.parse_args()


def load_data(args) -> pd.DataFrame:
    if args.data and os.path.exists(args.data):
        return pd.read_parquet(args.data)
    
    # Auto-detect
    for p in ["data/processed/SOLUSDT_processed.parquet", "data/processed/SOLUSDT_1m.parquet"]:
        if os.path.exists(p):
            return pd.read_parquet(p)
    
    log.error("No data found. Run: python main.py --mode download --symbol SOLUSDT --days 30")
    sys.exit(1)


def build_markets(sol_data: pd.DataFrame, duration_minutes: int) -> List[Dict]:
    """Chop SOL data into N-minute markets."""
    sol_data = sol_data.sort_values("ts").reset_index(drop=True)
    ts = sol_data["ts"].values
    interval_ms = duration_minutes * 60_000
    
    start_ts = int(ts[0])
    end_ts = int(ts[-1])
    first_market = (start_ts // interval_ms + 1) * interval_ms
    
    markets = []
    current = first_market
    
    while current + interval_ms <= end_ts:
        mask = (ts >= current) & (ts < current + interval_ms)
        indices = np.where(mask)[0]
        
        if len(indices) >= 3:
            ptb = float(sol_data.iloc[indices[0]]["close"])
            final = float(sol_data.iloc[indices[-1]]["close"])
            outcome = "UP" if final >= ptb else "DOWN"
            
            markets.append({
                "start_ts": current,
                "end_ts": current + interval_ms,
                "ptb": ptb,
                "final_price": final,
                "outcome": outcome,
                "bar_indices": indices,
                "outcome_up": 1 if outcome == "UP" else 0,
            })
        
        current += interval_ms
    
    return markets


def main():
    args = parse_args()
    t0 = time.time()
    
    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║  HONEST TRAIN/TEST BACKTEST — Hold to Expiry           ║")
    log.info("╠══════════════════════════════════════════════════════════╣")
    log.info(f"║  Duration:      {args.duration}m markets")
    log.info(f"║  Train/Test:    {args.split*100:.0f}% / {(1-args.split)*100:.0f}%")
    log.info(f"║  Entry Price:   ${args.entry_price:.2f}")
    log.info(f"║  Confidence:    >{args.conf_threshold:.2f}")
    log.info(f"║  Capital:       ${args.capital:.0f}, bet ${args.bet_size:.0f}/trade")
    log.info("╚══════════════════════════════════════════════════════════╝")
    
    # 1. Load data
    sol_data = load_data(args)
    log.info(f"\nLoaded {len(sol_data):,} bars of SOL data")
    
    # 2. Build markets
    markets = build_markets(sol_data, args.duration)
    log.info(f"Generated {len(markets)} markets ({args.duration}m each)")
    
    up_count = sum(1 for m in markets if m["outcome"] == "UP")
    log.info(f"Outcome balance: {up_count}/{len(markets)} UP ({up_count/len(markets)*100:.1f}%)")
    
    # 3. Build features (per-market, using first bar of each market)
    log.info("\nBuilding features...")
    ds = TrainingDataset()
    dataset = ds.build_shares_dataset(sol_data, duration_minutes=args.duration)
    
    X = dataset["X"]
    y_dir = dataset["y_direction"]
    feature_names = dataset["feature_names"]
    timestamps = dataset["timestamps"]
    
    log.info(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # 4. Map samples to markets (each sample belongs to a market)
    # We need to pick ONE sample per market for the "entry decision"
    # Use the sample from ~20% into the market (early enough to enter, late enough for features)
    market_samples = []  # (market_idx, sample_idx)
    
    for mi, market in enumerate(markets):
        # Find samples whose timestamp falls in this market window
        mask = (timestamps >= market["start_ts"]) & (timestamps < market["end_ts"])
        sample_indices = np.where(mask)[0]
        
        if len(sample_indices) == 0:
            continue
        
        # Pick sample at ~20% into market (early entry)
        entry_idx = sample_indices[max(0, len(sample_indices) // 5)]
        market_samples.append((mi, entry_idx))
    
    log.info(f"Markets with features: {len(market_samples)}")
    
    # 5. Train/test split — BY MARKET (time-ordered, no leakage)
    split_point = int(len(market_samples) * args.split)
    train_markets = market_samples[:split_point]
    test_markets = market_samples[split_point:]
    
    log.info(f"\nSplit: {len(train_markets)} train markets, {len(test_markets)} test markets")
    log.info(f"Train markets: #{train_markets[0][0]}–#{train_markets[-1][0]}")
    log.info(f"Test markets:  #{test_markets[0][0]}–#{test_markets[-1][0]}")
    
    # 6. Train model on TRAIN markets only
    train_sample_indices = [si for _, si in train_markets]
    X_train = X[train_sample_indices]
    y_train = y_dir[train_sample_indices]
    
    log.info(f"\nTraining on {len(X_train)} samples ({np.mean(y_train)*100:.1f}% UP)...")
    
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train LightGBM
    import lightgbm as lgb
    
    lgbm_train = lgb.Dataset(
        pd.DataFrame(X_train_scaled, columns=feature_names),
        label=y_train,
    )
    
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "n_jobs": -1,
    }
    
    model = lgb.train(
        params, lgbm_train,
        num_boost_round=300,
        valid_sets=[lgbm_train],
        callbacks=[lgb.log_evaluation(0)],
    )
    
    # Train accuracy
    train_preds = model.predict(X_train_scaled)
    train_acc = np.mean((train_preds > 0.5) == y_train)
    log.info(f"Train accuracy: {train_acc:.3f}")
    
    # 7. Test on UNSEEN markets
    test_sample_indices = [si for _, si in test_markets]
    X_test = X[test_sample_indices]
    y_test = y_dir[test_sample_indices]
    X_test_scaled = scaler.transform(X_test)
    
    test_probs = model.predict(X_test_scaled)  # probability of UP
    test_preds = (test_probs > 0.5).astype(int)
    test_acc = np.mean(test_preds == y_test)
    
    log.info(f"Test accuracy:  {test_acc:.3f} (on {len(test_markets)} unseen markets)")
    
    # 8. Simulate trading on TEST markets
    capital = args.capital
    initial_capital = capital
    trades = []
    equity_curve = [capital]
    
    for (mi, si), prob in zip(test_markets, test_probs):
        market = markets[mi]
        
        # Model predicts direction
        model_says_up = prob > 0.5
        confidence = abs(prob - 0.5) * 2  # 0 to 1
        
        # Confidence filter
        if max(prob, 1 - prob) < args.conf_threshold:
            continue
        
        if capital < args.bet_size:
            break
        
        # Entry price depends on direction
        # If buying the favored side, price = prob (theoretical fair value)
        # In real market you'd pay slightly more (spread), but we use fixed entry price
        entry_price = args.entry_price
        
        # What we're buying
        if model_says_up:
            direction = "UP"
            won = market["outcome"] == "UP"
        else:
            direction = "DOWN"
            won = market["outcome"] == "DOWN"
        
        # Execute trade
        bet = min(args.bet_size, capital)
        shares = bet / entry_price
        capital -= bet
        
        if won:
            # Shares pay $1 each
            payout = shares * 1.0
            pnl = payout - bet
            exit_price = 1.0
        else:
            # Shares worth $0
            payout = 0.0
            pnl = -bet
            exit_price = 0.0
        
        capital += payout
        pnl_pct = (exit_price - entry_price) / entry_price * 100
        
        # Compute entry timing (what % into market lifecycle)
        entry_ts = timestamps[si]
        market_duration_ms = market["end_ts"] - market["start_ts"]
        time_into_market_ms = entry_ts - market["start_ts"]
        entry_pct = time_into_market_ms / max(market_duration_ms, 1) * 100
        time_remaining_min = (market["end_ts"] - entry_ts) / 60_000
        
        trades.append({
            "market_idx": mi,
            "direction": direction,
            "confidence": confidence,
            "model_prob": prob,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "bet_usd": bet,
            "pnl_usd": pnl,
            "pnl_pct": pnl_pct,
            "won": won,
            "outcome": market["outcome"],
            "ptb": market["ptb"],
            "final_sol": market["final_price"],
            "entry_pct": entry_pct,
            "time_remaining_min": time_remaining_min,
        })
        
        equity_curve.append(capital)
    
    # 9. Results
    if not trades:
        log.error("No trades executed. Lower --confidence-threshold?")
        return
    
    trades_df = pd.DataFrame(trades)
    wins = trades_df[trades_df["won"]]
    losses = trades_df[~trades_df["won"]]
    
    total_pnl = trades_df["pnl_usd"].sum()
    win_rate = len(wins) / len(trades_df) * 100
    
    # Sharpe
    returns = trades_df["pnl_pct"].values
    sharpe = np.mean(returns) / max(np.std(returns), 1e-9) * np.sqrt(252) if len(returns) > 1 else 0
    
    # Max drawdown
    eq = np.array(equity_curve)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak * 100
    max_dd = dd.min()
    
    # Profit factor
    gross_profit = wins["pnl_usd"].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses["pnl_usd"].sum()) if len(losses) > 0 else 0.01
    pf = gross_profit / max(gross_loss, 0.01)
    
    # Confidence breakdown
    high_conf = trades_df[trades_df["confidence"] > 0.3]
    high_conf_wr = high_conf["won"].mean() * 100 if len(high_conf) > 0 else 0
    
    up_trades = trades_df[trades_df["direction"] == "UP"]
    dn_trades = trades_df[trades_df["direction"] == "DOWN"]
    up_wr = up_trades["won"].mean() * 100 if len(up_trades) > 0 else 0
    dn_wr = dn_trades["won"].mean() * 100 if len(dn_trades) > 0 else 0
    
    elapsed = time.time() - t0
    
    # Feature importance
    importance = model.feature_importance(importance_type="gain")
    top_features = sorted(zip(feature_names, importance), key=lambda x: -x[1])[:10]
    
    log.info("\n")
    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║     HONEST BACKTEST RESULTS (UNSEEN DATA)              ║")
    log.info("╠══════════════════════════════════════════════════════════╣")
    log.info(f"║  Train Markets:     {len(train_markets)}")
    log.info(f"║  Test Markets:      {len(test_markets)}")
    log.info(f"║  Model Test Acc:    {test_acc:.1%}")
    log.info(f"║  ────────────────────────────────────────────────────")
    log.info(f"║  Total Trades:      {len(trades_df)}")
    log.info(f"║  Win Rate:          {win_rate:.1f}%")
    log.info(f"║  Wins / Losses:     {len(wins)} / {len(losses)}")
    log.info(f"║  ────────────────────────────────────────────────────")
    log.info(f"║  Starting Capital:  ${initial_capital:.2f}")
    log.info(f"║  Final Capital:     ${capital:.2f}")
    log.info(f"║  Total PnL:         ${total_pnl:+.2f}")
    log.info(f"║  Return:            {(capital/initial_capital - 1)*100:+.1f}%")
    log.info(f"║  Max Drawdown:      {max_dd:.1f}%")
    log.info(f"║  Sharpe Ratio:      {sharpe:.2f}")
    log.info(f"║  Profit Factor:     {pf:.2f}")
    log.info(f"║  ────────────────────────────────────────────────────")
    log.info(f"║  UP trades:         {len(up_trades)} ({up_wr:.1f}% WR)")
    log.info(f"║  DOWN trades:       {len(dn_trades)} ({dn_wr:.1f}% WR)")
    log.info(f"║  High conf (>0.3):  {len(high_conf)} ({high_conf_wr:.1f}% WR)")
    log.info(f"║  Avg confidence:    {trades_df['confidence'].mean():.3f}")
    log.info(f"║  ────────────────────────────────────────────────────")
    log.info(f"║  Entry price:       ${args.entry_price:.2f} (fixed)")
    log.info(f"║  Win payout:        ${1.0/args.entry_price:.2f}x")
    log.info(f"║  Time:              {elapsed:.1f}s")
    log.info(f"║  ────────────────────────────────────────────────────")
    log.info(f"║  Top features:")
    for fname, fval in top_features[:5]:
        log.info(f"║    {fname:<35} {fval:.0f}")
    log.info("╚══════════════════════════════════════════════════════════╝")
    
    # ── Entry Timing Analysis ──
    log.info(f"\n  ⏱ Entry Timing (% into market lifecycle):")
    log.info(f"    Avg entry at:    {trades_df['entry_pct'].mean():.1f}% into market")
    log.info(f"    Avg time left:   {trades_df['time_remaining_min'].mean():.1f} min")
    log.info(f"    Min/Max entry:   {trades_df['entry_pct'].min():.0f}% – {trades_df['entry_pct'].max():.0f}%")
    
    # Timing breakdown
    early = trades_df[trades_df['entry_pct'] < 33]
    mid = trades_df[(trades_df['entry_pct'] >= 33) & (trades_df['entry_pct'] < 66)]
    late = trades_df[trades_df['entry_pct'] >= 66]
    log.info(f"    Early (<33%):    {len(early)} trades, {early['won'].mean()*100:.1f}% WR" if len(early) > 0 else "    Early (<33%):    0 trades")
    log.info(f"    Mid (33-66%):    {len(mid)} trades, {mid['won'].mean()*100:.1f}% WR" if len(mid) > 0 else "    Mid (33-66%):    0 trades")
    log.info(f"    Late (>66%):     {len(late)} trades, {late['won'].mean()*100:.1f}% WR" if len(late) > 0 else "    Late (>66%):     0 trades")
    
    # ── Confidence breakdown ──
    log.info(f"\n  🎯 Win Rate by Model Confidence:")
    for lo, hi, label in [(0.50, 0.55, "Low  0.50-0.55"), (0.55, 0.65, "Med  0.55-0.65"), 
                           (0.65, 0.80, "High 0.65-0.80"), (0.80, 1.01, "Ultra 0.80+")]:
        mask = (trades_df['model_prob'].apply(lambda p: max(p, 1-p)) >= lo) & \
               (trades_df['model_prob'].apply(lambda p: max(p, 1-p)) < hi)
        subset = trades_df[mask]
        if len(subset) > 0:
            log.info(f"    {label}: {len(subset):>4} trades, {subset['won'].mean()*100:.1f}% WR")
    
    # ── Entry Price Sensitivity — FULL SIMULATION ──
    log.info(f"\n  💰 Entry Price Simulation (same {len(trades_df)} trades, same {win_rate:.1f}% WR):")
    log.info(f"    {'Price':>6} {'Payout':>7} {'PnL':>10} {'Return':>8} {'EV/bet':>8}")
    for ep in [0.35, 0.40, 0.45, 0.50, 0.55]:
        sim_capital = args.capital
        for _, row in trades_df.iterrows():
            bet = min(args.bet_size, sim_capital)
            if sim_capital < bet:
                break
            shares = bet / ep
            sim_capital -= bet
            if row['won']:
                sim_capital += shares * 1.0
            # else: shares worth $0, already subtracted
        sim_pnl = sim_capital - args.capital
        sim_ret = (sim_capital / args.capital - 1) * 100
        ev = win_rate/100 * (1.0/ep - 1) + (1 - win_rate/100) * (-1)
        log.info(f"    ${ep:.2f}  {1/ep:.2f}x  ${sim_pnl:>+8.0f}  {sim_ret:>+6.0f}%  {ev:>+.3f} {'✅' if ev > 0 else '❌'}")
    
    # Save results
    os.makedirs("results", exist_ok=True)
    trades_df.to_parquet("results/honest_trades.parquet")
    pd.DataFrame({"equity": equity_curve}).to_parquet("results/honest_equity.parquet")
    log.info(f"\nSaved → results/honest_trades.parquet ({len(trades_df)} trades)")


if __name__ == "__main__":
    main()
