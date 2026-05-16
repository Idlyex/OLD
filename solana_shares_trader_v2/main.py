"""Solana Shares Trader v2 — Prediction Market Shares Trading System.

Trades UP/DOWN shares on Polymarket-style prediction markets.
Uses Binance CEX data as features + Polymarket-specific signals.

Usage:
    # Download CEX data (Binance + Bybit) for features
    python main.py --mode download --symbol SOLUSDT --days 30

    # Download Polymarket historical markets
    python main.py --mode download-markets --days 30

    # Record real-time Polymarket data (REQUIRED for honest backtesting)
    python main.py --mode record --interval 5 --duration 24h
    python main.py --mode record --interval 3 --duration infinite

    # Show recorded data summary
    python main.py --mode show-recorded
    python main.py --mode show-recorded --date 2025-05-03

    # Replay backtest on REAL recorded data
    python main.py --mode backtest --replay --market-duration 15
    python main.py --mode backtest --replay --date 2025-05-03 --market-duration 5

    # Backtest on synthetic data (legacy, for testing)
    python main.py --mode backtest --shares --market-duration 15

    # Train on shares data (CEX features + shares targets)
    python main.py --mode train --shares --market-duration 15

    # Live trading on Polymarket
    python main.py --mode live
    python main.py --mode live --live  # real orders
"""

import sys
import os
import asyncio
import argparse
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import load_config, config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Solana Shares Trader v2 — Prediction Market Shares Trading"
    )
    parser.add_argument(
        "--mode",
        choices=["download", "download-markets", "record", "show-recorded", "train", "backtest", "live"],
        default=None,
        help="Mode: download | download-markets | record | show-recorded | train | backtest | live",
    )
    parser.add_argument("--live", action="store_true", help="Enable real order execution")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--data", type=str, default=None, help="Path to data file (parquet/CSV)")
    parser.add_argument(
        "--strategy",
        choices=["ml_hybrid", "microstructure", "regime", "shares_mispricing", "shares_momentum", "shares_hybrid"],
        default="shares_hybrid",
    )
    # Download args
    parser.add_argument("--symbol", type=str, default="SOLUSDT", help="CEX trading symbol")
    parser.add_argument("--days", type=int, default=30, help="Days of history to download")
    parser.add_argument("--granularity", type=str, default="1m", help="Kline interval (1s, 1m, 5m)")
    parser.add_argument("--include-trades", action="store_true", help="Also download raw aggTrades")
    # Training args
    parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter tuning")
    parser.add_argument("--walk-forward", action="store_true", dest="walk_forward", help="Walk-forward optimization")
    parser.add_argument("--forward-minutes", type=int, default=5, help="Target lookahead in minutes")
    # Shares-specific args
    parser.add_argument("--shares", action="store_true", help="Use shares dataset/backtester")
    parser.add_argument("--market-duration", type=int, default=15, dest="market_duration",
                        help="Market duration in minutes (5, 15, 60)")
    parser.add_argument("--slugs", type=str, nargs="+",
                        default=["sol-updown-15m", "sol-updown-5m"],
                        help="Polymarket base slugs")
    # Recorder args
    parser.add_argument("--interval", type=float, default=5.0,
                        help="Recording interval in seconds (default 5)")
    parser.add_argument("--duration", type=str, default="infinite",
                        help="Recording duration: 1h, 6h, 24h, 48h, infinite")
    # Replay backtest args
    parser.add_argument("--replay", action="store_true",
                        help="Use replay backtester on recorded data (instead of synthetic)")
    parser.add_argument("--date", type=str, default=None,
                        help="Date for recorded data (YYYY-MM-DD) or comma-separated dates")
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════
#  MODE: DOWNLOAD
# ═══════════════════════════════════════════════════════════

def run_download(args):
    """Download historical data from exchanges."""
    from core.utils.logger import log
    from data.collector import DataCollector

    log.info(f"═══ Data Download: {args.symbol} / {args.days} days / {args.granularity} ═══")

    collector = DataCollector(symbol=args.symbol)

    async def _run():
        await collector.download_all(
            days=args.days,
            granularity=args.granularity,
            include_trades=args.include_trades,
            trade_days=min(args.days, 7),
        )
        await collector.close()

    asyncio.run(_run())
    log.info("Download complete ✅")


# ═══════════════════════════════════════════════════════════
#  MODE: DOWNLOAD-MARKETS (Polymarket historical)
# ═══════════════════════════════════════════════════════════

def run_download_markets(args):
    """Download historical Polymarket markets."""
    from core.utils.logger import log
    from data.polymarket_collector import PolymarketCollector

    log.info(f"═══ Polymarket Markets Download: {args.days} days ═══")

    collector = PolymarketCollector()

    async def _run():
        await collector.download_all(days=args.days, slugs=args.slugs)
        await collector.close()

    asyncio.run(_run())
    log.info("Polymarket download complete ✅")


# ═══════════════════════════════════════════════════════════
#  MODE: RECORD (Real-time Polymarket data)
# ═══════════════════════════════════════════════════════════

def _parse_duration(s: str) -> Optional[float]:
    """Parse duration string like '1h', '24h', '48h', 'infinite' to seconds."""
    if not s or s.lower() in ("inf", "infinite", "forever"):
        return None
    s = s.strip().lower()
    if s.endswith("h"):
        return float(s[:-1]) * 3600
    elif s.endswith("m"):
        return float(s[:-1]) * 60
    elif s.endswith("d"):
        return float(s[:-1]) * 86400
    elif s.endswith("s"):
        return float(s[:-1])
    try:
        return float(s)
    except ValueError:
        return None


def run_record(args):
    """Record real-time Polymarket data."""
    from core.utils.logger import log
    from data.recorder import PolymarketRecorder

    duration = _parse_duration(args.duration)
    dur_str = f"{duration/3600:.0f}h" if duration else "infinite"
    log.info(f"═══ Recording Polymarket Data: interval={args.interval}s duration={dur_str} ═══")

    recorder = PolymarketRecorder(
        interval_sec=args.interval,
        market_slugs=args.slugs,
        duration_sec=duration,
    )

    try:
        asyncio.run(recorder.start())
    except KeyboardInterrupt:
        log.info("Recording stopped by user (Ctrl+C)")
        asyncio.run(recorder.stop())


# ═══════════════════════════════════════════════════════════
#  MODE: SHOW-RECORDED
# ═══════════════════════════════════════════════════════════

def run_show_recorded(args):
    """Show summary of recorded data."""
    from data.recorder import show_recorded_data
    show_recorded_data(date=args.date)


# ═══════════════════════════════════════════════════════════
#  MODE: TRAIN
# ═══════════════════════════════════════════════════════════

def run_train(args):
    """Train models on downloaded data."""
    from core.utils.logger import log
    import pandas as pd

    # Load data
    data = _load_data(args)
    if data.empty:
        log.error("No data available. Run: python main.py --mode download first.")
        sys.exit(1)

    if args.shares:
        # Shares-specific training: CEX features + shares targets
        log.info(f"═══ Shares Training: {args.market_duration}m markets ═══")
        from training.dataset import TrainingDataset
        from training.train import Trainer

        ds = TrainingDataset()
        dataset = ds.build_shares_dataset(
            data,
            duration_minutes=args.market_duration,
        )

        if dataset["X"].size == 0:
            log.error("Empty shares dataset — need more data")
            sys.exit(1)

        trainer = Trainer()
        trainer.train_full(
            data,
            forward_minutes=args.market_duration,
            walk_forward=True,
            save=True,
            prebuilt_dataset=dataset,
        )
        return

    if args.tune:
        from training.hyper_tuning import HyperTuner
        tuner = HyperTuner(
            n_trials=100,
            n_folds=5,
            forward_minutes=args.forward_minutes,
        )
        tuner.tune(data)

    elif args.walk_forward:
        from training.walk_forward import WalkForwardOptimizer
        wf = WalkForwardOptimizer(
            train_size=min(5000, len(data) // 3),
            test_size=min(1000, len(data) // 10),
            step=min(500, len(data) // 20),
            forward_minutes=args.forward_minutes,
        )
        wf.run(data)

    else:
        from training.train import Trainer
        trainer = Trainer()
        trainer.train_full(
            data,
            forward_minutes=args.forward_minutes,
            walk_forward=True,
            save=True,
        )


# ═══════════════════════════════════════════════════════════
#  MODE: BACKTEST
# ═══════════════════════════════════════════════════════════

def run_backtest(args):
    """Run backtester on historical data."""
    from core.utils.logger import log
    import pandas as pd
    import numpy as np

    data = _load_data(args)

    if args.replay:
        # ═══ REPLAY BACKTESTER (real recorded data) ═══
        from backtester.replay_engine import ReplayBacktester

        dates = args.date.split(",") if args.date else None
        log.info(f"═══ Replay Backtest: {args.market_duration}m | dates={dates or 'all'} ═══")

        strategy = _get_strategy(args.strategy)
        bt = ReplayBacktester()
        results = bt.run(
            dates=dates,
            duration_minutes=args.market_duration,
            strategy=strategy,
            verbose=True,
        )

        results_dir = config.get("backtester", {}).get("results_dir", "results")
        os.makedirs(results_dir, exist_ok=True)

        trades_df = bt.get_trades_df()
        if not trades_df.empty:
            trades_df.to_parquet(f"{results_dir}/replay_trades.parquet")
            log.info(f"Replay trades saved → {results_dir}/replay_trades.parquet")

        equity_df = bt.get_equity_df()
        if not equity_df.empty:
            equity_df.to_parquet(f"{results_dir}/replay_equity.parquet")
            log.info(f"Replay equity saved → {results_dir}/replay_equity.parquet")

        return

    if args.shares:
        # ═══ SHARES BACKTESTER (synthetic data) ═══
        from backtester.shares_engine import SharesBacktester

        log.info(f"═══ Shares Backtest (synthetic): {args.market_duration}m markets ═══")

        if data.empty:
            log.warning("No real data — generating synthetic SOL data for shares demo")
            data = _generate_synthetic_sol(n_bars=4000)

        strategy = _get_strategy(args.strategy)
        bt = SharesBacktester()
        results = bt.run(
            sol_data=data,
            strategy=strategy,
            duration_minutes=args.market_duration,
            verbose=True,
        )

        results_dir = config.get("backtester", {}).get("results_dir", "results")
        os.makedirs(results_dir, exist_ok=True)

        trades_df = bt.get_trades_df()
        if not trades_df.empty:
            trades_df.to_parquet(f"{results_dir}/shares_trades.parquet")
            log.info(f"Shares trades saved → {results_dir}/shares_trades.parquet")

        equity_df = bt.get_equity_df()
        if not equity_df.empty:
            equity_df.to_parquet(f"{results_dir}/shares_equity.parquet")
            log.info(f"Shares equity saved → {results_dir}/shares_equity.parquet")

        return

    # ═══ LEGACY CEX BACKTESTER ═══
    from backtester.engine import Backtester
    from dashboard.live_dashboard import LiveDashboard

    log.info("═══ Starting Legacy Backtester ═══")

    if data.empty:
        log.warning("No real data — generating synthetic SOL data for demo")
        data = _generate_synthetic_sol()

    strategy = _get_strategy(args.strategy)

    bt = Backtester()
    results = bt.run(data, strategy=strategy, verbose=True)

    dashboard = LiveDashboard()
    dashboard.print_backtest_results(
        results, trades_df=bt.get_trades_df(), equity_df=bt.get_equity_df(),
    )

    results_dir = config.get("backtester", {}).get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)

    trades_df = bt.get_trades_df()
    if not trades_df.empty:
        trades_df.to_parquet(f"{results_dir}/trades.parquet")
        log.info(f"Trades saved → {results_dir}/trades.parquet")

    equity_df = bt.get_equity_df()
    if not equity_df.empty:
        equity_df.to_parquet(f"{results_dir}/equity.parquet")
        log.info(f"Equity saved → {results_dir}/equity.parquet")


# ═══════════════════════════════════════════════════════════
#  MODE: LIVE
# ═══════════════════════════════════════════════════════════

def run_live(args):
    """Run ML live trader — buy cheap shares, hold to resolution."""
    from core.utils.logger import log
    from live_trader.ml_shares_trader import MLSharesTrader

    if args.live:
        config["dry_run"] = False
        log.warning("⚠️  LIVE MODE — real CLOB orders will be placed!")
    else:
        config["dry_run"] = True
        log.info("🔵 DRY RUN mode — no real money, paper trading only")

    trader = MLSharesTrader()

    try:
        asyncio.run(trader.start())
    except KeyboardInterrupt:
        log.info("Interrupted by user (Ctrl+C)")
        asyncio.run(trader.shutdown())


# ═══════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════

def _load_data(args) -> "pd.DataFrame":
    """Load data from --data path or auto-detect from data/processed/."""
    import pandas as pd

    # Explicit path
    if args.data and os.path.exists(args.data):
        if args.data.endswith(".parquet"):
            df = pd.read_parquet(args.data)
        else:
            df = pd.read_csv(args.data)
        from core.utils.logger import log
        log.info(f"Loaded {len(df):,} rows from {args.data}")
        return df

    # Auto-detect from processed dir
    processed_path = f"data/processed/{args.symbol}_processed.parquet"
    if os.path.exists(processed_path):
        df = pd.read_parquet(processed_path)
        from core.utils.logger import log
        log.info(f"Auto-loaded {len(df):,} rows from {processed_path}")
        return df

    return pd.DataFrame()


def _get_strategy(name: str):
    """Get strategy instance by name."""
    # Shares strategies
    if name.startswith("shares_"):
        from strategies.shares import SharesMispricingStrategy, SharesMomentumStrategy, SharesHybridStrategy
        shares_strategies = {
            "shares_mispricing": SharesMispricingStrategy,
            "shares_momentum": SharesMomentumStrategy,
            "shares_hybrid": SharesHybridStrategy,
        }
        cls = shares_strategies.get(name, SharesHybridStrategy)
        return cls()

    # Legacy CEX strategies
    from strategies.base import MLHybridStrategy, MicrostructureStrategy, RegimeAwareStrategy
    strategies = {
        "ml_hybrid": MLHybridStrategy,
        "microstructure": MicrostructureStrategy,
        "regime": RegimeAwareStrategy,
    }
    cls = strategies.get(name, MLHybridStrategy)
    return cls()


def _generate_synthetic_sol(n_bars: int = 2000) -> "pd.DataFrame":
    """Generate synthetic SOL OHLCV data for demo/testing."""
    import numpy as np
    import pandas as pd

    np.random.seed(42)
    price = 150.0
    prices = [price]
    for _ in range(n_bars - 1):
        price *= 1 + np.random.normal(0, 0.003)
        prices.append(price)
    prices = np.array(prices)
    return pd.DataFrame({
        "ts": np.arange(n_bars) * 60_000,
        "open": prices * (1 + np.random.normal(0, 0.001, n_bars)),
        "high": prices * (1 + np.abs(np.random.normal(0, 0.002, n_bars))),
        "low": prices * (1 - np.abs(np.random.normal(0, 0.002, n_bars))),
        "close": prices,
        "volume": np.random.exponential(1000, n_bars),
        "taker_buy_volume": np.random.exponential(500, n_bars),
    })


def main():
    args = parse_args()

    if args.config:
        load_config(args.config)

    mode = args.mode or config.get("mode", "backtest")

    from core.utils.logger import log

    if mode == "download":
        run_download(args)
    elif mode == "download-markets":
        run_download_markets(args)
    elif mode == "record":
        run_record(args)
    elif mode == "show-recorded":
        run_show_recorded(args)
    elif mode == "train":
        run_train(args)
    elif mode == "backtest":
        run_backtest(args)
    elif mode == "live":
        run_live(args)
    else:
        log.error(f"Unknown mode: {mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
