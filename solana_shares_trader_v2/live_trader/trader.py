"""Live Trader — main async event loop for real-time trading.

Orchestrates: data collection → feature computation → model prediction →
              execution → risk management → dashboard updates.

All in one async loop, single entry point.
"""

import asyncio
import time
import signal
import numpy as np
from collections import deque
from typing import Dict, Optional

from core.features.engine import FeatureEngine
from core.models.hybrid_model import HybridModel
from core.risk.manager import RiskManager
from core.execution.executor import ExecutionEngine
from infrastructure.api.polymarket_clob import PolymarketCLOB
from infrastructure.api.gamma_api import GammaAPI
from infrastructure.api.binance_ws import BinanceWS
from infrastructure.collectors.cex_collector import CEXCollector
from infrastructure.collectors.onchain_collector import OnchainCollector
from infrastructure.database.storage import ParquetStore, ClickHouseStore
from dashboard.live_dashboard import LiveDashboard
from strategies.base import MLHybridStrategy
from core.utils.logger import log
from config import config


class LiveTrader:
    """Main live trading orchestrator."""

    def __init__(self):
        # Components
        self.clob = PolymarketCLOB()
        self.gamma = GammaAPI()
        self.binance_ws = BinanceWS()
        self.cex_collector = CEXCollector()
        self.onchain_collector = OnchainCollector()
        self.feature_engine = FeatureEngine()
        self.model = HybridModel()
        self.risk_manager = RiskManager()
        self.execution = ExecutionEngine(self.clob, self.risk_manager, self.model)
        self.parquet = ParquetStore()
        self.clickhouse = ClickHouseStore()
        self.dashboard = LiveDashboard()
        self.strategy = MLHybridStrategy(self.model)

        # State
        self.active_markets = []
        self._running = False
        self._cycle = 0
        self._feature_sequences: Dict[str, deque] = {}
        self._analysis_interval = config.get("timing", {}).get("analysis_interval_ms", 500) / 1000

        # Config
        self._market_slugs = config.get("markets", {}).get("slugs", [])
        self._symbols = [s.upper() for s in config.get("infrastructure", {}).get("binance", {}).get("symbols", ["SOLUSDT"])]

    async def start(self):
        """Initialize all components and start the trading loop."""
        log.info("╔══════════════════════════════════════════════════╗")
        log.info("║  Solana Shares Trader v2                        ║")
        log.info(f"║  Mode: {'DRY RUN' if config.get('dry_run', True) else 'LIVE'}  │  Markets: {', '.join(self._market_slugs):<15}║")
        log.info("╚══════════════════════════════════════════════════╝")

        # Initialize CLOB client
        await self.clob.init()

        # Load models
        if self.model.load("latest"):
            log.info("Models loaded from disk")
        else:
            log.warning("No pre-trained models found — running with defaults")

        # Connect to ClickHouse
        await self.clickhouse.connect()

        # Wire Binance WS events to collector
        self.binance_ws.on("kline", self.cex_collector.on_kline)
        self.binance_ws.on("trade", self.cex_collector.on_trade)
        self.binance_ws.on("mark_price", self.cex_collector.on_mark_price)
        self.binance_ws.on("orderbook", self.cex_collector.on_orderbook)
        self.binance_ws.on("liquidation", self.cex_collector.on_liquidation)

        # Bind dashboard
        self.dashboard.bind(
            execution_engine=self.execution,
            risk_manager=self.risk_manager,
            feature_engine=self.feature_engine,
            model=self.model,
            cex_collector=self.cex_collector,
        )

        self._running = True

        # Start async tasks
        tasks = [
            asyncio.create_task(self.binance_ws.connect()),
            asyncio.create_task(self.onchain_collector.start()),
            asyncio.create_task(self._analysis_loop()),
            asyncio.create_task(self._market_refresh_loop()),
            asyncio.create_task(self._data_flush_loop()),
            asyncio.create_task(self.dashboard.start()),
        ]

        # Handle shutdown
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))
            except NotImplementedError:
                pass  # Windows doesn't support add_signal_handler

        log.info(f"🚀 Live trader started — {len(tasks)} async tasks")

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
        except KeyboardInterrupt:
            await self.shutdown()

    async def _analysis_loop(self):
        """Main analysis cycle: features → strategy → execution → monitoring."""
        log.info(f"Analysis loop: every {self._analysis_interval * 1000:.0f}ms")

        # Wait for initial data
        await asyncio.sleep(5)

        while self._running:
            try:
                self._cycle += 1
                await self._run_analysis()
            except Exception as e:
                log.error(f"Analysis loop error: {e}")

            await asyncio.sleep(self._analysis_interval)

    async def _run_analysis(self):
        """Single analysis cycle."""
        # Compute features for each symbol
        for symbol in self._symbols:
            current_price = self.cex_collector.get_latest_price(symbol)
            if current_price <= 0:
                continue

            # Compute all 82 features
            features = self.feature_engine.compute_all(
                symbol=symbol,
                cex_collector=self.cex_collector,
                onchain_collector=self.onchain_collector,
                current_price=current_price,
            )

            # Build feature sequence for transformer
            if symbol not in self._feature_sequences:
                seq_len = config.get("models", {}).get("primary", {}).get("sequence_length", 60)
                self._feature_sequences[symbol] = deque(maxlen=seq_len)

            fv = self.feature_engine.get_feature_vector(features)
            self._feature_sequences[symbol].append(fv)

            # Save features
            self.parquet.append_features(symbol, features)

            # Update risk tracking
            vol = features.get("vol_garman_klass", 0.003)
            self.risk_manager.update_volatility(vol)

            # Evaluate strategy on each active market
            for market_info in self.active_markets:
                market = market_info.get("market", {})
                slug = market.get("slug", "")
                sym = slug.split("-")[0].upper()

                if sym + "USDT" != symbol:
                    continue

                # Build signal
                signal_data = {
                    "slug": slug,
                    "token_id": market.get("yes_token_id", ""),
                    "yes_token_id": market.get("yes_token_id", ""),
                    "no_token_id": market.get("no_token_id", ""),
                    "condition_id": market.get("condition_id", ""),
                    "price_to_beat": market.get("price_to_beat"),
                    "end_date": market.get("end_date", ""),
                    "share_price": market.get("yes_price", 0.5),
                }

                # Get strategy signal
                signal = self.strategy.evaluate(features, {"price": current_price})
                if signal:
                    signal_data.update(signal)
                    signal_data["token_id"] = (
                        market.get("yes_token_id", "")
                        if signal["direction"] == "UP"
                        else market.get("no_token_id", "")
                    )
                    share_price = market.get("yes_price", 0.5)
                    signal_data["share_price"] = (
                        share_price if signal["direction"] == "UP"
                        else 1 - share_price
                    )

                    # Execute
                    seq = np.array(list(self._feature_sequences[symbol])) if len(self._feature_sequences[symbol]) >= 10 else None
                    capital = self.risk_manager._equity_curve[-1] if self.risk_manager._equity_curve else 100.0

                    await self.execution.evaluate_entry(
                        signal=signal_data,
                        features=features,
                        feature_sequence=seq,
                        capital=capital,
                    )

        # Monitor open positions
        def price_getter(token_id):
            # Simple price lookup — in real impl would use PM WS
            return 0.5  # placeholder

        def features_getter(symbol):
            if symbol in self._symbols:
                return self.feature_engine.compute_all(
                    symbol=symbol,
                    cex_collector=self.cex_collector,
                    current_price=self.cex_collector.get_latest_price(symbol),
                )
            return None

        await self.execution.monitor_positions(price_getter, features_getter)

        # Update equity
        equity = 100.0 + self.execution._total_pnl  # simplified
        self.risk_manager.update_equity(equity)

        # Status log every 30 cycles
        if self._cycle % 60 == 0:
            stats = self.execution.get_stats()
            risk = self.risk_manager.get_risk_metrics()
            fe_stats = self.feature_engine.get_stats()

            prices = []
            for sym in self._symbols:
                p = self.cex_collector.get_latest_price(sym)
                if p > 0:
                    prices.append(f"{sym}=${p:.2f}")

            wr = stats.get("win_rate", 0)
            log.info(
                f"⚡ {' '.join(prices)} │ "
                f"Pos={stats['open_positions']} T={stats['total_trades']} "
                f"WR={wr:.0f}% PnL=${stats['total_pnl']:+.2f} │ "
                f"DD={risk['max_drawdown_pct']:.1f}% │ "
                f"Features: {fe_stats['avg_time_ms']:.1f}ms"
            )

    async def _market_refresh_loop(self):
        """Refresh active markets periodically."""
        refresh_ms = config.get("timing", {}).get("market_refresh_ms", 12000) / 1000

        while self._running:
            try:
                self.active_markets = await self.gamma.get_all_nearest_markets()
                if self.active_markets:
                    slugs = [m["market"]["slug"] for m in self.active_markets]
                    log.debug(f"Markets: {len(self.active_markets)} active — {', '.join(slugs)}")
            except Exception as e:
                log.error(f"Market refresh error: {e}")

            await asyncio.sleep(refresh_ms)

    async def _data_flush_loop(self):
        """Periodically flush data to parquet/clickhouse."""
        while self._running:
            try:
                self.parquet.flush_all()
            except Exception as e:
                log.error(f"Data flush error: {e}")
            await asyncio.sleep(60)

    async def shutdown(self):
        """Graceful shutdown."""
        log.info("🛑 Shutting down...")
        self._running = False
        self.dashboard.stop()
        self.binance_ws.destroy()
        self.onchain_collector.stop()
        self.parquet.flush_all()
        await self.clob.close()
        log.info("Shutdown complete ✅")
