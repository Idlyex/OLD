"""ML Shares Live Trader — buy cheap shares, hold to resolution.

Strategy proven on 10-day honest backtest (unseen data):
  - 15m: 64.9% WR, Sharpe 4.95, $100→$268
  - 5m: 65.7% WR, Sharpe 5.25, $100→$614
  - Ultra confidence (>0.80): 70-78% WR

How it works:
  1. Collects rolling 60x 1m Binance klines (same as training)
  2. Discovers active Polymarket markets via Gamma API
  3. At ~20% into each market, computes 104 features → LGBM predicts direction
  4. If confidence > threshold AND share_price < max_price → BUY
  5. Hold to expiry: win = $1, lose = $0 (no stops needed)
  6. Also records all data for future replay backtesting

Usage:
  python main.py --mode live              # dry run (paper trade)
  python main.py --mode live --live       # real CLOB orders
"""

import asyncio
import time
import math
import json
import joblib
import numpy as np
import pandas as pd
import httpx
from pathlib import Path
from collections import deque
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()
import os

from core.utils.logger import log
from core.features.shares import compute_shares_features
from data.polymarket_collector import PolymarketCollector, SharesMarket
from api.clob_client import PolymarketCLOB, OrderResult
from config import config, trading_config

# ═══════════════════════════════════════════════════════════
#  DATA CLASSES
# ═══════════════════════════════════════════════════════════

@dataclass
class MLPosition:
    """Active ML-driven shares position."""
    market_slug: str
    token_id: str
    direction: str        # "UP" or "DOWN"
    entry_price: float
    shares: float
    size_usd: float
    entry_ts: float
    confidence: float     # directional probability (0.5-1.0)
    model_prob: float     # raw UP probability from model
    price_to_beat: float
    duration_minutes: int
    end_date: str
    sol_price_at_entry: float
    # ML metadata
    primary_model: str = "lgbm"  # which model made the decision
    inference_ms: float = 0.0    # ML inference time
    all_model_probs: dict = field(default_factory=dict)  # {model: prob_up}
    # Live tracking
    current_price: float = 0.0
    current_pnl_pct: float = 0.0
    neg_risk: bool = False
    peak_pnl_pct: float = 0.0
    sell_order_id: str = ""  # GTC sell limit @ $0.99 placed after buy
    resolved: bool = False
    outcome: str = ""     # "UP" or "DOWN" after resolution


@dataclass
class TradeResult:
    """Completed trade."""
    slug: str
    direction: str
    entry_price: float
    exit_price: float
    shares: float
    pnl_usd: float
    pnl_pct: float
    confidence: float
    model_prob: float
    hold_time_s: float
    reason: str           # "expiry_win", "expiry_loss", "timeout"
    sol_at_entry: float
    sol_at_exit: float
    ptb: float
    ts: float


# ═══════════════════════════════════════════════════════════
#  ML SHARES TRADER
# ═══════════════════════════════════════════════════════════

class MLSharesTrader:
    """Live ML-driven shares trader — buy cheap, hold to resolution."""

    def __init__(self):
        self.collector = PolymarketCollector()
        self.positions: List[MLPosition] = []
        self.completed: List[TradeResult] = []

        # Config — from trading.yaml
        _entry = trading_config.get("entry", {})
        _exec = trading_config.get("execution", {})
        _markets = trading_config.get("markets", {})

        self.order_size = _exec.get("order_size_usd", 2.0)
        self.max_positions = _exec.get("max_open_positions", 5)
        self.min_confidence = _entry.get("min_confidence", 0.75)
        self.max_share_price = _entry.get("max_share_price", 0.60)
        self.min_share_price = _entry.get("min_share_price", 0.10)
        self.min_entry_pct = _entry.get("min_entry_pct", 0.60)
        self.max_entry_pct = _entry.get("max_entry_pct", 1.00)
        self.max_spread = _entry.get("max_spread", 0.08)
        self.min_depth = _entry.get("min_depth_shares", 20)
        self.primary_model_name = trading_config.get("model", {}).get("primary", "xgboost")
        # DRY_RUN: check .env first (DRY_RUN=false → live), then config
        env_dry = os.getenv("DRY_RUN", "").strip().lower()
        if env_dry in ("false", "0", "no"):
            self.dry_run = False
        elif env_dry in ("true", "1", "yes"):
            self.dry_run = True
        else:
            self.dry_run = config.get("dry_run", True)
        self.trade_slugs = _markets.get("trade_slugs", ["sol-updown-5m"])
        self.record_slugs = _markets.get("record_slugs", _markets.get("trade_slugs", ["sol-updown-5m"]))
        self.slugs = list(set(self.trade_slugs + self.record_slugs))  # all slugs for discovery

        # CLOB trading client (real Polymarket orders)
        self._clob = PolymarketCLOB(dry_run=self.dry_run)

        # ML models (all loaded for multi-model prediction)
        self._models = {}  # name -> model object
        self._model = None  # primary (catboost) for entry decisions
        self._scaler = None
        self._feature_names = None

        # Real-time data
        self._sol_price: float = 0.0
        self._klines: Dict[int, Dict] = {}  # ts → kline bar (dedup by ts)
        self._sol_volatility: float = 0.003
        self._http: Optional[httpx.AsyncClient] = None

        # Market tracking
        self._active_markets: Dict[str, Dict] = {}  # slug → {market, ptb, discovered_ts}
        self._market_entry_done: set = set()  # markets where entry decision is final
        self._pending_signals: Dict[str, Dict] = {}  # slug → {direction, prob, dir_prob, ...} waiting for cheap price
        self._clob_updates: int = 0  # count of CLOB price refreshes

        # Per-trade snapshot buffer: slug -> list of {ts, sol, up_bid, up_ask, dn_bid, dn_ask, ...}
        self._trade_snapshots: Dict[str, List[Dict]] = {}

        # Pre-signed order cache: slug -> {signed_order, token_id, price, shares, amount_usd, ts}
        self._presigned_orders: Dict[str, Dict] = {}

        # ── Prediction single-source-of-truth (from recording loop only) ──
        self._latest_predictions: Dict[str, Dict] = {}   # slug → {result, market, ptb, ts}
        self._prediction_history: Dict[str, list] = {}    # slug → [{prob_up, direction, ts}, ...]
        self._SMOOTHING_WINDOW = 3  # require N consecutive same-direction ticks

        # Telegram bot
        self._tg = None  # initialized in start()

        # Control
        self._running = False
        self._start_ts = 0
        self._capital = 100.0  # virtual capital for dry run

    @staticmethod
    def _end_date_from_slug(slug: str) -> str:
        """Extract end_date ISO string from slug. Slug epoch = START time."""
        import re
        ts_m = re.search(r'-(\d{10})$', slug)
        dur_m = re.search(r'-(\d+)m', slug)
        if ts_m and dur_m:
            from datetime import datetime as _dt, timezone as _tz
            start_epoch = int(ts_m.group(1))
            dur_sec = int(dur_m.group(1)) * 60
            end_epoch = start_epoch + dur_sec
            return _dt.fromtimestamp(end_epoch, tz=_tz.utc).isoformat()
        return ""

    @staticmethod
    def _elapsed_pct(slug: str, duration_minutes: int) -> float:
        """Compute real-time elapsed percentage from slug epoch. Never stale.
        Slug epoch = START time. End = start + duration."""
        import re
        m = re.search(r'-(\d{10})$', slug)
        if not m:
            return 0.0
        start_epoch = int(m.group(1))
        end_epoch = start_epoch + duration_minutes * 60
        now = time.time()
        if now <= start_epoch:
            return 0.0
        if now >= end_epoch:
            return 1.0
        return (now - start_epoch) / (end_epoch - start_epoch)

    # ═══════════════════════════════════════════════════════════
    #  STARTUP
    # ═══════════════════════════════════════════════════════════

    async def start(self):
        """Start the ML live trader."""
        self._start_ts = time.time()
        self._http = httpx.AsyncClient(timeout=10.0)

        log.info("╔══════════════════════════════════════════════════════════╗")
        log.info(f"║  ML Shares Trader — {self.primary_model_name.upper()} PRIMARY")
        log.info("╠══════════════════════════════════════════════════════════╣")
        log.info(f"║  Mode:          {'🔵 DRY RUN' if self.dry_run else '🔴 LIVE ORDERS'}")
        log.info(f"║  Primary model: {self.primary_model_name}")
        log.info(f"║  Trade:         {', '.join(self.trade_slugs)}")
        log.info(f"║  Record:        {', '.join(self.record_slugs)}")
        log.info(f"║  Max share $:   ${self.max_share_price:.2f}")
        log.info(f"║  Min share $:   ${self.min_share_price:.2f}")
        log.info(f"║  Min confidence: {self.min_confidence:.0%}")
        log.info(f"║  Entry window:  {self.min_entry_pct:.0%}-{self.max_entry_pct:.0%}")
        log.info(f"║  Order size:    ${self.order_size:.2f}")
        log.info(f"║  Max positions: {self.max_positions}")
        log.info(f"║  Orderbook:     WebSocket (real-time)")
        log.info("╚══════════════════════════════════════════════════════════╝")

        # Load ML model
        self._load_model()

        # Initialize CLOB trading client
        await self._clob.init()
        if not self.dry_run and not self._clob.is_read_only:
            bal = await self._clob.get_balance()
            if bal is not None:
                log.info(f"║  💰 USDC Balance: ${bal:.2f}")
                self._capital = bal  # use real USDC balance
            else:
                log.warning("║  ⚠️ Could not fetch USDC balance")
        elif self._clob.is_read_only:
            log.warning(f"║  ⚠️ CLOB read-only: {self._clob._read_only_reason}")

        # Telegram bot
        try:
            from telegram_bot import TelegramNotifier
            self._tg = TelegramNotifier(self)
            await self._tg.start()
        except ImportError:
            log.warning("  ⚠️ python-telegram-bot not installed — TG notifications disabled")
            self._tg = None
        except Exception as e:
            log.warning(f"  ⚠️ TG bot error: {e}")
            self._tg = None

        # Load previous trades (survive restarts) — controlled by config
        _persist = trading_config.get("persistence", {})
        if _persist.get("load_trades_on_restart", False):
            self._load_trades()

        # Warm up kline buffer
        await self._warmup_klines()

        self._running = True

        # Launch concurrent loops
        tasks = [
            self._kline_loop(),           # Fetch 1m klines every 10s
            self._sol_price_loop(),       # SOL price every 3s
            self._market_discovery_loop(),# Find active markets every 12s
            self._orderbook_ws_loop(),    # WebSocket orderbook (real-time, ~1s updates)
            self._entry_loop(),           # Check entries every 5s
            self._recording_loop(),       # Record ML for ALL markets 0-100% every 5s
            self._resolution_loop(),      # Check market resolutions every 10s
            self._status_loop(),          # Print status every 30s
            self._trade_snapshot_loop(),  # Record per-trade orderbook every 3s
        ]

        # Auto-redeem resolved positions every 2 min (claims winnings via relayer)
        if not self.dry_run:
            from api.auto_redeem import start_auto_redeem_loop
            tasks.append(start_auto_redeem_loop(interval_s=120))
            log.info("  💰 Auto-redeem enabled: every 120s")

        # Always-on 1s recorder for replay testing later
        from data.recorder import PolymarketRecorder
        recorder = PolymarketRecorder(
            interval_sec=1.0,
            market_slugs=self.slugs,
            duration_sec=None,
        )
        tasks.append(recorder.start())
        log.info("  📹 Recorder enabled: 1s interval, all markets")

        await asyncio.gather(*tasks)

    async def shutdown(self):
        self._running = False
        # Cancel any remaining open CLOB orders
        if not self.dry_run and not self._clob.is_read_only:
            await self._clob.cancel_all()

        # Flush outcomes for all remaining active markets
        for s, d in list(self._active_markets.items()):
            mkt = d["market"]
            ptb_val = d.get("ptb", self._sol_price)
            outcome = "UP" if self._sol_price >= ptb_val else "DOWN"
            self._record_outcome(
                s, outcome,
                sol_start=ptb_val, sol_end=self._sol_price,
                ptb=ptb_val, dur_min=mkt.duration_minutes,
            )
        self._active_markets.clear()

        if self._tg:
            await self._tg.stop()
        if self._http:
            await self._http.aclose()
        await self.collector.close()
        self._print_summary()

    def _load_model(self):
        """Load ALL trained models + scaler + feature names."""
        model_dir = Path("training/model_registry/latest")
        meta_path = model_dir / "meta.json"
        scaler_path = model_dir / "scaler.pkl"

        if not model_dir.exists():
            log.warning("No model registry — training fresh model...")
            self._train_fresh_model()
            return

        # Load meta
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            self._feature_names = meta.get("feature_names")
            model_names = meta.get("models", ["lgbm"])
            log.info(f"  ✅ {len(self._feature_names)} features, models: {model_names}")
        else:
            model_names = ["lgbm"]

        # Load scaler
        if scaler_path.exists():
            self._scaler = joblib.load(scaler_path)
            log.info(f"  ✅ Loaded scaler")

        # Load ALL models
        for name in model_names:
            p = model_dir / f"{name}_cls.pkl"
            if p.exists():
                self._models[name] = joblib.load(p)
                log.info(f"  ✅ Loaded {name} model")

        # Primary model for entry decisions (CatBoost: 82.8% WR @ conf>=60%)
        self._model = self._models.get("catboost", self._models.get("lgbm"))
        if not self._model and self._models:
            self._model = next(iter(self._models.values()))

        log.info(f"  ✅ Total models loaded: {len(self._models)} ({list(self._models.keys())})")

    def _train_fresh_model(self):
        """Train a fresh model from available data."""
        from training.dataset import TrainingDataset
        import lightgbm as lgb
        from sklearn.preprocessing import RobustScaler

        # Load available data
        data_path = "data/processed/SOLUSDT_processed.parquet"
        if not Path(data_path).exists():
            log.error(f"No data at {data_path}. Run: python main.py --mode download --days 10")
            return

        sol_data = pd.read_parquet(data_path)
        log.info(f"  Training on {len(sol_data):,} bars...")

        ds = TrainingDataset()

        # Build for 5m markets (more data)
        dataset = ds.build_shares_dataset(sol_data, duration_minutes=5)
        X = dataset["X"]
        y = dataset["y_direction"]
        self._feature_names = dataset["feature_names"]

        if X.size == 0:
            log.error("Empty dataset")
            return

        self._scaler = RobustScaler()
        X_scaled = self._scaler.fit_transform(X)

        lgbm_ds = lgb.Dataset(
            pd.DataFrame(X_scaled, columns=self._feature_names),
            label=y,
        )
        params = {
            "objective": "binary", "metric": "binary_logloss",
            "num_leaves": 31, "learning_rate": 0.05,
            "feature_fraction": 0.8, "bagging_fraction": 0.8,
            "bagging_freq": 5, "verbose": -1, "n_jobs": -1,
        }
        self._model = lgb.train(params, lgbm_ds, num_boost_round=300,
                                callbacks=[lgb.log_evaluation(0)])

        # Save for future use
        model_dir = Path("training/model_registry/latest")
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, model_dir / "lgbm_cls.pkl")
        joblib.dump(self._scaler, model_dir / "scaler.pkl")
        with open(model_dir / "meta.json", "w") as f:
            json.dump({"feature_names": self._feature_names}, f)

        train_preds = self._model.predict(X_scaled)
        train_acc = np.mean((train_preds > 0.5) == y)
        log.info(f"  ✅ Fresh model trained: {X.shape}, acc={train_acc:.3f}")

    # ═══════════════════════════════════════════════════════════
    #  DATA COLLECTION (same as training)
    # ═══════════════════════════════════════════════════════════

    def _kline_buffer(self) -> list:
        """Sorted kline list from dict."""
        return [self._klines[k] for k in sorted(self._klines.keys())]

    async def _fetch_ptb_for_slug(self, slug: str) -> Optional[float]:
        """Fetch exact SOL price at market start from Pyth Network.

        Polymarket uses Chainlink (→ Pyth) SOL-USD for PTB.
        Pyth Hermes API gives historical prices at exact timestamps — free, no auth.
        Accuracy: ~$0.0003 vs actual PM PTB.
        """
        import re
        m = re.search(r'-(\d{10,})$', slug)
        if not m:
            return None
        start_ts_sec = int(m.group(1))

        # Pyth SOL/USD price feed ID
        pyth_sol_id = "0xef0d8b6fda2ceba41da15d4095d1da392a0d2f8ed0c6c7bc0f4cfac8c280b56d"

        # Method 1: Pyth historical price at exact market start timestamp
        try:
            r = await self._http.get(
                f"https://hermes.pyth.network/v2/updates/price/{start_ts_sec}",
                params={"ids[]": pyth_sol_id, "parsed": "true"},
            )
            data = r.json()
            if "parsed" in data and data["parsed"]:
                p = data["parsed"][0]["price"]
                price = int(p["price"]) * 10 ** int(p["expo"])
                return round(price, 6)
        except Exception as e:
            log.debug(f"  PTB Pyth fetch failed: {e}")

        # Method 2: Fallback to Binance 1m kline open price
        try:
            r = await self._http.get(
                "https://fapi.binance.com/fapi/v1/klines",
                params={
                    "symbol": "SOLUSDT",
                    "interval": "1m",
                    "startTime": start_ts_sec * 1000,
                    "limit": 1,
                },
            )
            klines = r.json()
            if klines and len(klines) > 0:
                return float(klines[0][1])  # open price
        except Exception as e:
            log.debug(f"  PTB Binance fetch failed: {e}")

        return None

    def _upsert_kline(self, k):
        """Insert or update a kline bar."""
        ts = int(k[0])
        self._klines[ts] = {
            "ts": ts,
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
            "taker_buy_volume": float(k[9]) if len(k) > 9 else float(k[5]) * 0.5,
        }
        # Keep only last 120 bars
        if len(self._klines) > 120:
            oldest = min(self._klines.keys())
            del self._klines[oldest]

    async def _warmup_klines(self):
        """Fetch last 65 1m klines to warm up buffer."""
        log.info("  Warming up kline buffer (65 x 1m bars)...")
        try:
            r = await self._http.get(
                "https://fapi.binance.com/fapi/v1/klines",
                params={"symbol": "SOLUSDT", "interval": "1m", "limit": 65},
            )
            data = r.json()
            if isinstance(data, list):
                for k in data:
                    self._upsert_kline(k)
                buf = self._kline_buffer()
                self._sol_price = buf[-1]["close"] if buf else 0
                # Compute volatility from close prices
                if len(buf) > 10:
                    closes = np.array([b["close"] for b in buf])
                    log_rets = np.diff(np.log(closes))
                    self._sol_volatility = float(np.std(log_rets))
                log.info(f"  ✅ Kline buffer: {len(self._klines)} bars, SOL=${self._sol_price:.2f}, vol={self._sol_volatility:.5f}")
            else:
                log.error(f"Kline warmup got unexpected response: {str(data)[:200]}")
        except Exception as e:
            log.error(f"Kline warmup failed: {e}")

    async def _kline_loop(self):
        """Fetch latest 1m klines every 10 seconds."""
        while self._running:
            try:
                r = await self._http.get(
                    "https://fapi.binance.com/fapi/v1/klines",
                    params={"symbol": "SOLUSDT", "interval": "1m", "limit": 3},
                )
                data = r.json()
                if isinstance(data, list):
                    for k in data:
                        self._upsert_kline(k)
                    # Update SOL price and volatility
                    buf = self._kline_buffer()
                    if buf:
                        self._sol_price = buf[-1]["close"]
                    if len(buf) > 10:
                        closes = np.array([b["close"] for b in buf[-60:]])
                        log_rets = np.diff(np.log(closes))
                        self._sol_volatility = float(np.std(log_rets))
            except Exception as e:
                log.debug(f"Kline fetch error: {e}")
            await asyncio.sleep(10.0)

    async def _sol_price_loop(self):
        """High-frequency SOL price update from Pyth (same oracle as Polymarket).

        Polymarket resolves via Chainlink ← Pyth. Binance is ~$0.03 lower.
        Using Pyth ensures our gap calculations match what PM actually sees.
        Binance kept as fallback if Pyth is down.
        """
        pyth_sol_id = "0xef0d8b6fda2ceba41da15d4095d1da392a0d2f8ed0c6c7bc0f4cfac8c280b56d"
        while self._running:
            try:
                # Primary: Pyth Network (same as PM oracle)
                r = await self._http.get(
                    "https://hermes.pyth.network/v2/updates/price/latest",
                    params={"ids[]": pyth_sol_id, "parsed": "true"},
                )
                data = r.json()
                if "parsed" in data and data["parsed"]:
                    p = data["parsed"][0]["price"]
                    self._sol_price = round(
                        int(p["price"]) * 10 ** int(p["expo"]), 6
                    )
                else:
                    raise ValueError("No parsed data")
            except Exception:
                # Fallback: Binance Futures
                try:
                    r = await self._http.get(
                        "https://fapi.binance.com/fapi/v1/ticker/bookTicker",
                        params={"symbol": "SOLUSDT"},
                    )
                    data = r.json()
                    self._sol_price = (float(data["bidPrice"]) + float(data["askPrice"])) / 2
                except Exception:
                    pass
            await asyncio.sleep(3.0)

    async def _market_discovery_loop(self):
        """Discover active Polymarket markets."""
        while self._running:
            try:
                now_sec = int(time.time())
                for slug in self.slugs:
                    markets = await self.collector.get_active_markets(slug)
                    for m in markets:
                        if not m.is_tradeable:
                            continue

                        # Skip markets not currently active
                        # Slug epoch = START time. End = start + duration.
                        import re
                        ts_match = re.search(r'-(\d{10})$', m.slug)
                        if ts_match:
                            start_epoch = int(ts_match.group(1))
                            end_epoch = start_epoch + m.duration_minutes * 60
                            if start_epoch > now_sec:
                                continue  # market hasn't started yet
                            if end_epoch <= now_sec:
                                continue  # market already ended

                        is_new = m.slug not in self._active_markets
                        existing = self._active_markets.get(m.slug, {})

                        if is_new:
                            # PTB priority: Gamma API → Pyth (same as Chainlink) → fallback
                            ptb_source = "?"
                            ptb = m.price_to_beat
                            if ptb is not None:
                                ptb_source = "gamma"
                            else:
                                ptb = await self._fetch_ptb_for_slug(m.slug)
                                if ptb is not None:
                                    ptb_source = "pyth"
                                else:
                                    ptb = self._sol_price
                                    ptb_source = "fallback"
                            stored_ptb = ptb
                        else:
                            stored_ptb = existing.get("ptb", self._sol_price)

                        # Ensure end_date is always set (fallback: slug start + duration)
                        if not m.end_date and ts_match:
                            from datetime import datetime as _dt, timezone as _tz
                            s_epoch = int(ts_match.group(1))
                            e_epoch = s_epoch + m.duration_minutes * 60
                            m.end_date = _dt.fromtimestamp(e_epoch, tz=_tz.utc).isoformat()
                            m.time_remaining_ms = max(0, e_epoch * 1000 - now_sec * 1000)

                        if is_new:
                            # New market: use full Gamma object
                            self._active_markets[m.slug] = {
                                "market": m,
                                "ptb": stored_ptb,
                                "discovered_ts": time.time(),
                            }
                        else:
                            # Existing market: preserve WS-updated prices, only refresh metadata
                            existing_market = existing["market"]
                            # Update timing (always refresh from slug epoch for accuracy)
                            if ts_match:
                                from datetime import datetime as _dt, timezone as _tz
                                s_epoch = int(ts_match.group(1))
                                e_epoch = s_epoch + existing_market.duration_minutes * 60
                                existing_market.time_remaining_ms = max(0, e_epoch * 1000 - now_sec * 1000)
                                existing_market.time_elapsed_ms = max(0, now_sec * 1000 - s_epoch * 1000)
                                if not existing_market.end_date:
                                    existing_market.end_date = _dt.fromtimestamp(e_epoch, tz=_tz.utc).isoformat()
                            # Update accepting_orders status
                            existing_market.accepting_orders = m.accepting_orders
                            # If WS hasn't updated prices yet (best_ask still 0), use Gamma
                            if existing_market.best_ask == 0 and m.best_ask > 0:
                                existing_market.best_ask = m.best_ask
                                existing_market.best_bid = m.best_bid
                                existing_market.yes_price = m.best_ask  # ASK = entry price
                                existing_market.spread = m.spread
                            elif existing_market.best_ask == 0 and m.yes_price > 0:
                                # Fallback: use Gamma outcomePrices if no best_ask at all
                                existing_market.yes_price = m.yes_price
                                existing_market.no_price = m.no_price

                        if is_new:
                            log.info(
                                f"  🔍 Market: {m.slug} | "
                                f"PTB=${stored_ptb:.4f} ({ptb_source}) | "
                                f"UP=${m.yes_price:.3f} DN=${m.no_price:.3f}"
                            )

                # Clean up expired markets — use slug epoch (reliable, not stale API data)
                # Slug epoch = START time. End = start + duration.
                expired = []
                for s, d in self._active_markets.items():
                    ts_m = re.search(r'-(\d{10})$', s)
                    if ts_m:
                        s_epoch = int(ts_m.group(1))
                        e_epoch = s_epoch + d["market"].duration_minutes * 60
                        if e_epoch <= now_sec:
                            expired.append(s)
                    elif d["market"].time_remaining_ms <= 0:
                        expired.append(s)

                for s in expired:
                    info = self._active_markets[s]
                    mkt = info["market"]
                    ptb_val = info.get("ptb", self._sol_price)
                    outcome = "UP" if self._sol_price >= ptb_val else "DOWN"
                    self._record_outcome(
                        s, outcome,
                        sol_start=ptb_val, sol_end=self._sol_price,
                        ptb=ptb_val, dur_min=mkt.duration_minutes,
                    )
                    del self._active_markets[s]
                    self._market_entry_done.discard(s)
                    self._pending_signals.pop(s, None)
                    self._latest_predictions.pop(s, None)
                    self._prediction_history.pop(s, None)

            except Exception as e:
                log.error(f"Market discovery error: {e}")
            await asyncio.sleep(12.0)

    async def _orderbook_ws_loop(self):
        """WebSocket orderbook — real-time bid/ask updates (~1s) from Polymarket CLOB.

        Subscribes to YES + NO token_ids for all active markets.
        Events: 'book' (full snapshot), 'price_change', 'best_bid_ask'.
        Replaces REST polling entirely — no rate limits, instant updates.
        """
        import websockets

        ws_url = config.get("infrastructure", {}).get("polymarket", {}).get(
            "ws_market_url", "wss://ws-subscriptions-clob.polymarket.com/ws/market"
        )
        _ws_cfg = trading_config.get("orderbook_ws", {})
        reconnect_base = _ws_cfg.get("reconnect_base_ms", 1000) / 1000
        reconnect_max = _ws_cfg.get("reconnect_max_ms", 30000) / 1000
        reconnect_delay = reconnect_base

        # Map token_id → (slug, side) for fast lookup
        self._token_to_market: Dict[str, tuple] = {}  # token_id → (slug, "yes"|"no")
        self._ws_subscribed: set = set()  # token_ids already subscribed

        await asyncio.sleep(5.0)  # Wait for market discovery

        while self._running:
            ws = None
            try:
                log.info("  🔌 Orderbook WS: connecting...")
                ws = await websockets.connect(ws_url, ping_interval=20, ping_timeout=10)
                log.info("  ✅ Orderbook WS: connected")
                reconnect_delay = reconnect_base
                self._ws_subscribed.clear()

                # Subscribe to known markets immediately
                await self._ws_subscribe_all(ws)

                # Process messages + periodically subscribe new markets
                sub_check_ts = time.time()
                async for raw_msg in ws:
                    if not self._running:
                        break

                    try:
                        msg = json.loads(raw_msg)
                        self._process_ws_orderbook(msg)
                    except (json.JSONDecodeError, Exception) as e:
                        log.debug(f"  WS parse error: {e}")

                    # Every 10s, subscribe to any NEW markets discovered
                    if time.time() - sub_check_ts > 10:
                        await self._ws_subscribe_all(ws)
                        sub_check_ts = time.time()

            except Exception as e:
                log.warning(f"  ⚠️ Orderbook WS error: {e}")
            finally:
                if ws:
                    try:
                        await ws.close()
                    except Exception:
                        pass

            # Reconnect with exponential backoff
            if self._running:
                log.info(f"  🔄 Orderbook WS: reconnecting in {reconnect_delay:.1f}s...")
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, reconnect_max)

    async def _ws_subscribe_all(self, ws):
        """Subscribe WS to all active market token_ids (YES + NO)."""
        new_tokens = []
        for slug, info in list(self._active_markets.items()):
            market = info["market"]
            if market.yes_token_id and market.yes_token_id not in self._ws_subscribed:
                new_tokens.append(market.yes_token_id)
                self._token_to_market[market.yes_token_id] = (slug, "yes")
                self._ws_subscribed.add(market.yes_token_id)
            if market.no_token_id and market.no_token_id not in self._ws_subscribed:
                new_tokens.append(market.no_token_id)
                self._token_to_market[market.no_token_id] = (slug, "no")
                self._ws_subscribed.add(market.no_token_id)

        if new_tokens:
            sub_msg = json.dumps({"assets_ids": new_tokens, "type": "market"})
            await ws.send(sub_msg)
            log.info(f"  📡 WS subscribed: +{len(new_tokens)} tokens ({len(self._ws_subscribed)} total)")

    def _process_ws_orderbook(self, msg: dict):
        """Process a single WS message and update market prices.

        Handles event types:
        - 'book': full orderbook snapshot (bids + asks arrays)
        - 'price_change': individual price level change with best_bid/best_ask
        - 'best_bid_ask': direct best bid/ask/spread update
        """
        event_type = msg.get("event_type", "")
        asset_id = msg.get("asset_id", "")

        # price_change has asset_id inside price_changes array
        if event_type == "price_change":
            changes = msg.get("price_changes", [])
            for ch in changes:
                aid = ch.get("asset_id", "")
                if aid not in self._token_to_market:
                    continue
                slug, side = self._token_to_market[aid]
                info = self._active_markets.get(slug)
                if not info:
                    continue
                market = info["market"]
                best_bid = float(ch.get("best_bid", 0))
                best_ask = float(ch.get("best_ask", 0))
                if best_bid > 0 and best_ask > 0:
                    self._apply_price_update(market, side, best_bid, best_ask)
            self._clob_updates += 1
            return

        if asset_id not in self._token_to_market:
            return
        slug, side = self._token_to_market[asset_id]
        info = self._active_markets.get(slug)
        if not info:
            return
        market = info["market"]

        if event_type == "book":
            # Full orderbook snapshot — capture ALL liquidity levels
            bids = msg.get("bids", [])
            asks = msg.get("asks", [])
            parsed_bids = [(float(b.get("price", 0)), float(b.get("size", 0))) for b in bids if float(b.get("price", 0)) > 0]
            parsed_asks = [(float(a.get("price", 0)), float(a.get("size", 0))) for a in asks if float(a.get("price", 0)) > 0]
            best_bid = max(p for p, _ in parsed_bids) if parsed_bids else 0
            best_ask = min(p for p, _ in parsed_asks) if parsed_asks else 0
            if best_bid > 0 and best_ask > 0:
                # Full depth: total ask volume (what's available to buy)
                total_ask_depth = sum(s for _, s in parsed_asks)
                total_bid_depth = sum(s for _, s in parsed_bids)
                # Depth at price levels: how many shares available within $X of best
                sorted_asks = sorted(parsed_asks, key=lambda x: x[0])
                depth_2usd = sum(s for p, s in sorted_asks if p * s <= 2.0 or s <= 2.0 / max(p, 0.01))
                depth_5usd = sum(s for p, s in sorted_asks if p * s <= 5.0 or s <= 5.0 / max(p, 0.01))

                self._apply_price_update(market, side, best_bid, best_ask, depth=total_ask_depth)
                # Store extended orderbook data on market object
                if side == "yes":
                    market.yes_bid_depth = total_bid_depth
                    market.yes_ask_depth = total_ask_depth
                    market.yes_levels = len(parsed_asks)
                    market.yes_bid_levels = len(parsed_bids)
                else:
                    market.no_bid_depth = total_bid_depth
                    market.no_ask_depth = total_ask_depth
                    market.no_levels = len(parsed_asks)
                    market.no_bid_levels = len(parsed_bids)
            self._clob_updates += 1

        elif event_type == "best_bid_ask":
            best_bid = float(msg.get("best_bid", 0))
            best_ask = float(msg.get("best_ask", 0))
            if best_bid > 0 and best_ask > 0:
                self._apply_price_update(market, side, best_bid, best_ask)
            self._clob_updates += 1

    def _apply_price_update(self, market: SharesMarket, side: str,
                            best_bid: float, best_ask: float, depth: float = None):
        """Apply a price update to the correct side of the market.

        YES side: market.yes_price = best_ask (what you pay to buy YES = bet UP)
        NO side:  market.no_price  = best_ask (what you pay to buy NO = bet DOWN)

        IMPORTANT: On SOL markets, NO orderbook is typically EMPTY — all liquidity
        is on YES side. So when YES updates, we also derive NO price:
          no_price = 1 - yes_bid  (buying NO ≈ selling YES at bid)
        This gives realistic DOWN entry cost even without NO-side WS events.
        """
        if side == "yes":
            market.best_bid = best_bid
            market.best_ask = best_ask
            market.spread = round(best_ask - best_bid, 4)
            market.yes_price = best_ask  # ASK = entry price for UP bet
            if depth is not None:
                market.yes_depth = depth
            # Derive NO price from YES bid (since NO orderbook is empty)
            # Buying NO ≈ selling YES → cost = 1 - yes_bid
            if market.no_best_ask == 0 and best_bid > 0:
                market.no_price = round(1.0 - best_bid, 4)
        elif side == "no":
            market.no_best_bid = best_bid
            market.no_best_ask = best_ask
            market.no_spread = round(best_ask - best_bid, 4)
            market.no_price = best_ask  # ASK = entry price for DOWN bet
            if depth is not None:
                market.no_depth = depth

    # ═══════════════════════════════════════════════════════════
    #  ML PREDICTION + ENTRY
    # ═══════════════════════════════════════════════════════════

    async def _entry_loop(self):
        """Evaluate markets using predictions from _recording_loop (single source of truth).

        NEVER re-computes ML. Reads cached predictions + requires direction
        stability (N consecutive ticks agree) before entering.
        If share > max_share_price, queues and re-checks each loop.
        """
        await asyncio.sleep(5.0)  # Wait for warmup

        while self._running:
            if self._model and len(self._klines) >= 61:
                # --- 1) Re-check queued signals (price was too high, waiting) ---
                for slug in list(self._pending_signals.keys()):
                    if slug in self._market_entry_done:
                        del self._pending_signals[slug]
                        continue
                    if len(self.positions) >= self.max_positions:
                        continue
                    if slug not in self._active_markets:
                        del self._pending_signals[slug]
                        continue

                    info = self._active_markets[slug]
                    market = info["market"]
                    elapsed_pct = self._elapsed_pct(slug, market.duration_minutes)

                    # Expired — past max_entry_pct
                    if elapsed_pct > self.max_entry_pct:
                        log.info(f"  ⌛ {slug}: entry window closed (elapsed={elapsed_pct:.0%} > {self.max_entry_pct:.0%})")
                        self._market_entry_done.add(slug)
                        del self._pending_signals[slug]
                        continue

                    # Re-evaluate using latest recorded prediction (single source of truth)
                    ptb = info.get("ptb", self._sol_price)
                    try:
                        await self._evaluate_entry(slug, market, ptb, is_recheck=True)
                    except Exception as e:
                        log.debug(f"  Recheck error {slug}: {e}")

                # --- 2) Evaluate NEW markets (first time ML prediction) ---
                for slug, info in list(self._active_markets.items()):
                    market = info["market"]

                    # Only trade markets in trade_slugs (not record-only markets)
                    base_slug = "-".join(slug.rsplit("-", 1)[0:1])  # sol-updown-5m-123 → sol-updown-5m
                    if not any(slug.startswith(ts) for ts in self.trade_slugs):
                        continue

                    if slug in self._market_entry_done or slug in self._pending_signals:
                        continue
                    if len(self.positions) >= self.max_positions:
                        continue

                    elapsed_pct = self._elapsed_pct(slug, market.duration_minutes)

                    if elapsed_pct < self.min_entry_pct:
                        continue  # Too early (before entry window)
                    if elapsed_pct > self.max_entry_pct:
                        continue  # Past entry window

                    ptb = info.get("ptb", self._sol_price)
                    log.info(
                        f"  ⏱ {slug}: elapsed={elapsed_pct:.0%} "
                        f"| UP=${market.yes_price:.3f} DN=${market.no_price:.3f} "
                        f"| bid=${market.best_bid:.3f} ask=${market.best_ask:.3f} spread={market.spread:.3f} "
                        f"| PTB=${ptb:.2f} SOL=${self._sol_price:.2f} | evaluating..."
                    )

                    try:
                        await self._evaluate_entry(slug, market, ptb)
                    except Exception as e:
                        log.error(f"  ❌ evaluate_entry error for {slug}: {e}")
                        import traceback
                        traceback.print_exc()
                        self._market_entry_done.add(slug)

            await asyncio.sleep(5.0)

    def _compute_predictions(self, market: SharesMarket, ptb: float):
        """Compute features + predict with ALL models. Pure computation, no side effects.

        Returns dict with keys: all_preds, prob_up, dir_prob, direction, share_price,
                                token_id, sol_price, best_model, ml_ms
        Or None if insufficient data.
        """
        from core.features.price_volume import PriceVolumeFeatures
        from core.features.technical import TechnicalFeatures
        from core.features.microstructure import MicrostructureFeatures
        from core.features.liquidation_funding import LiquidationFundingFeatures
        from core.features.regime import RegimeFeatures

        if not self._feature_names or len(self._klines) < 61:
            return None

        sol_price = self._sol_price

        # Build OHLCV from kline buffer (last 61 bars)
        bars = self._kline_buffer()[-61:]
        ohlcv = {
            "open": np.array([b["open"] for b in bars], dtype=np.float64),
            "high": np.array([b["high"] for b in bars], dtype=np.float64),
            "low": np.array([b["low"] for b in bars], dtype=np.float64),
            "close": np.array([b["close"] for b in bars], dtype=np.float64),
            "volume": np.array([b["volume"] for b in bars], dtype=np.float64),
            "taker_buy_volume": np.array([b["taker_buy_volume"] for b in bars], dtype=np.float64),
        }

        # Compute all 6 feature blocks (same as training)
        features = {}

        pv = PriceVolumeFeatures()
        features.update(pv.compute(ohlcv_1m=ohlcv, current_price=sol_price))

        tech = TechnicalFeatures()
        features.update(tech.compute(ohlcv=ohlcv, current_price=sol_price))

        micro = MicrostructureFeatures()
        features.update(micro.compute(current_price=sol_price))

        liq = LiquidationFundingFeatures()
        features.update(liq.compute(funding_rate=0, current_price=sol_price))

        # On-chain (zeros — not available live)
        for k in [
            "onchain_large_transfers_60s", "onchain_whale_activity",
            "onchain_dex_volume_spike", "onchain_jupiter_accel",
            "onchain_mev_bundles", "onchain_priority_fee_pressure",
            "onchain_token_creation_rate", "onchain_large_transfers_300s",
            "onchain_dex_volume_spike_300s", "onchain_jupiter_accel_300s",
        ]:
            features[k] = 0.0

        regime = RegimeFeatures()
        close_arr = ohlcv["close"]
        rets = np.diff(np.log(close_arr[close_arr > 0])) if len(close_arr) > 10 else None
        features.update(regime.compute(returns=rets, close_prices=close_arr))

        # Shares features
        from scipy.stats import norm
        dist = (sol_price - ptb) / ptb if ptb > 0 else 0
        t_rem_ms = market.time_remaining_ms
        t_elap_ms = market.time_elapsed_ms
        vol = self._sol_volatility
        t_min = max(t_rem_ms / 60_000, 0.01)
        vol_adj = max(vol, 0.001) * np.sqrt(t_min)
        d = dist / vol_adj
        yes_price = float(np.clip(norm.cdf(d), 0.02, 0.98))
        no_price = 1.0 - yes_price

        shares_feats = compute_shares_features(
            sol_price=sol_price,
            price_to_beat=ptb,
            yes_price=yes_price,
            no_price=no_price,
            time_remaining_ms=t_rem_ms,
            time_elapsed_ms=t_elap_ms,
            duration_minutes=market.duration_minutes,
            sol_volatility=vol,
        )
        features.update(shares_feats)

        # Pre-market lookback features (from kline buffer before market start)
        market_start_ts = market.time_elapsed_ms  # approx
        kline_ts_sorted = sorted(self._klines.keys())
        for lookback in [2, 5, 10, 15, 30]:
            key_ret = f"pre_mkt_ret_{lookback}m"
            key_vol = f"pre_mkt_vol_{lookback}m"
            # Use last N bars from buffer as proxy
            if len(kline_ts_sorted) >= lookback + 1:
                pre_bars = [self._klines[t] for t in kline_ts_sorted[-(lookback + 1):-1]]
                if pre_bars:
                    pre_closes = [float(b.get("close", sol_price)) for b in pre_bars]
                    if len(pre_closes) >= 2:
                        features[key_ret] = (pre_closes[-1] - pre_closes[0]) / (pre_closes[0] + 1e-10)
                        import math
                        log_rets = [math.log(pre_closes[i+1] / (pre_closes[i] + 1e-10))
                                    for i in range(len(pre_closes) - 1)]
                        features[key_vol] = float(np.std(log_rets)) if len(log_rets) > 1 else 0.0
                    else:
                        features[key_ret] = 0.0; features[key_vol] = 0.0
                else:
                    features[key_ret] = 0.0; features[key_vol] = 0.0
            else:
                features[key_ret] = 0.0; features[key_vol] = 0.0

        # Extra features (zeros for missing)
        features.setdefault("oi_change", 0.0)
        features.setdefault("long_short_ratio", 1.0)

        fv = np.array([features.get(k, 0.0) for k in self._feature_names], dtype=np.float64)
        fv = np.nan_to_num(fv, nan=0.0, posinf=0.0, neginf=0.0).reshape(1, -1)

        # Scale
        if self._scaler:
            fv = self._scaler.transform(fv)

        # Predict with ALL models (timed)
        import time as _t
        _t0_ml = _t.perf_counter()
        all_preds = {}
        for model_name, model_obj in self._models.items():
            try:
                if hasattr(model_obj, 'predict_proba'):
                    p = float(model_obj.predict_proba(fv)[:, 1][0])
                else:
                    p = float(model_obj.predict(fv)[0])
                all_preds[model_name] = p
            except Exception as e:
                log.debug(f"  {model_name} predict error: {e}")
        _ml_ms = (_t.perf_counter() - _t0_ml) * 1000

        if not all_preds:
            return None

        # Primary: use configured model (xgboost) or fallback
        prob_up = all_preds.get(self.primary_model_name, all_preds.get("catboost", all_preds.get(next(iter(all_preds), ""), 0.5)))
        model_says_up = prob_up > 0.5
        dir_prob = max(prob_up, 1 - prob_up)

        # Ensemble: average all models
        if len(all_preds) >= 2:
            ens_prob = np.mean(list(all_preds.values()))
            all_preds["ensemble"] = float(ens_prob)

        # Direction + prices
        real_yes = market.yes_price
        real_no = market.no_price
        if model_says_up:
            direction = "UP"
            share_price = real_yes
            token_id = market.yes_token_id
        else:
            direction = "DOWN"
            share_price = real_no
            token_id = market.no_token_id

        best_model = max(all_preds, key=lambda k: max(all_preds[k], 1 - all_preds[k])) if all_preds else "catboost"

        return {
            "all_preds": all_preds, "prob_up": prob_up, "dir_prob": dir_prob,
            "direction": direction, "share_price": share_price, "token_id": token_id,
            "sol_price": sol_price, "best_model": best_model, "ml_ms": _ml_ms,
        }

    async def _evaluate_entry(self, slug: str, market: SharesMarket, ptb: float, is_recheck: bool = False):
        """ML predict → decide entry.

        Uses prediction from _recording_loop (single source of truth).
        Never re-computes ML — reads the latest cached prediction instead.
        Requires direction stability: last N ticks must agree.
        """
        # ── Read prediction from recording loop cache ──
        cached = self._latest_predictions.get(slug)
        if cached is None or (time.time() - cached["ts"]) > 15:
            return  # No fresh prediction yet — wait for recording loop

        result = cached["result"]
        all_preds = result["all_preds"]
        prob_up = result["prob_up"]
        dir_prob = result["dir_prob"]
        direction = result["direction"]
        share_price = result["share_price"]
        token_id = result["token_id"]
        sol_price = result["sol_price"]
        best_model_name = result["best_model"]
        _ml_ms = result["ml_ms"]

        # ── Direction stability check (smoothing) ──
        history = self._prediction_history.get(slug, [])
        if len(history) >= self._SMOOTHING_WINDOW:
            dirs = [h["direction"] for h in history[-self._SMOOTHING_WINDOW:]]
            agreement = dirs.count(direction) / len(dirs)
            if agreement < 1.0:
                # Not all recent ticks agree on direction — too unstable
                if not is_recheck:
                    log.info(
                        f"  ⚠️ {slug}: direction unstable ({dirs}) — "
                        f"need {self._SMOOTHING_WINDOW}/{self._SMOOTHING_WINDOW} agreement, skipping"
                    )
                return
            # Use average confidence from stable window for extra robustness
            avg_dir_prob = sum(h["dir_prob"] for h in history[-self._SMOOTHING_WINDOW:]) / self._SMOOTHING_WINDOW
        else:
            # Not enough history yet — skip (wait for N ticks)
            if not is_recheck:
                log.info(f"  ⏳ {slug}: warming up ({len(history)}/{self._SMOOTHING_WINDOW} ticks) — waiting")
            return

        log.info(
            f"  🔮 {slug}: {direction} prob={dir_prob:.0%} avg={avg_dir_prob:.0%} [{_ml_ms:.0f}ms] "
            f"| models: {', '.join(f'{k}={v:.2f}' for k, v in all_preds.items())} "
            f"| share=${share_price:.3f} "
            f"| SOL=${sol_price:.2f} PTB=${ptb:.2f} Δ={sol_price-ptb:+.2f}"
            f" | stable={self._SMOOTHING_WINDOW}/{self._SMOOTHING_WINDOW}"
        )

        # Filter: confidence (use average for stability)
        if avg_dir_prob < self.min_confidence:
            log.info(f"  ❌ {slug}: avg_conf={avg_dir_prob:.0%} < {self.min_confidence:.0%} — SKIP")
            if is_recheck:
                self._pending_signals.pop(slug, None)
            return

        # Filter: share price too low (noise/dead market)
        if share_price < self.min_share_price:
            self._market_entry_done.add(slug)
            self._pending_signals.pop(slug, None)
            return

        # Filter: share price too high
        if share_price > self.max_share_price:
            # Queue — recording loop will keep updating predictions
            if slug not in self._pending_signals:
                self._pending_signals[slug] = {"queued_at": time.time()}
            if not is_recheck:
                log.info(
                    f"  ⏳ {slug}: {direction} @ ${share_price:.3f} > ${self.max_share_price:.2f} "
                    f"— queued (avg_conf={avg_dir_prob:.0%})"
                )
            return

        # Filter: spread too wide (illiquid)
        active_spread = market.spread if direction == "UP" else market.no_spread
        if active_spread > self.max_spread:
            log.info(f"  ❌ {slug}: spread ${active_spread:.3f} > ${self.max_spread:.2f} — SKIP")
            return

        # Filter: depth too thin (can't fill order)
        active_depth = market.yes_depth if direction == "UP" else market.no_depth
        shares_needed = self.order_size / share_price if share_price > 0 else 0
        if active_depth > 0 and active_depth < shares_needed:
            log.info(f"  ❌ {slug}: depth {active_depth:.0f} < needed {shares_needed:.0f} — SKIP")
            return

        # All filters passed + direction stable → enter now
        log.info(
            f"  💰 {slug}: {direction} @ ${share_price:.3f} | avg_conf={avg_dir_prob:.0%} "
            f"(raw={dir_prob:.0%}) | spread=${active_spread:.3f} depth={active_depth:.0f}"
            f"{' (recheck)' if is_recheck else ''}"
        )
        import sys; print(f">>> ENTRY SIGNAL: {slug} {direction} ${share_price:.3f} conf={avg_dir_prob:.0%} stable={self._SMOOTHING_WINDOW} dry={self.dry_run}", file=sys.stderr, flush=True)
        self._pending_signals.pop(slug, None)

        # Pre-sign order before execute (saves ~50ms at submission time)
        if not self.dry_run and not self._clob.is_read_only:
            _presign_price = max(share_price, round(1.0 - market.best_bid, 4) if direction == "DOWN" and market.best_bid > 0 else share_price)
            _presign_price = min(_presign_price, self.max_share_price + 0.01)
            signed = self._clob.presign_buy(token_id, _presign_price, self.order_size, neg_risk=market.neg_risk)
            if signed:
                self._presigned_orders[slug] = {
                    "signed_order": signed, "token_id": token_id,
                    "price": _presign_price, "shares": round(self.order_size / _presign_price, 2),
                    "amount_usd": self.order_size, "ts": time.time(),
                }

        await self._execute_entry(slug, market, direction, share_price, prob_up, avg_dir_prob, token_id,
                                   ml_ms=_ml_ms, all_preds=all_preds, best_model=best_model_name)

    async def _execute_entry(self, slug: str, market: SharesMarket, direction: str,
                             share_price: float, prob_up: float, confidence: float, token_id: str,
                             ml_ms: float = 0, all_preds: dict = None, best_model: str = "lgbm"):
        """Execute entry: GTC limit buy (instant, stays on book if no match).

        After fill confirmed → immediately place GTC sell limit @ $0.99.
        This sell acts as early exit: if filled before expiry = WIN.
        """
        bet = min(self.order_size, self._capital)
        if bet < 0.50:
            log.warning("  Not enough capital")
            return

        shares = bet / share_price
        ptb = self._active_markets.get(slug, {}).get("ptb", self._sol_price)

        # === LIVE ORDER via CLOB (GTC limit buy — posts immediately, fills when matched) ===
        baseline_shares = 0.0
        usdc_before = None
        if not self.dry_run and not self._clob.is_read_only:
            baseline_shares = await self._clob.get_share_balance(token_id)
            usdc_before = await self._clob.get_balance()

        # Compute crossing price: must reach effective ask for neg_risk mint-match.
        # For UP: effective ask = YES best_ask (direct fill)
        # For DOWN: effective ask = 1 - YES best_bid (mint-match: NO_price + YES_bid >= $1)
        if direction == "UP":
            crossing_price = market.best_ask if market.best_ask > 0 else share_price
        else:
            # NO book is typically empty on SOL markets; crossing requires 1 - YES_bid
            crossing_price = round(1.0 - market.best_bid, 4) if market.best_bid > 0 else share_price

        # Use the higher of evaluated share_price and crossing_price to ensure fill
        limit_price = max(share_price, crossing_price)
        # Cap at max_share_price + small buffer to avoid runaway
        limit_price = min(limit_price, self.max_share_price + 0.01)
        shares = bet / limit_price  # recalculate shares at actual limit price

        log.info(f"  📋 ORDER: {direction} limit=${limit_price:.3f} (eval=${share_price:.3f} cross=${crossing_price:.3f}) {shares:.1f} shares")
        import sys; print(f">>> ORDER: {slug} {direction} limit=${limit_price:.3f} eval=${share_price:.3f} cross=${crossing_price:.3f}", file=sys.stderr, flush=True)

        # Try pre-signed order first (instant post, ~20ms vs ~70ms)
        presigned = self._presigned_orders.pop(slug, None)
        if presigned and abs(presigned["price"] - limit_price) < 0.005 and presigned["token_id"] == token_id:
            log.info(f"  ⚡ Using pre-signed order for {slug}")
            order_result = await self._clob.post_presigned(
                presigned["signed_order"],
                token_id=token_id, price=presigned["price"],
                shares=presigned["shares"], amount_usd=presigned["amount_usd"],
            )
        else:
            order_result = await self._clob.buy_limit(
                token_id=token_id,
                price=limit_price,
                amount_usd=bet,
                neg_risk=market.neg_risk,
            )
        if not order_result.success:
            log.warning(f"  ⚠️ CLOB BUY failed for {slug}: {order_result.error}")
            import sys; print(f">>> CLOB BUY FAILED: {slug} error={order_result.error}", file=sys.stderr, flush=True)
            return
        import sys; print(f">>> CLOB BUY OK: {slug} shares={order_result.shares} price={order_result.price} id={order_result.order_id}", file=sys.stderr, flush=True)

        # Wait for fill (GTC might need a few seconds on thin books)
        if not self.dry_run and not self._clob.is_read_only:
            filled = await self._clob.wait_for_fill(
                token_id, order_result.shares, baseline=baseline_shares, timeout_s=10.0
            )
            if filled < 0.5:
                # Order didn't fill in 10s — cancel and skip
                log.warning(f"  ⚠️ BUY GTC not filled in 10s for {slug} — cancelling")
                if order_result.order_id:
                    await self._clob.cancel_order(order_result.order_id)
                return
            actual_shares = filled
            # Track ACTUAL cost via USDC balance change (handles price improvement)
            usdc_after = await self._clob.get_balance()
            if usdc_before is not None and usdc_after is not None:
                actual_usd = round(usdc_before - usdc_after, 4)
                actual_price = round(actual_usd / filled, 4) if filled > 0 else limit_price
                log.info(f"  📊 Fill: {filled:.1f} shares, USDC ${usdc_before:.2f}→${usdc_after:.2f} = ${actual_usd:.2f} spent (${actual_price:.4f}/share)")
            else:
                actual_price = round(bet / filled, 4) if filled > 0 else limit_price
                actual_usd = bet
        else:
            actual_shares = order_result.shares if order_result.shares > 0 else shares
            actual_price = order_result.price if order_result.price > 0 else limit_price
            actual_usd = order_result.amount_usd if order_result.amount_usd > 0 else bet

        self._capital -= actual_usd

        pos = MLPosition(
            market_slug=slug,
            token_id=token_id,
            direction=direction,
            entry_price=actual_price,
            shares=actual_shares,
            size_usd=actual_usd,
            entry_ts=time.time(),
            confidence=confidence,
            model_prob=prob_up,
            price_to_beat=ptb,
            duration_minutes=market.duration_minutes,
            end_date=market.end_date or self._end_date_from_slug(slug),
            sol_price_at_entry=self._sol_price,
            primary_model=best_model,
            inference_ms=ml_ms,
            all_model_probs=all_preds or {},
            current_price=actual_price,
        )

        # === Immediately place GTC SELL @ $0.99 (lock in profit if filled early) ===
        sell_order_id = ""
        if not self.dry_run and not self._clob.is_read_only:
            sell_r = await self._clob.sell_limit(
                token_id=token_id,
                price=0.99,
                shares=round(actual_shares, 2),
                neg_risk=market.neg_risk,
            )
            if sell_r.success:
                sell_order_id = sell_r.order_id
                log.info(f"  💰 SELL LIMIT $0.99 placed: {actual_shares:.1f} shares [{sell_order_id}]")
            else:
                log.warning(f"  ⚠️ SELL LIMIT $0.99 failed: {sell_r.error}")
        pos.sell_order_id = sell_order_id

        self.positions.append(pos)
        self._market_entry_done.add(slug)
        self._trade_snapshots[slug] = []  # start collecting snapshots

        mode_str = "[DRY]" if self.dry_run else "[LIVE]"
        log.info(
            f"  {mode_str} 🎯 BUY {direction} {actual_shares:.1f} shares @ ${actual_price:.3f} "
            f"| {slug} | {confidence:.0%} confident "
            f"| SOL=${self._sol_price:.2f} PTB=${ptb:.2f} Δ={self._sol_price-ptb:+.2f}"
        )
        # Log orderbook state at entry time
        log.info(
            f"  📊 OB: YES bid=${market.best_bid:.3f} ask=${market.best_ask:.3f} "
            f"spread={market.spread:.3f} depth={market.yes_depth:.0f} | "
            f"NO bid=${market.no_best_bid:.3f} ask=${market.no_best_ask:.3f} "
            f"spread={market.no_spread:.3f} depth={market.no_depth:.0f} | "
            f"yes=${market.yes_price:.3f} no=${market.no_price:.3f}"
        )

        # Telegram notification
        if self._tg:
            await self._tg.notify_entry(pos)

    # ═══════════════════════════════════════════════════════════
    #  PER-TRADE SNAPSHOT RECORDING (every 3s while position open)
    # ═══════════════════════════════════════════════════════════

    async def _trade_snapshot_loop(self):
        """Record CLOB orderbook + SOL price every 3s for each open position."""
        while self._running:
            for pos in self.positions:
                slug = pos.market_slug
                if slug not in self._trade_snapshots:
                    continue
                mdata = self._active_markets.get(slug, {})
                market = mdata.get("market") if mdata else None
                snap = {
                    "ts": round(time.time(), 3),
                    "sol": round(self._sol_price, 4),
                }
                if market:
                    snap.update({
                        "yes_bid": round(market.best_bid, 4),
                        "yes_ask": round(market.best_ask, 4),
                        "yes_price": round(market.yes_price, 4),
                        "yes_spread": round(market.spread, 4),
                        "yes_depth": round(market.yes_depth, 1),
                        "no_bid": round(market.no_best_bid, 4),
                        "no_ask": round(market.no_best_ask, 4),
                        "no_price": round(market.no_price, 4),
                        "no_spread": round(market.no_spread, 4),
                        "no_depth": round(market.no_depth, 1),
                    })
                self._trade_snapshots[slug].append(snap)
            await asyncio.sleep(3)

    # ═══════════════════════════════════════════════════════════
    #  RESOLUTION TRACKING
    # ═══════════════════════════════════════════════════════════

    async def _resolution_loop(self):
        """Check if any positions have resolved.

        Resolution triggers:
        1. Sell limit $0.99 filled (share balance → 0) → early WIN
        2. Market expired (time_remaining ≤ 0) → SOL vs PTB determines outcome
           - On loss: cancel sell order, shares worth $0
           - On win: sell already placed at entry; exit_price = $0.99
        """
        while self._running:
            positions_to_close = []

            for pos in self.positions:
                # --- CHECK 1: Early sell fill ($0.99 limit matched) ---
                sold_early = False
                if not self.dry_run and not self._clob.is_read_only and pos.sell_order_id:
                    try:
                        bal = await self._clob.get_share_balance(pos.token_id)
                        if bal < 0.5:  # shares gone → sell filled
                            sold_early = True
                    except Exception:
                        pass

                if sold_early:
                    exit_price = 0.99
                    outcome = pos.direction  # sold = win (we got $0.99/share)
                    won = True

                    self._record_outcome(
                        pos.market_slug, outcome,
                        sol_start=pos.sol_price_at_entry, sol_end=self._sol_price,
                        ptb=pos.price_to_beat, dur_min=getattr(pos, 'duration_minutes', 15)
                    )

                    pnl_usd = pos.shares * (exit_price - pos.entry_price)
                    pnl_pct = (exit_price - pos.entry_price) / max(pos.entry_price, 0.01) * 100
                    self._capital += pos.shares * exit_price

                    result = TradeResult(
                        slug=pos.market_slug, direction=pos.direction,
                        entry_price=pos.entry_price, exit_price=exit_price,
                        shares=pos.shares, pnl_usd=pnl_usd, pnl_pct=pnl_pct,
                        confidence=pos.confidence, model_prob=pos.model_prob,
                        hold_time_s=time.time() - pos.entry_ts,
                        reason="sell_099_win",
                        sol_at_entry=pos.sol_price_at_entry, sol_at_exit=self._sol_price,
                        ptb=pos.price_to_beat, ts=time.time(),
                    )
                    self.completed.append(result)
                    positions_to_close.append(pos)
                    self._log_trade_json(pos, result, outcome)

                    mode_str = "[DRY]" if self.dry_run else "[LIVE]"
                    log.info(
                        f"  {mode_str} ✅ {pos.direction} SOLD @ $0.99 (early) | "
                        f"PnL={pnl_pct:+.0f}% (${pnl_usd:+.2f}) | "
                        f"{pos.confidence:.0%} confident | entry=${pos.entry_price:.3f} "
                        f"| held {time.time()-pos.entry_ts:.0f}s | cap=${self._capital:.2f}"
                    )
                    if self._tg:
                        await self._tg.notify_resolution(pos, result, outcome)
                        await self._tg.send_trade_card(pos, result, outcome)
                    continue

                # --- CHECK 2: Market expiry (normal resolution) ---
                try:
                    markets = await self.collector.get_active_markets(
                        pos.market_slug.rsplit("-", 1)[0]  # base slug
                    )
                except Exception:
                    markets = []

                resolved = True
                for m in markets:
                    if m.slug == pos.market_slug:
                        resolved = (m.time_remaining_ms <= 0)
                        if pos.direction == "UP":
                            pos.current_price = m.yes_price
                        else:
                            pos.current_price = m.no_price
                        break

                if resolved:
                    outcome = "UP" if self._sol_price >= pos.price_to_beat else "DOWN"
                    won = (pos.direction == outcome)

                    self._record_outcome(
                        pos.market_slug, outcome,
                        sol_start=pos.sol_price_at_entry, sol_end=self._sol_price,
                        ptb=pos.price_to_beat, dur_min=getattr(pos, 'duration_minutes', 15)
                    )

                    if won:
                        # Sell limit $0.99 was placed at entry — use that as exit
                        exit_price = 0.99
                    else:
                        exit_price = 0.0
                        # Cancel the sell order (shares are worthless)
                        if pos.sell_order_id and not self.dry_run and not self._clob.is_read_only:
                            await self._clob.cancel_order(pos.sell_order_id)

                    pnl_usd = pos.shares * (exit_price - pos.entry_price)
                    pnl_pct = (exit_price - pos.entry_price) / max(pos.entry_price, 0.01) * 100
                    self._capital += pos.shares * exit_price

                    result = TradeResult(
                        slug=pos.market_slug, direction=pos.direction,
                        entry_price=pos.entry_price, exit_price=exit_price,
                        shares=pos.shares, pnl_usd=pnl_usd, pnl_pct=pnl_pct,
                        confidence=pos.confidence, model_prob=pos.model_prob,
                        hold_time_s=time.time() - pos.entry_ts,
                        reason="expiry_win" if won else "expiry_loss",
                        sol_at_entry=pos.sol_price_at_entry, sol_at_exit=self._sol_price,
                        ptb=pos.price_to_beat, ts=time.time(),
                    )
                    self.completed.append(result)
                    positions_to_close.append(pos)
                    self._log_trade_json(pos, result, outcome)

                    emoji = "✅" if won else "❌"
                    mode_str = "[DRY]" if self.dry_run else "[LIVE]"
                    log.info(
                        f"  {mode_str} {emoji} {pos.direction} resolved → {outcome} | "
                        f"PnL={pnl_pct:+.0f}% (${pnl_usd:+.2f}) | "
                        f"{pos.confidence:.0%} confident | entry=${pos.entry_price:.3f} "
                        f"| SOL=${self._sol_price:.2f} PTB=${pos.price_to_beat:.2f} "
                        f"Δ={self._sol_price-pos.price_to_beat:+.2f} | cap=${self._capital:.2f}"
                    )
                    if self._tg:
                        await self._tg.notify_resolution(pos, result, outcome)
                        await self._tg.send_trade_card(pos, result, outcome)

            for pos in positions_to_close:
                self.positions.remove(pos)

            # Save completed trades periodically
            if self.completed and len(self.completed) % 5 == 0:
                self._save_trades()

            await asyncio.sleep(10.0)

    # ═══════════════════════════════════════════════════════════
    #  STATUS + SUMMARY
    # ═══════════════════════════════════════════════════════════

    async def _status_loop(self):
        """Print status every 30 seconds."""
        await asyncio.sleep(15.0)
        while self._running:
            elapsed = time.time() - self._start_ts
            wins = sum(1 for t in self.completed if "win" in t.reason)
            total = len(self.completed)
            wr = wins / max(total, 1) * 100
            pnl = sum(t.pnl_usd for t in self.completed)

            n_pending = len(self._pending_signals)
            n_open = len(self.positions)

            log.info(
                f"  📊 {int(elapsed//60)}m | SOL=${self._sol_price:.2f} | "
                f"{len(self._active_markets)} markets | "
                f"{n_open} open | {n_pending} pending | "
                f"{wins}W/{total-wins}L ({wr:.0f}%) | "
                f"PnL=${pnl:+.2f} | cap=${self._capital:.2f} | "
                f"CLOB:{self._clob_updates}"
            )

            # Show queued signals (waiting for price ≤ max_share_price)
            for slug in list(self._pending_signals.keys()):
                info = self._active_markets.get(slug, {})
                market = info.get("market")
                if market:
                    total_ms = market.time_remaining_ms + market.time_elapsed_ms
                    pct = market.time_elapsed_ms / max(total_ms, 1)
                    log.info(
                        f"    ⏳ queued {slug} | "
                        f"UP=${market.yes_price:.3f} DN=${market.no_price:.3f} "
                        f"(need ≤${self.max_share_price:.2f}) | "
                        f"bid=${market.best_bid:.3f} ask=${market.best_ask:.3f} | "
                        f"{pct:.0%}/{self.max_entry_pct:.0%} elapsed"
                    )

            # Show open positions
            for pos in self.positions:
                t_left = ""
                if pos.end_date:
                    try:
                        end_ts = datetime.fromisoformat(
                            pos.end_date.replace("Z", "+00:00")
                        ).timestamp()
                        secs_left = max(0, end_ts - time.time())
                        t_left = f" | {secs_left/60:.0f}m left"
                    except (ValueError, TypeError):
                        pass
                delta = self._sol_price - pos.price_to_beat
                delta_dir = "✓" if (pos.direction == "UP" and delta > 0) or (pos.direction == "DOWN" and delta < 0) else "✗"
                log.info(
                    f"    📌 {pos.direction} {pos.market_slug} @ ${pos.entry_price:.3f} "
                    f"| {pos.confidence:.0%} | Δ={delta:+.2f} {delta_dir}{t_left}"
                )

            await asyncio.sleep(30.0)

    def _print_summary(self):
        """Final summary."""
        if not self.completed:
            log.info("No completed trades.")
            return

        wins = [t for t in self.completed if "win" in t.reason]
        losses = [t for t in self.completed if "loss" in t.reason]
        total_pnl = sum(t.pnl_usd for t in self.completed)

        elapsed = time.time() - self._start_ts
        avg_conf = np.mean([t.confidence for t in self.completed])

        log.info("")
        log.info("╔══════════════════════════════════════════════════════════╗")
        log.info("║            SESSION SUMMARY                             ║")
        log.info("╠══════════════════════════════════════════════════════════╣")
        log.info(f"║  Duration:     {int(elapsed//60)}m")
        log.info(f"║  Trades:       {len(self.completed)} ({len(wins)}W / {len(losses)}L)")
        log.info(f"║  Win Rate:     {len(wins)/max(len(self.completed),1)*100:.1f}%")
        log.info(f"║  PnL:          ${total_pnl:+.2f}")
        log.info(f"║  Capital:      $100.00 → ${self._capital:.2f} ({(self._capital/100-1)*100:+.1f}%)")
        log.info(f"║  Avg Conf:     {avg_conf:.0%}")
        if wins:
            log.info(f"║  Avg Win:      ${np.mean([t.pnl_usd for t in wins]):+.2f}")
        if losses:
            log.info(f"║  Avg Loss:     ${np.mean([t.pnl_usd for t in losses]):+.2f}")
        log.info("╚══════════════════════════════════════════════════════════╝")

        self._save_trades()

    # ═══════════════════════════════════════════════════════════
    #  LIVE RECORDING — JSONL (fast append, 0-100% of every market)
    # ═══════════════════════════════════════════════════════════

    def _get_tick_path(self):
        """Get JSONL file path for today's ticks."""
        Path("results").mkdir(exist_ok=True)
        return Path(f"results/live_ticks_{datetime.now().strftime('%Y-%m-%d')}.jsonl")

    def _get_outcomes_path(self):
        """Get JSONL file path for today's outcomes."""
        Path("results").mkdir(exist_ok=True)
        return Path(f"results/live_outcomes_{datetime.now().strftime('%Y-%m-%d')}.jsonl")

    def _record_tick(self, slug: str, market, ptb: float, pred_result: dict):
        """Append ONE tick to JSONL. Flat schema, no nesting. Ultra-fast (append only)."""
        elapsed_pct = self._elapsed_pct(slug, market.duration_minutes)
        all_preds = pred_result["all_preds"]

        record = {
            "ts": round(time.time(), 3),
            "slug": slug,
            "entry_pct": round(elapsed_pct, 4),
            "dur_min": market.duration_minutes,
            "sol": round(pred_result["sol_price"], 4),
            "ptb": round(ptb, 4),
            "gap_pct": round((pred_result["sol_price"] - ptb) / ptb * 100, 3) if ptb > 0 else 0,
            "yes": round(market.yes_price, 4),       # YES ask (entry price for UP)
            "no": round(market.no_price, 4),         # NO ask (entry price for DOWN)
            "yes_bid": round(market.best_bid, 4),
            "yes_ask": round(market.best_ask, 4),
            "yes_spread": round(market.spread, 4),
            "no_bid": round(market.no_best_bid, 4),
            "no_ask": round(market.no_best_ask, 4),
            "no_spread": round(market.no_spread, 4),
            "yes_depth": round(market.yes_depth, 1),
            "no_depth": round(market.no_depth, 1),
            # Full orderbook metrics (from WS book snapshots)
            "yes_bid_depth": round(getattr(market, 'yes_bid_depth', 0), 1),
            "yes_ask_depth": round(getattr(market, 'yes_ask_depth', 0), 1),
            "yes_levels": getattr(market, 'yes_levels', 0),
            "yes_bid_levels": getattr(market, 'yes_bid_levels', 0),
            "no_bid_depth": round(getattr(market, 'no_bid_depth', 0), 1),
            "no_ask_depth": round(getattr(market, 'no_ask_depth', 0), 1),
            "no_levels": getattr(market, 'no_levels', 0),
            "no_bid_levels": getattr(market, 'no_bid_levels', 0),
            "dir": pred_result["direction"],
            "sp": round(pred_result["share_price"], 4),
        }
        # Flat model predictions (prob_up for each) — no nesting
        for model_name, p_up in all_preds.items():
            record[model_name] = round(p_up, 4)

        try:
            with open(self._get_tick_path(), "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            log.debug(f"Record tick error: {e}")

    def _record_outcome(self, slug: str, outcome: str, sol_start: float, sol_end: float, ptb: float, dur_min: int):
        """Append ONE market outcome to JSONL."""
        record = {
            "ts": round(time.time(), 3),
            "slug": slug,
            "outcome": outcome,
            "sol_start": round(sol_start, 4),
            "sol_end": round(sol_end, 4),
            "ptb": round(ptb, 4),
            "dur_min": dur_min,
        }
        try:
            with open(self._get_outcomes_path(), "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            log.debug(f"Record outcome error: {e}")

    async def _recording_loop(self):
        """Record ML predictions for ALL active markets, 0-100%. Runs every 5s.

        This is the SOLE recording source. Entry loop does NOT record ticks.
        Separate from entry logic — pure observation.
        """
        await asyncio.sleep(8.0)  # Wait for models + klines to load
        log.info("  📊 Recording loop started — ALL markets, 0-100%")

        while self._running:
            if not self._model or len(self._klines) < 61:
                await asyncio.sleep(5.0)
                continue

            recorded = 0
            for slug, info in list(self._active_markets.items()):
                market = info["market"]
                ptb = info.get("ptb", self._sol_price)

                # Use real-time elapsed from slug epoch
                elapsed_pct = self._elapsed_pct(slug, market.duration_minutes)
                if elapsed_pct >= 1.0:
                    continue  # market ended, will be cleaned up by discovery

                try:
                    result = self._compute_predictions(market, ptb)
                    if result is None:
                        continue

                    # ── Store as single source of truth for entry loop ──
                    self._latest_predictions[slug] = {
                        "result": result, "market": market,
                        "ptb": ptb, "ts": time.time(),
                    }
                    # ── Prediction history for smoothing / stability ──
                    if slug not in self._prediction_history:
                        self._prediction_history[slug] = []
                    self._prediction_history[slug].append({
                        "prob_up": result["prob_up"],
                        "direction": result["direction"],
                        "dir_prob": result["dir_prob"],
                        "ts": time.time(),
                    })
                    # Keep only last N
                    self._prediction_history[slug] = self._prediction_history[slug][-self._SMOOTHING_WINDOW:]

                    self._record_tick(slug, market, ptb, result)
                    recorded += 1
                except Exception as e:
                    log.debug(f"  Recording error {slug}: {e}")

            if recorded > 0:
                log.debug(f"  📊 Recorded {recorded} ticks")

            await asyncio.sleep(5.0)

    def _log_trade_json(self, pos: MLPosition, result: TradeResult, outcome: str):
        """Append a single trade record to JSON log file."""
        Path("results").mkdir(exist_ok=True)
        json_path = Path("results/ml_live_trades.json")

        # Load existing trades
        trades = []
        if json_path.exists():
            try:
                trades = json.loads(json_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                trades = []

        # Build detailed record
        gap = pos.sol_price_at_entry - pos.price_to_beat
        try:
            end_ts = datetime.fromisoformat(pos.end_date.replace("Z", "+00:00")).timestamp()
            time_remaining_at_entry_s = max(0, end_ts - pos.entry_ts)
        except (ValueError, TypeError, AttributeError):
            time_remaining_at_entry_s = 0

        # Grab per-trade snapshots
        trade_snaps = self._trade_snapshots.pop(pos.market_slug, [])

        record = {
            "slug": pos.market_slug,
            "direction": pos.direction,
            "confidence": round(pos.confidence, 4),
            "model_prob_up": round(pos.model_prob, 4),
            "entry_price": round(pos.entry_price, 4),
            "ptb": round(pos.price_to_beat, 4),
            "sol_at_entry": round(pos.sol_price_at_entry, 4),
            "sol_at_exit": round(result.sol_at_exit, 4),
            "gap_at_entry": round(gap, 4),
            "duration_min": pos.duration_minutes,
            "time_remaining_at_entry_s": round(time_remaining_at_entry_s, 1),
            "hold_time_s": round(result.hold_time_s, 1),
            "outcome": outcome,
            "won": int("win" in result.reason),
            "pnl_pct": round(result.pnl_pct, 2),
            "pnl_usd": round(result.pnl_usd, 4),
            "shares": round(pos.shares, 4),
            "entry_ts": round(pos.entry_ts, 3),
            "exit_ts": round(result.ts, 3),
            "entry_time": datetime.fromtimestamp(pos.entry_ts).strftime("%Y-%m-%d %H:%M:%S"),
            "exit_time": datetime.fromtimestamp(result.ts).strftime("%Y-%m-%d %H:%M:%S"),
            "dry_run": self.dry_run,
            "snapshots": trade_snaps,
        }

        trades.append(record)
        json_path.write_text(json.dumps(trades, indent=2, ensure_ascii=False), encoding="utf-8")

    def _load_trades(self):
        """Load completed trades from disk to survive restarts."""
        json_path = Path("results/ml_live_trades.json")
        if not json_path.exists():
            return
        try:
            trades = json.loads(json_path.read_text(encoding="utf-8"))
            if not trades:
                return
            loaded = 0
            for t in trades:
                won = t.get("won", 0)
                reason = t.get("outcome", "expiry_win" if won else "expiry_loss")
                if "win" not in reason and "loss" not in reason:
                    reason = "expiry_win" if won else "expiry_loss"
                tr = TradeResult(
                    slug=t.get("slug", ""),
                    direction=t.get("direction", ""),
                    entry_price=t.get("entry_price", 0),
                    exit_price=t.get("entry_price", 0) + t.get("pnl_usd", 0) / max(t.get("shares", 1), 0.01),
                    shares=t.get("shares", 0),
                    pnl_usd=t.get("pnl_usd", 0),
                    pnl_pct=t.get("pnl_pct", 0),
                    confidence=t.get("confidence", 0),
                    model_prob=t.get("model_prob_up", t.get("model_prob", 0)),
                    hold_time_s=t.get("hold_time_s", 0),
                    reason=reason,
                    sol_at_entry=t.get("sol_at_entry", 0),
                    sol_at_exit=t.get("sol_at_exit", 0),
                    ptb=t.get("ptb", 0),
                    ts=t.get("exit_ts", t.get("ts", 0)),
                )
                self.completed.append(tr)
                loaded += 1

            total_pnl = sum(t.pnl_usd for t in self.completed)
            self._capital += total_pnl
            wins = sum(1 for t in self.completed if "win" in t.reason)
            log.info(
                f"  📂 Loaded {loaded} trades from disk: "
                f"{wins}W/{loaded-wins}L PnL=${total_pnl:+.2f} cap=${self._capital:.2f}"
            )
        except Exception as e:
            log.warning(f"  ⚠️ Could not load trades: {e}")

    def _save_trades(self):
        """Save trades to disk."""
        if not self.completed:
            return
        Path("results").mkdir(exist_ok=True)
        df = pd.DataFrame([asdict(t) for t in self.completed])
        df.to_parquet("results/ml_live_trades.parquet")
        df.to_csv("results/ml_live_trades.csv", index=False)
