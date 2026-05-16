"""Polymarket Real-Time Data Recorder.

Records real Polymarket shares data every N seconds for all active SOL markets.
Data is saved to partitioned parquet files for honest replay backtesting.

Architecture:
  1. Market Discovery Loop   — Gamma API every 12s → find active markets
  2. Snapshot Loop            — every interval (default 5s):
     - For each active market: fetch CLOB orderbooks (UP + DOWN tokens)
     - Fetch Binance SOL/USDT price
     - Compute real-time features (momentum, acceleration, orderflow)
     - Append to in-memory buffer
  3. Flush Loop               — every 60s flush buffer → parquet
  4. Console Loop             — Rich live display with progress bars

Output structure:
  data/recorded/shares/YYYY-MM-DD/5m/snapshots.parquet
  data/recorded/shares/YYYY-MM-DD/15m/snapshots.parquet
  data/recorded/shares/YYYY-MM-DD/60m/snapshots.parquet

Usage:
  python main.py --mode record --interval 5 --duration 24h
"""

import asyncio
import time
import math
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import httpx

from core.utils.logger import log
from config import config

_pm_cfg = config.get("infrastructure", {}).get("polymarket", {})
_bn_cfg = config.get("infrastructure", {}).get("binance", {})

GAMMA_HOST = _pm_cfg.get("gamma_host", "https://gamma-api.polymarket.com")
CLOB_HOST = _pm_cfg.get("clob_host", "https://clob.polymarket.com")
BINANCE_REST = _bn_cfg.get("rest_url", "https://fapi.binance.com")
RECORD_DIR = Path("data/recorded/shares")


# ═══════════════════════════════════════════════════════════
#  SNAPSHOT DATA
# ═══════════════════════════════════════════════════════════

SNAPSHOT_COLUMNS = [
    # Identity
    "ts", "market_id", "slug", "duration_min",
    "creation_time", "expiration_time", "time_remaining_sec", "time_elapsed_sec",
    "time_remaining_pct",
    # Resolution
    "price_to_beat",
    # UP shares
    "up_best_bid", "up_best_ask", "up_mid_price", "up_spread", "up_spread_pct",
    "up_bid_volume", "up_ask_volume", "up_total_volume",
    "up_bid_depth_5", "up_ask_depth_5",
    # DOWN shares
    "dn_best_bid", "dn_best_ask", "dn_mid_price", "dn_spread", "dn_spread_pct",
    "dn_bid_volume", "dn_ask_volume", "dn_total_volume",
    "dn_bid_depth_5", "dn_ask_depth_5",
    # Orderbook imbalance
    "up_ob_imbalance", "dn_ob_imbalance",
    # CEX
    "sol_price", "sol_bid", "sol_ask",
    # Gamma API prices (for comparison)
    "gamma_yes_price", "gamma_no_price",
    # Computed features
    "shares_momentum_30s", "shares_momentum_2m",
    "shares_acceleration",
    "liquidity_score",
    "volume_spike",
    "volume_imbalance",
    # Market state
    "accepting_orders", "outcome",
]


# ═══════════════════════════════════════════════════════════
#  MARKET TRACKER (per-market rolling state)
# ═══════════════════════════════════════════════════════════

class MarketTracker:
    """Tracks rolling state for a single market — momentum, volume history."""

    def __init__(self, slug: str, duration_min: int):
        self.slug = slug
        self.duration_min = duration_min
        self.mid_history: deque = deque(maxlen=600)   # (ts, up_mid) — 10 min at 1s
        self.volume_history: deque = deque(maxlen=600)  # (ts, total_vol)
        self.first_seen_ts: float = time.time()

    def add_snapshot(self, ts: float, up_mid: float, total_vol: float):
        self.mid_history.append((ts, up_mid))
        self.volume_history.append((ts, total_vol))

    def momentum(self, lookback_sec: float) -> float:
        """Price change over lookback period."""
        if len(self.mid_history) < 2:
            return 0.0
        now_ts, now_price = self.mid_history[-1]
        cutoff = now_ts - lookback_sec
        for ts, price in self.mid_history:
            if ts >= cutoff:
                return (now_price - price) / max(price, 0.001)
        return 0.0

    def volume_spike(self, lookback_sec: float = 600) -> float:
        """Current volume / average volume over lookback."""
        if len(self.volume_history) < 3:
            return 1.0
        now_ts, now_vol = self.volume_history[-1]
        cutoff = now_ts - lookback_sec
        past_vols = [v for t, v in self.volume_history if t >= cutoff and t < now_ts]
        if not past_vols:
            return 1.0
        avg = np.mean(past_vols)
        return now_vol / max(avg, 0.001)


# ═══════════════════════════════════════════════════════════
#  RECORDER
# ═══════════════════════════════════════════════════════════

class PolymarketRecorder:
    """Real-time Polymarket data recorder."""

    def __init__(
        self,
        interval_sec: float = 5.0,
        market_slugs: List[str] = None,
        duration_sec: float = None,  # None = infinite
        headless: bool = False,  # True = no Rich console (when TUI is active)
    ):
        self.interval_sec = interval_sec
        self.slugs = market_slugs or config.get("shares", {}).get(
            "market_slugs", ["sol-updown-15m", "sol-updown-5m"]
        )
        self.duration_sec = duration_sec

        self._http: Optional[httpx.AsyncClient] = None
        self._running = False

        # Active markets: slug → SharesMarket-like dict
        self._active_markets: Dict[str, Dict] = {}
        # Per-market trackers
        self._trackers: Dict[str, MarketTracker] = {}
        # Snapshot buffer (per duration)
        self._buffers: Dict[int, List[Dict]] = defaultdict(list)
        # Flush interval
        self._flush_interval = 60  # seconds
        self._last_flush_ts = time.time()

        # SOL price cache
        self._sol_price = 0.0
        self._sol_bid = 0.0
        self._sol_ask = 0.0

        # Stats
        self._total_snapshots = 0
        self._total_markets_seen = 0
        self._start_ts = 0
        self._errors = 0
        self._resolved_markets: List[Dict] = []

        self._headless = headless
        RECORD_DIR.mkdir(parents=True, exist_ok=True)

    async def _get_http(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(
                timeout=12.0,
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
                headers={"Accept": "application/json"},
            )
        return self._http

    async def close(self):
        if self._http and not self._http.is_closed:
            await self._http.aclose()

    async def _get(self, url: str, params: dict = None) -> Optional[Any]:
        http = await self._get_http()
        try:
            resp = await http.get(url, params=params)
            if resp.status_code != 200:
                return None
            return resp.json()
        except Exception as e:
            self._errors += 1
            return None

    # ═══════════════════════════════════════════════════════════
    #  START / STOP
    # ═══════════════════════════════════════════════════════════

    async def start(self):
        """Start recording."""
        self._running = True
        self._start_ts = time.time()

        log.info("╔══════════════════════════════════════════════════╗")
        log.info("║  Polymarket Recorder — Starting                 ║")
        log.info(f"║  Slugs: {', '.join(self.slugs):<38}║")
        log.info(f"║  Interval: {self.interval_sec}s{'':>35}║")
        dur_str = f"{self.duration_sec/3600:.0f}h" if self.duration_sec else "infinite"
        log.info(f"║  Duration: {dur_str:<37}║")
        log.info("╚══════════════════════════════════════════════════╝")

        try:
            loops = [
                self._market_discovery_loop(),
                self._snapshot_loop(),
                self._flush_loop(),
                self._duration_watchdog(),
            ]
            if not self._headless:
                loops.append(self._console_loop())
            await asyncio.gather(*loops)
        except asyncio.CancelledError:
            pass
        finally:
            # Final flush
            self._flush_all()
            await self.close()
            self._print_summary()

    async def stop(self):
        self._running = False

    # ═══════════════════════════════════════════════════════════
    #  MARKET DISCOVERY
    # ═══════════════════════════════════════════════════════════

    async def _market_discovery_loop(self):
        """Discover active markets every 12 seconds."""
        while self._running:
            try:
                await self._refresh_markets()
            except Exception as e:
                log.error(f"Market discovery error: {e}")
                self._errors += 1
            await asyncio.sleep(12.0)

    async def _refresh_markets(self):
        """Fetch all active markets from Gamma API."""
        import re
        new_markets = {}

        for base_slug in self.slugs:
            # Parse duration
            m = re.search(r'-(\d+)m$', base_slug)
            dur_min = int(m.group(1)) if m else 15
            interval_sec = dur_min * 60

            # Generate timestamped slugs
            now = int(time.time())
            current_start = (now // interval_sec) * interval_sec
            for i in range(-1, 5):
                ts = current_start + interval_sec * i
                full_slug = f"{base_slug}-{ts}"
                market = await self._fetch_gamma_event(full_slug, dur_min)
                if market:
                    new_markets[full_slug] = market
                    if full_slug not in self._trackers:
                        self._trackers[full_slug] = MarketTracker(full_slug, dur_min)
                        self._total_markets_seen += 1

        # Detect resolved markets
        for slug, market in self._active_markets.items():
            if slug not in new_markets:
                # Market disappeared — likely resolved
                market["outcome"] = self._determine_outcome(market)
                self._resolved_markets.append(market)
                if slug in self._trackers:
                    del self._trackers[slug]

        self._active_markets = new_markets

    async def _fetch_gamma_event(self, full_slug: str, dur_min: int) -> Optional[Dict]:
        """Fetch single event from Gamma API."""
        data = await self._get(f"{GAMMA_HOST}/events", {"slug": full_slug, "limit": 1})
        if not data or len(data) == 0:
            return None

        event = data[0]
        markets = event.get("markets", [])
        if not markets:
            return None

        market = markets[0]
        if market.get("closed") or not market.get("active", True):
            return None

        # Token IDs
        yes_token = no_token = None
        try:
            tokens = json.loads(market.get("clobTokenIds", "[]"))
            yes_token = tokens[0] if len(tokens) > 0 else None
            no_token = tokens[1] if len(tokens) > 1 else None
        except (json.JSONDecodeError, IndexError):
            pass
        if not yes_token or not no_token:
            return None

        # Prices from Gamma
        yes_price = no_price = 0.5
        try:
            prices = json.loads(market.get("outcomePrices", "[]"))
            yes_price = float(prices[0]) if len(prices) > 0 else 0.5
            no_price = float(prices[1]) if len(prices) > 1 else 0.5
        except (json.JSONDecodeError, IndexError, ValueError):
            pass

        # PriceToBeat
        meta = event.get("eventMetadata") or {}
        ptb = float(meta["priceToBeat"]) if meta.get("priceToBeat") is not None else None
        final_price = float(meta["finalPrice"]) if meta.get("finalPrice") is not None else None

        # Timing
        event_start = event.get("startTime") or market.get("eventStartTime")
        end_date = market.get("endDate") or market.get("endDateIso") or event.get("endDate")

        now_ms = int(time.time() * 1000)
        time_remaining_ms = time_elapsed_ms = 0
        creation_time = None
        expiration_time = None

        if end_date:
            try:
                end_dt = datetime.fromisoformat(str(end_date).replace("Z", "+00:00"))
                expiration_time = end_dt.isoformat()
                time_remaining_ms = max(0, int(end_dt.timestamp() * 1000) - now_ms)
            except (ValueError, TypeError):
                pass
        if event_start:
            try:
                start_dt = datetime.fromisoformat(str(event_start).replace("Z", "+00:00"))
                creation_time = start_dt.isoformat()
                time_elapsed_ms = max(0, now_ms - int(start_dt.timestamp() * 1000))
            except (ValueError, TypeError):
                pass

        return {
            "slug": full_slug,
            "condition_id": market.get("conditionId", ""),
            "yes_token_id": yes_token,
            "no_token_id": no_token,
            "duration_min": dur_min,
            "price_to_beat": ptb,
            "final_price": final_price,
            "creation_time": creation_time,
            "expiration_time": expiration_time,
            "time_remaining_ms": time_remaining_ms,
            "time_elapsed_ms": time_elapsed_ms,
            "gamma_yes_price": yes_price,
            "gamma_no_price": no_price,
            "accepting_orders": market.get("acceptingOrders", True),
            "neg_risk": bool(market.get("negRisk")),
        }

    def _determine_outcome(self, market: Dict) -> Optional[str]:
        """Determine outcome of a resolved market."""
        fp = market.get("final_price")
        ptb = market.get("price_to_beat")
        if fp is not None and ptb is not None:
            return "Up" if fp >= ptb else "Down"
        return None

    # ═══════════════════════════════════════════════════════════
    #  SNAPSHOT COLLECTION
    # ═══════════════════════════════════════════════════════════

    async def _snapshot_loop(self):
        """Main data collection loop — every interval_sec."""
        # Wait for first market discovery + initial SOL price
        await asyncio.sleep(3.0)
        await self._fetch_sol_price()

        while self._running:
            t0 = time.time()
            try:
                # Fetch SOL price in parallel with orderbooks
                tasks = [self._fetch_sol_price()]
                for slug, market in list(self._active_markets.items()):
                    tasks.append(self._collect_market_snapshot(slug, market))

                await asyncio.gather(*tasks, return_exceptions=True)

            except Exception as e:
                log.error(f"Snapshot loop error: {e}")
                self._errors += 1

            elapsed = time.time() - t0
            sleep_time = max(0.1, self.interval_sec - elapsed)
            await asyncio.sleep(sleep_time)

    async def _fetch_sol_price(self):
        """Fetch SOL/USD price from Pyth Hermes (same oracle as Polymarket), Binance fallback."""
        pyth_sol_id = "0xef0d8b6fda2ceba41da15d4095d1da392a0d2f8ed0c6c7bc0f4cfac8c280b56d"
        try:
            data = await self._get(
                "https://hermes.pyth.network/v2/updates/price/latest",
                {"ids[]": pyth_sol_id, "parsed": "true"},
            )
            if data and "parsed" in data and data["parsed"]:
                p = data["parsed"][0]["price"]
                price = round(int(p["price"]) * 10 ** int(p["expo"]), 6)
                self._sol_price = price
                self._sol_bid = price  # Pyth doesn't have bid/ask
                self._sol_ask = price
                return
        except Exception:
            pass
        # Fallback: Binance
        data = await self._get(
            f"{BINANCE_REST}/fapi/v1/ticker/bookTicker",
            {"symbol": "SOLUSDT"},
        )
        if data:
            self._sol_price = (float(data.get("bidPrice", 0)) + float(data.get("askPrice", 0))) / 2
            self._sol_bid = float(data.get("bidPrice", 0))
            self._sol_ask = float(data.get("askPrice", 0))

    async def _fetch_orderbook(self, token_id: str) -> Dict:
        """Fetch full CLOB orderbook for a token."""
        data = await self._get(f"{CLOB_HOST}/book", {"token_id": token_id})
        if not data:
            return {"best_bid": 0, "best_ask": 1, "mid": 0.5, "spread": 1,
                    "bid_volume": 0, "ask_volume": 0, "bid_depth_5": 0, "ask_depth_5": 0,
                    "imbalance": 0}

        bids = sorted(data.get("bids", []), key=lambda x: -float(x.get("price", 0)))
        asks = sorted(data.get("asks", []), key=lambda x: float(x.get("price", 999)))

        best_bid = float(bids[0]["price"]) if bids else 0
        best_ask = float(asks[0]["price"]) if asks else 1
        mid = (best_bid + best_ask) / 2 if best_bid > 0 and best_ask < 1 else 0.5
        spread = best_ask - best_bid

        bid_volume = sum(float(b.get("size", 0)) for b in bids)
        ask_volume = sum(float(a.get("size", 0)) for a in asks)
        bid_depth_5 = sum(float(b.get("size", 0)) for b in bids[:5])
        ask_depth_5 = sum(float(a.get("size", 0)) for a in asks[:5])

        total = bid_volume + ask_volume
        imbalance = (bid_volume - ask_volume) / total if total > 0 else 0

        return {
            "best_bid": best_bid, "best_ask": best_ask,
            "mid": mid, "spread": spread,
            "bid_volume": bid_volume, "ask_volume": ask_volume,
            "bid_depth_5": bid_depth_5, "ask_depth_5": ask_depth_5,
            "imbalance": imbalance,
        }

    async def _collect_market_snapshot(self, slug: str, market: Dict):
        """Collect a full snapshot for one market."""
        now_ts = time.time()
        dur_min = market["duration_min"]

        # Fetch orderbooks for UP and DOWN tokens in parallel
        up_ob, dn_ob = await asyncio.gather(
            self._fetch_orderbook(market["yes_token_id"]),
            self._fetch_orderbook(market["no_token_id"]),
        )

        # Compute time
        time_remaining_sec = market["time_remaining_ms"] / 1000 - (now_ts - self._start_ts) * 0
        # Recalculate from expiration time for accuracy
        if market.get("expiration_time"):
            try:
                end_dt = datetime.fromisoformat(market["expiration_time"])
                time_remaining_sec = max(0, end_dt.timestamp() - now_ts)
            except (ValueError, TypeError):
                time_remaining_sec = max(0, market["time_remaining_ms"] / 1000)
        else:
            time_remaining_sec = max(0, market["time_remaining_ms"] / 1000)

        time_elapsed_sec = dur_min * 60 - time_remaining_sec
        total_sec = dur_min * 60
        time_remaining_pct = time_remaining_sec / total_sec if total_sec > 0 else 0

        # Spread percentages
        up_mid = up_ob["mid"]
        dn_mid = dn_ob["mid"]
        up_spread_pct = up_ob["spread"] / up_mid if up_mid > 0.01 else 0
        dn_spread_pct = dn_ob["spread"] / dn_mid if dn_mid > 0.01 else 0

        # Total volumes
        up_total_vol = up_ob["bid_volume"] + up_ob["ask_volume"]
        dn_total_vol = dn_ob["bid_volume"] + dn_ob["ask_volume"]
        total_vol = up_total_vol + dn_total_vol

        # Update tracker
        tracker = self._trackers.get(slug)
        if tracker:
            tracker.add_snapshot(now_ts, up_mid, total_vol)

        # Computed features
        momentum_30s = tracker.momentum(30) if tracker else 0
        momentum_2m = tracker.momentum(120) if tracker else 0
        acceleration = momentum_30s - momentum_2m
        vol_spike = tracker.volume_spike(600) if tracker else 1.0
        vol_imbalance = abs(up_total_vol - dn_total_vol) / max(total_vol, 0.001)

        # Liquidity score: log(volume) * (1 / spread_pct)
        avg_spread_pct = (up_spread_pct + dn_spread_pct) / 2
        liq_score = math.log1p(total_vol) * (1.0 / max(avg_spread_pct, 0.001)) if total_vol > 0 else 0

        snapshot = {
            "ts": now_ts,
            "market_id": market.get("condition_id", ""),
            "slug": slug,
            "duration_min": dur_min,
            "creation_time": market.get("creation_time", ""),
            "expiration_time": market.get("expiration_time", ""),
            "time_remaining_sec": round(time_remaining_sec, 1),
            "time_elapsed_sec": round(time_elapsed_sec, 1),
            "time_remaining_pct": round(time_remaining_pct, 4),
            "price_to_beat": market.get("price_to_beat"),
            # UP
            "up_best_bid": up_ob["best_bid"],
            "up_best_ask": up_ob["best_ask"],
            "up_mid_price": up_mid,
            "up_spread": up_ob["spread"],
            "up_spread_pct": round(up_spread_pct, 4),
            "up_bid_volume": up_ob["bid_volume"],
            "up_ask_volume": up_ob["ask_volume"],
            "up_total_volume": up_total_vol,
            "up_bid_depth_5": up_ob["bid_depth_5"],
            "up_ask_depth_5": up_ob["ask_depth_5"],
            # DOWN
            "dn_best_bid": dn_ob["best_bid"],
            "dn_best_ask": dn_ob["best_ask"],
            "dn_mid_price": dn_mid,
            "dn_spread": dn_ob["spread"],
            "dn_spread_pct": round(dn_spread_pct, 4),
            "dn_bid_volume": dn_ob["bid_volume"],
            "dn_ask_volume": dn_ob["ask_volume"],
            "dn_total_volume": dn_total_vol,
            "dn_bid_depth_5": dn_ob["bid_depth_5"],
            "dn_ask_depth_5": dn_ob["ask_depth_5"],
            # OB imbalance
            "up_ob_imbalance": up_ob["imbalance"],
            "dn_ob_imbalance": dn_ob["imbalance"],
            # CEX
            "sol_price": self._sol_price,
            "sol_bid": self._sol_bid,
            "sol_ask": self._sol_ask,
            # Gamma prices
            "gamma_yes_price": market.get("gamma_yes_price", 0.5),
            "gamma_no_price": market.get("gamma_no_price", 0.5),
            # Computed features
            "shares_momentum_30s": round(momentum_30s, 6),
            "shares_momentum_2m": round(momentum_2m, 6),
            "shares_acceleration": round(acceleration, 6),
            "liquidity_score": round(liq_score, 2),
            "volume_spike": round(vol_spike, 4),
            "volume_imbalance": round(vol_imbalance, 4),
            # State
            "accepting_orders": market.get("accepting_orders", True),
            "outcome": None,
        }

        self._buffers[dur_min].append(snapshot)
        self._total_snapshots += 1

    # ═══════════════════════════════════════════════════════════
    #  FLUSH TO DISK
    # ═══════════════════════════════════════════════════════════

    async def _flush_loop(self):
        """Periodically flush buffers to parquet."""
        while self._running:
            await asyncio.sleep(self._flush_interval)
            self._flush_all()

    def _flush_all(self):
        """Flush all duration buffers to disk."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        for dur_min, records in self._buffers.items():
            if not records:
                continue

            dir_path = RECORD_DIR / today / f"{dur_min}m"
            dir_path.mkdir(parents=True, exist_ok=True)

            df = pd.DataFrame(records)
            fpath = dir_path / "snapshots.parquet"

            if fpath.exists():
                existing = pd.read_parquet(fpath)
                # Drop all-NA columns to avoid FutureWarning
                existing = existing.dropna(axis=1, how='all')
                df = df.dropna(axis=1, how='all')
                df = pd.concat([existing, df], ignore_index=True)

            df.to_parquet(fpath, engine="pyarrow", compression="snappy")
            log.debug(f"Flushed {len(records)} snapshots → {fpath} (total {len(df)})")

        # Clear buffers
        for dur_min in list(self._buffers.keys()):
            self._buffers[dur_min] = []

        # Save resolved markets
        if self._resolved_markets:
            resolved_path = RECORD_DIR / today / "resolved.parquet"
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            df_resolved = pd.DataFrame(self._resolved_markets)
            if resolved_path.exists():
                existing = pd.read_parquet(resolved_path)
                existing = existing.dropna(axis=1, how='all')
                df_resolved = df_resolved.dropna(axis=1, how='all')
                df_resolved = pd.concat([existing, df_resolved], ignore_index=True)
            df_resolved.to_parquet(resolved_path, engine="pyarrow", compression="snappy")
            self._resolved_markets = []

        self._last_flush_ts = time.time()

    # ═══════════════════════════════════════════════════════════
    #  CONSOLE DISPLAY
    # ═══════════════════════════════════════════════════════════

    async def _console_loop(self):
        """Rich live console with progress bars."""
        try:
            from rich.live import Live
            from rich.table import Table
            from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
            from rich.panel import Panel
            from rich.layout import Layout
            from rich.console import Console
            from rich.text import Text

            console = Console()

            with Live(console=console, refresh_per_second=2) as live:
                while self._running:
                    table = self._build_display_table()
                    live.update(table)
                    await asyncio.sleep(0.5)

        except ImportError:
            # Fallback: simple logging
            while self._running:
                await asyncio.sleep(10.0)
                n_markets = len(self._active_markets)
                elapsed = time.time() - self._start_ts
                log.info(
                    f"📊 Recording: {n_markets} markets | "
                    f"{self._total_snapshots} snapshots | "
                    f"{elapsed/60:.0f}m elapsed | "
                    f"{self._errors} errors"
                )

    def _build_display_table(self):
        from rich.table import Table
        from rich.panel import Panel
        from rich.text import Text
        from rich import box

        now = time.time()
        elapsed = now - self._start_ts
        elapsed_str = f"{int(elapsed//3600)}h {int((elapsed%3600)//60)}m {int(elapsed%60)}s"
        remaining_str = "∞"
        if self.duration_sec:
            rem = max(0, self.duration_sec - elapsed)
            remaining_str = f"{int(rem//3600)}h {int((rem%3600)//60)}m"

        # Header stats
        header = Text()
        header.append("🔴 RECORDING", style="bold red")
        header.append(f"  |  Elapsed: {elapsed_str}", style="white")
        header.append(f"  |  Remaining: {remaining_str}", style="dim")
        header.append(f"  |  Snapshots: {self._total_snapshots:,}", style="green")
        header.append(f"  |  Errors: {self._errors}", style="yellow" if self._errors else "dim")
        header.append(f"  |  SOL: ${self._sol_price:.2f}", style="cyan")

        # Markets table
        table = Table(
            box=box.ROUNDED, show_header=True, header_style="bold cyan",
            title=str(header), title_style="",
            expand=True,
        )
        table.add_column("Market", style="white", width=36)
        table.add_column("Dur", style="dim", width=4)
        table.add_column("PTB", style="yellow", width=8)
        table.add_column("UP Mid", style="green", width=7)
        table.add_column("DN Mid", style="red", width=7)
        table.add_column("Spread", style="dim", width=7)
        table.add_column("Vol", style="cyan", width=8)
        table.add_column("Mom30s", style="magenta", width=8)
        table.add_column("Time Left", style="white", width=10)
        table.add_column("Progress", width=22)

        # Sort by time remaining
        sorted_markets = sorted(
            self._active_markets.items(),
            key=lambda x: x[1].get("time_remaining_ms", 0),
        )

        for slug, market in sorted_markets:
            dur_min = market["duration_min"]
            ptb = market.get("price_to_beat")

            # Get latest snapshot data from tracker
            tracker = self._trackers.get(slug)
            up_mid = tracker.mid_history[-1][1] if tracker and tracker.mid_history else 0.5
            dn_mid = 1.0 - up_mid

            # Time remaining
            tr_sec = 0
            if market.get("expiration_time"):
                try:
                    end_dt = datetime.fromisoformat(market["expiration_time"])
                    tr_sec = max(0, end_dt.timestamp() - now)
                except (ValueError, TypeError):
                    tr_sec = max(0, market.get("time_remaining_ms", 0) / 1000)
            else:
                tr_sec = max(0, market.get("time_remaining_ms", 0) / 1000)

            total_sec = dur_min * 60
            progress_pct = max(0, min(1, 1 - tr_sec / total_sec)) if total_sec > 0 else 0
            progress_bar = self._make_progress_bar(progress_pct)

            # Momentum
            mom30 = tracker.momentum(30) if tracker else 0

            # Volume from latest buffer entry
            latest_vol = 0
            latest_spread = 0
            buf = self._buffers.get(dur_min, [])
            for snap in reversed(buf):
                if snap["slug"] == slug:
                    latest_vol = snap.get("up_total_volume", 0) + snap.get("dn_total_volume", 0)
                    latest_spread = snap.get("up_spread", 0)
                    break

            # Time display
            if tr_sec >= 60:
                time_str = f"{int(tr_sec//60)}m {int(tr_sec%60)}s"
            else:
                time_str = f"{tr_sec:.0f}s"

            # Color momentum
            mom_str = f"{mom30:+.4f}"
            mom_style = "green" if mom30 > 0 else "red" if mom30 < 0 else "dim"

            table.add_row(
                slug[-30:],
                f"{dur_min}m",
                f"${ptb:.2f}" if ptb else "—",
                f"${up_mid:.3f}",
                f"${dn_mid:.3f}",
                f"${latest_spread:.3f}",
                f"{latest_vol:.0f}",
                Text(mom_str, style=mom_style),
                time_str,
                progress_bar,
            )

        return Panel(table, border_style="blue")

    @staticmethod
    def _make_progress_bar(pct: float) -> "Text":
        from rich.text import Text
        width = 20
        filled = int(pct * width)
        remaining = width - filled

        bar = Text()
        bar.append("█" * filled, style="green")
        bar.append("░" * remaining, style="dim")
        bar.append(f" {pct*100:.0f}%", style="bold white")
        return bar

    # ═══════════════════════════════════════════════════════════
    #  WATCHDOG
    # ═══════════════════════════════════════════════════════════

    async def _duration_watchdog(self):
        """Stop recording after duration expires."""
        if not self.duration_sec:
            # Infinite — just sleep forever
            while self._running:
                await asyncio.sleep(60)
            return

        await asyncio.sleep(self.duration_sec)
        log.info(f"⏰ Recording duration ({self.duration_sec/3600:.1f}h) reached — stopping")
        self._running = False

    # ═══════════════════════════════════════════════════════════
    #  SUMMARY
    # ═══════════════════════════════════════════════════════════

    def _print_summary(self):
        elapsed = time.time() - self._start_ts
        log.info("\n╔══════════════════════════════════════════════════╗")
        log.info("║         RECORDING SUMMARY                       ║")
        log.info("╠══════════════════════════════════════════════════╣")
        log.info(f"║  Duration:     {elapsed/3600:.1f}h ({elapsed/60:.0f}m)")
        log.info(f"║  Snapshots:    {self._total_snapshots:,}")
        log.info(f"║  Markets seen: {self._total_markets_seen}")
        log.info(f"║  Resolved:     {len(self._resolved_markets)}")
        log.info(f"║  Errors:       {self._errors}")

        # Size per duration
        for dur_dir in sorted(RECORD_DIR.glob("*/*m")):
            snap_file = dur_dir / "snapshots.parquet"
            if snap_file.exists():
                df = pd.read_parquet(snap_file)
                size_mb = snap_file.stat().st_size / 1024 / 1024
                log.info(f"║  {dur_dir.name}: {len(df):,} rows ({size_mb:.1f} MB)")

        log.info("╚══════════════════════════════════════════════════╝")


# ═══════════════════════════════════════════════════════════
#  SHOW RECORDED DATA
# ═══════════════════════════════════════════════════════════

def show_recorded_data(date: str = None):
    """Display summary of recorded data."""
    if not RECORD_DIR.exists():
        log.info("No recorded data found.")
        return

    dates = sorted(d.name for d in RECORD_DIR.iterdir() if d.is_dir() and d.name != "__pycache__")
    if not dates:
        log.info("No recorded data found.")
        return

    if date:
        dates = [d for d in dates if d == date]

    log.info(f"\n{'='*60}")
    log.info(f"  Recorded Polymarket Data")
    log.info(f"{'='*60}")

    total_snapshots = 0
    total_markets = 0

    for d in dates:
        date_dir = RECORD_DIR / d
        log.info(f"\n  📅 {d}")

        for dur_dir in sorted(date_dir.iterdir()):
            if not dur_dir.is_dir():
                continue
            snap_file = dur_dir / "snapshots.parquet"
            if snap_file.exists():
                df = pd.read_parquet(snap_file)
                n_markets = df["slug"].nunique()
                n_snaps = len(df)
                size_mb = snap_file.stat().st_size / 1024 / 1024
                time_range = ""
                if "ts" in df.columns and len(df) > 0:
                    t0 = datetime.fromtimestamp(df["ts"].min(), tz=timezone.utc).strftime("%H:%M")
                    t1 = datetime.fromtimestamp(df["ts"].max(), tz=timezone.utc).strftime("%H:%M")
                    time_range = f" ({t0}–{t1} UTC)"

                log.info(
                    f"    {dur_dir.name}: {n_snaps:,} snapshots, "
                    f"{n_markets} markets, {size_mb:.1f}MB{time_range}"
                )
                total_snapshots += n_snaps
                total_markets += n_markets

        # Resolved
        resolved_file = date_dir / "resolved.parquet"
        if resolved_file.exists():
            df_r = pd.read_parquet(resolved_file)
            log.info(f"    resolved: {len(df_r)} markets")

    log.info(f"\n  Total: {total_snapshots:,} snapshots across {total_markets} markets")
    log.info(f"{'='*60}\n")
