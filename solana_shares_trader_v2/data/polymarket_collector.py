"""Polymarket Data Collector — async market discovery + historical shares data.

Ported from POLYx JS (gamma-api.js, clob-client.js, ws-manager.js) to Python async.

Provides:
  - Active market discovery via Gamma API (sol-updown-5m, sol-updown-15m)
  - Market metadata: PriceToBeat, token IDs, time-to-expiry, prices
  - Historical resolved markets (past shares price curves)
  - CLOB orderbook snapshots
  - WebSocket real-time prices

Usage:
  collector = PolymarketCollector()
  markets = await collector.get_active_markets("sol-updown-15m")
  history = await collector.download_historical_markets("sol-updown-15m", days=30)
"""

import asyncio
import time
import math
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
import httpx

from core.utils.logger import log
from config import config, trading_config

_pm_cfg = config.get("infrastructure", {}).get("polymarket", {})

GAMMA_HOST = _pm_cfg.get("gamma_host", "https://gamma-api.polymarket.com")
CLOB_HOST = _pm_cfg.get("clob_host", "https://clob.polymarket.com")
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


# ═══════════════════════════════════════════════════════════
#  DATA CLASSES
# ═══════════════════════════════════════════════════════════

@dataclass
class SharesMarket:
    """Single prediction market snapshot."""
    slug: str = ""
    condition_id: str = ""
    question: str = ""
    yes_token_id: str = ""
    no_token_id: str = ""
    yes_price: float = 0.5
    no_price: float = 0.5
    price_to_beat: Optional[float] = None
    final_price: Optional[float] = None
    event_start_time: Optional[str] = None
    end_date: Optional[str] = None
    duration_minutes: int = 15
    time_remaining_ms: int = 0
    time_elapsed_ms: int = 0
    time_remaining_pct: float = 1.0
    accepting_orders: bool = True
    best_bid: float = 0.0      # YES best bid
    best_ask: float = 0.0      # YES best ask (what you PAY to buy YES)
    spread: float = 0.0        # YES spread
    no_best_bid: float = 0.0   # NO best bid
    no_best_ask: float = 0.0   # NO best ask (what you PAY to buy NO)
    no_spread: float = 0.0     # NO spread
    yes_depth: float = 0.0     # ask volume available (YES side)
    no_depth: float = 0.0      # ask volume available (NO side)
    last_trade_price: float = 0.0
    neg_risk: bool = False
    volume: float = 0.0
    liquidity: float = 0.0
    resolved: bool = False
    outcome: Optional[str] = None  # "Up" or "Down" after resolution
    ts: int = 0

    @property
    def is_tradeable(self) -> bool:
        return (self.time_remaining_ms > 30_000 and
                self.accepting_orders and
                self.yes_token_id and self.no_token_id)

    @property
    def implied_up_prob(self) -> float:
        return self.yes_price

    @property
    def implied_down_prob(self) -> float:
        return self.no_price

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MarketTimeline:
    """Full lifecycle of a resolved market — shares prices over time."""
    slug: str
    price_to_beat: float
    final_price: float
    outcome: str  # "Up" or "Down"
    duration_minutes: int
    start_ts: int
    end_ts: int
    # Time series: list of {ts, yes_price, no_price, sol_price, time_remaining_pct}
    snapshots: List[Dict] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════
#  GAMMA API CLIENT
# ═══════════════════════════════════════════════════════════

class PolymarketCollector:
    """Async Polymarket data collector — markets, orderbooks, history."""

    def __init__(self):
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        self._http: Optional[httpx.AsyncClient] = None
        self._cache: Dict[str, Tuple[float, Any]] = {}
        self._cache_ttl = 10.0  # seconds

    async def _get_http(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(
                timeout=15.0,
                limits=httpx.Limits(max_connections=10),
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
            log.debug(f"Polymarket request error: {e}")
            return None

    # ─── Slug Generation ─────────────────────────────────────

    @staticmethod
    def generate_slugs(base_slug: str, count: int = 6) -> List[str]:
        """Generate timestamped slugs for current + upcoming market windows.

        Slug pattern: {base_slug}-{UNIX_TIMESTAMP}
        Timestamp = start of the interval window, aligned to interval boundaries.
        """
        now = int(time.time())

        # Parse duration from slug: sol-updown-15m → 15, sol-updown-5m → 5
        import re
        m = re.search(r'-(\d+)m$', base_slug)
        interval_sec = (int(m.group(1)) if m else 15) * 60

        current_start = (now // interval_sec) * interval_sec
        slugs = []
        for i in range(-1, count):
            ts = current_start + interval_sec * i
            slugs.append(f"{base_slug}-{ts}")
        return slugs

    # ─── Single Event Fetch ──────────────────────────────────

    async def _fetch_event(self, full_slug: str) -> Optional[SharesMarket]:
        """Fetch a single event by its timestamped slug."""
        url = f"{GAMMA_HOST}/events"
        data = await self._get(url, {"slug": full_slug, "limit": 1})
        if not data or len(data) == 0:
            return None

        event = data[0]
        markets = event.get("markets", [])
        if not markets:
            return None

        market = markets[0]
        if market.get("closed") or not market.get("active", True):
            return None

        # Parse token IDs
        yes_token = no_token = None
        try:
            tokens = json.loads(market.get("clobTokenIds", "[]"))
            yes_token = tokens[0] if len(tokens) > 0 else None
            no_token = tokens[1] if len(tokens) > 1 else None
        except (json.JSONDecodeError, IndexError):
            pass

        if not yes_token or not no_token:
            return None

        # Outcome prices
        yes_price = no_price = 0.5
        try:
            prices = json.loads(market.get("outcomePrices", "[]"))
            yes_price = float(prices[0]) if len(prices) > 0 else 0.5
            no_price = float(prices[1]) if len(prices) > 1 else 0.5
        except (json.JSONDecodeError, IndexError, ValueError):
            pass

        # Price to beat from event metadata
        ptb = None
        final_price = None
        meta = event.get("eventMetadata", {})
        if meta:
            if meta.get("priceToBeat") is not None:
                ptb = float(meta["priceToBeat"])
            if meta.get("finalPrice") is not None:
                final_price = float(meta["finalPrice"])

        # Timing
        event_start = event.get("startTime") or market.get("eventStartTime")
        end_date = market.get("endDate") or market.get("endDateIso") or event.get("endDate")

        now_ms = int(time.time() * 1000)
        time_remaining_ms = 0
        time_elapsed_ms = 0
        if end_date:
            try:
                end_ms = int(datetime.fromisoformat(end_date.replace("Z", "+00:00")).timestamp() * 1000)
                time_remaining_ms = max(0, end_ms - now_ms)
            except (ValueError, TypeError):
                pass
        if event_start:
            try:
                start_ms = int(datetime.fromisoformat(str(event_start).replace("Z", "+00:00")).timestamp() * 1000)
                time_elapsed_ms = max(0, now_ms - start_ms)
            except (ValueError, TypeError):
                pass

        # Duration from slug
        import re
        m = re.search(r'-(\d+)m', full_slug)
        duration_min = int(m.group(1)) if m else 15

        # If end_date missing, try multiple fallbacks
        if not end_date:
            # 1) Extract epoch from slug (e.g. sol-updown-5m-1778218500)
            #    Slug epoch = START time. End = start + duration.
            ts_match = re.search(r'-(\d{10})$', full_slug)
            if ts_match:
                start_epoch = int(ts_match.group(1))
                end_ts_epoch = start_epoch + duration_min * 60
                end_dt = datetime.fromtimestamp(end_ts_epoch, tz=timezone.utc)
                end_date = end_dt.isoformat()
                time_remaining_ms = max(0, end_ts_epoch * 1000 - now_ms)
            # 2) Compute from event_start + duration
            elif event_start:
                try:
                    start_dt = datetime.fromisoformat(str(event_start).replace("Z", "+00:00"))
                    end_dt = start_dt + timedelta(minutes=duration_min)
                    end_date = end_dt.isoformat()
                    time_remaining_ms = max(0, int(end_dt.timestamp() * 1000) - now_ms)
                except (ValueError, TypeError):
                    pass
            # 3) Last resort: derive from time_remaining_ms
            elif time_remaining_ms > 0:
                end_dt = datetime.fromtimestamp((now_ms + time_remaining_ms) / 1000, tz=timezone.utc)
                end_date = end_dt.isoformat()

        total_ms = time_remaining_ms + time_elapsed_ms
        time_remaining_pct = time_remaining_ms / total_ms if total_ms > 0 else 1.0

        return SharesMarket(
            slug=full_slug,
            condition_id=market.get("conditionId", ""),
            question=market.get("question", event.get("title", "")),
            yes_token_id=yes_token,
            no_token_id=no_token,
            yes_price=yes_price,
            no_price=no_price,
            price_to_beat=ptb,
            final_price=final_price,
            event_start_time=str(event_start) if event_start else None,
            end_date=str(end_date) if end_date else None,
            duration_minutes=duration_min,
            time_remaining_ms=time_remaining_ms,
            time_elapsed_ms=time_elapsed_ms,
            time_remaining_pct=time_remaining_pct,
            accepting_orders=market.get("acceptingOrders", True),
            best_bid=float(market.get("bestBid") or 0),
            best_ask=float(market.get("bestAsk") or 0),
            spread=float(market.get("spread") or 0),
            last_trade_price=float(market.get("lastTradePrice") or 0),
            neg_risk=bool(market.get("negRisk")),
            volume=float(market.get("volume") or 0),
            liquidity=float(market.get("liquidity") or 0),
            ts=now_ms,
        )

    # ─── Active Markets ──────────────────────────────────────

    async def get_active_markets(self, base_slug: str = "sol-updown-15m") -> List[SharesMarket]:
        """Fetch all active markets for a base slug."""
        cache_key = f"active_{base_slug}"
        if cache_key in self._cache:
            cached_ts, cached_data = self._cache[cache_key]
            if time.time() - cached_ts < self._cache_ttl:
                return cached_data

        slugs = self.generate_slugs(base_slug)
        tasks = [self._fetch_event(slug) for slug in slugs]
        results = await asyncio.gather(*tasks)
        markets = [m for m in results if m is not None]

        # Sort by end date (nearest first)
        markets.sort(key=lambda m: m.time_remaining_ms)

        self._cache[cache_key] = (time.time(), markets)
        return markets

    async def get_nearest_market(self, base_slug: str = "sol-updown-15m") -> Optional[SharesMarket]:
        """Get the nearest tradeable market."""
        markets = await self.get_active_markets(base_slug)
        for m in markets:
            if m.is_tradeable and m.time_remaining_ms <= m.duration_minutes * 60_000:
                return m
        return None

    async def get_all_nearest_markets(self) -> List[SharesMarket]:
        """Get nearest markets for all configured slugs."""
        _m = trading_config.get("markets", {})
        slugs = _m.get("record_slugs", _m.get("trade_slugs", ["sol-updown-5m"]))
        tasks = [self.get_nearest_market(slug) for slug in slugs]
        results = await asyncio.gather(*tasks)
        return [m for m in results if m is not None]

    # ─── CLOB Orderbook ──────────────────────────────────────

    async def get_orderbook(self, token_id: str) -> Optional[Dict]:
        """Fetch orderbook for a token from CLOB."""
        url = f"{CLOB_HOST}/book"
        data = await self._get(url, {"token_id": token_id})
        if not data:
            return None

        bids = data.get("bids", [])
        asks = data.get("asks", [])

        # PM book may NOT be sorted — find actual best bid (max) and best ask (min)
        parsed_bids = [float(b.get("price", 0)) for b in bids]
        parsed_asks = [float(a.get("price", 0)) for a in asks if float(a.get("price", 0)) > 0]
        best_bid = max(parsed_bids) if parsed_bids else 0
        best_ask = min(parsed_asks) if parsed_asks else 1

        # Sort for volume calc: bids desc, asks asc
        sorted_bids = sorted(bids, key=lambda b: float(b.get("price", 0)), reverse=True)
        sorted_asks = sorted(asks, key=lambda a: float(a.get("price", 0)))
        bid_volume = sum(float(b.get("size", 0)) for b in sorted_bids[:10])
        ask_volume = sum(float(a.get("size", 0)) for a in sorted_asks[:10])

        mid = (best_bid + best_ask) / 2 if best_bid and best_ask else 0.5
        spread = best_ask - best_bid if best_bid and best_ask else 0

        return {
            "bids": bids,
            "asks": asks,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid": mid,
            "spread": spread,
            "bid_volume": bid_volume,
            "ask_volume": ask_volume,
            "imbalance": (bid_volume - ask_volume) / max(bid_volume + ask_volume, 1e-9),
        }

    # ─── Historical Resolved Markets ─────────────────────────

    async def download_historical_markets(
        self,
        base_slug: str = "sol-updown-15m",
        days: int = 30,
    ) -> pd.DataFrame:
        """Download resolved markets from the past N days.

        For each resolved market, fetches: slug, priceToBeat, outcome, yes/no prices.
        This is the TRAINING DATA for the shares model.
        """
        import re
        m = re.search(r'-(\d+)m$', base_slug)
        interval_min = int(m.group(1)) if m else 15
        interval_sec = interval_min * 60

        log.info(f"📥 Downloading {base_slug} historical markets — {days} days ({interval_min}m intervals)")

        now = int(time.time())
        start_ts = now - days * 86400
        # Align to interval boundary
        start_ts = (start_ts // interval_sec) * interval_sec

        # Generate all slugs for the date range
        all_markets = []
        current_ts = start_ts
        total_slugs = (now - start_ts) // interval_sec
        fetched = 0
        errors = 0

        # Batch fetch — 10 concurrent requests
        batch_size = 10
        slug_list = []
        while current_ts < now:
            slug_list.append(f"{base_slug}-{current_ts}")
            current_ts += interval_sec

        log.info(f"  Total slugs to fetch: {len(slug_list)}")

        for i in range(0, len(slug_list), batch_size):
            batch = slug_list[i:i + batch_size]
            tasks = [self._fetch_resolved_event(slug) for slug in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for slug, result in zip(batch, results):
                if isinstance(result, Exception):
                    errors += 1
                    continue
                if result is not None:
                    all_markets.append(result)
                    fetched += 1

            if (i // batch_size) % 20 == 0 and i > 0:
                log.info(f"    Progress: {i}/{len(slug_list)} slugs, {fetched} markets found")

            await asyncio.sleep(0.05)  # gentle rate limit

        log.info(f"  Fetched {fetched} resolved markets ({errors} errors)")

        if not all_markets:
            return pd.DataFrame()

        df = pd.DataFrame(all_markets)
        df = df.sort_values("start_ts").reset_index(drop=True)

        # Save
        fname = f"{base_slug.replace('-', '_')}_history.parquet"
        path = RAW_DIR / fname
        df.to_parquet(path, engine="pyarrow", compression="snappy")
        log.info(f"  ✅ Saved {len(df)} markets → {path}")

        return df

    async def _fetch_resolved_event(self, full_slug: str) -> Optional[Dict]:
        """Fetch a resolved event and extract key data."""
        url = f"{GAMMA_HOST}/events"
        data = await self._get(url, {"slug": full_slug, "limit": 1})
        if not data or len(data) == 0:
            return None

        event = data[0]
        markets = event.get("markets", [])
        if not markets:
            return None

        market = markets[0]
        meta = event.get("eventMetadata", {}) or {}

        ptb = float(meta["priceToBeat"]) if meta.get("priceToBeat") is not None else None
        final_price = float(meta["finalPrice"]) if meta.get("finalPrice") is not None else None

        if ptb is None:
            return None

        # Parse outcome prices
        yes_price = no_price = 0.5
        try:
            prices = json.loads(market.get("outcomePrices", "[]"))
            yes_price = float(prices[0]) if len(prices) > 0 else 0.5
            no_price = float(prices[1]) if len(prices) > 1 else 0.5
        except (json.JSONDecodeError, IndexError, ValueError):
            pass

        # Outcome
        resolved = market.get("closed", False)
        outcome = None
        if final_price is not None and ptb is not None:
            outcome = "Up" if final_price >= ptb else "Down"

        # Timing
        event_start = event.get("startTime") or market.get("eventStartTime")
        end_date = market.get("endDate") or event.get("endDate")

        start_ts = end_ts = 0
        try:
            if event_start:
                start_ts = int(datetime.fromisoformat(str(event_start).replace("Z", "+00:00")).timestamp())
        except (ValueError, TypeError):
            pass
        try:
            if end_date:
                end_ts = int(datetime.fromisoformat(str(end_date).replace("Z", "+00:00")).timestamp())
        except (ValueError, TypeError):
            pass

        import re
        m_dur = re.search(r'-(\d+)m', full_slug)
        duration_min = int(m_dur.group(1)) if m_dur else 15

        return {
            "slug": full_slug,
            "price_to_beat": ptb,
            "final_price": final_price,
            "outcome": outcome,
            "resolved": resolved,
            "yes_price": yes_price,
            "no_price": no_price,
            "duration_minutes": duration_min,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "volume": float(market.get("volume") or 0),
            "liquidity": float(market.get("liquidity") or 0),
            "spread": float(market.get("spread") or 0),
            "best_bid": float(market.get("bestBid") or 0),
            "best_ask": float(market.get("bestAsk") or 0),
        }

    # ─── Combined Download ───────────────────────────────────

    async def download_all(
        self,
        days: int = 30,
        slugs: List[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Download all Polymarket historical data + current state.

        Returns dict of DataFrames keyed by slug.
        """
        if slugs is None:
            _m = trading_config.get("markets", {})
            slugs = _m.get("record_slugs", _m.get("trade_slugs", ["sol-updown-5m"]))

        log.info(f"╔══════════════════════════════════════════════════╗")
        log.info(f"║  Polymarket Download │ {days} days │ {len(slugs)} slugs    ║")
        log.info(f"╚══════════════════════════════════════════════════╝")

        results = {}
        for slug in slugs:
            try:
                df = await self.download_historical_markets(slug, days)
                results[slug] = df
            except Exception as e:
                log.error(f"Failed to download {slug}: {e}")

        # Also save current active markets
        for slug in slugs:
            try:
                active = await self.get_active_markets(slug)
                if active:
                    active_df = pd.DataFrame([m.to_dict() for m in active])
                    fname = f"{slug.replace('-', '_')}_active.parquet"
                    path = RAW_DIR / fname
                    active_df.to_parquet(path, engine="pyarrow", compression="snappy")
                    log.info(f"  ✅ Saved {len(active)} active markets → {path}")
            except Exception as e:
                log.warning(f"Active markets fetch failed for {slug}: {e}")

        return results


# ═══════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════

def load_historical_markets(slug: str = "sol-updown-15m") -> pd.DataFrame:
    """Load previously downloaded historical markets."""
    fname = f"{slug.replace('-', '_')}_history.parquet"
    path = RAW_DIR / fname
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def load_active_markets(slug: str = "sol-updown-15m") -> pd.DataFrame:
    """Load saved active market snapshots."""
    fname = f"{slug.replace('-', '_')}_active.parquet"
    path = RAW_DIR / fname
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()
