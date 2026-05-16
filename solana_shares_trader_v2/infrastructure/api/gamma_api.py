"""Gamma API — auto-discovery of active Solana prediction markets.
Port of POLYx gamma-api.js to async Python.
"""

import asyncio
import math
from typing import Optional, List, Dict
from datetime import datetime, timezone

import httpx
from core.utils.logger import log
from config import config

_cfg_poly = config.get("infrastructure", {}).get("polymarket", {})
GAMMA_HOST = _cfg_poly.get("gamma_host", "https://gamma-api.polymarket.com")


class GammaAPI:
    """Discovers and tracks active Solana prediction markets on Polymarket."""

    def __init__(self):
        self.host = GAMMA_HOST
        self._cache: Dict[str, dict] = {}
        self._cache_ttl_ms = config.get("timing", {}).get("market_refresh_ms", 12000)

    def _generate_slugs(self, base_slug: str) -> List[str]:
        """Generate timestamped slugs for current and upcoming windows."""
        import time

        now = int(time.time())
        match = None
        import re

        m = re.search(r"-(\d+)m$", base_slug)
        interval = (int(m.group(1)) if m else 15) * 60
        current_start = (now // interval) * interval

        slugs = []
        for i in range(-1, 5):
            ts = current_start + interval * i
            slugs.append(f"{base_slug}-{ts}")
        return slugs

    async def _fetch_event(self, full_slug: str) -> Optional[Dict]:
        """Fetch a single event by its full slug."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{self.host}/events",
                    params={"slug": full_slug, "limit": 1},
                    headers={"Accept": "application/json"},
                )
                if resp.status_code != 200:
                    return None

                events = resp.json()
                if not events or len(events) == 0:
                    return None

                event = events[0]
                markets = event.get("markets", [])
                if not markets:
                    return None

                market = markets[0]
                if market.get("closed") or not market.get("active"):
                    return None

                # Parse token IDs
                import json

                yes_token_id = None
                no_token_id = None
                clob_tokens = market.get("clobTokenIds", "")
                if clob_tokens:
                    try:
                        token_ids = json.loads(clob_tokens)
                        yes_token_id = token_ids[0] if len(token_ids) > 0 else None
                        no_token_id = token_ids[1] if len(token_ids) > 1 else None
                    except (json.JSONDecodeError, IndexError):
                        pass

                if not yes_token_id or not no_token_id:
                    return None

                # Parse prices
                yes_price = 0.5
                no_price = 0.5
                outcome_prices = market.get("outcomePrices", "")
                if outcome_prices:
                    try:
                        prices = json.loads(outcome_prices)
                        yes_price = float(prices[0])
                        no_price = float(prices[1])
                    except (json.JSONDecodeError, IndexError, ValueError):
                        pass

                # Resolution data
                price_to_beat = None
                final_price = None
                metadata = event.get("eventMetadata", {})
                if metadata:
                    if metadata.get("priceToBeat") is not None:
                        price_to_beat = float(metadata["priceToBeat"])
                    if metadata.get("finalPrice") is not None:
                        final_price = float(metadata["finalPrice"])

                end_date_str = (
                    market.get("endDate")
                    or market.get("endDateIso")
                    or event.get("endDate")
                )
                event_start_time = (
                    event.get("startTime")
                    or market.get("eventStartTime")
                )

                now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
                end_ms = 0
                if end_date_str:
                    end_ms = int(
                        datetime.fromisoformat(
                            end_date_str.replace("Z", "+00:00")
                        ).timestamp()
                        * 1000
                    )
                time_remaining_ms = max(0, end_ms - now_ms)

                start_ms = 0
                if event_start_time:
                    try:
                        start_ms = int(
                            datetime.fromisoformat(
                                event_start_time.replace("Z", "+00:00")
                            ).timestamp()
                            * 1000
                        )
                    except (ValueError, TypeError):
                        pass
                time_elapsed_ms = max(0, now_ms - start_ms) if start_ms else 0

                return {
                    "condition_id": market.get("conditionId"),
                    "slug": full_slug,
                    "base_slug": full_slug.rsplit("-", 1)[0]
                    if full_slug.count("-") > 2
                    else full_slug,
                    "question": market.get("question", event.get("title", "")),
                    "yes_token_id": yes_token_id,
                    "no_token_id": no_token_id,
                    "yes_price": yes_price,
                    "no_price": no_price,
                    "end_date": end_date_str,
                    "event_start_time": event_start_time,
                    "neg_risk": bool(market.get("negRisk")),
                    "accepting_orders": market.get("acceptingOrders", True),
                    "price_to_beat": price_to_beat,
                    "final_price": final_price,
                    "time_remaining_ms": time_remaining_ms,
                    "time_elapsed_ms": time_elapsed_ms,
                    "time_remaining_pct": (
                        time_remaining_ms / (time_remaining_ms + time_elapsed_ms)
                        if (time_remaining_ms + time_elapsed_ms) > 0
                        else 1.0
                    ),
                }
        except Exception as e:
            log.debug(f"Gamma fetch error for {full_slug}: {e}")
            return None

    async def get_active_markets(self, base_slug: str) -> List[Dict]:
        """Fetch all active markets for a base slug."""
        import time

        cache_key = base_slug
        cached = self._cache.get(cache_key)
        now_ms = int(time.time() * 1000)
        if cached and (now_ms - cached["ts"]) < self._cache_ttl_ms:
            return cached["data"]

        slugs = self._generate_slugs(base_slug)
        tasks = [self._fetch_event(s) for s in slugs]
        results = await asyncio.gather(*tasks)
        markets = [r for r in results if r is not None]
        markets.sort(key=lambda m: m.get("end_date", ""))

        self._cache[cache_key] = {"data": markets, "ts": now_ms}
        if markets:
            log.debug(f"Gamma: found {len(markets)} active markets for {base_slug}")
        return markets

    async def get_nearest_market(self, base_slug: str) -> Optional[Dict]:
        """Get the nearest tradeable market."""
        import re
        import time

        markets = await self.get_active_markets(base_slug)
        now_ms = int(time.time() * 1000)

        m_match = re.search(r"-(\d+)m", base_slug)
        duration_ms = (int(m_match.group(1)) if m_match else 15) * 60_000

        for mkt in markets:
            end_str = mkt.get("end_date", "")
            if not end_str:
                continue
            end_ms = int(
                datetime.fromisoformat(end_str.replace("Z", "+00:00")).timestamp()
                * 1000
            )
            closes_in = end_ms - now_ms
            if 30_000 < closes_in <= duration_ms and mkt.get("accepting_orders", True):
                return mkt
        return None

    async def get_all_nearest_markets(self) -> List[Dict]:
        """Get nearest markets for all configured slugs."""
        slugs = config.get("markets", {}).get("slugs", [])
        tasks = [self.get_nearest_market(s) for s in slugs]
        results = await asyncio.gather(*tasks)
        return [
            {"slug": slugs[i], "market": r}
            for i, r in enumerate(results)
            if r is not None
        ]

    def clear_cache(self):
        self._cache.clear()
