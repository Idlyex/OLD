"""Block 3: Microstructure & Order Flow Features (22 features) — THE MOST IMPORTANT.

- Bid-Ask imbalance (top-5, top-10, top-20 levels)
- Cumulative Delta (aggressive orders)
- VPIN (Volume-synchronized Probability of Informed Trading)
- Order Flow Toxicity
- Aggressive Buy/Sell volume ratio (last 30s, 60s, 180s)
- Book pressure (weighted pressure)
- Spoofing detection (large cancelled orders)
- Iceberg detection
- Delta divergence (price vs delta)
- Microstructure breakouts
"""

import numpy as np
import math
from typing import Dict, List, Optional
from collections import deque

from core.utils.helpers import safe_div


class MicrostructureFeatures:
    """Order flow and microstructure feature computation."""

    def __init__(self):
        # VPIN state: rolling volume buckets
        self._vpin_buckets: Dict[str, deque] = {}
        self._vpin_bucket_size = 50  # trades per bucket

        # Spoofing detection: track order book changes
        self._prev_books: Dict[str, Dict] = {}

        # Iceberg detection: track trade sizes vs displayed book
        self._iceberg_scores: Dict[str, deque] = {}

    def compute(
        self,
        depth: Optional[Dict] = None,
        recent_trades: Optional[List[Dict]] = None,
        cvd_state: Optional[Dict] = None,
        current_price: float = 0.0,
        symbol: str = "SOLUSDT",
    ) -> Dict[str, float]:
        """Compute all 22 microstructure features."""
        features = {}

        # === Bid-Ask Imbalance at multiple levels ===
        if depth and depth.get("bids") and depth.get("asks"):
            bids = depth["bids"]
            asks = depth["asks"]

            for levels in [5, 10, 20]:
                bid_vol = sum(b["qty"] for b in bids[:levels])
                ask_vol = sum(a["qty"] for a in asks[:levels])
                total = bid_vol + ask_vol
                imb = safe_div(bid_vol - ask_vol, total)
                features[f"ob_imbalance_{levels}"] = imb

            # Book pressure (weighted by distance from mid)
            mid = (bids[0]["price"] + asks[0]["price"]) / 2 if bids and asks else current_price
            bid_pressure = sum(
                b["qty"] * (1.0 / max(mid - b["price"], 0.0001))
                for b in bids[:20] if b["price"] < mid
            )
            ask_pressure = sum(
                a["qty"] * (1.0 / max(a["price"] - mid, 0.0001))
                for a in asks[:20] if a["price"] > mid
            )
            total_pressure = bid_pressure + ask_pressure
            features["book_pressure"] = safe_div(bid_pressure - ask_pressure, total_pressure)

            # Spread
            spread = asks[0]["price"] - bids[0]["price"] if bids and asks else 0
            features["spread_bps"] = (spread / mid * 10000) if mid > 0 else 0

            # Depth ratio
            bid_depth_total = sum(b["qty"] for b in bids[:20])
            ask_depth_total = sum(a["qty"] for a in asks[:20])
            features["depth_ratio"] = safe_div(bid_depth_total, ask_depth_total)

        else:
            features["ob_imbalance_5"] = 0.0
            features["ob_imbalance_10"] = 0.0
            features["ob_imbalance_20"] = 0.0
            features["book_pressure"] = 0.0
            features["spread_bps"] = 0.0
            features["depth_ratio"] = 1.0

        # === Cumulative Delta ===
        if cvd_state:
            features["cvd_value"] = cvd_state.get("value", 0.0)
            # CVD slope (recent 60 entries)
            hist = list(cvd_state.get("history", []))
            if len(hist) >= 10:
                cvd_recent = [h["value"] for h in hist[-30:]]
                cvd_older = [h["value"] for h in hist[-60:-30]] if len(hist) >= 60 else cvd_recent[:1]
                features["cvd_slope"] = safe_div(
                    np.mean(cvd_recent) - np.mean(cvd_older),
                    abs(np.mean(cvd_older)) + 1
                )
            else:
                features["cvd_slope"] = 0.0
        else:
            features["cvd_value"] = 0.0
            features["cvd_slope"] = 0.0

        # === Aggressive Buy/Sell Volume Ratio ===
        if recent_trades:
            for window_s, label in [(30, "30s"), (60, "60s"), (180, "180s")]:
                now_ms = recent_trades[-1]["ts"] if recent_trades else 0
                cutoff = now_ms - window_s * 1000
                in_window = [t for t in recent_trades if t["ts"] >= cutoff]

                buy_vol = sum(t["qty"] for t in in_window if t.get("is_buy", False))
                sell_vol = sum(t["qty"] for t in in_window if not t.get("is_buy", True))
                total = buy_vol + sell_vol

                features[f"aggr_buy_ratio_{label}"] = safe_div(buy_vol, total)
                features[f"aggr_sell_ratio_{label}"] = safe_div(sell_vol, total)
        else:
            for label in ["30s", "60s", "180s"]:
                features[f"aggr_buy_ratio_{label}"] = 0.5
                features[f"aggr_sell_ratio_{label}"] = 0.5

        # === VPIN (Volume-synchronized Probability of Informed Trading) ===
        features["vpin"] = self._compute_vpin(symbol, recent_trades)

        # === Order Flow Toxicity ===
        features["flow_toxicity"] = self._compute_flow_toxicity(recent_trades)

        # === Spoofing Detection ===
        features["spoofing_score"] = self._detect_spoofing(symbol, depth)

        # === Iceberg Detection ===
        features["iceberg_score"] = self._detect_iceberg(symbol, depth, recent_trades)

        # === Delta Divergence (price vs CVD) ===
        features["delta_divergence"] = self._compute_delta_divergence(
            recent_trades, cvd_state, current_price
        )

        # === Microstructure Breakout ===
        features["micro_breakout"] = self._detect_micro_breakout(depth, recent_trades, current_price)

        return features

    def _compute_vpin(self, symbol: str, trades: Optional[List[Dict]]) -> float:
        """VPIN: Probability of informed trading based on volume buckets."""
        if not trades or len(trades) < self._vpin_bucket_size:
            return 0.5

        if symbol not in self._vpin_buckets:
            self._vpin_buckets[symbol] = deque(maxlen=50)

        # Create volume buckets
        buy_vol = 0.0
        sell_vol = 0.0
        bucket_count = 0

        for t in trades[-200:]:
            if t.get("is_buy", False):
                buy_vol += t["qty"]
            else:
                sell_vol += t["qty"]
            bucket_count += 1

            if bucket_count >= self._vpin_bucket_size:
                total = buy_vol + sell_vol
                if total > 0:
                    self._vpin_buckets[symbol].append(abs(buy_vol - sell_vol) / total)
                buy_vol = 0.0
                sell_vol = 0.0
                bucket_count = 0

        buckets = list(self._vpin_buckets[symbol])
        if not buckets:
            return 0.5

        return float(np.mean(buckets))

    def _compute_flow_toxicity(self, trades: Optional[List[Dict]]) -> float:
        """Order flow toxicity: ratio of informed vs uninformed trading."""
        if not trades or len(trades) < 20:
            return 0.0

        # Simple toxicity: high delta relative to total volume = toxic
        buy_vol = sum(t["qty"] for t in trades[-100:] if t.get("is_buy"))
        sell_vol = sum(t["qty"] for t in trades[-100:] if not t.get("is_buy", True))
        total = buy_vol + sell_vol
        if total == 0:
            return 0.0

        # Signed: positive = buy-side toxic, negative = sell-side toxic
        return (buy_vol - sell_vol) / total

    def _detect_spoofing(self, symbol: str, depth: Optional[Dict]) -> float:
        """Detect potential spoofing: large orders that appear/disappear."""
        if not depth:
            return 0.0

        prev = self._prev_books.get(symbol)
        self._prev_books[symbol] = depth

        if not prev:
            return 0.0

        # Compare bid side: large orders that vanished
        prev_bids = {b["price"]: b["qty"] for b in prev.get("bids", [])[:10]}
        curr_bids = {b["price"]: b["qty"] for b in depth.get("bids", [])[:10]}

        vanished_qty = 0.0
        for price, qty in prev_bids.items():
            if price not in curr_bids and qty > 100:  # large order vanished
                vanished_qty += qty

        prev_asks = {a["price"]: a["qty"] for a in prev.get("asks", [])[:10]}
        curr_asks = {a["price"]: a["qty"] for a in depth.get("asks", [])[:10]}

        for price, qty in prev_asks.items():
            if price not in curr_asks and qty > 100:
                vanished_qty += qty

        total_book = sum(b["qty"] for b in depth.get("bids", [])[:10]) + \
                     sum(a["qty"] for a in depth.get("asks", [])[:10])

        return safe_div(vanished_qty, total_book + 1)

    def _detect_iceberg(
        self, symbol: str, depth: Optional[Dict], trades: Optional[List[Dict]]
    ) -> float:
        """Detect iceberg orders: large fills at same price level despite small book display."""
        if not trades or not depth:
            return 0.0

        if symbol not in self._iceberg_scores:
            self._iceberg_scores[symbol] = deque(maxlen=100)

        # Check if trade size significantly exceeds displayed book size
        score = 0.0
        for t in trades[-20:]:
            price = t["price"]
            qty = t["qty"]

            # Find displayed size at this price in the book
            displayed = 0.0
            for b in depth.get("bids", []):
                if abs(b["price"] - price) < 0.01:
                    displayed = b["qty"]
                    break
            for a in depth.get("asks", []):
                if abs(a["price"] - price) < 0.01:
                    displayed = max(displayed, a["qty"])
                    break

            if displayed > 0 and qty > displayed * 2:
                score += (qty - displayed) / (displayed + 1)

        normalized = min(score / 10, 1.0)
        self._iceberg_scores[symbol].append(normalized)
        return normalized

    def _compute_delta_divergence(
        self, trades: Optional[List[Dict]], cvd_state: Optional[Dict],
        current_price: float
    ) -> float:
        """Detect divergence between price movement and CVD."""
        if not trades or len(trades) < 20 or not cvd_state:
            return 0.0

        hist = list(cvd_state.get("history", []))
        if len(hist) < 10:
            return 0.0

        # Price direction (last 30 trades)
        prices = [t["price"] for t in trades[-30:]]
        price_change = safe_div(prices[-1] - prices[0], prices[0])

        # CVD direction
        cvd_vals = [h["value"] for h in hist[-30:]]
        cvd_change = cvd_vals[-1] - cvd_vals[0]
        cvd_norm = safe_div(cvd_change, abs(cvd_vals[0]) + 1)

        # Divergence: price up but CVD down (or vice versa)
        if price_change > 0 and cvd_norm < 0:
            return -abs(price_change - cvd_norm)  # bearish divergence
        elif price_change < 0 and cvd_norm > 0:
            return abs(cvd_norm - price_change)  # bullish divergence

        return 0.0

    def _detect_micro_breakout(
        self, depth: Optional[Dict], trades: Optional[List[Dict]],
        current_price: float
    ) -> float:
        """Detect microstructure breakouts: aggressive sweep through book levels."""
        if not depth or not trades or len(trades) < 5:
            return 0.0

        # Check if recent trades swept multiple ask/bid levels
        asks = depth.get("asks", [])
        bids = depth.get("bids", [])

        recent_prices = [t["price"] for t in trades[-10:]]
        if not recent_prices:
            return 0.0

        price_range = max(recent_prices) - min(recent_prices)
        levels_swept = 0

        # Count ask levels swept
        for a in asks[:5]:
            if max(recent_prices) >= a["price"]:
                levels_swept += 1

        # Count bid levels swept (downside)
        for b in bids[:5]:
            if min(recent_prices) <= b["price"]:
                levels_swept -= 1

        return float(np.clip(levels_swept / 3, -1, 1))
