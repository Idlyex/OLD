"""Block 7: Shares Market Features — Polymarket-specific signals.

Features (16 total):
  1.  time_remaining_pct        — % of market lifetime remaining
  2.  time_remaining_min        — minutes until expiry
  3.  time_elapsed_min          — minutes since market opened
  4.  life_phase                — 0=early, 0.5=mid, 1=late (nonlinear)
  5.  distance_from_ptb_pct     — (sol_price - price_to_beat) / price_to_beat * 100
  6.  distance_from_ptb_norm    — normalized by volatility
  7.  up_implied_prob           — yes_price (implied probability SOL > PTB)
  8.  mispricing_score          — |implied_prob - model_prob| — edge detection
  9.  shares_momentum_30s       — UP price change over last 30s
  10. shares_momentum_1m        — UP price change over last 1m
  11. shares_momentum_3m        — UP price change over last 3m
  12. volume_imbalance          — (up_volume - down_volume) / total
  13. liquidity_score           — log(bid_volume + ask_volume) normalized
  14. spread_normalized         — spread / mid_price
  15. mean_reversion_strength   — tendency to revert near expiry
  16. arbitrage_score           — yes_price + no_price deviation from 1.0
"""

import numpy as np
from typing import Dict, Optional, List


def compute_shares_features(
    sol_price: float,
    price_to_beat: float,
    yes_price: float,
    no_price: float,
    time_remaining_ms: int,
    time_elapsed_ms: int,
    duration_minutes: int,
    best_bid: float = 0.0,
    best_ask: float = 0.0,
    bid_volume: float = 0.0,
    ask_volume: float = 0.0,
    spread: float = 0.0,
    up_volume: float = 0.0,
    down_volume: float = 0.0,
    # Historical yes_prices for momentum (most recent last)
    yes_price_history: Optional[List[float]] = None,
    # Model probability for mispricing (from CEX features)
    model_up_prob: Optional[float] = None,
    # Recent volatility
    sol_volatility: float = 0.003,
) -> Dict[str, float]:
    """Compute all 16 shares-market features.

    Args:
        sol_price: Current Solana CEX price
        price_to_beat: Chainlink price at market open
        yes_price: Current UP shares price (0-1)
        no_price: Current DOWN shares price (0-1)
        time_remaining_ms: Milliseconds until market resolves
        time_elapsed_ms: Milliseconds since market opened
        duration_minutes: Market lifetime (5, 15, 60)
        best_bid: Best bid on UP shares
        best_ask: Best ask on UP shares
        bid_volume: Total bid volume (shares)
        ask_volume: Total ask volume (shares)
        spread: Bid-ask spread
        up_volume: Trading volume of UP shares
        down_volume: Trading volume of DOWN shares
        yes_price_history: List of recent yes_prices for momentum calculation
        model_up_prob: ML model's probability estimate (for mispricing detection)
        sol_volatility: Recent SOL price volatility (for normalization)

    Returns:
        Dict of 16 feature name → value pairs
    """
    features = {}
    total_ms = time_remaining_ms + time_elapsed_ms
    duration_ms = duration_minutes * 60_000

    # ── 1-4. Time features ──
    features["time_remaining_pct"] = time_remaining_ms / max(total_ms, 1)
    features["time_remaining_min"] = time_remaining_ms / 60_000
    features["time_elapsed_min"] = time_elapsed_ms / 60_000

    # Life phase: nonlinear — accelerates near expiry
    elapsed_pct = time_elapsed_ms / max(total_ms, 1)
    features["life_phase"] = np.clip(elapsed_pct ** 1.5, 0, 1)

    # ── 5-6. Distance from PriceToBeat ──
    if price_to_beat and price_to_beat > 0:
        dist_pct = (sol_price - price_to_beat) / price_to_beat * 100
        features["distance_from_ptb_pct"] = dist_pct

        # Normalize by volatility * sqrt(time_remaining)
        # Higher vol or more time → distance is less significant
        time_factor = np.sqrt(max(time_remaining_ms / 60_000, 0.1))  # sqrt(minutes)
        vol_factor = max(sol_volatility, 1e-6) * 100  # convert to %
        features["distance_from_ptb_norm"] = dist_pct / max(vol_factor * time_factor, 0.01)
    else:
        features["distance_from_ptb_pct"] = 0.0
        features["distance_from_ptb_norm"] = 0.0

    # ── 7. Implied probability ──
    features["up_implied_prob"] = np.clip(yes_price, 0.01, 0.99)

    # ── 8. Mispricing score ──
    if model_up_prob is not None:
        features["mispricing_score"] = model_up_prob - yes_price
    else:
        # Estimate from distance: if SOL > PTB, real prob should be > 0.5
        # Simple logistic estimate based on distance and time
        if price_to_beat and price_to_beat > 0:
            z = features["distance_from_ptb_norm"]
            naive_prob = 1.0 / (1.0 + np.exp(-z * 2))
            features["mispricing_score"] = naive_prob - yes_price
        else:
            features["mispricing_score"] = 0.0

    # ── 9-11. Shares momentum ──
    if yes_price_history and len(yes_price_history) >= 2:
        current = yes_price_history[-1]
        # 30s momentum (assuming ~1 snapshot/sec)
        idx_30s = max(0, len(yes_price_history) - 30)
        features["shares_momentum_30s"] = current - yes_price_history[idx_30s]

        # 1m momentum
        idx_1m = max(0, len(yes_price_history) - 60)
        features["shares_momentum_1m"] = current - yes_price_history[idx_1m]

        # 3m momentum
        idx_3m = max(0, len(yes_price_history) - 180)
        features["shares_momentum_3m"] = current - yes_price_history[idx_3m]
    else:
        features["shares_momentum_30s"] = 0.0
        features["shares_momentum_1m"] = 0.0
        features["shares_momentum_3m"] = 0.0

    # ── 12. Volume imbalance ──
    total_vol = up_volume + down_volume
    features["volume_imbalance"] = (up_volume - down_volume) / max(total_vol, 1e-9)

    # ── 13. Liquidity score ──
    total_book = bid_volume + ask_volume
    features["liquidity_score"] = np.log1p(total_book) / 10.0  # normalized

    # ── 14. Spread normalized ──
    mid = (best_bid + best_ask) / 2 if best_bid and best_ask else yes_price
    features["spread_normalized"] = spread / max(mid, 0.01)

    # ── 15. Mean-reversion strength ──
    # Near expiry, shares prices converge to 0 or 1.
    # This measures how far from 0.5 AND how close to expiry.
    distance_from_half = abs(yes_price - 0.5)
    expiry_factor = 1.0 - features["time_remaining_pct"]  # 0 at start, 1 at expiry
    features["mean_reversion_strength"] = distance_from_half * expiry_factor

    # ── 16. Arbitrage score ──
    # In a perfect market, yes_price + no_price = 1.0
    # Deviation indicates arbitrage or liquidity issues
    features["arbitrage_score"] = (yes_price + no_price) - 1.0

    return features


def compute_shares_features_from_row(
    row: dict,
    sol_price: float,
    yes_price_history: Optional[List[float]] = None,
    model_up_prob: Optional[float] = None,
    sol_volatility: float = 0.003,
) -> Dict[str, float]:
    """Convenience: compute features from a market dict/row."""
    return compute_shares_features(
        sol_price=sol_price,
        price_to_beat=row.get("price_to_beat", 0),
        yes_price=row.get("yes_price", 0.5),
        no_price=row.get("no_price", 0.5),
        time_remaining_ms=row.get("time_remaining_ms", 0),
        time_elapsed_ms=row.get("time_elapsed_ms", 0),
        duration_minutes=row.get("duration_minutes", 15),
        best_bid=row.get("best_bid", 0),
        best_ask=row.get("best_ask", 0),
        bid_volume=row.get("bid_volume", 0),
        ask_volume=row.get("ask_volume", 0),
        spread=row.get("spread", 0),
        up_volume=row.get("up_volume", 0),
        down_volume=row.get("down_volume", 0),
        yes_price_history=yes_price_history,
        model_up_prob=model_up_prob,
        sol_volatility=sol_volatility,
    )


# Feature names for export
SHARES_FEATURE_NAMES = [
    "time_remaining_pct",
    "time_remaining_min",
    "time_elapsed_min",
    "life_phase",
    "distance_from_ptb_pct",
    "distance_from_ptb_norm",
    "up_implied_prob",
    "mispricing_score",
    "shares_momentum_30s",
    "shares_momentum_1m",
    "shares_momentum_3m",
    "volume_imbalance",
    "liquidity_score",
    "spread_normalized",
    "mean_reversion_strength",
    "arbitrage_score",
]
