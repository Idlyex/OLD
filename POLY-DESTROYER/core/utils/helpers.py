"""Shared utilities: retry, timing, math helpers."""

import asyncio
import time
import math
import numpy as np
from typing import Callable, Any, Optional
from functools import wraps


async def retry_async(
    fn: Callable,
    retries: int = 3,
    base_delay: float = 0.3,
    max_delay: float = 5.0,
    exceptions: tuple = (Exception,),
) -> Any:
    """Async retry with exponential backoff."""
    last_err = None
    for attempt in range(retries + 1):
        try:
            return await fn()
        except exceptions as e:
            last_err = e
            if attempt < retries:
                delay = min(base_delay * (2 ** attempt), max_delay)
                await asyncio.sleep(delay)
    raise last_err


def retry_sync(
    fn: Callable,
    retries: int = 3,
    base_delay: float = 0.3,
) -> Any:
    """Sync retry with exponential backoff."""
    last_err = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(base_delay * (2 ** attempt))
    raise last_err


class TimingContext:
    """Context manager for measuring execution time."""

    def __init__(self, label: str = ""):
        self.label = label
        self.elapsed_ms = 0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    """Safe division avoiding ZeroDivisionError."""
    return a / b if b != 0 else default


def log_return(price_now: float, price_prev: float) -> float:
    """Compute log return."""
    if price_prev <= 0 or price_now <= 0:
        return 0.0
    return math.log(price_now / price_prev)


def simple_return(price_now: float, price_prev: float) -> float:
    """Compute simple return."""
    if price_prev <= 0:
        return 0.0
    return (price_now - price_prev) / price_prev


def ema(values: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average."""
    alpha = 2.0 / (span + 1)
    result = np.empty_like(values, dtype=np.float64)
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
    return result


def sma(values: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average with NaN padding."""
    result = np.full_like(values, np.nan, dtype=np.float64)
    if len(values) < window:
        return result
    cumsum = np.cumsum(values)
    result[window - 1:] = (cumsum[window - 1:] - np.concatenate([[0], cumsum[:-window]])) / window
    return result


def garman_klass_vol(
    high: np.ndarray, low: np.ndarray, open_: np.ndarray, close: np.ndarray
) -> float:
    """Garman-Klass volatility estimator."""
    n = len(high)
    if n == 0:
        return 0.0
    u = np.log(high / open_)
    d = np.log(low / open_)
    c = np.log(close / open_)
    return float(np.sqrt(np.mean(0.5 * (u - d) ** 2 - (2 * math.log(2) - 1) * c ** 2)))


def parkinson_vol(high: np.ndarray, low: np.ndarray) -> float:
    """Parkinson volatility estimator."""
    n = len(high)
    if n == 0:
        return 0.0
    hl = np.log(high / low)
    return float(np.sqrt(np.mean(hl ** 2) / (4 * math.log(2))))


def rogers_satchell_vol(
    high: np.ndarray, low: np.ndarray, open_: np.ndarray, close: np.ndarray
) -> float:
    """Rogers-Satchell volatility estimator."""
    n = len(high)
    if n == 0:
        return 0.0
    hc = np.log(high / close)
    ho = np.log(high / open_)
    lc = np.log(low / close)
    lo = np.log(low / open_)
    return float(np.sqrt(np.mean(hc * ho + lc * lo)))


def weighted_mid_price(bid: float, ask: float, bid_qty: float, ask_qty: float) -> float:
    """Volume-weighted mid price."""
    total = bid_qty + ask_qty
    if total <= 0:
        return (bid + ask) / 2 if (bid + ask) > 0 else 0.0
    return (bid * ask_qty + ask * bid_qty) / total


def kelly_fraction(win_prob: float, win_loss_ratio: float) -> float:
    """Kelly criterion for optimal position sizing."""
    if win_prob <= 0 or win_loss_ratio <= 0:
        return 0.0
    return max(0.0, win_prob - (1 - win_prob) / win_loss_ratio)


def clip(value: float, lo: float, hi: float) -> float:
    """Clip value to [lo, hi]."""
    return max(lo, min(hi, value))


def ts_now_ms() -> int:
    """Current timestamp in milliseconds."""
    return int(time.time() * 1000)


def format_pnl(pnl_pct: float) -> str:
    """Format PnL with color hint."""
    sign = "+" if pnl_pct >= 0 else ""
    return f"{sign}{pnl_pct:.2f}%"
