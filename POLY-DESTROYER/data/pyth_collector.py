"""Pyth Hermes Historical Data Collector.

Downloads SOL/USD OHLCV data from Pyth Network benchmarks API.
Pyth is the oracle used by Polymarket (via Chainlink) for price resolution.
Polymarket prices differ from Binance by ~2-3 cents — training on Pyth
data should improve entry-bar predictions.

Output: data/processed/SOLUSD_pyth_processed.parquet
  - Same column format as Binance processed data
  - Volume approximated (Pyth is oracle, not exchange)
  - Funding/OI/LS merged from Binance supplementary data

Usage:
    from data.pyth_collector import PythCollector
    collector = PythCollector()
    await collector.download_all(days=30)
"""

import asyncio
import time
import numpy as np
import pandas as pd
import httpx
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional

from core.utils.logger import log

BENCHMARKS_URL = "https://benchmarks.pyth.network/v1/shims/tradingview/history"
PYTH_SOL_SYMBOL = "Crypto.SOL/USD"
PROCESSED_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")


class PythCollector:
    """Download historical SOL/USD data from Pyth Network."""

    def __init__(self, symbol: str = "SOL/USD"):
        self.symbol = symbol
        self._http: Optional[httpx.AsyncClient] = None

    async def _get_http(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(timeout=30.0)
        return self._http

    async def close(self):
        if self._http and not self._http.is_closed:
            await self._http.aclose()

    async def download_ohlcv(self, days: int = 30, resolution: int = 1) -> pd.DataFrame:
        """Download OHLCV data from Pyth benchmarks API.

        Args:
            days: Number of days of history
            resolution: Candle resolution in minutes (1, 5, 15, 60)

        Returns:
            DataFrame with columns: ts, open, high, low, close, volume
        """
        http = await self._get_http()
        end_ts = int(time.time())
        start_ts = end_ts - days * 86400
        expected_bars = days * 24 * 60 // resolution

        log.info(f"📥 Downloading Pyth {PYTH_SOL_SYMBOL} {resolution}m — {days} days")
        log.info(f"  Expected ~{expected_bars:,} candles")

        all_bars = []
        chunk_size = 5000  # max bars per request
        chunk_seconds = chunk_size * resolution * 60
        cursor = start_ts

        while cursor < end_ts:
            chunk_end = min(cursor + chunk_seconds, end_ts)
            try:
                r = await http.get(
                    BENCHMARKS_URL,
                    params={
                        "symbol": PYTH_SOL_SYMBOL,
                        "resolution": str(resolution),
                        "from": str(cursor),
                        "to": str(chunk_end),
                    },
                )
                data = r.json()

                if data.get("s") != "ok" or "t" not in data:
                    log.warning(f"  Pyth API returned: {data.get('s', 'unknown')} — {data.get('errmsg', '')}")
                    # Try smaller chunk
                    chunk_seconds = max(chunk_seconds // 2, 3600)
                    await asyncio.sleep(1.0)
                    continue

                timestamps = data["t"]
                opens = data["o"]
                highs = data["h"]
                lows = data["l"]
                closes = data["c"]
                volumes = data.get("v", [0] * len(timestamps))

                for i in range(len(timestamps)):
                    all_bars.append({
                        "ts": timestamps[i] * 1000,  # convert to ms (match Binance format)
                        "open": opens[i],
                        "high": highs[i],
                        "low": lows[i],
                        "close": closes[i],
                        "volume": volumes[i] if i < len(volumes) else 0,
                    })

                log.info(f"  Fetched {len(timestamps)} bars [{datetime.fromtimestamp(cursor).strftime('%m-%d %H:%M')} → {datetime.fromtimestamp(chunk_end).strftime('%m-%d %H:%M')}]")

            except Exception as e:
                log.warning(f"  Pyth fetch error: {e}, retrying...")
                await asyncio.sleep(2.0)
                continue

            cursor = chunk_end
            await asyncio.sleep(0.3)  # rate limit

        if not all_bars:
            log.error("  No data received from Pyth")
            return pd.DataFrame()

        df = pd.DataFrame(all_bars)
        df = df.drop_duplicates(subset="ts").sort_values("ts").reset_index(drop=True)

        # Save raw
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        raw_path = RAW_DIR / "SOLUSD_pyth_1m_klines.parquet"
        df.to_parquet(raw_path, engine="pyarrow")
        log.info(f"  ✅ Saved {len(df):,} candles → {raw_path}")

        return df

    async def download_all(self, days: int = 30) -> Optional[pd.DataFrame]:
        """Download Pyth OHLCV + merge with Binance supplementary data.

        Creates a processed parquet matching Binance format for pipeline compatibility.
        """
        log.info("╔══════════════════════════════════════════════════╗")
        log.info(f"║  Pyth Hermes Data Download: SOL/USD │ {days}d     ║")
        log.info("╚══════════════════════════════════════════════════╝")

        t0 = time.time()

        # Step 1: Download Pyth OHLCV
        klines = await self.download_ohlcv(days=days, resolution=1)
        if klines.empty:
            return None

        # Step 2: Process into pipeline format
        df = klines.copy()
        df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df.set_index("datetime")

        # Approximate taker_buy_volume (Pyth doesn't have this; use volume/2)
        df["taker_buy_volume"] = df["volume"] / 2.0
        df["quote_volume"] = df["volume"] * df["close"]
        df["taker_buy_quote_volume"] = df["quote_volume"] / 2.0
        df["trades"] = 0  # not available from Pyth
        df["close_time"] = df["ts"] + 60000  # 1m candle close time

        # Step 3: Try to merge Binance supplementary data (funding, OI, LS)
        funding_rate = 0.0
        oi = 0.0
        ls_ratio = 1.0

        binance_supplementary = [
            ("SOLUSDT_funding_rate.parquet", "funding_rate"),
            ("SOLUSDT_open_interest_1h.parquet", "oi"),
            ("SOLUSDT_long_short_ratio.parquet", "ls"),
        ]

        for fname, dtype in binance_supplementary:
            fpath = RAW_DIR / fname
            if fpath.exists():
                try:
                    sup = pd.read_parquet(fpath)
                    sup["datetime"] = pd.to_datetime(sup["ts"], unit="ms", utc=True)
                    sup = sup.set_index("datetime")
                    sup = sup[~sup.index.duplicated(keep="last")]

                    if dtype == "funding_rate" and "funding_rate" in sup.columns:
                        df = df.join(sup[["funding_rate"]], how="left")
                        df["funding_rate"] = df["funding_rate"].ffill().fillna(0)
                        log.info(f"  ✅ Merged Binance funding rate from {fname}")
                    elif dtype == "oi":
                        oi_cols = [c for c in sup.columns if "open_interest" in c.lower()]
                        if oi_cols:
                            df = df.join(sup[oi_cols], how="left")
                            for c in oi_cols:
                                df[c] = df[c].ffill().fillna(0)
                            log.info(f"  ✅ Merged Binance OI from {fname}")
                    elif dtype == "ls" and "long_short_ratio" in sup.columns:
                        df = df.join(sup[["long_short_ratio"]], how="left")
                        df["long_short_ratio"] = df["long_short_ratio"].ffill().fillna(1.0)
                        log.info(f"  ✅ Merged Binance L/S ratio from {fname}")
                except Exception as e:
                    log.warning(f"  ⚠️ Could not merge {fname}: {e}")

        # Fill missing supplementary columns with defaults
        if "funding_rate" not in df.columns:
            df["funding_rate"] = 0.0
        if "sum_open_interest" not in df.columns:
            df["sum_open_interest"] = 0.0
        if "sum_open_interest_value" not in df.columns:
            df["sum_open_interest_value"] = 0.0
        if "long_short_ratio" not in df.columns:
            df["long_short_ratio"] = 1.0

        # Reset index
        df = df.reset_index()

        # Compute derived columns (same as Binance pipeline)
        df["volume_delta"] = df["taker_buy_volume"] - (df["volume"] - df["taker_buy_volume"])
        df["volume_ratio"] = df["volume"] / df["volume"].rolling(20, min_periods=1).mean()
        df["returns_1m"] = np.log(df["close"] / df["close"].shift(1)).fillna(0)
        df["returns_5m"] = np.log(df["close"] / df["close"].shift(5)).fillna(0)
        df["returns_15m"] = np.log(df["close"] / df["close"].shift(15)).fillna(0)
        df["oi_change"] = df["sum_open_interest"].pct_change().fillna(0)
        df["funding_rate_change"] = df["funding_rate"].diff().fillna(0)

        # Save processed
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        out_path = PROCESSED_DIR / "SOLUSD_pyth_processed.parquet"
        df.to_parquet(out_path, engine="pyarrow", compression="snappy")

        elapsed = time.time() - t0
        log.info(f"  ✅ Processed: {len(df):,} rows, {len(df.columns)} cols → {out_path}")
        log.info(f"  Date range: {df['datetime'].iloc[0]} → {df['datetime'].iloc[-1]}")
        log.info(f"  Price range: ${df['close'].min():.2f} — ${df['close'].max():.2f}")
        log.info(f"  Total time: {elapsed:.1f}s")

        return df
