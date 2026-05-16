"""Data Collector — async historical + live data downloader.

Supports:
  - Binance USDS-M Futures: 1s/1m klines, aggTrades, depth snapshots, liquidations, funding
  - Bybit Perpetual: 1m klines, trades, funding
  - Solana on-chain: large transfers, DEX volume (Helius / QuickNode)

Usage:
  python main.py --mode download --symbol SOLUSDT --days 730
  python main.py --mode download --symbol SOLUSDT --days 30 --granularity 1s

Data is saved to data/raw/ as parquet, then processed → data/processed/
"""

import asyncio
import os
import time
import math
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import httpx

from core.utils.logger import log
from config import config

_data_cfg = config.get("data_pipeline", {})
RAW_DIR = Path(_data_cfg.get("raw_dir", "data/raw"))
PROCESSED_DIR = Path(_data_cfg.get("processed_dir", "data/processed"))

BINANCE_FAPI = "https://fapi.binance.com"
BYBIT_API = "https://api.bybit.com"
RATE_LIMIT_DELAY = 0.12  # seconds between API calls


class DataCollector:
    """Async data downloader for historical and live market data."""

    def __init__(self, symbol: str = "SOLUSDT"):
        self.symbol = symbol.upper()
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        self._http: Optional[httpx.AsyncClient] = None
        self._total_requests = 0
        self._started_at = 0.0

    async def _get_http(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(timeout=30.0, limits=httpx.Limits(max_connections=10))
        return self._http

    async def _request(self, url: str, params: dict = None, _retries: int = 3) -> Optional[dict | list]:
        """Rate-limited HTTP GET with retry."""
        http = await self._get_http()
        self._total_requests += 1
        try:
            t0 = time.monotonic()
            resp = await http.get(url, params=params)
            elapsed = time.monotonic() - t0

            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 10))
                log.warning(f"Rate limited — sleeping {retry_after}s")
                await asyncio.sleep(retry_after)
                return await self._request(url, params, _retries)
            if resp.status_code in (418, 403) and _retries > 0:
                log.warning(f"HTTP {resp.status_code} (IP ban?) — backing off 5s (retries left: {_retries})")
                await asyncio.sleep(5)
                return await self._request(url, params, _retries - 1)
            if resp.status_code != 200:
                log.warning(f"HTTP {resp.status_code}: {url.split('/')[-1]} ({elapsed:.1f}s)")
                return None
            if elapsed > 5.0:
                log.warning(f"Slow response: {url.split('/')[-1]} took {elapsed:.1f}s")
            return resp.json()
        except httpx.TimeoutException:
            if _retries > 0:
                log.warning(f"Timeout on {url.split('/')[-1]} — retrying ({_retries} left)")
                await asyncio.sleep(2)
                return await self._request(url, params, _retries - 1)
            log.error(f"Timeout after retries: {url}")
            return None
        except Exception as e:
            log.error(f"Request error: {e}")
            return None

    @staticmethod
    def _period_ms(period: str) -> int:
        """Convert period string (5m, 15m, 1h, etc.) to milliseconds."""
        unit = period[-1]
        val = int(period[:-1])
        multipliers = {"s": 1_000, "m": 60_000, "h": 3_600_000, "d": 86_400_000}
        return val * multipliers.get(unit, 60_000)

    # ═══════════════════════════════════════════════════════════
    #  BINANCE FUTURES
    # ═══════════════════════════════════════════════════════════

    async def download_binance_klines(
        self,
        interval: str = "1m",
        days: int = 730,
        end_time: int = None,
    ) -> pd.DataFrame:
        """Download Binance Futures klines.

        Args:
            interval: 1s, 1m, 3m, 5m, 15m, 1h, etc.
            days: number of days to download
            end_time: end timestamp in ms (default: now)
        """
        log.info(f"📥 Downloading Binance {self.symbol} {interval} klines — {days} days")

        if end_time is None:
            end_time = int(time.time() * 1000)

        # Calculate interval in ms
        interval_ms_map = {
            "1s": 1000, "1m": 60_000, "3m": 180_000, "5m": 300_000,
            "15m": 900_000, "30m": 1_800_000, "1h": 3_600_000,
            "4h": 14_400_000, "1d": 86_400_000,
        }
        interval_ms = interval_ms_map.get(interval, 60_000)

        start_time = end_time - days * 86_400_000
        limit = 1500 if interval != "1s" else 1000
        all_candles = []
        current = start_time

        total_expected = (end_time - start_time) // interval_ms
        log.info(f"  Expected ~{total_expected:,} candles, fetching {limit} per request")

        batch = 0
        while current < end_time:
            batch += 1
            params = {
                "symbol": self.symbol,
                "interval": interval,
                "startTime": current,
                "endTime": end_time,
                "limit": limit,
            }
            data = await self._request(f"{BINANCE_FAPI}/fapi/v1/klines", params)
            if not data or len(data) == 0:
                break

            for c in data:
                all_candles.append({
                    "ts": int(c[0]),
                    "open": float(c[1]),
                    "high": float(c[2]),
                    "low": float(c[3]),
                    "close": float(c[4]),
                    "volume": float(c[5]),
                    "close_time": int(c[6]),
                    "quote_volume": float(c[7]),
                    "trades": int(c[8]),
                    "taker_buy_volume": float(c[9]),
                    "taker_buy_quote_volume": float(c[10]),
                })

            current = int(data[-1][0]) + interval_ms

            if batch % 50 == 0:
                pct = (current - start_time) / (end_time - start_time) * 100
                log.info(f"  {pct:.0f}% — {len(all_candles):,} candles ({batch} requests)")

            await asyncio.sleep(RATE_LIMIT_DELAY)

        if not all_candles:
            log.warning(f"No klines downloaded for {self.symbol} {interval}")
            return pd.DataFrame()

        df = pd.DataFrame(all_candles)
        df = df.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)

        # Save
        fname = f"{self.symbol}_{interval}_klines.parquet"
        path = RAW_DIR / fname
        df.to_parquet(path, engine="pyarrow", compression="snappy")
        log.info(f"  ✅ Saved {len(df):,} candles → {path} ({path.stat().st_size / 1e6:.1f}MB)")

        return df

    async def download_binance_agg_trades(
        self,
        days: int = 30,
        end_time: int = None,
    ) -> pd.DataFrame:
        """Download Binance Futures aggregate trades.
        Warning: very large datasets. Use days <= 30 for manageable size.
        """
        log.info(f"📥 Downloading Binance {self.symbol} aggTrades — {days} days")

        if end_time is None:
            end_time = int(time.time() * 1000)
        start_time = end_time - days * 86_400_000

        all_trades = []
        current = start_time
        batch = 0
        last_id = None

        while current < end_time:
            batch += 1
            params = {
                "symbol": self.symbol,
                "startTime": current,
                "endTime": min(current + 3_600_000, end_time),  # 1h chunks
                "limit": 1000,
            }
            if last_id:
                params["fromId"] = last_id + 1
                params.pop("startTime", None)
                params.pop("endTime", None)

            data = await self._request(f"{BINANCE_FAPI}/fapi/v1/aggTrades", params)
            if not data or len(data) == 0:
                current += 3_600_000
                last_id = None
                continue

            for t in data:
                all_trades.append({
                    "ts": int(t["T"]),
                    "agg_trade_id": int(t["a"]),
                    "price": float(t["p"]),
                    "qty": float(t["q"]),
                    "first_trade_id": int(t["f"]),
                    "last_trade_id": int(t["l"]),
                    "is_buyer_maker": t["m"],
                })

            last_id = int(data[-1]["a"])
            last_ts = int(data[-1]["T"])

            if len(data) < 1000:
                current = last_ts + 1
                last_id = None
            else:
                pass  # continue with fromId

            if batch % 100 == 0:
                pct = (current - start_time) / (end_time - start_time) * 100
                log.info(f"  {pct:.0f}% — {len(all_trades):,} trades ({batch} requests)")

            await asyncio.sleep(RATE_LIMIT_DELAY)

        if not all_trades:
            return pd.DataFrame()

        df = pd.DataFrame(all_trades)
        df = df.drop_duplicates(subset=["agg_trade_id"]).sort_values("ts").reset_index(drop=True)

        fname = f"{self.symbol}_agg_trades_{days}d.parquet"
        path = RAW_DIR / fname
        df.to_parquet(path, engine="pyarrow", compression="snappy")
        log.info(f"  ✅ Saved {len(df):,} trades → {path} ({path.stat().st_size / 1e6:.1f}MB)")

        return df

    async def download_binance_funding_rate(
        self,
        days: int = 730,
    ) -> pd.DataFrame:
        """Download historical funding rate data."""
        log.info(f"📥 Downloading Binance {self.symbol} funding rates — {days} days")

        end_time = int(time.time() * 1000)
        start_time = end_time - days * 86_400_000
        all_rates = []
        current = start_time

        while current < end_time:
            params = {
                "symbol": self.symbol,
                "startTime": current,
                "limit": 1000,
            }
            data = await self._request(f"{BINANCE_FAPI}/fapi/v1/fundingRate", params)
            if not data or len(data) == 0:
                break

            for r in data:
                all_rates.append({
                    "ts": int(r["fundingTime"]),
                    "funding_rate": float(r["fundingRate"]),
                    "mark_price": float(r.get("markPrice", 0)),
                })

            current = int(data[-1]["fundingTime"]) + 1
            await asyncio.sleep(RATE_LIMIT_DELAY)

        if not all_rates:
            return pd.DataFrame()

        df = pd.DataFrame(all_rates).drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)

        fname = f"{self.symbol}_funding_rate.parquet"
        path = RAW_DIR / fname
        df.to_parquet(path, engine="pyarrow", compression="snappy")
        log.info(f"  ✅ Saved {len(df):,} funding rates → {path}")

        return df

    async def download_binance_open_interest(
        self,
        days: int = 30,
        period: str = "5m",
    ) -> pd.DataFrame:
        """Download historical open interest data."""
        # Auto-upgrade period — these endpoints return ~5 records/request
        if period == "5m":
            if days > 30:
                period = "4h"
            elif days > 7:
                period = "1h"
            else:
                period = "15m"

        log.info(f"📥 Downloading Binance {self.symbol} open interest — {days} days ({period})")

        end_time = int(time.time() * 1000)
        start_time = end_time - days * 86_400_000
        all_oi = []
        current = start_time
        req_count = 0

        while current < end_time:
            params = {
                "symbol": self.symbol,
                "period": period,
                "startTime": current,
                "endTime": end_time,
                "limit": 500,
            }
            data = await self._request(f"{BINANCE_FAPI}/futures/data/openInterestHist", params)
            if not data or len(data) == 0:
                break

            req_count += 1
            for r in data:
                all_oi.append({
                    "ts": int(r["timestamp"]),
                    "sum_open_interest": float(r["sumOpenInterest"]),
                    "sum_open_interest_value": float(r["sumOpenInterestValue"]),
                })

            if req_count % 10 == 0:
                log.info(f"    OI: {len(all_oi):,} records ({req_count} reqs)")

            current = int(data[-1]["timestamp"]) + 1
            if len(data) < 3:  # tail end — no more bulk data
                break
            await asyncio.sleep(RATE_LIMIT_DELAY)

        if not all_oi:
            return pd.DataFrame()

        df = pd.DataFrame(all_oi).drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
        fname = f"{self.symbol}_open_interest_{period}.parquet"
        path = RAW_DIR / fname
        df.to_parquet(path, engine="pyarrow", compression="snappy")
        log.info(f"  ✅ Saved {len(df):,} OI records → {path} ({req_count} reqs)")
        return df

    async def download_binance_long_short_ratio(
        self,
        days: int = 30,
        period: str = "5m",
    ) -> pd.DataFrame:
        """Download top trader long/short ratio."""
        if period == "5m":
            if days > 30:
                period = "4h"
            elif days > 7:
                period = "1h"
            else:
                period = "15m"

        log.info(f"📥 Downloading Binance {self.symbol} long/short ratio — {days} days ({period})")

        end_time = int(time.time() * 1000)
        start_time = end_time - days * 86_400_000
        all_ratios = []
        current = start_time
        req_count = 0

        while current < end_time:
            params = {
                "symbol": self.symbol,
                "period": period,
                "startTime": current,
                "endTime": end_time,
                "limit": 500,
            }
            data = await self._request(f"{BINANCE_FAPI}/futures/data/topLongShortAccountRatio", params)
            if not data or len(data) == 0:
                break

            req_count += 1
            for r in data:
                all_ratios.append({
                    "ts": int(r["timestamp"]),
                    "long_account": float(r["longAccount"]),
                    "short_account": float(r["shortAccount"]),
                    "long_short_ratio": float(r["longShortRatio"]),
                })

            if req_count % 10 == 0:
                log.info(f"    L/S: {len(all_ratios):,} records ({req_count} reqs)")

            current = int(data[-1]["timestamp"]) + 1
            if len(data) < 3:  # tail end — no more bulk data
                break
            await asyncio.sleep(RATE_LIMIT_DELAY)

        if not all_ratios:
            return pd.DataFrame()

        df = pd.DataFrame(all_ratios).drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
        fname = f"{self.symbol}_long_short_ratio.parquet"
        path = RAW_DIR / fname
        df.to_parquet(path, engine="pyarrow", compression="snappy")
        log.info(f"  ✅ Saved {len(df):,} L/S ratio records → {path} ({req_count} reqs)")

        return df

    # ═══════════════════════════════════════════════════════════
    #  BYBIT PERPETUAL
    # ═══════════════════════════════════════════════════════════

    async def download_bybit_klines(
        self,
        interval: str = "1",
        days: int = 200,
    ) -> pd.DataFrame:
        """Download Bybit perpetual klines. Interval: 1,3,5,15,30,60,120,240,D."""
        log.info(f"📥 Downloading Bybit {self.symbol} {interval}m klines — {days} days")

        end_time = int(time.time() * 1000)
        start_time = end_time - days * 86_400_000
        all_candles = []
        current = start_time

        while current < end_time:
            params = {
                "category": "linear",
                "symbol": self.symbol,
                "interval": interval,
                "start": current,
                "end": min(current + 200 * 60_000 * int(interval), end_time),
                "limit": 200,
            }
            data = await self._request(f"{BYBIT_API}/v5/market/kline", params)
            if not data or data.get("retCode") != 0:
                break

            result_list = data.get("result", {}).get("list", [])
            if not result_list:
                break

            for c in result_list:
                all_candles.append({
                    "ts": int(c[0]),
                    "open": float(c[1]),
                    "high": float(c[2]),
                    "low": float(c[3]),
                    "close": float(c[4]),
                    "volume": float(c[5]),
                    "turnover": float(c[6]),
                })

            # Bybit returns newest first
            oldest_ts = min(int(c[0]) for c in result_list)
            newest_ts = max(int(c[0]) for c in result_list)
            current = newest_ts + int(interval) * 60_000

            if len(result_list) < 200:
                break

            await asyncio.sleep(RATE_LIMIT_DELAY)

        if not all_candles:
            return pd.DataFrame()

        df = pd.DataFrame(all_candles)
        df = df.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)

        fname = f"{self.symbol}_bybit_{interval}m_klines.parquet"
        path = RAW_DIR / fname
        df.to_parquet(path, engine="pyarrow", compression="snappy")
        log.info(f"  ✅ Saved {len(df):,} Bybit candles → {path}")

        return df

    # ═══════════════════════════════════════════════════════════
    #  FULL DOWNLOAD PIPELINE
    # ═══════════════════════════════════════════════════════════

    async def download_all(
        self,
        days: int = 730,
        granularity: str = "1m",
        include_trades: bool = False,
        trade_days: int = 7,
    ):
        """Download all data sources in parallel where possible.

        Args:
            days: days of kline history
            granularity: kline interval (1s, 1m, 5m, etc.)
            include_trades: download raw aggTrades (large!)
            trade_days: days of aggTrades (default: 7)
        """
        self._started_at = time.time()

        log.info("╔══════════════════════════════════════════════════╗")
        log.info(f"║  Data Download: {self.symbol:<10} │ {days} days │ {granularity:<4}     ║")
        log.info("╚══════════════════════════════════════════════════╝")

        klines_1m = pd.DataFrame()
        funding = pd.DataFrame()
        oi = pd.DataFrame()
        ls = pd.DataFrame()

        # Phase 1: Core kline data
        try:
            klines_1m = await self.download_binance_klines("1m", days)
        except Exception as e:
            log.error(f"1m klines failed: {e}")

        if granularity != "1m":
            try:
                await self.download_binance_klines(granularity, days)
            except Exception as e:
                log.warning(f"{granularity} klines failed: {e}")

        for tf in ["5m", "15m"]:
            if tf != granularity:
                try:
                    await self.download_binance_klines(tf, days)
                except Exception as e:
                    log.warning(f"{tf} klines failed: {e}")

        # Phase 2: Derivatives data
        try:
            funding = await self.download_binance_funding_rate(days)
        except Exception as e:
            log.warning(f"Funding rate failed: {e}")

        # OI + L/S ratio — sequential (Binance throttles concurrent /futures/data/)
        oi_days = min(days, 30)
        try:
            oi = await self.download_binance_open_interest(oi_days, "5m")
        except Exception as e:
            log.warning(f"OI download failed: {e}")

        try:
            ls = await self.download_binance_long_short_ratio(oi_days, "5m")
        except Exception as e:
            log.warning(f"L/S ratio failed: {e}")

        # Phase 3: Optional raw trades
        if include_trades:
            try:
                await self.download_binance_agg_trades(min(trade_days, 30))
            except Exception as e:
                log.warning(f"AggTrades failed: {e}")

        # Phase 4: Bybit cross-reference
        try:
            bybit_days = min(days, 200)
            await self.download_bybit_klines("1", bybit_days)
        except Exception as e:
            log.warning(f"Bybit klines failed: {e}")

        # Phase 5: Always process whatever we got
        log.info("📊 Processing data → features...")
        await self._process_raw_data(klines_1m, funding, oi, ls)

        elapsed = time.time() - self._started_at
        log.info(f"✅ All downloads complete — {self._total_requests} requests in {elapsed:.0f}s")

    async def _process_raw_data(
        self,
        klines: pd.DataFrame,
        funding: pd.DataFrame,
        oi: pd.DataFrame,
        ls_ratio: pd.DataFrame,
    ):
        """Merge raw data into a single processed dataset with aligned timestamps."""
        if klines.empty:
            log.warning("No klines to process")
            return

        df = klines.copy()
        df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df.set_index("datetime")

        # Merge funding rate (forward-fill to 1m)
        if not funding.empty:
            fr = funding.copy()
            fr["datetime"] = pd.to_datetime(fr["ts"], unit="ms", utc=True)
            fr = fr.set_index("datetime")[["funding_rate"]].rename(columns={"funding_rate": "funding_rate"})
            fr = fr[~fr.index.duplicated(keep="last")]
            df = df.join(fr, how="left")
            df["funding_rate"] = df["funding_rate"].ffill().fillna(0)
        else:
            df["funding_rate"] = 0.0

        # Merge OI
        if not oi.empty:
            oi_df = oi.copy()
            oi_df["datetime"] = pd.to_datetime(oi_df["ts"], unit="ms", utc=True)
            oi_df = oi_df.set_index("datetime")[["sum_open_interest", "sum_open_interest_value"]]
            oi_df = oi_df[~oi_df.index.duplicated(keep="last")]
            df = df.join(oi_df, how="left")
            df["sum_open_interest"] = df["sum_open_interest"].ffill().fillna(0)
            df["sum_open_interest_value"] = df["sum_open_interest_value"].ffill().fillna(0)
        else:
            df["sum_open_interest"] = 0.0
            df["sum_open_interest_value"] = 0.0

        # Merge Long/Short ratio
        if not ls_ratio.empty:
            ls_df = ls_ratio.copy()
            ls_df["datetime"] = pd.to_datetime(ls_df["ts"], unit="ms", utc=True)
            ls_df = ls_df.set_index("datetime")[["long_short_ratio"]]
            ls_df = ls_df[~ls_df.index.duplicated(keep="last")]
            df = df.join(ls_df, how="left")
            df["long_short_ratio"] = df["long_short_ratio"].ffill().fillna(1.0)
        else:
            df["long_short_ratio"] = 1.0

        # Reset index
        df = df.reset_index()

        # Compute derived columns used downstream
        df["volume_delta"] = df["taker_buy_volume"] - (df["volume"] - df["taker_buy_volume"])
        df["volume_ratio"] = df["volume"] / df["volume"].rolling(20, min_periods=1).mean()
        df["returns_1m"] = np.log(df["close"] / df["close"].shift(1)).fillna(0)
        df["returns_5m"] = np.log(df["close"] / df["close"].shift(5)).fillna(0)
        df["returns_15m"] = np.log(df["close"] / df["close"].shift(15)).fillna(0)

        # OI change
        df["oi_change"] = df["sum_open_interest"].pct_change().fillna(0)
        df["funding_rate_change"] = df["funding_rate"].diff().fillna(0)

        # Save processed
        out_path = PROCESSED_DIR / f"{self.symbol}_processed.parquet"
        df.to_parquet(out_path, engine="pyarrow", compression="snappy")
        log.info(f"  ✅ Processed dataset: {len(df):,} rows, {len(df.columns)} cols → {out_path}")

        # Print summary
        log.info(f"  Date range: {df['datetime'].iloc[0]} → {df['datetime'].iloc[-1]}")
        log.info(f"  Price range: ${df['close'].min():.2f} — ${df['close'].max():.2f}")
        log.info(f"  Columns: {', '.join(df.columns[:15])}...")

        return df

    async def close(self):
        if self._http and not self._http.is_closed:
            await self._http.aclose()


def load_processed_data(symbol: str = "SOLUSDT") -> pd.DataFrame:
    """Load processed dataset for training/backtesting."""
    path = PROCESSED_DIR / f"{symbol}_processed.parquet"
    if not path.exists():
        log.error(f"No processed data at {path}. Run: python main.py --mode download --symbol {symbol}")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    log.info(f"Loaded {len(df):,} rows from {path}")
    return df


def list_available_data() -> List[Dict]:
    """List available raw and processed datasets."""
    datasets = []
    for d in [RAW_DIR, PROCESSED_DIR]:
        if d.exists():
            for f in sorted(d.glob("*.parquet")):
                stat = f.stat()
                datasets.append({
                    "path": str(f),
                    "name": f.name,
                    "size_mb": stat.st_size / 1e6,
                    "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                    "type": "raw" if d == RAW_DIR else "processed",
                })
    return datasets
