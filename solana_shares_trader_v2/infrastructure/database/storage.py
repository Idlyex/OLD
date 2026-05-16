"""Data storage engine — Parquet files + ClickHouse integration."""

import os
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from core.utils.logger import log
from config import config

_cfg_data = config.get("infrastructure", {}).get("data", {})
PARQUET_DIR = Path(_cfg_data.get("parquet_dir", "data/parquet"))
TRADES_DIR = Path(_cfg_data.get("trades_dir", "data/trades"))
FEATURES_DIR = Path(_cfg_data.get("features_dir", "data/features"))
MODELS_DIR = Path(_cfg_data.get("models_dir", "data/models"))


class ParquetStore:
    """High-performance columnar storage for features, trades, and OHLCV data."""

    def __init__(self):
        for d in [PARQUET_DIR, TRADES_DIR, FEATURES_DIR, MODELS_DIR]:
            d.mkdir(parents=True, exist_ok=True)

        self._buffers: Dict[str, List[Dict]] = {}
        self._flush_interval = 60  # seconds
        self._last_flush: Dict[str, float] = {}
        self._max_buffer = 5000

    def append_features(self, symbol: str, features: Dict[str, Any], ts: int = None):
        """Buffer a feature row for batch writing."""
        key = f"features_{symbol}"
        if key not in self._buffers:
            self._buffers[key] = []

        row = {"ts": ts or int(time.time() * 1000), "symbol": symbol, **features}
        self._buffers[key].append(row)

        if len(self._buffers[key]) >= self._max_buffer:
            self.flush(key)

    def append_trade(self, trade: Dict[str, Any]):
        """Buffer a trade record."""
        key = "trades"
        if key not in self._buffers:
            self._buffers[key] = []
        self._buffers[key].append(trade)

    def append_ohlcv(self, symbol: str, interval: str, candle: Dict[str, Any]):
        """Buffer an OHLCV candle."""
        key = f"ohlcv_{symbol}_{interval}"
        if key not in self._buffers:
            self._buffers[key] = []
        self._buffers[key].append(candle)

        if len(self._buffers[key]) >= self._max_buffer:
            self.flush(key)

    def flush(self, key: str = None):
        """Write buffered data to parquet files."""
        keys = [key] if key else list(self._buffers.keys())

        for k in keys:
            buf = self._buffers.get(k, [])
            if not buf:
                continue

            try:
                df = pd.DataFrame(buf)
                date_str = datetime.now(timezone.utc).strftime("%Y%m%d")

                if k.startswith("features_"):
                    out_dir = FEATURES_DIR / k.replace("features_", "")
                elif k == "trades":
                    out_dir = TRADES_DIR
                elif k.startswith("ohlcv_"):
                    out_dir = PARQUET_DIR / "ohlcv"
                else:
                    out_dir = PARQUET_DIR / k

                out_dir.mkdir(parents=True, exist_ok=True)
                path = out_dir / f"{k}_{date_str}.parquet"

                if path.exists():
                    existing = pd.read_parquet(path)
                    df = pd.concat([existing, df], ignore_index=True)

                df.to_parquet(path, engine="pyarrow", compression="snappy")
                self._buffers[k] = []
                self._last_flush[k] = time.time()
                log.debug(f"Flushed {len(buf)} rows → {path}")

            except Exception as e:
                log.error(f"Parquet flush error [{k}]: {e}")

    def flush_all(self):
        """Flush all buffers."""
        self.flush()

    def load_features(
        self, symbol: str, start_ts: int = None, end_ts: int = None
    ) -> pd.DataFrame:
        """Load feature data from parquet files."""
        feature_dir = FEATURES_DIR / symbol
        if not feature_dir.exists():
            return pd.DataFrame()

        files = sorted(feature_dir.glob("*.parquet"))
        if not files:
            return pd.DataFrame()

        dfs = []
        for f in files:
            try:
                df = pd.read_parquet(f)
                if start_ts and "ts" in df.columns:
                    df = df[df["ts"] >= start_ts]
                if end_ts and "ts" in df.columns:
                    df = df[df["ts"] <= end_ts]
                dfs.append(df)
            except Exception as e:
                log.warning(f"Failed to read {f}: {e}")

        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)

    def load_trades(self, limit: int = None) -> pd.DataFrame:
        """Load trade history."""
        files = sorted(TRADES_DIR.glob("*.parquet"))
        if not files:
            return pd.DataFrame()

        dfs = [pd.read_parquet(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)
        if limit:
            df = df.tail(limit)
        return df

    def load_ohlcv(
        self, symbol: str, interval: str, limit: int = None
    ) -> pd.DataFrame:
        """Load OHLCV candle data."""
        ohlcv_dir = PARQUET_DIR / "ohlcv"
        if not ohlcv_dir.exists():
            return pd.DataFrame()

        key = f"ohlcv_{symbol}_{interval}"
        files = sorted(ohlcv_dir.glob(f"{key}_*.parquet"))
        if not files:
            return pd.DataFrame()

        dfs = [pd.read_parquet(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)
        if limit:
            df = df.tail(limit)
        return df


class ClickHouseStore:
    """ClickHouse integration for persistent time-series storage."""

    def __init__(self):
        self._client = None
        self._cfg = config.get("infrastructure", {}).get("clickhouse", {})
        self._connected = False

    async def connect(self):
        """Connect to ClickHouse and create tables if needed."""
        try:
            import clickhouse_connect

            self._client = clickhouse_connect.get_client(
                host=self._cfg.get("host", "localhost"),
                port=int(self._cfg.get("port", 8123)),
                database=self._cfg.get("database", "solana_trader"),
                username=self._cfg.get("username", "default"),
                password=self._cfg.get("password", ""),
            )
            self._create_tables()
            self._connected = True
            log.info("ClickHouse: connected ✅")
        except Exception as e:
            log.warning(f"ClickHouse: not available ({e}) — using parquet only")
            self._connected = False

    def _create_tables(self):
        """Create tables if they don't exist."""
        if not self._client:
            return

        self._client.command("""
            CREATE TABLE IF NOT EXISTS features (
                ts DateTime64(3),
                symbol String,
                feature_name String,
                value Float64
            ) ENGINE = MergeTree()
            ORDER BY (symbol, ts, feature_name)
        """)

        self._client.command("""
            CREATE TABLE IF NOT EXISTS trades (
                ts DateTime64(3),
                symbol String,
                slug String,
                direction String,
                entry_price Float64,
                exit_price Float64,
                shares Float64,
                pnl_pct Float64,
                pnl_usd Float64,
                hold_time_s Float64,
                exit_reason String,
                confidence Float64,
                regime String
            ) ENGINE = MergeTree()
            ORDER BY (ts, symbol)
        """)

        self._client.command("""
            CREATE TABLE IF NOT EXISTS ohlcv (
                ts DateTime64(3),
                symbol String,
                interval String,
                open Float64,
                high Float64,
                low Float64,
                close Float64,
                volume Float64,
                quote_volume Float64,
                trades UInt32
            ) ENGINE = MergeTree()
            ORDER BY (symbol, interval, ts)
        """)

        self._client.command("""
            CREATE TABLE IF NOT EXISTS orderbook_snapshots (
                ts DateTime64(3),
                symbol String,
                bid_1 Float64, ask_1 Float64,
                bid_qty_1 Float64, ask_qty_1 Float64,
                bid_5 Float64, ask_5 Float64,
                bid_qty_5 Float64, ask_qty_5 Float64,
                spread Float64,
                imbalance_5 Float64,
                imbalance_10 Float64,
                imbalance_20 Float64
            ) ENGINE = MergeTree()
            ORDER BY (symbol, ts)
        """)

    def insert_features(self, rows: List[Dict]):
        """Batch insert feature rows."""
        if not self._connected or not rows:
            return
        try:
            df = pd.DataFrame(rows)
            self._client.insert_df("features", df)
        except Exception as e:
            log.error(f"ClickHouse insert features: {e}")

    def insert_trade(self, trade: Dict):
        """Insert a single trade record."""
        if not self._connected:
            return
        try:
            df = pd.DataFrame([trade])
            self._client.insert_df("trades", df)
        except Exception as e:
            log.error(f"ClickHouse insert trade: {e}")

    def insert_ohlcv(self, rows: List[Dict]):
        """Batch insert OHLCV rows."""
        if not self._connected or not rows:
            return
        try:
            df = pd.DataFrame(rows)
            self._client.insert_df("ohlcv", df)
        except Exception as e:
            log.error(f"ClickHouse insert ohlcv: {e}")

    def query(self, sql: str) -> pd.DataFrame:
        """Run a SQL query and return DataFrame."""
        if not self._connected:
            return pd.DataFrame()
        try:
            return self._client.query_df(sql)
        except Exception as e:
            log.error(f"ClickHouse query error: {e}")
            return pd.DataFrame()

    @property
    def is_connected(self) -> bool:
        return self._connected
