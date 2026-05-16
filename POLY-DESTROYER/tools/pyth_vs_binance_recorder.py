"""
High-frequency Pyth vs Binance SOL price recorder.
Records both prices every ~300ms to measure lag/lead relationship.

Usage:
    python tools/pyth_vs_binance_recorder.py [--interval 0.3] [--hours 2]

Output:
    results/price_comparison/pyth_vs_binance_YYYY-MM-DD_HHMMSS.jsonl
"""
import asyncio
import json
import time
import signal
import sys
from datetime import datetime
from pathlib import Path

import httpx
import websockets

# ─── Config ──────────────────────────────────────────────────
INTERVAL = 0.3          # seconds between samples
MAX_HOURS = 24           # auto-stop after this
PYTH_SOL_ID = "0xef0d8b6fda2ceba41da15d4095d1da392a0d2f8ed0c6c7bc0f4cfac8c280b56d"
PYTH_WS_URL = "wss://hermes.pyth.network/ws"
BINANCE_WS_URL = "wss://fstream.binance.com/ws/solusdt@bookTicker"

OUT_DIR = Path("results/price_comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)


class PriceRecorder:
    def __init__(self, interval: float = INTERVAL, max_hours: float = MAX_HOURS):
        self.interval = interval
        self.max_hours = max_hours
        self._running = False
        
        # Prices (updated by websockets)
        self._pyth_price = 0.0
        self._pyth_ts = 0.0       # timestamp when pyth price was received
        self._pyth_publish_ts = 0  # pyth publish_time from oracle
        
        self._binance_price = 0.0
        self._binance_ts = 0.0    # timestamp when binance price was received
        self._binance_event_ts = 0  # binance event time (server)
        
        # Stats
        self._ticks_written = 0
        self._pyth_updates = 0
        self._binance_updates = 0
        self._start_ts = 0
        
        # Output file
        ts_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self._out_path = OUT_DIR / f"pyth_vs_binance_{ts_str}.jsonl"
        self._file = None

    async def start(self):
        self._running = True
        self._start_ts = time.time()
        self._file = open(self._out_path, "a", encoding="utf-8")
        
        print(f"╔══════════════════════════════════════════╗")
        print(f"║  Pyth vs Binance HF Recorder            ║")
        print(f"║  Interval: {self.interval}s | Max: {self.max_hours}h       ║")
        print(f"║  Output: {self._out_path.name}  ║")
        print(f"╚══════════════════════════════════════════╝")
        print()
        
        tasks = [
            asyncio.create_task(self._pyth_ws()),
            asyncio.create_task(self._binance_ws()),
            asyncio.create_task(self._record_loop()),
            asyncio.create_task(self._status_loop()),
        ]
        
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
        finally:
            self._running = False
            if self._file:
                self._file.close()
            print(f"\n✅ Done. {self._ticks_written} ticks → {self._out_path}")

    async def _pyth_ws(self):
        """Subscribe to Pyth Hermes WebSocket for real-time SOL price."""
        while self._running:
            try:
                async with websockets.connect(PYTH_WS_URL, ping_interval=20) as ws:
                    # Subscribe to SOL price feed
                    sub_msg = json.dumps({
                        "type": "subscribe",
                        "ids": [PYTH_SOL_ID]
                    })
                    await ws.send(sub_msg)
                    print("📡 Pyth WS connected")
                    
                    async for msg in ws:
                        if not self._running:
                            break
                        try:
                            data = json.loads(msg)
                            if data.get("type") == "price_update":
                                price_feed = data.get("price_feed", {})
                                price_data = price_feed.get("price", {})
                                price_raw = int(price_data.get("price", 0))
                                expo = int(price_data.get("expo", 0))
                                publish_time = int(price_data.get("publish_time", 0))
                                
                                if price_raw != 0:
                                    self._pyth_price = round(price_raw * (10 ** expo), 6)
                                    self._pyth_ts = time.time()
                                    self._pyth_publish_ts = publish_time
                                    self._pyth_updates += 1
                        except (json.JSONDecodeError, KeyError, ValueError):
                            pass
            except Exception as e:
                if self._running:
                    print(f"  ⚠️ Pyth WS error: {e}, reconnecting...")
                    await asyncio.sleep(1)

    async def _binance_ws(self):
        """Subscribe to Binance Futures bookTicker for SOL (real-time best bid/ask)."""
        while self._running:
            try:
                async with websockets.connect(BINANCE_WS_URL, ping_interval=20) as ws:
                    print("📡 Binance WS connected")
                    
                    async for msg in ws:
                        if not self._running:
                            break
                        try:
                            data = json.loads(msg)
                            # bookTicker: {"e":"bookTicker","u":123,"s":"SOLUSDT","b":"94.50","B":"100","a":"94.51","A":"50","T":1234,"E":1234}
                            if data.get("e") == "bookTicker" or "b" in data:
                                bid = float(data.get("b", 0))
                                ask = float(data.get("a", 0))
                                event_time = int(data.get("E", data.get("T", 0)))
                                if bid > 0 and ask > 0:
                                    self._binance_price = round((bid + ask) / 2, 6)  # mid price
                                    self._binance_ts = time.time()
                                    self._binance_event_ts = event_time
                                    self._binance_updates += 1
                        except (json.JSONDecodeError, KeyError, ValueError):
                            pass
            except Exception as e:
                if self._running:
                    print(f"  ⚠️ Binance WS error: {e}, reconnecting...")
                    await asyncio.sleep(1)

    async def _record_loop(self):
        """Write a tick every INTERVAL seconds."""
        # Wait for both prices to be available
        for _ in range(100):
            if self._pyth_price > 0 and self._binance_price > 0:
                break
            await asyncio.sleep(0.1)
        
        if self._pyth_price == 0 or self._binance_price == 0:
            print("❌ Failed to get initial prices")
            self._running = False
            return
        
        print(f"✅ Recording started: Pyth=${self._pyth_price:.4f} Bin=${self._binance_price:.4f}")
        
        while self._running:
            t0 = time.time()
            
            # Auto-stop
            if t0 - self._start_ts > self.max_hours * 3600:
                print(f"\n⏰ Max duration {self.max_hours}h reached, stopping")
                self._running = False
                break
            
            if self._pyth_price > 0 and self._binance_price > 0:
                now = time.time()
                diff = self._pyth_price - self._binance_price
                diff_pct = diff / self._binance_price * 100
                
                record = {
                    "ts": round(now, 4),
                    "pyth": round(self._pyth_price, 6),
                    "binance": round(self._binance_price, 6),
                    "diff": round(diff, 6),
                    "diff_pct": round(diff_pct, 5),
                    "pyth_age_ms": round((now - self._pyth_ts) * 1000, 1),
                    "bin_age_ms": round((now - self._binance_ts) * 1000, 1),
                    "pyth_publish_ts": self._pyth_publish_ts,
                    "bin_event_ts": self._binance_event_ts,
                }
                
                self._file.write(json.dumps(record) + "\n")
                self._ticks_written += 1
                
                # Flush every 100 ticks
                if self._ticks_written % 100 == 0:
                    self._file.flush()
            
            # Precise sleep
            elapsed = time.time() - t0
            sleep_time = max(0, self.interval - elapsed)
            await asyncio.sleep(sleep_time)

    async def _status_loop(self):
        """Print status every 30 seconds."""
        await asyncio.sleep(5)
        while self._running:
            elapsed = time.time() - self._start_ts
            diff = self._pyth_price - self._binance_price
            print(
                f"  [{int(elapsed//60):>3}m] "
                f"Pyth ${self._pyth_price:.4f} Bin ${self._binance_price:.4f} "
                f"Δ{diff:+.4f} | "
                f"{self._ticks_written} ticks | "
                f"Pyth {self._pyth_updates} upd Bin {self._binance_updates} upd"
            )
            await asyncio.sleep(30)

    def stop(self):
        self._running = False


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pyth vs Binance HF price recorder")
    parser.add_argument("--interval", type=float, default=0.3, help="Recording interval (seconds)")
    parser.add_argument("--hours", type=float, default=2, help="Max recording hours")
    args = parser.parse_args()
    
    recorder = PriceRecorder(interval=args.interval, max_hours=args.hours)
    
    # Graceful shutdown on Ctrl+C
    loop = asyncio.get_event_loop()
    
    def handle_stop(sig, frame):
        print("\n⏹ Stopping...")
        recorder.stop()
    
    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)
    
    await recorder.start()


if __name__ == "__main__":
    asyncio.run(main())
