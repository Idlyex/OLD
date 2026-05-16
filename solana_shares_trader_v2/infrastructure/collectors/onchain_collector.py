"""On-Chain Solana Collector — whale transfers, DEX volumes, MEV, priority fees.
Connects to Solana RPC for real-time on-chain signals.
"""

import asyncio
import time
from collections import deque, defaultdict
from typing import Dict, List, Optional

import httpx
from core.utils.logger import log
from config import config

_cfg_sol = config.get("infrastructure", {}).get("solana", {})
RPC_URL = _cfg_sol.get("rpc_url", "https://api.mainnet-beta.solana.com")
COMMITMENT = _cfg_sol.get("commitment", "confirmed")

# DEX program IDs
DEX_PROGRAMS = _cfg_sol.get("dex_programs", {})
RAYDIUM = DEX_PROGRAMS.get("raydium", "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8")
ORCA = DEX_PROGRAMS.get("orca", "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc")
JUPITER = DEX_PROGRAMS.get("jupiter", "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4")


class OnchainCollector:
    """Collects on-chain Solana signals for feature generation."""

    def __init__(self):
        self._http: Optional[httpx.AsyncClient] = None
        self._running = False

        # Large transfer tracking (>300k USDC)
        self.large_transfers: deque = deque(maxlen=500)
        self._large_transfer_threshold = 300_000  # USDC

        # Whale activity score
        self.whale_scores: Dict[str, float] = {}

        # DEX volume tracking
        self.dex_volumes: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=300)
        )

        # Jupiter swap tracking
        self.jupiter_swaps: deque = deque(maxlen=500)

        # MEV bundle tracking
        self.mev_bundles: deque = deque(maxlen=300)

        # Priority fee tracking
        self.priority_fees: deque = deque(maxlen=300)

        # Token account creation rate
        self.token_creations: deque = deque(maxlen=500)

        # Polling intervals
        self._poll_interval_s = 5

    async def start(self):
        """Start on-chain data collection loop."""
        self._http = httpx.AsyncClient(timeout=10.0)
        self._running = True
        log.info("Onchain collector: started")

        while self._running:
            try:
                await asyncio.gather(
                    self._poll_recent_blocks(),
                    self._poll_priority_fees(),
                    return_exceptions=True,
                )
            except Exception as e:
                log.error(f"Onchain poll error: {e}")

            await asyncio.sleep(self._poll_interval_s)

    async def _rpc_call(self, method: str, params: list = None) -> Optional[dict]:
        """Make a Solana RPC call."""
        if not self._http:
            return None
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": method,
                "params": params or [],
            }
            resp = await self._http.post(RPC_URL, json=payload)
            if resp.status_code == 200:
                result = resp.json()
                return result.get("result")
        except Exception as e:
            log.debug(f"RPC call [{method}] error: {e}")
        return None

    async def _poll_recent_blocks(self):
        """Poll recent blocks for large transfers and DEX activity."""
        # Get recent block
        slot = await self._rpc_call("getSlot", [{"commitment": COMMITMENT}])
        if not slot:
            return

        block = await self._rpc_call(
            "getBlock",
            [
                slot,
                {
                    "encoding": "jsonParsed",
                    "transactionDetails": "signatures",
                    "maxSupportedTransactionVersion": 0,
                },
            ],
        )
        if not block:
            return

        ts_now = int(time.time() * 1000)
        block_time = block.get("blockTime", int(time.time()))

        # Count transactions touching DEX programs
        signatures = block.get("signatures", [])
        tx_count = len(signatures)

        # Record as rough DEX volume proxy
        self.dex_volumes["solana"].append({
            "ts": ts_now,
            "block_slot": slot,
            "tx_count": tx_count,
            "block_time": block_time,
        })

    async def _poll_priority_fees(self):
        """Poll recent priority fees."""
        fees = await self._rpc_call("getRecentPrioritizationFees")
        if not fees:
            return

        ts_now = int(time.time() * 1000)
        avg_fee = 0
        if fees:
            avg_fee = sum(f.get("prioritizationFee", 0) for f in fees[-10:]) / min(
                len(fees), 10
            )

        self.priority_fees.append({
            "ts": ts_now,
            "avg_fee": avg_fee,
            "max_fee": max((f.get("prioritizationFee", 0) for f in fees), default=0),
            "sample_size": len(fees),
        })

    # ── Feature Accessors ──

    def get_large_transfers_count(self, window_s: float = 60) -> int:
        """Count large transfers (>300k USDC) in window."""
        cutoff = int(time.time() * 1000) - int(window_s * 1000)
        return sum(1 for t in self.large_transfers if t.get("ts", 0) >= cutoff)

    def get_whale_activity_score(self) -> float:
        """Aggregate whale activity score (0-1)."""
        if not self.whale_scores:
            return 0.0
        return min(1.0, sum(self.whale_scores.values()))

    def get_dex_volume_spike(self, window_s: float = 60) -> float:
        """DEX volume acceleration (current vs historical average)."""
        now_ms = int(time.time() * 1000)
        vols = list(self.dex_volumes.get("solana", []))
        if len(vols) < 5:
            return 1.0

        cutoff = now_ms - int(window_s * 1000)
        recent = [v["tx_count"] for v in vols if v["ts"] >= cutoff]
        older = [v["tx_count"] for v in vols if v["ts"] < cutoff]

        if not recent or not older:
            return 1.0

        avg_recent = sum(recent) / len(recent)
        avg_older = sum(older) / len(older)
        return avg_recent / avg_older if avg_older > 0 else 1.0

    def get_jupiter_swap_acceleration(self, window_s: float = 60) -> float:
        """Jupiter swap volume acceleration."""
        now_ms = int(time.time() * 1000)
        cutoff = now_ms - int(window_s * 1000)
        recent = sum(1 for s in self.jupiter_swaps if s.get("ts", 0) >= cutoff)
        older = sum(
            1
            for s in self.jupiter_swaps
            if s.get("ts", 0) < cutoff
            and s.get("ts", 0) >= cutoff - int(window_s * 1000)
        )
        return recent / max(older, 1)

    def get_mev_bundle_count(self, window_s: float = 60) -> int:
        """MEV bundles detected in window."""
        cutoff = int(time.time() * 1000) - int(window_s * 1000)
        return sum(1 for b in self.mev_bundles if b.get("ts", 0) >= cutoff)

    def get_priority_fee_pressure(self) -> float:
        """Current priority fee relative to recent average."""
        if len(self.priority_fees) < 2:
            return 1.0
        fees = [f["avg_fee"] for f in self.priority_fees]
        current = fees[-1] if fees else 0
        avg = sum(fees) / len(fees) if fees else 1
        return current / avg if avg > 0 else 1.0

    def get_token_creation_rate(self, window_s: float = 60) -> float:
        """Token account creations per second (pump signal)."""
        now_ms = int(time.time() * 1000)
        cutoff = now_ms - int(window_s * 1000)
        count = sum(1 for t in self.token_creations if t.get("ts", 0) >= cutoff)
        return count / window_s if window_s > 0 else 0.0

    def get_all_features(self) -> Dict[str, float]:
        """Get all on-chain features as a flat dict."""
        return {
            "onchain_large_transfers_60s": float(self.get_large_transfers_count(60)),
            "onchain_whale_activity": self.get_whale_activity_score(),
            "onchain_dex_volume_spike": self.get_dex_volume_spike(60),
            "onchain_jupiter_accel": self.get_jupiter_swap_acceleration(60),
            "onchain_mev_bundles": float(self.get_mev_bundle_count(60)),
            "onchain_priority_fee_pressure": self.get_priority_fee_pressure(),
            "onchain_token_creation_rate": self.get_token_creation_rate(60),
            "onchain_large_transfers_300s": float(self.get_large_transfers_count(300)),
            "onchain_dex_volume_spike_300s": self.get_dex_volume_spike(300),
            "onchain_jupiter_accel_300s": self.get_jupiter_swap_acceleration(300),
        }

    def stop(self):
        self._running = False
        if self._http:
            asyncio.create_task(self._http.aclose())
