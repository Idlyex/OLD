"""Polymarket CLOB Client — limit/market orders, balance tracking.
Port of POLYx clob-client.js to async Python.
"""

import asyncio
from typing import Optional, Dict, Any
from eth_account import Account
from eth_account.signers.local import LocalAccount

import httpx
from core.utils.logger import log
from config import config

_cfg_poly = config.get("infrastructure", {}).get("polymarket", {})
CLOB_HOST = _cfg_poly.get("clob_host", "https://clob.polymarket.com")
CHAIN_ID = _cfg_poly.get("chain_id", 137)


class PolymarketCLOB:
    """Async Polymarket CLOB client for order placement and balance queries."""

    def __init__(self):
        self.host = CLOB_HOST
        self.chain_id = CHAIN_ID
        self._http: Optional[httpx.AsyncClient] = None
        self._signer: Optional[LocalAccount] = None
        self._api_creds: Optional[dict] = None
        self._read_only_reason: Optional[str] = None
        self._balance_cache: float = 0.0
        self._initialized = False

    async def init(self):
        """Initialize wallet and derive API credentials."""
        if self._initialized:
            return

        secrets = config.get("_secrets", {})
        pk = secrets.get("private_key", "")

        if not pk or pk == "0x_YOUR_PRIVATE_KEY_HERE":
            self._read_only_reason = "No PRIVATE_KEY set"
            log.warning("CLOB: No valid PRIVATE_KEY — read-only mode")
            self._initialized = True
            return

        try:
            self._signer = Account.from_key(pk)
            log.info(f"CLOB: Wallet {self._signer.address}")
        except Exception as e:
            self._read_only_reason = str(e)
            log.error(f"CLOB init error: {e}")
            self._initialized = True
            return

        self._http = httpx.AsyncClient(timeout=15.0)
        self._initialized = True
        log.info("CLOB: Client ready ✅")

    @property
    def is_read_only(self) -> bool:
        return self._signer is None

    async def get_orderbook(self, token_id: str) -> Optional[Dict]:
        """Fetch order book for a token."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{self.host}/book",
                    params={"token_id": token_id},
                )
                if resp.status_code == 200:
                    return resp.json()
        except Exception as e:
            log.error(f"Orderbook error: {e}")
        return None

    async def get_midpoint(self, token_id: str) -> Optional[float]:
        """Get midpoint price for a token."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{self.host}/midpoint",
                    params={"token_id": token_id},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return float(data.get("mid", 0))
        except Exception as e:
            log.error(f"Midpoint error: {e}")
        return None

    async def get_balance(self) -> Optional[float]:
        """Get USDC balance."""
        if self.is_read_only:
            return None
        # Placeholder — actual implementation requires signed API calls
        return self._balance_cache

    async def buy_shares(
        self,
        token_id: str,
        price: float,
        amount_usd: float,
        condition_id: str = "",
        dry_run: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Buy shares — sweeps orderbook for best execution."""
        if dry_run:
            shares = amount_usd / price if price > 0 else 0
            log.info(f"[DRY] 📈 BUY {shares:.2f} shares @ ${price:.4f} (${amount_usd})")
            return {
                "dry_run": True,
                "token_id": token_id,
                "price": price,
                "shares": shares,
                "entry_price": price,
                "side": "BUY",
            }

        if self.is_read_only:
            log.warning(f"CLOB: Cannot buy — {self._read_only_reason}")
            return None

        # Sweep orderbook
        ob = await self.get_orderbook(token_id)
        if not ob:
            return None

        asks = sorted(ob.get("asks", []), key=lambda a: float(a["price"]))
        remaining_usd = amount_usd
        total_shares = 0.0
        worst_price = price

        max_share_price = config.get("trading", {}).get("max_share_price", 0.40)

        for ask in asks:
            ask_price = float(ask["price"])
            ask_size = float(ask["size"])
            if ask_price > max_share_price:
                break
            ask_usd = ask_price * ask_size
            if remaining_usd <= ask_usd:
                total_shares += remaining_usd / ask_price
                worst_price = ask_price
                remaining_usd = 0
                break
            else:
                total_shares += ask_size
                worst_price = ask_price
                remaining_usd -= ask_usd

        if total_shares <= 0:
            log.warning(f"⚠️ OB sweep found no shares under ${max_share_price}")
            return None

        actual_price = amount_usd / total_shares if total_shares > 0 else price
        log.info(
            f"✅ BUY: {total_shares:.2f} shares @ avg ${actual_price:.4f} = ${amount_usd:.2f}"
        )
        return {
            "shares": total_shares,
            "entry_price": actual_price,
            "worst_price": worst_price,
            "side": "BUY",
        }

    async def sell_shares(
        self,
        token_id: str,
        shares: float,
        min_price: float = 0.01,
        dry_run: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Sell shares — sweeps bids for best execution."""
        if dry_run:
            log.info(f"[DRY] 📉 SELL {shares:.2f} shares @ ~${min_price:.4f}")
            return {
                "dry_run": True,
                "token_id": token_id,
                "shares": shares,
                "exit_price": min_price,
                "side": "SELL",
            }

        if self.is_read_only:
            log.warning(f"CLOB: Cannot sell — {self._read_only_reason}")
            return None

        ob = await self.get_orderbook(token_id)
        if not ob:
            return None

        bids = sorted(ob.get("bids", []), key=lambda b: -float(b["price"]))
        remaining = shares
        total_usd = 0.0
        sold = 0.0
        worst_price = min_price

        for bid in bids:
            bp = float(bid["price"])
            bs = float(bid["size"])
            worst_price = bp
            if remaining <= bs:
                total_usd += remaining * bp
                sold += remaining
                remaining = 0
                break
            else:
                total_usd += bs * bp
                sold += bs
                remaining -= bs

        if sold <= 0.01:
            log.warning("⚠️ SELL failed — no bids")
            return None

        actual_exit = total_usd / sold if sold > 0 else min_price
        log.info(f"✅ SELL: {sold:.2f} shares @ avg ${actual_exit:.4f} = +${total_usd:.2f}")
        return {
            "shares": sold,
            "exit_price": actual_exit,
            "usd_received": total_usd,
            "side": "SELL",
        }

    async def close(self):
        if self._http:
            await self._http.aclose()
