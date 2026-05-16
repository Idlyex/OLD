"""Polymarket CLOB Client V2 -- migrated for CLOB V2 (April 28 2026).

Uses py-clob-client-v2 with new exchange contracts, EIP-712 v2 domain,
and FAK market orders (same approach as POLYx JS client).

Usage:
    clob = PolymarketCLOB()
    await clob.init()
    result = await clob.buy_market(token_id, price=0.50, amount_usd=2.0)
    result = await clob.sell_market(token_id, shares=4.0, worst_price=0.90)
    balance = await clob.get_balance()
"""

import os
import time
import asyncio
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from dotenv import load_dotenv
load_dotenv()

log = logging.getLogger("clob")

CLOB_HOST = "https://clob.polymarket.com"
CHAIN_ID = 137  # Polygon mainnet
TICK_SIZE = "0.01"


@dataclass
class OrderResult:
    """Result of a CLOB order."""
    success: bool = False
    order_id: str = ""
    side: str = ""
    token_id: str = ""
    price: float = 0.0
    shares: float = 0.0
    amount_usd: float = 0.0
    dry_run: bool = False
    error: str = ""
    raw: Optional[Dict] = None


class PolymarketCLOB:
    """Polymarket CLOB V2 trading client.

    - buy_limit:  GTC limit buy  (maker = 0% fee)
    - buy_market: FAK market buy (immediate fill)
    - sell_limit: GTC limit sell (maker = 0% fee)
    - sell_market: FAK market sell (urgent exits)
    - get_balance: pUSD balance
    - get_share_balance: conditional token balance
    - get_orderbook: fetch order book for a token
    """

    def __init__(self, dry_run: bool = True):
        self._client = None
        self._anon_client = None
        self._initialized = False
        self._read_only_reason = None
        self._wallet_address = None
        self._proxy_wallet = None
        self._dry_run = dry_run
        self._balance_cache = None
        self._balance_cache_ts = 0

    async def init(self):
        """Initialize CLOB V2 client -- derive API key."""
        if self._initialized:
            return

        try:
            from py_clob_client_v2.client import ClobClient

            pk = os.getenv("PRIVATE_KEY", "")
            if not pk or pk == "0x_YOUR_PRIVATE_KEY_HERE":
                self._read_only_reason = "No PRIVATE_KEY set in .env"
                log.warning("CLOB: %s -- read-only mode", self._read_only_reason)
                self._initialized = True
                return

            self._proxy_wallet = os.getenv("PROXY_WALLET", None) or None
            sig_type = 2 if self._proxy_wallet else None

            temp_client = ClobClient(CLOB_HOST, key=pk, chain_id=CHAIN_ID)
            try:
                self._api_creds = temp_client.derive_api_key()
                log.info("CLOB V2: API key derived")
            except Exception:
                try:
                    self._api_creds = temp_client.create_or_derive_api_key()
                    log.info("CLOB V2: API key created")
                except Exception as e:
                    if "400" in str(e):
                        self._read_only_reason = "Wallet not activated on polymarket.com"
                        log.warning("CLOB: %s", self._read_only_reason)
                        self._initialized = True
                        return
                    raise

            self._client = ClobClient(
                CLOB_HOST,
                key=pk,
                chain_id=CHAIN_ID,
                creds=self._api_creds,
                signature_type=sig_type,
                funder=self._proxy_wallet,
            )

            self._anon_client = ClobClient(CLOB_HOST, chain_id=CHAIN_ID)

            self._wallet_address = self._client.get_address()
            log.info("CLOB V2: Wallet %s", self._wallet_address)
            if self._proxy_wallet:
                log.info("CLOB V2: Proxy  %s", self._proxy_wallet)

            self._initialized = True
            log.info("CLOB V2: Client ready (server version: %s)", self._client.get_version())

        except Exception as e:
            self._read_only_reason = str(e)
            log.error("CLOB V2 init error: %s", e)
            self._initialized = True

    @property
    def is_read_only(self) -> bool:
        return self._client is None

    @property
    def wallet_address(self) -> Optional[str]:
        return self._wallet_address

    # ---------------------------------------------------------------
    #  BUY LIMIT (GTC -- maker = 0% fee)
    # ---------------------------------------------------------------

    async def buy_limit(self, token_id: str, price: float, amount_usd: float,
                        neg_risk: bool = False) -> OrderResult:
        """Place a GTC limit buy order. Maker fee = 0%.

        Args:
            token_id: Token to buy (yes_token_id or no_token_id)
            price: Limit price (0.01 - 0.99)
            amount_usd: Dollar amount to spend
            neg_risk: Whether market uses neg risk
        """
        shares = round(amount_usd / price, 2)

        if self._dry_run:
            log.info("[DRY] BUY %s shares @ $%.3f ($%.2f)", shares, price, amount_usd)
            return OrderResult(
                success=True, side="BUY", token_id=token_id,
                price=price, shares=shares, amount_usd=amount_usd, dry_run=True
            )

        if self.is_read_only:
            log.warning("CLOB: Cannot buy -- %s", self._read_only_reason)
            return OrderResult(success=False, error=self._read_only_reason or "read-only")

        try:
            from py_clob_client_v2.clob_types import OrderArgsV2, PartialCreateOrderOptions
            from py_clob_client_v2.order_builder.constants import BUY

            order_args = OrderArgsV2(
                token_id=token_id,
                price=price,
                size=shares,
                side=BUY,
            )
            options = PartialCreateOrderOptions(
                tick_size=TICK_SIZE,
                neg_risk=neg_risk,
            )

            result = self._client.create_and_post_order(order_args, options)

            order_id = ""
            if isinstance(result, dict):
                order_id = result.get("orderID", result.get("id", ""))
                if result.get("success") is False:
                    error_msg = result.get("errorMsg", "Order failed")
                    log.error("CLOB BUY error: %s", error_msg)
                    return OrderResult(success=False, error=error_msg, raw=result)

            log.info("BUY LIMIT: %.2f shares @ $%.4f = $%.2f [%s]", shares, price, amount_usd, order_id)
            return OrderResult(
                success=True, order_id=order_id, side="BUY",
                token_id=token_id, price=price, shares=shares,
                amount_usd=amount_usd, raw=result,
            )

        except Exception as e:
            error_msg = str(e)
            if "balance" in error_msg.lower():
                log.warning("Insufficient balance")
            else:
                log.error("CLOB BUY error: %s", error_msg[:200])
            return OrderResult(success=False, error=error_msg)

    # ---------------------------------------------------------------
    #  PRE-SIGN + FAST POST (split create_order & post_order)
    # ---------------------------------------------------------------

    def presign_buy(self, token_id: str, price: float, amount_usd: float,
                    neg_risk: bool = False):
        """Pre-build and sign a GTC buy order (offline, ~50ms).
        Returns the signed order object, ready for instant posting.
        Call post_presigned() when the signal fires.
        """
        if self.is_read_only or self._dry_run:
            return None
        try:
            from py_clob_client_v2.clob_types import OrderArgsV2, PartialCreateOrderOptions
            from py_clob_client_v2.order_builder.constants import BUY

            shares = round(amount_usd / price, 2)
            order_args = OrderArgsV2(
                token_id=token_id,
                price=price,
                size=shares,
                side=BUY,
            )
            options = PartialCreateOrderOptions(
                tick_size=TICK_SIZE,
                neg_risk=neg_risk,
            )
            signed_order = self._client.create_order(order_args, options)
            return signed_order
        except Exception as e:
            log.error("Presign error: %s", str(e)[:200])
            return None

    async def post_presigned(self, signed_order, token_id: str = "",
                             price: float = 0, shares: float = 0,
                             amount_usd: float = 0) -> OrderResult:
        """Post a pre-signed order (fast, ~20ms network only)."""
        if self.is_read_only:
            return OrderResult(success=False, error="read-only")
        if signed_order is None:
            return OrderResult(success=False, error="no signed order")
        try:
            result = self._client.post_order(signed_order)
            order_id = ""
            if isinstance(result, dict):
                order_id = result.get("orderID", result.get("id", ""))
                if result.get("success") is False:
                    error_msg = result.get("errorMsg", "Order failed")
                    log.error("POST presigned error: %s", error_msg)
                    return OrderResult(success=False, error=error_msg, raw=result)

            log.info("POST PRESIGNED: %.2f shares @ $%.4f [%s]", shares, price, order_id)
            return OrderResult(
                success=True, order_id=order_id, side="BUY",
                token_id=token_id, price=price, shares=shares,
                amount_usd=amount_usd, raw=result,
            )
        except Exception as e:
            log.error("POST presigned error: %s", str(e)[:200])
            return OrderResult(success=False, error=str(e))

    # ---------------------------------------------------------------
    #  BUY MARKET (FAK -- immediate fill, same as POLYx JS)
    # ---------------------------------------------------------------

    async def buy_market(self, token_id: str, price: float, amount_usd: float,
                         neg_risk: bool = False) -> OrderResult:
        """Buy shares for exact USD amount using FAK (Fill-And-Kill).

        FAK fills what's available at price levels up to max price,
        kills unfilled remainder. Average price <= max price guaranteed.

        Args:
            token_id: Token to buy
            price: Max price per share
            amount_usd: Exact USD to spend
            neg_risk: Whether market uses neg risk
        """
        shares = round(amount_usd / price, 2)

        if self._dry_run:
            log.info("[DRY] BUY MARKET %s shares @ max $%.3f ($%.2f)", shares, price, amount_usd)
            return OrderResult(
                success=True, side="BUY", token_id=token_id,
                price=price, shares=shares, amount_usd=amount_usd, dry_run=True
            )

        if self.is_read_only:
            log.warning("CLOB: Cannot buy -- %s", self._read_only_reason)
            return OrderResult(success=False, error=self._read_only_reason or "read-only")

        try:
            from py_clob_client_v2.clob_types import MarketOrderArgsV2, OrderType, PartialCreateOrderOptions
            from py_clob_client_v2.order_builder.constants import BUY

            order_args = MarketOrderArgsV2(
                token_id=token_id,
                amount=amount_usd,
                side=BUY,
                price=price,
                order_type=OrderType.FAK,
            )
            options = PartialCreateOrderOptions(
                tick_size=TICK_SIZE,
                neg_risk=neg_risk,
            )

            result = self._client.create_and_post_market_order(
                order_args, options, order_type=OrderType.FAK
            )

            order_id = ""
            if isinstance(result, dict):
                order_id = result.get("orderID", result.get("id", ""))
                if result.get("success") is False:
                    error_msg = result.get("errorMsg", "Order failed")
                    log.error("CLOB BUY MARKET error: %s", error_msg)
                    return OrderResult(success=False, error=error_msg, raw=result)

            log.info("BUY FAK: %.2f shares @ max $%.4f = $%.2f [%s]",
                     shares, price, amount_usd, order_id)
            return OrderResult(
                success=True, order_id=order_id, side="BUY",
                token_id=token_id, price=price, shares=shares,
                amount_usd=amount_usd, raw=result,
            )

        except Exception as e:
            error_msg = str(e)
            if "balance" in error_msg.lower():
                log.warning("Insufficient balance for $%.2f", amount_usd)
            else:
                log.error("CLOB BUY error: %s", error_msg[:200])
            return OrderResult(success=False, error=error_msg)

    # ---------------------------------------------------------------
    #  SELL LIMIT (GTC -- maker = 0% fee)
    # ---------------------------------------------------------------

    async def sell_limit(self, token_id: str, price: float, shares: float,
                         neg_risk: bool = False) -> OrderResult:
        """Place a GTC limit sell order. Maker fee = 0%."""
        if self._dry_run:
            log.info("[DRY] SELL %.2f shares @ $%.3f", shares, price)
            return OrderResult(
                success=True, side="SELL", token_id=token_id,
                price=price, shares=shares, amount_usd=shares * price, dry_run=True
            )

        if self.is_read_only:
            log.warning("CLOB: Cannot sell -- %s", self._read_only_reason)
            return OrderResult(success=False, error=self._read_only_reason or "read-only")

        try:
            from py_clob_client_v2.clob_types import OrderArgsV2, PartialCreateOrderOptions
            from py_clob_client_v2.order_builder.constants import SELL

            order_args = OrderArgsV2(
                token_id=token_id,
                price=price,
                size=shares,
                side=SELL,
            )
            options = PartialCreateOrderOptions(
                tick_size=TICK_SIZE,
                neg_risk=neg_risk,
            )

            result = self._client.create_and_post_order(order_args, options)

            order_id = ""
            if isinstance(result, dict):
                order_id = result.get("orderID", result.get("id", ""))
                if result.get("success") is False:
                    error_msg = result.get("errorMsg", "Order failed")
                    log.error("CLOB SELL error: %s", error_msg)
                    return OrderResult(success=False, error=error_msg, raw=result)

            amount = shares * price
            log.info("SELL LIMIT: %.2f shares @ $%.4f = $%.2f [%s]", shares, price, amount, order_id)
            return OrderResult(
                success=True, order_id=order_id, side="SELL",
                token_id=token_id, price=price, shares=shares,
                amount_usd=amount, raw=result,
            )

        except Exception as e:
            log.error("CLOB SELL error: %s", str(e)[:200])
            return OrderResult(success=False, error=str(e))

    # ---------------------------------------------------------------
    #  SELL MARKET (FAK -- immediate fill)
    # ---------------------------------------------------------------

    async def sell_market(self, token_id: str, shares: float,
                          worst_price: float = 0.01,
                          neg_risk: bool = False) -> OrderResult:
        """Place a FAK market sell. For urgent exits (loss cuts, reversals)."""
        if self._dry_run:
            log.info("[DRY] MARKET SELL %.2f shares @ min $%.3f", shares, worst_price)
            return OrderResult(
                success=True, side="SELL", token_id=token_id,
                price=worst_price, shares=shares,
                amount_usd=shares * worst_price, dry_run=True
            )

        if self.is_read_only:
            log.warning("CLOB: Cannot sell -- %s", self._read_only_reason)
            return OrderResult(success=False, error=self._read_only_reason or "read-only")

        try:
            from py_clob_client_v2.clob_types import MarketOrderArgsV2, OrderType, PartialCreateOrderOptions
            from py_clob_client_v2.order_builder.constants import SELL

            order_args = MarketOrderArgsV2(
                token_id=token_id,
                amount=shares,
                side=SELL,
                price=worst_price,
                order_type=OrderType.FAK,
            )
            options = PartialCreateOrderOptions(
                tick_size=TICK_SIZE,
                neg_risk=neg_risk,
            )

            result = self._client.create_and_post_market_order(
                order_args, options, order_type=OrderType.FAK
            )

            order_id = ""
            if isinstance(result, dict):
                order_id = result.get("orderID", result.get("id", ""))

            log.info("MARKET SELL (FAK): %.2f shares @ min $%.4f [%s]", shares, worst_price, order_id)
            return OrderResult(
                success=True, order_id=order_id, side="SELL",
                token_id=token_id, price=worst_price, shares=shares,
                amount_usd=shares * worst_price, raw=result,
            )

        except Exception as e:
            log.error("CLOB MARKET SELL error: %s", str(e)[:200])
            return OrderResult(success=False, error=str(e))

    # ---------------------------------------------------------------
    #  BALANCE / ORDERBOOK / ORDER MANAGEMENT
    # ---------------------------------------------------------------


    async def sell_limit_99(self, token_id: str, shares: float,
                            neg_risk: bool = False) -> OrderResult:
        """Place limit sell at $0.99 for winning shares (0% maker fee).
        
        Someone buys winning shares at $0.99 before resolution.
        Avoids needing to claim on-chain.
        """
        if self._dry_run:
            log.info("[DRY] SELL LIMIT $0.99 x %.2f shares", shares)
            return OrderResult(
                success=True, side="SELL", token_id=token_id,
                price=0.99, shares=shares, amount_usd=round(shares * 0.99, 2), dry_run=True
            )

        if self.is_read_only:
            return OrderResult(success=False, error="read-only")

        try:
            from py_clob_client_v2.clob_types import OrderArgsV2, OrderType, PartialCreateOrderOptions
            from py_clob_client_v2.order_builder.constants import SELL

            order_args = OrderArgsV2(
                token_id=token_id,
                price=0.99,
                size=round(shares, 2),
                side=SELL,
            )
            options = PartialCreateOrderOptions(tick_size=TICK_SIZE, neg_risk=neg_risk)
            result = self._client.create_and_post_order(
                order_args, options, order_type=OrderType.GTC
            )

            order_id = ""
            if isinstance(result, dict):
                order_id = result.get("orderID", result.get("id", ""))
                if result.get("success") is False:
                    return OrderResult(success=False, error=result.get("errorMsg", ""), raw=result)

            log.info("SELL LIMIT $0.99 x %.2f shares [%s]", shares, order_id)
            return OrderResult(
                success=True, order_id=order_id, side="SELL",
                token_id=token_id, price=0.99, shares=shares,
                amount_usd=round(shares * 0.99, 2), raw=result,
            )
        except Exception as e:
            log.error("SELL LIMIT error: %s", str(e)[:200])
            return OrderResult(success=False, error=str(e))

    async def get_balance(self) -> Optional[float]:
        """Get pUSD/USDC balance from CLOB API."""
        if self.is_read_only:
            return None
        try:
            from py_clob_client_v2.clob_types import BalanceAllowanceParams, AssetType
            params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
            result = self._client.get_balance_allowance(params)
            bal = float(result.get("balance", "0")) / 1e6  # USDC has 6 decimals
            self._balance_cache = bal
            self._balance_cache_ts = time.time()
            return bal
        except Exception as e:
            log.error("Balance error: %s", e)
            return self._balance_cache

    async def get_share_balance(self, token_id: str) -> float:
        """Get conditional token (share) balance for a specific position."""
        if self.is_read_only:
            return 0.0
        try:
            from py_clob_client_v2.clob_types import BalanceAllowanceParams, AssetType
            params = BalanceAllowanceParams(
                asset_type=AssetType.CONDITIONAL, token_id=token_id
            )
            result = self._client.get_balance_allowance(params)
            return float(result.get("balance", "0")) / 1e6
        except Exception as e:
            log.error("Share balance error: %s", e)
            return 0.0

    async def get_orderbook(self, token_id: str) -> Optional[Dict]:
        """Fetch orderbook for a token. No auth needed."""
        try:
            if not self._anon_client:
                from py_clob_client_v2.client import ClobClient
                self._anon_client = ClobClient(CLOB_HOST, chain_id=CHAIN_ID)
            book = self._anon_client.get_order_book(token_id)
            return book
        except Exception as e:
            log.error("Orderbook error: %s", e)
            return None

    async def get_midpoint(self, token_id: str) -> Optional[float]:
        """Get midpoint price for a token. No auth needed."""
        try:
            if not self._anon_client:
                from py_clob_client_v2.client import ClobClient
                self._anon_client = ClobClient(CLOB_HOST, chain_id=CHAIN_ID)
            mid = self._anon_client.get_midpoint(token_id)
            return float(mid) if mid else None
        except Exception as e:
            log.error("Midpoint error: %s", e)
            return None

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a specific open order."""
        if self._dry_run or self.is_read_only:
            return True
        try:
            from py_clob_client_v2.clob_types import OrderPayload
            self._client.cancel_order(OrderPayload(orderID=order_id))
            log.info("Order cancelled: %s", order_id)
            return True
        except Exception as e:
            log.error("Cancel error: %s", e)
            return False

    async def cancel_all(self) -> bool:
        """Cancel all open orders."""
        if self._dry_run or self.is_read_only:
            return True
        try:
            self._client.cancel_all()
            log.info("All orders cancelled")
            return True
        except Exception as e:
            log.error("Cancel all error: %s", e)
            return False

    async def get_open_orders(self) -> List[Dict]:
        """Get all open orders."""
        if self.is_read_only:
            return []
        try:
            orders = self._client.get_open_orders()
            return orders if isinstance(orders, list) else []
        except Exception as e:
            log.error("Open orders error: %s", e)
            return []

    async def wait_for_fill(self, token_id: str, expected_shares: float,
                            baseline: float = 0.0,
                            timeout_s: float = 10.0,
                            poll_interval: float = 0.5) -> float:
        """Wait for a buy order to fill by polling share balance."""
        if self._dry_run:
            return expected_shares

        deadline = time.time() + timeout_s
        while time.time() < deadline:
            bal = await self.get_share_balance(token_id)
            delta = bal - baseline
            if delta > 0.001:
                return round(delta, 2)
            await asyncio.sleep(poll_interval)

        log.warning("Fill timeout (%ss) -- balance didn't increase", timeout_s)
        return 0.0

    # ---------------------------------------------------------------
    #  APPROVAL
    # ---------------------------------------------------------------

    async def approve_usdc(self) -> bool:
        """Approve pUSD/USDC spending for CTF contract."""
        if self.is_read_only:
            log.warning("CLOB: Cannot approve -- read-only")
            return False
        try:
            from py_clob_client_v2.clob_types import BalanceAllowanceParams, AssetType
            params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
            self._client.update_balance_allowance(params)
            log.info("CLOB V2: pUSD approved")
            return True
        except Exception as e:
            log.error("pUSD approval failed: %s", e)
            return False
