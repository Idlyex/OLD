"""
Auto-redeem resolved Polymarket positions via the Relayer.

Checks every ~2 minutes for redeemable positions (resolved markets where
shares are still held) and claims them through Polymarket's gasless relayer.

Usage:
    - As standalone: python -m api.auto_redeem
    - Integrated:    from api.auto_redeem import redeem_all, start_auto_redeem_loop
"""

import os
import time
import asyncio
import logging
import requests
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from eth_abi import encode as eth_encode
from eth_utils import keccak

from py_builder_relayer_client.client import RelayClient
from py_builder_relayer_client.models import OperationType, SafeTransaction, TransactionType
from py_builder_signing_sdk.config import BuilderConfig, BuilderApiKeyCreds

load_dotenv()
log = logging.getLogger("auto_redeem")

# ═══════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════

REDEEM_INTERVAL_S = 120  # check every 2 minutes
RELAYER_RETRY_WAIT = 60  # wait on rate limit

# Contract addresses (Polygon)
USDC_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
CTF_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
NEG_RISK_ADAPTER = "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"
CTF_COLLATERAL_ADAPTER = "0xAdA100Db00Ca00073811820692005400218FcE1f"
NEG_RISK_COLLATERAL_ADAPTER = "0xadA2005600Dec949baf300f4C6120000bDB6eAab"

# Function selectors
REDEEM_SELECTOR = keccak(text="redeemPositions(address,bytes32,bytes32,uint256[])")[:4]
NEG_RISK_REDEEM_SELECTOR = keccak(text="redeemPositions(bytes32,uint256[])")[:4]
APPROVE_SELECTOR = keccak(text="setApprovalForAll(address,bool)")[:4]
IS_APPROVED_SELECTOR = keccak(text="isApprovedForAll(address,address)")[:4]


# ═══════════════════════════════════════════════════════════
#  REDEEM ALL
# ═══════════════════════════════════════════════════════════

def redeem_all(output_token: str = "pUSD") -> int:
    """Find and redeem all resolved positions. Returns count of redeemed positions."""
    ts = lambda: datetime.now().strftime("%H:%M:%S")

    private_key = os.getenv("PRIVATE_KEY", "")
    funder_address = os.getenv("PROXY_WALLET", "")
    builder_api_key = os.getenv("POLY_BUILDER_API_KEY", "")
    builder_secret = os.getenv("POLY_BUILDER_SECRET", "")
    builder_passphrase = os.getenv("POLY_BUILDER_PASSPHRASE", "")

    if not all([private_key, funder_address, builder_api_key, builder_secret, builder_passphrase]):
        log.warning("Auto-redeem: missing credentials (PRIVATE_KEY, PROXY_WALLET, POLY_BUILDER_*)")
        return 0

    # ── Client Setup ──
    try:
        client = RelayClient(
            "https://relayer-v2.polymarket.com",
            chain_id=137,
            private_key=private_key,
            builder_config=BuilderConfig(
                local_builder_creds=BuilderApiKeyCreds(
                    key=builder_api_key,
                    secret=builder_secret,
                    passphrase=builder_passphrase,
                )
            ),
        )
    except Exception as e:
        log.error(f"Auto-redeem: RelayClient init failed: {e}")
        return 0

    # ── Find Redeemable Positions ──
    try:
        response = requests.get(
            "https://data-api.polymarket.com/positions",
            params={"user": funder_address, "redeemable": "true", "sizeThreshold": 0},
            timeout=15,
        )
        if response.status_code in (429, 1015):
            log.warning(f"Auto-redeem: Data API rate limited, waiting {RELAYER_RETRY_WAIT}s...")
            time.sleep(RELAYER_RETRY_WAIT)
            response = requests.get(
                "https://data-api.polymarket.com/positions",
                params={"user": funder_address, "redeemable": "true", "sizeThreshold": 0},
                timeout=15,
            )
        positions = response.json()
    except Exception as e:
        log.error(f"Auto-redeem: Failed to fetch positions: {e}")
        return 0

    # Filter out zero-size positions
    positions = [p for p in positions if float(p.get("size", 0)) > 0]
    if not positions:
        return 0

    log.info(f"Auto-redeem: Found {len(positions)} redeemable positions")

    # ── Pre-flight: check approvals for pUSD adapters ──
    rpc_url = os.getenv("RPC_URL", "https://polygon-rpc.com")

    def _is_approved(adapter: str) -> bool:
        try:
            args = eth_encode(
                ["address", "address"],
                [bytes.fromhex(funder_address[2:]) if isinstance(funder_address, str) and funder_address.startswith("0x") else funder_address,
                 bytes.fromhex(adapter[2:])]
            )
        except Exception:
            # Use string addresses directly
            args = eth_encode(["address", "address"], [funder_address, adapter])
        payload = {
            "jsonrpc": "2.0",
            "method": "eth_call",
            "id": 1,
            "params": [
                {"to": CTF_ADDRESS, "data": "0x" + (IS_APPROVED_SELECTOR + args).hex()},
                "latest",
            ],
        }
        try:
            r = requests.post(rpc_url, json=payload, timeout=10).json()
            return int(r["result"], 16) == 1
        except Exception:
            return False

    def _approve_txn(adapter: str) -> SafeTransaction:
        args = eth_encode(["address", "bool"], [adapter, True])
        return SafeTransaction(
            to=CTF_ADDRESS,
            operation=OperationType.Call,
            data="0x" + (APPROVE_SELECTOR + args).hex(),
            value="0",
        )

    approved_adapters = set()
    if output_token == "pUSD":
        for adapter in (CTF_COLLATERAL_ADAPTER, NEG_RISK_COLLATERAL_ADAPTER):
            if _is_approved(adapter):
                approved_adapters.add(adapter)

    # ── Redeem Each Position ──
    redeemed = 0
    for pos in positions:
        cid = pos.get("conditionId", pos.get("condition_id", ""))
        if not cid:
            continue
        if not cid.startswith("0x"):
            cid = "0x" + cid

        market = pos.get("title", cid[:12])

        try:
            condition_bytes = bytes.fromhex(cid[2:])
            neg_risk = pos.get("negativeRisk")

            if neg_risk is True:
                if output_token == "pUSD":
                    args = eth_encode(
                        ["address", "bytes32", "bytes32", "uint256[]"],
                        [USDC_ADDRESS, b"\x00" * 32, condition_bytes, [1, 2]],
                    )
                    txn = SafeTransaction(
                        to=NEG_RISK_COLLATERAL_ADAPTER,
                        operation=OperationType.Call,
                        data="0x" + (REDEEM_SELECTOR + args).hex(),
                        value="0",
                    )
                    adapter_for_approval = NEG_RISK_COLLATERAL_ADAPTER
                else:
                    size_raw = int(float(pos.get("size", 0)) * 1e6)
                    outcome_index = int(pos.get("outcomeIndex", 0))
                    amounts = [0, 0]
                    amounts[outcome_index] = size_raw
                    args = eth_encode(["bytes32", "uint256[]"], [condition_bytes, amounts])
                    txn = SafeTransaction(
                        to=NEG_RISK_ADAPTER,
                        operation=OperationType.Call,
                        data="0x" + (NEG_RISK_REDEEM_SELECTOR + args).hex(),
                        value="0",
                    )
                    adapter_for_approval = None

            elif neg_risk is False:
                args = eth_encode(
                    ["address", "bytes32", "bytes32", "uint256[]"],
                    [USDC_ADDRESS, b"\x00" * 32, condition_bytes, [1, 2]],
                )
                txn = SafeTransaction(
                    to=CTF_COLLATERAL_ADAPTER if output_token == "pUSD" else CTF_ADDRESS,
                    operation=OperationType.Call,
                    data="0x" + (REDEEM_SELECTOR + args).hex(),
                    value="0",
                )
                adapter_for_approval = CTF_COLLATERAL_ADAPTER if output_token == "pUSD" else None
            else:
                log.warning(f"Auto-redeem: Skipping {market}: unsupported type (negativeRisk={neg_risk!r})")
                continue

            # Build call list (include approval if needed)
            calls = []
            if adapter_for_approval and adapter_for_approval not in approved_adapters:
                calls.append(_approve_txn(adapter_for_approval))
                approved_adapters.add(adapter_for_approval)
            calls.append(txn)

            try:
                resp = client.execute(calls, f"redeem {cid[:12]}")
                resp.wait()
            except Exception as relay_err:
                status = getattr(relay_err, "status_code", None)
                if status in (429, 1015):
                    log.warning(f"Auto-redeem: Relayer rate limited, waiting {RELAYER_RETRY_WAIT}s...")
                    time.sleep(RELAYER_RETRY_WAIT)
                    resp = client.execute(calls, f"redeem {cid[:12]}")
                    resp.wait()
                else:
                    raise

            redeemed += 1
            size = float(pos.get("size", 0))
            log.info(f"Auto-redeem: ✅ Redeemed: {market} ({size:.2f} shares)")

        except Exception as e:
            log.warning(f"Auto-redeem: ❌ Failed to redeem {market}: {e}")

    if redeemed > 0:
        log.info(f"Auto-redeem: Redeemed {redeemed}/{len(positions)} positions")
    return redeemed


# ═══════════════════════════════════════════════════════════
#  ASYNC LOOP (for integration with trader)
# ═══════════════════════════════════════════════════════════

async def start_auto_redeem_loop(interval_s: float = REDEEM_INTERVAL_S):
    """Run redeem_all() every `interval_s` seconds as an async background task."""
    log.info(f"Auto-redeem loop started (every {interval_s}s)")
    while True:
        try:
            # Run blocking redeem_all in executor to not block event loop
            loop = asyncio.get_event_loop()
            count = await loop.run_in_executor(None, redeem_all)
            if count > 0:
                log.info(f"Auto-redeem cycle: claimed {count} positions")
        except Exception as e:
            log.warning(f"Auto-redeem cycle error: {e}")
        await asyncio.sleep(interval_s)


# ═══════════════════════════════════════════════════════════
#  STANDALONE
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    print("Auto-redeem: checking for redeemable positions...")
    count = redeem_all()
    print(f"Done. Redeemed {count} positions.")
