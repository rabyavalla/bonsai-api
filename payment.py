"""
B.O.N.S.A.I. x402 Payment Verification Middleware
===================================================
Implements Coinbase's x402 protocol for USDC micropayments on Base.

Flow:
1. Agent calls endpoint without X-Payment header
2. Server returns 402 with payment requirements (wallet, price, network)
3. Agent pays via USDC on Base, gets tx hash
4. Agent retries with X-Payment header containing the tx hash
5. Server verifies payment on-chain, serves the response

References:
- https://www.x402.org/
- Coinbase x402 spec: payment header = base64-encoded JSON with tx hash
"""

import json
import base64
import time
from typing import Optional

import structlog
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from config import get_settings

logger = structlog.get_logger()

# ── Tier pricing map (endpoint path → price in USDC) ─────────────────────────

ENDPOINT_PRICING = {
    "/v1/query": 0.02,
    "/v1/protocol": 0.20,
    "/v1/lab-protocol": 0.45,
    "/v1/assess": 0.85,
}


def build_402_response(endpoint: str, wallet_address: str) -> JSONResponse:
    """Return a 402 Payment Required response per x402 spec."""
    price = ENDPOINT_PRICING.get(endpoint, 0.02)

    payment_requirements = {
        "x402Version": 1,
        "accepts": [
            {
                "scheme": "exact",
                "network": "base",
                "maxAmountRequired": str(int(price * 1_000_000)),  # USDC has 6 decimals
                "resource": endpoint,
                "description": f"B.O.N.S.A.I. health intelligence — {endpoint}",
                "mimeType": "application/json",
                "payTo": wallet_address,
                "maxTimeoutSeconds": 300,
                "asset": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",  # USDC on Base
                "extra": {
                    "name": "B.O.N.S.A.I. Health Intelligence API",
                    "pricing": f"${price} USDC",
                },
            }
        ],
    }

    return JSONResponse(
        status_code=402,
        content=payment_requirements,
        headers={"Content-Type": "application/json"},
    )


def decode_payment_header(header_value: str) -> Optional[dict]:
    """Decode the X-Payment header (base64-encoded JSON per x402 spec)."""
    try:
        decoded = base64.b64decode(header_value)
        return json.loads(decoded)
    except Exception:
        # Try plain JSON (some agents send it unencoded)
        try:
            return json.loads(header_value)
        except Exception:
            return None


async def verify_payment_onchain(payment_data: dict, expected_amount: float, wallet_address: str) -> bool:
    """
    Verify USDC payment on Base chain.

    In production, this checks:
    1. Transaction exists and is confirmed
    2. Recipient matches our wallet
    3. Amount >= expected price
    4. Token is USDC on Base
    5. Transaction is recent (within 5 minutes)

    For MVP, we use a simplified check. In production, wire this to
    web3.py or the Coinbase x402 facilitator SDK.
    """
    settings = get_settings()

    if settings.skip_payment_verification:
        logger.info("payment_verification_skipped", reason="dev_mode")
        return True

    tx_hash = payment_data.get("transaction") or payment_data.get("txHash") or payment_data.get("payload", {}).get("transaction")

    if not tx_hash:
        logger.warning("payment_missing_tx_hash", data=payment_data)
        return False

    try:
        from web3 import Web3

        w3 = Web3(Web3.HTTPProvider(settings.base_rpc_url))

        # Get transaction receipt
        receipt = w3.eth.get_transaction_receipt(tx_hash)
        if receipt is None:
            logger.warning("payment_tx_not_found", tx_hash=tx_hash)
            return False

        # Check confirmation
        if receipt.status != 1:
            logger.warning("payment_tx_failed", tx_hash=tx_hash)
            return False

        # Check the USDC Transfer event
        # USDC Transfer topic: keccak256("Transfer(address,address,uint256)")
        transfer_topic = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
        usdc_address = settings.usdc_contract_address.lower()

        for log_entry in receipt.logs:
            if log_entry.address.lower() != usdc_address:
                continue
            if len(log_entry.topics) < 3:
                continue
            if log_entry.topics[0].hex() != transfer_topic:
                continue

            # Decode recipient (topic[2] = to address, zero-padded)
            to_address = "0x" + log_entry.topics[2].hex()[-40:]
            if to_address.lower() != wallet_address.lower():
                continue

            # Decode amount (USDC has 6 decimals)
            amount_raw = int(log_entry.data.hex(), 16)
            amount_usdc = amount_raw / 1_000_000
            expected_micro = expected_amount

            if amount_usdc >= expected_micro:
                logger.info(
                    "payment_verified",
                    tx_hash=tx_hash,
                    amount=amount_usdc,
                    expected=expected_micro,
                )
                return True

        logger.warning("payment_amount_insufficient_or_wrong_recipient", tx_hash=tx_hash)
        return False

    except ImportError:
        logger.error("web3_not_installed", msg="pip install web3")
        # In development without web3, check if skip is enabled
        if settings.skip_payment_verification:
            return True
        return False
    except Exception as e:
        logger.error("payment_verification_error", error=str(e), tx_hash=tx_hash)
        return False


class X402PaymentMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware implementing x402 payment gating.

    - Non-API routes pass through
    - API routes without X-Payment header get a 402 response
    - API routes with X-Payment header get verified on-chain
    """

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Only gate /v1/* endpoints
        if not path.startswith("/v1/"):
            return await call_next(request)

        # Health check passthrough
        if path == "/v1/health":
            return await call_next(request)

        settings = get_settings()
        expected_price = ENDPOINT_PRICING.get(path)

        if expected_price is None:
            return await call_next(request)

        # Check for X-Payment header
        payment_header = request.headers.get("X-Payment")

        if not payment_header:
            logger.info("payment_required", endpoint=path, price=expected_price)
            return build_402_response(path, settings.bonsai_wallet_address)

        # Decode and verify payment
        payment_data = decode_payment_header(payment_header)
        if payment_data is None:
            raise HTTPException(status_code=400, detail="Invalid X-Payment header format")

        verified = await verify_payment_onchain(
            payment_data, expected_price, settings.bonsai_wallet_address
        )

        if not verified:
            raise HTTPException(status_code=402, detail="Payment verification failed")

        # Payment verified — proceed
        logger.info("payment_accepted", endpoint=path, price=expected_price)
        response = await call_next(request)
        return response
