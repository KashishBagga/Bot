"""
Thin wrapper around real Fyers order-placement calls.

First-cut scope: Market entry orders and SL-M stop orders only, at small
(1-lot) size. No order modification, no combo/multi-leg orders. Every call
here places or cancels a REAL order — this module must never be imported by
anything on the counterfactual/shadow-trade path.
"""
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Fyers v3 side / order-type codes
SIDE_BUY = 1
SIDE_SELL = -1
ORDER_TYPE_MARKET = 2
ORDER_TYPE_SL_M = 3


@dataclass
class OrderResult:
    success: bool
    order_id: Optional[str] = None
    fill_price: Optional[float] = None
    status: Optional[str] = None
    message: Optional[str] = None


class FyersOrderExecutor:
    """Places real orders through an initialized FyersClient's underlying fyersModel instance."""

    def __init__(self, fyers_client):
        self.fyers_client = fyers_client

    def place(self, symbol, qty, side, order_type="MARKET", stop_price=None, product_type="INTRADAY"):
        """Place a real order. side: 'BUY'/'SELL'. Returns OrderResult, never raises."""
        if not self.fyers_client or not self.fyers_client.fyers:
            return OrderResult(success=False, message="Fyers client not initialized")

        fyers_side = SIDE_BUY if side.upper() == "BUY" else SIDE_SELL
        fyers_type = ORDER_TYPE_SL_M if order_type.upper() == "SL-M" else ORDER_TYPE_MARKET

        data = {
            "symbol": symbol,
            "qty": int(qty),
            "type": fyers_type,
            "side": fyers_side,
            "productType": product_type,
            "limitPrice": 0,
            "stopPrice": float(stop_price) if stop_price is not None else 0,
            "validity": "DAY",
            "disclosedQty": 0,
            "offlineOrder": False,
        }

        try:
            self.fyers_client._rate_limit()
            response = self.fyers_client.fyers.place_order(data)
        except Exception as e:
            logger.error(f"❌ Live order placement raised for {symbol}: {e}")
            return OrderResult(success=False, message=str(e))

        if response and response.get("s") == "ok":
            order_id = response.get("id")
            logger.info(f"✅ Live order placed: {symbol} qty={qty} side={side} id={order_id}")
            status = self.get_order_status(order_id) if order_id else {}
            return OrderResult(
                success=True,
                order_id=order_id,
                fill_price=status.get("tradedPrice"),
                status=status.get("status"),
            )

        message = response.get("message") if response else "no response from Fyers API"
        logger.error(f"❌ Live order rejected: {symbol} qty={qty} side={side} — {message}")
        return OrderResult(success=False, message=message)

    def cancel(self, order_id) -> bool:
        if not self.fyers_client or not self.fyers_client.fyers:
            return False
        try:
            self.fyers_client._rate_limit()
            response = self.fyers_client.fyers.cancel_order({"id": order_id})
            return bool(response and response.get("s") == "ok")
        except Exception as e:
            logger.error(f"❌ Live order cancel raised for {order_id}: {e}")
            return False

    def get_order_status(self, order_id) -> dict:
        if not self.fyers_client or not self.fyers_client.fyers:
            return {}
        try:
            self.fyers_client._rate_limit()
            response = self.fyers_client.fyers.orderbook({"id": order_id})
            if not response or response.get("s") != "ok":
                return {}
            orders = response.get("orderBook", [])
            return orders[0] if orders else {}
        except Exception as e:
            logger.error(f"❌ Live order status lookup raised for {order_id}: {e}")
            return {}
