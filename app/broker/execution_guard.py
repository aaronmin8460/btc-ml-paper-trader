from dataclasses import dataclass
from typing import Literal

from app.config import ALLOWED_SYMBOL
from app.monitoring.logger import get_logger

OrderSide = Literal["buy", "sell"]


class BTCOnlyViolation(ValueError):
    pass


class LongOnlyViolation(ValueError):
    pass


@dataclass(frozen=True)
class GuardedOrder:
    symbol: str
    side: OrderSide
    notional: float | None = None
    qty: float | None = None


def assert_btc_only(symbol: str, context: str = "unknown") -> str:
    if symbol != ALLOWED_SYMBOL:
        get_logger().event("rejected_order", symbol=symbol, reason="non_btc_symbol", context=context)
        raise BTCOnlyViolation(f"Blocked symbol {symbol!r}; only {ALLOWED_SYMBOL} is allowed.")
    return symbol


def validate_order_request(
    symbol: str,
    side: str,
    *,
    notional: float | None = None,
    qty: float | None = None,
    current_position_qty: float = 0.0,
    context: str = "order_request",
) -> GuardedOrder:
    assert_btc_only(symbol, context=context)
    if side not in {"buy", "sell"}:
        get_logger().event("rejected_order", symbol=symbol, side=side, reason="invalid_side", context=context)
        raise LongOnlyViolation("Only buy and sell are allowed for long-only BTC.")
    if side == "sell" and current_position_qty <= 0:
        get_logger().event("rejected_order", symbol=symbol, side=side, reason="sell_without_position", context=context)
        raise LongOnlyViolation("Sell is allowed only to close an existing BTC position.")
    if side == "buy" and (notional is None or notional <= 0):
        get_logger().event("rejected_order", symbol=symbol, side=side, reason="invalid_notional", context=context)
        raise ValueError("Buy orders require a positive notional amount.")
    if side == "sell" and qty is not None and qty <= 0:
        get_logger().event("rejected_order", symbol=symbol, side=side, reason="invalid_qty", context=context)
        raise ValueError("Sell quantity must be positive.")
    return GuardedOrder(symbol=symbol, side=side, notional=notional, qty=qty)
