import math
from dataclasses import asdict, dataclass, replace
from typing import Any
from uuid import uuid4

from app.broker.execution_guard import assert_btc_only
from app.config import ALLOWED_SYMBOL


@dataclass(frozen=True)
class PaperOrderRequest:
    side: str
    symbol: str = ALLOWED_SYMBOL
    bid_price: float | None = None
    ask_price: float | None = None
    bid_size: float | None = None
    ask_size: float | None = None
    latest_price: float | None = None
    notional: float | None = None
    qty: float | None = None
    fee_bps: float = 0.0
    slippage_bps: float = 0.0
    limit_price: float | None = None
    order_type: str = "market"
    time_in_force: str = "gtc"


@dataclass(frozen=True)
class PaperFillResult:
    id: str
    status: str
    symbol: str
    side: str
    filled_qty: float
    filled_avg_price: float | None
    fee_amount: float
    slippage_amount: float
    order_type: str
    time_in_force: str
    requested_qty: float
    unfilled_qty: float
    filled_notional: float
    limit_price: float | None = None
    bid_price: float | None = None
    ask_price: float | None = None
    spread_amount: float | None = None
    spread_bps: float | None = None
    spread_cost_amount: float = 0.0
    execution_source: str = "local_simulated"
    slippage_applied: bool = True

    def to_order_response(self) -> dict[str, Any]:
        return asdict(self)


def simulate_market_order(
    request: PaperOrderRequest | None = None,
    **kwargs: Any,
) -> PaperFillResult:
    order = _coerce_request(request, kwargs, order_type="market")
    touch_price = _market_touch_price(order)
    if touch_price is None:
        return _canceled_result(order, requested_qty=_requested_qty(order, fallback_price=None))
    fill_price = _slipped_price(touch_price, side=order.side, slippage_bps=order.slippage_bps)
    return _fill_result(order, touch_price=touch_price, fill_price=fill_price)


def simulate_limit_ioc_order(
    request: PaperOrderRequest | None = None,
    **kwargs: Any,
) -> PaperFillResult:
    order = _coerce_request(request, kwargs, order_type="limit", time_in_force="ioc")
    limit_price = _positive_float(order.limit_price)
    if limit_price is None:
        raise ValueError("Limit IOC orders require a positive limit_price.")

    touch_price = _limit_touch_price(order)
    requested_qty = _requested_qty(order, fallback_price=touch_price)
    if touch_price is None or not _limit_crosses_touch(order.side, limit_price, touch_price):
        return _canceled_result(order, requested_qty=requested_qty)

    slipped_price = _slipped_price(touch_price, side=order.side, slippage_bps=order.slippage_bps)
    fill_price = min(slipped_price, limit_price) if order.side == "buy" else max(slipped_price, limit_price)
    return _fill_result(order, touch_price=touch_price, fill_price=fill_price)


def _coerce_request(
    request: PaperOrderRequest | None,
    kwargs: dict[str, Any],
    *,
    order_type: str,
    time_in_force: str | None = None,
) -> PaperOrderRequest:
    if request is not None and kwargs:
        raise TypeError("Pass either a PaperOrderRequest or keyword arguments, not both.")
    if request is None:
        values = dict(kwargs)
        values["order_type"] = order_type
        if time_in_force is not None:
            values["time_in_force"] = time_in_force
        request = PaperOrderRequest(**values)
    else:
        request = replace(
            request,
            order_type=order_type,
            time_in_force=time_in_force or request.time_in_force,
        )
    return _validated_request(request)


def _validated_request(request: PaperOrderRequest) -> PaperOrderRequest:
    assert_btc_only(request.symbol, context="local_paper_execution")
    side = request.side.strip().lower()
    if side not in {"buy", "sell"}:
        raise ValueError("Paper execution side must be buy or sell.")
    time_in_force = request.time_in_force.strip().lower()
    if request.order_type == "limit" and time_in_force != "ioc":
        raise ValueError("Local simulated limit orders support IOC time in force only.")
    _non_negative_float(request.fee_bps, name="fee_bps")
    _non_negative_float(request.slippage_bps, name="slippage_bps")
    if request.notional is not None:
        _positive_required_float(request.notional, name="notional")
    if request.qty is not None:
        _positive_required_float(request.qty, name="qty")
    if request.notional is None and request.qty is None:
        raise ValueError("Paper orders require a positive notional or qty.")
    for name in ("bid_size", "ask_size"):
        value = getattr(request, name)
        if value is not None:
            _non_negative_float(value, name=name)
    return replace(request, side=side, time_in_force=time_in_force)


def _fill_result(order: PaperOrderRequest, *, touch_price: float, fill_price: float) -> PaperFillResult:
    requested_qty = _requested_qty(order, fallback_price=fill_price)
    if requested_qty <= 0:
        return _canceled_result(order, requested_qty=requested_qty)
    available_qty = _available_qty(order)
    filled_qty = requested_qty if available_qty is None else min(requested_qty, available_qty)
    if filled_qty <= 0:
        return _canceled_result(order, requested_qty=requested_qty)

    filled_notional = filled_qty * fill_price
    fee_amount = filled_notional * (order.fee_bps / 10_000)
    slippage_amount = abs(fill_price - touch_price) * filled_qty
    spread_amount, spread_bps = _spread_metrics(order)
    status = "filled" if math.isclose(filled_qty, requested_qty, rel_tol=0, abs_tol=1e-12) else "partially_filled"
    return PaperFillResult(
        id=_paper_order_id(),
        status=status,
        symbol=order.symbol,
        side=order.side,
        filled_qty=filled_qty,
        filled_avg_price=fill_price,
        fee_amount=fee_amount,
        slippage_amount=slippage_amount,
        order_type=order.order_type,
        time_in_force=order.time_in_force,
        requested_qty=requested_qty,
        unfilled_qty=max(0.0, requested_qty - filled_qty),
        filled_notional=filled_notional,
        limit_price=order.limit_price,
        bid_price=_positive_float(order.bid_price),
        ask_price=_positive_float(order.ask_price),
        spread_amount=spread_amount,
        spread_bps=spread_bps,
        spread_cost_amount=_spread_cost_amount(order, filled_qty=filled_qty),
    )


def _canceled_result(order: PaperOrderRequest, *, requested_qty: float) -> PaperFillResult:
    spread_amount, spread_bps = _spread_metrics(order)
    return PaperFillResult(
        id=_paper_order_id(),
        status="canceled",
        symbol=order.symbol,
        side=order.side,
        filled_qty=0.0,
        filled_avg_price=None,
        fee_amount=0.0,
        slippage_amount=0.0,
        order_type=order.order_type,
        time_in_force=order.time_in_force,
        requested_qty=max(0.0, requested_qty),
        unfilled_qty=max(0.0, requested_qty),
        filled_notional=0.0,
        limit_price=order.limit_price,
        bid_price=_positive_float(order.bid_price),
        ask_price=_positive_float(order.ask_price),
        spread_amount=spread_amount,
        spread_bps=spread_bps,
        spread_cost_amount=0.0,
    )


def _market_touch_price(order: PaperOrderRequest) -> float | None:
    quote_price = order.ask_price if order.side == "buy" else order.bid_price
    return _positive_float(quote_price) or _positive_float(order.latest_price)


def _limit_touch_price(order: PaperOrderRequest) -> float | None:
    quote_price = order.ask_price if order.side == "buy" else order.bid_price
    return _positive_float(quote_price)


def _requested_qty(order: PaperOrderRequest, *, fallback_price: float | None) -> float:
    explicit_qty = _positive_float(order.qty)
    if explicit_qty is not None:
        return explicit_qty
    notional = _positive_float(order.notional)
    price = _positive_float(fallback_price)
    if notional is None or price is None:
        return 0.0
    return notional / price


def _available_qty(order: PaperOrderRequest) -> float | None:
    size = order.ask_size if order.side == "buy" else order.bid_size
    if size is None:
        return None
    return _non_negative_float(size, name="quote_size")


def _slipped_price(touch_price: float, *, side: str, slippage_bps: float) -> float:
    slippage = slippage_bps / 10_000
    return touch_price * (1 + slippage) if side == "buy" else touch_price * (1 - slippage)


def _limit_crosses_touch(side: str, limit_price: float, touch_price: float) -> bool:
    return limit_price >= touch_price if side == "buy" else limit_price <= touch_price


def _spread_metrics(order: PaperOrderRequest) -> tuple[float | None, float | None]:
    bid_price = _positive_float(order.bid_price)
    ask_price = _positive_float(order.ask_price)
    if bid_price is None or ask_price is None:
        return None, None
    spread_amount = ask_price - bid_price
    mid_price = (ask_price + bid_price) / 2
    return spread_amount, spread_amount / mid_price * 10_000


def _spread_cost_amount(order: PaperOrderRequest, *, filled_qty: float) -> float:
    spread_amount, _ = _spread_metrics(order)
    if spread_amount is None:
        return 0.0
    return max(0.0, spread_amount) * filled_qty / 2


def _positive_required_float(value: float, *, name: str) -> float:
    parsed = _positive_float(value)
    if parsed is None:
        raise ValueError(f"{name} must be a finite positive number.")
    return parsed


def _positive_float(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0 else None


def _non_negative_float(value: float, *, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-negative number.") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{name} must be a finite non-negative number.")
    return parsed


def _paper_order_id() -> str:
    return f"local-paper-{uuid4().hex}"
