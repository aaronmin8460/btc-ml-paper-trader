import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func

from app.config import ALLOWED_SYMBOL, Settings
from app.db.models import Order, Trade
from app.db.repository import Repository


@dataclass
class Execution:
    order: Order
    side: str
    qty: float
    price: float
    fee: float
    fee_from_broker: bool
    slippage_amount: float
    slippage_from_broker: bool


@dataclass
class OpenLot:
    qty: float
    price: float
    effective_price: float
    remaining_fee: float
    remaining_slippage: float
    opened_at: datetime


@dataclass
class RealizedExecution:
    qty: float
    entry_price: float
    exit_price: float
    gross_pnl: float
    net_pnl: float
    fee_amount: float
    slippage_amount: float
    hold_seconds: float


def record_filled_order_trade(repo: Repository, order: Order | None, settings: Settings) -> Trade | None:
    if order is None or not _is_filled_order(order):
        return None
    execution = _execution_from_order(order, settings)
    if execution is None or execution.side != "sell":
        return None

    realized = _realized_pnl_for_sell(repo, execution, settings)
    if realized is None:
        return None

    if realized.qty <= 0:
        return None
    existing = _existing_realized_trade(
        repo,
        order=order,
        qty=realized.qty,
        price=realized.exit_price,
        pnl=realized.net_pnl,
    )
    if existing is not None:
        return existing
    raw = _safe_json_loads(order.raw_response)
    return repo.add_realized_trade(
        side="sell",
        qty=realized.qty,
        price=realized.exit_price,
        pnl=realized.net_pnl,
        created_at=_ensure_utc(order.created_at),
        entry_price=realized.entry_price,
        exit_price=realized.exit_price,
        notional=realized.qty * realized.exit_price,
        gross_pnl=realized.gross_pnl,
        net_pnl=realized.net_pnl,
        fee_amount=realized.fee_amount,
        slippage_amount=realized.slippage_amount,
        hold_seconds=realized.hold_seconds,
        reason=str(raw.get("decision_reason")) if raw.get("decision_reason") else None,
    )


def local_simulated_position_from_orders(repo: Repository) -> dict[str, Any] | None:
    qty = 0.0
    avg_entry_price = 0.0
    current_price = 0.0
    orders = (
        repo.db.query(Order)
        .filter(Order.symbol == ALLOWED_SYMBOL, func.lower(Order.status).in_(_FILLED_ORDER_STATUSES))
        .order_by(Order.created_at, Order.id)
        .all()
    )
    for order in orders:
        raw = _safe_json_loads(order.raw_response)
        if raw.get("execution_source") != "local_simulated":
            continue
        side = str(raw.get("side") or order.side or "").strip().lower()
        filled_qty = _first_float(raw, "filled_qty", "filled_quantity")
        filled_avg_price = _first_float(raw, "filled_avg_price", "avg_fill_price", "filled_price")
        if side not in {"buy", "sell"} or not filled_qty or not filled_avg_price:
            continue
        if side == "buy":
            avg_entry_price = (qty * avg_entry_price + filled_qty * filled_avg_price) / (qty + filled_qty)
            qty += filled_qty
        else:
            qty = max(0.0, qty - filled_qty)
            if qty <= 1e-12:
                qty = 0.0
                avg_entry_price = 0.0
        current_price = filled_avg_price
    if qty <= 0:
        return None
    return {
        "symbol": ALLOWED_SYMBOL,
        "qty": qty,
        "avg_entry_price": avg_entry_price,
        "current_price": current_price or avg_entry_price,
        "market_value": qty * (current_price or avg_entry_price),
        "execution_source": "local_simulated",
    }


def _realized_pnl_for_sell(
    repo: Repository,
    sell_execution: Execution,
    settings: Settings,
) -> RealizedExecution | None:
    open_lots: list[OpenLot] = []
    target_order_id = sell_execution.order.id
    matched_qty_for_target = 0.0
    pnl_for_target = 0.0
    gross_pnl_for_target = 0.0
    entry_value_for_target = 0.0
    fee_for_target = 0.0
    slippage_for_target = 0.0
    weighted_hold_seconds = 0.0
    exit_price_for_target = _effective_sell_price(
        sell_execution.price,
        settings,
        slippage_from_broker=sell_execution.slippage_from_broker,
    )

    for order in _filled_orders_through(repo, sell_execution.order):
        execution = _execution_from_order(order, settings)
        if execution is None:
            continue
        if execution.side == "buy":
            open_lots.append(
                OpenLot(
                    qty=execution.qty,
                    price=execution.price,
                    effective_price=_effective_buy_price(
                        execution.price,
                        settings,
                        slippage_from_broker=execution.slippage_from_broker,
                    ),
                    remaining_fee=execution.fee,
                    remaining_slippage=execution.slippage_amount,
                    opened_at=_ensure_utc(order.created_at),
                )
            )
            continue
        if execution.side != "sell":
            continue

        remaining_sell_qty = execution.qty
        sell_fee_remaining = execution.fee
        sell_slippage_remaining = execution.slippage_amount
        effective_sell_price = _effective_sell_price(
            execution.price,
            settings,
            slippage_from_broker=execution.slippage_from_broker,
        )
        while remaining_sell_qty > 0 and open_lots:
            lot = open_lots[0]
            matched_qty = min(remaining_sell_qty, lot.qty)
            if matched_qty <= 0:
                break
            lot_fraction = matched_qty / lot.qty if lot.qty else 0.0
            sell_fraction = matched_qty / remaining_sell_qty if remaining_sell_qty else 0.0
            entry_fee = lot.remaining_fee * lot_fraction
            exit_fee = sell_fee_remaining * sell_fraction
            entry_slippage = lot.remaining_slippage * lot_fraction
            exit_slippage = sell_slippage_remaining * sell_fraction
            realized = (
                matched_qty * effective_sell_price
                - exit_fee
                - matched_qty * lot.effective_price
                - entry_fee
            )

            if order.id == target_order_id:
                matched_qty_for_target += matched_qty
                pnl_for_target += realized
                gross_pnl_for_target += matched_qty * (execution.price - lot.price)
                entry_value_for_target += matched_qty * lot.price
                fee_for_target += entry_fee + exit_fee
                slippage_for_target += entry_slippage + exit_slippage
                weighted_hold_seconds += matched_qty * max(
                    0.0,
                    (_ensure_utc(order.created_at) - lot.opened_at).total_seconds(),
                )

            lot.qty -= matched_qty
            lot.remaining_fee -= entry_fee
            lot.remaining_slippage -= entry_slippage
            remaining_sell_qty -= matched_qty
            sell_fee_remaining -= exit_fee
            sell_slippage_remaining -= exit_slippage
            if lot.qty <= 1e-12:
                open_lots.pop(0)

        if order.id == target_order_id:
            break

    if matched_qty_for_target <= 0:
        return None
    return RealizedExecution(
        qty=matched_qty_for_target,
        entry_price=entry_value_for_target / matched_qty_for_target,
        exit_price=exit_price_for_target,
        gross_pnl=gross_pnl_for_target,
        net_pnl=pnl_for_target,
        fee_amount=fee_for_target,
        slippage_amount=slippage_for_target,
        hold_seconds=weighted_hold_seconds / matched_qty_for_target,
    )


def _filled_orders_through(repo: Repository, order: Order) -> list[Order]:
    created_at = _ensure_utc(order.created_at)
    return (
        repo.db.query(Order)
        .filter(
            Order.symbol == ALLOWED_SYMBOL,
            func.lower(Order.status).in_(_FILLED_ORDER_STATUSES),
            (Order.created_at < created_at) | ((Order.created_at == created_at) & (Order.id <= order.id)),
        )
        .order_by(Order.created_at, Order.id)
        .all()
    )


def _execution_from_order(order: Order, settings: Settings) -> Execution | None:
    raw = _safe_json_loads(order.raw_response)
    side = str(raw.get("side") or order.side or "").strip().lower()
    if side not in {"buy", "sell"}:
        return None
    qty = _first_float(raw, "filled_qty", "filled_quantity", "qty", "quantity")
    if qty is None:
        qty = _safe_float(order.qty)
    notional = _first_float(raw, "filled_notional", "notional", "filled_value", "value")
    if notional is None:
        notional = _safe_float(order.notional)
    price = _first_float(
        raw,
        "filled_avg_price",
        "avg_fill_price",
        "average_fill_price",
        "filled_price",
        "price",
        "limit_price",
    )
    if price is None and qty and notional:
        price = notional / qty
    if qty is None or qty <= 0 or price is None or price <= 0:
        return None

    fee, fee_from_broker = _fee_from_raw(raw)
    if fee is None:
        fee_notional = abs(qty * price)
        fee = fee_notional * (settings.paper_fee_bps / 10_000)
        fee_from_broker = False
    slippage_from_broker = _slippage_from_raw(raw)
    slippage_amount = _first_float(raw, "slippage_amount")
    if slippage_amount is None:
        slippage_amount = 0.0 if slippage_from_broker else abs(qty * price) * (settings.paper_slippage_bps / 10_000)
    return Execution(
        order=order,
        side=side,
        qty=qty,
        price=price,
        fee=max(0.0, fee),
        fee_from_broker=fee_from_broker,
        slippage_amount=max(0.0, slippage_amount),
        slippage_from_broker=slippage_from_broker,
    )


def _is_filled_order(order: Order) -> bool:
    return order.symbol == ALLOWED_SYMBOL and str(order.status or "").strip().lower() in _FILLED_ORDER_STATUSES


def _effective_buy_price(price: float, settings: Settings, *, slippage_from_broker: bool = False) -> float:
    if slippage_from_broker:
        return price
    return price * (1 + settings.paper_slippage_bps / 10_000)


def _effective_sell_price(price: float, settings: Settings, *, slippage_from_broker: bool = False) -> float:
    if slippage_from_broker:
        return price
    return price * (1 - settings.paper_slippage_bps / 10_000)


def _existing_realized_trade(repo: Repository, *, order: Order, qty: float, price: float, pnl: float) -> Trade | None:
    created_at = _ensure_utc(order.created_at)
    candidates = (
        repo.db.query(Trade)
        .filter(Trade.symbol == ALLOWED_SYMBOL, Trade.side == "sell", Trade.created_at == created_at)
        .all()
    )
    for trade in candidates:
        if (
            math.isclose(float(trade.qty), qty, rel_tol=0, abs_tol=1e-12)
            and math.isclose(float(trade.price), price, rel_tol=0, abs_tol=1e-9)
            and math.isclose(float(trade.pnl), pnl, rel_tol=0, abs_tol=1e-9)
        ):
            return trade
    return None


def _fee_from_raw(raw: dict[str, Any]) -> tuple[float | None, bool]:
    for key in ("fee_amount", "fee", "fees", "commission", "commission_amount", "filled_fee"):
        if key not in raw:
            continue
        parsed = _fee_value(raw[key])
        if parsed is not None:
            return parsed, True
    return None, False


def _slippage_from_raw(raw: dict[str, Any]) -> bool:
    return raw.get("slippage_applied") is True or raw.get("execution_source") == "local_simulated"


def _fee_value(value: Any) -> float | None:
    if isinstance(value, dict):
        total = 0.0
        found = False
        for child in value.values():
            parsed = _fee_value(child)
            if parsed is not None:
                total += parsed
                found = True
        return total if found else None
    if isinstance(value, list):
        total = 0.0
        found = False
        for child in value:
            parsed = _fee_value(child)
            if parsed is not None:
                total += parsed
                found = True
        return total if found else None
    return _safe_float(value)


def _first_float(raw: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        parsed = _safe_float(raw.get(key))
        if parsed is not None:
            return parsed
    return None


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _safe_json_loads(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


_FILLED_ORDER_STATUSES = {"filled", "partially_filled"}
