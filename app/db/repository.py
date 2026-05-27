import json
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from app.config import ALLOWED_SYMBOL
from app.db.models import ModelRun, Order, RiskEvent, Signal, Trade
from app.risk.risk_manager import TradeFrequencyState
from app.utils.time import utc_now


class Repository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def add_signal(self, action: str, buy_probability: float, sell_probability: float, reason: str) -> Signal:
        signal = Signal(
            symbol=ALLOWED_SYMBOL,
            action=action,
            buy_probability=buy_probability,
            sell_probability=sell_probability,
            reason=reason,
        )
        self.db.add(signal)
        self.db.commit()
        self.db.refresh(signal)
        return signal

    def add_order(
        self,
        *,
        side: str,
        status: str,
        notional: float | None = None,
        qty: float | None = None,
        broker_order_id: str | None = None,
        raw_response: dict[str, Any] | None = None,
    ) -> Order:
        order = Order(
            symbol=ALLOWED_SYMBOL,
            side=side,
            notional=notional,
            qty=qty,
            status=status,
            broker_order_id=broker_order_id,
            raw_response=json.dumps(raw_response or {}),
        )
        self.db.add(order)
        self.db.commit()
        self.db.refresh(order)
        return order

    def add_trade(self, *, side: str, qty: float, price: float, pnl: float = 0.0) -> Trade:
        trade = Trade(
            symbol=ALLOWED_SYMBOL,
            side=side,
            qty=qty,
            price=price,
            pnl=pnl,
        )
        self.db.add(trade)
        self.db.commit()
        self.db.refresh(trade)
        return trade

    def latest_signal(self) -> Signal | None:
        return self.db.query(Signal).order_by(desc(Signal.created_at)).first()

    def recent_signals(self, limit: int = 100) -> list[Signal]:
        return self.db.query(Signal).order_by(desc(Signal.created_at)).limit(limit).all()

    def recent_orders(self, limit: int = 50) -> list[Order]:
        return self.db.query(Order).order_by(desc(Order.created_at)).limit(limit).all()

    def latest_order(self, *, side: str | None = None) -> Order | None:
        query = self.db.query(Order).filter(Order.symbol == ALLOWED_SYMBOL)
        if side is not None:
            query = query.filter(Order.side == side)
        return query.order_by(desc(Order.created_at)).first()

    def latest_trade(self) -> Trade | None:
        return (
            self.db.query(Trade)
            .filter(Trade.symbol == ALLOWED_SYMBOL)
            .order_by(desc(Trade.created_at))
            .first()
        )

    def recent_trades(self, limit: int = 100) -> list[Trade]:
        return (
            self.db.query(Trade)
            .filter(Trade.symbol == ALLOWED_SYMBOL)
            .order_by(desc(Trade.created_at))
            .limit(limit)
            .all()
        )

    def all_trades_ordered(self) -> list[Trade]:
        return (
            self.db.query(Trade)
            .filter(Trade.symbol == ALLOWED_SYMBOL)
            .order_by(Trade.created_at, Trade.id)
            .all()
        )

    def count_orders_by_side(self) -> dict[str, int]:
        rows = (
            self.db.query(Order.side, func.count(Order.id))
            .filter(Order.symbol == ALLOWED_SYMBOL)
            .group_by(Order.side)
            .all()
        )
        return {str(side): int(count) for side, count in rows}

    def order_summary(self) -> dict[str, int]:
        side_counts = self.count_orders_by_side()
        total_orders = int(
            self.db.query(func.count(Order.id))
            .filter(Order.symbol == ALLOWED_SYMBOL)
            .scalar()
            or 0
        )
        return {
            "total_orders": total_orders,
            "total_buy_orders": side_counts.get("buy", 0),
            "total_sell_orders": side_counts.get("sell", 0),
        }

    def recent_ioc_canceled_buy_count(self, *, limit: int = 10) -> int:
        orders = (
            self.db.query(Order)
            .filter(Order.symbol == ALLOWED_SYMBOL, Order.side == "buy")
            .order_by(desc(Order.created_at))
            .limit(limit)
            .all()
        )
        count = 0
        for order in orders:
            if order.status != "canceled":
                continue
            raw = _safe_json_loads(order.raw_response)
            order_type = str(raw.get("order_type") or raw.get("type") or "").lower()
            time_in_force = str(raw.get("time_in_force") or "").lower()
            if order_type == "limit" and time_in_force == "ioc":
                count += 1
        return count

    def add_model_run(self, *, model_version: str, status: str, metrics: dict[str, Any]) -> ModelRun:
        run = ModelRun(model_version=model_version, status=status, metrics_json=json.dumps(metrics, default=str))
        self.db.add(run)
        self.db.commit()
        self.db.refresh(run)
        return run

    def recent_risk_events(self, limit: int = 100) -> list[RiskEvent]:
        return (
            self.db.query(RiskEvent)
            .filter(RiskEvent.symbol == ALLOWED_SYMBOL)
            .order_by(desc(RiskEvent.created_at))
            .limit(limit)
            .all()
        )

    def recent_model_runs(self, limit: int = 100) -> list[ModelRun]:
        return self.db.query(ModelRun).order_by(desc(ModelRun.created_at)).limit(limit).all()

    def trade_frequency_state(self, *, now: datetime | None = None) -> TradeFrequencyState:
        current_time = now or utc_now()
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=UTC)
        hour_start = current_time - timedelta(hours=1)
        day_start = current_time.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)

        trades_last_hour = self._count_orders_since(hour_start)
        trades_today = self._count_orders_since(day_start)
        last_order = (
            self.db.query(Order)
            .filter(Order.symbol == ALLOWED_SYMBOL)
            .order_by(desc(Order.created_at))
            .first()
        )

        last_trade_at = last_order.created_at if last_order else None
        if last_trade_at and last_trade_at.tzinfo is None:
            last_trade_at = last_trade_at.replace(tzinfo=UTC)

        return TradeFrequencyState(
            trades_last_hour=trades_last_hour,
            trades_today=trades_today,
            consecutive_losses=self._consecutive_losses(),
            last_trade_at=last_trade_at,
        )

    def _count_orders_since(self, since: datetime) -> int:
        return int(
            self.db.query(func.count(Order.id))
            .filter(Order.symbol == ALLOWED_SYMBOL, Order.created_at >= since)
            .scalar()
            or 0
        )

    def _consecutive_losses(self) -> int:
        trades = (
            self.db.query(Trade)
            .filter(Trade.symbol == ALLOWED_SYMBOL)
            .order_by(desc(Trade.created_at))
            .limit(100)
            .all()
        )
        losses = 0
        for trade in trades:
            if trade.pnl < 0:
                losses += 1
                continue
            break
        return losses


def _safe_json_loads(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}
