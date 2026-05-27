import json
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from app.config import ALLOWED_SYMBOL
from app.db.models import Order, Signal, Trade
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

    def recent_orders(self, limit: int = 50) -> list[Order]:
        return self.db.query(Order).order_by(desc(Order.created_at)).limit(limit).all()

    def latest_order(self, *, side: str | None = None) -> Order | None:
        query = self.db.query(Order).filter(Order.symbol == ALLOWED_SYMBOL)
        if side is not None:
            query = query.filter(Order.side == side)
        return query.order_by(desc(Order.created_at)).first()

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
