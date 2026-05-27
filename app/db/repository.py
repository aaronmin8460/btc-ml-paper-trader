import json
from typing import Any

from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.config import ALLOWED_SYMBOL
from app.db.models import Order, Signal


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

    def latest_signal(self) -> Signal | None:
        return self.db.query(Signal).order_by(desc(Signal.created_at)).first()

    def recent_orders(self, limit: int = 50) -> list[Order]:
        return self.db.query(Order).order_by(desc(Order.created_at)).limit(limit).all()
