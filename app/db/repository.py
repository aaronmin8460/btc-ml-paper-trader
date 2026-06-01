import json
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from app.config import ALLOWED_SYMBOL
from app.db.models import AccountSnapshot, ModelRun, Order, RiskEvent, Signal, Trade
from app.risk.risk_manager import TradeFrequencyState
from app.utils.time import utc_now


SECRET_KEY_PARTS = ("secret", "token", "key", "webhook", "authorization", "password")


class Repository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def add_signal(
        self,
        action: str,
        buy_probability: float,
        sell_probability: float,
        reason: str,
        *,
        spread_bps: float | None = None,
        quote_imbalance: float | None = None,
        model_version: str | None = None,
    ) -> Signal:
        signal = Signal(
            symbol=ALLOWED_SYMBOL,
            action=action,
            buy_probability=buy_probability,
            sell_probability=sell_probability,
            reason=reason,
            spread_bps=_safe_float(spread_bps),
            quote_imbalance=_safe_float(quote_imbalance),
            model_version=str(model_version) if model_version else None,
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

    def add_trade(
        self,
        *,
        side: str,
        qty: float,
        price: float,
        pnl: float = 0.0,
        entry_price: float | None = None,
        exit_price: float | None = None,
        notional: float | None = None,
        gross_pnl: float | None = None,
        net_pnl: float | None = None,
        fee_amount: float | None = None,
        slippage_amount: float | None = None,
        hold_seconds: float | None = None,
        reason: str | None = None,
    ) -> Trade:
        trade = Trade(
            symbol=ALLOWED_SYMBOL,
            side=side,
            qty=qty,
            price=price,
            pnl=pnl,
            entry_price=entry_price,
            exit_price=exit_price,
            notional=notional,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            fee_amount=fee_amount,
            slippage_amount=slippage_amount,
            hold_seconds=hold_seconds,
            reason=reason,
        )
        self.db.add(trade)
        self.db.commit()
        self.db.refresh(trade)
        return trade

    def add_realized_trade(
        self,
        *,
        side: str,
        qty: float,
        price: float,
        pnl: float,
        created_at: datetime | None = None,
        entry_price: float | None = None,
        exit_price: float | None = None,
        notional: float | None = None,
        gross_pnl: float | None = None,
        net_pnl: float | None = None,
        fee_amount: float | None = None,
        slippage_amount: float | None = None,
        hold_seconds: float | None = None,
        reason: str | None = None,
    ) -> Trade:
        trade = Trade(
            symbol=ALLOWED_SYMBOL,
            side=side,
            qty=qty,
            price=price,
            pnl=pnl,
            created_at=created_at or utc_now(),
            entry_price=entry_price,
            exit_price=exit_price,
            notional=notional,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            fee_amount=fee_amount,
            slippage_amount=slippage_amount,
            hold_seconds=hold_seconds,
            reason=reason,
        )
        self.db.add(trade)
        self.db.commit()
        self.db.refresh(trade)
        return trade

    def add_account_snapshot(
        self,
        *,
        equity: float | None = None,
        cash: float | None = None,
        buying_power: float | None = None,
        portfolio_value: float | None = None,
        currency: str | None = None,
        raw_response: dict[str, Any] | None = None,
    ) -> AccountSnapshot:
        safe_raw_response = _redact_secrets(raw_response or {})
        snapshot = AccountSnapshot(
            equity=_safe_float(equity),
            cash=_safe_float(cash),
            buying_power=_safe_float(buying_power),
            portfolio_value=_safe_float(portfolio_value),
            currency=str(currency) if currency is not None else None,
            raw_response=json.dumps(safe_raw_response, default=str),
        )
        self.db.add(snapshot)
        self.db.commit()
        self.db.refresh(snapshot)
        return snapshot

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

    def recent_account_snapshots(self, limit: int = 500) -> list[AccountSnapshot]:
        return (
            self.db.query(AccountSnapshot)
            .order_by(desc(AccountSnapshot.created_at))
            .limit(max(1, min(int(limit), 1000)))
            .all()
        )

    def latest_account_snapshot(self) -> AccountSnapshot | None:
        return self.db.query(AccountSnapshot).order_by(desc(AccountSnapshot.created_at)).first()

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

    def recent_ioc_canceled_buy_count(
        self,
        *,
        now: datetime | None = None,
        lookback_seconds: int = 300,
        limit: int = 1000,
    ) -> int:
        return self.recent_ioc_canceled_count(
            side="buy",
            now=now,
            lookback_seconds=lookback_seconds,
            limit=limit,
        )

    def latest_ioc_canceled_buy_at(
        self,
        *,
        now: datetime | None = None,
        lookback_seconds: int = 300,
        limit: int = 1000,
    ) -> datetime | None:
        latest_order = self.latest_ioc_canceled_order(
            side="buy",
            now=now,
            lookback_seconds=lookback_seconds,
            limit=limit,
        )
        if latest_order is None:
            return None
        created_at = latest_order.created_at
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)
        return created_at

    def latest_ioc_canceled_order(
        self,
        side: str | None = None,
        *,
        now: datetime | None = None,
        lookback_seconds: int | None = None,
        limit: int = 1000,
    ) -> Order | None:
        orders = self._ioc_canceled_orders(
            side=side,
            now=now,
            lookback_seconds=lookback_seconds,
            limit=limit,
        )
        for order in orders:
            if _is_limit_ioc_order(order.raw_response):
                return order
        return None

    def recent_ioc_canceled_count(
        self,
        side: str | None = None,
        *,
        now: datetime | None = None,
        lookback_seconds: int = 300,
        limit: int = 1000,
    ) -> int:
        orders = self._ioc_canceled_orders(
            side=side,
            now=now,
            lookback_seconds=lookback_seconds,
            limit=limit,
        )
        return sum(1 for order in orders if _is_limit_ioc_order(order.raw_response))

    def _ioc_canceled_orders(
        self,
        *,
        side: str | None,
        now: datetime | None,
        lookback_seconds: int | None,
        limit: int,
    ) -> list[Order]:
        query_limit = max(1, limit)
        query = self.db.query(Order).filter(Order.symbol == ALLOWED_SYMBOL, Order.status == "canceled")
        if side is not None:
            query = query.filter(Order.side == side)
        if lookback_seconds is not None:
            current_time = now or utc_now()
            if current_time.tzinfo is None:
                current_time = current_time.replace(tzinfo=UTC)
            window_start = current_time - timedelta(seconds=max(0, lookback_seconds))
            query = query.filter(Order.created_at >= window_start)
        orders = query.order_by(desc(Order.created_at)).limit(query_limit).all()
        return list(orders)

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

    def add_risk_event(
        self,
        *,
        event_type: str,
        reason: str,
        created_at: datetime | None = None,
    ) -> RiskEvent:
        event = RiskEvent(
            symbol=ALLOWED_SYMBOL,
            event_type=event_type,
            reason=reason,
            created_at=created_at or utc_now(),
        )
        self.db.add(event)
        self.db.commit()
        self.db.refresh(event)
        return event

    def latest_scalping_kill_switch_event(self) -> RiskEvent | None:
        return (
            self.db.query(RiskEvent)
            .filter(
                RiskEvent.symbol == ALLOWED_SYMBOL,
                RiskEvent.event_type.in_({"scalping_kill_switch_pause", "scalping_kill_switch_resume"}),
            )
            .order_by(desc(RiskEvent.created_at), desc(RiskEvent.id))
            .first()
        )

    def recent_model_runs(self, limit: int = 100) -> list[ModelRun]:
        return self.db.query(ModelRun).order_by(desc(ModelRun.created_at)).limit(limit).all()

    def realized_trade_summary_since(self, since: datetime) -> dict[str, float | int | None]:
        trades = (
            self.db.query(Trade)
            .filter(Trade.symbol == ALLOWED_SYMBOL, func.lower(Trade.side) == "sell", Trade.created_at >= since)
            .all()
        )
        net_pnls = [float(trade.net_pnl if trade.net_pnl is not None else trade.pnl) for trade in trades]
        return {
            "realized_pnl": float(sum(net_pnls)),
            "total_trades": len(trades),
            "win_rate": (sum(1 for pnl in net_pnls if pnl > 0) / len(net_pnls)) if net_pnls else None,
        }

    def trade_frequency_state(self, now: datetime | None = None) -> TradeFrequencyState:
        """Compatibility alias for filled-order trade frequency limits."""
        return self.filled_trade_frequency_state(now=now)

    def order_attempt_frequency_state(self, now: datetime | None = None) -> TradeFrequencyState:
        """Counts all recent BTC/USD order rows, including canceled attempts."""
        current_time = now or utc_now()
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=UTC)
        hour_start = current_time - timedelta(hours=1)
        ten_minutes_start = current_time - timedelta(minutes=10)
        day_start = current_time.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)

        attempts_last_10_minutes = self.order_attempt_count_since(ten_minutes_start)
        attempts_last_hour = len(self.recent_order_attempts_since(hour_start))
        attempts_today = len(self.recent_order_attempts_since(day_start))
        last_order = self.latest_order()

        last_order_at = last_order.created_at if last_order else None
        if last_order_at and last_order_at.tzinfo is None:
            last_order_at = last_order_at.replace(tzinfo=UTC)

        return TradeFrequencyState(
            trades_last_10_minutes=attempts_last_10_minutes,
            trades_last_hour=attempts_last_hour,
            trades_today=attempts_today,
            last_trade_at=last_order_at,
        )

    def filled_trade_frequency_state(self, now: datetime | None = None) -> TradeFrequencyState:
        """Counts BTC/USD orders with at least one fill."""
        current_time = now or utc_now()
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=UTC)
        hour_start = current_time - timedelta(hours=1)
        ten_minutes_start = current_time - timedelta(minutes=10)
        day_start = current_time.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)

        filled_last_10_minutes = self.filled_trade_count_since(ten_minutes_start)
        filled_last_hour = len(self.recent_filled_orders_since(hour_start))
        filled_today = len(self.recent_filled_orders_since(day_start))
        last_filled_order = (
            self.db.query(Order)
            .filter(Order.symbol == ALLOWED_SYMBOL, _filled_order_filter())
            .order_by(desc(Order.created_at))
            .first()
        )

        last_filled_at = last_filled_order.created_at if last_filled_order else None
        if last_filled_at and last_filled_at.tzinfo is None:
            last_filled_at = last_filled_at.replace(tzinfo=UTC)

        return TradeFrequencyState(
            trades_last_10_minutes=filled_last_10_minutes,
            trades_last_hour=filled_last_hour,
            trades_today=filled_today,
            consecutive_losses=self._consecutive_losses(),
            realized_pnl_last_hour=self.realized_pnl_since(hour_start),
            last_trade_at=last_filled_at,
        )

    def order_attempt_count_last_10_minutes(self, now: datetime | None = None) -> int:
        return self.order_attempt_count_since(_ensure_utc(now or utc_now()) - timedelta(minutes=10))

    def filled_trade_count_last_10_minutes(self, now: datetime | None = None) -> int:
        return self.filled_trade_count_since(_ensure_utc(now or utc_now()) - timedelta(minutes=10))

    def realized_pnl_last_hour(self, now: datetime | None = None) -> float:
        return self.realized_pnl_since(_ensure_utc(now or utc_now()) - timedelta(hours=1))

    def order_attempt_count_since(self, since: datetime) -> int:
        return int(
            self.db.query(func.count(Order.id))
            .filter(Order.symbol == ALLOWED_SYMBOL, Order.created_at >= since)
            .scalar()
            or 0
        )

    def filled_trade_count_since(self, since: datetime) -> int:
        return int(
            self.db.query(func.count(Order.id))
            .filter(Order.symbol == ALLOWED_SYMBOL, Order.created_at >= since, _filled_order_filter())
            .scalar()
            or 0
        )

    def realized_pnl_since(self, since: datetime) -> float:
        return float(
            self.db.query(func.sum(Trade.pnl))
            .filter(Trade.symbol == ALLOWED_SYMBOL, func.lower(Trade.side) == "sell", Trade.created_at >= since)
            .scalar()
            or 0.0
        )

    def consecutive_ioc_canceled_count(self, *, limit: int = 1000) -> int:
        orders = (
            self.db.query(Order)
            .filter(Order.symbol == ALLOWED_SYMBOL)
            .order_by(desc(Order.created_at), desc(Order.id))
            .limit(max(1, limit))
            .all()
        )
        consecutive = 0
        for order in orders:
            if str(order.status or "").lower() == "canceled" and _is_limit_ioc_order(order.raw_response):
                consecutive += 1
                continue
            break
        return consecutive

    def recent_order_attempts_since(self, since: datetime) -> list[Order]:
        return (
            self.db.query(Order)
            .filter(Order.symbol == ALLOWED_SYMBOL, Order.created_at >= since)
            .order_by(desc(Order.created_at))
            .all()
        )

    def recent_filled_orders_since(self, since: datetime) -> list[Order]:
        return (
            self.db.query(Order)
            .filter(Order.symbol == ALLOWED_SYMBOL, Order.created_at >= since, _filled_order_filter())
            .order_by(desc(Order.created_at))
            .all()
        )

    def _consecutive_losses(self) -> int:
        trades = (
            self.db.query(Trade)
            .filter(Trade.symbol == ALLOWED_SYMBOL, func.lower(Trade.side) == "sell")
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


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed


def _redact_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        redacted = {}
        for key, child in value.items():
            if any(part in str(key).lower() for part in SECRET_KEY_PARTS):
                redacted[key] = "***"
            else:
                redacted[key] = _redact_secrets(child)
        return redacted
    if isinstance(value, list):
        return [_redact_secrets(item) for item in value]
    return value


def _is_limit_ioc_order(raw_response: str | None) -> bool:
    raw = _safe_json_loads(raw_response)
    order_type = str(raw.get("order_type") or raw.get("type") or "").lower()
    time_in_force = str(raw.get("time_in_force") or "").lower()
    return order_type == "limit" and time_in_force == "ioc"


def _filled_order_filter():
    return func.lower(Order.status).in_({"filled", "partially_filled"})


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
