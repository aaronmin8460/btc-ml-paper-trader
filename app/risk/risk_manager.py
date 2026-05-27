from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from app.config import ALLOWED_SYMBOL, Settings, get_settings
from app.monitoring.logger import get_logger
from app.utils.time import utc_now


@dataclass
class PositionState:
    symbol: str = ALLOWED_SYMBOL
    qty: float = 0.0
    avg_entry_price: float = 0.0
    market_value: float = 0.0
    opened_at: datetime | None = None
    highest_price: float = 0.0
    realized_pnl_today: float = 0.0
    drawdown_pct: float = 0.0
    last_loss_at: datetime | None = None

    @property
    def has_position(self) -> bool:
        return self.qty > 0


@dataclass
class TradeFrequencyState:
    trades_last_hour: int = 0
    trades_today: int = 0
    consecutive_losses: int = 0
    last_trade_at: datetime | None = None


class RiskManager:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.logger = get_logger()

    def approve_buy(
        self,
        *,
        notional: float,
        position: PositionState,
        latest_price: float,
        trade_frequency: TradeFrequencyState | None = None,
        now: datetime | None = None,
    ) -> tuple[bool, str]:
        current_time = now or utc_now()
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=UTC)
        if position.symbol != ALLOWED_SYMBOL:
            return self._block("non_btc_position")
        if position.has_position:
            return self._block("already_holding_btc")
        if notional > self.settings.max_position_notional_usd:
            return self._block("order_notional_exceeds_max_position")
        if notional > self.settings.max_total_exposure_usd:
            return self._block("order_notional_exceeds_total_exposure")
        if self.settings.order_notional_usd > self.settings.max_position_notional_usd:
            return self._block("configured_order_notional_too_large")
        if position.realized_pnl_today <= -abs(self.settings.max_daily_loss_usd):
            return self._block("max_daily_loss_reached")
        if position.drawdown_pct >= self.settings.max_drawdown_pct:
            return self._block("max_drawdown_reached")
        if position.last_loss_at and datetime.now(position.last_loss_at.tzinfo) - position.last_loss_at < timedelta(minutes=30):
            return self._block("cooldown_after_loss")
        if latest_price <= 0:
            return self._block("invalid_price")
        approved, reason = self._approve_trade_frequency(trade_frequency or TradeFrequencyState(), now=current_time)
        if not approved:
            return self._block(reason)
        return True, "approved"

    def should_force_sell(self, *, position: PositionState, latest_price: float, now: datetime) -> tuple[bool, str]:
        if not position.has_position:
            return False, "no_position"
        entry = position.avg_entry_price
        if latest_price <= entry * (1 - self.settings.stop_loss_pct):
            return True, "stop_loss"
        if latest_price >= entry * (1 + self.settings.take_profit_pct):
            return True, "take_profit"
        high = max(position.highest_price or entry, latest_price)
        if latest_price <= high * (1 - self.settings.trailing_stop_pct):
            return True, "trailing_stop"
        if position.opened_at and now - position.opened_at >= timedelta(minutes=self.settings.max_holding_minutes):
            return True, "max_holding_time"
        return False, "hold"

    def approve_sell(self, position: PositionState) -> tuple[bool, str]:
        if not position.has_position:
            return self._block("sell_without_position")
        return True, "approved"

    def _approve_trade_frequency(self, state: TradeFrequencyState, *, now: datetime) -> tuple[bool, str]:
        if state.trades_last_hour >= self.settings.max_trades_per_hour:
            return False, "max_trades_per_hour_reached"
        if state.trades_today >= self.settings.max_daily_trades:
            return False, "max_daily_trades_reached"
        if state.consecutive_losses >= self.settings.max_consecutive_losses:
            return False, "max_consecutive_losses_reached"
        if state.last_trade_at:
            last_trade_at = state.last_trade_at
            if last_trade_at.tzinfo is None:
                last_trade_at = last_trade_at.replace(tzinfo=UTC)
            elapsed = now - last_trade_at
            if elapsed < timedelta(seconds=self.settings.min_seconds_between_trades):
                return False, "trade_cooldown_active"
        return True, "approved"

    def _block(self, reason: str) -> tuple[bool, str]:
        self.logger.event("risk_block", symbol=ALLOWED_SYMBOL, reason=reason)
        return False, reason
