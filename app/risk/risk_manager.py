from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import math
from typing import Any

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


@dataclass
class AccountState:
    available: bool = False
    equity: float | None = None
    cash: float | None = None
    buying_power: float | None = None
    portfolio_value: float | None = None
    last_equity: float | None = None
    daily_change_usd: float | None = None
    daily_change_pct: float | None = None
    drawdown_pct: float | None = None
    status: str | None = None
    paper: bool | None = None


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
        account_state: AccountState | None = None,
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
        approved, reason = self.approve_account_for_buy(notional=notional, account_state=account_state)
        if not approved:
            return self._block(reason)
        approved, reason = self._approve_trade_frequency(trade_frequency or TradeFrequencyState(), now=current_time)
        if not approved:
            return self._block(reason)
        return True, "approved"

    def approve_account_for_buy(
        self,
        *,
        notional: float,
        account_state: AccountState | None,
    ) -> tuple[bool, str]:
        state = account_state or AccountState()
        if not state.available:
            if self.settings.require_account_data_for_trading:
                return False, "account_data_required_unavailable"
            return True, "approved"
        if state.buying_power is not None and state.buying_power < notional:
            return False, "buying_power_too_low"
        if not self.settings.pause_trading_on_account_drawdown:
            return True, "approved"
        if state.daily_change_usd is not None and state.daily_change_usd <= -abs(self.settings.max_account_daily_loss_usd):
            return False, "account_daily_loss_usd_reached"
        if state.daily_change_pct is not None and state.daily_change_pct <= -abs(self.settings.max_account_daily_loss_pct):
            return False, "account_daily_loss_pct_reached"
        if state.drawdown_pct is not None and state.drawdown_pct >= self.settings.max_account_drawdown_pct:
            return False, "account_drawdown_reached"
        return True, "approved"

    def should_force_sell(self, *, position: PositionState, latest_price: float, now: datetime) -> tuple[bool, str]:
        if not position.has_position:
            return False, "no_position"
        entry = position.avg_entry_price
        stop_loss_pct = self.settings.scalping_stop_loss_pct if self.settings.scalping_mode_enabled else self.settings.stop_loss_pct
        take_profit_pct = self.settings.scalping_take_profit_pct if self.settings.scalping_mode_enabled else self.settings.take_profit_pct
        trailing_stop_pct = self.settings.scalping_trailing_stop_pct if self.settings.scalping_mode_enabled else self.settings.trailing_stop_pct
        stop_reason = "scalping_stop_loss" if self.settings.scalping_mode_enabled else "stop_loss"
        take_profit_reason = "scalping_take_profit" if self.settings.scalping_mode_enabled else "take_profit"
        trailing_reason = "scalping_trailing_stop" if self.settings.scalping_mode_enabled else "trailing_stop"
        max_position_reason = "scalping_max_position_seconds" if self.settings.scalping_mode_enabled else "max_holding_time"

        if latest_price <= entry * (1 - stop_loss_pct):
            return True, stop_reason
        if latest_price >= entry * (1 + take_profit_pct):
            return True, take_profit_reason
        high = max(position.highest_price or entry, latest_price)
        if latest_price <= high * (1 - trailing_stop_pct):
            return True, trailing_reason
        if position.opened_at:
            max_holding = (
                timedelta(seconds=self.settings.scalping_max_position_seconds)
                if self.settings.scalping_mode_enabled
                else timedelta(minutes=self.settings.max_holding_minutes)
            )
            if now - position.opened_at >= max_holding:
                return True, max_position_reason
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


def account_state_from_payload(account: dict[str, Any] | None) -> AccountState:
    if not account:
        return AccountState()
    if account.get("credentials") == "missing":
        return AccountState()
    equity = _safe_float(account.get("equity") or account.get("portfolio_value"))
    portfolio_value = _safe_float(account.get("portfolio_value") or account.get("equity"))
    cash = _safe_float(account.get("cash"))
    buying_power = _safe_float(account.get("buying_power"))
    last_equity = _safe_float(account.get("last_equity") or account.get("last_portfolio_value"))
    daily_change_usd = None
    daily_change_pct = None
    drawdown_pct = None
    if equity is not None and last_equity is not None and last_equity > 0:
        daily_change_usd = equity - last_equity
        daily_change_pct = daily_change_usd / last_equity
        drawdown_pct = max(0.0, (last_equity - equity) / last_equity)
    return AccountState(
        available=True,
        equity=equity,
        cash=cash,
        buying_power=buying_power,
        portfolio_value=portfolio_value,
        last_equity=last_equity,
        daily_change_usd=daily_change_usd,
        daily_change_pct=daily_change_pct,
        drawdown_pct=drawdown_pct,
        status=str(account.get("status")) if account.get("status") is not None else None,
        paper=account.get("paper") if isinstance(account.get("paper"), bool) else None,
    )


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None
