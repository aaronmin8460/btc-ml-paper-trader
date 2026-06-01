from datetime import UTC, datetime, timedelta

import pandas as pd

from app.broker.execution_guard import assert_btc_only
from app.config import ALLOWED_SYMBOL, Settings, get_settings
from app.risk.position_sizer import fixed_notional
from app.risk.risk_manager import AccountState, PositionState, RiskManager, TradeFrequencyState
from app.strategy.decision_engine import Decision
from app.strategy.scalping_rules import ScalpingMarketState, ScalpingRules
from app.utils.time import utc_now


HARD_EXIT_REASONS = {
    "scalping_emergency_stop_loss",
    "scalping_stop_loss",
    "scalping_take_profit",
    "scalping_trailing_stop",
    "scalping_max_position_seconds",
    "scalping_stale_data_exit",
}

STALE_DATA_REASONS = {
    "stale_market_data",
    "scalping_stale_data_exit",
}


class ScalpingDecisionEngine:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.risk = RiskManager(self.settings)
        self.rules = ScalpingRules(self.settings)

    def decide(
        self,
        *,
        prediction: dict,
        feature_row: pd.Series,
        position: PositionState,
        trading_enabled: bool | None = None,
        trade_frequency: TradeFrequencyState | None = None,
        order_attempt_frequency: TradeFrequencyState | None = None,
        filled_trade_frequency: TradeFrequencyState | None = None,
        quote: dict | None = None,
        api_budget: dict | None = None,
        account_state: AccountState | None = None,
        recent_ioc_canceled_buys: int = 0,
        latest_ioc_canceled_buy_at: datetime | None = None,
        now: datetime | None = None,
    ) -> Decision:
        current_time = now or utc_now()
        symbol = prediction.get("symbol", ALLOWED_SYMBOL)
        assert_btc_only(symbol, context="scalping_strategy_decision")
        buy_probability = float(prediction["buy_probability"])
        sell_probability = _sell_probability(prediction, buy_probability=buy_probability)
        market = self.rules.market_state(feature_row, quote=quote, now=current_time)
        expected_exit_price = self.risk.estimated_exit_price(
            quote=quote,
            latest_price=market.latest_price,
        ) or market.latest_price

        if position.has_position:
            return self._decide_exit(
                symbol=symbol,
                position=position,
                market=market,
                sell_probability=sell_probability,
                buy_probability=buy_probability,
                expected_exit_price=expected_exit_price,
                now=current_time,
            )
        return self._decide_entry(
            symbol=symbol,
            prediction=prediction,
            position=position,
            market=market,
            feature_row=feature_row,
            buy_probability=buy_probability,
            sell_probability=sell_probability,
            trading_enabled=trading_enabled,
            trade_frequency=filled_trade_frequency or trade_frequency,
            order_attempt_frequency=order_attempt_frequency,
            api_budget=api_budget,
            account_state=account_state,
            recent_ioc_canceled_buys=recent_ioc_canceled_buys,
            latest_ioc_canceled_buy_at=latest_ioc_canceled_buy_at,
            now=current_time,
        )

    def _decide_entry(
        self,
        *,
        symbol: str,
        prediction: dict,
        position: PositionState,
        market: ScalpingMarketState,
        feature_row: pd.Series,
        buy_probability: float,
        sell_probability: float,
        trading_enabled: bool | None,
        trade_frequency: TradeFrequencyState | None,
        order_attempt_frequency: TradeFrequencyState | None,
        api_budget: dict | None,
        account_state: AccountState | None,
        recent_ioc_canceled_buys: int,
        latest_ioc_canceled_buy_at: datetime | None,
        now: datetime,
    ) -> Decision:
        if _api_hard_budget_exhausted(api_budget):
            return Decision(symbol, "hold", "api_budget_exhausted")
        if _model_unavailable(prediction):
            return Decision(symbol, "hold", "model_unavailable")
        allowed, reason = self.rules.allow_entry_market(market)
        if not allowed:
            return Decision(symbol, "hold", reason)
        if buy_probability < self.settings.scalping_buy_probability_floor:
            return Decision(symbol, "hold", "scalping_buy_probability_below_floor")
        if buy_probability - sell_probability < self.settings.scalping_confidence_gap_required:
            return Decision(symbol, "hold", "scalping_confidence_gap_too_small")
        if not self.rules.entry_confirmed(feature_row):
            return Decision(symbol, "hold", "scalping_entry_not_confirmed")
        if recent_ioc_canceled_buys >= self.settings.max_recent_ioc_cancels:
            return Decision(symbol, "hold", "recent_ioc_cancels_too_high")
        if _timestamp_within_seconds(
            latest_ioc_canceled_buy_at,
            now=now,
            seconds=self.settings.ioc_cancel_cooldown_seconds,
        ):
            return Decision(symbol, "hold", "ioc_cancel_cooldown_active")
        notional = fixed_notional(self.settings)
        approved, risk_reason = self.risk.approve_buy(
            notional=notional,
            position=position,
            latest_price=market.latest_price,
            trade_frequency=trade_frequency,
            order_attempt_frequency=order_attempt_frequency,
            account_state=account_state,
            now=now,
        )
        if not approved:
            return Decision(symbol, "hold", risk_reason)
        if trading_enabled is False:
            return Decision(symbol, "hold", "trading_disabled")
        return Decision(symbol, "buy", "scalping_entry_approved", notional=notional)

    def _decide_exit(
        self,
        *,
        symbol: str,
        position: PositionState,
        market: ScalpingMarketState,
        sell_probability: float,
        buy_probability: float,
        expected_exit_price: float,
        now: datetime,
    ) -> Decision:
        hard_exit_reason = self._hard_exit_reason(position=position, latest_price=market.latest_price, now=now)
        if hard_exit_reason is not None:
            return self._sell_decision(
                symbol=symbol,
                position=position,
                reason=hard_exit_reason,
                expected_exit_price=expected_exit_price,
                bypass_profit_guard=True,
            )
        if not market.market_data_fresh:
            return self._sell_decision(
                symbol=symbol,
                position=position,
                reason="scalping_stale_data_exit",
                expected_exit_price=expected_exit_price,
                bypass_profit_guard=True,
            )
        if not self.rules.held_for_minimum(position.opened_at, now=now):
            return Decision(symbol, "hold", "scalping_min_hold_active")
        if self.settings.scalping_sell_on_weak_quote and self.rules.quote_strongly_unfavorable(market):
            return self._sell_decision(
                symbol=symbol,
                position=position,
                reason="scalping_weak_quote_exit",
                expected_exit_price=expected_exit_price,
            )
        if (
            sell_probability >= self.settings.scalping_sell_probability_floor
            and sell_probability - buy_probability >= self.settings.scalping_exit_confidence_gap_required
        ):
            return self._sell_decision(
                symbol=symbol,
                position=position,
                reason="scalping_model_sell_signal",
                expected_exit_price=expected_exit_price,
            )
        return Decision(symbol, "hold", "scalping_holding_position")

    def _hard_exit_reason(self, *, position: PositionState, latest_price: float, now: datetime) -> str | None:
        force_sell, reason = self.risk.should_force_sell(position=position, latest_price=latest_price, now=now)
        if force_sell:
            return reason
        if (
            position.avg_entry_price > 0
            and latest_price > 0
            and latest_price <= position.avg_entry_price * (1 - self.settings.scalping_stop_loss_pct)
        ):
            return "scalping_stop_loss"
        return None

    def _sell_decision(
        self,
        *,
        symbol: str,
        position: PositionState,
        reason: str,
        expected_exit_price: float,
        bypass_profit_guard: bool = False,
    ) -> Decision:
        approved, risk_reason = self.risk.approve_sell(position)
        if not approved:
            return Decision(symbol, "hold", risk_reason)
        if self.settings.scalping_profit_guard_enabled and not bypass_profit_guard:
            approved, guard_reason = self.risk.approve_profit_guarded_sell(
                position,
                expected_exit_price=expected_exit_price,
                requires_profit=True,
            )
            if not approved:
                return Decision(symbol, "hold", guard_reason)
        return Decision(symbol, "sell", reason, qty=position.qty)


def _model_unavailable(prediction: dict) -> bool:
    if prediction.get("model_available") is False:
        return True
    source = str(prediction.get("prediction_source") or prediction.get("source") or "").lower()
    return source.startswith("fallback")


def _sell_probability(prediction: dict, *, buy_probability: float) -> float:
    try:
        return float(prediction["sell_probability"])
    except (KeyError, TypeError, ValueError):
        return max(0.0, min(1.0, 1.0 - buy_probability))


def _api_hard_budget_exhausted(api_budget: dict | None) -> bool:
    if not api_budget:
        return False
    if api_budget.get("api_budget_status") == "hard_stop":
        return True
    try:
        return api_budget.get("budget_remaining") is not None and int(api_budget["budget_remaining"]) <= 0
    except (TypeError, ValueError):
        return False


def _timestamp_within_seconds(value: datetime | None, *, now: datetime, seconds: int | float) -> bool:
    if value is None:
        return False
    timestamp = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return now - timestamp < timedelta(seconds=max(0, seconds))
