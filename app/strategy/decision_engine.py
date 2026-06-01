from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal

import pandas as pd

from app.broker.execution_guard import assert_btc_only
from app.config import ALLOWED_SYMBOL, Settings, get_settings
from app.risk.position_sizer import fixed_notional
from app.risk.risk_manager import AccountState, PositionState, RiskManager, TradeFrequencyState
from app.strategy.rule_filter import RuleFilter
from app.utils.time import utc_now

Action = Literal["buy", "sell", "hold"]
EMERGENCY_EXIT_REASONS = {"emergency_stop_loss", "scalping_emergency_stop_loss"}
MAX_HOLDING_EXIT_REASONS = {"max_holding_time", "scalping_max_position_seconds"}
PROFIT_GUARD_HOLD_REASONS = {"profit_guard_holding_until_profitable", "profit_guard_holding_at_loss"}


@dataclass(frozen=True)
class Decision:
    symbol: str
    action: Action
    reason: str
    notional: float | None = None
    qty: float | None = None


class DecisionEngine:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.risk = RiskManager(self.settings)
        self.rules = RuleFilter(self.settings)

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
    ) -> Decision:
        symbol = prediction.get("symbol", ALLOWED_SYMBOL)
        assert_btc_only(symbol, context="strategy_decision")
        buy_probability = float(prediction["buy_probability"])
        sell_probability = _sell_probability(prediction, buy_probability=buy_probability)
        latest_price = float(feature_row["close"])
        market_price = _quote_mid_price(quote) or latest_price
        expected_exit_price = self.risk.estimated_exit_price(quote=quote, latest_price=market_price) or 0.0

        if self.settings.scalping_mode_enabled:
            return self._decide_scalping(
                symbol=symbol,
                buy_probability=buy_probability,
                sell_probability=sell_probability,
                feature_row=feature_row,
                position=position,
                trading_enabled=trading_enabled,
                trade_frequency=filled_trade_frequency or trade_frequency,
                order_attempt_frequency=order_attempt_frequency,
                latest_price=market_price,
                expected_exit_price=expected_exit_price,
                api_budget=api_budget,
                account_state=account_state,
                recent_ioc_canceled_buys=recent_ioc_canceled_buys,
                latest_ioc_canceled_buy_at=latest_ioc_canceled_buy_at,
                model_unavailable=_model_unavailable(
                    prediction,
                    self.settings,
                ),
            )

        force_sell, sell_reason = self.risk.should_force_sell(position=position, latest_price=market_price, now=utc_now())
        if sell_reason in PROFIT_GUARD_HOLD_REASONS:
            return Decision(symbol, "hold", sell_reason)
        if force_sell:
            return self._sell_decision(
                symbol=symbol,
                position=position,
                reason=sell_reason,
                expected_exit_price=expected_exit_price,
                requires_profit=self._sell_reason_requires_profit(sell_reason),
                emergency=sell_reason in EMERGENCY_EXIT_REASONS,
            )

        sell_gap = sell_probability - buy_probability
        if position.has_position and sell_probability >= self.settings.min_sell_probability and sell_gap >= self.settings.confidence_gap_required:
            if _model_unavailable(prediction, self.settings):
                return Decision(symbol, "hold", "model_unavailable")
            return self._sell_decision(
                symbol=symbol,
                position=position,
                reason="ml_sell_signal",
                expected_exit_price=expected_exit_price,
                requires_profit=self.settings.model_sell_requires_profit,
            )

        if not position.has_position:
            if _model_unavailable(prediction, self.settings):
                return Decision(symbol, "hold", "model_unavailable")
            if buy_probability < self.settings.min_buy_probability:
                return Decision(symbol, "hold", "buy_probability_below_threshold")
            if buy_probability - sell_probability < self.settings.confidence_gap_required:
                return Decision(symbol, "hold", "confidence_gap_too_small")
            allowed, rule_reason = self.rules.allow_buy(feature_row, buy_probability=buy_probability)
            if not allowed:
                return Decision(symbol, "hold", rule_reason)
            notional = fixed_notional(self.settings)
            approved, risk_reason = self.risk.approve_buy(
                notional=notional,
                position=position,
                latest_price=latest_price,
                trade_frequency=filled_trade_frequency or trade_frequency,
                order_attempt_frequency=order_attempt_frequency,
                account_state=account_state,
            )
            if not approved:
                return Decision(symbol, "hold", risk_reason)
            if trading_enabled is False:
                return Decision(symbol, "hold", "trading_disabled")
            return Decision(symbol, "buy", "ml_and_rules_approved", notional=notional)

        return Decision(symbol, "hold", "holding_position_no_sell_signal")

    def _decide_scalping(
        self,
        *,
        symbol: str,
        buy_probability: float,
        sell_probability: float,
        feature_row: pd.Series,
        position: PositionState,
        trading_enabled: bool | None,
        trade_frequency: TradeFrequencyState | None,
        order_attempt_frequency: TradeFrequencyState | None,
        latest_price: float,
        expected_exit_price: float,
        api_budget: dict | None,
        account_state: AccountState | None,
        recent_ioc_canceled_buys: int,
        latest_ioc_canceled_buy_at: datetime | None,
        model_unavailable: bool,
    ) -> Decision:
        if latest_price <= 0:
            return Decision(symbol, "hold", "invalid_price")

        now = utc_now()
        force_sell, sell_reason = self.risk.should_force_sell(position=position, latest_price=latest_price, now=now)
        if sell_reason in PROFIT_GUARD_HOLD_REASONS:
            return Decision(symbol, "hold", sell_reason)
        if force_sell:
            return self._sell_decision(
                symbol=symbol,
                position=position,
                reason=sell_reason,
                expected_exit_price=expected_exit_price,
                requires_profit=self._sell_reason_requires_profit(sell_reason),
                emergency=sell_reason in EMERGENCY_EXIT_REASONS,
            )

        spread = float(feature_row.get("orderbook_spread", 0) or 0)
        spread_bps = spread * 10_000
        quote_imbalance = float(feature_row.get("quote_imbalance", 0) or 0)
        if position.has_position:
            if self.settings.scalping_sell_on_weak_quote and (
                spread_bps > self.settings.max_spread_bps
                or quote_imbalance < self.settings.scalping_quote_imbalance_exit
            ):
                if not _held_long_enough_for_weak_quote_exit(
                    position,
                    now=now,
                    min_hold_seconds=self.settings.min_hold_seconds_before_weak_quote_exit,
                ):
                    return Decision(symbol, "hold", "weak_quote_exit_min_hold_active")
                if model_unavailable:
                    return Decision(symbol, "hold", "model_unavailable")
                return self._sell_decision(
                    symbol=symbol,
                    position=position,
                    reason="scalping_weak_quote_exit",
                    expected_exit_price=expected_exit_price,
                    requires_profit=self.settings.weak_quote_sell_requires_profit,
                )
            return Decision(symbol, "hold", "scalping_holding_position")

        if _api_hard_budget_exhausted(api_budget):
            return Decision(symbol, "hold", "api_budget_exhausted")

        if model_unavailable:
            return Decision(symbol, "hold", "model_unavailable")

        if buy_probability <= self.settings.scalping_buy_probability_floor:
            return Decision(symbol, "hold", "scalping_buy_probability_below_floor")

        if buy_probability - sell_probability < self.settings.scalping_confidence_gap_required:
            return Decision(symbol, "hold", "scalping_confidence_gap_too_small")

        allowed, rule_reason = self.rules.allow_buy(feature_row, buy_probability=buy_probability)
        if not allowed:
            return Decision(symbol, "hold", rule_reason)

        momentum = float(feature_row.get("log_return_3", 0) or 0)
        if momentum < self.settings.scalping_min_momentum_pct:
            return Decision(symbol, "hold", "scalping_momentum_too_weak")

        if recent_ioc_canceled_buys >= self.settings.max_recent_ioc_cancels:
            return Decision(symbol, "hold", "recent_ioc_cancels_too_high")

        if _timestamp_within_seconds(
            latest_ioc_canceled_buy_at,
            now=now,
            seconds=self.settings.ioc_cancel_cooldown_seconds,
        ):
            return Decision(symbol, "hold", "ioc_cancel_cooldown_active")

        sma_20_distance = float(feature_row.get("sma_20_distance", 0) or 0)
        log_return_5 = float(feature_row.get("log_return_5", 0) or 0)
        dip_confirmed = (
            -sma_20_distance >= self.settings.scalping_entry_dip_pct
            or -log_return_5 >= self.settings.scalping_entry_dip_pct
        )
        quote_favorable = quote_imbalance >= 0.0
        if not (dip_confirmed or quote_favorable):
            return Decision(symbol, "hold", "scalping_entry_not_confirmed")

        notional = fixed_notional(self.settings)
        approved, risk_reason = self.risk.approve_buy(
            notional=notional,
            position=position,
            latest_price=latest_price,
            trade_frequency=trade_frequency,
            order_attempt_frequency=order_attempt_frequency,
            account_state=account_state,
        )
        if not approved:
            return Decision(symbol, "hold", risk_reason)
        if trading_enabled is False:
            return Decision(symbol, "hold", "trading_disabled")
        return Decision(symbol, "buy", "scalping_dip_entry" if dip_confirmed else "scalping_quote_entry", notional=notional)

    def _sell_decision(
        self,
        *,
        symbol: str,
        position: PositionState,
        reason: str,
        expected_exit_price: float,
        requires_profit: bool = True,
        emergency: bool = False,
    ) -> Decision:
        approved, risk_reason = self.risk.approve_sell(position)
        if not approved:
            return Decision(symbol, "hold", risk_reason)
        approved, guard_reason = self.risk.approve_profit_guarded_sell(
            position,
            expected_exit_price=expected_exit_price,
            requires_profit=requires_profit,
            emergency=emergency,
        )
        if not approved:
            return Decision(symbol, "hold", guard_reason)
        return Decision(symbol, "sell", reason, qty=position.qty)

    def _sell_reason_requires_profit(self, reason: str) -> bool:
        if reason in EMERGENCY_EXIT_REASONS:
            return False
        if reason in MAX_HOLDING_EXIT_REASONS:
            return self.settings.max_holding_sell_requires_profit
        return True


def _api_hard_budget_exhausted(api_budget: dict | None) -> bool:
    if not api_budget:
        return False
    remaining = api_budget.get("budget_remaining")
    status = api_budget.get("api_budget_status")
    if status == "hard_stop":
        return True
    try:
        return remaining is not None and int(remaining) <= 0
    except (TypeError, ValueError):
        return False


def _quote_mid_price(quote: dict | None) -> float | None:
    if not quote:
        return None
    bid = _quote_float(quote, "bid_price", "bp", "bid")
    ask = _quote_float(quote, "ask_price", "ap", "ask")
    if bid is not None and ask is not None:
        return (bid + ask) / 2
    return _quote_float(quote, "mid_price", "mid")


def _quote_float(quote: dict | None, *keys: str) -> float | None:
    if not quote:
        return None
    for key in keys:
        value = quote.get(key)
        if value is None or value == "":
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return None


def _model_unavailable(prediction: dict, settings: Settings) -> bool:
    source = str(prediction.get("prediction_source") or prediction.get("source") or "").lower()
    return source.startswith("fallback") and not settings.allow_fallback_trading


def _sell_probability(prediction: dict, *, buy_probability: float) -> float:
    try:
        return float(prediction["sell_probability"])
    except (KeyError, TypeError, ValueError):
        return max(0.0, min(1.0, 1.0 - buy_probability))


def _timestamp_within_seconds(value: datetime | None, *, now: datetime, seconds: int | float) -> bool:
    if value is None:
        return False
    timestamp = value
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return now - timestamp < timedelta(seconds=max(0, seconds))


def _held_long_enough_for_weak_quote_exit(
    position: PositionState,
    *,
    now: datetime,
    min_hold_seconds: int | float,
) -> bool:
    if min_hold_seconds <= 0 or position.opened_at is None:
        return True
    opened_at = position.opened_at
    if opened_at.tzinfo is None:
        opened_at = opened_at.replace(tzinfo=UTC)
    return now - opened_at >= timedelta(seconds=min_hold_seconds)
