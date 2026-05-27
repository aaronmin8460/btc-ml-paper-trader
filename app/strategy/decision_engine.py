from dataclasses import dataclass
from typing import Literal

import pandas as pd

from app.broker.execution_guard import assert_btc_only
from app.config import ALLOWED_SYMBOL, Settings, get_settings
from app.risk.position_sizer import fixed_notional
from app.risk.risk_manager import AccountState, PositionState, RiskManager, TradeFrequencyState
from app.strategy.rule_filter import RuleFilter
from app.utils.time import utc_now

Action = Literal["buy", "sell", "hold"]


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
        quote: dict | None = None,
        api_budget: dict | None = None,
        account_state: AccountState | None = None,
        recent_ioc_canceled_buys: int = 0,
    ) -> Decision:
        symbol = prediction.get("symbol", ALLOWED_SYMBOL)
        assert_btc_only(symbol, context="strategy_decision")
        buy_probability = float(prediction["buy_probability"])
        sell_probability = float(prediction["sell_probability"])
        latest_price = float(feature_row["close"])

        if self.settings.scalping_mode_enabled:
            return self._decide_scalping(
                symbol=symbol,
                buy_probability=buy_probability,
                sell_probability=sell_probability,
                feature_row=feature_row,
                position=position,
                trading_enabled=trading_enabled,
                trade_frequency=trade_frequency,
                latest_price=_quote_mid_price(quote) or latest_price,
                api_budget=api_budget,
                account_state=account_state,
                recent_ioc_canceled_buys=recent_ioc_canceled_buys,
            )

        force_sell, sell_reason = self.risk.should_force_sell(position=position, latest_price=latest_price, now=utc_now())
        if force_sell:
            approved, reason = self.risk.approve_sell(position)
            return Decision(symbol, "sell" if approved else "hold", sell_reason if approved else reason, qty=position.qty if approved else None)

        sell_gap = sell_probability - buy_probability
        if position.has_position and sell_probability >= self.settings.min_sell_probability and sell_gap >= self.settings.confidence_gap_required:
            approved, reason = self.risk.approve_sell(position)
            return Decision(symbol, "sell" if approved else "hold", "ml_sell_signal" if approved else reason, qty=position.qty if approved else None)

        if not position.has_position:
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
                trade_frequency=trade_frequency,
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
        latest_price: float,
        api_budget: dict | None,
        account_state: AccountState | None,
        recent_ioc_canceled_buys: int,
    ) -> Decision:
        if latest_price <= 0:
            return Decision(symbol, "hold", "invalid_price")

        force_sell, sell_reason = self.risk.should_force_sell(position=position, latest_price=latest_price, now=utc_now())
        if force_sell:
            approved, reason = self.risk.approve_sell(position)
            return Decision(symbol, "sell" if approved else "hold", sell_reason if approved else reason, qty=position.qty if approved else None)

        spread = float(feature_row.get("orderbook_spread", 0) or 0)
        spread_bps = spread * 10_000
        quote_imbalance = float(feature_row.get("quote_imbalance", 0) or 0)
        if position.has_position:
            if self.settings.scalping_sell_on_weak_quote and (
                spread_bps > self.settings.max_spread_bps
                or quote_imbalance < self.settings.scalping_quote_imbalance_exit
            ):
                approved, reason = self.risk.approve_sell(position)
                return Decision(symbol, "sell" if approved else "hold", "scalping_weak_quote_exit" if approved else reason, qty=position.qty if approved else None)
            return Decision(symbol, "hold", "scalping_holding_position")

        if _api_hard_budget_exhausted(api_budget):
            return Decision(symbol, "hold", "api_budget_exhausted")

        if buy_probability < self.settings.scalping_buy_probability_floor:
            return Decision(symbol, "hold", "scalping_buy_probability_below_floor")

        allowed, rule_reason = self.rules.allow_buy(feature_row, buy_probability=buy_probability)
        if not allowed:
            return Decision(symbol, "hold", rule_reason)

        momentum = float(feature_row.get("log_return_3", 0) or 0)
        if momentum < self.settings.scalping_min_momentum_pct:
            return Decision(symbol, "hold", "scalping_momentum_too_weak")

        if recent_ioc_canceled_buys >= 3:
            return Decision(symbol, "hold", "recent_ioc_cancels_too_high")

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
            account_state=account_state,
        )
        if not approved:
            return Decision(symbol, "hold", risk_reason)
        if trading_enabled is False:
            return Decision(symbol, "hold", "trading_disabled")
        return Decision(symbol, "buy", "scalping_dip_entry" if dip_confirmed else "scalping_quote_entry", notional=notional)


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


def _quote_float(quote: dict, *keys: str) -> float | None:
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
