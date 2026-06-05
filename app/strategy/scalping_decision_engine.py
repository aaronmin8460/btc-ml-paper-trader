from datetime import UTC, datetime, timedelta

import pandas as pd

from app.broker.execution_guard import assert_btc_only
from app.config import ALLOWED_SYMBOL, Settings, get_settings
from app.risk.position_sizer import fixed_notional
from app.risk.risk_manager import AccountState, PositionState, RiskManager, TradeFrequencyState
from app.strategy.decision_engine import Decision
from app.strategy.scalping_rules import ScalpingMarketState, ScalpingRules
from app.strategy.strategies import (
    MarketContext,
    MarketRegime,
    MarketRegimeFilter,
    MeanReversionScalpingStrategy,
    MomentumBreakoutStrategy,
    StrategySignal,
)
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
        self.regime_filter = MarketRegimeFilter(self.settings)
        self.strategies = [
            MeanReversionScalpingStrategy(self.settings),
            MomentumBreakoutStrategy(self.settings),
        ]

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
            quote=quote,
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
        quote: dict | None,
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
            return _blocked_decision(symbol, "api_budget_exhausted", "api_budget")
        allowed, reason = self.rules.allow_entry_market(market)
        if not allowed:
            return _blocked_decision(symbol, reason, _blocked_by_for_reason(reason))

        regime = self.regime_filter.detect(feature_row, quote=quote)
        if regime.regime in {"too_volatile", "not_tradeable"}:
            return _blocked_decision(
                symbol,
                regime.reason,
                "regime_filter",
                regime=regime,
                metadata={"regime": regime.to_dict()},
            )

        candidates = self._strategy_candidates(
            feature_row=feature_row,
            prediction=prediction,
            position=position,
            quote=quote,
            regime=regime,
        )
        selected_strategy = self._select_strategy_candidate(candidates, regime=regime)
        candidate_payload = [candidate.to_dict() for candidate in candidates]
        if selected_strategy is None:
            return _blocked_decision(
                symbol,
                _quant_strategy_block_reason(candidates),
                "quant_strategy",
                regime=regime,
                strategy_candidates=candidate_payload,
                metadata={"regime": regime.to_dict()},
            )

        ml_confirmation = _ml_confirmation(
            prediction,
            buy_probability=buy_probability,
            sell_probability=sell_probability,
            buy_floor=self.settings.scalping_buy_probability_floor,
            confidence_gap_required=self.settings.scalping_confidence_gap_required,
        )
        if _active_model_invalid(prediction):
            return _blocked_decision(
                symbol,
                "active_model_invalid",
                "active_model_invalid",
                regime=regime,
                selected_strategy=selected_strategy,
                strategy_candidates=candidate_payload,
                ml_confirmation=ml_confirmation,
                metadata={"active_model_reason": prediction.get("active_model_reason")},
            )
        if _model_unavailable(prediction):
            return _blocked_decision(
                symbol,
                "model_unavailable",
                "ml_filter",
                regime=regime,
                selected_strategy=selected_strategy,
                strategy_candidates=candidate_payload,
                ml_confirmation=ml_confirmation,
            )
        if buy_probability < self.settings.scalping_buy_probability_floor:
            return _blocked_decision(
                symbol,
                "scalping_buy_probability_below_floor",
                "ml_filter",
                regime=regime,
                selected_strategy=selected_strategy,
                strategy_candidates=candidate_payload,
                ml_confirmation=ml_confirmation,
            )
        if buy_probability - sell_probability < self.settings.scalping_confidence_gap_required:
            return _blocked_decision(
                symbol,
                "scalping_confidence_gap_too_small",
                "ml_filter",
                regime=regime,
                selected_strategy=selected_strategy,
                strategy_candidates=candidate_payload,
                ml_confirmation=ml_confirmation,
            )
        if recent_ioc_canceled_buys >= self.settings.max_recent_ioc_cancels:
            return _blocked_decision(
                symbol,
                "recent_ioc_cancels_too_high",
                "ioc_cancel_guard",
                regime=regime,
                selected_strategy=selected_strategy,
                strategy_candidates=candidate_payload,
                ml_confirmation=ml_confirmation,
            )
        if _timestamp_within_seconds(
            latest_ioc_canceled_buy_at,
            now=now,
            seconds=self.settings.ioc_cancel_cooldown_seconds,
        ):
            return _blocked_decision(
                symbol,
                "ioc_cancel_cooldown_active",
                "ioc_cancel_guard",
                regime=regime,
                selected_strategy=selected_strategy,
                strategy_candidates=candidate_payload,
                ml_confirmation=ml_confirmation,
            )
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
            return _blocked_decision(
                symbol,
                risk_reason,
                _blocked_by_for_reason(risk_reason),
                regime=regime,
                selected_strategy=selected_strategy,
                strategy_candidates=candidate_payload,
                ml_confirmation=ml_confirmation,
            )
        if trading_enabled is False:
            return _blocked_decision(
                symbol,
                "trading_disabled",
                "risk_manager",
                regime=regime,
                selected_strategy=selected_strategy,
                strategy_candidates=candidate_payload,
                ml_confirmation=ml_confirmation,
            )
        return Decision(
            symbol,
            "buy",
            "scalping_entry_approved",
            notional=notional,
            strategy_name=selected_strategy.strategy_name,
            strategy_score=selected_strategy.score,
            strategy_confidence=selected_strategy.confidence,
            regime=regime.regime,
            regime_confidence=regime.confidence,
            ml_confirmation=ml_confirmation,
            strategy_candidates=candidate_payload,
            metadata={
                "entry_reason": selected_strategy.reason,
                "regime": regime.to_dict(),
                "strategy": selected_strategy.to_dict(),
            },
        )

    def _strategy_candidates(
        self,
        *,
        feature_row: pd.Series,
        prediction: dict,
        position: PositionState,
        quote: dict | None,
        regime: MarketRegime,
    ) -> list[StrategySignal]:
        context = MarketContext(regime=regime, risk_permits_evaluation=True)
        return [
            strategy.generate_signal(
                feature_row=feature_row,
                prediction=prediction,
                position=position,
                quote=quote,
                market_context=context,
            )
            for strategy in self.strategies
        ]

    def _select_strategy_candidate(self, candidates: list[StrategySignal], *, regime: MarketRegime) -> StrategySignal | None:
        allowed: list[StrategySignal] = []
        for candidate in candidates:
            if candidate.action != "buy":
                continue
            regime_allowed, _ = self.regime_filter.allows(regime, candidate.strategy_name)
            if regime_allowed:
                allowed.append(candidate)
        if not allowed:
            return None
        return max(allowed, key=lambda signal: (signal.score, signal.confidence, signal.strategy_name))

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


def _active_model_invalid(prediction: dict) -> bool:
    if prediction.get("active_model_valid") is not False:
        return False
    if prediction.get("active_model_path") or prediction.get("model_path"):
        return True
    return str(prediction.get("prediction_source") or "").lower() == "fallback_invalid_model"


def _ml_confirmation(
    prediction: dict,
    *,
    buy_probability: float,
    sell_probability: float,
    buy_floor: float,
    confidence_gap_required: float,
) -> dict:
    confidence_gap = buy_probability - sell_probability
    return {
        "buy_probability": buy_probability,
        "sell_probability": sell_probability,
        "confidence_gap": confidence_gap,
        "buy_floor": buy_floor,
        "confidence_gap_required": confidence_gap_required,
        "model_available": bool(prediction.get("model_available")),
        "prediction_source": prediction.get("prediction_source") or prediction.get("source"),
        "passed": (
            not _model_unavailable(prediction)
            and buy_probability >= buy_floor
            and confidence_gap >= confidence_gap_required
        ),
    }


def _blocked_decision(
    symbol: str,
    reason: str,
    blocked_by: str,
    *,
    regime: MarketRegime | None = None,
    selected_strategy: StrategySignal | None = None,
    strategy_candidates: list[dict] | None = None,
    ml_confirmation: dict | None = None,
    metadata: dict | None = None,
) -> Decision:
    return Decision(
        symbol,
        "hold",
        reason,
        blocked_by=blocked_by,
        block_reason=reason,
        strategy_name=selected_strategy.strategy_name if selected_strategy else None,
        strategy_score=selected_strategy.score if selected_strategy else None,
        strategy_confidence=selected_strategy.confidence if selected_strategy else None,
        regime=regime.regime if regime else None,
        regime_confidence=regime.confidence if regime else None,
        ml_confirmation=ml_confirmation,
        strategy_candidates=strategy_candidates,
        metadata=metadata or ({"regime": regime.to_dict()} if regime else None),
    )


def _quant_strategy_block_reason(candidates: list[StrategySignal]) -> str:
    buy_candidates = [candidate for candidate in candidates if candidate.action == "buy"]
    if buy_candidates:
        return "strategy_candidate_blocked_by_regime"
    if candidates:
        return candidates[0].reason
    return "no_quant_strategy_candidate"


def _blocked_by_for_reason(reason: str) -> str:
    if reason in {"stale_market_data", "scalping_stale_data_exit"}:
        return "stale_market_data"
    if reason in {"spread_too_wide", "spread_unavailable"}:
        return "spread"
    if reason in {"quote_imbalance_too_weak", "quote_imbalance_unavailable"}:
        return "quote_imbalance"
    if reason == "api_budget_exhausted":
        return "api_budget"
    if reason in {"recent_ioc_cancels_too_high", "ioc_cancel_cooldown_active"}:
        return "ioc_cancel_guard"
    if reason in {"trade_cooldown_active", "cooldown_after_loss"}:
        return "cooldown"
    if reason == "active_model_invalid":
        return "active_model_invalid"
    if reason.startswith("scalping_buy_probability") or reason.startswith("scalping_confidence"):
        return "ml_filter"
    return "risk_manager"


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
