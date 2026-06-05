import asyncio
from datetime import UTC, datetime, timedelta
import math
from typing import Any

from app.accounting.trade_accounting import local_simulated_position_from_orders, record_filled_order_trade
from app.broker.alpaca_client import AlpacaClient
from app.config import ALLOWED_SYMBOL, Settings, get_settings
from app.data.feature_engineering import latest_feature_row
from app.data.market_data import MarketDataClient
from app.data.scalping_features import latest_scalping_feature_row
from app.db.database import SessionLocal, init_db
from app.db.repository import Repository
from app.ml.predict import Predictor
from app.monitoring.logger import get_logger
from app.notifications.discord import DiscordNotifier
from app.risk.risk_manager import AccountState, PositionState, account_state_from_payload
from app.strategy.decision_engine import Decision, DecisionEngine
from app.strategy.scalping_decision_engine import HARD_EXIT_REASONS as SCALPING_HARD_EXIT_REASONS
from app.strategy.scalping_decision_engine import ScalpingDecisionEngine
from app.utils.rate_limiter import get_alpaca_rate_limiter


KILL_SWITCH_REASONS = {
    "max_order_attempts_per_hour_reached",
    "max_order_attempts_per_10_minutes_reached",
    "max_order_attempts_per_day_reached",
    "max_trades_per_hour_reached",
    "max_trades_per_10_minutes_reached",
    "max_daily_trades_reached",
    "max_consecutive_losses_reached",
    "trade_cooldown_active",
    "order_in_flight",
    "already_holding_btc",
    "sell_without_position",
    "api_budget_exhausted",
    "account_daily_loss_usd_reached",
    "account_daily_loss_pct_reached",
    "account_drawdown_reached",
    "account_data_required_unavailable",
    "buying_power_too_low",
    "recent_ioc_cancels_too_high",
    "model_not_profitable_after_costs",
    "stale_market_data",
    "scalping_kill_switch:hourly_loss_limit",
    "scalping_kill_switch:ioc_cancel_streak",
    "scalping_kill_switch:loss_streak",
    "scalping_kill_switch:runtime_errors",
    "spread_too_wide",
    "spread_unavailable",
    "quote_imbalance_unavailable",
    "quote_imbalance_too_weak",
}

PROFIT_GUARD_HOLD_REASONS = {
    "profit_guard_holding_until_profitable",
    "profit_guard_holding_at_loss",
}

RISK_ALERT_REASONS = KILL_SWITCH_REASONS | PROFIT_GUARD_HOLD_REASONS

HARD_RISK_EXIT_REASONS = {
    "emergency_stop_loss",
    "take_profit",
    "trailing_stop",
    "max_holding_time",
} | SCALPING_HARD_EXIT_REASONS


class Trader:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.market_data = MarketDataClient(self.settings)
        self.predictor = Predictor(self.settings)
        self.decision_engine = (
            ScalpingDecisionEngine(self.settings)
            if self.settings.scalping_mode_enabled
            else DecisionEngine(self.settings)
        )
        self.broker = AlpacaClient(self.settings)
        self.logger = get_logger()
        self.notifier = DiscordNotifier(self.settings)
        self._order_lock = asyncio.Lock()
        self._order_lock_started_at: datetime | None = None
        self._position_highest_price = 0.0
        self._last_risk_alert_reason: str | None = None
        self._last_risk_alert_sent_at: datetime | None = None
        self._last_order_attempt_bar_by_side: dict[str, str] = {}
        self._ioc_cancel_escalated_until: datetime | None = None

    async def get_position_state(self) -> PositionState:
        position = await self.broker.get_position(ALLOWED_SYMBOL)
        if not position and self.settings.paper_execution_mode == "local_simulated":
            position = self._restore_local_simulated_position()
        if not position:
            self._position_highest_price = 0.0
            return PositionState()
        qty = float(position.get("qty", 0) or 0)
        avg_entry = float(position.get("avg_entry_price", 0) or 0)
        market_value = abs(float(position.get("market_value", 0) or 0))
        current_price = float(position.get("current_price", avg_entry) or avg_entry)
        self._position_highest_price = max(self._position_highest_price, avg_entry, current_price)
        return PositionState(
            qty=qty,
            avg_entry_price=avg_entry,
            market_value=market_value,
            opened_at=self._position_opened_at(),
            highest_price=self._position_highest_price,
        )

    def _restore_local_simulated_position(self) -> dict[str, Any] | None:
        try:
            with SessionLocal() as db:
                position = local_simulated_position_from_orders(Repository(db))
            if position is None:
                return None
            hydrate = getattr(self.broker, "hydrate_local_position", None)
            if callable(hydrate):
                hydrate(
                    qty=position["qty"],
                    avg_entry_price=position["avg_entry_price"],
                    current_price=position["current_price"],
                )
            return position
        except Exception as exc:
            try:
                self.logger.event("local_position_restore_failed", error_type=type(exc).__name__)
            except Exception:
                pass
            return None

    async def run_once(self) -> dict:
        scalping_kill_switch_reason = None
        try:
            init_db()
            bars = await self.market_data.fetch_bars(ALLOWED_SYMBOL)
            quote = await self.market_data.fetch_latest_quote(ALLOWED_SYMBOL)
            prediction = self.predictor.predict(bars, quote=quote)
            feature_row = self._latest_strategy_feature_row(bars, quote=quote)
            latest_bar_timestamp = _latest_bar_timestamp(feature_row, prediction)
            position = await self.get_position_state()
            quote_mid_price = _mid_price_from_quote(quote)
            if position.has_position and quote_mid_price is not None:
                position.highest_price = max(position.highest_price or 0.0, quote_mid_price)
            account_payload = await self._account_payload()
            account_state = account_state_from_payload(account_payload)
            api_budget = get_alpaca_rate_limiter(self.settings).snapshot()
            with SessionLocal() as db:
                repo = Repository(db)
                self._store_account_snapshot(repo, account_payload)
                order_attempt_frequency = self._order_attempt_frequency_state(repo)
                filled_trade_frequency = self._filled_trade_frequency_state(repo)
                scalping_kill_switch_reason = self._scalping_kill_switch_reason(repo)
                recent_ioc_canceled_buys, latest_ioc_canceled_buy_at = self._ioc_cancel_state(repo)
                decision = self.decision_engine.decide(
                    prediction=prediction,
                    feature_row=feature_row,
                    position=position,
                    trading_enabled=self.settings.trading_enabled,
                    trade_frequency=filled_trade_frequency,
                    order_attempt_frequency=order_attempt_frequency,
                    filled_trade_frequency=filled_trade_frequency,
                    quote=quote,
                    api_budget=api_budget,
                    account_state=account_state,
                    recent_ioc_canceled_buys=recent_ioc_canceled_buys,
                    latest_ioc_canceled_buy_at=latest_ioc_canceled_buy_at,
                )
                decision, order_lock_acquired = await self._guard_order_decision(
                    decision,
                    position,
                    latest_bar_timestamp=latest_bar_timestamp,
                    scalping_kill_switch_reason=scalping_kill_switch_reason,
                )
                market_context = self._alert_market_context(
                    feature_row,
                    quote,
                    latest_bar_timestamp=latest_bar_timestamp,
                )
                risk_context = self._risk_block_context(
                    decision,
                    market_context=market_context,
                    position=position,
                    order_attempt_frequency=order_attempt_frequency,
                    filled_trade_frequency=filled_trade_frequency,
                    account_state=account_state,
                    api_budget=api_budget,
                    recent_ioc_canceled_buys=recent_ioc_canceled_buys,
                    latest_ioc_canceled_buy_at=latest_ioc_canceled_buy_at,
                )
                self.logger.event(
                    "signal",
                    symbol=decision.symbol,
                    action=decision.action,
                    reason=decision.reason,
                    timestamp=datetime.now(UTC).isoformat(),
                    latest_bar_timestamp=latest_bar_timestamp,
                    bar_age_seconds=market_context.get("bar_age_seconds"),
                    quote_age_seconds=market_context.get("quote_age_seconds"),
                    buy_probability=prediction["buy_probability"],
                    sell_probability=prediction["sell_probability"],
                    ml_buy_probability=prediction["buy_probability"],
                    ml_sell_probability=prediction["sell_probability"],
                    sell_probability_source=prediction.get("sell_probability_source", "unspecified"),
                    prediction_source=prediction.get("prediction_source"),
                    model_version=_prediction_model_version(prediction),
                    strategy_name=decision.strategy_name,
                    strategy_score=decision.strategy_score,
                    strategy_confidence=decision.strategy_confidence,
                    quant_score=decision.strategy_score,
                    quant_confidence=decision.strategy_confidence,
                    regime=decision.regime,
                    regime_confidence=decision.regime_confidence,
                    blocked_by=decision.blocked_by,
                    block_reason=decision.block_reason,
                    candidate_strategy_count=len(decision.strategy_candidates or []),
                    strategy_candidates=decision.strategy_candidates,
                    selected_strategy_signal={
                        "strategy_name": decision.strategy_name,
                        "score": decision.strategy_score,
                        "confidence": decision.strategy_confidence,
                        "reason": (decision.metadata or {}).get("entry_reason") if decision.metadata else None,
                    }
                    if decision.strategy_name
                    else None,
                    selected_strategy_reason=(decision.metadata or {}).get("entry_reason") if decision.metadata else None,
                    ml_confirmation_result=decision.ml_confirmation,
                    final_decision=decision.action,
                    spread_bps=market_context.get("spread_bps"),
                    quote_imbalance=market_context.get("quote_imbalance"),
                    momentum=market_context.get("momentum"),
                    volatility=market_context.get("volatility"),
                    position_qty=position.qty,
                    avg_entry_price=position.avg_entry_price,
                    unrealized_pnl_pct=_unrealized_pnl_pct(position, market_context.get("mid_price") or market_context.get("latest_price")),
                    risk_block_reason=risk_context.get("reason") if risk_context else None,
                    api_budget_status=api_budget.get("api_budget_status"),
                    alpaca_calls_last_minute=api_budget.get("calls_last_minute"),
                )
                if risk_context:
                    self.logger.event("risk_block", symbol=decision.symbol, **risk_context)
                await self._send_signal_alert(decision, prediction, market_context)
                await self._send_risk_alert(decision, risk_context=risk_context)
                _store_signal(
                    repo,
                    decision=decision,
                    prediction=prediction,
                    market_context=market_context,
                )
                order_response = None
                try:
                    if decision.action in {"buy", "sell"}:
                        self._remember_order_attempt_bar(decision, latest_bar_timestamp)
                        try:
                            order_response = await self.broker.submit_order(
                                symbol=decision.symbol,
                                side=decision.action,
                                notional=decision.notional,
                                qty=decision.qty,
                                current_position_qty=position.qty,
                                quote=quote,
                                latest_price=quote_mid_price or float(feature_row["close"]),
                            )
                        except Exception as exc:
                            self.logger.event(
                                "order",
                                symbol=decision.symbol,
                                **_order_log_payload(
                                    decision,
                                    {"status": "failed", "cancel_reason": type(exc).__name__},
                                    local_order_id=None,
                                    default_order_type=self.settings.order_type,
                                    default_time_in_force=self.settings.time_in_force,
                                ),
                            )
                            raise
                        self.broker.invalidate_position_cache(decision.symbol)
                        stored_order_response = {
                            **order_response,
                            "decision_reason": decision.reason,
                        }
                        order = repo.add_order(
                            side=decision.action,
                            status=order_response.get("status", "submitted"),
                            notional=decision.notional,
                            qty=decision.qty,
                            broker_order_id=order_response.get("id"),
                            raw_response=stored_order_response,
                        )
                        local_order_id = getattr(order, "id", None)
                        self.logger.event(
                            "order",
                            symbol=decision.symbol,
                            **_order_log_payload(
                                decision,
                                order_response,
                                local_order_id=local_order_id,
                                default_order_type=self.settings.order_type,
                                default_time_in_force=self.settings.time_in_force,
                            ),
                        )
                        await self._send_order_alert(decision, order_response, local_order_id=local_order_id)
                        if str(order_response.get("status") or "").lower() in {"filled", "partially_filled"}:
                            record_filled_order_trade(repo, order, self.settings)
                        scalping_kill_switch_reason = self._scalping_kill_switch_reason(repo)
                finally:
                    if order_lock_acquired and self._order_lock.locked():
                        self._order_lock.release()
                        self._order_lock_started_at = None
            return {
                "prediction": prediction,
                "decision": decision.__dict__,
                "latest_strategy_signal": {
                    "strategy_name": decision.strategy_name,
                    "strategy_score": decision.strategy_score,
                    "strategy_confidence": decision.strategy_confidence,
                    "strategy_candidates": decision.strategy_candidates,
                    "selected_strategy_reason": (decision.metadata or {}).get("entry_reason") if decision.metadata else None,
                },
                "regime": {
                    "regime": decision.regime,
                    "confidence": decision.regime_confidence,
                    "metadata": (decision.metadata or {}).get("regime") if decision.metadata else None,
                },
                "ml_confirmation": decision.ml_confirmation,
                "order": order_response,
                "scalping_kill_switch_reason": scalping_kill_switch_reason,
            }
        except Exception as exc:
            self.logger.event(
                "runtime_error",
                component="trader.run_once",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            await self._send_error_alert("trader.run_once", exc)
            raise

    async def _account_state(self) -> AccountState:
        return account_state_from_payload(await self._account_payload())

    def _latest_strategy_feature_row(self, bars, *, quote: dict):
        if self.settings.scalping_mode_enabled:
            return latest_scalping_feature_row(bars, quote=quote).iloc[-1]
        return latest_feature_row(bars, quote=quote).iloc[-1]

    async def _account_payload(self) -> dict[str, Any] | None:
        credentials_available = getattr(self.broker, "credentials_available", lambda: False)
        if not credentials_available():
            return None
        try:
            return await self.broker.get_account()
        except Exception:
            return None

    def _store_account_snapshot(self, repo: Repository, account_payload: dict[str, Any] | None) -> None:
        if not account_payload:
            return
        try:
            repo.add_account_snapshot(
                equity=account_payload.get("equity") or account_payload.get("portfolio_value"),
                cash=account_payload.get("cash"),
                buying_power=account_payload.get("buying_power"),
                portfolio_value=account_payload.get("portfolio_value") or account_payload.get("equity"),
                currency=account_payload.get("currency"),
                raw_response=account_payload,
            )
        except Exception as exc:
            try:
                self.logger.event(
                    "account_snapshot_failed",
                    error_type=type(exc).__name__,
                )
            except Exception:
                pass

    def _order_attempt_frequency_state(self, repo: Repository):
        if hasattr(repo, "order_attempt_frequency_state"):
            return repo.order_attempt_frequency_state()
        return repo.trade_frequency_state() if hasattr(repo, "trade_frequency_state") else None

    def _filled_trade_frequency_state(self, repo: Repository):
        if hasattr(repo, "filled_trade_frequency_state"):
            return repo.filled_trade_frequency_state()
        return repo.trade_frequency_state() if hasattr(repo, "trade_frequency_state") else None

    def _scalping_kill_switch_reason(self, repo: Repository) -> str | None:
        if not self.settings.scalping_mode_enabled or not self.settings.scalping_kill_switch_enabled:
            return None
        if hasattr(repo, "realized_pnl_last_hour"):
            hourly_pnl = repo.realized_pnl_last_hour()
            if hourly_pnl <= -abs(self.settings.max_loss_usd_per_hour):
                return "scalping_kill_switch:hourly_loss_limit"
        if hasattr(repo, "consecutive_ioc_canceled_count"):
            consecutive_ioc_cancels = repo.consecutive_ioc_canceled_count()
            if consecutive_ioc_cancels >= self.settings.max_consecutive_ioc_cancels:
                return "scalping_kill_switch:ioc_cancel_streak"
        return None

    def _position_opened_at(self) -> datetime:
        try:
            with SessionLocal() as db:
                order = Repository(db).latest_order(side="buy")
                if order and order.created_at:
                    opened_at = order.created_at
                    if opened_at.tzinfo is None:
                        opened_at = opened_at.replace(tzinfo=UTC)
                    return opened_at
        except Exception:
            pass
        return datetime.now(UTC)

    def _ioc_cancel_state(self, repo: Repository) -> tuple[int, datetime | None]:
        recent_ioc_canceled_buys = (
            repo.recent_ioc_canceled_count(
                side="buy",
                lookback_seconds=self.settings.ioc_cancel_lookback_seconds,
            )
            if hasattr(repo, "recent_ioc_canceled_count")
            else (
                repo.recent_ioc_canceled_buy_count(
                    lookback_seconds=self.settings.ioc_cancel_lookback_seconds,
                )
                if hasattr(repo, "recent_ioc_canceled_buy_count")
                else 0
            )
        )
        latest_lookup_seconds = max(
            self.settings.ioc_cancel_lookback_seconds,
            self.settings.ioc_cancel_cooldown_seconds,
            self.settings.ioc_cancel_escalation_cooldown_seconds,
        )
        latest_ioc_canceled_buy_at = None
        if hasattr(repo, "latest_ioc_canceled_order"):
            latest_ioc_canceled_buy = repo.latest_ioc_canceled_order(
                side="buy",
                lookback_seconds=latest_lookup_seconds,
            )
            latest_ioc_canceled_buy_at = (
                _ensure_utc(latest_ioc_canceled_buy.created_at)
                if latest_ioc_canceled_buy is not None
                else None
            )
        elif hasattr(repo, "latest_ioc_canceled_buy_at"):
            latest_ioc_canceled_buy_at = repo.latest_ioc_canceled_buy_at(
                lookback_seconds=latest_lookup_seconds,
            )
        now = datetime.now(UTC)
        if recent_ioc_canceled_buys >= self.settings.max_recent_ioc_cancels:
            escalation_anchor = latest_ioc_canceled_buy_at or now
            escalation_until = escalation_anchor + timedelta(
                seconds=self.settings.ioc_cancel_escalation_cooldown_seconds
            )
            self._ioc_cancel_escalated_until = escalation_until if now < escalation_until else None
        elif self._ioc_cancel_escalated_until is not None:
            if now < self._ioc_cancel_escalated_until:
                recent_ioc_canceled_buys = max(
                    recent_ioc_canceled_buys,
                    self.settings.max_recent_ioc_cancels,
                )
            else:
                self._ioc_cancel_escalated_until = None
        return recent_ioc_canceled_buys, latest_ioc_canceled_buy_at

    async def _guard_order_decision(
        self,
        decision: Decision,
        position: PositionState,
        *,
        latest_bar_timestamp: str | None = None,
        scalping_kill_switch_reason: str | None = None,
    ) -> tuple[Decision, bool]:
        if decision.action == "buy" and position.has_position:
            return _blocked_runtime_decision(decision, "already_holding_btc", "risk_manager"), False
        if decision.action == "sell" and not position.has_position:
            return _blocked_runtime_decision(decision, "sell_without_position", "risk_manager"), False
        if decision.action == "buy" and scalping_kill_switch_reason:
            self.logger.event(
                "order_blocked",
                symbol=decision.symbol,
                side=decision.action,
                reason=scalping_kill_switch_reason,
            )
            return _blocked_runtime_decision(decision, scalping_kill_switch_reason, "risk_manager"), False
        if decision.action not in {"buy", "sell"}:
            return decision, False
        if self._is_duplicate_bar_order_attempt(decision, latest_bar_timestamp):
            self.logger.event(
                "order_blocked",
                symbol=decision.symbol,
                side=decision.action,
                reason="duplicate_order_bar",
                latest_bar_timestamp=latest_bar_timestamp,
            )
            return _blocked_runtime_decision(decision, "duplicate_order_bar", "cooldown"), False
        if self._order_lock.locked():
            age_seconds = self._order_lock_age_seconds()
            self.logger.event(
                "order_blocked",
                symbol=decision.symbol,
                side=decision.action,
                reason="order_in_flight",
                lock_age_seconds=age_seconds,
                timeout_seconds=self.settings.order_in_flight_timeout_seconds,
            )
            return _blocked_runtime_decision(decision, "order_in_flight", "risk_manager"), False
        try:
            await asyncio.wait_for(self._order_lock.acquire(), timeout=0.001)
        except asyncio.TimeoutError:
            age_seconds = self._order_lock_age_seconds()
            self.logger.event(
                "order_blocked",
                symbol=decision.symbol,
                side=decision.action,
                reason="order_in_flight",
                lock_age_seconds=age_seconds,
                timeout_seconds=self.settings.order_in_flight_timeout_seconds,
            )
            return _blocked_runtime_decision(decision, "order_in_flight", "risk_manager"), False
        self._order_lock_started_at = datetime.now(UTC)
        return decision, True

    def _is_duplicate_bar_order_attempt(self, decision: Decision, latest_bar_timestamp: str | None) -> bool:
        if latest_bar_timestamp is None or decision.reason in HARD_RISK_EXIT_REASONS:
            return False
        return self._last_order_attempt_bar_by_side.get(decision.action) == latest_bar_timestamp

    def _remember_order_attempt_bar(self, decision: Decision, latest_bar_timestamp: str | None) -> None:
        if latest_bar_timestamp is None or decision.action not in {"buy", "sell"}:
            return
        self._last_order_attempt_bar_by_side[decision.action] = latest_bar_timestamp

    def _order_lock_age_seconds(self) -> float | None:
        if self._order_lock_started_at is None:
            return None
        return max(0.0, (datetime.now(UTC) - self._order_lock_started_at).total_seconds())

    def _discord_alerts_enabled(self) -> bool:
        return bool(self.settings.discord_alerts_enabled and self.settings.discord_webhook_url.strip())

    async def _send_signal_alert(self, decision: Decision, prediction: dict, context: dict) -> None:
        if not self._discord_alerts_enabled() or not self.settings.discord_alert_on_signal:
            return
        if decision.action == "hold" and not self.settings.discord_alert_on_hold:
            return
        try:
            await self.notifier.signal_alert(
                decision.symbol,
                decision.action,
                decision.reason,
                prediction["buy_probability"],
                prediction["sell_probability"],
                spread_bps=context.get("spread_bps"),
                quote_imbalance=context.get("quote_imbalance"),
                latest_price=context.get("latest_price"),
                mid_price=context.get("mid_price"),
                bar_age_seconds=context.get("bar_age_seconds"),
                quote_age_seconds=context.get("quote_age_seconds"),
                prediction_source=prediction.get("prediction_source"),
                model_version=_prediction_model_version(prediction),
            )
        except Exception as exc:
            self._log_discord_alert_failure("signal", exc)

    async def _send_order_alert(
        self,
        decision: Decision,
        order_response: dict,
        *,
        local_order_id: str | int | None = None,
    ) -> None:
        if not self._discord_alerts_enabled() or not self.settings.discord_alert_on_order:
            return
        try:
            optional_fields = {
                "local_order_id": local_order_id,
                "limit_price": order_response.get("limit_price"),
                "filled_qty": order_response.get("filled_qty"),
                "filled_avg_price": order_response.get("filled_avg_price"),
                "fee_amount": order_response.get("fee_amount"),
                "slippage_amount": order_response.get("slippage_amount"),
                "cancel_reason": _cancel_reason(order_response, default_order_type=self.settings.order_type, default_time_in_force=self.settings.time_in_force),
            }
            await self.notifier.order_alert(
                decision.action,
                order_response.get("status", "submitted"),
                decision.notional,
                decision.qty,
                broker_order_id=order_response.get("id"),
                order_type=order_response.get("order_type", self.settings.order_type),
                time_in_force=order_response.get("time_in_force", self.settings.time_in_force),
                **{name: value for name, value in optional_fields.items() if value is not None},
            )
        except Exception as exc:
            self._log_discord_alert_failure("order", exc)

    async def _send_error_alert(self, where: str, error: Exception) -> None:
        if not self._discord_alerts_enabled() or not self.settings.discord_alert_on_error:
            return
        try:
            await self.notifier.error_alert(where, error)
        except Exception as exc:
            self._log_discord_alert_failure("error", exc)

    async def _send_risk_alert(self, decision: Decision, *, risk_context: dict[str, Any] | None = None) -> None:
        if decision.action != "hold" or decision.reason not in RISK_ALERT_REASONS:
            return
        if not self._discord_alerts_enabled():
            return
        now = datetime.now(UTC)
        if (
            self._last_risk_alert_reason == decision.reason
            and self._last_risk_alert_sent_at is not None
            and (now - self._last_risk_alert_sent_at).total_seconds()
            < self.settings.discord_risk_alert_cooldown_seconds
        ):
            return
        try:
            details = {
                name: value
                for name, value in {
                    "relevant_limit": (risk_context or {}).get("relevant_limit"),
                    "current_value": (risk_context or {}).get("current_value"),
                    "reset_time": (risk_context or {}).get("reset_time"),
                }.items()
                if value is not None
            }
            try:
                await self.notifier.risk_alert(decision.reason, **details)
            except TypeError:
                await self.notifier.risk_alert(decision.reason)
            self._last_risk_alert_reason = decision.reason
            self._last_risk_alert_sent_at = now
        except Exception as exc:
            self._log_discord_alert_failure("risk", exc)

    def _log_discord_alert_failure(self, alert_type: str, error: Exception) -> None:
        try:
            self.logger.event("discord_alert_failed", alert_type=alert_type, error_type=type(error).__name__)
        except Exception:
            pass

    def _alert_market_context(
        self,
        feature_row,
        quote: dict,
        *,
        latest_bar_timestamp: str | None = None,
    ) -> dict:
        spread_bps = _feature_float(feature_row, "scalping_spread_bps")
        if spread_bps is None:
            spread_pct = _feature_float(feature_row, "orderbook_spread") or 0.0
            spread_bps = spread_pct * 10_000
        quote_imbalance = _feature_float(feature_row, "scalping_quote_imbalance")
        if quote_imbalance is None:
            quote_imbalance = _feature_float(feature_row, "quote_imbalance") or 0.0
        latest_price = float(feature_row.get("close", 0) or 0)
        bid = _float_quote(quote, "bid_price", "bp", "bid")
        ask = _float_quote(quote, "ask_price", "ap", "ask")
        mid_price = (bid + ask) / 2 if bid is not None and ask is not None else _float_quote(quote, "mid_price", "mid")
        quote_age_seconds = _quote_age_seconds(quote)
        if quote_age_seconds is None and hasattr(self.market_data, "quote_cache_age_seconds"):
            quote_age_seconds = self.market_data.quote_cache_age_seconds(symbol=ALLOWED_SYMBOL)
        return {
            "spread_bps": spread_bps,
            "quote_imbalance": quote_imbalance,
            "latest_price": latest_price,
            "mid_price": mid_price,
            "bar_age_seconds": _age_seconds(latest_bar_timestamp),
            "quote_age_seconds": quote_age_seconds,
            "momentum": _feature_float(feature_row, "scalping_momentum_3", "scalping_log_return_3", "log_return_3"),
            "volatility": _feature_float(feature_row, "scalping_volatility_5", "scalping_volatility_3", "volatility_20"),
        }

    def _risk_block_context(
        self,
        decision: Decision,
        *,
        market_context: dict[str, Any],
        position: PositionState,
        order_attempt_frequency,
        filled_trade_frequency,
        account_state: AccountState,
        api_budget: dict[str, Any],
        recent_ioc_canceled_buys: int,
        latest_ioc_canceled_buy_at: datetime | None,
    ) -> dict[str, Any] | None:
        if decision.action != "hold" or decision.reason not in RISK_ALERT_REASONS:
            return None
        return _risk_block_details(
            self.settings,
            decision.reason,
            market_context=market_context,
            position=position,
            order_attempt_frequency=order_attempt_frequency,
            filled_trade_frequency=filled_trade_frequency,
            account_state=account_state,
            api_budget=api_budget,
            recent_ioc_canceled_buys=recent_ioc_canceled_buys,
            latest_ioc_canceled_buy_at=latest_ioc_canceled_buy_at,
        )


def _float_quote(quote: dict, *keys: str) -> float | None:
    for key in keys:
        value = quote.get(key)
        if value is None or value == "":
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _store_signal(repo: Repository, *, decision: Decision, prediction: dict, market_context: dict) -> None:
    args = (
        decision.action,
        prediction["buy_probability"],
        prediction["sell_probability"],
        decision.reason,
    )
    try:
        repo.add_signal(
            *args,
            spread_bps=market_context.get("spread_bps"),
            quote_imbalance=market_context.get("quote_imbalance"),
            model_version=_prediction_model_version(prediction),
        )
    except TypeError:
        # Keep custom repositories and older integrations compatible during rollout.
        repo.add_signal(*args)


def _feature_float(feature_row, *keys: str) -> float | None:
    for key in keys:
        value = feature_row.get(key)
        if value is None:
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed):
            return parsed
    return None


def _mid_price_from_quote(quote: dict) -> float | None:
    bid = _float_quote(quote, "bid_price", "bp", "bid")
    ask = _float_quote(quote, "ask_price", "ap", "ask")
    if bid is not None and ask is not None:
        return (bid + ask) / 2
    return _float_quote(quote, "mid_price", "mid")


def _latest_bar_timestamp(feature_row, prediction: dict) -> str | None:
    value = None
    try:
        value = feature_row.get("timestamp")
    except AttributeError:
        value = None
    if value is None:
        value = prediction.get("timestamp")
    if value is None:
        return None
    try:
        return value.isoformat()
    except AttributeError:
        return str(value)


def _ensure_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _order_log_payload(
    decision: Decision,
    order_response: dict[str, Any],
    *,
    local_order_id: str | int | None,
    default_order_type: str,
    default_time_in_force: str,
) -> dict[str, Any]:
    return {
        "order_id": order_response.get("id"),
        "local_order_id": local_order_id,
        "side": decision.action,
        "order_type": order_response.get("order_type") or order_response.get("type") or default_order_type,
        "time_in_force": order_response.get("time_in_force") or default_time_in_force,
        "requested_notional": order_response.get("requested_notional", decision.notional),
        "requested_qty": order_response.get("requested_qty", decision.qty),
        "limit_price": order_response.get("limit_price"),
        "filled_qty": order_response.get("filled_qty"),
        "filled_avg_price": order_response.get("filled_avg_price"),
        "fee_amount": order_response.get("fee_amount"),
        "slippage_amount": order_response.get("slippage_amount"),
        "status": order_response.get("status", "submitted"),
        "cancel_reason": _cancel_reason(
            order_response,
            default_order_type=default_order_type,
            default_time_in_force=default_time_in_force,
        ),
    }


def _risk_block_details(
    settings: Settings,
    reason: str,
    *,
    market_context: dict[str, Any],
    position: PositionState,
    order_attempt_frequency,
    filled_trade_frequency,
    account_state: AccountState,
    api_budget: dict[str, Any],
    recent_ioc_canceled_buys: int,
    latest_ioc_canceled_buy_at: datetime | None,
) -> dict[str, Any]:
    details: dict[str, Any] = {
        "reason": reason,
        "relevant_limit": None,
        "current_value": None,
        "reset_time": None,
    }
    limit_and_value = {
        "max_order_attempts_per_10_minutes_reached": (
            settings.max_order_attempts_per_10_minutes,
            _state_value(order_attempt_frequency, "trades_last_10_minutes"),
        ),
        "max_order_attempts_per_hour_reached": (
            settings.max_order_attempts_per_hour,
            _state_value(order_attempt_frequency, "trades_last_hour"),
        ),
        "max_order_attempts_per_day_reached": (
            settings.max_order_attempts_per_day,
            _state_value(order_attempt_frequency, "trades_today"),
        ),
        "max_trades_per_10_minutes_reached": (
            settings.max_trades_per_10_minutes,
            _state_value(filled_trade_frequency, "trades_last_10_minutes"),
        ),
        "max_trades_per_hour_reached": (
            settings.max_trades_per_hour,
            _state_value(filled_trade_frequency, "trades_last_hour"),
        ),
        "max_daily_trades_reached": (
            settings.max_daily_trades,
            _state_value(filled_trade_frequency, "trades_today"),
        ),
        "max_consecutive_losses_reached": (
            settings.max_consecutive_losses,
            _state_value(filled_trade_frequency, "consecutive_losses"),
        ),
        "scalping_kill_switch:hourly_loss_limit": (
            -abs(settings.max_loss_usd_per_hour),
            _state_value(filled_trade_frequency, "realized_pnl_last_hour"),
        ),
        "scalping_kill_switch:ioc_cancel_streak": (
            settings.max_consecutive_ioc_cancels,
            recent_ioc_canceled_buys,
        ),
        "recent_ioc_cancels_too_high": (
            settings.max_recent_ioc_cancels,
            recent_ioc_canceled_buys,
        ),
        "spread_too_wide": (settings.max_spread_bps, market_context.get("spread_bps")),
        "stale_market_data": (settings.scalping_max_data_age_seconds, market_context.get("bar_age_seconds")),
        "quote_imbalance_too_weak": (settings.min_quote_imbalance, market_context.get("quote_imbalance")),
        "api_budget_exhausted": (
            api_budget.get("hard_stop_calls_per_minute"),
            api_budget.get("calls_last_minute"),
        ),
        "buying_power_too_low": (settings.order_notional_usd, account_state.buying_power),
        "account_daily_loss_usd_reached": (-abs(settings.max_account_daily_loss_usd), account_state.daily_change_usd),
        "account_daily_loss_pct_reached": (-abs(settings.max_account_daily_loss_pct), account_state.daily_change_pct),
        "account_drawdown_reached": (settings.max_account_drawdown_pct, account_state.drawdown_pct),
        "profit_guard_holding_until_profitable": (
            settings.min_net_exit_profit_pct,
            _unrealized_pnl_pct(position, market_context.get("mid_price") or market_context.get("latest_price")),
        ),
        "profit_guard_holding_at_loss": (
            settings.min_net_exit_profit_pct,
            _unrealized_pnl_pct(position, market_context.get("mid_price") or market_context.get("latest_price")),
        ),
    }
    if reason in limit_and_value:
        details["relevant_limit"], details["current_value"] = limit_and_value[reason]
    if reason == "trade_cooldown_active":
        details["relevant_limit"] = settings.min_seconds_between_trades
        last_trade_at = _state_value(filled_trade_frequency, "last_trade_at")
        if last_trade_at is not None:
            details["reset_time"] = (_ensure_utc(last_trade_at) + timedelta(seconds=settings.min_seconds_between_trades)).isoformat()
    if reason == "ioc_cancel_cooldown_active" and latest_ioc_canceled_buy_at is not None:
        details["relevant_limit"] = settings.ioc_cancel_cooldown_seconds
        details["current_value"] = recent_ioc_canceled_buys
        details["reset_time"] = (
            _ensure_utc(latest_ioc_canceled_buy_at) + timedelta(seconds=settings.ioc_cancel_cooldown_seconds)
        ).isoformat()
    return details


def _blocked_runtime_decision(decision: Decision, reason: str, blocked_by: str) -> Decision:
    return Decision(
        decision.symbol,
        "hold",
        reason,
        blocked_by=blocked_by,
        block_reason=reason,
        strategy_name=decision.strategy_name,
        strategy_score=decision.strategy_score,
        strategy_confidence=decision.strategy_confidence,
        regime=decision.regime,
        regime_confidence=decision.regime_confidence,
        ml_confirmation=decision.ml_confirmation,
        strategy_candidates=decision.strategy_candidates,
        metadata=decision.metadata,
    )


def _cancel_reason(
    order_response: dict[str, Any],
    *,
    default_order_type: str,
    default_time_in_force: str,
) -> str | None:
    explicit = order_response.get("cancel_reason") or order_response.get("reason")
    if explicit:
        return str(explicit)
    status = str(order_response.get("status") or "").lower()
    order_type = str(order_response.get("order_type") or order_response.get("type") or default_order_type).lower()
    time_in_force = str(order_response.get("time_in_force") or default_time_in_force).lower()
    if status in {"canceled", "cancelled"} and order_type == "limit" and time_in_force == "ioc":
        return "ioc_no_fill"
    if status in {"canceled", "cancelled"}:
        return "broker_canceled"
    return None


def _state_value(state: Any, name: str) -> Any:
    return getattr(state, name, None) if state is not None else None


def _prediction_model_version(prediction: dict[str, Any]) -> str | None:
    version = prediction.get("active_model_version") or prediction.get("model_version")
    if version:
        return str(version)
    model_path = prediction.get("model_path")
    if not model_path:
        return None
    return str(model_path).rsplit("/", 1)[-1].removesuffix(".joblib")


def _quote_age_seconds(quote: dict[str, Any]) -> float | None:
    for name in ("timestamp", "t", "quote_timestamp"):
        age_seconds = _age_seconds(quote.get(name))
        if age_seconds is not None:
            return age_seconds
    return None


def _age_seconds(value: Any) -> float | None:
    if value is None:
        return None
    try:
        timestamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    timestamp = _ensure_utc(timestamp)
    return max(0.0, (datetime.now(UTC) - timestamp).total_seconds())


def _unrealized_pnl_pct(position: PositionState, latest_price: Any) -> float | None:
    try:
        parsed_price = float(latest_price)
    except (TypeError, ValueError):
        return None
    if not position.has_position or position.avg_entry_price <= 0 or not math.isfinite(parsed_price):
        return None
    return (parsed_price - position.avg_entry_price) / position.avg_entry_price
