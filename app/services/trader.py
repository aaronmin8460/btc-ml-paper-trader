import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

from app.accounting.trade_accounting import record_filled_order_trade
from app.broker.alpaca_client import AlpacaClient
from app.config import ALLOWED_SYMBOL, Settings, get_settings
from app.data.feature_engineering import latest_feature_row
from app.data.market_data import MarketDataClient
from app.db.database import SessionLocal, init_db
from app.db.repository import Repository
from app.ml.predict import Predictor
from app.monitoring.logger import get_logger
from app.notifications.discord import DiscordNotifier
from app.risk.risk_manager import AccountState, PositionState, account_state_from_payload
from app.strategy.decision_engine import Decision, DecisionEngine
from app.utils.rate_limiter import get_alpaca_rate_limiter


KILL_SWITCH_REASONS = {
    "max_order_attempts_per_hour_reached",
    "max_order_attempts_per_day_reached",
    "max_trades_per_hour_reached",
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
    "ioc_cancel_cooldown_active",
    "model_not_profitable_after_costs",
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
    "scalping_emergency_stop_loss",
    "scalping_take_profit",
    "scalping_trailing_stop",
    "scalping_max_position_seconds",
}


class Trader:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.market_data = MarketDataClient(self.settings)
        self.predictor = Predictor(self.settings)
        self.decision_engine = DecisionEngine(self.settings)
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

    async def run_once(self) -> dict:
        try:
            init_db()
            bars = await self.market_data.fetch_bars(ALLOWED_SYMBOL)
            quote = await self.market_data.fetch_latest_quote(ALLOWED_SYMBOL)
            prediction = self.predictor.predict(bars, quote=quote)
            feature_row = latest_feature_row(bars, quote=quote).iloc[-1]
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
                )
                self.logger.event(
                    "signal",
                    symbol=decision.symbol,
                    action=decision.action,
                    reason=decision.reason,
                    buy_probability=prediction["buy_probability"],
                    sell_probability=prediction["sell_probability"],
                    **self._alert_market_context(feature_row, quote),
                    api_budget_status=api_budget.get("api_budget_status"),
                    alpaca_calls_last_minute=api_budget.get("calls_last_minute"),
                )
                await self._send_signal_alert(decision, prediction, self._alert_market_context(feature_row, quote))
                await self._send_risk_alert(decision)
                repo.add_signal(
                    decision.action,
                    prediction["buy_probability"],
                    prediction["sell_probability"],
                    decision.reason,
                )
                order_response = None
                try:
                    if decision.action in {"buy", "sell"}:
                        self._remember_order_attempt_bar(decision, latest_bar_timestamp)
                        order_response = await self.broker.submit_order(
                            symbol=decision.symbol,
                            side=decision.action,
                            notional=decision.notional,
                            qty=decision.qty,
                            current_position_qty=position.qty,
                            quote=quote,
                            latest_price=quote_mid_price or float(feature_row["close"]),
                        )
                        self.broker.invalidate_position_cache(decision.symbol)
                        await self._send_order_alert(decision, order_response)
                        order = repo.add_order(
                            side=decision.action,
                            status=order_response.get("status", "submitted"),
                            notional=decision.notional,
                            qty=decision.qty,
                            broker_order_id=order_response.get("id"),
                            raw_response=order_response,
                        )
                        if str(order_response.get("status") or "").lower() == "filled":
                            record_filled_order_trade(repo, order, self.settings)
                finally:
                    if order_lock_acquired and self._order_lock.locked():
                        self._order_lock.release()
                        self._order_lock_started_at = None
            return {"prediction": prediction, "decision": decision.__dict__, "order": order_response}
        except Exception as exc:
            await self._send_error_alert("trader.run_once", exc)
            raise

    async def _account_state(self) -> AccountState:
        return account_state_from_payload(await self._account_payload())

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
    ) -> tuple[Decision, bool]:
        if decision.action == "buy" and position.has_position:
            return Decision(decision.symbol, "hold", "already_holding_btc"), False
        if decision.action == "sell" and not position.has_position:
            return Decision(decision.symbol, "hold", "sell_without_position"), False
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
            return Decision(decision.symbol, "hold", "duplicate_order_bar"), False
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
            return Decision(decision.symbol, "hold", "order_in_flight"), False
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
            return Decision(decision.symbol, "hold", "order_in_flight"), False
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
            )
        except Exception as exc:
            self._log_discord_alert_failure("signal", exc)

    async def _send_order_alert(self, decision: Decision, order_response: dict) -> None:
        if not self._discord_alerts_enabled() or not self.settings.discord_alert_on_order:
            return
        try:
            await self.notifier.order_alert(
                decision.action,
                order_response.get("status", "submitted"),
                decision.notional,
                decision.qty,
                broker_order_id=order_response.get("id"),
                order_type=order_response.get("order_type", self.settings.order_type),
                time_in_force=order_response.get("time_in_force", self.settings.time_in_force),
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

    async def _send_risk_alert(self, decision: Decision) -> None:
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

    def _alert_market_context(self, feature_row, quote: dict) -> dict:
        spread_bps = float(feature_row.get("orderbook_spread", 0) or 0) * 10_000
        quote_imbalance = float(feature_row.get("quote_imbalance", 0) or 0)
        latest_price = float(feature_row.get("close", 0) or 0)
        bid = _float_quote(quote, "bid_price", "bp", "bid")
        ask = _float_quote(quote, "ask_price", "ap", "ask")
        mid_price = (bid + ask) / 2 if bid is not None and ask is not None else _float_quote(quote, "mid_price", "mid")
        return {
            "spread_bps": spread_bps,
            "quote_imbalance": quote_imbalance,
            "latest_price": latest_price,
            "mid_price": mid_price,
        }


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
