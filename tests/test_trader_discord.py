from datetime import timedelta

import pandas as pd
import pytest

from app.config import Settings
from app.risk.risk_manager import PositionState
from app.services.trader import Trader
from app.strategy.decision_engine import Decision


class FakeLogger:
    def __init__(self) -> None:
        self.events = []

    def event(self, event_type: str, **payload) -> None:
        self.events.append((event_type, payload))


class FakeMarketData:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    async def fetch_bars(self, symbol):
        if self.error:
            raise self.error
        return pd.DataFrame()

    async def fetch_latest_quote(self, symbol):
        return {}


class FakePredictor:
    def predict(self, bars, quote=None):
        return {
            "symbol": "BTC/USD",
            "buy_probability": 0.75,
            "sell_probability": 0.15,
            "sell_probability_source": "independent_sell_model",
        }


class FakeDecisionEngine:
    def __init__(self, decision: Decision) -> None:
        self.decision = decision

    def decide(self, **kwargs):
        return self.decision


class FakeBroker:
    def __init__(
        self,
        events: list[str],
        order_response: dict | None = None,
        position: dict | None = None,
        account_response: dict | None = None,
    ) -> None:
        self.events = events
        self.order_response = order_response or {"id": "paper-order-1", "status": "submitted"}
        self.position = position
        self.account_response = account_response

    async def get_position(self, symbol):
        return self.position

    def credentials_available(self):
        return self.account_response is not None

    async def get_account(self):
        if self.account_response is None:
            raise AssertionError("get_account should not be called without credentials")
        return self.account_response

    async def submit_market_order(self, **kwargs):
        self.events.append("submit_order")
        if isinstance(self.order_response, Exception):
            raise self.order_response
        return self.order_response

    async def submit_order(self, **kwargs):
        return await self.submit_market_order(**kwargs)

    def invalidate_position_cache(self, symbol=None):
        self.events.append("invalidate_position_cache")


class FakeNotifier:
    def __init__(self, events: list[str] | None = None, fail: bool = False) -> None:
        self.events = events if events is not None else []
        self.calls = []
        self.fail = fail

    async def signal_alert(self, *args, **kwargs):
        if self.fail:
            raise RuntimeError("discord failed")
        self.events.append("signal_alert")
        self.calls.append(("signal", args, kwargs))

    async def order_alert(self, *args, **kwargs):
        if self.fail:
            raise RuntimeError("discord failed")
        self.events.append("order_alert")
        self.calls.append(("order", args, kwargs))

    async def error_alert(self, *args):
        if self.fail:
            raise RuntimeError("discord failed")
        self.events.append("error_alert")
        self.calls.append(("error", args))

    async def risk_alert(self, *args):
        if self.fail:
            raise RuntimeError("discord failed")
        self.events.append("risk_alert")
        self.calls.append(("risk", args))


class RaisingNotifier:
    async def signal_alert(self, *args, **kwargs):
        raise AssertionError("Discord signal alert should not be called")

    async def order_alert(self, *args, **kwargs):
        raise AssertionError("Discord order alert should not be called")

    async def error_alert(self, *args):
        raise AssertionError("Discord error alert should not be called")

    async def risk_alert(self, *args):
        raise AssertionError("Discord risk alert should not be called")


class FakeSession:
    def __enter__(self):
        return object()

    def __exit__(self, exc_type, exc, traceback):
        return False


class FakeRepository:
    signals = []
    orders = []
    account_snapshots = []
    trade_frequency = None
    order_to_return = None

    def __init__(self, db) -> None:
        pass

    @classmethod
    def reset(cls) -> None:
        cls.signals = []
        cls.orders = []
        cls.account_snapshots = []
        cls.trade_frequency = None
        cls.order_to_return = None

    def add_signal(self, *args) -> None:
        self.signals.append(args)

    def add_order(self, **kwargs):
        self.orders.append(kwargs)
        return self.order_to_return

    def add_account_snapshot(self, **kwargs):
        self.account_snapshots.append(kwargs)

    def trade_frequency_state(self):
        return self.trade_frequency


@pytest.fixture
def trader_factory(monkeypatch):
    def build_trader(
        *,
        settings: Settings,
        decision: Decision,
        notifier=None,
        events: list[str] | None = None,
        market_error: Exception | None = None,
        trade_frequency=None,
        stored_order=None,
        account_response: dict | None = None,
    ) -> Trader:
        FakeRepository.reset()
        FakeRepository.trade_frequency = trade_frequency
        FakeRepository.order_to_return = stored_order
        monkeypatch.setattr("app.services.trader.init_db", lambda: None)
        monkeypatch.setattr("app.services.trader.SessionLocal", FakeSession)
        monkeypatch.setattr("app.services.trader.Repository", FakeRepository)
        monkeypatch.setattr(
            "app.services.trader.latest_feature_row",
            lambda bars, quote=None: pd.DataFrame([{"close": 65000.0}]),
        )

        flow_events = events if events is not None else []
        trader = Trader(settings)
        trader.market_data = FakeMarketData(error=market_error)
        trader.predictor = FakePredictor()
        trader.decision_engine = FakeDecisionEngine(decision)
        position = None
        if decision.action == "sell":
            position = {"qty": "0.01", "avg_entry_price": "65000", "market_value": "650", "current_price": "65000"}
        trader.broker = FakeBroker(flow_events, position=position, account_response=account_response)
        trader.logger = FakeLogger()
        trader.notifier = notifier if notifier is not None else FakeNotifier(flow_events)
        return trader

    return build_trader


def _risk_alert_settings(**overrides) -> Settings:
    defaults = {
        "_env_file": None,
        "discord_alerts_enabled": True,
        "discord_webhook_url": "https://discord.example/webhook",
        "discord_risk_alert_cooldown_seconds": 300,
    }
    defaults.update(overrides)
    return Settings(**defaults)


@pytest.mark.anyio
@pytest.mark.parametrize("action", ["buy", "sell"])
async def test_signal_alert_called_for_buy_sell(trader_factory, action):
    settings = Settings(
        _env_file=None,
        trading_enabled=True,
        discord_alerts_enabled=True,
        discord_webhook_url="https://discord.example/webhook",
        discord_alert_on_signal=True,
        discord_alert_on_order=False,
    )
    decision = Decision(
        "BTC/USD",
        action,
        "test_signal",
        notional=25 if action == "buy" else None,
        qty=0.01 if action == "sell" else None,
    )
    notifier = FakeNotifier()
    trader = trader_factory(settings=settings, decision=decision, notifier=notifier)

    await trader.run_once()

    assert notifier.calls[0][0] == "signal"
    assert notifier.calls[0][1] == ("BTC/USD", action, "test_signal", 0.75, 0.15)
    signal_events = [payload for event_type, payload in trader.logger.events if event_type == "signal"]
    assert signal_events[0]["sell_probability_source"] == "independent_sell_model"


@pytest.mark.anyio
async def test_hold_alert_skipped_by_default(trader_factory):
    settings = Settings(
        _env_file=None,
        discord_alerts_enabled=True,
        discord_webhook_url="https://discord.example/webhook",
        discord_alert_on_signal=True,
        discord_alert_on_hold=False,
    )
    decision = Decision("BTC/USD", "hold", "trading_disabled")
    notifier = FakeNotifier()
    trader = trader_factory(settings=settings, decision=decision, notifier=notifier)

    result = await trader.run_once()

    assert result["decision"]["action"] == "hold"
    assert notifier.calls == []


@pytest.mark.anyio
async def test_order_alert_called_after_order_submission(trader_factory):
    settings = Settings(
        _env_file=None,
        trading_enabled=True,
        discord_alerts_enabled=True,
        discord_webhook_url="https://discord.example/webhook",
        discord_alert_on_signal=False,
        discord_alert_on_order=True,
    )
    decision = Decision("BTC/USD", "buy", "ml_and_rules_approved", notional=25)
    events = []
    notifier = FakeNotifier(events)
    trader = trader_factory(settings=settings, decision=decision, notifier=notifier, events=events)

    result = await trader.run_once()

    assert result["order"] == {"id": "paper-order-1", "status": "submitted"}
    assert events == ["submit_order", "invalidate_position_cache", "order_alert"]
    assert notifier.calls == [
            (
                "order",
                ("buy", "submitted", 25, None),
                {"broker_order_id": "paper-order-1", "order_type": "limit", "time_in_force": "ioc"},
            )
        ]


@pytest.mark.anyio
async def test_filled_order_updates_trade_accounting(trader_factory, monkeypatch):
    settings = Settings(_env_file=None, trading_enabled=True, discord_alert_on_order=False)
    decision = Decision("BTC/USD", "buy", "ml_and_rules_approved", notional=25)
    stored_order = object()
    calls = []

    def fake_record_filled_order_trade(repo, order, settings):
        calls.append((repo, order, settings))

    monkeypatch.setattr("app.services.trader.record_filled_order_trade", fake_record_filled_order_trade)
    trader = trader_factory(
        settings=settings,
        decision=decision,
        stored_order=stored_order,
    )
    trader.broker.order_response = {
        "id": "paper-order-1",
        "status": "filled",
        "filled_qty": "0.001",
        "filled_avg_price": "65000",
    }

    await trader.run_once()

    assert len(calls) == 1
    assert calls[0][1] is stored_order
    assert calls[0][2] is settings


@pytest.mark.anyio
async def test_partially_filled_order_updates_trade_accounting(trader_factory, monkeypatch):
    settings = Settings(_env_file=None, trading_enabled=True, discord_alert_on_order=False)
    decision = Decision("BTC/USD", "buy", "ml_and_rules_approved", notional=25)
    stored_order = object()
    calls = []

    monkeypatch.setattr(
        "app.services.trader.record_filled_order_trade",
        lambda repo, order, settings: calls.append((repo, order, settings)),
    )
    trader = trader_factory(settings=settings, decision=decision, stored_order=stored_order)
    trader.broker.order_response = {
        "id": "paper-order-1",
        "status": "partially_filled",
        "filled_qty": "0.0005",
        "filled_avg_price": "65000",
    }

    await trader.run_once()

    assert len(calls) == 1
    assert calls[0][1] is stored_order


def test_trader_surfaces_ioc_cancel_streak_kill_switch_reason():
    class FakeKillSwitchRepository:
        def realized_pnl_last_hour(self):
            return 0

        def consecutive_ioc_canceled_count(self):
            return 5

    trader = Trader(
        Settings(
            _env_file=None,
            scalping_mode_enabled=True,
            max_consecutive_ioc_cancels=5,
        )
    )

    assert trader._scalping_kill_switch_reason(FakeKillSwitchRepository()) == "scalping_kill_switch:ioc_cancel_streak"


def test_trader_surfaces_hourly_loss_kill_switch_reason():
    class FakeKillSwitchRepository:
        def realized_pnl_last_hour(self):
            return -5

        def consecutive_ioc_canceled_count(self):
            return 0

    trader = Trader(
        Settings(
            _env_file=None,
            scalping_mode_enabled=True,
            max_loss_usd_per_hour=5,
        )
    )

    assert trader._scalping_kill_switch_reason(FakeKillSwitchRepository()) == "scalping_kill_switch:hourly_loss_limit"


@pytest.mark.anyio
async def test_account_snapshot_saved_when_account_payload_available(trader_factory):
    settings = Settings(_env_file=None, trading_enabled=False)
    account = {
        "equity": "1000.50",
        "cash": "900.25",
        "buying_power": "800.75",
        "portfolio_value": "1001.00",
        "currency": "USD",
        "secret_key": "must-not-leak",
    }
    trader = trader_factory(
        settings=settings,
        decision=Decision("BTC/USD", "hold", "dashboard_test"),
        account_response=account,
    )

    await trader.run_once()

    assert FakeRepository.account_snapshots == [
        {
            "equity": "1000.50",
            "cash": "900.25",
            "buying_power": "800.75",
            "portfolio_value": "1001.00",
            "currency": "USD",
            "raw_response": account,
        }
    ]


@pytest.mark.anyio
async def test_error_alert_called_when_run_once_raises(trader_factory):
    settings = Settings(
        _env_file=None,
        discord_alerts_enabled=True,
        discord_webhook_url="https://discord.example/webhook",
        discord_alert_on_error=True,
    )
    notifier = FakeNotifier()
    trader = trader_factory(
        settings=settings,
        decision=Decision("BTC/USD", "hold", "not_reached"),
        notifier=notifier,
        market_error=RuntimeError("market data failed"),
    )

    with pytest.raises(RuntimeError, match="market data failed"):
        await trader.run_once()

    assert len(notifier.calls) == 1
    call_type, args = notifier.calls[0]
    assert call_type == "error"
    assert args[0] == "trader.run_once"
    assert isinstance(args[1], RuntimeError)
    assert str(args[1]) == "market data failed"


@pytest.mark.anyio
@pytest.mark.parametrize(
    "reason",
    [
        "max_trades_per_hour_reached",
        "max_order_attempts_per_hour_reached",
        "profit_guard_holding_until_profitable",
        "profit_guard_holding_at_loss",
    ],
)
async def test_kill_switch_block_sends_risk_alert(trader_factory, reason):
    settings = Settings(
        _env_file=None,
        discord_alerts_enabled=True,
        discord_webhook_url="https://discord.example/webhook",
        discord_alert_on_signal=True,
        discord_alert_on_hold=False,
    )
    notifier = FakeNotifier()
    trader = trader_factory(
        settings=settings,
        decision=Decision("BTC/USD", "hold", reason),
        notifier=notifier,
    )

    await trader.run_once()

    assert notifier.calls == [("risk", (reason,))]


@pytest.mark.anyio
async def test_first_risk_alert_is_sent(trader_factory):
    notifier = FakeNotifier()
    trader = trader_factory(
        settings=_risk_alert_settings(),
        decision=Decision("BTC/USD", "hold", "trade_cooldown_active"),
        notifier=notifier,
    )

    await trader._send_risk_alert(Decision("BTC/USD", "hold", "trade_cooldown_active"))

    assert notifier.calls == [("risk", ("trade_cooldown_active",))]


@pytest.mark.anyio
async def test_repeated_same_risk_alert_within_cooldown_is_skipped(trader_factory):
    notifier = FakeNotifier()
    trader = trader_factory(
        settings=_risk_alert_settings(),
        decision=Decision("BTC/USD", "hold", "trade_cooldown_active"),
        notifier=notifier,
    )
    decision = Decision("BTC/USD", "hold", "trade_cooldown_active")

    await trader._send_risk_alert(decision)
    await trader._send_risk_alert(decision)

    assert notifier.calls == [("risk", ("trade_cooldown_active",))]


@pytest.mark.anyio
async def test_different_risk_reason_sends_immediately(trader_factory):
    notifier = FakeNotifier()
    trader = trader_factory(
        settings=_risk_alert_settings(),
        decision=Decision("BTC/USD", "hold", "trade_cooldown_active"),
        notifier=notifier,
    )

    await trader._send_risk_alert(Decision("BTC/USD", "hold", "trade_cooldown_active"))
    await trader._send_risk_alert(Decision("BTC/USD", "hold", "api_budget_exhausted"))

    assert notifier.calls == [
        ("risk", ("trade_cooldown_active",)),
        ("risk", ("api_budget_exhausted",)),
    ]


@pytest.mark.anyio
async def test_same_risk_reason_after_cooldown_sends_again(trader_factory):
    notifier = FakeNotifier()
    settings = _risk_alert_settings(discord_risk_alert_cooldown_seconds=300)
    trader = trader_factory(
        settings=settings,
        decision=Decision("BTC/USD", "hold", "trade_cooldown_active"),
        notifier=notifier,
    )
    decision = Decision("BTC/USD", "hold", "trade_cooldown_active")

    await trader._send_risk_alert(decision)
    assert trader._last_risk_alert_sent_at is not None
    trader._last_risk_alert_sent_at -= timedelta(
        seconds=settings.discord_risk_alert_cooldown_seconds + 1,
    )
    await trader._send_risk_alert(decision)

    assert notifier.calls == [
        ("risk", ("trade_cooldown_active",)),
        ("risk", ("trade_cooldown_active",)),
    ]


@pytest.mark.anyio
async def test_discord_disabled_keeps_existing_trader_behavior(trader_factory):
    settings = Settings(
        _env_file=None,
        trading_enabled=True,
        discord_alerts_enabled=False,
        discord_webhook_url="https://discord.example/webhook",
    )
    decision = Decision("BTC/USD", "buy", "ml_and_rules_approved", notional=25)
    trader = trader_factory(settings=settings, decision=decision, notifier=RaisingNotifier())

    result = await trader.run_once()

    assert result["decision"]["action"] == "buy"
    assert result["order"] == {"id": "paper-order-1", "status": "submitted"}
    assert FakeRepository.signals == [("buy", 0.75, 0.15, "ml_and_rules_approved")]
    assert FakeRepository.orders[0]["side"] == "buy"


@pytest.mark.anyio
async def test_discord_failure_does_not_interrupt_trading(trader_factory):
    settings = Settings(
        _env_file=None,
        trading_enabled=True,
        discord_alerts_enabled=True,
        discord_webhook_url="https://discord.example/webhook",
        discord_alert_on_signal=True,
        discord_alert_on_order=True,
    )
    decision = Decision("BTC/USD", "buy", "ml_and_rules_approved", notional=25)
    trader = trader_factory(settings=settings, decision=decision, notifier=FakeNotifier(fail=True))

    result = await trader.run_once()

    assert result["order"] == {"id": "paper-order-1", "status": "submitted"}
    assert [event[0] for event in trader.logger.events] == [
        "signal",
        "discord_alert_failed",
        "order",
        "discord_alert_failed",
    ]
    assert "discord.example" not in str(trader.logger.events)


@pytest.mark.anyio
async def test_structured_signal_log_contains_required_observability_fields(trader_factory):
    trader = trader_factory(
        settings=Settings(_env_file=None),
        decision=Decision(
            "BTC/USD",
            "hold",
            "dashboard_test",
            blocked_by="ml_filter",
            block_reason="ml_buy_probability_below_threshold",
            strategy_name="mean_reversion_scalping",
            strategy_score=0.62,
            strategy_confidence=0.74,
            regime="mean_reverting",
            ml_confirmation={"passed": False},
            strategy_candidates=[{"strategy_name": "mean_reversion_scalping", "score": 0.62}],
            metadata={"entry_reason": "mean_reversion_buy_candidate"},
        ),
    )

    await trader.run_once()

    signal = next(payload for event_type, payload in trader.logger.events if event_type == "signal")
    assert {
        "symbol",
        "action",
        "reason",
        "timestamp",
        "latest_bar_timestamp",
        "bar_age_seconds",
        "quote_age_seconds",
        "buy_probability",
        "sell_probability",
        "ml_buy_probability",
        "ml_sell_probability",
        "prediction_source",
        "model_version",
        "strategy_name",
        "quant_score",
        "quant_confidence",
        "regime",
        "blocked_by",
        "block_reason",
        "candidate_strategy_count",
        "strategy_candidates",
        "selected_strategy_signal",
        "selected_strategy_reason",
        "ml_confirmation_result",
        "final_decision",
        "spread_bps",
        "quote_imbalance",
        "momentum",
        "volatility",
        "position_qty",
        "avg_entry_price",
        "unrealized_pnl_pct",
        "risk_block_reason",
        "api_budget_status",
    }.issubset(signal)
    assert signal["strategy_name"] == "mean_reversion_scalping"
    assert signal["quant_score"] == 0.62
    assert signal["quant_confidence"] == 0.74
    assert signal["regime"] == "mean_reverting"
    assert signal["blocked_by"] == "ml_filter"
    assert signal["block_reason"] == "ml_buy_probability_below_threshold"
    assert signal["candidate_strategy_count"] == 1
    assert signal["strategy_candidates"] == [{"strategy_name": "mean_reversion_scalping", "score": 0.62}]
    assert signal["selected_strategy_signal"]["strategy_name"] == "mean_reversion_scalping"
    assert signal["selected_strategy_reason"] == "mean_reversion_buy_candidate"
    assert signal["ml_confirmation_result"] == {"passed": False}
    assert signal["final_decision"] == "hold"


@pytest.mark.anyio
async def test_structured_order_log_contains_required_execution_fields(trader_factory):
    trader = trader_factory(
        settings=Settings(_env_file=None, trading_enabled=True),
        decision=Decision("BTC/USD", "buy", "ml_and_rules_approved", notional=25),
    )

    await trader.run_once()

    order = next(payload for event_type, payload in trader.logger.events if event_type == "order")
    assert {
        "order_id",
        "local_order_id",
        "side",
        "order_type",
        "time_in_force",
        "requested_notional",
        "requested_qty",
        "limit_price",
        "filled_qty",
        "filled_avg_price",
        "fee_amount",
        "slippage_amount",
        "status",
        "cancel_reason",
    }.issubset(order)


@pytest.mark.anyio
async def test_structured_risk_block_log_contains_limit_value_and_reset_fields(trader_factory):
    trader = trader_factory(
        settings=Settings(_env_file=None),
        decision=Decision("BTC/USD", "hold", "max_trades_per_hour_reached"),
    )

    await trader.run_once()

    risk_block = next(payload for event_type, payload in trader.logger.events if event_type == "risk_block")
    assert risk_block["reason"] == "max_trades_per_hour_reached"
    assert {"relevant_limit", "current_value", "reset_time"}.issubset(risk_block)


@pytest.mark.anyio
async def test_canceled_ioc_order_log_has_clear_cancel_reason(trader_factory):
    trader = trader_factory(
        settings=Settings(_env_file=None, trading_enabled=True),
        decision=Decision("BTC/USD", "buy", "ml_and_rules_approved", notional=25),
    )
    trader.broker.order_response = {
        "id": "paper-order-1",
        "status": "canceled",
        "order_type": "limit",
        "time_in_force": "ioc",
    }

    await trader.run_once()

    order = next(payload for event_type, payload in trader.logger.events if event_type == "order")
    assert order["cancel_reason"] == "ioc_no_fill"


@pytest.mark.anyio
async def test_failed_order_attempt_is_logged_before_runtime_error(trader_factory):
    trader = trader_factory(
        settings=Settings(_env_file=None, trading_enabled=True),
        decision=Decision("BTC/USD", "buy", "ml_and_rules_approved", notional=25),
    )
    trader.broker.order_response = RuntimeError("submit failed")

    with pytest.raises(RuntimeError, match="submit failed"):
        await trader.run_once()

    event_types = [event_type for event_type, _ in trader.logger.events]
    assert "order" in event_types
    assert "runtime_error" in event_types
    order = next(payload for event_type, payload in trader.logger.events if event_type == "order")
    assert order["status"] == "failed"
    assert order["cancel_reason"] == "RuntimeError"


@pytest.mark.anyio
async def test_duplicate_order_lock_blocks_concurrent_attempt(trader_factory):
    settings = Settings(_env_file=None, trading_enabled=True)
    decision = Decision("BTC/USD", "buy", "ml_and_rules_approved", notional=25)
    trader = trader_factory(settings=settings, decision=decision)
    await trader._order_lock.acquire()

    result = await trader.run_once()

    assert result["decision"]["action"] == "hold"
    assert result["decision"]["reason"] == "order_in_flight"
    trader._order_lock.release()


@pytest.mark.anyio
async def test_sell_signal_without_position_is_blocked_before_order(trader_factory):
    settings = Settings(_env_file=None, trading_enabled=True)
    trader = trader_factory(settings=settings, decision=Decision("BTC/USD", "sell", "ml_sell_signal", qty=0.01))

    decision, acquired = await trader._guard_order_decision(
        Decision("BTC/USD", "sell", "ml_sell_signal", qty=0.01),
        PositionState(),
        latest_bar_timestamp="2026-05-29T12:00:00+00:00",
    )

    assert acquired is False
    assert decision.action == "hold"
    assert decision.reason == "sell_without_position"


@pytest.mark.anyio
async def test_duplicate_buy_on_same_latest_bar_is_blocked(trader_factory):
    settings = Settings(_env_file=None, trading_enabled=True)
    first = Decision("BTC/USD", "buy", "scalping_dip_entry", notional=25)
    trader = trader_factory(settings=settings, decision=first)
    trader._remember_order_attempt_bar(first, "2026-05-29T12:00:00+00:00")

    decision, acquired = await trader._guard_order_decision(
        Decision("BTC/USD", "buy", "scalping_dip_entry", notional=25),
        PositionState(),
        latest_bar_timestamp="2026-05-29T12:00:00+00:00",
    )

    assert acquired is False
    assert decision.action == "hold"
    assert decision.reason == "duplicate_order_bar"


@pytest.mark.anyio
async def test_duplicate_non_hard_sell_on_same_latest_bar_is_blocked(trader_factory):
    settings = Settings(_env_file=None, trading_enabled=True)
    first = Decision("BTC/USD", "sell", "scalping_weak_quote_exit", qty=0.01)
    trader = trader_factory(settings=settings, decision=first)
    trader._remember_order_attempt_bar(first, "2026-05-29T12:00:00+00:00")

    decision, acquired = await trader._guard_order_decision(
        Decision("BTC/USD", "sell", "scalping_weak_quote_exit", qty=0.01),
        PositionState(qty=0.01),
        latest_bar_timestamp="2026-05-29T12:00:00+00:00",
    )

    assert acquired is False
    assert decision.action == "hold"
    assert decision.reason == "duplicate_order_bar"


@pytest.mark.anyio
async def test_hard_risk_exit_sell_bypasses_duplicate_bar_guard(trader_factory):
    settings = Settings(_env_file=None, trading_enabled=True)
    first = Decision("BTC/USD", "sell", "scalping_weak_quote_exit", qty=0.01)
    trader = trader_factory(settings=settings, decision=first)
    trader._remember_order_attempt_bar(first, "2026-05-29T12:00:00+00:00")

    decision, acquired = await trader._guard_order_decision(
        Decision("BTC/USD", "sell", "scalping_emergency_stop_loss", qty=0.01),
        PositionState(qty=0.01),
        latest_bar_timestamp="2026-05-29T12:00:00+00:00",
    )

    assert acquired is True
    assert decision.action == "sell"
    trader._order_lock.release()


def test_ioc_cancel_escalation_cooldown_keeps_guard_active(trader_factory):
    class FakeIocRepository:
        def __init__(self, count: int) -> None:
            self.count = count

        def recent_ioc_canceled_buy_count(self, **kwargs):
            return self.count

        def latest_ioc_canceled_buy_at(self, **kwargs):
            return None

    settings = Settings(
        _env_file=None,
        max_recent_ioc_cancels=3,
        ioc_cancel_escalation_cooldown_seconds=600,
    )
    trader = trader_factory(
        settings=settings,
        decision=Decision("BTC/USD", "hold", "not_reached"),
    )

    first_count, _ = trader._ioc_cancel_state(FakeIocRepository(3))
    cooled_count, _ = trader._ioc_cancel_state(FakeIocRepository(0))

    assert first_count == 3
    assert cooled_count == 3
