import pandas as pd
import pytest

from app.config import Settings
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
        return {"symbol": "BTC/USD", "buy_probability": 0.75, "sell_probability": 0.15}


class FakeDecisionEngine:
    def __init__(self, decision: Decision) -> None:
        self.decision = decision

    def decide(self, **kwargs):
        return self.decision


class FakeBroker:
    def __init__(self, events: list[str], order_response: dict | None = None) -> None:
        self.events = events
        self.order_response = order_response or {"id": "paper-order-1", "status": "submitted"}

    async def get_position(self, symbol):
        return None

    async def submit_market_order(self, **kwargs):
        self.events.append("submit_order")
        return self.order_response

    async def submit_order(self, **kwargs):
        return await self.submit_market_order(**kwargs)


class FakeNotifier:
    def __init__(self, events: list[str] | None = None, fail: bool = False) -> None:
        self.events = events if events is not None else []
        self.calls = []
        self.fail = fail

    async def signal_alert(self, *args):
        if self.fail:
            raise RuntimeError("discord failed")
        self.events.append("signal_alert")
        self.calls.append(("signal", args))

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
    async def signal_alert(self, *args):
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
    trade_frequency = None

    def __init__(self, db) -> None:
        pass

    @classmethod
    def reset(cls) -> None:
        cls.signals = []
        cls.orders = []
        cls.trade_frequency = None

    def add_signal(self, *args) -> None:
        self.signals.append(args)

    def add_order(self, **kwargs) -> None:
        self.orders.append(kwargs)

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
    ) -> Trader:
        FakeRepository.reset()
        FakeRepository.trade_frequency = trade_frequency
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
        trader.broker = FakeBroker(flow_events)
        trader.logger = FakeLogger()
        trader.notifier = notifier if notifier is not None else FakeNotifier(flow_events)
        return trader

    return build_trader


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
    assert events == ["submit_order", "order_alert"]
    assert notifier.calls == [
        ("order", ("buy", "submitted", 25, None), {"broker_order_id": "paper-order-1"})
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
async def test_kill_switch_block_sends_risk_alert(trader_factory):
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
        decision=Decision("BTC/USD", "hold", "max_trades_per_hour_reached"),
        notifier=notifier,
    )

    await trader.run_once()

    assert notifier.calls == [("risk", ("max_trades_per_hour_reached",))]


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
        "discord_alert_failed",
    ]
    assert "discord.example" not in str(trader.logger.events)
