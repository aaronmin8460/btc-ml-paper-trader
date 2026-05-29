import pytest

from app.config import Settings
from app.notifications.discord import (
    DISCORD_GREEN,
    DISCORD_ORANGE,
    DISCORD_RED,
    DiscordNotifier,
    prepare_fields,
)


class FakeLogger:
    def __init__(self) -> None:
        self.events = []

    def event(self, event_type: str, **payload) -> None:
        self.events.append((event_type, payload))


class FakeResponse:
    status_code = 204

    def raise_for_status(self) -> None:
        return None


class RecordingAsyncClient:
    requests = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None

    async def post(self, url, json):
        self.requests.append({"url": url, "json": json})
        return FakeResponse()


class FailingAsyncClient:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None

    async def post(self, url, json):
        raise RuntimeError("discord unavailable")


@pytest.fixture
def fake_logger(monkeypatch):
    logger = FakeLogger()
    monkeypatch.setattr("app.notifications.discord.get_logger", lambda: logger)
    return logger


@pytest.fixture
def recording_http(monkeypatch):
    RecordingAsyncClient.requests = []
    monkeypatch.setattr("app.notifications.discord.httpx.AsyncClient", RecordingAsyncClient)
    return RecordingAsyncClient


def enabled_settings(**overrides):
    defaults = {
        "_env_file": None,
        "discord_alerts_enabled": True,
        "discord_webhook_url": "https://discord.example/webhook",
    }
    defaults.update(overrides)
    return Settings(**defaults)


@pytest.mark.anyio
async def test_no_http_request_when_disabled(recording_http, fake_logger):
    settings = Settings(
        _env_file=None,
        discord_alerts_enabled=False,
        discord_webhook_url="https://discord.example/webhook",
    )
    notifier = DiscordNotifier(settings)

    await notifier.send_embed("Test")

    assert notifier.enabled is False
    assert recording_http.requests == []


@pytest.mark.anyio
async def test_old_send_content_still_works(recording_http, fake_logger):
    notifier = DiscordNotifier(enabled_settings())

    await notifier.send("x" * 2005)

    assert len(recording_http.requests) == 1
    payload = recording_http.requests[0]["json"]
    assert payload == {
        "username": "BTC Paper Trader",
        "content": "x" * 2000,
        "allowed_mentions": {"parse": []},
    }


@pytest.mark.anyio
async def test_send_embed_payload_structure(recording_http, fake_logger):
    notifier = DiscordNotifier(enabled_settings())

    await notifier.send_embed(
        "Test Embed",
        description="hello",
        fields=[{"name": "Field", "value": "Value", "inline": False}],
        color=123,
        footer="footer text",
    )

    payload = recording_http.requests[0]["json"]
    assert payload["username"] == "BTC Paper Trader"
    assert payload["allowed_mentions"] == {"parse": []}
    assert "content" not in payload
    assert len(payload["embeds"]) == 1
    embed = payload["embeds"][0]
    assert embed["title"] == "Test Embed"
    assert embed["description"] == "hello"
    assert embed["fields"] == [{"name": "Field", "value": "Value", "inline": False}]
    assert embed["color"] == 123
    assert embed["footer"] == {"text": "footer text"}
    assert "timestamp" in embed


@pytest.mark.anyio
async def test_hold_signal_is_skipped_when_hold_alerts_disabled(recording_http, fake_logger):
    notifier = DiscordNotifier(enabled_settings(discord_alert_on_hold=False))

    await notifier.signal_alert("BTC/USD", "hold", "no edge", 0.45, 0.43)

    assert recording_http.requests == []


@pytest.mark.anyio
async def test_signal_alert_uses_embed(recording_http, fake_logger):
    notifier = DiscordNotifier(enabled_settings())

    await notifier.signal_alert(
        "BTC/USD",
        "buy",
        "scalping_entry_approved",
        0.53,
        0.47,
        spread_bps=5.12,
        quote_imbalance=0.0123,
        latest_price=75000.12,
        mid_price=75001.25,
    )

    embed = recording_http.requests[0]["json"]["embeds"][0]
    fields = {item["name"]: item["value"] for item in embed["fields"]}
    assert embed["title"] == "BTC/USD Signal: BUY"
    assert embed["color"] == DISCORD_GREEN
    assert fields["Action"] == "BUY"
    assert fields["Buy Probability"] == "0.5300 (53.00%)"
    assert fields["Latest Price"] == "$75,000.12"
    assert fields["Spread bps"] == "5.12"


@pytest.mark.anyio
async def test_order_alert_uses_embed(recording_http, fake_logger):
    notifier = DiscordNotifier(enabled_settings())

    await notifier.order_alert(
        "sell",
        "filled",
        notional=None,
        qty=0.01,
        broker_order_id="order-123",
        order_type="limit",
        time_in_force="ioc",
    )

    embed = recording_http.requests[0]["json"]["embeds"][0]
    fields = {item["name"]: item["value"] for item in embed["fields"]}
    assert embed["title"] == "BTC/USD Paper Order"
    assert embed["color"] == DISCORD_RED
    assert fields["Side"] == "SELL"
    assert fields["Status"] == "filled"
    assert fields["Quantity"] == "0.01000000"
    assert fields["Order Type"] == "limit"
    assert fields["Broker Order ID"] == "order-123"


@pytest.mark.anyio
async def test_error_alert_uses_embed(recording_http, fake_logger):
    notifier = DiscordNotifier(enabled_settings())

    await notifier.error_alert("trader.run_once", RuntimeError("boom"))

    embed = recording_http.requests[0]["json"]["embeds"][0]
    fields = {item["name"]: item["value"] for item in embed["fields"]}
    assert embed["title"] == "Trading Bot Error"
    assert embed["color"] == DISCORD_RED
    assert fields["Where"] == "trader.run_once"
    assert fields["Error Type"] == "RuntimeError"
    assert fields["Error Message"] == "boom"


@pytest.mark.anyio
async def test_risk_alert_uses_embed(recording_http, fake_logger):
    notifier = DiscordNotifier(enabled_settings())

    await notifier.risk_alert("trade_cooldown_active")

    embed = recording_http.requests[0]["json"]["embeds"][0]
    fields = {item["name"]: item["value"] for item in embed["fields"]}
    assert embed["title"] == "Risk Guard Triggered"
    assert embed["color"] == DISCORD_ORANGE
    assert fields["Symbol"] == "BTC/USD"
    assert fields["Reason"] == "trade_cooldown_active"


@pytest.mark.anyio
async def test_auto_trading_paused_alert_uses_embed(recording_http, fake_logger):
    notifier = DiscordNotifier(enabled_settings())

    await notifier.auto_trading_paused_alert("repeated_risk_block:trade_cooldown_active")

    embed = recording_http.requests[0]["json"]["embeds"][0]
    fields = {item["name"]: item["value"] for item in embed["fields"]}
    assert embed["title"] == "Auto trading paused"
    assert embed["color"] == DISCORD_RED
    assert fields["Symbol"] == "BTC/USD"
    assert fields["Reason"] == "repeated_risk_block:trade_cooldown_active"


@pytest.mark.anyio
async def test_model_alert_uses_embed(recording_http, fake_logger):
    notifier = DiscordNotifier(enabled_settings(discord_alert_on_model=True))

    await notifier.model_alert("models/btc.joblib", accepted=False, reason="precision_below_threshold", metrics={"precision": 0.1})

    embed = recording_http.requests[0]["json"]["embeds"][0]
    fields = {item["name"]: item["value"] for item in embed["fields"]}
    assert embed["title"] == "Model Rejected"
    assert embed["color"] == DISCORD_RED
    assert fields["Model Path"] == "models/btc.joblib"
    assert fields["Reason"] == "precision_below_threshold"
    assert '"precision": 0.1' in fields["Metrics"]


def test_field_truncation_and_limit():
    fields = [{"name": f"Field {index}", "value": "x" * 2000} for index in range(30)]

    prepared = prepare_fields(fields)

    assert len(prepared) == 25
    assert len(prepared[0]["value"]) == 1024
    assert prepared[0]["value"].endswith("…")


@pytest.mark.anyio
async def test_exceptions_from_discord_do_not_propagate(monkeypatch, fake_logger):
    monkeypatch.setattr("app.notifications.discord.httpx.AsyncClient", FailingAsyncClient)
    notifier = DiscordNotifier(enabled_settings())

    await notifier.send_embed("hello")

    assert fake_logger.events == [("discord_alert_failed", {"error_type": "RuntimeError"})]
    assert "discord.example" not in str(fake_logger.events)
