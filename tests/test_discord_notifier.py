import pytest

from app.config import Settings
from app.notifications.discord import DiscordNotifier


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


@pytest.mark.anyio
async def test_no_http_request_when_disabled(monkeypatch, fake_logger):
    RecordingAsyncClient.requests = []
    monkeypatch.setattr("app.notifications.discord.httpx.AsyncClient", RecordingAsyncClient)
    settings = Settings(
        _env_file=None,
        discord_alerts_enabled=False,
        discord_webhook_url="https://discord.example/webhook",
    )
    notifier = DiscordNotifier(settings)

    await notifier.send("hello")

    assert notifier.enabled is False
    assert RecordingAsyncClient.requests == []


@pytest.mark.anyio
async def test_http_request_when_enabled(monkeypatch, fake_logger):
    RecordingAsyncClient.requests = []
    monkeypatch.setattr("app.notifications.discord.httpx.AsyncClient", RecordingAsyncClient)
    settings = Settings(
        _env_file=None,
        discord_alerts_enabled=True,
        discord_webhook_url="https://discord.example/webhook",
    )
    notifier = DiscordNotifier(settings)

    await notifier.send("x" * 2005)

    assert len(RecordingAsyncClient.requests) == 1
    request = RecordingAsyncClient.requests[0]
    assert request["url"] == "https://discord.example/webhook"
    assert request["json"] == {
        "username": "BTC Paper Trader",
        "content": "x" * 2000,
        "allowed_mentions": {"parse": []},
    }


@pytest.mark.anyio
async def test_hold_signal_is_skipped_when_hold_alerts_disabled(monkeypatch, fake_logger):
    RecordingAsyncClient.requests = []
    monkeypatch.setattr("app.notifications.discord.httpx.AsyncClient", RecordingAsyncClient)
    settings = Settings(
        _env_file=None,
        discord_alerts_enabled=True,
        discord_webhook_url="https://discord.example/webhook",
        discord_alert_on_hold=False,
    )
    notifier = DiscordNotifier(settings)

    await notifier.signal_alert("BTC/USD", "hold", "no edge", 0.45, 0.43)

    assert RecordingAsyncClient.requests == []


@pytest.mark.anyio
async def test_exceptions_from_discord_do_not_propagate(monkeypatch, fake_logger):
    monkeypatch.setattr("app.notifications.discord.httpx.AsyncClient", FailingAsyncClient)
    settings = Settings(
        _env_file=None,
        discord_alerts_enabled=True,
        discord_webhook_url="https://discord.example/webhook",
    )
    notifier = DiscordNotifier(settings)

    await notifier.send("hello")

    assert fake_logger.events == [("discord_alert_failed", {"error_type": "RuntimeError"})]
    assert "discord.example" not in str(fake_logger.events)
