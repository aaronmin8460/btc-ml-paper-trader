from fastapi.testclient import TestClient

from app.config import Settings


class DisabledNotifier:
    instances = []

    def __init__(self, settings) -> None:
        self.settings = settings
        self.enabled = False
        self.sent_messages = []
        self.instances.append(self)

    async def send(self, content: str) -> None:
        self.sent_messages.append(content)


class EnabledNotifier:
    instances = []

    def __init__(self, settings) -> None:
        self.settings = settings
        self.enabled = True
        self.sent_messages = []
        self.instances.append(self)

    async def send(self, content: str) -> None:
        self.sent_messages.append(content)


def test_discord_test_endpoint_requires_admin_token(monkeypatch):
    from app import main

    DisabledNotifier.instances = []
    monkeypatch.setattr(main, "settings", Settings(_env_file=None, api_admin_token="secret"))
    monkeypatch.setattr(main, "DiscordNotifier", DisabledNotifier)

    response = TestClient(main.app).post("/alerts/discord/test")

    assert response.status_code == 401
    assert DisabledNotifier.instances == []


def test_discord_test_endpoint_returns_disabled(monkeypatch):
    from app import main

    DisabledNotifier.instances = []
    monkeypatch.setattr(
        main,
        "settings",
        Settings(
            _env_file=None,
            api_admin_token="secret",
            discord_alerts_enabled=False,
            discord_webhook_url="https://discord.example/webhook-secret",
        ),
    )
    monkeypatch.setattr(main, "DiscordNotifier", DisabledNotifier)

    response = TestClient(main.app).post("/alerts/discord/test", headers={"X-Admin-Token": "secret"})

    assert response.status_code == 200
    assert response.json() == {"sent": False, "reason": "discord_disabled"}
    assert DisabledNotifier.instances[0].sent_messages == []
    assert "webhook-secret" not in response.text


def test_discord_test_endpoint_calls_notifier_when_enabled(monkeypatch):
    from app import main

    EnabledNotifier.instances = []
    monkeypatch.setattr(
        main,
        "settings",
        Settings(
            _env_file=None,
            app_env="test",
            api_admin_token="secret",
            discord_alerts_enabled=True,
            discord_webhook_url="https://discord.example/webhook-secret",
        ),
    )
    monkeypatch.setattr(main, "DiscordNotifier", EnabledNotifier)

    response = TestClient(main.app).post("/alerts/discord/test", headers={"X-Admin-Token": "secret"})

    assert response.status_code == 200
    assert response.json() == {"sent": True}
    sent_messages = EnabledNotifier.instances[0].sent_messages
    assert len(sent_messages) == 1
    message = sent_messages[0]
    assert "App: btc-ml-paper-trader" in message
    assert "Environment: test" in message
    assert "Symbol: BTC/USD" in message
    assert "Paper trading only: True" in message
    assert "Timestamp:" in message
    assert "webhook-secret" not in message
    assert "webhook-secret" not in response.text
