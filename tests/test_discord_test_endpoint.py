from fastapi.testclient import TestClient

from app.config import Settings


class DisabledNotifier:
    instances = []

    def __init__(self, settings) -> None:
        self.settings = settings
        self.enabled = False
        self.sent_messages = []
        self.sent_embeds = []
        self.instances.append(self)

    async def send(self, content: str) -> None:
        self.sent_messages.append(content)

    async def send_embed(self, *args, **kwargs) -> None:
        self.sent_embeds.append((args, kwargs))


class EnabledNotifier:
    instances = []

    def __init__(self, settings) -> None:
        self.settings = settings
        self.enabled = True
        self.sent_messages = []
        self.sent_embeds = []
        self.instances.append(self)

    async def send(self, content: str) -> None:
        self.sent_messages.append(content)

    async def send_embed(self, *args, **kwargs) -> None:
        self.sent_embeds.append((args, kwargs))


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
    assert DisabledNotifier.instances[0].sent_embeds == []
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
    assert EnabledNotifier.instances[0].sent_messages == []
    sent_embeds = EnabledNotifier.instances[0].sent_embeds
    assert len(sent_embeds) == 1
    args, kwargs = sent_embeds[0]
    assert args == ()
    assert kwargs["title"] == "Discord Test Alert"
    fields = {item["name"]: item["value"] for item in kwargs["fields"]}
    assert fields["App"] == "btc-ml-paper-trader"
    assert fields["Environment"] == "test"
    assert fields["Symbol"] == "BTC/USD"
    assert fields["Paper Trading Only"] == "True"
    assert "Timestamp" in fields
    assert "webhook-secret" not in str(kwargs)
    assert "webhook-secret" not in response.text
