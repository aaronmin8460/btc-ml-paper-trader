from fastapi.testclient import TestClient

from app.config import Settings


DISCORD_ENV_VARS = [
    "DISCORD_WEBHOOK_URL",
    "DISCORD_ALERTS_ENABLED",
    "DISCORD_ALERT_ON_HOLD",
    "DISCORD_ALERT_ON_SIGNAL",
    "DISCORD_ALERT_ON_ORDER",
    "DISCORD_ALERT_ON_ERROR",
    "DISCORD_ALERT_ON_MODEL",
]


def test_discord_settings_are_disabled_by_default(monkeypatch):
    for env_var in DISCORD_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)

    settings = Settings(_env_file=None)

    assert settings.discord_webhook_url == ""
    assert settings.discord_alerts_enabled is False
    assert settings.discord_alert_on_hold is False
    assert settings.discord_alert_on_signal is True
    assert settings.discord_alert_on_order is True
    assert settings.discord_alert_on_error is True
    assert settings.discord_alert_on_model is False


def test_safe_dict_masks_discord_webhook_url():
    settings = Settings(_env_file=None, discord_webhook_url="https://discord.com/api/webhooks/example/secret")

    safe_config = settings.safe_dict()

    assert safe_config["discord_webhook_url"] == "***"
    assert "example/secret" not in str(safe_config)


def test_safe_config_endpoint_masks_discord_webhook_url(monkeypatch):
    from app import main

    settings = Settings(_env_file=None, discord_webhook_url="https://discord.com/api/webhooks/example/secret")
    monkeypatch.setattr(main, "settings", settings)

    response = TestClient(main.app).get("/config/safe")

    assert response.status_code == 200
    assert response.json()["discord_webhook_url"] == "***"
    assert "example/secret" not in response.text
