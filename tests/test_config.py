import pytest
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


def test_scalping_settings_load_conservative_defaults():
    settings = Settings(_env_file=None)

    assert settings.order_type == "market"
    assert settings.time_in_force == "gtc"
    assert settings.limit_price_offset_bps == 2
    assert settings.scalping_mode_enabled is False
    assert settings.max_spread_bps == 8
    assert settings.max_slippage_bps == 10
    assert settings.min_quote_imbalance == -0.25
    assert settings.max_trades_per_hour == 10
    assert settings.max_daily_trades == 30
    assert settings.max_consecutive_losses == 3
    assert settings.min_seconds_between_trades == 30
    assert settings.taker_fee_bps == 25
    assert settings.maker_fee_bps == 15
    assert settings.slippage_bps == 10
    assert settings.backtest_use_taker_fees is True


def test_config_still_rejects_unsafe_symbol_and_non_paper_mode():
    with pytest.raises(ValueError):
        Settings(_env_file=None, symbol="ETH/USD")

    with pytest.raises(ValueError):
        Settings(_env_file=None, paper_trading_only=False)

    with pytest.raises(ValueError):
        Settings(_env_file=None, order_type="stop")

    with pytest.raises(ValueError):
        Settings(_env_file=None, time_in_force="day")

    with pytest.raises(ValueError):
        Settings(_env_file=None, taker_fee_bps=-1)


def test_safe_dict_masks_discord_webhook_url():
    settings = Settings(_env_file=None, discord_webhook_url="https://discord.com/api/webhooks/example/secret")

    safe_config = settings.safe_dict()

    assert safe_config["discord_webhook_url"] == "***"
    assert "example/secret" not in str(safe_config)


def test_safe_config_endpoint_masks_discord_webhook_url(monkeypatch):
    from app import main

    settings = Settings(
        _env_file=None,
        api_admin_token="secret",
        discord_webhook_url="https://discord.com/api/webhooks/example/secret",
    )
    monkeypatch.setattr(main, "settings", settings)

    response = TestClient(main.app).get("/config/safe", headers={"X-Admin-Token": "secret"})

    assert response.status_code == 200
    assert response.json()["discord_webhook_url"] == "***"
    assert "example/secret" not in response.text
