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
    "DISCORD_RISK_ALERT_COOLDOWN_SECONDS",
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
    assert settings.discord_risk_alert_cooldown_seconds == 300
    assert settings.circuit_breaker_enabled is True
    assert settings.max_same_risk_blocks_before_pause == 20
    assert settings.max_runtime_errors_before_pause == 10
    assert settings.circuit_breaker_window_seconds == 900


def test_scalping_settings_load_conservative_defaults():
    settings = Settings(_env_file=None)

    assert settings.trading_enabled is False
    assert settings.auto_trade_enabled is False
    assert settings.order_type == "limit"
    assert settings.time_in_force == "ioc"
    assert settings.limit_price_offset_bps == 2
    assert settings.ioc_cancel_lookback_seconds == 300
    assert settings.max_recent_ioc_cancels == 3
    assert settings.ioc_cancel_cooldown_seconds == 120
    assert settings.ioc_cancel_escalation_cooldown_seconds == 600
    assert settings.scalping_mode_enabled is False
    assert settings.max_spread_bps == 10
    assert settings.max_slippage_bps == 8
    assert settings.min_quote_imbalance == -0.05
    assert settings.max_trades_per_hour == 1000
    assert settings.max_daily_trades == 10000
    assert settings.max_order_attempts_per_hour == 30
    assert settings.max_order_attempts_per_day == 100
    assert settings.max_consecutive_losses == 3
    assert settings.min_seconds_between_trades == 0
    assert settings.taker_fee_bps == 25
    assert settings.maker_fee_bps == 15
    assert settings.slippage_bps == 10
    assert settings.backtest_use_taker_fees is True
    assert settings.paper_fee_bps == 0
    assert settings.paper_slippage_bps == 0
    assert settings.alpaca_rate_limit_enabled is True
    assert settings.alpaca_max_calls_per_minute == 180
    assert settings.alpaca_api_budget_target_per_minute == 170
    assert settings.alpaca_api_budget_hard_stop_per_minute == 195
    assert settings.market_bars_cache_seconds == 30
    assert settings.position_cache_seconds == 2
    assert settings.account_equity_cache_seconds == 5
    assert settings.quote_cache_seconds == 0
    assert settings.scalping_entry_dip_pct == 0.0005
    assert settings.scalping_take_profit_pct == 0.0015
    assert settings.scalping_stop_loss_pct == 0.001
    assert settings.scalping_trailing_stop_pct == 0.0008
    assert settings.scalping_min_momentum_pct == -0.0005
    assert settings.scalping_max_position_seconds == 90
    assert settings.scalping_buy_probability_floor == 0.50
    assert settings.scalping_confidence_gap_required == 0.04
    assert settings.scalping_sell_on_weak_quote is True
    assert settings.scalping_quote_imbalance_exit == -0.10
    assert settings.min_hold_seconds_before_weak_quote_exit == 30
    assert settings.allow_fallback_trading is False
    assert settings.order_in_flight_timeout_seconds == 15
    assert settings.order_status_check_enabled is True
    assert settings.order_status_check_delay_seconds == 0.5
    assert settings.pause_trading_on_account_drawdown is True
    assert settings.max_account_daily_loss_usd == 25
    assert settings.max_account_daily_loss_pct == 0.01
    assert settings.max_account_drawdown_pct == 0.03
    assert settings.require_account_data_for_trading is False
    assert settings.min_backtest_net_return_pct == 0.001
    assert settings.max_backtest_drawdown_pct == 0.01
    assert settings.min_backtest_profit_factor == 1.05
    assert settings.min_backtest_trades == 20
    assert settings.model_promotion_require_positive_net_return is True


def test_conservative_paper_scalping_profile_values_load():
    settings = Settings(
        _env_file=None,
        trading_enabled=True,
        auto_trade_enabled=True,
        scalping_mode_enabled=True,
        scan_interval_seconds=5,
        order_type="limit",
        time_in_force="ioc",
        order_notional_usd=25,
        max_spread_bps=6,
        max_slippage_bps=8,
        min_quote_imbalance=0.00,
        scalping_buy_probability_floor=0.57,
        scalping_confidence_gap_required=0.06,
        min_seconds_between_trades=15,
        max_trades_per_hour=30,
        max_daily_trades=150,
        max_order_attempts_per_hour=30,
        max_order_attempts_per_day=100,
        ioc_cancel_lookback_seconds=300,
        max_recent_ioc_cancels=3,
        ioc_cancel_cooldown_seconds=120,
        ioc_cancel_escalation_cooldown_seconds=600,
        min_hold_seconds_before_weak_quote_exit=30,
        discord_alert_on_signal=False,
        discord_alert_on_order=True,
        discord_alert_on_error=True,
        discord_risk_alert_cooldown_seconds=300,
        circuit_breaker_enabled=True,
        max_same_risk_blocks_before_pause=20,
        max_runtime_errors_before_pause=10,
        circuit_breaker_window_seconds=900,
    )

    assert settings.trading_enabled is True
    assert settings.auto_trade_enabled is True
    assert settings.scalping_mode_enabled is True
    assert settings.scan_interval_seconds == 5
    assert settings.max_spread_bps == 6
    assert settings.scalping_buy_probability_floor == 0.57
    assert settings.scalping_confidence_gap_required == 0.06
    assert settings.min_seconds_between_trades == 15
    assert settings.max_trades_per_hour == 30
    assert settings.max_daily_trades == 150
    assert settings.max_order_attempts_per_hour == 30
    assert settings.max_order_attempts_per_day == 100
    assert settings.ioc_cancel_cooldown_seconds == 120
    assert settings.ioc_cancel_escalation_cooldown_seconds == 600
    assert settings.min_hold_seconds_before_weak_quote_exit == 30
    assert settings.discord_alert_on_signal is False
    assert settings.circuit_breaker_enabled is True
    assert settings.max_same_risk_blocks_before_pause == 20
    assert settings.max_runtime_errors_before_pause == 10
    assert settings.circuit_breaker_window_seconds == 900


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

    with pytest.raises(ValueError):
        Settings(_env_file=None, paper_fee_bps=-1)

    with pytest.raises(ValueError):
        Settings(_env_file=None, paper_slippage_bps=-1)

    with pytest.raises(ValueError):
        Settings(_env_file=None, alpaca_max_calls_per_minute=0)

    with pytest.raises(ValueError):
        Settings(_env_file=None, circuit_breaker_window_seconds=-1)


def test_safe_dict_masks_discord_webhook_url():
    settings = Settings(_env_file=None, discord_webhook_url="https://discord.example/api/webhooks/example/secret")

    safe_config = settings.safe_dict()

    assert safe_config["discord_webhook_url"] == "***"
    assert "example/secret" not in str(safe_config)


def test_safe_config_endpoint_masks_discord_webhook_url(monkeypatch):
    from app import main

    settings = Settings(
        _env_file=None,
        api_admin_token="secret",
        discord_webhook_url="https://discord.example/api/webhooks/example/secret",
    )
    monkeypatch.setattr(main, "settings", settings)

    response = TestClient(main.app).get("/config/safe", headers={"X-Admin-Token": "secret"})

    assert response.status_code == 200
    assert response.json()["discord_webhook_url"] == "***"
    assert "example/secret" not in response.text
