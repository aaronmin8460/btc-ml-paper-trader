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
    assert settings.discord_alert_cooldown_seconds == 60
    assert settings.discord_risk_alert_cooldown_seconds == 300
    assert settings.circuit_breaker_enabled is True
    assert settings.max_same_risk_blocks_before_pause == 20
    assert settings.max_runtime_errors_before_pause == 10
    assert settings.circuit_breaker_window_seconds == 900


def test_scalping_settings_load_conservative_defaults():
    settings = Settings(_env_file=None)

    assert settings.trading_enabled is False
    assert settings.auto_trade_enabled is False
    assert settings.strategy_mode == "rule_scalping"
    assert settings.order_type == "limit"
    assert settings.time_in_force == "ioc"
    assert settings.limit_price_offset_bps == 2
    assert settings.ioc_cancel_lookback_seconds == 300
    assert settings.max_recent_ioc_cancels == 3
    assert settings.ioc_cancel_cooldown_seconds == 120
    assert settings.ioc_cancel_escalation_cooldown_seconds == 600
    assert settings.scalping_mode_enabled is False
    assert settings.max_spread_bps == 5
    assert settings.max_slippage_bps == 8
    assert settings.min_quote_imbalance == 0.0
    assert settings.max_trades_per_hour == 5
    assert settings.max_trades_per_10_minutes == 10
    assert settings.max_daily_trades == 20
    assert settings.max_order_attempts_per_hour == 30
    assert settings.max_order_attempts_per_10_minutes == 20
    assert settings.max_order_attempts_per_day == 100
    assert settings.max_consecutive_losses == 2
    assert settings.max_loss_usd_per_hour == 5
    assert settings.max_consecutive_ioc_cancels == 5
    assert settings.min_seconds_between_trades == 180
    assert settings.scalping_kill_switch_enabled is True
    assert settings.scalping_pause_after_loss_streak_seconds == 900
    assert settings.scalping_pause_after_runtime_errors_seconds == 900
    assert settings.taker_fee_bps == 25
    assert settings.maker_fee_bps == 15
    assert settings.slippage_bps == 10
    assert settings.backtest_use_taker_fees is True
    assert settings.paper_execution_mode == "alpaca_paper"
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
    assert settings.scalping_take_profit_pct == 0.006
    assert settings.scalping_stop_loss_pct == 0.003
    assert settings.scalping_trailing_stop_pct == 0.002
    assert settings.scalping_label_horizon_bars == 3
    assert settings.scalping_label_take_profit_pct == 0.0012
    assert settings.scalping_label_stop_loss_pct == 0.0008
    assert settings.scalping_label_min_net_profit_pct == 0.0002
    assert settings.label_fee_bps_per_side == 15
    assert settings.label_slippage_bps_per_side == 0
    assert settings.label_spread_bps == 0
    assert settings.label_min_net_profit_pct == 0.0
    assert settings.label_horizon_bars == 6
    assert settings.scalping_min_momentum_pct == -0.0005
    assert settings.scalping_max_position_seconds == 900
    assert settings.scalping_max_data_age_seconds == 120
    assert settings.scalping_min_hold_seconds == 0
    assert settings.scalping_buy_probability_floor == 0.50
    assert settings.scalping_confidence_gap_required == 0.04
    assert settings.scalping_sell_probability_floor == 0.55
    assert settings.scalping_exit_confidence_gap_required == 0.04
    assert settings.scalping_sell_on_weak_quote is True
    assert settings.scalping_quote_imbalance_exit == -0.10
    assert settings.scalping_profit_guard_enabled is False
    assert settings.min_hold_seconds_before_weak_quote_exit == 30
    assert settings.regime_no_trade_volatility_threshold == 0.020
    assert settings.regime_no_trade_short_return_threshold == 0.015
    assert settings.regime_trend_strength_threshold == 0.800
    assert settings.regime_breakout_threshold == 0.001
    assert settings.regime_mean_reversion_short_return_threshold == 0.003
    assert settings.regime_mean_reversion_low_breakout_threshold == 0.006
    assert settings.rule_rsi_min == 40
    assert settings.rule_rsi_max == 60
    assert settings.rule_min_normalized_volume == 1.1
    assert settings.rule_ema_touch_tolerance_pct == 0.0015
    assert settings.rule_vwap_touch_tolerance_pct == 0.0015
    assert settings.rule_max_vwap_distance_pct == 0.01
    assert settings.rule_require_vwap_above is True
    assert settings.rule_min_score_to_buy == 10
    assert settings.rule_trend_5m_required is True
    assert settings.rule_trend_15m_required is True
    assert settings.profit_only_exit_enabled is True
    assert settings.min_net_exit_profit_pct == 0.002
    assert settings.exit_profit_buffer_bps == 5
    assert settings.allow_emergency_stop_loss is True
    assert settings.emergency_stop_loss_pct == 0.006
    assert settings.trailing_stop_arm_profit_pct == 0.002
    assert settings.model_sell_requires_profit is True
    assert settings.weak_quote_sell_requires_profit is True
    assert settings.max_holding_sell_requires_profit is True
    assert settings.allow_fallback_trading is False
    assert settings.auto_train_enabled is False
    assert settings.min_buy_positive_labels == 50
    assert settings.min_buy_positive_label_pct == 0.03
    assert settings.auto_train_interval_seconds == 21600
    assert settings.auto_train_startup_delay_seconds == 300
    assert settings.auto_train_min_bars == 3000
    assert settings.auto_train_send_discord_alerts is True
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
    assert settings.min_backtest_profit_factor == 1.2
    assert settings.min_backtest_trades == 30
    assert settings.max_backtest_ambiguous_candle_ratio == 0.10
    assert settings.model_promotion_require_positive_net_return is True


def test_rule_scalping_strategy_mode_is_accepted():
    settings = Settings(_env_file=None, strategy_mode="rule_scalping")

    assert settings.strategy_mode == "rule_scalping"


@pytest.mark.parametrize("strategy_mode", ["ml", "hybrid", "invalid"])
def test_invalid_strategy_modes_are_rejected(strategy_mode):
    with pytest.raises(ValueError):
        Settings(_env_file=None, strategy_mode=strategy_mode)


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
        profit_only_exit_enabled=True,
        min_net_exit_profit_pct=0.002,
        exit_profit_buffer_bps=5,
        trailing_stop_arm_profit_pct=0.002,
        allow_emergency_stop_loss=True,
        emergency_stop_loss_pct=0.006,
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
    assert settings.profit_only_exit_enabled is True
    assert settings.min_net_exit_profit_pct == 0.002
    assert settings.exit_profit_buffer_bps == 5
    assert settings.trailing_stop_arm_profit_pct == 0.002
    assert settings.allow_emergency_stop_loss is True
    assert settings.emergency_stop_loss_pct == 0.006
    assert settings.discord_alert_on_signal is False
    assert settings.circuit_breaker_enabled is True
    assert settings.max_same_risk_blocks_before_pause == 20
    assert settings.max_runtime_errors_before_pause == 10
    assert settings.circuit_breaker_window_seconds == 900


def test_active_profit_guarded_paper_scalping_profile_values_load():
    settings = Settings(
        _env_file=None,
        scalping_mode_enabled=True,
        scan_interval_seconds=1,
        min_seconds_between_trades=10,
        max_trades_per_hour=60,
        max_daily_trades=300,
        max_order_attempts_per_hour=120,
        max_order_attempts_per_day=500,
        scalping_buy_probability_floor=0.58,
        scalping_confidence_gap_required=0.08,
        min_quote_imbalance=-0.005,
        max_spread_bps=6,
        profit_only_exit_enabled=True,
        min_net_exit_profit_pct=0.002,
        exit_profit_buffer_bps=5,
        scalping_sell_on_weak_quote=False,
        trailing_stop_arm_profit_pct=0.002,
        allow_emergency_stop_loss=True,
        emergency_stop_loss_pct=0.006,
    )

    assert settings.scan_interval_seconds == 1
    assert settings.min_seconds_between_trades == 10
    assert settings.max_trades_per_hour == 60
    assert settings.max_daily_trades == 300
    assert settings.max_order_attempts_per_hour == 120
    assert settings.max_order_attempts_per_day == 500
    assert settings.scalping_buy_probability_floor == 0.58
    assert settings.scalping_confidence_gap_required == 0.08
    assert settings.min_quote_imbalance == -0.005
    assert settings.max_spread_bps == 6
    assert settings.profit_only_exit_enabled is True
    assert settings.min_net_exit_profit_pct == 0.002
    assert settings.exit_profit_buffer_bps == 5
    assert settings.scalping_sell_on_weak_quote is False
    assert settings.trailing_stop_arm_profit_pct == 0.002
    assert settings.allow_emergency_stop_loss is True
    assert settings.emergency_stop_loss_pct == 0.006


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
        Settings(_env_file=None, paper_execution_mode="live")

    with pytest.raises(ValueError):
        Settings(_env_file=None, alpaca_paper_base_url="https://api.alpaca.markets")

    with pytest.raises(ValueError):
        Settings(_env_file=None, alpaca_paper_base_url="https://paper-api.alpaca.markets.example.com")

    with pytest.raises(ValueError):
        Settings(_env_file=None, alpaca_max_calls_per_minute=0)

    with pytest.raises(ValueError):
        Settings(_env_file=None, circuit_breaker_window_seconds=-1)

    with pytest.raises(ValueError):
        Settings(_env_file=None, max_loss_usd_per_hour=-1)

    with pytest.raises(ValueError):
        Settings(_env_file=None, scalping_label_horizon_bars=0)

    with pytest.raises(ValueError):
        Settings(_env_file=None, scalping_label_horizon_bars=4)

    with pytest.raises(ValueError):
        Settings(_env_file=None, label_horizon_bars=0)

    with pytest.raises(ValueError):
        Settings(_env_file=None, label_fee_bps_per_side=-1)

    with pytest.raises(ValueError):
        Settings(_env_file=None, label_min_net_profit_pct=1.01)

    with pytest.raises(ValueError):
        Settings(_env_file=None, max_backtest_ambiguous_candle_ratio=1.01)

    with pytest.raises(ValueError):
        Settings(_env_file=None, min_buy_positive_label_pct=1.01)


def test_safe_dict_masks_discord_webhook_url():
    settings = Settings(_env_file=None, discord_webhook_url="https://discord.example/api/webhooks/example/secret")

    safe_config = settings.safe_dict()

    assert safe_config["discord_webhook_url"] == "***"
    assert "example/secret" not in str(safe_config)


def test_safe_config_endpoint_masks_discord_webhook_url(monkeypatch, tmp_path):
    from app import main

    settings = Settings(
        _env_file=None,
        api_admin_token="secret",
        discord_webhook_url="https://discord.example/api/webhooks/example/secret",
        model_dir=str(tmp_path / "models"),
    )
    monkeypatch.setattr(main, "settings", settings)

    response = TestClient(main.app).get("/config/safe", headers={"X-Admin-Token": "secret"})

    assert response.status_code == 200
    assert response.json()["discord_webhook_url"] == "***"
    assert response.json()["active_model_status"] == "stale"
    assert response.json()["active_model_valid"] is False
    assert response.json()["active_model_invalid_reason"] == "no_active_model"
    assert response.json()["registry_metadata_matches_joblib"] is False
    assert "example/secret" not in response.text
