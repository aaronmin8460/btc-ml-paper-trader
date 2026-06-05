from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from app.config import Settings
from app.risk.risk_manager import PositionState, TradeFrequencyState
from app.strategy.scalping_decision_engine import ScalpingDecisionEngine
from scripts.apply_candidate_config import write_candidate_config
from scripts.sweep_strategy_params import (
    build_sweep_summary,
    reject_config,
    settings_with_overrides,
    write_sweep_outputs,
)


def _settings(**overrides) -> Settings:
    values = {
        "_env_file": None,
        "scalping_mode_enabled": True,
        "trading_enabled": False,
        "auto_trade_enabled": False,
        "allow_fallback_trading": False,
        "min_seconds_between_trades": 30,
        "max_trades_per_hour": 20,
    }
    values.update(overrides)
    return Settings(**values)


def _good_metrics(**overrides):
    metrics = {
        "number_of_trades": 25,
        "max_drawdown_pct": 0.01,
        "profit_factor_net": 1.20,
        "net_return_pct": 0.02,
        "expectancy": 0.0008,
        "ambiguous_candle_ratio": 0.01,
    }
    metrics.update(overrides)
    return metrics


def test_parameter_sweep_writes_csv_and_json_without_modifying_env(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("TRADING_ENABLED=false\n", encoding="utf-8")
    rows = [
        {
            "strategy_name": "mean_reversion_scalping",
            "parameter_set_id": "ps_000_baseline",
            "number_of_signals": 10,
            "number_of_trades": 0,
            "net_return_pct": 0.0,
            "gross_return_pct": 0.0,
            "profit_factor_net": 0.0,
            "max_drawdown_pct": 0.0,
            "win_rate_net": 0.0,
            "expectancy": 0.0,
            "average_trade_net_return": 0.0,
            "average_hold_bars": 0.0,
            "canceled_orders": 0,
            "ambiguous_candle_ratio": 0.0,
            "blocked_signal_count": 10,
            "top_block_reason": "ml_buy_probability_below_threshold",
            "train_period": "a..b",
            "validation_period": "c..d",
            "accepted": False,
            "rejection_reasons": "number_of_trades_below_min",
            "rank_score": -1_000_000.0,
        }
    ]
    configs = {
        "ps_000_baseline": {
            "parameter_set_id": "ps_000_baseline",
            "env_overrides": {
                "PAPER_TRADING_ONLY": True,
                "SYMBOL": "BTC/USD",
                "TRADING_ENABLED": False,
                "AUTO_TRADE_ENABLED": False,
                "ALLOW_FALLBACK_TRADING": False,
            },
        }
    }
    csv_path = tmp_path / "logs" / "strategy_param_sweep.csv"
    summary_path = tmp_path / "logs" / "strategy_param_sweep_summary.json"

    summary = build_sweep_summary(rows, configs, _settings(), csv_path=csv_path, summary_path=summary_path)
    write_sweep_outputs(rows, summary, csv_path=csv_path, summary_path=summary_path)

    assert csv_path.exists()
    assert summary_path.exists()
    assert "strategy_name,parameter_set_id" in csv_path.read_text(encoding="utf-8")
    assert "best_candidate_configs" in summary_path.read_text(encoding="utf-8")
    assert env_path.read_text(encoding="utf-8") == "TRADING_ENABLED=false\n"


def test_apply_candidate_config_writes_only_env_candidate(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("TRADING_ENABLED=false\n", encoding="utf-8")
    summary_path = tmp_path / "strategy_param_sweep_summary.json"
    summary_path.write_text(
        """
        {
          "configurations_by_id": {
            "ps_001_ml_confirmation": {
              "env_overrides": {
                "PAPER_TRADING_ONLY": true,
                "SYMBOL": "BTC/USD",
                "TRADING_ENABLED": false,
                "AUTO_TRADE_ENABLED": false,
                "ALLOW_FALLBACK_TRADING": false,
                "SCALPING_MODE_ENABLED": true,
                "SCALPING_BUY_PROBABILITY_FLOOR": 0.58
              }
            }
          }
        }
        """,
        encoding="utf-8",
    )
    output_path = tmp_path / ".env.candidate"

    result = write_candidate_config(
        parameter_set_id="ps_001_ml_confirmation",
        summary_path=summary_path,
        output_path=output_path,
        current_env_path=env_path,
    )

    assert output_path.exists()
    assert env_path.read_text(encoding="utf-8") == "TRADING_ENABLED=false\n"
    text = output_path.read_text(encoding="utf-8")
    assert "PAPER_TRADING_ONLY=true" in text
    assert "SYMBOL=BTC/USD" in text
    assert "TRADING_ENABLED=false" in text
    assert "ALLOW_FALLBACK_TRADING=false" in text
    assert "SCALPING_BUY_PROBABILITY_FLOOR=0.58" in text
    assert "- TRADING_ENABLED=false" in result["diff_summary"]
    assert "+ TRADING_ENABLED=false" in result["diff_summary"]


def test_apply_candidate_config_refuses_to_write_env(tmp_path):
    summary_path = tmp_path / "strategy_param_sweep_summary.json"
    summary_path.write_text('{"configurations_by_id": {"ps_000": {"env_overrides": {}}}}', encoding="utf-8")

    with pytest.raises(ValueError):
        write_candidate_config(
            parameter_set_id="ps_000",
            summary_path=summary_path,
            output_path=tmp_path / ".env",
            current_env_path=tmp_path / ".env",
        )


def test_overfit_config_is_rejected_for_single_trade_concentration():
    reasons = reject_config(
        _good_metrics(),
        trade_details=[
            {"net_return_pct": 0.018},
            {"net_return_pct": 0.002},
        ],
    )

    assert "single_trade_return_concentration_too_high" in reasons


def test_config_with_too_few_trades_is_rejected():
    reasons = reject_config(_good_metrics(number_of_trades=19))

    assert "number_of_trades_below_min" in reasons


def test_config_with_high_drawdown_is_rejected():
    reasons = reject_config(_good_metrics(max_drawdown_pct=0.031))

    assert "max_drawdown_too_high" in reasons


def test_config_with_fallback_ml_predictions_is_rejected():
    reasons = reject_config(_good_metrics(), fallback_prediction_used=True)

    assert "fallback_prediction_not_allowed" in reasons


def test_sweep_settings_keep_btc_only_paper_only_and_no_fallback():
    settings = settings_with_overrides(
        _settings(),
        {
            "allow_fallback_trading": True,
            "trading_enabled": True,
            "auto_trade_enabled": True,
            "max_spread_bps": 8,
        },
    )

    assert settings.symbol == "BTC/USD"
    assert settings.paper_trading_only is True
    assert settings.trading_enabled is False
    assert settings.auto_trade_enabled is False
    assert settings.allow_fallback_trading is False
    assert settings.max_spread_bps == 8


def test_risk_controls_remain_final_authority_for_sweep_settings():
    settings = _settings(
        trading_enabled=True,
        scalping_buy_probability_floor=0.54,
        scalping_confidence_gap_required=0.04,
    )
    decision = ScalpingDecisionEngine(settings).decide(
        prediction={
            "symbol": "BTC/USD",
            "buy_probability": 0.8,
            "sell_probability": 0.1,
            "model_available": True,
            "prediction_source": "model",
            "active_model_valid": True,
        },
        feature_row=pd.Series(
                {
                    "timestamp": datetime.now(UTC),
                    "close": 100.0,
                "scalping_spread_bps": 2.0,
                "scalping_quote_imbalance": 0.1,
                "scalping_log_return_3": -0.001,
                "scalping_momentum_3": -0.001,
                "scalping_high_breakout_5": -0.001,
                "scalping_low_breakout_5": -0.001,
                "scalping_ema_5_distance": -0.001,
                "scalping_vwap_distance": -0.001,
                "scalping_rsi_3": 38.0,
                "scalping_volatility_10": 0.001,
            }
        ),
        position=PositionState(),
        trading_enabled=True,
        trade_frequency=TradeFrequencyState(trades_last_hour=settings.max_trades_per_hour),
    )

    assert decision.action == "hold"
    assert decision.blocked_by == "risk_manager"
