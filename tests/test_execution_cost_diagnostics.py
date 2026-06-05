import pandas as pd
import pytest

from app.backtest.scalping import estimated_round_trip_execution_cost_pct, promotion_required_return_pct
from app.config import Settings
from scripts.diagnose_execution_costs import build_execution_cost_report, take_profit_is_unsafe


def _settings(**overrides) -> Settings:
    values = {
        "_env_file": None,
        "scalping_mode_enabled": True,
        "taker_fee_bps": 25,
        "maker_fee_bps": 15,
        "slippage_bps": 10,
        "max_spread_bps": 6,
        "backtest_use_taker_fees": True,
        "min_backtest_net_return_pct": 0.001,
        "scalping_take_profit_pct": 0.003,
        "scalping_stop_loss_pct": 0.002,
        "scalping_label_take_profit_pct": 0.0012,
    }
    values.update(overrides)
    return Settings(**values)


def test_cost_diagnostic_computes_round_trip_cost_correctly():
    settings = _settings()

    assert estimated_round_trip_execution_cost_pct(settings) == pytest.approx(0.0076)
    assert promotion_required_return_pct(settings) == pytest.approx(0.0086)


def test_take_profit_lower_than_round_trip_cost_is_flagged_unsafe():
    bars = pd.DataFrame(
        {
            "close": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            "high": [100.0, 100.2, 100.3, 100.4, 100.5, 100.6, 100.7],
        }
    )

    report = build_execution_cost_report(bars, _settings())

    assert take_profit_is_unsafe(0.003, report["round_trip_estimated_cost_pct"]) is True
    assert report["scalping_take_profit_is_unsafe"] is True
    assert report["training_label_take_profit_is_unsafe"] is True
    assert report["minimum_take_profit_net_positive_pct"] == pytest.approx(0.0076)
