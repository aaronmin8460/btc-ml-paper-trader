from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from app.ml.buy_the_dip_labels import (
    BUY_THE_DIP_CHRONOLOGICAL,
    BUY_THE_DIP_ENTRY_LABEL,
    BUY_THE_DIP_EXIT_LABEL,
    BUY_THE_DIP_EXIT_REASON,
    BuyTheDipLabelConfig,
    generate_buy_the_dip_labels,
    validate_buy_the_dip_label_config,
)


def _safe_config(**overrides) -> BuyTheDipLabelConfig:
    values = {
        "timeframe": "5Min",
        "take_profit_pct": 0.0125,
        "stop_loss_pct": 0.006,
        "max_hold_bars": 6,
        "round_trip_estimated_cost_pct": 0.0076,
        "promotion_required_return_pct": 0.0086,
        "number_of_research_trades": 25,
        "minimum_research_trades": 20,
        "economically_viable_config": True,
        "synthetic_data_used": False,
        "source_used": "collected_market_data",
        "data_age_minutes": 5.0,
    }
    values.update(overrides)
    return BuyTheDipLabelConfig(**values)


def _bars() -> pd.DataFrame:
    start = datetime(2026, 6, 7, 9, 0, tzinfo=UTC)
    closes = [100.0, 100.2, 101.4, 100.5, 99.8, 99.0, 98.5, 100.0, 101.5, 102.0]
    rows = []
    previous = closes[0]
    for index, close in enumerate(closes):
        open_ = previous
        rows.append(
            {
                "timestamp": start + timedelta(minutes=5 * index),
                "open": open_,
                "high": max(open_, close) * 1.002,
                "low": min(open_, close) * 0.998,
                "close": close,
                "volume": 10.0 + index,
            }
        )
        previous = close
    return pd.DataFrame(rows)


def test_buy_the_dip_label_config_requires_viable_real_fresh_collected_data():
    reasons = validate_buy_the_dip_label_config(
        _safe_config(
            take_profit_pct=0.0076,
            number_of_research_trades=1,
            economically_viable_config=False,
            synthetic_data_used=True,
            source_used="synthetic_explicit_test_demo_mode",
            data_age_minutes=999.0,
        )
    )

    assert "take_profit_not_above_round_trip_cost" in reasons
    assert "number_of_trades_below_minimum" in reasons
    assert "no_economically_viable_research_config" in reasons
    assert "synthetic_data_not_allowed" in reasons
    assert "data_source_not_collected_market_data" in reasons
    assert "stale_data" in reasons


def test_buy_the_dip_labels_are_chronological_and_long_only():
    labeled = generate_buy_the_dip_labels(_bars(), _safe_config())

    assert labeled["timestamp"].is_monotonic_increasing
    assert set(labeled[BUY_THE_DIP_ENTRY_LABEL].dropna().unique()).issubset({0, 1})
    assert set(labeled[BUY_THE_DIP_EXIT_LABEL].dropna().unique()).issubset({0, 1})
    assert labeled[BUY_THE_DIP_CHRONOLOGICAL].all()
    assert "short" not in " ".join(str(value).lower() for value in labeled[BUY_THE_DIP_EXIT_REASON])


def test_buy_the_dip_label_generation_refuses_unvalidated_research_config():
    with pytest.raises(ValueError) as exc:
        generate_buy_the_dip_labels(
            _bars(),
            _safe_config(
                take_profit_pct=0.006,
                number_of_research_trades=5,
                economically_viable_config=False,
            ),
        )

    assert "take_profit_not_above_round_trip_cost" in str(exc.value)
    assert "number_of_trades_below_minimum" in str(exc.value)
