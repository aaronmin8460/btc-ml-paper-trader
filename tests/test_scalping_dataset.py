from datetime import UTC, datetime

import pandas as pd
import pytest

from app.data.dataset_builder import build_training_dataset, training_feature_columns
from app.data.feature_engineering import BAR_FEATURE_COLUMNS, add_features
from app.data.market_data import MarketDataClient
from app.data.scalping_features import SCALPING_BAR_FEATURE_COLUMNS, SCALPING_FEATURE_COLUMNS
from app.ml.labels import net_profit_scalping_labels, triple_barrier_labels


def test_scalping_dataset_uses_scalping_feature_columns():
    dataset = build_training_dataset(
        MarketDataClient.synthetic_btc_bars(180),
        scalping_mode_enabled=True,
    )

    assert not dataset.empty
    assert set(SCALPING_FEATURE_COLUMNS).issubset(dataset.columns)
    assert training_feature_columns(True) == SCALPING_BAR_FEATURE_COLUMNS


def test_scalping_dataset_horizon_is_configurable():
    bars = MarketDataClient.synthetic_btc_bars(180)

    one_bar_dataset = build_training_dataset(
        bars,
        scalping_mode_enabled=True,
        scalping_label_horizon_bars=1,
    )
    three_bar_dataset = build_training_dataset(
        bars,
        scalping_mode_enabled=True,
        scalping_label_horizon_bars=3,
    )

    assert len(one_bar_dataset) == len(three_bar_dataset) + 2


@pytest.mark.parametrize("horizon_bars", [0, 4])
def test_scalping_dataset_rejects_non_scalping_horizons(horizon_bars):
    with pytest.raises(ValueError, match="between 1 and 3"):
        build_training_dataset(
            MarketDataClient.synthetic_btc_bars(180),
            scalping_mode_enabled=True,
            scalping_label_horizon_bars=horizon_bars,
        )


def test_scalping_ambiguous_candle_resolves_to_stop_loss_first():
    bars = pd.DataFrame(
        {
            "timestamp": pd.date_range(datetime(2026, 5, 27, tzinfo=UTC), periods=3, freq="min"),
            "open": [100.0, 100.0, 100.0],
            "high": [100.0, 100.2, 100.0],
            "low": [100.0, 99.9, 100.0],
            "close": [100.0, 100.0, 100.0],
            "volume": [1.0, 1.0, 1.0],
        }
    )

    labeled = net_profit_scalping_labels(
        bars,
        horizon_bars=1,
        take_profit_pct=0.0012,
        stop_loss_pct=0.0008,
    )

    assert int(labeled.loc[0, "buy_quality_label"]) == 0
    assert int(labeled.loc[0, "sell_quality_label"]) == 1
    assert labeled.loc[0, "buy_exit_reason"] == "ambiguous_stop_first"


def test_non_scalping_dataset_behavior_remains_unchanged():
    bars = MarketDataClient.synthetic_btc_bars(180)
    expected = triple_barrier_labels(add_features(bars)).dropna(
        subset=BAR_FEATURE_COLUMNS + ["buy_quality_label", "sell_quality_label"]
    ).reset_index(drop=True)

    actual = build_training_dataset(bars)

    pd.testing.assert_frame_equal(actual, expected)
    assert training_feature_columns(False) == BAR_FEATURE_COLUMNS
