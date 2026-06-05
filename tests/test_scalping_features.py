import numpy as np
import pytest

from app.data.market_data import MarketDataClient
from app.data.scalping_features import (
    SCALPING_BAR_FEATURE_COLUMNS,
    SCALPING_FEATURE_COLUMNS,
    SCALPING_QUOTE_FEATURE_COLUMNS,
    build_scalping_features,
    latest_scalping_feature_row,
)


def test_scalping_feature_columns_exist():
    bars = MarketDataClient.synthetic_btc_bars(40)
    features = build_scalping_features(bars)

    for column in SCALPING_FEATURE_COLUMNS:
        assert column in features.columns


def test_scalping_features_never_contain_infinite_values():
    bars = MarketDataClient.synthetic_btc_bars(40)
    bars.loc[5, "open"] = 0
    bars.loc[6, "high"] = bars.loc[6, "low"]
    bars.loc[7, "volume"] = 0
    features = build_scalping_features(
        bars,
        quote={"bid_price": 65_000, "ask_price": 65_010, "bid_size": 2, "ask_size": 1},
    )

    numeric = features[SCALPING_FEATURE_COLUMNS].to_numpy(dtype=float)
    assert not np.isinf(numeric).any()


def test_scalping_volume_acceleration_is_finite_after_zero_previous_volume():
    bars = MarketDataClient.synthetic_btc_bars(40)
    bars.loc[10, "volume"] = 0
    bars.loc[11, "volume"] = 12

    features = build_scalping_features(bars)

    assert not np.isnan(features.loc[11, "scalping_volume_acceleration"])
    assert features.loc[11, "scalping_volume_acceleration"] == pytest.approx(np.log1p(12))


def test_scalping_quote_features_use_latest_valid_quote():
    bars = MarketDataClient.synthetic_btc_bars(40)
    latest_close = float(bars.iloc[-1]["close"])
    features = build_scalping_features(
        bars,
        quote={"bid_price": 65_000, "ask_price": 65_010, "bid_size": 2, "ask_size": 1},
    )
    latest = features.iloc[-1]
    expected_mid = 65_005
    expected_spread_pct = 10 / expected_mid

    assert latest["scalping_spread_pct"] == pytest.approx(expected_spread_pct)
    assert latest["scalping_spread_bps"] == pytest.approx(expected_spread_pct * 10_000)
    assert latest["scalping_quote_imbalance"] == pytest.approx(1 / 3)
    assert latest["scalping_bid_ask_mid_distance"] == pytest.approx((expected_mid - latest_close) / latest_close)
    assert features.iloc[:-1][SCALPING_QUOTE_FEATURE_COLUMNS].isna().all(axis=None)


def test_missing_quote_leaves_runtime_scalping_features_nan():
    bars = MarketDataClient.synthetic_btc_bars(40)
    features = build_scalping_features(bars)

    assert features[SCALPING_QUOTE_FEATURE_COLUMNS].isna().all(axis=None)


def test_latest_scalping_feature_row_requires_enough_bars():
    with pytest.raises(ValueError, match="Not enough bars"):
        latest_scalping_feature_row(MarketDataClient.synthetic_btc_bars(10))

    latest = latest_scalping_feature_row(MarketDataClient.synthetic_btc_bars(40))
    assert len(latest) == 1
    assert latest[SCALPING_BAR_FEATURE_COLUMNS].notna().all(axis=None)


def test_scalping_bar_features_do_not_use_future_data():
    bars = MarketDataClient.synthetic_btc_bars(50)
    original = build_scalping_features(bars).loc[25, SCALPING_BAR_FEATURE_COLUMNS].copy()
    changed = bars.copy()
    changed.loc[35:, "close"] = changed.loc[35:, "close"] * 10
    after_change = build_scalping_features(changed).loc[25, SCALPING_BAR_FEATURE_COLUMNS]

    assert original.equals(after_change)
