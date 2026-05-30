from app.data.feature_engineering import BAR_FEATURE_COLUMNS, RUNTIME_ONLY_FEATURE_COLUMNS, add_features
from app.data.market_data import MarketDataClient


def test_feature_engineering_returns_expected_columns():
    bars = MarketDataClient.synthetic_btc_bars(120)
    features = add_features(bars).dropna()
    for column in BAR_FEATURE_COLUMNS + RUNTIME_ONLY_FEATURE_COLUMNS:
        assert column in features.columns
    assert "orderbook_spread" not in BAR_FEATURE_COLUMNS
    assert "quote_imbalance" not in BAR_FEATURE_COLUMNS
    assert not features.empty


def test_feature_engineering_does_not_use_future_data():
    bars = MarketDataClient.synthetic_btc_bars(140)
    original = add_features(bars).loc[80, BAR_FEATURE_COLUMNS].copy()
    changed = bars.copy()
    changed.loc[100:, "close"] = changed.loc[100:, "close"] * 10
    after_change = add_features(changed).loc[80, BAR_FEATURE_COLUMNS]
    assert original.equals(after_change)
