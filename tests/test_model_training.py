from app.data.dataset_builder import build_training_dataset
from app.data.feature_engineering import FEATURE_COLUMNS
from app.data.market_data import MarketDataClient
from app.ml.model import MLSignalModel


def test_model_training_produces_probabilities():
    bars = MarketDataClient.synthetic_btc_bars(260)
    dataset = build_training_dataset(bars)
    model = MLSignalModel(feature_columns=FEATURE_COLUMNS).train(dataset)
    probabilities = model.predict_proba(dataset.tail(5))
    assert len(probabilities) == 5
    assert all(0 <= value <= 1 for value in probabilities)


def test_labels_drop_rows_where_future_unavailable():
    bars = MarketDataClient.synthetic_btc_bars(120)
    dataset = build_training_dataset(bars)
    assert len(dataset) < len(bars)
    assert "buy_quality_label" in dataset.columns
