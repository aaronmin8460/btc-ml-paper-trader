import pandas as pd

from app.data.feature_engineering import FEATURE_COLUMNS, add_features
from app.ml.labels import triple_barrier_labels


def build_training_dataset(
    bars: pd.DataFrame,
    *,
    horizon_bars: int = 12,
    take_profit_pct: float = 0.03,
    stop_loss_pct: float = 0.015,
) -> pd.DataFrame:
    featured = add_features(bars)
    labeled = triple_barrier_labels(
        featured,
        horizon_bars=horizon_bars,
        take_profit_pct=take_profit_pct,
        stop_loss_pct=stop_loss_pct,
    )
    return labeled.dropna(subset=FEATURE_COLUMNS + ["buy_quality_label", "sell_quality_label"]).reset_index(drop=True)
