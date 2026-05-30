import pandas as pd

from app.data.feature_engineering import BAR_FEATURE_COLUMNS, add_features
from app.ml.labels import net_profit_scalping_labels, triple_barrier_labels


def build_training_dataset(
    bars: pd.DataFrame,
    *,
    horizon_bars: int = 12,
    take_profit_pct: float = 0.03,
    stop_loss_pct: float = 0.015,
    scalping_mode_enabled: bool = False,
    trailing_stop_pct: float = 0.0,
    trailing_stop_arm_profit_pct: float = 0.0,
    fee_bps_per_side: float = 0.0,
    slippage_bps_per_side: float = 0.0,
    spread_cost_pct: float = 0.0,
    min_net_exit_profit_pct: float = 0.0,
    exit_profit_buffer_bps: float = 0.0,
) -> pd.DataFrame:
    featured = add_features(bars)
    if scalping_mode_enabled:
        labeled = net_profit_scalping_labels(
            featured,
            horizon_bars=horizon_bars,
            take_profit_pct=take_profit_pct,
            stop_loss_pct=stop_loss_pct,
            trailing_stop_pct=trailing_stop_pct,
            trailing_stop_arm_profit_pct=trailing_stop_arm_profit_pct,
            fee_bps_per_side=fee_bps_per_side,
            slippage_bps_per_side=slippage_bps_per_side,
            spread_cost_pct=spread_cost_pct,
            min_net_exit_profit_pct=min_net_exit_profit_pct,
            exit_profit_buffer_bps=exit_profit_buffer_bps,
        )
    else:
        labeled = triple_barrier_labels(
            featured,
            horizon_bars=horizon_bars,
            take_profit_pct=take_profit_pct,
            stop_loss_pct=stop_loss_pct,
        )
    return labeled.dropna(subset=BAR_FEATURE_COLUMNS + ["buy_quality_label", "sell_quality_label"]).reset_index(drop=True)
