import numpy as np
import pandas as pd


def triple_barrier_labels(
    df: pd.DataFrame,
    *,
    horizon_bars: int = 12,
    take_profit_pct: float = 0.03,
    stop_loss_pct: float = 0.015,
) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    labels: list[int | float] = []
    sell_labels: list[int | float] = []
    for i in range(len(out)):
        if i + horizon_bars >= len(out):
            labels.append(np.nan)
            sell_labels.append(np.nan)
            continue
        entry = out.loc[i, "close"]
        tp = entry * (1 + take_profit_pct)
        sl = entry * (1 - stop_loss_pct)
        label = 0
        sell_label = 0
        future = out.iloc[i + 1 : i + horizon_bars + 1]
        for _, row in future.iterrows():
            hit_tp = row["high"] >= tp
            hit_sl = row["low"] <= sl
            if hit_tp and hit_sl:
                label = 0
                sell_label = 1
                break
            if hit_tp:
                label = 1
                sell_label = 0
                break
            if hit_sl:
                label = 0
                sell_label = 1
                break
        labels.append(label)
        sell_labels.append(sell_label)
    out["buy_quality_label"] = labels
    out["sell_quality_label"] = sell_labels
    return out.dropna(subset=["buy_quality_label", "sell_quality_label"]).reset_index(drop=True)
