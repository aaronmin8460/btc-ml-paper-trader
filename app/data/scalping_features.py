import math
from typing import Any

import numpy as np
import pandas as pd


SCALPING_BAR_FEATURE_COLUMNS = [
    "scalping_log_return_1",
    "scalping_log_return_2",
    "scalping_log_return_3",
    "scalping_log_return_5",
    "scalping_volatility_3",
    "scalping_volatility_5",
    "scalping_volatility_10",
    "scalping_momentum_3",
    "scalping_momentum_5",
    "scalping_momentum_10",
    "scalping_body_pct",
    "scalping_upper_wick_pct",
    "scalping_lower_wick_pct",
    "scalping_range_pct",
    "scalping_close_position_in_candle",
    "scalping_volume_ratio_5",
    "scalping_volume_zscore_10",
    "scalping_volume_acceleration",
    "scalping_high_breakout_5",
    "scalping_low_breakout_5",
    "scalping_vwap_distance",
    "scalping_ema_5_distance",
    "scalping_ema_10_distance",
    "scalping_rsi_3",
    "scalping_rsi_5",
    "scalping_stochastic_k_5",
]

SCALPING_QUOTE_FEATURE_COLUMNS = [
    "scalping_spread_pct",
    "scalping_spread_bps",
    "scalping_quote_imbalance",
    "scalping_bid_ask_mid_distance",
]

SCALPING_FEATURE_COLUMNS = SCALPING_BAR_FEATURE_COLUMNS + SCALPING_QUOTE_FEATURE_COLUMNS

REQUIRED_BAR_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def build_scalping_features(bars: pd.DataFrame, quote: dict | None = None) -> pd.DataFrame:
    """Build BTC/USD scalping features without changing the legacy model schema.

    Quote-derived values are runtime-only. A quote describes the current market,
    so its values are attached only to the latest bar instead of being copied
    onto historical rows.
    """
    missing = [column for column in REQUIRED_BAR_COLUMNS if column not in bars.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")

    data = bars.copy().sort_values("timestamp").reset_index(drop=True)
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    for column in ["open", "high", "low", "close", "volume"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    open_price = data["open"]
    high = data["high"]
    low = data["low"]
    close = data["close"]
    volume = data["volume"]
    positive_close = close.where(close > 0)
    log_close = np.log(positive_close)
    log_return_1 = log_close.diff()

    for period in [1, 2, 3, 5]:
        data[f"scalping_log_return_{period}"] = log_close.diff(period)

    for window in [3, 5, 10]:
        data[f"scalping_volatility_{window}"] = log_return_1.rolling(window).std()
        data[f"scalping_momentum_{window}"] = close.pct_change(periods=window, fill_method=None)

    candle_denominator = positive_close
    data["scalping_body_pct"] = _safe_divide(close - open_price, open_price.where(open_price > 0))
    data["scalping_upper_wick_pct"] = _safe_divide(
        high - pd.concat([open_price, close], axis=1).max(axis=1),
        candle_denominator,
    )
    data["scalping_lower_wick_pct"] = _safe_divide(
        pd.concat([open_price, close], axis=1).min(axis=1) - low,
        candle_denominator,
    )
    candle_range = high - low
    data["scalping_range_pct"] = _safe_divide(candle_range, candle_denominator)
    data["scalping_close_position_in_candle"] = _safe_divide(close - low, candle_range)

    volume_mean_5 = volume.rolling(5).mean()
    volume_mean_10 = volume.rolling(10).mean()
    volume_std_10 = volume.rolling(10).std()
    data["scalping_volume_ratio_5"] = _safe_divide(volume, volume_mean_5)
    data["scalping_volume_zscore_10"] = _safe_divide(volume - volume_mean_10, volume_std_10)
    data["scalping_volume_acceleration"] = np.log1p(volume.clip(lower=0)).diff().fillna(0.0)

    prior_high_5 = high.shift(1).rolling(5).max()
    prior_low_5 = low.shift(1).rolling(5).min()
    data["scalping_high_breakout_5"] = _safe_divide(close - prior_high_5, prior_high_5)
    data["scalping_low_breakout_5"] = _safe_divide(close - prior_low_5, prior_low_5)

    typical_price = (high + low + close) / 3
    utc_date = data["timestamp"].dt.date
    cumulative_typical_volume = (typical_price * volume).groupby(utc_date).cumsum()
    cumulative_volume = volume.groupby(utc_date).cumsum()
    vwap = _safe_divide(cumulative_typical_volume, cumulative_volume)
    ema_5 = close.ewm(span=5, adjust=False).mean()
    ema_10 = close.ewm(span=10, adjust=False).mean()
    data["scalping_vwap_distance"] = _safe_divide(close - vwap, vwap)
    data["scalping_ema_5_distance"] = _safe_divide(close - ema_5, ema_5)
    data["scalping_ema_10_distance"] = _safe_divide(close - ema_10, ema_10)

    data["scalping_rsi_3"] = _compute_rsi(close, window=3)
    data["scalping_rsi_5"] = _compute_rsi(close, window=5)
    rolling_low_5 = low.rolling(5).min()
    rolling_high_5 = high.rolling(5).max()
    data["scalping_stochastic_k_5"] = 100 * _safe_divide(close - rolling_low_5, rolling_high_5 - rolling_low_5)

    for column in SCALPING_QUOTE_FEATURE_COLUMNS:
        data[column] = np.nan
    if not data.empty:
        latest_close = _positive_float(close.iloc[-1])
        for column, value in _quote_features(quote, latest_close=latest_close).items():
            data.loc[data.index[-1], column] = value

    return data.replace([np.inf, -np.inf], np.nan)


def latest_scalping_feature_row(bars: pd.DataFrame, quote: dict | None = None) -> pd.DataFrame:
    features = build_scalping_features(bars, quote=quote).dropna(subset=SCALPING_BAR_FEATURE_COLUMNS)
    if features.empty:
        raise ValueError("Not enough bars to compute scalping features.")
    return features.tail(1)


def _compute_rsi(close: pd.Series, *, window: int) -> pd.Series:
    delta = close.diff()
    average_gain = delta.clip(lower=0).rolling(window).mean()
    average_loss = (-delta.clip(upper=0)).rolling(window).mean()
    return 100 * _safe_divide(average_gain, average_gain + average_loss)


def _quote_features(quote: dict | None, *, latest_close: float | None) -> dict[str, float]:
    values = {column: math.nan for column in SCALPING_QUOTE_FEATURE_COLUMNS}
    if quote is None:
        return values

    bid = _quote_float(quote, "bid_price", "bp", "bid")
    ask = _quote_float(quote, "ask_price", "ap", "ask")
    bid_size = _quote_float(quote, "bid_size", "bs", allow_zero=True)
    ask_size = _quote_float(quote, "ask_size", "as", allow_zero=True)

    if bid is not None and ask is not None and ask >= bid:
        mid = (bid + ask) / 2
        spread_pct = (ask - bid) / mid if mid > 0 else math.nan
        values["scalping_spread_pct"] = spread_pct
        values["scalping_spread_bps"] = spread_pct * 10_000
        if latest_close is not None:
            values["scalping_bid_ask_mid_distance"] = (mid - latest_close) / latest_close

    if bid_size is not None and ask_size is not None:
        total_size = bid_size + ask_size
        if total_size > 0:
            values["scalping_quote_imbalance"] = (bid_size - ask_size) / total_size

    return values


def _quote_float(quote: dict[str, Any], *keys: str, allow_zero: bool = False) -> float | None:
    for key in keys:
        parsed = _finite_float(quote.get(key))
        if parsed is None:
            continue
        if parsed > 0 or (allow_zero and parsed == 0):
            return parsed
    return None


def _positive_float(value: Any) -> float | None:
    parsed = _finite_float(value)
    return parsed if parsed is not None and parsed > 0 else None


def _finite_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)
