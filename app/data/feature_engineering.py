import numpy as np
import pandas as pd


BAR_FEATURE_COLUMNS = [
    "log_return_1",
    "log_return_3",
    "log_return_5",
    "log_return_10",
    "log_return_20",
    "volatility_5",
    "volatility_10",
    "volatility_20",
    "volatility_50",
    "rolling_mean_return_10",
    "ema_fast_distance",
    "ema_slow_distance",
    "sma_20_distance",
    "sma_50_distance",
    "rsi_14",
    "macd",
    "macd_hist",
    "bb_width",
    "bb_close_position",
    "atr_14",
    "normalized_volume",
    "volume_zscore_20",
    "high_low_range_pct",
    "close_open_pct",
    "rolling_max_drawdown_50",
    "trend_strength_20",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
]

RUNTIME_ONLY_FEATURE_COLUMNS = [
    "orderbook_spread",
    "quote_imbalance",
]

FEATURE_COLUMNS = BAR_FEATURE_COLUMNS


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [(df["high"] - df["low"]), (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window).mean()


def add_features(df: pd.DataFrame, quote: dict | None = None) -> pd.DataFrame:
    data = df.copy().sort_values("timestamp").reset_index(drop=True)
    close = data["close"]
    log_close = np.log(close)
    log_ret = log_close.diff()

    for period in [1, 3, 5, 10, 20]:
        data[f"log_return_{period}"] = log_close.diff(period)
    for window in [5, 10, 20, 50]:
        data[f"volatility_{window}"] = log_ret.rolling(window).std()

    data["rolling_mean_return_10"] = log_ret.rolling(10).mean()
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    data["ema_fast_distance"] = (close - ema_fast) / close
    data["ema_slow_distance"] = (close - ema_slow) / close
    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()
    data["sma_20_distance"] = (close - sma_20) / close
    data["sma_50_distance"] = (close - sma_50) / close
    data["rsi_14"] = compute_rsi(close)
    data["macd"] = (ema_fast - ema_slow) / close
    macd_signal = (ema_fast - ema_slow).ewm(span=9, adjust=False).mean()
    data["macd_hist"] = ((ema_fast - ema_slow) - macd_signal) / close
    bb_std = close.rolling(20).std()
    bb_upper = sma_20 + 2 * bb_std
    bb_lower = sma_20 - 2 * bb_std
    data["bb_width"] = (bb_upper - bb_lower) / close
    data["bb_close_position"] = (close - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)
    data["atr_14"] = compute_atr(data) / close
    volume_mean_20 = data["volume"].rolling(20).mean()
    volume_std_20 = data["volume"].rolling(20).std()
    data["normalized_volume"] = data["volume"] / volume_mean_20.replace(0, np.nan)
    data["volume_zscore_20"] = (data["volume"] - volume_mean_20) / volume_std_20.replace(0, np.nan)
    data["high_low_range_pct"] = (data["high"] - data["low"]) / close
    data["close_open_pct"] = (data["close"] - data["open"]) / data["open"]
    rolling_peak = close.rolling(50).max()
    data["rolling_max_drawdown_50"] = (close / rolling_peak) - 1
    data["trend_strength_20"] = close.pct_change(20) / data["volatility_20"].replace(0, np.nan)
    data["hour_of_day"] = data["timestamp"].dt.hour
    data["day_of_week"] = data["timestamp"].dt.dayofweek
    data["is_weekend"] = data["day_of_week"].isin([5, 6]).astype(int)

    spread = np.nan
    imbalance = np.nan
    if quote:
        bid = quote.get("bid_price") or quote.get("bp")
        ask = quote.get("ask_price") or quote.get("ap")
        bid_size = quote.get("bid_size") or quote.get("bs")
        ask_size = quote.get("ask_size") or quote.get("as")
        if bid and ask:
            mid = (float(bid) + float(ask)) / 2
            spread = (float(ask) - float(bid)) / mid if mid else np.nan
        if bid_size and ask_size:
            denom = float(bid_size) + float(ask_size)
            imbalance = (float(bid_size) - float(ask_size)) / denom if denom else np.nan
    data["orderbook_spread"] = spread
    data["quote_imbalance"] = imbalance
    data[["orderbook_spread", "quote_imbalance"]] = data[["orderbook_spread", "quote_imbalance"]].fillna(0)
    return data.replace([np.inf, -np.inf], np.nan)


def latest_feature_row(df: pd.DataFrame, quote: dict | None = None) -> pd.DataFrame:
    features = add_features(df, quote=quote).dropna(subset=BAR_FEATURE_COLUMNS)
    if features.empty:
        raise ValueError("Not enough bars to compute features.")
    return features.tail(1)
