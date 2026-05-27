import pandas as pd

from app.config import Settings, get_settings


class RuleFilter:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def allow_buy(self, feature_row: pd.Series, *, buy_probability: float) -> tuple[bool, str]:
        vol = float(feature_row.get("volatility_20", 0))
        spread = float(feature_row.get("orderbook_spread", 0))
        range_pct = float(feature_row.get("high_low_range_pct", 0))
        ema_slow_distance = float(feature_row.get("ema_slow_distance", 0))
        if vol > 0.035:
            return False, "volatility_abnormally_high"
        if spread > 0.004:
            return False, "spread_too_wide"
        if range_pct > 0.06:
            return False, "recent_candle_extreme_range"
        if ema_slow_distance < 0 and buy_probability < self.settings.min_buy_probability + 0.05:
            return False, "below_slow_ema_with_marginal_confidence"
        return True, "allowed"
