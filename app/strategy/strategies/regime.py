from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import pandas as pd

from app.config import Settings, get_settings
from app.strategy.strategies.base import clamp, first_finite, quote_imbalance, spread_bps


RegimeName = Literal["trending", "mean_reverting", "ranging", "too_volatile", "not_tradeable"]


@dataclass(frozen=True)
class MarketRegime:
    regime: RegimeName
    confidence: float
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MarketRegimeFilter:
    """Classify current market regime from row-local rolling features only."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def detect(self, feature_row: pd.Series, *, quote: dict | None = None) -> MarketRegime:
        latest_price = first_finite(feature_row, "close", default=0.0) or 0.0
        spread = spread_bps(feature_row, quote, self.settings)
        imbalance = quote_imbalance(feature_row, quote)
        volatility = first_finite(
            feature_row,
            "scalping_volatility_10",
            "scalping_volatility_5",
            "volatility_20",
            "volatility_10",
            default=0.0,
        ) or 0.0
        trend_strength = abs(
            first_finite(
                feature_row,
                "trend_strength_20",
                "scalping_momentum_10",
                "log_return_20",
                "scalping_log_return_5",
                default=0.0,
            )
            or 0.0
        )
        short_return = first_finite(
            feature_row,
            "scalping_log_return_3",
            "scalping_momentum_3",
            "log_return_3",
            default=0.0,
        ) or 0.0
        breakout = first_finite(feature_row, "scalping_high_breakout_5", "high_breakout_5", default=0.0) or 0.0
        low_breakout = first_finite(feature_row, "scalping_low_breakout_5", "low_breakout_5", default=0.0) or 0.0
        metadata = {
            "latest_price": latest_price,
            "spread_bps": spread,
            "quote_imbalance": imbalance,
            "volatility": volatility,
            "trend_strength": trend_strength,
            "short_return": short_return,
            "high_breakout": breakout,
            "low_breakout": low_breakout,
        }

        if latest_price <= 0:
            return MarketRegime("not_tradeable", 1.0, "invalid_price", metadata)
        if spread is None:
            return MarketRegime("not_tradeable", 0.9, "spread_unavailable", metadata)
        if spread > self.settings.max_spread_bps:
            return MarketRegime("not_tradeable", 0.9, "spread_too_wide", metadata)
        if imbalance is None:
            return MarketRegime("not_tradeable", 0.8, "quote_imbalance_unavailable", metadata)
        if (
            volatility >= self.settings.regime_no_trade_volatility_threshold
            or abs(short_return) >= self.settings.regime_no_trade_short_return_threshold
        ):
            return MarketRegime("too_volatile", 0.85, "volatility_too_high", metadata)
        if trend_strength >= self.settings.regime_trend_strength_threshold or breakout > self.settings.regime_breakout_threshold:
            confidence = clamp(0.55 + min(0.35, trend_strength / 4) + max(0.0, min(0.1, breakout * 20)))
            return MarketRegime("trending", confidence, "trend_or_breakout_detected", metadata)
        if (
            abs(short_return) <= self.settings.regime_mean_reversion_short_return_threshold
            and abs(low_breakout) <= self.settings.regime_mean_reversion_low_breakout_threshold
        ):
            confidence = clamp(
                0.55
                + min(
                    0.25,
                    (self.settings.regime_mean_reversion_low_breakout_threshold - abs(short_return)) * 20,
                )
            )
            return MarketRegime("mean_reverting", confidence, "range_or_mean_reversion_detected", metadata)
        return MarketRegime("ranging", 0.55, "range_conditions_detected", metadata)

    def allows(self, regime: MarketRegime, strategy_name: str) -> tuple[bool, str]:
        if regime.regime in {"too_volatile", "not_tradeable"}:
            return False, regime.reason
        if strategy_name == "mean_reversion_scalping":
            if regime.regime in {"mean_reverting", "ranging"}:
                return True, "allowed"
            return False, "regime_not_mean_reverting"
        if strategy_name in {"momentum_breakout", "trend_pullback"}:
            if regime.regime == "trending":
                return True, "allowed"
            return False, "regime_not_trending"
        return False, "unknown_strategy"
