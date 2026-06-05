from __future__ import annotations

import pandas as pd

from app.config import Settings, get_settings
from app.risk.risk_manager import PositionState
from app.strategy.strategies.base import MarketContext, StrategySignal, clamp, first_finite, hold_signal, spread_bps


class TrendPullbackStrategy:
    name = "trend_pullback"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def generate_signal(
        self,
        *,
        feature_row: pd.Series,
        prediction: dict | None = None,
        position: PositionState,
        quote: dict | None = None,
        market_context: MarketContext | None = None,
    ) -> StrategySignal:
        context = market_context or MarketContext()
        if position.has_position:
            return hold_signal(self.name, "already_holding_btc")
        if not context.risk_permits_evaluation:
            return hold_signal(self.name, context.risk_reason or "risk_context_blocked")

        regime = getattr(context.regime, "regime", None)
        if regime != "trending":
            return hold_signal(self.name, "regime_not_trending", metadata={"regime": regime})

        spread = spread_bps(feature_row, quote, self.settings)
        if spread is None:
            return hold_signal(self.name, "spread_unavailable")
        if spread > self.settings.max_spread_bps:
            return hold_signal(self.name, "spread_too_wide", metadata={"spread_bps": spread})

        trend_strength = first_finite(feature_row, "trend_strength_20", "log_return_20", default=0.0) or 0.0
        rsi = first_finite(feature_row, "rsi_14", default=50.0) or 50.0
        macd_hist = first_finite(feature_row, "macd_hist", "macd", default=0.0) or 0.0
        atr = first_finite(feature_row, "atr_14", "volatility_20", default=0.0) or 0.0
        volume_zscore = first_finite(feature_row, "volume_zscore_20", "normalized_volume", default=0.0) or 0.0
        ema_fast_distance = first_finite(feature_row, "ema_fast_distance", "ema_20_distance", default=0.0) or 0.0
        ema_slow_distance = first_finite(feature_row, "ema_slow_distance", "ema_50_distance", default=0.0) or 0.0
        pullback_return = first_finite(feature_row, "log_return_3", "log_return_5", default=0.0) or 0.0

        trend_ok = trend_strength > 0 and ema_slow_distance >= -0.003
        pullback_ok = 38 <= rsi <= 58 and -0.018 <= pullback_return <= 0.004 and ema_fast_distance <= 0.006
        macd_ok = macd_hist >= -0.0015
        volatility_ok = 0.001 <= atr <= 0.035
        volume_ok = volume_zscore >= -1.0
        if not trend_ok:
            return hold_signal(
                self.name,
                "trend_not_confirmed",
                metadata={"trend_strength": trend_strength, "ema_slow_distance": ema_slow_distance},
            )
        if not pullback_ok:
            return hold_signal(
                self.name,
                "pullback_not_constructive",
                metadata={"rsi": rsi, "pullback_return": pullback_return, "ema_fast_distance": ema_fast_distance},
            )
        if not macd_ok:
            return hold_signal(self.name, "macd_not_supportive", metadata={"macd_hist": macd_hist})
        if not volatility_ok:
            return hold_signal(self.name, "volatility_not_tradeable", metadata={"atr": atr})
        if not volume_ok:
            return hold_signal(self.name, "volume_not_supportive", metadata={"volume_zscore": volume_zscore})

        trend_score = clamp(abs(trend_strength) / 2.5)
        pullback_score = clamp((58 - abs(rsi - 48)) / 58)
        macd_score = clamp((macd_hist + 0.0015) / 0.006)
        volatility_score = clamp(1.0 - abs(atr - 0.012) / 0.025)
        volume_score = clamp((volume_zscore + 1.0) / 3.0)
        spread_score = 1.0 - clamp(spread / max(0.01, self.settings.max_spread_bps))
        score = clamp(
            0.30 * trend_score
            + 0.20 * pullback_score
            + 0.20 * macd_score
            + 0.15 * volatility_score
            + 0.10 * volume_score
            + 0.05 * spread_score
        )
        confidence = clamp(0.45 + score * 0.45)
        return StrategySignal(
            action="buy",
            score=score,
            confidence=confidence,
            reason="trend_pullback_buy_candidate",
            strategy_name=self.name,
            metadata={
                "regime": regime,
                "spread_bps": spread,
                "trend_strength": trend_strength,
                "rsi": rsi,
                "macd_hist": macd_hist,
                "atr": atr,
                "volume_zscore": volume_zscore,
                "ema_fast_distance": ema_fast_distance,
                "ema_slow_distance": ema_slow_distance,
                "pullback_return": pullback_return,
            },
        )
