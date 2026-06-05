from __future__ import annotations

import pandas as pd

from app.config import Settings, get_settings
from app.risk.risk_manager import PositionState
from app.strategy.strategies.base import (
    MarketContext,
    StrategySignal,
    clamp,
    first_finite,
    hold_signal,
    quote_imbalance,
    spread_bps,
)


class MomentumBreakoutStrategy:
    name = "momentum_breakout"

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

        spread = spread_bps(feature_row, quote, self.settings)
        imbalance = quote_imbalance(feature_row, quote)
        if spread is None:
            return hold_signal(self.name, "spread_unavailable")
        if spread > self.settings.max_spread_bps:
            return hold_signal(self.name, "spread_too_wide", metadata={"spread_bps": spread})
        if imbalance is None:
            return hold_signal(self.name, "quote_imbalance_unavailable")
        if imbalance < max(self.settings.min_quote_imbalance, -0.05):
            return hold_signal(self.name, "quote_imbalance_not_supportive", metadata={"quote_imbalance": imbalance})

        regime = getattr(context.regime, "regime", None)
        if regime != "trending":
            return hold_signal(self.name, "regime_not_trending", metadata={"regime": regime})

        short_return = first_finite(feature_row, "scalping_log_return_3", "scalping_momentum_3", "log_return_3", default=0.0) or 0.0
        longer_return = first_finite(feature_row, "scalping_log_return_5", "scalping_momentum_5", "log_return_5", default=0.0) or 0.0
        trend_strength = first_finite(feature_row, "trend_strength_20", "scalping_momentum_10", default=0.0) or 0.0
        macd = first_finite(feature_row, "macd_hist", "macd", default=0.0) or 0.0
        volume_zscore = first_finite(feature_row, "scalping_volume_zscore_10", "volume_zscore_20", default=0.0) or 0.0
        volatility = first_finite(feature_row, "scalping_volatility_10", "scalping_volatility_5", "atr_14", "volatility_20", default=0.0) or 0.0
        breakout = first_finite(feature_row, "scalping_high_breakout_5", "high_breakout_5", default=0.0) or 0.0

        momentum_ok = short_return > 0 and longer_return >= -0.001
        breakout_ok = breakout > 0 or trend_strength > 0.8 or macd > 0
        volume_ok = volume_zscore >= -0.5
        volatility_ok = volatility < 0.02
        if not momentum_ok:
            return hold_signal(self.name, "momentum_not_positive", metadata={"short_return": short_return})
        if not breakout_ok:
            return hold_signal(self.name, "breakout_not_confirmed", metadata={"breakout": breakout})
        if not volume_ok:
            return hold_signal(self.name, "volume_not_supportive", metadata={"volume_zscore": volume_zscore})
        if not volatility_ok:
            return hold_signal(self.name, "volatility_too_high", metadata={"volatility": volatility})

        momentum_score = clamp(max(short_return, longer_return, trend_strength / 10, macd * 100) / 0.004)
        breakout_score = clamp(max(0.0, breakout) / 0.003)
        volume_score = clamp((volume_zscore + 0.5) / 2.5)
        spread_score = 1.0 - clamp(spread / max(0.01, self.settings.max_spread_bps))
        imbalance_score = clamp((imbalance + 0.05) / 1.05)
        score = clamp(
            0.35 * momentum_score
            + 0.25 * breakout_score
            + 0.15 * volume_score
            + 0.15 * spread_score
            + 0.10 * imbalance_score
        )
        confidence = clamp(0.45 + score * 0.45)
        return StrategySignal(
            action="buy",
            score=score,
            confidence=confidence,
            reason="momentum_breakout_buy_candidate",
            strategy_name=self.name,
            metadata={
                "short_return": short_return,
                "longer_return": longer_return,
                "trend_strength": trend_strength,
                "macd": macd,
                "volume_zscore": volume_zscore,
                "volatility": volatility,
                "breakout": breakout,
                "spread_bps": spread,
                "quote_imbalance": imbalance,
                "regime": regime,
            },
        )
