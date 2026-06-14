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


EXTREMELY_STRONG_TREND = 1.5
MIN_MOMENTUM_SHORT_RETURN = 0.0008
MIN_MOMENTUM_BREAKOUT = 0.0005


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
            return hold_signal(self.name, "momentum_spread_too_wide", metadata={"spread_bps": spread})
        if imbalance is None:
            return hold_signal(self.name, "quote_imbalance_unavailable")
        if imbalance < self.settings.min_quote_imbalance:
            return hold_signal(
                self.name,
                "momentum_quote_imbalance_too_weak",
                metadata={"quote_imbalance": imbalance},
            )

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

        extremely_strong_trend = trend_strength >= EXTREMELY_STRONG_TREND
        volatility_limit = min(0.02, max(0.0, float(self.settings.regime_no_trade_volatility_threshold)))
        if volatility >= volatility_limit:
            return hold_signal(self.name, "volatility_too_high", metadata={"volatility": volatility})
        if volume_zscore < 0:
            return hold_signal(self.name, "momentum_volume_too_weak", metadata={"volume_zscore": volume_zscore})
        if not extremely_strong_trend and short_return < MIN_MOMENTUM_SHORT_RETURN:
            return hold_signal(
                self.name,
                "momentum_short_return_too_weak",
                metadata={"short_return": short_return, "required_short_return": MIN_MOMENTUM_SHORT_RETURN},
            )
        if not extremely_strong_trend and breakout <= MIN_MOMENTUM_BREAKOUT:
            return hold_signal(
                self.name,
                "momentum_breakout_too_weak",
                metadata={"breakout": breakout, "required_breakout": MIN_MOMENTUM_BREAKOUT},
            )
        momentum_ok = short_return >= MIN_MOMENTUM_SHORT_RETURN or extremely_strong_trend
        breakout_ok = breakout > MIN_MOMENTUM_BREAKOUT or extremely_strong_trend
        volume_ok = volume_zscore >= 0
        if not momentum_ok:
            return hold_signal(self.name, "momentum_short_return_too_weak", metadata={"short_return": short_return})
        if not breakout_ok:
            return hold_signal(self.name, "momentum_breakout_too_weak", metadata={"breakout": breakout})
        if not volume_ok:
            return hold_signal(self.name, "momentum_volume_too_weak", metadata={"volume_zscore": volume_zscore})

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
