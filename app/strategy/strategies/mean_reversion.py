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


class MeanReversionScalpingStrategy:
    name = "mean_reversion_scalping"

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
        if imbalance < self.settings.min_quote_imbalance:
            return hold_signal(self.name, "quote_imbalance_too_weak", metadata={"quote_imbalance": imbalance})

        regime = getattr(context.regime, "regime", None)
        if regime not in {"mean_reverting", "ranging"}:
            return hold_signal(self.name, "regime_not_mean_reverting", metadata={"regime": regime})

        rsi = first_finite(feature_row, "scalping_rsi_3", "scalping_rsi_5", "rsi_14", default=50.0) or 50.0
        bb_position = first_finite(feature_row, "bb_close_position")
        ema_distance = first_finite(feature_row, "scalping_ema_5_distance", "scalping_ema_10_distance", "ema_fast_distance", "sma_20_distance", default=0.0) or 0.0
        vwap_distance = first_finite(feature_row, "scalping_vwap_distance", "vwap_distance", default=0.0) or 0.0
        short_return = first_finite(feature_row, "scalping_log_return_3", "scalping_momentum_3", "log_return_3", default=0.0) or 0.0
        low_breakout = first_finite(feature_row, "scalping_low_breakout_5", "low_breakout_5", default=0.0) or 0.0
        high_breakout = first_finite(feature_row, "scalping_high_breakout_5", "high_breakout_5", default=0.0) or 0.0
        volatility = first_finite(feature_row, "scalping_volatility_10", "scalping_volatility_5", "atr_14", "volatility_20", default=0.0) or 0.0

        stretched_down = (
            rsi <= 45
            or (bb_position is not None and bb_position <= 0.35)
            or ema_distance <= -self.settings.scalping_entry_dip_pct
            or vwap_distance <= -self.settings.scalping_entry_dip_pct
            or short_return <= -self.settings.scalping_entry_dip_pct
            or low_breakout <= -self.settings.scalping_entry_dip_pct
            or high_breakout <= -self.settings.scalping_entry_dip_pct
        )
        crash_like = short_return <= -0.01 or volatility >= 0.02
        if not stretched_down:
            return hold_signal(self.name, "mean_reversion_stretch_absent", metadata={"rsi": rsi})
        if crash_like:
            return hold_signal(
                self.name,
                "mean_reversion_crash_regime",
                metadata={"short_return": short_return, "volatility": volatility},
            )

        stretch_score = max(
            clamp((45 - rsi) / 25),
            clamp(abs(min(0.0, ema_distance)) / 0.004),
            clamp(abs(min(0.0, vwap_distance)) / 0.004),
            clamp(abs(min(0.0, short_return)) / 0.004),
            clamp(abs(min(0.0, low_breakout)) / 0.004),
            clamp(abs(min(0.0, high_breakout)) / 0.004),
        )
        spread_score = 1.0 - clamp(spread / max(0.01, self.settings.max_spread_bps))
        imbalance_score = clamp((imbalance - self.settings.min_quote_imbalance) / max(0.01, 1 - self.settings.min_quote_imbalance))
        score = clamp(0.55 * stretch_score + 0.25 * spread_score + 0.20 * imbalance_score)
        confidence = clamp(0.45 + score * 0.45)
        return StrategySignal(
            action="buy",
            score=score,
            confidence=confidence,
            reason="mean_reversion_buy_candidate",
            strategy_name=self.name,
            metadata={
                "rsi": rsi,
                "bb_close_position": bb_position,
                "ema_distance": ema_distance,
                "vwap_distance": vwap_distance,
                "short_return": short_return,
                "low_breakout": low_breakout,
                "high_breakout": high_breakout,
                "volatility": volatility,
                "spread_bps": spread,
                "quote_imbalance": imbalance,
                "regime": regime,
            },
        )
