from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol

import math
import pandas as pd

from app.config import Settings
from app.risk.risk_manager import AccountState, PositionState, TradeFrequencyState


StrategyAction = Literal["buy", "sell", "hold"]


@dataclass(frozen=True)
class MarketContext:
    regime: Any | None = None
    account_state: AccountState | None = None
    trade_frequency: TradeFrequencyState | None = None
    order_attempt_frequency: TradeFrequencyState | None = None
    risk_permits_evaluation: bool = True
    risk_reason: str | None = None


@dataclass(frozen=True)
class StrategySignal:
    action: StrategyAction
    score: float
    confidence: float
    reason: str
    strategy_name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class QuantStrategy(Protocol):
    name: str

    def generate_signal(
        self,
        *,
        feature_row: pd.Series,
        prediction: dict | None,
        position: PositionState,
        quote: dict | None,
        market_context: MarketContext,
    ) -> StrategySignal:
        ...


def hold_signal(strategy_name: str, reason: str, *, metadata: dict[str, Any] | None = None) -> StrategySignal:
    return StrategySignal(
        action="hold",
        score=0.0,
        confidence=0.0,
        reason=reason,
        strategy_name=strategy_name,
        metadata=metadata or {},
    )


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if not math.isfinite(value):
        return low
    return max(low, min(high, value))


def first_finite(feature_row: pd.Series, *keys: str, default: float | None = None) -> float | None:
    for key in keys:
        value = feature_row.get(key)
        if value is None:
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed):
            return parsed
    return default


def spread_bps(feature_row: pd.Series, quote: dict | None, settings: Settings) -> float | None:
    spread = first_finite(feature_row, "scalping_spread_bps")
    if spread is not None:
        return max(0.0, spread)
    spread_pct = first_finite(feature_row, "scalping_spread_pct", "orderbook_spread")
    if spread_pct is not None:
        return max(0.0, spread_pct * 10_000)
    bid = quote_float(quote, "bid_price", "bp", "bid")
    ask = quote_float(quote, "ask_price", "ap", "ask")
    if bid is None or ask is None or ask < bid:
        return None
    mid = (bid + ask) / 2
    if mid <= 0:
        return None
    return ((ask - bid) / mid) * 10_000


def quote_imbalance(feature_row: pd.Series, quote: dict | None) -> float | None:
    imbalance = first_finite(feature_row, "scalping_quote_imbalance", "quote_imbalance")
    if imbalance is not None:
        return imbalance
    bid_size = quote_float(quote, "bid_size", "bs", allow_zero=True)
    ask_size = quote_float(quote, "ask_size", "as", allow_zero=True)
    if bid_size is None or ask_size is None or bid_size + ask_size <= 0:
        return None
    return (bid_size - ask_size) / (bid_size + ask_size)


def quote_float(quote: dict[str, Any] | None, *keys: str, allow_zero: bool = False) -> float | None:
    if not quote:
        return None
    for key in keys:
        value = quote.get(key)
        if value is None or value == "":
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed) and (parsed > 0 or (allow_zero and parsed == 0)):
            return parsed
    return None
