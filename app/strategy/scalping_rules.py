from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import math
from typing import Any

import pandas as pd

from app.config import Settings, get_settings
from app.utils.time import utc_now


@dataclass(frozen=True)
class ScalpingMarketState:
    latest_price: float
    spread_bps: float | None
    quote_imbalance: float | None
    market_data_fresh: bool


class ScalpingRules:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def market_state(
        self,
        feature_row: pd.Series,
        *,
        quote: dict | None = None,
        now: datetime | None = None,
    ) -> ScalpingMarketState:
        latest_price = _quote_mid_price(quote) or _finite_float(feature_row.get("close")) or 0.0
        return ScalpingMarketState(
            latest_price=latest_price,
            spread_bps=_spread_bps(feature_row, quote=quote),
            quote_imbalance=_quote_imbalance(feature_row, quote=quote),
            market_data_fresh=self.market_data_is_fresh(feature_row, now=now),
        )

    def market_data_is_fresh(self, feature_row: pd.Series, *, now: datetime | None = None) -> bool:
        timestamp = _utc_timestamp(feature_row.get("timestamp"))
        if timestamp is None:
            return False
        age_seconds = ((now or utc_now()) - timestamp).total_seconds()
        return -5 <= age_seconds <= self.settings.scalping_max_data_age_seconds

    def allow_entry_market(self, market: ScalpingMarketState) -> tuple[bool, str]:
        if market.latest_price <= 0:
            return False, "invalid_price"
        if not market.market_data_fresh:
            return False, "stale_market_data"
        if market.spread_bps is None:
            return False, "spread_unavailable"
        if market.spread_bps > self.settings.max_spread_bps:
            return False, "spread_too_wide"
        if self.settings.limit_price_offset_bps > self.settings.max_slippage_bps:
            return False, "slippage_too_high"
        if market.quote_imbalance is None:
            return False, "quote_imbalance_unavailable"
        if market.quote_imbalance < self.settings.min_quote_imbalance:
            return False, "quote_imbalance_too_weak"
        return True, "allowed"

    def entry_confirmed(self, feature_row: pd.Series) -> bool:
        momentum = _first_finite(
            feature_row,
            "scalping_momentum_3",
            "scalping_log_return_3",
            "log_return_3",
        )
        breakout = _first_finite(feature_row, "scalping_high_breakout_5")
        return (
            momentum is not None
            and momentum >= self.settings.scalping_min_momentum_pct
        ) or (breakout is not None and breakout > 0)

    def quote_strongly_unfavorable(self, market: ScalpingMarketState) -> bool:
        return (
            market.spread_bps is not None
            and market.spread_bps > self.settings.max_spread_bps
        ) or (
            market.quote_imbalance is not None
            and market.quote_imbalance < self.settings.scalping_quote_imbalance_exit
        )

    def held_for_minimum(self, position_opened_at: datetime | None, *, now: datetime | None = None) -> bool:
        if self.settings.scalping_min_hold_seconds <= 0 or position_opened_at is None:
            return True
        opened_at = _utc_timestamp(position_opened_at)
        if opened_at is None:
            return True
        return (now or utc_now()) - opened_at >= timedelta(seconds=self.settings.scalping_min_hold_seconds)


def _spread_bps(feature_row: pd.Series, *, quote: dict | None) -> float | None:
    spread_bps = _finite_float(feature_row.get("scalping_spread_bps"))
    if spread_bps is not None:
        return max(0.0, spread_bps)
    spread_pct = _first_finite(feature_row, "scalping_spread_pct", "orderbook_spread")
    if spread_pct is not None:
        return max(0.0, spread_pct * 10_000)
    bid = _quote_float(quote, "bid_price", "bp", "bid")
    ask = _quote_float(quote, "ask_price", "ap", "ask")
    if bid is None or ask is None or ask < bid:
        return None
    mid = (bid + ask) / 2
    return ((ask - bid) / mid) * 10_000 if mid > 0 else None


def _quote_imbalance(feature_row: pd.Series, *, quote: dict | None) -> float | None:
    imbalance = _first_finite(feature_row, "scalping_quote_imbalance", "quote_imbalance")
    if imbalance is not None:
        return imbalance
    bid_size = _quote_float(quote, "bid_size", "bs", allow_zero=True)
    ask_size = _quote_float(quote, "ask_size", "as", allow_zero=True)
    if bid_size is None or ask_size is None or bid_size + ask_size <= 0:
        return None
    return (bid_size - ask_size) / (bid_size + ask_size)


def _quote_mid_price(quote: dict | None) -> float | None:
    bid = _quote_float(quote, "bid_price", "bp", "bid")
    ask = _quote_float(quote, "ask_price", "ap", "ask")
    if bid is not None and ask is not None:
        return (bid + ask) / 2
    return _quote_float(quote, "mid_price", "mid")


def _first_finite(feature_row: pd.Series, *keys: str) -> float | None:
    for key in keys:
        value = _finite_float(feature_row.get(key))
        if value is not None:
            return value
    return None


def _quote_float(quote: dict[str, Any] | None, *keys: str, allow_zero: bool = False) -> float | None:
    if not quote:
        return None
    for key in keys:
        value = _finite_float(quote.get(key))
        if value is not None and (value > 0 or (allow_zero and value == 0)):
            return value
    return None


def _finite_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _utc_timestamp(value: Any) -> datetime | None:
    if value is None or pd.isna(value):
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.to_pydatetime().astimezone(UTC)
