import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd

from app.broker.execution_guard import assert_btc_only
from app.config import ALLOWED_SYMBOL, Settings, get_settings
from app.monitoring.logger import get_logger
from app.utils.rate_limiter import get_alpaca_rate_limiter


MAX_ALPACA_BARS_PER_REQUEST = 10_000
TIMEFRAME_PATTERN = re.compile(r"^(?P<count>\d+)\s*(?P<unit>min|hour|day)s?$", re.IGNORECASE)


class StaleMarketDataError(RuntimeError):
    pass


def parse_timeframe_duration(timeframe: str) -> timedelta:
    normalized = timeframe.strip().strip("\"'")
    match = TIMEFRAME_PATTERN.fullmatch(normalized)
    if not match:
        raise ValueError(f"Unsupported timeframe: {timeframe!r}")

    count = int(match.group("count"))
    if count <= 0:
        raise ValueError(f"Unsupported timeframe: {timeframe!r}")

    unit = match.group("unit").lower()
    if unit == "min":
        return timedelta(minutes=count)
    if unit == "hour":
        return timedelta(hours=count)
    if unit == "day":
        return timedelta(days=count)
    raise ValueError(f"Unsupported timeframe: {timeframe!r}")


def stale_threshold_for_timeframe(timeframe: str) -> timedelta:
    duration = parse_timeframe_duration(timeframe)
    if duration < timedelta(hours=1):
        return max(timedelta(minutes=10), duration * 2)
    if duration < timedelta(days=1):
        return max(timedelta(hours=3), duration * 2)
    return timedelta(days=2)


def request_limit_with_buffer(desired_limit: int) -> int:
    if desired_limit <= 0:
        raise ValueError("limit must be positive")
    buffer_bars = max(5, min(100, int(desired_limit * 0.10)))
    return min(desired_limit + buffer_bars, MAX_ALPACA_BARS_PER_REQUEST)


def calculate_request_start(end: datetime, timeframe: str, desired_limit: int) -> datetime:
    return end - (parse_timeframe_duration(timeframe) * request_limit_with_buffer(desired_limit))


def validate_bars_are_fresh(bars: pd.DataFrame, timeframe: str, *, now: datetime | None = None) -> None:
    current_time = pd.Timestamp(now or datetime.now(UTC))
    if current_time.tzinfo is None:
        current_time = current_time.tz_localize("UTC")
    else:
        current_time = current_time.tz_convert("UTC")

    if bars.empty:
        raise StaleMarketDataError(
            "No Alpaca bars fetched; "
            f"latest_timestamp=None current_utc_time={current_time.isoformat()} "
            f"timeframe={timeframe} bars_fetched=0"
        )

    latest_timestamp = pd.Timestamp(bars["timestamp"].iloc[-1])
    if latest_timestamp.tzinfo is None:
        latest_timestamp = latest_timestamp.tz_localize("UTC")
    else:
        latest_timestamp = latest_timestamp.tz_convert("UTC")

    if current_time - latest_timestamp > stale_threshold_for_timeframe(timeframe):
        raise StaleMarketDataError(
            "Latest Alpaca bar is stale; "
            f"latest_timestamp={latest_timestamp.isoformat()} "
            f"current_utc_time={current_time.isoformat()} "
            f"timeframe={timeframe} bars_fetched={len(bars)}"
        )


def _timestamp_to_iso(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


def _log_bars_fetched(
    *,
    symbol: str,
    timeframe: str,
    requested_start: datetime | None,
    requested_end: datetime | None,
    desired_limit: int,
    bars: pd.DataFrame,
) -> None:
    payload = {
        "symbol": symbol,
        "timeframe": timeframe,
        "requested_start": requested_start.isoformat() if requested_start else None,
        "requested_end": requested_end.isoformat() if requested_end else None,
        "desired_limit": desired_limit,
        "actual_bars_fetched": len(bars),
        "first_timestamp": _timestamp_to_iso(bars["timestamp"].iloc[0]) if not bars.empty else None,
        "latest_timestamp": _timestamp_to_iso(bars["timestamp"].iloc[-1]) if not bars.empty else None,
    }
    try:
        from app.monitoring.logger import get_logger

        get_logger().event("market_data_bars_fetched", **payload)
    except Exception:
        print(json.dumps({"event_type": "market_data_bars_fetched", **payload}, separators=(",", ":")))


class MarketDataClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.logger = get_logger()
        self._bars_cache: dict[tuple[str, str, int], tuple[datetime, pd.DataFrame]] = {}
        self._quote_cache: dict[str, tuple[datetime, dict[str, Any]]] = {}

    @property
    def headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.settings.alpaca_api_key,
            "APCA-API-SECRET-KEY": self.settings.alpaca_secret_key,
        }

    async def fetch_bars(
        self,
        symbol: str = ALLOWED_SYMBOL,
        *,
        timeframe: str | None = None,
        limit: int | None = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        assert_btc_only(symbol, context="market_data_fetch_bars")
        timeframe = timeframe or self.settings.timeframe
        desired_limit = self.settings.lookback_bars if limit is None else limit
        cache_key = (symbol, timeframe, desired_limit)
        cached = self._get_bars_cache(cache_key, force_refresh=force_refresh)
        if cached is not None:
            return cached
        budget_cached = self._get_budget_stale_bars_cache(cache_key, force_refresh=force_refresh)
        if budget_cached is not None:
            return budget_cached

        request_limit = request_limit_with_buffer(desired_limit)
        if self.settings.alpaca_api_key and self.settings.alpaca_secret_key:
            end = datetime.now(UTC)
            start = calculate_request_start(end, timeframe, desired_limit)
            params = {
                "symbols": symbol,
                "timeframe": timeframe,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "limit": request_limit,
            }
            async with httpx.AsyncClient(timeout=30) as client:
                await self._wait_for_alpaca(endpoint="crypto_bars")
                response = await client.get(
                    f"{self.settings.alpaca_data_base_url}/v1beta3/crypto/us/bars",
                    headers=self.headers,
                    params=params,
                )
                response.raise_for_status()
                data = response.json().get("bars", {}).get(symbol, [])
                fetched_bars = self._bars_to_frame(data)
                _log_bars_fetched(
                    symbol=symbol,
                    timeframe=timeframe,
                    requested_start=start,
                    requested_end=end,
                    desired_limit=desired_limit,
                    bars=fetched_bars,
                )
                validate_bars_are_fresh(fetched_bars, timeframe)
                bars = fetched_bars.tail(desired_limit).reset_index(drop=True)
                self._set_bars_cache(cache_key, bars)
                return bars
        bars = self.synthetic_btc_bars(limit=desired_limit)
        _log_bars_fetched(
            symbol=symbol,
            timeframe=timeframe,
            requested_start=None,
            requested_end=None,
            desired_limit=desired_limit,
            bars=bars,
        )
        self._set_bars_cache(cache_key, bars)
        return bars

    async def fetch_latest_quote(self, symbol: str = ALLOWED_SYMBOL, *, force_refresh: bool = False) -> dict[str, Any]:
        assert_btc_only(symbol, context="market_data_fetch_latest_quote")
        cached = self._get_quote_cache(symbol, force_refresh=force_refresh)
        if cached is not None:
            return cached
        if not (self.settings.alpaca_api_key and self.settings.alpaca_secret_key):
            quote = {"bid_price": None, "ask_price": None, "bid_size": None, "ask_size": None}
            self._set_quote_cache(symbol, quote)
            return quote
        async with httpx.AsyncClient(timeout=15) as client:
            await self._wait_for_alpaca(endpoint="latest_quote")
            response = await client.get(
                f"{self.settings.alpaca_data_base_url}/v1beta3/crypto/us/latest/quotes",
                headers=self.headers,
                params={"symbols": symbol},
            )
            response.raise_for_status()
            quote = response.json().get("quotes", {}).get(symbol, {})
            self._set_quote_cache(symbol, quote)
            return quote

    def load_csv(self, path: str | Path, symbol: str = ALLOWED_SYMBOL) -> pd.DataFrame:
        assert_btc_only(symbol, context="market_data_load_csv")
        df = pd.read_csv(path)
        return normalize_ohlcv(df)

    def _bars_to_frame(self, rows: list[dict[str, Any]]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        df = pd.DataFrame(rows).rename(
            columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
        )
        return normalize_ohlcv(df)

    async def _wait_for_alpaca(self, *, endpoint: str) -> None:
        await get_alpaca_rate_limiter(self.settings).acquire(endpoint=endpoint)

    def _get_bars_cache(
        self,
        cache_key: tuple[str, str, int],
        *,
        force_refresh: bool,
    ) -> pd.DataFrame | None:
        symbol, timeframe, desired_limit = cache_key
        cached = self._bars_cache.get(cache_key)
        age = self._cache_age_seconds(cached[0]) if cached else None
        cache_hit = (
            cached is not None
            and not force_refresh
            and self.settings.market_bars_cache_seconds > 0
            and age is not None
            and age <= self.settings.market_bars_cache_seconds
        )
        self._log_cache_event(
            endpoint="bars",
            symbol=symbol,
            cache_hit=cache_hit,
            cache_age_seconds=age,
            timeframe=timeframe,
            desired_limit=desired_limit,
        )
        if cache_hit:
            return cached[1].copy()
        return None

    def _set_bars_cache(self, cache_key: tuple[str, str, int], bars: pd.DataFrame) -> None:
        self._bars_cache[cache_key] = (datetime.now(UTC), bars.copy())

    def _get_budget_stale_bars_cache(
        self,
        cache_key: tuple[str, str, int],
        *,
        force_refresh: bool,
    ) -> pd.DataFrame | None:
        if force_refresh:
            return None
        cached = self._bars_cache.get(cache_key)
        if cached is None:
            return None
        limiter = get_alpaca_rate_limiter(self.settings)
        if not limiter.soft_budget_reached():
            return None
        symbol, timeframe, desired_limit = cache_key
        age = self._cache_age_seconds(cached[0])
        self._log_cache_event(
            endpoint="bars",
            symbol=symbol,
            cache_hit=True,
            cache_age_seconds=age,
            timeframe=timeframe,
            desired_limit=desired_limit,
            api_budget_status="soft_limit_stale_cache",
        )
        return cached[1].copy()

    def bars_cache_age_seconds(
        self,
        *,
        symbol: str = ALLOWED_SYMBOL,
        timeframe: str | None = None,
        limit: int | None = None,
    ) -> float | None:
        timeframe = timeframe or self.settings.timeframe
        desired_limit = self.settings.lookback_bars if limit is None else limit
        cached = self._bars_cache.get((symbol, timeframe, desired_limit))
        return self._cache_age_seconds(cached[0]) if cached else None

    def _get_quote_cache(self, symbol: str, *, force_refresh: bool) -> dict[str, Any] | None:
        cached = self._quote_cache.get(symbol)
        age = self._cache_age_seconds(cached[0]) if cached else None
        cache_hit = (
            cached is not None
            and not force_refresh
            and self.settings.quote_cache_seconds > 0
            and age is not None
            and age <= self.settings.quote_cache_seconds
        )
        self._log_cache_event(endpoint="quote", symbol=symbol, cache_hit=cache_hit, cache_age_seconds=age)
        if cache_hit:
            return dict(cached[1])
        return None

    def _set_quote_cache(self, symbol: str, quote: dict[str, Any]) -> None:
        self._quote_cache[symbol] = (datetime.now(UTC), dict(quote))

    @staticmethod
    def _cache_age_seconds(cached_at: datetime) -> float:
        return max(0.0, (datetime.now(UTC) - cached_at).total_seconds())

    def _log_cache_event(self, *, endpoint: str, symbol: str, cache_hit: bool, cache_age_seconds: float | None, **extra: Any) -> None:
        try:
            self.logger.event(
                "market_data_cache",
                endpoint=endpoint,
                symbol=symbol,
                cache_hit=cache_hit,
                cache_age_seconds=cache_age_seconds,
                **extra,
            )
        except Exception:
            pass

    @staticmethod
    def synthetic_btc_bars(limit: int = 1200) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        ts = pd.date_range(end=pd.Timestamp.now(tz="UTC").floor("15min"), periods=limit, freq="15min")
        returns = rng.normal(0.00015, 0.006, size=limit)
        close = 65000 * np.exp(np.cumsum(returns))
        spread = close * rng.uniform(0.0005, 0.004, size=limit)
        open_ = np.roll(close, 1)
        open_[0] = close[0] * (1 - returns[0])
        high = np.maximum(open_, close) + spread
        low = np.minimum(open_, close) - spread
        volume = rng.lognormal(mean=2.4, sigma=0.4, size=limit)
        return pd.DataFrame(
            {"timestamp": ts, "open": open_, "high": high, "low": low, "close": close, "volume": volume}
        )


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename_map = {c: c.lower() for c in out.columns}
    out = out.rename(columns=rename_map)
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out[required].dropna().sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
