from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from app.config import Settings
from app.data.market_data import (
    MarketDataClient,
    StaleMarketDataError,
    calculate_request_start,
    parse_timeframe_duration,
    validate_bars_are_fresh,
)


@pytest.mark.parametrize(
    ("timeframe", "expected"),
    [
        ("1Min", timedelta(minutes=1)),
        ("5Min", timedelta(minutes=5)),
        ("1Hour", timedelta(hours=1)),
        ("4Hour", timedelta(hours=4)),
        ("1Day", timedelta(days=1)),
    ],
)
def test_parse_timeframe_duration(timeframe, expected):
    assert parse_timeframe_duration(timeframe) == expected


def test_one_minute_lookback_start_is_not_years_in_the_past():
    end = datetime(2026, 5, 27, 12, 0, tzinfo=UTC)

    start = calculate_request_start(end, "1Min", 1500)

    lookback = end - start
    assert timedelta(minutes=1500) < lookback <= timedelta(minutes=1600)
    assert lookback < timedelta(days=2)


def test_stale_alpaca_bars_raise_clear_error():
    now = datetime(2026, 5, 27, 12, 0, tzinfo=UTC)
    bars = pd.DataFrame(
        {
            "timestamp": [now - timedelta(minutes=11)],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [1.0],
        }
    )

    with pytest.raises(StaleMarketDataError) as exc_info:
        validate_bars_are_fresh(bars, "1Min", now=now)

    message = str(exc_info.value)
    assert "latest_timestamp=2026-05-27T11:49:00+00:00" in message
    assert "current_utc_time=2026-05-27T12:00:00+00:00" in message
    assert "timeframe=1Min" in message
    assert "bars_fetched=1" in message


def test_stale_2022_market_data_is_rejected():
    now = datetime(2026, 5, 27, 12, 0, tzinfo=UTC)
    bars = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2022-12-25T16:32:00Z")],
            "open": [16811.06],
            "high": [16816.69],
            "low": [16811.03],
            "close": [16811.28],
            "volume": [1.851083],
        }
    )

    with pytest.raises(StaleMarketDataError):
        validate_bars_are_fresh(bars, "1Min", now=now)


@pytest.mark.anyio
async def test_alpaca_fetch_uses_timeframe_window_and_returns_desired_tail(monkeypatch):
    from app.data import market_data

    fixed_now = datetime(2026, 5, 27, 12, 0, tzinfo=UTC)
    rows = []
    for index in range(1600):
        timestamp = fixed_now - timedelta(minutes=1599 - index)
        rows.append(
            {
                "t": timestamp.isoformat(),
                "o": 100.0 + index,
                "h": 101.0 + index,
                "l": 99.0 + index,
                "c": 100.5 + index,
                "v": 1.0,
            }
        )

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now if tz else fixed_now.replace(tzinfo=None)

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"bars": {"BTC/USD": rows}}

    class FakeAsyncClient:
        params = None

        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def get(self, url, headers, params):
            self.__class__.params = params
            return FakeResponse()

    monkeypatch.setattr(market_data, "datetime", FixedDateTime)
    monkeypatch.setattr(market_data.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(market_data, "_log_bars_fetched", lambda **kwargs: None)

    settings = Settings(
        _env_file=None,
        alpaca_api_key="key",
        alpaca_secret_key="secret",
        timeframe="1Min",
        lookback_bars=1500,
    )

    bars = await MarketDataClient(settings).fetch_bars("BTC/USD")

    assert len(bars) == 1500
    assert bars["timestamp"].iloc[0].isoformat() == "2026-05-26T11:01:00+00:00"
    assert bars["timestamp"].iloc[-1].isoformat() == "2026-05-27T12:00:00+00:00"
    assert FakeAsyncClient.params["limit"] == 1600
    assert FakeAsyncClient.params["start"] == "2026-05-26T09:20:00+00:00"
