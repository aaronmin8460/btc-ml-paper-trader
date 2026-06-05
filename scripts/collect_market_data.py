from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import ALLOWED_SYMBOL, Settings, get_settings
from app.data.market_data import MarketDataClient
from app.db.database import SessionLocal, init_db
from app.db.models import CollectedMarketData


DEFAULT_TIMEFRAMES = ("1Min", "5Min", "15Min")


async def collect_market_data(
    settings: Settings | None = None,
    *,
    timeframes: Iterable[str] = DEFAULT_TIMEFRAMES,
    limit: int | None = None,
    client: MarketDataClient | None = None,
    session_factory: Any | None = None,
) -> dict[str, Any]:
    settings = settings or get_settings()
    _assert_collector_safety(settings)
    if session_factory is None:
        init_db()
        session_factory = SessionLocal
    client = client or MarketDataClient(settings)
    desired_limit = limit or max(100, min(settings.lookback_bars, 5000))
    collected_rows = 0
    timeframe_reports: list[dict[str, Any]] = []

    quote = await _fetch_quote(client, settings.symbol)
    for timeframe in timeframes:
        bars = await client.fetch_bars(settings.symbol, timeframe=timeframe, limit=desired_limit, force_refresh=True)
        rows_written = _write_bars(
            bars,
            settings=settings,
            timeframe=str(timeframe),
            quote=quote,
            session_factory=session_factory,
        )
        collected_rows += rows_written
        timeframe_reports.append(
            {
                "timeframe": str(timeframe),
                "bars_fetched": int(len(bars)),
                "rows_written": rows_written,
                "first_timestamp": _timestamp_text(bars["timestamp"].iloc[0]) if not bars.empty else None,
                "latest_timestamp": _timestamp_text(bars["timestamp"].iloc[-1]) if not bars.empty else None,
            }
        )

    return {
        "symbol": settings.symbol,
        "paper_trading_only": settings.paper_trading_only,
        "trading_enabled_required": False,
        "auto_trade_enabled_required": False,
        "trading_enabled": settings.trading_enabled,
        "auto_trade_enabled": settings.auto_trade_enabled,
        "orders_placed": 0,
        "timeframes": timeframe_reports,
        "total_rows_written": collected_rows,
        "quote_available": _quote_bid_ask(quote)[0] is not None and _quote_bid_ask(quote)[1] is not None,
        "note": "Market-data collection only; this script never submits orders or enables trading.",
    }


def _assert_collector_safety(settings: Settings) -> None:
    if settings.symbol != ALLOWED_SYMBOL:
        raise ValueError("Market data collection is BTC/USD-only.")
    if not settings.paper_trading_only:
        raise ValueError("Market data collection must remain paper-only.")


async def _fetch_quote(client: MarketDataClient, symbol: str) -> dict[str, Any]:
    try:
        quote = await client.fetch_latest_quote(symbol, force_refresh=True)
    except Exception:
        return {}
    return dict(quote or {})


def _write_bars(
    bars: pd.DataFrame,
    *,
    settings: Settings,
    timeframe: str,
    quote: dict[str, Any],
    session_factory: Any,
) -> int:
    if bars.empty:
        return 0
    ordered = bars.sort_values("timestamp").reset_index(drop=True)
    latest_timestamp = pd.Timestamp(ordered["timestamp"].iloc[-1])
    rows_written = 0
    with session_factory() as db:
        for _, row in ordered.iterrows():
            timestamp = _utc_datetime(row["timestamp"])
            quote_fields = _quote_fields_for_timestamp(timestamp, latest_timestamp, quote)
            existing = (
                db.query(CollectedMarketData)
                .filter(
                    CollectedMarketData.symbol == settings.symbol,
                    CollectedMarketData.timeframe == timeframe,
                    CollectedMarketData.timestamp == timestamp,
                )
                .first()
            )
            values = {
                "symbol": settings.symbol,
                "timeframe": timeframe,
                "timestamp": timestamp,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
                "collected_at": datetime.now(UTC),
                **quote_fields,
            }
            if existing is None:
                db.add(CollectedMarketData(**values))
                rows_written += 1
            else:
                for key, value in values.items():
                    setattr(existing, key, value)
        db.commit()
    return rows_written


def _quote_fields_for_timestamp(
    timestamp: datetime,
    latest_timestamp: pd.Timestamp,
    quote: dict[str, Any],
) -> dict[str, float | None]:
    latest = _utc_datetime(latest_timestamp)
    bid, ask = _quote_bid_ask(quote)
    bid_size = _quote_float(quote, "bid_size", "bs")
    ask_size = _quote_float(quote, "ask_size", "as")
    if timestamp != latest or bid is None or ask is None:
        return {
            "bid": None,
            "ask": None,
            "bid_size": None,
            "ask_size": None,
            "spread_bps": None,
            "quote_imbalance": None,
        }
    mid = (bid + ask) / 2
    spread_bps = ((ask - bid) / mid * 10_000) if mid > 0 and ask >= bid else None
    imbalance = None
    if bid_size is not None and ask_size is not None and bid_size + ask_size > 0:
        imbalance = (bid_size - ask_size) / (bid_size + ask_size)
    return {
        "bid": bid,
        "ask": ask,
        "bid_size": bid_size,
        "ask_size": ask_size,
        "spread_bps": spread_bps,
        "quote_imbalance": imbalance,
    }


def _quote_bid_ask(quote: dict[str, Any]) -> tuple[float | None, float | None]:
    return _quote_float(quote, "bid_price", "bp", "bid"), _quote_float(quote, "ask_price", "ap", "ask")


def _quote_float(quote: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = quote.get(key)
        if value is None or value == "":
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed) and parsed >= 0:
            return parsed
    return None


def _utc_datetime(value: Any) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.to_pydatetime()


def _timestamp_text(value: Any) -> str:
    return _utc_datetime(value).isoformat()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect BTC/USD market bars and latest quote into SQLite.")
    parser.add_argument("--timeframes", nargs="+", default=list(DEFAULT_TIMEFRAMES), help="Research/data timeframes.")
    parser.add_argument("--limit", type=int, default=None, help="Bars per timeframe.")
    return parser.parse_args()


async def main() -> None:
    args = _parse_args()
    report = await collect_market_data(timeframes=args.timeframes, limit=args.limit)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
