from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import ALLOWED_SYMBOL, Settings, get_settings
from app.data.market_data import parse_timeframe_duration
from app.db.database import Base, connect_args_for_database_url, run_sqlite_schema_migrations
from scripts.auto_research_train import (
    _env_text,
    _json_safe,
    evaluate_environment_safety,
    load_effective_env,
)


DEFAULT_TIMEFRAMES = ("1Min", "5Min", "15Min")
DEFAULT_DAYS = 30
DEFAULT_LIMIT_PER_REQUEST = 10_000
MAX_LIMIT_PER_REQUEST = 10_000
REPORT_NAME = "backfill_market_data_report.json"
LATEST_REPORT_NAME = "backfill_market_data_report_latest.json"
TRUE_ACTIONS = {"insert", "update", "skip_duplicate"}


@dataclass(frozen=True)
class BackfillWindow:
    start: datetime
    end: datetime


@dataclass(frozen=True)
class ValidBar:
    symbol: str
    timeframe: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class AlpacaHistoricalProvider:
    def __init__(self, settings: Settings, *, sleep_seconds: float = 0.20, max_retries: int = 3) -> None:
        self.settings = settings
        self.sleep_seconds = sleep_seconds
        self.max_retries = max_retries

    @property
    def headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.settings.alpaca_api_key,
            "APCA-API-SECRET-KEY": self.settings.alpaca_secret_key,
        }

    async def fetch_bars(
        self,
        symbol: str,
        *,
        timeframe: str,
        start: datetime,
        end: datetime,
        limit_per_request: int,
    ) -> pd.DataFrame:
        if not (self.settings.alpaca_api_key and self.settings.alpaca_secret_key):
            raise RuntimeError("alpaca_credentials_missing")
        rows: list[dict[str, Any]] = []
        next_page_token: str | None = None
        request_count = 0
        async with httpx.AsyncClient(timeout=30) as client:
            while True:
                params: dict[str, Any] = {
                    "symbols": symbol,
                    "timeframe": timeframe,
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "limit": limit_per_request,
                }
                if next_page_token:
                    params["page_token"] = next_page_token
                payload = await self._request_page(client, params)
                request_count += 1
                rows.extend(payload.get("bars", {}).get(symbol, []) or [])
                next_page_token = payload.get("next_page_token")
                if not next_page_token:
                    break
                await asyncio.sleep(self.sleep_seconds)
        frame = _alpaca_rows_to_frame(rows)
        frame.attrs["provider_request_count"] = request_count
        return frame

    async def _request_page(self, client: httpx.AsyncClient, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.settings.alpaca_data_base_url.rstrip('/')}/v1beta3/crypto/us/bars"
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = await client.get(url, headers=self.headers, params=params)
                if response.status_code in {429, 500, 502, 503, 504} and attempt < self.max_retries:
                    await asyncio.sleep(min(2.0, self.sleep_seconds * attempt * 2))
                    continue
                response.raise_for_status()
                return response.json()
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    await asyncio.sleep(min(2.0, self.sleep_seconds * attempt * 2))
                    continue
                break
        raise RuntimeError(f"alpaca_provider_error: {last_error}") from last_error


async def run_backfill(
    *,
    symbol: str = ALLOWED_SYMBOL,
    timeframes: list[str] | tuple[str, ...] = DEFAULT_TIMEFRAMES,
    days: int | None = None,
    start: str | datetime | None = None,
    end: str | datetime | None = None,
    limit_per_request: int = DEFAULT_LIMIT_PER_REQUEST,
    dry_run: bool = True,
    source: str = "alpaca",
    database: str | Path = "data/trading.db",
    env: dict[str, str] | None = None,
    env_path: Path | None = None,
    provider: Any | None = None,
    now: datetime | None = None,
    write_reports: bool = True,
) -> tuple[dict[str, Any], int]:
    mode = "dry-run" if dry_run else "run"
    current_time = _utc_datetime(now or datetime.now(UTC))
    requested_window = _resolve_window(start=start, end=end, days=days, now=current_time)
    requested_timeframes = [str(timeframe) for timeframe in timeframes]
    effective_env = load_effective_env(env=env, env_path=env_path or ROOT / ".env")
    safety = evaluate_environment_safety(effective_env, inspection_only_dry_run=False)
    safety_flags = dict(safety.flags)
    fatal_reasons = list(safety.fatal_reasons)

    if symbol != ALLOWED_SYMBOL:
        fatal_reasons.append("symbol_argument_not_btc_usd")
    invalid_timeframes = [timeframe for timeframe in requested_timeframes if timeframe not in DEFAULT_TIMEFRAMES]
    if invalid_timeframes:
        fatal_reasons.append("unsupported_timeframe")
        safety_flags["unsupported_timeframes"] = invalid_timeframes
    if source != "alpaca":
        fatal_reasons.append("unsupported_source")
    if limit_per_request <= 0 or limit_per_request > MAX_LIMIT_PER_REQUEST:
        fatal_reasons.append("limit_per_request_out_of_range")

    safety_flags["fatal_reasons"] = fatal_reasons
    safety_flags["safety_gate_passed"] = not fatal_reasons
    safety_flags["backfill_trading_required"] = False
    safety_flags["backfill_places_orders"] = False

    report = _base_report(
        generated_at=current_time,
        mode=mode,
        symbol=symbol,
        timeframes=requested_timeframes,
        window=requested_window,
        source=source,
        safety_flags=safety_flags,
    )

    if fatal_reasons:
        report["final_recommendation"] = "validation_failed"
        report["brutally_honest_notes"] = _brutally_honest_notes(report)
        if write_reports:
            write_report_files(report, log_dir=_log_dir_from_env(effective_env))
        return report, 1

    engine: Engine | None = None
    settings: Settings | None = None
    try:
        if provider is None:
            settings = get_settings()
            provider = AlpacaHistoricalProvider(settings)
        database_url = _database_url(database)
        if dry_run and not _sqlite_database_exists(database_url):
            engine = None
        else:
            if not dry_run:
                _ensure_sqlite_parent(database_url)
            engine = create_engine(database_url, connect_args=connect_args_for_database_url(database_url))
            if not dry_run:
                Base.metadata.create_all(bind=engine)
                run_sqlite_schema_migrations(engine)

        for timeframe in requested_timeframes:
            summary = await _backfill_timeframe(
                provider,
                engine=engine,
                symbol=symbol,
                timeframe=timeframe,
                window=requested_window,
                limit_per_request=limit_per_request,
                source=source,
                dry_run=dry_run,
                now=current_time,
            )
            report["per_timeframe_summary"][timeframe] = summary
    except Exception as exc:
        report["provider_errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        if engine is not None:
            engine.dispose()

    _finish_report(report)
    if write_reports:
        write_report_files(
            report,
            log_dir=_log_dir_from_env(effective_env if settings is None else {**effective_env, "LOG_DIR": settings.log_dir}),
        )
    exit_code = 0 if report["final_recommendation"] in {"backfill_ready", "collect_more_data"} else 1
    return report, exit_code


async def _backfill_timeframe(
    provider: Any,
    *,
    engine: Engine | None,
    symbol: str,
    timeframe: str,
    window: BackfillWindow,
    limit_per_request: int,
    source: str,
    dry_run: bool,
    now: datetime,
) -> dict[str, Any]:
    before_count = _count_rows(engine, symbol=symbol, timeframe=timeframe)
    summary = {
        "fetched_rows": 0,
        "inserted_rows": 0,
        "updated_rows": 0,
        "skipped_duplicate_rows": 0,
        "invalid_rows": 0,
        "would_insert_rows": 0,
        "would_update_rows": 0,
        "first_timestamp": None,
        "latest_timestamp": None,
        "provider_errors": [],
        "provider_warnings": [],
        "validation_errors": {},
        "before_row_count": before_count,
        "after_row_count": before_count,
    }
    try:
        effective_end = _complete_request_end(timeframe=timeframe, requested_end=window.end, now=now)
        summary["effective_fetch_end"] = effective_end.isoformat()
        bars = await provider.fetch_bars(
            symbol,
            timeframe=timeframe,
            start=window.start,
            end=effective_end,
            limit_per_request=limit_per_request,
        )
    except Exception as exc:
        summary["provider_errors"].append(f"{type(exc).__name__}: {exc}")
        return summary

    frame = _coerce_provider_frame(bars)
    summary["fetched_rows"] = int(len(frame))
    if frame.empty:
        summary["provider_errors"].append("provider_returned_no_data")
        return summary

    valid_bars, validation_errors, invalid_rows = _validate_bars(
        frame,
        symbol=symbol,
        timeframe=timeframe,
        now=now,
    )
    summary["invalid_rows"] = invalid_rows
    summary["validation_errors"] = validation_errors
    if valid_bars:
        valid_bars = _deduplicate_valid_bars(valid_bars, summary)
        ordered = sorted(valid_bars, key=lambda bar: bar.timestamp)
        summary["first_timestamp"] = ordered[0].timestamp.isoformat()
        summary["latest_timestamp"] = ordered[-1].timestamp.isoformat()
        summary["provider_warnings"].extend(_provider_window_warnings(ordered, timeframe=timeframe, window=window))
        storage = _store_or_preview_bars(
            ordered,
            engine=engine,
            source=source,
            dry_run=dry_run,
            requested_window=window,
            now=now,
        )
        summary.update(storage)
    else:
        summary["provider_errors"].append("provider_returned_no_valid_bars")
    summary["after_row_count"] = _count_rows(engine, symbol=symbol, timeframe=timeframe)
    return summary


def _store_or_preview_bars(
    bars: list[ValidBar],
    *,
    engine: Engine | None,
    source: str,
    dry_run: bool,
    requested_window: BackfillWindow,
    now: datetime,
) -> dict[str, int]:
    counts = {
        "inserted_rows": 0,
        "updated_rows": 0,
        "skipped_duplicate_rows": 0,
        "would_insert_rows": 0,
        "would_update_rows": 0,
    }
    if engine is None:
        counts["would_insert_rows"] = len(bars)
        return counts

    with engine.begin() as connection:
        columns = _table_columns(connection)
        for bar in bars:
            action, existing = _storage_action(connection, columns, bar)
            if action not in TRUE_ACTIONS:
                continue
            if action == "insert":
                if dry_run:
                    counts["would_insert_rows"] += 1
                else:
                    _insert_bar(connection, columns, bar, source=source, requested_window=requested_window, now=now)
                    counts["inserted_rows"] += 1
            elif action == "update":
                if dry_run:
                    counts["would_update_rows"] += 1
                else:
                    _update_bar(connection, columns, bar, existing, source=source, requested_window=requested_window, now=now)
                    counts["updated_rows"] += 1
            else:
                counts["skipped_duplicate_rows"] += 1
    return counts


def _storage_action(connection: Any, columns: set[str], bar: ValidBar) -> tuple[str, dict[str, Any]]:
    existing = _existing_bar(connection, columns, bar)
    if existing is None:
        return "insert", {}
    existing_is_backfilled = bool(existing.get("backfilled"))
    if not existing_is_backfilled:
        return "skip_duplicate", existing
    if _existing_bar_matches(existing, bar) and existing.get("source") and existing.get("source_used"):
        return "skip_duplicate", existing
    return "update", existing


def _insert_bar(
    connection: Any,
    columns: set[str],
    bar: ValidBar,
    *,
    source: str,
    requested_window: BackfillWindow,
    now: datetime,
) -> None:
    values = _bar_storage_values(bar, source=source, requested_window=requested_window, now=now)
    insert_columns = [column for column in values if column in columns]
    sql = (
        f"INSERT INTO collected_market_data ({', '.join(insert_columns)}) "
        f"VALUES ({', '.join(f':{column}' for column in insert_columns)})"
    )
    connection.execute(text(sql), {column: values[column] for column in insert_columns})


def _update_bar(
    connection: Any,
    columns: set[str],
    bar: ValidBar,
    existing: dict[str, Any],
    *,
    source: str,
    requested_window: BackfillWindow,
    now: datetime,
) -> None:
    values = _bar_storage_values(bar, source=source, requested_window=requested_window, now=now)
    update_columns = [
        column
        for column in (
            "open",
            "high",
            "low",
            "close",
            "volume",
            "collected_at",
            "source",
            "source_used",
            "backfilled",
            "provider_metadata",
        )
        if column in columns
    ]
    sql = (
        f"UPDATE collected_market_data SET {', '.join(f'{column} = :{column}' for column in update_columns)} "
        "WHERE id = :id"
    )
    params = {column: values[column] for column in update_columns}
    params["id"] = existing["id"]
    connection.execute(text(sql), params)


def _existing_bar(connection: Any, columns: set[str], bar: ValidBar) -> dict[str, Any] | None:
    if "collected_market_data" not in inspect(connection).get_table_names():
        return None
    selected = [
        "id",
        "open",
        "high",
        "low",
        "close",
        "volume",
        *(column for column in ("source", "source_used", "backfilled") if column in columns),
    ]
    row = connection.execute(
        text(
            f"SELECT {', '.join(selected)} FROM collected_market_data "
            "WHERE symbol = :symbol AND timeframe = :timeframe AND timestamp = :timestamp "
            "ORDER BY id LIMIT 1"
        ),
        {"symbol": bar.symbol, "timeframe": bar.timeframe, "timestamp": _sqlite_datetime(bar.timestamp)},
    ).mappings().first()
    return dict(row) if row is not None else None


def _existing_bar_matches(existing: dict[str, Any], bar: ValidBar) -> bool:
    return all(
        math.isclose(float(existing[column]), float(getattr(bar, column)), rel_tol=0.0, abs_tol=1e-12)
        for column in ("open", "high", "low", "close", "volume")
    )


def _bar_storage_values(
    bar: ValidBar,
    *,
    source: str,
    requested_window: BackfillWindow,
    now: datetime,
) -> dict[str, Any]:
    metadata = {
        "provider": source,
        "source_used": f"{source}_historical_backfill",
        "backfilled_at": now.isoformat(),
        "requested_start": requested_window.start.isoformat(),
        "requested_end": requested_window.end.isoformat(),
        "timeframe": bar.timeframe,
        "synthetic_data_used": False,
    }
    return {
        "collected_at": _sqlite_datetime(now),
        "symbol": bar.symbol,
        "timeframe": bar.timeframe,
        "timestamp": _sqlite_datetime(bar.timestamp),
        "open": bar.open,
        "high": bar.high,
        "low": bar.low,
        "close": bar.close,
        "volume": bar.volume,
        "source": source,
        "source_used": f"{source}_historical_backfill",
        "backfilled": True,
        "provider_metadata": json.dumps(metadata, sort_keys=True),
    }


def _validate_bars(
    frame: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str,
    now: datetime,
) -> tuple[list[ValidBar], dict[str, int], int]:
    valid: list[ValidBar] = []
    errors: dict[str, int] = {}
    invalid_rows = 0
    for _, row in frame.iterrows():
        bar, row_errors = _validate_bar(row, symbol=symbol, timeframe=timeframe, now=now)
        if row_errors:
            invalid_rows += 1
            for reason in row_errors:
                errors[reason] = errors.get(reason, 0) + 1
            continue
        if bar is not None:
            valid.append(bar)
    return valid, errors, invalid_rows


def _validate_bar(row: pd.Series, *, symbol: str, timeframe: str, now: datetime) -> tuple[ValidBar | None, list[str]]:
    errors: list[str] = []
    if symbol != ALLOWED_SYMBOL:
        errors.append("symbol_not_btc_usd")
    if timeframe not in DEFAULT_TIMEFRAMES:
        errors.append("unsupported_timeframe")

    timestamp = _coerce_timestamp(row.get("timestamp"))
    if timestamp is None:
        errors.append("timestamp_invalid")
    else:
        if timestamp > now:
            errors.append("timestamp_in_future")
        if timestamp + parse_timeframe_duration(timeframe) > now:
            errors.append("candle_incomplete")

    values: dict[str, float] = {}
    for column in ("open", "high", "low", "close", "volume"):
        value = _finite_float(row.get(column))
        if value is None:
            errors.append(f"{column}_invalid")
        else:
            values[column] = value

    if all(column in values for column in ("open", "high", "low", "close")):
        if values["open"] <= 0 or values["high"] <= 0 or values["low"] <= 0 or values["close"] <= 0:
            errors.append("ohlc_non_positive")
        if values["high"] < max(values["open"], values["close"]):
            errors.append("high_below_open_or_close")
        if values["low"] > min(values["open"], values["close"]):
            errors.append("low_above_open_or_close")
    if "volume" in values and values["volume"] < 0:
        errors.append("volume_negative")

    if errors or timestamp is None:
        return None, errors
    return (
        ValidBar(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=timestamp,
            open=values["open"],
            high=values["high"],
            low=values["low"],
            close=values["close"],
            volume=values["volume"],
        ),
        [],
    )


def _deduplicate_valid_bars(valid_bars: list[ValidBar], summary: dict[str, Any]) -> list[ValidBar]:
    by_timestamp: dict[datetime, ValidBar] = {}
    duplicate_count = 0
    for bar in valid_bars:
        if bar.timestamp in by_timestamp:
            duplicate_count += 1
        by_timestamp[bar.timestamp] = bar
    if duplicate_count:
        validation_errors = dict(summary.get("validation_errors") or {})
        validation_errors["duplicate_provider_timestamp"] = duplicate_count
        summary["validation_errors"] = validation_errors
        summary["invalid_rows"] = int(summary.get("invalid_rows", 0)) + duplicate_count
    return list(by_timestamp.values())


def _provider_window_warnings(ordered: list[ValidBar], *, timeframe: str, window: BackfillWindow) -> list[str]:
    warnings: list[str] = []
    duration = parse_timeframe_duration(timeframe)
    latest = ordered[-1].timestamp
    if latest + duration < window.end - duration:
        warnings.append("provider_latest_timestamp_before_requested_end")
    expected_rows = max(1, int((window.end - window.start).total_seconds() // duration.total_seconds()))
    if len(ordered) < max(1, int(expected_rows * 0.50)):
        warnings.append(f"provider_rows_below_half_of_expected_window:{len(ordered)}/{expected_rows}")
    return warnings


def _coerce_provider_frame(bars: Any) -> pd.DataFrame:
    if isinstance(bars, pd.DataFrame):
        frame = bars.copy()
    else:
        frame = pd.DataFrame(bars)
    if frame.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    rename_map = {column: str(column).lower() for column in frame.columns}
    frame = frame.rename(columns=rename_map)
    return frame


def _alpaca_rows_to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    return pd.DataFrame(rows).rename(
        columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
    )


def _count_rows(engine: Engine | None, *, symbol: str, timeframe: str) -> int:
    if engine is None:
        return 0
    with engine.connect() as connection:
        if "collected_market_data" not in inspect(connection).get_table_names():
            return 0
        return int(
            connection.execute(
                text(
                    "SELECT COUNT(*) FROM collected_market_data "
                    "WHERE symbol = :symbol AND timeframe = :timeframe"
                ),
                {"symbol": symbol, "timeframe": timeframe},
            ).scalar()
            or 0
        )


def _table_columns(connection: Any) -> set[str]:
    inspector = inspect(connection)
    if "collected_market_data" not in inspector.get_table_names():
        return set()
    return {column["name"] for column in inspector.get_columns("collected_market_data")}


def _base_report(
    *,
    generated_at: datetime,
    mode: str,
    symbol: str,
    timeframes: list[str],
    window: BackfillWindow,
    source: str,
    safety_flags: dict[str, Any],
) -> dict[str, Any]:
    return {
        "generated_at": generated_at.isoformat(),
        "mode": mode,
        "symbol": symbol,
        "timeframes": timeframes,
        "requested_start": window.start.isoformat(),
        "requested_end": window.end.isoformat(),
        "source": source,
        "safety_flags": safety_flags,
        "per_timeframe_summary": {},
        "provider_errors": [],
        "total_inserted_rows": 0,
        "total_updated_rows": 0,
        "total_invalid_rows": 0,
        "synthetic_data_used": False,
        "orders_placed": 0,
        "trading_remained_disabled": (
            safety_flags.get("trading_enabled") is False
            and safety_flags.get("auto_trade_enabled") is False
        ),
        "final_recommendation": "collect_more_data",
        "brutally_honest_notes": [],
    }


def _finish_report(report: dict[str, Any]) -> None:
    summaries = report.get("per_timeframe_summary") or {}
    report["total_inserted_rows"] = sum(int(summary.get("inserted_rows", 0) or 0) for summary in summaries.values())
    report["total_updated_rows"] = sum(int(summary.get("updated_rows", 0) or 0) for summary in summaries.values())
    report["total_invalid_rows"] = sum(int(summary.get("invalid_rows", 0) or 0) for summary in summaries.values())
    report["synthetic_data_used"] = False
    report["orders_placed"] = 0
    report["trading_remained_disabled"] = (
        (report.get("safety_flags") or {}).get("trading_enabled") is False
        and (report.get("safety_flags") or {}).get("auto_trade_enabled") is False
    )
    provider_errors = list(report.get("provider_errors") or [])
    for summary in summaries.values():
        provider_errors.extend(summary.get("provider_errors") or [])
    if provider_errors:
        report["final_recommendation"] = "provider_error"
    elif report["total_invalid_rows"] > 0:
        report["final_recommendation"] = "validation_failed"
    elif _has_backfill_candidates(report):
        report["final_recommendation"] = "backfill_ready"
    else:
        report["final_recommendation"] = "collect_more_data"
    report["brutally_honest_notes"] = _brutally_honest_notes(report)


def _has_backfill_candidates(report: dict[str, Any]) -> bool:
    summaries = report.get("per_timeframe_summary") or {}
    for summary in summaries.values():
        if int(summary.get("inserted_rows", 0) or 0) > 0:
            return True
        if int(summary.get("updated_rows", 0) or 0) > 0:
            return True
        if int(summary.get("would_insert_rows", 0) or 0) > 0:
            return True
        if int(summary.get("would_update_rows", 0) or 0) > 0:
            return True
        if int(summary.get("skipped_duplicate_rows", 0) or 0) > 0 and int(summary.get("fetched_rows", 0) or 0) > 0:
            return True
    return False


def _brutally_honest_notes(report: dict[str, Any]) -> list[str]:
    notes = [
        "Historical backfill is market-data storage only; it never submits, stages, or simulates orders.",
        "No trading flags were changed by this script, and .env is never written.",
        "Historical data can speed up research readiness, but it does not prove profitability.",
        "Paper-forward eligibility from later research is still not permission to enable auto trading.",
    ]
    if report.get("mode") == "dry-run":
        notes.append("Dry-run mode fetched and validated provider data but inserted zero rows.")
    if report.get("provider_errors"):
        notes.append("Provider errors occurred; treat the backfill as incomplete.")
    for timeframe, summary in (report.get("per_timeframe_summary") or {}).items():
        errors = summary.get("provider_errors") or []
        if errors:
            notes.append(f"{timeframe} provider errors: {', '.join(errors)}.")
        warnings = summary.get("provider_warnings") or []
        if warnings:
            notes.append(f"{timeframe} provider warnings: {', '.join(warnings)}.")
        validation_errors = summary.get("validation_errors") or {}
        if validation_errors:
            notes.append(f"{timeframe} rejected invalid bars: {validation_errors}.")
    if not report.get("trading_remained_disabled"):
        notes.append("Trading flags were already enabled in the inspected environment, so backfill is unsafe.")
    return notes


def write_report_files(report: dict[str, Any], *, log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_json_safe(report), indent=2, sort_keys=True, allow_nan=False)
    (log_dir / REPORT_NAME).write_text(payload + "\n", encoding="utf-8")
    (log_dir / LATEST_REPORT_NAME).write_text(payload + "\n", encoding="utf-8")


def _resolve_window(
    *,
    start: str | datetime | None,
    end: str | datetime | None,
    days: int | None,
    now: datetime,
) -> BackfillWindow:
    parsed_start = _utc_datetime(start) if start is not None else None
    parsed_end = _utc_datetime(end) if end is not None else None
    if days is not None and days <= 0:
        raise ValueError("days must be positive")
    if parsed_start is None and parsed_end is None and days is None:
        days = DEFAULT_DAYS
    if parsed_start is not None and parsed_end is not None:
        window_start = parsed_start
        window_end = parsed_end
    elif parsed_start is not None:
        window_start = parsed_start
        window_end = parsed_start + timedelta(days=days) if days is not None else now
    elif parsed_end is not None:
        window_end = parsed_end
        window_start = parsed_end - timedelta(days=days or DEFAULT_DAYS)
    else:
        window_end = now
        window_start = window_end - timedelta(days=days or DEFAULT_DAYS)
    window_end = min(window_end, now)
    if window_start >= window_end:
        raise ValueError("backfill start must be before end")
    return BackfillWindow(start=window_start, end=window_end)


def _complete_request_end(*, timeframe: str, requested_end: datetime, now: datetime) -> datetime:
    duration_seconds = int(parse_timeframe_duration(timeframe).total_seconds())
    current_epoch = int(now.timestamp())
    complete_boundary = datetime.fromtimestamp(
        (current_epoch // duration_seconds) * duration_seconds,
        tz=UTC,
    )
    return min(requested_end, complete_boundary)


def _coerce_timestamp(value: Any) -> datetime | None:
    try:
        if value is None or pd.isna(value):
            return None
        return _utc_datetime(value)
    except Exception:
        return None


def _utc_datetime(value: Any) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.to_pydatetime()


def _sqlite_datetime(value: Any) -> str:
    return _utc_datetime(value).strftime("%Y-%m-%d %H:%M:%S.%f")


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _database_url(database: str | Path) -> str:
    raw = str(database)
    if raw.startswith("sqlite"):
        return raw
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    return f"sqlite:///{path}"


def _sqlite_database_exists(database_url: str) -> bool:
    if not database_url.startswith("sqlite:///"):
        return True
    raw_path = database_url.removeprefix("sqlite:///")
    if raw_path in {":memory:", ""}:
        return True
    return Path(raw_path).exists()


def _ensure_sqlite_parent(database_url: str) -> None:
    if not database_url.startswith("sqlite:///"):
        return
    raw_path = database_url.removeprefix("sqlite:///")
    if raw_path in {":memory:", ""}:
        return
    Path(raw_path).parent.mkdir(parents=True, exist_ok=True)


def _log_dir_from_env(effective_env: dict[str, str]) -> Path:
    log_dir = Path(_env_text(effective_env, "LOG_DIR", "logs"))
    return log_dir if log_dir.is_absolute() else ROOT / log_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safely backfill real historical BTC/USD OHLCV data into SQLite.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Fetch and validate only. This is the default.")
    mode.add_argument("--run", action="store_true", help="Write valid historical bars to SQLite.")
    parser.add_argument("--symbol", default=ALLOWED_SYMBOL, help="Only BTC/USD is supported.")
    parser.add_argument("--timeframes", nargs="+", default=list(DEFAULT_TIMEFRAMES), help="Timeframes to backfill.")
    parser.add_argument("--days", type=int, default=None, help="Number of days to backfill.")
    parser.add_argument("--start", default=None, help="UTC start date or ISO timestamp.")
    parser.add_argument("--end", default=None, help="UTC end date or ISO timestamp.")
    parser.add_argument(
        "--limit-per-request",
        type=int,
        default=DEFAULT_LIMIT_PER_REQUEST,
        help="Maximum Alpaca bars per paginated request.",
    )
    parser.add_argument("--source", default="alpaca", choices=["alpaca"], help="Historical market data source.")
    parser.add_argument("--database", default="data/trading.db", help="SQLite database path or URL.")
    parser.add_argument("--json", action="store_true", help="Print the full JSON report.")
    return parser.parse_args()


def _format_text_summary(report: dict[str, Any]) -> str:
    lines = [
        "BTC/USD historical backfill report",
        f"mode: {report.get('mode')}",
        f"source: {report.get('source')}",
        f"requested_start: {report.get('requested_start')}",
        f"requested_end: {report.get('requested_end')}",
        f"total inserted rows: {report.get('total_inserted_rows')}",
        f"total invalid rows: {report.get('total_invalid_rows')}",
        f"final recommendation: {report.get('final_recommendation')}",
        f"trading remained disabled: {report.get('trading_remained_disabled')}",
        f"orders placed: {report.get('orders_placed')}",
    ]
    for timeframe, summary in (report.get("per_timeframe_summary") or {}).items():
        lines.append(
            f"{timeframe}: fetched={summary.get('fetched_rows')} inserted={summary.get('inserted_rows')} "
            f"would_insert={summary.get('would_insert_rows')} invalid={summary.get('invalid_rows')} "
            f"duplicates={summary.get('skipped_duplicate_rows')}"
        )
    return "\n".join(lines)


async def main() -> int:
    args = _parse_args()
    report, exit_code = await run_backfill(
        symbol=args.symbol,
        timeframes=args.timeframes,
        days=args.days,
        start=args.start,
        end=args.end,
        limit_per_request=args.limit_per_request,
        dry_run=not args.run,
        source=args.source,
        database=args.database,
    )
    if args.json:
        print(json.dumps(_json_safe(report), indent=2, sort_keys=True, allow_nan=False))
    else:
        print(_format_text_summary(report))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
