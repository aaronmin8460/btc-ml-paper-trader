from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable, Iterable


PROFILE_A_MAX_SPREAD_BPS = 4.0
PROFILE_A_MIN_QUOTE_IMBALANCE = 0.05
PROFILE_A_SCALPING_MAX_DATA_AGE_SECONDS = 120.0

SIGNAL_TABLES = ("signals", "decisions", "strategy_decisions", "trading_signals")
MARKET_DATA_TABLES = ("collected_market_data", "market_data")

SIGNAL_TIME_COLUMNS = ("created_at", "timestamp", "ts", "time", "updated_at")
MARKET_DATA_TIME_COLUMNS = ("collected_at", "created_at", "timestamp", "ts", "time")
BAR_TIMESTAMP_COLUMNS = ("bar_timestamp", "latest_bar_timestamp", "market_bar_timestamp", "timestamp")
REASON_COLUMNS = ("reason", "decision_reason", "block_reason")
SPREAD_COLUMNS = ("spread_bps", "orderbook_spread_bps", "scalping_spread_bps")
QUOTE_IMBALANCE_COLUMNS = ("quote_imbalance", "imbalance", "scalping_quote_imbalance")
BAR_AGE_COLUMNS = ("bar_age_seconds", "latest_bar_age_seconds", "data_age_seconds")
JSON_COLUMNS = (
    "raw_response",
    "metadata",
    "metadata_json",
    "payload",
    "payload_json",
    "decision",
    "decision_json",
    "event",
    "event_json",
)

SPREAD_BPS_BUCKETS = ("<=2", "2-4", "4-6", "6-10", ">10")
QUOTE_IMBALANCE_BUCKETS = ("<-0.10", "-0.10-0.00", "0.00-0.05", "0.05-0.10", ">0.10")
BAR_AGE_BUCKETS = ("<=60", "60-120", "120-180", "180-240", ">240")
MISSING = "(missing)"


class WarningCollector:
    def __init__(self) -> None:
        self.items: list[str] = []
        self._seen: set[str] = set()

    def add(self, message: str) -> None:
        if message not in self._seen:
            self._seen.add(message)
            self.items.append(message)


def build_market_quality_report(
    db_path: str | Path,
    *,
    hours: float = 48,
    now: datetime | None = None,
    max_spread_bps: float = PROFILE_A_MAX_SPREAD_BPS,
    min_quote_imbalance: float = PROFILE_A_MIN_QUOTE_IMBALANCE,
    scalping_max_data_age_seconds: float = PROFILE_A_SCALPING_MAX_DATA_AGE_SECONDS,
) -> dict[str, Any]:
    current_time = _ensure_utc(now or datetime.now(UTC))
    window_start = current_time - timedelta(hours=max(0.0, float(hours)))
    path = Path(db_path)
    warnings = WarningCollector()
    tables: dict[str, list[str]] = {}
    signals: list[dict[str, Any]] = []
    market_data: list[dict[str, Any]] = []
    sources = {"signals": None, "market_data": None}

    if not path.exists():
        warnings.add(f"Database file not found: {path}")
    else:
        try:
            with sqlite3.connect(path) as connection:
                connection.row_factory = sqlite3.Row
                tables = _inspect_tables(connection)
                if not tables:
                    warnings.add(f"No tables found in SQLite database: {path}")

                signal_table = _resolve_table(tables, SIGNAL_TABLES, "signal", warnings)
                market_data_table = _resolve_table(tables, MARKET_DATA_TABLES, "market data", warnings)

                if signal_table:
                    sources["signals"] = signal_table
                    rows = _load_recent_rows(
                        connection,
                        signal_table,
                        tables[signal_table],
                        window_start=window_start,
                        timestamp_candidates=SIGNAL_TIME_COLUMNS,
                        purpose="signal",
                        warnings=warnings,
                    )
                    signals = [_normalize_signal(row, tables[signal_table]) for row in rows]
                    _warn_for_signal_gaps(signal_table, tables[signal_table], signals, warnings)

                if market_data_table:
                    sources["market_data"] = market_data_table
                    rows = _load_recent_rows(
                        connection,
                        market_data_table,
                        tables[market_data_table],
                        window_start=window_start,
                        timestamp_candidates=MARKET_DATA_TIME_COLUMNS,
                        purpose="market data",
                        warnings=warnings,
                    )
                    market_data = [_normalize_market_data(row, tables[market_data_table]) for row in rows]
                    _warn_for_market_data_gaps(market_data_table, tables[market_data_table], market_data, warnings)
        except sqlite3.DatabaseError as exc:
            warnings.add(f"Could not read SQLite database {path}: {exc}")

    observations = signals + market_data
    thresholds = {
        "MAX_SPREAD_BPS": float(max_spread_bps),
        "MIN_QUOTE_IMBALANCE": float(min_quote_imbalance),
        "SCALPING_MAX_DATA_AGE_SECONDS": float(scalping_max_data_age_seconds),
    }
    metrics = _build_metrics(signals, observations, thresholds)
    breakdowns = _build_breakdowns(observations)
    return {
        "generated_at": current_time.isoformat(),
        "db_path": str(path),
        "window": {
            "hours": float(hours),
            "start": window_start.isoformat(),
            "end": current_time.isoformat(),
        },
        "profile": "BTC/USD Profile A market quality diagnostics",
        "thresholds": thresholds,
        "sources": sources,
        "source_counts": {
            "signals": len(signals) if sources["signals"] else None,
            "market_data": len(market_data) if sources["market_data"] else None,
            "market_quality_rows": len(observations),
        },
        "available_tables": {table: columns for table, columns in sorted(tables.items())},
        "warnings": warnings.items,
        "metrics": metrics,
        "breakdowns": breakdowns,
        "note": (
            "Local read-only market quality diagnostic. It does not loosen trading thresholds, add "
            "strategies, enable live trading, change position size, require AWS, submit orders, or "
            "modify runtime settings."
        ),
    }


def save_market_quality_report(report: dict[str, Any], output_dir: str | Path = "reports") -> Path:
    generated_at = _parse_datetime(report.get("generated_at")) or datetime.now(UTC)
    output_path = Path(output_dir) / f"market_quality_{generated_at.strftime('%Y%m%d_%H%M%S')}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report["output_path"] = str(output_path)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def format_market_quality_report(report: dict[str, Any]) -> str:
    metrics = report.get("metrics") or {}
    thresholds = report.get("thresholds") or {}
    lines = [
        "BTC/USD Profile A market quality diagnostics",
        "",
        f"database: {report['db_path']}",
        f"window: last {_format_number(report['window']['hours'])} hours "
        f"({report['window']['start']} to {report['window']['end']})",
        f"json_report: {report.get('output_path', 'not written')}",
        "",
        report["note"],
        "",
        "Thresholds",
        _format_table(
            ("Setting", "Value"),
            [
                ("MAX_SPREAD_BPS", _format_number(thresholds.get("MAX_SPREAD_BPS"))),
                ("MIN_QUOTE_IMBALANCE", _format_number(thresholds.get("MIN_QUOTE_IMBALANCE"))),
                (
                    "SCALPING_MAX_DATA_AGE_SECONDS",
                    _format_seconds(thresholds.get("SCALPING_MAX_DATA_AGE_SECONDS")),
                ),
            ],
        ),
        "",
    ]

    warnings = report.get("warnings") or []
    if warnings:
        lines.append("Warnings")
        lines.extend(f"  - {warning}" for warning in warnings)
        lines.append("")

    lines.append("Summary")
    lines.append(
        _format_table(
            ("Metric", "Value"),
            [
                ("total signals", _format_count(metrics.get("total_signals"))),
                ("market quality rows", _format_count(metrics.get("market_quality_rows"))),
                ("spread_too_wide count", _format_count(metrics.get("spread_too_wide_count"))),
                ("spread_too_wide percent", _format_pct(metrics.get("spread_too_wide_pct"))),
                ("stale_market_data count", _format_count(metrics.get("stale_market_data_count"))),
                ("stale_market_data percent", _format_pct(metrics.get("stale_market_data_pct"))),
                ("average spread_bps", _format_number(metrics.get("average_spread_bps"))),
                ("median spread_bps", _format_number(metrics.get("median_spread_bps"))),
                ("p75 spread_bps", _format_number(metrics.get("p75_spread_bps"))),
                ("p90 spread_bps", _format_number(metrics.get("p90_spread_bps"))),
                ("p95 spread_bps", _format_number(metrics.get("p95_spread_bps"))),
                ("min spread_bps", _format_number(metrics.get("min_spread_bps"))),
                ("max spread_bps", _format_number(metrics.get("max_spread_bps"))),
                ("average quote_imbalance", _format_number(metrics.get("average_quote_imbalance"))),
                ("median quote_imbalance", _format_number(metrics.get("median_quote_imbalance"))),
                (
                    "rows with spread_bps <= MAX_SPREAD_BPS",
                    _format_pct(metrics.get("pct_spread_bps_lte_max")),
                ),
                (
                    "rows with quote_imbalance >= MIN_QUOTE_IMBALANCE",
                    _format_pct(metrics.get("pct_quote_imbalance_gte_min")),
                ),
                ("rows passing both spread and quote filters", _format_pct(metrics.get("pct_passing_both_filters"))),
                ("average bar_age_seconds", _format_seconds(metrics.get("average_bar_age_seconds"))),
                ("p90 bar_age_seconds", _format_seconds(metrics.get("p90_bar_age_seconds"))),
                (
                    "rows passing SCALPING_MAX_DATA_AGE_SECONDS",
                    _format_pct(metrics.get("pct_bar_age_lte_max")),
                ),
            ],
        )
    )

    breakdowns = report.get("breakdowns") or {}
    labels = {
        "spread_bps_bucket": "Spread Bps Bucket",
        "quote_imbalance_bucket": "Quote Imbalance Bucket",
        "bar_age_seconds_bucket": "Bar Age Seconds Bucket",
    }
    for key, label in labels.items():
        lines.extend(["", label])
        rows = breakdowns.get(key) or []
        if rows:
            lines.append(
                _format_table(
                    ("Value", "Count", "Pct"),
                    [
                        (str(row["value"]), _format_count(row["count"]), _format_pct(row.get("pct")))
                        for row in rows
                    ],
                )
            )
        else:
            lines.append("  n/a")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a local BTC/USD Profile A market quality diagnostics report.")
    parser.add_argument("--db", required=True, help="Path to the local SQLite trading database.")
    parser.add_argument("--hours", type=float, default=48, help="Lookback window in hours. Defaults to 48.")
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Directory for the JSON report. Defaults to reports/.",
    )
    parser.add_argument(
        "--max-spread-bps",
        type=float,
        default=PROFILE_A_MAX_SPREAD_BPS,
        help="Spread threshold for diagnostics. Defaults to Profile A MAX_SPREAD_BPS=4.",
    )
    parser.add_argument(
        "--min-quote-imbalance",
        type=float,
        default=PROFILE_A_MIN_QUOTE_IMBALANCE,
        help="Quote imbalance threshold for diagnostics. Defaults to Profile A MIN_QUOTE_IMBALANCE=0.05.",
    )
    parser.add_argument(
        "--max-data-age-seconds",
        type=float,
        default=PROFILE_A_SCALPING_MAX_DATA_AGE_SECONDS,
        help="Bar age threshold for diagnostics. Defaults to SCALPING_MAX_DATA_AGE_SECONDS=120.",
    )
    args = parser.parse_args(argv)
    if args.hours <= 0:
        parser.error("--hours must be greater than 0")
    if args.max_data_age_seconds < 0:
        parser.error("--max-data-age-seconds must be non-negative")

    report = build_market_quality_report(
        args.db,
        hours=args.hours,
        max_spread_bps=args.max_spread_bps,
        min_quote_imbalance=args.min_quote_imbalance,
        scalping_max_data_age_seconds=args.max_data_age_seconds,
    )
    save_market_quality_report(report, args.output_dir)
    print(format_market_quality_report(report))
    return 0


def spread_bps_bucket(value: Any) -> str | None:
    parsed = _safe_float(value)
    if parsed is None:
        return None
    if parsed <= 2:
        return "<=2"
    if parsed <= 4:
        return "2-4"
    if parsed <= 6:
        return "4-6"
    if parsed <= 10:
        return "6-10"
    return ">10"


def quote_imbalance_bucket(value: Any) -> str | None:
    parsed = _safe_float(value)
    if parsed is None:
        return None
    if parsed < -0.10:
        return "<-0.10"
    if parsed < 0.00:
        return "-0.10-0.00"
    if parsed < 0.05:
        return "0.00-0.05"
    if parsed <= 0.10:
        return "0.05-0.10"
    return ">0.10"


def bar_age_seconds_bucket(value: Any) -> str | None:
    parsed = _safe_float(value)
    if parsed is None:
        return None
    if parsed <= 60:
        return "<=60"
    if parsed <= 120:
        return "60-120"
    if parsed <= 180:
        return "120-180"
    if parsed <= 240:
        return "180-240"
    return ">240"


def _build_metrics(
    signals: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    signal_reasons = [_normalize_text(signal.get("reason")).lower() for signal in signals]
    total_signals = len(signals)
    spread_too_wide_count = sum(1 for reason in signal_reasons if reason == "spread_too_wide")
    stale_market_data_count = sum(1 for reason in signal_reasons if reason == "stale_market_data")
    spread_values = _finite_values(observation.get("spread_bps") for observation in observations)
    quote_imbalance_values = _finite_values(observation.get("quote_imbalance") for observation in observations)
    bar_age_values = _finite_values(observation.get("bar_age_seconds") for observation in observations)
    both_filter_rows = [
        observation
        for observation in observations
        if _safe_float(observation.get("spread_bps")) is not None
        and _safe_float(observation.get("quote_imbalance")) is not None
    ]
    return {
        "total_signals": total_signals,
        "market_quality_rows": len(observations),
        "spread_sample_size": len(spread_values),
        "quote_imbalance_sample_size": len(quote_imbalance_values),
        "bar_age_sample_size": len(bar_age_values),
        "spread_too_wide_count": spread_too_wide_count,
        "spread_too_wide_pct": _ratio(spread_too_wide_count, total_signals),
        "stale_market_data_count": stale_market_data_count,
        "stale_market_data_pct": _ratio(stale_market_data_count, total_signals),
        "average_spread_bps": mean(spread_values) if spread_values else None,
        "median_spread_bps": median(spread_values) if spread_values else None,
        "p75_spread_bps": _percentile(spread_values, 0.75),
        "p90_spread_bps": _percentile(spread_values, 0.90),
        "p95_spread_bps": _percentile(spread_values, 0.95),
        "min_spread_bps": min(spread_values) if spread_values else None,
        "max_spread_bps": max(spread_values) if spread_values else None,
        "average_quote_imbalance": mean(quote_imbalance_values) if quote_imbalance_values else None,
        "median_quote_imbalance": median(quote_imbalance_values) if quote_imbalance_values else None,
        "pct_spread_bps_lte_max": _ratio(
            sum(1 for value in spread_values if value <= thresholds["MAX_SPREAD_BPS"]),
            len(spread_values),
        ),
        "pct_quote_imbalance_gte_min": _ratio(
            sum(1 for value in quote_imbalance_values if value >= thresholds["MIN_QUOTE_IMBALANCE"]),
            len(quote_imbalance_values),
        ),
        "pct_passing_both_filters": _ratio(
            sum(
                1
                for row in both_filter_rows
                if _safe_float(row.get("spread_bps")) <= thresholds["MAX_SPREAD_BPS"]
                and _safe_float(row.get("quote_imbalance")) >= thresholds["MIN_QUOTE_IMBALANCE"]
            ),
            len(both_filter_rows),
        ),
        "average_bar_age_seconds": mean(bar_age_values) if bar_age_values else None,
        "p90_bar_age_seconds": _percentile(bar_age_values, 0.90),
        "pct_bar_age_lte_max": _ratio(
            sum(1 for value in bar_age_values if value <= thresholds["SCALPING_MAX_DATA_AGE_SECONDS"]),
            len(bar_age_values),
        ),
    }


def _build_breakdowns(observations: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        "spread_bps_bucket": _bucket_breakdown(
            [observation.get("spread_bps") for observation in observations],
            spread_bps_bucket,
            SPREAD_BPS_BUCKETS,
        ),
        "quote_imbalance_bucket": _bucket_breakdown(
            [observation.get("quote_imbalance") for observation in observations],
            quote_imbalance_bucket,
            QUOTE_IMBALANCE_BUCKETS,
        ),
        "bar_age_seconds_bucket": _bucket_breakdown(
            [observation.get("bar_age_seconds") for observation in observations],
            bar_age_seconds_bucket,
            BAR_AGE_BUCKETS,
        ),
    }


def _inspect_tables(connection: sqlite3.Connection) -> dict[str, list[str]]:
    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    tables: dict[str, list[str]] = {}
    for row in rows:
        table = str(row["name"])
        columns = connection.execute(f"PRAGMA table_info({_quote_identifier(table)})").fetchall()
        tables[table] = [str(column["name"]) for column in columns]
    return tables


def _resolve_table(
    tables: dict[str, list[str]],
    candidates: tuple[str, ...],
    label: str,
    warnings: WarningCollector,
) -> str | None:
    lookup = {name.lower(): name for name in tables}
    for candidate in candidates:
        table = lookup.get(candidate.lower())
        if table is not None:
            return table
    warnings.add(f"No {label} table found. Looked for: {', '.join(candidates)}.")
    return None


def _load_recent_rows(
    connection: sqlite3.Connection,
    table: str,
    columns: list[str],
    *,
    window_start: datetime,
    timestamp_candidates: tuple[str, ...],
    purpose: str,
    warnings: WarningCollector,
) -> list[dict[str, Any]]:
    selected = ", ".join(_quote_identifier(column) for column in columns)
    rows = [dict(row) for row in connection.execute(f"SELECT {selected} FROM {_quote_identifier(table)}").fetchall()]
    timestamp_column = _first_column(columns, timestamp_candidates)
    if timestamp_column is None:
        warnings.add(f"{table} has no timestamp column for {purpose} lookback filtering; using all rows.")
        return rows

    recent: list[dict[str, Any]] = []
    unparseable = 0
    for row in rows:
        timestamp = _parse_datetime(row.get(timestamp_column))
        if timestamp is None:
            unparseable += 1
            continue
        if timestamp >= window_start:
            recent.append(row)
    if unparseable:
        warnings.add(
            f"{table}.{timestamp_column} had {unparseable} unparseable timestamp row(s); excluded from the lookback."
        )
    return recent


def _normalize_signal(row: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    return {
        "source": "signals",
        "reason": _normalize_optional_text(
            _extract_value(row, columns, REASON_COLUMNS, json_paths=("reason", "block_reason"), json_keys=REASON_COLUMNS)
        ),
        "spread_bps": _safe_float(
            _extract_value(row, columns, SPREAD_COLUMNS, json_paths=("spread_bps",), json_keys=SPREAD_COLUMNS)
        ),
        "quote_imbalance": _safe_float(
            _extract_value(
                row,
                columns,
                QUOTE_IMBALANCE_COLUMNS,
                json_paths=("quote_imbalance",),
                json_keys=QUOTE_IMBALANCE_COLUMNS,
            )
        ),
        "bar_age_seconds": _safe_float(
            _extract_value(
                row,
                columns,
                BAR_AGE_COLUMNS,
                json_paths=("bar_age_seconds", "latest_bar_age_seconds"),
                json_keys=BAR_AGE_COLUMNS,
            )
        ),
    }


def _normalize_market_data(row: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    bar_age_seconds = _safe_float(_first_column_value(row, columns, BAR_AGE_COLUMNS))
    if bar_age_seconds is None:
        collected_at = _parse_datetime(_first_column_value(row, columns, ("collected_at", "created_at")))
        bar_timestamp = _parse_datetime(_first_column_value(row, columns, BAR_TIMESTAMP_COLUMNS))
        if collected_at is not None and bar_timestamp is not None:
            bar_age_seconds = max(0.0, (collected_at - bar_timestamp).total_seconds())
    return {
        "source": "market_data",
        "reason": None,
        "spread_bps": _safe_float(_first_column_value(row, columns, SPREAD_COLUMNS)),
        "quote_imbalance": _safe_float(_first_column_value(row, columns, QUOTE_IMBALANCE_COLUMNS)),
        "bar_age_seconds": bar_age_seconds,
    }


def _warn_for_signal_gaps(
    table: str,
    columns: list[str],
    signals: list[dict[str, Any]],
    warnings: WarningCollector,
) -> None:
    if not _has_any_column(columns, REASON_COLUMNS) and not any(_has_value(signal.get("reason")) for signal in signals):
        warnings.add(f"{table} has no signal reason column; spread_too_wide and stale_market_data counts unavailable.")
    if not _has_any_column(columns, SPREAD_COLUMNS) and not any(signal.get("spread_bps") is not None for signal in signals):
        warnings.add(f"{table} has no spread_bps column or payload field; spread metrics may rely on market data only.")
    if not _has_any_column(columns, QUOTE_IMBALANCE_COLUMNS) and not any(
        signal.get("quote_imbalance") is not None for signal in signals
    ):
        warnings.add(
            f"{table} has no quote_imbalance column or payload field; quote imbalance metrics may rely on market data only."
        )
    if not _has_any_column(columns, BAR_AGE_COLUMNS) and not any(signal.get("bar_age_seconds") is not None for signal in signals):
        warnings.add(f"{table} has no bar_age_seconds column or payload field; signal bar-age metrics unavailable.")


def _warn_for_market_data_gaps(
    table: str,
    columns: list[str],
    market_data: list[dict[str, Any]],
    warnings: WarningCollector,
) -> None:
    if not _has_any_column(columns, SPREAD_COLUMNS) and not any(row.get("spread_bps") is not None for row in market_data):
        warnings.add(f"{table} has no spread_bps column; spread metrics may rely on signals only.")
    if not _has_any_column(columns, QUOTE_IMBALANCE_COLUMNS) and not any(
        row.get("quote_imbalance") is not None for row in market_data
    ):
        warnings.add(f"{table} has no quote_imbalance column; quote imbalance metrics may rely on signals only.")
    has_bar_age = _has_any_column(columns, BAR_AGE_COLUMNS)
    can_derive_bar_age = _has_any_column(columns, ("collected_at", "created_at")) and _has_any_column(
        columns, BAR_TIMESTAMP_COLUMNS
    )
    if not has_bar_age and not can_derive_bar_age and not any(row.get("bar_age_seconds") is not None for row in market_data):
        warnings.add(f"{table} has no bar-age column or collected_at/timestamp pair; data-age metrics unavailable.")


def _bucket_breakdown(
    values: Iterable[Any],
    bucket_func: Callable[[Any], str | None],
    bucket_order: tuple[str, ...],
) -> list[dict[str, Any]]:
    buckets = [bucket_func(value) or MISSING for value in values]
    if not buckets:
        return []
    total = len(buckets)
    counter = Counter(buckets)
    rows = [{"value": bucket, "count": counter.get(bucket, 0), "pct": counter.get(bucket, 0) / total} for bucket in bucket_order]
    for bucket, count in sorted(counter.items()):
        if bucket not in bucket_order:
            rows.append({"value": bucket, "count": count, "pct": count / total})
    return rows


def _extract_value(
    row: dict[str, Any],
    columns: list[str],
    column_names: tuple[str, ...],
    *,
    json_paths: Iterable[str | tuple[str, ...]] = (),
    json_keys: Iterable[str] = (),
) -> Any:
    value = _first_column_value(row, columns, column_names)
    if _has_value(value):
        return value
    payloads = _json_payloads(row, columns)
    for payload in payloads:
        for path in json_paths:
            value = _value_from_path(payload, path)
            if _has_value(value):
                return value
        value = _find_key(payload, json_keys)
        if _has_value(value):
            return value
    return None


def _json_payloads(row: dict[str, Any], columns: list[str]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for column in JSON_COLUMNS:
        value = _first_column_value(row, columns, (column,))
        if value is None:
            continue
        parsed = _safe_json_loads(value)
        if isinstance(parsed, dict):
            payloads.append(parsed)
    return payloads


def _safe_json_loads(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _value_from_path(payload: dict[str, Any], path: str | tuple[str, ...]) -> Any:
    parts = tuple(path.split(".")) if isinstance(path, str) else path
    current: Any = payload
    for part in parts:
        if not isinstance(current, dict):
            return None
        current = _dict_get(current, part)
        if current is None:
            return None
    return current


def _find_key(payload: Any, keys: Iterable[str]) -> Any:
    wanted = {key.lower() for key in keys}
    if not wanted:
        return None
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key.lower() in wanted and _has_value(value):
                return value
        for value in payload.values():
            found = _find_key(value, wanted)
            if _has_value(found):
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = _find_key(item, wanted)
            if _has_value(found):
                return found
    return None


def _dict_get(payload: dict[str, Any], key: str) -> Any:
    if key in payload:
        return payload[key]
    lower_key = key.lower()
    for current_key, value in payload.items():
        if current_key.lower() == lower_key:
            return value
    return None


def _first_column_value(row: dict[str, Any], columns: list[str], candidates: tuple[str, ...]) -> Any:
    column = _first_column(columns, candidates)
    if column is None:
        return None
    return row.get(column)


def _first_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lookup = {column.lower(): column for column in columns}
    for candidate in candidates:
        column = lookup.get(candidate.lower())
        if column is not None:
            return column
    return None


def _has_any_column(columns: list[str], candidates: tuple[str, ...]) -> bool:
    return _first_column(columns, candidates) is not None


def _parse_datetime(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return _ensure_utc(value)
    if isinstance(value, (int, float)):
        timestamp = float(value)
        if timestamp > 10_000_000_000:
            timestamp = timestamp / 1000
        try:
            return datetime.fromtimestamp(timestamp, tz=UTC)
        except (OSError, OverflowError, ValueError):
            return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    for candidate in (text, text.replace(" ", "T", 1)):
        try:
            return _ensure_utc(datetime.fromisoformat(candidate))
        except ValueError:
            continue
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    return None


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _finite_values(values: Iterable[Any]) -> list[float]:
    return [value for value in (_safe_float(value) for value in values) if value is not None]


def _percentile(values: Iterable[Any], percentile: float) -> float | None:
    parsed = sorted(_finite_values(values))
    if not parsed:
        return None
    if len(parsed) == 1:
        return parsed[0]
    rank = (len(parsed) - 1) * percentile
    lower_index = math.floor(rank)
    upper_index = math.ceil(rank)
    if lower_index == upper_index:
        return parsed[lower_index]
    lower = parsed[lower_index]
    upper = parsed[upper_index]
    return lower + (upper - lower) * (rank - lower_index)


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _has_value(value: Any) -> bool:
    return value is not None and value != ""


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_optional_text(value: Any) -> str | None:
    text = _normalize_text(value)
    return text or None


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _format_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    string_rows = [tuple(str(value) for value in row) for row in rows]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in string_rows)) if string_rows else len(headers[index])
        for index in range(len(headers))
    ]
    header_line = " | ".join(headers[index].ljust(widths[index]) for index in range(len(headers)))
    separator = "-+-".join("-" * width for width in widths)
    body = [" | ".join(row[index].ljust(widths[index]) for index in range(len(headers))) for row in string_rows]
    return "\n".join([header_line, separator, *body])


def _format_count(value: Any) -> str:
    parsed = _safe_float(value)
    if parsed is None:
        return "n/a"
    return str(int(parsed))


def _format_number(value: Any) -> str:
    parsed = _safe_float(value)
    if parsed is None:
        return "n/a"
    if parsed == 0:
        return "0"
    if abs(parsed) >= 100:
        return f"{parsed:.2f}"
    return f"{parsed:.6f}".rstrip("0").rstrip(".")


def _format_pct(value: Any) -> str:
    parsed = _safe_float(value)
    if parsed is None:
        return "n/a"
    return f"{parsed * 100:.2f}%"


def _format_seconds(value: Any) -> str:
    parsed = _safe_float(value)
    if parsed is None:
        return "n/a"
    return f"{parsed:.2f}s"


if __name__ == "__main__":
    raise SystemExit(main())
