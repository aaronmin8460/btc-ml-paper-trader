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


DECISION_TABLES = ("decisions", "signals", "strategy_decisions", "trading_signals")
TRADE_TABLES = ("trades", "completed_trades", "realized_trades")
ORDER_TABLES = ("orders", "order_events")

TIMESTAMP_COLUMNS = ("created_at", "timestamp", "ts", "time", "updated_at")
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

ACTION_COLUMNS = ("action", "decision_action", "final_decision", "side")
REASON_COLUMNS = ("reason", "decision_reason", "block_reason")
BLOCKED_BY_COLUMNS = ("blocked_by", "blocked_reason_source")
STRATEGY_COLUMNS = ("strategy_name", "strategy", "selected_strategy")
REGIME_COLUMNS = ("regime", "market_regime", "market_state")
BUY_PROBABILITY_COLUMNS = ("buy_probability", "ml_buy_probability")
SELL_PROBABILITY_COLUMNS = ("sell_probability", "ml_sell_probability")
CONFIDENCE_GAP_COLUMNS = ("confidence_gap", "ml_confidence_gap")
SPREAD_COLUMNS = ("spread_bps", "orderbook_spread_bps")
QUOTE_IMBALANCE_COLUMNS = ("quote_imbalance", "imbalance")
PNL_COLUMNS = ("net_pnl", "pnl", "realized_pnl", "gross_pnl")
HOLD_SECONDS_COLUMNS = ("hold_seconds", "holding_seconds", "holding_time_seconds", "duration_seconds")
SELL_REASON_COLUMNS = ("reason", "sell_reason", "exit_reason", "decision_reason")
STATUS_COLUMNS = ("status", "order_status")
ORDER_TYPE_COLUMNS = ("order_type", "type")
TIME_IN_FORCE_COLUMNS = ("time_in_force", "tif")
CANCEL_REASON_COLUMNS = ("cancel_reason", "reason")

MISSING = "(missing)"

BUY_PROBABILITY_BUCKETS = ("<0.50", "0.50-0.54", "0.55-0.59", "0.60-0.64", ">=0.65")
CONFIDENCE_GAP_BUCKETS = ("<0.02", "0.02-0.04", "0.04-0.08", ">=0.08")
SPREAD_BPS_BUCKETS = ("<=2", "2-4", "4-6", "6-10", ">10")
QUOTE_IMBALANCE_BUCKETS = ("<-0.10", "-0.10-0.00", "0.00-0.05", "0.05-0.10", ">0.10")
HOLDING_TIME_BUCKETS = ("<30s", "30-90s", "90-180s", "180-900s", ">900s")


class WarningCollector:
    def __init__(self) -> None:
        self.items: list[str] = []
        self._seen: set[str] = set()

    def add(self, message: str) -> None:
        if message not in self._seen:
            self._seen.add(message)
            self.items.append(message)


def buy_probability_bucket(value: Any) -> str | None:
    parsed = _safe_float(value)
    if parsed is None:
        return None
    if parsed < 0.50:
        return "<0.50"
    if parsed < 0.55:
        return "0.50-0.54"
    if parsed < 0.60:
        return "0.55-0.59"
    if parsed < 0.65:
        return "0.60-0.64"
    return ">=0.65"


def confidence_gap_bucket(value: Any) -> str | None:
    parsed = _safe_float(value)
    if parsed is None:
        return None
    if parsed < 0.02:
        return "<0.02"
    if parsed < 0.04:
        return "0.02-0.04"
    if parsed < 0.08:
        return "0.04-0.08"
    return ">=0.08"


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


def holding_time_bucket(value: Any) -> str | None:
    parsed = _safe_float(value)
    if parsed is None:
        return None
    if parsed < 30:
        return "<30s"
    if parsed < 90:
        return "30-90s"
    if parsed < 180:
        return "90-180s"
    if parsed <= 900:
        return "180-900s"
    return ">900s"


def calculate_pnl_metrics(pnls: Iterable[Any]) -> dict[str, float | int | None]:
    values = [value for value in (_safe_float(pnl) for pnl in pnls) if value is not None]
    if not values:
        return {
            "pnl_sample_size": 0,
            "win_rate": None,
            "average_win": None,
            "average_loss": None,
            "expectancy": None,
            "total_realized_pnl": None,
            "max_consecutive_losses": None,
        }

    wins = [value for value in values if value > 0]
    losses = [value for value in values if value < 0]
    total = len(values)
    win_rate = len(wins) / total
    loss_rate = len(losses) / total
    average_win = mean(wins) if wins else None
    average_loss = mean(losses) if losses else None
    expectancy = (win_rate * (average_win or 0.0)) + (loss_rate * (average_loss or 0.0))
    return {
        "pnl_sample_size": total,
        "win_rate": win_rate,
        "average_win": average_win,
        "average_loss": average_loss,
        "expectancy": expectancy,
        "total_realized_pnl": sum(values),
        "max_consecutive_losses": max_consecutive_losses(values),
    }


def max_consecutive_losses(pnls: Iterable[Any]) -> int:
    longest = 0
    current = 0
    for pnl in pnls:
        value = _safe_float(pnl)
        if value is not None and value < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def build_strategy_health_report(
    db_path: str | Path,
    *,
    hours: float = 48,
    now: datetime | None = None,
) -> dict[str, Any]:
    current_time = _ensure_utc(now or datetime.now(UTC))
    window_start = current_time - timedelta(hours=max(0.0, float(hours)))
    warnings = WarningCollector()
    path = Path(db_path)
    tables: dict[str, list[str]] = {}
    decisions: list[dict[str, Any]] = []
    completed_trades: list[dict[str, Any]] = []
    orders: list[dict[str, Any]] = []
    sources = {"decisions": None, "trades": None, "orders": None}

    if not path.exists():
        warnings.add(f"Database file not found: {path}")
    else:
        try:
            with sqlite3.connect(path) as connection:
                connection.row_factory = sqlite3.Row
                tables = _inspect_tables(connection)
                if not tables:
                    warnings.add(f"No tables found in SQLite database: {path}")

                decision_table = _resolve_table(tables, DECISION_TABLES, "decision", warnings)
                trade_table = _resolve_table(tables, TRADE_TABLES, "trade", warnings)
                order_table = _resolve_table(tables, ORDER_TABLES, "order", warnings)

                if decision_table:
                    sources["decisions"] = decision_table
                    rows = _load_recent_rows(
                        connection,
                        decision_table,
                        tables[decision_table],
                        window_start=window_start,
                        purpose="decision",
                        warnings=warnings,
                    )
                    decisions = [_normalize_decision(row, tables[decision_table]) for row in rows]
                    _warn_for_decision_gaps(decision_table, tables[decision_table], decisions, warnings)

                if trade_table:
                    sources["trades"] = trade_table
                    rows = _load_recent_rows(
                        connection,
                        trade_table,
                        tables[trade_table],
                        window_start=window_start,
                        purpose="trade",
                        warnings=warnings,
                    )
                    trades = [_normalize_trade(row, tables[trade_table]) for row in rows]
                    completed_trades = _completed_trade_rows(trades, trade_table, tables[trade_table], warnings)
                    _warn_for_trade_gaps(trade_table, tables[trade_table], completed_trades, warnings)

                if order_table:
                    sources["orders"] = order_table
                    rows = _load_recent_rows(
                        connection,
                        order_table,
                        tables[order_table],
                        window_start=window_start,
                        purpose="order",
                        warnings=warnings,
                    )
                    orders = [_normalize_order(row, tables[order_table]) for row in rows]
                    _warn_for_order_gaps(order_table, tables[order_table], orders, warnings)
        except sqlite3.DatabaseError as exc:
            warnings.add(f"Could not read SQLite database {path}: {exc}")

    metrics = _empty_metrics()
    if sources["decisions"]:
        metrics.update(_decision_metrics(decisions))
    if sources["trades"]:
        metrics.update(_trade_metrics(completed_trades))
    if sources["orders"]:
        metrics["ioc_cancellations"] = sum(1 for order in orders if _is_ioc_cancellation(order))

    breakdowns = _build_breakdowns(decisions, completed_trades, sources=sources, warnings=warnings)
    return {
        "generated_at": current_time.isoformat(),
        "db_path": str(path),
        "window": {
            "hours": float(hours),
            "start": window_start.isoformat(),
            "end": current_time.isoformat(),
        },
        "sources": sources,
        "available_tables": {table: columns for table, columns in sorted(tables.items())},
        "warnings": warnings.items,
        "metrics": metrics,
        "breakdowns": breakdowns,
        "note": (
            "Local diagnostic only. This report reads SQLite data and does not change trading logic, "
            "thresholds, settings, order submission, deployment, or live trading behavior."
        ),
    }


def save_strategy_health_report(report: dict[str, Any], output_dir: str | Path = "reports") -> Path:
    generated_at = _parse_datetime(report.get("generated_at")) or datetime.now(UTC)
    output_path = Path(output_dir) / f"strategy_health_{generated_at.strftime('%Y%m%d_%H%M%S')}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report["output_path"] = str(output_path)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def format_strategy_health_report(report: dict[str, Any]) -> str:
    lines = [
        "BTC/USD strategy health diagnostics",
        "",
        f"database: {report['db_path']}",
        f"window: last {_format_number(report['window']['hours'])} hours "
        f"({report['window']['start']} to {report['window']['end']})",
        f"json_report: {report.get('output_path', 'not written')}",
        "",
        report["note"],
        "",
    ]
    warnings = report.get("warnings") or []
    if warnings:
        lines.append("Warnings")
        lines.extend(f"  - {warning}" for warning in warnings)
        lines.append("")

    metrics = report.get("metrics") or {}
    lines.append("Summary")
    lines.append(
        _format_table(
            ("Metric", "Value"),
            [
                ("total decisions", _format_count(metrics.get("total_decisions"))),
                ("buy decisions", _format_count(metrics.get("buy_decisions"))),
                ("sell decisions", _format_count(metrics.get("sell_decisions"))),
                ("hold decisions", _format_count(metrics.get("hold_decisions"))),
                ("completed trades", _format_count(metrics.get("completed_trades"))),
                ("win rate", _format_pct(metrics.get("win_rate"))),
                ("average win", _format_money(metrics.get("average_win"))),
                ("average loss", _format_money(metrics.get("average_loss"))),
                ("expectancy", _format_money(metrics.get("expectancy"))),
                ("total realized PnL", _format_money(metrics.get("total_realized_pnl"))),
                ("max consecutive losses", _format_count(metrics.get("max_consecutive_losses"))),
                ("average holding time", _format_seconds(metrics.get("average_holding_time_seconds"))),
                ("median holding time", _format_seconds(metrics.get("median_holding_time_seconds"))),
                ("average spread_bps", _format_number(metrics.get("average_spread_bps"))),
                ("average quote_imbalance", _format_number(metrics.get("average_quote_imbalance"))),
                ("IOC cancellations", _format_count(metrics.get("ioc_cancellations"))),
            ],
        )
    )

    breakdowns = report.get("breakdowns") or {}
    labels = {
        "decision_action": "Decision Action",
        "decision_reason": "Decision Reason",
        "blocked_by": "Blocked By",
        "strategy_name": "Strategy Name",
        "market_regime": "Market Regime",
        "sell_reason": "Sell Reason",
        "buy_probability_bucket": "Buy Probability Bucket",
        "confidence_gap_bucket": "Confidence Gap Bucket",
        "spread_bps_bucket": "Spread Bps Bucket",
        "quote_imbalance_bucket": "Quote Imbalance Bucket",
        "holding_time_bucket": "Holding Time Bucket",
    }
    for key, label in labels.items():
        lines.extend(["", label])
        rows = breakdowns.get(key) or []
        if rows:
            lines.append(
                _format_table(
                    ("Value", "Count", "Pct"),
                    [
                        (
                            str(row["value"]),
                            _format_count(row["count"]),
                            _format_pct(row.get("pct")),
                        )
                        for row in rows
                    ],
                )
            )
        else:
            lines.append("  n/a")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a local SQLite strategy health diagnostics report.")
    parser.add_argument("--db", required=True, help="Path to the local SQLite trading database.")
    parser.add_argument("--hours", type=float, default=48, help="Lookback window in hours. Defaults to 48.")
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Directory for the JSON report. Defaults to reports/.",
    )
    args = parser.parse_args(argv)
    if args.hours <= 0:
        parser.error("--hours must be greater than 0")

    report = build_strategy_health_report(args.db, hours=args.hours)
    save_strategy_health_report(report, args.output_dir)
    print(format_strategy_health_report(report))
    return 0


def _empty_metrics() -> dict[str, Any]:
    return {
        "total_decisions": None,
        "buy_decisions": None,
        "sell_decisions": None,
        "hold_decisions": None,
        "completed_trades": None,
        "pnl_sample_size": None,
        "win_rate": None,
        "average_win": None,
        "average_loss": None,
        "expectancy": None,
        "total_realized_pnl": None,
        "max_consecutive_losses": None,
        "average_holding_time_seconds": None,
        "median_holding_time_seconds": None,
        "average_spread_bps": None,
        "average_quote_imbalance": None,
        "ioc_cancellations": None,
    }


def _decision_metrics(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    actions = [_normalize_text(decision.get("action")).lower() for decision in decisions]
    spread_values = [decision.get("spread_bps") for decision in decisions]
    quote_imbalance_values = [decision.get("quote_imbalance") for decision in decisions]
    return {
        "total_decisions": len(decisions),
        "buy_decisions": sum(1 for action in actions if action == "buy"),
        "sell_decisions": sum(1 for action in actions if action == "sell"),
        "hold_decisions": sum(1 for action in actions if action == "hold"),
        "average_spread_bps": _average(spread_values),
        "average_quote_imbalance": _average(quote_imbalance_values),
    }


def _trade_metrics(completed_trades: list[dict[str, Any]]) -> dict[str, Any]:
    pnl_values = [trade.get("pnl") for trade in completed_trades if trade.get("pnl") is not None]
    hold_values = [trade.get("hold_seconds") for trade in completed_trades if trade.get("hold_seconds") is not None]
    metrics = calculate_pnl_metrics(pnl_values)
    metrics["completed_trades"] = len(completed_trades)
    if not completed_trades:
        metrics["total_realized_pnl"] = 0.0
        metrics["max_consecutive_losses"] = 0
    metrics["average_holding_time_seconds"] = _average(hold_values)
    metrics["median_holding_time_seconds"] = median(hold_values) if hold_values else None
    return metrics


def _build_breakdowns(
    decisions: list[dict[str, Any]],
    completed_trades: list[dict[str, Any]],
    *,
    sources: dict[str, str | None],
    warnings: WarningCollector,
) -> dict[str, list[dict[str, Any]]]:
    sell_reasons = [trade.get("sell_reason") for trade in completed_trades]
    if not any(_has_value(reason) for reason in sell_reasons):
        sell_decision_reasons = [
            decision.get("reason")
            for decision in decisions
            if _normalize_text(decision.get("action")).lower() == "sell"
        ]
        if sell_decision_reasons:
            warnings.add("Trade sell reason unavailable; using sell decision.reason for sell_reason breakdown.")
            sell_reasons = sell_decision_reasons

    breakdowns = {
        "decision_action": _count_breakdown([decision.get("action") for decision in decisions]),
        "decision_reason": _count_breakdown([decision.get("reason") for decision in decisions]),
        "blocked_by": _count_breakdown(
            [
                decision.get("blocked_by")
                for decision in decisions
                if _has_value(decision.get("blocked_by"))
                or _normalize_text(decision.get("action")).lower() == "hold"
            ]
        ),
        "strategy_name": _count_breakdown([decision.get("strategy_name") for decision in decisions])
        if sources.get("decisions") and any(_has_value(decision.get("strategy_name")) for decision in decisions)
        else [],
        "market_regime": _count_breakdown([decision.get("market_regime") for decision in decisions])
        if sources.get("decisions") and any(_has_value(decision.get("market_regime")) for decision in decisions)
        else [],
        "sell_reason": _count_breakdown(sell_reasons),
        "buy_probability_bucket": _bucket_breakdown(
            [decision.get("buy_probability") for decision in decisions],
            buy_probability_bucket,
            BUY_PROBABILITY_BUCKETS,
        ),
        "confidence_gap_bucket": _bucket_breakdown(
            [decision.get("confidence_gap") for decision in decisions],
            confidence_gap_bucket,
            CONFIDENCE_GAP_BUCKETS,
        ),
        "spread_bps_bucket": _bucket_breakdown(
            [decision.get("spread_bps") for decision in decisions],
            spread_bps_bucket,
            SPREAD_BPS_BUCKETS,
        ),
        "quote_imbalance_bucket": _bucket_breakdown(
            [decision.get("quote_imbalance") for decision in decisions],
            quote_imbalance_bucket,
            QUOTE_IMBALANCE_BUCKETS,
        ),
        "holding_time_bucket": _bucket_breakdown(
            [trade.get("hold_seconds") for trade in completed_trades],
            holding_time_bucket,
            HOLDING_TIME_BUCKETS,
        ),
    }
    return breakdowns


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
    table_lookup = {name.lower(): name for name in tables}
    for candidate in candidates:
        if candidate.lower() in table_lookup:
            return table_lookup[candidate.lower()]
    warnings.add(
        f"No {label} table found. Looked for: {', '.join(candidates)}."
    )
    return None


def _load_recent_rows(
    connection: sqlite3.Connection,
    table: str,
    columns: list[str],
    *,
    window_start: datetime,
    purpose: str,
    warnings: WarningCollector,
) -> list[dict[str, Any]]:
    selected = ", ".join(_quote_identifier(column) for column in columns)
    rows = [dict(row) for row in connection.execute(f"SELECT {selected} FROM {_quote_identifier(table)}").fetchall()]
    timestamp_column = _first_column(columns, TIMESTAMP_COLUMNS)
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


def _normalize_decision(row: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    action = _normalize_text(
        _extract_value(
            row,
            columns,
            ACTION_COLUMNS,
            json_paths=("action", "final_decision", "decision.action"),
            json_keys=("action", "final_decision"),
        )
    ).lower()
    reason = _normalize_optional_text(
        _extract_value(
            row,
            columns,
            REASON_COLUMNS,
            json_paths=("reason", "block_reason", "decision.reason"),
            json_keys=("reason", "block_reason"),
        )
    )
    buy_probability = _safe_float(
        _extract_value(
            row,
            columns,
            BUY_PROBABILITY_COLUMNS,
            json_paths=("buy_probability", "ml_buy_probability", "ml_confirmation.buy_probability"),
            json_keys=("buy_probability", "ml_buy_probability"),
        )
    )
    sell_probability = _safe_float(
        _extract_value(
            row,
            columns,
            SELL_PROBABILITY_COLUMNS,
            json_paths=("sell_probability", "ml_sell_probability", "ml_confirmation.sell_probability"),
            json_keys=("sell_probability", "ml_sell_probability"),
        )
    )
    confidence_gap = _safe_float(
        _extract_value(
            row,
            columns,
            CONFIDENCE_GAP_COLUMNS,
            json_paths=("confidence_gap", "ml_confirmation.confidence_gap"),
            json_keys=("confidence_gap", "ml_confidence_gap"),
        )
    )
    if confidence_gap is None and buy_probability is not None and sell_probability is not None:
        confidence_gap = buy_probability - sell_probability
    blocked_by = _normalize_optional_text(
        _extract_value(
            row,
            columns,
            BLOCKED_BY_COLUMNS,
            json_paths=("blocked_by", "decision.blocked_by"),
            json_keys=("blocked_by",),
        )
    )
    if blocked_by is None and action == "hold":
        blocked_by = infer_blocked_by(reason)
    raw_strategy_name = _extract_value(
        row,
        columns,
        STRATEGY_COLUMNS,
        json_paths=(
            "strategy_name",
            "decision.strategy_name",
            "selected_strategy_signal.strategy_name",
            "strategy.strategy_name",
        ),
        json_keys=("strategy_name",),
    )
    if isinstance(raw_strategy_name, dict):
        raw_strategy_name = raw_strategy_name.get("strategy_name") or raw_strategy_name.get("name")
    strategy_name = _normalize_optional_text(raw_strategy_name)

    raw_market_regime = _extract_value(
        row,
        columns,
        REGIME_COLUMNS,
        json_paths=("regime.regime", "metadata.regime.regime", "market_regime", "decision.regime", "regime"),
        json_keys=("regime", "market_regime"),
    )
    if isinstance(raw_market_regime, dict):
        raw_market_regime = raw_market_regime.get("regime") or raw_market_regime.get("market_regime")
    market_regime = _normalize_optional_text(raw_market_regime)
    return {
        "action": action or None,
        "reason": reason,
        "blocked_by": blocked_by,
        "strategy_name": strategy_name,
        "market_regime": market_regime,
        "buy_probability": buy_probability,
        "sell_probability": sell_probability,
        "confidence_gap": confidence_gap,
        "spread_bps": _safe_float(
            _extract_value(
                row,
                columns,
                SPREAD_COLUMNS,
                json_paths=("spread_bps",),
                json_keys=("spread_bps",),
            )
        ),
        "quote_imbalance": _safe_float(
            _extract_value(
                row,
                columns,
                QUOTE_IMBALANCE_COLUMNS,
                json_paths=("quote_imbalance",),
                json_keys=("quote_imbalance",),
            )
        ),
    }


def _normalize_trade(row: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    return {
        "side": _normalize_optional_text(
            _extract_value(row, columns, ("side", "action"), json_paths=("side", "action"), json_keys=("side",))
        ),
        "pnl": _safe_float(
            _extract_value(row, columns, PNL_COLUMNS, json_paths=PNL_COLUMNS, json_keys=PNL_COLUMNS)
        ),
        "sell_reason": _normalize_optional_text(
            _extract_value(
                row,
                columns,
                SELL_REASON_COLUMNS,
                json_paths=("reason", "sell_reason", "exit_reason", "decision_reason"),
                json_keys=("sell_reason", "exit_reason", "decision_reason"),
            )
        ),
        "hold_seconds": _safe_float(
            _extract_value(
                row,
                columns,
                HOLD_SECONDS_COLUMNS,
                json_paths=("hold_seconds", "holding_seconds"),
                json_keys=("hold_seconds", "holding_seconds"),
            )
        ),
    }


def _normalize_order(row: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    return {
        "status": _normalize_optional_text(
            _extract_value(row, columns, STATUS_COLUMNS, json_paths=("status",), json_keys=("status",))
        ),
        "order_type": _normalize_optional_text(
            _extract_value(
                row,
                columns,
                ORDER_TYPE_COLUMNS,
                json_paths=("order_type", "type"),
                json_keys=("order_type", "type"),
            )
        ),
        "time_in_force": _normalize_optional_text(
            _extract_value(
                row,
                columns,
                TIME_IN_FORCE_COLUMNS,
                json_paths=("time_in_force", "tif"),
                json_keys=("time_in_force", "tif"),
            )
        ),
        "cancel_reason": _normalize_optional_text(
            _extract_value(
                row,
                columns,
                CANCEL_REASON_COLUMNS,
                json_paths=("cancel_reason", "reason"),
                json_keys=("cancel_reason",),
            )
        ),
    }


def _completed_trade_rows(
    trades: list[dict[str, Any]],
    table: str,
    columns: list[str],
    warnings: WarningCollector,
) -> list[dict[str, Any]]:
    if _has_any_column(columns, ("side", "action")) or any(_has_value(trade.get("side")) for trade in trades):
        return [trade for trade in trades if _normalize_text(trade.get("side")).lower() == "sell"]
    warnings.add(f"{table} has no side/action column; treating all trade rows as completed trades.")
    return trades


def _is_ioc_cancellation(order: dict[str, Any]) -> bool:
    status = _normalize_text(order.get("status")).lower()
    if status not in {"canceled", "cancelled"}:
        return False
    cancel_reason = _normalize_text(order.get("cancel_reason")).lower()
    time_in_force = _normalize_text(order.get("time_in_force")).lower()
    return time_in_force == "ioc" or cancel_reason == "ioc_no_fill"


def _warn_for_decision_gaps(
    table: str,
    columns: list[str],
    decisions: list[dict[str, Any]],
    warnings: WarningCollector,
) -> None:
    if not _has_any_column(columns, ACTION_COLUMNS) and not any(_has_value(decision.get("action")) for decision in decisions):
        warnings.add(f"{table} has no decision action column; decision action counts may be unavailable.")
    if not _has_any_column(columns, REASON_COLUMNS) and not any(_has_value(decision.get("reason")) for decision in decisions):
        warnings.add(f"{table} has no decision reason column; reason breakdown may be unavailable.")
    if not _has_any_column(columns, BLOCKED_BY_COLUMNS):
        warnings.add(f"{table} has no blocked_by column; blocked_by is inferred from decision.reason where possible.")
    if not _has_any_column(columns, STRATEGY_COLUMNS) and not any(_has_value(decision.get("strategy_name")) for decision in decisions):
        warnings.add(f"{table} has no strategy_name column or payload field; strategy_name breakdown unavailable.")
    if not _has_any_column(columns, REGIME_COLUMNS) and not any(_has_value(decision.get("market_regime")) for decision in decisions):
        warnings.add(f"{table} has no market regime column or payload field; market regime breakdown unavailable.")
    if not _has_any_column(columns, BUY_PROBABILITY_COLUMNS) and not any(
        decision.get("buy_probability") is not None for decision in decisions
    ):
        warnings.add(f"{table} has no buy_probability column; buy_probability buckets unavailable.")
    if not _has_any_column(columns, SELL_PROBABILITY_COLUMNS + CONFIDENCE_GAP_COLUMNS) and not any(
        decision.get("confidence_gap") is not None for decision in decisions
    ):
        warnings.add(f"{table} has no sell_probability or confidence_gap column; confidence_gap buckets unavailable.")
    if not _has_any_column(columns, SPREAD_COLUMNS) and not any(decision.get("spread_bps") is not None for decision in decisions):
        warnings.add(f"{table} has no spread_bps column; spread_bps metrics and buckets unavailable.")
    if not _has_any_column(columns, QUOTE_IMBALANCE_COLUMNS) and not any(
        decision.get("quote_imbalance") is not None for decision in decisions
    ):
        warnings.add(f"{table} has no quote_imbalance column; quote_imbalance metrics and buckets unavailable.")


def _warn_for_trade_gaps(
    table: str,
    columns: list[str],
    completed_trades: list[dict[str, Any]],
    warnings: WarningCollector,
) -> None:
    if not _has_any_column(columns, PNL_COLUMNS) and not any(trade.get("pnl") is not None for trade in completed_trades):
        warnings.add(f"{table} has no realized PnL column; win rate, expectancy, and PnL metrics unavailable.")
    if not _has_any_column(columns, HOLD_SECONDS_COLUMNS) and not any(
        trade.get("hold_seconds") is not None for trade in completed_trades
    ):
        warnings.add(f"{table} has no holding time column; holding time metrics and buckets unavailable.")
    if not _has_any_column(columns, SELL_REASON_COLUMNS) and not any(_has_value(trade.get("sell_reason")) for trade in completed_trades):
        warnings.add(f"{table} has no sell reason column; sell reason breakdown may be unavailable.")


def _warn_for_order_gaps(
    table: str,
    columns: list[str],
    orders: list[dict[str, Any]],
    warnings: WarningCollector,
) -> None:
    if not _has_any_column(columns, STATUS_COLUMNS) and not any(_has_value(order.get("status")) for order in orders):
        warnings.add(f"{table} has no status column; IOC cancellations unavailable.")
    if (
        not _has_any_column(columns, TIME_IN_FORCE_COLUMNS + CANCEL_REASON_COLUMNS)
        and not _has_any_column(columns, JSON_COLUMNS)
        and not any(_has_value(order.get("time_in_force")) or _has_value(order.get("cancel_reason")) for order in orders)
    ):
        warnings.add(f"{table} has no time_in_force or cancel_reason data; IOC cancellations may be undercounted.")


def infer_blocked_by(reason: Any) -> str | None:
    text = _normalize_text(reason)
    if not text:
        return None
    if text in {"stale_market_data", "scalping_stale_data_exit"}:
        return "stale_market_data"
    if text in {"spread_too_wide", "spread_unavailable"}:
        return "spread"
    if text in {"quote_imbalance_too_weak", "quote_imbalance_unavailable"}:
        return "quote_imbalance"
    if text == "api_budget_exhausted":
        return "api_budget"
    if text in {"recent_ioc_cancels_too_high", "ioc_cancel_cooldown_active"}:
        return "ioc_cancel_guard"
    if text in {"trade_cooldown_active", "cooldown_after_loss"}:
        return "cooldown"
    if text in {"active_model_invalid", "model_not_profitable_after_costs"}:
        return "active_model_invalid"
    if text == "model_unavailable" or text.startswith("scalping_buy_probability") or text.startswith("scalping_confidence"):
        return "ml_filter"
    if text in {
        "max_order_attempts_per_hour_reached",
        "max_order_attempts_per_10_minutes_reached",
        "max_order_attempts_per_day_reached",
        "max_trades_per_hour_reached",
        "max_trades_per_10_minutes_reached",
        "max_daily_trades_reached",
        "max_consecutive_losses_reached",
        "account_daily_loss_usd_reached",
        "account_daily_loss_pct_reached",
        "account_drawdown_reached",
        "account_data_required_unavailable",
        "buying_power_too_low",
        "trading_disabled",
        "already_holding_btc",
        "sell_without_position",
        "order_in_flight",
    }:
        return "risk_manager"
    return None


def _count_breakdown(values: Iterable[Any]) -> list[dict[str, Any]]:
    labels = [_label(value) for value in values]
    if not labels:
        return []
    total = len(labels)
    counter = Counter(labels)
    return [
        {"value": value, "count": count, "pct": count / total if total else None}
        for value, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


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
    rows = [
        {"value": bucket, "count": counter.get(bucket, 0), "pct": counter.get(bucket, 0) / total}
        for bucket in bucket_order
    ]
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


def _average(values: Iterable[Any]) -> float | None:
    parsed = [value for value in (_safe_float(value) for value in values) if value is not None]
    return mean(parsed) if parsed else None


def _has_value(value: Any) -> bool:
    return value is not None and value != ""


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_optional_text(value: Any) -> str | None:
    text = _normalize_text(value)
    return text or None


def _label(value: Any) -> str:
    text = _normalize_text(value)
    return text if text else MISSING


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


def _format_money(value: Any) -> str:
    parsed = _safe_float(value)
    if parsed is None:
        return "n/a"
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
