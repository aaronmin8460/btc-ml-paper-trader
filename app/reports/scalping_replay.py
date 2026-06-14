from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

from app.config import Settings, get_settings
from app.data.scalping_features import build_scalping_features
from app.risk.risk_manager import AccountState, PositionState, TradeFrequencyState
from app.strategy.scalping_decision_engine import ScalpingDecisionEngine


MARKET_TABLES = ("collected_market_data", "market_bars")
TIMESTAMP_COLUMNS = ("timestamp", "created_at", "time")
REQUIRED_BAR_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")


@dataclass
class ReplayPosition:
    entry_time: datetime
    entry_close: float
    entry_price: float
    qty: float
    strategy_name: str | None
    regime: str | None
    highest_close: float
    entry_cost_pct: float


@dataclass
class ReplayTrade:
    entry_time: str
    exit_time: str
    entry_close: float
    exit_close: float
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    return_pct: float
    holding_seconds: float
    exit_reason: str
    strategy_name: str | None
    regime: str | None
    entry_cost_pct: float
    exit_cost_pct: float


def build_scalping_replay_report(
    db_path: str | Path,
    *,
    hours: float = 72,
    settings: Settings | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    settings = settings or get_settings()
    current_time = _ensure_utc(now or datetime.now(UTC))
    warnings: list[str] = []
    bars, source = load_recent_market_data(db_path, hours=hours, settings=settings, now=current_time, warnings=warnings)
    trades: list[ReplayTrade] = []
    max_open_positions = 0
    open_position_at_end = False

    if bars.empty:
        warnings.append("No BTC/USD 1-minute market data found for the requested replay window.")
        return _report(
            db_path=db_path,
            hours=hours,
            settings=settings,
            generated_at=current_time,
            source=source,
            bars=bars,
            trades=trades,
            warnings=warnings,
            max_open_positions=max_open_positions,
            open_position_at_end=open_position_at_end,
        )

    features = build_replay_features(bars, settings=settings, warnings=warnings)
    if len(features) < 12:
        warnings.append("Not enough feature-ready bars for a meaningful scalping replay.")

    engine = ScalpingDecisionEngine(settings)
    position: ReplayPosition | None = None
    last_row: pd.Series | None = None

    for _, row in features.iterrows():
        timestamp = _row_timestamp(row)
        close = _safe_float(row.get("close"))
        if timestamp is None or close is None or close <= 0:
            continue
        last_row = row

        if position is not None:
            position.highest_close = max(position.highest_close, close)
            exit_reason = replay_exit_reason(position, row, settings=settings, now=timestamp)
            if exit_reason is not None:
                trades.append(close_position(position, row, settings=settings, exit_reason=exit_reason))
                position = None
                continue

        if position is None:
            decision = replay_entry_decision(engine, row, settings=settings, now=timestamp)
            if decision.action != "buy":
                continue
            entry_cost_pct = estimated_entry_cost_pct(row, settings)
            entry_price = close * (1 + entry_cost_pct)
            if entry_price <= 0:
                continue
            position = ReplayPosition(
                entry_time=timestamp,
                entry_close=close,
                entry_price=entry_price,
                qty=float(settings.order_notional_usd) / entry_price,
                strategy_name=decision.strategy_name,
                regime=decision.regime,
                highest_close=close,
                entry_cost_pct=entry_cost_pct,
            )
            max_open_positions = max(max_open_positions, 1)

    if position is not None:
        open_position_at_end = True
        if last_row is not None:
            trades.append(close_position(position, last_row, settings=settings, exit_reason="end_of_replay"))
            position = None

    return _report(
        db_path=db_path,
        hours=hours,
        settings=settings,
        generated_at=current_time,
        source=source,
        bars=bars,
        trades=trades,
        warnings=warnings,
        max_open_positions=max_open_positions,
        open_position_at_end=open_position_at_end,
    )


def load_recent_market_data(
    db_path: str | Path,
    *,
    hours: float,
    settings: Settings,
    now: datetime,
    warnings: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = Path(db_path)
    source: dict[str, Any] = {"table": None, "rows_loaded": 0}
    if not path.exists():
        warnings.append(f"Database file not found: {path}")
        return pd.DataFrame(), source

    try:
        with sqlite3.connect(path) as connection:
            connection.row_factory = sqlite3.Row
            tables = _inspect_tables(connection)
            table = _first_existing_table(tables, MARKET_TABLES)
            if table is None:
                warnings.append(f"No market data table found. Looked for: {', '.join(MARKET_TABLES)}.")
                return pd.DataFrame(), source
            columns = tables[table]
            missing = [column for column in REQUIRED_BAR_COLUMNS if column not in columns]
            if missing:
                warnings.append(f"{table} is missing OHLCV columns required for replay: {missing}.")
                return pd.DataFrame(), {"table": table, "rows_loaded": 0}
            selected_columns = [
                column
                for column in (
                    "timestamp",
                    "symbol",
                    "timeframe",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "bid",
                    "ask",
                    "bid_size",
                    "ask_size",
                    "spread_bps",
                    "quote_imbalance",
                )
                if column in columns
            ]
            query = f"SELECT {', '.join(_quote_identifier(column) for column in selected_columns)} FROM {_quote_identifier(table)}"
            rows = [dict(row) for row in connection.execute(query).fetchall()]
    except sqlite3.DatabaseError as exc:
        warnings.append(f"Could not read SQLite database {path}: {exc}")
        return pd.DataFrame(), source

    frame = pd.DataFrame(rows)
    source["table"] = table
    if frame.empty:
        return frame, source
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"])
    if "symbol" in frame.columns:
        frame = frame[frame["symbol"].astype(str) == settings.symbol]
    if "timeframe" in frame.columns:
        frame = frame[frame["timeframe"].map(_is_one_minute_timeframe)]
    window_start = now - timedelta(hours=max(0.0, float(hours)))
    frame = frame[frame["timestamp"] >= pd.Timestamp(window_start)]
    for column in ["open", "high", "low", "close", "volume", "bid", "ask", "bid_size", "ask_size", "spread_bps", "quote_imbalance"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["open", "high", "low", "close", "volume"])
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    source["rows_loaded"] = int(len(frame))
    return frame, source


def build_replay_features(bars: pd.DataFrame, *, settings: Settings, warnings: list[str]) -> pd.DataFrame:
    features = build_scalping_features(bars[list(REQUIRED_BAR_COLUMNS)])
    features = features.reset_index(drop=True)
    bars = bars.reset_index(drop=True)
    spread_values = _spread_bps_series(bars)
    if spread_values.isna().all():
        warnings.append("Spread data is missing; using configured MAX_SPREAD_BPS as a replay fallback.")
        spread_values = pd.Series(float(settings.max_spread_bps), index=features.index)
    else:
        missing_count = int(spread_values.isna().sum())
        if missing_count:
            warnings.append(f"Spread data is missing on {missing_count} bar(s); filling with configured MAX_SPREAD_BPS.")
            spread_values = spread_values.fillna(float(settings.max_spread_bps))

    imbalance_values = _quote_imbalance_series(bars)
    if imbalance_values.isna().all():
        fallback = max(float(settings.min_quote_imbalance), 0.0)
        warnings.append("Quote imbalance data is missing; using a neutral/local-config fallback for replay entries.")
        imbalance_values = pd.Series(fallback, index=features.index)
    else:
        missing_count = int(imbalance_values.isna().sum())
        if missing_count:
            fallback = max(float(settings.min_quote_imbalance), 0.0)
            warnings.append(f"Quote imbalance data is missing on {missing_count} bar(s); filling with {fallback}.")
            imbalance_values = imbalance_values.fillna(fallback)

    features["spread_bps"] = spread_values.to_numpy()
    features["scalping_spread_bps"] = features["spread_bps"]
    features["scalping_spread_pct"] = features["spread_bps"] / 10_000
    features["orderbook_spread"] = features["scalping_spread_pct"]
    features["quote_imbalance"] = imbalance_values.to_numpy()
    features["scalping_quote_imbalance"] = features["quote_imbalance"]
    return features.iloc[min(10, max(0, len(features))) :].reset_index(drop=True)


def replay_entry_decision(
    engine: ScalpingDecisionEngine,
    row: pd.Series,
    *,
    settings: Settings,
    now: datetime,
):
    prediction = replay_prediction(row, settings)
    return engine.decide(
        prediction=prediction,
        feature_row=row,
        position=PositionState(),
        trading_enabled=settings.trading_enabled,
        trade_frequency=TradeFrequencyState(),
        order_attempt_frequency=TradeFrequencyState(),
        filled_trade_frequency=TradeFrequencyState(),
        quote=_quote_from_row(row),
        api_budget={"api_budget_status": "ok", "budget_remaining": 999_999},
        account_state=AccountState(),
        recent_ioc_canceled_buys=0,
        latest_ioc_canceled_buy_at=None,
        now=now,
    )


def replay_prediction(row: pd.Series, settings: Settings) -> dict[str, Any]:
    required_gap = max(0.0, float(settings.scalping_confidence_gap_required))
    buy_probability = min(0.95, max(float(settings.scalping_buy_probability_floor), 0.50) + required_gap + 0.02)
    sell_probability = max(0.01, buy_probability - required_gap - 0.02)
    return {
        "symbol": settings.symbol,
        "buy_probability": buy_probability,
        "sell_probability": sell_probability,
        "model_available": True,
        "prediction_source": "local_replay_synthetic_confirmation",
        "timestamp": str(row.get("timestamp")),
    }


def replay_exit_reason(
    position: ReplayPosition,
    row: pd.Series,
    *,
    settings: Settings,
    now: datetime,
) -> str | None:
    close = _safe_float(row.get("close"))
    if close is None or close <= 0:
        return None
    if close <= position.entry_price * (1 - float(settings.scalping_stop_loss_pct)):
        return "scalping_stop_loss"
    if close >= position.entry_price * (1 + float(settings.scalping_take_profit_pct)):
        return "scalping_take_profit"
    trailing_armed = position.highest_close >= position.entry_price * (1 + float(settings.trailing_stop_arm_profit_pct))
    if trailing_armed and close <= position.highest_close * (1 - float(settings.scalping_trailing_stop_pct)):
        return "scalping_trailing_stop"
    if now - position.entry_time >= timedelta(seconds=int(settings.scalping_max_position_seconds)):
        return "scalping_max_position_seconds"
    return None


def close_position(
    position: ReplayPosition,
    row: pd.Series,
    *,
    settings: Settings,
    exit_reason: str,
) -> ReplayTrade:
    timestamp = _row_timestamp(row)
    if timestamp is None:
        timestamp = position.entry_time
    close = _safe_float(row.get("close")) or position.entry_close
    exit_cost_pct = estimated_exit_cost_pct(row, settings)
    exit_price = close * (1 - exit_cost_pct)
    pnl = position.qty * (exit_price - position.entry_price)
    return_pct = (exit_price - position.entry_price) / position.entry_price if position.entry_price > 0 else 0.0
    holding_seconds = max(0.0, (timestamp - position.entry_time).total_seconds())
    return ReplayTrade(
        entry_time=position.entry_time.isoformat(),
        exit_time=timestamp.isoformat(),
        entry_close=position.entry_close,
        exit_close=close,
        entry_price=position.entry_price,
        exit_price=exit_price,
        qty=position.qty,
        pnl=pnl,
        return_pct=return_pct,
        holding_seconds=holding_seconds,
        exit_reason=exit_reason,
        strategy_name=position.strategy_name,
        regime=position.regime,
        entry_cost_pct=position.entry_cost_pct,
        exit_cost_pct=exit_cost_pct,
    )


def estimated_entry_cost_pct(row: pd.Series, settings: Settings) -> float:
    return _estimated_side_cost_pct(row, settings)


def estimated_exit_cost_pct(row: pd.Series, settings: Settings) -> float:
    return _estimated_side_cost_pct(row, settings)


def save_scalping_replay_report(report: dict[str, Any], output_dir: str | Path = "reports") -> Path:
    generated_at = _parse_datetime(report.get("generated_at")) or datetime.now(UTC)
    output_path = Path(output_dir) / f"scalping_replay_{generated_at.strftime('%Y%m%d_%H%M%S')}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report["output_path"] = str(output_path)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def format_scalping_replay_report(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    lines = [
        "BTC/USD scalping replay",
        "",
        f"database: {report['db_path']}",
        f"window_hours: {_format_number(report['hours'])}",
        f"source_table: {report['source'].get('table') or 'n/a'}",
        f"bars_loaded: {report['source'].get('rows_loaded', 0)}",
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
    lines.extend(
        [
            "Summary",
            _format_table(
                ("Metric", "Value"),
                [
                    ("number of trades", _format_count(metrics["number_of_trades"])),
                    ("win rate", _format_pct(metrics["win_rate"])),
                    ("average win", _format_money(metrics["average_win"])),
                    ("average loss", _format_money(metrics["average_loss"])),
                    ("expectancy", _format_money(metrics["expectancy"])),
                    ("total simulated PnL", _format_money(metrics["total_simulated_pnl"])),
                    ("max drawdown", _format_money(metrics["max_drawdown"])),
                    ("average holding time", _format_seconds(metrics["average_holding_time_seconds"])),
                    ("max open positions", _format_count(metrics["max_open_positions"])),
                ],
            ),
            "",
            "Exit Reason Breakdown",
            _format_breakdown(report["breakdowns"]["exit_reasons"]),
            "",
            "Strategy Breakdown",
            _format_breakdown(report["breakdowns"]["strategies"]),
            "",
            "Regime Breakdown",
            _format_breakdown(report["breakdowns"]["regimes"]),
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a lightweight local BTC/USD scalping replay.")
    parser.add_argument("--db", required=True, help="Path to the local SQLite trading database.")
    parser.add_argument("--hours", type=float, default=72, help="Lookback window in hours. Defaults to 72.")
    parser.add_argument("--output-dir", default="reports", help="Directory for the JSON report. Defaults to reports/.")
    args = parser.parse_args(argv)
    if args.hours <= 0:
        parser.error("--hours must be greater than 0")
    report = build_scalping_replay_report(args.db, hours=args.hours)
    save_scalping_replay_report(report, args.output_dir)
    print(format_scalping_replay_report(report))
    return 0


def _report(
    *,
    db_path: str | Path,
    hours: float,
    settings: Settings,
    generated_at: datetime,
    source: dict[str, Any],
    bars: pd.DataFrame,
    trades: list[ReplayTrade],
    warnings: list[str],
    max_open_positions: int,
    open_position_at_end: bool,
) -> dict[str, Any]:
    metrics = replay_metrics(trades, max_open_positions=max_open_positions)
    return {
        "generated_at": generated_at.isoformat(),
        "db_path": str(db_path),
        "hours": float(hours),
        "source": source,
        "settings": {
            "symbol": settings.symbol,
            "order_notional_usd": float(settings.order_notional_usd),
            "scalping_take_profit_pct": float(settings.scalping_take_profit_pct),
            "scalping_stop_loss_pct": float(settings.scalping_stop_loss_pct),
            "scalping_trailing_stop_pct": float(settings.scalping_trailing_stop_pct),
            "scalping_max_position_seconds": int(settings.scalping_max_position_seconds),
        },
        "warnings": _dedupe(warnings),
        "metrics": metrics,
        "breakdowns": {
            "exit_reasons": _counter_breakdown([trade.exit_reason for trade in trades]),
            "strategies": _counter_breakdown([trade.strategy_name or "(unknown)" for trade in trades]),
            "regimes": _counter_breakdown([trade.regime or "(unknown)" for trade in trades]),
        },
        "trades": [asdict(trade) for trade in trades],
        "bar_count": int(len(bars)),
        "open_position_at_end": open_position_at_end,
        "note": (
            "Lightweight local replay only. This reads SQLite data, uses current config values, "
            "and does not change live trading behavior or replace production backtesting."
        ),
    }


def replay_metrics(trades: list[ReplayTrade], *, max_open_positions: int) -> dict[str, Any]:
    pnls = [trade.pnl for trade in trades]
    wins = [pnl for pnl in pnls if pnl > 0]
    losses = [pnl for pnl in pnls if pnl < 0]
    total = len(trades)
    cumulative = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for pnl in pnls:
        cumulative += pnl
        peak = max(peak, cumulative)
        max_drawdown = min(max_drawdown, cumulative - peak)
    win_rate = len(wins) / total if total else None
    average_win = mean(wins) if wins else None
    average_loss = mean(losses) if losses else None
    loss_rate = len(losses) / total if total else 0.0
    expectancy = None
    if total:
        expectancy = (len(wins) / total) * (average_win or 0.0) + loss_rate * (average_loss or 0.0)
    return {
        "number_of_trades": total,
        "win_rate": win_rate,
        "average_win": average_win,
        "average_loss": average_loss,
        "expectancy": expectancy,
        "total_simulated_pnl": sum(pnls),
        "max_drawdown": abs(max_drawdown),
        "average_holding_time_seconds": mean([trade.holding_seconds for trade in trades]) if trades else None,
        "max_open_positions": max_open_positions,
    }


def _estimated_side_cost_pct(row: pd.Series, settings: Settings) -> float:
    spread_bps = _safe_float(row.get("spread_bps"))
    if spread_bps is None:
        spread_bps = float(settings.max_spread_bps)
    bps = (max(0.0, spread_bps) / 2) + max(0.0, float(settings.paper_slippage_bps)) + max(0.0, float(settings.paper_fee_bps))
    return bps / 10_000


def _spread_bps_series(bars: pd.DataFrame) -> pd.Series:
    if "spread_bps" in bars.columns:
        return pd.to_numeric(bars["spread_bps"], errors="coerce")
    if {"bid", "ask"} <= set(bars.columns):
        bid = pd.to_numeric(bars["bid"], errors="coerce")
        ask = pd.to_numeric(bars["ask"], errors="coerce")
        mid = (bid + ask) / 2
        return ((ask - bid) / mid.replace(0, math.nan)) * 10_000
    return pd.Series(math.nan, index=bars.index)


def _quote_imbalance_series(bars: pd.DataFrame) -> pd.Series:
    if "quote_imbalance" in bars.columns:
        return pd.to_numeric(bars["quote_imbalance"], errors="coerce")
    if {"bid_size", "ask_size"} <= set(bars.columns):
        bid_size = pd.to_numeric(bars["bid_size"], errors="coerce")
        ask_size = pd.to_numeric(bars["ask_size"], errors="coerce")
        total = bid_size + ask_size
        return (bid_size - ask_size) / total.replace(0, math.nan)
    return pd.Series(math.nan, index=bars.index)


def _quote_from_row(row: pd.Series) -> dict[str, Any]:
    close = _safe_float(row.get("close"))
    spread_bps = _safe_float(row.get("spread_bps"))
    bid = _safe_float(row.get("bid"))
    ask = _safe_float(row.get("ask"))
    if (bid is None or ask is None) and close is not None and spread_bps is not None:
        half_spread = close * (spread_bps / 10_000) / 2
        bid = close - half_spread
        ask = close + half_spread
    return {
        "bid_price": bid,
        "ask_price": ask,
        "bid_size": _safe_float(row.get("bid_size")) or 1.0,
        "ask_size": _safe_float(row.get("ask_size")) or 1.0,
    }


def _row_timestamp(row: pd.Series) -> datetime | None:
    value = row.get("timestamp")
    if value is None or pd.isna(value):
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.to_pydatetime().astimezone(UTC)


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


def _first_existing_table(tables: dict[str, list[str]], candidates: tuple[str, ...]) -> str | None:
    lookup = {table.lower(): table for table in tables}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def _is_one_minute_timeframe(value: Any) -> bool:
    normalized = str(value or "").strip().lower()
    return normalized in {"1min", "1m", "1", "one_minute"}


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return _ensure_utc(parsed)


def _counter_breakdown(values: list[str]) -> list[dict[str, Any]]:
    if not values:
        return []
    total = len(values)
    counter = Counter(values)
    return [
        {"value": value, "count": count, "pct": count / total}
        for value, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _format_breakdown(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "  n/a"
    return _format_table(
        ("Value", "Count", "Pct"),
        [(str(row["value"]), _format_count(row["count"]), _format_pct(row["pct"])) for row in rows],
    )


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


def _format_money(value: Any) -> str:
    parsed = _safe_float(value)
    if parsed is None:
        return "n/a"
    return f"{parsed:.6f}".rstrip("0").rstrip(".")


def _format_number(value: Any) -> str:
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
