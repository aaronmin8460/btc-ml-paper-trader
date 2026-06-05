from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from app.backtest.scalping import backtest_assumptions
from app.config import Settings, get_settings
from app.data.market_data import MarketDataClient
from app.ml.training_diagnostics import (
    build_training_dataset_with_diagnostics,
    buy_positive_label_warning,
    label_config_comparison,
)


def build_diagnosis_report(bars: pd.DataFrame, settings: Settings) -> dict[str, Any]:
    _, summary = build_training_dataset_with_diagnostics(bars, settings)
    return {
        "settings": settings.safe_dict(),
        "summary": summary,
        "conservative_backtest_assumptions": backtest_assumptions(
            settings,
            spread_available=bool(summary.get("training_label_assumptions", {}).get("spread_bps", 0)),
        ),
        "label_config_comparison": label_config_comparison(bars, settings),
        "warning": buy_positive_label_warning(summary, settings),
    }


def format_diagnosis_report(report: dict[str, Any], settings: Settings) -> str:
    summary = report["summary"]
    lines = [
        "BTC/USD label diagnosis",
        "",
        "Resolved settings:",
        json.dumps(report["settings"], indent=2, sort_keys=True, default=str),
        "",
        f"SYMBOL={settings.symbol}",
        f"TIMEFRAME={settings.timeframe}",
        f"SCALPING_MODE_ENABLED={settings.scalping_mode_enabled}",
        f"LOOKBACK_BARS={settings.lookback_bars}",
        f"MIN_TRAINING_ROWS={settings.min_training_rows}",
        f"LABEL_HORIZON_BARS={settings.label_horizon_bars}",
        f"LABEL_FEE_BPS_PER_SIDE={settings.label_fee_bps_per_side}",
        f"LABEL_SLIPPAGE_BPS_PER_SIDE={settings.label_slippage_bps_per_side}",
        f"LABEL_SPREAD_BPS={settings.label_spread_bps}",
        f"LABEL_MIN_NET_PROFIT_PCT={settings.label_min_net_profit_pct}",
        "",
        "Label semantics:",
        "sell_quality_label is a compatibility alias for exit_quality_label. It means exit an existing long BTC/USD position; it is not a short-entry label.",
        "",
        "Current production label assumptions:",
        _format_assumptions(summary["current_production_label_assumptions"]),
        "",
        "Proposed training label assumptions:",
        _format_assumptions(summary["training_label_assumptions"]),
        "",
        "Conservative promotion/backtest assumptions:",
        _format_backtest_assumptions(report["conservative_backtest_assumptions"]),
        "",
        f"raw bars count: {summary['raw_bars']}",
        f"first timestamp: {summary['first_timestamp']}",
        f"latest timestamp: {summary['latest_timestamp']}",
        f"featured rows: {summary['featured_rows']}",
        f"labeled rows: {summary['labeled_rows']}",
        f"trainable rows after dropna: {summary['trainable_rows']}",
        f"buy_quality_label distribution: {_format_distribution(summary['buy_quality_label_distribution'])}",
        f"exit_quality_label distribution: {_format_distribution(summary['exit_quality_label_distribution'])}",
        f"sell_quality_label alias distribution: {_format_distribution(summary['sell_quality_label_distribution'])}",
        f"buy positive count: {summary['buy_positive_label_count']}",
        f"buy positive percentage: {_format_pct(summary['buy_positive_label_pct'])}",
        f"exit/sell positive count: {summary['exit_positive_label_count']}",
        f"exit/sell positive percentage: {_format_pct(summary['exit_positive_label_pct'])}",
        f"buy/sell imbalance ratio: {_format_ratio(summary['buy_sell_imbalance_ratio'])}",
        "buy_exit_reason distribution:",
        _format_reason_distribution(summary["buy_exit_reason_distribution"]),
        "",
        "top NaN columns among required training columns:",
    ]
    lines.extend(_format_top_nan_columns(summary["top_nan_columns"]))
    lines.extend(
        [
            "",
            "estimated required exit return using proposed training label assumptions: "
            f"{summary['training_estimated_required_exit_return_pct']:.6f} "
            f"({_format_pct(summary['training_estimated_required_exit_return_pct'])})",
            "estimated required exit return using conservative promotion assumptions: "
            f"{summary['conservative_promotion_estimated_required_exit_return_pct']:.6f} "
            f"({_format_pct(summary['conservative_promotion_estimated_required_exit_return_pct'])})",
            "estimated required exit return using current production label assumptions: "
            f"{summary['current_production_estimated_required_exit_return_pct']:.6f} "
            f"({_format_pct(summary['current_production_estimated_required_exit_return_pct'])})",
        ]
    )
    if report.get("warning"):
        lines.append(f"WARNING: {report['warning']}")
    lines.extend(
        [
            "",
            "Label configuration comparison:",
            _format_comparison_table(report["label_config_comparison"]),
            "",
            "Note: comparison configs are diagnostics only; they do not enable trading or promotion.",
        ]
    )
    return "\n".join(lines)


async def main() -> None:
    settings = get_settings()
    bars = await MarketDataClient(settings).fetch_bars(
        settings.symbol,
        limit=max(settings.lookback_bars, settings.min_training_rows + 300),
    )
    report = build_diagnosis_report(bars, settings)
    print(format_diagnosis_report(report, settings))


def _format_distribution(distribution: dict[int, int]) -> str:
    return "{" + ", ".join(f"{label}: {distribution.get(label, 0)}" for label in sorted(distribution)) + "}"


def _format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _format_top_nan_columns(top_nan_columns: list[dict[str, Any]]) -> list[str]:
    if not top_nan_columns:
        return ["  none"]
    return [f"  {item['column']}: {item['nan_count']}" for item in top_nan_columns]


def _format_assumptions(values: dict[str, Any]) -> str:
    keys = [
        "name",
        "horizon_bars",
        "take_profit_pct",
        "stop_loss_pct",
        "fee_bps_per_side",
        "slippage_bps_per_side",
        "spread_bps",
        "min_net_exit_profit_pct",
        "exit_profit_buffer_bps",
        "estimated_required_exit_return_pct",
    ]
    return "\n".join(f"  {key}: {values.get(key)}" for key in keys)


def _format_backtest_assumptions(values: dict[str, Any]) -> str:
    keys = [
        "fee_model",
        "fee_bps_per_side",
        "taker_fee_bps",
        "slippage_bps_per_side",
        "spread_source",
        "execution_model",
        "ambiguous_candle_behavior",
        "order_notional_usd",
        "return_metrics_unit",
    ]
    return "\n".join(f"  {key}: {values.get(key)}" for key in keys)


def _format_reason_distribution(distribution: dict[str, int]) -> str:
    if not distribution:
        return "  none"
    return "\n".join(f"  {reason}: {count}" for reason, count in sorted(distribution.items()))


def _format_ratio(value: object) -> str:
    if value is None:
        return "undefined"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "undefined"


def _format_comparison_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "config",
        "horizon_bars",
        "required_exit_return",
        "trainable_rows",
        "buy_1",
        "buy_1_pct",
        "exit_1",
        "exit_1_pct",
        "buy_sell_ratio",
    ]
    formatted_rows = [
        {
            **row,
            "buy_1_pct": _format_pct(float(row["buy_1_pct"])),
            "exit_1_pct": _format_pct(float(row["exit_1_pct"])),
            "required_exit_return": _format_pct(float(row["estimated_required_exit_return_pct"])),
            "buy_sell_ratio": _format_ratio(row["buy_sell_imbalance_ratio"]),
        }
        for row in rows
    ]
    widths = {
        header: max(len(header), *(len(str(row[header])) for row in formatted_rows)) if formatted_rows else len(header)
        for header in headers
    }
    table_lines = ["  " + "  ".join(header.ljust(widths[header]) for header in headers)]
    table_lines.append("  " + "  ".join("-" * widths[header] for header in headers))
    for row in formatted_rows:
        table_lines.append("  " + "  ".join(str(row[header]).ljust(widths[header]) for header in headers))
    return "\n".join(table_lines)


if __name__ == "__main__":
    asyncio.run(main())
