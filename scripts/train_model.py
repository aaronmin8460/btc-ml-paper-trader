import asyncio
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.data.market_data import MarketDataClient
from app.ml.train import train_model_from_bars
from app.ml.training_diagnostics import next_recommended_action


async def main() -> None:
    settings = get_settings()
    bars = await MarketDataClient(settings).fetch_bars(settings.symbol, limit=max(settings.lookback_bars, settings.min_training_rows + 300))
    result = train_model_from_bars(bars, settings)
    print(format_training_report(result, settings))


def format_training_report(result: dict[str, Any], settings) -> str:
    metrics = result.get("metrics") or {}
    promotion_reason = metrics.get("promotion_reason") or result.get("reason")
    rejection_reason = None if result.get("accepted") else result.get("reason")
    fee_valid = metrics.get("fee_aware_backtest_valid")
    fee_reason = metrics.get("fee_aware_backtest_reason")
    lines = [
        "BTC/USD model training report",
        f"accepted: {result.get('accepted')}",
        f"rejection reason: {rejection_reason or 'none'}",
        f"promotion reason: {promotion_reason or 'none'}",
        f"raw bars: {metrics.get('raw_bars', 'unknown')}",
        f"trainable rows: {metrics.get('trainable_rows', metrics.get('rows', 'unknown'))}",
        f"buy positive label count: {metrics.get('buy_positive_label_count', 'unknown')}",
        f"buy positive label pct: {_format_pct(metrics.get('buy_positive_label_pct'))}",
        f"sell positive label count: {metrics.get('sell_positive_label_count', 'unknown')}",
        "top NaN columns:",
        *_format_top_nan_columns(metrics.get("top_nan_columns") or []),
        f"fee-aware backtest valid: {fee_valid if fee_valid is not None else 'not_run'}",
        f"fee-aware backtest valid/invalid reason: {fee_reason or ('valid' if fee_valid else 'not_run')}",
        f"net_return_pct: {_format_number(metrics.get('net_return_pct'))}",
        f"profit_factor_net: {_format_number(metrics.get('profit_factor_net'))}",
        f"number_of_trades: {metrics.get('number_of_trades', 'not_run')}",
        f"max_drawdown_pct: {_format_number(metrics.get('max_drawdown_pct'))}",
        f"ambiguous_candle_ratio: {_format_number(metrics.get('ambiguous_candle_ratio'))}",
        f"model_path: {result.get('model_path') or 'none'}",
        f"next recommended action: {next_recommended_action(result, settings)}",
    ]
    return "\n".join(lines)


def _format_pct(value: object) -> str:
    if value is None:
        return "unknown"
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "unknown"


def _format_number(value: object) -> str:
    if value is None:
        return "not_run"
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def _format_top_nan_columns(top_nan_columns: list[dict[str, Any]]) -> list[str]:
    if not top_nan_columns:
        return ["  none"]
    return [f"  {item['column']}: {item['nan_count']}" for item in top_nan_columns]


if __name__ == "__main__":
    asyncio.run(main())
