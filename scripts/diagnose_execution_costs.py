from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from app.backtest.scalping import (
    estimated_round_trip_execution_cost_pct,
    execution_cost_breakdown,
    minimum_take_profit_net_positive_pct,
    promotion_required_return_pct,
)
from app.config import Settings, get_settings
from app.data.market_data import MarketDataClient


HISTORICAL_WINDOWS = {
    "1m": 1,
    "3m": 3,
    "6m": 6,
}


def build_execution_cost_report(bars: pd.DataFrame, settings: Settings) -> dict[str, Any]:
    round_trip_cost = estimated_round_trip_execution_cost_pct(settings)
    minimum_take_profit = minimum_take_profit_net_positive_pct(settings)
    promotion_required_return = promotion_required_return_pct(settings)
    cost_breakdown = execution_cost_breakdown(settings)
    return {
        "symbol": settings.symbol,
        "timeframe": "1Min",
        "bar_count": int(len(bars)),
        "taker_fee_bps": float(settings.taker_fee_bps),
        "maker_fee_bps": float(settings.maker_fee_bps),
        "slippage_bps": float(settings.slippage_bps),
        "max_spread_bps": float(settings.max_spread_bps),
        "fee_model": cost_breakdown["fee_model"],
        "round_trip_estimated_cost_pct": round_trip_cost,
        "round_trip_estimated_cost_bps": round_trip_cost * 10_000,
        "scalping_take_profit_pct": float(settings.scalping_take_profit_pct),
        "scalping_stop_loss_pct": float(settings.scalping_stop_loss_pct),
        "training_label_take_profit_pct": float(settings.scalping_label_take_profit_pct),
        "promotion_required_return_pct": promotion_required_return,
        "promotion_required_return_bps": promotion_required_return * 10_000,
        "minimum_take_profit_net_positive_pct": minimum_take_profit,
        "minimum_take_profit_net_positive_bps": minimum_take_profit * 10_000,
        "scalping_take_profit_is_unsafe": take_profit_is_unsafe(
            settings.scalping_take_profit_pct,
            round_trip_cost,
        ),
        "training_label_take_profit_is_unsafe": take_profit_is_unsafe(
            settings.scalping_label_take_profit_pct,
            round_trip_cost,
        ),
        "historical_window_clear_rates": historical_window_clear_rates(
            bars,
            minimum_take_profit_pct=minimum_take_profit,
            promotion_required_return_pct=promotion_required_return,
        ),
        "note": (
            "This is a diagnostic only. It does not lower promotion costs, enable trading, "
            "or claim profitability."
        ),
    }


def take_profit_is_unsafe(take_profit_pct: float, round_trip_cost_pct: float) -> bool:
    return float(take_profit_pct) <= float(round_trip_cost_pct)


def historical_window_clear_rates(
    bars: pd.DataFrame,
    *,
    minimum_take_profit_pct: float,
    promotion_required_return_pct: float,
) -> dict[str, dict[str, Any]]:
    return {
        label: _window_clear_rate(
            bars,
            window_bars=window_bars,
            minimum_take_profit_pct=minimum_take_profit_pct,
            promotion_required_return_pct=promotion_required_return_pct,
        )
        for label, window_bars in HISTORICAL_WINDOWS.items()
    }


def format_execution_cost_report(report: dict[str, Any]) -> str:
    lines = [
        "BTC/USD execution cost diagnosis",
        "",
        f"symbol: {report['symbol']}",
        f"timeframe: {report['timeframe']}",
        f"bars: {report['bar_count']}",
        f"fee_model: {report['fee_model']}",
        f"taker_fee_bps: {report['taker_fee_bps']}",
        f"maker_fee_bps: {report['maker_fee_bps']}",
        f"slippage_bps: {report['slippage_bps']}",
        f"max_spread_bps: {report['max_spread_bps']}",
        f"round-trip estimated cost: {_format_pct(report['round_trip_estimated_cost_pct'])} "
        f"({report['round_trip_estimated_cost_bps']:.2f} bps)",
        f"scalping_take_profit_pct: {_format_pct(report['scalping_take_profit_pct'])}",
        f"scalping_stop_loss_pct: {_format_pct(report['scalping_stop_loss_pct'])}",
        f"training label take profit: {_format_pct(report['training_label_take_profit_pct'])}",
        f"promotion required return: {_format_pct(report['promotion_required_return_pct'])} "
        f"({report['promotion_required_return_bps']:.2f} bps)",
        f"minimum take profit needed to be net positive: "
        f"{_format_pct(report['minimum_take_profit_net_positive_pct'])} "
        f"({report['minimum_take_profit_net_positive_bps']:.2f} bps)",
        f"scalping take profit unsafe vs round-trip cost: {report['scalping_take_profit_is_unsafe']}",
        f"training label take profit unsafe vs round-trip cost: {report['training_label_take_profit_is_unsafe']}",
        "",
        "Historical windows that can clear required returns:",
    ]
    for label, values in report["historical_window_clear_rates"].items():
        lines.append(
            "  "
            f"{label}: "
            f"net-positive={_format_pct(values['clear_minimum_take_profit_pct'])} "
            f"({values['clear_minimum_take_profit_count']}/{values['eligible_windows']}), "
            f"promotion={_format_pct(values['clear_promotion_required_return_pct'])} "
            f"({values['clear_promotion_required_return_count']}/{values['eligible_windows']})"
        )
    lines.extend(["", report["note"], "", "JSON summary:", json.dumps(report, indent=2, default=str)])
    return "\n".join(lines)


async def main() -> None:
    settings = get_settings()
    bars = await MarketDataClient(settings).fetch_bars(
        settings.symbol,
        timeframe="1Min",
        limit=max(settings.lookback_bars, settings.min_training_rows + 300),
    )
    report = build_execution_cost_report(bars, settings)
    print(format_execution_cost_report(report))


def _window_clear_rate(
    bars: pd.DataFrame,
    *,
    window_bars: int,
    minimum_take_profit_pct: float,
    promotion_required_return_pct: float,
) -> dict[str, Any]:
    if bars.empty or window_bars <= 0:
        return _empty_window_rate(window_bars)
    close = pd.to_numeric(bars.get("close"), errors="coerce")
    high = pd.to_numeric(bars.get("high", close), errors="coerce")
    future_returns: list[float] = []
    for index in range(0, len(bars) - window_bars):
        entry_close = close.iloc[index]
        if pd.isna(entry_close) or entry_close <= 0:
            continue
        future_high = high.iloc[index + 1 : index + 1 + window_bars].max()
        if pd.isna(future_high):
            continue
        future_returns.append(float(future_high / entry_close - 1))

    eligible = len(future_returns)
    if eligible == 0:
        return _empty_window_rate(window_bars)
    minimum_count = sum(value >= minimum_take_profit_pct for value in future_returns)
    promotion_count = sum(value >= promotion_required_return_pct for value in future_returns)
    return {
        "window_bars": int(window_bars),
        "eligible_windows": int(eligible),
        "clear_minimum_take_profit_count": int(minimum_count),
        "clear_minimum_take_profit_pct": float(minimum_count / eligible),
        "clear_promotion_required_return_count": int(promotion_count),
        "clear_promotion_required_return_pct": float(promotion_count / eligible),
        "max_forward_return_pct": float(max(future_returns)),
        "average_forward_return_pct": float(sum(future_returns) / eligible),
    }


def _empty_window_rate(window_bars: int) -> dict[str, Any]:
    return {
        "window_bars": int(window_bars),
        "eligible_windows": 0,
        "clear_minimum_take_profit_count": 0,
        "clear_minimum_take_profit_pct": 0.0,
        "clear_promotion_required_return_count": 0,
        "clear_promotion_required_return_pct": 0.0,
        "max_forward_return_pct": 0.0,
        "average_forward_return_pct": 0.0,
    }


def _format_pct(value: float) -> str:
    return f"{float(value) * 100:.4f}%"


if __name__ == "__main__":
    asyncio.run(main())
