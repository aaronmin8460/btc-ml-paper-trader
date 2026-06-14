from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from typing import Any

from app.config import Settings, get_settings


EPSILON = 1e-12
MIN_TP_COST_MULTIPLE = 1.2
MIN_TP_SL_RATIO = 1.2
LOCAL_DIAGNOSTIC_MAX_TRADES_PER_HOUR = 10
LOCAL_DIAGNOSTIC_MIN_SECONDS_BETWEEN_TRADES = 60


@dataclass(frozen=True)
class ConfigCheck:
    name: str
    status: str
    explanation: str
    recommended_fix: str
    actual: Any
    expected: Any


def build_config_health_report(settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    execution = execution_settings(settings)
    labels = label_settings(settings)
    round_trip_cost_pct = estimated_local_round_trip_cost_pct(settings)
    checks = [
        label_take_profit_matches_execution(settings),
        label_stop_loss_matches_execution(settings),
        max_holding_time_matches_label_horizon(settings),
        take_profit_covers_local_cost(settings, round_trip_cost_pct=round_trip_cost_pct),
        take_profit_stop_loss_ratio_is_healthy(settings),
        stop_loss_is_not_larger_than_take_profit(settings),
        max_trades_per_hour_is_local_diagnostic_safe(settings),
        min_seconds_between_trades_is_local_diagnostic_safe(settings),
    ]
    warnings = [check for check in checks if check.status == "WARN"]
    return {
        "overall_status": "WARN" if warnings else "PASS",
        "summary": {
            "pass_count": sum(1 for check in checks if check.status == "PASS"),
            "warn_count": len(warnings),
        },
        "execution_settings": execution,
        "label_settings": labels,
        "cost_estimate": {
            "model": "local_paper_fee_slippage_plus_max_spread",
            "paper_fee_bps_per_side": float(settings.paper_fee_bps),
            "paper_slippage_bps_per_side": float(settings.paper_slippage_bps),
            "max_spread_bps": float(settings.max_spread_bps),
            "round_trip_estimated_cost_pct": round_trip_cost_pct,
            "round_trip_estimated_cost_bps": round_trip_cost_pct * 10_000,
            "minimum_take_profit_cost_multiple": MIN_TP_COST_MULTIPLE,
        },
        "checks": [asdict(check) for check in checks],
        "note": (
            "Read-only local config diagnostic. It prints warnings only and does not change trading "
            "behavior, thresholds, credentials, AWS resources, or deployment settings."
        ),
    }


def execution_settings(settings: Settings) -> dict[str, Any]:
    return {
        "TRADING_ENABLED": bool(settings.trading_enabled),
        "AUTO_TRADE_ENABLED": bool(settings.auto_trade_enabled),
        "SCALPING_MODE_ENABLED": bool(settings.scalping_mode_enabled),
        "SYMBOL": settings.symbol,
        "ORDER_NOTIONAL_USD": float(settings.order_notional_usd),
        "ORDER_TYPE": settings.order_type,
        "TIME_IN_FORCE": settings.time_in_force,
        "LIMIT_PRICE_OFFSET_BPS": float(settings.limit_price_offset_bps),
        "MAX_SPREAD_BPS": float(settings.max_spread_bps),
        "MAX_SLIPPAGE_BPS": float(settings.max_slippage_bps),
        "MIN_QUOTE_IMBALANCE": float(settings.min_quote_imbalance),
        "SCALPING_TAKE_PROFIT_PCT": float(settings.scalping_take_profit_pct),
        "SCALPING_STOP_LOSS_PCT": float(settings.scalping_stop_loss_pct),
        "SCALPING_TRAILING_STOP_PCT": float(settings.scalping_trailing_stop_pct),
        "SCALPING_MAX_POSITION_SECONDS": int(settings.scalping_max_position_seconds),
        "SCALPING_MIN_HOLD_SECONDS": int(settings.scalping_min_hold_seconds),
        "MAX_TRADES_PER_HOUR": int(settings.max_trades_per_hour),
        "MIN_SECONDS_BETWEEN_TRADES": int(settings.min_seconds_between_trades),
    }


def label_settings(settings: Settings) -> dict[str, Any]:
    return {
        "SCALPING_LABEL_HORIZON_BARS": int(settings.scalping_label_horizon_bars),
        "SCALPING_LABEL_TAKE_PROFIT_PCT": float(settings.scalping_label_take_profit_pct),
        "SCALPING_LABEL_STOP_LOSS_PCT": float(settings.scalping_label_stop_loss_pct),
        "SCALPING_LABEL_MIN_NET_PROFIT_PCT": float(settings.scalping_label_min_net_profit_pct),
    }


def estimated_local_round_trip_cost_pct(settings: Settings) -> float:
    return (
        2 * max(0.0, float(settings.paper_fee_bps)) / 10_000
        + 2 * max(0.0, float(settings.paper_slippage_bps)) / 10_000
        + max(0.0, float(settings.max_spread_bps)) / 10_000
    )


def label_take_profit_matches_execution(settings: Settings) -> ConfigCheck:
    execution_value = float(settings.scalping_take_profit_pct)
    label_value = float(settings.scalping_label_take_profit_pct)
    passed = _close(label_value, execution_value)
    return ConfigCheck(
        name="label_take_profit_matches_execution",
        status=_status(passed),
        explanation=(
            "ML labels and execution use the same take-profit target."
            if passed
            else "ML labels are training on a different take-profit target than execution uses."
        ),
        recommended_fix=(
            "No change needed."
            if passed
            else "Set SCALPING_LABEL_TAKE_PROFIT_PCT equal to SCALPING_TAKE_PROFIT_PCT."
        ),
        actual={
            "SCALPING_LABEL_TAKE_PROFIT_PCT": label_value,
            "SCALPING_TAKE_PROFIT_PCT": execution_value,
        },
        expected="equal values",
    )


def label_stop_loss_matches_execution(settings: Settings) -> ConfigCheck:
    execution_value = float(settings.scalping_stop_loss_pct)
    label_value = float(settings.scalping_label_stop_loss_pct)
    passed = _close(label_value, execution_value)
    return ConfigCheck(
        name="label_stop_loss_matches_execution",
        status=_status(passed),
        explanation=(
            "ML labels and execution use the same stop-loss target."
            if passed
            else "ML labels are training on a different stop-loss target than execution uses."
        ),
        recommended_fix=(
            "No change needed."
            if passed
            else "Set SCALPING_LABEL_STOP_LOSS_PCT equal to SCALPING_STOP_LOSS_PCT."
        ),
        actual={
            "SCALPING_LABEL_STOP_LOSS_PCT": label_value,
            "SCALPING_STOP_LOSS_PCT": execution_value,
        },
        expected="equal values",
    )


def max_holding_time_matches_label_horizon(settings: Settings) -> ConfigCheck:
    max_position_seconds = int(settings.scalping_max_position_seconds)
    recommended_max_seconds = int(settings.scalping_label_horizon_bars) * 60 * 2
    passed = max_position_seconds <= recommended_max_seconds
    return ConfigCheck(
        name="max_holding_time_matches_label_horizon",
        status=_status(passed),
        explanation=(
            "Execution max hold is within 2x the ML label horizon."
            if passed
            else "Execution may hold positions much longer than the horizon used to train scalping labels."
        ),
        recommended_fix=(
            "No change needed."
            if passed
            else "Lower SCALPING_MAX_POSITION_SECONDS or increase the label horizon only after updating labels/retraining."
        ),
        actual={"SCALPING_MAX_POSITION_SECONDS": max_position_seconds},
        expected=f"<= {recommended_max_seconds} seconds",
    )


def take_profit_covers_local_cost(settings: Settings, *, round_trip_cost_pct: float | None = None) -> ConfigCheck:
    take_profit_pct = float(settings.scalping_take_profit_pct)
    cost_pct = estimated_local_round_trip_cost_pct(settings) if round_trip_cost_pct is None else round_trip_cost_pct
    required_pct = cost_pct * MIN_TP_COST_MULTIPLE
    passed = take_profit_pct > required_pct
    return ConfigCheck(
        name="take_profit_covers_local_cost",
        status=_status(passed),
        explanation=(
            "Configured take profit has room above estimated local paper round-trip cost."
            if passed
            else "Configured take profit is too close to, or below, estimated local paper round-trip cost."
        ),
        recommended_fix=(
            "No change needed."
            if passed
            else "Increase SCALPING_TAKE_PROFIT_PCT or reduce local paper fee/slippage/spread assumptions before testing."
        ),
        actual={
            "SCALPING_TAKE_PROFIT_PCT": take_profit_pct,
            "round_trip_estimated_cost_pct": cost_pct,
            "take_profit_to_cost_multiple": take_profit_pct / cost_pct if cost_pct > 0 else None,
        },
        expected=f"> {MIN_TP_COST_MULTIPLE:.2f}x estimated round-trip cost ({required_pct:.6f})",
    )


def take_profit_stop_loss_ratio_is_healthy(settings: Settings) -> ConfigCheck:
    take_profit_pct = float(settings.scalping_take_profit_pct)
    stop_loss_pct = float(settings.scalping_stop_loss_pct)
    ratio = take_profit_pct / stop_loss_pct if stop_loss_pct > 0 else None
    passed = ratio is not None and ratio >= MIN_TP_SL_RATIO
    return ConfigCheck(
        name="take_profit_stop_loss_ratio_is_healthy",
        status=_status(passed),
        explanation=(
            "TP/SL ratio is large enough for this local diagnostic guardrail."
            if passed
            else "TP/SL ratio is too compressed; small fees, spread, or missed fills can overwhelm edge."
        ),
        recommended_fix=(
            "No change needed."
            if passed
            else "Raise SCALPING_TAKE_PROFIT_PCT, lower SCALPING_STOP_LOSS_PCT, or revisit the strategy target."
        ),
        actual={"tp_sl_ratio": ratio},
        expected=f">= {MIN_TP_SL_RATIO:.2f}",
    )


def stop_loss_is_not_larger_than_take_profit(settings: Settings) -> ConfigCheck:
    take_profit_pct = float(settings.scalping_take_profit_pct)
    stop_loss_pct = float(settings.scalping_stop_loss_pct)
    passed = stop_loss_pct <= take_profit_pct
    return ConfigCheck(
        name="stop_loss_is_not_larger_than_take_profit",
        status=_status(passed),
        explanation=(
            "Stop loss is not larger than take profit."
            if passed
            else "Stop loss is larger than take profit, so losses can outweigh wins before costs."
        ),
        recommended_fix=(
            "No change needed."
            if passed
            else "Set SCALPING_STOP_LOSS_PCT at or below SCALPING_TAKE_PROFIT_PCT."
        ),
        actual={
            "SCALPING_STOP_LOSS_PCT": stop_loss_pct,
            "SCALPING_TAKE_PROFIT_PCT": take_profit_pct,
        },
        expected="SCALPING_STOP_LOSS_PCT <= SCALPING_TAKE_PROFIT_PCT",
    )


def max_trades_per_hour_is_local_diagnostic_safe(settings: Settings) -> ConfigCheck:
    value = int(settings.max_trades_per_hour)
    passed = value <= LOCAL_DIAGNOSTIC_MAX_TRADES_PER_HOUR
    return ConfigCheck(
        name="max_trades_per_hour_is_local_diagnostic_safe",
        status=_status(passed),
        explanation=(
            "Hourly trade cap is restrained for local diagnostics."
            if passed
            else "Hourly trade cap is high for local diagnostic mode and can mask overtrading problems."
        ),
        recommended_fix=(
            "No change needed."
            if passed
            else f"Set MAX_TRADES_PER_HOUR to {LOCAL_DIAGNOSTIC_MAX_TRADES_PER_HOUR} or lower for local diagnostics."
        ),
        actual={"MAX_TRADES_PER_HOUR": value},
        expected=f"<= {LOCAL_DIAGNOSTIC_MAX_TRADES_PER_HOUR}",
    )


def min_seconds_between_trades_is_local_diagnostic_safe(settings: Settings) -> ConfigCheck:
    value = int(settings.min_seconds_between_trades)
    passed = value >= LOCAL_DIAGNOSTIC_MIN_SECONDS_BETWEEN_TRADES
    return ConfigCheck(
        name="min_seconds_between_trades_is_local_diagnostic_safe",
        status=_status(passed),
        explanation=(
            "Trade cooldown is long enough for local diagnostics."
            if passed
            else "Trade cooldown is short for local diagnostics and may allow rapid-fire churn."
        ),
        recommended_fix=(
            "No change needed."
            if passed
            else f"Set MIN_SECONDS_BETWEEN_TRADES to at least {LOCAL_DIAGNOSTIC_MIN_SECONDS_BETWEEN_TRADES}."
        ),
        actual={"MIN_SECONDS_BETWEEN_TRADES": value},
        expected=f">= {LOCAL_DIAGNOSTIC_MIN_SECONDS_BETWEEN_TRADES}",
    )


def format_config_health_report(report: dict[str, Any]) -> str:
    lines = [
        "BTC/USD scalping config health",
        "",
        f"overall_status: {report['overall_status']}",
        report["note"],
        "",
        "Current scalping execution settings",
        _format_key_values(report["execution_settings"]),
        "",
        "Current scalping ML label settings",
        _format_key_values(report["label_settings"]),
        "",
        "Estimated local round-trip cost",
        _format_key_values(
            {
                "model": report["cost_estimate"]["model"],
                "round_trip_estimated_cost_pct": _format_pct(
                    report["cost_estimate"]["round_trip_estimated_cost_pct"]
                ),
                "round_trip_estimated_cost_bps": _format_number(
                    report["cost_estimate"]["round_trip_estimated_cost_bps"]
                ),
            }
        ),
        "",
        "Checks",
        _format_checks(report["checks"]),
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check local BTC scalping config health.")
    parser.add_argument("--json", action="store_true", help="Print the config health report as JSON.")
    args = parser.parse_args(argv)
    report = build_config_health_report()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_config_health_report(report))
    return 0


def _status(passed: bool) -> str:
    return "PASS" if passed else "WARN"


def _close(left: float, right: float) -> bool:
    return abs(left - right) <= EPSILON


def _format_key_values(values: dict[str, Any]) -> str:
    width = max(len(str(key)) for key in values) if values else 0
    return "\n".join(f"  {key.ljust(width)} : {value}" for key, value in values.items())


def _format_checks(checks: list[dict[str, Any]]) -> str:
    rows = []
    for check in checks:
        rows.append(
            (
                check["status"],
                check["name"],
                check["explanation"],
                check["recommended_fix"],
            )
        )
    return _format_table(("Status", "Check", "Explanation", "Recommended Fix"), rows)


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


def _format_number(value: Any) -> str:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if parsed == 0:
        return "0"
    if abs(parsed) >= 100:
        return f"{parsed:.2f}"
    return f"{parsed:.6f}".rstrip("0").rstrip(".")


def _format_pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.4f}%"
    except (TypeError, ValueError):
        return "n/a"


if __name__ == "__main__":
    raise SystemExit(main())
