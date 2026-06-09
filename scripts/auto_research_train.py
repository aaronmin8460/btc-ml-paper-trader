from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


ALLOWED_SYMBOL = "BTC/USD"
TIMEFRAMES = ("1Min", "5Min", "15Min")
RESEARCH_TIMEFRAMES = ("5Min", "15Min")
DEFAULT_MIN_ROWS = {"1Min": 5000, "5Min": 1000, "15Min": 1000}
DEFAULT_MAX_AGE_MINUTES = {"1Min": 10.0, "5Min": 10.0, "15Min": 30.0}
REPORT_NAME = "auto_research_train_report.json"
LATEST_REPORT_NAME = "auto_research_train_report_latest.json"

TRUE_VALUES = {"1", "true", "yes", "on", "enabled"}
FALSE_VALUES = {"0", "false", "no", "off", "disabled"}
SHORT_FLAG_NAMES = (
    "SHORT_SELLING_ENABLED",
    "ALLOW_SHORT_SELLING",
    "ENABLE_SHORTS",
    "SHORT_ENABLED",
)
MARGIN_FLAG_NAMES = (
    "MARGIN_ENABLED",
    "ALLOW_MARGIN",
    "ENABLE_MARGIN",
    "USE_MARGIN",
    "LEVERAGE_ENABLED",
)
MULTI_SYMBOL_FLAG_NAMES = (
    "MULTI_SYMBOL_ENABLED",
    "ALLOW_MULTI_SYMBOL",
    "MULTI_ASSET_ENABLED",
)
MULTI_SYMBOL_LIST_NAMES = ("SYMBOLS", "TRADE_SYMBOLS", "WATCHLIST_SYMBOLS", "WATCHLIST")


@dataclass(frozen=True)
class SafetyEvaluation:
    passed: bool
    fatal_reasons: list[str]
    warnings: list[str]
    flags: dict[str, Any]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Safely inspect collected BTC/USD data, run offline research diagnostics, "
            "and optionally run guarded model training without enabling trading."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Inspect only. This is the default.")
    mode.add_argument("--run", action="store_true", help="Run gated diagnostics/research/training.")
    parser.add_argument(
        "--force-research",
        action="store_true",
        help="Run research even when readiness gates are not met; invalid data remains invalid for trading decisions.",
    )
    parser.add_argument("--json", action="store_true", help="Print the full JSON report.")
    return parser.parse_args()


async def main() -> int:
    args = _parse_args()
    mode = "run" if args.run else "dry-run"
    report, exit_code = await build_auto_research_train_report(
        mode=mode,
        force_research=bool(args.force_research),
        explicit_dry_run=bool(args.dry_run or not args.run),
    )
    if args.json:
        print(json.dumps(_json_safe(report), indent=2, sort_keys=True, allow_nan=False))
    else:
        print(_format_text_summary(report))
    return exit_code


async def build_auto_research_train_report(
    *,
    mode: str = "dry-run",
    force_research: bool = False,
    explicit_dry_run: bool = True,
    env: dict[str, str] | None = None,
    env_path: Path | None = None,
    session_factory: Any | None = None,
    now: datetime | None = None,
    write_reports: bool = True,
) -> tuple[dict[str, Any], int]:
    if mode not in {"dry-run", "run"}:
        raise ValueError("mode must be dry-run or run")

    current_time = _utc_datetime(now or datetime.now(UTC))
    env_path = env_path if env_path is not None else ROOT / ".env"
    effective_env = load_effective_env(env=env, env_path=env_path)
    safety = evaluate_environment_safety(
        effective_env,
        inspection_only_dry_run=mode == "dry-run" and explicit_dry_run,
    )

    settings = None
    settings_error = None
    if safety.passed:
        try:
            from app.config import get_settings

            settings = get_settings()
        except Exception as exc:  # pragma: no cover - defensive guard for malformed deployments.
            settings_error = f"{type(exc).__name__}: {exc}"
            fatal_reasons = safety.fatal_reasons + ["settings_validation_failed"]
            safety = SafetyEvaluation(
                passed=False,
                fatal_reasons=fatal_reasons,
                warnings=safety.warnings,
                flags={
                    **safety.flags,
                    "settings_validation_error": settings_error,
                    "safety_gate_passed": False,
                    "fatal_reasons": fatal_reasons,
                },
            )

    if settings is None:
        report = _base_report(
            generated_at=current_time,
            mode=mode,
            force_research=force_research,
            safety=safety,
        )
        report.update(
            {
                "data_readiness_by_timeframe": {},
                "diagnostics_summary": {
                    "error": settings_error or "environment_safety_gate_failed",
                    "active_model_valid": False,
                    "active_model_invalid_reason": "not_inspected",
                },
                "research_summary": _empty_research_summary("not_run_environment_safety_gate_failed"),
                "training_gate_results": {
                    "all_gates_passed": False,
                    "blocked_reasons": list(safety.fatal_reasons),
                },
                "training_was_run": False,
                "train_model_result": {"status": "not_run", "reason": "environment_safety_gate_failed"},
                "backtest_result": {"status": "not_run"},
                "model_registry_status": {
                    "registry_exists": False,
                    "active_model_valid": False,
                    "active_model_invalid_reason": "not_inspected",
                },
            }
        )
        _finish_report(report)
        if write_reports:
            write_report_files(report, log_dir=_log_dir_from_env(effective_env))
        return report, 1

    data_readiness = build_data_readiness_by_timeframe(
        settings,
        effective_env=effective_env,
        session_factory=session_factory,
        now=current_time,
    )
    model_registry_status = inspect_model_registry(settings)
    diagnostics_summary = build_diagnostics_summary(
        settings,
        data_readiness_by_timeframe=data_readiness,
        model_registry_status=model_registry_status,
        session_factory=session_factory,
    )

    research_should_run = should_run_research(data_readiness, force_research=force_research)
    if research_should_run:
        research_summary = await run_research_pipeline(
            settings,
            session_factory=session_factory,
            now=current_time,
        )
    else:
        research_summary = _empty_research_summary("not_run_no_ready_5min_or_15min_data")
    diagnostics_summary = merge_research_into_diagnostics(diagnostics_summary, research_summary)

    training_gates = evaluate_training_gates(
        mode=mode,
        settings=settings,
        safety=safety,
        data_readiness_by_timeframe=data_readiness,
        diagnostics_summary=diagnostics_summary,
        research_summary=research_summary,
    )

    training_was_run = False
    train_model_result: dict[str, Any] = {
        "status": "not_run",
        "reason": "dry_run" if mode == "dry-run" else "training_gates_failed",
        "blocked_reasons": list(training_gates["blocked_reasons"]),
    }
    if mode == "run" and training_gates["all_gates_passed"]:
        training_timeframe = str(settings.timeframe)
        training_rows = load_collected_bars(
            settings,
            timeframe=training_timeframe,
            limit=_training_bar_limit(settings, data_readiness.get(training_timeframe, {})),
            session_factory=session_factory,
        )
        training_was_run = True
        train_model_result = run_train_model(settings, training_rows)
        # Re-read after train_model so promotion status reflects only existing strict logic.
        model_registry_status = inspect_model_registry(settings)

    report = _base_report(
        generated_at=current_time,
        mode=mode,
        force_research=force_research,
        safety=safety,
    )
    report.update(
        {
            "data_readiness_by_timeframe": data_readiness,
            "diagnostics_summary": diagnostics_summary,
            "research_summary": research_summary,
            "training_gate_results": training_gates,
            "training_was_run": training_was_run,
            "train_model_result": train_model_result,
            "backtest_result": build_backtest_result(train_model_result),
            "model_registry_status": model_registry_status,
        }
    )
    _finish_report(report)
    if write_reports:
        write_report_files(report, log_dir=Path(settings.log_dir))
    return report, 0 if safety.passed else 1


def load_effective_env(*, env: dict[str, str] | None = None, env_path: Path | None = None) -> dict[str, str]:
    effective = {
        "PAPER_TRADING_ONLY": "true",
        "TRADING_ENABLED": "false",
        "AUTO_TRADE_ENABLED": "false",
        "ALLOW_FALLBACK_TRADING": "false",
        "SYMBOL": ALLOWED_SYMBOL,
        "MAX_OPEN_POSITIONS": "1",
        "ALPACA_PAPER_BASE_URL": "https://paper-api.alpaca.markets",
        "PAPER_EXECUTION_MODE": "alpaca_paper",
        "LOG_DIR": "logs",
    }
    if env_path is not None:
        effective.update(_read_dotenv_values(env_path))
    source_env = os.environ if env is None else env
    effective.update({str(key): str(value) for key, value in source_env.items()})
    return effective


def evaluate_environment_safety(
    effective_env: dict[str, str],
    *,
    inspection_only_dry_run: bool,
) -> SafetyEvaluation:
    fatal_reasons: list[str] = []
    warnings: list[str] = []

    paper_trading_only = _env_bool(effective_env, "PAPER_TRADING_ONLY", default=True)
    trading_enabled = _env_bool(effective_env, "TRADING_ENABLED", default=False)
    auto_trade_enabled = _env_bool(effective_env, "AUTO_TRADE_ENABLED", default=False)
    allow_fallback_trading = _env_bool(effective_env, "ALLOW_FALLBACK_TRADING", default=False)
    symbol = _env_text(effective_env, "SYMBOL", ALLOWED_SYMBOL)
    max_open_positions = _env_int(effective_env, "MAX_OPEN_POSITIONS", default=1)
    paper_execution_mode = _env_text(effective_env, "PAPER_EXECUTION_MODE", "alpaca_paper").lower()

    if paper_trading_only is not True:
        fatal_reasons.append("paper_trading_only_not_true")
    if symbol != ALLOWED_SYMBOL:
        fatal_reasons.append("symbol_not_btc_usd")
    if allow_fallback_trading:
        fatal_reasons.append("fallback_trading_enabled")
    if trading_enabled:
        if inspection_only_dry_run:
            warnings.append("trading_enabled_true_inspection_only_dry_run")
        else:
            fatal_reasons.append("trading_enabled_true")
    if auto_trade_enabled:
        if inspection_only_dry_run:
            warnings.append("auto_trade_enabled_true_inspection_only_dry_run")
        else:
            fatal_reasons.append("auto_trade_enabled_true")
    if max_open_positions != 1:
        fatal_reasons.append("max_open_positions_not_one")
    if paper_execution_mode not in {"alpaca_paper", "local_simulated"}:
        fatal_reasons.append("paper_execution_mode_not_paper_safe")

    live_trading_indicated = _live_trading_indicated(effective_env)
    margin_enabled = _any_env_flag(effective_env, MARGIN_FLAG_NAMES)
    short_selling_enabled = _any_env_flag(effective_env, SHORT_FLAG_NAMES)
    multi_symbol_enabled = _multi_symbol_enabled(effective_env)
    if live_trading_indicated:
        fatal_reasons.append("live_trading_setting_detected")
    if margin_enabled:
        fatal_reasons.append("margin_setting_detected")
    if short_selling_enabled:
        fatal_reasons.append("short_selling_setting_detected")
    if multi_symbol_enabled:
        fatal_reasons.append("multi_symbol_setting_detected")

    flags = {
        "paper_trading_only": paper_trading_only,
        "trading_enabled": trading_enabled,
        "auto_trade_enabled": auto_trade_enabled,
        "allow_fallback_trading": allow_fallback_trading,
        "symbol": symbol,
        "btc_usd_only": symbol == ALLOWED_SYMBOL,
        "long_only": not short_selling_enabled,
        "margin_enabled": margin_enabled,
        "short_selling_enabled": short_selling_enabled,
        "multi_symbol_enabled": multi_symbol_enabled,
        "max_open_positions": max_open_positions,
        "paper_execution_mode": paper_execution_mode,
        "live_trading_indicated": live_trading_indicated,
        "inspection_only_dry_run": inspection_only_dry_run,
        "safety_gate_passed": not fatal_reasons,
        "fatal_reasons": fatal_reasons,
        "warnings": warnings,
    }
    return SafetyEvaluation(
        passed=not fatal_reasons,
        fatal_reasons=fatal_reasons,
        warnings=warnings,
        flags=flags,
    )


def build_data_readiness_by_timeframe(
    settings: Any,
    *,
    effective_env: dict[str, str],
    session_factory: Any | None = None,
    now: datetime | None = None,
) -> dict[str, dict[str, Any]]:
    current_time = _utc_datetime(now or datetime.now(UTC))
    min_rows = _minimum_rows_by_timeframe(effective_env)
    max_age = _max_age_minutes_by_timeframe(effective_env)
    reports: dict[str, dict[str, Any]] = {}
    for timeframe in TIMEFRAMES:
        stats = collected_data_stats(settings, timeframe=timeframe, session_factory=session_factory)
        row_count = int(stats["row_count"])
        first_timestamp = stats["first_timestamp"]
        latest_timestamp = stats["latest_timestamp"]
        latest = _utc_datetime(latest_timestamp) if latest_timestamp is not None else None
        data_age_minutes = (
            max(0.0, (current_time - latest).total_seconds() / 60.0) if latest is not None else None
        )
        rejection_reasons: list[str] = []
        if row_count < min_rows[timeframe]:
            rejection_reasons.append("row_count_below_required")
        if latest is None:
            rejection_reasons.append("no_valid_real_data_source")
        elif data_age_minutes is not None and data_age_minutes > max_age[timeframe]:
            rejection_reasons.append("stale_latest_timestamp")
        if "no_valid_real_data_source" in rejection_reasons:
            rejection_reasons.append("synthetic_fallback_not_allowed")

        ready = not rejection_reasons
        source_used = "collected_market_data" if row_count > 0 else "no_valid_real_data_source"
        reports[timeframe] = {
            "timeframe": timeframe,
            "symbol": settings.symbol,
            "source_used": source_used,
            "row_count": row_count,
            "first_timestamp": _iso_or_none(first_timestamp),
            "latest_timestamp": _iso_or_none(latest_timestamp),
            "data_age_minutes": data_age_minutes,
            "minimum_required_rows": min_rows[timeframe],
            "freshness_required_minutes": max_age[timeframe],
            "ready_for_research": ready,
            "ready_for_training": ready,
            "rejection_reason": ";".join(rejection_reasons) if rejection_reasons else None,
            "rejection_reasons": rejection_reasons,
            "invalid_for_trading_decisions": not ready,
            "synthetic_data_used": False,
        }
    return reports


def collected_data_stats(
    settings: Any,
    *,
    timeframe: str,
    session_factory: Any | None = None,
) -> dict[str, Any]:
    if session_factory is None:
        from app.db.database import SessionLocal, init_db

        init_db()
        session_factory = SessionLocal

    from sqlalchemy import func

    from app.db.models import CollectedMarketData

    with session_factory() as db:
        row = (
            db.query(
                func.count(CollectedMarketData.id),
                func.min(CollectedMarketData.timestamp),
                func.max(CollectedMarketData.timestamp),
            )
            .filter(
                CollectedMarketData.symbol == settings.symbol,
                CollectedMarketData.timeframe == timeframe,
            )
            .one()
        )
    return {"row_count": int(row[0] or 0), "first_timestamp": row[1], "latest_timestamp": row[2]}


def load_collected_bars(
    settings: Any,
    *,
    timeframe: str,
    limit: int,
    session_factory: Any | None = None,
) -> pd.DataFrame:
    if session_factory is None:
        from app.db.database import SessionLocal, init_db

        init_db()
        session_factory = SessionLocal

    from app.data.market_data import normalize_ohlcv
    from app.db.models import CollectedMarketData

    with session_factory() as db:
        rows = (
            db.query(CollectedMarketData)
            .filter(
                CollectedMarketData.symbol == settings.symbol,
                CollectedMarketData.timeframe == timeframe,
            )
            .order_by(CollectedMarketData.timestamp.desc())
            .limit(max(1, int(limit)))
            .all()
        )
    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    records = [
        {
            "timestamp": row.timestamp,
            "open": row.open,
            "high": row.high,
            "low": row.low,
            "close": row.close,
            "volume": row.volume,
        }
        for row in reversed(rows)
    ]
    return normalize_ohlcv(pd.DataFrame(records))


def build_diagnostics_summary(
    settings: Any,
    *,
    data_readiness_by_timeframe: dict[str, dict[str, Any]],
    model_registry_status: dict[str, Any],
    session_factory: Any | None = None,
) -> dict[str, Any]:
    from scripts.diagnose_execution_costs import build_execution_cost_report, take_profit_is_unsafe

    one_min_report = data_readiness_by_timeframe.get("1Min", {})
    one_min_bars = load_collected_bars(
        settings,
        timeframe="1Min",
        limit=_training_bar_limit(settings, one_min_report),
        session_factory=session_factory,
    )
    execution_cost_report = build_execution_cost_report(one_min_bars, settings)
    current_target_take_profit_pct = (
        float(settings.scalping_take_profit_pct)
        if bool(settings.scalping_mode_enabled)
        else float(settings.take_profit_pct)
    )
    target_vs_cost_unsafe = take_profit_is_unsafe(
        current_target_take_profit_pct,
        float(execution_cost_report["round_trip_estimated_cost_pct"]),
    )

    label_report = build_label_diagnostics_summary(
        settings,
        data_readiness_by_timeframe=data_readiness_by_timeframe,
        session_factory=session_factory,
    )
    return {
        "round_trip_estimated_cost_pct": execution_cost_report["round_trip_estimated_cost_pct"],
        "promotion_required_return_pct": execution_cost_report["promotion_required_return_pct"],
        "scalping_take_profit_pct": execution_cost_report["scalping_take_profit_pct"],
        "current_target_take_profit_pct": current_target_take_profit_pct,
        "current_target_safe_vs_cost": not target_vs_cost_unsafe,
        "target_vs_cost_unsafe": target_vs_cost_unsafe,
        "execution_cost_report": execution_cost_report,
        "label_diagnostics": label_report,
        "paper_forward_eligible_config_count": 0,
        "economically_viable_config_count": 0,
        "research_result_valid": False,
        "synthetic_data_used": False,
        "source_used": {
            timeframe: report.get("source_used")
            for timeframe, report in data_readiness_by_timeframe.items()
        },
        "active_model_valid": model_registry_status.get("active_model_valid", False),
        "active_model_invalid_reason": model_registry_status.get("active_model_invalid_reason"),
        "active_model_status": model_registry_status.get("active_model_status"),
    }


def build_label_diagnostics_summary(
    settings: Any,
    *,
    data_readiness_by_timeframe: dict[str, dict[str, Any]],
    session_factory: Any | None = None,
) -> dict[str, Any]:
    training_timeframe = str(settings.timeframe)
    readiness = data_readiness_by_timeframe.get(training_timeframe, {})
    bars = load_collected_bars(
        settings,
        timeframe=training_timeframe,
        limit=_training_bar_limit(settings, readiness),
        session_factory=session_factory,
    )
    if bars.empty:
        return {
            "available": True,
            "training_timeframe": training_timeframe,
            "raw_bars": 0,
            "trainable_rows": 0,
            "buy_positive_label_count": 0,
            "buy_positive_label_pct": 0.0,
            "min_buy_positive_labels": int(settings.min_buy_positive_labels),
            "min_buy_positive_label_pct": float(settings.min_buy_positive_label_pct),
            "enough_buy_labels": False,
            "warning": "not_enough_bars_for_label_diagnostics",
        }
    try:
        from scripts.diagnose_labels import build_diagnosis_report

        report = build_diagnosis_report(bars, settings)
        summary = report["summary"]
        buy_positive_label_count = int(summary.get("buy_positive_label_count", 0))
        buy_positive_label_pct = float(summary.get("buy_positive_label_pct", 0.0))
        enough_buy_labels = (
            buy_positive_label_count >= int(settings.min_buy_positive_labels)
            and buy_positive_label_pct >= float(settings.min_buy_positive_label_pct)
        )
        return {
            "available": True,
            "training_timeframe": training_timeframe,
            "raw_bars": int(summary.get("raw_bars", 0)),
            "trainable_rows": int(summary.get("trainable_rows", 0)),
            "buy_positive_label_count": buy_positive_label_count,
            "buy_positive_label_pct": buy_positive_label_pct,
            "min_buy_positive_labels": int(settings.min_buy_positive_labels),
            "min_buy_positive_label_pct": float(settings.min_buy_positive_label_pct),
            "enough_buy_labels": enough_buy_labels,
            "warning": report.get("warning"),
            "summary": summary,
        }
    except Exception as exc:
        return {
            "available": False,
            "training_timeframe": training_timeframe,
            "raw_bars": int(len(bars)),
            "trainable_rows": 0,
            "buy_positive_label_count": 0,
            "buy_positive_label_pct": 0.0,
            "min_buy_positive_labels": int(settings.min_buy_positive_labels),
            "min_buy_positive_label_pct": float(settings.min_buy_positive_label_pct),
            "enough_buy_labels": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def should_run_research(
    data_readiness_by_timeframe: dict[str, dict[str, Any]],
    *,
    force_research: bool,
) -> bool:
    if force_research:
        return True
    return any(
        bool(data_readiness_by_timeframe.get(timeframe, {}).get("ready_for_research"))
        for timeframe in RESEARCH_TIMEFRAMES
    )


async def run_research_pipeline(
    settings: Any,
    *,
    session_factory: Any | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    try:
        from scripts.research_higher_timeframe import run_higher_timeframe_research

        summary = await run_higher_timeframe_research(
            settings,
            bar_limit=max(1500, int(settings.min_training_rows) + 500),
            output_dir=Path(settings.log_dir),
            session_factory=session_factory,
            allow_synthetic_fallback=False,
            now=now,
        )
        return normalize_research_summary(summary, status="run")
    except Exception as exc:
        summary = _empty_research_summary("research_exception")
        summary["status"] = "error"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        return summary


def normalize_research_summary(summary: dict[str, Any], *, status: str) -> dict[str, Any]:
    synthetic_data_used = bool(summary.get("synthetic_data_used"))
    research_result_valid = bool(summary.get("research_result_valid")) and not synthetic_data_used
    paper_forward_count = int(summary.get("paper_forward_eligible_config_count", 0) or 0)
    economic_count = int(summary.get("economically_viable_config_count", 0) or 0)
    if synthetic_data_used:
        research_result_valid = False
        paper_forward_count = 0
        economic_count = 0
    out = dict(summary)
    out.update(
        {
            "status": status,
            "research_result_valid": research_result_valid,
            "synthetic_data_used": synthetic_data_used,
            "paper_forward_eligible_config_count": paper_forward_count,
            "economically_viable_config_count": economic_count,
            "invalid_for_trading_decisions": (not research_result_valid) or synthetic_data_used,
        }
    )
    return out


def merge_research_into_diagnostics(
    diagnostics_summary: dict[str, Any],
    research_summary: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(diagnostics_summary)
    merged["paper_forward_eligible_config_count"] = int(
        research_summary.get("paper_forward_eligible_config_count", 0) or 0
    )
    merged["economically_viable_config_count"] = int(
        research_summary.get("economically_viable_config_count", 0) or 0
    )
    merged["research_result_valid"] = bool(research_summary.get("research_result_valid"))
    merged["synthetic_data_used"] = bool(research_summary.get("synthetic_data_used"))
    source_used = dict(merged.get("source_used") or {})
    for timeframe, source in (research_summary.get("source_used") or {}).items():
        source_used[str(timeframe)] = source
    merged["source_used"] = source_used
    active_model_status = research_summary.get("active_model_status") or {}
    if isinstance(active_model_status, dict) and active_model_status:
        merged["active_model_valid"] = active_model_status.get(
            "active_model_valid",
            merged.get("active_model_valid", False),
        )
        merged["active_model_invalid_reason"] = active_model_status.get(
            "active_model_invalid_reason",
            merged.get("active_model_invalid_reason"),
        )
        merged["active_model_status"] = active_model_status.get(
            "active_model_status",
            merged.get("active_model_status"),
        )
    return merged


def evaluate_training_gates(
    *,
    mode: str,
    settings: Any,
    safety: SafetyEvaluation,
    data_readiness_by_timeframe: dict[str, dict[str, Any]],
    diagnostics_summary: dict[str, Any],
    research_summary: dict[str, Any],
) -> dict[str, Any]:
    training_timeframe = str(settings.timeframe)
    readiness = data_readiness_by_timeframe.get(training_timeframe)
    label_diagnostics = diagnostics_summary.get("label_diagnostics") or {}
    research_synthetic = bool(research_summary.get("synthetic_data_used"))
    fallback_prediction_used = research_synthetic or _research_used_fallback_predictions(research_summary)
    strict_promotion_logic_available = True
    economic_count = int(research_summary.get("economically_viable_config_count", 0) or 0)

    gates = {
        "run_mode": {
            "passed": mode == "run",
            "reason": None if mode == "run" else "dry_run_never_runs_train_model",
        },
        "environment_safety": {
            "passed": safety.passed,
            "reason": ";".join(safety.fatal_reasons) if safety.fatal_reasons else None,
        },
        "trading_disabled": {
            "passed": (
                safety.flags.get("trading_enabled") is False
                and safety.flags.get("auto_trade_enabled") is False
            ),
            "reason": (
                None
                if safety.flags.get("trading_enabled") is False
                and safety.flags.get("auto_trade_enabled") is False
                else "trading_or_auto_trade_enabled"
            ),
        },
        "data_readiness": {
            "passed": bool(readiness and readiness.get("ready_for_training")),
            "timeframe": training_timeframe,
            "reason": None
            if readiness and readiness.get("ready_for_training")
            else (readiness or {}).get("rejection_reason", "unsupported_training_timeframe"),
        },
        "buy_label_floor": {
            "passed": bool(label_diagnostics.get("enough_buy_labels")),
            "reason": None if label_diagnostics.get("enough_buy_labels") else "buy_positive_labels_too_low",
            "buy_positive_label_count": label_diagnostics.get("buy_positive_label_count"),
            "buy_positive_label_pct": label_diagnostics.get("buy_positive_label_pct"),
            "min_buy_positive_labels": label_diagnostics.get("min_buy_positive_labels"),
            "min_buy_positive_label_pct": label_diagnostics.get("min_buy_positive_label_pct"),
        },
        "execution_cost_target": {
            "passed": diagnostics_summary.get("current_target_safe_vs_cost") is True,
            "reason": None
            if diagnostics_summary.get("current_target_safe_vs_cost") is True
            else "target_vs_cost_unsafe",
        },
        "research_or_strict_promotion": {
            "passed": economic_count > 0 or strict_promotion_logic_available,
            "reason": None
            if economic_count > 0
            else "no_economically_viable_research_config_train_model_strict_promotion_guard_required",
            "economically_viable_config_count": economic_count,
            "strict_promotion_logic_available": strict_promotion_logic_available,
        },
        "fallback_prediction_not_used": {
            "passed": not fallback_prediction_used,
            "reason": None if not fallback_prediction_used else "fallback_prediction_not_allowed",
        },
        "synthetic_research_not_used": {
            "passed": not research_synthetic,
            "reason": None if not research_synthetic else "synthetic_data_used",
        },
    }
    all_gates_passed = all(bool(gate["passed"]) for gate in gates.values())
    blocked_reasons = [
        str(gate["reason"])
        for gate in gates.values()
        if not gate["passed"] and gate.get("reason")
    ]
    return {
        "all_gates_passed": all_gates_passed,
        "blocked_reasons": blocked_reasons,
        "training_timeframe": training_timeframe,
        "gates": gates,
    }


def run_train_model(settings: Any, bars: pd.DataFrame) -> dict[str, Any]:
    from app.ml.train import train_model_from_bars

    result = train_model_from_bars(bars, settings)
    return {"status": "run", **_json_safe(result)}


def inspect_model_registry(settings: Any) -> dict[str, Any]:
    try:
        from app.ml.registry import ModelRegistry

        registry = ModelRegistry(settings)
        status = registry.validate_active_model().to_dict()
        return {
            "registry_exists": registry.path.exists(),
            **status,
        }
    except Exception as exc:  # pragma: no cover - defensive guard.
        return {
            "registry_exists": False,
            "active_model_path": None,
            "active_model_status": "error",
            "active_model_valid": False,
            "active_model_invalid_reason": f"{type(exc).__name__}: {exc}",
            "active_model_net_return_pct": None,
            "active_model_profit_factor_net": None,
            "active_model_number_of_trades": None,
            "active_model_promotion_reason": None,
        }


def build_backtest_result(train_model_result: dict[str, Any]) -> dict[str, Any]:
    if train_model_result.get("status") != "run":
        return {"status": "not_run", "reason": train_model_result.get("reason")}
    metrics = train_model_result.get("metrics") or {}
    return {
        "status": "run",
        "accepted": bool(train_model_result.get("accepted")),
        "reason": train_model_result.get("reason"),
        "fee_aware_backtest_valid": metrics.get("fee_aware_backtest_valid"),
        "fee_aware_backtest_reason": metrics.get("fee_aware_backtest_reason"),
        "net_return_pct": metrics.get("net_return_pct"),
        "profit_factor_net": metrics.get("profit_factor_net"),
        "number_of_trades": metrics.get("number_of_trades"),
        "max_drawdown_pct": metrics.get("max_drawdown_pct"),
        "ambiguous_candle_ratio": metrics.get("ambiguous_candle_ratio"),
    }


def write_report_files(report: dict[str, Any], *, log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_json_safe(report), indent=2, sort_keys=True, allow_nan=False)
    (log_dir / REPORT_NAME).write_text(payload + "\n", encoding="utf-8")
    (log_dir / LATEST_REPORT_NAME).write_text(payload + "\n", encoding="utf-8")


def _base_report(
    *,
    generated_at: datetime,
    mode: str,
    force_research: bool,
    safety: SafetyEvaluation,
) -> dict[str, Any]:
    return {
        "generated_at": generated_at.isoformat(),
        "mode": mode,
        "force_research": force_research,
        "safety_flags": safety.flags,
        "environment_safety_gate_passed": safety.passed,
        "environment_safety_gate_reasons": list(safety.fatal_reasons),
        "orders_placed": 0,
    }


def _finish_report(report: dict[str, Any]) -> None:
    safety_flags = report.get("safety_flags") or {}
    model_status = report.get("model_registry_status") or {}
    research_summary = report.get("research_summary") or {}
    data_readiness = report.get("data_readiness_by_timeframe") or {}
    training_was_run = bool(report.get("training_was_run"))
    train_result = report.get("train_model_result") or {}
    trading_remained_disabled = (
        safety_flags.get("trading_enabled") is False
        and safety_flags.get("auto_trade_enabled") is False
    )
    report["trading_remained_disabled"] = trading_remained_disabled
    report["orders_placed"] = 0
    strategy_training_summary = build_strategy_training_summary(report)
    report["strategy_training_summary"] = strategy_training_summary
    report.update(strategy_training_summary)
    report["final_recommendation"] = final_recommendation(
        safety_flags=safety_flags,
        data_readiness_by_timeframe=data_readiness,
        research_summary=research_summary,
        training_was_run=training_was_run,
        train_model_result=train_result,
    )
    report["recommendations"] = _recommendations(report)
    report["brutally_honest_notes"] = brutally_honest_notes(
        report,
        active_model_valid=bool(model_status.get("active_model_valid")),
    )


def build_strategy_training_summary(report: dict[str, Any]) -> dict[str, Any]:
    research_summary = report.get("research_summary") or {}
    diagnostics = report.get("diagnostics_summary") or {}
    gates = report.get("training_gate_results") or {}
    blocked_reasons = list(gates.get("blocked_reasons") or [])
    trade_summary = research_summary.get("buy_the_dip_mean_reversion_trade_summary") or {}
    current_scalping_blocked = (
        diagnostics.get("target_vs_cost_unsafe") is True
        or "target_vs_cost_unsafe" in blocked_reasons
    )
    configs_tested = _summary_int(research_summary, "buy_the_dip_configs_tested", trade_summary.get("configs_tested", 0))
    twenty_plus = _summary_int(
        research_summary,
        "buy_the_dip_20_plus_trade_configs",
        trade_summary.get("configs_with_20_plus_trades", 0),
    )
    profitable_twenty_plus = _summary_int(
        research_summary,
        "buy_the_dip_profitable_20_plus_trade_configs",
        trade_summary.get("profitable_configs_with_20_plus_trades", 0),
    )
    economic_count = _summary_int(
        research_summary,
        "buy_the_dip_economically_viable_count",
        trade_summary.get("economically_viable_count", 0),
    )
    best_20_plus = (
        research_summary.get("buy_the_dip_best_config_20_plus_trades")
        or trade_summary.get("best_config_20_plus_trades")
    )
    no_trainable_strategy = (
        report.get("training_was_run") is False
        and current_scalping_blocked
        and economic_count == 0
    )
    return {
        "current_scalping_training_blocked_by_target_vs_cost": current_scalping_blocked,
        "buy_the_dip_research_available": bool(configs_tested and research_summary.get("research_result_valid")),
        "buy_the_dip_configs_tested": configs_tested,
        "buy_the_dip_20_plus_trade_configs": twenty_plus,
        "buy_the_dip_profitable_20_plus_trade_configs": profitable_twenty_plus,
        "buy_the_dip_economically_viable_count": economic_count,
        "buy_the_dip_best_config_20_plus_trades": best_20_plus,
        "training_skipped_no_trainable_strategy_exists_yet": no_trainable_strategy,
    }


def final_recommendation(
    *,
    safety_flags: dict[str, Any],
    data_readiness_by_timeframe: dict[str, dict[str, Any]],
    research_summary: dict[str, Any],
    training_was_run: bool,
    train_model_result: dict[str, Any],
) -> str:
    if training_was_run and train_model_result.get("accepted") is True:
        return "keep_auto_trading_disabled"
    if safety_flags.get("trading_enabled") or safety_flags.get("auto_trade_enabled"):
        return "keep_auto_trading_disabled"
    if any(not report.get("ready_for_training") for report in data_readiness_by_timeframe.values()):
        return "collect_more_data"
    research_recommendation = str((research_summary.get("concise_summary") or {}).get("recommendation") or "")
    if research_recommendation in {"collect_more_data", "run_longer_backfill", "improve_strategy"}:
        return research_recommendation
    if int(research_summary.get("buy_the_dip_configs_tested", 0) or 0) > 0:
        if int(research_summary.get("buy_the_dip_20_plus_trade_configs", 0) or 0) == 0:
            return "run_longer_backfill"
        if int(research_summary.get("buy_the_dip_economically_viable_count", 0) or 0) == 0:
            return "improve_strategy"
    if int(research_summary.get("economically_viable_config_count", 0) or 0) > 0:
        return "keep_auto_trading_disabled"
    return "keep_auto_trading_disabled"


def brutally_honest_notes(report: dict[str, Any], *, active_model_valid: bool) -> list[str]:
    notes = [
        "This pipeline is research/training automation only; it does not submit, stage, or simulate broker orders.",
        "Paper-forward eligibility is not permission to enable auto trading.",
    ]
    safety_flags = report.get("safety_flags") or {}
    if safety_flags.get("trading_enabled") or safety_flags.get("auto_trade_enabled"):
        notes.append("Trading flags are enabled in the inspected environment; --run is blocked until they are false.")
    if not active_model_valid:
        notes.append("There is no valid active model according to the model registry checks.")
    for timeframe, readiness in (report.get("data_readiness_by_timeframe") or {}).items():
        if not readiness.get("ready_for_training"):
            notes.append(
                f"{timeframe} is not training-ready: {readiness.get('rejection_reason') or 'unknown_reason'}."
            )
    research_summary = report.get("research_summary") or {}
    if research_summary.get("synthetic_data_used"):
        notes.append("Synthetic research data was detected, so the research output is invalid for trading decisions.")
    if int(research_summary.get("economically_viable_config_count", 0) or 0) == 0:
        notes.append("No economically viable higher-timeframe config was found by this run.")
    if report.get("training_was_run") is False:
        blocked = (report.get("training_gate_results") or {}).get("blocked_reasons") or []
        if blocked:
            notes.append(f"Training did not run because gates blocked it: {', '.join(blocked)}.")
    notes.append("Auto trading should remain disabled until a human reviews reports, registry status, and paper-only risk.")
    return notes


def _recommendations(report: dict[str, Any]) -> list[str]:
    values = ["keep_auto_trading_disabled"]
    primary = report.get("final_recommendation")
    if primary and primary not in values:
        values.append(str(primary))
    return values


def _empty_research_summary(reason: str) -> dict[str, Any]:
    return {
        "status": "not_run",
        "reason": reason,
        "paper_forward_eligible_config_count": 0,
        "economically_viable_config_count": 0,
        "buy_the_dip_configs_tested": 0,
        "buy_the_dip_20_plus_trade_configs": 0,
        "buy_the_dip_profitable_20_plus_trade_configs": 0,
        "buy_the_dip_economically_viable_count": 0,
        "buy_the_dip_best_config_20_plus_trades": None,
        "research_result_valid": False,
        "synthetic_data_used": False,
        "invalid_for_trading_decisions": True,
        "source_used": {},
        "data_sources": {},
        "timeframe_data": {},
        "active_model_status": {},
        "all_results": [],
    }


def _format_text_summary(report: dict[str, Any]) -> str:
    readiness = report.get("data_readiness_by_timeframe") or {}
    five = readiness.get("5Min", {})
    fifteen = readiness.get("15Min", {})
    gates = report.get("training_gate_results") or {}
    research = report.get("research_summary") or {}
    lines = [
        "BTC/USD auto research/train report",
        f"mode: {report.get('mode')}",
        f"safety gate passed: {report.get('environment_safety_gate_passed')}",
        f"training was run: {report.get('training_was_run')}",
        f"training would run now: {gates.get('all_gates_passed')}",
        f"5Min ready: {five.get('ready_for_research')} ({five.get('rejection_reason') or 'ok'})",
        f"15Min ready: {fifteen.get('ready_for_research')} ({fifteen.get('rejection_reason') or 'ok'})",
        f"paper-forward eligible configs: {research.get('paper_forward_eligible_config_count', 0)}",
        f"final recommendation: {report.get('final_recommendation')}",
        f"orders placed: {report.get('orders_placed')}",
    ]
    blocked = gates.get("blocked_reasons") or []
    if blocked:
        lines.append(f"training blocked by: {', '.join(blocked)}")
    return "\n".join(lines)


def _read_dotenv_values(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def _minimum_rows_by_timeframe(effective_env: dict[str, str]) -> dict[str, int]:
    return {
        "1Min": _env_int(effective_env, "AUTO_RESEARCH_MIN_1MIN_ROWS", DEFAULT_MIN_ROWS["1Min"]),
        "5Min": _env_int(effective_env, "AUTO_RESEARCH_MIN_5MIN_ROWS", DEFAULT_MIN_ROWS["5Min"]),
        "15Min": _env_int(effective_env, "AUTO_RESEARCH_MIN_15MIN_ROWS", DEFAULT_MIN_ROWS["15Min"]),
    }


def _max_age_minutes_by_timeframe(effective_env: dict[str, str]) -> dict[str, float]:
    return {
        "1Min": _env_float(effective_env, "AUTO_RESEARCH_MAX_1MIN_AGE_MINUTES", DEFAULT_MAX_AGE_MINUTES["1Min"]),
        "5Min": _env_float(effective_env, "AUTO_RESEARCH_MAX_5MIN_AGE_MINUTES", DEFAULT_MAX_AGE_MINUTES["5Min"]),
        "15Min": _env_float(effective_env, "AUTO_RESEARCH_MAX_15MIN_AGE_MINUTES", DEFAULT_MAX_AGE_MINUTES["15Min"]),
    }


def _training_bar_limit(settings: Any, readiness: dict[str, Any]) -> int:
    row_count = int(readiness.get("row_count", 0) or 0)
    desired = max(
        int(getattr(settings, "lookback_bars", 0) or 0),
        int(getattr(settings, "min_training_rows", 0) or 0) + 300,
        int(readiness.get("minimum_required_rows", 0) or 0),
    )
    return max(1, min(max(row_count, desired), 20_000))


def _summary_int(summary: dict[str, Any], key: str, default: Any = 0) -> int:
    try:
        return int(summary.get(key, default) or 0)
    except (TypeError, ValueError):
        return 0


def _research_used_fallback_predictions(research_summary: dict[str, Any]) -> bool:
    if research_summary.get("synthetic_data_used"):
        return True
    for row in research_summary.get("all_results") or []:
        if isinstance(row, dict) and row.get("fallback_prediction_used"):
            return True
    return False


def _live_trading_indicated(effective_env: dict[str, str]) -> bool:
    paper_base_url = _env_text(effective_env, "ALPACA_PAPER_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
    if paper_base_url != "https://paper-api.alpaca.markets":
        return True
    for key in ("ALPACA_BASE_URL", "ALPACA_TRADING_BASE_URL", "BROKER_BASE_URL"):
        value = effective_env.get(key)
        if value and "paper-api.alpaca.markets" not in value and "api.alpaca.markets" in value:
            return True
    return _env_bool(effective_env, "LIVE_TRADING_ENABLED", default=False) is True


def _any_env_flag(effective_env: dict[str, str], names: tuple[str, ...]) -> bool:
    return any(_env_bool(effective_env, name, default=False) for name in names)


def _multi_symbol_enabled(effective_env: dict[str, str]) -> bool:
    if _any_env_flag(effective_env, MULTI_SYMBOL_FLAG_NAMES):
        return True
    for name in MULTI_SYMBOL_LIST_NAMES:
        raw = effective_env.get(name)
        if not raw:
            continue
        symbols = [part.strip() for part in raw.replace(";", ",").split(",") if part.strip()]
        if not symbols:
            continue
        if symbols != [ALLOWED_SYMBOL]:
            return True
    return False


def _env_bool(effective_env: dict[str, str], name: str, *, default: bool) -> bool:
    raw = effective_env.get(name)
    if raw is None or raw == "":
        return default
    normalized = str(raw).strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    return default


def _env_text(effective_env: dict[str, str], name: str, default: str) -> str:
    value = effective_env.get(name)
    return str(value).strip() if value is not None and str(value).strip() else default


def _env_int(effective_env: dict[str, str], name: str, default: int) -> int:
    value = effective_env.get(name)
    if value is None or value == "":
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _env_float(effective_env: dict[str, str], name: str, default: float) -> float:
    value = effective_env.get(name)
    if value is None or value == "":
        return float(default)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if math.isfinite(parsed) else float(default)


def _log_dir_from_env(effective_env: dict[str, str]) -> Path:
    return Path(_env_text(effective_env, "LOG_DIR", "logs"))


def _iso_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return _utc_datetime(value).isoformat()


def _utc_datetime(value: Any) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.to_pydatetime()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
    return value


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
