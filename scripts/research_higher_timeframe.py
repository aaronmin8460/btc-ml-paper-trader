from __future__ import annotations

import asyncio
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from app.backtest.scalping import backtest_assumptions, calculate_fee_aware_metrics
from app.config import ALLOWED_SYMBOL, Settings, get_settings
from app.data.feature_engineering import add_features
from app.data.market_data import (
    MarketDataClient,
    StaleMarketDataError,
    normalize_ohlcv,
    stale_threshold_for_timeframe,
)
from app.db.database import SessionLocal, init_db
from app.db.models import CollectedMarketData
from app.ml.registry import ModelRegistry
from app.risk.risk_manager import PositionState
from app.strategy.strategies import MarketContext, MarketRegimeFilter, TrendPullbackStrategy


RESEARCH_TIMEFRAMES = ("5Min", "15Min")
TAKE_PROFIT_VALUES = (0.008, 0.01, 0.015, 0.02)
STOP_LOSS_VALUES = (0.003, 0.005, 0.008)
MAX_HOLD_BARS_VALUES = (6, 12, 24, 48)
MIN_RESEARCH_TRADES = 20
MIN_RESEARCH_PROFIT_FACTOR_NET = 1.05
MAX_SINGLE_TRADE_RETURN_SHARE = 0.60


@dataclass(frozen=True)
class ResearchConfig:
    parameter_set_id: str
    timeframe: str
    take_profit_pct: float
    stop_loss_pct: float
    max_hold_bars: int


@dataclass(frozen=True)
class ResearchDataReport:
    timeframe: str
    source_used: str
    latest_timestamp: str | None
    data_age_minutes: float | None
    row_count: int
    synthetic_data_used: bool
    research_result_valid: bool
    rejection_reason: str | None = None
    rejected_sources: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class ResearchBarsResult:
    bars: pd.DataFrame
    report: ResearchDataReport

    def __iter__(self):
        yield self.bars
        yield self.report.source_used


async def main() -> None:
    settings = research_settings(get_settings())
    bar_limit = int(os.getenv("RESEARCH_BAR_LIMIT", str(max(1500, settings.min_training_rows + 500))))
    report = await run_higher_timeframe_research(
        settings,
        bar_limit=bar_limit,
        output_dir=Path(settings.log_dir),
        allow_synthetic_fallback=_synthetic_research_mode_enabled(),
    )
    print(json.dumps(report, indent=2, default=str))


async def run_higher_timeframe_research(
    base_settings: Settings,
    *,
    bar_limit: int = 1500,
    client: MarketDataClient | None = None,
    output_dir: Path | None = None,
    session_factory: Any | None = None,
    allow_synthetic_fallback: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    settings = research_settings(base_settings)
    client = client or MarketDataClient(settings)
    active_model = ModelRegistry(settings).validate_active_model()
    bars_by_timeframe: dict[str, pd.DataFrame] = {}
    data_source_reports: dict[str, ResearchDataReport] = {}
    current_time = _utc_timestamp(now or datetime.now(UTC))
    for timeframe in RESEARCH_TIMEFRAMES:
        result = await _fetch_research_bars(
            client,
            settings,
            timeframe=timeframe,
            limit=bar_limit,
            session_factory=session_factory,
            allow_synthetic_fallback=allow_synthetic_fallback,
            now=current_time,
        )
        bars_by_timeframe[timeframe] = result.bars
        data_source_reports[timeframe] = result.report
    rows = evaluate_research_configs(
        bars_by_timeframe,
        settings,
        active_model_valid=active_model.valid,
        active_model_status=active_model.to_dict(),
        data_source_reports=data_source_reports,
    )
    output_path = output_dir or Path(settings.log_dir)
    csv_path = output_path / "higher_timeframe_research.csv"
    summary_path = output_path / "higher_timeframe_research_summary.json"
    summary = build_research_summary(
        rows,
        settings,
        data_source_reports=data_source_reports,
        csv_path=csv_path,
        summary_path=summary_path,
        active_model_status=active_model.to_dict(),
    )
    write_research_outputs(rows, summary, csv_path=csv_path, summary_path=summary_path)
    return summary


def research_settings(settings: Settings) -> Settings:
    data = settings.model_dump()
    data.update(
        {
            "symbol": ALLOWED_SYMBOL,
            "paper_trading_only": True,
            "trading_enabled": False,
            "auto_trade_enabled": False,
            "allow_fallback_trading": False,
            "max_open_positions": 1,
        }
    )
    return Settings(_env_file=None, **data)


def evaluate_research_configs(
    bars_by_timeframe: dict[str, pd.DataFrame],
    settings: Settings,
    *,
    active_model_valid: bool,
    active_model_status: dict[str, Any] | None = None,
    data_source_reports: dict[str, ResearchDataReport | dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_reports = {
        timeframe: _coerce_data_report(timeframe, report)
        for timeframe, report in (data_source_reports or {}).items()
    }
    for config in generate_research_configs():
        candidate_settings = research_settings(
            Settings(
                _env_file=None,
                **{
                    **settings.model_dump(),
                    "take_profit_pct": config.take_profit_pct,
                    "stop_loss_pct": config.stop_loss_pct,
                    "label_horizon_bars": config.max_hold_bars,
                },
            )
        )
        bars = bars_by_timeframe.get(config.timeframe, pd.DataFrame())
        source_report = source_reports.get(config.timeframe)
        source_valid = source_report.research_result_valid if source_report is not None else True
        synthetic_data_used = source_report.synthetic_data_used if source_report is not None else False
        if bars.empty or not source_valid:
            metrics = _empty_research_metrics("invalid_research_data_source" if not source_valid else "no_bars")
        else:
            trades, signal_frame = build_research_trades(bars, candidate_settings, config)
            metrics = calculate_fee_aware_metrics(trades, candidate_settings, signal_frame=signal_frame)
        readiness = paper_forward_readiness_gate(
            metrics,
            candidate_settings,
            fallback_prediction_used=synthetic_data_used,
            active_model_valid=active_model_valid,
            research_result_valid=source_valid,
        )
        rows.append(
            {
                "parameter_set_id": config.parameter_set_id,
                "strategy_name": TrendPullbackStrategy.name,
                "timeframe": config.timeframe,
                "take_profit_pct": config.take_profit_pct,
                "stop_loss_pct": config.stop_loss_pct,
                "max_hold_bars": config.max_hold_bars,
                "number_of_trades": int(metrics.get("number_of_trades", 0) or 0),
                "gross_return_pct": _metric_float(metrics.get("gross_return_pct")),
                "net_return_pct": _metric_float(metrics.get("net_return_pct")),
                "profit_factor_net": _profit_factor_value(metrics.get("profit_factor_net")),
                "max_drawdown_pct": _metric_float(metrics.get("max_drawdown_pct")),
                "win_rate_net": _metric_float(metrics.get("win_rate_net")),
                "expectancy": _metric_float(metrics.get("expectancy")),
                "round_trip_estimated_cost_pct": _metric_float(metrics.get("round_trip_estimated_cost_pct")),
                "promotion_required_return_pct": _metric_float(metrics.get("promotion_required_return_pct")),
                "gross_winners_became_net_losers": int(metrics.get("gross_winners_became_net_losers", 0) or 0),
                "single_trade_return_concentration": single_trade_return_concentration(
                    metrics.get("trade_details", [])
                ),
                "source_used": source_report.source_used if source_report is not None else "unknown",
                "latest_timestamp": source_report.latest_timestamp if source_report is not None else None,
                "data_age_minutes": source_report.data_age_minutes if source_report is not None else None,
                "row_count": source_report.row_count if source_report is not None else int(len(bars)),
                "synthetic_data_used": synthetic_data_used,
                "research_result_valid": source_valid,
                "data_rejection_reason": source_report.rejection_reason if source_report is not None else None,
                "rejected_sources": list(source_report.rejected_sources) if source_report is not None else [],
                "fallback_prediction_used": synthetic_data_used,
                "active_model_valid": bool(active_model_valid),
                "active_model_status": (active_model_status or {}).get("active_model_status"),
                "economically_viable": readiness["economically_viable"],
                "paper_forward_eligible": readiness["paper_forward_eligible"],
                "rejection_reasons": ";".join(readiness["rejection_reasons"]),
                "rank_score": research_rank_score(metrics, readiness),
            }
        )
    return rows


def generate_research_configs() -> list[ResearchConfig]:
    configs: list[ResearchConfig] = []
    index = 0
    for timeframe, take_profit_pct, stop_loss_pct, max_hold_bars in product(
        RESEARCH_TIMEFRAMES,
        TAKE_PROFIT_VALUES,
        STOP_LOSS_VALUES,
        MAX_HOLD_BARS_VALUES,
    ):
        configs.append(
            ResearchConfig(
                parameter_set_id=f"htf_{index:03d}",
                timeframe=timeframe,
                take_profit_pct=float(take_profit_pct),
                stop_loss_pct=float(stop_loss_pct),
                max_hold_bars=int(max_hold_bars),
            )
        )
        index += 1
    return configs


def build_research_trades(
    bars: pd.DataFrame,
    settings: Settings,
    config: ResearchConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = add_features(bars).dropna().reset_index(drop=True)
    if features.empty:
        return pd.DataFrame(), pd.DataFrame()
    features["orderbook_spread"] = max(0.0, float(settings.max_spread_bps)) / 10_000
    features["quote_imbalance"] = 0.0
    features["scalping_spread_bps"] = max(0.0, float(settings.max_spread_bps))
    features["scalping_quote_imbalance"] = 0.0
    regime_filter = MarketRegimeFilter(settings)
    strategy = TrendPullbackStrategy(settings)
    signal_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    index = 0
    while index < len(features) - config.max_hold_bars - 1:
        row = features.iloc[index]
        regime = regime_filter.detect(row)
        allowed_by_regime, regime_reason = regime_filter.allows(regime, strategy.name)
        context = MarketContext(regime=regime, risk_permits_evaluation=allowed_by_regime, risk_reason=regime_reason)
        signal = strategy.generate_signal(
            feature_row=row,
            prediction=None,
            position=PositionState(),
            quote=None,
            market_context=context,
        )
        signal_row = _signal_row(row, signal=signal, regime=regime.regime, config=config)
        if signal.action == "buy":
            exit_result = resolve_research_exit(features, index, config)
            trade_row = {
                **signal_row,
                "buy_quality_label": int(exit_result["gross_return"] > 0),
                "buy_exit_return_pct": float(exit_result["gross_return"]),
                "buy_exit_reason": exit_result["exit_reason"],
                "buy_hold_bars": float(exit_result["hold_bars"]),
                "backtest_exit_high": float(exit_result["exit_high"]),
                "backtest_exit_low": float(exit_result["exit_low"]),
            }
            trade_rows.append(trade_row)
            signal_row["entry_allowed"] = True
            signal_rows.append(signal_row)
            index += max(1, int(exit_result["hold_bars"]))
            continue
        signal_row["entry_allowed"] = False
        signal_rows.append(signal_row)
        index += 1
    return pd.DataFrame(trade_rows), pd.DataFrame(signal_rows)


def resolve_research_exit(features: pd.DataFrame, entry_index: int, config: ResearchConfig) -> dict[str, Any]:
    entry_close = float(features.iloc[entry_index]["close"])
    take_profit_price = entry_close * (1 + config.take_profit_pct)
    stop_loss_price = entry_close * (1 - config.stop_loss_pct)
    max_exit_index = min(len(features) - 1, entry_index + config.max_hold_bars)
    for offset, row_index in enumerate(range(entry_index + 1, max_exit_index + 1), start=1):
        row = features.iloc[row_index]
        high = float(row["high"])
        low = float(row["low"])
        hit_take_profit = high >= take_profit_price
        hit_stop_loss = low <= stop_loss_price
        if hit_take_profit and hit_stop_loss:
            return _exit_result(-config.stop_loss_pct, "ambiguous_stop_first", offset, high, low)
        if hit_stop_loss:
            return _exit_result(-config.stop_loss_pct, "research_stop_loss", offset, high, low)
        if hit_take_profit:
            return _exit_result(config.take_profit_pct, "research_take_profit", offset, high, low)
    exit_row = features.iloc[max_exit_index]
    gross_return = float(exit_row["close"] / entry_close - 1)
    return _exit_result(
        gross_return,
        "research_max_hold",
        max(1, max_exit_index - entry_index),
        float(exit_row["high"]),
        float(exit_row["low"]),
    )


def paper_forward_readiness_gate(
    metrics: dict[str, Any],
    settings: Settings,
    *,
    fallback_prediction_used: bool,
    active_model_valid: bool,
    research_result_valid: bool = True,
) -> dict[str, Any]:
    reasons: list[str] = []
    if not research_result_valid:
        reasons.append("research_data_source_invalid")
    if _metric_float(metrics.get("net_return_pct")) <= 0:
        reasons.append("net_return_not_positive")
    if _profit_factor_value(metrics.get("profit_factor_net")) < MIN_RESEARCH_PROFIT_FACTOR_NET:
        reasons.append("profit_factor_net_below_1_05")
    if int(metrics.get("number_of_trades", 0) or 0) < MIN_RESEARCH_TRADES:
        reasons.append("number_of_trades_below_20")
    if _metric_float(metrics.get("max_drawdown_pct")) > float(settings.max_backtest_drawdown_pct):
        reasons.append("max_drawdown_above_configured_limit")
    if single_trade_return_concentration(metrics.get("trade_details", [])) > MAX_SINGLE_TRADE_RETURN_SHARE:
        reasons.append("single_trade_return_concentration_too_high")
    if fallback_prediction_used:
        reasons.append("fallback_prediction_not_allowed")
    economically_viable = not reasons
    if not active_model_valid:
        reasons.append("active_model_invalid")
    return {
        "economically_viable": economically_viable,
        "paper_forward_eligible": not reasons,
        "rejection_reasons": reasons,
    }


def build_research_summary(
    rows: list[dict[str, Any]],
    settings: Settings,
    *,
    data_sources: dict[str, str] | None = None,
    data_source_reports: dict[str, ResearchDataReport | dict[str, Any]] | None = None,
    csv_path: Path,
    summary_path: Path,
    active_model_status: dict[str, Any],
) -> dict[str, Any]:
    ranked = sorted(rows, key=lambda row: float(row.get("rank_score", -1_000_000.0)), reverse=True)
    raw_economically_viable = [row for row in ranked if row.get("economically_viable")]
    source_reports = _summary_source_reports(data_source_reports=data_source_reports, data_sources=data_sources)
    source_report_payload = {timeframe: _data_report_to_dict(report) for timeframe, report in source_reports.items()}
    data_source_names = {timeframe: report.source_used for timeframe, report in source_reports.items()}
    synthetic_data_used = any(report.synthetic_data_used for report in source_reports.values()) or any(
        bool(row.get("synthetic_data_used")) for row in ranked
    )
    research_result_valid = (
        bool(source_reports)
        and all(report.research_result_valid for report in source_reports.values())
        and not synthetic_data_used
    )
    economically_viable = [] if synthetic_data_used or not research_result_valid else raw_economically_viable
    eligible = [] if synthetic_data_used or not research_result_valid else [
        row for row in ranked if row.get("paper_forward_eligible")
    ]
    notes = [
        "Higher-timeframe research is offline analysis only and never enables trading.",
        "No configuration is auto-applied or promoted from this report.",
        "Paper-forward eligibility also requires the existing active model registry to validate.",
        "Research validity requires fresh non-synthetic data for every timeframe.",
    ]
    if synthetic_data_used:
        notes.append(
            "Synthetic bars were used only because explicit test/demo mode allowed it; "
            "this report is invalid for trading decisions."
        )
    return _json_safe(
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "symbol": settings.symbol,
            "paper_trading_only": settings.paper_trading_only,
            "btc_usd_only": settings.symbol == ALLOWED_SYMBOL,
            "long_only": True,
            "trading_enabled": settings.trading_enabled,
            "auto_trade_enabled": settings.auto_trade_enabled,
            "fallback_trading_allowed": settings.allow_fallback_trading,
            "auto_apply_best_config": False,
            "paper_forward_eligible_config_count": len(eligible),
            "economically_viable_config_count": len(economically_viable),
            "source_used": data_source_names,
            "latest_timestamp": {
                timeframe: report.latest_timestamp for timeframe, report in source_reports.items()
            },
            "data_age_minutes": {
                timeframe: report.data_age_minutes for timeframe, report in source_reports.items()
            },
            "row_count": {
                timeframe: report.row_count for timeframe, report in source_reports.items()
            },
            "synthetic_data_used": synthetic_data_used,
            "research_result_valid": research_result_valid,
            "csv_path": str(csv_path),
            "summary_path": str(summary_path),
            "data_sources": data_source_names,
            "timeframe_data": source_report_payload,
            "parameter_space": {
                "timeframes": list(RESEARCH_TIMEFRAMES),
                "take_profit_pct": list(TAKE_PROFIT_VALUES),
                "stop_loss_pct": list(STOP_LOSS_VALUES),
                "max_hold_bars": list(MAX_HOLD_BARS_VALUES),
            },
            "active_model_status": active_model_status,
            "conservative_backtest_assumptions": backtest_assumptions(settings, spread_available=True),
            "best_economically_viable_configs": economically_viable[:10],
            "paper_forward_eligible_configs": eligible[:10],
            "rejected_configs": [row for row in ranked if not row.get("paper_forward_eligible")][:50],
            "all_results": ranked,
            "notes": notes,
        }
    )


def write_research_outputs(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    csv_path: Path,
    summary_path: Path,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _csv_fieldnames(rows)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")


async def _fetch_research_bars(
    client: Any,
    settings: Settings,
    *,
    timeframe: str,
    limit: int,
    session_factory: Any | None = None,
    allow_synthetic_fallback: bool = False,
    now: datetime | None = None,
) -> ResearchBarsResult:
    current_time = _utc_timestamp(now or datetime.now(UTC))
    min_rows = _minimum_research_rows(settings, limit)
    rejected_sources: list[dict[str, Any]] = []

    sqlite_bars = _load_collected_market_data(
        settings,
        timeframe=timeframe,
        limit=limit,
        session_factory=session_factory,
    )
    sqlite_report = _assess_research_bars(
        sqlite_bars,
        source_used="collected_market_data",
        timeframe=timeframe,
        min_rows=min_rows,
        now=current_time,
        synthetic_data_used=False,
    )
    if sqlite_report.research_result_valid:
        return ResearchBarsResult(sqlite_bars, sqlite_report)
    rejected_sources.append(_data_report_to_rejected_source(sqlite_report))

    if isinstance(client, MarketDataClient) and not (settings.alpaca_api_key and settings.alpaca_secret_key):
        rejected_sources.append(
            {
                "source": "market_data_client",
                "status": "unavailable",
                "reason": "alpaca_credentials_missing",
                "latest_timestamp": None,
                "data_age_minutes": None,
                "row_count": 0,
            }
        )
    else:
        try:
            client_bars = await client.fetch_bars(settings.symbol, timeframe=timeframe, limit=limit, force_refresh=True)
            client_report = _assess_research_bars(
                client_bars,
                source_used="market_data_client",
                timeframe=timeframe,
                min_rows=min_rows,
                now=current_time,
                synthetic_data_used=False,
            )
            if client_report.research_result_valid:
                return ResearchBarsResult(client_bars, client_report)
            rejected_sources.append(_data_report_to_rejected_source(client_report))
        except Exception as exc:
            latest_timestamp = _latest_timestamp_from_error(exc)
            rejected_sources.append(
                {
                    "source": "market_data_client",
                    "status": "stale" if isinstance(exc, StaleMarketDataError) else "fetch_error",
                    "reason": "stale_latest_timestamp" if isinstance(exc, StaleMarketDataError) else type(exc).__name__,
                    "latest_timestamp": latest_timestamp,
                    "data_age_minutes": _data_age_minutes_from_iso(latest_timestamp, current_time),
                    "row_count": 0,
                }
            )

    if allow_synthetic_fallback:
        synthetic_bars = MarketDataClient.synthetic_btc_bars(limit=limit, timeframe=timeframe)
        synthetic_report = _assess_research_bars(
            synthetic_bars,
            source_used="synthetic_explicit_test_demo_mode",
            timeframe=timeframe,
            min_rows=min_rows,
            now=current_time,
            synthetic_data_used=True,
            force_invalid_reason="synthetic_data_not_valid_for_research_decisions",
        )
        return ResearchBarsResult(
            synthetic_bars,
            _with_rejected_sources(synthetic_report, rejected_sources),
        )

    report = ResearchDataReport(
        timeframe=timeframe,
        source_used="no_valid_real_data_source",
        latest_timestamp=None,
        data_age_minutes=None,
        row_count=0,
        synthetic_data_used=False,
        research_result_valid=False,
        rejection_reason="no_fresh_real_bars_available",
        rejected_sources=tuple(rejected_sources),
    )
    return ResearchBarsResult(pd.DataFrame(), report)


def _load_collected_market_data(
    settings: Settings,
    *,
    timeframe: str,
    limit: int,
    session_factory: Any | None,
) -> pd.DataFrame:
    if session_factory is None:
        init_db()
        session_factory = SessionLocal
    with session_factory() as db:
        rows = (
            db.query(CollectedMarketData)
            .filter(
                CollectedMarketData.symbol == settings.symbol,
                CollectedMarketData.timeframe == timeframe,
            )
            .order_by(CollectedMarketData.timestamp.desc())
            .limit(limit)
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


def _assess_research_bars(
    bars: pd.DataFrame,
    *,
    source_used: str,
    timeframe: str,
    min_rows: int,
    now: datetime,
    synthetic_data_used: bool,
    force_invalid_reason: str | None = None,
) -> ResearchDataReport:
    normalized = normalize_ohlcv(bars) if not bars.empty else bars
    latest = _latest_timestamp(normalized)
    age_minutes = _data_age_minutes(latest, now)
    rejection_reasons: list[str] = []
    if len(normalized) < min_rows:
        rejection_reasons.append(f"row_count_below_required_{min_rows}")
    if latest is None:
        rejection_reasons.append("latest_timestamp_missing")
    elif age_minutes is not None:
        max_age_minutes = stale_threshold_for_timeframe(timeframe).total_seconds() / 60
        if age_minutes > max_age_minutes:
            rejection_reasons.append(f"stale_latest_timestamp_age_minutes_gt_{max_age_minutes:g}")
    if synthetic_data_used:
        rejection_reasons.append(force_invalid_reason or "synthetic_data_not_valid_for_research_decisions")
    elif force_invalid_reason:
        rejection_reasons.append(force_invalid_reason)
    return ResearchDataReport(
        timeframe=timeframe,
        source_used=source_used,
        latest_timestamp=latest.isoformat() if latest is not None else None,
        data_age_minutes=age_minutes,
        row_count=int(len(normalized)),
        synthetic_data_used=synthetic_data_used,
        research_result_valid=not rejection_reasons,
        rejection_reason=";".join(rejection_reasons) if rejection_reasons else None,
    )


def _minimum_research_rows(settings: Settings, limit: int) -> int:
    return max(1, min(int(limit), max(MIN_RESEARCH_TRADES * 10, int(settings.min_training_rows))))


def _latest_timestamp(bars: pd.DataFrame) -> pd.Timestamp | None:
    if bars.empty or "timestamp" not in bars:
        return None
    latest = pd.Timestamp(bars["timestamp"].iloc[-1])
    if latest.tzinfo is None:
        latest = latest.tz_localize("UTC")
    else:
        latest = latest.tz_convert("UTC")
    return latest


def _data_age_minutes(latest: pd.Timestamp | None, now: datetime) -> float | None:
    if latest is None:
        return None
    current_time = _utc_timestamp(now)
    return max(0.0, (pd.Timestamp(current_time) - latest).total_seconds() / 60)


def _utc_timestamp(value: Any) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.to_pydatetime()


def _data_report_to_rejected_source(report: ResearchDataReport) -> dict[str, Any]:
    return {
        "source": report.source_used,
        "status": "stale" if _report_is_stale(report) else "rejected",
        "reason": report.rejection_reason,
        "latest_timestamp": report.latest_timestamp,
        "data_age_minutes": report.data_age_minutes,
        "row_count": report.row_count,
    }


def _report_is_stale(report: ResearchDataReport) -> bool:
    return "stale_latest_timestamp" in (report.rejection_reason or "")


def _with_rejected_sources(
    report: ResearchDataReport,
    rejected_sources: list[dict[str, Any]],
) -> ResearchDataReport:
    return ResearchDataReport(
        timeframe=report.timeframe,
        source_used=report.source_used,
        latest_timestamp=report.latest_timestamp,
        data_age_minutes=report.data_age_minutes,
        row_count=report.row_count,
        synthetic_data_used=report.synthetic_data_used,
        research_result_valid=report.research_result_valid,
        rejection_reason=report.rejection_reason,
        rejected_sources=tuple(rejected_sources),
    )


def _coerce_data_report(timeframe: str, value: ResearchDataReport | dict[str, Any]) -> ResearchDataReport:
    if isinstance(value, ResearchDataReport):
        return value
    source_used = str(value.get("source_used", value.get("source", "unknown")))
    synthetic_data_used = bool(value.get("synthetic_data_used", source_used.startswith("synthetic")))
    return ResearchDataReport(
        timeframe=timeframe,
        source_used=source_used,
        latest_timestamp=value.get("latest_timestamp"),
        data_age_minutes=value.get("data_age_minutes"),
        row_count=int(value.get("row_count", 0) or 0),
        synthetic_data_used=synthetic_data_used,
        research_result_valid=bool(value.get("research_result_valid", not synthetic_data_used)),
        rejection_reason=value.get("rejection_reason"),
        rejected_sources=tuple(value.get("rejected_sources", ())),
    )


def _summary_source_reports(
    *,
    data_source_reports: dict[str, ResearchDataReport | dict[str, Any]] | None,
    data_sources: dict[str, str] | None,
) -> dict[str, ResearchDataReport]:
    if data_source_reports:
        return {
            timeframe: _coerce_data_report(timeframe, report)
            for timeframe, report in data_source_reports.items()
        }
    return {
        timeframe: ResearchDataReport(
            timeframe=timeframe,
            source_used=source,
            latest_timestamp=None,
            data_age_minutes=None,
            row_count=0,
            synthetic_data_used=source.startswith("synthetic"),
            research_result_valid=not source.startswith("synthetic"),
        )
        for timeframe, source in (data_sources or {}).items()
    }


def _data_report_to_dict(report: ResearchDataReport) -> dict[str, Any]:
    return {
        "source_used": report.source_used,
        "latest_timestamp": report.latest_timestamp,
        "data_age_minutes": report.data_age_minutes,
        "row_count": report.row_count,
        "synthetic_data_used": report.synthetic_data_used,
        "research_result_valid": report.research_result_valid,
        "rejection_reason": report.rejection_reason,
        "rejected_sources": list(report.rejected_sources),
    }


def _latest_timestamp_from_error(exc: Exception) -> str | None:
    marker = "latest_timestamp="
    message = str(exc)
    if marker not in message:
        return None
    value = message.split(marker, 1)[1].split(" ", 1)[0]
    return value or None


def _data_age_minutes_from_iso(value: str | None, now: datetime) -> float | None:
    if value is None:
        return None
    return _data_age_minutes(pd.Timestamp(value), now)


def _synthetic_research_mode_enabled() -> bool:
    return _env_flag("RESEARCH_ALLOW_SYNTHETIC_FALLBACK") or _env_flag("RESEARCH_DEMO_MODE")


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _signal_row(
    row: pd.Series,
    *,
    signal: Any,
    regime: str,
    config: ResearchConfig,
) -> dict[str, Any]:
    return {
        "timestamp": row["timestamp"],
        "close": float(row["close"]),
        "orderbook_spread": float(row.get("orderbook_spread", 0.0)),
        "quote_imbalance": float(row.get("quote_imbalance", 0.0)),
        "scalping_spread_bps": float(row.get("scalping_spread_bps", 0.0)),
        "scalping_quote_imbalance": float(row.get("scalping_quote_imbalance", 0.0)),
        "strategy_name": signal.strategy_name,
        "entry_reason": signal.reason,
        "strategy_score": float(signal.score),
        "strategy_confidence": float(signal.confidence),
        "quant_score": float(signal.score),
        "quant_confidence": float(signal.confidence),
        "regime": regime,
        "blocked_by": None if signal.action == "buy" else _research_block_bucket(signal.reason),
        "block_reason": None if signal.action == "buy" else signal.reason,
        "ml_buy_probability": 1.0 if signal.action == "buy" else 0.0,
        "ml_sell_probability": 0.0 if signal.action == "buy" else 1.0,
        "_probability": 1.0 if signal.action == "buy" else 0.0,
        "research_timeframe": config.timeframe,
        "research_take_profit_pct": config.take_profit_pct,
        "research_stop_loss_pct": config.stop_loss_pct,
        "research_max_hold_bars": config.max_hold_bars,
    }


def _research_block_bucket(reason: str) -> str:
    if reason.startswith("regime") or reason in {"trend_not_confirmed", "volatility_not_tradeable"}:
        return "regime_filter"
    if "spread" in reason:
        return "spread"
    return "quant_strategy"


def _exit_result(gross_return: float, exit_reason: str, hold_bars: int, exit_high: float, exit_low: float) -> dict[str, Any]:
    return {
        "gross_return": float(gross_return),
        "exit_reason": exit_reason,
        "hold_bars": int(hold_bars),
        "exit_high": float(exit_high),
        "exit_low": float(exit_low),
    }


def _empty_research_metrics(reason: str) -> dict[str, Any]:
    return {
        "valid": False,
        "reason": reason,
        "number_of_trades": 0,
        "net_return_pct": 0.0,
        "gross_return_pct": 0.0,
        "profit_factor_net": 0.0,
        "max_drawdown_pct": 0.0,
        "win_rate_net": 0.0,
        "expectancy": 0.0,
        "trade_details": [],
    }


def single_trade_return_concentration(trade_details: list[dict[str, Any]]) -> float:
    returns = [_metric_float(trade.get("net_return_pct", trade.get("net_return"))) for trade in trade_details]
    positive_returns = [value for value in returns if value > 0]
    total_positive_return = sum(positive_returns)
    if total_positive_return <= 0:
        return 0.0
    return max(positive_returns) / total_positive_return


def research_rank_score(metrics: dict[str, Any], readiness: dict[str, Any]) -> float:
    base = _metric_float(metrics.get("net_return_pct")) * 10_000
    base += min(5.0, _profit_factor_value(metrics.get("profit_factor_net"))) * 10
    base += int(metrics.get("number_of_trades", 0) or 0) * 0.1
    base -= _metric_float(metrics.get("max_drawdown_pct")) * 1_000
    if not readiness.get("economically_viable"):
        base -= 1_000_000
    if not readiness.get("paper_forward_eligible"):
        base -= 10_000
    return base


def _metric_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def _profit_factor_value(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isinf(parsed):
        return 1_000_000.0 if parsed > 0 else 0.0
    if math.isnan(parsed):
        return 0.0
    return parsed


def _csv_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "parameter_set_id",
        "strategy_name",
        "timeframe",
        "take_profit_pct",
        "stop_loss_pct",
        "max_hold_bars",
        "number_of_trades",
        "gross_return_pct",
        "net_return_pct",
        "profit_factor_net",
        "max_drawdown_pct",
        "win_rate_net",
        "expectancy",
        "round_trip_estimated_cost_pct",
        "promotion_required_return_pct",
        "gross_winners_became_net_losers",
        "single_trade_return_concentration",
        "fallback_prediction_used",
        "active_model_valid",
        "economically_viable",
        "paper_forward_eligible",
        "rejection_reasons",
        "rank_score",
    ]
    extras = sorted({key for row in rows for key in row} - set(preferred))
    return [field for field in preferred if any(field in row for row in rows)] + extras


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(_json_safe(value), sort_keys=True)
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_json_safe(child) for child in value]
    if isinstance(value, tuple):
        return [_json_safe(child) for child in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value
    return value


if __name__ == "__main__":
    asyncio.run(main())
