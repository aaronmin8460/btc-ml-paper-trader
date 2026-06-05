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
from app.data.market_data import MarketDataClient
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


async def main() -> None:
    settings = research_settings(get_settings())
    bar_limit = int(os.getenv("RESEARCH_BAR_LIMIT", str(max(1500, settings.min_training_rows + 500))))
    report = await run_higher_timeframe_research(settings, bar_limit=bar_limit, output_dir=Path(settings.log_dir))
    print(json.dumps(report, indent=2, default=str))


async def run_higher_timeframe_research(
    base_settings: Settings,
    *,
    bar_limit: int = 1500,
    client: MarketDataClient | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    settings = research_settings(base_settings)
    client = client or MarketDataClient(settings)
    active_model = ModelRegistry(settings).validate_active_model()
    bars_by_timeframe: dict[str, pd.DataFrame] = {}
    data_sources: dict[str, str] = {}
    for timeframe in RESEARCH_TIMEFRAMES:
        bars, source = await _fetch_research_bars(client, settings, timeframe=timeframe, limit=bar_limit)
        bars_by_timeframe[timeframe] = bars
        data_sources[timeframe] = source
    rows = evaluate_research_configs(
        bars_by_timeframe,
        settings,
        active_model_valid=active_model.valid,
        active_model_status=active_model.to_dict(),
    )
    output_path = output_dir or Path(settings.log_dir)
    csv_path = output_path / "higher_timeframe_research.csv"
    summary_path = output_path / "higher_timeframe_research_summary.json"
    summary = build_research_summary(
        rows,
        settings,
        data_sources=data_sources,
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
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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
        if bars.empty:
            metrics = _empty_research_metrics("no_bars")
        else:
            trades, signal_frame = build_research_trades(bars, candidate_settings, config)
            metrics = calculate_fee_aware_metrics(trades, candidate_settings, signal_frame=signal_frame)
        readiness = paper_forward_readiness_gate(
            metrics,
            candidate_settings,
            fallback_prediction_used=False,
            active_model_valid=active_model_valid,
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
                "fallback_prediction_used": False,
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
) -> dict[str, Any]:
    reasons: list[str] = []
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
    data_sources: dict[str, str],
    csv_path: Path,
    summary_path: Path,
    active_model_status: dict[str, Any],
) -> dict[str, Any]:
    ranked = sorted(rows, key=lambda row: float(row.get("rank_score", -1_000_000.0)), reverse=True)
    economically_viable = [row for row in ranked if row.get("economically_viable")]
    eligible = [row for row in ranked if row.get("paper_forward_eligible")]
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
            "csv_path": str(csv_path),
            "summary_path": str(summary_path),
            "data_sources": data_sources,
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
            "notes": [
                "Higher-timeframe research is offline analysis only and never enables trading.",
                "No configuration is auto-applied or promoted from this report.",
                "Paper-forward eligibility also requires the existing active model registry to validate.",
            ],
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
    client: MarketDataClient,
    settings: Settings,
    *,
    timeframe: str,
    limit: int,
) -> tuple[pd.DataFrame, str]:
    if isinstance(client, MarketDataClient) and not (settings.alpaca_api_key and settings.alpaca_secret_key):
        return MarketDataClient.synthetic_btc_bars(limit=limit, timeframe=timeframe), "synthetic_no_alpaca_credentials"
    try:
        bars = await client.fetch_bars(settings.symbol, timeframe=timeframe, limit=limit, force_refresh=True)
        return bars, "market_data_client"
    except Exception:
        return MarketDataClient.synthetic_btc_bars(limit=limit, timeframe=timeframe), "synthetic_fallback_after_fetch_error"


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
