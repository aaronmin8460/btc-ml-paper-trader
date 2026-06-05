from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from app.broker.paper_execution import PaperFillResult, simulate_limit_ioc_order, simulate_market_order
from app.config import Settings
from app.data.feature_engineering import BAR_FEATURE_COLUMNS
from app.ml.model import MLSignalModel
from app.risk.risk_manager import PositionState
from app.strategy.strategies import MarketContext, MarketRegimeFilter, MeanReversionScalpingStrategy, MomentumBreakoutStrategy


BLOCKED_SIGNAL_BUCKETS = (
    "regime_filter",
    "quant_strategy",
    "ml_filter",
    "risk_manager",
    "spread",
    "quote_imbalance",
    "api_budget",
    "cooldown",
    "ioc_cancel_guard",
    "stale_market_data",
    "active_model_invalid",
    "fallback_prediction_not_allowed",
)

ZERO_BACKTEST_METRICS = {
    "number_of_trades": 0,
    "number_of_canceled_orders": 0,
    "partial_fill_count": 0,
    "evaluated_signal_count": 0,
    "ambiguous_candle_count": 0,
    "ambiguous_candle_ratio": 0.0,
    "win_rate": 0.0,
    "win_rate_net": 0.0,
    "gross_return": 0.0,
    "net_return": 0.0,
    "gross_return_pct": 0.0,
    "net_return_pct": 0.0,
    "average_gross_win": 0.0,
    "average_gross_loss": 0.0,
    "average_net_win": 0.0,
    "average_net_loss": 0.0,
    "average_trade_pnl": 0.0,
    "expectancy": 0.0,
    "best_trade_pnl": 0.0,
    "worst_trade_pnl": 0.0,
    "profit_factor_gross": 0.0,
    "profit_factor_net": 0.0,
    "max_drawdown": 0.0,
    "max_drawdown_pct": 0.0,
    "fees_paid_estimate": 0.0,
    "slippage_paid_estimate": 0.0,
    "spread_paid_estimate": 0.0,
    "total_fees": 0.0,
    "total_slippage": 0.0,
    "total_spread_cost": 0.0,
    "average_hold_bars": 0.0,
    "starting_equity": None,
    "ending_equity": None,
    "trade_details": [],
    "strategy_level_metrics": {},
    "regime_level_metrics": {},
    "blocked_signal_metrics": {name: 0 for name in BLOCKED_SIGNAL_BUCKETS},
}


def backtest_assumptions(settings: Settings, *, spread_available: bool = False) -> dict[str, Any]:
    fee_bps = settings.taker_fee_bps if settings.backtest_use_taker_fees else settings.maker_fee_bps
    spread_source = "unavailable"
    if spread_available:
        spread_source = "scalping_spread_pct" if settings.scalping_mode_enabled else "orderbook_spread"
    if not spread_available and settings.scalping_mode_enabled:
        spread_source = "configured_max_spread_bps"
    return {
        "symbol": settings.symbol,
        "paper_trading_only": settings.paper_trading_only,
        "long_only": True,
        "fee_model": "taker" if settings.backtest_use_taker_fees else "maker",
        "fee_bps_per_side": fee_bps,
        "taker_fee_bps": settings.taker_fee_bps,
        "maker_fee_bps": settings.maker_fee_bps,
        "slippage_bps_per_side": settings.slippage_bps,
        "fee_and_slippage_applied_on": "entry_and_exit",
        "spread_source": spread_source,
        "spread_cost_model": "simulated_touch_price_half_spread_paid_per_fill",
        "execution_model": "shared_local_paper_execution_simulator",
        "limit_order_behavior": "ioc_cancel_or_partial_fill",
        "partial_fill_scoring": "matched_quantity_return_with_all_simulated_fill_costs_charged",
        "ambiguous_candle_behavior": "stop_loss_first",
        "hold_duration_model": "row_hold_bars_when_available_else_label_horizon",
        "order_notional_usd": settings.order_notional_usd,
        "return_metrics_unit": "fraction_of_traded_notional",
        "paid_estimates_unit": "usd_from_simulated_fills",
        "gross_return_model": (
            "net_profit_scalping_labels" if settings.scalping_mode_enabled else "triple_barrier_take_profit_or_stop_loss"
        ),
    }


def calculate_fee_aware_metrics(
    trades: pd.DataFrame,
    settings: Settings,
    *,
    signal_frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    signals = _normalise_signal_frame(signal_frame if signal_frame is not None else trades)
    if trades.empty:
        return {
            "valid": False,
            "reason": "no_trades",
            **ZERO_BACKTEST_METRICS,
            "evaluated_signal_count": int(len(signals)),
            **_signal_observability_metrics([], signals, {}, {}),
        }
    if "buy_quality_label" not in trades.columns:
        return {
            "valid": False,
            "reason": "missing_buy_quality_label",
            **ZERO_BACKTEST_METRICS,
            "evaluated_signal_count": int(len(signals)),
            **_signal_observability_metrics([], signals, {}, {}),
        }

    take_profit_pct = _take_profit_pct(settings)
    stop_loss_pct = _stop_loss_pct(settings)
    fee_bps = settings.taker_fee_bps if settings.backtest_use_taker_fees else settings.maker_fee_bps
    notional = float(settings.order_notional_usd)
    gross_returns: list[float] = []
    net_trade_returns: list[float] = []
    equity_returns: list[float] = []
    hold_bars: list[float] = []
    total_fees = 0.0
    total_slippage = 0.0
    total_spread_cost = 0.0
    canceled_orders = 0
    partial_fills = 0
    ambiguous_candles = 0
    trade_details: list[dict[str, Any]] = []
    strategy_execution: dict[str, dict[str, int]] = {}
    regime_execution: dict[str, dict[str, int]] = {}

    for _, row in trades.iterrows():
        strategy_name = str(row.get("strategy_name") or "ml_walk_forward")
        regime_name = str(row.get("regime") or "unknown")
        entry_reason = str(row.get("entry_reason") or row.get("strategy_entry_reason") or "ml_probability_threshold")
        entry_reference = _entry_reference_price(row)
        gross_exit_return, ambiguous = _resolved_gross_return(
            row,
            entry_reference=entry_reference,
            take_profit_pct=take_profit_pct,
            stop_loss_pct=stop_loss_pct,
        )
        ambiguous_candles += int(ambiguous)
        _increment_execution(strategy_execution, strategy_name, "entry_attempts")
        _increment_execution(regime_execution, regime_name, "entry_attempts")
        if ambiguous:
            _increment_execution(strategy_execution, strategy_name, "ambiguous_candles")
            _increment_execution(regime_execution, regime_name, "ambiguous_candles")
        entry_fill = _simulate_backtest_order(
            row,
            settings,
            side="buy",
            reference_price=entry_reference,
            fee_bps=fee_bps,
            notional=notional,
        )
        total_fees += entry_fill.fee_amount
        total_slippage += entry_fill.slippage_amount
        total_spread_cost += entry_fill.spread_cost_amount
        canceled_orders += int(entry_fill.status == "canceled")
        partial_fills += int(entry_fill.status == "partially_filled")
        _record_execution_fill(strategy_execution, strategy_name, entry_fill, side="entry")
        _record_execution_fill(regime_execution, regime_name, entry_fill, side="entry")
        if entry_fill.filled_qty <= 0:
            continue

        _increment_execution(strategy_execution, strategy_name, "exit_attempts")
        _increment_execution(regime_execution, regime_name, "exit_attempts")
        exit_reference = entry_reference * (1 + gross_exit_return)
        exit_fill = _simulate_backtest_order(
            row,
            settings,
            side="sell",
            reference_price=exit_reference,
            fee_bps=fee_bps,
            qty=entry_fill.filled_qty,
        )
        total_fees += exit_fill.fee_amount
        total_slippage += exit_fill.slippage_amount
        total_spread_cost += exit_fill.spread_cost_amount
        canceled_orders += int(exit_fill.status == "canceled")
        partial_fills += int(exit_fill.status == "partially_filled")
        _record_execution_fill(strategy_execution, strategy_name, exit_fill, side="exit")
        _record_execution_fill(regime_execution, regime_name, exit_fill, side="exit")

        entry_cost = entry_fill.fee_amount + entry_fill.slippage_amount + entry_fill.spread_cost_amount
        if exit_fill.filled_qty <= 0:
            if entry_cost > 0:
                equity_returns.append(-entry_cost / notional)
            continue

        matched_qty = min(entry_fill.filled_qty, exit_fill.filled_qty)
        filled_fraction = matched_qty / entry_fill.requested_qty
        trade_gross_return = gross_exit_return * filled_fraction
        trade_cost = entry_cost + exit_fill.fee_amount + exit_fill.slippage_amount + exit_fill.spread_cost_amount
        trade_net_return = trade_gross_return - trade_cost / notional
        trade_fees = entry_fill.fee_amount + exit_fill.fee_amount
        trade_slippage = entry_fill.slippage_amount + exit_fill.slippage_amount
        trade_spread_cost = entry_fill.spread_cost_amount + exit_fill.spread_cost_amount
        hold_bar_count = _hold_bars(row, settings)
        gross_returns.append(trade_gross_return)
        net_trade_returns.append(trade_net_return)
        equity_returns.append(trade_net_return)
        hold_bars.append(hold_bar_count)
        trade_details.append(
            {
                "strategy_name": strategy_name,
                "regime": regime_name,
                "entry_reason": entry_reason,
                "exit_reason": str(row.get("buy_exit_reason") or ("label_take_profit" if int(row.get("buy_quality_label", 0)) == 1 else "label_stop_or_timeout")),
                "blocked_by": _clean_optional(row.get("blocked_by")),
                "block_reason": _clean_optional(row.get("block_reason")),
                "ml_buy_probability": _row_float(row, "ml_buy_probability", "_probability"),
                "ml_sell_probability": _row_float(row, "ml_sell_probability"),
                "quant_score": _row_float(row, "quant_score", "strategy_score"),
                "quant_confidence": _row_float(row, "quant_confidence", "strategy_confidence"),
                "gross_return": float(trade_gross_return),
                "net_return": float(trade_net_return),
                "gross_return_pct": float(trade_gross_return),
                "net_return_pct": float(trade_net_return),
                "fees": float(trade_fees),
                "fee_amount": float(trade_fees),
                "slippage": float(trade_slippage),
                "slippage_amount": float(trade_slippage),
                "spread_cost": float(trade_spread_cost),
                "hold_bars": float(hold_bar_count),
                "filled_fraction": float(filled_fraction),
                "canceled_orders": int(entry_fill.status == "canceled") + int(exit_fill.status == "canceled"),
                "partial_fills": int(entry_fill.status == "partially_filled")
                + int(exit_fill.status == "partially_filled"),
                "ambiguous_candle": bool(ambiguous),
            }
        )

    gross_values = np.asarray(gross_returns, dtype=float)
    net_trade_values = np.asarray(net_trade_returns, dtype=float)
    equity_values = np.asarray(equity_returns, dtype=float)
    trade_pnls = net_trade_values * notional
    gross_return = float(gross_values.sum())
    net_return = gross_return - (total_fees + total_slippage + total_spread_cost) / notional
    max_drawdown = _max_drawdown(equity_values)
    number_of_trades = len(gross_values)
    metrics = {
        "valid": number_of_trades > 0,
        "reason": None if number_of_trades > 0 else "no_filled_trades",
        "number_of_trades": int(number_of_trades),
        "number_of_canceled_orders": canceled_orders,
        "partial_fill_count": partial_fills,
        "evaluated_signal_count": int(len(signals)),
        "ambiguous_candle_count": ambiguous_candles,
        "ambiguous_candle_ratio": ambiguous_candles / len(trades),
        "win_rate": float((gross_values > 0).mean()) if number_of_trades else 0.0,
        "win_rate_net": float((net_trade_values > 0).mean()) if number_of_trades else 0.0,
        "gross_return": gross_return,
        "net_return": net_return,
        "gross_return_pct": gross_return,
        "net_return_pct": net_return,
        "average_gross_win": _mean_or_zero(gross_values[gross_values > 0]),
        "average_gross_loss": _mean_or_zero(gross_values[gross_values < 0]),
        "average_net_win": _mean_or_zero(net_trade_values[net_trade_values > 0]),
        "average_net_loss": _mean_or_zero(net_trade_values[net_trade_values < 0]),
        "average_trade_pnl": _mean_or_zero(trade_pnls),
        "expectancy": _mean_or_zero(net_trade_values),
        "best_trade_pnl": float(trade_pnls.max()) if len(trade_pnls) else 0.0,
        "worst_trade_pnl": float(trade_pnls.min()) if len(trade_pnls) else 0.0,
        "profit_factor_gross": _profit_factor(gross_values),
        "profit_factor_net": _profit_factor(equity_values),
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown,
        "fees_paid_estimate": total_fees,
        "slippage_paid_estimate": total_slippage,
        "spread_paid_estimate": total_spread_cost,
        "total_fees": total_fees,
        "total_slippage": total_slippage,
        "total_spread_cost": total_spread_cost,
        "average_hold_bars": _mean_or_zero(np.asarray(hold_bars, dtype=float)),
        "starting_equity": notional,
        "ending_equity": notional * (1 + net_return),
        "trade_details": trade_details,
        **_signal_observability_metrics(trade_details, signals, strategy_execution, regime_execution),
    }

    return metrics


def walk_forward_fee_aware_backtest(
    df: pd.DataFrame,
    settings: Settings,
    *,
    min_train_rows: int,
    threshold: float,
    folds: int = 3,
    feature_columns: list[str] | None = None,
) -> dict[str, Any]:
    if len(df) < min_train_rows:
        return {
            "valid": False,
            "reason": "not_enough_rows",
            "rows": int(len(df)),
            "min_train_rows": int(min_train_rows),
            **ZERO_BACKTEST_METRICS,
        }

    prediction_frame = _walk_forward_prediction_frame(
        df,
        min_train_rows=min_train_rows,
        folds=folds,
        feature_columns=feature_columns,
    )
    if prediction_frame.empty:
        return {
            "valid": False,
            "reason": "no_validation_folds",
            "rows": int(len(df)),
            "min_train_rows": int(min_train_rows),
            **ZERO_BACKTEST_METRICS,
        }

    prediction_frame = _annotate_strategy_candidates(prediction_frame, settings, threshold=threshold)
    prediction_frame["entry_allowed"] = prediction_frame["blocked_by"].isna()
    trades = prediction_frame.loc[prediction_frame["entry_allowed"]].copy()
    metrics = calculate_fee_aware_metrics(trades, settings, signal_frame=prediction_frame)
    metrics["rows"] = int(len(df))
    metrics["validation_rows"] = int(len(prediction_frame))
    metrics["threshold"] = float(threshold)
    if not metrics["valid"] and metrics.get("reason") == "no_trades":
        metrics["reason"] = "no_trades_above_threshold"
    return metrics


def _walk_forward_prediction_frame(
    df: pd.DataFrame,
    *,
    min_train_rows: int,
    folds: int,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    feature_columns = feature_columns or BAR_FEATURE_COLUMNS
    fold_size = max(50, (len(df) - min_train_rows) // max(1, folds))
    validation_frames: list[pd.DataFrame] = []
    for fold in range(folds):
        train_end = min_train_rows + fold * fold_size
        valid_end = min(len(df), train_end + fold_size)
        if valid_end <= train_end + 10:
            continue
        train = df.iloc[:train_end]
        valid = df.iloc[train_end:valid_end]
        if "buy_quality_label" in train.columns and train["buy_quality_label"].astype(int).nunique() < 2:
            continue
        model = MLSignalModel(feature_columns=feature_columns).train(train)
        frame = valid.copy()
        frame["_probability"] = model.predict_proba(valid)
        frame["_walk_forward_split"] = fold
        frame["_train_start_timestamp"] = _frame_timestamp(train, first=True)
        frame["_train_end_timestamp"] = _frame_timestamp(train, first=False)
        frame["_validation_start_timestamp"] = _frame_timestamp(valid, first=True)
        frame["_validation_end_timestamp"] = _frame_timestamp(valid, first=False)
        validation_frames.append(frame)
    if not validation_frames:
        return pd.DataFrame()
    return pd.concat(validation_frames, ignore_index=True)


def _frame_timestamp(frame: pd.DataFrame, *, first: bool) -> str | None:
    if "timestamp" not in frame.columns or frame.empty:
        return None
    value = frame["timestamp"].iloc[0 if first else -1]
    try:
        timestamp = pd.Timestamp(value)
    except Exception:
        return str(value)
    if pd.isna(timestamp):
        return None
    return timestamp.isoformat()


def _annotate_strategy_candidates(
    frame: pd.DataFrame,
    settings: Settings,
    *,
    threshold: float | None = None,
) -> pd.DataFrame:
    annotated = frame.copy()
    threshold_value = (
        float(threshold)
        if threshold is not None
        else float(settings.scalping_buy_probability_floor if settings.scalping_mode_enabled else settings.min_buy_probability)
    )
    if annotated.empty or not settings.scalping_mode_enabled:
        if "strategy_name" not in annotated.columns:
            annotated["strategy_name"] = "ml_walk_forward"
        if "entry_reason" not in annotated.columns:
            annotated["entry_reason"] = "ml_probability_threshold"
        if "strategy_score" not in annotated.columns:
            annotated["strategy_score"] = 0.0
        if "strategy_confidence" not in annotated.columns:
            annotated["strategy_confidence"] = 0.0
        if "regime" not in annotated.columns:
            annotated["regime"] = "ml_only"
        return _apply_ml_signal_annotations(annotated, threshold_value)

    regime_filter = MarketRegimeFilter(settings)
    strategies = [MeanReversionScalpingStrategy(settings), MomentumBreakoutStrategy(settings)]
    strategy_names: list[str] = []
    entry_reasons: list[str] = []
    strategy_scores: list[float] = []
    strategy_confidences: list[float] = []
    regimes: list[str] = []
    blocked_by_values: list[str | None] = []
    block_reasons: list[str | None] = []
    for _, row in annotated.iterrows():
        candidate_row = _backtest_strategy_row(row, settings)
        regime = regime_filter.detect(candidate_row)
        context = MarketContext(regime=regime, risk_permits_evaluation=True)
        candidates = [
            strategy.generate_signal(
                feature_row=candidate_row,
                prediction=None,
                position=PositionState(),
                quote=None,
                market_context=context,
            )
            for strategy in strategies
        ]
        allowed = [
            candidate
            for candidate in candidates
            if candidate.action == "buy" and regime_filter.allows(regime, candidate.strategy_name)[0]
        ]
        if allowed:
            selected = max(allowed, key=lambda candidate: (candidate.score, candidate.confidence, candidate.strategy_name))
            strategy_names.append(selected.strategy_name)
            entry_reasons.append(selected.reason)
            strategy_scores.append(float(selected.score))
            strategy_confidences.append(float(selected.confidence))
            blocked_by_values.append(None)
            block_reasons.append(None)
        else:
            reason = _backtest_quant_block_reason(regime, candidates)
            strategy_names.append("quant_strategy_blocked")
            entry_reasons.append(reason)
            strategy_scores.append(0.0)
            strategy_confidences.append(0.0)
            blocked_by_values.append(_backtest_blocked_by_for_reason(reason))
            block_reasons.append(reason)
        regimes.append(regime.regime)
    annotated["strategy_name"] = strategy_names
    annotated["entry_reason"] = entry_reasons
    annotated["strategy_score"] = strategy_scores
    annotated["strategy_confidence"] = strategy_confidences
    annotated["regime"] = regimes
    annotated["blocked_by"] = blocked_by_values
    annotated["block_reason"] = block_reasons
    return _apply_ml_signal_annotations(annotated, threshold_value)


def _apply_ml_signal_annotations(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    annotated = frame.copy()
    if "blocked_by" not in annotated.columns:
        annotated["blocked_by"] = None
    if "block_reason" not in annotated.columns:
        annotated["block_reason"] = None
    probabilities = pd.to_numeric(
        annotated["_probability"] if "_probability" in annotated.columns else pd.Series(index=annotated.index, dtype=float),
        errors="coerce",
    )
    annotated["ml_buy_probability"] = probabilities
    annotated["ml_sell_probability"] = (1.0 - probabilities).clip(lower=0.0, upper=1.0)
    strategy_score = (
        annotated["strategy_score"] if "strategy_score" in annotated.columns else pd.Series(0.0, index=annotated.index)
    )
    strategy_confidence = (
        annotated["strategy_confidence"]
        if "strategy_confidence" in annotated.columns
        else pd.Series(0.0, index=annotated.index)
    )
    annotated["quant_score"] = pd.to_numeric(strategy_score, errors="coerce").fillna(0.0)
    annotated["quant_confidence"] = pd.to_numeric(strategy_confidence, errors="coerce").fillna(0.0)
    below_threshold = probabilities < threshold
    unblocked = annotated["blocked_by"].isna()
    annotated.loc[below_threshold & unblocked, "blocked_by"] = "ml_filter"
    annotated.loc[below_threshold & unblocked, "block_reason"] = "ml_buy_probability_below_threshold"
    return annotated


def _backtest_quant_block_reason(regime, candidates: list[Any]) -> str:
    if getattr(regime, "regime", None) in {"too_volatile", "not_tradeable"}:
        return str(getattr(regime, "reason", None) or "regime_filter_blocked")
    buy_candidates = [candidate for candidate in candidates if candidate.action == "buy"]
    if buy_candidates:
        return "strategy_candidate_blocked_by_regime"
    if candidates:
        return str(candidates[0].reason)
    return "no_quant_strategy_candidate"


def _backtest_blocked_by_for_reason(reason: str) -> str:
    if reason in {"spread_too_wide", "spread_unavailable"}:
        return "spread"
    if reason in {"quote_imbalance_too_weak", "quote_imbalance_unavailable", "quote_imbalance_not_supportive"}:
        return "quote_imbalance"
    if reason in {"volatility_too_high", "invalid_price", "regime_filter_blocked", "strategy_candidate_blocked_by_regime"}:
        return "regime_filter"
    if reason.startswith("regime_"):
        return "regime_filter"
    return "quant_strategy"


def _backtest_strategy_row(row: pd.Series, settings: Settings) -> pd.Series:
    out = row.copy()
    if _row_float(out, "scalping_spread_bps") is None and _row_float(out, "scalping_spread_pct") is None:
        out["scalping_spread_bps"] = max(0.0, float(settings.max_spread_bps))
    if _row_float(out, "scalping_quote_imbalance") is None:
        out["scalping_quote_imbalance"] = 0.0
    return out


def _simulate_backtest_order(
    row: pd.Series,
    settings: Settings,
    *,
    side: str,
    reference_price: float,
    fee_bps: float,
    notional: float | None = None,
    qty: float | None = None,
) -> PaperFillResult:
    quote = _execution_quote(row, settings, side=side, reference_price=reference_price)
    kwargs = {
        "symbol": settings.symbol,
        "side": side,
        "bid_price": quote["bid_price"],
        "ask_price": quote["ask_price"],
        "bid_size": quote["bid_size"],
        "ask_size": quote["ask_size"],
        "latest_price": reference_price,
        "notional": notional,
        "qty": qty,
        "fee_bps": fee_bps,
        "slippage_bps": settings.slippage_bps,
    }
    if settings.order_type == "limit":
        touch_price = quote["ask_price"] if side == "buy" else quote["bid_price"]
        kwargs["limit_price"] = _backtest_limit_price(row, settings, side=side, touch_price=touch_price)
        return simulate_limit_ioc_order(**kwargs)
    return simulate_market_order(**kwargs)


def _execution_quote(
    row: pd.Series,
    settings: Settings,
    *,
    side: str,
    reference_price: float,
) -> dict[str, float | None]:
    phase = "entry" if side == "buy" else "exit"
    bid_price = _row_positive_float(row, f"{phase}_bid_price")
    ask_price = _row_positive_float(row, f"{phase}_ask_price")
    if phase == "entry":
        bid_price = bid_price or _row_positive_float(row, "bid_price")
        ask_price = ask_price or _row_positive_float(row, "ask_price")

    spread_pct = _row_non_negative_float(row, f"{phase}_spread_pct", "scalping_spread_pct", "orderbook_spread")
    if spread_pct == 0 and settings.scalping_mode_enabled:
        spread_pct = None
    observed_spread_pct = _observed_quote_spread_pct(row)
    spread_pct = observed_spread_pct if observed_spread_pct is not None else spread_pct
    if bid_price is not None and ask_price is not None and ask_price >= bid_price:
        spread_pct = (ask_price - bid_price) / ((ask_price + bid_price) / 2)
    else:
        spread_pct = spread_pct if spread_pct is not None else _fallback_spread_pct(settings)
        bid_price = reference_price * (1 - spread_pct / 2)
        ask_price = reference_price * (1 + spread_pct / 2)

    bid_size = _row_non_negative_float(row, f"{phase}_bid_size")
    ask_size = _row_non_negative_float(row, f"{phase}_ask_size")
    bid_size = bid_size if bid_size is not None else _row_non_negative_float(row, "bid_size")
    ask_size = ask_size if ask_size is not None else _row_non_negative_float(row, "ask_size")
    return {
        "bid_price": bid_price,
        "ask_price": ask_price,
        "bid_size": bid_size,
        "ask_size": ask_size,
    }


def _backtest_limit_price(row: pd.Series, settings: Settings, *, side: str, touch_price: float) -> float:
    phase = "entry" if side == "buy" else "exit"
    explicit = _row_positive_float(row, f"{phase}_limit_price")
    if explicit is None and side == "buy":
        explicit = _row_positive_float(row, "limit_price")
    if explicit is not None:
        return explicit
    offset = max(0.0, float(settings.limit_price_offset_bps)) / 10_000
    return touch_price * (1 + offset) if side == "buy" else touch_price * (1 - offset)


def _resolved_gross_return(
    row: pd.Series,
    *,
    entry_reference: float,
    take_profit_pct: float,
    stop_loss_pct: float,
) -> tuple[float, bool]:
    reason = str(row.get("buy_exit_reason") or "")
    ambiguous = reason == "ambiguous_stop_first" or _explicit_exit_bar_is_ambiguous(
        row,
        entry_reference=entry_reference,
        take_profit_pct=take_profit_pct,
        stop_loss_pct=stop_loss_pct,
    )
    if ambiguous:
        return -abs(stop_loss_pct), True

    label = int(row.get("buy_quality_label", 0))
    fallback = take_profit_pct if label == 1 else -abs(stop_loss_pct)
    return _row_float(row, "buy_exit_return_pct") or fallback, False


def _explicit_exit_bar_is_ambiguous(
    row: pd.Series,
    *,
    entry_reference: float,
    take_profit_pct: float,
    stop_loss_pct: float,
) -> bool:
    high = _row_positive_float(row, "backtest_exit_high")
    low = _row_positive_float(row, "backtest_exit_low")
    if high is None or low is None:
        return False
    return high >= entry_reference * (1 + take_profit_pct) and low <= entry_reference * (1 - abs(stop_loss_pct))


def _entry_reference_price(row: pd.Series) -> float:
    return _row_positive_float(row, "close", "latest_price") or 100.0


def _fallback_spread_pct(settings: Settings) -> float:
    return max(0.0, float(settings.max_spread_bps)) / 10_000 if settings.scalping_mode_enabled else 0.0


def _observed_quote_spread_pct(row: pd.Series) -> float | None:
    bid_price = _row_positive_float(row, "bid_price")
    ask_price = _row_positive_float(row, "ask_price")
    if bid_price is None or ask_price is None or ask_price < bid_price:
        return None
    return (ask_price - bid_price) / ((ask_price + bid_price) / 2)


def _hold_bars(row: pd.Series, settings: Settings) -> float:
    value = _row_positive_float(row, "buy_hold_bars", "hold_bars")
    if value is not None:
        return value
    return float(settings.label_horizon_bars if settings.scalping_mode_enabled else 12)


def _row_positive_float(row: pd.Series, *names: str) -> float | None:
    for name in names:
        value = _row_float(row, name)
        if value is not None and value > 0:
            return value
    return None


def _row_non_negative_float(row: pd.Series, *names: str) -> float | None:
    for name in names:
        value = _row_float(row, name)
        if value is not None and value >= 0:
            return value
    return None


def _row_float(row: pd.Series, *names: str) -> float | None:
    for name in names:
        try:
            value = float(row.get(name))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return None


def _mean_or_zero(values: np.ndarray) -> float:
    return float(values.mean()) if len(values) else 0.0


def _profit_factor(returns: np.ndarray) -> float | None:
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    gross_win = float(wins.sum()) if len(wins) else 0.0
    gross_loss = abs(float(losses.sum())) if len(losses) else 0.0
    if gross_loss == 0:
        return float("inf") if gross_win > 0 else 0.0
    return float(gross_win / gross_loss)


def _normalise_signal_frame(frame: pd.DataFrame) -> pd.DataFrame:
    signals = frame.copy()
    if signals.empty:
        return signals
    if "strategy_name" not in signals.columns:
        signals["strategy_name"] = "ml_walk_forward"
    if "regime" not in signals.columns:
        signals["regime"] = "unknown"
    if "blocked_by" not in signals.columns:
        signals["blocked_by"] = None
    if "block_reason" not in signals.columns:
        signals["block_reason"] = None
    if "entry_allowed" not in signals.columns:
        signals["entry_allowed"] = signals["blocked_by"].isna()
    return signals


def _increment_execution(execution: dict[str, dict[str, int]], key: str, field: str, value: int = 1) -> None:
    bucket = execution.setdefault(
        key,
        {
            "entry_attempts": 0,
            "exit_attempts": 0,
            "filled_entries": 0,
            "filled_exits": 0,
            "canceled_orders": 0,
            "partial_fills": 0,
            "ambiguous_candles": 0,
        },
    )
    bucket[field] = int(bucket.get(field, 0)) + int(value)


def _record_execution_fill(
    execution: dict[str, dict[str, int]],
    key: str,
    fill: PaperFillResult,
    *,
    side: str,
) -> None:
    if fill.status == "canceled":
        _increment_execution(execution, key, "canceled_orders")
    if fill.status == "partially_filled":
        _increment_execution(execution, key, "partial_fills")
    if fill.filled_qty > 0:
        _increment_execution(execution, key, "filled_entries" if side == "entry" else "filled_exits")


def _signal_observability_metrics(
    trade_details: list[dict[str, Any]],
    signal_frame: pd.DataFrame,
    strategy_execution: dict[str, dict[str, int]],
    regime_execution: dict[str, dict[str, int]],
) -> dict[str, Any]:
    signals = _normalise_signal_frame(signal_frame)
    return {
        "strategy_level_metrics": _strategy_level_metrics(trade_details, signals, strategy_execution),
        "regime_level_metrics": _regime_level_metrics(trade_details, signals, regime_execution),
        "blocked_signal_metrics": _blocked_signal_metrics(signals),
    }


def _strategy_level_metrics(
    trade_details: list[dict[str, Any]],
    signal_frame: pd.DataFrame,
    execution: dict[str, dict[str, int]],
) -> dict[str, dict[str, Any]]:
    names = set(execution)
    if not signal_frame.empty and "strategy_name" in signal_frame.columns:
        names.update(str(value or "unknown") for value in signal_frame["strategy_name"].dropna().unique())
    names.update(str(trade.get("strategy_name") or "unknown") for trade in trade_details)
    metrics: dict[str, dict[str, Any]] = {}
    for strategy_name in sorted(names):
        signal_rows = (
            signal_frame.loc[signal_frame["strategy_name"].astype(str) == strategy_name]
            if not signal_frame.empty and "strategy_name" in signal_frame.columns
            else pd.DataFrame()
        )
        rows = [row for row in trade_details if str(row.get("strategy_name") or "unknown") == strategy_name]
        exec_counts = execution.get(strategy_name, {})
        metrics[strategy_name] = _group_performance_metrics(
            rows,
            number_of_signals=int(len(signal_rows)),
            number_of_allowed_signals=_allowed_signal_count(signal_rows),
            execution=exec_counts,
        )
    return metrics


def _regime_level_metrics(
    trade_details: list[dict[str, Any]],
    signal_frame: pd.DataFrame,
    execution: dict[str, dict[str, int]],
) -> dict[str, dict[str, Any]]:
    names = set(execution)
    if not signal_frame.empty and "regime" in signal_frame.columns:
        names.update(str(value or "unknown") for value in signal_frame["regime"].dropna().unique())
    names.update(str(trade.get("regime") or "unknown") for trade in trade_details)
    metrics: dict[str, dict[str, Any]] = {}
    for regime_name in sorted(names):
        signal_rows = (
            signal_frame.loc[signal_frame["regime"].astype(str) == regime_name]
            if not signal_frame.empty and "regime" in signal_frame.columns
            else pd.DataFrame()
        )
        rows = [row for row in trade_details if str(row.get("regime") or "unknown") == regime_name]
        allowed_signals = _allowed_signal_count(signal_rows)
        number_of_signals = int(len(signal_rows))
        base = _group_performance_metrics(
            rows,
            number_of_signals=number_of_signals,
            number_of_allowed_signals=allowed_signals,
            execution=execution.get(regime_name, {}),
        )
        metrics[regime_name] = {
            "number_of_signals": base["number_of_signals"],
            "number_of_allowed_signals": allowed_signals,
            "number_of_blocked_signals": max(0, number_of_signals - allowed_signals),
            "number_of_trades": base["number_of_trades"],
            "net_return_pct": base["net_return_pct"],
            "profit_factor_net": base["profit_factor_net"],
            "max_drawdown_pct": base["max_drawdown_pct"],
            "win_rate_net": base["win_rate_net"],
        }
    return metrics


def _group_performance_metrics(
    rows: list[dict[str, Any]],
    *,
    number_of_signals: int,
    number_of_allowed_signals: int,
    execution: dict[str, int],
) -> dict[str, Any]:
    gross = np.asarray([float(row.get("gross_return_pct", row.get("gross_return", 0.0))) for row in rows], dtype=float)
    net = np.asarray([float(row.get("net_return_pct", row.get("net_return", 0.0))) for row in rows], dtype=float)
    hold_values = np.asarray([float(row.get("hold_bars", 0.0)) for row in rows], dtype=float)
    entry_attempts = int(execution.get("entry_attempts", number_of_allowed_signals))
    ambiguous = int(execution.get("ambiguous_candles", sum(1 for row in rows if row.get("ambiguous_candle"))))
    profit_factor_net = _profit_factor(net)
    max_drawdown_pct = _max_drawdown(net)
    return {
        "number_of_signals": int(number_of_signals),
        "number_of_entries": int(execution.get("filled_entries", len(rows))),
        "number_of_exits": int(execution.get("filled_exits", len(rows))),
        "number_of_trades": int(len(rows)),
        "win_rate_net": float((net > 0).mean()) if len(net) else 0.0,
        "average_net_return_pct": _mean_or_zero(net),
        "expectancy": _mean_or_zero(net),
        "gross_return_pct": float(gross.sum()) if len(gross) else 0.0,
        "net_return_pct": float(net.sum()) if len(net) else 0.0,
        "max_drawdown_pct": max_drawdown_pct,
        "profit_factor_net": profit_factor_net,
        "average_hold_bars": _mean_or_zero(hold_values),
        "canceled_orders": int(execution.get("canceled_orders", sum(int(row.get("canceled_orders", 0)) for row in rows))),
        "partial_fills": int(execution.get("partial_fills", sum(int(row.get("partial_fills", 0)) for row in rows))),
        "ambiguous_candle_ratio": (ambiguous / entry_attempts) if entry_attempts else 0.0,
        "number_of_allowed_signals": int(number_of_allowed_signals),
        "number_of_blocked_signals": max(0, int(number_of_signals) - int(number_of_allowed_signals)),
        "gross_return": float(gross.sum()) if len(gross) else 0.0,
        "net_return": float(net.sum()) if len(net) else 0.0,
        "win_rate": float((net > 0).mean()) if len(net) else 0.0,
        "profit_factor": profit_factor_net,
        "max_drawdown": max_drawdown_pct,
        "fees": float(sum(float(row.get("fee_amount", row.get("fees", 0.0))) for row in rows)),
        "slippage": float(sum(float(row.get("slippage_amount", row.get("slippage", 0.0))) for row in rows)),
        "spread_cost": float(sum(float(row.get("spread_cost", 0.0)) for row in rows)),
    }


def _allowed_signal_count(signal_rows: pd.DataFrame) -> int:
    if signal_rows.empty:
        return 0
    if "entry_allowed" in signal_rows.columns:
        return int(signal_rows["entry_allowed"].fillna(False).astype(bool).sum())
    if "blocked_by" in signal_rows.columns:
        return int(signal_rows["blocked_by"].isna().sum())
    return int(len(signal_rows))


def _blocked_signal_metrics(signal_frame: pd.DataFrame) -> dict[str, int]:
    counts = {name: 0 for name in BLOCKED_SIGNAL_BUCKETS}
    if signal_frame.empty or "blocked_by" not in signal_frame.columns:
        return counts
    for raw_value in signal_frame["blocked_by"].dropna():
        bucket = _blocked_signal_bucket(str(raw_value))
        counts[bucket] = counts.get(bucket, 0) + 1
    return counts


def _blocked_signal_bucket(value: str) -> str:
    if value in BLOCKED_SIGNAL_BUCKETS:
        return value
    if value in {"model_unavailable", "fallback"}:
        return "fallback_prediction_not_allowed"
    return "risk_manager"


def _clean_optional(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    parsed = str(value)
    return parsed if parsed and parsed.lower() != "nan" else None


def _max_drawdown(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0
    equity = np.concatenate(([0.0], np.cumsum(returns)))
    peak = np.maximum.accumulate(equity)
    drawdown = equity - peak
    value = abs(float(drawdown.min()))
    return value if math.isfinite(value) else 0.0


def _take_profit_pct(settings: Settings) -> float:
    return settings.scalping_label_take_profit_pct if settings.scalping_mode_enabled else settings.take_profit_pct


def _stop_loss_pct(settings: Settings) -> float:
    return settings.scalping_label_stop_loss_pct if settings.scalping_mode_enabled else settings.stop_loss_pct
