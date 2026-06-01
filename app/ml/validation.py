import math

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from app.data.feature_engineering import BAR_FEATURE_COLUMNS
from app.ml.model import MLSignalModel


def trading_metrics(
    df: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float = 0.58,
    *,
    take_profit_pct: float = 0.03,
    stop_loss_pct: float = 0.015,
) -> dict:
    trades = df.loc[probabilities >= threshold].copy()
    if trades.empty:
        return {
            "number_of_trades": 0,
            "win_rate": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "expectancy": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
        }
    returns = _trade_returns(trades, take_profit_pct=take_profit_pct, stop_loss_pct=stop_loss_pct)
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    equity = np.cumsum(returns)
    peak = np.maximum.accumulate(equity)
    drawdown = equity - peak
    gross_win = wins.sum() if len(wins) else 0.0
    gross_loss = abs(losses.sum()) if len(losses) else 0.0
    return {
        "number_of_trades": int(len(trades)),
        "win_rate": float((returns > 0).mean()),
        "average_win": float(wins.mean()) if len(wins) else 0.0,
        "average_loss": float(losses.mean()) if len(losses) else 0.0,
        "expectancy": float(returns.mean()),
        "profit_factor": float(gross_win / gross_loss) if gross_loss else (float("inf") if gross_win else 0.0),
        "max_drawdown": float(abs(drawdown.min())) if len(drawdown) else 0.0,
    }


def walk_forward_validate(
    df: pd.DataFrame,
    *,
    min_train_rows: int,
    threshold: float = 0.58,
    folds: int = 3,
    take_profit_pct: float = 0.03,
    stop_loss_pct: float = 0.015,
    feature_columns: list[str] | None = None,
    sell_threshold: float = 0.55,
) -> dict:
    feature_columns = feature_columns or BAR_FEATURE_COLUMNS
    sell_class_balance = _normalized_class_balance(df["sell_quality_label"]) if "sell_quality_label" in df else {}
    if len(df) < min_train_rows:
        return {"valid": False, "reason": "not_enough_rows", "rows": len(df), "sell_class_balance": sell_class_balance}
    class_counts = _class_counts(df["buy_quality_label"])
    if len(class_counts) < 2:
        return {
            "valid": False,
            "reason": "target_class_diversity_too_low",
            "rows": len(df),
            "class_counts": class_counts,
            "sell_class_balance": sell_class_balance,
        }
    fold_size = max(50, (len(df) - min_train_rows) // max(1, folds))
    all_y: list[int] = []
    all_p: list[float] = []
    validation_frames: list[pd.DataFrame] = []
    all_sell_y: list[int] = []
    all_sell_p: list[float] = []
    skipped_folds = 0
    skipped_sell_folds = 0
    for fold in range(folds):
        train_end = min_train_rows + fold * fold_size
        valid_end = min(len(df), train_end + fold_size)
        if valid_end <= train_end + 10:
            continue
        train = df.iloc[:train_end]
        valid = df.iloc[train_end:valid_end]
        if len(_class_counts(train["buy_quality_label"])) < 2:
            skipped_folds += 1
            continue
        model = MLSignalModel(feature_columns=feature_columns).train(train)
        probs = model.predict_buy_proba(valid)
        all_y.extend(valid["buy_quality_label"].astype(int).tolist())
        all_p.extend(probs.tolist())
        frame = valid.copy()
        frame["_probability"] = probs
        validation_frames.append(frame)
        if model.supports_independent_sell_probability:
            sell_probs = model.predict_sell_proba(valid)
            all_sell_y.extend(valid["sell_quality_label"].astype(int).tolist())
            all_sell_p.extend(sell_probs.tolist())
        else:
            skipped_sell_folds += 1
    if not all_y:
        return {
            "valid": False,
            "reason": "no_trainable_validation_folds",
            "rows": len(df),
            "class_counts": class_counts,
            "skipped_folds": skipped_folds,
            "sell_class_balance": sell_class_balance,
            "skipped_sell_folds": skipped_sell_folds,
        }
    y = np.array(all_y)
    p = np.array(all_p)
    pred = (p >= threshold).astype(int)
    metrics = {
        "rows": int(len(df)),
        "validation_rows": int(len(y)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)) if len(set(y.tolist())) > 1 else math.nan,
        "skipped_folds": skipped_folds,
        "sell_precision": _sell_precision(all_sell_y, all_sell_p, threshold=sell_threshold),
        "sell_class_balance": sell_class_balance,
        "sell_validation_rows": len(all_sell_y),
        "skipped_sell_folds": skipped_sell_folds,
    }
    trade_df = pd.concat(validation_frames, ignore_index=True)
    metrics.update(
        trading_metrics(
            trade_df,
            trade_df["_probability"].to_numpy(),
            threshold=threshold,
            take_profit_pct=take_profit_pct,
            stop_loss_pct=stop_loss_pct,
        )
    )
    return metrics


def promotion_decision(
    metrics: dict,
    *,
    min_rows: int,
    min_precision: float,
    max_drawdown: float,
    max_trade_fraction: float,
    min_net_return_pct: float = 0.0,
    max_backtest_drawdown_pct: float | None = None,
    min_backtest_profit_factor: float = 1.2,
    min_backtest_trades: int = 30,
    max_ambiguous_candle_ratio: float = 0.10,
    require_positive_net_return: bool = False,
) -> tuple[bool, str]:
    if not metrics or metrics.get("validation_rows", 0) < min(50, min_rows):
        return False, "validation_rows_too_low"
    if metrics.get("precision", 0) < min_precision:
        return False, "precision_below_threshold"
    if metrics.get("profit_factor", 0) <= 1.05:
        return False, "profit_factor_too_low"
    if metrics.get("max_drawdown", 0) > max_drawdown:
        return False, "max_drawdown_too_high"
    trades = metrics.get("number_of_trades", 0)
    rows = metrics.get("validation_rows", 1)
    if trades == 0:
        return False, "zero_trades"
    if trades < min_backtest_trades:
        return False, "not_enough_backtest_trades"
    if trades / rows > max_trade_fraction:
        return False, "too_many_trades"
    if metrics.get("fee_aware_backtest_valid") is False:
        return False, metrics.get("fee_aware_backtest_reason") or "fee_aware_backtest_invalid"
    ambiguous_candle_ratio = _metric_float(metrics.get("ambiguous_candle_ratio", 0.0))
    if ambiguous_candle_ratio is None:
        return False, "ambiguous_candle_ratio_unavailable"
    if ambiguous_candle_ratio > max_ambiguous_candle_ratio:
        return False, "ambiguous_candle_ratio_too_high"
    net_return = _metric_float(metrics.get("net_return_pct"))
    if net_return is None:
        return False, "net_return_unavailable"
    if net_return <= 0:
        return False, "model_not_profitable_after_costs"
    if net_return < min_net_return_pct:
        return False, "net_return_below_threshold"
    backtest_drawdown = _metric_float(metrics.get("max_drawdown_pct"))
    if (
        max_backtest_drawdown_pct is not None
        and backtest_drawdown is not None
        and backtest_drawdown > max_backtest_drawdown_pct
    ):
        return False, "backtest_drawdown_too_high"
    profit_factor_net = _metric_float(metrics.get("profit_factor_net"), allow_infinite=True)
    if profit_factor_net is None:
        return False, "profit_factor_net_unavailable"
    if profit_factor_net < min_backtest_profit_factor:
        return False, "profit_factor_net_too_low"
    return True, "accepted"


def _class_counts(values: pd.Series) -> dict[int, int]:
    counts = values.astype(int).value_counts().sort_index()
    return {int(label): int(count) for label, count in counts.items()}


def _normalized_class_balance(values: pd.Series) -> dict[int, float]:
    counts = values.astype(int).value_counts(normalize=True).sort_index()
    return {int(label): float(fraction) for label, fraction in counts.items()}


def _sell_precision(labels: list[int], probabilities: list[float], *, threshold: float) -> float | None:
    if not labels:
        return None
    return float(precision_score(np.array(labels), np.array(probabilities) >= threshold, zero_division=0))


def _metric_float(value: object, *, allow_infinite: bool = False) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(parsed) or allow_infinite:
        return parsed
    return None


def _trade_returns(trades: pd.DataFrame, *, take_profit_pct: float, stop_loss_pct: float) -> np.ndarray:
    labels = trades["buy_quality_label"].to_numpy()
    fallback = np.where(labels == 1, take_profit_pct, -stop_loss_pct).astype(float)
    if "buy_exit_return_pct" not in trades.columns:
        return fallback
    returns = pd.to_numeric(trades["buy_exit_return_pct"], errors="coerce").to_numpy(dtype=float)
    return np.where(np.isfinite(returns), returns, fallback)
