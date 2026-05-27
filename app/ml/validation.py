import math

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from app.data.feature_engineering import FEATURE_COLUMNS
from app.ml.model import MLSignalModel


def trading_metrics(df: pd.DataFrame, probabilities: np.ndarray, threshold: float = 0.58) -> dict:
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
    returns = np.where(trades["buy_quality_label"].to_numpy() == 1, 0.03, -0.015)
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
) -> dict:
    if len(df) < min_train_rows:
        return {"valid": False, "reason": "not_enough_rows", "rows": len(df)}
    fold_size = max(50, (len(df) - min_train_rows) // max(1, folds))
    all_y: list[int] = []
    all_p: list[float] = []
    validation_frames: list[pd.DataFrame] = []
    for fold in range(folds):
        train_end = min_train_rows + fold * fold_size
        valid_end = min(len(df), train_end + fold_size)
        if valid_end <= train_end + 10:
            continue
        train = df.iloc[:train_end]
        valid = df.iloc[train_end:valid_end]
        model = MLSignalModel(feature_columns=FEATURE_COLUMNS).train(train)
        probs = model.predict_proba(valid)
        all_y.extend(valid["buy_quality_label"].astype(int).tolist())
        all_p.extend(probs.tolist())
        frame = valid.copy()
        frame["_probability"] = probs
        validation_frames.append(frame)
    if not all_y:
        return {"valid": False, "reason": "no_validation_folds", "rows": len(df)}
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
    }
    trade_df = pd.concat(validation_frames, ignore_index=True)
    metrics.update(trading_metrics(trade_df, trade_df["_probability"].to_numpy(), threshold=threshold))
    return metrics


def promotion_decision(metrics: dict, *, min_rows: int, min_precision: float, max_drawdown: float, max_trade_fraction: float) -> tuple[bool, str]:
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
    if trades / rows > max_trade_fraction:
        return False, "too_many_trades"
    return True, "accepted"
