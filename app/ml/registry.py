import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings


ACCEPTED_PROMOTION_REASONS = {"accepted", "approved"}


@dataclass(frozen=True)
class ActiveModelStatus:
    active_model_path: str | None
    status: str
    valid: bool
    invalid_reason: str | None = None
    promotion_reason: str | None = None
    net_return_pct: float | None = None
    profit_factor_net: float | None = None
    number_of_trades: int | None = None
    model_version: str | None = None
    registry_metadata_matches_joblib: bool = False

    @property
    def registry_mismatched(self) -> bool:
        return self.status == "registry-mismatched"

    @property
    def reason(self) -> str | None:
        return self.invalid_reason

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_model_path": self.active_model_path,
            "active_model_version": self.model_version,
            "active_model_status": self.status,
            "active_model_valid": self.valid,
            "active_model_invalid_reason": self.invalid_reason,
            "active_model_reason": self.invalid_reason,
            "active_model_promotion_reason": self.promotion_reason,
            "active_model_net_return_pct": self.net_return_pct,
            "active_model_profit_factor_net": self.profit_factor_net,
            "active_model_number_of_trades": self.number_of_trades,
            "registry_metadata_matches_joblib": self.registry_metadata_matches_joblib,
            "active_model_registry_mismatched": self.registry_mismatched,
        }


class ModelRegistry:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.path = Path(self.settings.model_dir) / "registry.json"

    def read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            parsed = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def promote(
        self,
        *,
        model_path: str,
        feature_columns: list[str],
        metrics: dict,
        thresholds: dict,
        training_start: str,
        training_end: str,
        supports_independent_sell_probability: bool = False,
    ) -> dict[str, Any]:
        registry = {
            "active_model_path": model_path,
            "created_at": datetime.now(UTC).isoformat(),
            "training_start": training_start,
            "training_end": training_end,
            "feature_columns": feature_columns,
            "metrics": metrics,
            "thresholds": thresholds,
            "model_version": Path(model_path).stem,
            "supports_independent_sell_probability": supports_independent_sell_probability,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
        return registry

    def active_model_path(self) -> Path | None:
        active = self.read().get("active_model_path")
        return Path(active) if active else None

    def validate_active_model(self) -> ActiveModelStatus:
        status, _ = self._load_active_model()
        return status

    def load_valid_active_model(self) -> tuple[Any | None, ActiveModelStatus]:
        status, model = self._load_active_model()
        return (model if status.valid else None), status

    def _load_active_model(self) -> tuple[ActiveModelStatus, Any | None]:
        registry = self.read()
        active = registry.get("active_model_path")
        if not active:
            return ActiveModelStatus(None, "stale", False, "no_active_model"), None

        active_path = Path(active)
        active_path_str = str(active_path)
        path_version = active_path.stem
        if not active_path.exists():
            return ActiveModelStatus(active_path_str, "stale", False, "active_model_missing", model_version=path_version), None

        try:
            from app.ml.model import MLSignalModel

            model = MLSignalModel.load(active_path)
        except Exception:
            return ActiveModelStatus(active_path_str, "stale", False, "active_model_load_failed", model_version=path_version), None

        registry_metrics = registry.get("metrics")
        metadata = getattr(model, "metadata", None) or {}
        metadata_metrics = metadata.get("validation_metrics")
        registry_version = registry.get("model_version")
        metadata_version = metadata.get("model_version") or metadata.get("version")
        model_version = str(metadata_version or registry_version or path_version)
        snapshot_metrics = metadata_metrics if isinstance(metadata_metrics, dict) else registry_metrics
        net_return, profit_factor_net, number_of_trades = _metric_snapshot(snapshot_metrics)
        ambiguous_candle_ratio = _ambiguous_candle_ratio(snapshot_metrics)

        if registry_version is not None and str(registry_version) != path_version:
            return ActiveModelStatus(
                active_path_str,
                "registry-mismatched",
                False,
                "registry_model_version_mismatch",
                net_return_pct=net_return,
                profit_factor_net=profit_factor_net,
                number_of_trades=number_of_trades,
                model_version=model_version,
            ), model
        if metadata_version is not None and str(metadata_version) != path_version:
            return ActiveModelStatus(
                active_path_str,
                "registry-mismatched",
                False,
                "model_metadata_version_mismatch",
                net_return_pct=net_return,
                profit_factor_net=profit_factor_net,
                number_of_trades=number_of_trades,
                model_version=model_version,
            ), model
        if not isinstance(registry_metrics, dict):
            return ActiveModelStatus(
                active_path_str,
                "registry-mismatched",
                False,
                "registry_metrics_missing",
                net_return_pct=net_return,
                profit_factor_net=profit_factor_net,
                number_of_trades=number_of_trades,
                model_version=model_version,
            ), model
        if not isinstance(metadata_metrics, dict):
            return ActiveModelStatus(
                active_path_str,
                "registry-mismatched",
                False,
                "metadata_validation_metrics_missing",
                net_return_pct=net_return,
                profit_factor_net=profit_factor_net,
                number_of_trades=number_of_trades,
                model_version=model_version,
            ), model
        if not _metrics_equal(registry_metrics, metadata_metrics):
            return ActiveModelStatus(
                active_path_str,
                "registry-mismatched",
                False,
                "validation_metrics_mismatch",
                net_return_pct=net_return,
                profit_factor_net=profit_factor_net,
                number_of_trades=number_of_trades,
                model_version=model_version,
            ), model

        registry_features = registry.get("feature_columns")
        model_features = getattr(model, "feature_columns", None)
        if registry_features is not None and list(registry_features) != list(model_features or []):
            return ActiveModelStatus(
                active_path_str,
                "registry-mismatched",
                False,
                "feature_columns_mismatch",
                net_return_pct=net_return,
                profit_factor_net=profit_factor_net,
                number_of_trades=number_of_trades,
                model_version=model_version,
            ), model

        promotion_reason = _promotion_reason(metadata, metadata_metrics, registry_metrics)
        if promotion_reason not in ACCEPTED_PROMOTION_REASONS:
            return ActiveModelStatus(
                active_path_str,
                "rejected",
                False,
                "promotion_reason_not_accepted",
                promotion_reason,
                net_return,
                profit_factor_net,
                number_of_trades,
                model_version,
                True,
            ), model
        if net_return is None:
            return ActiveModelStatus(
                active_path_str,
                "rejected",
                False,
                "net_return_unavailable",
                promotion_reason,
                net_return,
                profit_factor_net,
                number_of_trades,
                model_version,
                True,
            ), model
        if net_return <= 0:
            return ActiveModelStatus(
                active_path_str,
                "rejected",
                False,
                "model_not_profitable_after_costs",
                promotion_reason,
                net_return,
                profit_factor_net,
                number_of_trades,
                model_version,
                True,
            ), model
        if profit_factor_net is None:
            return ActiveModelStatus(
                active_path_str,
                "rejected",
                False,
                "profit_factor_net_unavailable",
                promotion_reason,
                net_return,
                profit_factor_net,
                number_of_trades,
                model_version,
                True,
            ), model
        if profit_factor_net < self.settings.min_backtest_profit_factor:
            return ActiveModelStatus(
                active_path_str,
                "rejected",
                False,
                "profit_factor_net_too_low",
                promotion_reason,
                net_return,
                profit_factor_net,
                number_of_trades,
                model_version,
                True,
            ), model
        if number_of_trades is None:
            return ActiveModelStatus(
                active_path_str,
                "rejected",
                False,
                "number_of_trades_unavailable",
                promotion_reason,
                net_return,
                profit_factor_net,
                number_of_trades,
                model_version,
                True,
            ), model
        if number_of_trades < self.settings.min_backtest_trades:
            return ActiveModelStatus(
                active_path_str,
                "rejected",
                False,
                "not_enough_backtest_trades",
                promotion_reason,
                net_return,
                profit_factor_net,
                number_of_trades,
                model_version,
                True,
            ), model
        if ambiguous_candle_ratio is None:
            return ActiveModelStatus(
                active_path_str,
                "rejected",
                False,
                "ambiguous_candle_ratio_unavailable",
                promotion_reason,
                net_return,
                profit_factor_net,
                number_of_trades,
                model_version,
                True,
            ), model
        if ambiguous_candle_ratio > self.settings.max_backtest_ambiguous_candle_ratio:
            return ActiveModelStatus(
                active_path_str,
                "rejected",
                False,
                "ambiguous_candle_ratio_too_high",
                promotion_reason,
                net_return,
                profit_factor_net,
                number_of_trades,
                model_version,
                True,
            ), model

        return ActiveModelStatus(
            active_path_str,
            "accepted",
            True,
            None,
            promotion_reason,
            net_return,
            profit_factor_net,
            number_of_trades,
            model_version,
            True,
        ), model


def _promotion_reason(*sources: dict[str, Any]) -> str | None:
    for source in sources:
        reason = source.get("promotion_reason")
        if reason is not None:
            return str(reason).strip().lower()
    return None


def _metric_snapshot(metrics: Any) -> tuple[float | None, float | None, int | None]:
    if not isinstance(metrics, dict):
        return None, None, None
    return (
        _metric_float(metrics.get("net_return_pct")),
        _metric_float(metrics.get("profit_factor_net"), allow_infinite=True),
        _metric_int(metrics.get("number_of_trades")),
    )


def _ambiguous_candle_ratio(metrics: Any) -> float | None:
    if not isinstance(metrics, dict):
        return None
    return _metric_float(metrics.get("ambiguous_candle_ratio", 0.0))


def _metric_float(value: object, *, allow_infinite: bool = False) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) or allow_infinite else None


def _metric_int(value: object) -> int | None:
    number = _metric_float(value)
    return int(number) if number is not None else None


def _metrics_equal(left: Any, right: Any) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left.keys()) != set(right.keys()):
            return False
        return all(_metrics_equal(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return False
        return all(_metrics_equal(left_item, right_item) for left_item, right_item in zip(left, right, strict=True))
    left_number = _coerce_number(left)
    right_number = _coerce_number(right)
    if left_number is not None and right_number is not None:
        if math.isnan(left_number) and math.isnan(right_number):
            return True
        if math.isinf(left_number) or math.isinf(right_number):
            return left_number == right_number
        return math.isclose(left_number, right_number, rel_tol=1e-12, abs_tol=1e-12)
    return left == right


def _coerce_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
