import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import get_settings


SIGNAL_LOG_FIELDS = (
    "symbol",
    "action",
    "reason",
    "timestamp",
    "latest_bar_timestamp",
    "bar_age_seconds",
    "quote_age_seconds",
    "buy_probability",
    "sell_probability",
    "ml_buy_probability",
    "ml_sell_probability",
    "prediction_source",
    "model_version",
    "strategy_name",
    "quant_score",
    "quant_confidence",
    "regime",
    "blocked_by",
    "block_reason",
    "candidate_strategy_count",
    "strategy_candidates",
    "selected_strategy_signal",
    "selected_strategy_reason",
    "ml_confirmation_result",
    "final_decision",
    "spread_bps",
    "quote_imbalance",
    "momentum",
    "volatility",
    "position_qty",
    "avg_entry_price",
    "unrealized_pnl_pct",
    "risk_block_reason",
    "api_budget_status",
)

ORDER_LOG_FIELDS = (
    "order_id",
    "local_order_id",
    "side",
    "order_type",
    "time_in_force",
    "requested_notional",
    "requested_qty",
    "limit_price",
    "filled_qty",
    "filled_avg_price",
    "fee_amount",
    "slippage_amount",
    "status",
    "cancel_reason",
)

RISK_BLOCK_LOG_FIELDS = (
    "reason",
    "relevant_limit",
    "current_value",
    "reset_time",
)


def _json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


class JsonlLogger:
    def __init__(self, name: str = "btc_ml_paper_trader") -> None:
        self.name = name
        self.settings = get_settings()
        self.file_path: Path | None = Path(self.settings.log_dir) / "events.jsonl"
        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            self.file_path = None
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(handler)

    def event(self, event_type: str, **payload: Any) -> None:
        timestamp = datetime.now(UTC).isoformat()
        record = {
            "ts": timestamp,
            "event_type": event_type,
            **normalize_event_payload(event_type, payload, timestamp=timestamp),
        }
        line = json.dumps(record, default=_json_default, separators=(",", ":"))
        if self.file_path is not None:
            try:
                with self.file_path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except OSError:
                pass
        self.logger.info(line)


def get_logger() -> JsonlLogger:
    return JsonlLogger()


def normalize_event_payload(event_type: str, payload: dict[str, Any], *, timestamp: str | None = None) -> dict[str, Any]:
    normalized = dict(payload)
    if event_type == "signal":
        normalized.setdefault("timestamp", timestamp or datetime.now(UTC).isoformat())
        _setdefault_fields(normalized, SIGNAL_LOG_FIELDS)
    elif event_type == "order":
        _setdefault_fields(normalized, ORDER_LOG_FIELDS)
    elif event_type == "risk_block":
        _setdefault_fields(normalized, RISK_BLOCK_LOG_FIELDS)
    return normalized


def _setdefault_fields(payload: dict[str, Any], fields: tuple[str, ...]) -> None:
    for name in fields:
        payload.setdefault(name, None)
