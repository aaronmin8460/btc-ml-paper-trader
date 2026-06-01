import json
from datetime import UTC, datetime
from typing import Any

import httpx

from app.config import ALLOWED_SYMBOL, Settings, get_settings
from app.monitoring.logger import get_logger


USERNAME = "BTC Paper Trader"
CONTENT_LIMIT = 2000
EMBED_TITLE_LIMIT = 256
EMBED_DESCRIPTION_LIMIT = 4096
EMBED_FIELD_NAME_LIMIT = 256
EMBED_FIELD_VALUE_LIMIT = 1024
EMBED_FOOTER_LIMIT = 2048
MAX_EMBED_FIELDS = 25

DISCORD_GREEN = 0x2ECC71
DISCORD_RED = 0xE74C3C
DISCORD_BLUE_GRAY = 0x5865F2
DISCORD_ORANGE = 0xF59E0B
DISCORD_GRAY = 0x95A5A6


class DiscordNotifier:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.logger = get_logger()
        self._last_alert_sent_at: dict[str, datetime] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.settings.discord_alerts_enabled and self.settings.discord_webhook_url.strip())

    async def send(self, content: str) -> None:
        if not self.enabled:
            return

        payload = {
            "username": USERNAME,
            "content": str(content)[:CONTENT_LIMIT],
            "allowed_mentions": {"parse": []},
        }

        await self._post_payload(payload)

    async def send_embed(
        self,
        title: str,
        description: str | None = None,
        fields: list[dict[str, Any]] | None = None,
        color: int | None = None,
        footer: str | None = None,
        *,
        alert_type: str | None = None,
        cooldown_key: str | None = None,
    ) -> bool:
        if not self.enabled:
            return False
        throttle_key = _alert_throttle_key(alert_type, cooldown_key)
        if throttle_key and self._alert_is_throttled(throttle_key):
            try:
                self.logger.event("discord_alert_throttled", alert_type=alert_type)
            except Exception:
                pass
            return False

        embed: dict[str, Any] = {
            "title": truncate(title, EMBED_TITLE_LIMIT),
            "timestamp": utc_timestamp(),
        }
        if description:
            embed["description"] = truncate(description, EMBED_DESCRIPTION_LIMIT)
        if color is not None:
            embed["color"] = int(color)
        prepared_fields = prepare_fields(fields or [])
        if prepared_fields:
            embed["fields"] = prepared_fields
        if footer:
            embed["footer"] = {"text": truncate(footer, EMBED_FOOTER_LIMIT)}

        payload = {
            "username": USERNAME,
            "embeds": [embed],
            "allowed_mentions": {"parse": []},
        }

        sent = await self._post_payload(payload)
        if sent and throttle_key:
            self._last_alert_sent_at[throttle_key] = datetime.now(UTC)
        return sent

    async def signal_alert(
        self,
        symbol: str,
        action: str,
        reason: str,
        buy_probability: float,
        sell_probability: float,
        *,
        spread_bps: float | None = None,
        quote_imbalance: float | None = None,
        latest_price: float | None = None,
        mid_price: float | None = None,
        bar_age_seconds: float | None = None,
        quote_age_seconds: float | None = None,
        prediction_source: str | None = None,
        model_version: str | None = None,
    ) -> None:
        if not self.settings.discord_alert_on_signal:
            return
        if action.lower() == "hold" and not self.settings.discord_alert_on_hold:
            return

        action_text = action.upper()
        await self.send_embed(
            title=f"{symbol} Signal: {action_text}",
            fields=[
                field("Alert Type", "signal"),
                field("Action", action_text),
                field("Reason", reason),
                field("Buy Probability", format_probability(buy_probability)),
                field("Sell Probability", format_probability(sell_probability)),
                field("Prediction Source", prediction_source),
                field("Model Version", model_version),
                field("Latest Price", format_usd(latest_price)),
                field("Mid Price", format_usd(mid_price)),
                field("Spread bps", format_number(spread_bps, digits=2)),
                field("Quote Imbalance", format_number(quote_imbalance, digits=4)),
                field("Bar Age Seconds", format_number(bar_age_seconds, digits=2)),
                field("Quote Age Seconds", format_number(quote_age_seconds, digits=2)),
                field("Timestamp", utc_timestamp()),
            ],
            color=action_color(action),
            footer="BTC/USD paper trading only",
            alert_type="signal",
            cooldown_key=f"{symbol}:{action.lower()}:{reason}",
        )

    async def order_alert(
        self,
        side: str,
        status: str,
        notional: float | None,
        qty: float | None,
        broker_order_id: str | None = None,
        *,
        order_type: str | None = None,
        time_in_force: str | None = None,
        local_order_id: str | int | None = None,
        limit_price: float | None = None,
        filled_qty: float | None = None,
        filled_avg_price: float | None = None,
        fee_amount: float | None = None,
        slippage_amount: float | None = None,
        cancel_reason: str | None = None,
    ) -> None:
        if not self.settings.discord_alert_on_order:
            return

        alert_type = _order_alert_type(side=side, status=status)
        await self.send_embed(
            title=f"{ALLOWED_SYMBOL} Paper Order",
            fields=[
                field("Alert Type", alert_type),
                field("Side", side.upper()),
                field("Status", status),
                field("Requested Notional", format_usd(notional)),
                field("Requested Quantity", format_number(qty, digits=8)),
                field("Quantity", format_number(qty, digits=8)),
                field("Order Type", order_type or "n/a"),
                field("Time in Force", time_in_force or "n/a"),
                field("Broker Order ID", broker_order_id or "n/a"),
                field("Local Order ID", local_order_id),
                field("Limit Price", format_usd(limit_price)),
                field("Filled Quantity", format_number(filled_qty, digits=8)),
                field("Filled Average Price", format_usd(filled_avg_price)),
                field("Fee Amount", format_usd(fee_amount)),
                field("Slippage Amount", format_usd(slippage_amount)),
                field("Cancel Reason", cancel_reason),
                field("Timestamp", utc_timestamp()),
            ],
            color=DISCORD_ORANGE if alert_type == "order_canceled" else action_color(side),
            footer="BTC/USD paper order",
            alert_type=alert_type,
            cooldown_key=str(broker_order_id or local_order_id or f"{side}:{status}"),
        )

    async def error_alert(self, where: str, error: Exception | str, *, force: bool = False) -> None:
        if not force and not self.settings.discord_alert_on_error:
            return

        error_type = type(error).__name__ if isinstance(error, Exception) else "Error"
        await self.send_embed(
            title="Trading Bot Error",
            fields=[
                field("Alert Type", "runtime_error"),
                field("Where", where),
                field("Error Type", error_type),
                field("Error Message", str(error)),
                field("Timestamp", utc_timestamp()),
            ],
            color=DISCORD_RED,
            footer="Notifier failure cannot stop trading flow",
            alert_type="runtime_error",
            cooldown_key=f"{where}:{error_type}:{error}",
        )

    async def risk_alert(
        self,
        reason: str,
        *,
        relevant_limit: Any = None,
        current_value: Any = None,
        reset_time: Any = None,
    ) -> None:
        alert_type = "stale_data_block" if reason == "stale_market_data" else "risk_block"
        await self.send_embed(
            title="Risk Guard Triggered",
            fields=[
                field("Alert Type", alert_type),
                field("Symbol", ALLOWED_SYMBOL),
                field("Reason", reason),
                field("Relevant Limit", relevant_limit),
                field("Current Value", current_value),
                field("Reset Time", reset_time),
                field("Timestamp", utc_timestamp()),
            ],
            color=DISCORD_ORANGE,
            footer="Paper trading risk guard",
            alert_type=alert_type,
            cooldown_key=reason,
        )

    async def auto_trading_paused_alert(self, reason: str) -> None:
        await self.send_embed(
            title="Auto trading paused",
            fields=[
                field("Alert Type", "kill_switch_pause"),
                field("Symbol", ALLOWED_SYMBOL),
                field("Reason", reason),
                field("Timestamp", utc_timestamp()),
            ],
            color=DISCORD_RED,
            footer="Runtime circuit breaker",
            alert_type="kill_switch_pause",
            cooldown_key=reason,
        )

    async def kill_switch_pause_alert(self, reason: str) -> None:
        await self.auto_trading_paused_alert(reason)

    async def model_alert(
        self,
        model_path: str,
        accepted: bool,
        reason: str,
        metrics: dict[str, Any] | None = None,
        *,
        force: bool = False,
    ) -> None:
        if not force and not self.settings.discord_alert_on_model:
            return

        await self.send_embed(
            title="Model Accepted" if accepted else "Model Rejected",
            fields=[
                field("Model Path", model_path),
                field("Reason", reason),
                field("Metrics", format_metrics(metrics)),
                field("Timestamp", utc_timestamp()),
            ],
            color=DISCORD_GREEN if accepted else DISCORD_RED,
            footer="BTC/USD paper model validation",
            alert_type="model_result",
            cooldown_key=f"{model_path}:{accepted}:{reason}",
        )

    async def _post_payload(self, payload: dict[str, Any]) -> bool:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(self.settings.discord_webhook_url.strip(), json=payload)
                response.raise_for_status()
            return True
        except Exception as exc:
            self._log_failure(exc)
            return False

    def _alert_is_throttled(self, throttle_key: str) -> bool:
        sent_at = self._last_alert_sent_at.get(throttle_key)
        if sent_at is None:
            return False
        elapsed = (datetime.now(UTC) - sent_at).total_seconds()
        return elapsed < self.settings.discord_alert_cooldown_seconds

    def _log_failure(self, exc: Exception) -> None:
        payload: dict[str, Any] = {"error_type": type(exc).__name__}
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
        if status_code is not None:
            payload["status_code"] = status_code

        try:
            self.logger.event("discord_alert_failed", **payload)
        except Exception:
            pass


def truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 1)] + "…"


def field(name: str, value: Any, *, inline: bool = True) -> dict[str, Any]:
    return {
        "name": truncate(str(name), EMBED_FIELD_NAME_LIMIT),
        "value": truncate(str(value if value not in {None, ""} else "n/a"), EMBED_FIELD_VALUE_LIMIT),
        "inline": inline,
    }


def prepare_fields(fields: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared = []
    for item in fields[:MAX_EMBED_FIELDS]:
        prepared.append(
            {
                "name": truncate(str(item.get("name", "Field")), EMBED_FIELD_NAME_LIMIT),
                "value": truncate(str(item.get("value", "n/a")), EMBED_FIELD_VALUE_LIMIT),
                "inline": bool(item.get("inline", True)),
            }
        )
    return prepared


def utc_timestamp() -> str:
    return datetime.now(UTC).isoformat()


def format_usd(value: float | int | None) -> str:
    parsed = safe_float(value)
    if parsed is None:
        return "n/a"
    return f"${parsed:,.2f}"


def format_probability(value: float | int | None) -> str:
    parsed = safe_float(value)
    if parsed is None:
        return "n/a"
    return f"{parsed:.4f} ({parsed * 100:.2f}%)"


def format_number(value: float | int | None, *, digits: int = 4) -> str:
    parsed = safe_float(value)
    if parsed is None:
        return "n/a"
    return f"{parsed:,.{digits}f}"


def format_metrics(metrics: dict[str, Any] | None) -> str:
    if not metrics:
        return "n/a"
    safe_metrics = {str(key): value for key, value in sorted(metrics.items())}
    return truncate(json.dumps(safe_metrics, sort_keys=True, default=str), EMBED_FIELD_VALUE_LIMIT)


def action_color(action: str) -> int:
    normalized = action.lower()
    if normalized == "buy":
        return DISCORD_GREEN
    if normalized == "sell":
        return DISCORD_RED
    if normalized == "hold":
        return DISCORD_BLUE_GRAY
    return DISCORD_GRAY


def safe_float(value: float | int | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _order_alert_type(*, side: str, status: str) -> str:
    normalized_status = str(status).lower()
    if normalized_status in {"canceled", "cancelled"}:
        return "order_canceled"
    if normalized_status in {"filled", "partially_filled"}:
        return "buy_order_filled" if str(side).lower() == "buy" else "sell_order_filled"
    return "order_submitted"


def _alert_throttle_key(alert_type: str | None, cooldown_key: str | None) -> str | None:
    if not alert_type:
        return None
    return f"{alert_type}:{cooldown_key or ''}"
