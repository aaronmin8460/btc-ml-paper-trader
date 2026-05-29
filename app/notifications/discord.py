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
    ) -> None:
        if not self.enabled:
            return

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

        await self._post_payload(payload)

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
    ) -> None:
        if not self.settings.discord_alert_on_signal:
            return
        if action.lower() == "hold" and not self.settings.discord_alert_on_hold:
            return

        action_text = action.upper()
        await self.send_embed(
            title=f"{symbol} Signal: {action_text}",
            fields=[
                field("Action", action_text),
                field("Reason", reason),
                field("Buy Probability", format_probability(buy_probability)),
                field("Sell Probability", format_probability(sell_probability)),
                field("Latest Price", format_usd(latest_price)),
                field("Mid Price", format_usd(mid_price)),
                field("Spread bps", format_number(spread_bps, digits=2)),
                field("Quote Imbalance", format_number(quote_imbalance, digits=4)),
                field("Timestamp", utc_timestamp()),
            ],
            color=action_color(action),
            footer="BTC/USD paper trading only",
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
    ) -> None:
        if not self.settings.discord_alert_on_order:
            return

        await self.send_embed(
            title=f"{ALLOWED_SYMBOL} Paper Order",
            fields=[
                field("Side", side.upper()),
                field("Status", status),
                field("Notional", format_usd(notional)),
                field("Quantity", format_number(qty, digits=8)),
                field("Order Type", order_type or "n/a"),
                field("Time in Force", time_in_force or "n/a"),
                field("Broker Order ID", broker_order_id or "n/a"),
                field("Timestamp", utc_timestamp()),
            ],
            color=action_color(side),
            footer="Alpaca paper order",
        )

    async def error_alert(self, where: str, error: Exception | str) -> None:
        if not self.settings.discord_alert_on_error:
            return

        error_type = type(error).__name__ if isinstance(error, Exception) else "Error"
        await self.send_embed(
            title="Trading Bot Error",
            fields=[
                field("Where", where),
                field("Error Type", error_type),
                field("Error Message", str(error)),
                field("Timestamp", utc_timestamp()),
            ],
            color=DISCORD_RED,
            footer="Notifier failure cannot stop trading flow",
        )

    async def risk_alert(self, reason: str) -> None:
        await self.send_embed(
            title="Risk Guard Triggered",
            fields=[
                field("Symbol", ALLOWED_SYMBOL),
                field("Reason", reason),
                field("Timestamp", utc_timestamp()),
            ],
            color=DISCORD_ORANGE,
            footer="Paper trading risk guard",
        )

    async def auto_trading_paused_alert(self, reason: str) -> None:
        await self.send_embed(
            title="Auto trading paused",
            fields=[
                field("Symbol", ALLOWED_SYMBOL),
                field("Reason", reason),
                field("Timestamp", utc_timestamp()),
            ],
            color=DISCORD_RED,
            footer="Runtime circuit breaker",
        )

    async def model_alert(
        self,
        model_path: str,
        accepted: bool,
        reason: str,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        if not self.settings.discord_alert_on_model:
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
        )

    async def _post_payload(self, payload: dict[str, Any]) -> None:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(self.settings.discord_webhook_url.strip(), json=payload)
                response.raise_for_status()
        except Exception as exc:
            self._log_failure(exc)

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
