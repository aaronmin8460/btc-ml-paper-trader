from typing import Any

import httpx

from app.config import Settings, get_settings
from app.monitoring.logger import get_logger


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
            "username": "BTC Paper Trader",
            "content": content[:2000],
            "allowed_mentions": {"parse": []},
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(self.settings.discord_webhook_url.strip(), json=payload)
                response.raise_for_status()
        except Exception as exc:
            self._log_failure(exc)

    async def signal_alert(
        self,
        symbol: str,
        action: str,
        reason: str,
        buy_probability: float,
        sell_probability: float,
    ) -> None:
        if not self.settings.discord_alert_on_signal:
            return
        if action.lower() == "hold" and not self.settings.discord_alert_on_hold:
            return

        await self.send(
            "\n".join(
                [
                    f"{symbol} signal: {action.upper()}",
                    f"Reason: {reason}",
                    f"Buy probability: {buy_probability:.4f}",
                    f"Sell probability: {sell_probability:.4f}",
                ]
            )
        )

    async def order_alert(
        self,
        side: str,
        status: str,
        notional: float | None,
        qty: float | None,
        broker_order_id: str | None = None,
    ) -> None:
        if not self.settings.discord_alert_on_order:
            return

        lines = [
            f"BTC/USD paper order: {side.upper()} {status}",
            f"Notional: {notional if notional is not None else 'n/a'}",
            f"Qty: {qty if qty is not None else 'n/a'}",
        ]
        if broker_order_id:
            lines.append(f"Broker order ID: {broker_order_id}")

        await self.send("\n".join(lines))

    async def error_alert(self, where: str, error: Exception | str) -> None:
        if not self.settings.discord_alert_on_error:
            return

        error_type = type(error).__name__ if isinstance(error, Exception) else "Error"
        await self.send(f"Error in {where}: {error_type}: {error}")

    async def risk_alert(self, reason: str) -> None:
        await self.send(f"Risk block: {reason}")

    async def model_alert(
        self,
        model_path: str,
        accepted: bool,
        reason: str,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        if not self.settings.discord_alert_on_model:
            return

        lines = [
            f"Model {'accepted' if accepted else 'rejected'}: {model_path}",
            f"Reason: {reason}",
        ]
        if metrics:
            metric_text = ", ".join(f"{key}={value}" for key, value in sorted(metrics.items()))
            lines.append(f"Metrics: {metric_text}")

        await self.send("\n".join(lines))

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
