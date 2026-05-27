from typing import Any

import httpx

from app.broker.execution_guard import assert_btc_only, validate_order_request
from app.config import ALLOWED_SYMBOL, Settings, get_settings
from app.monitoring.logger import get_logger


class AlpacaClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.logger = get_logger()

    @property
    def headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.settings.alpaca_api_key,
            "APCA-API-SECRET-KEY": self.settings.alpaca_secret_key,
        }

    def credentials_available(self) -> bool:
        return bool(self.settings.alpaca_api_key and self.settings.alpaca_secret_key)

    async def get_account(self) -> dict[str, Any]:
        if not self.credentials_available():
            return {"buying_power": "0", "paper": True, "credentials": "missing"}
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(f"{self.settings.alpaca_paper_base_url}/v2/account", headers=self.headers)
            response.raise_for_status()
            return response.json()

    async def get_position(self, symbol: str = ALLOWED_SYMBOL) -> dict[str, Any] | None:
        assert_btc_only(symbol, context="position_symbol_validation")
        if not self.credentials_available():
            return None
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(
                f"{self.settings.alpaca_paper_base_url}/v2/positions/{symbol}",
                headers=self.headers,
            )
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.json()

    async def submit_market_order(
        self,
        *,
        symbol: str,
        side: str,
        notional: float | None = None,
        qty: float | None = None,
        current_position_qty: float = 0.0,
    ) -> dict[str, Any]:
        order = validate_order_request(
            symbol,
            side,
            notional=notional,
            qty=qty,
            current_position_qty=current_position_qty,
            context="alpaca_submit_market_order",
        )
        if not self.settings.trading_enabled:
            self.logger.event(
                "submitted_order",
                symbol=order.symbol,
                side=order.side,
                notional=order.notional,
                qty=order.qty,
                status="dry_run_trading_disabled",
            )
            return {"id": None, "status": "dry_run_trading_disabled", "symbol": order.symbol, "side": order.side}
        if not self.credentials_available():
            raise RuntimeError("Alpaca credentials are required when TRADING_ENABLED=true.")

        payload: dict[str, Any] = {
            "symbol": order.symbol,
            "side": order.side,
            "type": "market",
            "time_in_force": "gtc",
        }
        if order.side == "buy":
            payload["notional"] = str(round(order.notional or 0, 2))
        else:
            payload["qty"] = str(order.qty or current_position_qty)

        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(
                f"{self.settings.alpaca_paper_base_url}/v2/orders",
                headers=self.headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            self.logger.event("submitted_order", symbol=order.symbol, side=order.side, broker_order_id=data.get("id"))
            return data
