import asyncio
import math
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

import httpx

from app.broker.execution_guard import assert_btc_only, validate_order_request
from app.broker.paper_execution import PaperOrderRequest, simulate_limit_ioc_order, simulate_market_order
from app.config import ALLOWED_SYMBOL, Settings, get_settings
from app.monitoring.logger import get_logger
from app.utils.rate_limiter import get_alpaca_rate_limiter


class AlpacaClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.logger = get_logger()
        self._position_cache: dict[str, tuple[datetime, dict[str, Any] | None]] = {}
        self._account_cache: tuple[datetime, dict[str, Any]] | None = None
        self._local_position_qty = 0.0
        self._local_avg_entry_price = 0.0
        self._local_current_price = 0.0

    @property
    def headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.settings.alpaca_api_key,
            "APCA-API-SECRET-KEY": self.settings.alpaca_secret_key,
        }

    def credentials_available(self) -> bool:
        return bool(self.settings.alpaca_api_key and self.settings.alpaca_secret_key)

    async def get_account(self, *, force_refresh: bool = False) -> dict[str, Any]:
        cached = self._get_account_cache(force_refresh=force_refresh)
        if cached is not None:
            return cached
        if not self.credentials_available():
            account = {"buying_power": "0", "paper": True, "credentials": "missing"}
            self._set_account_cache(account)
            return account
        async with httpx.AsyncClient(timeout=15) as client:
            await self._wait_for_alpaca(endpoint="account")
            response = await client.get(f"{self.settings.alpaca_paper_base_url}/v2/account", headers=self.headers)
            response.raise_for_status()
            account = response.json()
            self._set_account_cache(account)
            return account

    async def get_position(self, symbol: str = ALLOWED_SYMBOL, *, force_refresh: bool = False) -> dict[str, Any] | None:
        assert_btc_only(symbol, context="position_symbol_validation")
        if self.settings.paper_execution_mode == "local_simulated":
            return self._local_position_payload()
        cached = self._get_position_cache(symbol, force_refresh=force_refresh)
        if cached is not _CACHE_MISS:
            return cached
        if not self.credentials_available():
            return None
        async with httpx.AsyncClient(timeout=15) as client:
            response = await self._get_position_response(client, symbol)
            if response.status_code == 404:
                self._set_position_cache(symbol, None)
                return None
            response.raise_for_status()
            position = response.json()
            self._set_position_cache(symbol, position)
            return position

    async def _get_position_response(self, client: httpx.AsyncClient, symbol: str) -> httpx.Response:
        candidates = [quote(symbol, safe="")]
        legacy_crypto_symbol = symbol.replace("/", "")
        if legacy_crypto_symbol != symbol:
            candidates.append(legacy_crypto_symbol)

        response: httpx.Response | None = None
        for candidate in candidates:
            await self._wait_for_alpaca(endpoint="position")
            response = await client.get(
                f"{self.settings.alpaca_paper_base_url}/v2/positions/{candidate}",
                headers=self.headers,
            )
            if response.status_code != 404:
                return response
        if response is None:
            raise RuntimeError("No Alpaca position lookup candidate was attempted.")
        return response

    async def submit_market_order(
        self,
        *,
        symbol: str,
        side: str,
        notional: float | None = None,
        qty: float | None = None,
        current_position_qty: float = 0.0,
        quote: dict[str, Any] | None = None,
        latest_price: float | None = None,
    ) -> dict[str, Any]:
        return await self.submit_order(
            symbol=symbol,
            side=side,
            notional=notional,
            qty=qty,
            current_position_qty=current_position_qty,
            quote=quote,
            latest_price=latest_price,
            order_type="market",
            time_in_force="gtc",
        )

    async def submit_order(
        self,
        *,
        symbol: str,
        side: str,
        notional: float | None = None,
        qty: float | None = None,
        current_position_qty: float = 0.0,
        quote: dict[str, Any] | None = None,
        latest_price: float | None = None,
        order_type: str | None = None,
        time_in_force: str | None = None,
    ) -> dict[str, Any]:
        selected_order_type = (order_type or self.settings.order_type).strip().lower()
        selected_time_in_force = (time_in_force or self.settings.time_in_force).strip().lower()
        prevalidated_limit_price = 1.0 if selected_order_type == "limit" else None
        order = validate_order_request(
            symbol,
            side,
            notional=notional,
            qty=qty,
            order_type=selected_order_type,
            limit_price=prevalidated_limit_price,
            current_position_qty=current_position_qty,
            context="alpaca_submit_order",
        )
        if selected_time_in_force not in {"gtc", "ioc"}:
            self.logger.event(
                "rejected_order",
                symbol=symbol,
                side=side,
                reason="invalid_time_in_force",
                time_in_force=selected_time_in_force,
            )
            raise ValueError("Time in force must be gtc or ioc.")
        if selected_order_type == "limit":
            limit_price = self._limit_price_from_quote(side=side, quote=quote, latest_price=latest_price)
            order = replace(order, limit_price=limit_price)
        self.logger.event(
            "order_attempt",
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            time_in_force=selected_time_in_force,
            notional=order.notional,
            qty=order.qty,
            limit_price=order.limit_price,
        )
        if not self.settings.trading_enabled:
            self.logger.event(
                "submitted_order",
                symbol=order.symbol,
                side=order.side,
                notional=order.notional,
                qty=order.qty,
                order_type=order.order_type,
                limit_price=order.limit_price,
                status="dry_run_trading_disabled",
            )
            return {
                "id": None,
                "status": "dry_run_trading_disabled",
                "symbol": order.symbol,
                "side": order.side,
                "order_type": order.order_type,
                "time_in_force": selected_time_in_force,
                "limit_price": order.limit_price,
            }
        if self.settings.paper_execution_mode == "local_simulated":
            return self._simulate_local_order(
                order=order,
                time_in_force=selected_time_in_force,
                current_position_qty=current_position_qty,
                quote=quote,
                latest_price=latest_price,
            )
        if not self.credentials_available():
            raise RuntimeError("Alpaca credentials are required when TRADING_ENABLED=true.")

        payload: dict[str, Any] = {
            "symbol": order.symbol,
            "side": order.side,
            "type": order.order_type,
            "time_in_force": selected_time_in_force,
        }
        if order.order_type == "limit":
            payload["limit_price"] = str(order.limit_price)
        if order.side == "buy":
            payload["notional"] = str(round(order.notional or 0, 2))
        else:
            payload["qty"] = str(order.qty or current_position_qty)

        async with httpx.AsyncClient(timeout=20) as client:
            await self._wait_for_alpaca(endpoint="submit_order")
            response = await client.post(
                f"{self.settings.alpaca_paper_base_url}/v2/orders",
                headers=self.headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            data.setdefault("order_type", order.order_type)
            data.setdefault("time_in_force", selected_time_in_force)
            data.setdefault("limit_price", order.limit_price)
            self.invalidate_position_cache(order.symbol)
            data = await self._maybe_check_order_status(data)
            self.logger.event(
                "submitted_order",
                symbol=order.symbol,
                side=order.side,
                broker_order_id=data.get("id"),
                status=data.get("status"),
                order_type=order.order_type,
                time_in_force=selected_time_in_force,
            )
            return data

    def _simulate_local_order(
        self,
        *,
        order,
        time_in_force: str,
        current_position_qty: float,
        quote: dict[str, Any] | None,
        latest_price: float | None,
    ) -> dict[str, Any]:
        bid_size = self._quote_size_float(quote, "bid_size", "bs")
        ask_size = self._quote_size_float(quote, "ask_size", "as")
        qty = order.qty
        if order.side == "sell":
            qty = min(order.qty or current_position_qty, max(0.0, current_position_qty))
        request = PaperOrderRequest(
            symbol=order.symbol,
            side=order.side,
            bid_price=self._quote_float(quote, "bid_price", "bp", "bid"),
            ask_price=self._quote_float(quote, "ask_price", "ap", "ask"),
            bid_size=bid_size,
            ask_size=ask_size,
            latest_price=latest_price,
            notional=order.notional,
            qty=qty,
            fee_bps=self.settings.paper_fee_bps,
            slippage_bps=self.settings.paper_slippage_bps,
            limit_price=order.limit_price,
            order_type=order.order_type,
            time_in_force=time_in_force,
        )
        result = simulate_market_order(request) if order.order_type == "market" else simulate_limit_ioc_order(request)
        data = result.to_order_response()
        self._apply_local_fill(data)
        self.logger.event(
            "submitted_order",
            symbol=order.symbol,
            side=order.side,
            broker_order_id=data["id"],
            status=data["status"],
            order_type=order.order_type,
            time_in_force=time_in_force,
            filled_qty=data["filled_qty"],
            filled_avg_price=data["filled_avg_price"],
            fee_amount=data["fee_amount"],
            slippage_amount=data["slippage_amount"],
            execution_source=data["execution_source"],
        )
        return data

    def _apply_local_fill(self, order_response: dict[str, Any]) -> None:
        filled_qty = float(order_response.get("filled_qty") or 0)
        filled_avg_price = float(order_response.get("filled_avg_price") or 0)
        if filled_qty <= 0 or filled_avg_price <= 0:
            return
        if order_response.get("side") == "buy":
            old_notional = self._local_position_qty * self._local_avg_entry_price
            self._local_position_qty += filled_qty
            self._local_avg_entry_price = (old_notional + filled_qty * filled_avg_price) / self._local_position_qty
        else:
            self._local_position_qty = max(0.0, self._local_position_qty - filled_qty)
            if self._local_position_qty <= 1e-12:
                self._local_position_qty = 0.0
                self._local_avg_entry_price = 0.0
        self._local_current_price = filled_avg_price

    def _local_position_payload(self) -> dict[str, Any] | None:
        if self._local_position_qty <= 0:
            return None
        return {
            "symbol": ALLOWED_SYMBOL,
            "qty": str(self._local_position_qty),
            "avg_entry_price": str(self._local_avg_entry_price),
            "current_price": str(self._local_current_price or self._local_avg_entry_price),
            "market_value": str(self._local_position_qty * (self._local_current_price or self._local_avg_entry_price)),
            "execution_source": "local_simulated",
        }

    def hydrate_local_position(self, *, qty: float, avg_entry_price: float, current_price: float | None = None) -> None:
        if self.settings.paper_execution_mode != "local_simulated":
            return
        self._local_position_qty = max(0.0, float(qty))
        self._local_avg_entry_price = max(0.0, float(avg_entry_price))
        self._local_current_price = max(0.0, float(current_price or avg_entry_price))

    async def get_order(self, order_id: str) -> dict[str, Any]:
        if not self.credentials_available():
            raise RuntimeError("Alpaca credentials are required to fetch order status.")
        async with httpx.AsyncClient(timeout=15) as client:
            await self._wait_for_alpaca(endpoint="get_order")
            response = await client.get(
                f"{self.settings.alpaca_paper_base_url}/v2/orders/{order_id}",
                headers=self.headers,
            )
            response.raise_for_status()
            return response.json()

    def _limit_price_from_quote(
        self,
        *,
        side: str,
        quote: dict[str, Any] | None,
        latest_price: float | None,
    ) -> float:
        bid = self._quote_float(quote, "bid_price", "bp", "bid")
        ask = self._quote_float(quote, "ask_price", "ap", "ask")
        mid = self._quote_float(quote, "mid_price", "mid")
        if mid is None and bid is not None and ask is not None:
            mid = (bid + ask) / 2

        reference = ask if side == "buy" else bid
        reference = reference or mid
        if reference is None or not math.isfinite(reference) or reference <= 0:
            self.logger.event("rejected_order", symbol=ALLOWED_SYMBOL, side=side, reason="limit_order_quote_missing")
            raise ValueError("limit_order_quote_missing")

        offset = self.settings.limit_price_offset_bps / 10_000
        if side == "buy":
            limit_price = reference * (1 + offset)
        else:
            limit_price = reference * (1 - offset)
        if not math.isfinite(limit_price) or limit_price <= 0:
            self.logger.event("rejected_order", symbol=ALLOWED_SYMBOL, side=side, reason="invalid_limit_price")
            raise ValueError("Limit orders require a valid positive limit price.")
        return round(limit_price, 2)

    @staticmethod
    def _quote_float(quote: dict[str, Any] | None, *keys: str) -> float | None:
        if not quote:
            return None
        for key in keys:
            value = quote.get(key)
            if value is None or value == "":
                continue
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(parsed) and parsed > 0:
                return parsed
        return None

    @staticmethod
    def _quote_size_float(quote: dict[str, Any] | None, *keys: str) -> float | None:
        if not quote:
            return None
        for key in keys:
            value = quote.get(key)
            if value is None or value == "":
                continue
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(parsed) and parsed >= 0:
                return parsed
        return None

    async def _wait_for_alpaca(self, *, endpoint: str) -> None:
        await get_alpaca_rate_limiter(self.settings).acquire(endpoint=endpoint)

    async def _maybe_check_order_status(self, data: dict[str, Any]) -> dict[str, Any]:
        order_id = data.get("id")
        if not order_id or not self.settings.order_status_check_enabled:
            return data
        limiter = get_alpaca_rate_limiter(self.settings)
        if limiter.soft_budget_reached():
            data["status_check_skipped_reason"] = "api_soft_budget"
            self.logger.event("order_status_check_skipped", broker_order_id=order_id, reason="api_soft_budget")
            return data
        if self.settings.order_status_check_delay_seconds > 0:
            await asyncio.sleep(self.settings.order_status_check_delay_seconds)
        try:
            status_data = await self.get_order(str(order_id))
        except Exception as exc:
            self.logger.event("order_status_check_failed", broker_order_id=order_id, error_type=type(exc).__name__)
            return data
        data["status_check"] = status_data
        if status_data.get("status"):
            data["status"] = status_data["status"]
        return data

    def _get_position_cache(self, symbol: str, *, force_refresh: bool) -> Any:
        cached = self._position_cache.get(symbol)
        age = self._cache_age_seconds(cached[0]) if cached else None
        cache_hit = (
            cached is not None
            and not force_refresh
            and self.settings.position_cache_seconds > 0
            and age is not None
            and age <= self.settings.position_cache_seconds
        )
        self._log_cache_event(endpoint="position", symbol=symbol, cache_hit=cache_hit, cache_age_seconds=age)
        if cache_hit:
            return cached[1]
        return _CACHE_MISS

    def _set_position_cache(self, symbol: str, position: dict[str, Any] | None) -> None:
        self._position_cache[symbol] = (datetime.now(UTC), position)

    def invalidate_position_cache(self, symbol: str | None = None) -> None:
        if symbol is None:
            self._position_cache.clear()
            return
        self._position_cache.pop(symbol, None)

    def _get_account_cache(self, *, force_refresh: bool) -> dict[str, Any] | None:
        cached = self._account_cache
        age = self._cache_age_seconds(cached[0]) if cached else None
        cache_hit = (
            cached is not None
            and not force_refresh
            and self.settings.account_equity_cache_seconds > 0
            and age is not None
            and age <= self.settings.account_equity_cache_seconds
        )
        self._log_cache_event(endpoint="account", symbol=ALLOWED_SYMBOL, cache_hit=cache_hit, cache_age_seconds=age)
        if cache_hit:
            return dict(cached[1])
        if cached is not None and get_alpaca_rate_limiter(self.settings).soft_budget_reached():
            self._log_cache_event(
                endpoint="account",
                symbol=ALLOWED_SYMBOL,
                cache_hit=True,
                cache_age_seconds=age,
                api_budget_status="soft_limit_stale_cache",
            )
            return dict(cached[1])
        return None

    def _set_account_cache(self, account: dict[str, Any]) -> None:
        self._account_cache = (datetime.now(UTC), dict(account))

    @staticmethod
    def _cache_age_seconds(cached_at: datetime) -> float:
        return max(0.0, (datetime.now(UTC) - cached_at).total_seconds())

    def _log_cache_event(self, *, endpoint: str, symbol: str, cache_hit: bool, cache_age_seconds: float | None) -> None:
        try:
            self.logger.event(
                "alpaca_cache",
                endpoint=endpoint,
                symbol=symbol,
                cache_hit=cache_hit,
                cache_age_seconds=cache_age_seconds,
            )
        except Exception:
            pass


_CACHE_MISS = object()
