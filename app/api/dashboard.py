import json
import math
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from fastapi import APIRouter, Query, Request

from app.broker.alpaca_client import AlpacaClient
from app.config import ALLOWED_SYMBOL, Settings
from app.data.market_data import MarketDataClient
from app.db.database import SessionLocal
from app.db.models import Order, Signal, Trade
from app.db.repository import Repository
from app.risk.risk_manager import PositionState
from app.utils.time import iso_utc_now


router = APIRouter(prefix="/dashboard", tags=["dashboard"])

SECRET_KEY_PARTS = ("secret", "token", "key", "webhook", "authorization", "password")


@router.get("/summary")
async def dashboard_summary(request: Request) -> dict:
    settings = _settings_from_request(request)
    trader = _trader_from_request(request)
    scheduler = getattr(request.app.state, "scheduler", None)
    market = MarketDataClient(settings)
    broker = getattr(trader, "broker", AlpacaClient(settings))

    with SessionLocal() as db:
        repo = Repository(db)
        latest_signal = repo.latest_signal()
        latest_order = repo.latest_order()
        latest_trade = repo.latest_trade()
        trades = repo.all_trades_ordered()
        order_summary = repo.order_summary()

    position = await _current_position(trader)
    account_summary = await _account_summary(broker)
    market_summary = await _market_snapshot(settings, market)
    trade_metrics = _trade_metrics(trades, position)

    return {
        "app_status": "ok",
        "symbol": settings.symbol,
        "paper_trading_only": settings.paper_trading_only,
        "trading_enabled": settings.trading_enabled,
        "auto_trade_enabled": settings.auto_trade_enabled,
        "scheduler_running": getattr(scheduler, "running", None),
        "latest_btc_price": market_summary.get("latest_close"),
        "latest_signal": _serialize_signal(latest_signal) if latest_signal else None,
        "current_position": _serialize_position(position),
        "alpaca_account": account_summary,
        **order_summary,
        "total_trades": len(trades),
        **trade_metrics,
        "last_order": _serialize_order(latest_order) if latest_order else None,
        "last_trade": _serialize_trade(latest_trade) if latest_trade else None,
        "data_freshness": {
            "latest_timestamp": market_summary.get("latest_timestamp"),
            "current_utc_time": market_summary.get("current_utc_time"),
            "latest_bar_age_seconds": market_summary.get("latest_bar_age_seconds"),
            "cache_age_seconds": market_summary.get("cache_age_seconds"),
        },
    }


@router.get("/signals")
async def dashboard_signals(limit: int = Query(default=100, ge=1)) -> list[dict]:
    with SessionLocal() as db:
        rows = Repository(db).recent_signals(_cap_limit(limit))
        return [_serialize_signal(row) for row in rows]


@router.get("/orders")
async def dashboard_orders(limit: int = Query(default=100, ge=1)) -> list[dict]:
    with SessionLocal() as db:
        rows = Repository(db).recent_orders(_cap_limit(limit))
        return [_serialize_order(row) for row in rows]


@router.get("/trades")
async def dashboard_trades(limit: int = Query(default=100, ge=1)) -> list[dict]:
    with SessionLocal() as db:
        rows = Repository(db).recent_trades(_cap_limit(limit))
        return [_serialize_trade(row) for row in rows]


@router.get("/equity-curve")
async def dashboard_equity_curve() -> list[dict]:
    with SessionLocal() as db:
        trades = Repository(db).all_trades_ordered()
    return _equity_curve(trades)


@router.get("/market")
async def dashboard_market(request: Request) -> dict:
    settings = _settings_from_request(request)
    return await _market_snapshot(settings, MarketDataClient(settings))


@router.post("/run-once")
async def dashboard_run_once(request: Request) -> dict:
    trader = _trader_from_request(request)
    result = await trader.run_once()
    prediction = result.get("prediction") or {}
    decision = result.get("decision") or {}
    order = result.get("order") or {}
    features = prediction.get("features") or {}
    return {
        **result,
        "summary": {
            "action": decision.get("action"),
            "reason": decision.get("reason"),
            "buy_probability": _safe_float(prediction.get("buy_probability")),
            "sell_probability": _safe_float(prediction.get("sell_probability")),
            "order_status": order.get("status") if order else None,
            "latest_price": _safe_float(features.get("close")),
        },
    }


def _settings_from_request(request: Request) -> Settings:
    return getattr(request.app.state, "settings")


def _trader_from_request(request: Request) -> Any:
    return getattr(request.app.state, "trader")


async def _current_position(trader: Any) -> PositionState:
    try:
        return await trader.get_position_state()
    except Exception:
        return PositionState()


async def _account_summary(broker: Any) -> dict | None:
    credentials_available = getattr(broker, "credentials_available", lambda: False)
    if not credentials_available():
        return None
    try:
        account = await broker.get_account()
    except Exception:
        return None
    return {
        "status": account.get("status"),
        "currency": account.get("currency"),
        "buying_power": _safe_float(account.get("buying_power")),
        "cash": _safe_float(account.get("cash")),
        "equity": _safe_float(account.get("equity")),
        "portfolio_value": _safe_float(account.get("portfolio_value")),
        "paper": account.get("paper"),
    }


async def _market_snapshot(settings: Settings, market: MarketDataClient) -> dict:
    bars = pd.DataFrame()
    quote: dict[str, Any] | None = None
    cache_age_seconds = None
    try:
        bars = await market.fetch_bars(settings.symbol)
        if hasattr(market, "bars_cache_age_seconds"):
            cache_age_seconds = market.bars_cache_age_seconds(symbol=settings.symbol)
    except Exception:
        bars = pd.DataFrame()

    try:
        quote = await market.fetch_latest_quote(settings.symbol)
    except Exception:
        quote = None

    latest_bar = _latest_bar_snapshot(bars)
    return {
        "symbol": settings.symbol,
        "timeframe": settings.timeframe,
        **latest_bar,
        "current_utc_time": iso_utc_now(),
        **_quote_snapshot(quote),
        "cache_age_seconds": _safe_float(cache_age_seconds),
        "bars_count": len(bars),
    }


def _trade_metrics(trades: list[Trade], position: PositionState) -> dict:
    pnl_values = _known_trade_pnls(trades)
    total_realized_pnl = float(sum(pnl_values)) if pnl_values else 0.0
    winning_trades = [pnl for pnl in pnl_values if pnl > 0]
    return {
        "total_realized_pnl": total_realized_pnl,
        "total_return_pct": None,
        "unrealized_pnl": _unrealized_pnl(position),
        "win_rate": _safe_percentage(len(winning_trades), len(pnl_values)) if pnl_values else None,
        "average_trade_pnl": float(sum(pnl_values) / len(pnl_values)) if pnl_values else None,
        "best_trade_pnl": max(pnl_values) if pnl_values else None,
        "worst_trade_pnl": min(pnl_values) if pnl_values else None,
        "max_drawdown": _max_drawdown(pnl_values) if pnl_values else None,
    }


def _equity_curve(trades: list[Trade]) -> list[dict]:
    cumulative = 0.0
    peak = 0.0
    points = []
    for trade in trades:
        pnl = _safe_float(trade.pnl)
        if pnl is None:
            continue
        cumulative += pnl
        peak = max(peak, cumulative)
        points.append(
            {
                "timestamp": _serialize_timestamp(trade.created_at),
                "trade_pnl": pnl,
                "cumulative_realized_pnl": cumulative,
                "drawdown": cumulative - peak,
            }
        )
    return points


def _unrealized_pnl(position: PositionState) -> float | None:
    qty = _safe_float(position.qty)
    avg_entry = _safe_float(position.avg_entry_price)
    market_value = _safe_float(position.market_value)
    if qty is None or qty <= 0:
        return 0.0
    if avg_entry is None or avg_entry <= 0 or market_value is None:
        return None
    return market_value - (qty * avg_entry)


def _max_drawdown(pnl_values: list[float]) -> float:
    cumulative = 0.0
    peak = 0.0
    worst = 0.0
    for pnl in pnl_values:
        cumulative += pnl
        peak = max(peak, cumulative)
        worst = min(worst, cumulative - peak)
    return abs(worst)


def _serialize_signal(signal: Signal) -> dict:
    return {
        "created_at": _serialize_timestamp(signal.created_at),
        "symbol": signal.symbol,
        "action": signal.action,
        "buy_probability": _safe_float(signal.buy_probability),
        "sell_probability": _safe_float(signal.sell_probability),
        "reason": signal.reason,
    }


def _serialize_order(order: Order) -> dict:
    return {
        "created_at": _serialize_timestamp(order.created_at),
        "symbol": order.symbol,
        "side": order.side,
        "status": order.status,
        "notional": _safe_float(order.notional),
        "qty": _safe_float(order.qty),
        "broker_order_id": order.broker_order_id,
        "raw_response": _parse_order_raw_response(order.raw_response),
    }


def _serialize_trade(trade: Trade) -> dict:
    return {
        "created_at": _serialize_timestamp(trade.created_at),
        "symbol": trade.symbol,
        "side": trade.side,
        "qty": _safe_float(trade.qty),
        "price": _safe_float(trade.price),
        "pnl": _safe_float(trade.pnl),
    }


def _serialize_position(position: PositionState) -> dict:
    return {
        "symbol": position.symbol,
        "qty": _safe_float(position.qty),
        "avg_entry_price": _safe_float(position.avg_entry_price),
        "market_value": _safe_float(position.market_value),
        "opened_at": _serialize_timestamp(position.opened_at),
        "highest_price": _safe_float(position.highest_price),
        "realized_pnl_today": _safe_float(position.realized_pnl_today),
        "drawdown_pct": _safe_float(position.drawdown_pct),
        "last_loss_at": _serialize_timestamp(position.last_loss_at),
    }


def _parse_order_raw_response(raw_response: str | None) -> Any:
    if not raw_response:
        return {}
    try:
        parsed = json.loads(raw_response)
    except (TypeError, json.JSONDecodeError):
        return None
    return _redact_secrets(parsed)


def _sanitize_mapping(value: dict[str, Any]) -> dict[str, Any]:
    return _redact_secrets(dict(value))


def _redact_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        redacted = {}
        for key, child in value.items():
            if any(part in str(key).lower() for part in SECRET_KEY_PARTS):
                redacted[key] = "***"
            else:
                redacted[key] = _redact_secrets(child)
        return redacted
    if isinstance(value, list):
        return [_redact_secrets(item) for item in value]
    return value


def _serialize_timestamp(value: Any) -> str | None:
    if value is None:
        return None
    try:
        timestamp = pd.Timestamp(value)
    except Exception:
        return None
    if pd.isna(timestamp):
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(UTC)
    else:
        timestamp = timestamp.tz_convert(UTC)
    return timestamp.isoformat()


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _safe_percentage(numerator: int | float | None, denominator: int | float | None) -> float | None:
    numerator_value = _safe_float(numerator)
    denominator_value = _safe_float(denominator)
    if numerator_value is None or denominator_value is None or denominator_value == 0:
        return None
    return numerator_value / denominator_value


def _quote_float(quote: dict[str, Any] | None, *keys: str) -> float | None:
    if not quote:
        return None
    for key in keys:
        parsed = _safe_float(quote.get(key))
        if parsed is not None:
            return parsed
    return None


def _latest_bar_snapshot(bars: pd.DataFrame) -> dict:
    if bars.empty:
        return {"latest_close": None, "latest_timestamp": None, "latest_bar_age_seconds": None}
    latest_timestamp = bars["timestamp"].iloc[-1]
    return {
        "latest_close": _safe_float(bars["close"].iloc[-1]),
        "latest_timestamp": _serialize_timestamp(latest_timestamp),
        "latest_bar_age_seconds": _safe_float(_bar_age_seconds(latest_timestamp)),
    }


def _quote_snapshot(quote: dict[str, Any] | None) -> dict:
    bid = _quote_float(quote, "bid_price", "bp", "bid")
    ask = _quote_float(quote, "ask_price", "ap", "ask")
    bid_size = _quote_float(quote, "bid_size", "bs")
    ask_size = _quote_float(quote, "ask_size", "as")
    mid_price = (bid + ask) / 2 if bid is not None and ask is not None else None
    spread_bps = ((ask - bid) / mid_price * 10_000) if bid is not None and ask is not None and mid_price else None
    quote_imbalance = (
        (bid_size - ask_size) / (bid_size + ask_size)
        if bid_size is not None and ask_size is not None and (bid_size + ask_size) != 0
        else None
    )
    return {
        "latest_quote": _sanitize_mapping(quote or {}),
        "bid_price": bid,
        "ask_price": ask,
        "mid_price": mid_price,
        "spread_bps": _safe_float(spread_bps),
        "quote_imbalance": _safe_float(quote_imbalance),
    }


def _known_trade_pnls(trades: list[Trade]) -> list[float]:
    pnls = []
    for trade in trades:
        pnl = _safe_float(trade.pnl)
        if pnl is not None:
            pnls.append(pnl)
    return pnls


def _bar_age_seconds(value: Any) -> float | None:
    timestamp_iso = _serialize_timestamp(value)
    if timestamp_iso is None:
        return None
    timestamp = datetime.fromisoformat(timestamp_iso)
    return max(0.0, (datetime.now(UTC) - timestamp).total_seconds())


def _cap_limit(limit: int) -> int:
    return max(1, min(int(limit), 500))
