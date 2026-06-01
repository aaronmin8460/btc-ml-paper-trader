import json
import math
from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
from fastapi import APIRouter, Query, Request

from app.broker.alpaca_client import AlpacaClient
from app.config import ALLOWED_SYMBOL, Settings
from app.data.market_data import MarketDataClient
from app.db.database import SessionLocal
from app.db.models import AccountSnapshot, Order, Signal, Trade
from app.db.repository import Repository
from app.ml.registry import ModelRegistry
from app.risk.risk_manager import PositionState, RiskManager, account_state_from_payload
from app.utils.rate_limiter import get_alpaca_rate_limiter
from app.utils.time import iso_utc_now


router = APIRouter(prefix="/dashboard", tags=["dashboard"])

SECRET_KEY_PARTS = ("secret", "token", "key", "webhook", "authorization", "password")
RISK_BLOCK_REASONS = {
    "max_order_attempts_per_hour_reached",
    "max_order_attempts_per_10_minutes_reached",
    "max_order_attempts_per_day_reached",
    "max_trades_per_hour_reached",
    "max_trades_per_10_minutes_reached",
    "max_daily_trades_reached",
    "max_consecutive_losses_reached",
    "trade_cooldown_active",
    "order_in_flight",
    "already_holding_btc",
    "sell_without_position",
    "api_budget_exhausted",
    "account_daily_loss_usd_reached",
    "account_daily_loss_pct_reached",
    "account_drawdown_reached",
    "account_data_required_unavailable",
    "buying_power_too_low",
    "recent_ioc_cancels_too_high",
    "max_daily_loss_reached",
    "max_drawdown_reached",
    "cooldown_after_loss",
    "order_notional_exceeds_max_position",
    "order_notional_exceeds_total_exposure",
    "configured_order_notional_too_large",
    "model_unavailable",
    "profit_guard_holding_until_profitable",
    "profit_guard_holding_at_loss",
    "stale_market_data",
    "spread_too_wide",
    "spread_unavailable",
    "quote_imbalance_unavailable",
    "quote_imbalance_too_weak",
    "scalping_kill_switch:hourly_loss_limit",
    "scalping_kill_switch:ioc_cancel_streak",
    "scalping_kill_switch:loss_streak",
    "scalping_kill_switch:runtime_errors",
}


@router.get("/summary")
async def dashboard_summary(request: Request) -> dict:
    settings = _settings_from_request(request)
    trader = _trader_from_request(request)
    scheduler = getattr(request.app.state, "scheduler", None)
    training_status = _training_scheduler_status(getattr(request.app.state, "training_scheduler", None), settings)
    pause_status = _scheduler_pause_status(scheduler)
    market = MarketDataClient(settings)
    broker = getattr(trader, "broker", AlpacaClient(settings))

    with SessionLocal() as db:
        repo = Repository(db)
        latest_signal = repo.latest_signal()
        latest_order = repo.latest_order()
        latest_trade = repo.latest_trade()
        trades = repo.all_trades_ordered()
        order_summary = repo.order_summary()
        latest_model_run = repo.recent_model_runs(1)
        ioc_cancel_summary = _ioc_cancel_summary(settings, repo)
        today_trade_summary = repo.realized_trade_summary_since(_utc_day_start())

    position = await _current_position(trader)
    account_summary = await _account_summary(broker)
    market_summary = await _market_snapshot(settings, market)
    trade_metrics = _trade_metrics(trades, position)
    profit_guard = _profit_guard_summary(settings, position, market_summary)
    api_budget = get_alpaca_rate_limiter(settings).snapshot()
    latest_model = _latest_model_summary(latest_model_run[0] if latest_model_run else None)
    active_model = ModelRegistry(settings).validate_active_model().to_dict()

    return {
        "app_status": "ok",
        "symbol": settings.symbol,
        "paper_trading_only": settings.paper_trading_only,
        "trading_enabled": settings.trading_enabled,
        "auto_trade_enabled": settings.auto_trade_enabled,
        "scalping_mode_enabled": settings.scalping_mode_enabled,
        **training_status,
        "scheduler_running": getattr(scheduler, "running", None),
        "scheduler_paused": bool(pause_status.get("paused")),
        "pause_reason": pause_status.get("pause_reason"),
        "scheduler_pause_reason": pause_status.get("pause_reason"),
        "scheduler_paused_at": pause_status.get("paused_at"),
        "latest_btc_price": market_summary.get("latest_close"),
        "latest_signal": _serialize_signal(latest_signal) if latest_signal else None,
        "current_position": _serialize_position(position),
        "current_unrealized_pnl": trade_metrics.get("unrealized_pnl"),
        "realized_pnl_today": today_trade_summary["realized_pnl"],
        "total_trades_today": today_trade_summary["total_trades"],
        "win_rate_today": today_trade_summary["win_rate"],
        **profit_guard,
        "alpaca_account": account_summary,
        "alpaca_calls_last_minute": api_budget.get("calls_last_minute"),
        "alpaca_budget_remaining": api_budget.get("budget_remaining"),
        "alpaca_endpoint_counts": api_budget.get("endpoint_counts"),
        "api_budget_status": api_budget.get("api_budget_status"),
        "account_equity": account_summary.get("equity") if account_summary else None,
        "cash": account_summary.get("cash") if account_summary else None,
        "buying_power": account_summary.get("buying_power") if account_summary else None,
        "portfolio_value": account_summary.get("portfolio_value") if account_summary else None,
        "account_daily_change_usd": account_summary.get("daily_change_usd") if account_summary else None,
        "account_daily_change_pct": account_summary.get("daily_change_pct") if account_summary else None,
        "account_drawdown_pct": account_summary.get("drawdown_pct") if account_summary else None,
        "latest_model_net_return_pct": latest_model.get("latest_model_net_return_pct"),
        "latest_model_max_drawdown_pct": latest_model.get("latest_model_max_drawdown_pct"),
        "latest_model_profit_factor": latest_model.get("latest_model_profit_factor"),
        "latest_model_accepted": latest_model.get("latest_model_accepted"),
        "latest_model_rejected_reason": latest_model.get("latest_model_rejected_reason"),
        **active_model,
        **order_summary,
        "total_trades": len(trades),
        **trade_metrics,
        "latest_order": _serialize_order(latest_order) if latest_order else None,
        "last_order": _serialize_order(latest_order) if latest_order else None,
        "last_trade": _serialize_trade(latest_trade) if latest_trade else None,
        "ioc_cancel_guard": ioc_cancel_summary,
        "data_freshness": {
            "latest_timestamp": market_summary.get("latest_timestamp"),
            "current_utc_time": market_summary.get("current_utc_time"),
            "latest_bar_age_seconds": market_summary.get("latest_bar_age_seconds"),
            "quote_timestamp": market_summary.get("quote_timestamp"),
            "quote_age_seconds": market_summary.get("quote_age_seconds"),
            "cache_age_seconds": market_summary.get("cache_age_seconds"),
        },
    }


@router.get("/signals")
async def dashboard_signals(limit: int = Query(default=100, ge=1)) -> list[dict]:
    with SessionLocal() as db:
        rows = Repository(db).recent_signals(_cap_limit(limit))
        return [_serialize_signal(row) for row in rows]


@router.get("/orders")
async def dashboard_orders(limit: int = Query(default=50, ge=1)) -> list[dict]:
    with SessionLocal() as db:
        rows = Repository(db).recent_orders(_cap_limit(limit))
        return [_serialize_order(row) for row in rows]


@router.get("/trades")
async def dashboard_trades(limit: int = Query(default=50, ge=1)) -> list[dict]:
    with SessionLocal() as db:
        rows = Repository(db).recent_trades(_cap_limit(limit))
        return [_serialize_trade(row) for row in rows]


@router.get("/risk")
async def dashboard_risk(request: Request) -> dict:
    settings = _settings_from_request(request)
    scheduler = getattr(request.app.state, "scheduler", None)
    pause_status = _scheduler_pause_status(scheduler)
    now = datetime.now(UTC)
    with SessionLocal() as db:
        repo = Repository(db)
        recent_signals = repo.recent_signals(500)
        recent_risk_events = repo.recent_risk_events(100)
        trade_frequency = repo.filled_trade_frequency_state(now=now)
        order_attempt_frequency = repo.order_attempt_frequency_state(now=now)
        daily_trade_summary = repo.realized_trade_summary_since(_utc_day_start(now))
        hourly_realized_pnl = repo.realized_pnl_last_hour(now=now)
        latest_kill_switch_event = repo.latest_scalping_kill_switch_event()
        ioc_cancel_summary = _ioc_cancel_summary(settings, repo)
    return {
        "symbol": settings.symbol,
        "active_risk_blocks": _active_risk_blocks(
            recent_signals,
            paused=bool(pause_status.get("paused")),
            pause_reason=pause_status.get("pause_reason"),
            paused_at=pause_status.get("paused_at"),
        ),
        "recent_risk_block_counts": _recent_risk_block_counts(recent_signals, now=now),
        "trade_frequency_state": _frequency_state_payload(
            trade_frequency,
            ten_minute_limit=settings.max_trades_per_10_minutes,
            hourly_limit=settings.max_trades_per_hour,
            daily_limit=settings.max_daily_trades,
        ),
        "order_attempt_frequency_state": _frequency_state_payload(
            order_attempt_frequency,
            ten_minute_limit=settings.max_order_attempts_per_10_minutes,
            hourly_limit=settings.max_order_attempts_per_hour,
            daily_limit=settings.max_order_attempts_per_day,
        ),
        "daily_loss_state": _loss_state(
            realized_pnl=daily_trade_summary["realized_pnl"],
            limit_usd=settings.max_daily_loss_usd,
        ),
        "hourly_loss_state": _loss_state(
            realized_pnl=hourly_realized_pnl,
            limit_usd=settings.max_loss_usd_per_hour,
        ),
        "kill_switch_state": {
            "enabled": settings.scalping_mode_enabled and settings.scalping_kill_switch_enabled,
            "paused": bool(pause_status.get("paused")),
            "pause_reason": pause_status.get("pause_reason"),
            "paused_at": pause_status.get("paused_at"),
            "latest_persisted_event": _serialize_risk_event(latest_kill_switch_event),
        },
        "ioc_cancel_guard": ioc_cancel_summary,
        "recent_risk_events": [_serialize_risk_event(event) for event in recent_risk_events],
    }


@router.get("/model")
async def dashboard_model(request: Request) -> dict:
    return _dashboard_model_payload(_settings_from_request(request))


@router.get("/equity-curve")
async def dashboard_equity_curve() -> list[dict]:
    with SessionLocal() as db:
        trades = Repository(db).all_trades_ordered()
    return _equity_curve(trades)


@router.get("/account-snapshots")
async def dashboard_account_snapshots(limit: int = Query(default=500, ge=1)) -> list[dict]:
    with SessionLocal() as db:
        rows = Repository(db).recent_account_snapshots(_cap_limit(limit))
        return [_serialize_account_snapshot(row) for row in rows]


@router.get("/portfolio-curve")
async def dashboard_portfolio_curve() -> list[dict]:
    with SessionLocal() as db:
        rows = Repository(db).recent_account_snapshots(500)
    return _portfolio_curve(rows)


@router.get("/trading-status")
async def dashboard_trading_status(request: Request) -> dict:
    settings = _settings_from_request(request)
    scheduler = getattr(request.app.state, "scheduler", None)
    pause_status = _scheduler_pause_status(scheduler)
    with SessionLocal() as db:
        repo = Repository(db)
        latest_signal = repo.latest_signal()
        recent_signals = repo.recent_signals(50)
        ioc_cancel_summary = _ioc_cancel_summary(settings, repo)
    return _trading_status_payload(
        settings=settings,
        scheduler_running=getattr(scheduler, "running", None),
        latest_signal=latest_signal,
        recent_signals=recent_signals,
        ioc_cancel_summary=ioc_cancel_summary,
        paused=bool(pause_status.get("paused")),
        pause_reason=pause_status.get("pause_reason"),
        paused_at=pause_status.get("paused_at"),
    )


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
    state = account_state_from_payload(account)
    return {
        "status": account.get("status"),
        "currency": account.get("currency"),
        "buying_power": state.buying_power,
        "cash": state.cash,
        "equity": state.equity,
        "portfolio_value": state.portfolio_value,
        "last_equity": state.last_equity,
        "daily_change_usd": state.daily_change_usd,
        "daily_change_pct": state.daily_change_pct,
        "drawdown_pct": state.drawdown_pct,
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

    quote_cache_age_seconds = None
    if hasattr(market, "quote_cache_age_seconds"):
        quote_cache_age_seconds = market.quote_cache_age_seconds(symbol=settings.symbol)
    latest_bar = _latest_bar_snapshot(bars)
    return {
        "symbol": settings.symbol,
        "timeframe": settings.timeframe,
        **latest_bar,
        "current_utc_time": iso_utc_now(),
        **_quote_snapshot(quote, cache_age_seconds=quote_cache_age_seconds),
        "cache_age_seconds": _safe_float(cache_age_seconds),
        "bars_count": len(bars),
    }


def _trade_metrics(trades: list[Trade], position: PositionState) -> dict:
    pnl_values = _closed_trade_pnls(trades)
    total_realized_pnl = float(sum(pnl_values)) if pnl_values else 0.0
    winning_trades = [pnl for pnl in pnl_values if pnl > 0]
    return {
        "closed_trades": len(pnl_values),
        "total_realized_pnl": total_realized_pnl,
        "total_return_pct": None,
        "unrealized_pnl": _unrealized_pnl(position),
        "win_rate": _safe_percentage(len(winning_trades), len(pnl_values)) if pnl_values else None,
        "average_trade_pnl": float(sum(pnl_values) / len(pnl_values)) if pnl_values else None,
        "best_trade_pnl": max(pnl_values) if pnl_values else None,
        "worst_trade_pnl": min(pnl_values) if pnl_values else None,
        "max_drawdown": _max_drawdown(pnl_values) if pnl_values else None,
    }


def _profit_guard_summary(settings: Settings, position: PositionState, market_summary: dict) -> dict:
    risk = RiskManager(settings)
    quote = market_summary.get("latest_quote") if isinstance(market_summary.get("latest_quote"), dict) else None
    latest_price = market_summary.get("mid_price") or market_summary.get("latest_close")
    estimated_exit_price = risk.estimated_exit_price(quote=quote, latest_price=_safe_float(latest_price))
    snapshot = risk.profit_guard_snapshot(position, estimated_exit_price=estimated_exit_price)
    return {
        "profit_guard_enabled": snapshot.profit_guard_enabled,
        "min_net_exit_profit_pct": snapshot.min_net_exit_profit_pct,
        "current_unrealized_pnl_pct": snapshot.current_unrealized_pnl_pct,
        "profit_guard_exit_allowed": snapshot.profit_guard_exit_allowed,
        "estimated_exit_price": snapshot.estimated_exit_price,
        "minimum_profitable_exit_price": snapshot.minimum_profitable_exit_price,
    }


def _equity_curve(trades: list[Trade]) -> list[dict]:
    cumulative = 0.0
    peak = 0.0
    points = []
    for trade in trades:
        if str(trade.side).lower() != "sell":
            continue
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


def _portfolio_curve(snapshots: list[AccountSnapshot]) -> list[dict]:
    return [
        {
            "timestamp": _serialize_timestamp(snapshot.created_at),
            "equity": _safe_float(snapshot.equity),
            "cash": _safe_float(snapshot.cash),
            "buying_power": _safe_float(snapshot.buying_power),
            "portfolio_value": _safe_float(snapshot.portfolio_value),
        }
        for snapshot in sorted(snapshots, key=lambda row: row.created_at)
    ]


def _trading_status_payload(
    *,
    settings: Settings,
    scheduler_running: bool | None,
    latest_signal: Signal | None,
    recent_signals: list[Signal],
    ioc_cancel_summary: dict,
    paused: bool = False,
    pause_reason: str | None = None,
    paused_at: str | None = None,
) -> dict:
    latest_risk_reason = _latest_risk_block_reason(recent_signals)
    active_model = ModelRegistry(settings).validate_active_model().to_dict()
    model_available = bool(active_model["active_model_valid"])
    prediction_source = "model" if model_available else "fallback"
    state, state_tone = _trading_state(
        settings=settings,
        scheduler_running=scheduler_running,
        paused=paused,
        latest_decision_action=latest_signal.action if latest_signal else None,
        latest_risk_block_reason=latest_risk_reason,
        ioc_cooldown_active=bool(ioc_cancel_summary.get("cooldown_active")),
    )
    return {
        "state": state,
        "state_tone": state_tone,
        "paused": paused,
        "pause_reason": pause_reason,
        "paused_at": paused_at,
        "latest_decision_action": latest_signal.action if latest_signal else None,
        "latest_decision_reason": latest_signal.reason if latest_signal else None,
        "latest_risk_block_reason": latest_risk_reason,
        "current_ioc_cancel_count": ioc_cancel_summary.get("recent_buy_ioc_cancel_count"),
        "ioc_cancel_lookback_seconds": ioc_cancel_summary.get("lookback_seconds"),
        "ioc_cooldown_active": bool(ioc_cancel_summary.get("cooldown_active")),
        "ioc_cooldown_expires_at": ioc_cancel_summary.get("cooldown_expires_at"),
        "scheduler_running": scheduler_running,
        "auto_trade_enabled": settings.auto_trade_enabled,
        "trading_enabled": settings.trading_enabled,
        "paper_trading_only": settings.paper_trading_only,
        "model_available": model_available,
        "prediction_source": prediction_source,
        **active_model,
        "fallback_trading_allowed": settings.allow_fallback_trading,
    }


def _latest_risk_block_reason(signals: list[Signal]) -> str | None:
    for signal in signals:
        if signal.action == "hold" and signal.reason in RISK_BLOCK_REASONS:
            return signal.reason
    return None


def _model_available(settings: Settings) -> bool:
    return ModelRegistry(settings).validate_active_model().valid


def _scheduler_pause_status(scheduler: Any) -> dict:
    if scheduler is None:
        return {"paused": False, "pause_reason": None, "paused_at": None}
    if hasattr(scheduler, "status"):
        try:
            status = scheduler.status()
            return {
                "paused": bool(status.get("paused")),
                "pause_reason": status.get("pause_reason"),
                "paused_at": _serialize_timestamp(status.get("paused_at")),
            }
        except Exception:
            pass
    return {
        "paused": bool(getattr(scheduler, "paused", False)),
        "pause_reason": getattr(scheduler, "pause_reason", None),
        "paused_at": _serialize_timestamp(getattr(scheduler, "paused_at", None)),
    }


def _training_scheduler_status(training_scheduler: Any, settings: Settings) -> dict:
    if training_scheduler is None or not hasattr(training_scheduler, "status"):
        return {
            "auto_train_enabled": settings.auto_train_enabled,
            "training_scheduler_running": False,
            "last_training_started_at": None,
            "last_training_finished_at": None,
            "last_training_status": None,
            "last_training_reason": None,
            "last_training_model_path": None,
            "last_training_accepted": None,
            "last_training_metrics": None,
        }
    try:
        status = training_scheduler.status()
    except Exception:
        return {
            "auto_train_enabled": settings.auto_train_enabled,
            "training_scheduler_running": False,
            "last_training_started_at": None,
            "last_training_finished_at": None,
            "last_training_status": "unavailable",
            "last_training_reason": "training_scheduler_status_unavailable",
            "last_training_model_path": None,
            "last_training_accepted": None,
            "last_training_metrics": None,
        }
    return {
        "auto_train_enabled": bool(status.get("auto_train_enabled")),
        "training_scheduler_running": bool(status.get("running")),
        "last_training_started_at": _serialize_timestamp(status.get("last_training_started_at")),
        "last_training_finished_at": _serialize_timestamp(status.get("last_training_finished_at")),
        "last_training_status": status.get("last_training_status"),
        "last_training_reason": status.get("last_training_reason"),
        "last_training_model_path": status.get("last_training_model_path"),
        "last_training_accepted": status.get("last_training_accepted"),
        "last_training_metrics": status.get("last_training_metrics"),
    }


def _trading_state(
    *,
    settings: Settings,
    scheduler_running: bool | None,
    paused: bool,
    latest_decision_action: str | None,
    latest_risk_block_reason: str | None,
    ioc_cooldown_active: bool,
) -> tuple[str, str]:
    if paused:
        return "paused", "red"
    if not settings.trading_enabled:
        return "disabled", "gray"
    if not settings.auto_trade_enabled:
        return "paused", "red"
    if scheduler_running is False:
        return "stopped", "red"
    if ioc_cooldown_active:
        return "cooling_down", "yellow"
    if latest_risk_block_reason and latest_decision_action == "hold":
        return "blocked", "red"
    if latest_decision_action == "hold":
        return "waiting", "yellow"
    return "running", "green"


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
    timestamp = _serialize_timestamp(signal.created_at)
    return {
        "timestamp": timestamp,
        "created_at": timestamp,
        "symbol": signal.symbol,
        "action": signal.action,
        "buy_probability": _safe_float(signal.buy_probability),
        "sell_probability": _safe_float(signal.sell_probability),
        "reason": signal.reason,
        "spread_bps": _safe_float(signal.spread_bps),
        "quote_imbalance": _safe_float(signal.quote_imbalance),
        "model_version": signal.model_version,
    }


def _serialize_order(order: Order) -> dict:
    raw_response = _parse_order_raw_response(order.raw_response)
    raw = raw_response if isinstance(raw_response, dict) else {}
    timestamp = _serialize_timestamp(order.created_at)
    return {
        "local_order_id": order.id,
        "order_id": order.broker_order_id,
        "timestamp": timestamp,
        "created_at": timestamp,
        "symbol": order.symbol,
        "side": order.side,
        "status": order.status,
        "notional": _safe_float(order.notional),
        "qty": _safe_float(order.qty),
        "broker_order_id": order.broker_order_id,
        "order_type": raw.get("order_type") or raw.get("type"),
        "time_in_force": raw.get("time_in_force"),
        "limit_price": _safe_float(raw.get("limit_price")),
        "filled_qty": _safe_float(raw.get("filled_qty")),
        "filled_avg_price": _safe_float(raw.get("filled_avg_price")),
        "fee_amount": _safe_float(raw.get("fee_amount")),
        "slippage_amount": _safe_float(raw.get("slippage_amount")),
        "cancel_reason": _order_cancel_reason(order.status, raw),
        "raw_response": raw_response,
    }


def _serialize_trade(trade: Trade) -> dict:
    timestamp = _serialize_timestamp(trade.created_at)
    qty = _safe_float(trade.qty)
    price = _safe_float(trade.price)
    entry_price = _safe_float(trade.entry_price)
    exit_price = _safe_float(trade.exit_price)
    if entry_price is None and str(trade.side).lower() == "buy":
        entry_price = price
    if exit_price is None and str(trade.side).lower() == "sell":
        exit_price = price
    return {
        "timestamp": timestamp,
        "created_at": timestamp,
        "symbol": trade.symbol,
        "side": trade.side,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "qty": qty,
        "notional": _safe_float(trade.notional) or _multiply(qty, exit_price or entry_price),
        "gross_pnl": _safe_float(trade.gross_pnl),
        "net_pnl": _safe_float(trade.net_pnl if trade.net_pnl is not None else trade.pnl),
        "fee_amount": _safe_float(trade.fee_amount),
        "slippage_amount": _safe_float(trade.slippage_amount),
        "hold_seconds": _safe_float(trade.hold_seconds),
        "reason": trade.reason,
        "price": price,
        "pnl": _safe_float(trade.pnl),
    }


def _serialize_risk_event(event: Any | None) -> dict | None:
    if event is None:
        return None
    return {
        "timestamp": _serialize_timestamp(getattr(event, "created_at", None)),
        "event_type": getattr(event, "event_type", None),
        "reason": getattr(event, "reason", None),
    }


def _serialize_account_snapshot(snapshot: AccountSnapshot) -> dict:
    return {
        "created_at": _serialize_timestamp(snapshot.created_at),
        "equity": _safe_float(snapshot.equity),
        "cash": _safe_float(snapshot.cash),
        "buying_power": _safe_float(snapshot.buying_power),
        "portfolio_value": _safe_float(snapshot.portfolio_value),
        "currency": snapshot.currency,
        "raw_response": _parse_order_raw_response(snapshot.raw_response),
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


def _order_cancel_reason(status: str, raw_response: dict[str, Any]) -> str | None:
    explicit = raw_response.get("cancel_reason") or raw_response.get("reason")
    if explicit:
        return str(explicit)
    if str(status or "").lower() not in {"canceled", "cancelled"}:
        return None
    order_type = str(raw_response.get("order_type") or raw_response.get("type") or "").lower()
    time_in_force = str(raw_response.get("time_in_force") or "").lower()
    return "ioc_no_fill" if order_type == "limit" and time_in_force == "ioc" else "broker_canceled"


def _latest_model_summary(model_run: Any | None) -> dict:
    if model_run is None:
        return {
            "latest_model_net_return_pct": None,
            "latest_model_max_drawdown_pct": None,
            "latest_model_profit_factor": None,
            "latest_model_accepted": None,
            "latest_model_rejected_reason": None,
        }
    metrics = _parse_order_raw_response(getattr(model_run, "metrics_json", None))
    if not isinstance(metrics, dict):
        metrics = {}
    status = getattr(model_run, "status", None)
    return {
        "latest_model_net_return_pct": _safe_float(metrics.get("net_return_pct")),
        "latest_model_max_drawdown_pct": _safe_float(metrics.get("max_drawdown_pct")),
        "latest_model_profit_factor": _safe_float(metrics.get("profit_factor_net")),
        "latest_model_accepted": status == "accepted" if status is not None else None,
        "latest_model_rejected_reason": None if status == "accepted" else metrics.get("promotion_reason") or metrics.get("reason"),
    }


def _dashboard_model_payload(settings: Settings) -> dict:
    registry = ModelRegistry(settings)
    registry_data = _sanitize_mapping(registry.read())
    active_model = registry.validate_active_model()
    metrics = registry_data.get("metrics") if isinstance(registry_data.get("metrics"), dict) else {}
    feature_columns = registry_data.get("feature_columns")
    return {
        "active_model_path": active_model.active_model_path,
        "active_model_status": active_model.status,
        "model_version": active_model.model_version or registry_data.get("model_version"),
        "trained_at": registry_data.get("created_at") or registry_data.get("training_end"),
        "feature_columns": list(feature_columns) if isinstance(feature_columns, list) else [],
        "validation_metrics": metrics,
        "promotion_reason": active_model.promotion_reason or metrics.get("promotion_reason") or metrics.get("reason"),
        "active_model_valid": active_model.valid,
        "active_model_invalid_reason": active_model.invalid_reason,
        "supports_independent_sell_probability": bool(
            registry_data.get("supports_independent_sell_probability")
        ),
    }


def _active_risk_blocks(
    signals: list[Signal],
    *,
    paused: bool,
    pause_reason: str | None,
    paused_at: str | None,
) -> list[dict]:
    blocks: list[dict] = []
    if paused and pause_reason:
        blocks.append(
            {
                "reason": pause_reason,
                "source": "scheduler_pause",
                "timestamp": paused_at,
            }
        )
    if signals:
        latest = signals[0]
        if latest.action == "hold" and latest.reason in RISK_BLOCK_REASONS:
            blocks.append(
                {
                    "reason": latest.reason,
                    "source": "latest_signal",
                    "timestamp": _serialize_timestamp(latest.created_at),
                }
            )
    return blocks


def _recent_risk_block_counts(signals: list[Signal], *, now: datetime) -> dict:
    window_seconds = 3600
    cutoff = _ensure_datetime_utc(now) - timedelta(seconds=window_seconds)
    counts = Counter(
        signal.reason
        for signal in signals
        if signal.action == "hold"
        and signal.reason in RISK_BLOCK_REASONS
        and (_ensure_datetime_utc(signal.created_at) or cutoff) >= cutoff
    )
    return {
        "window_seconds": window_seconds,
        "total": sum(counts.values()),
        "by_reason": dict(sorted(counts.items())),
    }


def _frequency_state_payload(
    state: Any,
    *,
    ten_minute_limit: int,
    hourly_limit: int,
    daily_limit: int,
) -> dict:
    last_10_minutes = int(getattr(state, "trades_last_10_minutes", 0) or 0)
    last_hour = int(getattr(state, "trades_last_hour", 0) or 0)
    today = int(getattr(state, "trades_today", 0) or 0)
    return {
        "last_10_minutes": last_10_minutes,
        "last_hour": last_hour,
        "today": today,
        "last_event_at": _serialize_timestamp(getattr(state, "last_trade_at", None)),
        "limits": {
            "last_10_minutes": ten_minute_limit,
            "last_hour": hourly_limit,
            "today": daily_limit,
        },
        "blocked": (
            last_10_minutes >= ten_minute_limit
            or last_hour >= hourly_limit
            or today >= daily_limit
        ),
    }


def _loss_state(*, realized_pnl: float | int | None, limit_usd: float) -> dict:
    pnl = _safe_float(realized_pnl) or 0.0
    limit = abs(float(limit_usd))
    return {
        "realized_pnl": pnl,
        "loss_limit_usd": limit,
        "blocked": pnl <= -limit,
    }


def _ioc_cancel_summary(settings: Settings, repo: Repository) -> dict:
    now = datetime.now(UTC)
    latest_buy_cancel = repo.latest_ioc_canceled_order(side="buy")
    latest_any_cancel = repo.latest_ioc_canceled_order()
    latest_buy_cancel_at = _ensure_datetime_utc(
        latest_buy_cancel.created_at if latest_buy_cancel is not None else None
    )
    recent_buy_count = repo.recent_ioc_canceled_count(
        side="buy",
        now=now,
        lookback_seconds=settings.ioc_cancel_lookback_seconds,
    )
    cooldown_remaining = _seconds_until(
        latest_buy_cancel_at,
        now=now,
        seconds=settings.ioc_cancel_cooldown_seconds,
    )
    cooldown_expires_at = (
        latest_buy_cancel_at + timedelta(seconds=max(0, settings.ioc_cancel_cooldown_seconds))
        if latest_buy_cancel_at is not None and cooldown_remaining > 0
        else None
    )
    return {
        "latest_ioc_cancel_at": _serialize_timestamp(
            latest_any_cancel.created_at if latest_any_cancel is not None else None
        ),
        "latest_buy_ioc_cancel_at": _serialize_timestamp(latest_buy_cancel_at),
        "recent_buy_ioc_cancel_count": recent_buy_count,
        "lookback_seconds": settings.ioc_cancel_lookback_seconds,
        "max_recent_ioc_cancels": settings.max_recent_ioc_cancels,
        "cooldown_active": cooldown_remaining > 0,
        "cooldown_seconds_remaining": cooldown_remaining if cooldown_remaining > 0 else 0,
        "cooldown_expires_at": _serialize_timestamp(cooldown_expires_at),
    }


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


def _ensure_datetime_utc(value: Any) -> datetime | None:
    timestamp_iso = _serialize_timestamp(value)
    if timestamp_iso is None:
        return None
    return datetime.fromisoformat(timestamp_iso)


def _seconds_until(value: datetime | None, *, now: datetime, seconds: int | float) -> float:
    if value is None:
        return 0.0
    ends_at = value + timedelta(seconds=max(0, seconds))
    return max(0.0, (ends_at - now).total_seconds())


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


def _multiply(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left * right


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


def _quote_snapshot(quote: dict[str, Any] | None, *, cache_age_seconds: float | None = None) -> dict:
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
    quote_timestamp = _quote_timestamp(quote)
    return {
        "latest_quote": _sanitize_mapping(quote or {}),
        "bid_price": bid,
        "ask_price": ask,
        "mid_price": mid_price,
        "spread_bps": _safe_float(spread_bps),
        "quote_imbalance": _safe_float(quote_imbalance),
        "quote_timestamp": quote_timestamp,
        "quote_age_seconds": _safe_float(_bar_age_seconds(quote_timestamp)) if quote_timestamp else _safe_float(cache_age_seconds),
    }


def _quote_timestamp(quote: dict[str, Any] | None) -> str | None:
    if not quote:
        return None
    for key in ("timestamp", "t", "quote_timestamp"):
        timestamp = _serialize_timestamp(quote.get(key))
        if timestamp is not None:
            return timestamp
    return None


def _closed_trade_pnls(trades: list[Trade]) -> list[float]:
    pnls = []
    for trade in trades:
        if str(trade.side).lower() != "sell":
            continue
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


def _utc_day_start(value: datetime | None = None) -> datetime:
    current = _ensure_datetime_utc(value or datetime.now(UTC)) or datetime.now(UTC)
    return current.replace(hour=0, minute=0, second=0, microsecond=0)


def _cap_limit(limit: int) -> int:
    return max(1, min(int(limit), 500))
