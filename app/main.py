import json
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy import text

from app.api.dashboard import router as dashboard_router
from app.config import Settings, get_settings
from app.data.market_data import MarketDataClient
from app.db.database import SessionLocal, init_db
from app.db.repository import Repository
from app.ml.registry import ModelRegistry
from app.ml.train import train_model_from_bars
from app.monitoring.logger import get_logger
from app.notifications.discord import DiscordNotifier
from app.risk.risk_manager import account_state_from_payload
from app.services.scheduler import TradingScheduler
from app.services.training_scheduler import TrainingScheduler
from app.services.trader import Trader
from app.utils.time import iso_utc_now

FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"
DASHBOARD_UI_PREFIX = "/dashboard-ui"

app = FastAPI(title="btc-ml-paper-trader", version="0.1.0")
settings = get_settings()
trader = Trader(settings)
scheduler = TradingScheduler(trader, settings)
training_scheduler = TrainingScheduler(settings, broker=trader.broker)
app.state.settings = settings
app.state.trader = trader
app.state.scheduler = scheduler
app.state.training_scheduler = training_scheduler


def serialize_model(row) -> dict:
    return {key: value for key, value in row.__dict__.items() if not key.startswith("_")}


def serialize_timestamp(value) -> str | None:
    return value.isoformat() if value is not None else None


def runtime_scheduler():
    return getattr(app.state, "scheduler", scheduler)


def runtime_training_scheduler():
    return getattr(app.state, "training_scheduler", training_scheduler)


def require_admin(x_admin_token: str | None = Header(default=None)) -> None:
    token = settings.api_admin_token
    if not token or x_admin_token != token:
        raise HTTPException(status_code=401, detail="Admin token required")


app.include_router(dashboard_router, dependencies=[Depends(require_admin)])


def configure_frontend_static(fastapi_app: FastAPI, static_dir: Path | str = FRONTEND_DIST) -> bool:
    dist = Path(static_dir)
    index_path = dist / "index.html"
    if not dist.is_dir() or not index_path.is_file():
        _log_frontend_static_event("frontend_static_unavailable", static_dir=str(dist))
        return False

    root = dist.resolve()
    index = index_path.resolve()

    @fastapi_app.get(DASHBOARD_UI_PREFIX, include_in_schema=False)
    @fastapi_app.get(f"{DASHBOARD_UI_PREFIX}/{{asset_path:path}}", include_in_schema=False)
    async def dashboard_ui(asset_path: str = "") -> FileResponse:
        if not asset_path:
            return FileResponse(index)

        candidate = (root / asset_path).resolve()
        if not candidate.is_relative_to(root):
            raise HTTPException(status_code=404, detail="Not found")
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(index)

    _log_frontend_static_event("frontend_static_enabled", static_dir=str(root), mount_path=DASHBOARD_UI_PREFIX)
    return True


def _log_frontend_static_event(event_type: str, **payload: str) -> None:
    try:
        get_logger().event(event_type, **payload)
    except Exception:
        pass


configure_frontend_static(app)


@app.on_event("startup")
async def startup() -> None:
    init_db()
    pause_restored = scheduler.restore_pause_state()
    scheduler_started = False
    if settings.auto_trade_enabled:
        scheduler_started = scheduler.start()
    if settings.auto_train_enabled:
        training_scheduler.start()
    _log_application_startup(
        scheduler_started=scheduler_started,
        pause_restored=pause_restored,
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "paper_trading_only": settings.paper_trading_only, "symbol": settings.symbol}


@app.get("/health/deep")
async def health_deep() -> JSONResponse:
    checks = _deep_health_checks()
    healthy = all(check.get("ok") is True for check in checks.values())
    return JSONResponse(
        status_code=200 if healthy else 503,
        content={
            "status": "ok" if healthy else "degraded",
            "paper_trading_only": settings.paper_trading_only,
            "symbol": settings.symbol,
            "checks": checks,
        },
    )


@app.get("/config/safe", dependencies=[Depends(require_admin)])
async def safe_config() -> dict:
    return {**settings.safe_dict(), **ModelRegistry(settings).validate_active_model().to_dict()}


@app.get("/position", dependencies=[Depends(require_admin)])
async def position() -> dict:
    state = await trader.get_position_state()
    return state.__dict__


@app.get("/signals/latest", dependencies=[Depends(require_admin)])
async def latest_signal() -> dict:
    with SessionLocal() as db:
        signal = Repository(db).latest_signal()
        return serialize_model(signal) if signal else {}


@app.get("/orders", dependencies=[Depends(require_admin)])
async def orders() -> list[dict]:
    with SessionLocal() as db:
        rows = Repository(db).recent_orders()
        return [serialize_model(row) for row in rows]


@app.post("/run-once", dependencies=[Depends(require_admin)])
async def run_once() -> dict:
    return await trader.run_once()


@app.get("/debug/latest-bars", dependencies=[Depends(require_admin)])
async def debug_latest_bars(force_refresh: bool = False) -> dict:
    market_data = MarketDataClient(settings)
    try:
        bars = await market_data.fetch_bars(settings.symbol, force_refresh=force_refresh)
    except TypeError:
        bars = await market_data.fetch_bars(settings.symbol)
    first_timestamp = bars["timestamp"].iloc[0] if not bars.empty else None
    latest_timestamp = bars["timestamp"].iloc[-1] if not bars.empty else None
    latest_close = float(bars["close"].iloc[-1]) if not bars.empty else None
    cache_age_seconds = None
    if hasattr(market_data, "bars_cache_age_seconds"):
        cache_age_seconds = market_data.bars_cache_age_seconds(symbol=settings.symbol)
    return {
        "symbol": settings.symbol,
        "timeframe": settings.timeframe,
        "count": len(bars),
        "first_timestamp": serialize_timestamp(first_timestamp),
        "latest_timestamp": serialize_timestamp(latest_timestamp),
        "current_utc_time": iso_utc_now(),
        "latest_close": latest_close,
        "cache_age_seconds": cache_age_seconds,
    }


@app.post("/auto/start", dependencies=[Depends(require_admin)])
async def auto_start() -> dict:
    active_scheduler = runtime_scheduler()
    started = active_scheduler.start()
    return {"started": started, "running": active_scheduler.running}


@app.post("/auto/stop", dependencies=[Depends(require_admin)])
async def auto_stop() -> dict:
    active_scheduler = runtime_scheduler()
    stopped = await active_scheduler.stop()
    return {"stopped": stopped, "running": active_scheduler.running}


@app.post("/admin/pause", dependencies=[Depends(require_admin)])
async def admin_pause() -> dict:
    active_scheduler = runtime_scheduler()
    changed = await active_scheduler.pause("manual_pause", send_alert=False)
    return {"changed": changed, **scheduler_status_payload(active_scheduler)}


@app.post("/admin/resume", dependencies=[Depends(require_admin)])
async def admin_resume() -> dict:
    active_scheduler = runtime_scheduler()
    changed = active_scheduler.resume()
    return {"changed": changed, **scheduler_status_payload(active_scheduler)}


@app.get("/admin/status", dependencies=[Depends(require_admin)])
async def admin_status() -> dict:
    return scheduler_status_payload(runtime_scheduler())


@app.get("/scheduler/status", dependencies=[Depends(require_admin)])
async def scheduler_status() -> dict:
    return scheduler_status_payload(runtime_scheduler())


@app.get("/admin/training/status", dependencies=[Depends(require_admin)])
async def admin_training_status() -> dict:
    return training_scheduler_status_payload(runtime_training_scheduler())


@app.post("/admin/training/run-now", dependencies=[Depends(require_admin)])
async def admin_training_run_now() -> dict:
    return training_scheduler_status_payload(await runtime_training_scheduler().run_now())


@app.post("/admin/training/start", dependencies=[Depends(require_admin)])
async def admin_training_start() -> dict:
    active_scheduler = runtime_training_scheduler()
    started = active_scheduler.start()
    return {"started": started, **training_scheduler_status_payload(active_scheduler)}


@app.post("/admin/training/stop", dependencies=[Depends(require_admin)])
async def admin_training_stop() -> dict:
    active_scheduler = runtime_training_scheduler()
    stopped = await active_scheduler.stop()
    return {"stopped": stopped, **training_scheduler_status_payload(active_scheduler)}


def scheduler_status_payload(active_scheduler) -> dict:
    if hasattr(active_scheduler, "status"):
        status = dict(active_scheduler.status())
    else:
        status = {
            "running": getattr(active_scheduler, "running", None),
            "paused": getattr(active_scheduler, "paused", False),
            "pause_reason": getattr(active_scheduler, "pause_reason", None),
            "paused_at": getattr(active_scheduler, "paused_at", None),
            "auto_trade_enabled": settings.auto_trade_enabled,
            "trading_enabled": settings.trading_enabled,
            "circuit_breaker_enabled": settings.circuit_breaker_enabled,
            "runtime_error_count_window": None,
            "runtime_error_window_seconds": settings.circuit_breaker_window_seconds,
            "last_successful_run_at": None,
            "last_runtime_error_at": None,
            "last_runtime_error": None,
            "last_stale_data_at": None,
            "last_stale_data_reason": None,
        }
    for key in ["paused_at", "last_successful_run_at", "last_runtime_error_at", "last_stale_data_at"]:
        status[key] = serialize_timestamp(status.get(key))
    return status


def _deep_health_checks() -> dict:
    checks = {
        "database": {"ok": False},
        "model_registry": {"ok": False},
        "market_data_client": {"ok": False},
        "scheduler": {"ok": False},
        "paper_trading_only": {"ok": settings.paper_trading_only is True},
        "symbol": {"ok": settings.symbol == "BTC/USD", "value": settings.symbol},
    }
    try:
        with SessionLocal() as db:
            db.execute(text("SELECT 1"))
        checks["database"] = {"ok": True}
    except Exception as exc:
        checks["database"] = {"ok": False, "error_type": type(exc).__name__}
    try:
        registry = ModelRegistry(settings)
        registry.read()
        if registry.path.exists():
            raw_registry = json.loads(registry.path.read_text(encoding="utf-8"))
            if not isinstance(raw_registry, dict):
                raise ValueError("Model registry must be a JSON object.")
        checks["model_registry"] = {"ok": True, "configured": registry.path.exists()}
    except Exception as exc:
        checks["model_registry"] = {"ok": False, "error_type": type(exc).__name__}
    try:
        market = MarketDataClient(settings)
        configured = market.settings.symbol == "BTC/USD"
        checks["market_data_client"] = {"ok": configured, "symbol": market.settings.symbol}
    except Exception as exc:
        checks["market_data_client"] = {"ok": False, "error_type": type(exc).__name__}
    try:
        status = scheduler_status_payload(runtime_scheduler())
        checks["scheduler"] = {"ok": True, **status}
    except Exception as exc:
        checks["scheduler"] = {"ok": False, "error_type": type(exc).__name__}
    return checks


def _log_application_startup(*, scheduler_started: bool, pause_restored: bool) -> None:
    try:
        get_logger().event(
            "application_startup",
            symbol=settings.symbol,
            paper_trading_only=settings.paper_trading_only,
            trading_enabled=settings.trading_enabled,
            auto_trade_enabled=settings.auto_trade_enabled,
            scheduler_started=scheduler_started,
            scheduler_paused=scheduler.paused,
            scheduler_pause_restored=pause_restored,
            scheduler_pause_reason=scheduler.pause_reason,
        )
    except Exception:
        pass


def training_scheduler_status_payload(value) -> dict:
    status = dict(value if isinstance(value, dict) else value.status())
    for key in ["last_training_started_at", "last_training_finished_at"]:
        status[key] = serialize_timestamp(status.get(key))
    return status


@app.post("/alerts/discord/test", dependencies=[Depends(require_admin)])
async def test_discord_alert() -> dict:
    notifier = DiscordNotifier(settings)
    if not notifier.enabled:
        return {"sent": False, "reason": "discord_disabled"}

    await notifier.send_embed(
        title="Discord Test Alert",
        fields=[
            {"name": "App", "value": "btc-ml-paper-trader", "inline": True},
            {"name": "Environment", "value": settings.app_env, "inline": True},
            {"name": "Symbol", "value": settings.symbol, "inline": True},
            {"name": "Paper Trading Only", "value": str(settings.paper_trading_only), "inline": True},
            {"name": "Timestamp", "value": iso_utc_now(), "inline": False},
        ],
        color=0xF7931A,
        footer="BTC/USD paper trading dashboard",
    )
    return {"sent": True}


@app.post("/train", dependencies=[Depends(require_admin)])
async def train() -> dict:
    bars = await MarketDataClient(settings).fetch_bars(settings.symbol, limit=max(settings.lookback_bars, settings.min_training_rows + 200))
    starting_equity = None
    try:
        account_state = account_state_from_payload(await trader.broker.get_account())
        starting_equity = account_state.equity or account_state.portfolio_value
    except Exception:
        starting_equity = None
    return train_model_from_bars(bars, settings, starting_equity=starting_equity)


@app.post("/backtest", dependencies=[Depends(require_admin)])
async def backtest() -> dict:
    from scripts.backtest import run_backtest

    return await run_backtest()
