from fastapi import Depends, FastAPI, Header, HTTPException

from app.config import Settings, get_settings
from app.data.market_data import MarketDataClient
from app.db.database import SessionLocal, init_db
from app.db.repository import Repository
from app.ml.train import train_model_from_bars
from app.notifications.discord import DiscordNotifier
from app.services.scheduler import TradingScheduler
from app.services.trader import Trader
from app.utils.time import iso_utc_now

app = FastAPI(title="btc-ml-paper-trader", version="0.1.0")
settings = get_settings()
trader = Trader(settings)
scheduler = TradingScheduler(trader, settings)


def serialize_model(row) -> dict:
    return {key: value for key, value in row.__dict__.items() if not key.startswith("_")}


def require_admin(x_admin_token: str | None = Header(default=None)) -> None:
    token = settings.api_admin_token
    if not token or x_admin_token != token:
        raise HTTPException(status_code=401, detail="Admin token required")


@app.on_event("startup")
async def startup() -> None:
    init_db()
    if settings.auto_trade_enabled:
        scheduler.start()


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "paper_trading_only": settings.paper_trading_only, "symbol": settings.symbol}


@app.get("/config/safe")
async def safe_config() -> dict:
    return settings.safe_dict()


@app.get("/position")
async def position() -> dict:
    state = await trader.get_position_state()
    return state.__dict__


@app.get("/signals/latest")
async def latest_signal() -> dict:
    with SessionLocal() as db:
        signal = Repository(db).latest_signal()
        return serialize_model(signal) if signal else {}


@app.get("/orders")
async def orders() -> list[dict]:
    with SessionLocal() as db:
        rows = Repository(db).recent_orders()
        return [serialize_model(row) for row in rows]


@app.post("/run-once", dependencies=[Depends(require_admin)])
async def run_once() -> dict:
    return await trader.run_once()


@app.post("/auto/start", dependencies=[Depends(require_admin)])
async def auto_start() -> dict:
    started = scheduler.start()
    return {"started": started, "running": scheduler.running}


@app.post("/auto/stop", dependencies=[Depends(require_admin)])
async def auto_stop() -> dict:
    stopped = await scheduler.stop()
    return {"stopped": stopped, "running": scheduler.running}


@app.post("/alerts/discord/test", dependencies=[Depends(require_admin)])
async def test_discord_alert() -> dict:
    notifier = DiscordNotifier(settings)
    if not notifier.enabled:
        return {"sent": False, "reason": "discord_disabled"}

    message = "\n".join(
        [
            "Discord test alert",
            "App: btc-ml-paper-trader",
            f"Environment: {settings.app_env}",
            f"Symbol: {settings.symbol}",
            f"Paper trading only: {settings.paper_trading_only}",
            f"Timestamp: {iso_utc_now()}",
        ]
    )
    await notifier.send(message)
    return {"sent": True}


@app.post("/train", dependencies=[Depends(require_admin)])
async def train() -> dict:
    bars = await MarketDataClient(settings).fetch_bars(settings.symbol, limit=max(settings.lookback_bars, settings.min_training_rows + 200))
    return train_model_from_bars(bars, settings)


@app.post("/backtest", dependencies=[Depends(require_admin)])
async def backtest() -> dict:
    from scripts.backtest import run_backtest

    return await run_backtest()
