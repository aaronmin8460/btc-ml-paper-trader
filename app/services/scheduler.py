import asyncio

from app.config import Settings, get_settings
from app.monitoring.logger import get_logger


class TradingScheduler:
    def __init__(self, trader, settings: Settings | None = None) -> None:
        self.trader = trader
        self.settings = settings or get_settings()
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self.logger = get_logger()

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> bool:
        if self.running:
            return False
        self._stop.clear()
        self._task = asyncio.create_task(self._loop())
        return True

    async def stop(self) -> bool:
        if not self.running:
            return False
        self._stop.set()
        await self._task
        return True

    async def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                if self.settings.auto_trade_enabled:
                    await self.trader.run_once()
            except Exception as exc:
                self.logger.event("runtime_error", component="scheduler", error=str(exc))
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.settings.scan_interval_seconds)
            except asyncio.TimeoutError:
                continue
