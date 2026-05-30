import json
import threading
import time
from datetime import UTC, datetime

import pytest

from app.config import Settings
from app.data.market_data import MarketDataClient
from app.ml.registry import ModelRegistry
from app.services.training_scheduler import TrainingScheduler


class FakeMarketData:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int | None]] = []

    async def fetch_bars(self, symbol: str, *, limit: int | None = None):
        self.calls.append((symbol, limit))
        return MarketDataClient.synthetic_btc_bars(limit or 140)


class FakeBroker:
    def __init__(self, equity: float | None = 1234.0) -> None:
        self.equity = equity

    def credentials_available(self) -> bool:
        return True

    async def get_account(self) -> dict:
        return {"equity": str(self.equity), "portfolio_value": str(self.equity), "paper": True}


class FakeNotifier:
    def __init__(self) -> None:
        self.model_alerts = []
        self.error_alerts = []

    async def model_alert(self, model_path, accepted, reason, metrics=None, *, force=False):
        self.model_alerts.append(
            {
                "model_path": model_path,
                "accepted": accepted,
                "reason": reason,
                "metrics": metrics,
                "force": force,
            }
        )

    async def error_alert(self, where, error, *, force=False):
        self.error_alerts.append({"where": where, "error": str(error), "force": force})


def _settings(tmp_path, **overrides) -> Settings:
    defaults = {
        "_env_file": None,
        "model_dir": str(tmp_path),
        "auto_train_min_bars": 120,
        "auto_train_interval_seconds": 60,
        "auto_train_startup_delay_seconds": 0,
        "min_training_rows": 50,
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _metrics(**overrides) -> dict:
    metrics = {
        "validation_rows": 200,
        "precision": 0.8,
        "profit_factor": 1.5,
        "max_drawdown": 0.005,
        "number_of_trades": 35,
        "net_return_pct": 0.01,
        "profit_factor_net": 1.2,
        "promotion_reason": "accepted",
    }
    metrics.update(overrides)
    return metrics


@pytest.mark.anyio
async def test_startup_starts_training_scheduler_when_enabled(monkeypatch, tmp_path):
    from app import main

    class FakeTrainingScheduler:
        starts = 0

        def start(self):
            self.starts += 1
            return True

    training = FakeTrainingScheduler()
    monkeypatch.setattr(main, "init_db", lambda: None)
    monkeypatch.setattr(main, "settings", _settings(tmp_path, auto_train_enabled=True))
    monkeypatch.setattr(main, "training_scheduler", training)

    await main.startup()

    assert training.starts == 1


@pytest.mark.anyio
async def test_startup_does_not_start_training_scheduler_when_disabled(monkeypatch, tmp_path):
    from app import main

    class FakeTrainingScheduler:
        starts = 0

        def start(self):
            self.starts += 1
            return True

    training = FakeTrainingScheduler()
    monkeypatch.setattr(main, "init_db", lambda: None)
    monkeypatch.setattr(main, "settings", _settings(tmp_path, auto_train_enabled=False))
    monkeypatch.setattr(main, "training_scheduler", training)

    await main.startup()

    assert training.starts == 0


def test_admin_training_run_now_endpoint_triggers_scheduler(monkeypatch):
    from app import main
    from fastapi.testclient import TestClient

    class FakeTrainingScheduler:
        def __init__(self) -> None:
            self.run_now_calls = 0
            self.running = False

        def status(self):
            return {
                "auto_train_enabled": True,
                "running": self.running,
                "last_training_started_at": datetime(2026, 5, 29, tzinfo=UTC),
                "last_training_finished_at": None,
                "last_training_status": "idle",
                "last_training_reason": None,
                "last_training_model_path": None,
                "last_training_accepted": None,
                "last_training_metrics": None,
            }

        async def run_now(self):
            self.run_now_calls += 1
            payload = self.status()
            payload["last_training_status"] = "accepted"
            return payload

    scheduler = FakeTrainingScheduler()
    monkeypatch.setattr(main, "settings", Settings(_env_file=None, api_admin_token="secret"))
    monkeypatch.setattr(main.app.state, "training_scheduler", scheduler, raising=False)

    response = TestClient(main.app).post("/admin/training/run-now", headers={"X-Admin-Token": "secret"})

    assert response.status_code == 200
    assert scheduler.run_now_calls == 1
    assert response.json()["last_training_status"] == "accepted"
    assert response.json()["last_training_started_at"] == "2026-05-29T00:00:00+00:00"


@pytest.mark.anyio
async def test_rejected_training_does_not_create_active_model(tmp_path):
    settings = _settings(tmp_path)
    market = FakeMarketData()
    notifier = FakeNotifier()

    def rejected_trainer(bars, settings, *, starting_equity=None):
        assert starting_equity == 1234.0
        return {
            "model_path": str(tmp_path / "rejected.joblib"),
            "accepted": False,
            "reason": "model_not_profitable_after_costs",
            "metrics": _metrics(net_return_pct=-0.01, promotion_reason="model_not_profitable_after_costs"),
            "registry": None,
        }

    scheduler = TrainingScheduler(
        settings,
        market_data=market,
        broker=FakeBroker(),
        notifier=notifier,
        trainer=rejected_trainer,
    )

    result = await scheduler.run_now()

    assert result["last_training_status"] == "rejected"
    assert result["last_training_accepted"] is False
    assert not (tmp_path / "registry.json").exists()
    assert notifier.model_alerts[0]["accepted"] is False
    assert notifier.model_alerts[0]["force"] is True


@pytest.mark.anyio
async def test_accepted_training_updates_active_model(tmp_path):
    settings = _settings(tmp_path)
    accepted_path = str(tmp_path / "accepted.joblib")

    def accepted_trainer(bars, settings, *, starting_equity=None):
        registry = ModelRegistry(settings).promote(
            model_path=accepted_path,
            feature_columns=["log_return_1"],
            metrics=_metrics(),
            thresholds={},
            training_start="2026-05-29T00:00:00+00:00",
            training_end="2026-05-29T00:01:00+00:00",
        )
        return {
            "model_path": accepted_path,
            "accepted": True,
            "reason": "accepted",
            "metrics": _metrics(),
            "registry": registry,
        }

    scheduler = TrainingScheduler(
        settings,
        market_data=FakeMarketData(),
        notifier=FakeNotifier(),
        trainer=accepted_trainer,
    )

    result = await scheduler.run_now()
    registry = json.loads((tmp_path / "registry.json").read_text(encoding="utf-8"))

    assert result["last_training_status"] == "accepted"
    assert result["last_training_accepted"] is True
    assert registry["active_model_path"] == accepted_path


@pytest.mark.anyio
async def test_concurrent_training_is_blocked(tmp_path):
    settings = _settings(tmp_path)
    started = threading.Event()
    release = threading.Event()

    def slow_trainer(bars, settings, *, starting_equity=None):
        started.set()
        release.wait(timeout=2)
        return {
            "model_path": str(tmp_path / "accepted.joblib"),
            "accepted": True,
            "reason": "accepted",
            "metrics": _metrics(),
            "registry": None,
        }

    scheduler = TrainingScheduler(
        settings,
        market_data=FakeMarketData(),
        notifier=FakeNotifier(),
        trainer=slow_trainer,
    )

    first = asyncio_create_task(scheduler.run_now())
    assert await wait_for_thread_event(started)

    busy = await scheduler.run_now()
    release.set()
    completed = await first

    assert busy["started"] is False
    assert busy["last_training_status"] == "busy"
    assert busy["last_training_reason"] == "training_already_running"
    assert completed["last_training_status"] == "accepted"


async def wait_for_thread_event(event: threading.Event) -> bool:
    import asyncio

    return await asyncio.to_thread(event.wait, 1)


def asyncio_create_task(coro):
    import asyncio

    return asyncio.create_task(coro)
