from datetime import UTC, datetime

from fastapi.testclient import TestClient

from app.config import Settings


class FakeScheduler:
    def __init__(self) -> None:
        self.running = True
        self.paused = False
        self.pause_reason = None
        self.paused_at = None

    async def pause(self, reason: str = "manual_pause", *, send_alert: bool = False) -> bool:
        if self.paused:
            return False
        self.paused = True
        self.pause_reason = reason
        self.paused_at = datetime(2026, 5, 29, tzinfo=UTC)
        return True

    def resume(self) -> bool:
        was_paused = self.paused
        self.paused = False
        self.pause_reason = None
        self.paused_at = None
        return was_paused

    def status(self) -> dict:
        return {
            "running": self.running,
            "paused": self.paused,
            "pause_reason": self.pause_reason,
            "paused_at": self.paused_at,
            "auto_trade_enabled": True,
            "trading_enabled": True,
            "circuit_breaker_enabled": True,
        }


class FakeTrainingScheduler:
    def __init__(self) -> None:
        self.running = False
        self.start_calls = 0
        self.stop_calls = 0

    def start(self) -> bool:
        self.start_calls += 1
        if self.running:
            return False
        self.running = True
        return True

    async def stop(self) -> bool:
        self.stop_calls += 1
        if not self.running:
            return False
        self.running = False
        return True

    def status(self) -> dict:
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


def test_admin_pause_resume_status(monkeypatch):
    from app import main

    settings = Settings(_env_file=None, api_admin_token="secret")
    scheduler = FakeScheduler()
    monkeypatch.setattr(main, "settings", settings)
    monkeypatch.setattr(main.app.state, "scheduler", scheduler, raising=False)

    client = TestClient(main.app)

    status = client.get("/admin/status", headers={"X-Admin-Token": "secret"})
    assert status.status_code == 200
    assert status.json()["paused"] is False

    paused = client.post("/admin/pause", headers={"X-Admin-Token": "secret"})
    assert paused.status_code == 200
    assert paused.json()["changed"] is True
    assert paused.json()["paused"] is True
    assert paused.json()["pause_reason"] == "manual_pause"
    assert paused.json()["paused_at"] == "2026-05-29T00:00:00+00:00"

    resumed = client.post("/admin/resume", headers={"X-Admin-Token": "secret"})
    assert resumed.status_code == 200
    assert resumed.json()["changed"] is True
    assert resumed.json()["paused"] is False
    assert resumed.json()["pause_reason"] is None
    assert resumed.json()["paused_at"] is None


def test_admin_training_start_stop_status(monkeypatch):
    from app import main

    settings = Settings(_env_file=None, api_admin_token="secret")
    training_scheduler = FakeTrainingScheduler()
    monkeypatch.setattr(main, "settings", settings)
    monkeypatch.setattr(main.app.state, "training_scheduler", training_scheduler, raising=False)

    client = TestClient(main.app)

    status = client.get("/admin/training/status", headers={"X-Admin-Token": "secret"})
    assert status.status_code == 200
    assert status.json()["running"] is False
    assert status.json()["last_training_started_at"] == "2026-05-29T00:00:00+00:00"

    started = client.post("/admin/training/start", headers={"X-Admin-Token": "secret"})
    assert started.status_code == 200
    assert started.json()["started"] is True
    assert started.json()["running"] is True
    assert training_scheduler.start_calls == 1

    stopped = client.post("/admin/training/stop", headers={"X-Admin-Token": "secret"})
    assert stopped.status_code == 200
    assert stopped.json()["stopped"] is True
    assert stopped.json()["running"] is False
    assert training_scheduler.stop_calls == 1
