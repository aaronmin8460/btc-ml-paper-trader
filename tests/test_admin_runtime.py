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
