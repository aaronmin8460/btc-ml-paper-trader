from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import Settings
from app.db.database import Base


def test_health_is_public(monkeypatch):
    from app import main

    monkeypatch.setattr(main, "settings", Settings(_env_file=None, api_admin_token="secret"))

    response = TestClient(main.app).get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["paper_trading_only"] is True
    assert response.json()["symbol"] == "BTC/USD"


def test_health_deep_returns_safe_runtime_checks_without_secrets(monkeypatch, tmp_path):
    from app import main

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)

    class FakeScheduler:
        def status(self):
            return {
                "running": False,
                "paused": True,
                "pause_reason": "manual_pause",
                "paused_at": None,
                "auto_trade_enabled": False,
                "trading_enabled": False,
                "circuit_breaker_enabled": True,
                "last_successful_run_at": None,
                "last_runtime_error_at": None,
                "last_runtime_error": None,
                "last_stale_data_at": None,
                "last_stale_data_reason": None,
            }

    settings = Settings(
        _env_file=None,
        api_admin_token="admin-secret",
        alpaca_api_key="paper-key",
        alpaca_secret_key="paper-secret",
        discord_webhook_url="https://discord.example/private-hook",
        model_dir=str(tmp_path / "models"),
    )
    monkeypatch.setattr(main, "settings", settings)
    monkeypatch.setattr(main, "SessionLocal", Session)
    monkeypatch.setattr(main.app.state, "scheduler", FakeScheduler(), raising=False)

    try:
        response = TestClient(main.app).get("/health/deep")
    finally:
        engine.dispose()

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["paper_trading_only"] is True
    assert body["symbol"] == "BTC/USD"
    assert body["checks"]["database"]["ok"] is True
    assert body["checks"]["model_registry"]["ok"] is True
    assert body["checks"]["market_data_client"]["ok"] is True
    assert body["checks"]["scheduler"]["pause_reason"] == "manual_pause"
    assert "admin-secret" not in response.text
    assert "paper-key" not in response.text
    assert "paper-secret" not in response.text
    assert "private-hook" not in response.text


def test_health_deep_reports_corrupt_model_registry(monkeypatch, tmp_path):
    from app import main

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "registry.json").write_text("not-json", encoding="utf-8")
    settings = Settings(_env_file=None, model_dir=str(model_dir))
    monkeypatch.setattr(main, "settings", settings)
    monkeypatch.setattr(main, "SessionLocal", Session)

    try:
        response = TestClient(main.app).get("/health/deep")
    finally:
        engine.dispose()

    assert response.status_code == 503
    assert response.json()["status"] == "degraded"
    assert response.json()["checks"]["model_registry"]["ok"] is False
    assert response.json()["checks"]["model_registry"]["error_type"] == "JSONDecodeError"


def test_runtime_read_endpoints_require_admin_token(monkeypatch):
    from app import main

    monkeypatch.setattr(main, "settings", Settings(_env_file=None, api_admin_token="secret"))

    client = TestClient(main.app)

    for path in [
        "/config/safe",
        "/position",
        "/signals/latest",
        "/orders",
        "/debug/latest-bars",
        "/admin/status",
        "/scheduler/status",
        "/admin/training/status",
    ]:
        response = client.get(path)
        assert response.status_code == 401, path

    for path in ["/admin/pause", "/admin/resume", "/admin/training/run-now", "/admin/training/start", "/admin/training/stop"]:
        response = client.post(path)
        assert response.status_code == 401, path


def test_scheduler_status_returns_safe_state(monkeypatch):
    from app import main

    class FakeScheduler:
        def status(self):
            return {
                "running": True,
                "paused": True,
                "pause_reason": "scalping_kill_switch:runtime_errors",
                "paused_at": None,
                "auto_trade_enabled": True,
                "trading_enabled": True,
                "circuit_breaker_enabled": True,
                "runtime_error_count_window": 3,
                "runtime_error_window_seconds": 900,
                "last_successful_run_at": None,
                "last_runtime_error_at": None,
                "last_runtime_error": "RuntimeError",
                "last_stale_data_at": None,
                "last_stale_data_reason": None,
            }

    monkeypatch.setattr(main, "settings", Settings(_env_file=None, api_admin_token="secret"))
    monkeypatch.setattr(main.app.state, "scheduler", FakeScheduler(), raising=False)

    response = TestClient(main.app).get("/scheduler/status", headers={"X-Admin-Token": "secret"})

    assert response.status_code == 200
    assert response.json()["paused"] is True
    assert response.json()["pause_reason"] == "scalping_kill_switch:runtime_errors"
    assert response.json()["last_runtime_error"] == "RuntimeError"
