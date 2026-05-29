from fastapi.testclient import TestClient

from app.config import Settings


def test_health_is_public(monkeypatch):
    from app import main

    monkeypatch.setattr(main, "settings", Settings(_env_file=None, api_admin_token="secret"))

    response = TestClient(main.app).get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_runtime_read_endpoints_require_admin_token(monkeypatch):
    from app import main

    monkeypatch.setattr(main, "settings", Settings(_env_file=None, api_admin_token="secret"))

    client = TestClient(main.app)

    for path in ["/config/safe", "/position", "/signals/latest", "/orders", "/debug/latest-bars", "/admin/status"]:
        response = client.get(path)
        assert response.status_code == 401, path

    for path in ["/admin/pause", "/admin/resume"]:
        response = client.post(path)
        assert response.status_code == 401, path
