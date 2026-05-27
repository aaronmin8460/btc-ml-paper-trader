from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import configure_frontend_static


def test_frontend_static_missing_build_does_not_break_api(tmp_path):
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    mounted = configure_frontend_static(app, tmp_path / "missing-dist")
    client = TestClient(app)

    assert mounted is False
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/dashboard-ui").status_code == 404


def test_frontend_static_serves_index_assets_and_spa_fallback(tmp_path):
    dist = tmp_path / "dist"
    assets = dist / "assets"
    assets.mkdir(parents=True)
    (dist / "index.html").write_text(
        '<html><body><div id="root"></div><script src="/dashboard-ui/assets/app.js"></script></body></html>',
        encoding="utf-8",
    )
    (assets / "app.js").write_text("console.log('dashboard');", encoding="utf-8")

    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    mounted = configure_frontend_static(app, dist)
    client = TestClient(app)

    assert mounted is True
    assert client.get("/health").status_code == 200
    assert "root" in client.get("/dashboard-ui").text
    assert "dashboard" in client.get("/dashboard-ui/assets/app.js").text
    assert "root" in client.get("/dashboard-ui/client/route").text
