import pandas as pd
from fastapi.testclient import TestClient

from app.config import Settings


class FakeMarketDataClient:
    def __init__(self, settings) -> None:
        self.settings = settings

    async def fetch_bars(self, symbol):
        assert symbol == "BTC/USD"
        return pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2026-05-27T11:58:00Z", "2026-05-27T11:59:00Z"],
                    utc=True,
                ),
                "open": [100.0, 101.0],
                "high": [101.0, 102.0],
                "low": [99.0, 100.0],
                "close": [100.5, 101.5],
                "volume": [1.0, 2.0],
            }
        )


def test_debug_latest_bars_requires_admin_token(monkeypatch):
    from app import main

    monkeypatch.setattr(main, "settings", Settings(_env_file=None, api_admin_token="secret"))

    response = TestClient(main.app).get("/debug/latest-bars")

    assert response.status_code == 401


def test_debug_latest_bars_returns_safe_bar_metadata(monkeypatch):
    from app import main

    monkeypatch.setattr(
        main,
        "settings",
        Settings(_env_file=None, api_admin_token="secret", timeframe="1Min"),
    )
    monkeypatch.setattr(main, "MarketDataClient", FakeMarketDataClient)

    response = TestClient(main.app).get("/debug/latest-bars", headers={"X-Admin-Token": "secret"})

    assert response.status_code == 200
    body = response.json()
    assert body["symbol"] == "BTC/USD"
    assert body["timeframe"] == "1Min"
    assert body["count"] == 2
    assert body["first_timestamp"] == "2026-05-27T11:58:00+00:00"
    assert body["latest_timestamp"] == "2026-05-27T11:59:00+00:00"
    assert body["latest_close"] == 101.5
    assert "current_utc_time" in body
    assert "secret" not in response.text
