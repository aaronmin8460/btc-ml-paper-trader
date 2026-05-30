from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import Settings
from app.db.database import Base
from app.db.models import AccountSnapshot, Order, Signal, Trade
from app.risk.risk_manager import PositionState


class FakeMarketDataClient:
    def __init__(self, settings) -> None:
        self.settings = settings

    async def fetch_bars(self, symbol):
        assert symbol == "BTC/USD"
        return pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2026-05-27T16:00:00Z", "2026-05-27T16:01:00Z"], utc=True),
                "open": [100.0, 101.0],
                "high": [101.0, 102.0],
                "low": [99.0, 100.0],
                "close": [100.5, 101.5],
                "volume": [1.0, 2.0],
            }
        )

    async def fetch_latest_quote(self, symbol):
        assert symbol == "BTC/USD"
        return {"bid_price": 101.0, "ask_price": 102.0, "bid_size": 3.0, "ask_size": 1.0}

    def bars_cache_age_seconds(self, *, symbol):
        assert symbol == "BTC/USD"
        return 2.5


class FakeBroker:
    def credentials_available(self):
        return True

    async def get_account(self):
        return {
            "status": "ACTIVE",
            "currency": "USD",
            "buying_power": "1000",
            "cash": "900",
            "equity": "1000",
            "portfolio_value": "1000",
            "last_equity": "990",
            "paper": True,
            "secret_key": "must-not-leak",
        }


class NoCredentialsBroker:
    def credentials_available(self):
        return False

    async def get_account(self):
        raise AssertionError("get_account should not be called without credentials")


class FakeTrader:
    def __init__(self) -> None:
        self.broker = FakeBroker()

    async def get_position_state(self):
        return PositionState()

    async def run_once(self):
        return {
            "prediction": {
                "buy_probability": 0.51,
                "sell_probability": 0.49,
                "features": {"close": 101.5},
            },
            "decision": {"action": "hold", "reason": "dashboard_test"},
            "order": None,
        }


class NoCredentialsTrader(FakeTrader):
    def __init__(self) -> None:
        self.broker = NoCredentialsBroker()


class FakeScheduler:
    running = True
    paused = False
    pause_reason = None
    paused_at = None


class FakeTrainingScheduler:
    running = False

    def status(self):
        return {
            "auto_train_enabled": False,
            "running": self.running,
            "last_training_started_at": datetime(2026, 5, 29, tzinfo=UTC),
            "last_training_finished_at": datetime(2026, 5, 29, 0, 1, tzinfo=UTC),
            "last_training_status": "rejected",
            "last_training_reason": "model_not_profitable_after_costs",
            "last_training_model_path": "models/rejected.joblib",
            "last_training_accepted": False,
            "last_training_metrics": {"net_return_pct": -0.01},
        }


class FailingMarketDataClient:
    def __init__(self, settings) -> None:
        self.settings = settings

    async def fetch_bars(self, symbol):
        raise RuntimeError("market data unavailable")

    async def fetch_latest_quote(self, symbol):
        raise RuntimeError("quote unavailable")


@pytest.fixture
def dashboard_client(monkeypatch, tmp_path):
    from app import main
    from app.api import dashboard

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)

    settings = Settings(
        _env_file=None,
        api_admin_token="secret",
        alpaca_api_key="paper-key",
        alpaca_secret_key="paper-secret",
        discord_webhook_url="https://discord.example/webhook",
        trading_enabled=False,
        auto_trade_enabled=False,
        lookback_bars=2,
        timeframe="1Min",
        model_dir=str(tmp_path / "models"),
    )
    monkeypatch.setattr(main, "settings", settings)
    monkeypatch.setattr(main.app.state, "settings", settings, raising=False)
    monkeypatch.setattr(main.app.state, "trader", FakeTrader(), raising=False)
    monkeypatch.setattr(main.app.state, "scheduler", FakeScheduler(), raising=False)
    monkeypatch.setattr(main.app.state, "training_scheduler", FakeTrainingScheduler(), raising=False)
    monkeypatch.setattr(dashboard, "SessionLocal", Session)
    monkeypatch.setattr(dashboard, "MarketDataClient", FakeMarketDataClient)

    try:
        yield TestClient(main.app), Session
    finally:
        engine.dispose()


def test_dashboard_endpoints_require_admin_token(dashboard_client):
    client, _ = dashboard_client

    for path in [
        "/dashboard/summary",
        "/dashboard/signals",
        "/dashboard/orders",
        "/dashboard/trades",
        "/dashboard/equity-curve",
        "/dashboard/account-snapshots",
        "/dashboard/portfolio-curve",
        "/dashboard/trading-status",
        "/dashboard/market",
    ]:
        response = client.get(path)
        assert response.status_code == 401, path

    response = client.post("/dashboard/run-once")
    assert response.status_code == 401


def test_dashboard_summary_returns_expected_structure_and_nulls(dashboard_client):
    client, _ = dashboard_client

    response = client.get("/dashboard/summary", headers={"X-Admin-Token": "secret"})

    assert response.status_code == 200
    body = response.json()
    assert body["app_status"] == "ok"
    assert body["symbol"] == "BTC/USD"
    assert body["paper_trading_only"] is True
    assert body["scheduler_running"] is True
    assert body["auto_train_enabled"] is False
    assert body["training_scheduler_running"] is False
    assert body["last_training_started_at"] == "2026-05-29T00:00:00+00:00"
    assert body["last_training_finished_at"] == "2026-05-29T00:01:00+00:00"
    assert body["last_training_status"] == "rejected"
    assert body["last_training_reason"] == "model_not_profitable_after_costs"
    assert body["last_training_model_path"] == "models/rejected.joblib"
    assert body["last_training_accepted"] is False
    assert body["last_training_metrics"] == {"net_return_pct": -0.01}
    assert body["latest_btc_price"] == 101.5
    assert body["profit_guard_enabled"] is True
    assert body["min_net_exit_profit_pct"] == 0.002
    assert body["current_unrealized_pnl_pct"] is None
    assert body["profit_guard_exit_allowed"] is False
    assert body["estimated_exit_price"] == pytest.approx(100.98)
    assert body["minimum_profitable_exit_price"] is None
    assert body["total_orders"] == 0
    assert body["total_trades"] == 0
    assert body["closed_trades"] == 0
    assert body["total_return_pct"] is None
    assert body["win_rate"] is None
    assert body["average_trade_pnl"] is None
    assert body["max_drawdown"] is None
    assert body["account_equity"] == 1000
    assert body["buying_power"] == 1000
    assert body["account_daily_change_usd"] == 10
    assert body["account_daily_change_pct"] == pytest.approx(10 / 990)
    assert body["account_drawdown_pct"] == 0
    assert body["alpaca_calls_last_minute"] is not None
    assert isinstance(body["alpaca_endpoint_counts"], dict)
    assert body["api_budget_status"] in {"ok", "soft_limit", "hard_stop", "disabled"}
    assert body["latest_model_net_return_pct"] is None
    assert body["latest_model_accepted"] is None
    assert body["active_model_status"] == "stale"
    assert body["active_model_valid"] is False
    assert body["active_model_invalid_reason"] == "no_active_model"
    assert body["active_model_reason"] == "no_active_model"
    assert body["active_model_version"] is None
    assert body["active_model_profit_factor_net"] is None
    assert body["active_model_number_of_trades"] is None
    assert body["registry_metadata_matches_joblib"] is False
    assert body["active_model_registry_mismatched"] is False
    assert body["latest_signal"] is None
    assert body["last_order"] is None
    assert body["last_trade"] is None
    assert body["ioc_cancel_guard"]["latest_buy_ioc_cancel_at"] is None
    assert body["ioc_cancel_guard"]["recent_buy_ioc_cancel_count"] == 0
    assert body["ioc_cancel_guard"]["cooldown_active"] is False
    assert body["data_freshness"]["latest_timestamp"] == "2026-05-27T16:01:00+00:00"


def test_dashboard_trading_status_returns_disabled_empty_state(dashboard_client):
    client, _ = dashboard_client

    response = client.get("/dashboard/trading-status", headers={"X-Admin-Token": "secret"})

    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "disabled"
    assert body["state_tone"] == "gray"
    assert body["paused"] is False
    assert body["pause_reason"] is None
    assert body["paused_at"] is None
    assert body["latest_decision_action"] is None
    assert body["latest_decision_reason"] is None
    assert body["latest_risk_block_reason"] is None
    assert body["current_ioc_cancel_count"] == 0
    assert body["ioc_cancel_lookback_seconds"] == 300
    assert body["ioc_cooldown_active"] is False
    assert body["ioc_cooldown_expires_at"] is None
    assert body["scheduler_running"] is True
    assert body["auto_trade_enabled"] is False
    assert body["trading_enabled"] is False
    assert body["model_available"] is False
    assert body["prediction_source"] == "fallback"
    assert body["active_model_status"] == "stale"
    assert body["active_model_valid"] is False
    assert body["active_model_invalid_reason"] == "no_active_model"
    assert body["active_model_reason"] == "no_active_model"
    assert body["registry_metadata_matches_joblib"] is False
    assert body["fallback_trading_allowed"] is False


def test_dashboard_trading_status_shows_risk_block_and_ioc_cooldown(dashboard_client, monkeypatch):
    from app import main

    client, Session = dashboard_client
    settings = Settings(
        _env_file=None,
        api_admin_token="secret",
        trading_enabled=True,
        auto_trade_enabled=True,
        ioc_cancel_lookback_seconds=300,
        ioc_cancel_cooldown_seconds=120,
        model_dir=main.settings.model_dir,
    )
    monkeypatch.setattr(main, "settings", settings)
    monkeypatch.setattr(main.app.state, "settings", settings, raising=False)
    with Session() as db:
        db.add_all(
            [
                Signal(
                    symbol="BTC/USD",
                    action="hold",
                    buy_probability=0.9,
                    sell_probability=0.1,
                    reason="ioc_cancel_cooldown_active",
                    created_at=datetime.now(UTC) - timedelta(seconds=5),
                ),
                Order(
                    symbol="BTC/USD",
                    side="buy",
                    status="canceled",
                    raw_response='{"order_type":"limit","time_in_force":"ioc"}',
                    created_at=datetime.now(UTC) - timedelta(seconds=20),
                ),
            ]
        )
        db.commit()

    response = client.get("/dashboard/trading-status", headers={"X-Admin-Token": "secret"})

    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "cooling_down"
    assert body["state_tone"] == "yellow"
    assert body["paused"] is False
    assert body["latest_decision_action"] == "hold"
    assert body["latest_decision_reason"] == "ioc_cancel_cooldown_active"
    assert body["latest_risk_block_reason"] is None
    assert body["current_ioc_cancel_count"] == 1
    assert body["ioc_cooldown_active"] is True
    assert body["ioc_cooldown_expires_at"] is not None
    assert body["scheduler_running"] is True
    assert body["auto_trade_enabled"] is True
    assert body["trading_enabled"] is True


def test_dashboard_trading_status_shows_runtime_pause(dashboard_client, monkeypatch):
    from app import main

    client, _ = dashboard_client
    settings = Settings(
        _env_file=None,
        api_admin_token="secret",
        trading_enabled=True,
        auto_trade_enabled=True,
    )

    class PausedScheduler:
        running = True
        paused = True
        pause_reason = "repeated_risk_block:trade_cooldown_active"
        paused_at = datetime(2026, 5, 29, tzinfo=UTC)

    monkeypatch.setattr(main, "settings", settings)
    monkeypatch.setattr(main.app.state, "settings", settings, raising=False)
    monkeypatch.setattr(main.app.state, "scheduler", PausedScheduler(), raising=False)

    response = client.get("/dashboard/trading-status", headers={"X-Admin-Token": "secret"})

    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "paused"
    assert body["state_tone"] == "red"
    assert body["paused"] is True
    assert body["pause_reason"] == "repeated_risk_block:trade_cooldown_active"
    assert body["paused_at"] == "2026-05-29T00:00:00+00:00"


def test_dashboard_summary_shows_active_ioc_cancel_cooldown(dashboard_client):
    client, Session = dashboard_client
    with Session() as db:
        db.add(
            Order(
                symbol="BTC/USD",
                side="buy",
                status="canceled",
                raw_response='{"order_type":"limit","time_in_force":"ioc"}',
                created_at=datetime.now(UTC) - timedelta(seconds=10),
            )
        )
        db.commit()

    response = client.get("/dashboard/summary", headers={"X-Admin-Token": "secret"})

    assert response.status_code == 200
    guard = response.json()["ioc_cancel_guard"]
    assert guard["latest_ioc_cancel_at"] is not None
    assert guard["latest_buy_ioc_cancel_at"] is not None
    assert guard["recent_buy_ioc_cancel_count"] == 1
    assert guard["cooldown_active"] is True
    assert guard["cooldown_seconds_remaining"] > 0


def test_dashboard_endpoints_do_not_expose_secrets(dashboard_client):
    client, Session = dashboard_client
    with Session() as db:
        db.add(
            Order(
                symbol="BTC/USD",
                side="buy",
                status="filled",
                notional=25,
                raw_response='{"api_key":"raw-key","nested":{"discord_webhook_url":"raw-hook"}}',
            )
        )
        db.commit()

    response = client.get("/dashboard/orders", headers={"X-Admin-Token": "secret"})

    assert response.status_code == 200
    body_text = response.text
    assert "paper-key" not in body_text
    assert "paper-secret" not in body_text
    assert "https://discord.example/webhook" not in body_text
    assert "raw-key" not in body_text
    assert "raw-hook" not in body_text
    raw_response = response.json()[0]["raw_response"]
    assert raw_response["api_key"] == "***"
    assert raw_response["nested"]["discord_webhook_url"] == "***"

    summary = client.get("/dashboard/summary", headers={"X-Admin-Token": "secret"})
    assert summary.status_code == 200
    assert "paper-key" not in summary.text
    assert "paper-secret" not in summary.text
    assert "https://discord.example/webhook" not in summary.text
    assert "raw-key" not in summary.text
    assert "raw-hook" not in summary.text


def test_dashboard_orders_invalid_raw_response_returns_null(dashboard_client):
    client, Session = dashboard_client
    with Session() as db:
        db.add(Order(symbol="BTC/USD", side="buy", status="filled", raw_response="not valid json"))
        db.commit()

    response = client.get("/dashboard/orders", headers={"X-Admin-Token": "secret"})

    assert response.status_code == 200
    assert response.json()[0]["raw_response"] is None


def test_dashboard_summary_works_without_alpaca_credentials(dashboard_client, monkeypatch):
    from app import main

    client, _ = dashboard_client
    settings = Settings(_env_file=None, api_admin_token="secret", alpaca_api_key="", alpaca_secret_key="")
    monkeypatch.setattr(main, "settings", settings)
    monkeypatch.setattr(main.app.state, "settings", settings, raising=False)
    monkeypatch.setattr(main.app.state, "trader", NoCredentialsTrader(), raising=False)

    response = client.get("/dashboard/summary", headers={"X-Admin-Token": "secret"})

    assert response.status_code == 200
    body = response.json()
    assert body["alpaca_account"] is None
    assert body["symbol"] == "BTC/USD"


def test_dashboard_summary_returns_null_market_fields_when_market_fetch_fails(dashboard_client, monkeypatch):
    from app.api import dashboard

    client, _ = dashboard_client
    monkeypatch.setattr(dashboard, "MarketDataClient", FailingMarketDataClient)

    response = client.get("/dashboard/summary", headers={"X-Admin-Token": "secret"})

    assert response.status_code == 200
    body = response.json()
    assert body["latest_btc_price"] is None
    assert body["data_freshness"]["latest_timestamp"] is None
    assert body["data_freshness"]["latest_bar_age_seconds"] is None


def test_dashboard_equity_curve_returns_empty_list_without_trades(dashboard_client):
    client, _ = dashboard_client

    response = client.get("/dashboard/equity-curve", headers={"X-Admin-Token": "secret"})

    assert response.status_code == 200
    assert response.json() == []


def test_dashboard_account_snapshots_empty_returns_empty_lists(dashboard_client):
    client, _ = dashboard_client

    snapshots = client.get("/dashboard/account-snapshots", headers={"X-Admin-Token": "secret"})
    curve = client.get("/dashboard/portfolio-curve", headers={"X-Admin-Token": "secret"})

    assert snapshots.status_code == 200
    assert snapshots.json() == []
    assert curve.status_code == 200
    assert curve.json() == []


def test_dashboard_account_snapshots_redact_secrets_and_build_portfolio_curve(dashboard_client):
    client, Session = dashboard_client
    now = datetime(2026, 5, 27, 16, 0, tzinfo=UTC)
    with Session() as db:
        db.add_all(
            [
                AccountSnapshot(
                    created_at=now,
                    equity=1000,
                    cash=900,
                    buying_power=800,
                    portfolio_value=1005,
                    currency="USD",
                    raw_response='{"api_key":"raw-key","nested":{"secret_token":"hidden"}}',
                ),
                AccountSnapshot(
                    created_at=now + timedelta(minutes=1),
                    equity=1002,
                    cash=902,
                    buying_power=802,
                    portfolio_value=1007,
                    currency="USD",
                    raw_response='{"status":"ACTIVE"}',
                ),
            ]
        )
        db.commit()

    snapshots = client.get("/dashboard/account-snapshots", headers={"X-Admin-Token": "secret"})
    curve = client.get("/dashboard/portfolio-curve", headers={"X-Admin-Token": "secret"})

    assert snapshots.status_code == 200
    body_text = snapshots.text
    assert "raw-key" not in body_text
    assert "hidden" not in body_text
    assert snapshots.json()[1]["raw_response"]["api_key"] == "***"
    assert snapshots.json()[1]["raw_response"]["nested"]["secret_token"] == "***"
    assert curve.status_code == 200
    assert curve.json() == [
        {
            "timestamp": "2026-05-27T16:00:00+00:00",
            "equity": 1000,
            "cash": 900,
            "buying_power": 800,
            "portfolio_value": 1005,
        },
        {
            "timestamp": "2026-05-27T16:01:00+00:00",
            "equity": 1002,
            "cash": 902,
            "buying_power": 802,
            "portfolio_value": 1007,
        },
    ]


def test_dashboard_trade_metrics_use_closed_sell_trades_only(dashboard_client):
    client, Session = dashboard_client
    now = datetime(2026, 5, 27, 16, 0, tzinfo=UTC)
    with Session() as db:
        db.add_all(
            [
                Trade(symbol="BTC/USD", side="buy", qty=0.01, price=10_000, pnl=100, created_at=now),
                Trade(symbol="BTC/USD", side="sell", qty=0.01, price=11_000, pnl=10, created_at=now + timedelta(seconds=1)),
                Trade(symbol="BTC/USD", side="sell", qty=0.01, price=9_500, pnl=-5, created_at=now + timedelta(seconds=2)),
            ]
        )
        db.commit()

    summary = client.get("/dashboard/summary", headers={"X-Admin-Token": "secret"})
    equity = client.get("/dashboard/equity-curve", headers={"X-Admin-Token": "secret"})

    assert summary.status_code == 200
    body = summary.json()
    assert body["total_trades"] == 3
    assert body["closed_trades"] == 2
    assert body["total_realized_pnl"] == 5
    assert body["win_rate"] == pytest.approx(0.5)
    assert body["average_trade_pnl"] == pytest.approx(2.5)
    assert body["best_trade_pnl"] == 10
    assert body["worst_trade_pnl"] == -5
    assert equity.status_code == 200
    assert [point["trade_pnl"] for point in equity.json()] == [10, -5]


def test_dashboard_limit_query_params_are_capped_at_500(dashboard_client):
    client, Session = dashboard_client
    now = datetime(2026, 5, 27, 16, 0, tzinfo=UTC)
    with Session() as db:
        db.add_all(
            [
                Signal(
                    symbol="BTC/USD",
                    action="hold",
                    buy_probability=0.5,
                    sell_probability=0.5,
                    reason="test",
                    created_at=now + timedelta(seconds=index),
                )
                for index in range(550)
            ]
        )
        db.commit()

    response = client.get("/dashboard/signals?limit=999", headers={"X-Admin-Token": "secret"})

    assert response.status_code == 200
    assert len(response.json()) == 500


def test_dashboard_run_once_includes_dashboard_summary(dashboard_client):
    client, _ = dashboard_client

    response = client.post("/dashboard/run-once", headers={"X-Admin-Token": "secret"})

    assert response.status_code == 200
    assert response.json()["summary"] == {
        "action": "hold",
        "reason": "dashboard_test",
        "buy_probability": 0.51,
        "sell_probability": 0.49,
        "order_status": None,
        "latest_price": 101.5,
    }
