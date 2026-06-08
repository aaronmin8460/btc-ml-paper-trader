import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import Settings, get_settings
from app.db.database import Base, connect_args_for_database_url
from app.db.models import CollectedMarketData
from scripts import auto_research_train as art


def _settings(tmp_path: Path, **overrides) -> Settings:
    values = {
        "_env_file": None,
        "database_url": f"sqlite:///{tmp_path / 'auto_research.db'}",
        "model_dir": str(tmp_path / "models"),
        "log_dir": str(tmp_path / "logs"),
        "timeframe": "15Min",
        "trading_enabled": False,
        "auto_trade_enabled": False,
        "allow_fallback_trading": False,
        "min_training_rows": 1000,
    }
    values.update(overrides)
    return Settings(**values)


def _safe_env(tmp_path: Path, **overrides) -> dict[str, str]:
    env = {
        "PAPER_TRADING_ONLY": "true",
        "TRADING_ENABLED": "false",
        "AUTO_TRADE_ENABLED": "false",
        "ALLOW_FALLBACK_TRADING": "false",
        "SYMBOL": "BTC/USD",
        "TIMEFRAME": "15Min",
        "DATABASE_URL": f"sqlite:///{tmp_path / 'auto_research.db'}",
        "MODEL_DIR": str(tmp_path / "models"),
        "LOG_DIR": str(tmp_path / "logs"),
        "MAX_OPEN_POSITIONS": "1",
        "ALPACA_PAPER_BASE_URL": "https://paper-api.alpaca.markets",
    }
    env.update(overrides)
    return env


def _apply_env(monkeypatch, env: dict[str, str]) -> None:
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()


def _session_factory(tmp_path: Path):
    database_url = f"sqlite:///{tmp_path / 'auto_research.db'}"
    engine = create_engine(database_url, connect_args=connect_args_for_database_url(database_url))
    Base.metadata.create_all(bind=engine)
    return engine, sessionmaker(bind=engine)


def _insert_collected_rows(Session, *, timeframe: str, latest: datetime, count: int, step_minutes: int) -> None:
    timestamps = [latest - timedelta(minutes=step_minutes * (count - 1 - index)) for index in range(count)]
    with Session() as db:
        for index, timestamp in enumerate(timestamps):
            price = 100.0 + index
            db.add(
                CollectedMarketData(
                    symbol="BTC/USD",
                    timeframe=timeframe,
                    timestamp=timestamp,
                    open=price,
                    high=price + 1.0,
                    low=price - 1.0,
                    close=price + 0.25,
                    volume=1.0 + index,
                    collected_at=latest,
                )
            )
        db.commit()


def test_auto_research_train_dry_run_does_not_run_train_model(monkeypatch, tmp_path):
    engine, Session = _session_factory(tmp_path)
    _apply_env(monkeypatch, _safe_env(tmp_path))

    def fail_train(*args, **kwargs):
        raise AssertionError("dry-run must not call train_model")

    monkeypatch.setattr(art, "run_train_model", fail_train)
    try:
        report, exit_code = asyncio.run(
            art.build_auto_research_train_report(
                mode="dry-run",
                explicit_dry_run=True,
                env=_safe_env(tmp_path),
                env_path=tmp_path / "missing.env",
                session_factory=Session,
            )
        )
    finally:
        engine.dispose()
        get_settings.cache_clear()

    assert exit_code == 0
    assert report["training_was_run"] is False
    assert report["train_model_result"]["status"] == "not_run"
    assert "dry_run_never_runs_train_model" in report["training_gate_results"]["blocked_reasons"]


def test_auto_research_train_never_enables_trading_or_modifies_env(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "PAPER_TRADING_ONLY=true",
                "TRADING_ENABLED=false",
                "AUTO_TRADE_ENABLED=false",
                "ALLOW_FALLBACK_TRADING=false",
                "SYMBOL=BTC/USD",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    before = env_file.read_text(encoding="utf-8")
    engine, Session = _session_factory(tmp_path)
    env = _safe_env(tmp_path)
    _apply_env(monkeypatch, env)

    try:
        report, _ = asyncio.run(
            art.build_auto_research_train_report(
                mode="dry-run",
                explicit_dry_run=True,
                env=env,
                env_path=env_file,
                session_factory=Session,
            )
        )
    finally:
        engine.dispose()
        get_settings.cache_clear()

    assert env_file.read_text(encoding="utf-8") == before
    assert report["safety_flags"]["trading_enabled"] is False
    assert report["safety_flags"]["auto_trade_enabled"] is False
    assert report["trading_remained_disabled"] is True


def test_auto_research_train_exits_if_paper_trading_only_false():
    safety = art.evaluate_environment_safety(
        art.load_effective_env(env={"PAPER_TRADING_ONLY": "false"}, env_path=Path("/missing")),
        inspection_only_dry_run=True,
    )

    assert safety.passed is False
    assert "paper_trading_only_not_true" in safety.fatal_reasons


def test_auto_research_train_exits_if_symbol_is_not_btc_usd():
    safety = art.evaluate_environment_safety(
        art.load_effective_env(env={"SYMBOL": "ETH/USD"}, env_path=Path("/missing")),
        inspection_only_dry_run=True,
    )

    assert safety.passed is False
    assert "symbol_not_btc_usd" in safety.fatal_reasons


def test_auto_research_train_exits_if_fallback_trading_enabled():
    safety = art.evaluate_environment_safety(
        art.load_effective_env(env={"ALLOW_FALLBACK_TRADING": "true"}, env_path=Path("/missing")),
        inspection_only_dry_run=True,
    )

    assert safety.passed is False
    assert "fallback_trading_enabled" in safety.fatal_reasons


def test_auto_research_train_blocks_training_when_15min_rows_below_minimum(tmp_path):
    now = datetime(2026, 6, 8, 6, 0, tzinfo=UTC)
    engine, Session = _session_factory(tmp_path)
    _insert_collected_rows(Session, timeframe="15Min", latest=now - timedelta(minutes=15), count=203, step_minutes=15)
    settings = _settings(tmp_path, timeframe="15Min")
    env = art.load_effective_env(env=_safe_env(tmp_path), env_path=tmp_path / "missing.env")

    try:
        readiness = art.build_data_readiness_by_timeframe(
            settings,
            effective_env=env,
            session_factory=Session,
            now=now,
        )
    finally:
        engine.dispose()

    safety = art.evaluate_environment_safety(env, inspection_only_dry_run=False)
    gates = art.evaluate_training_gates(
        mode="run",
        settings=settings,
        safety=safety,
        data_readiness_by_timeframe=readiness,
        diagnostics_summary={
            "current_target_safe_vs_cost": True,
            "label_diagnostics": {"enough_buy_labels": True},
        },
        research_summary={
            "synthetic_data_used": False,
            "economically_viable_config_count": 1,
            "all_results": [],
        },
    )

    assert readiness["15Min"]["ready_for_training"] is False
    assert readiness["15Min"]["rejection_reason"] == "row_count_below_required"
    assert gates["all_gates_passed"] is False
    assert "row_count_below_required" in gates["blocked_reasons"]


def test_auto_research_train_allows_research_when_5min_rows_are_sufficient(tmp_path):
    now = datetime(2026, 6, 8, 6, 0, tzinfo=UTC)
    engine, Session = _session_factory(tmp_path)
    _insert_collected_rows(Session, timeframe="5Min", latest=now - timedelta(minutes=5), count=1000, step_minutes=5)
    settings = _settings(tmp_path)
    env = art.load_effective_env(env=_safe_env(tmp_path), env_path=tmp_path / "missing.env")

    try:
        readiness = art.build_data_readiness_by_timeframe(
            settings,
            effective_env=env,
            session_factory=Session,
            now=now,
        )
    finally:
        engine.dispose()

    assert readiness["5Min"]["ready_for_research"] is True
    assert art.should_run_research(readiness, force_research=False) is True


def test_synthetic_fallback_cannot_produce_trading_valid_output():
    summary = art.normalize_research_summary(
        {
            "synthetic_data_used": True,
            "research_result_valid": True,
            "paper_forward_eligible_config_count": 3,
            "economically_viable_config_count": 2,
        },
        status="run",
    )

    assert summary["synthetic_data_used"] is True
    assert summary["research_result_valid"] is False
    assert summary["invalid_for_trading_decisions"] is True
    assert summary["paper_forward_eligible_config_count"] == 0
    assert summary["economically_viable_config_count"] == 0


def test_systemd_service_does_not_start_paper_trader_service():
    service = Path("deploy/systemd/btc-model-research-train.service").read_text(encoding="utf-8")

    assert "btc-paper-trader.service" not in service
    assert "systemctl" not in service
    assert "ExecStart=/home/ubuntu/btc-ml-paper-trader/.venv/bin/python scripts/auto_research_train.py --run --json" in service
    assert "User=ubuntu" in service
    assert "WorkingDirectory=/home/ubuntu/btc-ml-paper-trader" in service
    assert "NoNewPrivileges=true" in service
    assert "PrivateTmp=true" in service


def test_systemd_timer_does_not_enable_trading():
    timer = Path("deploy/systemd/btc-model-research-train.timer").read_text(encoding="utf-8")

    assert "btc-paper-trader.service" not in timer
    assert "TRADING_ENABLED" not in timer
    assert "AUTO_TRADE_ENABLED" not in timer
    assert "OnUnitActiveSec=6h" in timer


def test_reports_always_include_orders_placed_zero(monkeypatch, tmp_path):
    engine, Session = _session_factory(tmp_path)
    env = _safe_env(tmp_path)
    _apply_env(monkeypatch, env)

    try:
        report, _ = asyncio.run(
            art.build_auto_research_train_report(
                mode="dry-run",
                explicit_dry_run=True,
                env=env,
                env_path=tmp_path / "missing.env",
                session_factory=Session,
            )
        )
    finally:
        engine.dispose()
        get_settings.cache_clear()

    assert report["orders_placed"] == 0
    latest_report = tmp_path / "logs" / art.LATEST_REPORT_NAME
    assert latest_report.exists()
    assert '"orders_placed": 0' in latest_report.read_text(encoding="utf-8")


def test_btc_usd_only_and_long_only_constraints_remain_enforced():
    env = art.load_effective_env(
        env={
            "SYMBOL": "BTC/USD",
            "SHORT_SELLING_ENABLED": "false",
            "MULTI_SYMBOL_ENABLED": "false",
        },
        env_path=Path("/missing"),
    )
    safe = art.evaluate_environment_safety(env, inspection_only_dry_run=False)
    unsafe = art.evaluate_environment_safety(
        art.load_effective_env(
            env={
                "SYMBOL": "BTC/USD",
                "SHORT_SELLING_ENABLED": "true",
                "SYMBOLS": "BTC/USD,ETH/USD",
            },
            env_path=Path("/missing"),
        ),
        inspection_only_dry_run=False,
    )

    assert safe.passed is True
    assert safe.flags["btc_usd_only"] is True
    assert safe.flags["long_only"] is True
    assert unsafe.passed is False
    assert "short_selling_setting_detected" in unsafe.fatal_reasons
    assert "multi_symbol_setting_detected" in unsafe.fatal_reasons
