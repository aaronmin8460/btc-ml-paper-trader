from pathlib import Path
from types import SimpleNamespace

from app.monitoring import logger as logger_module
from app.monitoring.logger import JsonlLogger, ORDER_LOG_FIELDS, RISK_BLOCK_LOG_FIELDS, SIGNAL_LOG_FIELDS, normalize_event_payload


def test_signal_payload_normalization_adds_required_fields():
    payload = normalize_event_payload("signal", {"symbol": "BTC/USD", "action": "hold", "reason": "waiting"})

    assert set(SIGNAL_LOG_FIELDS).issubset(payload)
    assert payload["timestamp"] is not None
    assert payload["blocked_by"] is None
    assert payload["quant_score"] is None
    assert payload["candidate_strategy_count"] is None


def test_order_and_risk_payload_normalization_add_required_fields():
    order = normalize_event_payload("order", {"side": "buy", "status": "canceled"})
    risk = normalize_event_payload("risk_block", {"reason": "stale_market_data"})

    assert set(ORDER_LOG_FIELDS).issubset(order)
    assert set(RISK_BLOCK_LOG_FIELDS).issubset(risk)


def test_jsonl_logger_keeps_stdout_logging_when_file_write_fails(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(logger_module, "get_settings", lambda: SimpleNamespace(log_dir=str(tmp_path / "logs")))

    def fail_open(self, *args, **kwargs):
        raise PermissionError("read-only log directory")

    monkeypatch.setattr(Path, "open", fail_open)
    logger = JsonlLogger(name="stdout_fallback_test")

    logger.event("container_log_test", symbol="BTC/USD")

    assert '"event_type":"container_log_test"' in capsys.readouterr().err
