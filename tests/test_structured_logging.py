from app.monitoring.logger import ORDER_LOG_FIELDS, RISK_BLOCK_LOG_FIELDS, SIGNAL_LOG_FIELDS, normalize_event_payload


def test_signal_payload_normalization_adds_required_fields():
    payload = normalize_event_payload("signal", {"symbol": "BTC/USD", "action": "hold", "reason": "waiting"})

    assert set(SIGNAL_LOG_FIELDS).issubset(payload)
    assert payload["timestamp"] is not None


def test_order_and_risk_payload_normalization_add_required_fields():
    order = normalize_event_payload("order", {"side": "buy", "status": "canceled"})
    risk = normalize_event_payload("risk_block", {"reason": "stale_market_data"})

    assert set(ORDER_LOG_FIELDS).issubset(order)
    assert set(RISK_BLOCK_LOG_FIELDS).issubset(risk)
