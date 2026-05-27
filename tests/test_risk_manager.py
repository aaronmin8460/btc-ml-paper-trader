from app.risk.risk_manager import PositionState, RiskManager


def test_buy_cannot_happen_if_already_holding_btc():
    risk = RiskManager()
    approved, reason = risk.approve_buy(notional=25, position=PositionState(qty=0.01), latest_price=65000)
    assert approved is False
    assert reason == "already_holding_btc"


def test_risk_manager_blocks_oversized_orders():
    risk = RiskManager()
    approved, reason = risk.approve_buy(notional=10_000, position=PositionState(), latest_price=65000)
    assert approved is False
    assert "exceeds" in reason
