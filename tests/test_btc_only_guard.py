import pytest

from app.broker.execution_guard import BTCOnlyViolation, LongOnlyViolation, assert_btc_only, validate_order_request
from app.config import Settings


def test_btc_only_guard_blocks_non_btc_symbols():
    for symbol in ["ETH/USD", "BTCUSDT", "SOL/USD", "AAPL", "SPY"]:
        with pytest.raises(BTCOnlyViolation):
            assert_btc_only(symbol, context="test")


def test_btc_only_guard_allows_btc_usd():
    assert assert_btc_only("BTC/USD", context="test") == "BTC/USD"


def test_config_rejects_non_btc_symbol():
    with pytest.raises(ValueError):
        Settings(symbol="ETH/USD")


def test_sell_cannot_happen_without_position():
    with pytest.raises(LongOnlyViolation):
        validate_order_request("BTC/USD", "sell", qty=0.01, current_position_qty=0)


def test_env_defaults_are_safe():
    settings = Settings()
    assert settings.trading_enabled is False
    assert settings.auto_trade_enabled is False
    assert settings.paper_trading_only is True
