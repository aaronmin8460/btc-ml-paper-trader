import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.broker.execution_guard import BTCOnlyViolation, assert_btc_only, validate_order_request


def main() -> None:
    blocked = []
    for symbol in ["ETH/USD", "BTCUSDT", "SOL/USD", "AAPL", "SPY", "DOGE/USD"]:
        try:
            assert_btc_only(symbol, context="verify_btc_only")
        except BTCOnlyViolation:
            blocked.append(symbol)
    validate_order_request("BTC/USD", "buy", notional=25, context="verify_btc_only")
    print({"blocked_symbols": blocked, "allowed_symbol": "BTC/USD", "passed": len(blocked) == 6})
    if len(blocked) != 6:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
