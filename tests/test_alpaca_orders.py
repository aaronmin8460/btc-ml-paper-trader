import pytest

from app.broker.execution_guard import BTCOnlyViolation, LongOnlyViolation
from app.broker.alpaca_client import AlpacaClient
from app.config import Settings


class FakeResponse:
    def __init__(self, status_code: int = 200) -> None:
        self.status_code = status_code

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return {"id": "paper-order-1", "status": "accepted"}


class RecordingAsyncClient:
    requests = []
    get_status_codes = []

    def __init__(self, *args, **kwargs) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None

    async def post(self, url, headers, json):
        self.requests.append({"url": url, "headers": headers, "json": json})
        return FakeResponse()

    async def get(self, url, headers):
        self.requests.append({"url": url, "headers": headers, "json": None})
        status_code = self.get_status_codes.pop(0) if self.get_status_codes else 200
        return FakeResponse(status_code=status_code)


@pytest.fixture
def recording_httpx(monkeypatch):
    RecordingAsyncClient.requests = []
    RecordingAsyncClient.get_status_codes = []
    monkeypatch.setattr("app.broker.alpaca_client.httpx.AsyncClient", RecordingAsyncClient)
    return RecordingAsyncClient


def _live_paper_settings(**overrides):
    defaults = {
        "_env_file": None,
        "trading_enabled": True,
        "alpaca_api_key": "paper-key",
        "alpaca_secret_key": "paper-secret",
        "order_type": "market",
        "time_in_force": "gtc",
        "limit_price_offset_bps": 2,
        "order_status_check_enabled": False,
    }
    defaults.update(overrides)
    return Settings(**defaults)


@pytest.mark.anyio
async def test_market_order_payload_unchanged_by_default(recording_httpx):
    client = AlpacaClient(_live_paper_settings())

    await client.submit_order(symbol="BTC/USD", side="buy", notional=25, current_position_qty=0)

    assert recording_httpx.requests[0]["json"] == {
        "symbol": "BTC/USD",
        "side": "buy",
        "type": "market",
        "time_in_force": "gtc",
        "notional": "25",
    }


@pytest.mark.anyio
async def test_submit_market_order_keeps_legacy_market_gtc_payload(recording_httpx):
    client = AlpacaClient(_live_paper_settings(order_type="limit", time_in_force="ioc"))

    await client.submit_market_order(symbol="BTC/USD", side="buy", notional=25, current_position_qty=0)

    assert recording_httpx.requests[0]["json"] == {
        "symbol": "BTC/USD",
        "side": "buy",
        "type": "market",
        "time_in_force": "gtc",
        "notional": "25",
    }


@pytest.mark.anyio
async def test_limit_buy_payload_includes_limit_price(recording_httpx):
    client = AlpacaClient(_live_paper_settings(order_type="limit", time_in_force="ioc"))

    await client.submit_order(
        symbol="BTC/USD",
        side="buy",
        notional=10,
        current_position_qty=0,
        quote={"ask_price": 65000, "bid_price": 64990},
        latest_price=64995,
    )

    payload = recording_httpx.requests[0]["json"]
    assert payload["type"] == "limit"
    assert payload["time_in_force"] == "ioc"
    assert payload["notional"] == "10"
    assert float(payload["limit_price"]) == pytest.approx(65013.0)


@pytest.mark.anyio
async def test_limit_sell_payload_includes_limit_price(recording_httpx):
    client = AlpacaClient(_live_paper_settings(order_type="limit", time_in_force="ioc"))

    await client.submit_order(
        symbol="BTC/USD",
        side="sell",
        qty=0.01,
        current_position_qty=0.01,
        quote={"bid_price": 65000, "ask_price": 65010},
        latest_price=65005,
    )

    payload = recording_httpx.requests[0]["json"]
    assert payload["type"] == "limit"
    assert payload["time_in_force"] == "ioc"
    assert payload["qty"] == "0.01"
    assert float(payload["limit_price"]) == pytest.approx(64987.0)


@pytest.mark.anyio
async def test_non_btc_order_is_rejected(recording_httpx):
    client = AlpacaClient(_live_paper_settings())

    with pytest.raises(BTCOnlyViolation):
        await client.submit_order(symbol="ETH/USD", side="buy", notional=25)

    assert recording_httpx.requests == []


@pytest.mark.anyio
async def test_non_btc_limit_order_is_rejected_before_quote_validation(recording_httpx):
    client = AlpacaClient(_live_paper_settings(order_type="limit"))

    with pytest.raises(BTCOnlyViolation):
        await client.submit_order(symbol="ETH/USD", side="buy", notional=25, quote={})

    assert recording_httpx.requests == []


@pytest.mark.anyio
async def test_sell_without_position_is_rejected(recording_httpx):
    client = AlpacaClient(_live_paper_settings())

    with pytest.raises(LongOnlyViolation):
        await client.submit_order(symbol="BTC/USD", side="sell", qty=0.01, current_position_qty=0)

    assert recording_httpx.requests == []


@pytest.mark.anyio
async def test_limit_sell_without_position_is_rejected_before_quote_validation(recording_httpx):
    client = AlpacaClient(_live_paper_settings(order_type="limit"))

    with pytest.raises(LongOnlyViolation):
        await client.submit_order(symbol="BTC/USD", side="sell", qty=0.01, current_position_qty=0, quote={})

    assert recording_httpx.requests == []


@pytest.mark.anyio
async def test_missing_quote_blocks_limit_order_safely(recording_httpx):
    client = AlpacaClient(_live_paper_settings(order_type="limit"))

    with pytest.raises(ValueError, match="limit_order_quote_missing"):
        await client.submit_order(symbol="BTC/USD", side="buy", notional=25, quote={})

    assert recording_httpx.requests == []


@pytest.mark.anyio
async def test_soft_api_budget_skips_optional_order_status_check(recording_httpx, monkeypatch):
    class SoftBudgetLimiter:
        async def acquire(self, *, endpoint):
            return 0

        def soft_budget_reached(self):
            return True

    monkeypatch.setattr("app.broker.alpaca_client.get_alpaca_rate_limiter", lambda settings: SoftBudgetLimiter())
    client = AlpacaClient(
        _live_paper_settings(
            order_status_check_enabled=True,
            order_status_check_delay_seconds=0,
        )
    )

    result = await client.submit_order(symbol="BTC/USD", side="buy", notional=25, current_position_qty=0)

    assert result["status_check_skipped_reason"] == "api_soft_budget"
    assert len(recording_httpx.requests) == 1
    assert recording_httpx.requests[0]["json"]["type"] == "market"


@pytest.mark.anyio
async def test_get_position_url_encodes_crypto_symbol(recording_httpx):
    client = AlpacaClient(_live_paper_settings())

    await client.get_position("BTC/USD")

    assert recording_httpx.requests[0]["url"].endswith("/v2/positions/BTC%2FUSD")


@pytest.mark.anyio
async def test_get_position_falls_back_to_legacy_crypto_symbol(recording_httpx):
    recording_httpx.get_status_codes = [404, 200]
    client = AlpacaClient(_live_paper_settings())

    await client.get_position("BTC/USD")

    assert recording_httpx.requests[0]["url"].endswith("/v2/positions/BTC%2FUSD")
    assert recording_httpx.requests[1]["url"].endswith("/v2/positions/BTCUSD")
