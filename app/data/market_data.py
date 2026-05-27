from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd

from app.broker.execution_guard import assert_btc_only
from app.config import ALLOWED_SYMBOL, Settings, get_settings


class MarketDataClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    @property
    def headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.settings.alpaca_api_key,
            "APCA-API-SECRET-KEY": self.settings.alpaca_secret_key,
        }

    async def fetch_bars(
        self,
        symbol: str = ALLOWED_SYMBOL,
        *,
        timeframe: str | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        assert_btc_only(symbol, context="market_data_fetch_bars")
        timeframe = timeframe or self.settings.timeframe
        limit = limit or self.settings.lookback_bars
        if self.settings.alpaca_api_key and self.settings.alpaca_secret_key:
            end = datetime.now(UTC)
            start = end - timedelta(days=max(10, int(limit * 20 / 24)))
            params = {
                "symbols": symbol,
                "timeframe": timeframe,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "limit": limit,
            }
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(
                    f"{self.settings.alpaca_data_base_url}/v1beta3/crypto/us/bars",
                    headers=self.headers,
                    params=params,
                )
                response.raise_for_status()
                data = response.json().get("bars", {}).get(symbol, [])
                return self._bars_to_frame(data)
        return self.synthetic_btc_bars(limit=max(limit, self.settings.min_training_rows + 80))

    async def fetch_latest_quote(self, symbol: str = ALLOWED_SYMBOL) -> dict[str, Any]:
        assert_btc_only(symbol, context="market_data_fetch_latest_quote")
        if not (self.settings.alpaca_api_key and self.settings.alpaca_secret_key):
            return {"bid_price": None, "ask_price": None, "bid_size": None, "ask_size": None}
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(
                f"{self.settings.alpaca_data_base_url}/v1beta3/crypto/us/latest/quotes",
                headers=self.headers,
                params={"symbols": symbol},
            )
            response.raise_for_status()
            return response.json().get("quotes", {}).get(symbol, {})

    def load_csv(self, path: str | Path, symbol: str = ALLOWED_SYMBOL) -> pd.DataFrame:
        assert_btc_only(symbol, context="market_data_load_csv")
        df = pd.read_csv(path)
        return normalize_ohlcv(df)

    def _bars_to_frame(self, rows: list[dict[str, Any]]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        df = pd.DataFrame(rows).rename(
            columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
        )
        return normalize_ohlcv(df)

    @staticmethod
    def synthetic_btc_bars(limit: int = 1200) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        ts = pd.date_range(end=pd.Timestamp.now(tz="UTC").floor("15min"), periods=limit, freq="15min")
        returns = rng.normal(0.00015, 0.006, size=limit)
        close = 65000 * np.exp(np.cumsum(returns))
        spread = close * rng.uniform(0.0005, 0.004, size=limit)
        open_ = np.roll(close, 1)
        open_[0] = close[0] * (1 - returns[0])
        high = np.maximum(open_, close) + spread
        low = np.minimum(open_, close) - spread
        volume = rng.lognormal(mean=2.4, sigma=0.4, size=limit)
        return pd.DataFrame(
            {"timestamp": ts, "open": open_, "high": high, "low": low, "close": close, "volume": volume}
        )


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename_map = {c: c.lower() for c in out.columns}
    out = out.rename(columns=rename_map)
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out[required].dropna().sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
