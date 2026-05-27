import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.data.market_data import MarketDataClient
from app.ml.train import train_model_from_bars


async def main() -> None:
    settings = get_settings()
    bars = await MarketDataClient(settings).fetch_bars(settings.symbol, limit=max(settings.lookback_bars, settings.min_training_rows + 300))
    result = train_model_from_bars(bars, settings)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
