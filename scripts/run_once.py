import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.trader import Trader


async def main() -> None:
    result = await Trader().run_once()
    summary = {
        "symbol": result["prediction"]["symbol"],
        "buy_probability": result["prediction"]["buy_probability"],
        "sell_probability": result["prediction"]["sell_probability"],
        "decision": result["decision"],
        "order": result["order"],
    }
    print(summary)


if __name__ == "__main__":
    asyncio.run(main())
