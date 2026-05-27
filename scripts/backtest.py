import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.data.dataset_builder import build_training_dataset
from app.data.market_data import MarketDataClient
from app.ml.validation import walk_forward_validate


async def run_backtest() -> dict:
    settings = get_settings()
    bars = await MarketDataClient(settings).fetch_bars(settings.symbol, limit=max(1500, settings.min_training_rows + 500))
    dataset = build_training_dataset(
        bars,
        take_profit_pct=settings.take_profit_pct,
        stop_loss_pct=settings.stop_loss_pct,
    )
    metrics = walk_forward_validate(dataset, min_train_rows=settings.min_training_rows, threshold=settings.min_buy_probability)
    report = {
        "symbol": settings.symbol,
        "fee_assumption_pct": 0.001,
        "slippage_assumption_pct": 0.001,
        "metrics": metrics,
        "note": "Walk-forward backtest uses triple-barrier outcomes; no profitability is hardcoded.",
    }
    path = Path(settings.log_dir) / "backtest_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(asyncio.run(run_backtest()))
