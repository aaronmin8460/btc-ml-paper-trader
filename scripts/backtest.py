import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.backtest.scalping import backtest_assumptions, walk_forward_fee_aware_backtest
from app.data.dataset_builder import build_training_dataset
from app.data.market_data import MarketDataClient
from app.ml.validation import walk_forward_validate


async def run_backtest() -> dict:
    settings = get_settings()
    bars = await MarketDataClient(settings).fetch_bars(settings.symbol, limit=max(1500, settings.min_training_rows + 500))
    take_profit_pct = settings.scalping_take_profit_pct if settings.scalping_mode_enabled else settings.take_profit_pct
    stop_loss_pct = settings.scalping_stop_loss_pct if settings.scalping_mode_enabled else settings.stop_loss_pct
    threshold = settings.scalping_buy_probability_floor if settings.scalping_mode_enabled else settings.min_buy_probability
    dataset = build_training_dataset(
        bars,
        take_profit_pct=take_profit_pct,
        stop_loss_pct=stop_loss_pct,
    )
    metrics = walk_forward_validate(
        dataset,
        min_train_rows=settings.min_training_rows,
        threshold=threshold,
        take_profit_pct=take_profit_pct,
        stop_loss_pct=stop_loss_pct,
    )
    spread_available = "orderbook_spread" in dataset.columns and bool((dataset["orderbook_spread"] > 0).any())
    backtest_metrics = walk_forward_fee_aware_backtest(
        dataset,
        settings,
        min_train_rows=settings.min_training_rows,
        threshold=threshold,
    )
    report = {
        "symbol": settings.symbol,
        "assumptions": backtest_assumptions(settings, spread_available=spread_available),
        "walk_forward_validation": metrics,
        "metrics": backtest_metrics,
        "note": (
            "Fee-aware walk-forward backtest uses triple-barrier outcomes, subtracts configured fees, "
            "slippage, and available spread costs, and does not hardcode profitability."
        ),
    }
    path = Path(settings.log_dir) / "backtest_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(asyncio.run(run_backtest()))
