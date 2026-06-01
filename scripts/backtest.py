import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.backtest.scalping import backtest_assumptions, walk_forward_fee_aware_backtest
from app.data.dataset_builder import build_training_dataset, training_feature_columns
from app.data.market_data import MarketDataClient
from app.ml.validation import walk_forward_validate


async def run_backtest() -> dict:
    settings = get_settings()
    bars = await MarketDataClient(settings).fetch_bars(settings.symbol, limit=max(1500, settings.min_training_rows + 500))
    take_profit_pct = settings.scalping_label_take_profit_pct if settings.scalping_mode_enabled else settings.take_profit_pct
    stop_loss_pct = settings.scalping_label_stop_loss_pct if settings.scalping_mode_enabled else settings.stop_loss_pct
    threshold = settings.scalping_buy_probability_floor if settings.scalping_mode_enabled else settings.min_buy_probability
    feature_columns = training_feature_columns(settings.scalping_mode_enabled)
    dataset = build_training_dataset(
        bars,
        scalping_label_horizon_bars=settings.scalping_label_horizon_bars,
        take_profit_pct=take_profit_pct,
        stop_loss_pct=stop_loss_pct,
        scalping_mode_enabled=settings.scalping_mode_enabled,
        trailing_stop_pct=settings.scalping_trailing_stop_pct if settings.scalping_mode_enabled else settings.trailing_stop_pct,
        trailing_stop_arm_profit_pct=settings.trailing_stop_arm_profit_pct,
        fee_bps_per_side=settings.taker_fee_bps if settings.backtest_use_taker_fees else settings.maker_fee_bps,
        slippage_bps_per_side=settings.slippage_bps,
        spread_cost_pct=(settings.max_spread_bps / 10_000) if settings.scalping_mode_enabled else 0.0,
        min_net_exit_profit_pct=settings.scalping_label_min_net_profit_pct if settings.scalping_mode_enabled else 0.0,
        exit_profit_buffer_bps=settings.exit_profit_buffer_bps if settings.scalping_mode_enabled else 0.0,
    )
    metrics = walk_forward_validate(
        dataset,
        min_train_rows=settings.min_training_rows,
        threshold=threshold,
        take_profit_pct=take_profit_pct,
        stop_loss_pct=stop_loss_pct,
        feature_columns=feature_columns,
        sell_threshold=settings.min_sell_probability,
    )
    spread_column = "scalping_spread_pct" if settings.scalping_mode_enabled else "orderbook_spread"
    spread_available = spread_column in dataset.columns and bool((dataset[spread_column] > 0).any())
    backtest_metrics = walk_forward_fee_aware_backtest(
        dataset,
        settings,
        min_train_rows=settings.min_training_rows,
        threshold=threshold,
        feature_columns=feature_columns,
    )
    report = {
        "symbol": settings.symbol,
        "assumptions": backtest_assumptions(settings, spread_available=spread_available),
        "walk_forward_validation": metrics,
        "metrics": backtest_metrics,
        "note": (
            "Fee-aware walk-forward backtest runs shared local paper execution simulation for entry and exit, "
            "uses configured spread fallback when quote data is absent, counts IOC cancellations and partial fills, "
            "treats ambiguous candles as stop-loss first, and does not claim profitability."
        ),
    }
    path = Path(settings.log_dir) / "backtest_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(asyncio.run(run_backtest()))
