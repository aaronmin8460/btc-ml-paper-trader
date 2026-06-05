from app.config import Settings
from app.data.market_data import MarketDataClient
from app.ml.training_diagnostics import build_training_dataset_with_diagnostics
from scripts.diagnose_labels import build_diagnosis_report


def test_diagnose_labels_report_counts_buy_and_sell_distribution_correctly(tmp_path):
    bars = MarketDataClient.synthetic_btc_bars(120)
    settings = Settings(
        _env_file=None,
        model_dir=str(tmp_path),
        scalping_mode_enabled=True,
        min_buy_positive_labels=0,
        min_buy_positive_label_pct=0.0,
        taker_fee_bps=0,
        slippage_bps=0,
        max_spread_bps=0,
        label_fee_bps_per_side=0,
        label_slippage_bps_per_side=0,
        label_spread_bps=0,
        label_min_net_profit_pct=0.0,
        label_horizon_bars=6,
        scalping_label_take_profit_pct=0.0001,
        scalping_label_stop_loss_pct=0.01,
        scalping_label_min_net_profit_pct=0.0,
        exit_profit_buffer_bps=0,
    )

    trainable, _ = build_training_dataset_with_diagnostics(bars, settings)
    report = build_diagnosis_report(bars, settings)

    expected_buy = {
        0: int((trainable["buy_quality_label"].astype(int) == 0).sum()),
        1: int((trainable["buy_quality_label"].astype(int) == 1).sum()),
    }
    expected_sell = {
        0: int((trainable["sell_quality_label"].astype(int) == 0).sum()),
        1: int((trainable["sell_quality_label"].astype(int) == 1).sum()),
    }
    summary = report["summary"]
    assert summary["buy_quality_label_distribution"] == expected_buy
    assert summary["exit_quality_label_distribution"] == expected_sell
    assert summary["sell_quality_label_distribution"] == expected_sell
    assert summary["buy_positive_label_count"] == expected_buy[1]
    assert summary["exit_positive_label_count"] == expected_sell[1]
    assert summary["sell_positive_label_count"] == expected_sell[1]
    assert summary["training_label_assumptions"]["horizon_bars"] == 6
    assert "conservative_promotion_assumptions" in summary
    assert isinstance(summary["buy_exit_reason_distribution"], dict)
