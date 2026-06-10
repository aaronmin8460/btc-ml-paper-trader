# Safe Auto Research/Train Pipeline

`scripts/auto_research_train.py` is an offline BTC/USD research and model-training helper. It periodically inspects collected SQLite market data, runs diagnostics and higher-timeframe research when data gates are met, and may run `train_model.py` logic only in `--run` mode after safety and data gates pass.

It writes:

- `logs/auto_research_train_report.json`
- `logs/auto_research_train_report_latest.json`

## What It Does

- Checks `.env` and process environment safety flags before importing database or training code.
- Requires `PAPER_TRADING_ONLY=true`, `SYMBOL=BTC/USD`, and `ALLOW_FALLBACK_TRADING=false`.
- Refuses `--run` if `TRADING_ENABLED=true` or `AUTO_TRADE_ENABLED=true`.
- Checks `collected_market_data` row counts and freshness for `1Min`, `5Min`, and `15Min`.
- Runs execution-cost diagnostics, label diagnostics, and higher-timeframe research when ready.
- Runs training only through the existing strict model-promotion logic.
- Reports model registry status after training.

## What It Does Not Do

- It does not enable trading.
- It does not start `btc-paper-trader.service`.
- It does not edit `.env`.
- It does not modify production credentials.
- It does not place orders.
- It does not enable live trading, margin, short selling, fallback trading, or multi-symbol trading.
- It does not auto-apply strategy configs.
- It does not claim profitability or fabricate model performance.

## How This Differs From Auto Trading

Auto trading evaluates live signals and may submit paper orders when enabled. This pipeline only prepares diagnostics, research output, and possible model artifacts. Even if a model is promoted by existing strict rules, a human must review the reports and keep auto trading disabled until explicitly deciding otherwise.

## Manual Commands

Historical backfill can be used before these commands to speed up readiness checks without enabling trading. See `docs/historical_backfill.md`.

Dry-run inspection:

```bash
python3 scripts/auto_research_train.py --dry-run --json
```

Run gated research/training:

```bash
python3 scripts/auto_research_train.py --run --json
```

Force research even when a timeframe is not ready:

```bash
python3 scripts/auto_research_train.py --dry-run --force-research --json
```

Forced research remains invalid for trading decisions when row-count, freshness, or real-data-source requirements fail.

## Inspect Reports

```bash
jq . logs/auto_research_train_report_latest.json
jq '.data_readiness_by_timeframe' logs/auto_research_train_report_latest.json
jq '.training_gate_results' logs/auto_research_train_report_latest.json
jq '.model_registry_status' logs/auto_research_train_report_latest.json
jq '.buy_the_dip_20_plus_trade_configs, .buy_the_dip_profitable_20_plus_trade_configs, .buy_the_dip_economically_viable_count' logs/auto_research_train_report_latest.json
```

The report always includes `orders_placed: 0`.

## Install The Systemd Timer Manually

These templates are committed but not installed or enabled automatically.

```bash
sudo cp deploy/systemd/btc-model-research-train.service /etc/systemd/system/
sudo cp deploy/systemd/btc-model-research-train.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now btc-model-research-train.timer
systemctl status btc-model-research-train.timer
```

The service runs:

```bash
/home/ubuntu/btc-ml-paper-trader/.venv/bin/python scripts/auto_research_train.py --run --json
```

## Stop And Disable The Timer

```bash
sudo systemctl disable --now btc-model-research-train.timer
sudo systemctl reset-failed btc-model-research-train.service
```

This does not affect `btc-market-data-collector.timer`.

## Verify Trading Remains Disabled

```bash
grep -E '^(PAPER_TRADING_ONLY|TRADING_ENABLED|AUTO_TRADE_ENABLED|ALLOW_FALLBACK_TRADING|SYMBOL)=' .env
systemctl is-enabled btc-paper-trader.service || true
systemctl is-active btc-paper-trader.service || true
jq '.trading_remained_disabled, .orders_placed' logs/auto_research_train_report_latest.json
```

Expected safety posture:

```text
PAPER_TRADING_ONLY=true
TRADING_ENABLED=false
AUTO_TRADE_ENABLED=false
ALLOW_FALLBACK_TRADING=false
SYMBOL=BTC/USD
```

## Why 15Min Needs Around 1000 Rows

The 15Min research path needs enough real collected bars to cover indicator warmup, label horizons, trade simulation, and validation without leaning on stale market-data-client output or synthetic fallback. Around 1000 collected rows gives the research script enough sample depth to reject weak configs honestly instead of overreading a tiny window.

## Buy-The-Dip V2 Rejected

`buy_the_dip_mean_reversion` v2 is now a historical rejected strategy. It was evaluated on real roughly 180-day BTC/USD `collected_market_data`, generated enough trades, and still failed after conservative fee, slippage, and spread assumptions.

Gross return could look positive in some configs, but net return did not survive costs. It must not be connected to training, promotion, paper-forward, or trading.

The old command remains available only for historical reproduction:

```bash
.venv/bin/python scripts/research_higher_timeframe.py \
  --strategy buy_the_dip_mean_reversion \
  --max-rows-5min 8000 \
  --max-rows-15min 2880 \
  --max-buy-dip-configs 2000 \
  --json
```

The v2 report penalizes one-trade luck. Configs below 20 trades are marked `statistically_weak`, configs with one dominant winner are ranked lower, and the report separates best configs with any trades from best configs with 20+ and 50+ trades. The verified real-data outcome had zero economically viable configs.

Do not train on Buy-the-Dip v2.

## Strategy Research V3

Use v3 for new research:

```bash
.venv/bin/python scripts/research_higher_timeframe.py \
  --strategy all \
  --max-rows-15min 17274 \
  --max-rows-1h 5000 \
  --max-v3-configs 5000 \
  --walk-forward-splits 4 \
  --json
```

V5 reality audit mode reports `strategy_reality_audit_available`, `fifteen_min_rejected`, `rejected_strategy_families`, `best_htf_strategy`, `best_htf_config`, `baseline_comparison_available`, and `training_skipped_reason` in the auto research/train summary. Training remains blocked unless existing gates and the reality-audit gates pass.

V3/V5 supports `uptrend_pullback`, `volatility_breakout`, and conservative higher-timeframe templates on `15Min`, `1H`, `4H`, and `1D`. If raw higher-timeframe bars are unavailable, complete candles are derived chronologically from real lower-timeframe `collected_market_data`.

## Paper-Forward Is Not Trading Permission

Paper-forward eligibility only means a config survived offline checks in the report. It does not enable auto trading, does not apply that config, does not prove profitability, and does not authorize orders. Treat it as a candidate for manual review, not a switch to flip.

Never use a random split for trading time series. Train on older rows, validate/test on newer rows, and keep the most recent sample for validation, test, or paper-forward checks.
