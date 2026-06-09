# Safe Historical BTC/USD Backfill

`scripts/backfill_market_data.py` fetches real historical BTC/USD OHLCV bars and writes valid, complete candles into SQLite `collected_market_data`.

It is research/training data plumbing only. It does not enable trading, start services, edit `.env`, place orders, generate synthetic candles, lower thresholds, promote models, or claim profitability.

## What Backfill Does

- Fetches real Alpaca historical BTC/USD bars for `1Min`, `5Min`, and `15Min`.
- Defaults to dry-run mode unless `--run` is explicitly passed.
- Defaults to a conservative 30-day window when no `--days`, `--start`, or `--end` is provided.
- Excludes the currently forming candle for each timeframe.
- Normalizes timestamps to UTC and stores them in the same SQLite table used by the live collector.
- Validates OHLCV values before storage.
- Avoids duplicate rows by `symbol + timeframe + timestamp`.
- Records source metadata with `source`, `source_used`, `backfilled`, and `provider_metadata`.
- Writes reports to:
  - `logs/backfill_market_data_report.json`
  - `logs/backfill_market_data_report_latest.json`

## What Backfill Does Not Do

- It does not enable `TRADING_ENABLED`.
- It does not enable `AUTO_TRADE_ENABLED`.
- It does not start `btc-paper-trader.service`.
- It does not place or simulate broker orders.
- It does not use fallback trading.
- It does not use synthetic, generated, or fake candles.
- It does not trade non-BTC symbols.
- It does not allow margin, shorts, or multi-symbol mode.
- It does not lower research, training, or promotion thresholds.
- It does not auto-promote weak models.

## Backfill Versus Live Collection

The live collector periodically appends recent BTC/USD bars and quote fields as time passes. Historical backfill asks the provider for older completed OHLCV bars so the existing research/training readiness checks can evaluate a deeper history sooner.

Backfill rows still land in `collected_market_data`, so `scripts/auto_research_train.py` reads them through the existing path.

## Dry-Run

Dry-run is the default and inserts nothing:

```bash
.venv/bin/python scripts/backfill_market_data.py \
  --dry-run \
  --symbol BTC/USD \
  --timeframes 1Min 5Min 15Min \
  --days 30 \
  --json
```

Inspect:

```bash
jq . logs/backfill_market_data_report_latest.json
jq '.per_timeframe_summary' logs/backfill_market_data_report_latest.json
```

## Actual Backfill

Only run this after the dry-run report shows real provider data and safe flags:

```bash
.venv/bin/python scripts/backfill_market_data.py \
  --run \
  --symbol BTC/USD \
  --timeframes 1Min 5Min 15Min \
  --days 30 \
  --database data/trading.db \
  --json
```

## Inspect Row Counts

```bash
sqlite3 data/trading.db "
select timeframe, count(*), min(timestamp), max(timestamp)
from collected_market_data
where symbol = 'BTC/USD'
group by timeframe
order by timeframe;
"
```

## Verify Trading Stayed Off

```bash
grep -E '^(PAPER_TRADING_ONLY|TRADING_ENABLED|AUTO_TRADE_ENABLED|ALLOW_FALLBACK_TRADING|SYMBOL)=' .env
systemctl is-active btc-paper-trader.service || true
jq '.trading_remained_disabled, .orders_placed, .synthetic_data_used' logs/backfill_market_data_report_latest.json
```

Expected posture:

```text
PAPER_TRADING_ONLY=true
TRADING_ENABLED=false
AUTO_TRADE_ENABLED=false
ALLOW_FALLBACK_TRADING=false
SYMBOL=BTC/USD
```

## Run Auto Research/Train After Backfill

Dry-run first:

```bash
.venv/bin/python scripts/auto_research_train.py --dry-run --json
```

Run gated research/training only if the dry-run report is safe:

```bash
.venv/bin/python scripts/auto_research_train.py --run --json
```

This still does not enable auto trading. A promoted model is a research artifact that requires manual review.

## Time-Series Bias Warning

Never use a random train/test split for time-series trading data. Splits must preserve chronological order: train on older data, validate/test on newer data, and reserve the most recent data for validation, test, or paper-forward checks.

The current model validation path uses walk-forward style validation. Keep that property intact when changing training or research code. Historical data alone is not enough to enable trading, and paper-forward eligibility is not permission to turn on auto trading.
