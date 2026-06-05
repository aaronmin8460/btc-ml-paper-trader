# btc-ml-paper-trader

Automatic BTC/USD-only paper trading bot for Alpaca paper trading.

This project is intentionally paper-only, long-only, and BTC/USD-only. It does not implement live trading, short selling, margin trading, or multi-symbol trading.

## Safety Defaults

The default `.env.example` is safe:

```env
PAPER_TRADING_ONLY=true
TRADING_ENABLED=false
AUTO_TRADE_ENABLED=false
SCALPING_MODE_ENABLED=false
SYMBOL=BTC/USD
```

There is no live-trading switch. The Alpaca broker client uses only the paper API base URL and hard-validates every order request before submission.

## Configuration Profiles

Profile A is the safe default mode. It keeps the bot paper-only and BTC/USD-only, records signals, and leaves automatic order submission off:

```env
TRADING_ENABLED=false
AUTO_TRADE_ENABLED=false
SCALPING_MODE_ENABLED=false
```

Profile B is conservative BTC/USD paper scalping. Use it only with Alpaca paper credentials and an accepted model:

```env
TRADING_ENABLED=true
AUTO_TRADE_ENABLED=true
SCALPING_MODE_ENABLED=true
SCAN_INTERVAL_SECONDS=5
ORDER_TYPE=limit
TIME_IN_FORCE=ioc
ORDER_NOTIONAL_USD=25
MAX_SPREAD_BPS=6
MAX_SLIPPAGE_BPS=8
MIN_QUOTE_IMBALANCE=0.00
SCALPING_BUY_PROBABILITY_FLOOR=0.57
SCALPING_CONFIDENCE_GAP_REQUIRED=0.06
MIN_SECONDS_BETWEEN_TRADES=15
MAX_TRADES_PER_HOUR=30
MAX_DAILY_TRADES=150
MAX_ORDER_ATTEMPTS_PER_HOUR=30
MAX_ORDER_ATTEMPTS_PER_DAY=100
IOC_CANCEL_LOOKBACK_SECONDS=300
MAX_RECENT_IOC_CANCELS=3
IOC_CANCEL_COOLDOWN_SECONDS=120
IOC_CANCEL_ESCALATION_COOLDOWN_SECONDS=600
MIN_HOLD_SECONDS_BEFORE_WEAK_QUOTE_EXIT=30
PROFIT_ONLY_EXIT_ENABLED=true
MIN_NET_EXIT_PROFIT_PCT=0.002
EXIT_PROFIT_BUFFER_BPS=5
ALLOW_EMERGENCY_STOP_LOSS=true
EMERGENCY_STOP_LOSS_PCT=0.006
TRAILING_STOP_ARM_PROFIT_PCT=0.002
MODEL_SELL_REQUIRES_PROFIT=true
WEAK_QUOTE_SELL_REQUIRES_PROFIT=true
MAX_HOLDING_SELL_REQUIRES_PROFIT=true
DISCORD_ALERT_ON_SIGNAL=false
DISCORD_ALERT_ON_ORDER=true
DISCORD_ALERT_ON_ERROR=true
DISCORD_RISK_ALERT_COOLDOWN_SECONDS=300
CIRCUIT_BREAKER_ENABLED=true
MAX_SAME_RISK_BLOCKS_BEFORE_PAUSE=20
MAX_RUNTIME_ERRORS_BEFORE_PAUSE=10
CIRCUIT_BREAKER_WINDOW_SECONDS=900
ALLOW_FALLBACK_TRADING=false
PAPER_FEE_BPS=0
PAPER_SLIPPAGE_BPS=0
```

This is still paper trading, and it is not guaranteed profitable.

## Local setup

```bash
git clone https://github.com/aaronmin8460/btc-ml-paper-trader.git
cd btc-ml-paper-trader
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add Alpaca paper credentials to `.env` if you want real paper account data or paper orders. Without credentials, the bot can still train and produce signals using deterministic synthetic BTC-like bars for local development.

Keep these defaults for local safety:

```env
PAPER_TRADING_ONLY=true
TRADING_ENABLED=false
AUTO_TRADE_ENABLED=false
SCALPING_MODE_ENABLED=false
SYMBOL=BTC/USD
```

## Local dry-run

Run one BTC/USD decision locally without submitting an order:

```bash
source .venv/bin/activate
TRADING_ENABLED=false AUTO_TRADE_ENABLED=false python scripts/run_once.py
```

With `TRADING_ENABLED=false`, the bot fetches or synthesizes BTC/USD data, produces a signal, and does not submit a paper order.

## Quant strategy backtests and reports

Run the BTC/USD paper scalping backtest:

```bash
source .venv/bin/activate
python scripts/backtest.py
```

The script prints a JSON report and writes `logs/backtest_report.json`. The report keeps walk-forward validation, fee, slippage, spread, IOC cancellation, partial fill, and ambiguous candle handling. It is an evaluation artifact only; it does not enable live trading and does not prove future profitability.

Strategy-level metrics are under `metrics.strategy_level_metrics`. Use `number_of_signals` to see how often a strategy produced or was considered for a signal, `number_of_entries` and `number_of_trades` to see how many actually became simulated paper entries/trades, and `net_return_pct`, `profit_factor_net`, `win_rate_net`, `expectancy`, `max_drawdown_pct`, `canceled_orders`, `partial_fills`, and `ambiguous_candle_ratio` to compare cost-aware outcomes. These are fractions of traded notional, not account-level promises.

Regime-level metrics are under `metrics.regime_level_metrics`. They show how many signals were allowed or blocked in regimes such as `trending`, `mean_reverting`, `ranging`, `too_volatile`, and `not_tradeable`. A strong strategy should not only show favorable trade metrics; it should also avoid forcing entries in regimes where the filter says conditions are unsafe.

Blocked-signal counts are under `metrics.blocked_signal_metrics`. `blocked_by` names the gate that stopped a candidate signal, and `block_reason` gives the specific reason. Common gates include `regime_filter`, `quant_strategy`, `ml_filter`, `risk_manager`, `spread`, `quote_imbalance`, `api_budget`, `cooldown`, `ioc_cancel_guard`, `stale_market_data`, `active_model_invalid`, and `fallback_prediction_not_allowed`.

To compare `MeanReversionScalpingStrategy` and `MomentumBreakoutStrategy`, look at both the strategy-level section and the regime-level section. Mean reversion should mostly appear in `mean_reverting` or `ranging` regimes; momentum breakout should mostly appear in `trending` regimes. A strategy with few signals, few trades, high cancellation, high drawdown, or poor net returns after fees and spread is not strong evidence for paper auto-trading.

Backtests are historical simulations with simplifying assumptions and limited data. They can reveal broken labels, excessive costs, over-filtering, or unsafe trade selection, but they do not guarantee future results. Keep `TRADING_ENABLED=false` and `AUTO_TRADE_ENABLED=false` unless you are intentionally running paper-only automation with all safety guards intact.

## Parameter sweeps and paper-forward checks

Run the bounded strategy parameter sweep:

```bash
source .venv/bin/activate
python scripts/sweep_strategy_params.py
```

The sweep writes `logs/strategy_param_sweep.csv` and `logs/strategy_param_sweep_summary.json`. It evaluates BTC/USD paper-only, long-only strategy settings with walk-forward validation and cost-aware backtesting. It does not modify `.env`, does not enable fallback trading, does not auto-apply a best config, and does not claim guaranteed profitability.

To write a candidate file for manual review:

```bash
python scripts/apply_candidate_config.py --parameter-set-id ps_001_ml_confirmation
```

This writes `.env.candidate`, never `.env`, and prints a manual diff-style summary. Review the file yourself before using it. Candidate files force `PAPER_TRADING_ONLY=true`, `SYMBOL=BTC/USD`, `TRADING_ENABLED=false`, `AUTO_TRADE_ENABLED=false`, and `ALLOW_FALLBACK_TRADING=false`.

Paper-forward evaluation checklist:

1. Run the candidate in paper mode only.
2. Test in stages: 1 hour, then 6 hours, then 24 hours.
3. Monitor `strategy_name`, `blocked_by`, spread, quote imbalance, net PnL, drawdown, order cancels, API budget, and active model availability.
4. Stop immediately if `active_model_valid=false`, prediction source is fallback, max drawdown is breached, IOC cancels repeat, spread blocks become excessive, runtime errors appear, or market data is stale.
5. Treat every passing backtest as a hypothesis, not proof. A config is only worth paper-forward testing when it has enough trades, positive net expectancy after costs, tolerable drawdown, low ambiguous candle ratio, and stable walk-forward splits.

## Local API run

Start the FastAPI server locally:

```bash
source .venv/bin/activate
APP_ENV=development PORT=8000 API_ADMIN_TOKEN=replace-with-a-local-token python scripts/run_api.py
```

Check the local health endpoint:

```bash
curl -sS http://localhost:8000/health
```

Run one protected API decision locally:

```bash
curl -sS -X POST http://localhost:8000/run-once \
  -H 'X-Admin-Token: replace-with-a-local-token'
```

## Discord webhook setup

Discord alerts are optional. The app runs with no Discord environment variables configured.

1. In Discord, create a webhook for the channel that should receive bot alerts.
2. Copy the webhook URL.
3. Store it only in your local `.env` or deployment platform secret manager.
4. Never commit the webhook URL.

Alerts are sent as rich Discord embeds for signals, paper orders, risk guards, runtime errors, model validation, and the `/alerts/discord/test` endpoint. The legacy plain text sender remains available internally for compatibility. Mention parsing is disabled in webhook payloads so bot alerts cannot ping users or roles.

Local Discord alert configuration:

```env
DISCORD_WEBHOOK_URL=replace-with-your-discord-webhook-url
DISCORD_ALERTS_ENABLED=true
DISCORD_ALERT_ON_HOLD=false
DISCORD_ALERT_ON_SIGNAL=true
DISCORD_ALERT_ON_ORDER=true
DISCORD_ALERT_ON_ERROR=true
DISCORD_ALERT_ON_MODEL=false
DISCORD_RISK_ALERT_COOLDOWN_SECONDS=300
```

Send a local test alert after starting the API:

```bash
curl -sS -X POST http://localhost:8000/alerts/discord/test \
  -H 'X-Admin-Token: replace-with-a-local-token'
```

If Discord is disabled, the endpoint returns:

```json
{"sent":false,"reason":"discord_disabled"}
```

## Docker deployment

The container entrypoint runs `python scripts/run_api.py`. It binds FastAPI to `0.0.0.0` and reads the generic `PORT` environment variable, defaulting to `8000`.

Build and run the image locally:

```bash
docker build -t btc-ml-paper-trader .
docker run --rm --env-file .env -p 8000:8000 btc-ml-paper-trader
```

For persistent SQLite data, logs, and models, use Docker Compose:

```bash
cp .env.example .env
docker compose up -d --build
```

The included `docker-compose.yml` mounts local directories under `./runtime/` into `/app/data`, `/app/logs`, and `/app/models`. These are ordinary mounted volumes, not provider-specific paths.

## AWS deployment overview

The simplest AWS setup is a single EC2 instance with Docker Compose. Install Docker, clone this repository, create a private `.env`, and run:

```bash
docker compose up -d --build
```

Use one running scheduler instance. Multiple containers with `AUTO_TRADE_ENABLED=true` can duplicate scans and paper orders.

For a managed container deployment, publish the Docker image to Amazon ECR and run one ECS/Fargate task. Store secrets in AWS Secrets Manager or SSM Parameter Store, send container stdout/stderr to CloudWatch Logs, and configure the task or load balancer health check to call `/health`.

For longer-running production-like paper trading, prefer PostgreSQL on Amazon RDS over container-local SQLite. Local development can keep:

```env
DATABASE_URL=sqlite:///./data/trading.db
```

A PostgreSQL/RDS deployment can use:

```env
DATABASE_URL=postgresql+psycopg://replace-with-user:replace-with-password@replace-with-host:5432/replace-with-database
```

SQLite is also suitable for a single Docker host when `/app/data` is backed by a mounted volume. Logs should go to stdout/stderr for container aggregation; `LOG_DIR` remains available for JSONL application logs.

## Deployment environment

Minimum safe container or AWS configuration:

```env
APP_ENV=production
PORT=8000
DATABASE_URL=sqlite:///./data/trading.db
MODEL_DIR=./models
LOG_DIR=./logs
PAPER_TRADING_ONLY=true
TRADING_ENABLED=false
AUTO_TRADE_ENABLED=false
SCALPING_MODE_ENABLED=false
SYMBOL=BTC/USD
API_ADMIN_TOKEN=replace-with-a-long-random-secret
ALPACA_API_KEY=replace-with-your-alpaca-paper-key
ALPACA_SECRET_KEY=replace-with-your-alpaca-paper-secret
ALPACA_PAPER_BASE_URL=https://paper-api.alpaca.markets
ALPACA_DATA_BASE_URL=https://data.alpaca.markets
```

Add Discord settings only when alerts are needed. Keep Alpaca credentials, the admin token, and the Discord webhook in private environment variables or a secret manager.

## Safe first deployment

Start every deployment with trading and automatic scheduling disabled:

```env
PAPER_TRADING_ONLY=true
TRADING_ENABLED=false
AUTO_TRADE_ENABLED=false
SCALPING_MODE_ENABLED=false
```

Set the deployed API URL and admin token in your shell:

```bash
export API_URL="https://replace-with-your-api-domain"
export ADMIN_TOKEN="replace-with-your-api-admin-token"
```

Verify the public health endpoint:

```bash
curl -sS "$API_URL/health"
```

Verify database, model-registry, market-client, scheduler, and paper-only readiness:

```bash
curl -sS "$API_URL/health/deep"
```

Inspect the protected scheduler state:

```bash
curl -sS "$API_URL/scheduler/status" \
  -H "X-Admin-Token: $ADMIN_TOKEN"
```

Verify that protected config output masks Alpaca keys, the admin token, and the Discord webhook:

```bash
curl -sS "$API_URL/config/safe" \
  -H "X-Admin-Token: $ADMIN_TOKEN"
```

Run one manual BTC/USD decision:

```bash
curl -sS -X POST "$API_URL/run-once" \
  -H "X-Admin-Token: $ADMIN_TOKEN"
```

With `TRADING_ENABLED=false`, `/run-once` returns a BTC/USD decision with no submitted paper order. After validating health, masked config, logs, and optional Discord alerts, enable paper auto-trading deliberately:

```env
TRADING_ENABLED=true
AUTO_TRADE_ENABLED=true
```

Keep `PAPER_TRADING_ONLY=true`, `SYMBOL=BTC/USD`, and the Alpaca paper API URL unchanged. The scheduler runs every `SCAN_INTERVAL_SECONDS` and calls `Trader.run_once()`.

## Generated File Audit And Cleanup

Audit generated files before deleting anything:

```bash
python3 scripts/audit_generated_files.py
```

Run cleanup in dry-run mode. This is the default and deletes nothing:

```bash
python3 scripts/cleanup_generated_files.py
```

Apply cleanup only after reviewing the dry-run output:

```bash
python3 scripts/cleanup_generated_files.py --apply
```

Cleanup is intentionally narrow. It only removes `.pytest_cache/`, `__pycache__/`, `*.pyc`, `train_*.log`, `train_now.log`, and `app.log`, and it writes `logs/cleanup_report.json`. Never delete `.env`, `data/trading.db`, `data/`, `models/*.joblib`, `models/registry.json`, registry backups, source under `app/`, `scripts/`, or `tests/`, deployment docs, `requirements.txt`, `pyproject.toml`, or this README.

Model files and `models/registry.json` are preserved because they define what model, if any, the paper-trading runtime is allowed to consider active. Deleting or editing them by cleanup script could hide validation failures, break auditability, or make the runtime fall back to invalid/no model behavior.

## Runtime circuit breaker

The runtime circuit breaker is enabled by default for automatic paper trading:

```env
CIRCUIT_BREAKER_ENABLED=true
MAX_SAME_RISK_BLOCKS_BEFORE_PAUSE=20
MAX_RUNTIME_ERRORS_BEFORE_PAUSE=10
CIRCUIT_BREAKER_WINDOW_SECONDS=900
```

If the same risk-block reason repeats too many times inside the window, or scheduler runtime errors repeat too often, the scheduler enters a paused state and stops calling `Trader.run_once()`. The first automatic pause sends one Discord alert titled `Auto trading paused`; repeated pause attempts do not spam Discord. Pause reason, pause timestamp, last successful run, last runtime error type, and last stale-data event are stored in the database. A restart does not clear a persisted pause.

Inspect the state and resume only after reviewing logs and the cause:

```bash
curl -sS "$API_URL/scheduler/status" \
  -H "X-Admin-Token: $ADMIN_TOKEN"

curl -sS -X POST "$API_URL/admin/resume" \
  -H "X-Admin-Token: $ADMIN_TOKEN"
```

Manual pause and compatibility status controls remain available:

```bash
curl -sS "$API_URL/admin/status" \
  -H "X-Admin-Token: $ADMIN_TOKEN"

curl -sS -X POST "$API_URL/admin/pause" \
  -H "X-Admin-Token: $ADMIN_TOKEN"

curl -sS -X POST "$API_URL/admin/resume" \
  -H "X-Admin-Token: $ADMIN_TOKEN"
```

## Important safety warnings

- Run one scheduler-enabled container only. Multiple instances can duplicate scans and paper orders.
- Keep `API_ADMIN_TOKEN` secret.
- Never commit Alpaca keys.
- Never commit the Discord webhook URL.
- This project is paper-trading-only.
- This project is BTC/USD-only.
- This project is long-only.
- Do not add live trading, short selling, margin trading, or multi-symbol trading.

## BTC-Only Enforcement

BTC/USD is enforced in multiple layers:

- `app/config.py` rejects any `SYMBOL` except `BTC/USD`.
- `app/broker/execution_guard.py` blocks every non-BTC/USD symbol and logs `rejected_order`.
- Strategy decisions call the same guard before returning buy/sell decisions.
- Alpaca order submission validates symbol, side, notional, and current position.
- Market data fetchers reject non-BTC/USD requests.

Blocked examples include `ETH/USD`, `BTCUSDT`, `SOL/USD`, `AAPL`, and `SPY`.

Verify locally:

```bash
python scripts/verify_btc_only.py
```

## Training

```bash
python scripts/train_model.py
```

The trainer builds leakage-safe OHLCV features, applies entry and long-exit labels, runs walk-forward validation, saves a model under `models/`, and promotes it to `models/registry.json` only if validation metrics pass. Training label assumptions use `LABEL_FEE_BPS_PER_SIDE`, `LABEL_SLIPPAGE_BPS_PER_SIDE`, `LABEL_SPREAD_BPS`, `LABEL_MIN_NET_PROFIT_PCT`, and `LABEL_HORIZON_BARS`; promotion backtests still use conservative execution assumptions from `TAKER_FEE_BPS`, `SLIPPAGE_BPS`, and `MAX_SPREAD_BPS`.

`buy_quality_label` is the long-entry training label. `exit_quality_label` is the long-exit training label. `sell_quality_label` remains as a compatibility alias for `exit_quality_label`; in this long-only bot it means closing an existing BTC/USD long position, never opening a short. Model promotion still requires enough positive buy labels, a passing fee-aware backtest, positive net return above threshold, sufficient net profit factor, and enough trades. It does not fabricate profitability or hardcode a winning backtest.

## ML Model

`MLSignalModel` combines:

- LogisticRegression with StandardScaler
- RandomForestClassifier
- HistGradientBoostingClassifier
- Weighted average probability output

Feature columns are tracked and saved with the model. Random train/test splits are avoided; validation is walk-forward.

If no active model is available, `Predictor` still returns fallback probabilities for local diagnostics and labels the result with `prediction_source=fallback` and `model_available=false`. Automatic trading does not use fallback predictions unless `ALLOW_FALLBACK_TRADING=true` is explicitly set. In scalping mode, BUY entries require `buy_probability` to be greater than `SCALPING_BUY_PROBABILITY_FLOOR` and must satisfy `SCALPING_CONFIDENCE_GAP_REQUIRED`, so a coin-flip probability such as `0.50` cannot trigger a BUY.

## Order Execution

Orders default to limit/IOC paper orders for scalping-oriented paper execution:

```env
ORDER_TYPE=limit
TIME_IN_FORCE=ioc
LIMIT_PRICE_OFFSET_BPS=2
IOC_CANCEL_LOOKBACK_SECONDS=300
MAX_RECENT_IOC_CANCELS=3
IOC_CANCEL_COOLDOWN_SECONDS=120
IOC_CANCEL_ESCALATION_COOLDOWN_SECONDS=600
```

Limit prices are derived from the latest BTC/USD quote with `LIMIT_PRICE_OFFSET_BPS`; if a valid quote is unavailable, the order is blocked instead of falling back to market execution. Market/GTC paper orders are still available by explicitly setting `ORDER_TYPE=market` and `TIME_IN_FORCE=gtc`.

The IOC cancel guard counts canceled limit/IOC BUY orders inside `IOC_CANCEL_LOOKBACK_SECONDS` and blocks new buys when the count reaches `MAX_RECENT_IOC_CANCELS`. A single recent IOC cancel also pauses new buys for `IOC_CANCEL_COOLDOWN_SECONDS`; repeated cancels escalate the pause for `IOC_CANCEL_ESCALATION_COOLDOWN_SECONDS`. Older canceled orders age out of the guard instead of blocking paper trading for the rest of the day.

Filled paper orders are also fed into BTC-only, long-only trade accounting. Filled BUY orders establish open lots; a filled SELL matches those lots FIFO and creates a `Trade` row with realized PnL. Canceled, rejected, and dry-run orders do not create `Trade` rows. If the broker response does not include fee data, paper accounting can apply `PAPER_FEE_BPS`; `PAPER_SLIPPAGE_BPS` adjusts assumed entry/exit prices for realized PnL. These paper accounting settings are separate from backtest fee settings.

## Scalping Configuration

Scalping mode is a REST-polling paper-trading profile for faster BTC/USD experiments. It is not true HFT: every decision still uses ordinary Alpaca REST calls, shared API budgeting, paper account risk checks, and long-only BTC/USD guardrails.

Recommended conservative paper scalping settings:

```env
TRADING_ENABLED=true
AUTO_TRADE_ENABLED=true
SCALPING_MODE_ENABLED=true
TIMEFRAME=1Min
LOOKBACK_BARS=1500
SCAN_INTERVAL_SECONDS=5

ALPACA_RATE_LIMIT_ENABLED=true
ALPACA_MAX_CALLS_PER_MINUTE=180
ALPACA_API_BUDGET_TARGET_PER_MINUTE=170
ALPACA_API_BUDGET_HARD_STOP_PER_MINUTE=195

MARKET_BARS_CACHE_SECONDS=30
POSITION_CACHE_SECONDS=2
ACCOUNT_EQUITY_CACHE_SECONDS=5
QUOTE_CACHE_SECONDS=0

ORDER_TYPE=limit
TIME_IN_FORCE=ioc
LIMIT_PRICE_OFFSET_BPS=2
ORDER_NOTIONAL_USD=25

MAX_SPREAD_BPS=6
MAX_SLIPPAGE_BPS=8
MIN_QUOTE_IMBALANCE=0.00
SCALPING_BUY_PROBABILITY_FLOOR=0.57
SCALPING_CONFIDENCE_GAP_REQUIRED=0.06
MIN_SECONDS_BETWEEN_TRADES=15
MAX_TRADES_PER_HOUR=30
MAX_DAILY_TRADES=150
MAX_ORDER_ATTEMPTS_PER_HOUR=30
MAX_ORDER_ATTEMPTS_PER_DAY=100
IOC_CANCEL_LOOKBACK_SECONDS=300
MAX_RECENT_IOC_CANCELS=3
IOC_CANCEL_COOLDOWN_SECONDS=120
IOC_CANCEL_ESCALATION_COOLDOWN_SECONDS=600
MIN_HOLD_SECONDS_BEFORE_WEAK_QUOTE_EXIT=30
DISCORD_ALERT_ON_SIGNAL=false
DISCORD_ALERT_ON_ORDER=true
DISCORD_ALERT_ON_ERROR=true
DISCORD_RISK_ALERT_COOLDOWN_SECONDS=300
ALLOW_FALLBACK_TRADING=false
PAPER_FEE_BPS=0
PAPER_SLIPPAGE_BPS=0

SCALPING_ENTRY_DIP_PCT=0.0005
SCALPING_TAKE_PROFIT_PCT=0.0015
SCALPING_STOP_LOSS_PCT=0.001
SCALPING_TRAILING_STOP_PCT=0.0008
SCALPING_MAX_POSITION_SECONDS=90
```

More active profit-guarded paper scalping settings:

```env
SCAN_INTERVAL_SECONDS=1
MIN_SECONDS_BETWEEN_TRADES=10
MAX_TRADES_PER_HOUR=60
MAX_DAILY_TRADES=300
MAX_ORDER_ATTEMPTS_PER_HOUR=120
MAX_ORDER_ATTEMPTS_PER_DAY=500
SCALPING_BUY_PROBABILITY_FLOOR=0.58
SCALPING_CONFIDENCE_GAP_REQUIRED=0.08
MIN_QUOTE_IMBALANCE=-0.005
MAX_SPREAD_BPS=6
PROFIT_ONLY_EXIT_ENABLED=true
MIN_NET_EXIT_PROFIT_PCT=0.002
EXIT_PROFIT_BUFFER_BPS=5
SCALPING_SELL_ON_WEAK_QUOTE=false
TRAILING_STOP_ARM_PROFIT_PCT=0.002
ALLOW_EMERGENCY_STOP_LOSS=true
EMERGENCY_STOP_LOSS_PCT=0.006
```

The bot uses latest quote/mid price for scalping entries, limit prices, emergency stop-loss, take-profit, trailing stop, and weak-quote exits. Cached bars are used as background context for momentum, RSI, SMA distance, and volatility so the bot does not fetch 1500 bars every cycle. Weak-quote exits wait for `MIN_HOLD_SECONDS_BEFORE_WEAK_QUOTE_EXIT`. With `PROFIT_ONLY_EXIT_ENABLED=true`, model sells, weak-quote sells, trailing-stop sells, and max-holding sells are held until the estimated exit price is at least `avg_entry_price * (1 + MIN_NET_EXIT_PROFIT_PCT + EXIT_PROFIT_BUFFER_BPS / 10000)`, unless an enabled emergency stop-loss is breached. BUY and non-hard-risk SELL attempts are de-duplicated per latest bar timestamp.

If `ALLOW_EMERGENCY_STOP_LOSS=true`, a loss exit is allowed only after the loss exceeds `EMERGENCY_STOP_LOSS_PCT`. If it is false, loss exits hold with `profit_guard_holding_at_loss`. Trailing stops only arm after unrealized profit reaches `TRAILING_STOP_ARM_PROFIT_PCT`.

The shared Alpaca budget tracks calls by endpoint (`latest_quote`, `crypto_bars`, `position`, `account`, `submit_order`, `get_order`). Near the soft target, optional work is skipped first, such as order-status checks or bars refresh. At the hard stop, new buys are blocked with `api_budget_exhausted`; sells that reduce or close BTC exposure are still prioritized.

Short-term scalping can still lose money from spread, slippage, missed fills, and emergency stop-losses, even in paper trading. The profit guard prevents non-emergency loss exits; it is not a profit guarantee. This project does not support live trading; do not point it at a live Alpaca account.

Account-aware buy risk checks are paper-account only:

```env
MAX_SPREAD_BPS=10
MAX_SLIPPAGE_BPS=8
MIN_QUOTE_IMBALANCE=-0.05
MAX_TRADES_PER_HOUR=1000
MAX_DAILY_TRADES=10000
MAX_ORDER_ATTEMPTS_PER_HOUR=30
MAX_ORDER_ATTEMPTS_PER_DAY=100
MAX_CONSECUTIVE_LOSSES=3
MIN_SECONDS_BETWEEN_TRADES=0

PAUSE_TRADING_ON_ACCOUNT_DRAWDOWN=true
MAX_ACCOUNT_DAILY_LOSS_USD=25
MAX_ACCOUNT_DAILY_LOSS_PCT=0.01
MAX_ACCOUNT_DRAWDOWN_PCT=0.03
REQUIRE_ACCOUNT_DATA_FOR_TRADING=false
```

Account risk blocks only new buys. Sells needed to reduce or close BTC exposure are not blocked by daily trade counts, hourly trade counts, account drawdown, or soft API-budget pressure.

`MAX_TRADES_PER_HOUR` and `MAX_DAILY_TRADES` apply to filled orders. `MAX_ORDER_ATTEMPTS_PER_HOUR` and `MAX_ORDER_ATTEMPTS_PER_DAY` apply to all order rows, including canceled IOC attempts, so noisy retries can be throttled without treating canceled orders as executed trades.

## API

- `GET /health`
- `GET /health/deep`
- `GET /config/safe`
- `GET /position`
- `GET /signals/latest`
- `GET /orders`
- `GET /debug/latest-bars`
- `POST /run-once`
- `POST /auto/start`
- `POST /auto/stop`
- `GET /admin/status`
- `GET /scheduler/status`
- `POST /admin/pause`
- `POST /admin/resume`
- `POST /alerts/discord/test`
- `POST /train`
- `POST /backtest`

Protected endpoints require `X-Admin-Token` matching `API_ADMIN_TOKEN`.

## Dashboard API

The dashboard API provides read-focused backend data for an operator dashboard: bot status, trading status, recent signals, recent paper orders, IOC cancel guard state, realized trade PnL from closed `Trade.pnl` rows, Alpaca paper account snapshots, portfolio/equity history, win rate, average trade PnL, best/worst trade PnL, market freshness, Alpaca API budget telemetry, profit-guard exit status, paper account risk state, latest model promotion metrics, and a dashboard-friendly `/run-once` wrapper. It is for BTC/USD paper-trading analytics only. It does not enable live trading, short selling, margin, or multi-symbol trading.

When Alpaca paper credentials are available, each `run_once` attempts to store a redacted account snapshot with equity, cash, buying power, portfolio value, and currency. Snapshot storage is best-effort and does not interrupt trading decisions. Realized PnL and portfolio history are separate views: `/dashboard/equity-curve` is built from closed trade PnL, while `/dashboard/portfolio-curve` is built from Alpaca paper account snapshots.

`/dashboard/summary` includes `profit_guard_enabled`, `min_net_exit_profit_pct`, `current_unrealized_pnl_pct`, `profit_guard_exit_allowed`, `estimated_exit_price`, and `minimum_profitable_exit_price`. `/dashboard/trading-status` includes the latest decision, latest risk reason, IOC cooldown state, model/fallback status, scheduler running flag, and runtime pause reason so the operator can see why automatic trading is running, waiting, blocked, stopped, or paused.

All dashboard endpoints require the admin token. Do not expose `API_ADMIN_TOKEN` in a public frontend, browser bundle, logs, screenshots, or shared command history.

Endpoints:

- `GET /dashboard/summary`
- `GET /dashboard/signals?limit=100`
- `GET /dashboard/orders?limit=100`
- `GET /dashboard/trades?limit=100`
- `GET /dashboard/equity-curve`
- `GET /dashboard/account-snapshots?limit=500`
- `GET /dashboard/portfolio-curve`
- `GET /dashboard/trading-status`
- `GET /dashboard/market`
- `POST /dashboard/run-once`

Example setup:

```bash
export API_URL="http://localhost:8000"
export ADMIN_TOKEN="replace-with-your-admin-token"
```

Summary:

```bash
curl -sS "$API_URL/dashboard/summary" \
  -H "X-Admin-Token: $ADMIN_TOKEN"
```

Recent signals:

```bash
curl -sS "$API_URL/dashboard/signals?limit=100" \
  -H "X-Admin-Token: $ADMIN_TOKEN"
```

Recent paper orders:

```bash
curl -sS "$API_URL/dashboard/orders?limit=100" \
  -H "X-Admin-Token: $ADMIN_TOKEN"
```

Recent trades:

```bash
curl -sS "$API_URL/dashboard/trades?limit=100" \
  -H "X-Admin-Token: $ADMIN_TOKEN"
```

Equity curve from realized trade PnL:

```bash
curl -sS "$API_URL/dashboard/equity-curve" \
  -H "X-Admin-Token: $ADMIN_TOKEN"
```

Recent Alpaca paper account snapshots:

```bash
curl -sS "$API_URL/dashboard/account-snapshots?limit=500" \
  -H "X-Admin-Token: $ADMIN_TOKEN"
```

Portfolio curve from account snapshots:

```bash
curl -sS "$API_URL/dashboard/portfolio-curve" \
  -H "X-Admin-Token: $ADMIN_TOKEN"
```

Trading status:

```bash
curl -sS "$API_URL/dashboard/trading-status" \
  -H "X-Admin-Token: $ADMIN_TOKEN"
```

Latest market snapshot:

```bash
curl -sS "$API_URL/dashboard/market" \
  -H "X-Admin-Token: $ADMIN_TOKEN"
```

Run one paper-trading decision with dashboard summary:

```bash
curl -sS -X POST "$API_URL/dashboard/run-once" \
  -H "X-Admin-Token: $ADMIN_TOKEN"
```

Unavailable metrics return `null` or an empty list instead of fabricated values. For example, total return percentage is `null` unless it can be computed safely from real data, and the equity curve is empty when there are no recorded trades. Secret-like fields in stored order responses are redacted.

## Web Dashboard

The React web dashboard lives in `frontend/`. It is a private operator UI for BTC/USD paper-trading analytics and controls. It consumes the protected Dashboard API and never hardcodes or commits the admin token.

Install frontend dependencies:

```bash
cd frontend
npm install
```

Run the backend API locally in another terminal:

```bash
cd ..
APP_ENV=development PORT=8000 API_ADMIN_TOKEN=replace-with-a-local-token python scripts/run_api.py
```

Run the frontend locally:

```bash
cd frontend
npm run dev
```

Open the Vite dashboard at:

```text
http://localhost:5173/dashboard-ui/
```

By default, the frontend calls the same origin. During local Vite development, `vite.config.ts` proxies API paths to `http://localhost:8000`, so `VITE_API_BASE_URL` can be left unset.

To point the dashboard at a deployed API, set `VITE_API_BASE_URL`:

```bash
cd frontend
VITE_API_BASE_URL="http://YOUR_SERVER_IP:8000" npm run dev
```

Build the frontend:

```bash
cd frontend
npm run build
```

The production build writes static files to:

```text
frontend/dist
```

When `frontend/dist/index.html` exists, FastAPI serves the built dashboard at:

```text
http://YOUR_API_HOST:8000/dashboard-ui
```

If `frontend/dist` is missing, the backend still starts normally and all API routes continue to work. This keeps local backend development independent from the frontend build.

AWS/Lightsail production flow:

```bash
cd ~/btc-ml-paper-trader
git pull

cd frontend
npm install
npm run build

cd ..
sudo docker compose up -d --build --force-recreate
```

The Docker build copies the repository into the image. If `frontend/dist` exists before `docker compose up --build`, FastAPI can serve it from `/dashboard-ui`. `frontend/node_modules` is ignored by Docker and Git.

Required backend environment variables are still the normal paper-trading settings:

```env
APP_ENV=production
PAPER_TRADING_ONLY=true
API_ADMIN_TOKEN=replace-with-a-long-random-secret
ALPACA_API_KEY=replace-with-your-alpaca-paper-key
ALPACA_SECRET_KEY=replace-with-your-alpaca-paper-secret
ALPACA_PAPER_BASE_URL=https://paper-api.alpaca.markets
ALPACA_DATA_BASE_URL=https://data.alpaca.markets
SYMBOL=BTC/USD
```

The dashboard static files do not contain the admin token. Users enter `API_ADMIN_TOKEN` in the private access screen; protected API requests send it as `X-Admin-Token`.

Admin token login:

- The dashboard first shows a private access screen.
- Enter the same value used by backend `API_ADMIN_TOKEN`.
- The token is stored only in `sessionStorage`, not `localStorage`.
- The token is cleared when you click logout or close the browser tab/session.
- Do not expose the admin token in public frontend config, screenshots, logs, or browser bundles.
- In production, protect the server with HTTPS and restrict port `8000` to trusted IPs or place it behind a reverse proxy such as Nginx/Caddy.

If dashboard API data is missing, charts and tables show empty states instead of fake performance data.

## Release Verification Checklist

Before enabling automatic paper scalping, run the full local checks:

```bash
.venv/bin/python -m pytest

cd frontend
npm install
npm run build
```

Confirm these safety invariants before deployment:

- `PAPER_TRADING_ONLY=true`, `SYMBOL=BTC/USD`, and `ALPACA_PAPER_BASE_URL=https://paper-api.alpaca.markets`.
- No Alpaca keys, Discord webhook URLs, or admin tokens are committed; use placeholders in docs and real values only in private environment variables.
- The strategy remains long-only: BUY only opens a BTC position, SELL only closes an existing BTC position, with no margin, short selling, or multi-symbol support.
- IOC cancel protection is enabled with `IOC_CANCEL_LOOKBACK_SECONDS`, `MAX_RECENT_IOC_CANCELS`, `IOC_CANCEL_COOLDOWN_SECONDS`, and `IOC_CANCEL_ESCALATION_COOLDOWN_SECONDS`.
- Discord risk alerts use `DISCORD_RISK_ALERT_COOLDOWN_SECONDS`, and the runtime circuit breaker can pause repeated loops with `CIRCUIT_BREAKER_ENABLED=true`.
- Dashboard trading status should show the latest decision, risk reason, model/fallback status, IOC cooldown, scheduler state, and runtime pause reason.
- Realized PnL comes only from closed `Trade` rows created from filled BUY/SELL paper order accounting; account equity/portfolio curves come from Alpaca paper account snapshots.

## Risk Design

The bot is long-only:

- Buy is allowed only when no BTC position exists.
- Sell is allowed only to close an existing BTC position.
- No averaging down.
- No short selling.
- No margin.
- Max one open BTC position.

Risk controls include max order notional, max total exposure, daily loss block, drawdown block, profit-only exit guard, explicit emergency stop-loss, take profit, trailing stop, max holding time, and cooldown after loss.

## Feature Engineering

Features are built from BTC/USD OHLCV bars only and use rolling/past data:

- Log returns
- Rolling volatility
- EMA/SMA distances
- RSI, MACD, Bollinger bands, ATR
- Volume normalization and z-score
- Candle range/body features
- Rolling drawdown and trend strength
- Hour, day of week, weekend flag
- Optional spread and quote imbalance when available

Rows without enough historical context are dropped.

## Labels

Labels use a triple-barrier method:

- Look forward `N=12` bars by default.
- Positive if take-profit is hit first.
- Negative if stop-loss is hit first.
- Neutral/no-hit rows become non-buy labels.
- Rows without enough future bars are dropped.

Features are computed without future values; labels are used only for training and validation.

## Backtesting

```bash
python scripts/backtest.py
```

The script fetches or synthesizes BTC/USD bars, builds features and labels, runs walk-forward validation, includes fee/slippage assumptions in the report, and writes:

```text
logs/backtest_report.json
```

Backtest cost assumptions are configurable:

```env
TAKER_FEE_BPS=25
MAKER_FEE_BPS=15
SLIPPAGE_BPS=10
BACKTEST_USE_TAKER_FEES=true
```

The report separates gross and net return metrics. Net metrics subtract configured round-trip fees, configured round-trip slippage, and available `orderbook_spread` costs; if there is not enough validation data, the report returns a clear reason instead of profitability metrics.

Model promotion also checks account-style net performance after costs:

```env
MIN_BACKTEST_NET_RETURN_PCT=0.001
MAX_BACKTEST_DRAWDOWN_PCT=0.01
MIN_BACKTEST_PROFIT_FACTOR=1.05
MIN_BACKTEST_TRADES=20
MODEL_PROMOTION_REQUIRE_POSITIVE_NET_RETURN=true
```

The model does not use account equity as a direct BTC price-prediction feature. Account/equity data is used for risk controls, model evaluation, model promotion/rejection, and dashboard analytics.

## Logs

Structured JSONL events are written to:

```text
logs/events.jsonl
```

Logged events include signals, predictions, rejected orders, submitted orders, risk blocks, model promotions/rejections, training runs, and runtime errors.

## Tests

```bash
pytest
```

Coverage focuses on BTC-only enforcement, long-only behavior, safe defaults, feature generation, leakage resistance, model training output, decision thresholds, and risk blocks.
