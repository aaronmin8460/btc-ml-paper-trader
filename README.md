# btc-ml-paper-trader

Automatic BTC/USD-only paper trading bot for Alpaca paper trading.

This project is intentionally paper-only, long-only, and BTC/USD-only. It does not implement live trading, short selling, margin trading, or multi-symbol trading.

## Safety Defaults

The default `.env.example` is safe:

```env
PAPER_TRADING_ONLY=true
TRADING_ENABLED=false
AUTO_TRADE_ENABLED=false
SYMBOL=BTC/USD
```

There is no live-trading switch. The Alpaca broker client uses only the paper API base URL and hard-validates every order request before submission.

## Setup

```bash
cd btc-ml-paper-trader
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add Alpaca paper credentials to `.env` if you want real paper account data or paper orders. Without credentials, the bot can still train and produce signals using deterministic synthetic BTC-like bars for local development.

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

The trainer builds leakage-safe OHLCV features, applies triple-barrier labels, runs walk-forward validation, saves a model under `models/`, and promotes it to `models/registry.json` only if validation metrics pass. It does not fabricate profitability or hardcode a winning backtest.

## ML Model

`MLSignalModel` combines:

- LogisticRegression with StandardScaler
- RandomForestClassifier
- HistGradientBoostingClassifier
- Weighted average probability output

Feature columns are tracked and saved with the model. Random train/test splits are avoided; validation is walk-forward.

## Run Once

```bash
python scripts/run_once.py
```

With `TRADING_ENABLED=false`, this produces a BTC/USD signal and does not submit a paper order.

To allow a manual paper order through the API, set:

```env
TRADING_ENABLED=true
AUTO_TRADE_ENABLED=false
API_ADMIN_TOKEN=change-me
```

Then call:

```bash
python scripts/run_api.py
curl -X POST http://localhost:8000/run-once -H 'X-Admin-Token: change-me'
```

Only BTC/USD paper market orders can pass the broker guard.

## Automatic Paper Trading

Set:

```env
TRADING_ENABLED=true
AUTO_TRADE_ENABLED=true
API_ADMIN_TOKEN=change-me
```

Start the API:

```bash
python scripts/run_api.py
```

The in-process scheduler runs every `SCAN_INTERVAL_SECONDS`, catches exceptions, logs runtime errors, and calls `Trader.run_once()`.

## Railway Deployment

This FastAPI server can be deployed to Railway or a similar platform with the included `railway.toml`. Railway should use Nixpacks automatically and start the app with:

```bash
python scripts/run_api.py
```

The server binds to `0.0.0.0` and reads the platform-provided `PORT` environment variable. Local development still defaults to port `8000`.

Local development can use the default SQLite database:

```env
DATABASE_URL=sqlite:///./data/trading.db
```

Recommended Railway steps:

1. Create a new Railway project from this GitHub repository.
2. Add the required environment variables in Railway.
3. Deploy with one replica/instance only.
4. Confirm the health check at `/health`.

Required environment variables for deployment:

```env
APP_ENV=production
PAPER_TRADING_ONLY=true
TRADING_ENABLED=false
AUTO_TRADE_ENABLED=false
SYMBOL=BTC/USD
API_ADMIN_TOKEN=change-me
```

Add Alpaca paper credentials only if you want the deployment to read real paper account data or submit paper orders:

```env
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
ALPACA_PAPER_BASE_URL=https://paper-api.alpaca.markets
ALPACA_DATA_BASE_URL=https://data.alpaca.markets
```

For SQLite on Railway, attach a Railway Volume mounted at `/data` and use:

```env
DATABASE_URL=sqlite:////data/trading.db
MODEL_DIR=/data/models
LOG_DIR=/data/logs
```

Future production analytics can use PostgreSQL with SQLAlchemy's psycopg driver, for example:

```env
DATABASE_URL=postgresql+psycopg://user:password@host:5432/database
```

Keep `PAPER_TRADING_ONLY=true` and `SYMBOL=BTC/USD`. The app rejects live trading URLs, non-BTC/USD symbols, and configurations with more than one open position. Do not run more than one Railway replica/instance: automatic paper trading uses an in-process scheduler, so multiple instances could run duplicate scans.

## API

- `GET /health`
- `GET /config/safe`
- `GET /position`
- `GET /signals/latest`
- `GET /orders`
- `POST /run-once`
- `POST /auto/start`
- `POST /auto/stop`
- `POST /train`
- `POST /backtest`

Write endpoints require `X-Admin-Token` matching `API_ADMIN_TOKEN`.

## Risk Design

The bot is long-only:

- Buy is allowed only when no BTC position exists.
- Sell is allowed only to close an existing BTC position.
- No averaging down.
- No short selling.
- No margin.
- Max one open BTC position.

Risk controls include max order notional, max total exposure, daily loss block, drawdown block, stop loss, take profit, trailing stop, max holding time, and cooldown after loss.

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
# btc-ml-paper-trader
