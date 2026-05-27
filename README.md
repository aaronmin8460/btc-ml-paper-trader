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
SYMBOL=BTC/USD
```

## Local dry-run

Run one BTC/USD decision locally without submitting an order:

```bash
source .venv/bin/activate
TRADING_ENABLED=false AUTO_TRADE_ENABLED=false python scripts/run_once.py
```

With `TRADING_ENABLED=false`, the bot fetches or synthesizes BTC/USD data, produces a signal, and does not submit a paper order.

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
3. Store it only in your local `.env` or Railway environment variables.
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

## Railway deployment

This FastAPI server is deployable on Railway with the included `railway.toml`.

Railway should use Nixpacks and start the app with:

```bash
python scripts/run_api.py
```

The server binds to `0.0.0.0` and reads Railway's `PORT` environment variable. In production, set `APP_ENV=production` so reload stays disabled.

Recommended Railway steps:

1. Create a new Railway project from this GitHub repository.
2. Add the environment variables from the sections below.
3. Attach a Railway Volume if using SQLite persistence.
4. Deploy with exactly one replica.
5. Verify `/health`.
6. Test Discord alerts.
7. Run `/run-once` manually before enabling automatic paper trading.

## Railway environment variables

Use Railway's Variables UI. Values below are examples or placeholders; do not paste real keys into the README or commit them to git.

Minimum safe deployment:

```env
APP_ENV=production
PAPER_TRADING_ONLY=true
TRADING_ENABLED=false
AUTO_TRADE_ENABLED=false
SYMBOL=BTC/USD
API_ADMIN_TOKEN=replace-with-a-long-random-secret
ALPACA_PAPER_BASE_URL=https://paper-api.alpaca.markets
ALPACA_DATA_BASE_URL=https://data.alpaca.markets
```

Add Alpaca paper credentials only as Railway variables:

```env
ALPACA_API_KEY=replace-with-your-alpaca-paper-key
ALPACA_SECRET_KEY=replace-with-your-alpaca-paper-secret
```

Optional Discord variables:

```env
DISCORD_WEBHOOK_URL=replace-with-your-discord-webhook-url
DISCORD_ALERTS_ENABLED=true
DISCORD_ALERT_ON_HOLD=false
DISCORD_ALERT_ON_SIGNAL=true
DISCORD_ALERT_ON_ORDER=true
DISCORD_ALERT_ON_ERROR=true
DISCORD_ALERT_ON_MODEL=false
```

Recommended BTC/USD scalping paper-trading variables:

```env
TIMEFRAME=1Min
LOOKBACK_BARS=1500
SCAN_INTERVAL_SECONDS=5
ORDER_NOTIONAL_USD=10
STOP_LOSS_PCT=0.0025
TAKE_PROFIT_PCT=0.006
TRAILING_STOP_PCT=0.003
MAX_HOLDING_MINUTES=15
```

## Railway Volume setup for SQLite persistence

Local development can use SQLite at the default path:

```env
DATABASE_URL=sqlite:///./data/trading.db
```

For Railway SQLite persistence:

1. Add a Railway Volume to the service.
2. Mount it at `/data`.
3. Set these Railway variables:

```env
DATABASE_URL=sqlite:////data/trading.db
MODEL_DIR=/data/models
LOG_DIR=/data/logs
```

PostgreSQL can be used later for production analytics, but it is not required for the paper-trading deployment:

```env
DATABASE_URL=postgresql+psycopg://replace-with-user:replace-with-password@replace-with-host:5432/replace-with-database
```

## Safe first deployment

Deploy first with trading and automatic scheduling disabled:

```env
TRADING_ENABLED=false
AUTO_TRADE_ENABLED=false
```

This lets you verify health, config masking, Discord alerts, and one manual `/run-once` without submitting paper orders.

## Testing deployed server

Set your Railway URL and admin token in your shell:

```bash
export API_URL="https://replace-with-your-railway-domain"
export ADMIN_TOKEN="replace-with-your-api-admin-token"
```

Health check:

```bash
curl -sS "$API_URL/health"
```

Safe config check. This must not expose Alpaca keys, the admin token, or the Discord webhook URL:

```bash
curl -sS "$API_URL/config/safe"
```

Discord test alert:

```bash
curl -sS -X POST "$API_URL/alerts/discord/test" \
  -H "X-Admin-Token: $ADMIN_TOKEN"
```

Manual run-once test:

```bash
curl -sS -X POST "$API_URL/run-once" \
  -H "X-Admin-Token: $ADMIN_TOKEN"
```

Expected safety signals:

- `/health` returns `status: ok`, `paper_trading_only: true`, and `symbol: BTC/USD`.
- `/config/safe` masks secrets with `***`.
- `/alerts/discord/test` returns either `{"sent":true}` or `{"sent":false,"reason":"discord_disabled"}`.
- `/run-once` returns a BTC/USD decision. With `TRADING_ENABLED=false`, `order` should be `null`.

## Enabling automatic paper trading

After the safe first deployment succeeds, enable paper trading and the in-process scheduler:

```env
TRADING_ENABLED=true
AUTO_TRADE_ENABLED=true
```

Keep these safety variables unchanged:

```env
PAPER_TRADING_ONLY=true
SYMBOL=BTC/USD
ALPACA_PAPER_BASE_URL=https://paper-api.alpaca.markets
```

Redeploy after changing Railway variables. The scheduler runs every `SCAN_INTERVAL_SECONDS` and calls `Trader.run_once()`.

## Important safety warnings

- Use one Railway replica only. The app has an in-process scheduler, and multiple replicas can duplicate scans.
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

The trainer builds leakage-safe OHLCV features, applies triple-barrier labels, runs walk-forward validation, saves a model under `models/`, and promotes it to `models/registry.json` only if validation metrics pass. It does not fabricate profitability or hardcode a winning backtest.

## ML Model

`MLSignalModel` combines:

- LogisticRegression with StandardScaler
- RandomForestClassifier
- HistGradientBoostingClassifier
- Weighted average probability output

Feature columns are tracked and saved with the model. Random train/test splits are avoided; validation is walk-forward.

## Order Execution

Orders default to market/GTC paper orders:

```env
ORDER_TYPE=market
TIME_IN_FORCE=gtc
LIMIT_PRICE_OFFSET_BPS=2
```

Optional limit/IOC paper orders can be enabled with `ORDER_TYPE=limit` and `TIME_IN_FORCE=ioc`. Limit prices are derived from the latest BTC/USD quote with `LIMIT_PRICE_OFFSET_BPS`; if a valid quote is unavailable, the order is blocked instead of falling back to market execution.

## Scalping Configuration

Scalping mode is a configuration profile for faster BTC/USD paper-trading experiments. It is disabled by default and these settings are not yet enforced in strategy logic; they are available for a later risk/strategy phase without changing the current trading behavior.

Recommended paper-trading settings for a scalping experiment:

```env
SCALPING_MODE_ENABLED=true
TIMEFRAME=1Min
LOOKBACK_BARS=1500
SCAN_INTERVAL_SECONDS=5
ORDER_NOTIONAL_USD=10
STOP_LOSS_PCT=0.0025
TAKE_PROFIT_PCT=0.006
TRAILING_STOP_PCT=0.003
MAX_HOLDING_MINUTES=15
```

The scalping guard configuration is intentionally conservative by default:

```env
MAX_SPREAD_BPS=8
MAX_SLIPPAGE_BPS=10
MIN_QUOTE_IMBALANCE=-0.25
MAX_TRADES_PER_HOUR=10
MAX_DAILY_TRADES=30
MAX_CONSECUTIVE_LOSSES=3
MIN_SECONDS_BETWEEN_TRADES=30
```

## API

- `GET /health`
- `GET /config/safe`
- `GET /position`
- `GET /signals/latest`
- `GET /orders`
- `GET /debug/latest-bars`
- `POST /run-once`
- `POST /auto/start`
- `POST /auto/stop`
- `POST /alerts/discord/test`
- `POST /train`
- `POST /backtest`

Protected endpoints require `X-Admin-Token` matching `API_ADMIN_TOKEN`.

## Dashboard API

The dashboard API provides read-focused backend data for an operator dashboard: bot status, recent signals, recent paper orders, realized trade PnL from stored `Trade.pnl` rows, market freshness, and a dashboard-friendly `/run-once` wrapper. It is for BTC/USD paper-trading analytics only. It does not enable live trading, short selling, margin, or multi-symbol trading.

All dashboard endpoints require the admin token. Do not expose `API_ADMIN_TOKEN` in a public frontend, browser bundle, logs, screenshots, or shared command history.

Endpoints:

- `GET /dashboard/summary`
- `GET /dashboard/signals?limit=100`
- `GET /dashboard/orders?limit=100`
- `GET /dashboard/trades?limit=100`
- `GET /dashboard/equity-curve`
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

Backtest cost assumptions are configurable:

```env
TAKER_FEE_BPS=25
MAKER_FEE_BPS=15
SLIPPAGE_BPS=10
BACKTEST_USE_TAKER_FEES=true
```

The report separates gross and net return metrics. Net metrics subtract configured round-trip fees, configured round-trip slippage, and available `orderbook_spread` costs; if there is not enough validation data, the report returns a clear reason instead of profitability metrics.

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
