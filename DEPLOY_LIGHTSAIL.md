# Deploy On AWS Lightsail With Docker Compose

This guide deploys the BTC/USD-only, long-only, paper-trading-only FastAPI bot on an AWS Lightsail VPS.

Do not commit real Alpaca keys, Discord webhook URLs, admin tokens, or `.env` files.

## Clone The Repository

```bash
git clone https://github.com/aaronmin8460/btc-ml-paper-trader.git
cd btc-ml-paper-trader
```

## Create The Environment File

```bash
cp .env.example .env
```

Generate a strong admin token:

```bash
openssl rand -hex 32
```

Edit `.env` and set `API_ADMIN_TOKEN` to the generated value. Add Alpaca paper credentials and an optional Discord webhook URL only in `.env`.

Keep the first deployment safe:

```env
PAPER_TRADING_ONLY=true
TRADING_ENABLED=false
AUTO_TRADE_ENABLED=false
SYMBOL=BTC/USD
ALPACA_PAPER_BASE_URL=https://paper-api.alpaca.markets
```

For two-second REST scalping paper trading, use settings like:

```env
TIMEFRAME=1Min
LOOKBACK_BARS=1500
SCAN_INTERVAL_SECONDS=2
SCALPING_MODE_ENABLED=true
ORDER_TYPE=limit
TIME_IN_FORCE=ioc
ALPACA_RATE_LIMIT_ENABLED=true
ALPACA_MAX_CALLS_PER_MINUTE=160
MARKET_BARS_CACHE_SECONDS=20
POSITION_CACHE_SECONDS=5
QUOTE_CACHE_SECONDS=0
MIN_SECONDS_BETWEEN_TRADES=10
MAX_TRADES_PER_HOUR=20
MAX_DAILY_TRADES=100
```

This is still REST polling, not true HFT. The cache and rate limiter are there to keep the bot below Alpaca API limits while scanning quickly.

## Create Runtime Directories

```bash
mkdir -p runtime/data runtime/logs runtime/models
```

These directories are mounted into the container for SQLite data, logs, and model artifacts.

## Build And Run

```bash
docker compose up -d --build
```

Check container logs:

```bash
docker compose logs -f
```

On some Lightsail hosts, Docker is run with `sudo`:

```bash
sudo docker compose logs -f
tail -f runtime/logs/events.jsonl
```

## Test The Server

Health check:

```bash
curl http://YOUR_STATIC_IP:8000/health
```

Latest BTC/USD bars check:

```bash
curl -X GET "http://YOUR_STATIC_IP:8000/debug/latest-bars" \
  -H "X-Admin-Token: YOUR_TOKEN"
```

The `latest_timestamp` should be near the current UTC time, not 2022.

Manual run-once check:

```bash
curl -X POST "http://YOUR_STATIC_IP:8000/run-once" \
  -H "X-Admin-Token: YOUR_TOKEN"
```

With `TRADING_ENABLED=false`, this should return a BTC/USD decision without submitting a paper order.

Discord test:

```bash
curl -X POST "http://YOUR_STATIC_IP:8000/alerts/discord/test" \
  -H "X-Admin-Token: YOUR_TOKEN"
```

Recommended safe startup sequence:

1. Keep `TRADING_ENABLED=false` and `AUTO_TRADE_ENABLED=false`.
2. Test `/debug/latest-bars`.
3. Test `/run-once`.
4. Test Discord.
5. Set `TRADING_ENABLED=true` and keep `AUTO_TRADE_ENABLED=false`.
6. Manually call `/run-once` and verify the Alpaca paper order.
7. Set `AUTO_TRADE_ENABLED=true` only after the manual paper order works.

## Enable Automatic Paper Trading Later

After health, config masking, latest-bars, Discord alerts, and manual run-once all work, you may enable automatic paper trading by updating `.env`:

```env
TRADING_ENABLED=true
AUTO_TRADE_ENABLED=true
```

Then restart:

```bash
docker compose up -d
```

This still uses Alpaca paper trading only. Do not add or use Alpaca live trading URLs.

Stop automation without stopping the container:

```bash
curl -X POST "http://YOUR_STATIC_IP:8000/auto/stop" \
  -H "X-Admin-Token: YOUR_TOKEN"
```

If API pressure is high, reduce scan frequency and the rate budget:

```env
SCAN_INTERVAL_SECONDS=5
ALPACA_MAX_CALLS_PER_MINUTE=120
```

## Network Safety

Port `8000` should ideally be restricted by the Lightsail firewall to trusted IP addresses. A later production setup should place the service behind Nginx with HTTPS.

Keep one running instance only. The app has an in-process scheduler, and multiple instances can duplicate paper-trading scans.
