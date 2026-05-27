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

## Network Safety

Port `8000` should ideally be restricted by the Lightsail firewall to trusted IP addresses. A later production setup should place the service behind Nginx with HTTPS.

Keep one running instance only. The app has an in-process scheduler, and multiple instances can duplicate paper-trading scans.
