# Safe Lightsail Systemd Rollout and Rollback

This checklist is for the BTC/USD paper-only deployment. It does not add live trading, margin, short selling, multi-symbol trading, fallback trading, or automatic systemd changes.

Assumptions:

```bash
export APP_DIR=/opt/btc-ml-paper-trader
export SERVICE=btc-ml-paper-trader
export API_URL=http://127.0.0.1:8000
export ADMIN_TOKEN=replace-with-your-runtime-token
```

## 1. Backup `.env`

```bash
cd "$APP_DIR"
mkdir -p backups
cp .env "backups/.env.$(date -u +%Y%m%dT%H%M%SZ).bak"
ls -lh backups/.env.*.bak | tail
```

Do not edit production `.env` during rollout unless you are intentionally making a reviewed configuration change.

## 2. Stop Service

```bash
sudo systemctl status "$SERVICE" --no-pager
sudo systemctl stop "$SERVICE"
sudo systemctl status "$SERVICE" --no-pager
```

## 3. Pull Code

```bash
cd "$APP_DIR"
git status --short
git rev-parse HEAD > backups/previous_git_sha.txt
git pull --ff-only
git rev-parse HEAD
```

If `git pull --ff-only` fails, stop and inspect the server worktree.

## 4. Install and Test

```bash
cd "$APP_DIR"
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

Do not continue if tests fail.

## 5. Diagnose Labels

```bash
python scripts/diagnose_labels.py
```

Stop if labels are obviously unhealthy, positive buy labels are too low, or required training columns are broken.

## 6. Train Model

```bash
python scripts/train_model.py
```

Training may reject the model. A rejection is safer than promoting a weak model. Do not override promotion guards.

## 7. Verify `registry.json`

```bash
test -s models/registry.json
python - <<'PY'
import json
from pathlib import Path
path = Path("models/registry.json")
data = json.loads(path.read_text())
print("registry:", path)
print("active_model_path:", data.get("active_model_path"))
print("model_version:", data.get("model_version"))
print("promotion_reason:", data.get("promotion_reason") or data.get("metrics", {}).get("promotion_reason"))
PY
```

If no valid registry exists, keep automatic paper trading disabled until the model registry is healthy.

## 8. Verify Safe Config

Start the API temporarily only if needed for config verification, or run this after the service is started in a later step.

```bash
curl -fsS "$API_URL/health"
curl -fsS "$API_URL/health/deep"
curl -fsS "$API_URL/config/safe" \
  -H "X-Admin-Token: $ADMIN_TOKEN" | python -m json.tool
```

Confirm:

- `paper_trading_only=true`
- `symbol=BTC/USD`
- `allow_fallback_trading=false`
- `active_model_valid=true` before enabling automatic paper trading
- no secrets are printed

## 9. Run One Safe Decision

Run a local one-shot decision with order submission disabled by environment override:

```bash
cd "$APP_DIR"
source .venv/bin/activate
TRADING_ENABLED=false AUTO_TRADE_ENABLED=false ALLOW_FALLBACK_TRADING=false python scripts/run_once.py
```

Do not use API `/run-once` as a pre-start smoke test unless you have explicitly verified production trading flags cannot submit an order.

## 10. Start Service

```bash
sudo systemctl start "$SERVICE"
sudo systemctl status "$SERVICE" --no-pager
```

## 11. Monitor

```bash
journalctl -u "$SERVICE" -n 100 --no-pager
journalctl -u "$SERVICE" -f
```

Optional read-only runtime checks:

```bash
python scripts/health_check_runtime.py --base-url "$API_URL" --admin-token "$ADMIN_TOKEN"
python scripts/monitor_paper_trading.py --base-url "$API_URL" --admin-token "$ADMIN_TOKEN" --watch
```

Watch for:

- fallback prediction source
- `active_model_valid=false`
- `model_unavailable`
- stale market data
- repeated IOC cancels
- spread or quote imbalance blocks
- API budget hard stops
- account drawdown blocks
- circuit breaker pauses
- runtime errors

## Rollback

Use rollback if tests fail, runtime health degrades, the model registry is invalid, or safety guards begin blocking repeatedly for unexpected reasons.

```bash
cd "$APP_DIR"
sudo systemctl stop "$SERVICE"

# Restore previous code.
git fetch --all --prune
git checkout "$(cat backups/previous_git_sha.txt)"

# Restore the last known-good environment backup manually.
ls -lh backups/.env.*.bak | tail
cp backups/.env.YYYYMMDDTHHMMSSZ.bak .env

source .venv/bin/activate
pip install -r requirements.txt
pytest
python scripts/diagnose_labels.py
python scripts/train_model.py

sudo systemctl start "$SERVICE"
sudo systemctl status "$SERVICE" --no-pager
journalctl -u "$SERVICE" -n 100 --no-pager
```

Rollback does not guarantee profitability. It only returns code and configuration to a previously selected state. Keep the deployment paper-only, BTC/USD-only, long-only, and fallback-trading disabled.
