from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from typing import Any


DEFAULT_BASE_URL = "http://127.0.0.1:8000"


def main() -> None:
    args = _parse_args()
    token = args.admin_token or os.getenv("ADMIN_TOKEN") or os.getenv("API_ADMIN_TOKEN") or ""
    if not token:
        print(json.dumps({"ok": False, "error": "admin_token_missing"}, indent=2))
        raise SystemExit(1)

    while True:
        report = build_monitor_report(args.base_url, admin_token=token, timeout=args.timeout)
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
        if not args.watch:
            if report["alerts"]:
                raise SystemExit(1)
            return
        time.sleep(max(1.0, args.interval_seconds))


def build_monitor_report(base_url: str, *, admin_token: str, timeout: float = 5.0) -> dict[str, Any]:
    base_url = base_url.rstrip("/")
    summary = _request_json(f"{base_url}/dashboard/summary", admin_token=admin_token, timeout=timeout)
    trading_status = _request_json(f"{base_url}/dashboard/trading-status", admin_token=admin_token, timeout=timeout)
    summary_body = summary.get("body") if isinstance(summary.get("body"), dict) else {}
    status_body = trading_status.get("body") if isinstance(trading_status.get("body"), dict) else {}
    snapshot = {
        "timestamp": datetime.now(UTC).isoformat(),
        "symbol": summary_body.get("symbol"),
        "paper_trading_only": summary_body.get("paper_trading_only"),
        "trading_enabled": summary_body.get("trading_enabled"),
        "auto_trade_enabled": summary_body.get("auto_trade_enabled"),
        "fallback_trading_allowed": summary_body.get("fallback_trading_allowed"),
        "active_model_valid": summary_body.get("active_model_valid"),
        "active_model_invalid_reason": summary_body.get("active_model_invalid_reason"),
        "strategy_name": summary_body.get("latest_strategy_name"),
        "regime": summary_body.get("latest_regime"),
        "final_decision": summary_body.get("final_decision"),
        "blocked_by": summary_body.get("blocked_by"),
        "block_reason": summary_body.get("block_reason"),
        "spread_bps": _nested(summary_body, "latest_signal", "spread_bps"),
        "quote_imbalance": _nested(summary_body, "latest_signal", "quote_imbalance"),
        "current_unrealized_pnl": summary_body.get("current_unrealized_pnl"),
        "realized_pnl_today": summary_body.get("realized_pnl_today"),
        "account_drawdown_pct": summary_body.get("account_drawdown_pct"),
        "api_budget_status": summary_body.get("api_budget_status"),
        "alpaca_budget_remaining": summary_body.get("alpaca_budget_remaining"),
        "ioc_cancel_guard": summary_body.get("ioc_cancel_guard"),
        "trading_state": status_body.get("state"),
        "pause_reason": status_body.get("pause_reason"),
        "latest_decision_reason": status_body.get("latest_decision_reason"),
        "latest_risk_block_reason": status_body.get("latest_risk_block_reason"),
        "data_freshness": summary_body.get("data_freshness"),
    }
    alerts = _alerts(snapshot, summary_ok=summary["ok"], status_ok=trading_status["ok"])
    return {
        "ok": summary["ok"] and trading_status["ok"] and not alerts,
        "base_url": base_url,
        "snapshot": snapshot,
        "alerts": alerts,
        "notes": [
            "Read-only monitor. This script does not modify systemd or .env.",
            "Alerts are stop-and-investigate conditions for paper trading.",
        ],
    }


def _alerts(snapshot: dict[str, Any], *, summary_ok: bool, status_ok: bool) -> list[str]:
    alerts: list[str] = []
    if not summary_ok:
        alerts.append("dashboard_summary_unavailable")
    if not status_ok:
        alerts.append("dashboard_trading_status_unavailable")
    if snapshot.get("paper_trading_only") is not True:
        alerts.append("paper_trading_only_false")
    if snapshot.get("symbol") != "BTC/USD":
        alerts.append("symbol_not_btc_usd")
    if snapshot.get("fallback_trading_allowed") is not False:
        alerts.append("fallback_trading_allowed")
    if snapshot.get("active_model_valid") is not True:
        alerts.append("active_model_invalid")
    if snapshot.get("api_budget_status") == "hard_stop":
        alerts.append("api_budget_hard_stop")
    if snapshot.get("blocked_by") in {"stale_market_data", "active_model_invalid", "fallback_prediction_not_allowed"}:
        alerts.append(f"blocked_by_{snapshot['blocked_by']}")
    ioc_guard = snapshot.get("ioc_cancel_guard") if isinstance(snapshot.get("ioc_cancel_guard"), dict) else {}
    if ioc_guard.get("cooldown_active"):
        alerts.append("ioc_cancel_cooldown_active")
    data_freshness = snapshot.get("data_freshness") if isinstance(snapshot.get("data_freshness"), dict) else {}
    bar_age = data_freshness.get("latest_bar_age_seconds")
    if isinstance(bar_age, (int, float)) and bar_age > 300:
        alerts.append("market_data_stale")
    return alerts


def _request_json(url: str, *, admin_token: str, timeout: float = 5.0) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "X-Admin-Token": admin_token},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return {
                "ok": 200 <= response.status < 300,
                "status_code": response.status,
                "body": json.loads(raw) if raw else None,
            }
    except urllib.error.HTTPError as exc:
        return {"ok": False, "status_code": exc.code, "error": str(exc), "body": _decode_error_body(exc)}
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {"ok": False, "status_code": None, "error": str(exc), "body": None}


def _decode_error_body(exc: urllib.error.HTTPError) -> Any:
    try:
        raw = exc.read().decode("utf-8")
        return json.loads(raw) if raw else None
    except Exception:
        return None


def _nested(mapping: dict[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only BTC/USD paper trading runtime monitor.")
    parser.add_argument("--base-url", default=os.getenv("API_URL", DEFAULT_BASE_URL))
    parser.add_argument("--admin-token", default="")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--watch", action="store_true", help="Poll continuously.")
    parser.add_argument("--interval-seconds", type=float, default=30.0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
