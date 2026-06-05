from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any


DEFAULT_BASE_URL = "http://127.0.0.1:8000"


def main() -> None:
    args = _parse_args()
    token = args.admin_token or os.getenv("ADMIN_TOKEN") or os.getenv("API_ADMIN_TOKEN") or ""
    report = build_runtime_health_report(args.base_url, admin_token=token, timeout=args.timeout)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    if not report["ok"]:
        raise SystemExit(1)


def build_runtime_health_report(base_url: str, *, admin_token: str = "", timeout: float = 5.0) -> dict[str, Any]:
    base_url = base_url.rstrip("/")
    public_health = _request_json(f"{base_url}/health", timeout=timeout)
    deep_health = _request_json(f"{base_url}/health/deep", timeout=timeout)
    safe_config = (
        _request_json(f"{base_url}/config/safe", admin_token=admin_token, timeout=timeout)
        if admin_token
        else {"ok": False, "error": "admin_token_missing", "body": None}
    )
    checks = {
        "health_endpoint_ok": public_health["ok"],
        "deep_health_endpoint_ok": deep_health["ok"],
        "config_safe_endpoint_ok": safe_config["ok"],
        "paper_trading_only": _body_value(public_health, "paper_trading_only") is True
        and _body_value(safe_config, "paper_trading_only") is True,
        "btc_usd_only": _body_value(public_health, "symbol") == "BTC/USD"
        and _body_value(safe_config, "symbol") == "BTC/USD",
        "fallback_trading_disabled": _body_value(safe_config, "allow_fallback_trading") is False,
        "no_live_trading_switch": _body_value(safe_config, "paper_trading_only") is True,
    }
    active_model_valid = _body_value(safe_config, "active_model_valid")
    checks["active_model_valid"] = active_model_valid is True
    checks["safe_for_paper_auto_trading"] = all(
        checks[name]
        for name in [
            "health_endpoint_ok",
            "deep_health_endpoint_ok",
            "config_safe_endpoint_ok",
            "paper_trading_only",
            "btc_usd_only",
            "fallback_trading_disabled",
            "active_model_valid",
        ]
    )
    return {
        "ok": checks["health_endpoint_ok"] and checks["config_safe_endpoint_ok"],
        "base_url": base_url,
        "checks": checks,
        "health": public_health,
        "deep_health": deep_health,
        "safe_config_summary": _safe_config_summary(safe_config.get("body")),
        "notes": [
            "Read-only check. This script does not modify systemd or .env.",
            "safe_for_paper_auto_trading=false means investigate before enabling scheduler automation.",
        ],
    }


def _request_json(url: str, *, admin_token: str = "", timeout: float = 5.0) -> dict[str, Any]:
    headers = {"Accept": "application/json"}
    if admin_token:
        headers["X-Admin-Token"] = admin_token
    request = urllib.request.Request(url, headers=headers, method="GET")
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


def _body_value(response: dict[str, Any], key: str) -> Any:
    body = response.get("body")
    return body.get(key) if isinstance(body, dict) else None


def _safe_config_summary(body: Any) -> dict[str, Any]:
    if not isinstance(body, dict):
        return {}
    keys = [
        "paper_trading_only",
        "symbol",
        "trading_enabled",
        "auto_trade_enabled",
        "scalping_mode_enabled",
        "allow_fallback_trading",
        "active_model_valid",
        "active_model_invalid_reason",
        "active_model_version",
    ]
    return {key: body.get(key) for key in keys}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only runtime health check for the BTC/USD paper trader API.")
    parser.add_argument("--base-url", default=os.getenv("API_URL", DEFAULT_BASE_URL))
    parser.add_argument("--admin-token", default="")
    parser.add_argument("--timeout", type=float, default=5.0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
