from functools import lru_cache
from pathlib import Path

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


ALLOWED_SYMBOL = "BTC/USD"
ALPACA_CRYPTO_SYMBOL = "BTC/USD"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "development"
    paper_trading_only: bool = True
    trading_enabled: bool = False
    auto_trade_enabled: bool = False
    api_admin_token: str = ""

    discord_webhook_url: str = ""
    discord_alerts_enabled: bool = False
    discord_alert_on_hold: bool = False
    discord_alert_on_signal: bool = True
    discord_alert_on_order: bool = True
    discord_alert_on_error: bool = True
    discord_alert_on_model: bool = False
    discord_risk_alert_cooldown_seconds: int = 300

    circuit_breaker_enabled: bool = True
    max_same_risk_blocks_before_pause: int = 20
    max_runtime_errors_before_pause: int = 10
    circuit_breaker_window_seconds: int = 900

    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_paper_base_url: str = "https://paper-api.alpaca.markets"
    alpaca_data_base_url: str = "https://data.alpaca.markets"
    alpaca_rate_limit_enabled: bool = True
    alpaca_max_calls_per_minute: int = 180
    alpaca_api_budget_target_per_minute: int = 170
    alpaca_api_budget_hard_stop_per_minute: int = 195

    symbol: str = ALLOWED_SYMBOL
    timeframe: str = "1Min"
    lookback_bars: int = 1500
    scan_interval_seconds: int = 1
    market_bars_cache_seconds: int = 30
    position_cache_seconds: int = 2
    account_equity_cache_seconds: int = 5
    quote_cache_seconds: int = 0

    database_url: str = "sqlite:///./data/trading.db"
    model_dir: str = "models"
    log_dir: str = "logs"

    order_notional_usd: float = 25
    order_type: str = "limit"
    time_in_force: str = "ioc"
    limit_price_offset_bps: float = 2
    ioc_cancel_lookback_seconds: int = 300
    max_recent_ioc_cancels: int = 3
    ioc_cancel_cooldown_seconds: int = 120
    ioc_cancel_escalation_cooldown_seconds: int = 600
    max_position_notional_usd: float = 100
    max_total_exposure_usd: float = 100
    max_daily_loss_usd: float = 20
    max_drawdown_pct: float = 0.05
    max_open_positions: int = 1

    scalping_mode_enabled: bool = False
    max_spread_bps: float = 10
    max_slippage_bps: float = 8
    min_quote_imbalance: float = -0.05
    max_trades_per_hour: int = 1000
    max_daily_trades: int = 10000
    max_order_attempts_per_hour: int = 30
    max_order_attempts_per_day: int = 100
    max_consecutive_losses: int = 3
    min_seconds_between_trades: int = 0

    scalping_entry_dip_pct: float = 0.0005
    scalping_take_profit_pct: float = 0.0015
    scalping_stop_loss_pct: float = 0.001
    scalping_trailing_stop_pct: float = 0.0008
    scalping_min_momentum_pct: float = -0.0005
    scalping_max_position_seconds: int = 90
    scalping_buy_probability_floor: float = 0.50
    scalping_confidence_gap_required: float = 0.04
    scalping_sell_on_weak_quote: bool = True
    scalping_quote_imbalance_exit: float = -0.10
    min_hold_seconds_before_weak_quote_exit: int = 30

    profit_only_exit_enabled: bool = True
    min_net_exit_profit_pct: float = 0.002
    exit_profit_buffer_bps: float = 5
    allow_emergency_stop_loss: bool = True
    emergency_stop_loss_pct: float = 0.006
    trailing_stop_arm_profit_pct: float = 0.002
    model_sell_requires_profit: bool = True
    weak_quote_sell_requires_profit: bool = True
    max_holding_sell_requires_profit: bool = True

    order_in_flight_timeout_seconds: int = 15
    order_status_check_enabled: bool = True
    order_status_check_delay_seconds: float = 0.5

    pause_trading_on_account_drawdown: bool = True
    max_account_daily_loss_usd: float = 25
    max_account_daily_loss_pct: float = 0.01
    max_account_drawdown_pct: float = 0.03
    require_account_data_for_trading: bool = False

    taker_fee_bps: float = 25
    maker_fee_bps: float = 15
    slippage_bps: float = 10
    backtest_use_taker_fees: bool = True
    paper_fee_bps: float = 0
    paper_slippage_bps: float = 0

    min_buy_probability: float = 0.58
    min_sell_probability: float = 0.55
    confidence_gap_required: float = 0.08

    stop_loss_pct: float = 0.015
    take_profit_pct: float = 0.03
    trailing_stop_pct: float = 0.02
    max_holding_minutes: int = 720

    ml_enabled: bool = True
    allow_fallback_trading: bool = False
    model_retrain_enabled: bool = True
    retrain_every_hours: int = 24
    min_training_rows: int = 1000
    optuna_enabled: bool = False

    min_precision_for_promotion: float = 0.52
    max_validation_drawdown_pct: float = 0.20
    max_trade_fraction: float = 0.40
    min_backtest_net_return_pct: float = 0.001
    max_backtest_drawdown_pct: float = 0.01
    min_backtest_profit_factor: float = 1.05
    min_backtest_trades: int = 20
    model_promotion_require_positive_net_return: bool = True

    @field_validator("symbol")
    @classmethod
    def symbol_must_be_btc_usd(cls, value: str) -> str:
        if value != ALLOWED_SYMBOL:
            raise ValueError("This project is hard-limited to BTC/USD only.")
        return value

    @field_validator("order_type")
    @classmethod
    def order_type_must_be_supported(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"market", "limit"}:
            raise ValueError("ORDER_TYPE must be market or limit.")
        return normalized

    @field_validator("time_in_force")
    @classmethod
    def time_in_force_must_be_supported(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"gtc", "ioc"}:
            raise ValueError("TIME_IN_FORCE must be gtc or ioc.")
        return normalized

    @field_validator("taker_fee_bps", "maker_fee_bps", "slippage_bps", "paper_fee_bps", "paper_slippage_bps")
    @classmethod
    def cost_bps_must_be_non_negative(cls, value: float) -> float:
        if value < 0:
            raise ValueError("Fee and slippage bps values must be non-negative.")
        return value

    @field_validator(
        "alpaca_max_calls_per_minute",
        "alpaca_api_budget_target_per_minute",
        "alpaca_api_budget_hard_stop_per_minute",
        "market_bars_cache_seconds",
        "position_cache_seconds",
        "account_equity_cache_seconds",
        "quote_cache_seconds",
        "order_in_flight_timeout_seconds",
        "order_status_check_delay_seconds",
        "ioc_cancel_lookback_seconds",
        "max_recent_ioc_cancels",
        "ioc_cancel_cooldown_seconds",
        "ioc_cancel_escalation_cooldown_seconds",
        "discord_risk_alert_cooldown_seconds",
        "max_same_risk_blocks_before_pause",
        "max_runtime_errors_before_pause",
        "circuit_breaker_window_seconds",
        "max_order_attempts_per_hour",
        "max_order_attempts_per_day",
        "scalping_max_position_seconds",
        "min_hold_seconds_before_weak_quote_exit",
        "max_account_daily_loss_usd",
        "max_account_daily_loss_pct",
        "max_account_drawdown_pct",
        "min_backtest_net_return_pct",
        "max_backtest_drawdown_pct",
        "min_backtest_profit_factor",
        "min_backtest_trades",
        "min_net_exit_profit_pct",
        "exit_profit_buffer_bps",
        "emergency_stop_loss_pct",
        "trailing_stop_arm_profit_pct",
    )
    @classmethod
    def runtime_timing_values_must_be_non_negative(cls, value: float) -> float:
        if value < 0:
            raise ValueError("Runtime timing and rate limit values must be non-negative.")
        return value

    @field_validator("alpaca_max_calls_per_minute")
    @classmethod
    def alpaca_rate_limit_must_be_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("ALPACA_MAX_CALLS_PER_MINUTE must be positive.")
        return value

    @model_validator(mode="after")
    def enforce_api_budget_sanity(self) -> "Settings":
        if self.alpaca_api_budget_hard_stop_per_minute >= 200:
            raise ValueError("ALPACA_API_BUDGET_HARD_STOP_PER_MINUTE must stay below 200.")
        if self.alpaca_api_budget_target_per_minute >= self.alpaca_api_budget_hard_stop_per_minute:
            raise ValueError("ALPACA_API_BUDGET_TARGET_PER_MINUTE must be below hard stop.")
        return self

    @model_validator(mode="after")
    def enforce_paper_only(self) -> "Settings":
        if not self.paper_trading_only:
            raise ValueError("PAPER_TRADING_ONLY must remain true; live trading is not implemented.")
        if "paper-api.alpaca.markets" not in self.alpaca_paper_base_url:
            raise ValueError("Only Alpaca paper trading base URL is allowed.")
        if self.max_open_positions != 1:
            raise ValueError("BTC-only MVP supports exactly one open position.")
        return self

    @property
    def model_path(self) -> Path:
        return Path(self.model_dir)

    @property
    def log_path(self) -> Path:
        return Path(self.log_dir)

    def safe_dict(self) -> dict:
        data = self.model_dump()
        data["alpaca_api_key"] = "***" if self.alpaca_api_key else ""
        data["alpaca_secret_key"] = "***" if self.alpaca_secret_key else ""
        data["api_admin_token"] = "***" if self.api_admin_token else ""
        data["discord_webhook_url"] = "***" if self.discord_webhook_url else ""
        return data


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    Path(settings.model_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.log_dir).mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(parents=True, exist_ok=True)
    return settings
