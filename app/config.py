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

    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_paper_base_url: str = "https://paper-api.alpaca.markets"
    alpaca_data_base_url: str = "https://data.alpaca.markets"

    symbol: str = ALLOWED_SYMBOL
    timeframe: str = "15Min"
    lookback_bars: int = 500
    scan_interval_seconds: int = 60

    database_url: str = "sqlite:///./data/trading.db"
    model_dir: str = "models"
    log_dir: str = "logs"

    order_notional_usd: float = 25
    max_position_notional_usd: float = 100
    max_total_exposure_usd: float = 100
    max_daily_loss_usd: float = 20
    max_drawdown_pct: float = 0.05
    max_open_positions: int = 1

    min_buy_probability: float = 0.58
    min_sell_probability: float = 0.55
    confidence_gap_required: float = 0.08

    stop_loss_pct: float = 0.015
    take_profit_pct: float = 0.03
    trailing_stop_pct: float = 0.02
    max_holding_minutes: int = 720

    ml_enabled: bool = True
    model_retrain_enabled: bool = True
    retrain_every_hours: int = 24
    min_training_rows: int = 1000
    optuna_enabled: bool = False

    min_precision_for_promotion: float = 0.52
    max_validation_drawdown_pct: float = 0.20
    max_trade_fraction: float = 0.40

    @field_validator("symbol")
    @classmethod
    def symbol_must_be_btc_usd(cls, value: str) -> str:
        if value != ALLOWED_SYMBOL:
            raise ValueError("This project is hard-limited to BTC/USD only.")
        return value

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
        return data


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    Path(settings.model_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.log_dir).mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(parents=True, exist_ok=True)
    return settings
