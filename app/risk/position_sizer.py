from app.config import Settings, get_settings
from app.utils.math import clamp


def fixed_notional(settings: Settings | None = None) -> float:
    settings = settings or get_settings()
    return min(settings.order_notional_usd, settings.max_position_notional_usd)


def volatility_adjusted_notional(volatility: float, settings: Settings | None = None) -> float:
    settings = settings or get_settings()
    base = fixed_notional(settings)
    if volatility <= 0:
        return base
    scale = clamp(0.01 / volatility, 0.25, 1.0)
    return min(base * scale, settings.max_position_notional_usd)
