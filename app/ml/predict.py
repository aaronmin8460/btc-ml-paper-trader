import pandas as pd

from app.config import Settings, get_settings
from app.data.feature_engineering import latest_feature_row
from app.ml.registry import ModelRegistry
from app.monitoring.logger import get_logger


class Predictor:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def predict(self, bars: pd.DataFrame, quote: dict | None = None) -> dict:
        row = latest_feature_row(bars, quote=quote)
        model, active_model_status = ModelRegistry(self.settings).load_valid_active_model()
        model_available = model is not None
        if model_available:
            buy_probability = float(model.predict_proba(row)[0])
            prediction_source = "model"
        else:
            buy_probability = self._fallback_probability(row)
            prediction_source = "fallback_invalid_model" if active_model_status.active_model_path else "fallback"
        sell_probability = float(max(0.0, min(1.0, 1.0 - buy_probability)))
        result = {
            "symbol": self.settings.symbol,
            "timestamp": row.iloc[-1]["timestamp"].isoformat(),
            "buy_probability": buy_probability,
            "sell_probability": sell_probability,
            "features": row.iloc[-1].to_dict(),
            "model_path": active_model_status.active_model_path,
            "prediction_source": prediction_source,
            "model_available": model_available,
            **active_model_status.to_dict(),
        }
        get_logger().event(
            "prediction",
            symbol=self.settings.symbol,
            buy_probability=buy_probability,
            sell_probability=sell_probability,
            model_path=result["model_path"],
            prediction_source=prediction_source,
            model_available=model_available,
            active_model_status=result["active_model_status"],
            active_model_reason=result["active_model_reason"],
        )
        return result

    def _fallback_probability(self, row: pd.DataFrame) -> float:
        rsi = float(row.iloc[-1]["rsi_14"])
        trend = float(row.iloc[-1]["trend_strength_20"])
        score = 0.5
        if 35 <= rsi <= 65:
            score += 0.03
        if trend > 0:
            score += 0.04
        return max(0.05, min(0.95, score))
