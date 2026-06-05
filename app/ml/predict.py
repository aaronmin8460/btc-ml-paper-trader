import pandas as pd

from app.config import Settings, get_settings
from app.data.feature_engineering import latest_feature_row
from app.data.scalping_features import SCALPING_BAR_FEATURE_COLUMNS, latest_scalping_feature_row
from app.ml.registry import ModelRegistry
from app.monitoring.logger import get_logger


class Predictor:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def predict(self, bars: pd.DataFrame, quote: dict | None = None) -> dict:
        model, active_model_status = ModelRegistry(self.settings).load_valid_active_model()
        model_available = model is not None
        if model_available:
            row = self._model_feature_row(bars, quote=quote, feature_columns=model.feature_columns)
            buy_probability = float(self._predict_buy_probability(model, row))
            independent_sell_probability = self._predict_independent_sell_probability(model, row)
            if independent_sell_probability is None:
                sell_probability = self._complement_probability(buy_probability)
                sell_probability_source = "buy_probability_complement"
                prediction_source = "legacy_sell_probability_fallback"
            else:
                sell_probability = independent_sell_probability
                sell_probability_source = "independent_sell_model"
                prediction_source = "model"
        else:
            row = latest_feature_row(bars, quote=quote)
            buy_probability = self._fallback_probability(row)
            sell_probability = self._complement_probability(buy_probability)
            sell_probability_source = "fallback_buy_probability_complement"
            prediction_source = "fallback_invalid_model" if active_model_status.active_model_path else "fallback"
        result = {
            "symbol": self.settings.symbol,
            "timestamp": row.iloc[-1]["timestamp"].isoformat(),
            "buy_probability": buy_probability,
            "sell_probability": sell_probability,
            "exit_probability": sell_probability,
            "sell_probability_source": sell_probability_source,
            "exit_probability_source": sell_probability_source,
            "sell_probability_semantics": "exit_existing_long_position",
            "supports_independent_sell_probability": sell_probability_source == "independent_sell_model",
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
            sell_probability_source=sell_probability_source,
            model_path=result["model_path"],
            prediction_source=prediction_source,
            model_available=model_available,
            active_model_status=result["active_model_status"],
            active_model_reason=result["active_model_reason"],
        )
        return result

    @staticmethod
    def _predict_buy_probability(model, row: pd.DataFrame) -> float:
        predictor = getattr(model, "predict_buy_proba", None)
        if callable(predictor):
            return float(predictor(row)[0])
        return float(model.predict_proba(row)[0])

    @staticmethod
    def _predict_independent_sell_probability(model, row: pd.DataFrame) -> float | None:
        supports_independent_sell_probability = getattr(model, "supports_independent_sell_probability", False)
        if callable(supports_independent_sell_probability):
            supports_independent_sell_probability = supports_independent_sell_probability()
        predictor = getattr(model, "predict_sell_proba", None)
        if not supports_independent_sell_probability or not callable(predictor):
            return None
        return float(predictor(row)[0])

    @staticmethod
    def _complement_probability(buy_probability: float) -> float:
        return float(max(0.0, min(1.0, 1.0 - buy_probability)))

    @staticmethod
    def _model_feature_row(bars: pd.DataFrame, *, quote: dict | None, feature_columns: list[str]) -> pd.DataFrame:
        if feature_columns == SCALPING_BAR_FEATURE_COLUMNS:
            return latest_scalping_feature_row(bars, quote=quote)
        return latest_feature_row(bars, quote=quote)

    def _fallback_probability(self, row: pd.DataFrame) -> float:
        rsi = float(row.iloc[-1]["rsi_14"])
        trend = float(row.iloc[-1]["trend_strength_20"])
        score = 0.5
        if 35 <= rsi <= 65:
            score += 0.03
        if trend > 0:
            score += 0.04
        return max(0.05, min(0.95, score))
