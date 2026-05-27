from datetime import UTC, datetime

from app.broker.alpaca_client import AlpacaClient
from app.config import ALLOWED_SYMBOL, Settings, get_settings
from app.data.feature_engineering import latest_feature_row
from app.data.market_data import MarketDataClient
from app.db.database import SessionLocal, init_db
from app.db.repository import Repository
from app.ml.predict import Predictor
from app.monitoring.logger import get_logger
from app.risk.risk_manager import PositionState
from app.strategy.decision_engine import DecisionEngine


class Trader:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.market_data = MarketDataClient(self.settings)
        self.predictor = Predictor(self.settings)
        self.decision_engine = DecisionEngine(self.settings)
        self.broker = AlpacaClient(self.settings)
        self.logger = get_logger()

    async def get_position_state(self) -> PositionState:
        position = await self.broker.get_position(ALLOWED_SYMBOL)
        if not position:
            return PositionState()
        qty = float(position.get("qty", 0) or 0)
        avg_entry = float(position.get("avg_entry_price", 0) or 0)
        market_value = abs(float(position.get("market_value", 0) or 0))
        return PositionState(
            qty=qty,
            avg_entry_price=avg_entry,
            market_value=market_value,
            opened_at=datetime.now(UTC),
            highest_price=max(avg_entry, float(position.get("current_price", avg_entry) or avg_entry)),
        )

    async def run_once(self) -> dict:
        init_db()
        bars = await self.market_data.fetch_bars(ALLOWED_SYMBOL)
        quote = await self.market_data.fetch_latest_quote(ALLOWED_SYMBOL)
        prediction = self.predictor.predict(bars, quote=quote)
        feature_row = latest_feature_row(bars, quote=quote).iloc[-1]
        position = await self.get_position_state()
        decision = self.decision_engine.decide(
            prediction=prediction,
            feature_row=feature_row,
            position=position,
            trading_enabled=self.settings.trading_enabled,
        )
        self.logger.event(
            "signal",
            symbol=decision.symbol,
            action=decision.action,
            reason=decision.reason,
            buy_probability=prediction["buy_probability"],
            sell_probability=prediction["sell_probability"],
        )
        with SessionLocal() as db:
            repo = Repository(db)
            repo.add_signal(decision.action, prediction["buy_probability"], prediction["sell_probability"], decision.reason)
            order_response = None
            if decision.action in {"buy", "sell"}:
                order_response = await self.broker.submit_market_order(
                    symbol=decision.symbol,
                    side=decision.action,
                    notional=decision.notional,
                    qty=decision.qty,
                    current_position_qty=position.qty,
                )
                repo.add_order(
                    side=decision.action,
                    status=order_response.get("status", "submitted"),
                    notional=decision.notional,
                    qty=decision.qty,
                    broker_order_id=order_response.get("id"),
                    raw_response=order_response,
                )
        return {"prediction": prediction, "decision": decision.__dict__, "order": order_response}
