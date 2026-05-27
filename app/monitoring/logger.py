import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import get_settings


def _json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


class JsonlLogger:
    def __init__(self, name: str = "btc_ml_paper_trader") -> None:
        self.name = name
        self.settings = get_settings()
        Path(self.settings.log_dir).mkdir(parents=True, exist_ok=True)
        self.file_path = Path(self.settings.log_dir) / "events.jsonl"
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(handler)

    def event(self, event_type: str, **payload: Any) -> None:
        record = {
            "ts": datetime.now(UTC).isoformat(),
            "event_type": event_type,
            **payload,
        }
        line = json.dumps(record, default=_json_default, separators=(",", ":"))
        with self.file_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        self.logger.info(line)


def get_logger() -> JsonlLogger:
    return JsonlLogger()
