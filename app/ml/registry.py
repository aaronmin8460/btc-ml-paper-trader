import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings


class ModelRegistry:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.path = Path(self.settings.model_dir) / "registry.json"

    def read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def promote(
        self,
        *,
        model_path: str,
        feature_columns: list[str],
        metrics: dict,
        thresholds: dict,
        training_start: str,
        training_end: str,
    ) -> dict[str, Any]:
        registry = {
            "active_model_path": model_path,
            "created_at": datetime.now(UTC).isoformat(),
            "training_start": training_start,
            "training_end": training_end,
            "feature_columns": feature_columns,
            "metrics": metrics,
            "thresholds": thresholds,
            "model_version": Path(model_path).stem,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
        return registry

    def active_model_path(self) -> Path | None:
        active = self.read().get("active_model_path")
        return Path(active) if active else None
