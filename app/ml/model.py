from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class MLSignalModel:
    feature_columns: list[str]
    models: dict[str, Any] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=lambda: {"logistic": 0.25, "random_forest": 0.4, "gradient_boosting": 0.35})
    metadata: dict[str, Any] = field(default_factory=dict)

    def train(self, df: pd.DataFrame, target_col: str = "buy_quality_label", tuned_params: dict[str, dict] | None = None) -> "MLSignalModel":
        x = df[self.feature_columns]
        y = df[target_col].astype(int)
        tuned_params = tuned_params or {}
        self.models = {
            "logistic": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
                ]
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=tuned_params.get("random_forest", {}).get("n_estimators", 160),
                max_depth=tuned_params.get("random_forest", {}).get("max_depth", 8),
                min_samples_leaf=tuned_params.get("random_forest", {}).get("min_samples_leaf", 8),
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            ),
            "gradient_boosting": HistGradientBoostingClassifier(
                max_iter=tuned_params.get("gradient_boosting", {}).get("max_iter", 120),
                learning_rate=tuned_params.get("gradient_boosting", {}).get("learning_rate", 0.05),
                max_leaf_nodes=tuned_params.get("gradient_boosting", {}).get("max_leaf_nodes", 24),
                random_state=42,
            ),
        }
        for model in self.models.values():
            model.fit(x, y)
        self.metadata.update(
            {
                "trained_at": datetime.now(UTC).isoformat(),
                "rows": int(len(df)),
                "target_col": target_col,
                "class_balance": y.value_counts(normalize=True).to_dict(),
            }
        )
        return self

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        if not self.models:
            raise ValueError("Model has not been trained or loaded.")
        x = rows[self.feature_columns]
        total_weight = 0.0
        weighted = np.zeros(len(rows), dtype=float)
        for name, model in self.models.items():
            weight = self.weights.get(name, 1.0)
            proba = model.predict_proba(x)[:, 1]
            weighted += weight * proba
            total_weight += weight
        return weighted / total_weight

    def feature_importance(self) -> dict[str, float]:
        rf = self.models.get("random_forest")
        if rf is None or not hasattr(rf, "feature_importances_"):
            return {}
        return dict(zip(self.feature_columns, [float(x) for x in rf.feature_importances_], strict=True))

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)
        return target

    @staticmethod
    def load(path: str | Path) -> "MLSignalModel":
        return joblib.load(path)


def tune_tree_params(df: pd.DataFrame, feature_columns: list[str], target_col: str = "buy_quality_label", n_trials: int = 15) -> dict[str, dict]:
    try:
        import optuna
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return {}

    split = int(len(df) * 0.8)
    train = df.iloc[:split]
    valid = df.iloc[split:]
    x_train, y_train = train[feature_columns], train[target_col].astype(int)
    x_valid, y_valid = valid[feature_columns], valid[target_col].astype(int)
    if len(set(y_valid.tolist())) < 2:
        return {}

    def objective(trial: optuna.Trial) -> float:
        rf = RandomForestClassifier(
            n_estimators=trial.suggest_int("rf_n_estimators", 80, 220),
            max_depth=trial.suggest_int("rf_max_depth", 4, 12),
            min_samples_leaf=trial.suggest_int("rf_min_samples_leaf", 4, 20),
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(x_train, y_train)
        return float(roc_auc_score(y_valid, rf.predict_proba(x_valid)[:, 1]))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    params = study.best_params
    return {
        "random_forest": {
            "n_estimators": params["rf_n_estimators"],
            "max_depth": params["rf_max_depth"],
            "min_samples_leaf": params["rf_min_samples_leaf"],
        }
    }
