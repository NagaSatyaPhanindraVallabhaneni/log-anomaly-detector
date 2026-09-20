"""IsolationForest-based anomaly detector over log feature vectors.

The model is trained on a baseline of "normal" logs. At inference time each
log line gets an anomaly score defined as ``-decision_function(x)`` (higher
means more anomalous) and a binary label from the IsolationForest prediction.
The scaler, model, and baseline template counts are persisted together with
joblib so a service restart reloads identical state.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from log_anomaly_detector.features import (
    FEATURE_NAMES,
    build_template_counts,
    featurize,
    mine_template,
)
from log_anomaly_detector.parser import parse_log_line


@dataclass
class DetectorConfig:
    contamination: float = 0.02
    n_estimators: int = 200
    random_state: int = 42


@dataclass
class ScoredLog:
    raw: str
    service: str
    level: str
    template: str
    score: float  # higher = more anomalous
    is_anomaly: bool


class LogAnomalyDetector:
    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()
        self._scaler = StandardScaler()
        self._model = IsolationForest(
            n_estimators=self.config.n_estimators,
            contamination=self.config.contamination,
            random_state=self.config.random_state,
        )
        self._template_counts: dict[str, int] = {}
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def template_counts(self) -> dict[str, int]:
        return dict(self._template_counts)

    def fit(self, lines: list[str]) -> LogAnomalyDetector:
        """Train on baseline log lines assumed to be (mostly) normal."""
        parsed = [parse_log_line(line) for line in lines if line.strip()]
        if not parsed:
            raise ValueError("fit() needs at least one non-empty log line")
        self._template_counts = build_template_counts([p.message for p in parsed])
        X = featurize(parsed, self._template_counts)
        X_scaled = self._scaler.fit_transform(X)
        self._model.fit(X_scaled)
        self._fitted = True
        return self

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Detector is not fitted yet — call fit() or load() first")

    def score_lines(self, lines: list[str]) -> list[ScoredLog]:
        """Score log lines; higher score = more anomalous."""
        self._require_fitted()
        parsed = [parse_log_line(line) for line in lines if line.strip()]
        if not parsed:
            return []
        X = featurize(parsed, self._template_counts)
        X_scaled = self._scaler.transform(X)
        raw_scores = -self._model.decision_function(X_scaled)  # flip: higher = worse
        predictions = self._model.predict(X_scaled)  # -1 = anomaly, 1 = normal
        return [
            ScoredLog(
                raw=p.raw,
                service=p.service,
                level=p.level,
                template=mine_template(p.message),
                score=float(score),
                is_anomaly=bool(pred == -1),
            )
            for p, score, pred in zip(parsed, raw_scores, predictions)
        ]

    def save(self, path: str | Path) -> Path:
        """Persist scaler, model, and template stats to disk."""
        self._require_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "scaler": self._scaler,
            "model": self._model,
            "template_counts": self._template_counts,
            "feature_names": FEATURE_NAMES,
            "config": self.config,
            "version": 1,
        }
        joblib.dump(payload, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> LogAnomalyDetector:
        """Load a detector previously saved with :meth:`save`."""
        payload: dict[str, Any] = joblib.load(path)
        detector = cls(config=payload.get("config") or DetectorConfig())
        detector._scaler = payload["scaler"]
        detector._model = payload["model"]
        detector._template_counts = payload["template_counts"]
        detector._fitted = True
        return detector

    # Backwards-compat alias used by older call sites in this repo.
    score = score_lines
