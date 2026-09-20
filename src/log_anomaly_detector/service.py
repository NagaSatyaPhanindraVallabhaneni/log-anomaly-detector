"""FastAPI service exposing the log anomaly detector over HTTP."""

from __future__ import annotations

from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from log_anomaly_detector.model import DetectorConfig, LogAnomalyDetector, ScoredLog

DEFAULT_MODEL_PATH = Path("models/detector.joblib")
MAX_BUFFER = 10_000


class LinesPayload(BaseModel):
    lines: str | list[str] = Field(description="One log line or a list of log lines")

    def normalized(self) -> list[str]:
        if isinstance(self.lines, str):
            return [self.lines]
        return list(self.lines)


class TrainPayload(LinesPayload):
    contamination: float = Field(default=0.02, gt=0.0, lt=0.5)


class ScoredLine(BaseModel):
    raw: str
    service: str
    level: str
    template: str
    score: float
    is_anomaly: bool


def _to_response(scored: ScoredLog) -> ScoredLine:
    return ScoredLine(
        raw=scored.raw,
        service=scored.service,
        level=scored.level,
        template=scored.template,
        score=scored.score,
        is_anomaly=scored.is_anomaly,
    )


def create_app(model_path: str | Path = DEFAULT_MODEL_PATH) -> FastAPI:
    model_path = Path(model_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if model_path.exists():
            app.state.detector = LogAnomalyDetector.load(model_path)
        else:
            app.state.detector = None
        app.state.recent = deque(maxlen=MAX_BUFFER)
        app.state.ingested = 0
        app.state.anomalies = 0
        yield

    app = FastAPI(title="Log Anomaly Detector", version="0.1.0", lifespan=lifespan)

    def _detector_or_400() -> LogAnomalyDetector:
        detector = app.state.detector
        if detector is None:
            raise HTTPException(
                status_code=400,
                detail="No model trained yet — POST /train with baseline log lines first.",
            )
        return detector

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model_loaded": app.state.detector is not None}

    @app.get("/stats")
    def stats() -> dict:
        ingested = app.state.ingested
        anomalies = app.state.anomalies
        return {
            "ingested": ingested,
            "anomalies": anomalies,
            "anomaly_rate": (anomalies / ingested) if ingested else 0.0,
            "model_trained": app.state.detector is not None,
            "templates_known": len(app.state.detector.template_counts) if app.state.detector else 0,
        }

    @app.post("/train")
    def train(payload: TrainPayload) -> dict:
        lines = [ln for ln in payload.normalized() if ln.strip()]
        if not lines:
            raise HTTPException(status_code=422, detail="Provide at least one log line.")
        detector = LogAnomalyDetector(DetectorConfig(contamination=payload.contamination))
        detector.fit(lines)
        detector.save(model_path)
        app.state.detector = detector
        return {
            "trained_on": len(lines),
            "templates": len(detector.template_counts),
            "model_path": str(model_path),
        }

    @app.post("/score", response_model=list[ScoredLine])
    def score(payload: LinesPayload) -> list[ScoredLine]:
        """Score lines without storing them."""
        detector = _detector_or_400()
        lines = [ln for ln in payload.normalized() if ln.strip()]
        return [_to_response(s) for s in detector.score_lines(lines)]

    @app.post("/ingest", response_model=list[ScoredLine])
    def ingest(payload: LinesPayload) -> list[ScoredLine]:
        """Score lines and add them to the rolling buffer / stats."""
        detector = _detector_or_400()
        lines = [ln for ln in payload.normalized() if ln.strip()]
        scored = detector.score_lines(lines)
        for item in scored:
            app.state.recent.append(item)
            app.state.ingested += 1
            if item.is_anomaly:
                app.state.anomalies += 1
        return [_to_response(s) for s in scored]

    return app


app = create_app()
