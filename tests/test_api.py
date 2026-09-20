"""Tests for the FastAPI service."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from generate_logs import generate  # noqa: E402

from log_anomaly_detector.service import create_app  # noqa: E402


@pytest.fixture()
def client(tmp_path: Path):
    app = create_app(model_path=tmp_path / "detector.joblib")
    with TestClient(app) as test_client:
        yield test_client


def test_health_before_training(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": False}


def test_score_before_training_is_400(client: TestClient):
    response = client.post("/score", json={"lines": ["2026-09-20T04:00:01Z INFO auth ok"]})
    assert response.status_code == 400


def test_train_then_ingest_and_stats(client: TestClient):
    train_lines = generate(500, mode="normal", seed=21)
    response = client.post("/train", json={"lines": train_lines})
    assert response.status_code == 200
    body = response.json()
    assert body["trained_on"] == 500
    assert body["templates"] > 0

    assert client.get("/health").json()["model_loaded"] is True

    mixed = generate(60, mode="mixed", seed=22)
    response = client.post("/ingest", json={"lines": mixed})
    assert response.status_code == 200
    scored = response.json()
    assert len(scored) == 60
    assert all({"raw", "score", "is_anomaly", "template"} <= set(item) for item in scored)
    assert any(item["is_anomaly"] for item in scored)

    stats = client.get("/stats").json()
    assert stats["ingested"] == 60
    assert stats["anomalies"] == sum(1 for item in scored if item["is_anomaly"])
    assert stats["model_trained"] is True


def test_ingest_accepts_single_string(client: TestClient):
    client.post("/train", json={"lines": generate(300, mode="normal", seed=23)})
    line = "2026-09-20T04:00:01.123Z INFO auth login successful user_id=1 latency_ms=12 status=200"
    response = client.post("/ingest", json={"lines": line})
    assert response.status_code == 200
    assert len(response.json()) == 1


def test_score_does_not_change_stats(client: TestClient):
    client.post("/train", json={"lines": generate(300, mode="normal", seed=24)})
    line = "2026-09-20T04:00:01.123Z INFO auth ok latency_ms=5 status=200"
    client.post("/score", json={"lines": [line]})
    assert client.get("/stats").json()["ingested"] == 0
