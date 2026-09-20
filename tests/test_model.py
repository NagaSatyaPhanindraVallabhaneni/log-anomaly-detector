"""Tests for model training, scoring, and persistence."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from generate_logs import generate, generate_with_labels  # noqa: E402

from log_anomaly_detector.model import DetectorConfig, LogAnomalyDetector  # noqa: E402


@pytest.fixture(scope="module")
def trained_detector() -> LogAnomalyDetector:
    detector = LogAnomalyDetector(DetectorConfig(contamination=0.05, random_state=42))
    detector.fit(generate(1500, mode="normal", seed=11))
    return detector


def test_fit_requires_nonempty_input():
    with pytest.raises(ValueError):
        LogAnomalyDetector().fit([])


def test_score_before_fit_raises():
    with pytest.raises(RuntimeError):
        LogAnomalyDetector().score_lines(["2026-09-20T04:00:01Z INFO auth ok"])


def test_detector_flags_injected_anomalies(trained_detector: LogAnomalyDetector):
    stream, labels = generate_with_labels(300, seed=123)
    scored = trained_detector.score_lines(stream)
    assert len(scored) == len(stream)
    assert all(isinstance(s.score, float) for s in scored)

    anomaly_scores = [s.score for s, y in zip(scored, labels) if y]
    normal_scores = [s.score for s, y in zip(scored, labels) if not y]
    # Injected anomalies should score higher on average than normal lines.
    assert sum(anomaly_scores) / len(anomaly_scores) > sum(normal_scores) / len(normal_scores)
    # And a clear majority of injected anomalies should be flagged.
    flagged = [s for s, y in zip(scored, labels) if y and s.is_anomaly]
    assert len(flagged) / len(anomaly_scores) >= 0.6


def test_save_load_roundtrip_preserves_scores(trained_detector: LogAnomalyDetector, tmp_path: Path):
    stream, _ = generate_with_labels(50, seed=321)
    before = [s.score for s in trained_detector.score_lines(stream)]

    model_path = tmp_path / "detector.joblib"
    trained_detector.save(model_path)
    assert model_path.exists()

    reloaded = LogAnomalyDetector.load(model_path)
    after = [s.score for s in reloaded.score_lines(stream)]
    assert before == pytest.approx(after)
    assert len(reloaded.template_counts) == len(trained_detector.template_counts)
