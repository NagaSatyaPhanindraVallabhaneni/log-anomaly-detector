"""Tests for template mining and featurization."""

import numpy as np

from log_anomaly_detector.features import (
    FEATURE_NAMES,
    build_template_counts,
    extract_features,
    featurize,
    mine_template,
    template_rarity,
)
from log_anomaly_detector.parser import parse_log_line


def test_mine_template_masks_numbers():
    a = mine_template("login failed user_id=4821 latency_ms=42")
    b = mine_template("login failed user_id=9917 latency_ms=7")
    assert a == b
    assert "4821" not in a
    assert "<NUM>" in a


def test_mine_template_masks_ip_and_uuid():
    masked = mine_template("conn from 10.0.0.5 id=123e4567-e89b-12d3-a456-426614174000 done")
    assert "10.0.0.5" not in masked
    assert "123e4567-e89b-12d3-a456-426614174000" not in masked
    assert "<IP>" in masked
    assert "<UUID>" in masked


def test_template_rarity_unknown_is_max():
    assert template_rarity("never seen", {"a": 10}) == 1.0
    assert template_rarity("never seen", None) == 1.0
    assert template_rarity("never seen", {}) == 1.0


def test_template_rarity_decreases_with_count():
    counts = {"common": 99, "rare": 1}
    assert template_rarity("common", counts) < template_rarity("rare", counts)
    assert 0.0 < template_rarity("common", counts) <= 1.0


def test_extract_features_fixed_length_and_names():
    parsed = parse_log_line(
        "2026-09-20T04:00:01.123Z ERROR payments database timeout latency_ms=5000 status=500"
    )
    feats = extract_features(parsed, {"x": 1})
    assert sorted(feats.keys()) == sorted(FEATURE_NAMES)
    assert feats["level_ERROR"] == 1.0
    assert feats["level_INFO"] == 0.0
    assert feats["status_5xx"] == 1.0
    assert feats["latency_ms_log"] > 0.0
    assert feats["template_rarity"] == 1.0  # unseen template


def test_featurize_matrix_shape():
    lines = [
        "2026-09-20T04:00:01.123Z INFO auth ok latency_ms=10 status=200",
        "2026-09-20T04:00:02.123Z WARN auth slow latency_ms=900 status=200",
    ]
    parsed = [parse_log_line(line) for line in lines]
    counts = build_template_counts([p.message for p in parsed])
    X = featurize(parsed, counts)
    assert isinstance(X, np.ndarray)
    assert X.shape == (2, len(FEATURE_NAMES))
    assert np.all(np.isfinite(X))
