#!/usr/bin/env python3
"""End-to-end CLI demo: train on normal logs, score a mixed stream.

Steps:
  1. Generate 2,000 normal log lines and train the detector on them.
  2. Generate a mixed stream (400 lines, ~8% injected anomalies) with labels.
  3. Score the stream and print the top anomalies plus detection recall
     against the injected ground truth.

Run with the project venv active::

    python scripts/demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_logs import generate, generate_with_labels  # noqa: E402

from log_anomaly_detector.model import DetectorConfig, LogAnomalyDetector  # noqa: E402

TRAIN_N = 2000
STREAM_N = 400


def main() -> int:
    print("== Log Anomaly Detector demo ==\n")

    print(f"[1/3] Generating {TRAIN_N} normal log lines for training...")
    train_lines = generate(TRAIN_N, mode="normal", seed=7)

    print("[2/3] Training IsolationForest on the baseline...")
    # contamination matches the ~8% anomaly rate injected into the demo stream.
    detector = LogAnomalyDetector(DetectorConfig(contamination=0.08))
    detector.fit(train_lines)
    print(f"      learned {len(detector.template_counts)} log templates\n")

    print(f"[3/3] Scoring a mixed stream of {STREAM_N} lines (~8% injected anomalies)...")
    stream, labels = generate_with_labels(STREAM_N, seed=99)
    scored = detector.score_lines(stream)

    ranked = sorted(zip(scored, labels), key=lambda pair: pair[0].score, reverse=True)

    print("\nTop 10 most anomalous lines:")
    print("-" * 100)
    for item, _ in ranked[:10]:
        flag = "ANOMALY " if item.is_anomaly else "normal  "
        print(f"[{flag}] score={item.score:7.3f} {item.raw[:86]}")
    print("-" * 100)

    # Honest evaluation on the synthetic labels.
    predicted = [s.is_anomaly for s in scored]
    tp = sum(1 for p, y in zip(predicted, labels) if p and y)
    fp = sum(1 for p, y in zip(predicted, labels) if p and not y)
    fn = sum(1 for p, y in zip(predicted, labels) if not p and y)
    n_injected = sum(labels)
    recall = tp / n_injected if n_injected else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0

    print(f"\nInjected anomalies: {n_injected}")
    print(
        f"Flagged as anomalous: {sum(predicted)} "
        f"(true positives={tp}, false positives={fp}, missed={fn})"
    )
    print(f"Recall on injected anomalies:    {recall:.1%}")
    print(f"Precision on flagged lines:      {precision:.1%}")
    print("\nNote: these numbers are on synthetic data — they show the pipeline works,")
    print("not how it would perform on any real production logs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
