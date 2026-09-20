#!/usr/bin/env python3
"""Synthetic application-log generator.

Produces realistic-looking service logs for demos and tests — no external
data needed. Two modes:

* ``normal``  — healthy traffic: INFO-heavy, low latency, 2xx/3xx statuses.
* ``mixed``   — normal traffic with injected anomalies: error bursts,
  latency spikes, and rare failure templates.

Every run is seeded, so output is reproducible.
"""

from __future__ import annotations

import argparse
import random
import sys
import uuid
from datetime import datetime, timedelta, timezone

WORDS = ["alpha", "bravo", "chart", "delta", "ember", "flux", "grove", "harbor"]
RESOURCES = ["orders", "users", "sessions", "payments", "reports", "inventory"]
REASONS = ["bad_password", "expired_token", "rate_limited"]


def _ts(base: datetime, seconds: float) -> str:
    ts = base + timedelta(seconds=seconds)
    return ts.strftime("%Y-%m-%dT%H:%M:%S.") + f"{int(ts.microsecond / 1000):03d}Z"


def _normal_line(rng: random.Random, base: datetime, offset: float) -> str:
    service = rng.choices(
        ["auth", "payments", "search", "gateway", "cache"],
        weights=[25, 20, 20, 25, 10],
    )[0]
    latency = max(1, int(rng.lognormvariate(3.4, 0.55)))
    ts = _ts(base, offset)

    if service == "auth":
        if rng.random() < 0.93:
            return (
                f"{ts} INFO auth login successful user_id={rng.randint(1000, 9999)} "
                f"latency_ms={latency} status=200"
            )
        return (
            f"{ts} WARN auth login failed user_id={rng.randint(1000, 9999)} "
            f"reason={rng.choice(REASONS)} latency_ms={latency + 40} status=401"
        )
    if service == "payments":
        if rng.random() < 0.97:
            return (
                f"{ts} INFO payments payment processed order_id={uuid.uuid4()} "
                f"amount_cents={rng.randint(500, 20000)} latency_ms={latency + 30} status=200"
            )
        return (
            f"{ts} WARN payments payment declined order_id={uuid.uuid4()} "
            f"reason=insufficient_funds latency_ms={latency + 60} status=402"
        )
    if service == "search":
        return (
            f'{ts} INFO search query served q="{rng.choice(WORDS)}" '
            f"hits={rng.randint(0, 500)} latency_ms={latency} status=200"
        )
    if service == "gateway":
        status = rng.choices([200, 200, 200, 200, 304, 404], weights=[80, 0, 0, 0, 12, 8])[0]
        return (
            f"{ts} INFO gateway request completed method=GET "
            f"path=/api/v1/{rng.choice(RESOURCES)} latency_ms={latency} status={status}"
        )
    # cache
    if rng.random() < 0.7:
        return f"{ts} INFO cache cache hit key=user:{rng.randint(1, 5000)} latency_ms=2 status=200"
    return (
        f"{ts} INFO cache cache miss key=user:{rng.randint(1, 5000)} "
        f"latency_ms={latency} status=200"
    )


def _anomaly_line(rng: random.Random, base: datetime, offset: float) -> str:
    ts = _ts(base, offset)
    kind = rng.random()
    if kind < 0.35:  # error burst: DB timeouts
        return (
            f"{ts} ERROR payments database connection timeout after "
            f"{rng.randint(2000, 8000)}ms order_id={uuid.uuid4()} "
            f"latency_ms={rng.randint(2000, 8000)} status=500"
        )
    if kind < 0.55:  # upstream timeout
        return (
            f"{ts} ERROR gateway upstream timeout service=search "
            f"latency_ms={rng.randint(5000, 15000)} status=504"
        )
    if kind < 0.70:  # latency spike on an otherwise normal template
        return (
            f"{ts} INFO auth login successful user_id={rng.randint(1000, 9999)} "
            f"latency_ms={rng.randint(4000, 12000)} status=200"
        )
    if kind < 0.85:  # rare failure template
        return (
            f"{ts} ERROR cache eviction storm keys_evicted={rng.randint(50000, 200000)} "
            f"latency_ms={rng.randint(800, 2500)} status=200"
        )
    # TLS failures
    return (
        f"{ts} ERROR gateway tls handshake failed client_ip=10.{rng.randint(0, 255)}."
        f"{rng.randint(0, 255)}.{rng.randint(1, 254)} latency_ms={rng.randint(50, 300)} "
        f"status=525"
    )


def generate(n: int, mode: str = "normal", seed: int = 42, anomaly_rate: float = 0.08) -> list[str]:
    """Generate ``n`` log lines. ``mode`` is ``normal`` or ``mixed``."""
    rng = random.Random(seed)
    base = datetime(2026, 9, 20, 4, 0, 0, tzinfo=timezone.utc)
    lines: list[str] = []
    offset = 0.0
    for _ in range(n):
        offset += rng.uniform(0.05, 1.5)
        if mode == "mixed" and rng.random() < anomaly_rate:
            lines.append(_anomaly_line(rng, base, offset))
        else:
            lines.append(_normal_line(rng, base, offset))
    return lines


def generate_with_labels(
    n: int, seed: int = 42, anomaly_rate: float = 0.08
) -> tuple[list[str], list[bool]]:
    """Generate a mixed stream plus ground-truth anomaly labels (for the demo)."""
    rng = random.Random(seed)
    base = datetime(2026, 9, 20, 4, 0, 0, tzinfo=timezone.utc)
    lines: list[str] = []
    labels: list[bool] = []
    offset = 0.0
    for _ in range(n):
        offset += rng.uniform(0.05, 1.5)
        is_anomaly = rng.random() < anomaly_rate
        if is_anomaly:
            lines.append(_anomaly_line(rng, base, offset))
        else:
            lines.append(_normal_line(rng, base, offset))
        labels.append(is_anomaly)
    # Shuffle so anomalies are not clustered (bursts still occur by chance).
    order = list(range(n))
    rng.shuffle(order)
    return [lines[i] for i in order], [labels[i] for i in order]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic application logs.")
    parser.add_argument("--n", type=int, default=1000, help="Number of lines to generate")
    parser.add_argument("--mode", choices=["normal", "mixed"], default="normal")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--anomaly-rate", type=float, default=0.08)
    parser.add_argument("--out", default="-", help="Output file ('-' for stdout)")
    args = parser.parse_args(argv)

    lines = generate(args.n, mode=args.mode, seed=args.seed, anomaly_rate=args.anomaly_rate)
    text = "\n".join(lines) + "\n"
    if args.out == "-":
        sys.stdout.write(text)
    else:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"Wrote {len(lines)} lines to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
