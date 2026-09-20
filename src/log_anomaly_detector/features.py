"""Featurization: lightweight template mining plus numeric log features.

Template mining masks volatile tokens (numbers, IPs, UUIDs, hex, quoted
strings) so that ``"login failed user_id=4821"`` and ``"login failed
user_id=9917"`` map to the same template ``"login failed user_id=<NUM>"``.
Template rarity is derived from baseline template counts learned at train
time: a template never seen during training gets the maximum rarity of 1.0.
"""

from __future__ import annotations

import math
import re

import numpy as np

from log_anomaly_detector.parser import ParsedLog

_MASK_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b"), "<IP>"),  # IPv4 (+port)
    (
        re.compile(
            r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
        ),
        "<UUID>",
    ),
    (re.compile(r"\b0x[0-9a-fA-F]+\b"), "<HEX>"),
    (re.compile(r'"[^"]*"'), "<STR>"),  # quoted strings
    (re.compile(r"\b\d+\b"), "<NUM>"),  # bare numbers last
]

_LEVELS = ["DEBUG", "INFO", "WARN", "ERROR"]

FEATURE_NAMES: list[str] = [
    "level_DEBUG",
    "level_INFO",
    "level_WARN",
    "level_ERROR",
    "level_OTHER",
    "latency_ms_log",
    "status_2xx",
    "status_4xx",
    "status_5xx",
    "status_other",
    "hour_sin",
    "hour_cos",
    "msg_len_log",
    "token_count_log",
    "digit_ratio",
    "template_rarity",
]


def mine_template(message: str) -> str:
    """Reduce a log message to its template by masking volatile tokens."""
    template = message
    for pattern, replacement in _MASK_PATTERNS:
        template = pattern.sub(replacement, template)
    # Collapse values of key=<MASK> pairs and stray '=' spacing for stability.
    template = re.sub(r"\s+", " ", template).strip()
    return template


def build_template_counts(messages: list[str]) -> dict[str, int]:
    """Count templates over a baseline corpus (used at train time)."""
    counts: dict[str, int] = {}
    for message in messages:
        template = mine_template(message)
        counts[template] = counts.get(template, 0) + 1
    return counts


def template_rarity(template: str, template_counts: dict[str, int] | None) -> float:
    """Rarity in (0, 1]: 1.0 for templates never seen in the baseline."""
    if not template_counts:
        return 1.0
    return 1.0 / (1.0 + template_counts.get(template, 0))


def _safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_features(
    parsed: ParsedLog, template_counts: dict[str, int] | None = None
) -> dict[str, float]:
    """Extract a fixed, named feature vector from one parsed log line."""
    level = parsed.level.upper()
    if level == "WARNING":
        level = "WARN"
    level_feats = {f"level_{name}": 0.0 for name in ["DEBUG", "INFO", "WARN", "ERROR", "OTHER"]}
    level_feats[f"level_{level}" if level in _LEVELS else "level_OTHER"] = 1.0

    latency = _safe_float(parsed.attrs.get("latency_ms"))
    latency_feat = math.log1p(latency) if latency is not None and latency >= 0 else 0.0

    status = _safe_float(parsed.attrs.get("status") or parsed.attrs.get("status_code"))
    status_feats = {"status_2xx": 0.0, "status_4xx": 0.0, "status_5xx": 0.0, "status_other": 0.0}
    if status is not None:
        code = int(status)
        if 200 <= code < 300:
            status_feats["status_2xx"] = 1.0
        elif 400 <= code < 500:
            status_feats["status_4xx"] = 1.0
        elif 500 <= code < 600:
            status_feats["status_5xx"] = 1.0
        else:
            status_feats["status_other"] = 1.0
    else:
        status_feats["status_other"] = 1.0

    if parsed.timestamp is not None:
        hour = parsed.timestamp.hour + parsed.timestamp.minute / 60.0
    else:
        hour = 12.0  # neutral fallback when the timestamp is missing/unparseable
    hour_angle = 2.0 * math.pi * hour / 24.0

    message = parsed.message or ""
    msg_len = len(message)
    digit_chars = sum(ch.isdigit() for ch in message)

    template = mine_template(message)

    return {
        **level_feats,
        "latency_ms_log": latency_feat,
        "status_2xx": status_feats["status_2xx"],
        "status_4xx": status_feats["status_4xx"],
        "status_5xx": status_feats["status_5xx"],
        "status_other": status_feats["status_other"],
        "hour_sin": math.sin(hour_angle),
        "hour_cos": math.cos(hour_angle),
        "msg_len_log": math.log1p(msg_len),
        "token_count_log": math.log1p(len(message.split())),
        "digit_ratio": (digit_chars / msg_len) if msg_len else 0.0,
        "template_rarity": template_rarity(template, template_counts),
    }


def featurize(
    parsed_logs: list[ParsedLog], template_counts: dict[str, int] | None = None
) -> np.ndarray:
    """Featurize parsed logs into an (n, len(FEATURE_NAMES)) float matrix."""
    rows = [
        [extract_features(parsed, template_counts)[name] for name in FEATURE_NAMES]
        for parsed in parsed_logs
    ]
    return np.asarray(rows, dtype=np.float64)
