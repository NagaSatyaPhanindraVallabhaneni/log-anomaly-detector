"""Parsing of single log lines into structured records.

Expected line format (space-separated prefix, free-form message)::

    <ISO-8601 timestamp> <LEVEL> <service> <message with optional key=value attrs>

Example::

    2026-09-20T04:00:01.123Z INFO auth login successful user_id=4821 latency_ms=42 status=200

Lines that do not match the format are still returned as :class:`ParsedLog`
with ``level="UNKNOWN"`` / ``service="unknown"`` instead of raising, so a
single malformed line can never break a streaming pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

LOG_PATTERN = re.compile(
    r"^(?P<ts>\S+)\s+(?P<level>[A-Z]+)\s+(?P<service>[\w\-.]+)\s+(?P<message>.*)$"
)
KV_PATTERN = re.compile(r"(?P<key>[A-Za-z_][\w.]*)=(?P<value>[^\s]+)")


@dataclass
class ParsedLog:
    raw: str
    timestamp: datetime | None
    level: str
    service: str
    message: str
    attrs: dict[str, str] = field(default_factory=dict)


def _parse_timestamp(value: str) -> datetime | None:
    candidate = value.strip()
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(candidate)
    except ValueError:
        return None


def parse_log_line(line: str) -> ParsedLog:
    """Parse one log line. Never raises on malformed input."""
    raw = line.rstrip("\n")
    if not raw.strip():
        return ParsedLog(
            raw=raw,
            timestamp=None,
            level="UNKNOWN",
            service="unknown",
            message="",
            attrs={},
        )
    match = LOG_PATTERN.match(raw)
    if not match:
        return ParsedLog(
            raw=raw,
            timestamp=None,
            level="UNKNOWN",
            service="unknown",
            message=raw,
            attrs={},
        )
    message = match.group("message")
    attrs = {m.group("key"): m.group("value") for m in KV_PATTERN.finditer(message)}
    return ParsedLog(
        raw=raw,
        timestamp=_parse_timestamp(match.group("ts")),
        level=match.group("level"),
        service=match.group("service"),
        message=message,
        attrs=attrs,
    )
