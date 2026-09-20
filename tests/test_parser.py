"""Tests for log-line parsing."""

from log_anomaly_detector.parser import parse_log_line


def test_parse_wellformed_line():
    line = (
        "2026-09-20T04:00:01.123Z INFO auth login successful user_id=4821 latency_ms=42 status=200"
    )
    parsed = parse_log_line(line)
    assert parsed.level == "INFO"
    assert parsed.service == "auth"
    assert parsed.timestamp is not None
    assert parsed.timestamp.year == 2026
    assert parsed.attrs["user_id"] == "4821"
    assert parsed.attrs["latency_ms"] == "42"
    assert parsed.attrs["status"] == "200"
    assert "login successful" in parsed.message


def test_parse_malformed_line_does_not_raise():
    parsed = parse_log_line("this is not a log line at all !!!")
    assert parsed.level == "UNKNOWN"
    assert parsed.service == "unknown"
    assert parsed.timestamp is None
    assert parsed.raw == "this is not a log line at all !!!"


def test_parse_empty_line_does_not_raise():
    parsed = parse_log_line("   ")
    assert parsed.level == "UNKNOWN"
    assert parsed.message == ""


def test_parse_bad_timestamp_yields_none():
    parsed = parse_log_line("not-a-timestamp INFO auth hello world")
    assert parsed.timestamp is None
    assert parsed.level == "INFO"
    assert parsed.service == "auth"
