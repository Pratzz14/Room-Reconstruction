"""Explicit free-tier quota snapshots; example limits never authorize live calls."""

import math
from datetime import datetime
from zoneinfo import ZoneInfo

DEFAULTS = {
    "version": "change-review-v1",
    "model": "gemini-3.1-flash-lite",
    "project_scope": "gemini-project-test",
    "quota_verified": False,
    "free_tier_verified": False,
    "quota_verified_at": None,
    "remaining_requests": None,
    "requests_per_minute": None,
    "input_tokens_per_minute": None,
    "requests_per_day": None,
    "max_input_tokens": 125000,
    "min_generation_interval_sec": 65,
    "max_output_tokens": 24576,
    "timeout_sec": 240,
    "batch_size": 4,
}
REVIEW_MODELS = (
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
)


def configuration(value=None):
    cfg = {**DEFAULTS, **(value or {})}
    if set(cfg) != set(DEFAULTS) or cfg["version"] != DEFAULTS["version"]:
        raise ValueError("Unknown change-review configuration")
    if (
        cfg["model"] not in REVIEW_MODELS
        or not isinstance(cfg["project_scope"], str)
        or not cfg["project_scope"].strip()
    ):
        raise ValueError(
            "Use a supported review model and a stable shared project scope"
        )
    for key in (
        "max_input_tokens",
        "min_generation_interval_sec",
        "max_output_tokens",
        "timeout_sec",
        "batch_size",
    ):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], int) or cfg[key] <= 0:
            raise ValueError("Invalid review limit: " + key)
    if cfg["min_generation_interval_sec"] < 65 or cfg["batch_size"] > 4:
        raise ValueError(
            "Review requires >=65-second spacing and <=4 identities per batch"
        )
    if cfg["max_input_tokens"] > 125000:
        raise ValueError("Review input ceiling must not exceed 125000 tokens")
    for key in (
        "requests_per_minute",
        "input_tokens_per_minute",
        "requests_per_day",
        "remaining_requests",
    ):
        v = cfg[key]
        if v is not None and (
            isinstance(v, bool)
            or not isinstance(v, int)
            or v < (0 if key == "remaining_requests" else 1)
        ):
            raise ValueError("Invalid verified quota: " + key)
    if any(type(cfg[k]) is not bool for k in ("quota_verified", "free_tier_verified")):
        raise ValueError("Quota verification flags must be booleans")
    return cfg


def live_limits(cfg, now):
    configuration(cfg)
    if not cfg["quota_verified"] or not cfg["free_tier_verified"]:
        raise ValueError(
            "Live review requires verified AI Studio free-tier quotas; preparation/offline review is available"
        )
    stamp = cfg["quota_verified_at"]
    if (
        isinstance(stamp, bool)
        or not isinstance(stamp, (int, float))
        or not math.isfinite(stamp)
    ):
        raise ValueError(
            "quota_verified_at must be a Unix timestamp of the AI Studio allowance snapshot"
        )
    day = lambda t: datetime.fromtimestamp(t, ZoneInfo("America/Los_Angeles")).date()
    if stamp > now or day(stamp) != day(now):
        raise ValueError("Refresh the quota snapshot for the current Pacific day")
    if any(
        cfg[k] is None
        for k in (
            "requests_per_minute",
            "input_tokens_per_minute",
            "requests_per_day",
            "remaining_requests",
        )
    ):
        raise ValueError("Verified quotas and remaining request allowance are required")
    if cfg["remaining_requests"] > cfg["requests_per_day"]:
        raise ValueError("Remaining allowance exceeds daily quota")
    return {
        "requests_per_minute": max(1, math.floor(cfg["requests_per_minute"] * 0.9)),
        "input_tokens_per_minute": max(
            1, math.floor(cfg["input_tokens_per_minute"] * 0.9)
        ),
    }
