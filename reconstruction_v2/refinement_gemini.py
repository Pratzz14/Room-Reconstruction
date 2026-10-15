"""Bounded Gemini requests with durable receipts and local project quota accounting."""

import base64
import json
import os
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

from .core import digest, identity, read_json, write_json


@contextmanager
def exclusive(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise ValueError(
            f"Writer lock exists: {path}. Check the owning process before removing a stale lock."
        ) from None
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield
    finally:
        path.unlink(missing_ok=True)


def validate_config(config):
    if (
        config.get("version") != "refinement-v1"
        or config.get("model") != "gemini-3.1-flash-lite"
    ):
        raise ValueError(
            "Unsupported refinement version/model; model switches must be deliberate"
        )
    for key in (
        "max_generation_attempts",
        "daily_generation_cap",
        "requests_per_minute",
        "input_tokens_per_minute",
        "requests_per_day",
        "max_input_tokens",
        "min_generation_interval_sec",
        "max_output_tokens",
        "timeout_sec",
        "context_frames_per_capture",
        "crop_size",
        "geometry_max_spread_extents",
    ):
        value = config[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0 < value < float("inf")
        ):
            raise ValueError("Invalid refinement limit: " + key)
    if config["max_input_tokens"] > config["input_tokens_per_minute"] / 2:
        raise ValueError("Input token ceiling must reserve half the minute quota")
    if config["daily_generation_cap"] > config["requests_per_day"]:
        raise ValueError("Local daily cap exceeds the project quota")
    if not 0 < config["geometry_overlap"] <= 1:
        raise ValueError("Invalid geometry overlap")
    if any(
        int(config[k]) != config[k]
        for k in ("max_generation_attempts", "context_frames_per_capture", "crop_size")
    ):
        raise ValueError("Counts must be integers")
    if (
        not isinstance(config.get("project_scope"), str)
        or not config["project_scope"].strip()
    ):
        raise ValueError("A stable local project scope is required")
    categories = config.get("focus_categories", [])
    if (
        not isinstance(categories, list)
        or any(not isinstance(c, str) or not c.strip() for c in categories)
        or len(categories) != len(set(categories))
    ):
        raise ValueError("Focus categories must be unique nonempty strings")
    details = config.get("detail_regions_per_capture", 0)
    if (
        isinstance(details, bool)
        or not isinstance(details, int)
        or not 0 <= details <= 12
    ):
        raise ValueError("Detail region budget must be between 0 and 12 per capture")
    reviews = config.get("label_review_regions_per_capture", 0)
    if (
        isinstance(reviews, bool)
        or not isinstance(reviews, int)
        or not 0 <= reviews <= 12
    ):
        raise ValueError("Label review budget must be between 0 and 12 per capture")
    return config


def validate_output(reconstruction, objects, output):
    destination = Path(output).resolve()
    for source in (reconstruction, objects):
        source = Path(source).resolve()
        if destination.is_relative_to(source) or source.is_relative_to(destination):
            raise ValueError(
                "Refinement output and source artifacts must be separate sibling trees"
            )


class Gemini:
    def __init__(
        self,
        output,
        config,
        credential_source=None,
        *,
        transport=None,
        clock=time.time,
        sleep=time.sleep,
    ):
        self.output, self.config = Path(output), validate_config(config)
        self.cache = (
            self.output.parent
            / "refinement_cache"
            / identity(config["project_scope"])[:16]
        )
        self.cache.mkdir(parents=True, exist_ok=True)
        self.transport = transport or requests.post
        self.clock, self.sleep = clock, sleep
        self.credential_source = credential_source
        self.key = None
        self.generation_calls = self.token_count_calls = self.cache_hits = 0

    def _credential(self):
        if self.key is None:
            from scripts.compare_contact_sheets_gemini import credential

            self.key, _ = credential(
                Path(self.credential_source or "__no_credential_source__")
            )
        return self.key

    def _clean(self, value):
        from scripts.compare_contact_sheets_gemini import clean

        return clean(value, self.key or "__NO_SECRET__")

    def _day(self, timestamp):
        # Gemini RPD resets at midnight Pacific, including daylight-saving changes.
        return (
            datetime.fromtimestamp(timestamp, ZoneInfo("America/Los_Angeles"))
            .date()
            .isoformat()
        )

    def _send(self, method, payload, tokens=0):
        key = self._credential()  # Missing credentials never consume an attempt.
        cfg = self.config
        with exclusive(self.cache / ".quota.lock"):
            ledger_path = self.cache / "quota.json"
            entries = read_json(ledger_path) if ledger_path.exists() else []
            now = self.clock()
            entries = [e for e in entries if now - e["time"] < 172800]
            daily = [e for e in entries if self._day(e["time"]) == self._day(now)]
            if (
                sum(e["method"] == "generateContent" for e in daily)
                >= cfg["daily_generation_cap"]
            ):
                raise ValueError(
                    "Local daily generation cap reached; resume after Pacific midnight"
                )
            if len(daily) >= cfg["requests_per_day"]:
                raise ValueError("Local daily API request cap reached")
            if method == "generateContent":
                budget_path = self.output / "budget.json"
                budget = (
                    read_json(budget_path)
                    if budget_path.exists()
                    else {"generation_attempts": 0}
                )
                if budget["generation_attempts"] >= cfg["max_generation_attempts"]:
                    raise ValueError(
                        "Analysis generation attempt budget exhausted (including failures)"
                    )
            # All calls count toward local RPM; generation input tokens are reserved before sending.
            wait = 0.0
            recent = [e for e in entries if now - e["time"] < 60]
            if len(recent) >= cfg["requests_per_minute"]:
                wait = max(wait, min(e["time"] for e in recent) + 65 - now)
            if method == "generateContent":
                previous = [e["time"] for e in entries if e["method"] == method]
                if previous:
                    wait = max(
                        wait, max(previous) + cfg["min_generation_interval_sec"] - now
                    )
                if (
                    tokens + sum(e["tokens"] for e in recent)
                    > cfg["input_tokens_per_minute"]
                ):
                    wait = max(wait, max(e["time"] for e in recent) + 65 - now)
            if wait > 0:
                print(f"Quota pacing: waiting {wait:.0f}s before {method}", flush=True)
                deadline = self.clock() + wait
                while self.clock() < deadline:
                    self.sleep(min(1.0, deadline - self.clock()))
            if method == "generateContent":
                budget["generation_attempts"] += 1
                write_json(budget_path, budget)
            entries.append({"time": self.clock(), "method": method, "tokens": tokens})
            write_json(ledger_path, entries)
            if method == "generateContent":
                self.generation_calls += 1
            else:
                self.token_count_calls += 1
            # Keep the project lock through the request so sibling CLI processes cannot overlap.
            endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{cfg['model']}:{method}"
            try:
                response = self.transport(
                    endpoint,
                    headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                    json=payload,
                    timeout=(20, cfg["timeout_sec"]),
                )
                result = self._clean(response.json())
            except Exception as exc:  # noqa: BLE001 -- redact provider/transport errors before persistence
                raise ValueError(
                    self._clean(
                        f"Gemini request failed; no automatic retry: {type(exc).__name__}: {exc}"
                    )
                ) from None
            if response.status_code != 200:
                details = result.get("error", {})
                raise ValueError(
                    f"Gemini HTTP {response.status_code}; no automatic retry: {json.dumps(details)}"
                )
            return result

    def generate(self, system, prompt, schema, images, *, retry_failed=False):
        request_identity = identity(
            [
                self.config["model"],
                system,
                prompt,
                schema,
                [(str(p.name), digest(p)) for p in images],
                self.config["max_output_tokens"],
            ]
        )
        receipt_path = self.output / "requests" / (request_identity + ".json")
        shared = self.cache / (request_identity + ".json")
        if shared.exists():
            cached = read_json(shared)
            self.cache_hits += 1
            write_json(receipt_path, {**cached, "cache_hit": True})
            return cached["answer"]
        if receipt_path.exists() and not retry_failed:
            previous = read_json(receipt_path)
            if previous.get("status") != "completed":
                raise ValueError(
                    "Previous request failed or has unknown outcome; inspect receipt and explicitly use --retry-failed"
                )
        budget_path = self.output / "budget.json"
        if (
            budget_path.exists()
            and read_json(budget_path)["generation_attempts"]
            >= self.config["max_generation_attempts"]
        ):
            raise ValueError(
                "Analysis generation attempt budget exhausted (including failures)"
            )
        parts = [{"text": prompt}]
        for path in images:
            parts.append(
                {
                    "inlineData": {
                        "mimeType": "image/jpeg",
                        "data": base64.b64encode(path.read_bytes()).decode(),
                    }
                }
            )
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseJsonSchema": schema,
                "maxOutputTokens": self.config["max_output_tokens"],
                "thinkingConfig": {"thinkingLevel": "HIGH"},
            },
        }
        if len(json.dumps(payload).encode()) > 19_000_000:
            raise ValueError(
                "Prepared inline request exceeds the 19 MB safety limit; no evidence was dropped"
            )
        token_path = self.cache / (request_identity + ".tokens.json")
        if token_path.exists():
            count = read_json(token_path)["totalTokens"]
        else:
            # Count the exact generation request, including system/schema configuration.
            count_result = self._send(
                "countTokens",
                {
                    "generateContentRequest": {
                        "model": "models/" + self.config["model"],
                        **payload,
                    }
                },
            )
            count = count_result["totalTokens"]
            write_json(token_path, {"totalTokens": count})
        if not isinstance(count, int) or count > self.config["max_input_tokens"]:
            raise ValueError(
                f"Input token count {count} exceeds configured ceiling; no generation sent"
            )
        print(
            f"Gemini preflight: {count:,} input tokens; one combined generation request",
            flush=True,
        )
        receipt = {
            "identity": request_identity,
            "model": self.config["model"],
            "input_tokens": count,
            "status": "started",
            "cache_hit": False,
            "started": self.clock(),
        }
        write_json(receipt_path, receipt)
        try:
            result = self._send("generateContent", payload, tokens=count)
            receipt.update(response=result, usage=result.get("usageMetadata", {}))
            candidate = result.get("candidates", [{}])[0]
            if candidate.get("finishReason") != "STOP":
                raise ValueError(
                    "Incomplete/blocked Gemini response: "
                    + str(candidate.get("finishReason"))
                )
            answer = json.loads(
                "\n".join(
                    p["text"]
                    for p in candidate.get("content", {}).get("parts", [])
                    if p.get("text") and not p.get("thought")
                )
            )
            receipt.update(status="completed", answer=answer)
            write_json(receipt_path, receipt)
            write_json(shared, receipt)
            return answer
        except Exception as exc:
            receipt.update(status="failed", error=self._clean(str(exc)))
            write_json(receipt_path, receipt)
            raise
