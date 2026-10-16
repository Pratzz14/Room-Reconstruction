"""Two independent room scenes controlled by one shared-identity selection."""

import json
import mimetypes
import secrets
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from .core import read_json, safe_path
from .region_reference import RegionReferences
from .sync_viewer import _SyncState

CAPTURES = ("before", "after")

UNCERTAINTY_REASONS = {
    "invalid_vlm_evidence": "Gemini returned an invalid image region or evidence reference",
    "local_visual_contradiction": "A source-picture check rejected the model claim",
    "visual_and_location_evidence_conflict": "Pictures and the estimated location change disagree",
    "incompatible_semantic_identity_assignment": "Gemini linked different types of objects",
    "identity_ambiguous": "Competing object identities remain unresolved",
    "competing_visual_identity": "Gemini proposed competing identity matches",
    "uncertain_capture_identity": "The source detection is low confidence or a group of objects",
    "new_counterpart_requires_closeup_confirmation": "A newly found counterpart still needs close-up confirmation",
    "presence_change_requires_closeup_confirmation": "Addition or removal still needs close-up confirmation",
    "opposing_region_absence_not_supported": "The pictures do not establish that the missing side is empty",
    "unresolved_inventoried_counterpart": "The original opposite-side object has not been accounted for",
    "no_valid_review_for_reconciled_identity": "No valid visual review supports this final identity",
    "not_yet_reviewed": "An additional competing observation has not been independently reviewed",
    "visual_movement_unresolved": "The pictures do not resolve whether this object moved",
    "alignment_not_accepted": "Room alignment was not accepted",
}


def uncertainty_messages(change):
    codes = change.get("uncertainties", [])
    ordered = sorted(
        codes,
        key=lambda c: (
            list(UNCERTAINTY_REASONS).index(c.split(":")[0])
            if c.split(":")[0] in UNCERTAINTY_REASONS
            else len(UNCERTAINTY_REASONS)
        ),
    )
    return [
        {
            "code": code,
            "message": UNCERTAINTY_REASONS.get(
                code.split(":")[0], code.replace("_", " ")
            ),
        }
        for code in ordered
    ]


def readable_explanation(change):
    result = change.get("explanation", "Change classification unavailable.")
    for reason in sorted(uncertainty_messages(change), key=lambda r: -len(r["code"])):
        result = result.replace(reason["code"], reason["message"])
    return result


class ComparisonState:
    def __init__(self, root, inventory, comparison, maximum=200000, report=None):
        self.report_root = Path(report).resolve() if report else None
        self.report = read_json(self.report_root / "report.json") if report else None
        self.report_files = (
            read_json(self.report_root / "manifest.json")["artifacts"] if report else {}
        )
        report_events = (
            {e["event_id"]: e for e in self.report["events"]} if report else {}
        )
        self.lock = threading.RLock()
        self.session, self.sequence, self.selected = None, -1, None
        self.views = {
            capture: _SyncState(root, [capture], maximum, inventory, static=True)
            for capture in CAPTURES
        }
        for view in self.views.values():
            view.overlay = False
            view.settings[view.capture]["show_path"] = False
        self.items = {}
        for identity in getattr(
            inventory,
            "review_identities",
            read_json(Path(comparison) / "identities.json")["identities"],
        ):
            sides = {}
            for capture in CAPTURES:
                oid = identity[capture + "_id"]
                if oid is None:
                    sides[capture] = None
                    continue
                obj, _ = inventory.object(capture, oid)
                inspection = self.views[capture].inspection
                observations = obj.get(
                    "geometry_observation_ids", obj["observation_ids"]
                )
                context = ["context:" + eid for eid in obj.get("context_evidence", [])]
                evidence = list(
                    dict.fromkeys(
                        (observations + context + obj["observation_ids"])
                        if obj["point_count"]
                        else (context + obj["observation_ids"])
                    )
                )
                frames = []
                for eid in evidence:
                    record = inspection.evidence_record(capture, eid)
                    frames.append(
                        {
                            "id": eid,
                            "frame_id": record["frame_id"],
                            "timestamp_sec": record["timestamp_sec"],
                            "context": record.get("status") == "context",
                        }
                    )
                if identity["shared_id"] in report_events:
                    role = "object_" + capture
                    preferred = [
                        v["source_id"]
                        for v in report_events[identity["shared_id"]]["evidence"][role][
                            "views"
                        ]
                    ]
                    frames.sort(
                        key=lambda f: (
                            preferred.index(f["id"])
                            if f["id"] in preferred
                            else len(preferred)
                        )
                    )
                sides[capture] = {
                    "id": oid,
                    "point_count": obj["point_count"],
                    "frames": frames,
                }
            label_obj = next(
                inventory.object(c, identity[c + "_id"])[0]
                for c in CAPTURES
                if identity[c + "_id"]
            )
            change = label_obj.get("change", {})
            self.items[identity["shared_id"]] = {
                "shared_id": identity["shared_id"],
                "label": label_obj["label"],
                "status": identity["status"],
                "type": change.get("type", "not_evaluated"),
                "trust": change.get("trust", "uncertain"),
                "explanation": readable_explanation(change),
                "uncertainty_reasons": uncertainty_messages(change),
                "review_status": change.get("review_status"),
                "original_proposals": [
                    {"type": p["type"], "shared_id": p["shared_id"]}
                    for p in change.get("original_proposals", [])
                ],
                "sides": sides,
            }
        self.counts = inventory.comparison.get("change_counts", {})
        self.alignment = inventory.comparison["alignment"]["quality"]["status"]
        self.review_coverage = inventory.comparison.get("review_coverage")
        self.review_status = inventory.comparison.get("review_status")
        primary_reasons = Counter(
            item["uncertainty_reasons"][0]["message"]
            if item["uncertainty_reasons"]
            else "Insufficient object evidence"
            for item in self.items.values()
            if item["type"] == "uncertain"
        )
        self.uncertainty_summary = [
            {"reason": reason, "count": count}
            for reason, count in sorted(
                primary_reasons.items(), key=lambda row: (-row[1], row[0])
            )
        ]
        self.event_aliases = {}
        for item in self.items.values():
            for original in item["original_proposals"]:
                self.event_aliases.setdefault(original["shared_id"], []).append(
                    item["shared_id"]
                )
        self.region_frames = RegionReferences(inventory, self.views)
        self.references = dict.fromkeys(CAPTURES)

    def begin_session(self):
        with self.lock:
            self.session, self.sequence = secrets.token_urlsafe(24), -1
            for view in self.views.values():
                view.begin_session()
            return {
                "session": self.session,
                "items": list(self.items.values()),
                "counts": self.counts,
                "alignment": self.alignment,
                "report_available": self.report is not None,
                "review_coverage": self.review_coverage,
                "review_status": self.review_status,
                "uncertainty_summary": self.uncertainty_summary,
                "event_aliases": self.event_aliases,
                **self.snapshot(),
            }

    def snapshot(self):
        with self.lock:
            return {
                "sequence": self.sequence,
                "selected": self.items.get(self.selected),
                "region_references": self.references,
                "views": {
                    capture: view.snapshot() for capture, view in self.views.items()
                },
            }

    def update(self, payload):
        with self.lock:
            if payload.get("session") != self.session or self.session is None:
                raise PermissionError(
                    "Another page controls this viewer. Reload to take control."
                )
            sequence = payload.get("sequence")
            if type(sequence) is not int or sequence < 0:
                raise ValueError("Non-negative integer sequence required")
            if sequence <= self.sequence:
                return {**self.snapshot(), "stale": True}
            if set(payload) - {
                "session",
                "sequence",
                "select",
                "frames",
                "settings",
                "fit",
                "focus",
            }:
                raise ValueError("Unknown comparison action")
            selected = payload.get("select", self.selected)
            if selected is not None and selected not in self.items:
                raise ValueError("Unknown shared identity")
            frames, settings = payload.get("frames", {}), payload.get("settings", {})
            if not isinstance(frames, dict) or set(frames) - set(CAPTURES):
                raise ValueError("Invalid source frames")
            if not isinstance(settings, dict) or set(settings) - {
                "crop_enabled",
                "show_path",
            }:
                raise ValueError("Invalid room settings")
            if (
                any(type(v) is not bool for v in settings.values())
                or type(payload.get("fit", False)) is not bool
                or type(payload.get("focus", False)) is not bool
            ):
                raise ValueError("Room controls must be boolean")
            item = self.items.get(selected)
            actions = {}
            references = dict.fromkeys(CAPTURES)
            # First validate genuine object frames. Opposing region references
            # remain separate from object evidence and never select an object.
            if item:
                for capture in CAPTURES:
                    side = item["sides"][capture]
                    if (
                        side
                        and capture in frames
                        and frames[capture] not in {f["id"] for f in side["frames"]}
                    ):
                        raise ValueError("Frame does not support the selected object")
                for capture in CAPTURES:
                    if item["sides"][capture] is not None:
                        continue
                    source_capture = "after" if capture == "before" else "before"
                    side = item["sides"][source_capture]
                    source_id = frames.get(source_capture)
                    if source_id is None:
                        if "select" in payload:
                            source_id = (
                                side["frames"][0]["id"] if side["frames"] else None
                            )
                        else:
                            source_id = self.views[
                                source_capture
                            ].inspection.selected_evidence
                    obj, _ = self.views[source_capture].inspection.inventory.object(
                        source_capture, side["id"]
                    )
                    bundle = self.region_frames.candidates(
                        source_capture, obj, source_id, capture
                    )
                    previous = self.references[capture]
                    chosen = frames.get(capture)
                    if (
                        chosen is None
                        and previous
                        and previous["token"] == bundle["token"]
                    ):
                        chosen = (
                            previous["selected"]["id"] if previous["selected"] else None
                        )
                    if chosen is None and bundle["frames"]:
                        chosen = bundle["frames"][0]["id"]
                    record = next(
                        (r for r in bundle["frames"] if r["id"] == chosen), None
                    )
                    if capture in frames and record is None:
                        raise ValueError("Unknown region reference frame")
                    references[capture] = {**bundle, "selected": record}
            # Validate both sides before either scene changes.
            for capture in CAPTURES:
                side = item["sides"][capture] if item else None
                action = {}
                if "select" in payload:
                    action = {"select": side["id"]} if side else {"clear": True}
                    if side and side["frames"]:
                        action["evidence"] = side["frames"][0]["id"]
                if capture in frames:
                    if side is None and references[capture] is not None:
                        actions[capture] = action
                        continue
                    allowed = {f["id"] for f in side["frames"]} if side else set()
                    if frames[capture] not in allowed:
                        raise ValueError("Frame does not support the selected object")
                    action["evidence"] = frames[capture]
                if payload.get("focus") and side and side["point_count"]:
                    action["focus"] = True
                actions[capture] = action
            for capture, view in self.views.items():
                view.update(
                    capture,
                    0,
                    sequence,
                    view.session,
                    settings=settings,
                    fit=payload.get("fit", False),
                    object_action=actions[capture] or None,
                )
            self.selected, self.sequence = selected, sequence
            self.references = references
            return self.snapshot()


def make_comparison_handler(state, scene_ports):
    page = Path(__file__).with_name("comparison.html").read_bytes()
    for capture, port in scene_ports.items():
        page = page.replace(
            ("__" + capture.upper() + "_PORT__").encode(), str(port).encode()
        )

    class Handler(BaseHTTPRequestHandler):
        def send(self, body, content_type="application/json", status=200):
            if content_type == "application/json" and not isinstance(body, bytes):
                body = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            try:
                if path == "/":
                    self.send(page, "text/html; charset=utf-8")
                elif path == "/status":
                    self.send(state.snapshot())
                elif state.report_root is not None and (
                    path in ("/report", "/report.html", "/report.json", "/summary.md")
                    or path.startswith("/assets/")
                ):
                    relative = (
                        "report.html"
                        if path == "/report"
                        else unquote(path.lstrip("/"))
                    )
                    if relative not in state.report_files:
                        raise ValueError("Unknown report asset")
                    body = safe_path(state.report_root, relative).read_bytes()
                    mime = (
                        mimetypes.guess_type(relative)[0] or "application/octet-stream"
                    )
                    self.send(
                        body,
                        mime + ("; charset=utf-8" if mime.startswith("text/") else ""),
                    )
                elif path == "/favicon.ico":
                    self.send(b"", "image/x-icon", 204)
                elif path.startswith("/region/"):
                    parts = path.strip("/").split("/")
                    if len(parts) != 4 or parts[1] not in CAPTURES:
                        raise ValueError("Unknown region reference")
                    data = state.region_frames.image(
                        parts[1], parts[2], unquote(parts[3])
                    )
                    self.send(data, "image/png")
                elif path.startswith("/evidence/"):
                    parts = path.strip("/").split("/")
                    if len(parts) != 4 or parts[1] not in CAPTURES:
                        raise ValueError("Unknown evidence")
                    data = state.views[parts[1]].inspection.evidence(
                        parts[1], unquote(parts[2]), parts[3]
                    )
                    self.send(data, "image/png")
                else:
                    self.send({"error": "Not found"}, status=404)
            except (KeyError, ValueError, FileNotFoundError) as exc:
                self.send({"error": str(exc)}, status=404)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16384:
                    raise ValueError("Invalid request length")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise TypeError("Expected JSON object")
                path = urlparse(self.path).path
                if path == "/session":
                    self.send(state.begin_session())
                elif path == "/inspect":
                    self.send(state.update(payload))
                else:
                    self.send({"error": "Not found"}, status=404)
            except PermissionError as exc:
                self.send({"error": str(exc)}, status=409)
            except (KeyError, ValueError, TypeError) as exc:
                self.send({"error": str(exc)}, status=400)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def log_message(self, *_):
            pass

    return Handler


def serve_comparison(root, inventory, comparison, port, maximum, report=None):
    import viser

    state = ComparisonState(root, inventory, comparison, maximum, report=report)
    servers, http = [], None
    scene_ports = {capture: port + i + 1 for i, capture in enumerate(CAPTURES)}
    try:
        for capture, view in state.views.items():
            server = viser.ViserServer(host="127.0.0.1", port=scene_ports[capture])
            servers.append(server)
            server.gui.configure_theme(
                dark_mode=True,
                show_logo=False,
                show_share_button=False,
                control_layout="floating",
            )
            server.gui.main_panel.minimize()
            server.scene.world_axes.visible = False
            server.scene.set_up_direction("-y")
            view.load(server)
            server.on_client_connect(view.connect)
            server.on_client_disconnect(view.disconnect)
        http = ThreadingHTTPServer(
            ("127.0.0.1", port), make_comparison_handler(state, scene_ports)
        )
        print(
            f"Before / After comparison: http://127.0.0.1:{port} (Ctrl+C to stop)",
            flush=True,
        )
        http.serve_forever()
    except KeyboardInterrupt:
        print("Stopping comparison viewer.", flush=True)
    finally:
        if http:
            http.server_close()
        for server in servers:
            server.stop()
