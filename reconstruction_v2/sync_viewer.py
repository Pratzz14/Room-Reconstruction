"""Local video-driven progressive reconstruction, with a fitted free-orbit view."""

import json
import math
import secrets
import socket
import threading
import time
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from .core import read_json, verify_capture
from .sync import (
    first_observation_frames,
    fit_room,
    load_timeline,
    room_bounds,
    sample_indices,
    validate_video_source,
)
from .sync_media import byte_range, prepare_playback
from .viewer import ceiling_crop_indices

DEFAULT_SETTINGS = {"show_path": True, "point_size": 0.003, "crop_enabled": False,
                    "crop_direction": "-Y", "crop_percent": 15.0}


class _SyncState:
    def __init__(self, root, captures, maximum=200000, inventory=None, static=False):
        self.root, self.captures = Path(root), captures
        self.capture, self.playback_time, self.end = captures[0], 0.0, -1
        self.lock = threading.RLock()
        self.server = self.scene = None
        self.timeline = {name: load_timeline(root, name) for name in captures}
        self.static = static
        self.comparison = getattr(inventory, "comparison", None)
        self.after_transform = getattr(inventory, "after_transform", None)
        self.overlay = self.after_transform is not None
        if static:
            self.end = len(self.timeline[self.capture].frames) - 1
            self.playback_time = self.timeline[self.capture].timestamps[-1]
        self.points, self.colors, self.observation_frames, self.cameras = {}, {}, {}, {}
        self.groups, self.camera_handles, self.path_handles = {}, {}, {}
        self.display_indices, self.bounds, self.clients = {}, {}, {}
        self.settings = {name: dict(DEFAULT_SETTINGS) for name in captures}
        if static and not self.comparison:
            for settings in self.settings.values():
                settings["show_path"] = False
        self.maximum = int(maximum)
        if self.maximum < 1:
            raise ValueError("Display point limit must be positive")
        self.session, self.sequence, self.geometry_updates = None, -1, 0
        from .object_inspection import ObjectInspection
        self.inspection = ObjectInspection(inventory) if inventory is not None else None
        self.object_refresh_key = None
        self.frame_point_indices = {}

    def atomic(self):
        return self.server.atomic() if self.server else nullcontext()

    def begin_session(self):
        with self.lock:
            self.session, self.sequence = secrets.token_urlsafe(24), -1
            return {"session": self.session, "captures": self.captures, "initial": self.captures[0],
                    "timestamps": {n: list(t.timestamps) for n, t in self.timeline.items()},
                    "settings": self.settings, "maximum_points": self.maximum,
                    "objects_available": self.inspection is not None,
                    "view_mode": "objects" if self.static else "playback",
                    "comparison": self.comparison["alignment"] if self.comparison else None,
                    "overlay": self.overlay}

    def snapshot(self):
        with self.lock:
            end = len(self.timeline[self.capture].frames) - 1 if self.static else self.timeline[self.capture].frame_index(self.playback_time)
            indices = self.display_indices.get(self.capture, np.array([], dtype=int))
            first = self.observation_frames.get(self.capture, np.array([], dtype=int))
            return {"capture": self.capture, "time": self.playback_time, "sequence": self.sequence,
                    "frame_count": len(self.timeline[self.capture].frames), "frame_index": end,
                    "revealed_frames": end + 1,
                    "visible_points": int(np.count_nonzero(first[indices] <= end)) if len(indices) else 0,
                    "display_points": len(indices), "empty_crop": self.bounds.get(self.capture) is None,
                    # An inactive tab may never acknowledge its camera fit; it must not block active viewers.
                    "ready": any(c["framed"] for c in self.clients.values()),
                    "geometry_updates": self.geometry_updates,
                    "overlay": self.overlay,
                    "settings": self.settings[self.capture],
                    "inspection": self.inspection.snapshot() if self.inspection else None,
                    "cameras": [{"id": key, "manual": c["manual"], "aspect": c["aspect"],
                                 "position": list(c["client"].camera.position),
                                 "look_at": list(c["client"].camera.look_at)} for key, c in self.clients.items()]}

    def load(self, server):
        self.server, self.scene = server, server.scene
        if self.inspection:
            self.inspection.scene = server.scene
            self.inspection.source_points_provider = self.source_frame_points
            self.inspection.clear(self.capture)
        for name in self.captures:
            with np.load(self.root / name / "cloud.npz", allow_pickle=False) as data:
                self.points[name], self.colors[name] = data["points"], data["colors"]
            if self.static:
                self.observation_frames[name] = np.zeros(len(self.points[name]), dtype=np.int32)
            else:
                with np.load(self.root / name / "observations.npz", allow_pickle=False) as data:
                    self.observation_frames[name] = first_observation_frames(data, len(self.points[name]))
            self.cameras[name] = read_json(self.root / name / "cameras.json")
            if name == "after" and self.after_transform:
                from .alignment import apply_similarity
                from .comparison import transformed_cameras
                self.points[name] = apply_similarity(self.points[name], self.after_transform).astype(np.float32)
                self.cameras[name] = transformed_cameras(self.cameras[name], self.after_transform)
            self._build_groups(name)
            self._build_cameras(name)
            if self.inspection:
                self.inspection.support_cameras[name] = self.cameras[name]
                self.inspection.camera_scales[name] = max(float(np.linalg.norm(
                    room_bounds(self.points[name])[1] - room_bounds(self.points[name])[0])) * 0.04, 0.001)
        self._visibility()

    def source_frame_points(self, capture, frame_index):
        """Existing cloud points with any retained observation in this frame."""
        if capture not in self.captures or not 0 <= frame_index < len(self.timeline[capture].frames):
            raise ValueError("Unknown source frame")
        key = (capture, frame_index)
        if key not in self.frame_point_indices:
            with np.load(self.root / capture / "observations.npz", allow_pickle=False) as data:
                rows = data["observation_frame_index"] == frame_index
                indices = np.unique(data["cloud_point_index"][rows])
            if len(self.frame_point_indices) >= 8:
                self.frame_point_indices.pop(next(iter(self.frame_point_indices)))
            self.frame_point_indices[key] = indices
        # self.points already uses the displayed coordinate system, including
        # an optional After transform when comparison is explicitly enabled.
        return self.points[capture][self.frame_point_indices[key]]

    def _build_groups(self, capture):
        for _, handle, _ in self.groups.get(capture, []):
            handle.remove()
        settings, points = self.settings[capture], self.points[capture]
        retained = (ceiling_crop_indices(points, settings["crop_direction"], settings["crop_percent"])
                    if settings["crop_enabled"] else np.arange(len(points), dtype=np.int64))
        if settings["crop_enabled"] and settings["crop_percent"] == 100:
            retained = np.empty(0, dtype=np.int64)
        self.bounds[capture] = room_bounds(points[retained])
        indices = sample_indices(retained, self.maximum)
        self.display_indices[capture] = indices
        self.groups[capture] = []
        first = self.observation_frames[capture]
        for frame in (range(1) if self.static else range(len(self.timeline[capture].frames))):
            selected = indices[first[indices] == frame]
            if not len(selected):
                continue
            handle = self.scene.add_point_cloud(
                f"/sync/{capture}/points_{frame}", points[selected],
                np.tile((80, 160, 255) if capture == "before" else (255, 170, 70), (len(selected), 1)).astype(np.uint8) if self.overlay else self.colors[capture][selected],
                point_size=settings["point_size"], precision="float32", visible=False)
            self.groups[capture].append((frame, handle, len(selected)))
        self.geometry_updates += 1

    def _build_cameras(self, capture):
        self.camera_handles[capture], self.path_handles[capture] = [], []
        bounds = room_bounds(self.points[capture])
        scale = max(float(np.linalg.norm(bounds[1] - bounds[0])) * 0.025, 0.001)
        positions = []
        for index, camera in enumerate(self.cameras[capture]):
            width, height = camera["model_width"], camera["model_height"]
            corners = np.array([[-0.5, -0.5, 1], [width - 0.5, -0.5, 1],
                                [width - 0.5, height - 0.5, 1], [-0.5, height - 0.5, 1]])
            corners = corners @ np.linalg.inv(np.array(camera["intrinsics"])).T * scale
            lines = np.concatenate((np.stack((np.zeros_like(corners), corners), axis=1),
                                    np.stack((corners, np.roll(corners, -1, axis=0)), axis=1)))
            pose = np.asarray(camera["camera_to_world"])
            positions.append(pose[:3, 3])
            lines = lines @ pose[:3, :3].T + pose[:3, 3]
            self.camera_handles[capture].append(self.scene.add_line_segments(
                f"/sync/{capture}/camera_{index}", lines.astype(np.float32), (80, 190, 255),
                thickness=2, thickness_units="screen", visible=False))
            if index:
                segment = np.array([[positions[-2], positions[-1]]], dtype=np.float32)
                self.path_handles[capture].append(self.scene.add_line_segments(
                    f"/sync/{capture}/path_{index}", segment, (255, 159, 67),
                    thickness=2, thickness_units="screen", visible=False))

    def _visibility(self):
        isolated = self.inspection is not None and self.inspection.isolate
        for name in self.captures:
            active = (name == self.capture or self.overlay) and not isolated
            for frame, handle, _ in self.groups.get(name, []):
                handle.visible = active and frame <= self.end
            show = active and self.settings[name]["show_path"]
            for index, handle in enumerate(self.camera_handles.get(name, [])):
                handle.visible = show and (self.static or index == self.end)
            for index, handle in enumerate(self.path_handles.get(name, []), 1):
                handle.visible = show and (self.static or index <= self.end)

    def fit(self, client=None):
        for record in list(self.clients.values()):
            if client is not None and record["client"] != client:
                continue
            camera = record["client"].camera
            if self.inspection and self.inspection.focus_active and self.inspection.selected:
                fitted = self.inspection.fit(record["client"])
                if fitted is not None:
                    record.update(manual=False, aspect=camera.aspect, fitted=fitted, awaiting_fit=True)
                continue
            fitted = fit_room(self.bounds.get(self.capture), camera.aspect)
            if fitted is None:
                record["framed"] = True  # An explicitly empty crop still permits changing controls.
                continue
            record.update(manual=False, aspect=camera.aspect, fitted=fitted, awaiting_fit=True)
            with record["client"].atomic():
                camera.fov = fitted["fov"]
                camera.up_direction = tuple(fitted["up_direction"])
                # Position precedes look_at: position also translates Viser's orbit target.
                camera.position = tuple(fitted["position"])
                camera.look_at = tuple(fitted["look_at"])
                camera.near, camera.far = fitted["near"], fitted["far"]
                camera.min_orbit_distance = fitted["radius"] * 0.01
                camera.max_orbit_distance = fitted["far"] * 10

    def connect(self, client):
        with self.lock:
            client.gui.main_panel.minimize()
            self.clients[client.client_id] = {"client": client, "manual": False,
                                              "aspect": client.camera.aspect, "fitted": None,
                                              "awaiting_fit": False, "framed": False}
            self.fit(client)

        @client.camera.on_update
        def camera_changed(camera):
            with self.lock:
                record = self.clients.get(client.client_id)
                if not record:
                    return
                resized = not math.isclose(record["aspect"], camera.aspect, rel_tol=0.001)
                if resized:
                    record["aspect"] = camera.aspect
                    # Viser switches to a bottom sheet on narrow screens; collapse that view too.
                    client.gui.main_panel.minimize()
                    if not record["manual"]:
                        self.fit(client)
                    return
                fitted = record["fitted"]
                matches = fitted is not None and (
                    np.allclose(camera.position, fitted["position"], rtol=0.001, atol=fitted["radius"] * 0.001)
                    and np.allclose(camera.look_at, fitted["look_at"], rtol=0.001, atol=fitted["radius"] * 0.001))
                if record["awaiting_fit"]:
                    # Camera reports already in flight when we fit are not user orbit gestures.
                    if matches:
                        record["awaiting_fit"] = False
                        record["framed"] = True
                    return
                if fitted and not matches:
                    record["manual"] = True

    def disconnect(self, client):
        with self.lock:
            self.clients.pop(client.client_id, None)

    def update(self, capture, playback_time, sequence=None, session=None, settings=None, fit=False, object_action=None):
        if capture not in self.captures:
            raise ValueError(f"Unknown capture: {capture}")
        if self.static:
            playback_time = self.timeline[capture].timestamps[-1]
        end = self.timeline[capture].frame_index(float(playback_time))
        if sequence is not None and (type(sequence) is not int or sequence < 0):
            raise ValueError("Sequence must be a non-negative integer")
        if type(fit) is not bool:
            raise ValueError("Fit must be boolean")
        with self.lock:
            if session is not None and session != self.session:
                raise PermissionError("Another page controls this viewer. Reload to take control.")
            if sequence is not None and sequence <= self.sequence:
                return {**self.snapshot(), "stale": True}
            if object_action is not None:
                if (not self.inspection or not isinstance(object_action, dict)
                        or set(object_action) - {"select", "clear", "options", "focus", "inventory_mode", "overlay", "evidence"}):
                    raise ValueError("Object inspection is unavailable or action is invalid")
                if "inventory_mode" in object_action:
                    if not hasattr(self.inspection.inventory, "use_mode") or object_action["inventory_mode"] not in ("original", "refined"):
                        raise ValueError("Inventory mode is unavailable")
                    if "select" in object_action:
                        raise ValueError("Change inventory mode before selecting an object")
                if "select" in object_action:
                    self.inspection.inventory.object(capture, object_action["select"])
                if "overlay" in object_action and (not self.after_transform or type(object_action["overlay"]) is not bool):
                    raise ValueError("Aligned overlay is unavailable or invalid")
                for key in ("clear", "focus"):
                    if key in object_action and type(object_action[key]) is not bool:
                        raise ValueError("Object action must be boolean")
                if "options" in object_action and not isinstance(object_action["options"], dict):
                    raise ValueError("Invalid object display options")
                if self.static and object_action.get("options", {}).get("mode", "full") != "full":
                    raise ValueError("Object inspection always shows full saved geometry")
            chosen = dict(self.settings[capture])
            if settings is not None:
                if not isinstance(settings, dict) or set(settings) - set(DEFAULT_SETTINGS):
                    raise ValueError("Invalid display settings")
                chosen.update(settings)
                if (type(chosen["show_path"]) is not bool or type(chosen["crop_enabled"]) is not bool
                        or chosen["crop_direction"] not in ("-X", "+X", "-Y", "+Y", "-Z", "+Z")
                        or not isinstance(chosen["point_size"], (int, float))
                        or not 0.0005 <= chosen["point_size"] <= 0.03
                        or not isinstance(chosen["crop_percent"], (int, float))
                        or not 0 <= chosen["crop_percent"] <= 100):
                    raise ValueError("Display settings outside supported range")
            if object_action and "inventory_mode" in object_action:
                self.inspection.clear(capture)
                self.inspection.inventory.use_mode(object_action["inventory_mode"])
            if object_action and "overlay" in object_action:
                self.overlay = object_action["overlay"]
                if self.scene is not None:
                    for name in self.captures:
                        self._build_groups(name)
            previous = self.settings[capture]
            cropped = any(chosen[k] != previous[k] for k in ("crop_enabled", "crop_percent", "crop_direction"))
            changed = capture != self.capture
            visibility = changed or self.end != end or chosen["show_path"] != previous["show_path"]
            self.capture, self.playback_time, self.end = capture, float(playback_time), end
            self.settings[capture] = chosen
            if self.inspection:
                if changed:
                    self.inspection.clear(capture)
                if object_action:
                    if object_action.get("clear"):
                        self.inspection.clear(capture)
                    if "select" in object_action:
                        self.inspection.select(capture, object_action["select"])
                    if "evidence" in object_action:
                        self.inspection.set_evidence(object_action["evidence"])
                    if "options" in object_action:
                        self.inspection.options(object_action["options"])
                if fit:
                    self.inspection.focus_active = False
            if sequence is not None:
                self.sequence = sequence
            if self.scene is not None:
                with self.atomic():
                    if cropped:
                        self._build_groups(capture)
                    elif chosen["point_size"] != previous["point_size"]:
                        for _, handle, _ in self.groups[capture]:
                            handle.point_size = chosen["point_size"]
                    if visibility or cropped:
                        self._visibility()
                    if self.inspection:
                        import json as _json
                        key = (capture, end, self.inspection.identifier, _json.dumps(chosen, sort_keys=True),
                               _json.dumps(self.inspection.snapshot(), sort_keys=True, default=str))
                        if key != self.object_refresh_key:
                            from .object_inspection import crop_plane
                            self.inspection.refresh(end, {**chosen, "plane": crop_plane(self.points[capture], chosen)}, chosen["point_size"])
                            # Store the post-refresh state so visible-count updates do not trigger an extra rebuild.
                            self.object_refresh_key = (capture, end, self.inspection.identifier, _json.dumps(chosen, sort_keys=True),
                                                       _json.dumps(self.inspection.snapshot(), sort_keys=True, default=str))
                        self._visibility()
                if changed or cropped or fit:
                    self.fit()
                if object_action and object_action.get("focus"):
                    if not self.inspection.selected or not self.inspection.selected.get("point_count"):
                        raise ValueError("Select an object before focusing")
                    self.inspection.focus_active = True
                    self.fit()
            return self.snapshot()


def make_handler(state, videos, media_records, scene_port, page_name="sync.html"):
    page = (Path(__file__).parent / page_name).read_bytes().replace(b"__SCENE_PORT__", str(scene_port).encode())

    class Handler(BaseHTTPRequestHandler):
        def json_response(self, value, status=200):
            body = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_HEAD(self):
            self.serve(head=True)

        def do_GET(self):
            self.serve()

        def serve(self, head=False):
            path = urlparse(self.path).path
            try:
                if path == "/":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(page)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    if not head:
                        self.wfile.write(page)
                elif path == "/status" and not head:
                    self.json_response(state.snapshot())
                elif path.startswith("/objects/") and not head:
                    capture = path.removeprefix("/objects/")
                    if capture not in state.captures:
                        self.send_error(404)
                        return
                    self.json_response(state.inspection.inventory.listing(capture) if state.inspection else
                                       {"objects": [], "unresolved": [], "status": "unavailable"})
                elif path.startswith("/evidence/") and not head:
                    parts = path.strip("/").split("/")
                    if len(parts) != 4 or not state.inspection:
                        self.send_error(404)
                        return
                    try:
                        from urllib.parse import unquote
                        data = state.inspection.evidence(parts[1], unquote(parts[2]), parts[3])
                    except (ValueError, KeyError):
                        self.send_error(404)
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                elif path.startswith("/video/"):
                    name = path.removeprefix("/video/")
                    if name not in videos:
                        self.send_error(404)
                        return
                    video = videos[name]
                    size = video.stat().st_size
                    try:
                        start, end, status = byte_range(self.headers.get("Range"), size)
                    except ValueError:
                        self.send_response(416)
                        self.send_header("Content-Range", f"bytes */{size}")
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    self.send_response(status)
                    self.send_header("Content-Type", "video/mp4")
                    self.send_header("Content-Length", str(end - start + 1))
                    self.send_header("Accept-Ranges", "bytes")
                    if status == 206:
                        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                    self.end_headers()
                    if not head:
                        with video.open("rb") as stream:
                            stream.seek(start)
                            remaining = end - start + 1
                            while remaining:
                                chunk = stream.read(min(1024 * 1024, remaining))
                                if not chunk:
                                    break
                                self.wfile.write(chunk)
                                remaining -= len(chunk)
                elif path == "/favicon.ico":
                    self.send_response(204)
                    self.end_headers()
                else:
                    self.send_error(404)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def do_POST(self):
            path = urlparse(self.path).path
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 <= length <= 16384:
                    raise ValueError("Invalid request length")
                payload = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(payload, dict):
                    raise TypeError("Expected JSON object")
                if path == "/session":
                    self.json_response({**state.begin_session(), "media": media_records})
                elif path == "/sync" or (path == "/inspect" and state.static):
                    if not isinstance(payload.get("session"), str) or type(payload.get("sequence")) is not int:
                        raise ValueError("Session and sequence are required")
                    started = time.perf_counter()
                    result = state.update(payload["capture"], payload.get("time", 0) if state.static else payload["time"], payload["sequence"],
                                          payload["session"], payload.get("settings"), payload.get("fit", False), payload.get("object_action"))
                    self.json_response({**result, "server_ms": (time.perf_counter() - started) * 1000})
                else:
                    self.send_error(404)
            except PermissionError as exc:
                self.json_response({"error": str(exc)}, 409)
            except (KeyError, ValueError, TypeError, OverflowError) as exc:
                self.json_response({"error": str(exc)}, 400)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def log_message(self, *_):
            return

    return Handler


def check_viewer_ports(port):
    if not 1 <= port <= 65534:
        raise ValueError("Choose a port between 1 and 65534 (the scene uses the next port)")
    # Viser otherwise silently chooses a new port, breaking the embedded page.
    sockets = []
    try:
        for requested in (port, port + 1):
            probe = socket.socket()
            sockets.append(probe)
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            probe.bind(("127.0.0.1", requested))
    except OSError as exc:
        raise RuntimeError(f"Viewer port {requested} is occupied. Choose another --port.") from exc
    finally:
        for probe in sockets:
            probe.close()


def synchronized_view(root, port=8092, before_video=None, after_video=None, objects=None):
    root = Path(root).resolve()
    check_viewer_ports(port)
    manifest = read_json(root / "run_manifest.json")
    captures = [name for name in ("before", "after") if manifest["captures"][name]["status"] == "completed"]
    if not captures:
        raise ValueError("No completed captures are available; run GPU inference first")
    videos, media = {}, {}
    overrides = {"before": before_video, "after": after_video}
    for name in captures:
        print(f"Verifying saved {name} capture and source video...", flush=True)
        verify_capture(root, name)
        source = validate_video_source(root, name, overrides[name])
        source_hash = read_json(root / name / "source.json")["source_video_sha256"]
        videos[name], media[name] = prepare_playback(source, source_hash, root.parent / "sync_cache")
    maximum = read_json(root / "config.json").get("viewer", {}).get("max_points", 200000)
    from .object_contracts import ObjectInventory
    inventory = ObjectInventory(root, objects) if objects else None
    state = _SyncState(root, captures, maximum, inventory)
    serve_viewer(state, port, videos, media)


def serve_viewer(state, port, videos=None, media=None, page_name="sync.html", title="Synchronized viewer"):
    import viser

    scene_server, http = None, None
    try:
        scene_server = viser.ViserServer(host="127.0.0.1", port=port + 1)
        scene_server.gui.configure_theme(dark_mode=True, show_logo=False, show_share_button=False,
                                         control_layout="floating")
        scene_server.gui.main_panel.float(x=-8, y=8, width=180)
        scene_server.gui.main_panel.minimize()
        scene_server.scene.world_axes.visible = False
        scene_server.scene.set_up_direction("-y")
        state.load(scene_server)
        scene_server.on_client_connect(state.connect)
        scene_server.on_client_disconnect(state.disconnect)
        http = ThreadingHTTPServer(("127.0.0.1", port), make_handler(state, videos or {}, media or {}, port + 1, page_name))
        print(f"{title}: http://127.0.0.1:{port} (Ctrl+C to stop)", flush=True)
        http.serve_forever()
    except KeyboardInterrupt:
        print(f"Stopping {title.lower()}.", flush=True)
    finally:
        if http:
            http.server_close()
        if scene_server:
            scene_server.stop()
