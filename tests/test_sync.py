import json
import math
import socket
import tempfile
import threading
import unittest
from contextlib import nullcontext
from fractions import Fraction
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from reconstruction import core
from reconstruction.sync import (
    FrameTimeline,
    first_observation_frames,
    fit_room,
    room_bounds,
    sample_indices,
    validate_video_source,
)
from reconstruction.sync_media import byte_range, prepare_playback
from reconstruction.sync_viewer import _SyncState, make_handler, synchronized_view
from tests.fixtures import PROJECT, source_frames


def workspace_temp():
    directory = PROJECT / "outputs/.test_work"
    directory.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=directory, prefix="sync-")


class TimelineTests(unittest.TestCase):
    def test_reveals_saved_frames_at_timestamp_boundaries(self):
        timeline = FrameTimeline([
            {"elapsed_timestamp_sec": 0.4},
            {"elapsed_timestamp_sec": 1.2},
            {"elapsed_timestamp_sec": 2.0},
        ])
        self.assertEqual(timeline.frame_index(0.0), -1)
        self.assertEqual(timeline.frame_index(0.4), 0)
        self.assertEqual(list(timeline.revealed_indices(1.2)), [0, 1])
        self.assertEqual(timeline.frame_index(10), 2)

    def test_rejects_invalid_timeline(self):
        with self.assertRaises(ValueError):
            FrameTimeline([])
        with self.assertRaises(ValueError):
            FrameTimeline([{"elapsed_timestamp_sec": 2}, {"elapsed_timestamp_sec": 1}])
        with self.assertRaises(ValueError):
            FrameTimeline([{"elapsed_timestamp_sec": -1}])
        with self.assertRaises(ValueError):
            FrameTimeline([{"elapsed_timestamp_sec": math.nan}])


class VideoSourceTests(unittest.TestCase):
    def test_source_path_is_hash_checked(self):
        with workspace_temp() as tmp:
            root = Path(tmp)
            video = root / "before.mp4"
            video.write_bytes(b"synthetic video")
            core.write_json(root / "before" / "source.json", {
                "source_video_path": str(video),
                "source_video_sha256": core.digest(video),
            })
            self.assertEqual(validate_video_source(root, "before"), video.resolve())

    def test_missing_or_changed_source_fails(self):
        with workspace_temp() as tmp:
            root = Path(tmp)
            video = root / "before.mp4"
            video.write_bytes(b"synthetic video")
            core.write_json(root / "before" / "source.json", {
                "source_video_path": str(video),
                "source_video_sha256": "wrong",
            })
            with self.assertRaises(ValueError):
                validate_video_source(root, "before")


class SyncStateTests(unittest.TestCase):
    def test_capture_switch_and_backward_seek_recompute_reveal_count(self):
        with workspace_temp() as tmp:
            root = Path(tmp)
            source = root / "source"
            source_frames(source)
            state = _SyncState(source, ["before", "after"])
            self.assertEqual(state.update("before", 1.0)["revealed_frames"], 2)
            self.assertEqual(state.update("before", 0.0)["revealed_frames"], 1)
            result = state.update("after", 10.0)
            self.assertEqual(result["capture"], "after")
            self.assertEqual(result["revealed_frames"], 2)


class GeometryTests(unittest.TestCase):
    def test_earliest_support_is_independent_of_representative(self):
        observations = {"cloud_point_index": np.array([0, 1, 0, 1]),
                        "observation_frame_index": np.array([1, 3, 0, 2]),
                        "representative_observation": np.array([0, 1])}
        np.testing.assert_array_equal(first_observation_frames(observations, 2), [0, 2])
        with self.assertRaises(ValueError):
            first_observation_frames(observations, 3)

    def test_framing_projects_translated_room_inside_every_viewport(self):
        for center in ([0, 0, 0], [150, -77, 400]):
            for size in (0.001, 1, 100):
                low, high = np.array(center) - np.array([2, 1, 3]) * size, np.array(center) + np.array([2, 1, 3]) * size
                corners = np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])])
                for aspect in (0.2, 1, 2.5):
                    fitted = fit_room([low, high], aspect)
                    forward = fitted["look_at"] - fitted["position"]
                    forward /= np.linalg.norm(forward)
                    right = np.cross(forward, fitted["up_direction"])
                    right /= np.linalg.norm(right)
                    up = np.cross(right, forward)
                    relative = corners - fitted["position"]
                    z = relative @ forward
                    self.assertTrue(np.all(z > fitted["near"]))
                    self.assertTrue(np.all(z < fitted["far"]))
                    self.assertTrue(np.all(np.abs(relative @ right) / z <= math.tan(fitted["fov"] / 2) * aspect))
                    self.assertTrue(np.all(np.abs(relative @ up) / z <= math.tan(fitted["fov"] / 2)))

    def test_outliers_empty_and_deterministic_point_budget(self):
        points = np.tile(np.array([[0., 0, 0], [2, 3, 4]]), (100, 1))
        bounds = room_bounds(np.concatenate((points, [[1e9, 1e9, 1e9]])))
        self.assertLess(bounds.max(), 5)
        self.assertIsNone(room_bounds(np.empty((0, 3))))
        self.assertIsNone(fit_room(None, 1))
        ids = np.arange(100)
        chosen = sample_indices(ids, 12)
        np.testing.assert_array_equal(chosen, sample_indices(ids, 12))
        self.assertEqual(len(set(chosen)), 12)
        self.assertEqual(len(sample_indices(np.array([], dtype=int), 12)), 0)


class FakeScene:
    def __init__(self):
        self.added = 0

    def add_point_cloud(self, name, points, colors, **kwargs):
        self.added += 1
        return SimpleNamespace(visible=False, point_size=kwargs["point_size"], remove=lambda: None)


class CameraTests(unittest.TestCase):
    def test_fits_on_connect_preserves_manual_view_and_recovers(self):
        class Camera:
            aspect, position, look_at = 1.0, (0, 0, 0), (0, 0, 1)
            def on_update(self, callback):
                self.callback = callback
        with workspace_temp() as tmp:
            source_frames(Path(tmp) / "source")
            state = _SyncState(Path(tmp) / "source", ["before"])
            state.bounds["before"] = np.array([[-1., -1, -1], [1., 1, 1]])
            client = SimpleNamespace(client_id=3, camera=Camera(), atomic=nullcontext,
                                     gui=SimpleNamespace(main_panel=SimpleNamespace(minimize=lambda: None)))
            state.connect(client)
            self.assertFalse(state.snapshot()["ready"])
            inactive = SimpleNamespace(client_id=4, camera=Camera(), atomic=nullcontext, gui=client.gui)
            state.connect(inactive)
            self.assertFalse(state.snapshot()["ready"])
            original = np.array(client.camera.position)
            client.camera.position = (0, 0, 0)  # Delayed report from before the server fitted the room.
            client.camera.callback(client.camera)
            self.assertFalse(state.clients[3]["manual"])
            client.camera.position = original
            client.camera.callback(client.camera)  # Acknowledge the fitted view.
            self.assertTrue(state.snapshot()["ready"])
            self.assertFalse(state.clients[4]["framed"])
            client.camera.aspect = 0.3
            client.camera.callback(client.camera)
            self.assertGreater(np.linalg.norm(client.camera.position), np.linalg.norm(original))
            client.camera.callback(client.camera)
            client.camera.position = (100, 0, 0)
            client.camera.callback(client.camera)
            self.assertTrue(state.clients[3]["manual"])
            state.update("before", 1)
            self.assertEqual(client.camera.position, (100, 0, 0))
            client.camera.aspect = 2
            client.camera.callback(client.camera)
            self.assertEqual(client.camera.position, (100, 0, 0))
            state.fit()
            self.assertFalse(state.clients[3]["manual"])
            self.assertLess(np.linalg.norm(client.camera.position), 100)
            np.testing.assert_allclose(client.camera.look_at, [0, 0, 0])


class ProgressiveSceneTests(unittest.TestCase):
    def test_seek_visibility_stale_updates_and_no_geometry_retransmission(self):
        with workspace_temp() as tmp:
            source_frames(Path(tmp) / "source")
            state = _SyncState(Path(tmp) / "source", ["before", "after"], maximum=4)
            state.scene = FakeScene()
            for name in state.captures:
                state.points[name] = np.array([[0., 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
                state.colors[name] = np.ones((5, 3), np.uint8)
                state.observation_frames[name] = np.array([0, 0, 1, 1, 1])
                state._build_groups(name)
            session = state.begin_session()["session"]
            result = state.update("before", 1, sequence=3, session=session)
            self.assertEqual(result["visible_points"], 4)
            count = state.scene.added
            state.update("before", 1.5, sequence=4, session=session)
            self.assertEqual(state.scene.added, count)
            result = state.update("after", 0, sequence=5, session=session)
            self.assertEqual(result["visible_points"], 2)
            self.assertFalse(any(h.visible for _, h, _ in state.groups["before"]))
            self.assertTrue(state.update("before", 1, sequence=2, session=session)["stale"])
            self.assertEqual(state.capture, "after")
            state.update("before", 0, sequence=6, session=session)
            self.assertEqual(state.snapshot()["visible_points"], 2)
            state.update("before", 0, sequence=7, session=session, settings={"crop_enabled": True, "crop_percent": 100})
            self.assertEqual(state.snapshot()["visible_points"], 0)
            self.assertTrue(state.snapshot()["empty_crop"])
            state.begin_session()
            with self.assertRaises(PermissionError):
                state.update("before", 1, sequence=8, session=session)


class RangeAndHttpTests(unittest.TestCase):
    def test_occupied_port_fails_before_media_or_inference_work(self):
        with socket.socket() as occupied:
            occupied.bind(("127.0.0.1", 0))
            occupied.listen()
            port = occupied.getsockname()[1]
            with (patch("reconstruction.sync_viewer.verify_capture", side_effect=AssertionError("Must fail before verification")),
                  self.assertRaisesRegex(RuntimeError, "occupied")):
                synchronized_view(PROJECT, port)

    def test_range_forms(self):
        for header, expected in ((None, (0, 99, 200)), ("bytes=10-19", (10, 19, 206)),
                                 ("bytes=80-", (80, 99, 206)), ("bytes=-8", (92, 99, 206)),
                                 ("bytes=80-999", (80, 99, 206))):
            self.assertEqual(byte_range(header, 100), expected)
        for header in ("bytes=100-", "bytes=-0", "bytes=20-10", "bytes=0-3,6-8", "bytes=", "bytes=-", "bytes=nan-9"):
            with self.assertRaises(ValueError):
                byte_range(header, 100)

    def test_actual_http_ranges_and_invalid_sync(self):
        with workspace_temp() as tmp:
            root = Path(tmp)
            source_frames(root / "source")
            state = _SyncState(root / "source", ["before", "after"])
            video = root / "playback.mp4"
            video.write_bytes(b"0123456789")
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(state, {"before": video}, {}, 9999))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            connection = HTTPConnection("127.0.0.1", server.server_port)
            try:
                connection.request("GET", "/video/before", headers={"Range": "bytes=-3"})
                response = connection.getresponse()
                self.assertEqual(response.status, 206)
                self.assertEqual(response.getheader("Content-Range"), "bytes 7-9/10")
                self.assertEqual(response.read(), b"789")
                connection.request("HEAD", "/video/before")
                response = connection.getresponse()
                self.assertEqual(response.getheader("Content-Length"), "10")
                self.assertEqual(response.read(), b"")
                connection.request("GET", "/video/before", headers={"Range": "bytes=100-"})
                response = connection.getresponse()
                self.assertEqual(response.status, 416)
                self.assertEqual(response.getheader("Content-Range"), "bytes */10")
                response.read()
                connection.request("POST", "/session", "{}")
                response = connection.getresponse()
                session = json.loads(response.read())["session"]
                payload = {"session": session, "sequence": 1, "capture": "before", "time": -1}
                connection.request("POST", "/sync", json.dumps(payload))
                response = connection.getresponse()
                self.assertEqual(response.status, 400)
                response.read()
                self.assertEqual(state.playback_time, 0)
            finally:
                connection.close()
                server.shutdown()
                server.server_close()
                thread.join()


class PlaybackCopyTests(unittest.TestCase):
    def test_variable_timestamps_nonzero_origin_and_cache_reuse(self):
        import av
        with workspace_temp() as tmp:
            root = Path(tmp)
            source = root / "source.mp4"
            with av.open(str(source), "w") as container:
                stream = container.add_stream("libx264", rate=25)
                audio = container.add_stream("aac", rate=48000)
                audio.layout = "stereo"
                stream.width, stream.height, stream.pix_fmt = 32, 24, "yuv420p"
                stream.codec_context.time_base = Fraction(1, 1000)
                stream.codec_context.max_b_frames = 0
                for index, pts in enumerate((2000, 2040, 2130, 2210)):
                    frame = av.VideoFrame.from_ndarray(np.full((24, 32, 3), index * 50, np.uint8), format="rgb24")
                    frame.pts, frame.time_base = pts, Fraction(1, 1000)
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode(None):
                    container.mux(packet)
                samples = np.sin(np.arange(12000) * (2 * np.pi * 440 / 48000)).astype(np.float32) * 0.1
                sound = av.AudioFrame.from_ndarray(np.stack([samples, samples]), format="fltp", layout="stereo")
                sound.sample_rate, sound.pts, sound.time_base = 48000, 96000, Fraction(1, 48000)
                for packet in audio.encode(sound):
                    container.mux(packet)
                for packet in audio.encode(None):
                    container.mux(packet)
            output, record = prepare_playback(source, core.digest(source), root / "cache")
            with av.open(str(output)) as decoded:
                times = [float(f.pts * f.time_base) for f in decoded.decode(video=0)]
            np.testing.assert_allclose(times, [0, 0.04, 0.13, 0.21], atol=0.0001)
            with av.open(str(output)) as decoded:
                self.assertEqual(decoded.streams.audio[0].codec_context.name, "aac")
                sounds = list(decoded.decode(audio=0))
                self.assertGreater(np.abs(sounds[1].to_ndarray()).mean(), 0.01)
            self.assertEqual(record["frame_count"], 4)
            self.assertAlmostEqual(record["duration"], 0.25, places=4)
            modified = output.stat().st_mtime_ns
            with patch("av.open", side_effect=AssertionError("Cache must not decode or reencode")):
                self.assertEqual(prepare_playback(source, core.digest(source), root / "cache")[0], output)
            self.assertEqual(output.stat().st_mtime_ns, modified)
            output.write_bytes(b"corrupt")
            rebuilt, _ = prepare_playback(source, core.digest(source), root / "cache")
            self.assertGreater(rebuilt.stat().st_size, len(b"corrupt"))
