"""Shared scene/state tests with explicit synthetic evidence and real HTTP routes."""

import json
import tempfile
import threading
import unittest
from contextlib import nullcontext
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from reconstruction_v2 import core
from reconstruction_v2.object_association import consolidate
from reconstruction_v2.object_contracts import ObjectInventory
from reconstruction_v2.object_inspection import (
    ObjectInspection,
    SavedObjectPanel,
    crop_plane,
)
from reconstruction_v2.object_pipeline import run
from reconstruction_v2.sync_viewer import _SyncState, make_handler
from tests.fixtures import PROJECT, completed
from tests.test_objects import AnalyticModels
from tests.test_viewer_ceiling import Gui, Handle


class Scene:
    def __init__(self):
        self.nodes = {}

    def add(self, name, *args, **kwargs):
        handle = Handle(
            visible=True, **{k: v for k, v in kwargs.items() if k != "visible"}
        )
        self.nodes[name] = handle
        return handle

    add_point_cloud = add_label = add_line_segments = add


class InspectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scratch = PROJECT / "outputs/.test_work"
        scratch.mkdir(parents=True, exist_ok=True)
        cls.temporary = tempfile.TemporaryDirectory(prefix="inspection-", dir=scratch)
        cls.base = Path(cls.temporary.name)
        cls.reconstruction = completed(cls.base / "run")
        config = core.read_json(PROJECT / "configs/objects.json")
        config["device"] = "cpu"
        lock = {
            "status": "synthetic",
            "models": {
                kind: {"fixture": True}
                for kind in ("detector", "segmenter", "appearance")
            },
        }
        cls.objects = cls.base / "objects"
        result = run(
            cls.reconstruction, cls.objects, config, lock, models=AnalyticModels()
        )
        if result["status"] == "failed":
            raise AssertionError(result)
        cls.inventory = ObjectInventory(cls.reconstruction, cls.objects)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_full_selection_ignores_global_sampling_and_playback_uses_object_support(
        self,
    ):
        inspection = ObjectInspection(self.inventory, Scene())
        first = self.inventory.listing("before")["objects"][0]
        inspection.select("before", first["id"])
        inspection.refresh(-1)
        self.assertEqual(inspection.visible_count, first["point_count"])
        inspection.options({"mode": "playback"})
        inspection.refresh(-1)
        self.assertEqual(inspection.visible_count, 0)
        inspection.refresh(1)
        self.assertEqual(inspection.visible_count, first["point_count"])
        np.testing.assert_array_equal(
            inspection.selected["centroid"], first["centroid"]
        )
        inspection.options({"isolate": True})
        inspection.clear("after")
        self.assertFalse(inspection.isolate)
        self.assertIsNone(inspection.selected)
        with self.assertRaises(ValueError):
            inspection.select("after", first["id"])

    def test_crop_bypass_and_explicit_selection_crop(self):
        inspection = ObjectInspection(self.inventory, Scene())
        inspection.select(
            "before", self.inventory.listing("before")["objects"][0]["id"]
        )
        settings = {"crop_enabled": True, "crop_direction": "-Y", "crop_percent": 100}
        points = inspection.geometry["points"]
        inspection.refresh(
            crop_settings={**settings, "plane": crop_plane(points, settings)}
        )
        self.assertGreater(inspection.visible_count, 0)
        inspection.options({"crop_selection": True})
        inspection.refresh(
            crop_settings={**settings, "plane": crop_plane(points, settings)}
        )
        self.assertEqual(inspection.visible_count, 0)

    def test_evidence_scope_and_image_types(self):
        inspection = ObjectInspection(self.inventory)
        obj = self.inventory.listing("before")["objects"][0]
        observation = obj["observation_ids"][0]
        for kind in ("original", "overlay", "crop"):
            self.assertTrue(
                inspection.evidence("before", observation, kind).startswith(b"\x89PNG")
            )
        with self.assertRaises(ValueError):
            inspection.evidence("after", observation)
        with self.assertRaises(ValueError):
            inspection.evidence("before", "../../mask")

    def test_session_sequence_selection_isolation_and_capture_switch(self):
        server = SimpleNamespace(scene=Scene(), atomic=lambda: nullcontext())
        state = _SyncState(
            self.reconstruction,
            ["before", "after"],
            maximum=1,
            inventory=self.inventory,
        )
        state.load(server)
        session = state.begin_session()["session"]
        first = self.inventory.listing("before")["objects"][0]
        result = state.update(
            "before",
            0,
            1,
            session,
            object_action={"select": first["id"], "options": {"isolate": True}},
        )
        self.assertEqual(result["inspection"]["selected"]["id"], first["id"])
        self.assertGreater(result["inspection"]["visible_points"], 1)
        self.assertTrue(all(not h.visible for _, h, _ in state.groups["before"]))
        stale = state.update("after", 0, 0, session)
        self.assertTrue(stale["stale"])
        self.assertEqual(state.capture, "before")
        result = state.update("after", 0, 2, session)
        self.assertIsNone(result["inspection"]["selected"])
        self.assertFalse(result["inspection"]["isolate"])
        with self.assertRaises(PermissionError):
            state.update("before", 0, 3, "old-session")

    def test_http_inventory_evidence_and_session_actions(self):
        state = _SyncState(
            self.reconstruction, ["before", "after"], inventory=self.inventory
        )
        http = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(state, {}, {}, 8093))
        thread = threading.Thread(target=http.serve_forever, daemon=True)
        thread.start()
        connection = HTTPConnection("127.0.0.1", http.server_port)
        try:
            connection.request("GET", "/objects/before")
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            result = json.loads(response.read())
            self.assertEqual(len(result["objects"]), 2)
            self.assertTrue(result["synthetic"])
            observation = result["objects"][0]["observation_ids"][0]
            connection.request("GET", f"/evidence/before/{observation}/overlay")
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            self.assertTrue(response.read().startswith(b"\x89PNG"))
            connection.request("GET", f"/evidence/after/{observation}/overlay")
            response = connection.getresponse()
            self.assertEqual(response.status, 404)
            response.read()
        finally:
            connection.close()
            http.shutdown()
            http.server_close()
            thread.join()

    def test_standard_viewer_uses_same_membership_and_resets_capture(self):
        class TextGui(Gui):
            add_text = Gui.control

        context = []
        server = SimpleNamespace(scene=Scene(), gui=TextGui(), get_clients=dict)
        points = self.inventory.captures["before"]["geometry"]["points"]
        panel = SavedObjectPanel(
            server,
            self.inventory,
            context.append,
            lambda: {"crop_enabled": False},
            lambda: points,
            lambda: 0.003,
        )
        panel.capture("before")
        label = next(iter(panel.labels))
        panel.selection.change(label)
        self.assertIsNotNone(panel.inspection.selected)
        panel.isolation.change(True)
        self.assertTrue(context[-1])
        panel.capture("after")
        self.assertIsNone(panel.inspection.selected)
        self.assertFalse(context[-1])

    def test_static_inspection_reveals_complete_capture_without_time_advancing(self):
        state = _SyncState(
            self.reconstruction,
            ["before", "after"],
            maximum=100,
            inventory=self.inventory,
            static=True,
        )
        state.load(SimpleNamespace(scene=Scene(), atomic=lambda: nullcontext()))
        self.assertGreater(state.snapshot()["visible_points"], 0)
        self.assertTrue(all(handle.visible for _, handle, _ in state.groups["before"]))
        self.assertTrue(
            all(not handle.visible for _, handle, _ in state.groups["after"])
        )
        session = state.begin_session()
        self.assertEqual(session["view_mode"], "objects")
        result = state.update("after", 0, 1, session["session"])
        self.assertEqual(result["frame_index"], 1)
        self.assertEqual(result["visible_points"], result["display_points"])
        self.assertTrue(all(handle.visible for _, handle, _ in state.groups["after"]))
        with self.assertRaises(ValueError):
            state.update(
                "after",
                0,
                2,
                session["session"],
                object_action={"options": {"mode": "playback"}},
            )

    def test_static_page_and_actions_have_no_media_dependency(self):
        state = _SyncState(
            self.reconstruction,
            ["before", "after"],
            inventory=self.inventory,
            static=True,
        )
        http = ThreadingHTTPServer(
            ("127.0.0.1", 0), make_handler(state, {}, {}, 8095, "objects.html")
        )
        thread = threading.Thread(target=http.serve_forever, daemon=True)
        thread.start()
        connection = HTTPConnection("127.0.0.1", http.server_port)
        try:
            connection.request("GET", "/")
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            page = response.read().decode()
            self.assertIn('id="object-list"', page)
            self.assertIn('id="scene"', page)
            self.assertNotIn("<video", page)
            self.assertNotIn('id="play"', page)
            self.assertNotIn('id="seek"', page)
            self.assertNotIn('id="object-mode"', page)
            connection.request(
                "POST",
                "/session",
                body="{}",
                headers={"Content-Type": "application/json"},
            )
            session = json.loads(connection.getresponse().read())
            payload = {"capture": "after", "session": session["session"], "sequence": 1}
            connection.request(
                "POST",
                "/inspect",
                body=json.dumps(payload),
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            self.assertEqual(json.loads(response.read())["frame_index"], 1)
            connection.request("GET", "/video/before")
            response = connection.getresponse()
            self.assertEqual(response.status, 404)
            response.read()
        finally:
            connection.close()
            http.shutdown()
            http.server_close()
            thread.join()

    def test_static_startup_does_not_validate_or_transcode_source_video(self):
        from reconstruction_v2.inspection_viewer import object_view

        with (
            patch("reconstruction_v2.inspection_viewer.check_viewer_ports"),
            patch("reconstruction_v2.inspection_viewer.serve_viewer") as serve,
            patch(
                "reconstruction_v2.sync_viewer.validate_video_source",
                side_effect=AssertionError("Video must not be needed"),
            ),
            patch(
                "reconstruction_v2.sync_viewer.prepare_playback",
                side_effect=AssertionError("No transcoding"),
            ),
        ):
            object_view(self.reconstruction, self.objects)
        self.assertTrue(serve.call_args.args[0].static)
        self.assertEqual(serve.call_args.kwargs["page_name"], "objects.html")


class AssociationTests(unittest.TestCase):
    def associate(self, frames, vectors, points=None):
        settings = core.read_json(PROJECT / "configs/objects.json")["association"]
        observations = [
            {"id": str(i), "status": "usable", "label": "chair", "frame_index": frame}
            for i, frame in enumerate(frames)
        ]
        cloud = np.array(
            [
                [x, y, 2]
                for x in np.linspace(0, 0.5, 12)
                for y in np.linspace(0, 0.5, 12)
            ]
        )
        arrays = [
            {"points": cloud if points is None else points[i]}
            for i in range(len(frames))
        ]
        with patch(
            "reconstruction_v2.object_association.reprojection",
            return_value={"samples": 50, "fraction": 1.0},
        ):
            return consolidate(
                observations,
                arrays,
                np.array(vectors),
                None,
                None,
                None,
                None,
                settings,
                0.05,
            )

    def test_distinct_covisible_instances_do_not_merge(self):
        groups, _ = self.associate([0, 0], [[1, 0], [1, 0]])
        self.assertEqual(groups, [[0], [1]])

    def test_bridge_cannot_override_group_conflict(self):
        theta = np.deg2rad([0, 30, 60])
        groups, decisions = self.associate(
            [0, 1, 2], np.c_[np.cos(theta), np.sin(theta)]
        )
        self.assertEqual(groups, [[0], [1], [2]])
        self.assertTrue(
            any("competing_identity_support" in d["reasons"] for d in decisions)
        )

    def test_disjoint_surface_views_can_merge_with_positive_reprojection(self):
        cloud = np.array(
            [
                [x, y, 2]
                for x in np.linspace(0, 0.5, 12)
                for y in np.linspace(0, 0.5, 12)
            ]
        )
        groups, _ = self.associate(
            [0, 1], [[1, 0], [1, 0]], [cloud, cloud + [0, 0, 0.3]]
        )
        self.assertEqual(groups, [[0, 1]])


if __name__ == "__main__":
    unittest.main()
