"""Shared selection across separate scenes, using saved analytic object evidence."""

import copy
import io
import json
import tempfile
import threading
import unittest
from contextlib import nullcontext
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from reconstruction_v2.comparison_viewer import (
    ComparisonState,
    make_comparison_handler,
    readable_explanation,
    uncertainty_messages,
)
from reconstruction_v2.core import read_json, write_json
from reconstruction_v2.object_contracts import ObjectInventory
from reconstruction_v2.object_pipeline import run
from reconstruction_v2.region_reference import project_region
from tests.fixtures import PROJECT, completed
from tests.test_object_inspection import Scene
from tests.test_objects import AnalyticModels


class ComparisonViewerTests(unittest.TestCase):
    def test_uncertainty_reasons_are_readable_and_prioritize_invalid_evidence(self):
        change = {
            "explanation": "Uncertain: identity_ambiguous, invalid_vlm_evidence:invalid_full_frame_box.",
            "uncertainties": [
                "identity_ambiguous",
                "invalid_vlm_evidence:invalid_full_frame_box",
            ],
        }
        messages = uncertainty_messages(change)
        self.assertIn("invalid image region", messages[0]["message"])
        explanation = readable_explanation(change)
        self.assertNotIn("invalid_full_frame_box", explanation)
        self.assertNotIn("identity_ambiguous", explanation)
        self.assertIn("Competing object identities", explanation)

    @classmethod
    def setUpClass(cls):
        scratch = PROJECT / "outputs/.test_work"
        scratch.mkdir(parents=True, exist_ok=True)
        cls.temp = tempfile.TemporaryDirectory(prefix="comparison-view-", dir=scratch)
        cls.base = Path(cls.temp.name)
        cls.root = completed(cls.base / "reconstruction")
        cfg = read_json(PROJECT / "configs/objects.json")
        cfg["device"] = "cpu"
        lock = {
            "models": {
                k: {"fixture": True} for k in ("detector", "segmenter", "appearance")
            }
        }
        run(cls.root, cls.base / "objects", cfg, lock, models=AnalyticModels())
        cls.original = ObjectInventory(cls.root, cls.base / "objects")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        original = self.original

        class Inventory(ObjectInventory):
            def object(self, capture, identifier):
                obj, geometry = super().object(capture, identifier)
                if not obj["point_count"]:
                    geometry = {k: v[:0] for k, v in geometry.items()}
                return obj, geometry

        inventory = Inventory.__new__(Inventory)
        inventory.__dict__.update(original.__dict__)
        inventory.captures = copy.deepcopy(original.captures)
        inventory.comparison = {
            "alignment": {"quality": {"status": "accepted"}},
            "change_counts": {"unchanged": 1},
        }
        inventory.after_transform = None
        before, after = (inventory.captures[c]["objects"] for c in ("before", "after"))
        self.image_id = after[1]["id"]
        after[1].update(point_count=0, centroid=None, bounds=None)
        rows = [
            {
                "shared_id": "both",
                "before_id": before[0]["id"],
                "after_id": after[0]["id"],
                "status": "matched",
            },
            {
                "shared_id": "before-only",
                "before_id": before[1]["id"],
                "after_id": None,
                "status": "unmatched",
            },
            {
                "shared_id": "after-image",
                "before_id": None,
                "after_id": after[1]["id"],
                "status": "unmatched",
            },
        ]
        self.comparison = self.base / "comparison"
        write_json(self.comparison / "identities.json", {"identities": rows})
        self.state = ComparisonState(self.root, inventory, self.comparison, maximum=100)
        self.scenes = {}
        for capture, view in self.state.views.items():
            scene = self.scenes[capture] = Scene()
            view.load(SimpleNamespace(scene=scene, atomic=lambda: nullcontext()))
        self.session = self.state.begin_session()["session"]

    def update(self, sequence, **action):
        return self.state.update(
            {"session": self.session, "sequence": sequence, **action}
        )

    def test_shared_selection_highlights_both_and_auto_opens_evidence(self):
        response = self.update(1, select="both")
        for capture, view in self.state.views.items():
            self.assertFalse(view.overlay)
            self.assertEqual(view.captures, [capture])
            self.assertIsNotNone(
                response["views"][capture]["inspection"]["source_frame"]
            )
            self.assertGreater(view.inspection.visible_count, 0)
            self.assertIn("/object/selected", self.scenes[capture].nodes)

    def test_switching_to_one_side_clears_other_then_clear_clears_both(self):
        self.update(1, select="both")
        before_handles = list(self.state.views["before"].inspection.handles)
        response = self.update(2, select="after-image")
        self.assertIsNone(response["views"]["before"]["inspection"]["selected"])
        self.assertTrue(all(handle.removed for handle in before_handles))
        after = response["views"]["after"]["inspection"]
        self.assertEqual(after["selected"]["id"], self.image_id)
        self.assertIsNone(after["selected"]["centroid"])
        self.assertGreater(after["source_point_count"], 0)
        self.assertIn("/object/source_points", self.scenes["after"].nodes)
        reference = response["region_references"]["before"]
        self.assertIsNotNone(reference["selected"])
        self.assertTrue(reference["frames"])
        response = self.update(3, select="before-only")
        self.assertIsNone(response["views"]["after"]["inspection"]["selected"])
        response = self.update(4, select=None)
        self.assertTrue(all(r is None for r in response["region_references"].values()))
        self.assertTrue(
            all(v["inspection"]["selected"] is None for v in response["views"].values())
        )

    def test_frame_change_updates_only_that_side_and_invalid_frame_is_atomic(self):
        response = self.update(1, select="both")
        side = response["selected"]["sides"]["before"]
        target = side["frames"][-1]["id"]
        after_source = response["views"]["after"]["inspection"]["source_frame"]
        changed = self.update(2, frames={"before": target})
        self.assertEqual(
            changed["views"]["before"]["inspection"]["source_frame"]["id"], target
        )
        self.assertEqual(
            changed["views"]["after"]["inspection"]["source_frame"], after_source
        )
        with self.assertRaises(ValueError):
            self.update(3, select="before-only", frames={"after": target})
        self.assertEqual(self.state.selected, "both")
        self.assertEqual(self.state.sequence, 2)

    def test_stale_actions_and_old_sessions_do_not_change_selection(self):
        self.update(2, select="both")
        self.assertTrue(self.update(1, select=None)["stale"])
        self.assertEqual(self.state.selected, "both")
        self.state.begin_session()
        with self.assertRaises(PermissionError):
            self.update(3, select=None)

    def test_missing_side_has_region_photo_without_object_or_classification_change(
        self,
    ):
        response = self.update(1, select="before-only")
        reference = response["region_references"]["after"]
        self.assertIsNone(response["views"]["after"]["inspection"]["selected"])
        self.assertIsNone(response["selected"]["sides"]["after"])
        self.assertEqual(response["selected"]["type"], "not_evaluated")
        image = self.state.region_frames.image(
            "after", reference["token"], reference["selected"]["id"]
        )
        self.assertGreater(Image.open(io.BytesIO(image)).width, 0)
        chosen = reference["frames"][-1]["id"]
        changed = self.update(2, frames={"after": chosen})
        self.assertEqual(
            changed["region_references"]["after"]["selected"]["id"], chosen
        )
        self.assertEqual(
            changed["views"]["before"]["inspection"]["source_frame"],
            response["views"]["before"]["inspection"]["source_frame"],
        )
        with self.assertRaises(ValueError):
            self.state.region_frames.image("before", reference["token"], chosen)

    def test_source_frame_change_updates_missing_side_reference(self):
        response = self.update(1, select="before-only")
        source = response["selected"]["sides"]["before"]
        chosen = source["frames"][-1]["id"]
        changed = self.update(2, frames={"before": chosen})
        source_frame = changed["views"]["before"]["inspection"]["source_frame"]
        self.assertEqual(
            changed["region_references"]["after"]["source_frame_id"],
            source_frame["frame_id"],
        )
        if len(source["frames"]) > 1:
            self.assertNotEqual(
                changed["region_references"]["after"]["token"],
                response["region_references"]["after"]["token"],
            )

    def test_rejected_alignment_keeps_manual_reference_frames(self):
        self.state.views["before"].inspection.inventory.comparison["alignment"][
            "quality"
        ]["status"] = "rejected"
        response = self.update(1, select="before-only")
        frames = response["region_references"]["after"]["frames"]
        self.assertEqual(len(frames), len(self.original.captures["after"]["frames"]))
        self.assertTrue(
            all(
                r["box_original"] is None and r["method"] == "manual_reference"
                for r in frames
            )
        )

    def test_image_region_uses_filtered_source_rows_in_the_display_coordinate_frame(
        self,
    ):
        helper = self.state.region_frames
        helper.source_arrays["after"] = {
            "observation_frame_index": np.array([0, 0, 0]),
            "original_pixel_xy": np.array([[1, 1], [5, 5], [20, 20]]),
            "cloud_point_index": np.array([0, 1, 2]),
        }
        self.state.views["after"].points["after"] = np.array(
            [[10, 2, 3], [20, 2, 3], [900, 2, 3]]
        )
        points = helper.region_points(
            "after",
            {"point_count": 0},
            {"frame_index": 0, "source_region_box": [0, 0, 10, 10]},
        )
        np.testing.assert_array_equal(points, [[10, 2, 3], [20, 2, 3]])

    def test_ambiguous_pair_stays_labelled_and_cannot_change_classification(self):
        self.state.items["both"]["status"] = "ambiguous"
        response = self.update(1, select="both")
        self.assertEqual(response["selected"]["status"], "ambiguous")
        self.assertEqual(response["selected"]["type"], "not_evaluated")
        self.assertTrue(
            all(v["inspection"]["selected"] for v in response["views"].values())
        )

    def test_http_two_scenes_evidence_and_controls(self):
        server = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            make_comparison_handler(self.state, {"before": 9001, "after": 9002}),
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=10)
        try:
            connection.request("GET", "/")
            response = connection.getresponse()
            page = response.read().decode()
            self.assertIn('title="Before 3D room"', page)
            self.assertIn('title="After 3D room"', page)
            self.assertIn("127.0.0.1:9001", page)
            self.assertIn("127.0.0.1:9002", page)
            connection.request(
                "POST",
                "/inspect",
                body=json.dumps(
                    {"session": self.session, "sequence": 1, "select": "both"}
                ),
            )
            response = connection.getresponse()
            selected = json.loads(response.read())
            self.assertEqual(response.status, 200)
            frame = selected["views"]["before"]["inspection"]["source_frame"]["id"]
            connection.request("GET", f"/evidence/before/{frame}/overlay")
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            self.assertTrue(response.read().startswith(b"\x89PNG"))
            connection.request("GET", f"/evidence/after/{frame}/overlay")
            response = connection.getresponse()
            self.assertEqual(response.status, 404)
            response.read()
            connection.request(
                "POST",
                "/inspect",
                body=json.dumps(
                    {"session": self.session, "sequence": 2, "select": "before-only"}
                ),
            )
            response = connection.getresponse()
            reference = json.loads(response.read())["region_references"]["after"]
            connection.request(
                "GET",
                f"/region/after/{reference['token']}/{reference['selected']['id']}",
            )
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            self.assertTrue(response.read().startswith(b"\x89PNG"))
        finally:
            connection.close()
            server.shutdown()
            server.server_close()
            thread.join()


class RegionProjectionTests(unittest.TestCase):
    def test_projection_uses_model_to_original_pixels_and_rejects_behind_camera(self):
        camera = {
            "world_to_camera": np.eye(4),
            "intrinsics": [[100, 0, 50], [0, 100, 50], [0, 0, 1]],
        }
        mapping = {"model_to_original": [[2, 0, 3], [0, 3, 5], [0, 0, 1]]}
        frame = {"width": 300, "height": 300}
        box, coverage = project_region([[0, 0, 2]], camera, mapping, frame)
        np.testing.assert_allclose(box, [95, 147, 111, 163])
        self.assertEqual(coverage, 1)
        self.assertEqual(
            project_region([[0, 0, -2]], camera, mapping, frame), (None, 0)
        )
        self.assertEqual(
            project_region([[100, 100, 1]], camera, mapping, frame), (None, 0)
        )


if __name__ == "__main__":
    unittest.main()
