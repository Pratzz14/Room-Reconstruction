"""Analytic checks for simple presence changes and supporting-frame mean proxies."""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from reconstruction.change_detection import match_objects
from reconstruction.change_stage import build_changes, frame_cloud_means
from tests.fixtures import PROJECT


def obj(identifier, center=None, frames=("f0",), **values):
    return {
        "id": identifier,
        "label": "pressure cooker",
        "centroid": center,
        "frame_ids": list(frames),
        "observation_ids": [],
        "point_count": 10 if center is not None else 0,
        "geometry_status": "supported" if center is not None else "missing",
        **values,
    }


def identities(before, after, status="matched"):
    return [
        {
            "shared_id": "shared",
            "before_id": before,
            "after_id": after,
            "status": status,
        }
    ]


class ChangeStageTests(unittest.TestCase):
    def setUp(self):
        scratch = PROJECT / "outputs/.test_work"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="changes-", dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inventory = SimpleNamespace(
            reconstruction=self.root,
            captures={"before": {"objects": []}, "after": {"objects": []}},
        )
        self.alignment = {
            "quality": {"status": "accepted"},
            "transform": {
                "scale": 1,
                "rotation": np.eye(3),
                "translation": np.zeros(3),
            },
        }

    def cloud(self, capture, points, membership=(0, 0, 1, 2), frames=(0, 1, 1, 1)):
        directory = self.root / capture
        directory.mkdir(exist_ok=True)
        np.savez(directory / "cloud.npz", points=np.asarray(points, float))
        np.savez(
            directory / "observations.npz",
            frame_ids=np.array(["f0", "f1"]),
            cloud_point_index=np.asarray(membership),
            observation_frame_index=np.asarray(frames),
        )

    def classify(self, before, after, status="matched", extent=10):
        self.inventory.captures["before"]["objects"] = before
        self.inventory.captures["after"]["objects"] = after
        bs, ats = before[0]["id"] if before else None, after[0]["id"] if after else None
        matching = {
            "matches": [{"before_id": bs, "after_id": ats, "status": status}]
            if bs and ats
            else []
        }
        return build_changes(
            self.inventory,
            identities(bs, ats, status),
            matching,
            self.alignment,
            extent,
        )

    def test_frame_means_use_unique_cloud_union_and_all_observations(self):
        self.cloud("before", [[0, 0, 0], [3, 0, 0], [9, 0, 0]])
        item = obj("b", frames=("f0", "f1", "f1"))
        mean = frame_cloud_means(self.inventory, "before", [item])["b"]
        self.assertEqual(mean["mean"], [4, 0, 0])
        self.assertEqual(mean["point_count"], 3)
        self.assertEqual(mean["frame_ids"], ["f0", "f1"])
        # f1 includes point 0 even though its representative might be from f0.
        item["frame_ids"] = ["f1"]
        self.assertEqual(
            frame_cloud_means(self.inventory, "before", [item])["b"]["mean"], [4, 0, 0]
        )

    def test_geometry_transform_applied_once_and_threshold_equality_unchanged(self):
        self.alignment["transform"] = {
            "scale": 2,
            "rotation": [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
            "translation": [4, 3, 0],
        }
        before, after = [obj("b", [2, 3, 0])], [obj("a", [0, 1, 0])]
        sources = copy.deepcopy((before, after))
        event = self.classify(before, after)["events"][0]
        self.assertEqual(event["type"], "unchanged")
        self.assertEqual(event["object_displacement"], 0)
        self.assertEqual(event["trust"], "reliable")
        self.assertEqual((before, after), sources)
        after[0]["centroid"] = [0, 0, 0]
        event = self.classify(before, after)["events"][0]
        self.assertEqual(event["object_displacement"], event["threshold"])
        self.assertEqual(event["type"], "unchanged")
        after[0]["centroid"] = [0, -0.1, 0]
        self.assertEqual(self.classify(before, after)["events"][0]["type"], "moved")

    def test_mixed_pair_uses_frame_means_on_both_sides_without_mutating_centroid(self):
        self.cloud("before", [[0, 0, 0], [3, 0, 0], [9, 0, 0]])
        self.cloud("after", [[1, 0, 0], [4, 0, 0], [10, 0, 0]])
        self.alignment["transform"]["translation"] = [5, 0, 0]
        before = [obj("b", [1000, 0, 0], frames=("f0", "f1"))]
        after = [obj("a", frames=("f0", "f1"))]
        event = self.classify(before, after, status="proposed_visual")["events"][0]
        self.assertEqual(event["type"], "moved")
        self.assertEqual(event["basis"], "frame_cloud_mean")
        self.assertEqual(event["trust"], "estimated")
        self.assertEqual(event["before_location"], [4, 0, 0])
        self.assertEqual(event["after_location"], [10, 0, 0])
        self.assertEqual(event["frame_cloud_distance"], 6)
        self.assertIsNone(event["object_displacement"])
        self.assertIsNone(after[0]["centroid"])

    def test_image_only_same_proxy_is_unchanged(self):
        for capture in ("before", "after"):
            self.cloud(capture, [[0, 0, 0], [3, 0, 0], [9, 0, 0]])
        event = self.classify([obj("b")], [obj("a")], "proposed_visual")["events"][0]
        self.assertEqual(event["type"], "unchanged")
        self.assertEqual(event["frame_cloud_distance"], 0)

    def test_observation_and_context_frames_join_the_mean(self):
        self.cloud("before", [[0, 0, 0], [3, 0, 0], [9, 0, 0]])
        self.inventory.captures["before"]["observations"] = [
            {"id": "ob", "frame_id": "f0"}
        ]
        self.inventory.context_record = lambda *_: {"frame_id": "f1"}
        item = obj("b", frames=(), observation_ids=["ob"], context_evidence=["ctx"])
        mean = frame_cloud_means(self.inventory, "before", [item])["b"]
        self.assertEqual(mean["frame_ids"], ["f0", "f1"])
        self.assertEqual(mean["mean"], [4, 0, 0])

    def test_presence_only_added_removed_without_cloud_or_visibility(self):
        for before, after, expected in (
            ([obj("b")], [], "removed"),
            ([], [obj("a")], "added"),
        ):
            event = self.classify(before, after, "unmatched")["events"][0]
            self.assertEqual(event["type"], expected)
            self.assertEqual(event["basis"], "presence_only")
            self.assertEqual(event["trust"], "estimated")

    def test_rejected_alignment_blocks_all_kinds(self):
        self.alignment["quality"]["status"] = "rejected"
        for before, after in (
            ([obj("b", [0, 0, 0])], [obj("a", [9, 0, 0])]),
            ([], [obj("a")]),
            ([obj("b")], []),
        ):
            result = self.classify(before, after)
            self.assertIsNone(result["coordinate_frame"])
            self.assertEqual(result["events"][0]["type"], "uncertain")
            self.assertIsNone(result["events"][0]["object_displacement"])

    def test_empty_nonfinite_unknown_and_missing_frame_support(self):
        cases = [
            ([], (), (), ("f0",), "empty_frame_cloud"),
            ([[float("nan"), 0, 0]], (0,), (0,), ("f0",), "nonfinite_frame_cloud"),
            ([[0, 0, 0]], (0,), (0,), ("unknown",), "unknown_supporting_frame"),
            ([[0, 0, 0]], (0,), (0,), (), "missing_supporting_frames"),
        ]
        self.cloud("after", [[0, 0, 0]], (0,), (0,))
        for points, membership, frames, supporting, reason in cases:
            with self.subTest(reason=reason):
                self.cloud("before", points, membership, frames)
                event = self.classify([obj("b", frames=supporting)], [obj("a")])[
                    "events"
                ][0]
                self.assertEqual(event["type"], "uncertain")
                self.assertIn("before_" + reason, event["uncertainties"])

    def test_unmatched_competitor_is_uncertain_not_removed(self):
        before = [obj(x, [0, 0, 0], descriptor=[1, 0]) for x in ("b1", "b2")]
        after = [obj("a1", [9, 0, 0], descriptor=[1, 0])]
        matching = match_objects(before, after)
        paired = matching["matches"][0]
        items = identities(paired["before_id"], paired["after_id"], paired["status"])
        items.append(
            {
                "shared_id": "other",
                "before_id": matching["unmatched_before"][0],
                "after_id": None,
                "status": "unmatched",
            }
        )
        self.inventory.captures["before"]["objects"] = before
        self.inventory.captures["after"]["objects"] = after
        events = build_changes(self.inventory, items, matching, self.alignment, 10)[
            "events"
        ]
        self.assertEqual([e["type"] for e in events], ["uncertain", "uncertain"])
        self.assertTrue(all("identity_ambiguous" in e["uncertainties"] for e in events))

    def test_invalid_fraction_and_extent(self):
        for extent, fraction in ((0, 0.2), (float("nan"), 0.2), (10, 0), (10, 2)):
            with self.assertRaises(ValueError):
                build_changes(
                    self.inventory,
                    [],
                    {"matches": []},
                    self.alignment,
                    extent,
                    fraction,
                )


if __name__ == "__main__":
    unittest.main()
