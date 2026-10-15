"""Analytic alignment/matching checks; independent of the real-video acceptance run."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from reconstruction_v2.alignment import apply_similarity, inverse_similarity
from reconstruction_v2.change_detection import classify_changes, match_objects
from reconstruction_v2.comparison import (
    ComparisonInventory,
    run,
    transformed_cameras,
    verify,
)
from reconstruction_v2.core import digest, read_json, write_json
from reconstruction_v2.object_contracts import ObjectInventory
from reconstruction_v2.object_pipeline import run as detect
from reconstruction_v2.static_alignment import extract_correspondences, fit_static
from tests.fixtures import PROJECT, completed
from tests.test_alignment_change import object_record
from tests.test_objects import AnalyticModels


def analytic_correspondences():
    rng = np.random.default_rng(11)
    target = rng.uniform(-0.8, 0.8, (360, 3))
    target[:180, 2] = 0
    target[180:, 0] = 0.8
    angle = 0.45
    transform = {
        "scale": 1.7,
        "rotation": np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        ),
        "translation": np.array([0.3, -0.2, 0.7]),
    }
    source = apply_similarity(target, inverse_similarity(transform))
    pairs = np.c_[np.arange(360) % 12, np.arange(360) % 9]
    held = (pairs[:, 0] % 5 == 0) | (pairs[:, 1] % 5 == 0)
    c = {
        "source": source,
        "target": target,
        "frame_pairs": pairs,
        "held_out": held,
        "region_ids": np.zeros(360, int),
    }
    return c, transform


class StaticAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.cfg = read_json(PROJECT / "configs/comparison.json")
        self.cfg["ransac_iterations"] = 150

    def test_outliers_scale_and_whole_frame_holdouts(self):
        c, expected = analytic_correspondences()
        c["source"][::9] += 3
        result, q = fit_static(c, ["room_shell"], 2.5, self.cfg)
        self.assertEqual(q["status"], "accepted", q)
        self.assertGreaterEqual(q["held_out"]["shell_normal_directions"], 2)
        self.assertAlmostEqual(result["scale"], expected["scale"], places=6)
        np.testing.assert_allclose(result["rotation"], expected["rotation"], atol=1e-6)

    def test_wrong_withheld_geometry_rejects_good_training_fit(self):
        c, _ = analytic_correspondences()
        c["source"][c["held_out"]] += [2, 0, 0]
        _, q = fit_static(c, ["room_shell"], 2.5, self.cfg)
        self.assertEqual(q["status"], "rejected")
        self.assertFalse(q["location_comparison_allowed"])

    def test_one_wall_cannot_pass_as_distributed_room_support(self):
        c, _ = analytic_correspondences()
        c["target"][:, 2] = 0
        c["source"] = c["target"].copy()
        _, q = fit_static(c, ["room_shell"], 2.5, self.cfg)
        self.assertEqual(q["status"], "rejected")

    def test_missing_matches_preserve_failed_stage_diagnostics(self):
        c = {
            "source": np.empty((0, 3)),
            "target": np.empty((0, 3)),
            "held_out": np.empty(0, bool),
        }
        transform, quality = fit_static(c, ["room_shell"], 2.5, self.cfg)
        self.assertIsNone(transform)
        self.assertEqual(quality["status"], "rejected")

    def test_named_regions_never_match_across_symmetric_instances(self):
        f = {
            "points": np.array([[0.0, 0, 1], [1, 0, 1]]),
            "descriptors": np.eye(2, dtype=np.float32),
            "pixels": np.zeros((2, 2), int),
            "regions": np.array([1, 1]),
        }
        other = {**f, "regions": np.array([2, 2])}
        inventory = SimpleNamespace(
            captures={"before": {"objects": []}, "after": {"objects": []}}
        )
        with patch(
            "reconstruction_v2.static_alignment.frame_features",
            side_effect=[[f], [other]],
        ):
            c, _ = extract_correspondences(inventory, self.cfg)
        self.assertEqual(len(c["source"]), 0)


class IdentityMatchingTests(unittest.TestCase):
    def test_equal_capture_counts_have_real_unmatched_choices(self):
        b = [
            object_record("b1", "chair", [0, 0, 0], [1, 0]),
            object_record("b2", "laptop", [0, 0, 0], [0, 1]),
        ]
        a = [
            object_record("a1", "chair", [5, 0, 0], [1, 0]),
            object_record("a2", "cooker", [0, 0, 0], [0, 1]),
        ]
        m = match_objects(b, a)
        self.assertEqual(
            [(x["before_id"], x["after_id"]) for x in m["matches"]], [("b1", "a1")]
        )
        self.assertEqual(m["unmatched_before"], ["b2"])
        self.assertEqual(m["unmatched_after"], ["a2"])

    def test_two_before_instances_competing_for_one_after_stay_ambiguous(self):
        b = [object_record(x, "chair", [0, 0, 0], [1, 0]) for x in ("b1", "b2")]
        a = [object_record("a1", "chair", [0, 0, 0], [1, 0])]
        m = match_objects(b, a)
        self.assertEqual(m["matches"][0]["status"], "ambiguous")
        self.assertTrue(m["matches"][0]["alternative_before_ids"])

    def test_image_only_bottles_keep_proposed_identities_and_null_geometry(self):
        b = [
            {
                "id": "blue-b",
                "label": "drinking bottle",
                "centroid": None,
                "bounds": None,
            }
        ]
        a = [
            {
                "id": "blue-a",
                "label": "drinking bottle",
                "centroid": None,
                "bounds": None,
            }
        ]
        m = match_objects(
            b,
            a,
            visual_links=[
                {"before_id": "blue-b", "after_id": "blue-a", "status": "proposed"}
            ],
        )
        self.assertEqual(m["matches"][0]["status"], "proposed_visual")
        self.assertIsNone(m["matches"][0]["features"]["centroid_distance"])

    def test_rejected_alignment_never_contributes_positions_or_changes(self):
        b = [object_record("b", "chair", [0, 0, 0], [1, 0])]
        a = [object_record("a", "chair", [5, 0, 0], [1, 0])]
        transform = {"scale": 2, "rotation": np.eye(3), "translation": np.zeros(3)}
        m = match_objects(b, a, after_transform=transform, alignment_accepted=False)
        self.assertIsNone(m["matches"][0]["features"]["position_cost"])
        self.assertIsNone(m["matches"][0]["features"]["shape_cost"])
        events = classify_changes(
            b, a, m, after_transform=transform, tolerance=1, alignment_status="rejected"
        )["events"]
        self.assertEqual(events[0]["type"], "uncertain")
        self.assertIsNone(events[0]["displacement"])

    def test_visible_empty_sets_do_not_prove_removal(self):
        b = [object_record("b", "chair", [0, 0, 0], [1, 0])]
        m = match_objects(b, [])
        r = classify_changes(
            b,
            [],
            m,
            tolerance=1,
            alignment_status="accepted",
            before_visible={"b"},
            after_visible=set(),
        )
        self.assertEqual(r["events"][0]["type"], "uncertain")

    def test_shape_dimensions_use_alignment_scale(self):
        b = [object_record("b", "chair", [0, 0, 0], [1, 0], extent=(2, 4, 6))]
        a = [object_record("a", "chair", [0, 0, 0], [1, 0], extent=(1, 2, 3))]
        m = match_objects(
            b,
            a,
            after_transform={
                "scale": 2,
                "rotation": np.eye(3),
                "translation": np.zeros(3),
            },
            alignment_accepted=True,
        )
        self.assertAlmostEqual(m["matches"][0]["features"]["shape_cost"], 0)


class ComparisonWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(
            prefix="comparison-", dir=PROJECT / "outputs/.test_work"
        )
        cls.base = Path(cls.temp.name)
        cls.reconstruction = completed(cls.base / "reconstruction")
        cls.objects = cls.base / "objects"
        cfg = read_json(PROJECT / "configs/objects.json")
        cfg["device"] = "cpu"
        lock = {
            "status": "synthetic",
            "models": {
                k: {"fixture": True} for k in ("detector", "segmenter", "appearance")
            },
        }
        detect(cls.reconstruction, cls.objects, cfg, lock, models=AnalyticModels())

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_saved_pipeline_exports_provenance_and_reuses_correspondences(self):
        cfg = read_json(PROJECT / "configs/comparison.json")
        cfg["ransac_iterations"] = 100
        c, _ = analytic_correspondences()
        output = self.base / "comparison"
        with patch(
            "reconstruction_v2.comparison.extract_correspondences",
            return_value=(c, ["room_shell"]),
        ) as extract:
            report = run(self.reconstruction, self.objects, None, output, cfg)
            self.assertEqual(extract.call_count, 1)
        self.assertTrue(report["synthetic"])
        self.assertEqual(report["version"], "comparison-v2")
        changes = read_json(output / "changes.json")
        self.assertEqual(sum(changes["counts"].values()), len(changes["events"]))
        inventory = ObjectInventory(self.reconstruction, self.objects)
        verify(inventory, output)
        reader = ComparisonInventory(inventory, output)
        self.assertEqual(
            reader.captures["after"]["objects"][0]["coordinate_frame"], "before_world"
        )
        self.assertIn("change", reader.captures["after"]["objects"][0])
        # Rehashing a changed measurement still cannot pass semantic verification.
        changed = read_json(output / "changes.json")
        changed["events"][0]["threshold"] += 1
        write_json(output / "changes.json", changed)
        report["artifacts"]["changes.json"] = digest(output / "changes.json")
        write_json(output / "comparison.json", report)
        with self.assertRaisesRegex(ValueError, "Changes disagree"):
            verify(inventory, output)
        write_json(output / "changes.json", changes)
        report["artifacts"]["changes.json"] = digest(output / "changes.json")
        write_json(output / "comparison.json", report)
        with patch(
            "reconstruction_v2.comparison.extract_correspondences",
            side_effect=AssertionError("cache should be reused"),
        ):
            repeated = run(self.reconstruction, self.objects, None, output, cfg)
        self.assertEqual(read_json(output / "changes.json"), changes)
        self.assertEqual(repeated["change_counts"], report["change_counts"])
        # Prior comparison versions remain readable, but run cannot overwrite them.
        legacy = read_json(output / "comparison.json")
        legacy["version"] = "comparison-v1"
        legacy["change_status"] = "not_evaluated"
        legacy["artifacts"].pop("changes.json")
        write_json(output / "comparison.json", legacy)
        self.assertEqual(verify(inventory, output)["version"], "comparison-v1")
        original_digest = digest(output / "comparison.json")
        with self.assertRaisesRegex(ValueError, "use a new output directory"):
            run(self.reconstruction, self.objects, None, output, cfg)
        self.assertEqual(digest(output / "comparison.json"), original_digest)
        write_json(output / "comparison.json", repeated)
        source = read_json(output / "after/objects.json")
        source["objects"][0]["centroid"][0] += 1
        write_json(output / "after/objects.json", source)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            verify(inventory, output)

    def test_scaled_camera_pose_and_depth_preserve_projection(self):
        camera = read_json(self.reconstruction / "after/cameras.json")[0]
        _, transform = analytic_correspondences()
        result = transformed_cameras([camera], transform)[0]
        pose = np.asarray(camera["camera_to_world"])
        local = np.array([0.2, -0.1, 3.0])
        world = pose[:3, :3] @ local + pose[:3, 3]
        aligned = apply_similarity(world, transform)
        newlocal = np.asarray(result["world_to_camera"]) @ np.r_[aligned, 1]
        np.testing.assert_allclose(newlocal[:3], local * transform["scale"], atol=1e-7)
        np.testing.assert_allclose(newlocal[:2] / newlocal[2], local[:2] / local[2])


if __name__ == "__main__":
    unittest.main()
