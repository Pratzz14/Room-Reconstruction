"""Analytic object membership/association/transport tests; never real model evidence."""

import copy
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from reconstruction import core
from reconstruction.object_archive import import_archive, pack
from reconstruction.object_contracts import ObjectInventory, atomic_move
from reconstruction.object_geometry import (
    CaptureGeometry,
    geometry_quality,
    voxel_geometry,
)
from reconstruction.object_pipeline import run
from tests.fixtures import PROJECT, completed


class AnalyticModels:
    def __init__(self):
        self.runtime = {"backend": "SYNTHETIC_MASK_FIXTURE", "packages": {}}
        self.calls = []

    def activate(self, kind):
        self.calls.append(kind)

    def detect(self, image):
        return [
            {
                "box": [0, 0, image.width / 2, image.height],
                "score": 0.9,
                "label": "chair",
                "raw_label": "chair",
            },
            {
                "box": [image.width / 2, 0, image.width, image.height],
                "score": 0.8,
                "label": "cabinet",
                "raw_label": "cabinet",
            },
        ]

    def segment(self, image, box):
        mask = np.zeros((image.height, image.width), bool)
        mask[:, int(box[0]) : int(box[2])] = True
        return mask, 0.9

    def embed(self, image):
        return np.array([1.0, 0.0, 0.0], np.float32)

    def close(self):
        pass


class ObjectWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scratch = PROJECT / "outputs/.test_work"
        scratch.mkdir(parents=True, exist_ok=True)
        cls.scratch = tempfile.TemporaryDirectory(prefix="objects-", dir=scratch)
        cls.base = Path(cls.scratch.name)
        cls.reconstruction = completed(cls.base / "reconstruction")
        cls.config = core.read_json(PROJECT / "configs/objects.json")
        cls.config["device"] = "cpu"
        cls.lock = {
            "status": "synthetic",
            "models": {
                kind: {"fixture": True}
                for kind in ("detector", "segmenter", "appearance")
            },
        }

    @classmethod
    def tearDownClass(cls):
        cls.scratch.cleanup()

    def process(self, name, config=None, **kwargs):
        models = AnalyticModels()
        output = self.base / name
        result = run(
            self.reconstruction,
            output,
            config or self.config,
            self.lock,
            models=models,
            **kwargs,
        )
        self.assertNotEqual(result["status"], "failed", result)
        return output, models, result

    def test_full_inventory_geometry_provenance_and_resume(self):
        before = {
            p: core.digest(p) for p in self.reconstruction.rglob("*") if p.is_file()
        }
        output, models, _ = self.process("full")
        inventory = ObjectInventory(self.reconstruction, output, require_complete=True)
        for name in ("before", "after"):
            listing = inventory.listing(name)
            self.assertEqual(len(listing["objects"]), 2)
            self.assertEqual(len(listing["unresolved"]), 0)
            self.assertTrue(
                all(o["supporting_frames"] == 2 for o in listing["objects"])
            )
            for obj in listing["objects"]:
                _, geometry = inventory.object(name, obj["id"])
                np.testing.assert_allclose(geometry["points"].mean(0), obj["centroid"])
                self.assertTrue(obj["id"].startswith(name + "_obj_"))
        identities = [o["id"] for o in inventory.listing("before")["objects"]]
        calls = list(models.calls)
        run(
            self.reconstruction,
            output,
            self.config,
            self.lock,
            models=models,
            resume=True,
        )
        self.assertEqual(models.calls, calls)
        self.assertEqual(
            identities,
            [
                o["id"]
                for o in ObjectInventory(self.reconstruction, output).listing("before")[
                    "objects"
                ]
            ],
        )
        self.assertEqual(before, {p: core.digest(p) for p in before})

    def test_association_settings_reuse_model_caches(self):
        output, models, _ = self.process("cache")
        updated = copy.deepcopy(self.config)
        updated["association"]["appearance_similarity"] = 0.85
        models.calls.clear()
        result = run(
            self.reconstruction, output, updated, self.lock, resume=True, models=models
        )
        self.assertNotEqual(result["status"], "failed", result)
        self.assertEqual(models.calls, [])
        ObjectInventory(self.reconstruction, output, require_complete=True)

    def test_sparse_masks_are_preserved_as_unresolved(self):
        class Sparse(AnalyticModels):
            def segment(self, image, box):
                return np.zeros((image.height, image.width), bool), 0.5

        output = self.base / "sparse"
        result = run(
            self.reconstruction, output, self.config, self.lock, models=Sparse()
        )
        self.assertEqual(result["status"], "completed_with_warnings")
        inventory = ObjectInventory(self.reconstruction, output)
        self.assertEqual(len(inventory.listing("before")["objects"]), 0)
        self.assertGreater(len(inventory.listing("before")["unresolved"]), 0)
        for count in (0, 1, 2):
            self.assertIn(
                "insufficient_depth_support",
                geometry_quality(
                    np.zeros((count, 3)), 100, self.config["geometry"], 0.01
                ),
            )

    def test_corruption_rejected_and_repaired_without_model_rerun(self):
        output, models, _ = self.process("corruption")
        (output / "before/geometry.npz").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            ObjectInventory(self.reconstruction, output)
        models.calls.clear()
        result = run(
            self.reconstruction,
            output,
            self.config,
            self.lock,
            resume=True,
            models=models,
        )
        self.assertNotEqual(result["status"], "failed", result)
        self.assertEqual(models.calls, [])
        ObjectInventory(self.reconstruction, output)

    def test_archive_roundtrip_and_unsafe_member(self):
        output, _, _ = self.process("transport", frame_limit=1)
        archive = self.base / "objects.zip"
        pack(self.reconstruction, output, archive)
        imported = self.base / "imported"
        import_archive(archive, self.reconstruction, imported)
        inventory = ObjectInventory(self.reconstruction, imported)
        self.assertEqual(len(inventory.listing("before")["objects"]), 2)
        with zipfile.ZipFile(self.base / "unsafe.zip", "w") as bundle:
            bundle.writestr("../escape.txt", "bad")
        with self.assertRaises(ValueError):
            import_archive(
                self.base / "unsafe.zip", self.reconstruction, self.base / "refused"
            )
        self.assertFalse((self.base / "refused").exists())

    def test_mask_original_pixels_and_xyz_use_observations(self):
        geometry = CaptureGeometry(self.reconstruction, "before")
        mask = np.zeros((24, 32), bool)
        mask[:, :8] = True
        selected = geometry.mask_rows(0, mask)
        xy = np.floor(
            geometry.observations["original_pixel_xy"][selected] + 0.5
        ).astype(int)
        self.assertTrue(np.all(xy[:, 0] < 8))
        self.assertGreater(len(selected), 0)
        points = geometry.points(selected)
        camera = geometry.cameras[0]
        projected, z = core.project(
            points, camera["intrinsics"], camera["camera_to_world"]
        )
        pi = geometry.observations["model_pixel_index"][selected]
        np.testing.assert_allclose(
            projected,
            np.c_[pi % camera["model_width"], pi // camera["model_width"]],
            atol=1e-4,
        )
        np.testing.assert_allclose(z, geometry.observations["depth"][selected])
        obj = voxel_geometry(geometry, selected, geometry.voxel * 2)
        self.assertTrue(np.isin(obj["representative_observation"], selected).all())
        with self.assertRaises(ValueError):
            geometry.mask_rows(0, np.ones((3, 3)))

    def test_failure_of_one_capture_preserves_other(self):
        class Failure(AnalyticModels):
            def __init__(self):
                super().__init__()
                self.count = 0

            def activate(self, kind):
                if kind == "detector":
                    self.count += 1
                    if self.count == 2:
                        raise RuntimeError("deliberate After failure")
                super().activate(kind)

        output = self.base / "one_failure"
        result = run(
            self.reconstruction, output, self.config, self.lock, models=Failure()
        )
        self.assertEqual(result["captures"]["before"]["status"], "completed")
        self.assertEqual(result["captures"]["after"]["status"], "failed")
        inventory = ObjectInventory(self.reconstruction, output)
        self.assertIn("before", inventory.captures)
        self.assertNotIn("after", inventory.captures)


class PublicationTests(unittest.TestCase):
    def test_transient_windows_sharing_error_is_retried(self):
        busy = PermissionError("sharing violation")
        busy.winerror = 32
        with (
            patch.object(
                Path, "replace", side_effect=[busy, Path("destination")]
            ) as move,
            patch("reconstruction.core.time.sleep"),
        ):
            atomic_move(Path("source"), Path("destination"))
            self.assertEqual(move.call_count, 2)

    def test_permanent_file_error_is_not_suppressed(self):
        with (
            patch.object(Path, "replace", side_effect=OSError("disk error")) as move,
            patch("reconstruction.core.time.sleep"),
        ):
            with self.assertRaises(OSError):
                atomic_move(Path("source"), Path("destination"))
            self.assertEqual(move.call_count, 1)


if __name__ == "__main__":
    unittest.main()
