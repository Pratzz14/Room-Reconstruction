import importlib.util
import shutil
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from reconstruction_v2 import core
from reconstruction_v2.archive import import_archive, pack
from reconstruction_v2.inference import cache_compatible, run, run_lock
from tests.fixtures import PROJECT, completed, predictions, prepared


def workspace_temporary_directory():
    root = (PROJECT / "outputs/.test_work").resolve()
    if not root.is_relative_to(PROJECT.resolve()):
        raise ValueError("Temporary test path escaped workspace")
    root.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="case-", dir=root)


class ReconstructionTests(unittest.TestCase):
    def test_known_camera_coordinates_and_pose_direction(self):
        k = np.array([[4, 0, 1.5], [0, 5, 0.5], [0, 0, 1]])
        pose = np.array([[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]])
        point = core.unproject(np.full((2, 3), 2.0), k, pose)[1, 2]
        np.testing.assert_allclose(point, [0.8, 2.25, 5])
        pixels, z = core.project(point[None], k, pose)
        np.testing.assert_allclose(pixels, [[2, 1]])
        np.testing.assert_allclose(z, [2])
        with self.assertRaises(ValueError):
            core.validate_pose(np.diag([1, 1, -1, 1]))

    def test_real_upstream_preprocessing_reference(self):
        # Execute the pinned upstream PIL preprocessing with CPU NumPy tensor stubs.
        # The actual PyTorch tensor equality check additionally runs in every GPU worker.
        torch = types.ModuleType("torch")
        torch.Tensor = np.ndarray
        torch.stack = np.stack
        torch.nn = types.SimpleNamespace(functional=types.SimpleNamespace(pad=lambda value, pads, mode, value_: None))
        def pad(tensor, pads, mode="constant", **kwargs):
            left, right, top, bottom = pads
            return np.pad(tensor, ((0, 0), (top, bottom), (left, right)), constant_values=kwargs["value"])
        torch.nn.functional.pad = pad
        tv = types.ModuleType("torchvision")
        tv.transforms = types.SimpleNamespace(ToTensor=lambda: lambda image: np.asarray(image).transpose(2, 0, 1).astype(np.float32) / 255)
        path = PROJECT / "reference/vggt_load_fn.py"
        self.assertEqual(core.digest(path), core.read_json(PROJECT / "configs/models.lock.json")["preprocessing_source_sha256"])
        with workspace_temporary_directory() as temporary, patch.dict("sys.modules", {"torch": torch, "torchvision": tv}):
            spec = importlib.util.spec_from_file_location("upstream_reference", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            for mode in ("crop", "pad"):
                paths, ours = [], []
                for index, dimensions in enumerate(((1920, 1080), (1080, 1920))):
                    v, u = np.indices(dimensions[::-1])
                    image = Image.fromarray(np.stack((u % 256, v % 256, (u + v) % 256), -1).astype(np.uint8))
                    file = Path(temporary) / f"{index}.png"; image.save(file); paths.append(str(file))
                    result, valid, transform = core.preprocess(image, mode=mode)
                    ours.append(np.asarray(result).transpose(2, 0, 1).astype(np.float32) / 255)
                    np.testing.assert_allclose(np.array(transform["model_to_original"]) @ np.array(transform["original_to_model"]), np.eye(3), atol=1e-12)
                    self.assertEqual(valid.shape, (result.height, result.width))
                reference = module.load_and_preprocess_images(paths, mode=mode)
                max_h = max(a.shape[1] for a in ours)
                padded = []
                for value in ours:
                    amount = max_h - value.shape[1]
                    padded.append(np.pad(value, ((0, 0), (amount // 2, amount - amount // 2), (0, 0)), constant_values=1))
                np.testing.assert_array_equal(reference, np.stack(padded))

    def test_voxel_membership_highest_confidence_and_stable_ties(self):
        points = np.array([[0.02, 0, 0], [0.03, 0, 0], [0.04, 0, 0], [1.2, 0, 0]])
        representatives, membership = core.reduce_voxels(points, np.array([1, 4, 4, 2]), 1)
        np.testing.assert_array_equal(representatives, [1, 3])
        np.testing.assert_array_equal(membership, [0, 0, 0, 1])

    def test_run_lock_prevents_concurrent_manifest_writers(self):
        with workspace_temporary_directory() as temporary:
            root = Path(temporary)
            with run_lock(root):
                with self.assertRaises(ValueError):
                    pack(root, root.parent / "active_run.zip")
                with self.assertRaises(RuntimeError):
                    with run_lock(root):
                        pass
            self.assertFalse((root / ".run.lock").exists())

    def test_export_invalid_depth_provenance_and_exact_rgb(self):
        with workspace_temporary_directory() as temporary:
            root = completed(Path(temporary) / "run")
            report = core.verify_capture(root, "before")
            self.assertTrue(report["all_points_have_observations"])
            self.assertTrue(report["point_color_source_verified"])
            with np.load(root / "before/valid.npz") as valid:
                self.assertFalse(valid["valid"][0, 0, :4].any())
            with np.load(root / "before/observations.npz") as observations:
                self.assertGreater(np.max(np.diff(observations["point_observation_offsets"])), 1)
            self.assertTrue(cache_compatible(root, "before"))
            self.assertTrue(cache_compatible(root, "after"))
            # Offline resume never requires an installed model or checkpoint.
            result = run(root, Path(temporary) / "missing-source", Path(temporary) / "missing-weights", resume=True)
            self.assertEqual(result["status"], "completed")

    def test_corrupt_capture_does_not_invalidate_other_capture(self):
        with workspace_temporary_directory() as temporary:
            root = completed(Path(temporary) / "run")
            (root / "after/cloud.ply").write_bytes(b"corrupt")
            self.assertFalse(cache_compatible(root, "after"))
            self.assertTrue(cache_compatible(root, "before"))
            with self.assertRaises(ValueError):
                core.verify_capture(root, "after")

    def test_checkpoint_identity_and_input_changes_cannot_reuse_geometry(self):
        with workspace_temporary_directory() as temporary:
            root = completed(Path(temporary) / "run")
            lock = core.read_json(root / "models.lock.json")
            lock["checkpoint_sha256"] = "0" * 64
            core.write_json(root / "models.lock.json", lock)
            self.assertFalse(cache_compatible(root, "before"))
            with self.assertRaises(ValueError):
                run(root, Path("none"), Path("none"), resume=True)

    def test_no_observations_does_not_publish_success(self):
        with workspace_temporary_directory() as temporary:
            root = prepared(Path(temporary) / "run")
            values = list(predictions(root, "before")); values[0][:] = np.nan
            with self.assertRaises(ValueError):
                core.export_predictions(root, "before", *values, {"backend": "synthetic"})
            self.assertEqual(core.read_json(root / "run_manifest.json")["captures"]["before"]["status"], "prepared")
            self.assertFalse((root / "before/cloud.ply").exists())

    def test_semantic_provenance_corruption_is_detected_without_hash_checks(self):
        with workspace_temporary_directory() as temporary:
            root = completed(Path(temporary) / "run")
            path = root / "before/observations.npz"
            with np.load(path) as data:
                values = {name: data[name] for name in data.files}
            values["original_pixel_xy"][0] += 10
            np.savez_compressed(path, **values)
            with self.assertRaises(ValueError):
                core.verify_capture(root, "before", check_hashes=False)

    def test_safe_portable_roundtrip_and_traversal_rejection(self):
        with workspace_temporary_directory() as temporary:
            root = completed(Path(temporary) / "run")
            archive = Path(temporary) / "results.zip"
            pack(root, archive)
            moved = Path(temporary) / "relocated"
            import_archive(archive, moved)
            self.assertEqual(core.verify_capture(moved, "after")["internal_consistency"], "passed")
            malicious = Path(temporary) / "bad.zip"
            with zipfile.ZipFile(malicious, "w") as bundle:
                bundle.writestr("../escape.txt", "bad")
            with self.assertRaises(ValueError):
                import_archive(malicious, Path(temporary) / "bad_import")


if __name__ == "__main__":
    unittest.main()
