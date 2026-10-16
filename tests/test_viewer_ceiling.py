import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from reconstruction import core, viewer
from tests.fixtures import PROJECT, completed


class CeilingCropTests(unittest.TestCase):
    def test_signed_axes_keep_original_ids_and_leave_points_unchanged(self):
        coordinates = np.arange(-5, 6, dtype=np.float32)
        points = np.column_stack((coordinates, coordinates, coordinates))
        original = points.copy()
        for axis in "XYZ":
            np.testing.assert_array_equal(viewer.ceiling_crop_indices(points, "-" + axis, 50), np.arange(5, 11))
            np.testing.assert_array_equal(viewer.ceiling_crop_indices(points, "+" + axis, 50), np.arange(6))
        np.testing.assert_array_equal(points, original)
        np.testing.assert_array_equal(viewer.ceiling_crop_indices(points, percent=0), np.arange(11))

    def test_empty_constant_cloud_and_invalid_settings(self):
        self.assertEqual(len(viewer.ceiling_crop_indices(np.empty((0, 3)))), 0)
        np.testing.assert_array_equal(viewer.ceiling_crop_indices(np.ones((3, 3))), np.arange(3))
        for direction, percent in (("up", 15), ("-Y", -1), ("+Y", 101)):
            with self.assertRaises(ValueError):
                viewer.ceiling_crop_indices(np.ones((3, 3)), direction, percent)

    def test_ceiling_cut_uses_robust_extent(self):
        points = np.zeros((101, 3), dtype=np.float32)
        points[:, 1] = np.r_[np.arange(100), 1000000]
        kept = viewer.ceiling_crop_indices(points, "+Y", 50)
        self.assertGreater(len(kept), 40)
        self.assertLess(len(kept), 60)
        self.assertNotIn(100, kept)


class Handle:
    def __init__(self, value=None, **kwargs):
        self.value, self.callbacks = value, []
        self.__dict__.update(kwargs)

    def on_update(self, callback):
        self.callbacks.append(callback)
        return callback

    on_click = on_update

    def change(self, value):
        self.value = value
        for callback in self.callbacks:
            callback(types.SimpleNamespace(client=None))

    def remove(self):
        self.removed = True


class Gui:
    def __init__(self):
        self.controls = {}

    def control(self, label, initial_value=None, **kwargs):
        handle = Handle(initial_value, **kwargs)
        self.controls[label] = handle
        return handle

    add_dropdown = add_number = add_slider = add_checkbox = add_button = control

    def add_markdown(self, content):
        return Handle(content=content)

    def add_image(self, image, label):
        return self.control(label, image=image)


class Scene:
    def add_point_cloud(self, name, points, colors, **kwargs):
        self.cloud = Handle(points=points, colors=colors, **kwargs)
        return self.cloud

    def add_line_segments(self, *args, **kwargs):
        return Handle()


class ViewerCeilingIntegrationTests(unittest.TestCase):
    def test_controls_restore_capture_settings_and_preserve_saved_evidence(self):
        import viser

        scratch = PROJECT / "outputs/.test_work"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="ceiling-viewer-", dir=scratch) as tmp:
            root = completed(Path(tmp) / "run")
            hashes = {p: core.digest(p) for p in root.rglob("*") if p.is_file()}
            server = types.SimpleNamespace(gui=Gui(), scene=Scene(), get_clients=lambda: {},
                                           on_client_connect=lambda callback: callback)
            with patch.object(viser, "ViserServer", return_value=server), \
                    patch.object(viewer.time, "sleep", side_effect=InterruptedError("end test event loop")):
                with self.assertRaises(InterruptedError):
                    viewer.view(root)
            controls = server.gui.controls
            original_points = server.scene.cloud.points.copy()
            controls["Hide ceiling"].change(True)
            self.assertLess(len(server.scene.cloud.points), len(original_points))
            self.assertTrue(controls["Ceiling crop (%)"].visible)
            controls["Ceiling crop (%)"].change(50)
            clipped = server.scene.cloud.points.copy()
            self.assertLess(len(clipped), len(original_points))
            # Saved IDs still address the full observation table after display clipping.
            controls["Saved cloud point ID"].value = len(original_points) - 1
            controls["Show point source"].change(None)
            controls["Capture"].change("after")
            self.assertFalse(controls["Hide ceiling"].value)
            controls["Capture"].change("before")
            self.assertTrue(controls["Hide ceiling"].value)
            self.assertEqual(controls["Ceiling crop (%)"].value, 50)
            np.testing.assert_array_equal(server.scene.cloud.points, clipped)
            controls["Hide ceiling"].change(False)
            np.testing.assert_array_equal(server.scene.cloud.points, original_points)
            self.assertFalse(controls["Ceiling direction"].visible)
            for path, expected in hashes.items():
                self.assertEqual(core.digest(path), expected)


if __name__ == "__main__":
    unittest.main()
