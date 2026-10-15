"""Exact original-mask membership and capture-local observation geometry."""

import numpy as np
from PIL import Image

from .core import read_json
from .object_contracts import load_arrays


class MaskStore:
    """Small decoded-mask cache; original-resolution inventories can contain many masks."""

    def __init__(self, directory, observations, maximum=8):
        self.directory, self.maximum = directory, maximum
        self.paths = {o["id"]: o["mask_file"] for o in observations}
        self.cache = {}

    def __getitem__(self, identifier):
        if identifier not in self.cache:
            if len(self.cache) >= self.maximum:
                self.cache.pop(next(iter(self.cache)))
            with Image.open(self.directory / self.paths[identifier]) as image:
                self.cache[identifier] = np.asarray(image) > 0
        return self.cache[identifier]


class CaptureGeometry:
    def __init__(self, root, capture):
        self.root, self.capture = root, capture
        self.observations = load_arrays(root / capture / "observations.npz")
        self.cameras = read_json(root / capture / "cameras.json")
        self.frames = read_json(root / capture / "frames.json")["frames"]
        self.transforms = read_json(root / capture / "image_transforms.json")
        # Avoid retaining another copy of every dense XYZ on top of the saved arrays.
        self.voxel = float(self.observations["voxel_size"])
        self.frame_rows = [
            np.flatnonzero(self.observations["observation_frame_index"] == i)
            for i in range(len(self.frames))
        ]

    def points(self, rows):
        rows = np.asarray(rows, dtype=np.int64)
        source = self.observations
        output = np.empty((len(rows), 3), np.float32)
        for fi in np.unique(source["observation_frame_index"][rows]):
            chosen = np.flatnonzero(source["observation_frame_index"][rows] == fi)
            camera = self.cameras[int(fi)]
            pi = source["model_pixel_index"][rows[chosen]]
            uv = np.c_[
                pi % camera["model_width"],
                pi // camera["model_width"],
                np.ones(len(pi)),
            ]
            xyz = (uv @ np.linalg.inv(np.array(camera["intrinsics"])).T) * source[
                "depth"
            ][rows[chosen], None]
            pose = np.array(camera["camera_to_world"])
            output[chosen] = xyz @ pose[:3, :3].T + pose[:3, 3]
        return output

    def mask_rows(self, frame_index, mask):
        frame = self.frames[frame_index]
        mask = np.asarray(mask, bool)
        if mask.shape != (frame["height"], frame["width"]):
            raise ValueError("Mask must be in original-image coordinates")
        rows = self.frame_rows[frame_index]
        xy = np.floor(self.observations["original_pixel_xy"][rows] + 0.5).astype(
            np.int64
        )
        valid = (
            (xy[:, 0] >= 0)
            & (xy[:, 0] < mask.shape[1])
            & (xy[:, 1] >= 0)
            & (xy[:, 1] < mask.shape[0])
        )
        result = rows[valid][mask[xy[valid, 1], xy[valid, 0]]]
        return result[np.isfinite(self.points(result)).all(1)]

    def colors(self, rows):
        output = np.empty((len(rows), 3), np.uint8)
        source = self.observations
        for fi in np.unique(source["observation_frame_index"][rows]):
            chosen = np.flatnonzero(source["observation_frame_index"][rows] == fi)
            with Image.open(
                self.root / self.frames[int(fi)]["model_image_path"]
            ) as image:
                rgb = np.asarray(image.convert("RGB"))
            pi = source["model_pixel_index"][rows[chosen]]
            output[chosen] = rgb.reshape(-1, 3)[pi]
        return output


def voxel_geometry(capture, rows, size):
    rows = np.unique(rows).astype(np.int64)
    points = capture.points(rows)
    cells = np.floor(points / size).astype(np.int64)
    # Stable evidence order determines representatives, not frame frequency.
    _, representatives, inverse = np.unique(
        cells, axis=0, return_index=True, return_inverse=True
    )
    selected = rows[representatives]
    first = np.full(len(representatives), len(capture.frames), dtype=np.int64)
    np.minimum.at(first, inverse, capture.observations["observation_frame_index"][rows])
    return {
        "points": points[representatives],
        "colors": capture.colors(selected),
        "representative_observation": selected,
        "first_frame": first,
    }


def geometry_quality(points, mask_area, settings, voxel):
    warnings = []
    if mask_area < settings["min_mask_pixels"]:
        warnings.append("small_mask")
    if len(points) < max(3, settings["min_points"]):
        warnings.append("insufficient_depth_support")
    if len(points) >= 3:
        extent = np.sort(np.ptp(points, axis=0))
        if extent[-2] < settings["min_extent_voxels"] * voxel:
            warnings.append("degenerate_geometry")
    return warnings


def masked_crop(image, mask, background):
    rgb = np.array(image.convert("RGB"))
    mask = np.asarray(mask, bool)
    if mask.shape != rgb.shape[:2]:
        raise ValueError("Appearance mask dimensions mismatch")
    y, x = np.where(mask)
    if not len(x):
        return Image.new("RGB", (16, 16), tuple(background))
    rgb[~mask] = background
    return Image.fromarray(rgb).crop(
        (int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1)
    )
