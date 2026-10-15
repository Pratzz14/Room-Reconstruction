"""Opposing-capture region photos for visual inspection, never absence evidence."""

import io

import numpy as np
from PIL import Image, ImageDraw

from .core import identity, read_json, safe_path


def project_region(points, camera, mapping, frame):
    """Project common-world samples to original pixels, retaining camera/FOV checks."""
    points = np.asarray(points, float)
    if not len(points):
        return None, 0.0
    pose = np.asarray(camera["world_to_camera"], float)
    local = points @ pose[:3, :3].T + pose[:3, 3]
    front = local[:, 2] > 1e-6
    pixels = local[front] @ np.asarray(camera["intrinsics"], float).T
    pixels = pixels[:, :2] / pixels[:, 2:3]
    original = (
        np.c_[pixels, np.ones(len(pixels))]
        @ np.asarray(mapping["model_to_original"], float).T
    )
    original = original[:, :2] / original[:, 2:3]
    inside = (
        np.isfinite(original).all(1)
        & (original[:, 0] >= 0)
        & (original[:, 0] < frame["width"])
        & (original[:, 1] >= 0)
        & (original[:, 1] < frame["height"])
    )
    coverage = float(inside.sum() / len(points))
    if not inside.any():
        return None, coverage
    lo, hi = original[inside].min(0), original[inside].max(0)
    padding = np.maximum((hi - lo) * 0.08, 8)
    lo = np.maximum(lo - padding, 0)
    hi = np.minimum(hi + padding, [frame["width"] - 1, frame["height"] - 1])
    return np.r_[lo, hi].tolist(), coverage


class RegionReferences:
    def __init__(self, inventory, views):
        self.inventory, self.views = inventory, views
        self.transforms = {
            c: read_json(inventory.reconstruction / c / "image_transforms.json")
            for c in views
        }
        self.cache, self.images = {}, {}
        self.source_arrays = {}

    def region_points(self, capture, obj, source):
        if obj["point_count"]:
            _, geometry = self.inventory.object(capture, obj["id"])
            points = geometry["points"]
        elif source is not None:
            if capture not in self.source_arrays:
                with np.load(
                    self.inventory.reconstruction / capture / "observations.npz",
                    allow_pickle=False,
                ) as saved:
                    self.source_arrays[capture] = {
                        k: saved[k]
                        for k in (
                            "observation_frame_index",
                            "original_pixel_xy",
                            "cloud_point_index",
                        )
                    }
            arrays = self.source_arrays[capture]
            rows = np.flatnonzero(
                arrays["observation_frame_index"] == source["frame_index"]
            )
            xy = arrays["original_pixel_xy"][rows]
            box = source.get("source_region_box") or source.get("box")
            if box:
                lo, hi = np.asarray(box[:2]), np.asarray(box[2:])
                rows = rows[((xy >= lo) & (xy <= hi)).all(1)]
            indices = np.unique(arrays["cloud_point_index"][rows])
            # Display clouds are already in Before coordinates, including After's
            # single alignment transform. Never transform these points again.
            points = self.views[capture].points[capture][indices]
        else:
            return np.empty((0, 3))
        points = points[np.isfinite(points).all(1)]
        if len(points) > 512:
            points = points[np.linspace(0, len(points) - 1, 512).astype(int)]
        return points

    def candidates(self, source_capture, obj, source_id, target_capture):
        token = identity([source_capture, obj["id"], source_id, target_capture])[:24]
        if token in self.cache:
            return self.cache[token]
        source = (
            self.views[source_capture].inspection.evidence_record(
                source_capture, source_id
            )
            if source_id
            else None
        )
        accepted = (
            self.inventory.comparison["alignment"]["quality"]["status"] == "accepted"
        )
        points = (
            self.region_points(source_capture, obj, source)
            if accepted
            else np.empty((0, 3))
        )
        source_camera = (
            self.views[source_capture].cameras[source_capture][source["frame_index"]]
            if source
            else None
        )
        rows = []
        for i, (frame, camera, mapping) in enumerate(
            zip(
                self.inventory.captures[target_capture]["frames"],
                self.views[target_capture].cameras[target_capture],
                self.transforms[target_capture],
            )
        ):
            box, coverage = project_region(points, camera, mapping, frame)
            projected = box is not None and coverage >= 0.2
            score = coverage
            if source_camera is not None and accepted:
                a, b = (
                    np.asarray(c["camera_to_world"], float)
                    for c in (source_camera, camera)
                )
                orientation = float(np.clip(a[:3, 2] @ b[:3, 2], -1, 1))
                proximity = 1 / (1 + np.linalg.norm(a[:3, 3] - b[:3, 3]))
                score = 0.4 * coverage + 0.4 * ((orientation + 1) / 2) + 0.2 * proximity
            rows.append(
                {
                    "id": "region:" + frame["frame_id"],
                    "frame_id": frame["frame_id"],
                    "frame_index": i,
                    "timestamp_sec": frame["elapsed_timestamp_sec"],
                    "box_original": box if projected else None,
                    "method": "projected_region" if projected else "manual_reference",
                    "coverage": coverage,
                    "score": float(score),
                    "note": "Projected region reference; inspect visibility and absence visually."
                    if projected
                    else "Region could not be located in this frame; inspect manually.",
                }
            )
        rows.sort(
            key=lambda r: (
                r["method"] != "projected_region",
                -r["score"],
                r["frame_index"],
            )
        )
        result = {
            "token": token,
            "capture": target_capture,
            "source_capture": source_capture,
            "source_frame_id": source["frame_id"] if source else None,
            "frames": rows,
        }
        self.cache[token] = result
        return result

    def image(self, capture, token, identifier):
        bundle = self.cache.get(token)
        if not bundle or bundle["capture"] != capture:
            raise ValueError("Unknown region reference")
        record = next((r for r in bundle["frames"] if r["id"] == identifier), None)
        if record is None:
            raise ValueError("Unknown reference frame")
        key = (token, identifier)
        if key in self.images:
            return self.images[key]
        frame = self.inventory.captures[capture]["frames"][record["frame_index"]]
        with Image.open(
            safe_path(self.inventory.reconstruction, frame["image_path"])
        ) as saved:
            image = saved.convert("RGB")
        if record["box_original"] is not None:
            ImageDraw.Draw(image).rectangle(
                record["box_original"],
                outline=(204, 165, 255),
                width=max(3, image.width // 250),
            )
        image.thumbnail((1280, 960))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        if len(self.images) >= 12:
            self.images.pop(next(iter(self.images)))
        self.images[key] = buffer.getvalue()
        return self.images[key]
