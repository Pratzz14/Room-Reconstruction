"""Deterministic saved-image selection and source-faithful evidence exports."""

import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .core import digest, identity, read_json, safe_path
from .object_inspection import ObjectInspection
from .region_reference import RegionReferences

ROLES = {
    "object_before": ("before", "before"),
    "object_after": ("after", "after"),
    "old_location_after": ("before", "after"),
    "new_location_before": ("after", "before"),
}
DEFAULTS = {"max_views": 3, "crop_padding": 0.1, "minimum_projection_coverage": 0.2}
LIMITATIONS = {
    "no_supporting_view",
    "outside_frame",
    "suspected_occlusion",
    "visibility_unknown",
    "alignment_unusable",
    "region_unavailable",
}


def crop_box(box, size, padding=0.1):
    if box is None or len(box) != 4 or not np.isfinite(box).all():
        return None
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        return None
    dx, dy = (x1 - x0) * padding, (y1 - y0) * padding
    result = [
        max(0, math.floor(x0 - dx)),
        max(0, math.floor(y0 - dy)),
        min(size[0], math.ceil(x1 + dx)),
        min(size[1], math.ceil(y1 + dy)),
    ]
    return result if result[2] > result[0] and result[3] > result[1] else None


def sharpness(image, box):
    image = image.crop(tuple(box)) if box else image.copy()
    image.thumbnail((512, 512))
    gray = np.asarray(image.convert("L"), float)
    if min(gray.shape) < 3:
        return 0.0
    laplace = (
        gray[1:-1, :-2]
        + gray[1:-1, 2:]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        - 4 * gray[1:-1, 1:-1]
    )
    return float(np.var(laplace))


def depth_visibility(points, camera, depth, valid, margin):
    """Depth is only a heuristic: closer surfaces flag possible occlusion."""
    if not len(points):
        return {"status": "unknown", "samples": 0, "occluded_fraction": None}
    pose = np.asarray(camera["world_to_camera"], float)
    local = np.asarray(points) @ pose[:3, :3].T + pose[:3, 3]
    local = local[np.isfinite(local).all(1) & (local[:, 2] > 1e-6)]
    uv = local @ np.asarray(camera["intrinsics"]).T
    uv = np.rint(uv[:, :2] / uv[:, 2:3]).astype(int)
    inside = (
        (uv[:, 0] >= 0)
        & (uv[:, 0] < depth.shape[1])
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < depth.shape[0])
    )
    uv, local = uv[inside], local[inside]
    samples = depth[uv[:, 1], uv[:, 0]]
    good = valid[uv[:, 1], uv[:, 0]] & np.isfinite(samples) & (samples > 0)
    # Count distinct model pixels, so many projected points cannot fabricate support.
    _, unique = np.unique(uv[good], axis=0, return_index=True)
    z, samples = local[good, 2][unique], samples[good][unique]
    n = len(z)
    occluded = float(np.mean(samples + margin < z)) if n else None
    return {
        "status": "suspected_occlusion" if n >= 8 and occluded > 0.5 else "unknown",
        "samples": n,
        "occluded_fraction": occluded,
        "consistent_fraction": float(np.mean(np.abs(samples - z) <= margin))
        if n
        else None,
        "depth_margin": float(margin),
        "independently_validated": False,
    }


class EvidenceSelector:
    def __init__(self, inventory, alignment, config=None):
        self.inventory, self.alignment = inventory, alignment
        self.config = {**DEFAULTS, **(config or {})}
        if (
            set(self.config) != set(DEFAULTS)
            or type(self.config["max_views"]) is not int
            or not 1 <= self.config["max_views"] <= 3
        ):
            raise ValueError("Evidence max_views must be between one and three")
        if (
            not 0 <= self.config["crop_padding"] <= 0.5
            or not 0 < self.config["minimum_projection_coverage"] <= 1
        ):
            raise ValueError("Invalid evidence padding/coverage")
        self.inspection = ObjectInspection(inventory)
        self.regions = RegionReferences(inventory, alignment=alignment)
        self.frame_quality, self.depths, self.support = {}, {}, {}
        self.association = read_json(inventory.root / "config.json")["association"]

    def frame(self, capture, index):
        return self.inventory.captures[capture]["frames"][index]

    def mask_box(self, record, frame):
        if record.get("status") == "context":
            return record.get("source_region_box"), None
        path = safe_path(self.inventory.root, record["mask_path"])
        with Image.open(path) as image:
            mask = np.asarray(image) > 0
        if mask.shape != (frame["height"], frame["width"]):
            raise ValueError("Evidence mask has different source dimensions")
        y, x = np.where(mask)
        box = (
            [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1]
            if len(x)
            else None
        )
        return box, mask

    def object_candidates(self, capture, obj):
        if obj is None:
            return []
        ids = obj.get("observation_ids", []) + [
            "context:" + eid for eid in obj.get("context_evidence", [])
        ]
        output = []
        for eid in dict.fromkeys(ids):
            record = self.inspection.evidence_record(capture, eid)
            frame = self.frame(capture, record["frame_index"])
            box, mask = self.mask_box(record, frame)
            context = record.get("status") == "context"
            # Refined image-only identities may retain a parent shelf/container
            # observation for context. Its mask is not a mask of the named item.
            mask_is_context = (
                "geometry_observation_ids" in obj
                and eid not in obj["geometry_observation_ids"]
            )
            if mask_is_context:
                context = True
            item = self.candidate(
                capture,
                record["frame_index"],
                box,
                "context_region" if context else "object_mask",
                eid,
                coverage=float(mask.mean()) if mask is not None else 0.0,
                support=record.get("depth_support", 0),
            )
            item["source_quality"] = sorted(
                set(obj.get("quality", []) + record.get("quality", []))
            )
            item["mask_path"] = record.get("mask_path") if not context else None
            item["source_mask_sha256"] = (
                digest(safe_path(self.inventory.root, item["mask_path"]))
                if item["mask_path"]
                else None
            )
            if mask_is_context:
                item["notes"].append(
                    "Inherited segmentation is context only; it does not isolate the named object."
                )
            if (
                not context
                and obj.get("point_count")
                and self.alignment["quality"]["status"] == "accepted"
            ):
                points = self.regions.region_points(capture, obj, record)
                item["visibility"] = self.occlusion(
                    capture, record["frame_index"], points
                )
                if item["visibility"]["status"] == "suspected_occlusion":
                    item["limitations"].append("suspected_occlusion")
                    item["ranking"]["occlusion_penalty"] = 1.0
            output.append(item)
        return output

    def candidate(
        self, capture, index, box, method, source_id, coverage=0.0, support=0
    ):
        frame = self.frame(capture, index)
        box = [float(x) for x in box] if box is not None else None
        crop = crop_box(
            box, (frame["width"], frame["height"]), self.config["crop_padding"]
        )
        if crop is None:
            box = None
        key = (capture, index, tuple(crop or []))
        if key not in self.frame_quality:
            with Image.open(
                safe_path(self.inventory.reconstruction, frame["image_path"])
            ) as image:
                self.frame_quality[key] = sharpness(image, crop)
        blur = self.frame_quality[key]
        clipped = box is not None and (
            box[0] <= 1
            or box[1] <= 1
            or box[2] >= frame["width"] - 1
            or box[3] >= frame["height"] - 1
        )
        metrics = {
            "coverage": float(coverage),
            "border_clearance": float(not clipped),
            "sharpness": blur,
            "sharpness_normalized": blur / (blur + 100),
            "geometry_support": int(support),
            "geometry_support_normalized": min(support / 100, 1),
            "occlusion_penalty": 0.0,
        }
        return {
            "capture": capture,
            "frame_index": index,
            "frame_id": frame["frame_id"],
            "source_id": source_id,
            "method": method,
            "region_box": box,
            "crop_box": crop,
            "ranking": metrics,
            "visibility": {"status": "unknown", "samples": 0},
            "limitations": ["visibility_unknown"]
            + (["region_unavailable"] if box is None else []),
            "notes": [
                "Context and projected outlines are regions to inspect, not object masks or absence evidence."
            ]
            if method != "object_mask"
            else [],
        }

    def occlusion(self, capture, index, points):
        if capture not in self.depths:
            root = self.inventory.reconstruction / capture
            with np.load(root / "depth.npz", allow_pickle=False) as saved:
                depth = saved["depth"]
            with np.load(root / "valid.npz", allow_pickle=False) as saved:
                valid = saved["valid"]
            with np.load(root / "observations.npz", allow_pickle=False) as saved:
                voxel = float(saved["voxel_size"])
            scale = self.alignment["transform"]["scale"] if capture == "after" else 1
            self.depths[capture] = depth * scale, valid, voxel * scale
        depth, valid, voxel = self.depths[capture]
        return depth_visibility(
            points,
            self.regions.cameras_for(capture)[index],
            depth[index],
            valid[index],
            self.association["depth_margin_voxels"] * voxel,
        )

    def location_candidates(self, source_capture, obj, source_views, target):
        if obj is None or not source_views:
            return [], ["no_supporting_view"]
        if self.alignment["quality"]["status"] != "accepted" or not self.alignment.get(
            "transform"
        ):
            return [], ["alignment_unusable"]
        source_id = source_views[0]["source_id"]
        source = self.inspection.evidence_record(source_capture, source_id)
        points = self.regions.region_points(source_capture, obj, source)
        if not len(points):
            return [], ["region_unavailable"]
        bundle = self.regions.candidates(source_capture, obj, source_id, target)
        output = []
        for row in bundle["frames"]:
            # A random full frame is not evidence for an unlocated region.
            if (
                row["box_original"] is None
                or row["coverage"] < self.config["minimum_projection_coverage"]
            ):
                continue
            item = self.candidate(
                target,
                row["frame_index"],
                row["box_original"],
                "projected_region",
                source_id,
                coverage=row["coverage"],
                support=len(points),
            )
            item["source_capture"] = source_capture
            item["visibility"] = self.occlusion(target, row["frame_index"], points)
            if item["visibility"]["status"] == "suspected_occlusion":
                item["limitations"].append("suspected_occlusion")
                item["ranking"]["occlusion_penalty"] = 1.0
            output.append(item)
        return output, [] if output else ["outside_frame"]

    def select(self, candidates):
        for row in candidates:
            m = row["ranking"]
            row["score"] = float(
                0.35 * m["coverage"]
                + 0.2 * m["border_clearance"]
                + 0.3 * m["sharpness_normalized"]
                + 0.15 * m["geometry_support_normalized"]
                - 0.6 * m["occlusion_penalty"]
            )
            row["selection_reasons"] = [
                "direct_observation"
                if row["method"] == "object_mask"
                else row["method"],
                "ranked_by_coverage_clipping_sharpness_geometry_and_occlusion",
            ]
        ranked = sorted(
            candidates,
            key=lambda r: (
                r["region_box"] is None,
                r["method"] == "context_region",
                -r["score"],
                r["frame_id"],
                r["source_id"],
            ),
        )
        selected = []
        while ranked and len(selected) < self.config["max_views"]:
            chosen = ranked[0]
            if selected:
                # Prefer an alternative camera direction/position, then score; never duplicate a frame.
                def diverse(row):
                    pose = np.asarray(
                        self.regions.cameras_for(row["capture"])[row["frame_index"]][
                            "camera_to_world"
                        ]
                    )
                    for old in selected:
                        other = np.asarray(
                            self.regions.cameras_for(old["capture"])[
                                old["frame_index"]
                            ]["camera_to_world"]
                        )
                        if (
                            np.linalg.norm(pose[:3, 3] - other[:3, 3]) < 0.03
                            and np.dot(pose[:3, 2], other[:3, 2]) > 0.98
                        ):
                            return False
                    return True

                alternatives = [
                    r
                    for r in ranked
                    if r["method"] == chosen["method"] and r["region_box"] is not None
                ]
                varied = next((r for r in alternatives if diverse(r)), None)
                if varied is not None:
                    chosen = varied
                    chosen["selection_reasons"].append(
                        "alternative_viewpoint_preferred"
                    )
            selected.append(chosen)
            ranked = [r for r in ranked if r["frame_id"] != chosen["frame_id"]]
        for i, row in enumerate(selected):
            row["primary"] = i == 0
        return selected

    def event(self, before, after):
        objects = {"before": before, "after": after}
        direct = {
            c: self.select(self.object_candidates(c, obj)) for c, obj in objects.items()
        }
        result = {}
        for role, (source, target) in ROLES.items():
            if source == target:
                views, limitations = (
                    direct[source],
                    [] if direct[source] else ["no_supporting_view"],
                )
            else:
                candidates, limitations = self.location_candidates(
                    source, objects[source], direct[source], target
                )
                views = self.select(candidates)
            result[role] = {
                "capture": target,
                "views": views,
                "limitations": sorted(
                    set(limitations + [x for v in views for x in v["limitations"]])
                ),
            }
        return result

    def export(self, row, output):
        frame = self.frame(row["capture"], row["frame_index"])
        source = safe_path(self.inventory.reconstruction, frame["image_path"])
        sha = digest(source)
        if sha != frame["image_sha256"]:
            raise ValueError("Source image changed during evidence generation")
        assets = Path(output) / "assets"
        assets.mkdir(exist_ok=True)
        original = "assets/" + sha + ".png"
        if not (Path(output) / original).exists():
            shutil.copyfile(source, Path(output) / original)
        images = {"full": original, "highlight": None, "crop": None}
        if row["region_box"] is not None:
            key = identity(
                [
                    sha,
                    row["method"],
                    row["region_box"],
                    row["crop_box"],
                    row.get("source_mask_sha256"),
                ]
            )
            images.update(
                highlight=f"assets/{key}_highlight.png", crop=f"assets/{key}_crop.png"
            )
            if not (Path(output) / images["highlight"]).exists():
                with Image.open(source) as saved:
                    rgb = np.array(saved.convert("RGB"))
                if row["method"] == "object_mask":
                    with Image.open(
                        safe_path(self.inventory.root, row["mask_path"])
                    ) as saved:
                        mask = np.asarray(saved) > 0
                    rgb[mask] = (
                        rgb[mask] * 0.5 + np.array([0, 220, 255]) * 0.5
                    ).astype(np.uint8)
                image = Image.fromarray(rgb)
                if row["method"] != "object_mask":
                    ImageDraw.Draw(image).rectangle(
                        row["region_box"],
                        outline=(255, 185, 50),
                        width=max(3, image.width // 250),
                    )
                image.save(Path(output) / images["highlight"])
                # Close-ups retain the original pixels; highlight is separately selectable.
                with Image.open(source) as saved:
                    saved.crop(tuple(row["crop_box"])).save(
                        Path(output) / images["crop"]
                    )
        provenance = {
            key: frame.get(key)
            for key in (
                "decoded_frame_index",
                "source_pts",
                "time_base_numerator",
                "time_base_denominator",
                "source_timestamp_sec",
                "elapsed_timestamp_sec",
                "source_sha256",
                "image_sha256",
                "width",
                "height",
            )
        }
        return {
            **{k: v for k, v in row.items() if k != "mask_path"},
            **provenance,
            "source_image_path": frame["image_path"],
            "images": images,
            "image_coordinate_frame": "original_pixels",
            "crop_padding": self.config["crop_padding"],
        }
