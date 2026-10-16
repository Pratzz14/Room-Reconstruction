"""Versioned, reconstruction-bound CPU object artifacts; no model imports."""

from pathlib import Path

import numpy as np
from PIL import Image

from .core import (
    atomic_replace,
    check_artifacts,
    digest,
    identity,
    read_json,
    safe_path,
)

VERSION = "objects-v2-1"
CAPTURES = ("before", "after")


def atomic_move(source, destination):
    return atomic_replace(source, destination)


def stable_id(prefix, evidence):
    return prefix + "_" + identity([VERSION, sorted(evidence)])[:16]


def validate_config(config):
    if config.get("schema_version") != "1.0" or config.get("device") not in (
        "cpu",
        "cuda",
        "auto",
    ):
        raise ValueError("Unsupported object configuration")
    d, g, a = (config[k] for k in ("detection", "geometry", "association"))
    if not d["vocabulary"] or any(
        not isinstance(x, str) or not x.strip() for x in d["vocabulary"]
    ):
        raise ValueError("Object vocabulary must contain nonempty labels")
    probabilities = [
        d["box_threshold"],
        d["text_threshold"],
        d["duplicate_mask_iou"],
        config["segmentation"]["mask_threshold"],
        config["segmentation"]["overlap_warning_fraction"],
        a["overlap_fraction"],
        a["reprojection_fraction"],
        a["contradiction_fraction"],
    ]
    if any(not np.isfinite(x) or not 0 < x <= 1 for x in probabilities):
        raise ValueError("Invalid probability/overlap threshold")
    if not -1 <= a["appearance_similarity"] <= 1:
        raise ValueError("Invalid appearance threshold")
    for value in (
        g["min_points"],
        g["min_mask_pixels"],
        g["voxel_multiplier"],
        g["min_extent_voxels"],
        d["max_detections_per_frame"],
        a["max_pair_points"],
        a["depth_margin_voxels"],
        a["candidate_extent_multiplier"],
        config["viewer"]["max_object_points"],
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Object sizes and budgets must be positive")
    return config


def reconstruction_binding(root, capture, *, verify=False):
    from .core import verify_capture

    root = Path(root)
    if capture not in CAPTURES:
        raise ValueError("Unknown capture")
    manifest = read_json(root / "run_manifest.json")
    record = manifest["captures"][capture]
    if record["status"] != "completed":
        raise ValueError(f"{capture}: reconstruction is not completed")
    if verify:
        verify_capture(root, capture)
    # Use authoritative reconstruction artifact hashes, independent of archive path.
    check_artifacts(root, record)
    return {
        "capture": capture,
        "coordinate_frame": record["coordinate_frame"],
        "units": "relative",
        "frame_ids": record["frame_ids"],
        "source_video_sha256": record["source_video_sha256"],
        "artifacts": record["artifacts"],
        "config_identity": manifest["config_identity"],
        "model_identity": manifest["model_identity"],
    }


def load_arrays(path):
    with np.load(path, allow_pickle=False) as values:
        return {k: values[k] for k in values.files}


def check_indices(values, count, description):
    values = np.asarray(values)
    if (
        values.ndim != 1
        or values.dtype.kind not in "iu"
        or np.any(values >= count)
        or np.any(values < 0)
    ):
        raise ValueError(f"Invalid {description} indices")
    if len(values) and (np.diff(values.astype(np.int64)) <= 0).any():
        raise ValueError(f"Unsorted or duplicate {description} indices")


def offsets(values, length, rows, name):
    if (
        values.shape != (rows + 1,)
        or values.dtype.kind not in "iu"
        or values[0] != 0
        or values[-1] != length
        or np.any(np.diff(values.astype(np.int64)) < 0)
    ):
        raise ValueError(f"Invalid {name} offsets")


class ObjectInventory:
    """Fully checked inventory and lazy per-capture geometry for saved inspection."""

    def __init__(self, reconstruction, root, *, require_complete=False, captures=None):
        self.reconstruction, self.root = (
            Path(reconstruction).resolve(),
            Path(root).resolve(),
        )
        if (self.root / ".objects.lock").exists():
            raise ValueError(
                "Object writer is active; open a completed inventory after it stops"
            )
        self.manifest = read_json(self.root / "manifest.json")
        if self.manifest.get("version") != VERSION:
            raise ValueError("Unsupported object artifact version")
        if (
            identity(read_json(self.root / "config.json"))
            != self.manifest["config_identity"]
        ):
            raise ValueError("Object configuration identity mismatch")
        if (
            identity(read_json(self.root / "models.lock.json"))
            != self.manifest["model_identity"]
        ):
            raise ValueError("Object model identity mismatch")
        self.config = validate_config(read_json(self.root / "config.json"))
        self.captures = {}
        for capture, record in self.manifest["captures"].items():
            if capture not in CAPTURES:
                raise ValueError("Unknown object capture")
            if captures is not None and capture not in captures:
                continue
            if record.get("status") not in ("completed", "completed_with_warnings"):
                if require_complete:
                    raise ValueError(f"{capture}: object processing incomplete")
                continue
            current = reconstruction_binding(self.reconstruction, capture)
            if record["binding"] != current:
                raise ValueError(f"{capture}: objects belong to another reconstruction")
            required = {
                f"{capture}/{name}"
                for name in (
                    "objects.json",
                    "observations.json",
                    "membership.npz",
                    "geometry.npz",
                    "descriptors.npz",
                    "diagnostics.json",
                )
            }
            if not required.issubset(record["artifacts"]) or any(
                not name.startswith(capture + "/") for name in record["artifacts"]
            ):
                raise ValueError(
                    "Object artifact inventory is incomplete or crosses captures"
                )
            for relative, expected in record["artifacts"].items():
                if digest(safe_path(self.root, relative)) != expected:
                    raise ValueError(f"Object artifact hash mismatch: {relative}")
            self.captures[capture] = self._load(capture, current)
        if require_complete and set(self.captures) != set(CAPTURES):
            raise ValueError("Both object captures are required")

    def _load(self, capture, binding):
        directory = self.root / capture
        observations = read_json(directory / "observations.json")["observations"]
        objects = read_json(directory / "objects.json")["objects"]
        members = load_arrays(directory / "membership.npz")
        geometry = load_arrays(directory / "geometry.npz")
        descriptors = load_arrays(directory / "descriptors.npz")["descriptors"]
        source = load_arrays(self.reconstruction / capture / "observations.npz")
        n = len(source["depth"])
        for prefix in ("observation", "object"):
            values = members[prefix + "_indices"]
            if (
                values.ndim != 1
                or values.dtype.kind not in "iu"
                or np.any(values < 0)
                or np.any(values >= n)
            ):
                raise ValueError("Invalid source observation membership")
            if not np.array_equal(
                members[prefix + "_cloud_indices"],
                source["cloud_point_index"][members[prefix + "_indices"]],
            ):
                raise ValueError(
                    "Object cloud-row mapping differs from source membership"
                )
        offsets(
            members["observation_offsets"],
            len(members["observation_indices"]),
            len(observations),
            "observation",
        )
        offsets(
            members["object_offsets"],
            len(members["object_indices"]),
            len(objects),
            "object",
        )
        offsets(geometry["offsets"], len(geometry["points"]), len(objects), "geometry")
        if (
            descriptors.ndim != 2
            or descriptors.shape[0] != len(observations)
            or not np.isfinite(descriptors).all()
        ):
            raise ValueError("Invalid descriptors")
        count = len(geometry["points"])
        if (
            geometry["points"].shape != (count, 3)
            or geometry["colors"].shape != (count, 3)
            or not np.isfinite(geometry["points"]).all()
            or geometry["first_frame"].shape != (count,)
            or geometry["representative_observation"].shape != (count,)
        ):
            raise ValueError("Invalid object geometry")
        ids = [o["id"] for o in observations]
        if len(ids) != len(set(ids)) or len({o["id"] for o in objects}) != len(objects):
            raise ValueError("Duplicate inventory IDs")
        index = {o["id"]: i for i, o in enumerate(observations)}
        frames = read_json(self.reconstruction / capture / "frames.json")["frames"]
        for i, observation in enumerate(observations):
            fi = observation["frame_index"]
            if (
                not isinstance(fi, int)
                or not 0 <= fi < len(frames)
                or observation["frame_id"] != frames[fi]["frame_id"]
            ):
                raise ValueError("Object observation/frame mismatch")
            with Image.open(safe_path(self.root, observation["mask_path"])) as mask:
                if mask.size != (frames[fi]["width"], frames[fi]["height"]):
                    raise ValueError("Object mask dimensions mismatch")
                mask_values = np.asarray(mask) > 0
            if observation["mask_path"] not in self.manifest["captures"][capture][
                "artifacts"
            ] or observation["status"] not in ("usable", "unresolved"):
                raise ValueError("Unbound object mask or invalid observation status")
            lo, hi = members["observation_offsets"][i : i + 2]
            selected = members["observation_indices"][lo:hi]
            check_indices(selected, n, "observation membership")
            if not np.all(source["observation_frame_index"][selected] == fi):
                raise ValueError("Object membership crosses source frames")
            xy = np.floor(source["original_pixel_xy"][selected] + 0.5).astype(np.int64)
            mask = mask_values
            if (
                np.any(xy < 0)
                or np.any(xy[:, 0] >= mask.shape[1])
                or np.any(xy[:, 1] >= mask.shape[0])
                or not mask[xy[:, 1], xy[:, 0]].all()
            ):
                raise ValueError("Object membership falls outside its mask")
        for i, obj in enumerate(objects):
            if (
                obj["capture"] != capture
                or obj["coordinate_frame"] != binding["coordinate_frame"]
                or obj["units"] != "relative"
            ):
                raise ValueError("Object coordinate mismatch")
            if not obj["observation_ids"] or any(
                x not in index for x in obj["observation_ids"]
            ):
                raise ValueError("Unknown object observation")
            if obj["id"] != stable_id(capture + "_obj", obj["observation_ids"]):
                raise ValueError("Object ID/evidence identity mismatch")
            a, b = members["object_offsets"][i : i + 2]
            selected = members["object_indices"][a:b]
            check_indices(selected, n, "object membership")
            expected = np.unique(
                np.concatenate(
                    [
                        members["observation_indices"][
                            slice(
                                *members["observation_offsets"][index[x] : index[x] + 2]
                            )
                        ]
                        for x in obj["observation_ids"]
                    ]
                )
            )
            if not np.array_equal(selected, expected):
                raise ValueError("Object/group membership mismatch")
            a, b = geometry["offsets"][i : i + 2]
            pts, rep = (
                geometry["points"][a:b],
                geometry["representative_observation"][a:b],
            )
            if len(pts) < 3 or not np.isin(rep, selected).all():
                raise ValueError("Object geometry lacks source support")
            if (
                obj["point_count"] != len(pts)
                or geometry["first_frame"].dtype.kind not in "iu"
                or rep.dtype.kind not in "iu"
            ):
                raise ValueError("Invalid object geometry count or index type")
            if not np.allclose(
                pts.mean(0), obj["centroid"], atol=1e-6
            ) or not np.allclose([pts.min(0), pts.max(0)], obj["bounds"], atol=1e-6):
                raise ValueError("Object centroid/bounds mismatch")
            if np.any(geometry["first_frame"][a:b] >= len(frames)) or np.any(
                geometry["first_frame"][a:b] < 0
            ):
                raise ValueError("Invalid object playback support")
        # Check XYZ/color provenance against actual observations, including mixed room voxels.
        if count:
            from .object_geometry import CaptureGeometry

            capture_geometry = CaptureGeometry(self.reconstruction, capture)
            sample = np.unique(np.linspace(0, count - 1, min(5000, count)).astype(int))
            rows = geometry["representative_observation"][sample]
            if not np.allclose(
                geometry["points"][sample], capture_geometry.points(rows), atol=1e-6
            ) or not np.array_equal(
                geometry["colors"][sample], capture_geometry.colors(rows)
            ):
                raise ValueError("Object geometry differs from its source observations")
        return {
            "observations": observations,
            "objects": objects,
            "members": members,
            "geometry": geometry,
            "frames": frames,
            "index": index,
        }

    def listing(self, capture):
        data = self.captures.get(capture)
        if data is None:
            record = self.manifest["captures"].get(capture, {})
            return {
                "objects": [],
                "unresolved": [],
                "status": record.get("status", "unavailable"),
            }
        return {
            "objects": data["objects"],
            "unresolved": [o for o in data["observations"] if o["status"] != "usable"],
            "observations": [
                {
                    k: o[k]
                    for k in (
                        "id",
                        "frame_index",
                        "frame_id",
                        "timestamp_sec",
                        "label",
                        "status",
                        "quality",
                    )
                }
                for o in data["observations"]
            ],
            "status": self.manifest["captures"][capture]["status"],
            "synthetic": self.manifest.get("synthetic", False),
        }

    def object(self, capture, identifier):
        data = self.captures.get(capture, {})
        for i, obj in enumerate(data.get("objects", [])):
            if obj["id"] == identifier:
                a, b = data["geometry"]["offsets"][i : i + 2]
                return obj, {
                    k: v[a:b] for k, v in data["geometry"].items() if k != "offsets"
                }
        raise ValueError("Object ID is unavailable in the selected capture")
