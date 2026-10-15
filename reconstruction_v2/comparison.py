"""Saved-run alignment, checked exports, global identities and viewer adapter."""

import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .alignment import apply_similarity, transform_bounds, validate_similarity
from .change_detection import match_objects
from .change_stage import ambiguous_objects, build_changes, saved_changes
from .core import digest, identity, read_json, safe_path, write_json
from .object_contracts import ObjectInventory, load_arrays
from .refined_inventory import RefinedInventory
from .refinement_evidence import binding
from .refinement_gemini import exclusive, validate_output
from .static_alignment import extract_correspondences, fit_static, scene_extent


def load_inventory(reconstruction, objects, refinement=None):
    print("Verifying saved source inventories...", flush=True)
    raw = ObjectInventory(reconstruction, objects, require_complete=True)
    return RefinedInventory(raw, refinement) if refinement else raw


def source_binding(inventory):
    return {
        "raw": binding(inventory),
        "refinement_sha256": digest(inventory.refinement_root / "refinement.json")
        if hasattr(inventory, "refinement_root")
        else None,
    }


def matching_objects(inventory, capture):
    data = inventory.captures[capture]
    descriptors = load_arrays(inventory.root / capture / "descriptors.npz")[
        "descriptors"
    ]
    output = []
    for obj in data["objects"]:
        item = copy.deepcopy(obj)
        # Do not borrow the DINO descriptor of a parent shelf for an image-only bottle.
        observation_ids = obj.get("geometry_observation_ids", obj["observation_ids"])
        rows = [data["index"][oid] for oid in observation_ids]
        desc = descriptors[rows]
        if len(desc):
            normalized = desc / np.maximum(
                np.linalg.norm(desc, axis=1, keepdims=True), 1e-12
            )
            item["descriptor"] = normalized.mean(0)
        else:
            item["descriptor"] = []
        output.append(item)
    return output


def transformed_cameras(cameras, transform):
    scale, rotation, _ = validate_similarity(transform)
    output = copy.deepcopy(cameras)
    for camera in output:
        pose = np.array(camera["camera_to_world"])
        pose[:3, 3] = apply_similarity(pose[:3, 3], transform)
        pose[:3, :3] = rotation @ pose[:3, :3]
        camera.update(
            camera_to_world=pose.tolist(),
            world_to_camera=np.linalg.inv(pose).tolist(),
            coordinate_frame="before_world",
            depth_scale_from_source=scale,
        )
    return output


def aligned_objects(inventory, transform, accepted):
    objects = copy.deepcopy(inventory.captures["after"]["objects"])
    for obj in objects:
        obj["source_centroid"] = obj["centroid"]
        obj["source_coordinate_frame"] = obj["coordinate_frame"]
        obj["coordinate_frame"] = "before_world"
        obj["alignment_status"] = "accepted" if accepted else "diagnostic_preview"
        if obj["centroid"] is not None:
            obj["centroid"] = apply_similarity(obj["centroid"], transform).tolist()
            obj["bounds"] = transform_bounds(obj["bounds"], transform).tolist()
        obj["location_comparison_allowed"] = accepted and obj.get(
            "movement_centroid_usable", False
        )
    return objects


def export_aligned(inventory, output, transform, accepted):
    directory = output / "after"
    directory.mkdir(exist_ok=True)
    with np.load(
        inventory.reconstruction / "after/cloud.npz", allow_pickle=False
    ) as archive:
        cloud = {k: archive[k] for k in archive.files}
    cloud["points"] = apply_similarity(cloud["points"], transform).astype(np.float32)
    cloud["coordinate_frame"] = np.array("before_world")
    np.savez_compressed(directory / "cloud.npz", **cloud)
    raw_cameras = read_json(inventory.reconstruction / "after/cameras.json")
    write_json(directory / "cameras.json", transformed_cameras(raw_cameras, transform))
    with np.load(
        inventory.reconstruction / "after/depth.npz", allow_pickle=False
    ) as archive:
        depth = {k: archive[k] for k in archive.files}
    depth["depth"] *= transform["scale"]
    np.savez_compressed(directory / "depth.npz", **depth)
    geometry = dict(inventory.captures["after"]["geometry"])
    geometry["points"] = apply_similarity(geometry["points"], transform).astype(
        np.float32
    )
    np.savez_compressed(directory / "geometry.npz", **geometry)
    write_json(
        directory / "objects.json",
        {"objects": aligned_objects(inventory, transform, accepted)},
    )
    write_json(
        directory / "provenance.json",
        {
            "source_reconstruction": str(inventory.reconstruction),
            "source_capture": "after",
            "source_objects": str(
                getattr(inventory, "refinement_root", inventory.root)
            ),
            "cloud_rows": "identical row order and colors; source cloud_point_index remains valid",
            "object_membership": "source membership, offsets, representative observation IDs and frame IDs unchanged",
            "observation_coordinates": "source pixels and membership refer to original files; XYZ transforms by saved Sim(3)",
            "depth_contract": "optical-axis Z scaled with camera translations; intrinsics and pixels unchanged",
            "raw_sources_modified": False,
            "trust": "accepted_geometry_checks"
            if accepted
            else "diagnostic_preview_only",
        },
    )


def validate_config(config):
    if config.get("version") not in ("comparison-v1", "comparison-v2"):
        raise ValueError("Unknown comparison config version")
    positive = (
        "sift_features",
        "max_matches_per_pair_region",
        "ransac_iterations",
        "min_correspondences",
        "min_frame_pairs",
        "min_regions",
    )
    if (
        any(not isinstance(config[k], int) or config[k] < 1 for k in positive)
        or config["holdout_stride"] < 2
    ):
        raise ValueError("Invalid comparison sample/iteration settings")
    if any(
        not 0 < config[k] < 1
        for k in (
            "ratio_test",
            "residual_extent_fraction",
            "min_inlier_fraction",
            "min_support_extent_fraction",
        )
    ):
        raise ValueError("Invalid comparison fractions")
    if not 0 < config["minimum_scale"] < config["maximum_scale"]:
        raise ValueError("Invalid similarity scale limits")
    fraction = config.get("movement_extent_fraction", 0.2)
    if isinstance(fraction, bool) or not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("Movement extent fraction must be in (0, 1]")
    return config


def run(reconstruction, objects, refinement, output, config):
    output = Path(output).resolve()
    validate_config(config)
    validate_output(reconstruction, objects, output)
    if refinement and (
        output == Path(refinement).resolve()
        or output in Path(refinement).resolve().parents
        or Path(refinement).resolve() in output.parents
    ):
        raise ValueError("Comparison output must be separate from refinement")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()) and not (output / "inputs.json").exists():
        raise ValueError("Nonempty output is not a comparison workspace")
    if (output / "comparison.json").exists() and read_json(
        output / "comparison.json"
    ).get("version") != "comparison-v2":
        raise ValueError("Preserve the older comparison; use a new output directory")
    with exclusive(output / ".comparison.lock"):
        inventory = load_inventory(reconstruction, objects, refinement)
        inputs = {"sources": source_binding(inventory), "config": config}
        path = output / "inputs.json"
        if path.exists() and read_json(path) != inputs:
            raise ValueError(
                "Comparison inputs/config changed; use a new output directory"
            )
        write_json(path, inputs)
        feature_binding = identity(
            [inputs, digest(Path(__file__).with_name("static_alignment.py"))]
        )
        cached = output / "correspondence_index.json"
        if (
            cached.exists()
            and read_json(cached).get("feature_binding") == feature_binding
        ):
            index = read_json(cached)
            if digest(output / "correspondences.npz") != index["sha256"]:
                raise ValueError("Cached correspondences were modified")
            correspondences, regions = (
                load_arrays(output / "correspondences.npz"),
                index["regions"],
            )
        else:
            correspondences, regions = extract_correspondences(inventory, config)
            np.savez_compressed(output / "correspondences.npz", **correspondences)
            write_json(
                cached,
                {
                    "feature_binding": feature_binding,
                    "sha256": digest(output / "correspondences.npz"),
                    "regions": regions,
                    "frames": {
                        c: [
                            {
                                k: f[k]
                                for k in (
                                    "frame_id",
                                    "decoded_frame_index",
                                    "elapsed_timestamp_sec",
                                    "model_image_path",
                                )
                            }
                            for f in inventory.captures[c]["frames"]
                        ]
                        for c in ("before", "after")
                    },
                    "columns": {
                        "source": "after_world XYZ",
                        "target": "before_world XYZ",
                        "frame_pairs": "after_index,before_index",
                        "model_pixels": "after_x,after_y,before_x,before_y",
                    },
                    "validation_partition": "Every fifth frame in either capture is withheld entirely from fitting",
                },
            )
        with np.load(
            inventory.reconstruction / "before/cloud.npz", allow_pickle=False
        ) as archive:
            extent = scene_extent(archive["points"])
        print(
            f"Fitting alignment from {len(correspondences['source'])} static feature pairs...",
            flush=True,
        )
        transform, quality = fit_static(correspondences, regions, extent, config)
        serial = (
            {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in transform.items()
            }
            if transform
            else None
        )
        accepted = quality["status"] == "accepted"
        alignment = {
            "version": "alignment-v2",
            "direction": "after_to_before",
            "coordinate_frame": "before_world",
            "units": "relative",
            "transform": serial,
            "quality": quality,
            "sources": inputs["sources"],
            "interpretation": "Geometry checks on separate image frames; not independent depth calibration or ground truth",
        }
        write_json(output / "alignment.json", alignment)
        if transform:
            print(
                "Exporting After geometry, depths, cameras and centroids in Before coordinates...",
                flush=True,
            )
            export_aligned(inventory, output, transform, accepted)
        before, after = (matching_objects(inventory, c) for c in ("before", "after"))
        matching = match_objects(
            before,
            after,
            after_transform=transform,
            alignment_accepted=accepted,
            visual_links=getattr(inventory, "report", {}).get("links", []),
            config={**config["matching"], "position_scale": extent},
        )
        identities = []
        for match in matching["matches"]:
            identities.append(
                {
                    **match,
                    "shared_id": "shared_"
                    + identity([match["before_id"], match["after_id"]])[:16],
                }
            )
        for cap in ("before", "after"):
            for oid in matching["unmatched_" + cap]:
                identities.append(
                    {
                        "shared_id": "shared_" + identity([oid])[:16],
                        "before_id": oid if cap == "before" else None,
                        "after_id": oid if cap == "after" else None,
                        "status": "unmatched",
                        "reasons": ["no_supported_identity_match"],
                    }
                )
        affected = ambiguous_objects(identities, matching)
        for item in identities:
            if {item["before_id"], item["after_id"]} & affected:
                item["status"] = "ambiguous"
                item["reasons"] = sorted(
                    set(item.get("reasons", []) + ["competing_identity_group"])
                )
        changes = build_changes(
            inventory,
            identities,
            matching,
            alignment,
            extent,
            config.get("movement_extent_fraction", 0.2),
        )
        write_json(output / "changes.json", changes)
        write_json(
            output / "identities.json", {"identities": identities, "matching": matching}
        )
        files = [
            "inputs.json",
            "alignment.json",
            "identities.json",
            "changes.json",
            "correspondences.npz",
            "correspondence_index.json",
        ]
        files += (
            [
                str(p.relative_to(output)).replace("\\", "/")
                for p in (output / "after").glob("*")
            ]
            if transform
            else []
        )
        report = {
            "version": "comparison-v2",
            "sources": inputs["sources"],
            "source_paths": {
                "reconstruction": str(Path(reconstruction).resolve()),
                "objects": str(Path(objects).resolve()),
                "refinement": str(Path(refinement).resolve()) if refinement else None,
            },
            "synthetic": inventory.manifest.get("synthetic", False),
            "alignment": alignment,
            "identity_counts": dict(Counter(x["status"] for x in identities)),
            "change_status": "evaluated" if accepted else "uncertain_alignment",
            "change_counts": changes["counts"],
            "provider_calls": 0,
            "code": {
                name: digest(Path(__file__).with_name(name))
                for name in (
                    "comparison.py",
                    "static_alignment.py",
                    "alignment.py",
                    "change_detection.py",
                    "change_stage.py",
                )
            },
            "artifacts": {p: digest(output / p) for p in files},
        }
        write_json(output / "comparison.json", report)
        summary(output, report, identities, inventory)
        return report


def summary(output, report, identities, inventory):
    q = report["alignment"]["quality"]
    labels = {
        o["id"]: o["label"] for d in inventory.captures.values() for o in d["objects"]
    }
    rows = [
        "# Shared-frame alignment and object identities",
        "",
        f"Alignment: **{q['status']}**. Relative units, After → Before.",
        "",
        f"Identity counts: {report['identity_counts']}. Provider calls: 0. Changes: {report['change_status']}.",
        "",
        "Rejected alignment permits a diagnostic overlay only; location comparisons remain blocked.",
        "",
        "## Alignment diagnostics",
        "",
        "```json",
        json.dumps(q, indent=2),
        "```",
        "",
        "## Identity proposals",
        "",
        "| Shared identity | Label | Before | After | Status |",
        "|---|---|---|---|---|",
    ]
    for obj in identities:
        rows.append(
            f"| {obj['shared_id']} | {labels.get(obj['before_id'] or obj['after_id'])} | {obj['before_id'] or '—'} | {obj['after_id'] or '—'} | {obj['status']} |"
        )
    rows += [
        "",
        "See identities.json for costs, semantic/appearance/shape/position evidence and alternatives. Image evidence remains linked through capture-specific IDs and supporting frames in the source inventory.",
        "",
        "Image-only objects keep null centroids. Reliable geometry is an MVP evidence basis, not independent metric validation.",
        "",
    ]
    changes = read_json(output / "changes.json")
    rows += [
        "## Object changes",
        "",
        f"Counts: {changes['counts']}. Threshold: {changes['threshold']:.6g} relative units ({changes['movement_extent_fraction']:.0%} of room extent).",
        "",
        *["> " + assumption for assumption in changes["assumptions"]],
        "",
        "| Label | Shared identity | Change | Trust | Basis | Explanation |",
        "|---|---|---|---|---|---|",
    ]
    for event in changes["events"]:
        cells = [
            event[k]
            for k in ("label", "shared_id", "type", "trust", "basis", "explanation")
        ]
        rows.append(
            "| "
            + " | ".join(
                str(cell).replace("|", "\\|").replace("\n", " ") for cell in cells
            )
            + " |"
        )
    rows.append("")
    (output / "summary.md").write_text("\n".join(rows), encoding="utf-8")


def verify(inventory, output):
    output = Path(output)
    if (output / ".comparison.lock").exists():
        raise ValueError("Comparison writer active")
    report = read_json(output / "comparison.json")
    if report.get("version") not in ("comparison-v1", "comparison-v2") or report[
        "sources"
    ] != source_binding(inventory):
        raise ValueError("Comparison belongs to different source artifacts")
    required = {
        "inputs.json",
        "alignment.json",
        "identities.json",
        "correspondences.npz",
        "correspondence_index.json",
    }
    if report["version"] == "comparison-v2":
        required.add("changes.json")
    if report["alignment"]["transform"]:
        required |= {
            "after/cloud.npz",
            "after/geometry.npz",
            "after/depth.npz",
            "after/cameras.json",
            "after/objects.json",
            "after/provenance.json",
        }
    if not required.issubset(report["artifacts"]):
        raise ValueError("Comparison is missing required bound artifacts")
    for path, sha in report["artifacts"].items():
        if digest(safe_path(output, path)) != sha:
            raise ValueError("Comparison artifact hash mismatch: " + path)
    if (
        read_json(output / "alignment.json") != report["alignment"]
        or report["alignment"]["sources"] != report["sources"]
    ):
        raise ValueError("Alignment and comparison manifests disagree")
    identities = read_json(output / "identities.json")["identities"]
    if len({i["shared_id"] for i in identities}) != len(identities):
        raise ValueError("Duplicate shared identity")
    for cap in ("before", "after"):
        assigned = [i[cap + "_id"] for i in identities if i[cap + "_id"] is not None]
        expected_ids = {o["id"] for o in inventory.captures[cap]["objects"]}
        if len(assigned) != len(set(assigned)) or set(assigned) != expected_ids:
            raise ValueError(
                "Shared identities must cover each capture object exactly once"
            )
    if dict(Counter(i["status"] for i in identities)) != report["identity_counts"]:
        raise ValueError("Shared identity counts disagree")
    if report["version"] == "comparison-v2":
        inputs = read_json(output / "inputs.json")
        validate_config(inputs["config"])
        if inputs["sources"] != report["sources"]:
            raise ValueError("Comparison input sources disagree")
        expected_changes = saved_changes(
            inventory,
            identities,
            read_json(output / "identities.json")["matching"],
            report["alignment"],
            inputs["config"],
        )
        accepted = report["alignment"]["quality"]["status"] == "accepted"
        if (
            read_json(output / "changes.json") != expected_changes
            or report.get("change_counts") != expected_changes["counts"]
            or report["change_status"]
            != ("evaluated" if accepted else "uncertain_alignment")
        ):
            raise ValueError(
                "Changes disagree with source locations, identities or threshold"
            )
    transform = report["alignment"]["transform"]
    if transform:
        validate_similarity(transform)
        with (
            np.load(
                inventory.reconstruction / "after/cloud.npz", allow_pickle=False
            ) as raw,
            np.load(output / "after/cloud.npz", allow_pickle=False) as aligned,
        ):
            sample = np.unique(
                np.linspace(
                    0, len(raw["points"]) - 1, min(len(raw["points"]), 5000)
                ).astype(int)
            )
            if (
                len(raw["points"]) != len(aligned["points"])
                or not np.allclose(
                    apply_similarity(raw["points"][sample], transform),
                    aligned["points"][sample],
                    atol=1e-6,
                )
                or not np.array_equal(raw["colors"], aligned["colors"])
            ):
                raise ValueError(
                    "Aligned cloud lost source row/color/coordinate provenance"
                )
        original_geometry = inventory.captures["after"]["geometry"]
        geometry = load_arrays(output / "after/geometry.npz")
        for key, value in original_geometry.items():
            expected_value = (
                apply_similarity(value, transform).astype(np.float32)
                if key == "points"
                else value
            )
            agrees = (
                np.allclose(geometry[key], expected_value, atol=1e-6)
                if key == "points"
                else np.array_equal(geometry[key], expected_value)
            )
            if not agrees:
                raise ValueError(
                    "Aligned object geometry changed source membership or coordinates"
                )
        with (
            np.load(
                inventory.reconstruction / "after/depth.npz", allow_pickle=False
            ) as raw,
            np.load(output / "after/depth.npz", allow_pickle=False) as aligned,
        ):
            if not np.array_equal(
                raw["frame_ids"], aligned["frame_ids"]
            ) or not np.allclose(
                raw["depth"] * transform["scale"], aligned["depth"], equal_nan=True
            ):
                raise ValueError("Aligned depths disagree with camera/coordinate scale")
        expected = aligned_objects(
            inventory, transform, report["alignment"]["quality"]["status"] == "accepted"
        )
        if read_json(output / "after/objects.json")["objects"] != expected:
            raise ValueError("Aligned object coordinates disagree with transform")
        cameras = read_json(inventory.reconstruction / "after/cameras.json")
        if read_json(output / "after/cameras.json") != transformed_cameras(
            cameras, transform
        ):
            raise ValueError("Aligned camera coordinates disagree with transform")
    return report


class ComparisonInventory:
    def __init__(self, inventory, output):
        self.__dict__.update(inventory.__dict__)
        self.base = inventory
        self.comparison = verify(inventory, output)
        self.after_transform = self.comparison["alignment"]["transform"]
        self.captures = copy.copy(inventory.captures)
        for cap in self.captures:
            self.captures[cap] = {
                **self.captures[cap],
                "objects": copy.deepcopy(self.captures[cap]["objects"]),
            }
        if self.after_transform:
            self.captures["after"]["objects"] = read_json(
                Path(output) / "after/objects.json"
            )["objects"]
            self.captures["after"]["geometry"] = load_arrays(
                Path(output) / "after/geometry.npz"
            )
        identities = read_json(Path(output) / "identities.json")["identities"]
        changes = (
            read_json(Path(output) / "changes.json")["events"]
            if self.comparison["version"] == "comparison-v2"
            else []
        )
        for data in self.captures.values():
            for obj in data["objects"]:
                related = [
                    i
                    for i in identities
                    if obj["id"] in (i["before_id"], i["after_id"])
                ]
                obj["identity_links"] = [
                    {
                        **i,
                        "id": i["shared_id"],
                        "matching_status": i["status"],
                        "status": "proposed"
                        if i["status"] in ("matched", "proposed_visual")
                        else "ambiguous",
                    }
                    for i in related
                    if i["before_id"] and i["after_id"]
                ]
                obj["shared_identity"] = related[0]["shared_id"] if related else None
                event = next(
                    (e for e in changes if e["shared_id"] == obj["shared_identity"]),
                    None,
                )
                if event:
                    obj["change"] = event
                    obj["change_status"] = event["type"]

    def listing(self, capture):
        result = self.base.listing(capture)
        result.pop("available_modes", None)
        result.update(
            objects=self.captures[capture]["objects"], comparison=self.comparison
        )
        return result

    def object(self, capture, identifier):
        return ObjectInventory.object(self, capture, identifier)

    def context_record(self, capture, identifier):
        return self.base.context_record(capture, identifier)
