"""Resumable object inference and conservative CPU inventory publication."""

import contextlib
import os
import time
import traceback
import uuid
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .core import digest, identity, read_json, safe_path, write_json
from .object_association import consolidate
from .object_contracts import (
    CAPTURES,
    VERSION,
    ObjectInventory,
    atomic_move,
    load_arrays,
    reconstruction_binding,
    stable_id,
    validate_config,
)
from .object_geometry import (
    CaptureGeometry,
    MaskStore,
    geometry_quality,
    masked_crop,
    voxel_geometry,
)


@contextlib.contextmanager
def object_lock(root):
    path = root / ".objects.lock"
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(str(os.getpid()))
    except FileExistsError:
        raise RuntimeError(
            "Another object writer owns .objects.lock; confirm it has stopped before removing a stale lock"
        ) from None
    try:
        yield
    finally:
        path.unlink()


def save_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    atomic_move(temporary, path)


def stage(root, capture, kind, signature, producer):
    directory = root / "cache" / capture / kind / identity(signature)
    receipt = directory / "cache.json"
    if receipt.is_file():
        record = read_json(receipt)
        if all(
            safe_path(directory, name).is_file()
            and digest(safe_path(directory, name)) == expected
            for name, expected in record["files"].items()
        ):
            print(f"{capture}: reuse {kind}", flush=True)
            return directory, record["result"]
    directory.mkdir(parents=True, exist_ok=True)
    print(f"{capture}: {kind}", flush=True)
    result = producer(directory)
    files = {
        p.relative_to(directory).as_posix(): digest(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file() and p.name != "cache.json"
    }
    write_json(receipt, {"signature": signature, "files": files, "result": result})
    return directory, result


def mask_iou(a, b):
    return float(np.count_nonzero(a & b) / max(1, np.count_nonzero(a | b)))


def pack_rows(groups):
    return (
        np.concatenate(groups).astype(np.int64) if groups else np.empty(0, np.int64),
        np.r_[0, np.cumsum([len(g) for g in groups])].astype(np.int64),
    )


def process_capture(
    reconstruction, output, capture, config, lock, models, binding, frame_limit=None
):
    geom = CaptureGeometry(reconstruction, capture)
    frames = geom.frames
    chosen = list(range(len(frames)))
    if frame_limit:
        chosen = np.unique(
            np.linspace(0, len(frames) - 1, min(frame_limit, len(frames))).astype(int)
        ).tolist()
    common = {
        "binding": identity(binding),
        "frames": chosen,
        "version": VERSION,
        "pipeline": digest(Path(__file__)),
        "adapter": digest(Path(__file__).with_name("object_models.py")),
        "runtime": models.runtime.get("packages", {}),
        "seed": config["seed"],
        "device": config["device"],
    }
    signatures = {}
    warnings = ["partial_frame_coverage"] if len(chosen) < len(frames) else []
    d = config["detection"]

    def detect(directory):
        models.activate("detector")
        rows = []
        for fi in chosen:
            with Image.open(reconstruction / frames[fi]["image_path"]) as image:
                proposals = models.detect(image.convert("RGB"))
            exceeded = len(proposals) > d["max_detections_per_frame"]
            for k, proposal in enumerate(proposals[: d["max_detections_per_frame"]]):
                box = np.asarray(proposal["box"], float)
                if (
                    box.shape != (4,)
                    or not np.isfinite(box).all()
                    or box[2] <= box[0]
                    or box[3] <= box[1]
                ):
                    raise ValueError("Detector returned an invalid box")
                if not np.isfinite(proposal["score"]):
                    raise ValueError("Detector returned a nonfinite score")
                rows.append(
                    {
                        **proposal,
                        "id": stable_id(
                            capture + "_observation",
                            [frames[fi]["frame_id"], str(k), identity(proposal)],
                        ),
                        "frame_index": fi,
                        "frame_id": frames[fi]["frame_id"],
                        "timestamp_sec": frames[fi]["elapsed_timestamp_sec"],
                    }
                )
            if exceeded:
                rows.append({"budget_warning": True, "frame_index": fi})
        models.close()
        write_json(directory / "detections.json", rows)
        return rows

    signatures["detect"] = {**common, "config": d, "model": lock["models"]["detector"]}
    _, proposals = stage(output, capture, "detect", signatures["detect"], detect)
    if any(p.get("budget_warning") for p in proposals):
        warnings.append("detection_budget_exhausted")
    proposals = [p for p in proposals if not p.get("budget_warning")]

    def segment(directory):
        if proposals:
            models.activate("segmenter")
        rows, accepted_masks, duplicates = [], {}, []
        previous_frame = None
        for proposal in proposals:
            fi = proposal["frame_index"]
            if fi != previous_frame:
                accepted_masks.clear()
                previous_frame = fi
            with Image.open(reconstruction / frames[fi]["image_path"]) as image:
                mask, score = models.segment(image.convert("RGB"), proposal["box"])
            mask = np.asarray(mask, bool)
            if mask.shape != (
                frames[fi]["height"],
                frames[fi]["width"],
            ) or not np.isfinite(score):
                raise ValueError(
                    "Segmenter returned inconsistent mask dimensions or score"
                )
            same = [
                r
                for r in rows
                if r["frame_index"] == fi and r["label"] == proposal["label"]
            ]
            duplicate = next(
                (
                    r
                    for r in same
                    if mask_iou(mask, accepted_masks[r["id"]])
                    >= d["duplicate_mask_iou"]
                ),
                None,
            )
            if duplicate:
                duplicates.append(
                    {"suppressed": proposal["id"], "retained": duplicate["id"]}
                )
                continue
            conflicts = []
            for previous in (r for r in rows if r["frame_index"] == fi):
                other = accepted_masks[previous["id"]]
                shared = np.count_nonzero(mask & other) / max(
                    1, min(np.count_nonzero(mask), np.count_nonzero(other))
                )
                if shared >= config["segmentation"]["overlap_warning_fraction"]:
                    conflicts.append(previous["id"])
                    previous["mask_conflicts"].append(proposal["id"])
            accepted_masks[proposal["id"]] = mask
            mask_path = proposal["id"] + ".png"
            Image.fromarray(mask.astype(np.uint8) * 255).save(directory / mask_path)
            rows.append(
                {
                    **proposal,
                    "mask_score": float(score),
                    "mask_file": mask_path,
                    "mask_area": int(mask.sum()),
                    "mask_conflicts": conflicts,
                }
            )
        models.close()
        write_json(
            directory / "segmentation.json",
            {"observations": rows, "duplicates": duplicates},
        )
        return {"observations": rows, "duplicates": duplicates}

    signatures["segment"] = {
        "input": identity(signatures["detect"]),
        "model": lock["models"]["segmenter"],
        "config": config["segmentation"],
    }
    segment_dir, segmented = stage(
        output, capture, "segment", signatures["segment"], segment
    )
    observations = segmented["observations"]
    masks = MaskStore(segment_dir, observations)

    def appearance(directory):
        descriptors = []
        if observations:
            models.activate("appearance")
        for observation in observations:
            fi = observation["frame_index"]
            with Image.open(reconstruction / frames[fi]["image_path"]) as image:
                crop = masked_crop(
                    image,
                    masks[observation["id"]],
                    config["appearance"]["background_rgb"],
                )
                descriptor = np.asarray(models.embed(crop), np.float32)
            if (
                descriptor.ndim != 1
                or not len(descriptor)
                or not np.isfinite(descriptor).all()
                or np.linalg.norm(descriptor) < 1e-8
            ):
                raise ValueError("Appearance model returned an invalid descriptor")
            descriptors.append(descriptor)
        models.close()
        values = np.stack(descriptors) if descriptors else np.empty((0, 0), np.float32)
        save_npz(directory / "descriptors.npz", descriptors=values)
        return {"count": len(descriptors)}

    signatures["appearance"] = {
        "input": identity(signatures["segment"]),
        "config": config["appearance"],
        "model": lock["models"]["appearance"],
    }
    appearance_dir, _ = stage(
        output, capture, "appearance", signatures["appearance"], appearance
    )
    descriptors = load_arrays(appearance_dir / "descriptors.npz")["descriptors"]

    def lift(directory):
        groups, quality = [], []
        for observation in observations:
            rows = geom.mask_rows(observation["frame_index"], masks[observation["id"]])
            points = geom.points(rows)
            problems = geometry_quality(
                points, observation["mask_area"], config["geometry"], geom.voxel
            )
            groups.append(rows)
            quality.append(problems)
        indices, ranges = pack_rows(groups)
        save_npz(directory / "lifting.npz", indices=indices, offsets=ranges)
        write_json(directory / "quality.json", quality)
        return quality

    signatures["lift"] = {
        "input": identity(signatures["segment"]),
        "geometry": config["geometry"],
        "code": digest(Path(__file__).with_name("object_geometry.py")),
    }
    lift_dir, qualities = stage(output, capture, "lift", signatures["lift"], lift)
    lifted = load_arrays(lift_dir / "lifting.npz")
    arrays = []
    for i, observation in enumerate(observations):
        rows = lifted["indices"][slice(*lifted["offsets"][i : i + 2])]
        sample_rows = (
            rows[
                np.linspace(
                    0,
                    len(rows) - 1,
                    min(len(rows), config["association"]["max_pair_points"] * 4),
                ).astype(int)
            ]
            if len(rows)
            else rows
        )
        arrays.append({"rows": rows, "points": geom.points(sample_rows)})
        observation.update(
            status="unresolved" if qualities[i] else "usable",
            quality=qualities[i],
            depth_support=len(rows),
            mask_path=f"{capture}/masks/{observation['id']}.png",
        )
    depth = load_arrays(reconstruction / capture / "depth.npz")["depth"]
    valid = load_arrays(reconstruction / capture / "valid.npz")["valid"]
    voxel = geom.voxel * config["geometry"]["voxel_multiplier"]
    groups, decisions = consolidate(
        observations,
        arrays,
        descriptors,
        geom,
        masks,
        depth,
        valid,
        config["association"],
        voxel,
    )

    # Stage into a sibling below the output root, then publish one complete capture.
    staging = output / (".publish_" + capture + "_" + uuid.uuid4().hex)
    staging.mkdir()
    (staging / "masks").mkdir()
    (staging / "previews").mkdir()
    for observation in observations:
        mask = masks[observation["id"]]
        Image.fromarray(mask.astype(np.uint8) * 255).save(
            staging / "masks" / (observation["id"] + ".png")
        )
        with Image.open(
            reconstruction / frames[observation["frame_index"]]["image_path"]
        ) as image:
            image = image.convert("RGB")
            tinted = np.array(image)
            tinted[mask] = (
                tinted[mask].astype(float) * 0.5 + np.array([0, 220, 255]) * 0.5
            ).astype(np.uint8)
            preview = Image.fromarray(tinted)
            preview.thumbnail((640, 480))
            ImageDraw.Draw(preview).text(
                (6, 6),
                observation["label"],
                fill="white",
                stroke_width=1,
                stroke_fill="black",
            )
            preview.save(
                staging / "previews" / (observation["id"] + ".jpg"), quality=85
            )
    objects, object_members, geometries = [], [], []
    for group in groups:
        evidence = [observations[i]["id"] for i in group]
        rows = np.unique(np.concatenate([arrays[i]["rows"] for i in group]))
        geometry = voxel_geometry(geom, rows, voxel)
        if len(geometry["points"]) < 3:
            for i in group:
                observations[i].update(
                    status="unresolved",
                    quality=observations[i]["quality"]
                    + ["insufficient_consolidated_geometry"],
                )
            continue
        labels = sorted(
            {
                observations[i]["label"]
                for i in group
                if observations[i]["label"] != "unknown_object"
            }
        )
        center = geometry["points"].mean(0)
        view_centers = [arrays[i]["points"].mean(0) for i in group]
        alternatives = sorted(
            {
                x
                for decision in decisions
                if decision["relation"] in ("ambiguous", "group_conflict")
                and (decision["a"] in evidence or decision["b"] in evidence)
                for x in (decision["a"], decision["b"])
                if x not in evidence
            }
        )
        objects.append(
            {
                "id": stable_id(capture + "_obj", evidence),
                "capture": capture,
                "label": labels[0] if len(labels) == 1 else "unknown_object",
                "label_alternatives": labels,
                "coordinate_frame": binding["coordinate_frame"],
                "units": "relative",
                "observation_ids": evidence,
                "frame_ids": sorted({observations[i]["frame_id"] for i in group}),
                "centroid": center.tolist(),
                "bounds": [
                    geometry["points"].min(0).tolist(),
                    geometry["points"].max(0).tolist(),
                ],
                "point_count": len(geometry["points"]),
                "supporting_frames": len(
                    {observations[i]["frame_index"] for i in group}
                ),
                "quality": ["observed_surface_center", "uncalibrated_depth"]
                + (["possible_duplicate_fragments"] if alternatives else [])
                + (
                    ["overlapping_instance_masks"]
                    if any(observations[i]["mask_conflicts"] for i in group)
                    else []
                )
                + (["single_view"] if len(group) == 1 else []),
                "possible_observation_alternatives": alternatives,
                "view_centroid_max_deviation": float(
                    max(np.linalg.norm(v - center) for v in view_centers)
                ),
                "estimator": {
                    "method": "mean_of_unique_voxel_representatives",
                    "voxel_size": voxel,
                    "filtering": "saved_reconstruction_validity; no inferred hidden surfaces",
                },
            }
        )
        object_members.append(rows)
        geometries.append(geometry)
    oi, oo = pack_rows([x["rows"] for x in arrays])
    mi, mo = pack_rows(object_members)
    save_npz(
        staging / "membership.npz",
        observation_indices=oi,
        observation_offsets=oo,
        object_indices=mi,
        object_offsets=mo,
        observation_cloud_indices=geom.observations["cloud_point_index"][oi],
        object_cloud_indices=geom.observations["cloud_point_index"][mi],
    )
    values = {}
    for key, shape, dtype in (
        ("points", (0, 3), np.float32),
        ("colors", (0, 3), np.uint8),
        ("representative_observation", (0,), np.int64),
        ("first_frame", (0,), np.int64),
    ):
        values[key] = (
            np.concatenate([g[key] for g in geometries])
            if geometries
            else np.empty(shape, dtype)
        )
    save_npz(
        staging / "geometry.npz",
        offsets=np.r_[0, np.cumsum([len(g["points"]) for g in geometries])].astype(
            np.int64
        ),
        **values,
    )
    save_npz(staging / "descriptors.npz", descriptors=descriptors)
    write_json(
        staging / "observations.json",
        {"schema_version": "1.0", "observations": observations},
    )
    write_json(staging / "objects.json", {"schema_version": "1.0", "objects": objects})
    write_json(
        staging / "diagnostics.json",
        {
            "warnings": warnings,
            "frames_processed": chosen,
            "frame_count": len(frames),
            "objects": len(objects),
            "unresolved_observations": sum(
                o["status"] != "usable" for o in observations
            ),
            "associations": decisions,
            "duplicate_proposals": segmented["duplicates"],
            "real_review": "pending",
            "limitations": [
                "Vocabulary-limited detection",
                "Reconstruction distortion can fragment objects",
                "Centroids describe observed surfaces; no physical center guarantee",
            ],
        },
    )
    final = output / capture
    if final.exists():
        history = output / "history"
        history.mkdir(exist_ok=True)
        # Both resolved move targets stay inside the explicitly selected object output.
        assert final.resolve().is_relative_to(
            output
        ) and history.resolve().is_relative_to(output)
        atomic_move(final, history / (capture + "_" + uuid.uuid4().hex))
    atomic_move(staging, final)
    return {
        "status": "completed_with_warnings"
        if warnings or any(o["status"] != "usable" for o in observations)
        else "completed",
        "binding": binding,
        "objects": len(objects),
        "observations": len(observations),
        "real_review": "pending",
        "artifacts": {
            p.relative_to(output).as_posix(): digest(p)
            for p in sorted(final.rglob("*"))
            if p.is_file()
        },
    }


def run(
    reconstruction,
    output,
    config,
    lock,
    *,
    resume=False,
    captures=CAPTURES,
    frame_limit=None,
    models=None,
):
    from .object_models import ObjectModels

    reconstruction, output = Path(reconstruction).resolve(), Path(output).resolve()
    config = validate_config(config)
    if output.is_relative_to(reconstruction) or reconstruction.is_relative_to(output):
        raise ValueError(
            "Object output and reconstruction must be separate sibling trees"
        )
    if (
        not captures
        or any(c not in CAPTURES for c in captures)
        or (frame_limit is not None and frame_limit < 1)
    ):
        raise ValueError("Invalid capture selection or frame limit")
    output.mkdir(parents=True, exist_ok=True)
    existing = output / "manifest.json"
    if existing.exists() and not resume:
        raise ValueError("Object output exists; use --resume or a new directory")
    algorithm = identity(
        {
            p.name: digest(p)
            for p in Path(__file__).parent.glob("object_*.py")
            if p.name != "object_inspection.py"
        }
    )
    requested = {
        "config_identity": identity(config),
        "model_identity": identity(lock),
        "algorithm_identity": algorithm,
        "frame_limit": frame_limit,
    }
    manifest = (
        read_json(existing)
        if existing.exists()
        else {"version": VERSION, "captures": {}}
    )
    changed = any(manifest.get(k) != v for k, v in requested.items())
    if changed:
        manifest["captures"] = {c: {"status": "pending"} for c in CAPTURES}
    manifest.update(requested)
    manifest["synthetic"] = bool(
        read_json(reconstruction / "run_manifest.json").get("synthetic")
    )
    # Validate completed results before acquiring a writer lock (loader rejects active locks).
    reusable = {}
    if resume and not changed:
        for capture in captures:
            record = manifest["captures"].get(capture, {})
            if record.get("status") in ("completed", "completed_with_warnings"):
                try:
                    inventory = ObjectInventory(
                        reconstruction, output, captures=(capture,)
                    )
                    reusable[capture] = capture in inventory.captures
                except (ValueError, KeyError, OSError):
                    reusable[capture] = False
    with object_lock(output):
        write_json(output / "config.json", config)
        write_json(output / "models.lock.json", lock)
        write_json(existing, manifest)
        for capture in captures:
            if reusable.get(capture):
                print(f"{capture}: reuse completed inventory", flush=True)
                continue
            started = time.perf_counter()
            manifest["captures"][capture] = {"status": "running"}
            write_json(existing, manifest)
            adapter = models
            try:
                binding = reconstruction_binding(reconstruction, capture, verify=True)
                adapter = adapter or ObjectModels(config, lock)
                record = process_capture(
                    reconstruction,
                    output,
                    capture,
                    config,
                    lock,
                    adapter,
                    binding,
                    frame_limit,
                )
                manifest["captures"][capture] = record
            except Exception as exc:  # noqa: BLE001 - stage boundary retains diagnostics and the other capture
                manifest["captures"][capture] = {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "real_review": "pending",
                }
                write_json(
                    output / "logs" / f"{capture}_failure.json",
                    {"traceback": traceback.format_exc()},
                )
                print(
                    f"{capture}: {manifest['captures'][capture]['error']}", flush=True
                )
            finally:
                if adapter is not None:
                    adapter.close()
                log = {
                    "capture": capture,
                    "seconds": time.perf_counter() - started,
                    "runtime": adapter.runtime if adapter is not None else {},
                    "status": manifest["captures"][capture]["status"],
                }
                write_json(output / "logs" / f"{capture}.json", log)
                manifest["status"] = (
                    "failed"
                    if any(
                        r["status"] == "failed" for r in manifest["captures"].values()
                    )
                    else "partial"
                )
                if all(
                    manifest["captures"].get(c, {}).get("status")
                    in ("completed", "completed_with_warnings")
                    for c in CAPTURES
                ):
                    manifest["status"] = (
                        "completed_with_warnings"
                        if any(
                            r["status"] == "completed_with_warnings"
                            for r in manifest["captures"].values()
                        )
                        else "completed"
                    )
                write_json(existing, manifest)
    return manifest
