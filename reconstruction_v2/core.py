"""CPU preparation, geometry, provenance, diagnostics and artifact validation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import VERSION


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def safe_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError(f"Artifact escapes run root: {relative}")
    return path


def environment():
    packages = {}
    for name in ("numpy", "Pillow", "scipy", "viser", "torch", "torchvision", "huggingface-hub", "safetensors", "einops"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"created_at_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
            "platform": platform.platform(), "packages": packages, "pipeline_version": VERSION}


def code_identity():
    return identity({name: digest(Path(__file__).parent / name) for name in ("__init__.py", "core.py", "inference.py")})


def validate_config(config):
    p = config["preprocessing"]
    if p["mode"] not in ("crop", "pad") or p["target_size"] != 518 or p["patch_size"] != 14:
        raise ValueError("VGGT preparation requires the supported 518/14 crop or pad profile")
    c = config["confidence"]
    if not 0 <= c["percentile"] <= 100:
        raise ValueError("Confidence percentile must be between 0 and 100")
    if c["threshold"] is not None and not np.isfinite(c["threshold"]):
        raise ValueError("Confidence threshold must be finite")
    v = config["voxel"]
    if v["size_relative"] is not None and (not np.isfinite(v["size_relative"]) or v["size_relative"] <= 0):
        raise ValueError("Voxel size must be finite and positive")
    if not np.isfinite(v["median_depth_multiplier"]) or v["median_depth_multiplier"] <= 0:
        raise ValueError("Voxel multiplier must be finite and positive")


def preprocess(image, rotation=0, mode="crop"):
    """Match VGGT's PIL resize/14-rounding; record pixel-center maps exactly."""
    image = image.convert("RGB")
    width, height = image.size
    rotations = {
        0: np.eye(3), 90: np.array([[0, 1, 0], [-1, 0, width - 1], [0, 0, 1]]),
        180: np.array([[-1, 0, width - 1], [0, -1, height - 1], [0, 0, 1]]),
        270: np.array([[0, -1, height - 1], [1, 0, 0], [0, 0, 1]]),
    }
    if rotation not in rotations or mode not in ("crop", "pad"):
        raise ValueError("Unsupported rotation or preprocessing mode")
    if rotation:
        image = image.transpose({90: Image.Transpose.ROTATE_90, 180: Image.Transpose.ROTATE_180, 270: Image.Transpose.ROTATE_270}[rotation])
    oriented_width, oriented_height = image.size
    if mode == "pad" and oriented_height > oriented_width:
        new_height = 518
        new_width = round(oriented_width * (518 / oriented_height) / 14) * 14
    else:
        new_width = 518
        new_height = round(oriented_height * (518 / oriented_width) / 14) * 14
    if min(new_width, new_height) < 14:
        raise ValueError("Image aspect ratio produces a zero-sized model dimension")
    image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
    sx, sy = new_width / oriented_width, new_height / oriented_height
    matrix = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]]) @ rotations[rotation]
    crop_top = (new_height - 518) // 2 if mode == "crop" and new_height > 518 else 0
    final_height = min(new_height, 518) if mode == "crop" else new_height
    if crop_top:
        image = image.crop((0, crop_top, new_width, crop_top + 518))
    padding = [0, 0, 0, 0]
    valid = np.ones((final_height, new_width), bool)
    if mode == "pad":
        left, top = (518 - new_width) // 2, (518 - new_height) // 2
        padding = [left, top, 518 - new_width - left, 518 - new_height - top]
        canvas = Image.new("RGB", (518, 518), "white")
        canvas.paste(image, (left, top))
        image = canvas
        valid = np.zeros((518, 518), bool)
        valid[top:top + new_height, left:left + new_width] = True
    offset = np.array([[1, 0, padding[0]], [0, 1, padding[1] - crop_top], [0, 0, 1]])
    matrix = offset @ matrix
    record = {"original_width": width, "original_height": height, "rotation_counterclockwise_degrees": rotation,
              "resized_width": new_width, "resized_height": new_height, "resize_scale_xy": [sx, sy],
              "crop_xywh_resized": [0, crop_top, new_width, final_height], "padding_ltrb": padding,
              "batch_padding_ltrb": [0, 0, 0, 0], "model_width": image.width, "model_height": image.height,
              "original_to_model": matrix.tolist(), "model_to_original": np.linalg.inv(matrix).tolist(),
              "pixel_convention": "integer pixel centers; resize maps (x+0.5)*scale-0.5", "mode": mode}
    return image, valid, record


def artifacts(root, capture):
    return {p.relative_to(root).as_posix(): digest(p) for p in sorted((root / capture).rglob("*")) if p.is_file()}


def check_artifacts(root, record):
    if not record.get("artifacts"):
        raise ValueError("No capture artifact hashes recorded")
    for path, expected in record["artifacts"].items():
        file = safe_path(root, path)
        if not file.is_file() or digest(file) != expected:
            raise ValueError(f"Artifact missing or changed: {path}")


def prepare(frame_run, output, config, lock, *, selections=None, kind="full"):
    frame_run, output = Path(frame_run).resolve(), Path(output).resolve()
    validate_config(config)
    selected_manifest = read_json(frame_run / "frame_manifest.json")
    receipt = read_json(frame_run / "artifact_verification.json")
    if receipt["status"] != "passed" or receipt["frame_manifest_sha256"] != digest(frame_run / "frame_manifest.json"):
        raise ValueError("Frame-selection verification no longer matches its manifest")
    if output.exists():
        raise ValueError("Preparation requires a new output directory")
    expected = {a["path"]: a["sha256"] for a in selected_manifest["artifacts"]}
    for capture in ("before", "after"):
        frames_path = frame_run / capture / "frames.json"
        if digest(frames_path) != expected[f"{capture}/frames.json"]:
            raise ValueError("Selected-frame manifest changed")
        for frame in read_json(frames_path)["frames"]:
            if digest(safe_path(frame_run, frame["image_path"])) != frame["image_sha256"]:
                raise ValueError(f"Selected image changed: {frame['frame_id']}")
    output.mkdir(parents=True)
    write_json(output / "config.json", config)
    write_json(output / "models.lock.json", lock)
    write_json(output / "preparation_environment.json", environment())
    manifest = {"schema_version": "1.0", "pipeline_version": VERSION, "kind": kind, "status": "prepared",
                "created_at_utc": datetime.now(timezone.utc).isoformat(), "frame_selection_identity": digest(frame_run / "frame_manifest.json"),
                "frame_selection_warnings": [{"capture": c["capture_id"], "status": c["status"], "weak_pairs": c["weak_overlap_pairs"],
                                               "uncovered_time_bins": c["uncovered_time_bins"]} for c in selected_manifest["captures"]],
                "config_identity": identity(config), "model_identity": identity(lock), "captures": {},
                "units": "relative", "scale_status": "unvalidated", "alignment": "not_performed",
                "source_videos_included": False, "synthetic": False}
    capture_sources = {c["capture_id"]: c for c in selected_manifest["captures"]}
    for name in ("before", "after"):
        selected = read_json(frame_run / name / "frames.json")["frames"]
        if selections is not None:
            selected = [selected[i] for i in selections[name]]
        if len(selected) < 2:
            raise ValueError("Each capture requires at least two frames")
        directory = output / name
        (directory / "frames").mkdir(parents=True)
        (directory / "model_images").mkdir()
        rows, masks, images, transforms = [], [], [], []
        for original in selected:
            frame = dict(original)
            original_path = f"{name}/frames/{frame['frame_id']}.png"
            shutil.copyfile(safe_path(frame_run, frame["image_path"]), output / original_path)
            with Image.open(output / original_path) as image:
                model, valid, transform = preprocess(image, int(frame["orientation_counterclockwise_degrees"]), config["preprocessing"]["mode"])
            rows.append({**frame, "image_path": original_path, "model_image_path": f"{name}/model_images/{frame['frame_id']}.png"})
            images.append(model)
            masks.append(valid)
            transforms.append({"frame_id": frame["frame_id"], **transform})
        width, height = max(i.width for i in images), max(i.height for i in images)
        for row, image, mask, transform in zip(rows, images, masks, transforms):
            left, top = (width - image.width) // 2, (height - image.height) // 2
            batch = [left, top, width - image.width - left, height - image.height - top]
            if any(batch):
                canvas = Image.new("RGB", (width, height), "white")
                canvas.paste(image, (left, top))
                image = canvas
            new_mask = np.zeros((height, width), bool)
            new_mask[top:top + mask.shape[0], left:left + mask.shape[1]] = mask
            masks[rows.index(row)] = new_mask
            matrix = np.array([[1, 0, left], [0, 1, top], [0, 0, 1]]) @ np.array(transform["original_to_model"])
            transform.update({"batch_padding_ltrb": batch, "model_width": width, "model_height": height,
                              "original_to_model": matrix.tolist(), "model_to_original": np.linalg.inv(matrix).tolist()})
            image.save(output / row["model_image_path"])
            row["model_image_sha256"] = digest(output / row["model_image_path"])
        frame_ids = np.array([r["frame_id"] for r in rows])
        np.savez_compressed(directory / "padding_valid.npz", frame_ids=frame_ids, valid=np.stack(masks))
        write_json(directory / "frames.json", {"capture_id": name, "frames": rows})
        write_json(directory / "image_transforms.json", transforms)
        write_json(directory / "source.json", {"capture_id": name, "source_video_path": capture_sources[name]["source_path"],
                                              "source_video_sha256": capture_sources[name]["source_sha256"],
                                              "evidence": "Selected PNG pixels independently verified against original video before packaging"})
        hashes = artifacts(output, name)
        manifest["captures"][name] = {"status": "prepared", "frame_ids": frame_ids.tolist(), "frame_count": len(rows),
                                     "model_shape_hw": [height, width], "coordinate_frame": f"{name}_world" if kind == "full" else f"{name}_{kind}_world",
                                     "input_identity": identity({"frames": rows, "transforms": transforms}),
                                     "source_video_sha256": capture_sources[name]["source_sha256"],
                                     "artifacts": hashes, "prepared_artifacts": dict(hashes)}
    write_json(output / "run_manifest.json", manifest)
    return manifest


def validate_pose(pose):
    p = np.asarray(pose, dtype=np.float64)
    if p.shape != (4, 4) or not np.isfinite(p).all() or not np.allclose(p[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("Invalid finite homogeneous camera pose")
    if not np.allclose(p[:3, :3].T @ p[:3, :3], np.eye(3), atol=2e-4) or not np.isclose(np.linalg.det(p[:3, :3]), 1, atol=2e-4):
        raise ValueError("Camera rotation is not orthonormal/right handed")
    return p


def validate_intrinsics(k):
    k = np.asarray(k, dtype=np.float64)
    if k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1]) or abs(np.linalg.det(k)) < 1e-12:
        raise ValueError("Invalid camera intrinsics")
    return k


def unproject(depth, k, camera_to_world):
    k, pose = validate_intrinsics(k), validate_pose(camera_to_world)
    v, u = np.indices(depth.shape)
    camera = np.stack((u, v, np.ones_like(u)), -1) @ np.linalg.inv(k).T * depth[..., None]
    return camera @ pose[:3, :3].T + pose[:3, 3]


def project(points, k, camera_to_world):
    k, inverse = validate_intrinsics(k), np.linalg.inv(validate_pose(camera_to_world))
    camera = np.asarray(points, dtype=np.float64) @ inverse[:3, :3].T + inverse[:3, 3]
    homogeneous = camera @ k.T
    pixels = np.full((len(points), 2), np.nan)
    front = np.isfinite(camera).all(1) & (camera[:, 2] > 0)
    pixels[front] = homogeneous[front, :2] / homogeneous[front, 2:3]
    return pixels, camera[:, 2]


def reduce_voxels(points, confidence, size):
    if size <= 0 or not np.isfinite(size):
        raise ValueError("Invalid voxel size")
    coordinates = np.floor(points.astype(np.float64) / size)
    if np.any(np.abs(coordinates) >= 2 ** 63):
        raise ValueError("Voxel index overflow")
    voxels = coordinates.astype(np.int64)
    order = np.lexsort((np.arange(len(points)), -confidence, voxels[:, 2], voxels[:, 1], voxels[:, 0]))
    first = np.ones(len(points), bool)
    first[1:] = np.any(voxels[order[1:]] != voxels[order[:-1]], axis=1)
    representatives = order[first]
    membership = np.empty(len(points), np.int64)
    membership[order] = np.cumsum(first) - 1
    return representatives, membership


PLY_DTYPE = np.dtype([("xyz", "<f4", (3,)), ("rgb", "u1", (3,))])


def write_ply(path, points, colors):
    header = ("ply\nformat binary_little_endian 1.0\n" + f"element vertex {len(points)}\n" +
              "property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    values = np.empty(len(points), PLY_DTYPE)
    values["xyz"], values["rgb"] = points, colors
    with Path(path).open("wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(values.tobytes())


def load_ply(path):
    with Path(path).open("rb") as handle:
        header = []
        while True:
            line = handle.readline()
            if not line or len(header) > 32:
                raise ValueError("Truncated/invalid PLY header")
            header.append(line.decode("ascii").strip())
            if header[-1] == "end_header":
                break
        expected = ["property float x", "property float y", "property float z", "property uchar red", "property uchar green", "property uchar blue"]
        if "format binary_little_endian 1.0" not in header or [line for line in header if line.startswith("property")] != expected:
            raise ValueError("Unsupported PLY layout")
        count = int(next(line for line in header if line.startswith("element vertex ")).split()[-1])
        data = handle.read()
        if count <= 0 or len(data) != count * PLY_DTYPE.itemsize:
            raise ValueError("PLY vertex count mismatch")
        result = np.frombuffer(data, PLY_DTYPE)
        return result["xyz"].copy(), result["rgb"].copy()


def colorize(values, valid, low, high, confidence=False):
    # A sequential palette and explicit numeric range; invalid pixels are gray.
    knots = np.array([[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]], float)
    normalized = np.zeros(values.shape, float)
    normalized[valid] = np.clip((values[valid] - low) / max(high - low, 1e-12), 0, 1)
    x = np.linspace(0, 1, len(knots))
    colors = np.stack([np.interp(normalized, x, knots[:, channel]) for channel in range(3)], -1).astype(np.uint8)
    colors[~valid] = [128, 128, 128]
    return Image.fromarray(colors)


def previews(root, capture, frames, depth, confidence, valid):
    directory = root / capture / "previews"
    directory.mkdir(exist_ok=True)
    eligible = valid & np.isfinite(confidence)
    depth_range = np.percentile(depth[eligible], [2, 98]).tolist()
    confidence_range = np.percentile(confidence[eligible], [2, 98]).tolist()
    write_json(directory / "scales.json", {"depth_range_relative_z": depth_range, "confidence_range_uncalibrated": confidence_range,
                                         "range_scope": "this capture", "display_percentiles": [2, 98],
                                         "invalid_color": "gray", "raw_arrays_unchanged": True})
    font = ImageFont.load_default(size=13)
    rows, cards = [], []
    for index, frame in enumerate(frames):
        with Image.open(root / frame["model_image_path"]) as image:
            rgb = image.convert("RGB")
        depth_image = colorize(depth[index], valid[index], *depth_range)
        confidence_image = colorize(confidence[index], valid[index] & np.isfinite(confidence[index]), *confidence_range)
        depth_image.save(directory / f"{frame['frame_id']}_depth.png")
        confidence_image.save(directory / f"{frame['frame_id']}_confidence.png")
        panel = Image.new("RGB", (rgb.width * 3, rgb.height + 86), "white")
        draw = ImageDraw.Draw(panel)
        for column, image in enumerate((rgb, depth_image, confidence_image)):
            panel.paste(image, (column * rgb.width, 28))
        draw.text((8, 6), f"{capture}: frame {frame['decoded_frame_index']} | {frame['elapsed_timestamp_sec']:.3f}s", fill="black", font=font)
        y = rgb.height + 32
        labels = ("RGB model input", f"Z depth {depth_range[0]:.3g} - {depth_range[1]:.3g} (relative)",
                  f"Confidence {confidence_range[0]:.3g} - {confidence_range[1]:.3g} (uncalibrated)")
        for column, label in enumerate(labels):
            draw.text((column * rgb.width + 8, y), label, fill="black", font=font)
            if column:
                ramp = np.linspace(0, 1, rgb.width - 16)[None].repeat(12, axis=0)
                panel.paste(colorize(ramp, np.ones_like(ramp, bool), 0, 1), (column * rgb.width + 8, y + 24))
        panel.save(directory / f"{frame['frame_id']}_panel.png")
        small = panel.copy()
        small.thumbnail((900, 250))
        rows.append(small)
        cards.append(f'<option value="{frame["frame_id"]}_panel.png">Frame {frame["decoded_frame_index"]} — {frame["elapsed_timestamp_sec"]:.3f}s</option>')
    sheet = Image.new("RGB", (900, sum(r.height + 10 for r in rows)), "#ececec")
    top = 0
    for row in rows:
        sheet.paste(row, (0, top))
        top += row.height + 10
    sheet.save(root / capture / "depth_contact_sheet.png")
    viewer = f'''<!doctype html><html><head><meta charset="utf-8"><title>{capture} depth previews</title><style>body{{font:16px system-ui;margin:20px}}img{{max-width:100%}}</style></head><body><h1>{capture.title()} RGB / depth / confidence</h1><p>Relative Z depth. Scales are consistent within this capture, independent of the other capture. Gray = invalid. Confidence is not a probability.</p><select id="frame" onchange="document.getElementById('panel').src=this.value">{''.join(cards)}</select><p><img id="panel" src="{frames[0]['frame_id']}_panel.png" alt="RGB depth confidence"></p></body></html>'''
    (directory / "depth_preview.html").write_text(viewer, encoding="utf-8")


def cross_view_diagnostics(depth, valid, cameras, settings):
    """Held-out-view depth disagreement is diagnostic, never a calibration certificate."""
    pairs = list(zip(range(len(cameras) - 1), range(1, len(cameras))))
    pairs = [pairs[i] for i in np.unique(np.linspace(0, len(pairs) - 1, min(len(pairs), settings["cross_view_pairs"])).astype(int))]
    results = []
    for a, b in pairs:
        indices = np.flatnonzero(valid[a])
        indices = indices[np.unique(np.linspace(0, len(indices) - 1, min(len(indices), settings["cross_view_samples_per_pair"])).astype(int))] if len(indices) else indices
        points = unproject(np.where(valid[a], depth[a], 0), np.array(cameras[a]["intrinsics"]), np.array(cameras[a]["camera_to_world"])).reshape(-1, 3)[indices]
        pixels, z = project(points, np.array(cameras[b]["intrinsics"]), np.array(cameras[b]["camera_to_world"]))
        finite = (np.isfinite(pixels).all(1) & (z > 0) & (pixels[:, 0] >= -0.5) & (pixels[:, 0] < depth.shape[2] - 0.5)
                  & (pixels[:, 1] >= -0.5) & (pixels[:, 1] < depth.shape[1] - 0.5))
        u, v = np.zeros(len(points), int), np.zeros(len(points), int)
        u[finite], v[finite] = np.rint(pixels[finite, 0]).astype(int), np.rint(pixels[finite, 1]).astype(int)
        inside = finite & (z > 0) & (u >= 0) & (u < depth.shape[2]) & (v >= 0) & (v < depth.shape[1])
        selected = np.flatnonzero(inside)
        selected = selected[valid[b, v[selected], u[selected]]]
        error = np.abs(depth[b, v[selected], u[selected]] - z[selected]) / np.maximum(z[selected], 1e-12)
        results.append({"from_frame_id": cameras[a]["frame_id"], "to_frame_id": cameras[b]["frame_id"], "comparable_samples": len(error),
                        "median_relative_depth_disagreement": float(np.median(error)) if len(error) else None,
                        "p90_relative_depth_disagreement": float(np.percentile(error, 90)) if len(error) else None})
    return {"pairs": results, "interpretation": "Includes occlusion, view coverage, nearest-pixel sampling and prediction errors; no independent ground truth"}


def export_predictions(root, capture, depth, confidence, intrinsics, camera_to_world, runtime):
    root = Path(root)
    manifest = read_json(root / "run_manifest.json")
    config = read_json(root / "config.json")
    validate_config(config)
    frames = read_json(root / capture / "frames.json")["frames"]
    transforms = read_json(root / capture / "image_transforms.json")
    n, height, width = len(frames), *manifest["captures"][capture]["model_shape_hw"]
    depth, confidence = np.asarray(depth, np.float32), np.asarray(confidence, np.float32)
    if depth.shape != (n, height, width) or confidence.shape != depth.shape or np.shape(intrinsics) != (n, 3, 3) or np.shape(camera_to_world) != (n, 4, 4):
        raise ValueError("Predictions do not match ordered selected frames/model dimensions")
    frame_ids = np.array([f["frame_id"] for f in frames])
    with np.load(root / capture / "padding_valid.npz", allow_pickle=False) as data:
        if not np.array_equal(data["frame_ids"], frame_ids):
            raise ValueError("Padding-mask frame order differs")
        padding = data["valid"]
    eligible = padding & np.isfinite(depth) & (depth > 0) & np.isfinite(confidence)
    if not eligible.any():
        raise ValueError("No usable depth observations")
    c = config["confidence"]
    threshold = None if not c["enabled"] else float(c["threshold"] if c["threshold"] is not None else np.percentile(confidence[eligible], c["percentile"]))
    valid = eligible if threshold is None else eligible & (confidence >= threshold)
    points, colors, scores, frame_indices, pixel_indices, original_pixels = [], [], [], [], [], []
    cameras = []
    for i, (frame, transform) in enumerate(zip(frames, transforms)):
        k, pose = validate_intrinsics(intrinsics[i]), validate_pose(camera_to_world[i])
        with Image.open(root / frame["model_image_path"]) as image:
            rgb = np.array(image.convert("RGB"))
        flattened = np.flatnonzero(valid[i])
        xyz = unproject(np.where(valid[i], depth[i], 0), k, pose).reshape(-1, 3)[flattened]
        finite = np.isfinite(xyz).all(1) & (np.abs(xyz) <= np.finfo(np.float32).max).all(1)
        flattened, xyz = flattened[finite], xyz[finite]
        valid[i] = False
        valid[i].ravel()[flattened] = True
        uv = np.column_stack((flattened % width, flattened // width, np.ones(len(flattened))))
        original = uv @ np.array(transform["model_to_original"]).T
        points.append(xyz.astype(np.float32)); colors.append(rgb.reshape(-1, 3)[flattened])
        scores.append(confidence[i].ravel()[flattened]); frame_indices.append(np.full(len(flattened), i, np.uint32))
        pixel_indices.append(flattened.astype(np.uint32)); original_pixels.append(original[:, :2].astype(np.float32))
        cameras.append({"frame_id": frame["frame_id"], "capture_id": capture, "intrinsics": k.tolist(),
                        "model_width": width, "model_height": height, "camera_to_world": pose.tolist(), "world_to_camera": np.linalg.inv(pose).tolist(),
                        "coordinate_frame": manifest["captures"][capture]["coordinate_frame"], "camera_axes": "OpenCV: +X right, +Y down, +Z forward",
                        "depth_convention": "optical_axis_z", "pose_direction_export": "camera_to_world", "units": "relative",
                        "source_timestamp_sec": frame["source_timestamp_sec"], "elapsed_timestamp_sec": frame["elapsed_timestamp_sec"]})
    xyz, rgb, score = np.concatenate(points), np.concatenate(colors), np.concatenate(scores)
    if not len(xyz):
        raise ValueError("No finite retained geometry")
    obs_frame, obs_pixel, obs_original = np.concatenate(frame_indices), np.concatenate(pixel_indices), np.concatenate(original_pixels)
    size = config["voxel"]["size_relative"] or max(float(np.median(depth[valid])) * config["voxel"]["median_depth_multiplier"], 1e-6)
    representatives, membership = reduce_voxels(xyz, score, size)
    directory = root / capture
    for name, values in (("depth", depth), ("confidence", confidence), ("valid", valid)):
        np.savez_compressed(directory / f"{name}.npz", frame_ids=frame_ids, **{name: values})
    np.savez_compressed(directory / "cloud.npz", points=xyz[representatives], colors=rgb[representatives], units=np.array("relative"),
                        coordinate_frame=np.array(manifest["captures"][capture]["coordinate_frame"]))
    write_ply(directory / "cloud.ply", xyz[representatives], rgb[representatives])
    # One observation per retained depth pixel; membership indexes PLY/cloud.npz rows.
    grouped = np.argsort(membership, kind="stable")
    offsets = np.r_[0, np.cumsum(np.bincount(membership, minlength=len(representatives)))].astype(np.int64)
    np.savez_compressed(directory / "observations.npz", frame_ids=frame_ids, observation_frame_index=obs_frame,
                        model_pixel_index=obs_pixel, original_pixel_xy=obs_original, confidence=score,
                        depth=depth.reshape(n, -1)[obs_frame, obs_pixel], cloud_point_index=membership,
                        representative_observation=representatives, point_observation_offsets=offsets,
                        point_observation_indices=grouped, voxel_size=np.array(size), model_shape_hw=np.array([height, width]))
    write_json(directory / "cameras.json", cameras)
    write_json(directory / "inference_environment.json", {**runtime, "status": "completed", "stage": "completed"})
    previews(root, capture, frames, depth, confidence, valid)
    report = verify_capture(root, capture, check_hashes=False)
    report.update({"filtering": {"eligible_pixels": int(eligible.sum()), "retained_observations": len(xyz), "cloud_points": len(representatives),
                                 "confidence_threshold": threshold, "confidence_scope": "capture", "voxel_size_relative": size},
                   "cross_view": cross_view_diagnostics(depth, valid, cameras, config["verification"]),
                   "real_scene_accuracy": "not_independently_validated", "visual_review": "pending",
                   "limitations": ["Relative scale only; captures are unaligned", "Confidence is an uncalibrated backend score",
                                   "Self-reprojection tests mappings, not independent camera/depth accuracy", "Cross-view disagreement includes occlusion and view differences",
                                   "Inspect stretched/duplicated surfaces, camera paths and missing coverage manually"]})
    write_json(directory / "quality_report.json", report)
    record = manifest["captures"][capture]
    record.update({"status": "completed", "prediction_identity": identity({"input": record["input_identity"], "config": manifest["config_identity"],
                                                                           "model": manifest["model_identity"], "code": code_identity()}),
                   "pipeline_code_identity": code_identity(), "point_count": len(representatives), "observation_count": len(xyz),
                   "internal_consistency": "passed", "real_scene_accuracy": "not_independently_validated", "artifacts": artifacts(root, capture)})
    manifest["status"] = "completed" if all(r["status"] == "completed" for r in manifest["captures"].values()) else "partially_completed"
    write_json(root / "run_manifest.json", manifest)
    return report


def verify_capture(root, capture, *, check_hashes=True):
    root = Path(root)
    manifest = read_json(root / "run_manifest.json")
    record = manifest["captures"][capture]
    if check_hashes:
        check_artifacts(root, record)
        if record["status"] != "completed":
            raise ValueError("Capture has not completed inference")
    frames = read_json(root / capture / "frames.json")["frames"]
    transforms = read_json(root / capture / "image_transforms.json")
    cameras = read_json(root / capture / "cameras.json")
    ids = np.array([f["frame_id"] for f in frames])
    source = read_json(root / capture / "source.json")
    if source["source_video_sha256"] != record["source_video_sha256"] or any(f["source_sha256"] != record["source_video_sha256"] or f["capture_id"] != capture for f in frames):
        raise ValueError("Frame/video/capture provenance identity mismatch")
    if len(set(ids.tolist())) != len(ids) or [c["frame_id"] for c in cameras] != ids.tolist() or [t["frame_id"] for t in transforms] != ids.tolist():
        raise ValueError("Frame/camera/transform order mismatch")
    arrays = {}
    for name in ("depth", "confidence", "valid", "padding_valid"):
        with np.load(root / capture / f"{name}.npz", allow_pickle=False) as data:
            if not np.array_equal(data["frame_ids"], ids):
                raise ValueError("Dense archive frame order mismatch")
            arrays[name] = data["valid" if name == "padding_valid" else name]
    depth, confidence, valid, padding = (arrays[n] for n in ("depth", "confidence", "valid", "padding_valid"))
    if (depth.shape != tuple([len(frames)] + record["model_shape_hw"]) or depth.shape != confidence.shape
            or valid.shape != depth.shape or padding.shape != depth.shape or valid.dtype != bool or padding.dtype != bool
            or depth.dtype != np.float32 or confidence.dtype != np.float32):
        raise ValueError("Dense archive shape/dtype mismatch")
    if np.any(valid & (~padding | ~np.isfinite(depth) | (depth <= 0) | ~np.isfinite(confidence))):
        raise ValueError("Validity mask admits invalid/padded samples")
    images = []
    for frame, transform in zip(frames, transforms):
        if digest(root / frame["image_path"]) != frame["image_sha256"] or digest(root / frame["model_image_path"]) != frame["model_image_sha256"]:
            raise ValueError("Frame/model image identity changed")
        with Image.open(root / frame["model_image_path"]) as image:
            images.append(np.array(image.convert("RGB")))
        if not np.allclose(np.array(transform["model_to_original"]) @ np.array(transform["original_to_model"]), np.eye(3), atol=1e-10):
            raise ValueError("Pixel transforms are not inverses")
    with np.load(root / capture / "cloud.npz", allow_pickle=False) as data:
        points, colors = data["points"], data["colors"]
        if str(data["units"]) != "relative" or str(data["coordinate_frame"]) != record["coordinate_frame"]:
            raise ValueError("Cloud coordinate contract mismatch")
    ply_points, ply_colors = load_ply(root / capture / "cloud.ply")
    if points.ndim != 2 or points.shape[1] != 3 or colors.shape != points.shape or points.dtype != np.float32 or colors.dtype != np.uint8:
        raise ValueError("Numerical cloud shape/dtype mismatch")
    if not np.array_equal(ply_points, points) or not np.array_equal(ply_colors, colors) or not np.isfinite(points).all():
        raise ValueError("PLY/numerical cloud disagree or points are nonfinite")
    with np.load(root / capture / "observations.npz", allow_pickle=False) as data:
        observations = {name: data[name] for name in data.files}
    fi, pi, membership, representatives = (observations[name] for name in ("observation_frame_index", "model_pixel_index", "cloud_point_index", "representative_observation"))
    count = len(fi)
    height, width = depth.shape[1:]
    if not np.array_equal(observations["frame_ids"], ids) or len(pi) != count or len(membership) != count or count != int(valid.sum()):
        raise ValueError("Observation archive membership/count mismatch")
    if (np.any(fi < 0) or np.any(pi < 0) or np.any(membership < 0) or np.any(representatives < 0)
            or np.any(fi >= len(frames)) or np.any(pi >= width * height) or np.any(membership >= len(points))
            or len(representatives) != len(points) or np.any(representatives >= count)):
        raise ValueError("Observation indices out of range")
    if observations["original_pixel_xy"].shape != (count, 2) or not np.isfinite(observations["original_pixel_xy"]).all():
        raise ValueError("Original pixel-coordinate archive is invalid")
    if not np.array_equal(membership[representatives], np.arange(len(points))):
        raise ValueError("Representative observation does not bind the matching cloud row")
    if not np.array_equal(observations["depth"], depth.reshape(len(frames), -1)[fi, pi]) or not np.array_equal(observations["confidence"], confidence.reshape(len(frames), -1)[fi, pi]):
        raise ValueError("Observation depth/confidence disagree with dense arrays")
    if not np.all(valid.reshape(len(frames), -1)[fi, pi]):
        raise ValueError("Provenance includes a filtered pixel")
    keys = fi.astype(np.uint64) * (width * height) + pi
    if len(np.unique(keys)) != count:
        raise ValueError("Observation pixels are duplicated")
    offsets, grouped = observations["point_observation_offsets"], observations["point_observation_indices"]
    if len(offsets) != len(points) + 1 or offsets[0] != 0 or offsets[-1] != count or np.any(np.diff(offsets) <= 0) or not np.array_equal(np.sort(grouped), np.arange(count)):
        raise ValueError("Point-to-observation CSR membership is invalid")
    if not np.array_equal(membership[grouped], np.repeat(np.arange(len(points)), np.diff(offsets))):
        raise ValueError("Grouped observation memberships disagree")
    sample = np.unique(np.linspace(0, count - 1, min(count, read_json(root / "config.json")["verification"]["sample_observations"])).astype(int))
    maximum_pixel_error, maximum_original_error, maximum_cloud_pixel_error = 0.0, 0.0, 0.0
    for index, camera in enumerate(cameras):
        k, pose = validate_intrinsics(camera["intrinsics"]), validate_pose(camera["camera_to_world"])
        if not np.allclose(np.array(camera["world_to_camera"]) @ pose, np.eye(4), atol=1e-5):
            raise ValueError("Pose directions are inconsistent")
        if camera["coordinate_frame"] != record["coordinate_frame"] or camera["depth_convention"] != "optical_axis_z" or camera["units"] != "relative":
            raise ValueError("Camera conventions differ from capture contract")
        chosen = sample[fi[sample] == index]
        uv = np.column_stack((pi[chosen] % width, pi[chosen] // width))
        clean_depth = np.where(valid[index], depth[index], 0)
        world = unproject(clean_depth, k, pose).reshape(-1, 3)[pi[chosen]]
        projected, _ = project(world, k, pose)
        if len(chosen):
            maximum_pixel_error = max(maximum_pixel_error, float(np.max(np.abs(projected - uv))))
            original = np.column_stack((uv, np.ones(len(uv)))) @ np.array(transforms[index]["model_to_original"]).T
            maximum_original_error = max(maximum_original_error, float(np.max(np.abs(original[:, :2] - observations["original_pixel_xy"][chosen]))))
        represented = np.flatnonzero(fi[representatives] == index)
        rp = pi[representatives[represented]]
        expected_points = unproject(clean_depth, k, pose).reshape(-1, 3)[rp].astype(np.float32)
        if not np.allclose(expected_points, points[represented], atol=1e-6, rtol=2e-6) or not np.array_equal(images[index].reshape(-1, 3)[rp], colors[represented]):
            raise ValueError("Cloud point/RGB does not come from its representative source pixel")
        checked = represented[np.unique(np.linspace(0, len(represented) - 1, min(len(represented), 1000)).astype(int))] if len(represented) else represented
        if len(checked):
            cloud_uv, _ = project(points[checked], k, pose)
            expected_uv = np.column_stack((pi[representatives[checked]] % width, pi[representatives[checked]] // width))
            maximum_cloud_pixel_error = max(maximum_cloud_pixel_error, float(np.max(np.abs(cloud_uv - expected_uv))))
    if maximum_pixel_error > 1e-5 or maximum_original_error > 0.001 or maximum_cloud_pixel_error > 0.005:
        raise ValueError("Pixel mapping/self-reprojection error exceeds tolerance")
    return {"internal_consistency": "passed", "cloud_points": len(points), "observations": count, "sampled_self_reprojections": len(sample),
            "max_self_reprojection_error_model_pixels": maximum_pixel_error, "max_original_pixel_mapping_error": maximum_original_error,
            "max_sampled_cloud_reprojection_error_model_pixels": maximum_cloud_pixel_error,
            "point_color_source_verified": True, "all_points_have_observations": True, "units": "relative", "coordinate_frame": record["coordinate_frame"]}
