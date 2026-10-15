"""Explicit analytic test inputs. These are never VGGT or real-room predictions."""

import copy
from pathlib import Path

import numpy as np
from PIL import Image

from reconstruction_v2 import core

PROJECT = Path(__file__).resolve().parents[1]


def source_frames(directory):
    directory = Path(directory)
    directory.mkdir(parents=True)
    artifact_list, captures = [], []
    for name in ("before", "after"):
        (directory / name).mkdir()
        rows = []
        for index in range(2):
            v, u = np.indices((24, 32))
            rgb = np.stack(((u * 7 + index * 20) % 256, (v * 9) % 256, (u * 3 + v * 4) % 256), -1).astype(np.uint8)
            frame_id = f"SYNTHETIC_{name}_{index}"
            path = f"{name}/{frame_id}.png"
            Image.fromarray(rgb).save(directory / path)
            rows.append({"capture_id": name, "frame_id": frame_id, "source_sha256": core.identity(name),
                         "decoded_frame_index": index, "source_pts": index * 30, "time_base_numerator": 1, "time_base_denominator": 30,
                         "source_timestamp_sec": float(index), "elapsed_timestamp_sec": float(index),
                         "image_path": path, "image_sha256": core.digest(directory / path), "width": 32, "height": 24,
                         "orientation_counterclockwise_degrees": 0})
            artifact_list.append({"path": path, "sha256": core.digest(directory / path)})
        core.write_json(directory / name / "frames.json", {"frames": rows})
        artifact_list.append({"path": f"{name}/frames.json", "sha256": core.digest(directory / name / "frames.json")})
        captures.append({"capture_id": name, "source_path": f"SYNTHETIC_NO_VIDEO_{name}", "source_sha256": core.identity(name),
                         "status": "synthetic", "weak_overlap_pairs": [], "uncovered_time_bins": []})
    core.write_json(directory / "frame_manifest.json", {"captures": captures, "artifacts": artifact_list, "synthetic": True})
    core.write_json(directory / "artifact_verification.json", {"status": "passed", "frame_manifest_sha256": core.digest(directory / "frame_manifest.json"),
                                                             "scope": "Synthetic fixture identity only; no video decoding"})


def prepared(directory, config=None):
    directory = Path(directory)
    source = directory.parent / (directory.name + "_synthetic_source")
    source_frames(source)
    config = config or core.read_json(PROJECT / "configs/reconstruction.json")
    result = core.prepare(source, directory, config, core.read_json(PROJECT / "configs/models.lock.json"))
    result["synthetic"] = True
    result["fixture_description"] = "Analytic planes and RGB gradients; not model predictions or actual room data"
    core.write_json(directory / "run_manifest.json", result)
    return directory


def predictions(directory, name, *, invalid=False):
    record = core.read_json(directory / "run_manifest.json")["captures"][name]
    height, width = record["model_shape_hw"]
    depth = np.full((2, height, width), 2 if name == "before" else 3, np.float32)
    confidence = np.ones_like(depth) * 2
    confidence[1] = 3
    if invalid:
        depth[0, 0, :3] = [np.nan, np.inf, -1]
        confidence[0, 0, 3] = np.nan
    k = np.tile(np.array([[420, 0, width / 2], [0, 410, height / 2], [0, 0, 1]]), (2, 1, 1))
    poses = np.tile(np.eye(4), (2, 1, 1))
    poses[1, 0, 3] = 0.1
    return depth, confidence, k, poses


def completed(directory):
    directory = prepared(directory)
    for name in ("before", "after"):
        core.export_predictions(directory, name, *predictions(directory, name, invalid=True),
                                {"backend": "SYNTHETIC_ANALYTIC_PLANE", "gpu_inference": False})
    return directory
