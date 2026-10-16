"""Measured four/eight/sixteen-frame pilots, then separate full-capture inference."""

import argparse
import copy
import json
import shutil
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reconstruction.core import artifacts, code_identity, identity, read_json, write_json
from reconstruction.inference import assert_run_configuration, run


def pilot_identity(root):
    manifest = assert_run_configuration(root)
    return identity({"inputs": {n: r["input_identity"] for n, r in manifest["captures"].items()},
                     "config": manifest["config_identity"], "model": manifest["model_identity"],
                     "code": code_identity()})


def validate_pilot_report(root, report):
    expected = read_json(root / "config.json")["inference"]["pilot_frames"]
    pilots = report.get("pilots", [])
    if (report.get("status") != "passed" or report.get("run_identity") != pilot_identity(root)
            or [p.get("requested_frames") for p in pilots] != expected
            or not pilots or any(p.get("status") != "completed" or
                any(p.get("captures", {}).get(n, {}).get("status") != "completed" for n in ("before", "after"))
                for p in pilots)):
        raise RuntimeError("Pilot results are missing, incomplete, or from different inputs/configuration/code; rerun the pilot phase")


def capture_measurement(target, name, record):
    env_path = target / name / "inference_environment.json"
    environment = read_json(env_path) if env_path.is_file() else None
    measurement = {"status": record["status"], "frame_count": record["frame_count"],
                   "environment": environment, "error": record.get("error")}
    if record["status"] != "completed":
        log_path = target / "logs" / f"{name}.log"
        tail = ""
        if log_path.is_file():
            with log_path.open(encoding="utf-8", errors="replace") as handle:
                tail = "".join(deque(handle, maxlen=60))
        measurement.update({"worker_log": str(log_path), "worker_log_tail": tail,
                            "worker_error": environment.get("error") if environment else None})
        print(f"{name}: {record['status']} ({record['frame_count']} frames)\n"
              f"Stage: {(environment or {}).get('stage', 'unknown')}\n"
              f"Error: {measurement['worker_error'] or measurement['error']}\n"
              f"Worker log: {log_path}\nLast worker output:\n{tail or '(no worker output recorded)'}", flush=True)
    return measurement


def pilot_subset(source, destination, count):
    manifest = read_json(source / "run_manifest.json")
    if destination.exists():
        existing = read_json(destination / "run_manifest.json")
        if existing["parent_run_input_identity"] != identity({name: r["input_identity"] for name, r in manifest["captures"].items()}):
            raise ValueError("Pilot directory belongs to other inputs")
        if any(existing[key] != manifest[key] for key in ("config_identity", "model_identity")):
            raise ValueError("Pilot directory belongs to another configuration/model; use a new workspace")
        assert_run_configuration(destination)
        return
    destination.mkdir(parents=True)
    for file in ("config.json", "models.lock.json", "preparation_environment.json"):
        shutil.copyfile(source / file, destination / file)
    pilot = copy.deepcopy(manifest)
    pilot.update({"status": "prepared", "kind": f"pilot_{count}", "parent_run_input_identity": identity({name: r["input_identity"] for name, r in manifest["captures"].items()})})
    for name in ("before", "after"):
        rows = read_json(source / name / "frames.json")["frames"]
        indices = np.unique(np.linspace(0, len(rows) - 1, min(count, len(rows))).round().astype(int))
        selected = [rows[i] for i in indices]
        transforms = [read_json(source / name / "image_transforms.json")[i] for i in indices]
        (destination / name / "frames").mkdir(parents=True)
        (destination / name / "model_images").mkdir()
        for row in selected:
            for path in (row["image_path"], row["model_image_path"]):
                shutil.copyfile(source / path, destination / path)
        shutil.copyfile(source / name / "source.json", destination / name / "source.json")
        write_json(destination / name / "frames.json", {"capture_id": name, "frames": selected})
        write_json(destination / name / "image_transforms.json", transforms)
        with np.load(source / name / "padding_valid.npz", allow_pickle=False) as data:
            np.savez_compressed(destination / name / "padding_valid.npz", frame_ids=data["frame_ids"][indices], valid=data["valid"][indices])
        hashes = artifacts(destination, name)
        pilot["captures"][name] = {"status": "prepared", "frame_count": len(selected), "frame_ids": [f["frame_id"] for f in selected],
                                    "model_shape_hw": manifest["captures"][name]["model_shape_hw"], "coordinate_frame": f"{name}_pilot_{count}_world",
                                    "source_video_sha256": manifest["captures"][name]["source_video_sha256"],
                                    "input_identity": identity({"frames": selected, "transforms": transforms}), "artifacts": hashes, "prepared_artifacts": dict(hashes)}
    write_json(destination / "run_manifest.json", pilot)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--phase", choices=("pilot", "full"), required=True)
    args = parser.parse_args()
    source = args.workspace / "vggt"
    checkpoint = Path(read_json(args.workspace / "checkpoint.json")["checkpoint"])
    if args.phase == "pilot":
        results = {"status": "running", "run_identity": pilot_identity(args.run), "pilots": [],
                   "note": "Pilot success does not guarantee full-count memory feasibility or real-scene accuracy"}
        write_json(args.run / "pilot_report.json", results)
        for count in read_json(args.run / "config.json")["inference"]["pilot_frames"]:
            target = args.workspace / "pilots" / f"p{count:02d}"
            pilot_subset(args.run, target, count)
            manifest = run(target, source, checkpoint, resume=True)
            measurements = {name: capture_measurement(target, name, record)
                            for name, record in manifest["captures"].items()}
            results["pilots"].append({"requested_frames": count, "path": str(target), "status": manifest["status"], "captures": measurements})
            results["status"] = "running" if manifest["status"] == "completed" else "failed"
            write_json(args.run / "pilot_report.json", results)
            if results["status"] == "failed":
                raise RuntimeError(f"{count}-frame pilot failed; worker errors are printed above and saved in {args.run / 'pilot_report.json'}")
            print(f"{count}-frame pilot: both independent captures completed", flush=True)
        results["status"] = "passed"
        validate_pilot_report(args.run, results)
        write_json(args.run / "pilot_report.json", results)
        print(json.dumps({"pilot_status": results["status"], "sizes": [p["requested_frames"] for p in results["pilots"]]}, indent=2))
    else:
        report = read_json(args.run / "pilot_report.json")
        validate_pilot_report(args.run, report)
        result = run(args.run, source, checkpoint, resume=True)
        if result["status"] != "completed":
            for name, record in result["captures"].items():
                if record["status"] != "completed":
                    capture_measurement(args.run, name, record)
            raise RuntimeError("At least one capture failed; successful artifacts remain resumable")
        print("Both independent captures completed. Geometric accuracy still requires visual review.")
    freeze = args.workspace / "pip_freeze.txt"
    if freeze.is_file():
        shutil.copyfile(freeze, args.run / "environment_lock.txt")


if __name__ == "__main__":
    main()
