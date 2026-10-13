"""Read-only full-video validation; writes receipts and exact frame timestamps."""

import argparse
import csv
import hashlib
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import av


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def rational(value):
    return None if value is None else {"numerator": value.numerator, "denominator": value.denominator}


def validate(capture_id, path, output):
    started = time.perf_counter()
    record = {"capture_id": capture_id, "source_path": str(path.resolve()),
              "status": "failed", "errors": [], "warnings": []}
    decoder_logs = []
    try:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError("Input must be a nonempty regular file")
        initial_stat = path.stat()
        record["bytes"] = initial_stat.st_size
        record["sha256"] = sha256_file(path)
        print(f"{capture_id}: SHA-256 {record['sha256']}; decoding all frames", flush=True)
        rows, timestamps, intervals, dimensions, rotations = [], [], [], set(), set()
        corrupt_packets = corrupt_frames = missing_timestamps = backward = repeated = 0
        first = previous = None
        with av.logging.Capture() as decoder_logs:
            with av.open(str(path)) as container:
                if not container.streams.video:
                    raise ValueError("No video stream")
                stream = container.streams.video[0]
                stream.codec_context.thread_count = 1
                record.update({
                    "codec": stream.codec_context.name,
                    "video_stream_index": stream.index,
                    "video_stream_count": len(container.streams.video),
                    "declared_frames": stream.frames,
                    "container_duration_sec": None if container.duration is None else container.duration / av.time_base,
                    "stream_duration_sec": None if stream.duration is None else float(stream.duration * stream.time_base),
                    "stream_time_base": rational(stream.time_base),
                    "nominal_average_rate": rational(stream.average_rate),
                    "sample_aspect_ratio": rational(stream.sample_aspect_ratio),
                    "stream_metadata": dict(stream.metadata),
                })
                if len(container.streams.video) > 1:
                    record["warnings"].append("Multiple video streams; validated the first video stream only")
                for packet in container.demux(stream):
                    if packet.is_corrupt:
                        corrupt_packets += 1
                    for frame in packet.decode():
                        index = len(rows)
                        corrupt_frames += int(frame.is_corrupt)
                        dimensions.add((frame.width, frame.height))
                        rotation = float(frame.rotation) % 360
                        rotations.add(rotation)
                        row = {"capture_id": capture_id,
                               "frame_id": f"{capture_id}_{record['sha256']}_{index:08d}",
                               "decoded_frame_index": index, "pts": frame.pts,
                               "time_base_numerator": None, "time_base_denominator": None,
                               "source_time_sec": None, "elapsed_time_sec": None,
                               "width": frame.width, "height": frame.height,
                               "rotation_counterclockwise_degrees": rotation,
                               "is_corrupt": bool(frame.is_corrupt)}
                        if frame.pts is None or frame.time_base is None or frame.time_base <= 0:
                            missing_timestamps += 1
                        else:
                            timestamp = Fraction(frame.pts) * frame.time_base
                            if first is None:
                                first = timestamp
                            if previous is not None:
                                delta = timestamp - previous
                                backward += int(delta < 0)
                                repeated += int(delta == 0)
                                if delta > 0:
                                    intervals.append(delta)
                            previous = timestamp
                            timestamps.append(timestamp)
                            row.update({"time_base_numerator": frame.time_base.numerator,
                                        "time_base_denominator": frame.time_base.denominator,
                                        "source_time_sec": float(timestamp),
                                        "elapsed_time_sec": float(timestamp - first)})
                        rows.append(row)
                        if len(rows) % 300 == 0:
                            print(f"{capture_id}: decoded {len(rows)} frames", flush=True)
        if not rows:
            raise ValueError("No decodable frames")
        for count, message in ((corrupt_packets, "corrupt video packets"),
                               (corrupt_frames, "corrupt decoded frames"),
                               (missing_timestamps, "frames without usable timestamps"),
                               (backward, "backward timestamp transitions")):
            if count:
                record["errors"].append(f"{count} {message}")
        if len(dimensions) != 1 or len(rotations) != 1:
            record["errors"].append("Resolution or orientation changes within the video")
        if any(abs(((angle - round(angle / 90) * 90 + 180) % 360) - 180) > 0.01 for angle in rotations):
            record["errors"].append("Display rotation is not a supported quarter turn")
        if record["declared_frames"] and record["declared_frames"] != len(rows):
            record["errors"].append(f"Decoded count {len(rows)} differs from declared count {record['declared_frames']}")
        if len(timestamps) > 1 and timestamps[-1] <= timestamps[0]:
            record["errors"].append("No positive timestamp span")
        if repeated:
            record["warnings"].append(f"{repeated} repeated timestamps; later selection must resolve ties deterministically")
        median_interval = statistics.median(intervals) if intervals else Fraction(0)
        timestamp_span = timestamps[-1] - timestamps[0] if timestamps else None
        decoded_duration = None if timestamp_span is None else timestamp_span + median_interval
        long_gaps = sum(delta > 3 * median_interval for delta in intervals) if median_interval else 0
        if long_gaps:
            record["warnings"].append(f"{long_gaps} frame intervals exceed three times the median")
        width, height = sorted(dimensions)[0]
        rotation = sorted(rotations)[0]
        swap = round(rotation / 90) % 2
        record.update({
            "decoded_frames": len(rows), "decoded_dimensions": [list(item) for item in sorted(dimensions)],
            "encoded_width": width, "encoded_height": height,
            "orientation_counterclockwise_degrees": rotation,
            "orientation_values_counterclockwise_degrees": sorted(rotations),
            "display_pixel_width": height if swap else width,
            "display_pixel_height": width if swap else height,
            "orientation_source": "PyAV decoded frame display-matrix rotation",
            "source_start_timestamp_sec": None if first is None else float(first),
            "timestamp_span_sec": None if timestamp_span is None else float(timestamp_span),
            "decoded_duration_estimate_sec": None if decoded_duration is None else float(decoded_duration),
            "duration_estimation_method": "last minus first presentation timestamp plus median positive frame interval",
            "timestamps": {"missing": missing_timestamps, "backward_transitions": backward,
                           "repeated_transitions": repeated, "variable_positive_intervals": len(set(intervals)) > 1,
                           "median_interval_sec": float(median_interval),
                           "minimum_positive_interval_sec": float(min(intervals)) if intervals else None,
                           "maximum_interval_sec": float(max(intervals)) if intervals else None,
                           "long_gaps_over_3x_median": long_gaps},
            "decode": {"reached_end_of_stream": True, "corrupt_packets": corrupt_packets,
                       "corrupt_frames": corrupt_frames},
        })
        timestamp_path = output / f"{capture_id}_frame_timestamps.csv"
        with timestamp_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        record["frame_timestamps"] = {"path": timestamp_path.name,
                                     "sha256": sha256_file(timestamp_path), "rows": len(rows)}
        # A second hash binds the decoded observations to unchanged source bytes.
        final_hash = sha256_file(path)
        final_stat = path.stat()
        record["source_unchanged_during_validation"] = (
            final_hash == record["sha256"] and
            (initial_stat.st_size, initial_stat.st_mtime_ns) == (final_stat.st_size, final_stat.st_mtime_ns))
        if not record["source_unchanged_during_validation"]:
            record["errors"].append("Source changed during validation")
    except Exception as exc:
        record["errors"].append(f"{type(exc).__name__}: {exc}")
    log_rows = [{"level": level, "component": component, "message": message.strip()}
                for level, component, message in decoder_logs if level <= av.logging.WARNING]
    if log_rows:
        log_path = output / f"{capture_id}_decoder_log.json"
        write_json(log_path, log_rows)
        record["decoder_log_path"] = log_path.name
        error_logs = sum(item["level"] <= av.logging.ERROR for item in log_rows)
        if error_logs:
            record["errors"].append(f"{error_logs} FFmpeg error log entries; inspect decoder log")
        else:
            record["warnings"].append(f"{len(log_rows)} FFmpeg warning log entries; inspect decoder log")
    record["status"] = "failed" if record["errors"] else "passed_with_warnings" if record["warnings"] else "passed"
    record["validation_elapsed_sec"] = round(time.perf_counter() - started, 3)
    print(f"{capture_id}: {record['status']} ({record.get('decoded_frames', 0)} frames)", flush=True)
    return record


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, default=root / "data/before.mp4")
    parser.add_argument("--after", type=Path, default=root / "data/after.mp4")
    parser.add_argument("--output", type=Path, default=root / "outputs/input_validation")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error("Output already exists; use a new --output path to preserve the earlier receipt")
    if args.before.resolve() == args.after.resolve():
        parser.error("Before and after must reference different files")
    for source in (args.before.resolve(), args.after.resolve()):
        if source.is_relative_to(output):
            parser.error("Output must not contain a source video")
    output.mkdir(parents=True)
    av.logging.set_level(av.logging.WARNING)
    records = [validate(capture_id, path.resolve(), output)
               for capture_id, path in (("before", args.before), ("after", args.after))]
    if records[0].get("sha256") and records[0].get("sha256") == records[1].get("sha256"):
        for record in records:
            record["status"] = "failed"
            record["errors"].append("Before and after have identical source contents")
    script_path = Path(__file__).resolve()
    manifest = {
        "schema_version": "1.0", "stage": "input_validation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "failed" if any(r["status"] == "failed" for r in records) else
                  "passed_with_warnings" if any(r["warnings"] for r in records) else "passed",
        "scope": {"captures": ["before", "after"], "source_videos_modified": False,
                  "audio_validation": False, "image_extraction": False},
        "validator": {"script": str(script_path), "sha256": sha256_file(script_path),
                      "python_executable": sys.executable, "python_version": platform.python_version(),
                      "pyav_version": av.__version__, "ffmpeg_libraries": av.library_versions},
        "captures": records,
        "limitations": ["Decoding and timing checks do not establish scene coverage or 3D reconstruction quality.",
                        "Display orientation is metadata-derived; physical upright orientation is not visually verified.",
                        "Decoded duration is an estimate; container and stream durations are recorded separately.",
                        "Only the first video stream is validated; audio is not decoded."],
    }
    write_json(output / "input_manifest.json", manifest)
    write_json(output / "validation_report.json", {
        "schema_version": "1.0", "manifest_path": "input_manifest.json",
        "manifest_sha256": sha256_file(output / "input_manifest.json"),
        "status": manifest["status"],
        "captures": [{key: record[key] for key in ("capture_id", "status", "errors", "warnings")}
                     for record in records],
        "limitations": manifest["limitations"],
    })
    lines = ["# Input validation report", "", f"Overall result: **{manifest['status']}**", "",
             "Both source videos were read without rotation, transcoding, renaming, or other edits.", "",
             "| Check | Why we do it | Change performed |", "|---|---|---|",
             "| Full decoding | Find unreadable or truncated video before GPU work | Read all video frames; record decoder errors and counts |",
             "| Duration and timestamps | Select frames by actual presentation time and retain source evidence | Write exact PTS/time-base and elapsed timestamps to CSV |",
             "| Resolution and orientation | Keep later pixel-to-3D mappings consistent | Record encoded/display dimensions and metadata rotation; do not apply transforms |",
             "| SHA-256 identity | Bind future outputs to the exact input bytes | Hash each file before and after validation |",
             "| Capture IDs | Keep independent before/after records unambiguous | Assign logical IDs in metadata; keep filenames unchanged |", "",
             "| Capture | Result | Decoded frames | Duration estimate (s) | Encoded pixels | Rotation CCW |",
             "|---|---|---:|---:|---|---:|"]
    for record in records:
        duration = record.get("decoded_duration_estimate_sec")
        duration_text = "unavailable" if duration is None else f"{duration:.6f}"
        lines.append(f"| {record['capture_id']} | {record['status']} | {record.get('decoded_frames', 'unavailable')} | {duration_text} | {record.get('encoded_width', '?')} x {record.get('encoded_height', '?')} | {record.get('orientation_counterclockwise_degrees', 'unavailable')} |")
    for record in records:
        lines += ["", f"## {record['capture_id']}", "", f"Source: `{record['source_path']}`", "",
                  f"SHA-256: `{record.get('sha256', 'unavailable')}`", ""]
        lines += [f"- Error: {value}" for value in record["errors"]]
        lines += [f"- Warning: {value}" for value in record["warnings"]]
        if not record["errors"] and not record["warnings"]:
            lines.append("No validation errors or warnings recorded.")
    lines += ["", "## Limits", ""] + [f"- {value}" for value in manifest["limitations"]]
    lines += ["", "## Re-run", "", "Use the same interpreter with PyAV installed and a new output directory:", "",
              "```powershell", f"& '{sys.executable}' '{script_path}' --output '{output.parent / 'input_validation_repeat'}'", "```", ""]
    (output / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved receipts: {output}", flush=True)
    return 1 if manifest["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
