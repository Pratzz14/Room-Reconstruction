"""Independently bind exported PNG pixels and metadata to the source videos."""

import argparse
import csv
import hashlib
import json
from fractions import Fraction
from html.parser import HTMLParser
from pathlib import Path

import av
import numpy as np
from PIL import Image


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        self.links.extend(value for key, value in attrs if key in ("href", "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path(__file__).resolve().parents[1] / "outputs/frame_selection")
    args = parser.parse_args()
    root = args.run.resolve()
    manifest_path = root / "frame_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] not in ("completed", "completed_with_quality_warnings"):
        raise ValueError("Frame selection has not completed")
    validation_path = Path(manifest["input_manifest_path"])
    assert digest(validation_path) == manifest["input_manifest_sha256"]
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    assert digest(Path(manifest["runtime"]["script_path"])) == manifest["runtime"]["script_sha256"]
    sources = {c["capture_id"]: c for c in validation["captures"]}
    for artifact in manifest["artifacts"]:
        path = (root / artifact["path"]).resolve()
        assert path.is_relative_to(root) and path.is_file()
        assert path.stat().st_size == artifact["bytes"]
        assert digest(path) == artifact["sha256"], artifact["path"]
    checks = []
    for capture in manifest["captures"]:
        name = capture["capture_id"]
        validated = sources[name]
        source = Path(capture["source_path"])
        assert digest(source) == capture["source_sha256"] == validated["sha256"]
        with (validation_path.parent / validated["frame_timestamps"]["path"]).open(encoding="utf-8", newline="") as handle:
            timing_rows = list(csv.DictReader(handle))
        frames = json.loads((root / capture["frames_manifest_path"]).read_text(encoding="utf-8"))["frames"]
        assert len(frames) == capture["selected_count"] <= 32
        indices = [f["decoded_frame_index"] for f in frames]
        assert indices == sorted(set(indices))
        assert len(list((root / name / "frames").glob("*.png"))) == len(frames)
        if "overlap_graph_path" in capture:
            graph_record = json.loads((root / capture["overlap_graph_path"]).read_text(encoding="utf-8"))
            assert graph_record["retained_component_frame_indices"] == indices
            assert not set(indices).intersection(graph_record["excluded_disconnected_frame_indices"])
            neighbours = {index: set() for index in indices}
            for edge in graph_record["supported_edges"]:
                a, b = edge["from_frame_index"], edge["to_frame_index"]
                if a in neighbours and b in neighbours:
                    assert edge["overlap_supported"]
                    neighbours[a].add(b)
                    neighbours[b].add(a)
            reached, pending = set(), [indices[0]]
            while pending:
                index = pending.pop()
                if index not in reached:
                    reached.add(index)
                    pending.extend(neighbours[index] - reached)
            assert reached == set(indices), "Retained image-overlap graph is disconnected"
        wanted = {f["decoded_frame_index"]: f for f in frames}
        for frame in frames:
            row = timing_rows[frame["decoded_frame_index"]]
            assert row["frame_id"] == frame["frame_id"]
            assert row["capture_id"] == frame["capture_id"] == name
            assert int(row["pts"]) == frame["source_pts"]
            assert int(row["time_base_numerator"]) == frame["time_base_numerator"]
            assert int(row["time_base_denominator"]) == frame["time_base_denominator"]
            assert float(row["elapsed_time_sec"]) == frame["elapsed_timestamp_sec"]
            assert float(row["source_time_sec"]) == frame["source_timestamp_sec"]
            assert digest(root / frame["image_path"]) == frame["image_sha256"]
            assert frame["blur_score"] >= manifest["settings"]["blur_threshold"]
        links = Links()
        sheet = root / name / "contact_sheet.html"
        links.feed(sheet.read_text(encoding="utf-8"))
        for link in links.links:
            target = (sheet.parent / link).resolve()
            assert target.is_relative_to(root) and target.is_file(), link
        for sheet_name in ("selected_contact_sheet.png", "candidates_contact_sheet.png"):
            with Image.open(root / name / sheet_name) as image:
                image.verify()
        print(f"{name}: metadata, hashes, PNGs, and {len(links.links)} HTML links verified; comparing original decoded pixels", flush=True)
        matched = count = 0
        with av.open(str(source)) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 1
            for packet in container.demux(stream):
                assert not packet.is_corrupt
                for decoded in packet.decode():
                    index = count
                    count += 1
                    assert not decoded.is_corrupt
                    if index not in wanted:
                        continue
                    frame = wanted[index]
                    assert decoded.pts == frame["source_pts"]
                    assert decoded.time_base == Fraction(frame["time_base_numerator"], frame["time_base_denominator"])
                    rgb = decoded.to_ndarray(format="rgb24")
                    with Image.open(root / frame["image_path"]) as image:
                        assert image.mode == "RGB" and image.size == (frame["width"], frame["height"])
                        assert np.array_equal(np.asarray(image), rgb), frame["frame_id"]
                    matched += 1
                    if matched % 8 == 0:
                        print(f"{name}: {matched}/{len(frames)} original frame pixel comparisons passed", flush=True)
        assert count == validated["decoded_frames"] and matched == len(frames)
        assert digest(source) == validated["sha256"]
        assert not list((root / name).glob(".candidate_spool-*"))
        checks.append({"capture_id": name, "selected_pngs_verified": matched,
                       "pixel_exact_against_independent_source_decode": True,
                       "source_and_artifact_hashes_match": True,
                       "metadata_matches_validated_timestamp_archive": True,
                       "retained_overlap_graph_connected": capture.get("retained_overlap_graph_connected"),
                       "html_links_resolve": True, "temporary_spool_removed": True})
    receipt = {"status": "passed", "frame_manifest_sha256": digest(manifest_path),
               "selection_status": manifest["status"], "artifact_count_verified": len(manifest["artifacts"]),
               "checks": checks,
               "scope": "Source binding, exact PNG pixels, metadata, hashes and contact-sheet links; no 3D quality assertion"}
    (root / "artifact_verification.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
