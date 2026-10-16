"""Select traceable RGB frames from previously validated videos, without inference."""

import argparse
import csv
import hashlib
import html
import json
import math
import platform
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ALGORITHM = "timestamp-bins-sharpness-duplicates-overlap-connected-v2"
SETTINGS = {
    "initial_candidates": 128, "maximum_selected": 32,
    "quality_longest_edge": 640, "blur_threshold": 50.0,
    "duplicate_hash_distance": 4, "duplicate_gray_mad": 5.0,
    "overlap_min_matches": 20, "overlap_min_inliers": 15,
    "overlap_min_inlier_fraction": 0.4, "overlap_min_spatial_coverage": 0.05,
    "little_image_change_fraction": 0.01,
    "adaptive_candidate_grid": 256, "replacement_sweeps": 2,
}


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def time_of(row):
    return Fraction(int(row["pts"]) * int(row["time_base_numerator"]), int(row["time_base_denominator"]))


def time_bin(index, timings):
    elapsed = time_of(timings[index]) - time_of(timings[0])
    span = time_of(timings[-1]) - time_of(timings[0])
    return min(31, int(elapsed * 32 / span))


def uniform_indices(timings, count):
    from bisect import bisect_left
    timestamps = [time_of(row) for row in timings]
    indices = set()
    for step in range(count):
        target = timestamps[0] + (timestamps[-1] - timestamps[0]) * Fraction(step, count - 1)
        cursor = bisect_left(timestamps, target)
        choices = [i for i in (cursor - 1, cursor) if 0 <= i < len(timestamps)]
        indices.add(min(choices, key=lambda i: (abs(timestamps[i] - target), i)))
    return indices


def score_frame(rgb):
    height, width = rgb.shape[:2]
    factor = min(1, SETTINGS["quality_longest_edge"] / max(width, height))
    small = cv2.resize(rgb, (round(width * factor), round(height * factor)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    dhash_gray = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    dhash = 0
    for bit in (dhash_gray[:, 1:] > dhash_gray[:, :-1]).ravel():
        dhash = (dhash << 1) | int(bit)
    signature = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    points, descriptors = cv2.ORB_create(nfeatures=2500).detectAndCompute(gray, None)
    return sharpness, dhash, signature, small, np.array([p.pt for p in points], dtype=np.float32).reshape(-1, 2), descriptors


def extract(source, capture, timings, wanted, spool, candidates):
    print(f"{capture['capture_id']}: extracting {len(wanted)} candidates", flush=True)
    count = 0
    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        for packet in container.demux(stream):
            if packet.is_corrupt:
                raise ValueError("Corrupt packet during extraction")
            for frame in packet.decode():
                index = count
                count += 1
                if index >= len(timings):
                    raise ValueError("Source frame count changed")
                row = timings[index]
                if frame.is_corrupt or frame.pts != int(row["pts"]) or frame.time_base != Fraction(int(row["time_base_numerator"]), int(row["time_base_denominator"])):
                    raise ValueError(f"Corrupt frame or source timestamps changed at frame {index}")
                if (frame.width, frame.height) != (capture["encoded_width"], capture["encoded_height"]) or float(frame.rotation) % 360 != capture["orientation_counterclockwise_degrees"]:
                    raise ValueError("Source image geometry changed")
                if index not in wanted:
                    continue
                rgb = frame.to_ndarray(format="rgb24")
                raw = spool / f"{index:08d}.npy"
                np.save(raw, rgb, allow_pickle=False)
                sharpness, dhash, signature, thumbnail, points, descriptors = score_frame(rgb)
                candidates[index] = {"index": index, "timing": row, "bin": time_bin(index, timings),
                                     "sharpness": sharpness, "dhash": dhash, "signature": signature,
                                     "thumbnail": thumbnail, "points": points, "descriptors": descriptors,
                                     "raw": raw}
                if len(candidates) % 32 == 0:
                    print(f"{capture['capture_id']}: scored {len(candidates)} candidates", flush=True)
    if count != len(timings) or not wanted.issubset(candidates):
        raise ValueError("Frame count or requested candidate set changed")
    if digest(source) != capture["sha256"]:
        raise ValueError("Source hash changed during extraction")


def duplicates(a, b):
    return ((a["dhash"] ^ b["dhash"]).bit_count() <= SETTINGS["duplicate_hash_distance"] and
            float(np.abs(a["signature"].astype(np.int16) - b["signature"].astype(np.int16)).mean()) <= SETTINGS["duplicate_gray_mad"])


def initial_selection(candidates):
    bins = [[] for _ in range(32)]
    for candidate in candidates.values():
        if candidate["sharpness"] >= SETTINGS["blur_threshold"]:
            bins[candidate["bin"]].append(candidate)
    for group in bins:
        group.sort(key=lambda c: (-c["sharpness"], c["index"]))
    selected = []
    while len(selected) < 32 and any(bins):
        for group in bins:
            while group and len(selected) < 32:
                candidate = group.pop(0)
                if not any(duplicates(candidate, kept) for kept in selected):
                    selected.append(candidate)
                    break
    return sorted(selected, key=lambda c: c["index"])


def overlap(a, b, cache):
    key = (a["index"], b["index"])
    if key in cache:
        return cache[key]
    result = {"from_frame_index": key[0], "to_frame_index": key[1],
              "matches": 0, "inliers": 0, "inlier_fraction": 0.0,
              "spatial_coverage_from": 0.0, "spatial_coverage_to": 0.0,
              "median_image_displacement_fraction": None, "model": None,
              "overlap_supported": False, "little_image_change": False}
    if a["descriptors"] is not None and b["descriptors"] is not None and len(b["descriptors"]) >= 2:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        pairs = matcher.knnMatch(a["descriptors"], b["descriptors"], k=2)
        matches = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
        # Each destination feature supports at most one source feature.
        unique = {}
        for match in sorted(matches, key=lambda m: (m.distance, m.queryIdx, m.trainIdx)):
            unique.setdefault(match.trainIdx, match)
        matches = list(unique.values())
        result["matches"] = len(matches)
        if len(matches) >= 8:
            p = a["points"][[m.queryIdx for m in matches]]
            q = b["points"][[m.trainIdx for m in matches]]
            models = []
            cv2.setRNGSeed(0)
            matrix, mask = cv2.findHomography(p, q, cv2.RANSAC, 3.0, maxIters=2000, confidence=0.995)
            if matrix is not None and mask is not None:
                models.append(("homography", mask.ravel().astype(bool)))
            cv2.setRNGSeed(0)
            matrix, mask = cv2.findFundamentalMat(p, q, cv2.FM_RANSAC, 2.0, 0.995)
            if matrix is not None and mask is not None and len(mask) == len(matches):
                models.append(("fundamental", mask.ravel().astype(bool)))
            if models:
                name, mask = max(models, key=lambda item: (int(item[1].sum()), item[0]))
                inliers = int(mask.sum())
                height, width = a["thumbnail"].shape[:2]
                area = height * width
                coverage_a = float(cv2.contourArea(cv2.convexHull(p[mask]))) / area if inliers >= 3 else 0.0
                coverage_b = float(cv2.contourArea(cv2.convexHull(q[mask]))) / area if inliers >= 3 else 0.0
                displacement = float(np.median(np.linalg.norm(p[mask] - q[mask], axis=1))) / math.hypot(width, height) if inliers else None
                ratio = inliers / len(matches)
                supported = (len(matches) >= SETTINGS["overlap_min_matches"] and inliers >= SETTINGS["overlap_min_inliers"] and
                             ratio >= SETTINGS["overlap_min_inlier_fraction"] and
                             min(coverage_a, coverage_b) >= SETTINGS["overlap_min_spatial_coverage"])
                result.update({"inliers": inliers, "inlier_fraction": ratio, "model": name,
                               "spatial_coverage_from": coverage_a, "spatial_coverage_to": coverage_b,
                               "median_image_displacement_fraction": displacement, "overlap_supported": supported,
                               "little_image_change": displacement is not None and displacement < SETTINGS["little_image_change_fraction"]})
    cache[key] = result
    return result


def edge_score(result):
    support = min(result["inliers"] / 80, 1.0) * min(min(result["spatial_coverage_from"], result["spatial_coverage_to"]) / 0.2, 1.0)
    return 2 * int(result["overlap_supported"]) + support - 0.25 * int(result["little_image_change"])


def repair_selection(selected, candidates, cache):
    replacements = []
    for sweep in range(SETTINGS["replacement_sweeps"]):
        for pos in range(len(selected)):
            current = selected[pos]
            neighbours = ([selected[pos - 1]] if pos else []) + ([selected[pos + 1]] if pos + 1 < len(selected) else [])
            def score(candidate):
                edges = [overlap(neighbour, candidate, cache) if neighbour["index"] < candidate["index"] else overlap(candidate, neighbour, cache)
                         for neighbour in neighbours]
                return sum(edge_score(edge) for edge in edges) + 0.05 * math.log1p(candidate["sharpness"])
            edges = [overlap(neighbour, current, cache) if neighbour["index"] < current["index"] else overlap(current, neighbour, cache) for neighbour in neighbours]
            if edges and all(e["overlap_supported"] and not e["little_image_change"] for e in edges):
                continue
            alternatives = sorted((c for c in candidates.values() if c["bin"] == current["bin"] and c["sharpness"] >= SETTINGS["blur_threshold"]),
                                  key=lambda c: (-c["sharpness"], c["index"]))[:8]
            current_score = score(current)
            best, best_score = current, current_score
            for candidate in alternatives:
                others = selected[:pos] + selected[pos + 1:]
                if any(duplicates(candidate, other) for other in others):
                    continue
                if pos and candidate["index"] <= selected[pos - 1]["index"]:
                    continue
                if pos + 1 < len(selected) and candidate["index"] >= selected[pos + 1]["index"]:
                    continue
                candidate_score = score(candidate)
                if candidate_score > best_score + 1e-9:
                    best, best_score = candidate, candidate_score
            if best["index"] != current["index"] and best_score >= current_score + 0.1:
                selected[pos] = best
                replacements.append({"old_index": current["index"], "new_index": best["index"],
                                     "reason": "improved_neighbour_overlap_or_image_change", "score_gain": best_score - current_score})
    return selected, replacements


def connected_selection(selected, cache):
    """Keep a connected visual-overlap component rather than force the frame count."""
    edges = [overlap(a, b, cache) for a, b in zip(selected, selected[1:])]
    graph = {c["index"]: set() for c in selected}
    # A supported temporal chain already establishes image-feature connectivity.
    pairs = list(zip(selected, selected[1:])) if all(e["overlap_supported"] for e in edges) else [
        (a, b) for pos, a in enumerate(selected) for b in selected[pos + 1:]]
    supported_edges = []
    for a, b in pairs:
        edge = overlap(a, b, cache)
        if edge["overlap_supported"]:
            graph[a["index"]].add(b["index"])
            graph[b["index"]].add(a["index"])
            supported_edges.append(dict(edge))
    unseen, components = set(graph), []
    while unseen:
        pending, component = [min(unseen)], set()
        while pending:
            index = pending.pop()
            if index in component:
                continue
            component.add(index)
            pending.extend(graph[index] - component)
        unseen -= component
        components.append(sorted(component))
    kept = set(max(components, key=lambda component: (len(component), -component[0])))
    rejected = sorted(set(graph) - kept)
    if rejected:
        print(f"Excluded disconnected selected-frame components: {rejected}; kept {len(kept)} connected views", flush=True)
    return [c for c in selected if c["index"] in kept], {
        "method": "Supported temporal chain, or all selected pairs when temporal links are weak",
        "components_before_filtering": components,
        "excluded_disconnected_frame_indices": rejected,
        "retained_component_frame_indices": sorted(kept),
        "supported_edges": supported_edges,
        "limitations": "Image-feature connectivity does not establish calibrated poses or 3D accuracy",
    }


def contact_sheet(candidates, path, title, decisions):
    cols, tile_w, tile_h = 4, 340, 236
    sheet = Image.new("RGB", (cols * tile_w, math.ceil(len(candidates) / cols) * tile_h + 52), "#f3f0e9")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=14)
    draw.text((12, 16), title, fill="#252525", font=font)
    for pos, candidate in enumerate(candidates):
        x, y = pos % cols * tile_w, pos // cols * tile_h + 52
        thumb = Image.fromarray(candidate["thumbnail"])
        thumb.thumbnail((324, 182), Image.Resampling.LANCZOS)
        sheet.paste(thumb, (x + 8, y))
        row = candidate["timing"]
        reason = decisions[candidate["index"]]["reason"]
        fill = "#177345" if reason.startswith("selected") else "#444444"
        draw.text((x + 8, y + 186), f"frame {candidate['index']} | {float(row['elapsed_time_sec']):.3f}s", fill=fill, font=font)
        draw.text((x + 8, y + 209), f"sharpness {candidate['sharpness']:.1f} | {reason}", fill=fill, font=font)
    sheet.save(path)


def process(capture, validation_root, output):
    started = time.perf_counter()
    name = capture["capture_id"]
    source = Path(capture["source_path"])
    if digest(source) != capture["sha256"]:
        raise ValueError(f"{name}: source no longer matches input validation")
    timing_path = validation_root / capture["frame_timestamps"]["path"]
    if digest(timing_path) != capture["frame_timestamps"]["sha256"]:
        raise ValueError(f"{name}: validated timestamp archive changed")
    with timing_path.open(newline="", encoding="utf-8") as handle:
        timings = list(csv.DictReader(handle))
    if len(timings) != capture["decoded_frames"]:
        raise ValueError("Timestamp archive count mismatch")
    if capture["orientation_counterclockwise_degrees"] != 0:
        raise ValueError("This selector requires the validated 0-degree inputs; no implicit orientation transform")
    directory = output / name
    directory.mkdir()
    (directory / "frames").mkdir()
    (directory / "candidate_thumbnails").mkdir()
    candidates, cache = {}, {}
    with tempfile.TemporaryDirectory(prefix=".candidate_spool-", dir=directory) as temporary:
        extract(source, capture, timings, uniform_indices(timings, SETTINGS["initial_candidates"]), Path(temporary), candidates)
        selected = initial_selection(candidates)
        if not selected:
            raise ValueError("No candidate passes the recorded sharpness and duplicate policy")
        baseline_indices = [c["index"] for c in selected]
        selected, replacements = repair_selection(selected, candidates, cache)
        edges = [overlap(a, b, cache) for a, b in zip(selected, selected[1:])]
        weak_bins = set()
        for a, b, edge in zip(selected, selected[1:], edges):
            if not edge["overlap_supported"]:
                weak_bins.update(range(max(0, a["bin"] - 1), min(31, b["bin"] + 1) + 1))
        occupied_bins = {c["bin"] for c in selected}
        weak_bins.update(set(range(32)) - occupied_bins)
        adaptive = {index for index in uniform_indices(timings, SETTINGS["adaptive_candidate_grid"])
                    if index not in candidates and time_bin(index, timings) in weak_bins}
        if adaptive:
            print(f"{name}: denser sampling adds {len(adaptive)} candidates in {len(weak_bins)} affected time bins", flush=True)
            extract(source, capture, timings, adaptive, Path(temporary), candidates)
            new_selected = initial_selection(candidates)
            # Keep the repaired baseline when its overlap is stronger than the expanded selection.
            repaired, more_replacements = repair_selection(new_selected, candidates, cache)
            def selection_score(items):
                return (len(items), sum(edge_score(overlap(a, b, cache)) for a, b in zip(items, items[1:])))
            if selection_score(repaired) >= selection_score(selected):
                selected = repaired
                replacements += more_replacements
            else:
                selected, more_replacements = repair_selection(selected, candidates, cache)
                replacements += more_replacements
        selected, connectivity = connected_selection(selected, cache)
        disconnected_indices = set(connectivity["excluded_disconnected_frame_indices"])
        selected_indices = {c["index"] for c in selected}
        baseline_set = set(baseline_indices)
        decisions = {}
        frames = []
        for candidate in sorted(candidates.values(), key=lambda c: c["index"]):
            index, row = candidate["index"], candidate["timing"]
            thumbnail_path = f"{name}/candidate_thumbnails/{row['frame_id']}.png"
            Image.fromarray(candidate["thumbnail"]).save(output / thumbnail_path, compress_level=3)
            duplicate = next((other for other in selected if other["index"] != index and duplicates(candidate, other)), None)
            reason = ("selected_overlap_refinement" if index not in baseline_set else "selected") if index in selected_indices else (
                "disconnected_overlap_component" if index in disconnected_indices else
                "blur" if candidate["sharpness"] < SETTINGS["blur_threshold"] else
                "duplicate" if duplicate is not None else "replaced_for_overlap" if index in baseline_set else "frame_budget")
            decisions[index] = {"frame_id": row["frame_id"], "decoded_frame_index": index,
                                "timestamp_sec": float(row["elapsed_time_sec"]), "time_bin": candidate["bin"],
                                "blur_score": candidate["sharpness"], "difference_hash": f"{candidate['dhash']:016x}",
                                "reason": reason, "thumbnail_path": thumbnail_path,
                                "matching_selected_frame_index": duplicate["index"] if reason == "duplicate" else None,
                                "candidate_sampling": "initial" if index in uniform_indices(timings, SETTINGS["initial_candidates"]) else "adaptive"}
            if index in selected_indices:
                image_path = f"{name}/frames/{row['frame_id']}.png"
                rgb = np.load(candidate["raw"], allow_pickle=False)
                Image.fromarray(rgb).save(output / image_path, compress_level=3)
                with Image.open(output / image_path) as image:
                    if image.mode != "RGB" or not np.array_equal(np.asarray(image), rgb):
                        raise ValueError(f"PNG pixel round-trip failed: {image_path}")
                frames.append({"capture_id": name, "frame_id": row["frame_id"], "source_sha256": capture["sha256"],
                               "decoded_frame_index": index, "source_pts": int(row["pts"]),
                               "time_base_numerator": int(row["time_base_numerator"]), "time_base_denominator": int(row["time_base_denominator"]),
                               "source_timestamp_sec": float(row["source_time_sec"]), "elapsed_timestamp_sec": float(row["elapsed_time_sec"]),
                               "image_path": image_path, "image_sha256": digest(output / image_path),
                               "width": rgb.shape[1], "height": rgb.shape[0], "mode": "RGB",
                               "orientation_counterclockwise_degrees": 0, "original_to_exported_pixel_transform": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                               "blur_score": candidate["sharpness"], "difference_hash": f"{candidate['dhash']:016x}",
                               "time_bin": candidate["bin"], "selection_reason": reason, "png_pixel_roundtrip_verified": True})
        ordered = sorted(candidates.values(), key=lambda c: c["index"])
        contact_sheet(ordered, directory / "candidates_contact_sheet.png", f"{name.upper()} | {len(ordered)} candidates | green labels = selected", decisions)
        contact_sheet(selected, directory / "selected_contact_sheet.png", f"{name.upper()} | {len(selected)} selected frames | chronological order", decisions)
        cards = []
        for candidate in ordered:
            row = candidate["timing"]
            decision = decisions[candidate["index"]]
            thumb = "candidate_thumbnails/" + row["frame_id"] + ".png"
            link = "frames/" + row["frame_id"] + ".png" if candidate["index"] in selected_indices else thumb
            selected_attr = "selected" if candidate["index"] in selected_indices else "rejected"
            cards.append(f'<figure class="{selected_attr}"><a href="{html.escape(link, quote=True)}"><img src="{html.escape(thumb, quote=True)}" alt="Frame {candidate["index"]}"></a><figcaption>Frame {candidate["index"]} | {float(row["elapsed_time_sec"]):.3f}s<br>Sharpness {candidate["sharpness"]:.1f} | {decision["reason"]}</figcaption></figure>')
        (directory / "contact_sheet.html").write_text(f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{name.title()} frame selection</title><style>body{{font:16px system-ui;background:#f3f0e9;margin:24px;color:#252525}}main{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}}figure{{margin:0;padding:10px;background:white;border:2px solid transparent}}figure.selected{{border-color:#177345}}img{{width:100%;height:auto}}figcaption{{font-size:13px;line-height:1.7}}label{{display:block;margin:16px 0}}body.only-selected .rejected{{display:none}}</style></head><body><h1>{name.title()} frame selection</h1><p>{len(ordered)} candidates; {len(selected)} selected. Green borders identify selected frames. Click selected images for full-resolution PNGs.</p><p>Inspect shared landmarks, coverage, and camera translation. Feature matching alone does not verify physical parallax.</p><label><input type="checkbox" onchange="document.body.classList.toggle('only-selected',this.checked)"> Show selected frames only</label><main>{''.join(cards)}</main></body></html>''', encoding="utf-8")
        save_json(directory / "candidate_decisions.json", {"schema_version": "1.0", "capture_id": name, "candidates": list(decisions.values())})
        save_json(directory / "frames.json", {"schema_version": "1.0", "capture_id": name, "source_sha256": capture["sha256"], "paths_relative_to": "frame_selection run root", "frames": frames})
        final_edges = [overlap(a, b, cache) for a, b in zip(selected, selected[1:])]
        for edge, a, b in zip(final_edges, selected, selected[1:]):
            edge["from_frame_id"], edge["to_frame_id"] = a["timing"]["frame_id"], b["timing"]["frame_id"]
            edge["gap_sec"] = float(time_of(b["timing"]) - time_of(a["timing"]))
        save_json(directory / "overlap_diagnostics.json", {"capture_id": name, "method": "ORB ratio matches; robust homography/fundamental inliers; distributed image support", "pixel_coordinate_space": "640-pixel-longest-edge thumbnails", "physical_parallax_status": "unverified", "adjacent_selected_pairs": final_edges, "replacement_history": replacements})
        save_json(directory / "overlap_graph.json", {"capture_id": name, **connectivity})
        span = float(time_of(timings[-1]) - time_of(timings[0]))
        elapsed = [float(c["timing"]["elapsed_time_sec"]) for c in selected]
        gaps = np.diff([0.0] + elapsed + [span])
        occupied = sorted({c["bin"] for c in selected})
        summary = {"capture_id": name, "source_path": str(source), "source_sha256": capture["sha256"],
                   "candidate_count": len(candidates), "initial_candidate_count": 128, "adaptive_candidate_count": len(adaptive),
                   "selected_count": len(selected), "decision_counts": dict(Counter(d["reason"] for d in decisions.values())),
                   "covered_time_bins": occupied, "uncovered_time_bins": sorted(set(range(32)) - set(occupied)),
                   "first_selected_time_sec": elapsed[0], "last_selected_time_sec": elapsed[-1],
                   "largest_selected_gap_sec": float(max(gaps)), "overlap_supported_pairs": sum(e["overlap_supported"] for e in final_edges),
                   "adjacent_pair_count": len(final_edges), "weak_overlap_pairs": [{"from_index": e["from_frame_index"], "to_index": e["to_frame_index"]} for e in final_edges if not e["overlap_supported"]],
                   "little_image_change_pairs": [{"from_index": e["from_frame_index"], "to_index": e["to_frame_index"]} for e in final_edges if e["little_image_change"]],
                   "physical_parallax_status": "unverified", "source_unchanged_after_extraction": digest(source) == capture["sha256"],
                   "excluded_disconnected_frame_indices": sorted(disconnected_indices),
                   "retained_overlap_graph_connected": True,
                   "status": "completed_with_quality_warnings" if len(selected) < 32 or len(occupied) < 32 or any(not e["overlap_supported"] or e["little_image_change"] for e in final_edges) else "completed",
                   "elapsed_sec": round(time.perf_counter() - started, 3),
                   "frames_manifest_path": f"{name}/frames.json", "candidate_decisions_path": f"{name}/candidate_decisions.json",
                   "overlap_diagnostics_path": f"{name}/overlap_diagnostics.json",
                   "overlap_graph_path": f"{name}/overlap_graph.json",
                   "contact_sheet_paths": [f"{name}/candidates_contact_sheet.png", f"{name}/selected_contact_sheet.png", f"{name}/contact_sheet.html"]}
        if not summary["source_unchanged_after_extraction"]:
            raise ValueError("Source changed while exporting frames")
    summary["temporary_candidate_spool_removed"] = True
    print(f"{name}: selected {len(selected)}/{len(candidates)}; overlap supported for {summary['overlap_supported_pairs']}/{len(final_edges)} adjacent pairs", flush=True)
    return summary


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, default=root / "outputs/input_validation/input_manifest.json")
    parser.add_argument("--output", type=Path, default=root / "outputs/frame_selection")
    args = parser.parse_args()
    output = args.output.resolve()
    validation = args.validation.resolve()
    manifest = json.loads(validation.read_text(encoding="utf-8"))
    if manifest["status"] != "passed" or {c["capture_id"] for c in manifest["captures"]} != {"before", "after"}:
        parser.error("Requires a passed validation manifest for exactly before and after")
    report = json.loads((validation.parent / "validation_report.json").read_text(encoding="utf-8"))
    if report["manifest_sha256"] != digest(validation):
        parser.error("Validation report no longer binds the input manifest")
    if output.exists():
        parser.error("Output already exists; use a new --output path to preserve earlier results")
    if any(Path(c["source_path"]).resolve().is_relative_to(output) for c in manifest["captures"]):
        parser.error("Output must not contain the source videos")
    output.mkdir(parents=True)
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    av.logging.set_level(av.logging.ERROR)
    result = {"schema_version": "1.0", "algorithm": ALGORITHM, "status": "running",
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "input_manifest_path": str(validation),
              "input_manifest_sha256": digest(validation), "settings": SETTINGS,
              "source_videos_modified": False, "exported_image_space": "original decoded RGB; no resizing, cropping, or rotation",
              "runtime": {"python": platform.python_version(), "av": av.__version__, "opencv": cv2.__version__, "numpy": np.__version__,
                          "script_path": str(Path(__file__).resolve()), "script_sha256": digest(Path(__file__).resolve())},
              "captures": [], "errors": [],
              "limitations": ["Sharpness and duplicate thresholds are heuristics; review contact sheets.",
                              "Feature matches support image overlap but do not verify camera translation or physical parallax.",
                              "Temporal coverage does not guarantee complete room coverage or unseen surfaces.",
                              "RGB PNGs preserve decoded RGB pixels; they do not recover original sensor data or lost HEVC detail.",
                              "Model-image preprocessing is deferred to the reconstruction preparation stage."]}
    save_json(output / "frame_manifest.json", result)
    for capture in manifest["captures"]:
        try:
            result["captures"].append(process(capture, validation.parent, output))
        except Exception as exc:
            result["errors"].append({"capture_id": capture["capture_id"], "error": f"{type(exc).__name__}: {exc}"})
            print(f"{capture['capture_id']}: FAILED: {exc}", flush=True)
        save_json(output / "frame_manifest.json", result)
    result["status"] = "failed" if result["errors"] else "completed_with_quality_warnings" if any(c["status"] != "completed" for c in result["captures"]) else "completed"
    artifacts = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "frame_manifest.json":
            artifacts.append({"path": path.relative_to(output).as_posix(), "sha256": digest(path), "bytes": path.stat().st_size})
    result["artifacts"] = artifacts
    save_json(output / "frame_manifest.json", result)
    lines = ["# Reconstruction frame selection", "", f"Result: **{result['status']}**", "",
             "Original videos were read without edits. Selected frames are full-resolution decoded RGB PNGs.", "",
             "| Capture | Candidates | Selected | Time bins | Supported adjacent pairs | Largest gap (s) |",
             "|---|---:|---:|---:|---:|---:|"]
    for c in result["captures"]:
        lines.append(f"| {c['capture_id']} | {c['candidate_count']} | {c['selected_count']} | {len(c['covered_time_bins'])}/32 | {c['overlap_supported_pairs']}/{c['adjacent_pair_count']} | {c['largest_selected_gap_sec']:.3f} |")
    lines += ["", "## Why and what changed", "",
              "- Timestamp sampling distributes candidates through each recording and preserves links to validated source frames.",
              "- Sharpness and duplicate filtering avoid spending the reconstruction budget on blur or repeated views.",
              "- Thirty-two time bins distribute retained views. Feature matching flags disconnected neighbours; replacements and denser sampling are restricted to affected sections.",
              "- When temporal links are weak, all selected pairs are checked. Only the largest connected image-overlap component is retained; disconnected close-ups are excluded instead of forcing 32 frames.",
              "- Full-resolution PNGs, hashes, original frame IDs, exact PTS/time bases, timestamps, and identity pixel transforms preserve evidence for later depth lifting.",
              "- Candidate and selected contact sheets show coverage, rejections, and retained views. HTML links open selected originals.", "",
              "## Review status", "", "Visual review is recorded separately in visual_review.json after contact-sheet inspection. Physical parallax remains unverified until camera/geometry estimation."]
    for c in result["captures"]:
        lines += ["", f"## {c['capture_id']}", "", f"Decision counts: `{json.dumps(c['decision_counts'])}`.",
                  f"Uncovered time bins: `{c['uncovered_time_bins']}`.", f"Weak overlap pairs: `{c['weak_overlap_pairs']}`.",
                  f"Pairs with little image change: `{c['little_image_change_pairs']}`.",
                  f"Excluded disconnected views: `{c['excluded_disconnected_frame_indices']}`.",
                  f"[Interactive contact sheet]({c['capture_id']}/contact_sheet.html)"]
    lines += ["", "## Limitations", ""] + [f"- {item}" for item in result["limitations"]]
    if result["errors"]:
        lines += ["", "## Errors", "", json.dumps(result["errors"], indent=2)]
    (output / "selection_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved selection: {output}", flush=True)
    return int(bool(result["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
