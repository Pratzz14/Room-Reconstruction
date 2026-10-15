"""Align saved structural correspondences and compare saved object inventories."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reconstruction_v2.alignment import assess_alignment, ransac_similarity
from reconstruction_v2.change_detection import classify_changes, match_objects
from reconstruction_v2.core import read_json, write_json
from reconstruction_v2.object_contracts import ObjectInventory


def _objects(inventory, capture):
    from reconstruction_v2.comparison import matching_objects

    return matching_objects(inventory, capture)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "verify"):
        p = sub.add_parser(command)
        p.add_argument("--reconstruction", type=Path, required=True)
        p.add_argument("--objects", type=Path, required=True)
        p.add_argument("--refinement", type=Path)
        p.add_argument("--output", type=Path, required=True)
        if command == "run":
            p.add_argument(
                "--config",
                type=Path,
                default=Path(__file__).resolve().parents[1] / "configs/comparison.json",
            )
    align = sub.add_parser("align")
    align.add_argument(
        "--correspondences",
        type=Path,
        required=True,
        help="NPZ with source and target Nx3 arrays",
    )
    align.add_argument("--output", type=Path, required=True)
    align.add_argument("--threshold", type=float, required=True)
    align.add_argument("--max-residual", type=float, required=True)
    align.add_argument("--iterations", type=int, default=512)
    match = sub.add_parser("match")
    match.add_argument("--reconstruction", type=Path, required=True)
    match.add_argument("--objects", type=Path, required=True)
    match.add_argument("--output", type=Path, required=True)
    match.add_argument("--transform", type=Path)
    match.add_argument("--tolerance", type=float, required=True)
    match.add_argument("--position-scale", type=float, default=1.0)
    args = parser.parse_args()
    if args.command in ("run", "verify"):
        from reconstruction_v2.comparison import load_inventory, run, verify

        if args.command == "run":
            report = run(
                args.reconstruction,
                args.objects,
                args.refinement,
                args.output,
                read_json(args.config),
            )
        else:
            report = verify(
                load_inventory(args.reconstruction, args.objects, args.refinement),
                args.output,
            )
        print(
            json.dumps(
                {
                    "status": "completed" if args.command == "run" else "verified",
                    "alignment": report["alignment"]["quality"],
                    "identities": report["identity_counts"],
                    "changes": report["change_status"],
                    "provider_calls": 0,
                },
                indent=2,
            )
        )
        return 0
    if args.command == "align":
        with np.load(args.correspondences, allow_pickle=False) as values:
            source, target = values["source"], values["target"]
        result = ransac_similarity(
            source, target, threshold=args.threshold, iterations=args.iterations
        )
        quality = assess_alignment(
            source, target, result, max_residual=args.max_residual
        )
        write_json(
            args.output,
            {
                "schema_version": "alignment-v1",
                "direction": "after_to_before",
                "coordinate_frame": "before_world",
                "transform": {
                    key: value.tolist() if isinstance(value, np.ndarray) else value
                    for key, value in result["transform"].items()
                },
                "quality": quality.as_dict(),
                "inliers": result["inliers"].tolist(),
                "residuals": result["residuals"].tolist(),
            },
        )
        return 0 if quality.status != "rejected" else 2
    inventory = ObjectInventory(
        args.reconstruction, args.objects, require_complete=True
    )
    before, after = _objects(inventory, "before"), _objects(inventory, "after")
    transform, alignment_status = None, "unverified"
    if args.transform:
        record = read_json(args.transform)
        if record.get("direction") != "after_to_before":
            raise ValueError("Expected an After-to-Before transform")
        transform = record["transform"]
        alignment_status = record["quality"]["status"]
        from reconstruction_v2.comparison import source_binding

        if "sources" in record and record["sources"] != source_binding(inventory):
            raise ValueError("Transform belongs to different source artifacts")
        if "sources" not in record:
            alignment_status = "unverified"
    matching = match_objects(
        before,
        after,
        after_transform=transform,
        config={"position_scale": args.position_scale},
        alignment_accepted=alignment_status == "accepted",
    )
    result = classify_changes(
        before,
        after,
        matching,
        after_transform=transform,
        tolerance=args.tolerance,
        alignment_status=alignment_status,
    )
    result["matching"] = matching
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
