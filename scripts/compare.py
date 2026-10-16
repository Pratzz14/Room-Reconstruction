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
    for command in ("review", "verify-review"):
        p = sub.add_parser(command)
        p.add_argument("--reconstruction", type=Path, required=True)
        p.add_argument("--objects", type=Path, required=True)
        p.add_argument("--refinement", type=Path)
        p.add_argument("--comparison", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        if command == "review":
            p.add_argument(
                "--config",
                type=Path,
                default=Path(__file__).resolve().parents[1]
                / "configs/change_review.json",
            )
            p.add_argument("--prepare-only", action="store_true")
            p.add_argument(
                "--prompt-profile",
                choices=("legacy", "visual-grounding-v2", "visual-grounding-native-v3"),
                default="legacy",
            )
            p.add_argument(
                "--execution-prompt-profile",
                choices=("legacy", "visual-grounding-v2", "visual-grounding-native-v3"),
            )
            p.add_argument(
                "--confirm-ids",
                nargs="+",
                help="Prioritize one allowed follow-up for already reviewed identities",
            )
            p.add_argument(
                "--answers",
                type=Path,
                help="Import saved structured answers without provider calls",
            )
            p.add_argument("--retry-failed", action="store_true")
            p.add_argument(
                "--execution-config",
                type=Path,
                help="Explicit verified model/quota profile for unfinished requests; preserves saved answers",
            )
            p.add_argument(
                "--batch-size-limit",
                type=int,
                choices=range(1, 5),
                help="Use smaller execution batches when resuming after provider capacity failures",
            )
            p.add_argument(
                "--evidence-report",
                type=Path,
                help="Reuse a source-bound original report without rerunning evidence selection",
            )
            p.add_argument(
                "--credential-source",
                type=Path,
                help="Read an existing local Gemini credential without executing or copying its script",
            )
    for command in ("report", "verify-report", "pack-report", "refresh-report"):
        p = sub.add_parser(command)
        p.add_argument("--reconstruction", type=Path, required=True)
        p.add_argument("--objects", type=Path, required=True)
        p.add_argument("--refinement", type=Path)
        p.add_argument("--comparison", type=Path, required=True)
        p.add_argument("--review", type=Path)
        p.add_argument(
            "--output", type=Path, required=True, help="Separate report directory"
        )
        if command == "report":
            p.add_argument("--config", type=Path)
            p.add_argument("--viewer-url", default="http://127.0.0.1:8094/")
        if command == "pack-report":
            p.add_argument("--archive", type=Path, required=True)
        if command == "refresh-report":
            p.add_argument("--source-report", type=Path, required=True)
            p.add_argument("--viewer-url")
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
    if args.command in ("review", "verify-review"):
        from reconstruction_v2.change_review import run as review_run
        from reconstruction_v2.change_review import verify as verify_review
        from reconstruction_v2.comparison import load_inventory

        inventory = load_inventory(args.reconstruction, args.objects, args.refinement)
        result = (
            review_run(
                inventory,
                args.comparison,
                args.output,
                read_json(args.config),
                prepare_only=args.prepare_only,
                answers=args.answers,
                retry_failed=args.retry_failed,
                evidence_report=args.evidence_report,
                credential_source=args.credential_source,
                batch_limit=args.batch_size_limit,
                prompt_profile=args.prompt_profile,
                execution_prompt_profile=args.execution_prompt_profile,
                confirm_ids=args.confirm_ids,
                execution_config=read_json(args.execution_config)
                if args.execution_config
                else None,
            )
            if args.command == "review"
            else verify_review(args.output, inventory, args.comparison)
        )
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in ("status", "coverage", "usage", "stopped_reason")
                },
                indent=2,
            )
        )
        return 2 if result["stopped_reason"] else 0
    if args.command in ("report", "verify-report", "pack-report", "refresh-report"):
        from reconstruction_v2.change_report import (
            build_report,
            pack_report,
            refresh_report,
            verify_report,
        )
        from reconstruction_v2.comparison import load_inventory

        inventory = load_inventory(args.reconstruction, args.objects, args.refinement)
        if args.command == "pack-report":
            result = pack_report(
                args.output,
                args.archive,
                inventory,
                args.comparison,
                review=args.review,
            )
        else:
            if args.command == "report":
                config = read_json(args.config) if args.config else {}
                config.setdefault("viewer_url", args.viewer_url)
                report = build_report(
                    inventory, args.comparison, args.output, config, review=args.review
                )
            elif args.command == "refresh-report":
                report = refresh_report(
                    args.source_report,
                    args.output,
                    inventory,
                    args.comparison,
                    viewer_url=args.viewer_url,
                    review=args.review,
                )
            else:
                report = verify_report(
                    args.output, inventory, args.comparison, review=args.review
                )
            result = {
                "status": "completed"
                if args.command in ("report", "refresh-report")
                else "verified",
                "events": report["scene"]["event_count"],
                "counts": report["scene"]["change_counts"],
                "output": str(args.output),
                "provider_calls": 0,
            }
        print(json.dumps(result, indent=2))
        return 0
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
                    "change_counts": report.get("change_counts", {}),
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
