"""Refine saved inventories using bounded Gemini calls, then inspect derived objects."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reconstruction_v2.core import read_json
from reconstruction_v2.refinement_evidence import prepare
from reconstruction_v2.refinement_gemini import (
    exclusive,
    validate_config,
    validate_output,
)


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "run", "verify"):
        p = commands.add_parser(command)
        p.add_argument("--reconstruction", type=Path, required=True)
        p.add_argument("--objects", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument(
            "--config", type=Path, default=project / "configs/refinement.json"
        )
        if command == "run":
            p.add_argument(
                "--baseline",
                type=Path,
                help="Reuse a verified previous refinement of these exact source artifacts",
            )
            p.add_argument(
                "--answer-file",
                type=Path,
                help="Import a saved model JSON/receipt, normalize it, and rebuild without another generation",
            )
            p.add_argument(
                "--review-notes",
                type=Path,
                help="Optional evidence-linked operator checks for the bounded follow-up",
            )
            p.add_argument(
                "--credential-source",
                type=Path,
                default=project.parent
                / "Room Reconstruction/scripts/test_gemini_vlm.py",
            )
            p.add_argument(
                "--offline",
                action="store_true",
                help="Rebuild from saved answer.json; no provider calls",
            )
            p.add_argument(
                "--no-followup",
                action="store_true",
                help="Keep initial result without the optional targeted call",
            )
            p.add_argument(
                "--retry-failed",
                action="store_true",
                help="Explicitly retry a failed/unknown request within the original attempt budget",
            )
    args = parser.parse_args()
    try:
        config = validate_config(read_json(args.config))
        validate_output(args.reconstruction, args.objects, args.output)
        if args.command == "prepare":
            with exclusive(args.output / ".refinement.lock"):
                _, prepared = prepare(
                    args.reconstruction, args.objects, args.output, config
                )
            result = {
                "status": "prepared",
                "observations": sum(
                    s["observations"] for s in prepared["summaries"].values()
                ),
                "images": len(prepared["images"]),
                "generation_calls": 0,
            }
        elif args.command == "verify":
            from reconstruction_v2.object_contracts import ObjectInventory
            from reconstruction_v2.refined_inventory import RefinedInventory

            raw = ObjectInventory(
                args.reconstruction, args.objects, require_complete=True
            )
            refined = RefinedInventory(raw, args.output)
            result = {
                "status": "verified",
                "synthetic": refined.report["synthetic"],
                "captures": refined.report["captures"],
                "identity_proposals": len(refined.report["links"]),
                "change_status": "not_evaluated",
            }
        else:
            from reconstruction_v2.refinement import run

            report = run(
                args.reconstruction,
                args.objects,
                args.output,
                config,
                args.credential_source,
                offline=args.offline,
                followup=not args.no_followup,
                retry_failed=args.retry_failed,
                review_notes=args.review_notes,
                baseline=args.baseline,
                answer_file=args.answer_file,
            )
            result = {
                "status": report["status"],
                "captures": report["captures"],
                "identity_proposals": len(report["links"]),
                "repair_tasks": report["repair_tasks"],
                "api_usage": report["api_usage"],
                "change_status": "not_evaluated",
            }
        print(json.dumps(result, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001 -- CLI failure boundary, no automatic retries
        print(f"Refinement stopped: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
