"""Run, verify and transport capture-local object inventories."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reconstruction.core import read_json


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("resolve-models")
    p.add_argument(
        "--source", type=Path, default=project / "configs/objects.models.lock.json"
    )
    p.add_argument("--output", type=Path, required=True)
    for name in ("run", "verify", "pack", "import"):
        p = commands.add_parser(name)
        p.add_argument("--reconstruction", type=Path, required=True)
        if name in ("verify", "pack"):
            p.add_argument("--objects", type=Path, required=True)
        if name in ("run", "import"):
            p.add_argument("--output", type=Path, required=True)
        if name in ("pack", "import"):
            p.add_argument("--archive", type=Path, required=True)
        if name == "run":
            p.add_argument(
                "--config", type=Path, default=project / "configs/objects.json"
            )
            p.add_argument("--lock", type=Path, required=True)
            p.add_argument("--resume", action="store_true")
            p.add_argument("--capture", choices=("before", "after"))
            p.add_argument("--frames-per-capture", type=int)
    args = parser.parse_args()
    if args.command == "resolve-models":
        from reconstruction.object_models import resolve_models

        result = resolve_models(args.source, args.output)
    elif args.command == "run":
        from reconstruction.object_pipeline import run

        result = run(
            args.reconstruction,
            args.output,
            read_json(args.config),
            read_json(args.lock),
            resume=args.resume,
            captures=(args.capture,) if args.capture else ("before", "after"),
            frame_limit=args.frames_per_capture,
        )
        print(json.dumps(result, indent=2))
        return int(result["status"] == "failed")
    elif args.command == "verify":
        from reconstruction.object_contracts import ObjectInventory

        inventory = ObjectInventory(args.reconstruction, args.objects)
        result = {
            "verified_captures": {
                name: len(data["objects"]) for name, data in inventory.captures.items()
            },
            "real_review": "Separate visual acceptance required",
            "status": inventory.manifest["status"],
        }
        if not inventory.captures:
            raise ValueError("No completed inventory")
    else:
        from reconstruction.object_archive import import_archive, pack

        result = (
            pack(args.reconstruction, args.objects, args.archive)
            if args.command == "pack"
            else import_archive(args.archive, args.reconstruction, args.output)
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
