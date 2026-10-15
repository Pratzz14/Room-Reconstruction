"""Prepare, run, verify, package and inspect independent v2 reconstructions."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reconstruction_v2 import core


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--frames", type=Path, default=root / "outputs/frame_selection_connected")
    p.add_argument("--output", type=Path, default=root / "outputs/reconstruction")
    p.add_argument("--config", type=Path, default=root / "configs/reconstruction.json")
    p.add_argument("--lock", type=Path, default=root / "configs/models.lock.json")
    for name in ("run", "worker"):
        p = commands.add_parser(name)
        p.add_argument("--run", type=Path, required=True)
        p.add_argument("--source", type=Path, required=True)
        p.add_argument("--checkpoint", type=Path, required=True)
        if name == "run":
            p.add_argument("--resume", action="store_true")
            p.add_argument("--capture", choices=("before", "after"))
        else:
            p.add_argument("--capture", choices=("before", "after"), required=True)
    for name in ("verify", "view", "sync-view", "object-view", "pack", "preview"):
        p = commands.add_parser(name)
        p.add_argument("--run", type=Path, required=True)
        if name == "pack":
            p.add_argument("--archive", type=Path, required=True)
        if name == "view":
            p.add_argument("--port", type=int, default=8090)
            p.add_argument("--objects", type=Path)
        if name == "sync-view":
            p.add_argument("--port", type=int, default=8092)
            p.add_argument("--objects", type=Path)
            p.add_argument("--before-video", type=Path)
            p.add_argument("--after-video", type=Path)
        if name == "object-view":
            p.add_argument("--port", type=int, default=8094)
            p.add_argument("--objects", type=Path, required=True)
            p.add_argument("--refinement", type=Path)
            p.add_argument("--comparison", type=Path)
        if name == "preview":
            p.add_argument("--port", type=int, default=8091)
    p = commands.add_parser("import")
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = core.prepare(args.frames, args.output, core.read_json(args.config), core.read_json(args.lock))
    elif args.command in ("run", "worker"):
        from reconstruction_v2.inference import run, worker
        if args.command == "worker":
            worker(args.run, args.capture, args.source, args.checkpoint)
            return 0
        result = run(args.run, args.source, args.checkpoint, resume=args.resume,
                     captures=(args.capture,) if args.capture else ("before", "after"))
        print(json.dumps({"status": result["status"], "captures": {n: r["status"] for n, r in result["captures"].items()}}, indent=2))
        return int(any(r["status"] == "failed" for r in result["captures"].values()))
    elif args.command == "verify":
        result = {name: core.verify_capture(args.run, name) for name in ("before", "after")}
    elif args.command in ("pack", "import"):
        from reconstruction_v2.archive import import_archive, pack
        result = pack(args.run, args.archive) if args.command == "pack" else import_archive(args.archive, args.output)
    elif args.command == "view":
        from reconstruction_v2.viewer import view
        view(args.run, args.port, objects=args.objects)
        return 0
    elif args.command == "sync-view":
        from reconstruction_v2.sync_viewer import synchronized_view
        synchronized_view(args.run, args.port, args.before_video, args.after_video, objects=args.objects)
        return 0
    elif args.command == "object-view":
        from reconstruction_v2.inspection_viewer import object_view
        object_view(args.run, args.objects, args.port, refinement=args.refinement, comparison=args.comparison)
        return 0
    else:
        from functools import partial
        from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
        print(f"Depth previews: http://127.0.0.1:{args.port}/before/previews/depth_preview.html", flush=True)
        ThreadingHTTPServer(("127.0.0.1", args.port), partial(SimpleHTTPRequestHandler, directory=str(args.run.resolve()))).serve_forever()
        return 0
    print(json.dumps({"status": result.get("status", "verified"), "output": str(getattr(args, "output", getattr(args, "run", ""))),
                      "result": result if args.command in ("verify", "pack") else None}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
