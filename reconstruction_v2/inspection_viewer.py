"""Object list and complete saved 3D scene, without source-video dependencies."""

from pathlib import Path

from .core import read_json, verify_capture
from .object_contracts import ObjectInventory
from .sync_viewer import _SyncState, check_viewer_ports, serve_viewer


def object_view(root, objects, port=8094):
    root = Path(root).resolve()
    check_viewer_ports(port)
    manifest = read_json(root / "run_manifest.json")
    captures = [
        name
        for name in ("before", "after")
        if manifest["captures"][name]["status"] == "completed"
    ]
    if not captures:
        raise ValueError("No completed captures are available")
    for name in captures:
        print(f"Verifying saved {name} reconstruction...", flush=True)
        verify_capture(root, name)
    print("Verifying object inventories...", flush=True)
    inventory = ObjectInventory(root, objects)
    maximum = (
        read_json(root / "config.json").get("viewer", {}).get("max_points", 200000)
    )
    state = _SyncState(root, captures, maximum, inventory, static=True)
    serve_viewer(state, port, page_name="objects.html", title="Object inspection")
