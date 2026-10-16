"""Object list and complete saved 3D scene, without source-video dependencies."""

from pathlib import Path

from .core import read_json, verify_capture
from .object_contracts import ObjectInventory
from .sync_viewer import _SyncState, check_viewer_ports, serve_viewer


def object_view(
    root, objects, port=8094, refinement=None, comparison=None, report=None, review=None
):
    if (report or review) and not comparison:
        raise ValueError("--report/--review requires --comparison")
    if report and not review:
        review = read_json(Path(report) / "report.json").get("review_source")
    root = Path(root).resolve()
    check_viewer_ports(port, scene_count=2 if comparison else 1)
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
    if comparison:
        from .comparison import ComparisonInventory

        if refinement:
            from .refined_inventory import RefinedInventory

            inventory = RefinedInventory(inventory, refinement)
        reviewed = None
        if review:
            from .change_review import verify as verify_review

            reviewed = verify_review(review, inventory, comparison)
        if report:
            from .change_report import verify_report

            print("Verifying bound change report...", flush=True)
            verify_report(report, inventory, comparison, review=review)
            if (
                review
                and read_json(Path(report) / "report.json")["schema_version"]
                != "change-report-v2"
            ):
                raise ValueError("Use a report generated with the same --review")
        inventory = ComparisonInventory(inventory, comparison)
        if reviewed:
            from .change_review import apply_to_display

            inventory = apply_to_display(inventory, reviewed)
    elif refinement:
        from .refined_inventory import InventoryModes

        inventory = InventoryModes(inventory, refinement)
    maximum = (
        read_json(root / "config.json").get("viewer", {}).get("max_points", 200000)
    )
    if comparison:
        from .comparison_viewer import serve_comparison

        serve_comparison(root, inventory, comparison, port, maximum, report=report)
        return
    state = _SyncState(root, captures, maximum, inventory, static=True)
    serve_viewer(state, port, page_name="objects.html", title="Object inspection")
