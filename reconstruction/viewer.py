"""Local saved-run viewer with separate captures, exact frustums and point evidence."""

import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .core import read_json, verify_capture


def ceiling_crop_indices(points, direction="-Y", percent=15):
    """Return original row IDs below an adjustable display plane; never alter geometry."""
    if direction not in ("-X", "+X", "-Y", "+Y", "-Z", "+Z") or not 0 <= percent <= 100:
        raise ValueError("Choose a signed ceiling axis and a crop percentage from 0 to 100")
    indices = np.arange(len(points), dtype=np.int64)
    if percent == 0 or not len(points):
        return indices
    axis = "XYZ".index(direction[1])
    heights = points[:, axis] * (-1 if direction[0] == "-" else 1)
    low, high = np.percentile(heights, [2, 98])
    if high <= low:
        return indices
    return indices[heights <= high - (high - low) * percent / 100]


def view(root, port=8090, objects=None):
    import viser
    root = Path(root).resolve()
    manifest = read_json(root / "run_manifest.json")
    available = [name for name in ("before", "after") if manifest["captures"][name]["status"] == "completed"]
    if not available:
        raise ValueError("No completed captures are available; run GPU inference first")
    for name in available:
        verify_capture(root, name)
    server = viser.ViserServer(host="127.0.0.1", port=port)
    title = "SYNTHETIC TEST FIXTURE" if manifest.get("synthetic") else "Independent room reconstruction"
    server.gui.add_markdown(f"## {title}\nRelative units. Before/After are separate coordinate frames. Real-scene accuracy is unvalidated.")
    capture = server.gui.add_dropdown("Capture", options=tuple(available), initial_value=available[0])
    frame_select = server.gui.add_dropdown("Frame", options=("loading",), initial_value="loading")
    point_id = server.gui.add_number("Saved cloud point ID", initial_value=0, min=0, step=1)
    point_button = server.gui.add_button("Show point source")
    camera_visibility = server.gui.add_checkbox("Show cameras", initial_value=True)
    camera_state = server.gui.add_markdown("Camera display: shown")
    point_size = server.gui.add_slider("Point size", min=0.0005, max=0.03, step=0.0005, initial_value=0.003)
    hide_ceiling = server.gui.add_checkbox("Hide ceiling", initial_value=False)
    ceiling_direction = server.gui.add_dropdown("Ceiling direction", options=("-Y", "+Y", "-Z", "+Z", "-X", "+X"),
                                                initial_value="-Y", visible=False)
    ceiling_amount = server.gui.add_slider("Ceiling crop (%)", min=0, max=100, step=1, initial_value=15, visible=False)
    focus = server.gui.add_button("Focus on cloud")
    status = server.gui.add_markdown("")
    evidence = server.gui.add_markdown("Choose a saved point ID to inspect its source pixel and supporting observations.")
    source_image = server.gui.add_image(np.zeros((8, 8, 3), np.uint8), label="Original frame")
    panel_image = server.gui.add_image(np.zeros((8, 8, 3), np.uint8), label="RGB / depth / confidence")
    state = {"handles": [], "camera_handles": [], "labels": {}, "busy": False,
             "ceiling_settings": {name: {"enabled": False, "direction": "-Y", "percent": 15} for name in available}}
    object_panel = None

    def context_visibility(isolated):
        if "cloud_handle" in state:
            state["cloud_handle"].visible = not isolated
        for handle in state["camera_handles"]:
            handle.visible = camera_visibility.value and not isolated

    if objects:
        from .object_contracts import ObjectInventory
        from .object_inspection import SavedObjectPanel
        object_panel = SavedObjectPanel(server, ObjectInventory(root, objects), context_visibility,
            lambda: {"crop_enabled": hide_ceiling.value, "crop_direction": ceiling_direction.value, "crop_percent": ceiling_amount.value},
            lambda: state["points"], lambda: point_size.value)
    else:
        server.gui.add_markdown("Object inventory unavailable. Supply --objects after object processing to inspect IDs and masks.")

    def load_image(path):
        with Image.open(path) as image:
            return np.array(image.convert("RGB"))

    def show_frame(index):
        frame = state["frames"][index]
        source_image.image = load_image(root / frame["image_path"])
        panel_image.image = load_image(root / capture.value / "previews" / f"{frame['frame_id']}_panel.png")

    def focus_cloud(client=None):
        points = state["points"][state["retained_indices"]]
        if not len(points):
            return
        low, high = np.percentile(points, [2, 98], axis=0)
        center = (low + high) / 2
        distance = max(float(np.linalg.norm(high - low)), 0.1)
        clients = [client] if client else list(server.get_clients().values())
        for connection in clients:
            connection.camera.look_at = tuple(center)
            connection.camera.position = tuple(center + [0, -distance * 0.2, -distance])
            connection.camera.up_direction = (0, -1, 0)

    def refresh_cloud():
        points, colors = state["points"], state["colors"]
        retained = (ceiling_crop_indices(points, ceiling_direction.value, ceiling_amount.value)
                    if hide_ceiling.value else np.arange(len(points), dtype=np.int64))
        maximum = read_json(root / "config.json")["viewer"]["max_points"]
        sampled = retained[np.unique(np.linspace(0, len(retained) - 1, min(len(retained), maximum)).astype(int))] if len(retained) else retained
        if "cloud_handle" in state:
            previous = state["cloud_handle"]
            previous.remove()
            state["handles"].remove(previous)
        handle = server.scene.add_point_cloud("/capture/points", points[sampled], colors[sampled],
                                             point_size=point_size.value, precision="float32")
        state["cloud_handle"] = handle
        state["handles"].append(handle)
        state["retained_indices"], state["display_indices"] = retained, sampled
        name = capture.value
        crop_note = (f" {len(retained):,} remain after the ceiling crop. Manual display cut; change direction if needed."
                     if hide_ceiling.value else "")
        status.content = (f"**{name}** · {manifest['captures'][name]['coordinate_frame']} · "
                          f"{len(points):,} saved points; {len(sampled):,} shown.{crop_note}\n\n"
                          "All saved point IDs remain accessible through provenance.")

    def load_capture():
        state["busy"] = True
        try:
            for handle in state["handles"]:
                handle.remove()
            state["handles"], state["camera_handles"] = [], []
            state.pop("cloud_handle", None)
            name = capture.value
            settings = state["ceiling_settings"][name]
            hide_ceiling.value = settings["enabled"]
            ceiling_direction.value = settings["direction"]
            ceiling_amount.value = settings["percent"]
            ceiling_direction.visible = ceiling_amount.visible = hide_ceiling.value
            state["frames"] = read_json(root / name / "frames.json")["frames"]
            state["cameras"] = read_json(root / name / "cameras.json")
            with np.load(root / name / "cloud.npz", allow_pickle=False) as data:
                state["points"], state["colors"] = data["points"], data["colors"]
            with np.load(root / name / "observations.npz", allow_pickle=False) as data:
                state["observations"] = {key: data[key] for key in data.files}
            points = state["points"]
            refresh_cloud()
            scale = max(float(np.linalg.norm(np.percentile(points, 98, axis=0) - np.percentile(points, 2, axis=0))) * 0.03, 0.001)
            for i, camera in enumerate(state["cameras"]):
                width, height = camera["model_width"], camera["model_height"]
                pixel_corners = np.array([[-0.5, -0.5, 1], [width - 0.5, -0.5, 1], [width - 0.5, height - 0.5, 1], [-0.5, height - 0.5, 1]])
                corners = pixel_corners @ np.linalg.inv(np.array(camera["intrinsics"])).T * scale
                lines = np.concatenate((np.stack((np.zeros_like(corners), corners), axis=1), np.stack((corners, np.roll(corners, -1, axis=0)), axis=1)))
                pose = np.array(camera["camera_to_world"])
                lines = lines @ pose[:3, :3].T + pose[:3, 3]
                camera_handle = server.scene.add_line_segments(f"/capture/camera_{i}", lines.astype(np.float32), (80, 110, 230), thickness=1.5, thickness_units="screen", visible=camera_visibility.value)
                state["camera_handles"].append(camera_handle)
                state["handles"].append(camera_handle)
            positions = np.array([np.array(c["camera_to_world"])[:3, 3] for c in state["cameras"]])
            path = server.scene.add_line_segments("/capture/camera_path", np.stack((positions[:-1], positions[1:]), axis=1).astype(np.float32), (255, 130, 20), thickness=2, thickness_units="screen", visible=camera_visibility.value)
            state["camera_handles"].append(path); state["handles"].append(path)
            state["labels"] = {f"{frame['decoded_frame_index']} | {frame['elapsed_timestamp_sec']:.3f}s": i for i, frame in enumerate(state["frames"])}
            frame_select.options = tuple(state["labels"])
            frame_select.value = next(iter(state["labels"]))
            point_id.max = len(points) - 1
            point_id.value = 0
            evidence.content = "Choose a saved point ID to inspect source evidence."
            show_frame(0)
            if object_panel:
                object_panel.capture(name)
            focus_cloud()
        finally:
            state["busy"] = False

    @capture.on_update
    def _capture(event):
        load_capture()

    @frame_select.on_update
    def _frame(event):
        if not state["busy"]:
            show_frame(state["labels"][frame_select.value])

    @camera_visibility.on_update
    def _cameras(event):
        context_visibility(bool(object_panel and object_panel.inspection.isolate))
        camera_state.content = "Camera display: " + ("shown" if camera_visibility.value else "hidden")

    @point_size.on_update
    def _size(event):
        if "cloud_handle" in state:
            state["cloud_handle"].point_size = point_size.value
        if object_panel:
            object_panel.refresh()

    @hide_ceiling.on_update
    @ceiling_direction.on_update
    @ceiling_amount.on_update
    def _ceiling(event):
        if state["busy"]:
            return
        ceiling_direction.visible = ceiling_amount.visible = hide_ceiling.value
        state["ceiling_settings"][capture.value] = {"enabled": hide_ceiling.value,
                                                   "direction": ceiling_direction.value,
                                                   "percent": ceiling_amount.value}
        refresh_cloud()
        if object_panel:
            object_panel.refresh()

    @focus.on_click
    def _focus(event):
        focus_cloud(event.client)

    @point_button.on_click
    def _source(event):
        index = int(point_id.value)
        observations = state["observations"]
        if not 0 <= index < len(state["points"]):
            return
        representative = int(observations["representative_observation"][index])
        frame_index = int(observations["observation_frame_index"][representative])
        frame = state["frames"][frame_index]
        frame_select.value = next(label for label, i in state["labels"].items() if i == frame_index)
        show_frame(frame_index)
        x, y = observations["original_pixel_xy"][representative]
        with Image.open(root / frame["image_path"]) as image:
            image = image.convert("RGB")
            draw = ImageDraw.Draw(image)
            radius = max(image.width / 150, 3)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline="cyan", width=3)
            source_image.image = np.array(image)
        offsets = observations["point_observation_offsets"]
        supporting = observations["point_observation_indices"][offsets[index]:offsets[index + 1]]
        support_ids = np.unique(observations["observation_frame_index"][supporting])
        pi = int(observations["model_pixel_index"][representative])
        width = state["cameras"][frame_index]["model_width"]
        source = read_json(root / capture.value / "source.json")
        # Markdown hard breaks keep complete identifiers readable in the narrow sidebar.
        def identifier_lines(value):
            return "  \n".join(f"`{value[start:start + 24]}`" for start in range(0, len(value), 24))

        evidence.content = (f"**Cloud point {index}** · {len(supporting)} observations across {len(support_ids)} frames\n\n"
                            f"Representative: frame {frame['decoded_frame_index']} at {frame['elapsed_timestamp_sec']:.3f}s\n\n"
                            f"Model pixel: ({pi % width}, {pi // width}); original pixel: ({x:.2f}, {y:.2f})\n\n"
                            f"Z depth: {float(observations['depth'][representative]):.5g} relative; confidence: {float(observations['confidence'][representative]):.5g}\n\n"
                            f"Frame ID (join lines):\n\n{identifier_lines(frame['frame_id'])}\n\n"
                            f"Video SHA-256 (join lines):\n\n{identifier_lines(source['source_video_sha256'])}")

    @server.on_client_connect
    def _connect(client):
        if "points" in state:
            focus_cloud(client)

    load_capture()
    print(f"Viewer: http://127.0.0.1:{port} (Ctrl+C to stop)", flush=True)
    while True:
        time.sleep(0.25)
