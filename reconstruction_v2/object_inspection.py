"""Shared CPU object selection, exact highlights, camera fitting and evidence."""

import io
import math

import numpy as np
from PIL import Image

from .core import safe_path
from .sync import fit_room


def box_lines(bounds):
    low, high = np.asarray(bounds)
    corners = np.array(
        [[high[k] if i & (1 << k) else low[k] for k in range(3)] for i in range(8)]
    )
    return np.array(
        [
            [corners[i], corners[i ^ (1 << axis)]]
            for i in range(8)
            for axis in range(3)
            if not i & (1 << axis)
        ],
        np.float32,
    )


class ObjectInspection:
    def __init__(self, inventory, scene=None):
        self.inventory, self.scene = inventory, scene
        self.capture = None
        self.selected = None
        self.identifier = None
        self.geometry = None
        self.isolate = False
        self.mode = "full"
        self.show_centroid = self.show_bounds = True
        self.crop_selection = False
        self.focus_active = False
        self.handles = []
        self.visible_count = 0
        self.evidence_cache = {}

    def clear(self, capture=None):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.selected = self.identifier = self.geometry = None
        self.isolate = self.focus_active = False
        self.mode = "full"
        if capture is not None:
            self.capture = capture

    def select(self, capture, identifier):
        # Validate before changing current state.
        obj, geometry = self.inventory.object(capture, identifier)
        self.clear(capture)
        self.selected, self.identifier, self.geometry = obj, identifier, geometry

    def options(self, values):
        allowed = {"mode", "isolate", "show_centroid", "show_bounds", "crop_selection"}
        if not isinstance(values, dict) or set(values) - allowed:
            raise ValueError("Invalid object display options")
        for key, value in values.items():
            if (key == "mode" and value not in ("full", "playback")) or (
                key != "mode" and type(value) is not bool
            ):
                raise ValueError("Invalid object display option")
        if values.get("isolate") and self.selected is None:
            raise ValueError("Select an object before isolating")
        for key, value in values.items():
            setattr(self, key, value)

    def snapshot(self):
        return {
            "selected": self.selected,
            "mode": self.mode,
            "isolate": self.isolate,
            "show_centroid": self.show_centroid,
            "show_bounds": self.show_bounds,
            "crop_selection": self.crop_selection,
            "visible_points": self.visible_count,
            "note": "Full saved object; all supporting views; ceiling crop bypassed"
            if self.selected and self.mode == "full" and not self.crop_selection
            else "Saved full-object centroid; visible geometry follows selected support"
            if self.selected
            else "Choose an object",
        }

    def refresh(self, frame_index=None, crop_settings=None, point_size=0.003):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.visible_count = 0
        if self.selected is None:
            return
        geometry = self.geometry
        selected = np.arange(len(geometry["points"]))
        if self.mode == "playback" and frame_index is not None:
            selected = selected[geometry["first_frame"][selected] <= frame_index]
        if self.crop_selection and crop_settings and crop_settings["crop_enabled"]:
            # Clip with the ROOM plane, not a separately computed object-relative plane.
            plane = crop_settings.get("plane")
            if plane is not None:
                axis, sign, limit = plane
                selected = selected[geometry["points"][selected, axis] * sign <= limit]
        self.visible_count = len(selected)
        if not len(selected) or self.scene is None:
            return
        maximum = self.inventory.config["viewer"]["max_object_points"]
        sampled = selected[
            np.linspace(0, len(selected) - 1, min(maximum, len(selected))).astype(int)
        ]
        self.handles.append(
            self.scene.add_point_cloud(
                "/object/selected",
                geometry["points"][sampled],
                (30, 230, 255),
                point_size=point_size * 1.3,
                precision="float32",
            )
        )
        if self.show_centroid:
            self.handles.append(
                self.scene.add_point_cloud(
                    "/object/centroid",
                    np.asarray([self.selected["centroid"]], np.float32),
                    (255, 215, 80),
                    point_size=point_size * 5,
                    precision="float32",
                )
            )
            self.handles.append(
                self.scene.add_label(
                    "/object/label",
                    self.selected["label"] + " · observed-surface centroid",
                    position=tuple(self.selected["centroid"]),
                )
            )
        if self.show_bounds:
            self.handles.append(
                self.scene.add_line_segments(
                    "/object/bounds",
                    box_lines(self.selected["bounds"]),
                    (30, 230, 255),
                    thickness=1.5,
                    thickness_units="screen",
                )
            )

    def fit(self, client):
        if self.selected is None:
            return None
        camera = client.camera
        fitted = fit_room(np.array(self.selected["bounds"]), camera.aspect)
        if fitted is None:
            return None
        center = np.array(fitted["look_at"])
        direction = np.array(camera.position) - np.array(camera.look_at)
        length = float(np.linalg.norm(direction))
        if length > 1e-8:
            # A sphere fit remains valid after preserving an arbitrary user viewing direction.
            half_angle = min(
                fitted["fov"] / 2,
                math.atan(math.tan(fitted["fov"] / 2) * camera.aspect),
            )
            distance = fitted["radius"] * 1.15 / max(math.sin(half_angle), 1e-6)
            fitted["position"] = (center + direction / length * distance).tolist()
            fitted["far"] = distance + fitted["radius"] * 10
        with client.atomic():
            camera.fov = fitted["fov"]
            camera.position = tuple(fitted["position"])
            camera.look_at = tuple(center)
            camera.near, camera.far = fitted["near"], fitted["far"]
            camera.min_orbit_distance = fitted["radius"] * 0.01
            camera.max_orbit_distance = fitted["far"] * 10
        self.focus_active = True
        return fitted

    def evidence_record(self, capture, observation_id):
        data = self.inventory.captures.get(capture)
        if data is None or observation_id not in data["index"]:
            raise ValueError("Unknown source observation in capture")
        return data["observations"][data["index"][observation_id]]

    def evidence(self, capture, observation_id, kind="overlay"):
        key = (capture, observation_id, kind)
        if key in self.evidence_cache:
            return self.evidence_cache[key]
        if kind not in ("overlay", "crop", "original"):
            raise ValueError("Unsupported evidence image")
        record = self.evidence_record(capture, observation_id)
        data = self.inventory.captures[capture]
        frame = data["frames"][record["frame_index"]]
        with Image.open(
            safe_path(self.inventory.reconstruction, frame["image_path"])
        ) as image:
            rgb = np.array(image.convert("RGB"))
        with Image.open(
            safe_path(self.inventory.root, record["mask_path"])
        ) as mask_image:
            mask = np.asarray(mask_image) > 0
        if kind != "original":
            rgb[mask] = (
                rgb[mask].astype(float) * 0.5 + np.array([0, 220, 255]) * 0.5
            ).astype(np.uint8)
        image = Image.fromarray(rgb)
        if kind == "crop" and mask.any():
            y, x = np.where(mask)
            pad = max(4, int(max(x.max() - x.min(), y.max() - y.min()) * 0.1))
            image = image.crop(
                (
                    max(0, int(x.min()) - pad),
                    max(0, int(y.min()) - pad),
                    min(image.width, int(x.max()) + pad + 1),
                    min(image.height, int(y.max()) + pad + 1),
                )
            )
        image.thumbnail((1280, 960))
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        if len(self.evidence_cache) >= 12:
            self.evidence_cache.pop(next(iter(self.evidence_cache)))
        self.evidence_cache[key] = stream.getvalue()
        return self.evidence_cache[key]


def crop_plane(points, settings):
    if not settings.get("crop_enabled") or not len(points):
        return None
    direction = settings["crop_direction"]
    axis, sign = "XYZ".index(direction[1]), -1 if direction[0] == "-" else 1
    low, high = np.percentile(points[:, axis] * sign, [2, 98])
    limit = low + (high - low) * (1 - settings["crop_percent"] / 100)
    if settings["crop_percent"] == 100:
        limit = -float("inf")
    return axis, sign, limit


class SavedObjectPanel:
    """Viser controls using the same selection/geometry controller as synchronized replay."""

    def __init__(self, server, inventory, context, settings, points, size):
        self.server, self.context, self.settings, self.points, self.size = (
            server,
            context,
            settings,
            points,
            size,
        )
        self.inspection = ObjectInspection(inventory, server.scene)
        self.busy, self.capture_name = False, None
        gui = server.gui
        gui.add_markdown("### Objects · capture-local inspection")
        self.search = gui.add_text("Search objects", initial_value="")
        self.selection = gui.add_dropdown(
            "Object ID and label", options=("None",), initial_value="None"
        )
        self.focus_button = gui.add_button("Focus object", disabled=True)
        self.clear_button = gui.add_button("Clear object selection")
        self.isolation = gui.add_checkbox("Isolate object", initial_value=False)
        self.centroid = gui.add_checkbox("Show object centroid", initial_value=True)
        self.bounds = gui.add_checkbox("Show object bounds", initial_value=True)
        self.crop = gui.add_checkbox(
            "Apply ceiling crop to object", initial_value=False
        )
        self.details = gui.add_markdown("Choose an object")
        self.evidence_select = gui.add_dropdown(
            "Object supporting frame", options=("None",), initial_value="None"
        )
        self.unresolved = gui.add_dropdown(
            "Unresolved 2D observation", options=("None",), initial_value="None"
        )
        self.image = gui.add_image(
            np.zeros((8, 8, 3), np.uint8), label="Object mask on original frame"
        )
        self.crop_image = gui.add_image(
            np.zeros((8, 8, 3), np.uint8), label="Object close-up"
        )
        self.evidence_details = gui.add_markdown("")
        self.labels, self.evidence_labels, self.unresolved_labels = {}, {}, {}
        self.search.on_update(lambda _: self.filter())
        self.selection.on_update(self.select)
        self.clear_button.on_click(lambda _: self.clear())
        self.focus_button.on_click(self.focus)
        self.evidence_select.on_update(
            lambda _: self.show_evidence(
                self.evidence_labels.get(self.evidence_select.value)
            )
        )
        self.unresolved.on_update(self.show_unresolved)
        for control in (self.isolation, self.centroid, self.bounds, self.crop):
            control.on_update(lambda _: self.refresh())

    def capture(self, name):
        self.busy = True
        try:
            self.capture_name = name
            self.inspection.clear(name)
            self.isolation.value = False
            self.search.value = ""
            self.selection.value = "None"
            self.evidence_select.options = self.unresolved.options = ("None",)
            self.evidence_select.value = self.unresolved.value = "None"
            self.evidence_labels = {}
            listing = self.inspection.inventory.listing(name)
            self.labels = {
                f"{o['label']} · {o['id']}": o["id"] for o in listing["objects"]
            }
            self.unresolved_labels = {
                f"{o['label']} · {o['id']}": o["id"] for o in listing["unresolved"]
            }
            self.unresolved.options = ("None", *self.unresolved_labels)
            self.selection.options = ("None", *self.labels)
            self.details.content = f"{len(self.labels)} objects; {len(self.unresolved_labels)} unresolved observations. {listing['status']}."
            self.image.image = self.crop_image.image = np.zeros((8, 8, 3), np.uint8)
            self.evidence_details.content = ""
            self.focus_button.disabled = True
        finally:
            self.busy = False
        self.context(False)

    def filter(self):
        if self.busy:
            return
        query = self.search.value.lower()
        previous = self.selection.value
        options = ("None", *(label for label in self.labels if query in label.lower()))
        self.busy = True
        self.selection.options = options
        self.selection.value = previous if previous in options else "None"
        self.busy = False

    def clear(self):
        self.capture(self.capture_name)

    def select(self, _):
        if self.busy:
            return
        identifier = self.labels.get(self.selection.value)
        if identifier is None:
            self.clear()
            return
        self.inspection.select(self.capture_name, identifier)
        obj = self.inspection.selected
        data = self.inspection.inventory.captures[self.capture_name]
        self.evidence_labels = {
            f"{data['observations'][data['index'][x]]['timestamp_sec']:.3f}s · {x}": x
            for x in obj["observation_ids"]
        }
        self.busy = True
        self.isolation.value = False
        self.evidence_select.options = ("None", *self.evidence_labels)
        self.evidence_select.value = "None"
        self.unresolved.value = "None"
        self.image.image = self.crop_image.image = np.zeros((8, 8, 3), np.uint8)
        self.evidence_details.content = ""
        self.busy = False
        self.focus_button.disabled = False
        self.refresh()

    def refresh(self):
        if self.busy:
            return
        self.inspection.options(
            {
                "isolate": self.isolation.value
                and self.inspection.selected is not None,
                "show_centroid": self.centroid.value,
                "show_bounds": self.bounds.value,
                "crop_selection": self.crop.value,
            }
        )
        settings = self.settings()
        self.inspection.refresh(
            crop_settings={**settings, "plane": crop_plane(self.points(), settings)},
            point_size=self.size(),
        )
        self.context(self.inspection.isolate)
        obj = self.inspection.selected
        if obj:
            self.details.content = (
                f"**{obj['label']}** · `{obj['id']}`\n\n{obj['coordinate_frame']} · relative units\n\n"
                f"Observed-surface centroid: {obj['centroid']}\n\n{obj['supporting_frames']} supporting frames; "
                f"{obj['point_count']:,} saved points; {self.inspection.visible_count:,} visible.\n\n"
                + ", ".join(obj["quality"])
                + "\n\nFull saved object; ceiling crop "
                + ("applied." if self.crop.value else "bypassed.")
            )

    def focus(self, event):
        clients = [event.client] if event.client else self.server.get_clients().values()
        for client in clients:
            self.inspection.fit(client)

    def show_unresolved(self, _):
        if self.busy:
            return
        identifier = self.unresolved_labels.get(self.unresolved.value)
        if identifier:
            self.inspection.clear(self.capture_name)
            self.busy = True
            self.isolation.value = False
            self.selection.value = "None"
            self.evidence_select.options = ("None",)
            self.evidence_select.value = "None"
            self.busy = False
            self.context(False)
            self.focus_button.disabled = True
            self.details.content = (
                "Unresolved 2D observation; 3D focus/isolation unavailable."
            )
            self.show_evidence(identifier)

    def show_evidence(self, identifier):
        if self.busy or not identifier:
            return
        record = self.inspection.evidence_record(self.capture_name, identifier)
        self.image.image = np.array(
            Image.open(
                io.BytesIO(self.inspection.evidence(self.capture_name, identifier))
            )
        )
        self.crop_image.image = np.array(
            Image.open(
                io.BytesIO(
                    self.inspection.evidence(self.capture_name, identifier, "crop")
                )
            )
        )
        self.evidence_details.content = f"`{record['frame_id']}` · {record['timestamp_sec']:.3f}s\n\n{record['label']} · {record['status']}"
