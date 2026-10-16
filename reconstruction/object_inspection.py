"""Shared CPU object selection, exact highlights, camera fitting and evidence."""

import io
import math

import numpy as np
from PIL import Image, ImageDraw

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
        self.selected_evidence = None
        self.support_cameras = {}
        self.camera_scales = {}
        self.source_points_provider = None
        self.source_point_count = self.source_display_points = 0

    def clear(self, capture=None):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.selected = self.identifier = self.geometry = None
        self.selected_evidence = None
        self.visible_count = 0
        self.source_point_count = self.source_display_points = 0
        self.isolate = self.focus_active = False
        self.mode = "full"
        if capture is not None:
            self.capture = capture

    def select(self, capture, identifier):
        # Validate before changing current state.
        obj, geometry = self.inventory.object(capture, identifier)
        self.clear(capture)
        self.selected, self.identifier, self.geometry = obj, identifier, geometry
        if obj.get("point_count"):
            evidence = obj.get("geometry_observation_ids") or obj.get("observation_ids")
            evidence = evidence or [
                "context:" + eid for eid in obj.get("context_evidence", [])
            ]
        else:
            evidence = ["context:" + eid for eid in obj.get("context_evidence", [])]
            evidence += obj.get("observation_ids", [])
        if evidence:
            self.set_evidence(evidence[0])

    def set_evidence(self, identifier):
        if self.selected is None or not isinstance(identifier, str):
            raise ValueError("Select an object before choosing its supporting frame")
        allowed = set(self.selected.get("observation_ids", []))
        allowed.update(
            "context:" + eid for eid in self.selected.get("context_evidence", [])
        )
        if identifier and identifier not in allowed:
            raise ValueError("Frame is not evidence for the selected object")
        if identifier:
            self.evidence_record(self.capture, identifier)
        self.selected_evidence = identifier or None

    def source_frame(self):
        if not self.selected_evidence:
            return None
        record = self.evidence_record(self.capture, self.selected_evidence)
        return {
            key: record.get(key)
            for key in (
                "id",
                "frame_index",
                "frame_id",
                "timestamp_sec",
                "status",
                "source_region_box",
            )
        }

    def highlight_source_frame(self):
        # Segmented objects show their source photo without highlighting the
        # source camera or the wider frame in the 3D scene.
        if self.selected is None or self.selected.get("point_count"):
            return
        record = self.source_frame()
        cameras = self.support_cameras.get(self.capture, [])
        if not record or self.scene is None or not cameras:
            return
        camera = cameras[record["frame_index"]]
        if camera["frame_id"] != record["frame_id"]:
            raise ValueError("Source frame does not match its saved camera")
        width, height = camera["model_width"], camera["model_height"]
        corners = np.array(
            [
                [-0.5, -0.5, 1],
                [width - 0.5, -0.5, 1],
                [width - 0.5, height - 0.5, 1],
                [-0.5, height - 0.5, 1],
            ]
        )
        corners = corners @ np.linalg.inv(np.asarray(camera["intrinsics"])).T
        corners *= self.camera_scales.get(self.capture, 0.05)
        lines = np.concatenate(
            (
                np.stack((np.zeros_like(corners), corners), axis=1),
                np.stack((corners, np.roll(corners, -1, axis=0)), axis=1),
            )
        )
        pose = np.asarray(camera["camera_to_world"])
        lines = lines @ pose[:3, :3].T + pose[:3, 3]
        self.handles.append(
            self.scene.add_line_segments(
                "/object/source_camera",
                lines.astype(np.float32),
                (255, 185, 50),
                thickness=4,
                thickness_units="screen",
            )
        )
        self.handles.append(
            self.scene.add_label(
                "/object/source_frame",
                f"Source camera · {self.capture.title()} · frame {record['frame_id'].rsplit('_', 1)[-1].lstrip('0') or '0'} · {record['timestamp_sec']:.2f}s",
                position=tuple(pose[:3, 3]),
                depth_test=False,
            )
        )

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
        if values.get("isolate") and not self.selected.get("point_count"):
            raise ValueError(
                "This object has image evidence only; isolation needs geometry"
            )
        for key, value in values.items():
            setattr(self, key, value)

    def highlight_source_points(self, crop_settings, point_size):
        if (
            self.selected.get("point_count")
            or not self.selected_evidence
            or self.source_points_provider is None
        ):
            return
        record = self.source_frame()
        points = self.source_points_provider(self.capture, record["frame_index"])
        if self.crop_selection and crop_settings and crop_settings.get("crop_enabled"):
            plane = crop_settings.get("plane")
            if plane is not None:
                axis, sign, limit = plane
                points = points[points[:, axis] * sign <= limit]
        self.source_point_count = len(points)
        maximum = self.inventory.config["viewer"]["max_object_points"]
        self.source_display_points = min(len(points), maximum)
        if not len(points) or self.scene is None:
            return
        rows = np.linspace(0, len(points) - 1, self.source_display_points).astype(int)
        self.handles.append(
            self.scene.add_point_cloud(
                "/object/source_points",
                points[rows],
                (255, 185, 50),
                point_size=point_size * 1.5,
                precision="float32",
            )
        )

    def snapshot(self):
        return {
            "selected": self.selected,
            "mode": self.mode,
            "isolate": self.isolate,
            "show_centroid": self.show_centroid,
            "show_bounds": self.show_bounds,
            "crop_selection": self.crop_selection,
            "visible_points": self.visible_count,
            "source_frame": self.source_frame(),
            "source_point_count": self.source_point_count,
            "source_display_points": self.source_display_points,
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
        self.source_point_count = self.source_display_points = 0
        if self.selected is None:
            return
        self.highlight_source_frame()
        self.highlight_source_points(crop_settings, point_size)
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
        if self.selected is None or not self.selected.get("point_count"):
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
        if observation_id.startswith("context:") and hasattr(
            self.inventory, "context_record"
        ):
            return self.inventory.context_record(capture, observation_id)
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
        if record.get("status") == "context":
            mask = np.zeros(rgb.shape[:2], bool)
        else:
            with Image.open(
                safe_path(self.inventory.root, record["mask_path"])
            ) as mask_image:
                mask = np.asarray(mask_image) > 0
        if kind != "original":
            rgb[mask] = (
                rgb[mask].astype(float) * 0.5 + np.array([0, 220, 255]) * 0.5
            ).astype(np.uint8)
        image = Image.fromarray(rgb)
        if kind != "original" and record.get("status") == "context":
            # A context window is not an object mask. Outline it without tinting
            # pixels that have not been segmented as belonging to this object.
            box = record.get("source_region_box") or [0, 0, image.width, image.height]
            x0, y0, x1, y1 = box
            ImageDraw.Draw(image).rectangle(
                (
                    max(0, x0),
                    max(0, y0),
                    min(image.width - 1, x1 - 1),
                    min(image.height - 1, y1 - 1),
                ),
                outline=(255, 185, 50),
                width=max(3, image.width // 250),
            )
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
        elif (
            kind == "crop"
            and record.get("status") == "context"
            and record.get("source_region_box")
        ):
            image = image.crop(tuple(record["source_region_box"]))
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
