"""Timestamp and provenance helpers for synchronized reconstruction playback."""

import math
from bisect import bisect_right
from pathlib import Path

import numpy as np

from .core import digest, read_json

DEFAULT_FOV = math.radians(50)


class FrameTimeline:
    """Validated capture-local timeline for mapping video time to saved frames."""

    def __init__(self, frames):
        if not frames:
            raise ValueError("A reconstruction capture must contain at least one frame")
        self.frames = tuple(frames)
        self.timestamps = tuple(float(frame["elapsed_timestamp_sec"]) for frame in frames)
        if any(not math.isfinite(timestamp) or timestamp < 0 for timestamp in self.timestamps):
            raise ValueError("Frame timestamps must be non-negative")
        if any(right < left for left, right in zip(self.timestamps, self.timestamps[1:])):
            raise ValueError("Frame timestamps must be sorted")

    def frame_index(self, playback_time):
        """Return the last saved frame reached at playback_time, or -1 before the first."""
        if not math.isfinite(playback_time) or playback_time < 0:
            raise ValueError("Playback time must be non-negative")
        return bisect_right(self.timestamps, float(playback_time)) - 1

    def revealed_indices(self, playback_time):
        end = self.frame_index(playback_time)
        return range(end + 1)


def load_timeline(root, capture):
    return FrameTimeline(read_json(Path(root) / capture / "frames.json")["frames"])


def validate_video_source(root, capture, override=None):
    """Resolve and hash-check the original video referenced by a saved capture."""
    root = Path(root).resolve()
    source = read_json(root / capture / "source.json")
    path = Path(override or source["source_video_path"]).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Source video is missing: {path}")
    actual = digest(path)
    if actual != source["source_video_sha256"]:
        raise ValueError(f"Source video hash mismatch for {capture}: {path}")
    return path


def first_observation_frames(observations, point_count):
    """Earliest supporting frame, independently of the representative's confidence."""
    first = np.full(point_count, np.iinfo(np.int32).max, dtype=np.int32)
    np.minimum.at(first, observations["cloud_point_index"], observations["observation_frame_index"])
    if np.any(first == np.iinfo(np.int32).max):
        raise ValueError("A saved cloud point has no supporting observation")
    return first


def sample_indices(indices, maximum):
    if maximum < 1:
        raise ValueError("Display point limit must be positive")
    return indices[np.linspace(0, len(indices) - 1, min(len(indices), maximum), dtype=np.int64)] if len(indices) else indices


def room_bounds(points):
    """Exclude isolated depth outliers when framing; never modify saved geometry."""
    points = np.asarray(points)
    if len(points) == 0:
        return None
    if not np.isfinite(points).all():
        raise ValueError("Cannot frame non-finite geometry")
    return np.percentile(points, [2, 98], axis=0)


def fit_room(bounds, aspect, fov=DEFAULT_FOV):
    """Fit all robust-box corners using both horizontal and vertical field of view."""
    if bounds is None:
        return None
    if not math.isfinite(aspect) or aspect <= 0 or not 0 < fov < math.pi:
        raise ValueError("Invalid viewport aspect or field of view")
    low, high = np.asarray(bounds, dtype=float)
    center = (low + high) / 2
    radius = max(float(np.linalg.norm(high - low)) / 2, 0.001)
    outward = np.array([1.0, -0.65, -1.0])
    outward /= np.linalg.norm(outward)
    forward = -outward
    up = np.array([0.0, -1.0, 0.0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    vertical = np.cross(right, forward)
    corners = np.array([[x, y, z] for x in (low[0], high[0])
                        for y in (low[1], high[1]) for z in (low[2], high[2])])
    offsets = corners - center
    depths = offsets @ forward
    tangent = math.tan(fov / 2)
    distance = max(float(np.max(np.abs(offsets @ right) * 1.15 / (tangent * aspect) - depths)),
                   float(np.max(np.abs(offsets @ vertical) * 1.15 / tangent - depths)),
                   float(np.max(-depths)) + radius * 0.1)
    return {"position": center + outward * distance, "look_at": center,
            "up_direction": up, "near": max(radius * 0.001, 0.000001),
            "far": distance + radius * 10, "radius": radius, "fov": fov}
