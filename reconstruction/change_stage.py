"""Simple saved-inventory changes; full-frame means are explicitly estimates."""

from collections import Counter

import numpy as np

from .alignment import apply_similarity


def supporting_frames(inventory, capture, obj):
    frames = set(obj.get("frame_ids", []))
    data = inventory.captures[capture]
    observations = {o["id"]: o for o in data.get("observations", [])}
    for oid in obj.get("observation_ids", []):
        if oid in observations:
            frames.add(observations[oid]["frame_id"])
    for eid in obj.get("context_evidence", []):
        if hasattr(inventory, "context_record"):
            frames.add(inventory.context_record(capture, "context:" + eid)["frame_id"])
    return sorted(frames)


def frame_cloud_means(inventory, capture, objects):
    """Mean the union of cloud rows, including non-representative observations.

    Read original, capture-local clouds only. Display sampling, cropping and the
    aligned exports must never affect this calculation.
    """
    if not objects:
        return {}
    root = inventory.reconstruction / capture
    with np.load(root / "cloud.npz", allow_pickle=False) as saved:
        points = saved["points"]
    with np.load(root / "observations.npz", allow_pickle=False) as saved:
        frame_ids = saved["frame_ids"].tolist()
        observation_frames = saved["observation_frame_index"]
        membership = saved["cloud_point_index"]
    indices = {fid: i for i, fid in enumerate(frame_ids)}
    cache, result = {}, {}
    for obj in objects:
        frames = supporting_frames(inventory, capture, obj)
        reason = None
        if not frames:
            reason = "missing_supporting_frames"
        elif any(fid not in indices for fid in frames):
            reason = "unknown_supporting_frame"
        rows = np.empty(0, dtype=np.int64)
        if reason is None:
            for fid in frames:
                if fid not in cache:
                    cache[fid] = np.unique(
                        membership[observation_frames == indices[fid]]
                    )
            rows = np.unique(np.concatenate([cache[fid] for fid in frames]))
            if not len(rows):
                reason = "empty_frame_cloud"
            elif not np.isfinite(points[rows]).all():
                reason = "nonfinite_frame_cloud"
        mean = (
            np.mean(points[rows], axis=0, dtype=np.float64) if reason is None else None
        )
        result[obj["id"]] = {
            "mean": mean.tolist() if mean is not None else None,
            "frame_ids": frames,
            "point_count": len(rows),
            "reason": reason,
        }
    return result


def ambiguous_objects(identities, matching):
    """Propagate competing assignments to selected pairs and unmatched alternatives."""
    affected = set()
    links = []
    for item in identities:
        pair = {item[k] for k in ("before_id", "after_id") if item.get(k)}
        links.append(pair)
        if item["status"] == "ambiguous":
            affected.update(pair)
    for match in matching["matches"]:
        group = {match["before_id"], match["after_id"]}
        group.update(match.get("alternatives", []))
        group.update(match.get("alternative_before_ids", []))
        for pair in match.get("global_alternative_pairs", []):
            group.update(x for x in pair if x is not None)
        links.append(group)
        if match["status"] == "ambiguous":
            affected.update(group)
    while True:
        expanded = affected.union(*(group for group in links if group & affected))
        if expanded == affected:
            return affected
        affected = expanded


def usable_centroid(obj):
    value = obj.get("centroid")
    if (
        value is None
        or not obj.get("movement_centroid_usable", True)
        or obj.get("geometry_status") in ("missing", "inconsistent")
        or obj.get("point_count", 3) < 3
    ):
        return None
    value = np.asarray(value, float)
    return value if value.shape == (3,) and np.isfinite(value).all() else None


def build_changes(inventory, identities, matching, alignment, extent, fraction=0.2):
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError("Positive finite room extent required for changes")
    if not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("Movement extent fraction must be in (0, 1]")
    tolerance = float(extent * fraction)
    accepted = (
        alignment["quality"]["status"] == "accepted"
        and alignment["transform"] is not None
    )
    objects = {
        c: {o["id"]: o for o in inventory.captures[c]["objects"]}
        for c in ("before", "after")
    }
    ambiguous = ambiguous_objects(identities, matching)
    needed = {c: {} for c in objects}
    if accepted:
        for item in identities:
            left, right = (objects[c].get(item[c + "_id"]) for c in objects)
            if (
                left
                and right
                and not ({left["id"], right["id"]} & ambiguous)
                and (usable_centroid(left) is None or usable_centroid(right) is None)
            ):
                needed["before"][left["id"]] = left
                needed["after"][right["id"]] = right
    means = {
        c: frame_cloud_means(inventory, c, list(needed[c].values())) for c in objects
    }
    events = []
    for item in sorted(identities, key=lambda i: i["shared_id"]):
        left, right = (objects[c].get(item[c + "_id"]) for c in objects)
        event = {
            "shared_id": item["shared_id"],
            "before_id": item["before_id"],
            "after_id": item["after_id"],
            "label": (left or right)["label"],
            "type": "uncertain",
            "trust": "uncertain",
            "basis": "insufficient_evidence",
            "before_location": None,
            "after_location": None,
            "object_displacement": None,
            "frame_cloud_distance": None,
            "threshold": tolerance,
            "supporting_frame_ids": {
                c: supporting_frames(inventory, c, obj) if obj else []
                for c, obj in (("before", left), ("after", right))
            },
            "frame_cloud_support": {},
            "uncertainties": [],
        }
        reasons = event["uncertainties"]
        if not accepted:
            reasons.append("alignment_not_accepted")
        if {item["before_id"], item["after_id"]} & ambiguous:
            reasons.append("identity_ambiguous")
        if any(
            o and (o.get("confidence") == "low" or o.get("kind") == "collection")
            for o in (left, right)
        ):
            reasons.append("uncertain_capture_identity")
        if reasons:
            event["explanation"] = "Uncertain: " + ", ".join(reasons) + "."
        elif not left or not right:
            event.update(
                type="removed" if left else "added",
                trust="estimated",
                basis="presence_only",
                explanation="Observed only "
                + ("before" if left else "after")
                + "; estimated presence change without an empty-region check.",
            )
        else:
            old, new = usable_centroid(left), usable_centroid(right)
            geometry = old is not None and new is not None
            if geometry:
                event["basis"] = "object_centroids"
            else:
                event["basis"] = "frame_cloud_mean"
                for c, obj in (("before", left), ("after", right)):
                    support = means[c][obj["id"]]
                    event["frame_cloud_support"][c] = {
                        k: v for k, v in support.items() if k != "mean"
                    }
                    if support["reason"]:
                        reasons.append(c + "_" + support["reason"])
                old, new = (
                    means[c][obj["id"]]["mean"]
                    for c, obj in (("before", left), ("after", right))
                )
            if old is None or new is None:
                event["explanation"] = (
                    "Uncertain: supporting-frame location unavailable."
                )
            else:
                # Both inputs are original capture-local values, never aligned exports.
                new = apply_similarity(new, alignment["transform"])
                distance = float(np.linalg.norm(np.asarray(old) - new))
                moved = distance > tolerance and not np.isclose(
                    distance, tolerance, rtol=1e-12, atol=0
                )
                event.update(
                    type="moved" if moved else "unchanged",
                    trust="reliable" if geometry else "estimated",
                    before_location=np.asarray(old).tolist(),
                    after_location=new.tolist(),
                )
                event["object_displacement" if geometry else "frame_cloud_distance"] = (
                    distance
                )
                measurement = (
                    "Object displacement"
                    if geometry
                    else "Estimated frame-cloud distance"
                )
                event["explanation"] = (
                    f"{measurement}: {distance:.6g} relative units; "
                    f"{'above' if moved else 'at or below'} threshold {tolerance:.6g}."
                )
        events.append(event)
    return {
        "schema_version": "changes-v1",
        "coordinate_frame": "before_world" if accepted else None,
        "units": "relative",
        "alignment_status": alignment["quality"]["status"],
        "scene_extent": float(extent),
        "movement_extent_fraction": float(fraction),
        "threshold": tolerance,
        "counts": dict(sorted(Counter(e["type"] for e in events).items())),
        "assumptions": [
            "One-sided presence counts as added/removed without visibility checks.",
            "Whole supporting-frame cloud means estimate image-only locations.",
            "Reliable denotes usable object geometry for this MVP, not independent metric validation.",
            "Unchanged means no location difference above threshold; rotation is not evaluated.",
        ],
        "events": events,
    }


def saved_changes(inventory, identities, matching, alignment, config):
    from .static_alignment import scene_extent

    with np.load(
        inventory.reconstruction / "before/cloud.npz", allow_pickle=False
    ) as saved:
        extent = scene_extent(saved["points"])
    return build_changes(
        inventory,
        identities,
        matching,
        alignment,
        extent,
        config.get("movement_extent_fraction", 0.2),
    )
