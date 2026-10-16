"""Global one-to-one identity assignment with unmatched and ambiguous cases."""

import re

import numpy as np
from scipy.optimize import linear_sum_assignment

from .alignment import apply_similarity


def label_family(label):
    label = re.sub(r"\s*\(instance candidate \d+\)$", "", label.lower()).strip()
    return {
        "sofa": "bed",
        "notebook computer": "laptop",
        "water bottle": "drinking bottle",
        "flask": "drinking bottle",
        "spoon": "utensils",
        "cooking utensil": "utensils",
        "bin": "waste bin",
        "shelf": "bookshelf",
        "bookcase": "bookshelf",
    }.get(label, label)


def _cosine(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if (
        a.ndim != 1
        or b.shape != a.shape
        or not len(a)
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        return None
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.clip(np.dot(a, b) / norm, -1, 1)) if norm > 1e-12 else None


def _center(obj):
    value = obj.get("centroid")
    if value is None:
        return None
    value = np.asarray(value, float)
    return value if value.shape == (3,) and np.isfinite(value).all() else None


def _shape_cost(left, right, scale):
    if any(
        o.get("bounds") is None
        or o.get("geometry_status") in ("inconsistent", "missing")
        for o in (left, right)
    ):
        return None
    a = np.sort(np.ptp(np.asarray(left["bounds"], float), axis=0))
    b = np.sort(np.ptp(np.asarray(right["bounds"], float), axis=0)) * scale
    return min(
        float(np.linalg.norm(a - b) / max(np.linalg.norm(a), np.linalg.norm(b), 1e-6)),
        1.0,
    )


def match_objects(
    before,
    after,
    *,
    after_transform=None,
    config=None,
    visual_links=None,
    alignment_accepted=False,
):
    """Search the whole room. Position is only a small cost after accepted alignment.

    An N+M assignment gives every object an unmatched option even for equal capture
    counts. Check alternatives in both directions and alternative global solutions.
    """
    config = config or {}
    threshold, margin = (
        float(config.get("unmatched_cost", 0.55)),
        float(config.get("ambiguity_margin", 0.06)),
    )
    if not 0 < threshold < 1 or not 0 <= margin < threshold:
        raise ValueError("Invalid matching thresholds")
    before, after = list(before), list(after)
    if any(len({o["id"] for o in group}) != len(group) for group in (before, after)):
        raise ValueError("Duplicate object IDs cannot enter identity assignment")
    n, m = len(before), len(after)
    hints = {(l["before_id"], l["after_id"]): l for l in visual_links or []}
    costs, features = np.full((n, m), 1e6), {}
    for i, left in enumerate(before):
        for j, right in enumerate(after):
            l, r = label_family(left["label"]), label_family(right["label"])
            unknown = any(
                x in ("unknown", "unknown_object", "unidentified") for x in (l, r)
            )
            if (l != r and not unknown) or left.get("kind", "individual") != right.get(
                "kind", "individual"
            ):
                continue
            appearance = _cosine(
                left.get("descriptor", []), right.get("descriptor", [])
            )
            hint = hints.get((left["id"], right["id"]))
            if appearance is None and not hint:
                continue
            appearance_cost = (1 - appearance) / 2 if appearance is not None else 0.5
            shape = (
                _shape_cost(left, right, after_transform["scale"])
                if after_transform and alignment_accepted
                else None
            )
            lc, rc = _center(left), _center(right)
            position, distance = None, None
            if (
                alignment_accepted
                and after_transform
                and lc is not None
                and rc is not None
                and all(o.get("movement_centroid_usable", True) for o in (left, right))
            ):
                distance = float(
                    np.linalg.norm(lc - apply_similarity(rc, after_transform))
                )
                position = min(
                    distance / max(float(config.get("position_scale", 1)), 1e-9), 1.0
                )
            value = (
                0.60 * appearance_cost
                + (0.2 if unknown else 0)
                + (0.10 * shape if shape is not None else 0.05)
            )
            value += 0.05 * position if position is not None else 0.025
            value += (
                -0.12
                if hint and hint.get("status") == "proposed"
                else 0.10
                if hint and hint.get("status") == "ambiguous"
                else 0
            )
            if appearance is not None and appearance < 0.25:
                value += 0.25
            costs[i, j] = max(0, value)
            features[i, j] = {
                "semantic_family": l,
                "appearance_cosine": appearance,
                "shape_cost": shape,
                "position_cost": position,
                "centroid_distance": distance,
                "visual_proposal": hint.get("status") if hint else None,
                "geometry_used": shape is not None or position is not None,
                "evidence_before": left.get("frame_ids", []),
                "evidence_after": right.get("frame_ids", []),
            }
    matrix = np.full((n + m, n + m), 1e6)
    matrix[:n, :m] = costs
    matrix[np.arange(n), m + np.arange(n)] = threshold / 2
    matrix[n + np.arange(m), np.arange(m)] = threshold / 2
    matrix[n:, m:] = 0
    rows, cols = linear_sum_assignment(matrix)
    chosen = dict(zip(rows.tolist(), cols.tolist()))
    optimal = float(matrix[rows, cols].sum())
    matches, candidates, mb, ma = [], [], set(), set()
    for i in range(n):
        for j in range(m):
            if costs[i, j] < threshold:
                candidates.append(
                    {
                        "before_id": before[i]["id"],
                        "after_id": after[j]["id"],
                        "cost": float(costs[i, j]),
                        "features": features[i, j],
                    }
                )
    for i, j in zip(rows, cols):
        if i >= n or j >= m or costs[i, j] >= threshold:
            continue
        alternate = matrix.copy()
        alternate[i, j] = 1e6
        ar, ac = linear_sum_assignment(alternate)
        gap = float(alternate[ar, ac].sum() - optimal)
        row_alts = [
            after[k]["id"]
            for k in range(m)
            if k != j and costs[i, k] <= costs[i, j] + margin
        ]
        col_alts = [
            before[k]["id"]
            for k in range(n)
            if k != i and costs[k, j] <= costs[i, j] + margin
        ]
        hint = hints.get((before[i]["id"], after[j]["id"]))
        reasons = []
        if gap <= margin or row_alts or col_alts:
            reasons.append("competing_global_assignments")
        if hint and (hint.get("status") == "ambiguous" or hint.get("alternatives")):
            reasons.append("ambiguous_visual_proposal")
        if any(
            o.get("confidence") == "low" or o.get("kind") == "collection"
            for o in (before[i], after[j])
        ):
            reasons.append("uncertain_capture_identity")
        f = features[i, j]
        visual_only = f["appearance_cosine"] is None
        if not visual_only and f["appearance_cosine"] < 0.45:
            reasons.append("weak_appearance_support")
        matches.append(
            {
                "before_id": before[i]["id"],
                "after_id": after[j]["id"],
                "status": "ambiguous"
                if reasons
                else "proposed_visual"
                if visual_only
                else "matched",
                "cost": float(costs[i, j]),
                "global_alternative_gap": gap,
                "features": f,
                "alternatives": row_alts,
                "alternative_before_ids": col_alts,
                "global_alternative_pairs": [
                    [before[row]["id"], after[col]["id"] if col < m else None]
                    for row, col in zip(ar.tolist(), ac.tolist())
                    if row < n and col != chosen[row]
                ]
                if gap <= margin
                else [],
                "reasons": reasons,
                "geometry_verified": False,
            }
        )
        mb.add(i)
        ma.add(j)
    return {
        "matches": matches,
        "unmatched_before": [o["id"] for i, o in enumerate(before) if i not in mb],
        "unmatched_after": [o["id"] for j, o in enumerate(after) if j not in ma],
        "alternatives": [r for r in matches if r["status"] == "ambiguous"],
        "candidates": candidates,
    }


def classify_changes(
    before,
    after,
    matching,
    *,
    after_transform=None,
    tolerance,
    alignment_status="unverified",
    before_visible=None,
    after_visible=None,
    before_free=None,
    after_free=None,
):
    """Legacy standalone helper; the connected run uses change_stage's MVP rules."""
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Positive movement tolerance required")
    b, a = ({o["id"]: o for o in group} for group in (before, after))
    events = []
    for match in matching["matches"]:
        left, right = b[match["before_id"]], a[match["after_id"]]
        old, new = _center(left), _center(right)
        reason, distance = "alignment_not_accepted", None
        if (
            alignment_status == "accepted"
            and after_transform is not None
            and old is not None
            and new is not None
        ):
            new = apply_similarity(new, after_transform)
            distance, reason = float(np.linalg.norm(new - old)), "centroid_displacement"
        eligible = (
            distance is not None
            and match["status"] == "matched"
            and all(o.get("movement_centroid_usable", True) for o in (left, right))
        )
        kind = (
            ("moved" if distance > tolerance else "unchanged")
            if eligible
            else "uncertain"
        )
        events.append(
            {
                "type": kind,
                "trust": "provisional" if eligible else "uncertain",
                "before_id": left["id"],
                "after_id": right["id"],
                "before_centroid": old.tolist() if old is not None else None,
                "after_centroid": new.tolist()
                if new is not None and distance is not None
                else None,
                "displacement": distance,
                "tolerance": tolerance,
                "reason": reason
                if eligible or distance is None
                else "identity_or_geometry_uncertain",
            }
        )
    for cap, identifiers, free in (
        ("before", matching["unmatched_before"], after_free),
        ("after", matching["unmatched_after"], before_free),
    ):
        for oid in identifiers:
            supported = (
                alignment_status == "accepted" and free is not None and oid in free
            )
            events.append(
                {
                    "type": ("removed" if cap == "before" else "added")
                    if supported
                    else "uncertain",
                    "trust": "provisional" if supported else "uncertain",
                    "before_id": oid if cap == "before" else None,
                    "after_id": oid if cap == "after" else None,
                    "reason": "opposing_free_space_evidence"
                    if supported
                    else "unmatched_without_absence_evidence",
                }
            )
    return {
        "schema_version": "change-v2",
        "coordinate_frame": "before_world",
        "units": "relative",
        "events": events,
    }
