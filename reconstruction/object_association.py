"""Conservative capture-local associations with explicit conflicts and alternatives."""

import numpy as np
from scipy.spatial import cKDTree

from .core import project


def sample(points, maximum):
    return points[
        np.linspace(0, len(points) - 1, min(maximum, len(points))).astype(int)
    ]


def reprojection(source, target, capture, masks, depth, valid, settings, voxel):
    fi = target["frame_index"]
    camera = capture.cameras[fi]
    points = sample(source["points"], settings["max_pair_points"])
    uv, z = project(points, camera["intrinsics"], camera["camera_to_world"])
    finite = np.isfinite(uv).all(1) & np.isfinite(z) & (z > 0)
    uv, z = uv[finite], z[finite]
    pixels = np.floor(uv + 0.5).astype(np.int64)
    inside = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 0] < camera["model_width"])
        & (pixels[:, 1] < camera["model_height"])
    )
    uv, z, pixels = uv[inside], z[inside], pixels[inside]
    target_depth = depth[fi, pixels[:, 1], pixels[:, 0]]
    supported = (
        valid[fi, pixels[:, 1], pixels[:, 0]]
        & np.isfinite(target_depth)
        & (target_depth > 0)
    )
    # Occluded/unknown points do not vote against identity. Only surfaces at matching depth vote.
    supported &= np.abs(z - target_depth) <= settings["depth_margin_voxels"] * voxel
    uv = uv[supported]
    original = (
        np.c_[uv, np.ones(len(uv))]
        @ np.asarray(capture.transforms[fi]["model_to_original"]).T
    )
    xy = np.floor(original[:, :2] / original[:, 2:3] + 0.5).astype(np.int64)
    mask = masks[target["id"]]
    inside = (
        (xy[:, 0] >= 0)
        & (xy[:, 1] >= 0)
        & (xy[:, 0] < mask.shape[1])
        & (xy[:, 1] < mask.shape[0])
    )
    xy = xy[inside]
    return {
        "samples": len(xy),
        "fraction": float(mask[xy[:, 1], xy[:, 0]].mean()) if len(xy) else None,
    }


def consolidate(
    observations, arrays, descriptors, capture, masks, depth, valid, settings, voxel
):
    usable = [i for i, o in enumerate(observations) if o["status"] == "usable"]
    if not usable:
        return [], []
    features = {}
    for i in usable:
        pts = arrays[i]["points"]
        features[i] = {
            **observations[i],
            "points": pts,
            "center": np.median(pts, axis=0),
            "extent": float(np.linalg.norm(np.ptp(pts, axis=0))),
        }
    centers = np.array([features[i]["center"] for i in usable])
    maximum_extent = max(features[i]["extent"] for i in usable)
    tree = cKDTree(centers)
    pairs = set()
    for k, i in enumerate(usable):
        radius = (
            settings["candidate_extent_multiplier"]
            * (features[i]["extent"] + maximum_extent)
            + 4 * voxel
        )
        pairs.update(
            (min(i, usable[j]), max(i, usable[j]))
            for j in tree.query_ball_point(centers[k], radius)
            if usable[j] != i
        )
    decisions, relations = [], {}
    for i, j in sorted(pairs):
        a, b = features[i], features[j]
        record = {"a": a["id"], "b": b["id"], "relation": "unrelated", "reasons": []}
        label_ok = a["label"] == b["label"] or "unknown_object" in (
            a["label"],
            b["label"],
        )
        if not label_ok:
            record.update(relation="conflict", reasons=["incompatible_labels"])
        elif a["frame_index"] == b["frame_index"]:
            record.update(
                relation="conflict", reasons=["distinct_same_frame_instances"]
            )
        elif (
            np.linalg.norm(a["center"] - b["center"])
            <= settings["candidate_extent_multiplier"] * (a["extent"] + b["extent"])
            + 4 * voxel
        ):
            da, db = descriptors[i], descriptors[j]
            cosine = (
                float(
                    np.dot(da, db) / max(np.linalg.norm(da) * np.linalg.norm(db), 1e-12)
                )
                if len(da) and len(db)
                else None
            )
            pa, pb = (
                sample(a["points"], settings["max_pair_points"]),
                sample(b["points"], settings["max_pair_points"]),
            )
            overlap = min(
                float(np.mean(cKDTree(pb).query(pa)[0] <= 2 * voxel)),
                float(np.mean(cKDTree(pa).query(pb)[0] <= 2 * voxel)),
            )
            projections = [
                reprojection(a, b, capture, masks, depth, valid, settings, voxel),
                reprojection(b, a, capture, masks, depth, valid, settings, voxel),
            ]
            votes = [p["fraction"] for p in projections if p["samples"] >= 8]
            record.update(appearance=cosine, overlap=overlap, reprojection=projections)
            if votes and min(votes) < settings["contradiction_fraction"]:
                record.update(
                    relation="conflict", reasons=["visible_mask_contradiction"]
                )
            elif cosine is not None and cosine < settings["appearance_similarity"]:
                record.update(relation="conflict", reasons=["appearance_disagreement"])
            elif cosine is not None and (
                (
                    overlap >= settings["overlap_fraction"]
                    and (not votes or min(votes) >= settings["reprojection_fraction"])
                )
                or (len(votes) == 2 and min(votes) >= settings["reprojection_fraction"])
            ):
                record.update(
                    relation="match", reasons=["appearance_and_geometry_support"]
                )
            else:
                record.update(
                    relation="ambiguous", reasons=["insufficient_joint_support"]
                )
        relations[i, j] = record["relation"]
        decisions.append(record)
    # A view that supports two mutually conflicting identities is an alternative,
    # not permission to choose whichever edge sorts first.
    ambiguous_edges = set()
    for middle in usable:
        neighbors = [
            b if a == middle else a
            for (a, b), relation in relations.items()
            if relation == "match" and middle in (a, b)
        ]
        for k, a in enumerate(neighbors):
            for b in neighbors[k + 1 :]:
                incompatible = (
                    observations[a]["frame_index"] == observations[b]["frame_index"]
                    or relations.get(tuple(sorted((a, b)))) == "conflict"
                    or (
                        observations[a]["label"] != observations[b]["label"]
                        and "unknown_object"
                        not in (observations[a]["label"], observations[b]["label"])
                    )
                )
                if incompatible:
                    ambiguous_edges.update(
                        (tuple(sorted((middle, a))), tuple(sorted((middle, b))))
                    )
    for edge in ambiguous_edges:
        relations[edge] = "ambiguous"
    for decision in decisions:
        if any(
            {decision["a"], decision["b"]}
            == {observations[a]["id"], observations[b]["id"]}
            for a, b in ambiguous_edges
        ):
            decision.update(
                relation="ambiguous", reasons=["competing_identity_support"]
            )
    groups = [{i} for i in usable]
    for (i, j), relation in sorted(
        relations.items(), key=lambda p: (p[1] != "match", p[0])
    ):
        if relation != "match":
            continue
        first, second = (
            next(g for g in groups if i in g),
            next(g for g in groups if j in g),
        )
        if first is second:
            continue
        # A supporting bridge cannot override any known pairwise conflict.
        if any(
            relations.get(tuple(sorted((a, b)))) == "conflict"
            or observations[a]["frame_index"] == observations[b]["frame_index"]
            or (
                observations[a]["label"] != observations[b]["label"]
                and "unknown_object"
                not in (observations[a]["label"], observations[b]["label"])
            )
            for a in first
            for b in second
        ):
            decisions.append(
                {
                    "a": observations[i]["id"],
                    "b": observations[j]["id"],
                    "relation": "group_conflict",
                    "reasons": ["group_contains_conflict"],
                }
            )
            continue
        first.update(second)
        groups.remove(second)
    return [sorted(g) for g in sorted(groups, key=lambda g: min(g))], decisions
