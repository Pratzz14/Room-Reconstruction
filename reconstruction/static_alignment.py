"""Image correspondences lifted through saved depth, restricted to named static regions.

No object centroids, camera-path pairing, or VLM coordinates are fitting landmarks.
Frames withheld from fitting are used for a separate geometric acceptance check.
"""

from collections import Counter

import cv2
import numpy as np
from PIL import Image

from .alignment import apply_similarity, ransac_similarity
from .core import read_json, safe_path


def scene_extent(points):
    return float(
        np.linalg.norm(
            np.percentile(points, 95, axis=0) - np.percentile(points, 5, axis=0)
        )
    )


def static_assignments(inventory, config):
    """Only unique visual proposals may seed named fixed-instance regions.

    The unmasked room shell is a separate region; all other detected masks are
    excluded. Fixed-label assumptions are configuration, not proof of immobility.
    """
    maps = {c: {} for c in ("before", "after")}
    objects = {o["id"]: o for d in inventory.captures.values() for o in d["objects"]}
    regions = ["room_shell"]
    for link in getattr(inventory, "report", {}).get("links", []):
        left, right = objects[link["before_id"]], objects[link["after_id"]]
        if (
            link["status"] != "proposed"
            or left["label"] != right["label"]
            or left["label"] not in config["fixed_labels"]
            or any(
                o.get("confidence") != "high" or o.get("kind") != "individual"
                for o in (left, right)
            )
        ):
            continue
        name = left["label"] + ":" + link["id"]
        regions.append(name)
        for cap, obj in (("before", left), ("after", right)):
            for oid in obj.get("geometry_observation_ids", []):
                maps[cap][oid] = len(regions) - 1
    return maps, regions


def frame_features(inventory, capture, assignments, config):
    cv2.setNumThreads(2)
    root = inventory.reconstruction
    data = inventory.captures[capture]
    cameras = read_json(root / capture / "cameras.json")
    transforms = read_json(root / capture / "image_transforms.json")
    with np.load(root / capture / "depth.npz", allow_pickle=False) as archive:
        depth = archive["depth"]
    # Use only depth pixels that actually survived the reconstruction's filters.
    with np.load(root / capture / "observations.npz", allow_pickle=False) as archive:
        valid = np.zeros(depth.shape, bool)
        fi, pi = archive["observation_frame_index"], archive["model_pixel_index"]
        valid.reshape(len(valid), -1)[fi, pi] = True
    sift = cv2.SIFT_create(nfeatures=int(config["sift_features"]))
    frames = []
    for fi, (frame, camera, mapping) in enumerate(
        zip(data["frames"], cameras, transforms)
    ):
        image = cv2.imread(
            str(safe_path(root, frame["model_image_path"])), cv2.IMREAD_GRAYSCALE
        )
        h, w = image.shape
        regions = np.zeros((h, w), np.int16)
        excluded = np.zeros((h, w), bool)
        named = []
        for ob in data["observations"]:
            if ob["frame_index"] != fi:
                continue
            with Image.open(safe_path(inventory.root, ob["mask_path"])) as mask:
                original = np.asarray(mask, dtype=np.uint8)
            mask = (
                cv2.warpPerspective(
                    original,
                    np.array(mapping["original_to_model"]),
                    (w, h),
                    flags=cv2.INTER_NEAREST,
                )
                > 0
            )
            if ob["id"] in assignments:
                named.append((mask, assignments[ob["id"]]))
            else:
                excluded |= mask
        regions[excluded] = -1
        for mask, region in named:
            # Conflicting named instances are excluded rather than assigned twice.
            conflict = mask & (regions > 0) & (regions != region)
            regions[mask & ~excluded] = region
            regions[conflict] = -1
        allowed = (regions >= 0) & valid[fi]
        # Avoid boundaries/occlusions and unreliable discontinuities in lifted depth.
        finite_depth = np.where(np.isfinite(depth[fi]) & (depth[fi] > 0), depth[fi], 0)
        near, far = (
            cv2.erode(finite_depth, np.ones((3, 3), np.uint8)),
            cv2.dilate(finite_depth, np.ones((3, 3), np.uint8)),
        )
        allowed &= (far - near) <= 0.1 * np.maximum(finite_depth, 1e-6)
        allowed = cv2.erode(allowed.astype(np.uint8), np.ones((3, 3), np.uint8))
        kp, descriptors = sift.detectAndCompute(image, allowed * 255)
        if not kp:
            frames.append(
                {
                    "points": np.empty((0, 3)),
                    "descriptors": np.empty((0, 128), np.float32),
                    "pixels": np.empty((0, 2), int),
                    "regions": np.empty(0, int),
                }
            )
            continue
        pixels = np.rint([k.pt for k in kp]).astype(int)
        pixels[:, 0] = np.clip(pixels[:, 0], 0, w - 1)
        pixels[:, 1] = np.clip(pixels[:, 1], 0, h - 1)
        uv = np.c_[pixels, np.ones(len(pixels))]
        points = (uv @ np.linalg.inv(np.array(camera["intrinsics"])).T) * depth[
            fi, pixels[:, 1], pixels[:, 0], None
        ]
        pose = np.array(camera["camera_to_world"])
        frames.append(
            {
                "points": points @ pose[:3, :3].T + pose[:3, 3],
                "pixels": pixels,
                "descriptors": descriptors,
                "regions": regions[pixels[:, 1], pixels[:, 0]],
            }
        )
    return frames


def mutual_matches(left, right, ratio):
    if min(len(left), len(right)) < 2:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward, reverse = (
        matcher.knnMatch(left, right, k=2),
        matcher.knnMatch(right, left, k=2),
    )
    back = {
        m.queryIdx: m.trainIdx for m, n in reverse if m.distance < ratio * n.distance
    }
    return [
        m
        for m, n in forward
        if m.distance < ratio * n.distance and back.get(m.trainIdx) == m.queryIdx
    ]


def extract_correspondences(inventory, config):
    assignments, names = static_assignments(inventory, config)
    features = {}
    for cap in ("before", "after"):
        print(f"Extracting static image features: {cap}", flush=True)
        features[cap] = frame_features(inventory, cap, assignments[cap], config)
    source, target, frame_pairs, region_ids, pixels, scores = [], [], [], [], [], []
    for ai, after in enumerate(features["after"]):
        for bi, before in enumerate(features["before"]):
            for region in sorted(set(after["regions"]) & set(before["regions"])):
                if region < 0:
                    continue
                ar, br = (
                    np.flatnonzero(after["regions"] == region),
                    np.flatnonzero(before["regions"] == region),
                )
                matches = mutual_matches(
                    after["descriptors"][ar],
                    before["descriptors"][br],
                    config["ratio_test"],
                )
                for match in sorted(matches, key=lambda m: m.distance)[
                    : config["max_matches_per_pair_region"]
                ]:
                    a, b = ar[match.queryIdx], br[match.trainIdx]
                    source.append(after["points"][a])
                    target.append(before["points"][b])
                    frame_pairs.append([ai, bi])
                    region_ids.append(region)
                    pixels.append([*after["pixels"][a], *before["pixels"][b]])
                    scores.append(match.distance)
    pairs = np.asarray(frame_pairs, np.int32).reshape(-1, 2)
    holdout = (pairs[:, 0] % config["holdout_stride"] == 0) | (
        pairs[:, 1] % config["holdout_stride"] == 0
    )
    return {
        "source": np.asarray(source, float).reshape(-1, 3),
        "target": np.asarray(target, float).reshape(-1, 3),
        "frame_pairs": pairs,
        "region_ids": np.asarray(region_ids, int),
        "model_pixels": np.asarray(pixels, int).reshape(-1, 4),
        "descriptor_distances": np.asarray(scores),
        "held_out": holdout,
    }, names


def structural_patches(c, names, extent, config):
    """Find Before shell planes from fitting frames; label withheld points separately.

    Patches measure distributed support, without manufacturing correspondences.
    """
    target, fitting = c["target"], ~c["held_out"]
    remaining = np.flatnonzero((c["region_ids"] == 0) & fitting)
    rng, models = np.random.default_rng(config["seed"]), []
    tolerance = extent * 0.012
    for _ in range(5):
        if len(remaining) < config["min_correspondences"]:
            break
        points, best = target[remaining], None
        for _ in range(256):
            sample = points[rng.choice(len(points), 3, replace=False)]
            normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
            length = np.linalg.norm(normal)
            if length < extent * extent * 1e-5:
                continue
            normal /= length
            offset = -normal @ sample[0]
            mask = np.abs(points @ normal + offset) < tolerance
            if best is None or mask.sum() > best[0].sum():
                best = mask, normal, offset
        if best is None or best[0].sum() < max(30, config["min_correspondences"]):
            break
        selected = points[best[0]]
        center = np.median(selected, axis=0)
        _, _, vh = np.linalg.svd(selected - center, full_matrices=False)
        normal, offset = vh[-1], -vh[-1] @ center
        if scene_extent(selected) < 0.15 * extent:
            break
        models.append((normal, offset))
        remaining = remaining[np.abs(points @ normal + offset) >= tolerance]
    labels = c["region_ids"].copy()
    labels[labels == 0] = -1
    patch_names = list(names)
    if models:
        distance = np.column_stack([np.abs(target @ n + d) for n, d in models])
        nearest = distance.argmin(1)
        for i, _ in enumerate(models):
            chosen = (
                (c["region_ids"] == 0) & (nearest == i) & (distance[:, i] < tolerance)
            )
            labels[chosen] = len(patch_names)
            patch_names.append(f"room_shell_plane_{i}")
    return (
        labels,
        patch_names,
        [{"normal": n.tolist(), "offset": float(d)} for n, d in models],
    )


def fit_static(correspondences, names, extent, config):
    c = correspondences
    source, target = c["source"], c["target"]
    fit, held = ~c["held_out"], c["held_out"]
    threshold = extent * config["residual_extent_fraction"]
    reasons = []
    quality = {
        "status": "rejected",
        "reasons": reasons,
        "threshold": threshold,
        "scene_extent": extent,
        "threshold_source": "configured_fraction_of_before_extent_not_independent_calibration",
        "fit_count": int(fit.sum()),
        "held_out_count": int(held.sum()),
        "location_comparison_allowed": False,
    }
    if fit.sum() < config["min_correspondences"]:
        reasons.append("insufficient_static_correspondences")
        return None, quality
    try:
        result = ransac_similarity(
            source[fit],
            target[fit],
            threshold=threshold,
            iterations=config["ransac_iterations"],
            seed=config["seed"],
        )
    except ValueError as exc:
        reasons.append(str(exc))
        return None, quality
    transform = result["transform"]
    residuals = np.linalg.norm(apply_similarity(source, transform) - target, axis=1)
    inliers = residuals <= threshold
    patches, patch_names, planes = structural_patches(c, names, extent, config)
    quality["structural_planes_before"] = planes
    if not config["minimum_scale"] <= transform["scale"] <= config["maximum_scale"]:
        reasons.append("scale_outside_limits")
    for label, subset in (("fit", fit), ("held_out", held)):
        good = subset & inliers
        fraction = float(good.sum() / max(1, subset.sum()))
        pairs = len(np.unique(c["frame_pairs"][good], axis=0))
        supported_patches = [
            p
            for p in np.unique(patches[good & (patches >= 0)])
            if np.sum(good & (patches == p)) >= config["min_correspondences"]
        ]
        regions = len(supported_patches)
        normals = []
        for p in supported_patches:
            if p < len(names):
                continue
            normal = np.asarray(planes[p - len(names)]["normal"])
            if all(abs(normal @ n) < np.cos(np.deg2rad(20)) for n in normals):
                normals.append(normal)
        support = scene_extent(target[good]) / extent if good.sum() >= 3 else 0
        quality[label] = {
            "inlier_count": int(good.sum()),
            "inlier_fraction": fraction,
            "frame_pairs": pairs,
            "regions": regions,
            "support_extent_fraction": support,
            "shell_normal_directions": len(normals),
            "median_residual_all": float(np.median(residuals[subset]))
            if subset.any()
            else None,
            "p95_inlier_residual": float(np.percentile(residuals[good], 95))
            if good.any()
            else None,
        }
        if good.sum() < config["min_correspondences"]:
            reasons.append(label + "_insufficient_inliers")
        if fraction < config["min_inlier_fraction"]:
            reasons.append(label + "_low_inlier_fraction")
        if subset.any() and np.median(residuals[subset]) > threshold:
            reasons.append(label + "_median_residual_too_high")
        if pairs < config["min_frame_pairs"] or regions < config["min_regions"]:
            reasons.append(label + "_insufficient_distributed_static_support")
        if not np.any(c["region_ids"][good] > 0) and len(normals) < 2:
            reasons.append(label + "_shell_requires_multiple_plane_directions")
        if support < config["min_support_extent_fraction"]:
            reasons.append(label + "_support_too_local")
    quality["structural_patches"] = [
        {
            "name": name,
            "fit_inliers": int(np.sum((patches == ri) & inliers & fit)),
            "held_out_inliers": int(np.sum((patches == ri) & inliers & held)),
            "median_residual": float(np.median(residuals[patches == ri])),
        }
        for ri, name in enumerate(patch_names)
        if np.any(patches == ri)
    ]
    # Report every named region, including contradictory/missing support.
    quality["regions"] = []
    for ri, name in enumerate(names):
        rows = c["region_ids"] == ri
        if not rows.any():
            continue
        quality["regions"].append(
            {
                "name": name,
                "correspondences": int(rows.sum()),
                "inlier_fraction": float(inliers[rows].mean()),
                "median_residual": float(np.median(residuals[rows])),
            }
        )
        if (
            rows.sum() >= config["min_correspondences"]
            and inliers[rows].mean() < config["min_inlier_fraction"]
        ):
            reasons.append("contradictory_static_region:" + name)
    # Expose scale stability under dispersed frame-pair subsampling.
    from .alignment import estimate_similarity

    scales = []
    rng = np.random.default_rng(config["seed"])
    pair_ids = c["frame_pairs"][fit & inliers]
    unique_pairs = np.unique(pair_ids, axis=0)
    for _ in range(16):
        if len(unique_pairs) < 4:
            break
        chosen = unique_pairs[
            rng.choice(len(unique_pairs), max(2, len(unique_pairs) // 2), replace=False)
        ]
        rows = (
            fit
            & inliers
            & np.any(np.all(c["frame_pairs"][:, None] == chosen, axis=2), axis=1)
        )
        try:
            scales.append(estimate_similarity(source[rows], target[rows])["scale"])
        except ValueError:
            pass
    variation = (
        float(np.ptp(np.percentile(scales, [5, 95])) / transform["scale"])
        if scales
        else None
    )
    quality["scale_variation"] = variation
    if variation is None or variation > 0.1:
        reasons.append("unstable_scale")
    quality["status"] = "accepted" if not reasons else "rejected"
    quality["location_comparison_allowed"] = not reasons
    quality["inlier_regions"] = dict(
        Counter(names[r] for r in c["region_ids"][inliers])
    )
    c["residuals"], c["inliers"] = residuals, inliers
    return transform, quality
