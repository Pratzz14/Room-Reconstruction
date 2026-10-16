"""Robust similarity alignment between independent reconstruction frames."""

from dataclasses import dataclass

import numpy as np


def _validate_points(source, target, minimum=3):
    source, target = np.asarray(source, float), np.asarray(target, float)
    if source.ndim != 2 or source.shape != target.shape or source.shape[1] != 3:
        raise ValueError("Alignment points must be matching Nx3 arrays")
    if (
        len(source) < minimum
        or not np.isfinite(source).all()
        or not np.isfinite(target).all()
    ):
        raise ValueError(
            f"Alignment points must contain at least {minimum} finite pairs"
        )
    return source, target


def estimate_similarity(source, target):
    """Return ``target ~= scale * rotation @ source + translation``."""
    source, target = _validate_points(source, target)
    source_mean, target_mean = source.mean(0), target.mean(0)
    x, y = source - source_mean, target - target_mean
    if min(np.linalg.matrix_rank(x), np.linalg.matrix_rank(y)) < 2:
        raise ValueError("Collinear correspondences cannot constrain room alignment")
    covariance = (y.T @ x) / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1
    rotation = u @ correction @ vt
    variance = np.mean(np.sum(x * x, axis=1))
    if variance <= np.finfo(float).eps:
        raise ValueError("Alignment source points are degenerate")
    scale = float(np.sum(singular * np.diag(correction)) / variance)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Alignment scale is invalid")
    translation = target_mean - scale * (rotation @ source_mean)
    return {"scale": scale, "rotation": rotation, "translation": translation}


def validate_similarity(transform):
    scale = float(transform["scale"])
    rotation = np.asarray(transform["rotation"], float)
    translation = np.asarray(transform["translation"], float)
    if (
        not np.isfinite(scale)
        or scale <= 0
        or rotation.shape != (3, 3)
        or translation.shape != (3,)
        or not np.isfinite(rotation).all()
        or not np.isfinite(translation).all()
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)
    ):
        raise ValueError(
            "Expected a finite positive scale, proper rotation and 3D translation"
        )
    return scale, rotation, translation


def apply_similarity(points, transform):
    points = np.asarray(points, float)
    if points.ndim == 1:
        points = points[None, :]
        squeeze = True
    else:
        squeeze = False
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Points must be finite Nx3 geometry")
    scale, rotation, translation = validate_similarity(transform)
    result = scale * (points @ rotation.T) + translation
    return result[0] if squeeze else result


def inverse_similarity(transform):
    scale, rotation, translation = validate_similarity(transform)
    inverse_rotation = rotation.T
    inverse_scale = 1.0 / scale
    return {
        "scale": inverse_scale,
        "rotation": inverse_rotation,
        "translation": -inverse_scale * (inverse_rotation @ translation),
    }


def transform_bounds(bounds, transform):
    bounds = np.asarray(bounds, float)
    if bounds.shape != (2, 3):
        raise ValueError("Bounds must be [low, high]")
    if not np.isfinite(bounds).all() or np.any(bounds[0] > bounds[1]):
        raise ValueError("Bounds must be finite and ordered")
    corners = np.array(
        [[x, y, z] for x in bounds[:, 0] for y in bounds[:, 1] for z in bounds[:, 2]]
    )
    aligned = apply_similarity(corners, transform)
    return np.array([aligned.min(0), aligned.max(0)])


def _residuals(source, target, transform):
    return np.linalg.norm(apply_similarity(source, transform) - target, axis=1)


def ransac_similarity(source, target, *, threshold, iterations=512, seed=0):
    """Fit a similarity transform while rejecting bad structural correspondences."""
    source, target = _validate_points(source, target)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("RANSAC threshold must be positive")
    rng = np.random.default_rng(seed)
    best = None
    sample_size = 3
    for _ in range(max(1, int(iterations))):
        sample = rng.choice(len(source), sample_size, replace=False)
        try:
            candidate = estimate_similarity(source[sample], target[sample])
        except ValueError:
            continue
        residuals = _residuals(source, target, candidate)
        inliers = residuals <= threshold
        score = (
            int(inliers.sum()),
            -float(np.median(residuals[inliers])) if inliers.any() else -np.inf,
        )
        if best is None or score > best[0]:
            best = (score, inliers, candidate)
    if best is None or int(best[1].sum()) < 3:
        raise ValueError("Unable to estimate a non-degenerate similarity transform")
    transform = estimate_similarity(source[best[1]], target[best[1]])
    residuals = _residuals(source, target, transform)
    inliers = residuals <= threshold
    return {
        "transform": transform,
        "inliers": inliers,
        "residuals": residuals,
        "iterations": int(iterations),
        "threshold": float(threshold),
    }


@dataclass(frozen=True)
class AlignmentQuality:
    status: str
    inlier_count: int
    correspondence_count: int
    residual_p95: float
    support_fraction: float
    held_out_p95: float | None
    reasons: tuple[str, ...]

    def as_dict(self):
        return {
            "status": self.status,
            "inlier_count": self.inlier_count,
            "correspondence_count": self.correspondence_count,
            "residual_p95": self.residual_p95,
            "support_fraction": self.support_fraction,
            "held_out_p95": self.held_out_p95,
            "reasons": list(self.reasons),
        }


def assess_alignment(
    source,
    target,
    result,
    *,
    max_residual,
    min_inlier_fraction=0.6,
    held_out_source=None,
    held_out_target=None,
):
    source, target = _validate_points(source, target)
    residuals = np.asarray(result["residuals"], float)
    inliers = np.asarray(result["inliers"], bool)
    if residuals.shape != (len(source),) or inliers.shape != (len(source),):
        raise ValueError("Alignment result does not match correspondences")
    reasons = []
    support = float(inliers.mean())
    residual_p95 = (
        float(np.percentile(residuals[inliers], 95)) if inliers.any() else np.inf
    )
    held_out_p95 = None
    if support < min_inlier_fraction:
        reasons.append("insufficient_inlier_support")
    if not np.isfinite(residual_p95) or residual_p95 > max_residual:
        reasons.append("static_residual_too_high")
    if held_out_source is not None or held_out_target is not None:
        if held_out_source is None or held_out_target is None:
            raise ValueError("Held-out source and target must be provided together")
        held_out_source, held_out_target = _validate_points(
            held_out_source, held_out_target, minimum=1
        )
        held_out_p95 = float(
            np.percentile(
                _residuals(held_out_source, held_out_target, result["transform"]), 95
            )
        )
        if held_out_p95 > max_residual:
            reasons.append("held_out_residual_too_high")
    status = "accepted" if not reasons else "rejected"
    if status == "accepted" and held_out_p95 is None:
        status = "provisional"
    return AlignmentQuality(
        status,
        int(inliers.sum()),
        len(source),
        residual_p95,
        support,
        held_out_p95,
        tuple(reasons),
    )


def align_geometry(points, bounds, transform):
    """Transform object geometry and recompute axis-aligned bounds correctly."""
    points = apply_similarity(points, transform)
    return points, transform_bounds(bounds, transform)
