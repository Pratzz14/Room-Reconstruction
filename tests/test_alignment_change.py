import unittest

import numpy as np

from reconstruction.alignment import (
    apply_similarity,
    assess_alignment,
    estimate_similarity,
    inverse_similarity,
    ransac_similarity,
    transform_bounds,
)
from reconstruction.change_detection import classify_changes, match_objects


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.source = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3], [2, 1, 1]], float
        )
        angle = np.pi / 2
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        self.transform = {
            "scale": 2.5,
            "rotation": rotation,
            "translation": np.array([4, -2, 7.0]),
        }
        self.target = apply_similarity(self.source, self.transform)

    def test_umeyama_recovers_similarity_and_inverse(self):
        recovered = estimate_similarity(self.source, self.target)
        np.testing.assert_allclose(
            apply_similarity(self.source, recovered), self.target, atol=1e-8
        )
        np.testing.assert_allclose(
            apply_similarity(self.target, inverse_similarity(recovered)),
            self.source,
            atol=1e-8,
        )

    def test_ransac_rejects_outlier_and_assesses_held_out_support(self):
        target = self.target.copy()
        target[0] += [20, 20, 20]
        result = ransac_similarity(
            self.source, target, threshold=1e-6, iterations=100, seed=4
        )
        self.assertEqual(int(result["inliers"].sum()), 4)
        quality = assess_alignment(
            self.source,
            target,
            result,
            max_residual=1e-5,
            held_out_source=self.source[1:3],
            held_out_target=self.target[1:3],
        )
        self.assertEqual(quality.status, "accepted")

    def test_bounds_transform_uses_all_corners(self):
        bounds = np.array([[0, 0, 0], [1, 2, 3]], float)
        transformed = transform_bounds(bounds, self.transform)
        points = apply_similarity(
            np.array([[x, y, z] for x in (0, 1) for y in (0, 2) for z in (0, 3)]),
            self.transform,
        )
        np.testing.assert_allclose(transformed, [points.min(0), points.max(0)])


def object_record(identifier, label, centroid, descriptor, extent=(1, 1, 1)):
    center = np.asarray(centroid, float)
    half = np.asarray(extent, float) / 2
    return {
        "id": identifier,
        "label": label,
        "centroid": center.tolist(),
        "bounds": [(center - half).tolist(), (center + half).tolist()],
        "descriptor": np.asarray(descriptor, float),
    }


class ChangeDetectionTests(unittest.TestCase):
    def test_moved_object_matches_globally_and_is_classified_once(self):
        before = [object_record("b-chair", "chair", [0, 0, 0], [1, 0, 0])]
        after = [object_record("a-chair", "chair", [5, 0, 0], [1, 0, 0])]
        matching = match_objects(before, after, config={"position_scale": 1})
        self.assertEqual(matching["unmatched_before"], [])
        self.assertEqual(matching["unmatched_after"], [])
        identity = {"scale": 1, "rotation": np.eye(3), "translation": np.zeros(3)}
        result = classify_changes(
            before,
            after,
            matching,
            tolerance=1,
            after_transform=identity,
            alignment_status="accepted",
        )
        self.assertEqual([x["type"] for x in result["events"]], ["moved"])

    def test_unmatched_objects_require_opposing_visibility(self):
        before = [object_record("b-box", "box", [0, 0, 0], [1, 0, 0])]
        after = []
        matching = match_objects(before, after)
        uncertain = classify_changes(before, after, matching, tolerance=1)
        self.assertEqual(uncertain["events"][0]["type"], "uncertain")
        removed = classify_changes(
            before,
            after,
            matching,
            tolerance=1,
            before_visible={"b-box"},
            after_visible=set(),
            alignment_status="accepted",
            after_free={"b-box"},
        )
        self.assertEqual(removed["events"][0]["type"], "removed")

    def test_ambiguous_identity_is_preserved(self):
        before = [object_record("b-chair", "chair", [0, 0, 0], [1, 0, 0])]
        after = [
            object_record("a-one", "chair", [0, 0, 0], [1, 0, 0]),
            object_record("a-two", "chair", [0, 0, 0], [1, 0, 0]),
        ]
        matching = match_objects(before, after, config={"ambiguity_margin": 0.1})
        self.assertEqual(matching["matches"][0]["status"], "ambiguous")
        result = classify_changes(before, after, matching, tolerance=1)
        self.assertEqual(result["events"][0]["type"], "uncertain")


if __name__ == "__main__":
    unittest.main()
