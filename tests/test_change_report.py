"""Saved-input evidence, report integrity, coordinate and HTTP regression checks."""

import copy
import json
import tempfile
import threading
import unittest
import zipfile
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageFilter

from reconstruction.alignment import apply_similarity
from reconstruction.change_report import (
    build_report,
    event_record,
    pack_report,
    refresh_report,
    verify_report,
)
from reconstruction.comparison import ComparisonInventory, run
from reconstruction.comparison_viewer import ComparisonState, make_comparison_handler
from reconstruction.core import digest, read_json, write_json
from reconstruction.object_contracts import ObjectInventory
from reconstruction.object_pipeline import run as detect
from reconstruction.region_reference import RegionReferences
from reconstruction.report_evidence import (
    EvidenceSelector,
    crop_box,
    depth_visibility,
    sharpness,
)
from reconstruction.report_render import (
    comparison_views,
    render_html,
    render_markdown,
)
from tests.fixtures import PROJECT, completed
from tests.test_comparison import analytic_correspondences
from tests.test_objects import AnalyticModels


class EvidenceUnitTests(unittest.TestCase):
    def test_comparison_pair_reuses_location_views_for_missing_objects(self):
        def bundle(capture, fid):
            return {
                "capture": capture,
                "views": [{"frame_id": fid}] if fid else [],
                "limitations": ["visibility_unknown"],
            }

        event = {
            "evidence": {
                "object_before": bundle("before", "before-object"),
                "object_after": bundle("after", None),
                "old_location_after": bundle("after", "after-location"),
                "new_location_before": bundle("before", "before-location"),
            }
        }
        original = copy.deepcopy(event)
        pairs = comparison_views(event)
        self.assertEqual(pairs["before"]["views"][0]["frame_id"], "before-object")
        self.assertEqual(pairs["after"]["views"][0]["frame_id"], "after-location")
        self.assertFalse(pairs["before"]["reference_only"])
        self.assertTrue(pairs["after"]["reference_only"])
        self.assertEqual(event, original)
        event["evidence"]["object_before"]["views"] = []
        self.assertEqual(
            comparison_views(event)["before"]["role"], "new_location_before"
        )
        event["evidence"]["new_location_before"]["views"] = []
        self.assertEqual(comparison_views(event)["before"]["views"], [])

    def test_ranking_prefers_sharp_unclipped_direct_views_and_deduplicates(self):
        selector = EvidenceSelector.__new__(EvidenceSelector)
        selector.config = {"max_views": 3}
        selector.regions = SimpleNamespace(
            cameras_for=lambda c: [{"camera_to_world": np.eye(4)}] * 4
        )

        def row(fid, blur, clipped=False, method="object_mask"):
            return {
                "frame_id": fid,
                "source_id": fid,
                "frame_index": 0,
                "capture": "before",
                "region_box": [0, 0, 5, 5],
                "method": method,
                "ranking": {
                    "coverage": 0.1,
                    "border_clearance": float(not clipped),
                    "sharpness_normalized": blur,
                    "geometry_support_normalized": 1,
                    "occlusion_penalty": 0,
                },
            }

        rows = [
            row("blurred", 0.01),
            row("sharp", 0.9),
            row("clipped", 0.9, True),
            row("context", 1, method="context_region"),
            row("sharp", 0.2),
        ]
        result = selector.select(rows)
        self.assertEqual(result[0]["frame_id"], "sharp")
        self.assertEqual(result[1]["frame_id"], "clipped")
        self.assertEqual(len({r["frame_id"] for r in result}), len(result))
        self.assertEqual(
            selector.select(list(reversed(copy.deepcopy(rows))))[0]["frame_id"], "sharp"
        )

    def test_clipped_invalid_and_padded_crops(self):
        self.assertEqual(crop_box([1, 1, 9, 9], (10, 10)), [0, 0, 10, 10])
        for box in ([5, 5, 1, 1], [np.nan, 0, 1, 1], [20, 20, 22, 22], None):
            self.assertIsNone(crop_box(box, (10, 10)))

    def test_sharpness_detects_blur(self):
        y, x = np.indices((100, 100))
        image = Image.fromarray((((x // 4 + y // 4) % 2) * 255).astype(np.uint8))
        self.assertGreater(
            sharpness(image, None),
            sharpness(image.filter(ImageFilter.GaussianBlur(4)), None),
        )

    def test_occlusion_needs_valid_distinct_pixels_and_never_proves_absence(self):
        camera = {"world_to_camera": np.eye(4), "intrinsics": np.eye(3)}
        y, x = np.indices((5, 5))
        points = np.c_[x.ravel() * 2, y.ravel() * 2, np.full(25, 2)]
        result = depth_visibility(
            points, camera, np.ones((6, 6)), np.ones((6, 6), bool), 0.1
        )
        self.assertEqual(result["status"], "suspected_occlusion")
        self.assertFalse(result["independently_validated"])
        duplicate = np.repeat(points[:1], 100, axis=0)
        self.assertEqual(
            depth_visibility(
                duplicate, camera, np.ones((6, 6)), np.ones((6, 6), bool), 0.1
            )["status"],
            "unknown",
        )
        self.assertEqual(
            depth_visibility(
                points, camera, np.ones((6, 6)), np.zeros((6, 6), bool), 0.1
            )["samples"],
            0,
        )

    def test_proxy_never_becomes_an_object_centroid_or_displacement(self):
        event = {
            "shared_id": "shared",
            "before_id": "b",
            "after_id": "a",
            "label": "chair",
            "type": "moved",
            "trust": "estimated",
            "basis": "frame_cloud_mean",
            "object_displacement": None,
            "frame_cloud_distance": 3,
            "before_location": [0, 0, 0],
            "after_location": [3, 0, 0],
            "threshold": 1,
            "explanation": "estimate",
            "uncertainties": [],
        }
        obj = {
            "centroid": None,
            "coordinate_frame": "before_world",
            "label": "chair",
            "id": "b",
        }
        record = event_record(
            event,
            {"status": "matched"},
            {"before": obj, "after": {**obj, "id": "a"}},
            {
                "quality": {"status": "accepted"},
                "transform": {
                    "scale": 2,
                    "rotation": np.eye(3),
                    "translation": [1, 0, 0],
                },
            },
            "http://localhost:8094/",
        )
        self.assertIsNone(record["measurements"]["displacement"])
        self.assertEqual(
            record["measurements"]["centroids"], {"before": None, "after": None}
        )
        self.assertEqual(record["measurements"]["frame_cloud_estimate"]["distance"], 3)
        self.assertEqual(record["trust"], "estimated")


class ReportWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scratch = PROJECT / "outputs/.test_work"
        scratch.mkdir(parents=True, exist_ok=True)
        cls.temp = tempfile.TemporaryDirectory(prefix="report-", dir=scratch)
        cls.base = Path(cls.temp.name)
        cls.reconstruction = completed(cls.base / "reconstruction")
        cls.objects = cls.base / "objects"
        cfg = read_json(PROJECT / "configs/objects.json")
        cfg["device"] = "cpu"
        lock = {
            "models": {
                k: {"fixture": True} for k in ("detector", "segmenter", "appearance")
            }
        }
        detect(cls.reconstruction, cls.objects, cfg, lock, models=AnalyticModels())
        cls.inventory = ObjectInventory(cls.reconstruction, cls.objects)
        cfg = read_json(PROJECT / "configs/comparison.json")
        cfg["ransac_iterations"] = 100
        c, _ = analytic_correspondences()
        cls.comparison = cls.base / "comparison"
        with patch(
            "reconstruction.comparison.extract_correspondences",
            return_value=(c, ["room_shell"]),
        ):
            cls.compared = run(
                cls.reconstruction, cls.objects, None, cls.comparison, cfg
            )
        cls.output = cls.base / "report"
        cls.report = build_report(cls.inventory, cls.comparison, cls.output)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_saved_reports_are_deterministic_and_do_not_mutate_sources(self):
        second = self.base / "repeat"
        before = digest(self.comparison / "changes.json")
        repeated = build_report(self.inventory, self.comparison, second)
        self.assertEqual(repeated, self.report)
        self.assertEqual(before, digest(self.comparison / "changes.json"))
        verify_report(second, self.inventory, self.comparison)
        with self.assertRaises(FileExistsError):
            build_report(self.inventory, self.comparison, second)
        packed = self.base / "report.zip"
        pack_report(second, packed, self.inventory, self.comparison)
        with zipfile.ZipFile(packed) as zipped:
            self.assertIn("report.html", zipped.namelist())
            self.assertIsNone(zipped.testzip())
        with self.assertRaises(FileExistsError):
            pack_report(second, packed)

    def test_refresh_presentation_preserves_evidence_and_source_export(self):
        original_hash = digest(self.output / "manifest.json")
        destination = self.base / "paired"
        refreshed = refresh_report(
            self.output,
            destination,
            self.inventory,
            self.comparison,
            viewer_url="http://127.0.0.1:8120/",
        )
        self.assertEqual(digest(self.output / "manifest.json"), original_hash)
        for old, new in zip(self.report["events"], refreshed["events"]):
            self.assertEqual(old["evidence"], new["evidence"])
            self.assertEqual(old["type"], new["type"])
            self.assertEqual(old["trust"], new["trust"])
            self.assertTrue(
                new["visualization"]["viewer_url"].startswith("http://127.0.0.1:8120/")
            )
        verify_report(destination, self.inventory, self.comparison)
        rendered = (destination / "report.html").read_text(encoding="utf-8")
        self.assertEqual(
            rendered.count('class="paired-comparison"'), len(refreshed["events"])
        )
        self.assertEqual(
            rendered.count('class="role comparison-side"'), 2 * len(refreshed["events"])
        )
        self.assertNotIn("No suitable saved view.", rendered)
        with self.assertRaises(FileExistsError):
            refresh_report(self.output, destination, self.inventory, self.comparison)

    def test_full_frames_retain_bytes_and_all_exports_share_counts(self):
        report = verify_report(self.output, self.inventory, self.comparison)
        self.assertEqual(
            report["scene"]["change_counts"],
            read_json(self.comparison / "changes.json")["counts"],
        )
        self.assertIn(
            json.dumps(report["scene"]["change_counts"], sort_keys=True),
            render_markdown(report),
        )
        self.assertIn("Scene summary", render_html(report))
        for event in report["events"]:
            for bundle in event["evidence"].values():
                for view in bundle["views"]:
                    self.assertEqual(
                        digest(self.output / view["images"]["full"]),
                        view["image_sha256"],
                    )
                    with Image.open(self.output / view["images"]["full"]) as image:
                        self.assertEqual(image.size, (view["width"], view["height"]))

    def test_rejected_alignment_blocks_opposing_projection_and_missing_objects(self):
        alignment = copy.deepcopy(self.compared["alignment"])
        alignment["quality"]["status"] = "rejected"
        selector = EvidenceSelector(self.inventory, alignment)
        before = self.inventory.captures["before"]["objects"][0]
        bundles = selector.event(before, None)
        self.assertTrue(bundles["object_before"]["views"])
        self.assertEqual(
            bundles["old_location_after"]["limitations"], ["alignment_unusable"]
        )
        self.assertEqual(bundles["object_after"]["limitations"], ["no_supporting_view"])

    def test_masks_context_invalid_region_and_no_geometry_remain_distinct(self):
        inventory = copy.copy(self.inventory)
        inventory.captures = copy.deepcopy(self.inventory.captures)
        obj = inventory.captures["before"]["objects"][0]
        observation = obj["observation_ids"][0]
        record = inventory.captures["before"]["observations"][
            inventory.captures["before"]["index"][observation]
        ]
        context = {**record, "status": "context", "source_region_box": [4, 4, 12, 12]}
        inventory.context_record = lambda c, eid: context
        obj.update(observation_ids=[observation], context_evidence=["test"])
        selector = EvidenceSelector(inventory, self.compared["alignment"])
        rows = selector.object_candidates("before", obj)
        self.assertEqual(rows[0]["method"], "object_mask")
        self.assertEqual(rows[1]["method"], "context_region")
        self.assertIsNone(rows[1]["source_mask_sha256"])
        context["source_region_box"] = [-4, -4, -1, -1]
        row = selector.object_candidates("before", obj)[1]
        self.assertIsNone(row["region_box"])
        exported = selector.export(row, self.base)
        self.assertIsNone(exported["images"]["crop"])
        obj.update(
            point_count=0,
            observation_ids=[],
            geometry_observation_ids=[],
            context_evidence=[],
        )
        self.assertEqual(
            selector.event(obj, None)["old_location_after"]["limitations"],
            ["no_supporting_view"],
        )

    def test_semantic_event_tampering_is_rejected_even_with_updated_hashes(self):
        original = self.report
        altered = copy.deepcopy(original)
        altered["events"][0]["type"] = "invented"
        manifest = read_json(self.output / "manifest.json")
        changed = copy.deepcopy(manifest)
        try:
            write_json(self.output / "report.json", altered)
            (self.output / "summary.md").write_text(
                render_markdown(altered), encoding="utf-8"
            )
            (self.output / "report.html").write_text(
                render_html(altered), encoding="utf-8"
            )
            for file in ("report.json", "summary.md", "report.html"):
                changed["artifacts"][file] = digest(self.output / file)
            write_json(self.output / "manifest.json", changed)
            with self.assertRaisesRegex(ValueError, "measurements/classification"):
                verify_report(self.output, self.inventory, self.comparison)
        finally:
            write_json(self.output / "report.json", original)
            (self.output / "summary.md").write_text(
                render_markdown(original), encoding="utf-8"
            )
            (self.output / "report.html").write_text(
                render_html(original), encoding="utf-8"
            )
            write_json(self.output / "manifest.json", manifest)

    def test_standalone_projection_matches_viewer_and_transforms_after_once(self):
        selector = EvidenceSelector(self.inventory, self.compared["alignment"])
        obj = self.inventory.captures["after"]["objects"][0]
        raw = self.inventory.object("after", obj["id"])[1]["points"]
        actual = selector.regions.region_points("after", obj, None)
        expected = apply_similarity(raw, self.compared["alignment"]["transform"])
        if len(expected) > 512:
            expected = expected[np.linspace(0, len(expected) - 1, 512).astype(int)]
        np.testing.assert_allclose(actual, expected)
        adapted = ComparisonInventory(self.inventory, self.comparison)
        views = {
            c: SimpleNamespace(
                points={c: np.empty((0, 3))},
                cameras={c: selector.regions.cameras_for(c)},
            )
            for c in ("before", "after")
        }
        existing = RegionReferences(adapted, views)
        aligned_obj = adapted.object("after", obj["id"])[0]
        np.testing.assert_allclose(
            existing.region_points("after", aligned_obj, None), actual, atol=1e-6
        )

    def test_corrupt_asset_stale_sources_and_escaping_paths_are_rejected(self):
        manifest = read_json(self.output / "manifest.json")
        path = next(x for x in manifest["artifacts"] if x.startswith("assets/"))
        target = self.output / path
        saved = target.read_bytes()
        try:
            target.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_report(self.output)
        finally:
            target.write_bytes(saved)
        bad = copy.deepcopy(manifest)
        bad["artifacts"]["assets/../../escaped.png"] = "x"
        try:
            write_json(self.output / "manifest.json", bad)
            with self.assertRaises(ValueError):
                verify_report(self.output)
        finally:
            write_json(self.output / "manifest.json", manifest)
        altered = copy.deepcopy(self.report)
        altered["sources"]["comparison_sha256"] = "stale"
        try:
            write_json(self.output / "report.json", altered)
            with self.assertRaises(ValueError):
                verify_report(self.output, self.inventory, self.comparison)
        finally:
            write_json(self.output / "report.json", self.report)

    def test_escape_labels_and_http_allowlist_and_report_primary_selection(self):
        altered = copy.deepcopy(self.report)
        altered["events"][0]["label"] = '<script>alert("unsafe")</script> @@COUNTS@@'
        rendered = render_html(altered)
        self.assertNotIn('<script>alert("unsafe")', rendered)
        self.assertIn("&lt;script&gt;", rendered)
        self.assertIn("@@COUNTS@@", rendered)
        inventory = ComparisonInventory(self.inventory, self.comparison)
        state = ComparisonState(
            self.reconstruction,
            inventory,
            self.comparison,
            maximum=100,
            report=self.output,
        )
        event = self.report["events"][0]
        for c in ("before", "after"):
            preferred = event["evidence"]["object_" + c]["views"]
            if preferred:
                self.assertEqual(
                    state.items[event["event_id"]]["sides"][c]["frames"][0]["id"],
                    preferred[0]["source_id"],
                )
        http = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            make_comparison_handler(state, {"before": 9001, "after": 9002}),
        )
        thread = threading.Thread(target=http.serve_forever, daemon=True)
        thread.start()
        connection = HTTPConnection("127.0.0.1", http.server_port)
        try:
            for path, expected in (
                ("/report", 200),
                ("/report.json", 200),
                ("/summary.md", 200),
                ("/assets/../../report.json", 404),
                ("/assets/not-bound.png", 404),
            ):
                connection.request("GET", path)
                response = connection.getresponse()
                response.read()
                self.assertEqual(response.status, expected, path)
        finally:
            connection.close()
            http.shutdown()
            http.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
