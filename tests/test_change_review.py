"""Visual-review replay, evidence validation, quota and saved-workflow regressions."""

import copy
import json
import tempfile
import unittest
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from reconstruction.change_review import (
    apply_to_display,
    needs_followup,
    run,
    verify,
)
from reconstruction.change_stage import saved_changes
from reconstruction.core import digest, identity, read_json, write_json
from reconstruction.refinement_gemini import Gemini, RequestTooLarge, exclusive
from reconstruction.review_analysis import reconcile, validate_finding
from reconstruction.review_config import configuration, live_limits
from reconstruction.review_evidence import (
    SYSTEM,
    candidate_checks,
    candidate_confirmed,
    request,
    request_schema,
    schema,
    system_instruction,
)
from tests.fixtures import PROJECT


def obj(capture, name, x=0):
    return {
        "id": name,
        "capture": capture,
        "label": "chair",
        "centroid": [x, 0, 0],
        "point_count": 3,
        "bounds": [[x, 0, 0], [x + 0.1, 0.1, 0.1]],
        "observation_ids": [],
        "frame_ids": [capture + "0"],
        "coordinate_frame": capture + "_world",
        "confidence": "high",
    }


def presence(capture, oid, state="present"):
    return {
        "presence": state,
        "object_id": oid,
        "region_visible": True,
        "evidence": [{"frame_id": capture + "0", "box": [100, 100, 300, 400]}],
    }


def finding(sid="s1", before="b1", after="a1", movement="moved"):
    return {
        "shared_id": sid,
        "before": presence("before", before, "present" if before else "absent"),
        "after": presence("after", after, "present" if after else "absent"),
        "identity": "same",
        "movement": movement,
        "alternatives": [],
        "explanation": "Distinctive chair visible beside the same fixed wall features.",
    }


class AnalysisTests(unittest.TestCase):
    def test_presence_confirmation_accepts_supplied_alternate_view_not_unseen_or_new_counterpart(
        self,
    ):
        from reconstruction.review_evidence import supplied_presence_confirmation

        index = self.index()
        f = finding(after=None, movement="unclear")
        step = {
            "followup": True,
            "prompt_profile": "visual-grounding-native-v3",
            "images": [],
        }
        index["events"]["s1"]["evidence"] = {}
        for cap, role in [("before", "object_before"), ("after", "old_location_after")]:
            sha = cap + "_image"
            index["events"]["s1"]["evidence"][role] = {
                "views": [
                    {
                        "frame_id": cap + "0",
                        "region_box": [10, 8, 30, 32],
                        "width": 100,
                        "height": 80,
                        "image_sha256": sha,
                    }
                ]
            }
            step["images"].append(
                {"path": "assets/" + identity([sha, "full"]) + ".jpg"}
            )
        self.assertTrue(supplied_presence_confirmation(f, "s1", index, step))
        missing = copy.deepcopy(step)
        missing["images"].pop()
        self.assertFalse(supplied_presence_confirmation(f, "s1", index, missing))
        wrong = copy.deepcopy(f)
        wrong["after"]["evidence"][0]["box"] = [700, 700, 900, 900]
        self.assertFalse(supplied_presence_confirmation(wrong, "s1", index, step))
        new = copy.deepcopy(f)
        new["before"]["object_id"] = None
        self.assertFalse(supplied_presence_confirmation(new, "s1", index, step))
        step["followup"] = False
        self.assertFalse(supplied_presence_confirmation(f, "s1", index, step))

    def test_native_gemini_box_order_is_decoded_without_altering_receipt_answer(self):
        from reconstruction.change_review import _load_findings

        raw = finding()
        for cap in ("before", "after"):
            raw[cap]["evidence"][0]["box"] = [100, 100, 400, 300]
        original = copy.deepcopy(raw)
        step = {
            "ids": ["s1"],
            "followup": False,
            "answer": {"reviews": [raw]},
            "prompt_profile": "visual-grounding-native-v3",
        }
        decoded = _load_findings([step], self.index())["s1"]
        self.assertFalse(decoded["errors"])
        self.assertEqual(
            decoded["finding"]["before"]["evidence"][0]["box"], [100, 100, 300, 400]
        )
        self.assertEqual(raw, original)

    def test_legacy_coordinate_recovery_is_explicit_source_bound_and_replayable(self):
        from reconstruction.change_review import _load_findings
        from reconstruction.review_analysis import recover_supplied_pixel_boxes

        row, index = finding(), self.index()
        box = [1200, 120, 1600, 700]
        index["events"]["s1"]["evidence"] = {}
        for c in ("before", "after"):
            index["frames"][c + "0"].update(width=1920, height=1080)
            row[c]["evidence"][0]["box"] = list(box)
            index["events"]["s1"]["evidence"]["object_" + c] = {
                "views": [{"frame_id": c + "0", "region_box": list(box)}]
            }
        step = {"ids": ["s1"], "followup": False, "answer": {"reviews": [row]}}
        self.assertEqual(
            _load_findings([step], index)["s1"]["errors"], ["invalid_full_frame_box"]
        )
        original = copy.deepcopy(row)
        step["recover_supplied_pixel_boxes"] = ["s1"]
        fixed = _load_findings([step], index)["s1"]
        self.assertFalse(fixed["errors"])
        self.assertEqual(fixed, _load_findings([step], index)["s1"])
        self.assertEqual(row, original)
        self.assertEqual(len(fixed["coordinate_repairs"]), 2)
        self.assertAlmostEqual(fixed["finding"]["before"]["evidence"][0]["box"][0], 625)
        for bad in (
            [1200, 700, 1600, 120],
            [1200, 120, 2000, 700],
            [100, 100, 300, 400],
            [float("nan"), 120, 1600, 700],
        ):
            invalid = copy.deepcopy(row)
            invalid["after"]["evidence"][0]["box"] = bad
            with self.assertRaises(ValueError):
                recover_supplied_pixel_boxes(invalid, "s1", index)
        with self.assertRaises(ValueError):
            recover_supplied_pixel_boxes(fixed["finding"], "s1", index)
        step["recover_supplied_pixel_boxes"] = ["unknown"]
        with self.assertRaises(ValueError):
            _load_findings([step], index)

    def test_local_visual_validation_only_downgrades_with_source_reference(self):
        from reconstruction.change_review import apply_local_validation

        analysis = {
            "changes": {
                "events": [
                    {
                        "source_event_ids": ["s"],
                        "type": "unchanged",
                        "trust": "estimated",
                        "review_status": "confirmed",
                        "uncertainties": [],
                        "explanation": "Model claim.",
                    }
                ],
                "counts": {"unchanged": 1},
            },
            "coverage": {"uncertain": 0},
        }
        index = {"events": {"s": {}}, "frames": {"f": {"sha256": "bound-image"}}}
        validation = {
            "schema_version": "local-visual-validation-v1",
            "rejections": [
                {
                    "shared_id": "s",
                    "frame_id": "f",
                    "image_sha256": "bound-image",
                    "reason": "Claimed object is not supported by the cited picture.",
                }
            ],
        }
        actual = apply_local_validation(copy.deepcopy(analysis), validation, index)
        self.assertEqual(actual["changes"]["counts"], {"uncertain": 1})
        self.assertEqual(actual["coverage"]["uncertain"], 1)
        self.assertEqual(actual["changes"]["events"][0]["review_status"], "uncertain")
        validation["rejections"][0]["image_sha256"] = "different-image"
        with self.assertRaises(ValueError):
            apply_local_validation(copy.deepcopy(analysis), validation, index)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=PROJECT / "outputs/.test_work")
        self.root = Path(self.temp.name)
        captures = {}
        for cap in ("before", "after"):
            (self.root / cap).mkdir()
            pts = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10]], float)
            np.savez(self.root / cap / "cloud.npz", points=pts)
            np.savez(
                self.root / cap / "observations.npz",
                frame_ids=[cap + "0"],
                observation_frame_index=[0, 0, 0, 0, 0],
                cloud_point_index=[0, 0, 1, 2, 3],
            )
            captures[cap] = {
                "objects": [
                    obj(
                        cap,
                        "b1" if cap == "before" else "a1",
                        0 if cap == "before" else 8,
                    )
                ],
                "frames": [
                    {
                        "frame_id": cap + "0",
                        "width": 100,
                        "height": 80,
                        "elapsed_timestamp_sec": 0,
                    }
                ],
                "observations": [],
                "geometry": {"offsets": np.array([0, 0]), "points": np.empty((0, 3))},
            }
        self.inv = SimpleNamespace(reconstruction=self.root, captures=captures)
        self.align = {
            "quality": {"status": "accepted"},
            "transform": {
                "scale": 1.0,
                "rotation": np.eye(3).tolist(),
                "translation": [0, 0, 0],
            },
        }
        self.originals = [
            {
                "shared_id": "s1",
                "before_id": "b1",
                "after_id": "a1",
                "status": "matched",
            }
        ]
        self.matching = {"matches": []}

    def tearDown(self):
        self.temp.cleanup()

    def index(self):
        return {
            "frames": {c + "0": {"capture": c} for c in self.inv.captures},
            "catalog": {
                c: [{"id": o["id"]} for o in d["objects"]]
                for c, d in self.inv.captures.items()
            },
            "events": {o["shared_id"]: {**o, "label": "chair"} for o in self.originals},
        }

    def evaluate(self, rows, *, localized=True, confirmed=False, rejected=None):
        changes = saved_changes(self.inv, self.originals, self.matching, self.align, {})
        return reconcile(
            self.inv,
            changes,
            self.originals,
            self.matching,
            self.align,
            {
                r["shared_id"]: {
                    "finding": r,
                    "localized_followup": localized,
                    **({"identity_confirmation": True} if confirmed else {}),
                }
                for r in rows
            },
            self.index(),
            rejected=rejected,
        )

    def test_rejected_visual_claim_does_not_create_derived_counterpart(self):
        f = finding()
        f["after"]["object_id"] = None
        self.assertTrue(self.evaluate([f])["derived_observations"])
        rejected = self.evaluate([f], rejected={"s1"})
        self.assertFalse(rejected["derived_observations"])
        self.assertEqual(rejected["changes"]["events"][0]["type"], "uncertain")

    def test_moved_unchanged_and_visual_conflict(self):
        result = self.evaluate([finding()])
        event = result["changes"]["events"][0]
        self.assertEqual(
            (event["type"], event["review_status"], event["object_displacement"]),
            ("moved", "confirmed", 8),
        )
        event = self.evaluate([finding(movement="unchanged")])["changes"]["events"][0]
        self.assertEqual(event["type"], "uncertain")
        self.assertIn("visual_and_location_evidence_conflict", event["uncertainties"])
        self.inv.captures["after"]["objects"][0]["centroid"] = [0, 0, 0]
        self.assertEqual(
            self.evaluate([finding(movement="unchanged")])["changes"]["events"][0][
                "type"
            ],
            "unchanged",
        )

    def test_targeted_identity_confirmation_can_resolve_source_prior_but_not_collections(
        self,
    ):
        self.inv.captures["before"]["objects"][0]["confidence"] = "low"
        self.assertEqual(
            self.evaluate([finding()])["changes"]["events"][0]["type"], "uncertain"
        )
        self.assertEqual(
            self.evaluate([finding()], confirmed=True, localized=False)["changes"][
                "events"
            ][0]["type"],
            "uncertain",
        )
        confirmed = self.evaluate([finding()], confirmed=True)["changes"]["events"][0]
        self.assertEqual(confirmed["type"], "moved")
        self.assertTrue(confirmed["identity_confirmations"])
        self.assertEqual(self.inv.captures["before"]["objects"][0]["confidence"], "low")
        self.inv.captures["before"]["objects"][0]["kind"] = "collection"
        self.assertEqual(
            self.evaluate([finding()], confirmed=True)["changes"]["events"][0]["type"],
            "uncertain",
        )

    def test_rejected_alignment_stays_uncertain_and_pending_is_explicit(self):
        self.align["quality"]["status"] = "rejected"
        self.assertEqual(
            self.evaluate([finding()])["changes"]["events"][0]["type"], "uncertain"
        )
        pending = self.evaluate([])
        self.assertEqual(pending["coverage"]["pending"], 1)
        self.assertEqual(
            pending["changes"]["events"][0]["review_status"], "not_reviewed"
        )
        self.assertFalse(
            needs_followup(self.evaluate([finding()])["changes"]["events"][0])
        )

    def test_fixed_inventory_quality_alone_does_not_need_another_vlm_call(self):
        self.inv.captures["before"]["objects"][0]["confidence"] = "low"
        event = self.evaluate([finding()])["changes"]["events"][0]
        self.assertFalse(needs_followup(event))
        row = finding()
        row["after"]["evidence"][0]["box"] = [0, 0, 1500, 1500]
        self.assertTrue(needs_followup(self.evaluate([row])["changes"]["events"][0]))

    def test_exact_threshold_and_missing_frame_support(self):
        threshold = saved_changes(
            self.inv, self.originals, self.matching, self.align, {}
        )["threshold"]
        self.inv.captures["after"]["objects"][0]["centroid"] = [threshold, 0, 0]
        self.assertEqual(
            self.evaluate([finding(movement="unchanged")])["changes"]["events"][0][
                "type"
            ],
            "unchanged",
        )
        self.inv.captures["after"]["objects"][0].update(
            centroid=None, geometry_status="missing", frame_ids=[]
        )
        self.assertEqual(
            self.evaluate([finding(movement="unchanged")])["changes"]["events"][0][
                "type"
            ],
            "uncertain",
        )

    def test_removed_requires_visible_opposing_region_and_added_is_symmetric(self):
        self.inv.captures["after"]["objects"] = []
        self.originals[0]["after_id"] = None
        self.originals[0]["status"] = "unmatched"
        row = finding(after=None)
        event = self.evaluate([row])["changes"]["events"][0]
        self.assertEqual(
            (event["type"], event["basis"], event["trust"]),
            ("removed", "visually_supported_presence", "estimated"),
        )
        unconfirmed = self.evaluate([row], localized=False)["changes"]["events"][0]
        self.assertEqual(unconfirmed["type"], "uncertain")
        self.assertIn(
            "presence_change_requires_closeup_confirmation",
            unconfirmed["uncertainties"],
        )
        checks = candidate_checks({"s1": {"finding": row, "errors": []}}, ["s1"])
        self.assertTrue(candidate_confirmed(row, checks["s1"]))
        wrong_region = copy.deepcopy(row)
        wrong_region["after"]["evidence"][0]["box"] = [800, 800, 900, 900]
        self.assertFalse(candidate_confirmed(wrong_region, checks["s1"]))
        for state in ("occluded", "outside_frame", "unclear"):
            row["after"]["presence"] = state
            self.assertEqual(
                self.evaluate([row])["changes"]["events"][0]["type"], "uncertain"
            )
        self.inv.captures["before"]["objects"] = []
        self.inv.captures["after"]["objects"] = [obj("after", "a1")]
        self.originals[0].update(before_id=None, after_id="a1")
        self.assertEqual(
            self.evaluate([finding(before=None)])["changes"]["events"][0]["type"],
            "added",
        )

    def test_presence_claim_must_account_for_an_existing_matched_counterpart(self):
        events = self.evaluate([finding(after=None)])["changes"]["events"]
        self.assertTrue(all(e["type"] == "uncertain" for e in events))
        self.assertTrue(
            any(
                "unresolved_inventoried_counterpart" in e["uncertainties"]
                for e in events
            )
        )

    def test_invalid_hallucinated_nonfinite_and_wrong_capture_evidence(self):
        for mutate in (
            lambda r: r["after"]["evidence"][0].update(frame_id="missing"),
            lambda r: r["after"]["evidence"][0].update(frame_id="before0"),
            lambda r: r["before"]["evidence"][0].update(box=[0, 0, float("nan"), 4]),
            lambda r: r["after"].update(object_id="b1"),
            lambda r: r["after"].update(evidence=[]),
        ):
            row = finding()
            mutate(row)
            self.assertTrue(validate_finding(row, "s1", self.index())["errors"])
            self.assertEqual(
                self.evaluate([row])["changes"]["events"][0]["type"], "uncertain"
            )

    def test_cross_batch_identity_correction_merges_add_remove_once(self):
        self.originals = [
            {
                "shared_id": "s1",
                "before_id": "b1",
                "after_id": None,
                "status": "unmatched",
            },
            {
                "shared_id": "s2",
                "before_id": None,
                "after_id": "a1",
                "status": "unmatched",
            },
        ]
        result = self.evaluate([finding(), finding(sid="s2")])
        self.assertEqual(len(result["changes"]["events"]), 1)
        self.assertEqual(result["changes"]["events"][0]["type"], "moved")
        self.assertEqual(result["changes"]["events"][0]["review_status"], "corrected")
        self.assertEqual(
            result["changes"]["events"][0]["source_event_ids"], ["s1", "s2"]
        )

    def test_competitors_including_unmatched_are_uncertain(self):
        self.inv.captures["before"]["objects"].append(obj("before", "b2"))
        self.originals.append(
            {
                "shared_id": "s2",
                "before_id": "b2",
                "after_id": None,
                "status": "unmatched",
            }
        )
        rows = [finding(), finding(sid="s2", before="b2")]
        events = self.evaluate(rows)["changes"]["events"]
        self.assertTrue(all(e["type"] == "uncertain" for e in events))
        row = finding()
        row["alternatives"] = [{"capture": "before", "object_id": "b2"}]
        self.assertTrue(
            all(
                e["type"] == "uncertain"
                for e in self.evaluate([row])["changes"]["events"]
            )
        )

    def test_vlm_cannot_assign_a_supporting_shelf_as_the_object(self):
        shelf = obj("before", "b2")
        shelf["label"] = "bookshelf"
        self.inv.captures["before"]["objects"].append(shelf)
        self.originals.append(
            {
                "shared_id": "s2",
                "before_id": "b2",
                "after_id": None,
                "status": "unmatched",
            }
        )
        events = self.evaluate([finding(sid="s2", before="b2")])["changes"]["events"]
        self.assertTrue(all(e["type"] == "uncertain" for e in events))
        self.assertTrue(
            any(
                "incompatible_semantic_identity_assignment" in e["uncertainties"]
                for e in events
            )
        )

    def test_missed_counterpart_is_image_only_and_both_means_used(self):
        self.inv.captures["after"]["objects"] = []
        self.originals[0].update(after_id=None, status="unmatched")
        row = finding(movement="unchanged")
        row["after"]["object_id"] = None
        result = self.evaluate([row])
        derived = result["derived_observations"][0]["object"]
        self.assertIsNone(derived["centroid"])
        self.assertEqual(derived["point_count"], 0)
        event = result["changes"]["events"][0]
        self.assertEqual(
            (event["type"], event["basis"], event["frame_cloud_distance"]),
            ("unchanged", "frame_cloud_mean", 0),
        )
        self.assertEqual(event["before_location"], [2.5, 2.5, 2.5])
        self.assertIsNone(event["object_displacement"])
        self.assertEqual(derived["context_evidence"], [derived["id"]])
        unconfirmed = self.evaluate([row], localized=False)["changes"]["events"][0]
        self.assertEqual(unconfirmed["type"], "uncertain")
        self.assertIn(
            "new_counterpart_requires_closeup_confirmation",
            unconfirmed["uncertainties"],
        )


class ReviewQuotaTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=PROJECT / "outputs/.test_work")
        self.root = Path(self.temp.name)
        self.now = 1792022400.0
        self.calls = []
        self.cfg = configuration(
            {
                "quota_verified": True,
                "free_tier_verified": True,
                "quota_verified_at": self.now,
                "remaining_requests": 100,
                "requests_per_day": 100,
                "requests_per_minute": 10,
                "input_tokens_per_minute": 100000,
            }
        )

    def tearDown(self):
        self.temp.cleanup()

    def transport(self, url, **kwargs):
        self.calls.append((url, self.now, kwargs["json"]))
        payload = (
            {"totalTokens": 1234}
            if url.endswith(":countTokens")
            else {
                "candidates": [
                    {"finishReason": "STOP", "content": {"parts": [{"text": "{}"}]}}
                ]
            }
        )
        return SimpleNamespace(status_code=200, json=lambda: payload)

    def sleep(self, sec):
        self.now += sec

    def client(self, name="review", cfg=None):
        client = Gemini(
            self.root / name,
            cfg or self.cfg,
            transport=self.transport,
            clock=lambda: self.now,
            sleep=self.sleep,
        )
        client.key = "SYNTHETIC_SECRET"
        return client

    def test_more_than_two_calls_cached_and_spaced(self):
        client = self.client()
        for n in range(12):
            client.generate(SYSTEM, str(n), schema(), [])
        times = [t for u, t, _ in self.calls if u.endswith(":generateContent")]
        self.assertEqual(len(times), 12)
        self.assertTrue(all(b - a >= 65 for a, b in pairwise(times)))
        calls = len(self.calls)
        self.client("other", configuration()).generate(SYSTEM, "0", schema(), [])
        self.assertEqual(calls, len(self.calls))

    def test_unverified_stale_or_exhausted_profile_never_sends(self):
        for cfg in (
            configuration(),
            {**self.cfg, "quota_verified_at": self.now - 90000},
            {**self.cfg, "remaining_requests": 0},
        ):
            with self.assertRaises(ValueError):
                self.client(
                    str(len(self.calls)) + str(cfg["remaining_requests"]), cfg
                ).generate(SYSTEM, "x", schema(), [])
        self.assertFalse(self.calls)

    def test_all_requests_count_and_shared_daily_allowance_survives_outputs(self):
        cfg = {**self.cfg, "remaining_requests": 3}
        self.client(cfg=cfg).generate(SYSTEM, "1", schema(), [])
        with self.assertRaisesRegex(ValueError, "allowance exhausted"):
            self.client("second", cfg).generate(SYSTEM, "2", schema(), [])
        self.assertEqual(len(self.calls), 3)

    def test_explicit_model_continuation_has_separate_cache_and_shared_quota(self):
        first = self.client(cfg={**self.cfg, "remaining_requests": 2})
        first.generate(SYSTEM, "same evidence", schema(), [])
        second = self.client(
            "alternate",
            {**self.cfg, "model": "gemini-3.5-flash-lite", "remaining_requests": 2},
        )
        self.assertEqual(first.cache, second.cache)
        second.generate(SYSTEM, "same evidence", schema(), [])
        generations = [u for u, _, _ in self.calls if u.endswith(":generateContent")]
        self.assertEqual(len(generations), 2)
        self.assertIn("gemini-3.5-flash-lite", generations[-1])
        with self.assertRaisesRegex(ValueError, "allowance exhausted"):
            second.generate(SYSTEM, "new evidence", schema(), [])
        self.assertEqual(len(self.calls), 4)

    def test_mime_limits_pacific_reset_and_concurrent_lock(self):
        bounded = request_schema()["properties"]["reviews"]["items"]["properties"][
            "before"
        ]["properties"]["evidence"]["items"]["properties"]["box"]["anyOf"][0]["items"]
        self.assertEqual((bounded["minimum"], bounded["maximum"]), (0, 1000))
        client = self.client()
        image = self.root / "picture.png"
        Image.new("RGB", (10, 10)).save(image)
        client.generate(SYSTEM, "image", schema(), [image])
        self.assertEqual(
            self.calls[-1][2]["contents"][0]["parts"][1]["inlineData"]["mimeType"],
            "image/png",
        )
        with (
            exclusive(client.cache / ".quota.lock"),
            self.assertRaisesRegex(ValueError, "Writer lock"),
        ):
            client.generate(SYSTEM, "concurrent", schema(), [])
        self.assertNotEqual(client._day(self.now), client._day(self.now + 90000))
        with self.assertRaises(ValueError):
            live_limits(self.cfg, self.now + 90000)
        self.assertEqual(live_limits(self.cfg, self.now)["requests_per_minute"], 9)

    def test_token_overflow_failure_persistence_and_no_automatic_retry(self):
        small = self.client(cfg={**self.cfg, "max_input_tokens": 100})
        with self.assertRaises(RequestTooLarge):
            small.generate(SYSTEM, "large", schema(), [])
        self.assertTrue(all(u.endswith(":countTokens") for u, _, _ in self.calls))
        client = self.client("failed")

        def failure(url, **kwargs):
            if url.endswith(":countTokens"):
                return self.transport(url, **kwargs)
            self.calls.append((url, self.now, {}))
            return SimpleNamespace(
                status_code=429,
                json=lambda: {"error": {"message": "SYNTHETIC_SECRET quota"}},
            )

        client.transport = failure
        with self.assertRaisesRegex(ValueError, "429"):
            client.generate(SYSTEM, "fail", schema(), [])
        count = len(self.calls)
        with self.assertRaisesRegex(ValueError, "Previous request"):
            client.generate(SYSTEM, "fail", schema(), [])
        self.assertEqual(count, len(self.calls))
        self.assertEqual(
            read_json(client.output / "budget.json")["generation_attempts"], 1
        )
        self.assertNotIn(
            client.key, "".join(p.read_text() for p in client.output.rglob("*.json"))
        )
        client.transport = self.transport
        client.generate(SYSTEM, "fail", schema(), [], retry_failed=True)
        history = list((client.output / "requests" / "attempts").glob("*.json"))
        self.assertTrue(history)
        self.assertTrue(any("429" in read_json(p).get("error", "") for p in history))


class SavedReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.test_change_report import ReportWorkflowTests

        cls.fixture = ReportWorkflowTests
        cls.fixture.setUpClass()

    @classmethod
    def tearDownClass(cls):
        cls.fixture.tearDownClass()

    def test_offline_prepare_resume_reports_viewer_and_tamper(self):
        from reconstruction.change_report import (
            build_report,
            pack_report,
            verify_report,
        )
        from reconstruction.comparison import ComparisonInventory
        from reconstruction.comparison_viewer import ComparisonState

        f = self.fixture
        output = f.base / "review"
        before = digest(f.comparison / "changes.json")
        pending = run(f.inventory, f.comparison, output, prepare_only=True)
        self.assertEqual(pending["coverage"]["reviewed"], 0)
        verify(output, f.inventory, f.comparison)
        partial_report = build_report(
            f.inventory, f.comparison, f.base / "pending-report", review=output
        )
        self.assertTrue(
            any("review is incomplete" in w for w in partial_report["warnings"])
        )
        pending_display = apply_to_display(
            ComparisonInventory(f.inventory, f.comparison), pending
        )
        self.assertEqual(
            ComparisonState(
                f.reconstruction, pending_display, f.comparison
            ).review_status,
            "partial",
        )
        # A newer renderer must not invalidate the source-bound JSON/image evidence.
        with patch(
            "reconstruction.report_render.render_html",
            return_value="<p>New report presentation</p>",
        ):
            self.assertEqual(verify(output, f.inventory, f.comparison), pending)
        index = read_json(output / "evidence_index.json")
        prompt, images = request(index, sorted(index["events"]), output)
        self.assertNotIn('"type":', prompt)
        self.assertNotIn('"distance":', prompt)
        self.assertTrue(images)
        answer = {"reviews": []}
        originals = read_json(f.comparison / "changes.json")["events"]
        for e in originals:
            row = finding(
                e["shared_id"],
                e["before_id"],
                e["after_id"],
                e["type"] if e["type"] in ("moved", "unchanged") else "unclear",
            )
            for cap in ("before", "after"):
                view = index["events"][e["shared_id"]]["evidence"]["object_" + cap][
                    "views"
                ][0]
                row[cap]["evidence"][0]["frame_id"] = view["frame_id"]
            answer["reviews"].append(row)
        answer_path = f.base / "answers.json"
        write_json(answer_path, answer)
        reviewed = run(f.inventory, f.comparison, output, answers=answer_path)
        self.assertEqual(reviewed["status"], "completed")
        self.assertEqual(reviewed, verify(output, f.inventory, f.comparison))
        self.assertEqual(before, digest(f.comparison / "changes.json"))
        self.assertEqual(
            run(f.inventory, f.comparison, output, answers=answer_path), reviewed
        )

        class FakeClient:
            def __init__(self, fail=False, split=False, unclear=False, discovery=False):
                self.calls = []
                self.fail, self.split, self.unclear = fail, split, unclear
                self.discovery = discovery

            def _clean(self, value):
                return value

            def generate(self, system, prompt, response_schema, images, **kwargs):
                payload = json.loads(prompt)
                requested = [r["shared_id"] for r in payload["requested"]]
                self.calls.append(requested)
                if self.split and len(requested) > 1:
                    raise RequestTooLarge("synthetic oversized batch")
                if self.fail and len(self.calls) == 2:
                    raise ValueError("synthetic quota exhaustion")
                rows = copy.deepcopy(
                    [r for r in answer["reviews"] if r["shared_id"] in requested]
                )
                if self.unclear and "additional" not in payload["task"]:
                    for row in rows:
                        row["movement"] = "unclear"
                if self.discovery and "additional" not in payload["task"]:
                    rows[0]["after"]["object_id"] = None
                return {"reviews": rows}

        split_client = FakeClient(split=True)
        split_root = f.base / "split-review"
        report_hash = digest(f.output / "manifest.json")
        split = run(
            f.inventory,
            f.comparison,
            split_root,
            client=split_client,
            evidence_report=f.output,
        )
        self.assertEqual(report_hash, digest(f.output / "manifest.json"))
        self.assertEqual(split["status"], "completed")
        self.assertEqual(split["changes"], reviewed["changes"])
        self.assertEqual(len(split_client.calls), 3)
        verify(split_root, f.inventory, f.comparison)
        v2_root = f.base / "v2-prompt-review"
        v2 = run(
            f.inventory,
            f.comparison,
            v2_root,
            client=FakeClient(),
            prompt_profile="visual-grounding-v2",
        )
        self.assertEqual(v2["changes"], reviewed["changes"])
        self.assertEqual(verify(v2_root, f.inventory, f.comparison), v2)
        v2_index = read_json(v2_root / "evidence_index.json")
        v2_prompt, v2_images = request(
            v2_index,
            sorted(v2_index["events"]),
            v2_root,
            profile="visual-grounding-v2",
            render=False,
        )
        self.assertNotIn("_original_pixels", v2_prompt)
        self.assertTrue(any("gallery_" in p.name for p in v2_images))
        self.assertEqual(v2["inputs"]["prompt_profile"], "visual-grounding-v2")
        interrupted_root = f.base / "interrupted-review"
        partial = run(
            f.inventory,
            f.comparison,
            interrupted_root,
            client=FakeClient(fail=True),
            batch_limit=1,
        )
        self.assertEqual(
            (partial["status"], partial["coverage"]["reviewed"]), ("partial", 1)
        )
        prepared_partial = run(
            f.inventory, f.comparison, interrupted_root, prepare_only=True
        )
        self.assertEqual(prepared_partial["status"], "partial")
        self.assertEqual(prepared_partial["stopped_reason"], partial["stopped_reason"])
        resumed_client = FakeClient()
        # A process can die after a receipt update but before publishing its manifest.
        write_json(
            interrupted_root / "budget.json",
            {"generation_attempts": 1, "token_count_attempts": 1},
        )
        resumed = run(
            f.inventory,
            f.comparison,
            interrupted_root,
            client=resumed_client,
            batch_limit=1,
            execution_config={"model": "gemini-3.5-flash-lite"},
        )
        self.assertEqual(resumed["status"], "completed")
        self.assertEqual(len(resumed_client.calls), 1)
        saved_steps = read_json(interrupted_root / "steps.json")
        self.assertEqual(
            saved_steps[0]["provider_config"]["model"], "gemini-3.1-flash-lite"
        )
        self.assertEqual(
            saved_steps[1]["provider_config"]["model"], "gemini-3.5-flash-lite"
        )
        self.assertEqual(
            read_json(interrupted_root / "inputs.json")["model"],
            "gemini-3.1-flash-lite",
        )
        verify(interrupted_root, f.inventory, f.comparison)
        # Replay a mixed-model review using synthetic provider receipts. Old batches
        # without provider_config keep the original model, including after resume.
        from reconstruction.change_review import publish

        receipt_root = f.base / "mixed-model-receipts"
        run(f.inventory, f.comparison, receipt_root, prepare_only=True)
        receipt_index = read_json(receipt_root / "evidence_index.json")
        receipt_steps = []
        for n, row in enumerate(answer["reviews"]):
            cfg = configuration({"model": "gemini-3.5-flash-lite"} if n else {})
            ids = [row["shared_id"]]
            profile = "visual-grounding-native-v3" if n else "legacy"
            raw = copy.deepcopy(row)
            if n:
                for cap in ("before", "after"):
                    b = raw[cap]["evidence"][0]["box"]
                    raw[cap]["evidence"][0]["box"] = [b[1], b[0], b[3], b[2]]
            prompt, images = request(receipt_index, ids, receipt_root, profile=profile)
            step = {
                "ids": ids,
                "followup": False,
                "answer": {"reviews": [raw]},
                "origin": "gemini",
                "prompt_sha256": identity(prompt),
                "response_schema": request_schema(profile),
                "images": [
                    {
                        "path": p.relative_to(receipt_root).as_posix(),
                        "sha256": digest(p),
                    }
                    for p in images
                ],
            }
            if n:
                step["provider_config"] = cfg
                step["prompt_profile"] = profile
            rid = identity(
                [
                    cfg["model"],
                    system_instruction(profile),
                    prompt,
                    request_schema(profile),
                    [(p.name, digest(p)) for p in images],
                    cfg["max_output_tokens"],
                ]
            )
            write_json(
                receipt_root / "requests" / (rid + ".json"),
                {
                    "status": "completed",
                    "model": cfg["model"],
                    "answer": step["answer"],
                },
            )
            receipt_steps.append(step)
        mixed = publish(
            receipt_root,
            f.inventory,
            f.comparison,
            configuration(),
            receipt_steps,
            receipt_index,
        )
        self.assertEqual(verify(receipt_root, f.inventory, f.comparison), mixed)
        archived = copy.deepcopy(receipt_steps[0])
        archived["active_ids"] = []
        merged_steps = [archived, *receipt_steps]
        merged = publish(
            receipt_root,
            f.inventory,
            f.comparison,
            configuration(),
            merged_steps,
            receipt_index,
        )
        self.assertEqual(verify(receipt_root, f.inventory, f.comparison), merged)
        self.assertEqual(merged["changes"], mixed["changes"])
        archived["active_ids"] = ["unknown"]
        with self.assertRaises(ValueError):
            publish(
                receipt_root,
                f.inventory,
                f.comparison,
                configuration(),
                merged_steps,
                receipt_index,
            )
        receipt_steps[-1]["provider_config"]["model"] = "gemini-3.1-flash-lite"
        publish(
            receipt_root,
            f.inventory,
            f.comparison,
            configuration(),
            receipt_steps,
            receipt_index,
        )
        with self.assertRaises((ValueError, FileNotFoundError)):
            verify(receipt_root, f.inventory, f.comparison)
        follow_client = FakeClient(unclear=True)
        with patch("reconstruction.change_review.has_extra", return_value=True):
            follow = run(
                f.inventory,
                f.comparison,
                f.base / "follow-review",
                client=follow_client,
            )
        self.assertEqual(follow["status"], "completed")
        self.assertEqual(len(follow_client.calls), 2)
        self.assertEqual(follow["changes"], reviewed["changes"])
        discovery_client = FakeClient(discovery=True)
        checked = run(
            f.inventory,
            f.comparison,
            f.base / "checked-discovery",
            client=discovery_client,
        )
        self.assertEqual(len(discovery_client.calls), 2)
        self.assertFalse(checked["derived_observations"])
        self.assertEqual(checked["changes"]["counts"], reviewed["changes"]["counts"])
        report_root = f.base / "review-report"
        report = build_report(f.inventory, f.comparison, report_root, review=output)
        verify_report(report_root, f.inventory, f.comparison, review=output)
        self.assertEqual(
            report["scene"]["change_counts"], reviewed["changes"]["counts"]
        )
        pack_report(
            report_root,
            f.base / "review-report.zip",
            f.inventory,
            f.comparison,
            review=output,
        )
        display = apply_to_display(
            ComparisonInventory(f.inventory, f.comparison), reviewed
        )
        state = ComparisonState(
            f.reconstruction, display, f.comparison, report=report_root
        )
        self.assertEqual(state.counts, reviewed["changes"]["counts"])
        self.assertEqual(state.review_status, "completed")
        self.assertTrue(all(item["review_status"] for item in state.items.values()))
        discovery_answers = copy.deepcopy(answer)
        discovery_answers["reviews"][0]["after"]["object_id"] = None
        discovery_path = f.base / "discovery-answers.json"
        write_json(discovery_path, discovery_answers)
        discovery_root = f.base / "discovery-review"
        discovery = run(
            f.inventory, f.comparison, discovery_root, answers=discovery_path
        )
        self.assertEqual(len(discovery["derived_observations"]), 1)
        derived_row = discovery_answers["reviews"][0]
        checks = candidate_checks(
            {derived_row["shared_id"]: {"finding": derived_row, "errors": []}},
            [derived_row["shared_id"]],
        )
        self.assertTrue(
            candidate_confirmed(derived_row, checks[derived_row["shared_id"]])
        )
        wrong = copy.deepcopy(derived_row)
        wrong["after"]["evidence"][0]["box"] = [800, 800, 900, 900]
        self.assertFalse(candidate_confirmed(wrong, checks[derived_row["shared_id"]]))
        verify(discovery_root, f.inventory, f.comparison)
        derived_report = build_report(
            f.inventory,
            f.comparison,
            f.base / "discovery-report",
            review=discovery_root,
        )
        derived = discovery["derived_observations"][0]["object"]
        derived_event = next(
            e for e in derived_report["events"] if e["after_id"] == derived["id"]
        )
        self.assertTrue(derived_event["evidence"]["object_after"]["views"])
        self.assertIsNone(derived_event["measurements"]["source_centroids"]["after"])
        display = apply_to_display(
            ComparisonInventory(f.inventory, f.comparison), discovery
        )
        derived_state = ComparisonState(f.reconstruction, display, f.comparison)
        session = derived_state.begin_session()["session"]
        snapshot = derived_state.update(
            {"session": session, "sequence": 0, "select": derived_event["shared_id"]}
        )
        self.assertEqual(snapshot["selected"]["sides"]["after"]["id"], derived["id"])
        edited = copy.deepcopy(reviewed)
        edited["changes"]["events"][0]["type"] = "added"
        write_json(output / "review.json", edited)
        manifest = read_json(output / "manifest.json")
        manifest["artifacts"]["review.json"] = digest(output / "review.json")
        write_json(output / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "replayed"):
            verify(output, f.inventory, f.comparison)


if __name__ == "__main__":
    unittest.main()
