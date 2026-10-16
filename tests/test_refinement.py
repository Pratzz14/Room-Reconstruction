"""Synthetic refinement, provenance and provider-budget regressions; no live calls."""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from reconstruction import core
from reconstruction.object_inspection import ObjectInspection
from reconstruction.object_pipeline import run as object_run
from reconstruction.refined_inventory import InventoryModes, RefinedInventory
from reconstruction.refinement import (
    build,
    category_coverage,
    merge_inventory,
    normalize_evidence,
    run,
    validate_answer,
)
from reconstruction.refinement_evidence import prepare, schema
from reconstruction.refinement_gemini import Gemini, validate_output
from tests.fixtures import PROJECT, completed
from tests.test_objects import AnalyticModels


def config():
    return core.read_json(PROJECT / "configs/refinement.json")


class InventoryVisibilityTests(unittest.TestCase):
    def test_unknowns_are_excluded_in_both_modes_without_mutating_saved_lists(self):
        unknowns = [
            {"id": str(i), "label": label}
            for i, label in enumerate(["unknown_object", "Unknown", " unidentified "])
        ]
        named = {"id": "before_bottle", "label": "bottle"}
        raw_objects = unknowns + [named]
        original = SimpleNamespace(listing=lambda _: {"objects": raw_objects})
        refined = SimpleNamespace(listing=lambda _: {"objects": [named]})
        reader = InventoryModes.__new__(InventoryModes)
        reader.original, reader.refined, reader.mode = original, refined, "refined"
        self.assertEqual(reader.listing("before")["objects"], [named])
        reader.use_mode("original")
        self.assertEqual(reader.listing("before")["objects"], [named])
        self.assertEqual(len(raw_objects), 4)


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=PROJECT / "outputs/.test_work")
        self.base = Path(self.temp.name)
        self.now = 1792022400.0
        self.calls = []

    def tearDown(self):
        self.temp.cleanup()

    def sleep(self, seconds):
        self.now += seconds

    def transport(self, url, **kwargs):
        self.calls.append((url, self.now))
        if url.endswith(":countTokens"):
            result = {"totalTokens": 1234}
        else:
            result = {
                "candidates": [
                    {"finishReason": "STOP", "content": {"parts": [{"text": "{}"}]}}
                ],
                "usageMetadata": {"promptTokenCount": 1234},
            }
        return SimpleNamespace(status_code=200, json=lambda: result)

    def client(self, output="analysis", cfg=None):
        client = Gemini(
            self.base / output,
            cfg or config(),
            transport=self.transport,
            clock=lambda: self.now,
            sleep=self.sleep,
        )
        client.key = "SYNTHETIC_SECRET_DO_NOT_LOG"
        return client

    def test_one_call_cache_and_pacing(self):
        client = self.client()
        self.assertEqual(client.generate("sys", "prompt", schema(), []), {})
        self.assertEqual(client.generate("sys", "prompt", schema(), []), {})
        self.assertEqual(len(self.calls), 2)  # token preflight + one generation
        client.generate("sys", "targeted followup", schema(), [])
        generations = [t for url, t in self.calls if url.endswith(":generateContent")]
        self.assertGreaterEqual(generations[1] - generations[0], 65)
        with self.assertRaisesRegex(ValueError, "budget exhausted"):
            client.generate("sys", "third generation", schema(), [])
        self.assertEqual(
            len([u for u, _ in self.calls if u.endswith(":generateContent")]), 2
        )

    def test_token_limit_prevents_generation(self):
        cfg = config()
        cfg["max_input_tokens"] = 1000
        with self.assertRaisesRegex(ValueError, "token count"):
            self.client(cfg=cfg).generate("sys", "too large", schema(), [])
        self.assertTrue(all(u.endswith(":countTokens") for u, _ in self.calls))

    def test_rate_limit_not_retried_and_budget_persisted(self):
        client = self.client()
        original = self.transport

        def limited(url, **kwargs):
            if url.endswith(":countTokens"):
                return original(url, **kwargs)
            self.calls.append((url, self.now))
            return SimpleNamespace(
                status_code=429, json=lambda: {"error": {"message": "quota exhausted"}}
            )

        client.transport = limited
        with self.assertRaisesRegex(ValueError, "429"):
            client.generate("sys", "rate limited", schema(), [])
        self.assertEqual(
            core.read_json(client.output / "budget.json")["generation_attempts"], 1
        )
        before = len(self.calls)
        with self.assertRaisesRegex(ValueError, "Previous request"):
            client.generate("sys", "rate limited", schema(), [])
        self.assertEqual(len(self.calls), before)
        self.assertNotIn(
            client.key, "".join(p.read_text() for p in client.output.rglob("*.json"))
        )

    def test_pacific_daily_cap(self):
        cfg = config()
        cfg["daily_generation_cap"] = 1
        self.client(cfg=cfg).generate("sys", "first", schema(), [])
        with self.assertRaisesRegex(ValueError, "daily generation cap"):
            self.client(output="other_analysis", cfg=cfg).generate(
                "sys", "next", schema(), []
            )

    def test_nested_output_rejected_without_writing_sources(self):
        with self.assertRaisesRegex(ValueError, "separate sibling"):
            validate_output(
                self.base / "reconstruction",
                self.base / "raw",
                self.base / "raw/refinement",
            )
        with self.assertRaisesRegex(ValueError, "separate sibling"):
            validate_output(self.base / "reconstruction", self.base / "raw", self.base)

    def test_timeout_and_incomplete_response_require_explicit_retry(self):
        client = self.client()
        original = self.transport

        def interrupted(url, **kwargs):
            if url.endswith(":countTokens"):
                return original(url, **kwargs)
            raise TimeoutError("Synthetic timeout")

        client.transport = interrupted
        with self.assertRaisesRegex(ValueError, "no automatic retry"):
            client.generate("sys", "timeout", schema(), [])
        self.assertEqual(
            core.read_json(client.output / "budget.json")["generation_attempts"], 1
        )
        client.transport = lambda *a, **kw: SimpleNamespace(
            status_code=200,
            json=lambda: {"candidates": [{"finishReason": "MAX_TOKENS"}]},
        )
        # Saved token preflight is reused and the explicit retry consumes the second attempt.
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            client.generate("sys", "timeout", schema(), [], retry_failed=True)
        self.assertEqual(
            core.read_json(client.output / "budget.json")["generation_attempts"], 2
        )


class RefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scratch = PROJECT / "outputs/.test_work"
        scratch.mkdir(parents=True, exist_ok=True)
        cls.temp = tempfile.TemporaryDirectory(prefix="refinement-", dir=scratch)
        cls.base = Path(cls.temp.name)
        cls.reconstruction = completed(cls.base / "reconstruction")
        cls.raw = cls.base / "raw"
        cfg = core.read_json(PROJECT / "configs/objects.json")
        cfg["device"] = "cpu"
        lock = {
            "status": "synthetic",
            "models": {
                k: {"fixture": True} for k in ("detector", "segmenter", "appearance")
            },
        }
        report = object_run(
            cls.reconstruction, cls.raw, cfg, lock, models=AnalyticModels()
        )
        if report["status"] == "failed":
            raise AssertionError(report)
        cls.config = config()
        cls.config.update(
            crop_size=64,
            context_frames_per_capture=2,
            focus_categories=[],
            detail_regions_per_capture=0,
            label_review_regions_per_capture=0,
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def evidence(self):
        output = self.base / self._testMethodName
        inventory, prepared = prepare(
            self.reconstruction, self.raw, output, self.config
        )
        records = core.read_json(output / "evidence_index.json")
        entities = []
        for cap in ("before", "after"):
            for label in ("chair", "cabinet"):
                ids = [
                    eid
                    for eid, rec in records.items()
                    if rec["capture"] == cap and rec.get("label") == label
                ]
                entities.append(
                    {
                        "key": cap + label,
                        "capture": cap,
                        "label": label,
                        "kind": "individual",
                        "confidence": "high",
                        "description": "SYNTHETIC analytic object",
                        "observations": [
                            {"evidence_id": e, "role": "whole"} for e in ids
                        ],
                        "context_evidence": [],
                        "reason": "Synthetic fixture identity",
                    }
                )
        answer = {
            "entities": entities,
            "links": [
                {
                    "before_key": "beforechair",
                    "after_key": "afterchair",
                    "confidence": "high",
                    "reason": "Same synthetic identity at different depth",
                    "alternatives": [],
                }
            ],
            "unresolved_evidence": [],
            "alignment_candidates": [],
            "limitations": ["SYNTHETIC analytic test, not room evidence"],
        }
        core.write_json(output / "answer.json", answer)
        return output, inventory, prepared, records, answer

    def test_duplicates_provenance_and_global_identity(self):
        output, inventory, prepared, _records, answer = self.evidence()
        before = {p: core.digest(p) for p in self.raw.rglob("*") if p.is_file()}
        result = build(inventory, output, prepared, answer)
        self.assertEqual(result["captures"]["before"]["refined_objects"], 2)
        self.assertEqual(result["links"][0]["status"], "proposed")
        checked = RefinedInventory(inventory, output)
        self.assertEqual(len(checked.listing("after")["objects"]), 2)
        self.assertEqual(
            before, {p: core.digest(p) for p in self.raw.rglob("*") if p.is_file()}
        )
        with patch(
            "reconstruction.refinement.Gemini.generate",
            side_effect=AssertionError("Provider must not be called"),
        ):
            rebuilt = run(
                self.reconstruction, self.raw, output, self.config, offline=True
            )
            run(self.reconstruction, self.raw, output, self.config, followup=False)
        self.assertEqual(rebuilt["links"], result["links"])
        modes = InventoryModes(inventory, output)
        modes.use_mode("original")
        self.assertEqual(modes.listing("before")["inventory_mode"], "original")
        modes.use_mode("refined")
        self.assertEqual(modes.listing("before")["inventory_mode"], "refined")

    def test_mixed_chair_backpack_separate_without_duplicated_geometry(self):
        output, inventory, prepared, _records, answer = self.evidence()
        chair = answer["entities"][0]
        for ob in chair["observations"]:
            ob["role"] = "mixed"
        backpack = {**copy.deepcopy(chair), "key": "beforebag", "label": "backpack"}
        answer["entities"].append(backpack)
        core.write_json(output / "answer.json", answer)
        result = build(inventory, output, prepared, answer)
        checked = RefinedInventory(inventory, output)
        objects = checked.listing("before")["objects"]
        selected = [o for o in objects if o["label"] in ("chair", "backpack")]
        self.assertEqual(len(selected), 2)
        self.assertTrue(
            all(o["centroid"] is None and o["point_count"] == 0 for o in selected)
        )
        inspection = ObjectInspection(checked)
        inspection.select("before", selected[0]["id"])
        with self.assertRaises(ValueError):
            inspection.options({"isolate": True})
        self.assertIsNone(inspection.fit(None))
        self.assertGreater(result["repair_tasks"], 0)

    def test_disjoint_same_frame_instances_not_trusted_as_one(self):
        output, inventory, prepared, _records, answer = self.evidence()
        answer["entities"][0]["observations"] += answer["entities"][1]["observations"]
        answer["entities"].pop(1)
        core.write_json(output / "answer.json", answer)
        build(inventory, output, prepared, answer)
        objects = core.read_json(output / "before/objects.json")["objects"]
        self.assertEqual(len(objects), 2)
        self.assertTrue(all(o["confidence"] == "low" for o in objects))
        self.assertTrue(all(not o["movement_centroid_usable"] for o in objects))
        self.assertTrue(all(o["identity_status"] == "uncertain" for o in objects))

    def test_missing_evidence_stays_unresolved_and_bad_references_rejected(self):
        output, inventory, prepared, records, answer = self.evidence()
        lost = answer["entities"][0]["observations"].pop()["evidence_id"]
        self.assertIn(lost, validate_answer(answer, records))
        core.write_json(output / "answer.json", answer)
        build(inventory, output, prepared, answer)
        listing = RefinedInventory(inventory, output).listing("before")
        self.assertIn(
            records[lost]["observation_id"], {o["id"] for o in listing["unresolved"]}
        )
        answer["entities"][0]["observations"][0]["evidence_id"] = "NOT_SUPPLIED"
        with self.assertRaisesRegex(ValueError, "evidence"):
            validate_answer(answer, records)

    def test_competing_matches_stay_ambiguous(self):
        output, inventory, prepared, _records, answer = self.evidence()
        answer["links"].append({**answer["links"][0], "after_key": "aftercabinet"})
        core.write_json(output / "answer.json", answer)
        report = build(inventory, output, prepared, answer)
        self.assertTrue(all(l["status"] == "ambiguous" for l in report["links"]))

    def test_collection_context_only_and_tamper_detection(self):
        output, inventory, prepared, records, answer = self.evidence()
        eid = next(
            e
            for e, r in records.items()
            if r["capture"] == "before" and "observation_id" not in r
        )
        answer["entities"].append(
            {
                "key": "books",
                "capture": "before",
                "label": "books",
                "kind": "collection",
                "confidence": "medium",
                "description": "Synthetic collection",
                "observations": [],
                "context_evidence": [eid],
                "reason": "Small clutter group",
            }
        )
        core.write_json(output / "answer.json", answer)
        build(inventory, output, prepared, answer)
        checked = RefinedInventory(inventory, output)
        books = next(
            o for o in checked.listing("before")["objects"] if o["label"] == "books"
        )
        self.assertIsNone(books["centroid"])
        self.assertEqual(
            checked.context_record("before", "context:" + eid)["frame_id"],
            records[eid]["frame_id"],
        )
        png = ObjectInspection(checked).evidence("before", "context:" + eid)
        self.assertTrue(png.startswith(b"\x89PNG"))
        core.write_json(output / "before/objects.json", {"objects": []})
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            RefinedInventory(inventory, output)

    def test_visible_laptop_cannot_hide_in_electronics_collection(self):
        output, inventory, prepared, _records, answer = self.evidence()
        source = answer["entities"][0]
        source.update(label="electronics", kind="collection")
        core.write_json(output / "answer.json", answer)
        report = build(inventory, output, prepared, answer)
        objects = core.read_json(output / "before/objects.json")["objects"]
        recovered = [
            o for o in objects if o["label"].startswith("electronics item candidate")
        ]
        self.assertEqual(len(recovered), len(source["observations"]))
        self.assertTrue(
            all(o["confidence"] == "low" and o["point_count"] == 0 for o in recovered)
        )
        self.assertTrue(all(o["observation_ids"] for o in recovered))
        self.assertGreater(report["repair_tasks"], 0)

    def test_category_review_requires_individual_names_and_context(self):
        _, _, _, records, answer = self.evidence()
        before = next(
            e
            for e, r in records.items()
            if r["capture"] == "before" and "observation_id" not in r
        )
        after = next(
            e
            for e, r in records.items()
            if r["capture"] == "after" and "observation_id" not in r
        )
        answer["entities"][0]["label"] = "closed laptop"
        answer["category_reviews"] = [
            {
                "capture": "before",
                "category": "laptop",
                "status": "visible",
                "entity_keys": ["beforechair"],
                "evidence_ids": [before],
                "reason": "Distinct laptop visible",
            },
            {
                "capture": "after",
                "category": "laptop",
                "status": "not_observed",
                "entity_keys": [],
                "evidence_ids": [after],
                "reason": "No laptop discernible in inspected views",
            },
        ]
        self.assertEqual(category_coverage(answer, records, ["laptop"]), [])
        answer["entities"][0]["label"] = "electronics"
        answer["entities"][0]["kind"] = "collection"
        self.assertTrue(
            any(
                "individually named" in x
                for x in category_coverage(answer, records, ["laptop"])
            )
        )
        answer["category_reviews"].pop()
        self.assertTrue(
            any(
                "after/laptop: category not reviewed" == x
                for x in category_coverage(answer, records, ["laptop"])
            )
        )

    def test_unsegmented_oil_bottle_is_named_without_inheriting_shelf_points(self):
        output, inventory, prepared, records, answer = self.evidence()
        context = next(
            e
            for e, r in records.items()
            if r["capture"] == "before" and "observation_id" not in r
        )
        answer["entities"].append(
            {
                "key": "oil",
                "capture": "before",
                "label": "oil bottle",
                "kind": "individual",
                "confidence": "high",
                "description": "Synthetic visible bottle on a shelf",
                "observations": [],
                "context_evidence": [context],
                "reason": "Visible but no bottle-specific mask exists",
            }
        )
        core.write_json(output / "answer.json", answer)
        build(inventory, output, prepared, answer)
        checked = RefinedInventory(inventory, output)
        bottle = next(
            o
            for o in checked.listing("before")["objects"]
            if o["label"] == "oil bottle"
        )
        self.assertEqual(bottle["geometry_status"], "missing")
        self.assertIsNone(bottle["centroid"])
        self.assertEqual(bottle["point_count"], 0)
        self.assertEqual(bottle["observation_ids"], [])
        self.assertTrue(
            any(
                t["label"] == "oil bottle"
                for t in core.read_json(output / "repair_tasks.json")
            )
        )

    def test_context_references_cannot_become_clean_geometry(self):
        _, inventory, _, records, answer = self.evidence()
        ent = answer["entities"][0]
        eid = ent["observations"][0]["evidence_id"]
        ent["observations"] = []
        ent["context_evidence"] = [eid]
        normalized = normalize_evidence(answer, records, inventory)
        self.assertEqual(normalized["entities"][0]["observations"], [])
        self.assertTrue(normalized["entities"][0]["context_evidence"])
        self.assertTrue(
            all(
                "observation_id" not in records[e]
                for e in normalized["entities"][0]["context_evidence"]
            )
        )
        validate_answer(normalized, records)

    def test_child_bottle_cannot_reuse_whole_cabinet_mask(self):
        _, inventory, _, records, answer = self.evidence()
        ent = answer["entities"][1]
        ent["label"] = "oil bottle"
        for item in ent["observations"]:
            item["role"] = "part"
        normalized = normalize_evidence(answer, records, inventory)
        self.assertTrue(
            all(
                o["role"] == "uncertain"
                for o in normalized["entities"][1]["observations"]
            )
        )

    def test_two_unsegmented_bottles_can_share_a_context_frame(self):
        output, inventory, prepared, records, answer = self.evidence()
        context = next(
            e
            for e, r in records.items()
            if r["capture"] == "before" and "observation_id" not in r
        )
        for key, appearance in [
            ("blue", "blue transparent bottle"),
            ("silver", "silver metal flask"),
        ]:
            answer["entities"].append(
                {
                    "key": key,
                    "capture": "before",
                    "label": "drinking bottle",
                    "kind": "individual",
                    "confidence": "high",
                    "description": appearance,
                    "observations": [],
                    "context_evidence": [context],
                    "reason": "Two distinct visible vessels",
                }
            )
        core.write_json(output / "answer.json", answer)
        build(inventory, output, prepared, answer)
        bottles = [
            o
            for o in core.read_json(output / "before/objects.json")["objects"]
            if o["label"] == "drinking bottle"
        ]
        self.assertEqual(len({o["id"] for o in bottles}), 2)
        self.assertTrue(all(o["centroid"] is None for o in bottles))

    def test_targeted_answer_preserves_prior_inventory_and_corrects_masks(self):
        _, _, _, _, old = self.evidence()
        new = copy.deepcopy(old)
        new["entities"] = [new["entities"][0]]
        new["entities"][0].update(key="laptop", label="laptop")
        new["entities"][0]["observations"] = new["entities"][0]["observations"][:1]
        new["links"] = []
        merged = merge_inventory(old, new)
        self.assertIn("beforecabinet", {e["key"] for e in merged["entities"]})
        claimed = new["entities"][0]["observations"][0]["evidence_id"]
        self.assertEqual(
            [
                e["key"]
                for e in merged["entities"]
                if any(o["evidence_id"] == claimed for o in e["observations"])
            ],
            ["laptop"],
        )

    def test_context_only_spoon_is_not_duplicated_under_utensils_label(self):
        _, _, _, records, old = self.evidence()
        context = next(e for e, r in records.items() if "observation_id" not in r)
        ent = copy.deepcopy(old["entities"][0])
        ent.update(
            key="oldspoon", label="spoon", observations=[], context_evidence=[context]
        )
        old["entities"] = [ent]
        new = copy.deepcopy(old)
        new["entities"][0].update(key="newutensils", label="utensils")
        new["links"] = []
        merged = merge_inventory(old, new)
        self.assertEqual(len(merged["entities"]), 1)


if __name__ == "__main__":
    unittest.main()
