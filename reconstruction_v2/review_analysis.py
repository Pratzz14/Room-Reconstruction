"""Validate visual claims and reconcile identities without inventing geometry."""

import copy
import math
from collections import Counter, defaultdict

import numpy as np

from .change_detection import label_family
from .change_stage import saved_changes
from .core import identity
from .object_contracts import ObjectInventory

CAPTURES = ("before", "after")


def recover_supplied_pixel_boxes(finding, sid, index):
    """Recover an explicitly marked legacy response only when its units are evidenced.

    Every box must match a pixel region actually supplied for this identity/frame.
    Out-of-range, reversed, invented or ambiguous boxes are never guessed or clipped.
    """
    recovered = copy.deepcopy(finding)
    repairs = []
    for capture in CAPTURES:
        for ref in recovered[capture]["evidence"]:
            box = ref["box"]
            if box is None:
                continue
            frame = index["frames"].get(ref["frame_id"], {})
            if (
                frame.get("capture") != capture
                or not isinstance(box, list)
                or len(box) != 4
                or any(
                    isinstance(x, bool)
                    or not isinstance(x, (int, float))
                    or not math.isfinite(x)
                    for x in box
                )
                or not 0 <= box[0] < box[2] <= frame["width"]
                or not 0 <= box[1] < box[3] <= frame["height"]
            ):
                raise ValueError("Unsafe legacy pixel-coordinate recovery")
            views = [
                v
                for role in index["events"][sid]["evidence"].values()
                for v in role["views"]
                if v["frame_id"] == ref["frame_id"] and v.get("region_box") is not None
            ]
            if not any(
                max(abs(a - b) for a, b in zip(box, v["region_box"])) <= 2
                for v in views
            ):
                raise ValueError("Legacy box does not match supplied pixel evidence")
            normalized = [
                box[i] * 1000 / frame["width" if i % 2 == 0 else "height"]
                for i in range(4)
            ]
            repairs.append(
                {
                    "capture": capture,
                    "frame_id": ref["frame_id"],
                    "original_pixel_box": box,
                    "normalized_box": normalized,
                }
            )
            ref["box"] = normalized
    if not repairs or not any(max(r["original_pixel_box"]) > 1000 for r in repairs):
        raise ValueError("Legacy response has ambiguous coordinate units")
    return recovered, repairs


def validate_finding(finding, sid, index):
    errors = []
    try:
        if (
            set(finding)
            != {
                "shared_id",
                "before",
                "after",
                "identity",
                "movement",
                "alternatives",
                "explanation",
            }
            or finding["shared_id"] != sid
        ):
            raise ValueError("invalid_review_fields")
        if finding["identity"] not in ("same", "different", "ambiguous") or finding[
            "movement"
        ] not in ("moved", "unchanged", "unclear"):
            raise ValueError("invalid_identity_or_movement")
        if (
            not isinstance(finding["explanation"], str)
            or not finding["explanation"].strip()
        ):
            raise ValueError("missing_explanation")
        anchored = False
        for capture in CAPTURES:
            side = finding[capture]
            ids = {o["id"] for o in index["catalog"][capture]}
            if set(side) != {"presence", "object_id", "region_visible", "evidence"}:
                raise ValueError("invalid_presence_fields")
            if (
                side["presence"]
                not in ("present", "absent", "occluded", "outside_frame", "unclear")
                or type(side["region_visible"]) is not bool
            ):
                raise ValueError("invalid_presence_or_visibility")
            oid = side["object_id"]
            if oid is not None and (not isinstance(oid, str) or oid not in ids):
                raise ValueError("unknown_capture_object")
            anchored |= oid is not None and oid == index["events"][sid][capture + "_id"]
            if not isinstance(side["evidence"], list):
                raise TypeError("invalid_evidence_list")
            for ref in side["evidence"]:
                if (
                    set(ref) != {"frame_id", "box"}
                    or index["frames"].get(ref["frame_id"], {}).get("capture")
                    != capture
                ):
                    raise ValueError("unknown_or_wrong_capture_frame")
                box = ref["box"]
                if box is not None and (
                    not isinstance(box, list)
                    or len(box) != 4
                    or any(
                        isinstance(x, bool)
                        or not isinstance(x, (int, float))
                        or not math.isfinite(x)
                        or not 0 <= x <= 1000
                        for x in box
                    )
                    or box[0] >= box[2]
                    or box[1] >= box[3]
                ):
                    raise ValueError("invalid_full_frame_box")
            if side["presence"] in ("present", "absent") and (
                not side["evidence"]
                or not any(r["box"] is not None for r in side["evidence"])
            ):
                raise ValueError("unsupported_presence_claim")
            if side["presence"] == "absent" and (
                not side["region_visible"] or oid is not None
            ):
                raise ValueError("absence_requires_visible_region_and_null_id")
        if not anchored:
            raise ValueError("review_does_not_reference_target_object")
        if not isinstance(finding["alternatives"], list):
            raise TypeError("invalid_alternatives")
        for alt in finding["alternatives"]:
            if (
                set(alt) != {"capture", "object_id"}
                or alt["capture"] not in CAPTURES
                or alt["object_id"]
                not in {o["id"] for o in index["catalog"][alt["capture"]]}
            ):
                raise ValueError("unknown_identity_alternative")
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(str(exc))
    return {"finding": finding, "errors": errors}


class AugmentedInventory:
    """Read-only wrapper adding image-only observations; source arrays are retained."""

    def __init__(self, source, derived):
        self.__dict__.update(source.__dict__)
        self.review_base = source
        self.review_context = {}
        self.captures = {
            c: {
                **d,
                "objects": copy.deepcopy(d["objects"]),
                "geometry": dict(d["geometry"]),
            }
            for c, d in source.captures.items()
        }
        for item in derived:
            obj, record = item["object"], item["context"]
            data = self.captures[obj["capture"]]
            data["objects"].append(copy.deepcopy(obj))
            data["geometry"]["offsets"] = np.append(
                data["geometry"]["offsets"], data["geometry"]["offsets"][-1]
            )
            for context in item.get("contexts", [record]):
                self.review_context[context["id"]] = context

    def object(self, capture, identifier):
        return ObjectInventory.object(self, capture, identifier)

    def context_record(self, capture, identifier):
        if identifier in self.review_context:
            record = self.review_context[identifier]
            if record["capture"] != capture:
                raise ValueError("Wrong capture for reviewed evidence")
            return record
        return self.review_base.context_record(capture, identifier)

    def listing(self, capture):
        return {
            **self.review_base.listing(capture),
            "objects": self.captures[capture]["objects"],
            **({"comparison": self.comparison} if hasattr(self, "comparison") else {}),
        }


def derived_observation(capture, side, label, inventory):
    refs = sorted(
        (r for r in side["evidence"] if r["box"] is not None),
        key=lambda r: (r["frame_id"], r["box"]),
    )
    # One stable observation per identity/capture, with all visible supporting frames.
    key = "review_" + identity([capture, label, refs])[:20]
    frame_map = {
        f["frame_id"]: (i, f)
        for i, f in enumerate(inventory.captures[capture]["frames"])
    }
    contexts = []
    for position, ref in enumerate(refs):
        index, frame = frame_map[ref["frame_id"]]
        box = ref["box"]
        eid = key if position == 0 else key + "_" + str(position)
        contexts.append(
            {
                **frame,
                "capture": capture,
                "id": "context:" + eid,
                "frame_index": index,
                "timestamp_sec": frame["elapsed_timestamp_sec"],
                "status": "context",
                "source_region_box": [
                    box[0] * frame["width"] / 1000,
                    box[1] * frame["height"] / 1000,
                    box[2] * frame["width"] / 1000,
                    box[3] * frame["height"] / 1000,
                ],
                "label": label,
                "quality": ["vlm_image_only_observation"],
            }
        )
    obj = {
        "id": key,
        "capture": capture,
        "label": label,
        "kind": "object",
        "confidence": "high",
        "reason": "VLM located a missed counterpart in saved pictures; no measured geometry.",
        "centroid": None,
        "bounds": None,
        "coordinate_frame": capture + "_world",
        "point_count": 0,
        "movement_centroid_usable": False,
        "geometry_status": "missing",
        "observation_ids": [],
        "geometry_observation_ids": [],
        "context_evidence": [r["id"].removeprefix("context:") for r in contexts],
        "frame_ids": sorted({r["frame_id"] for r in refs}),
        "quality": ["vlm_image_only_observation"],
    }
    return {"object": obj, "context": contexts[0], "contexts": contexts}


def reconcile(inventory, changes, originals, matching, alignment, findings, index):
    """Replay saved findings into one canonical, conservative reviewed analysis."""
    source_ids = {
        (c, o["id"]) for c, d in inventory.captures.items() for o in d["objects"]
    }
    original_by_vertex = {
        (c, i[c + "_id"]): i["shared_id"]
        for i in originals
        for c in CAPTURES
        if i[c + "_id"]
    }
    object_labels = {
        (c, o["id"]): label_family(o["label"])
        for c, d in inventory.captures.items()
        for o in d["objects"]
    }
    semantic_conflicts = set()
    proposals, derived, blocked, claims = [], {}, set(), defaultdict(set)
    valid = {
        sid: {
            **validate_finding(f["finding"], sid, index),
            "localized_followup": f.get("localized_followup", False),
            **(
                {"coordinate_repairs": f["coordinate_repairs"]}
                if f.get("coordinate_repairs")
                else {}
            ),
        }
        for sid, f in findings.items()
    }
    for original in originals:
        sid = original["shared_id"]
        result = valid.get(sid)
        if not result or result["errors"]:
            continue
        f = result["finding"]
        sides = {}
        for c in CAPTURES:
            side = f[c]
            oid = side["object_id"] if side["presence"] == "present" else None
            if (
                side["presence"] == "present"
                and oid is None
                and f["identity"] == "same"
            ):
                item = derived_observation(
                    c, side, index["events"][sid]["label"], inventory
                )
                oid = item["object"]["id"]
                derived[oid] = item
            sides[c] = oid
        vertices = {(c, oid) for c, oid in sides.items() if oid}
        original_vertices = {
            (c, original[c + "_id"]) for c in CAPTURES if original[c + "_id"]
        }
        if f["identity"] == "ambiguous" or f["alternatives"]:
            blocked.update(
                original_vertices
                | vertices
                | {(a["capture"], a["object_id"]) for a in f["alternatives"]}
            )
        if f["identity"] == "same" and len(vertices) == 2:
            families = {
                object_labels[(c, oid)]
                for c, oid in vertices
                if (c, oid) in object_labels
            } - {"unknown", "unknown_object", "unidentified"}
            if len(families) > 1:
                # A shelf/container ID must not become the identity of an item it supports.
                blocked.update(vertices)
                semantic_conflicts.update(vertices)
            proposals.append((sides["before"], sides["after"], sid))
            claims[("before", sides["before"])].add(sides["after"])
            claims[("after", sides["after"])].add(sides["before"])
        elif len(vertices) == 1:
            proposals.append((sides["before"], sides["after"], sid))
            for capture, oid in vertices:
                claims[(capture, oid)].add(None)
        elif f["identity"] == "different":
            blocked.update(original_vertices | vertices)
    blocked.update(v for v, candidates in claims.items() if len(candidates) > 1)
    discovered = list(derived.values())
    for i, left in enumerate(discovered):
        for right in discovered[i + 1 :]:
            a, b = left["context"], right["context"]
            if a["capture"] != b["capture"] or a["frame_id"] != b["frame_id"]:
                continue
            x, y = a["source_region_box"], b["source_region_box"]
            overlap = max(0, min(x[2], y[2]) - max(x[0], y[0])) * max(
                0, min(x[3], y[3]) - max(x[1], y[1])
            )
            union = (
                (x[2] - x[0]) * (x[3] - x[1]) + (y[2] - y[0]) * (y[3] - y[1]) - overlap
            )
            if union > 0 and overlap / union >= 0.5:
                blocked.update(
                    (item["object"]["capture"], item["object"]["id"])
                    for item in (left, right)
                )
    # Competing assignments contaminate every participant, including unmatched candidates.
    while True:
        expanded = set(blocked)
        for b, a, _ in proposals:
            pair = {("before", b), ("after", a)} - {("before", None), ("after", None)}
            if pair & blocked:
                expanded.update(pair)
        if expanded == blocked:
            break
        blocked = expanded
    identities, used, event_reviews = [], set(), {}

    def add(b, a, sids, status):
        vertices = {(c, oid) for c, oid in (("before", b), ("after", a)) if oid}
        if vertices & used:
            return
        sids = sorted(
            set(sids)
            | {original_by_vertex[v] for v in vertices if v in original_by_vertex}
        )
        same = next(
            (o for o in originals if o["before_id"] == b and o["after_id"] == a), None
        )
        shared = same["shared_id"] if same else "reviewed_" + identity([b, a])[:20]
        identities.append(
            {
                "shared_id": shared,
                "before_id": b,
                "after_id": a,
                "status": status,
                "source_event_ids": sids,
            }
        )
        event_reviews[shared] = sids
        used.update(vertices)

    for b, a, sid in sorted(
        set(proposals), key=lambda x: (x[0] or "", x[1] or "", x[2])
    ):
        if {(c, oid) for c, oid in (("before", b), ("after", a)) if oid} & blocked:
            continue
        add(b, a, [sid], "matched" if b and a else "unmatched")
    for o in originals:
        vertices = {(c, o[c + "_id"]) for c in CAPTURES if o[c + "_id"]}
        if not vertices & used:
            add(
                o["before_id"],
                o["after_id"],
                [o["shared_id"]],
                "ambiguous"
                if vertices & blocked or o["status"] == "ambiguous"
                else o["status"],
            )
    for c, oid in sorted(
        source_ids | {(i["object"]["capture"], oid) for oid, i in derived.items()}
    ):
        if (c, oid) not in used:
            add(
                oid if c == "before" else None,
                oid if c == "after" else None,
                [],
                "ambiguous" if (c, oid) in blocked else "unmatched",
            )
    augmented = AugmentedInventory(inventory, list(derived.values()))
    evaluated = saved_changes(
        augmented,
        identities,
        {"matches": []},
        alignment,
        {"movement_extent_fraction": changes["movement_extent_fraction"]},
    )
    original_events = {e["shared_id"]: e for e in changes["events"]}
    resolved_presence = set()
    for row in valid.values():
        f = row["finding"]
        if (
            not row["errors"]
            and f["identity"] == "same"
            and not f["alternatives"]
            and all(f[c]["presence"] == "present" for c in CAPTURES)
        ):
            resolved_presence.update(
                (c, f[c]["object_id"])
                for c in CAPTURES
                if f[c]["object_id"] is not None
                and (c, f[c]["object_id"]) not in blocked
            )
    for event in evaluated["events"]:
        sids = event_reviews[event["shared_id"]]
        results = [valid[s] for s in sids if s in valid]
        original = [original_events[s] for s in sids]
        event["source_event_ids"] = sids
        event["original_proposals"] = original
        event["review_findings"] = results
        pending = not results
        reasons = []
        if {(c, event[c + "_id"]) for c in CAPTURES} & semantic_conflicts:
            reasons.append("incompatible_semantic_identity_assignment")
        applicable = []
        for r in results:
            f = r["finding"]
            if r["errors"]:
                reasons += ["invalid_vlm_evidence:" + x for x in r["errors"]]
                continue
            # Only use a review that actually assessed this final pair/presence.
            if all(
                not event[c + "_id"]
                or f[c]["presence"] == "present"
                and (
                    f[c]["object_id"] == event[c + "_id"]
                    or f[c]["object_id"] is None
                    and event[c + "_id"].startswith("review_")
                )
                for c in CAPTURES
            ):
                applicable.append(f)
        if not pending:
            if not applicable:
                reasons.append("no_valid_review_for_reconciled_identity")
            if any(event[c + "_id"] in derived for c in CAPTURES) and not any(
                r.get("localized_followup") for r in results if not r["errors"]
            ):
                reasons.append("new_counterpart_requires_closeup_confirmation")
            for f in applicable:
                if f["alternatives"] or f["identity"] == "ambiguous":
                    reasons.append("competing_visual_identity")
                if event["before_id"] and event["after_id"]:
                    if f["identity"] != "same" or f["movement"] == "unclear":
                        reasons.append("visual_movement_unresolved")
                    elif (
                        event["type"] in ("moved", "unchanged")
                        and f["movement"] != event["type"]
                    ):
                        reasons.append("visual_and_location_evidence_conflict")
                else:
                    absent = "after" if event["before_id"] else "before"
                    if not any(
                        row.get("localized_followup") and row["finding"] == f
                        for row in results
                    ):
                        reasons.append("presence_change_requires_closeup_confirmation")
                    if any(
                        o["before_id"]
                        and o["after_id"]
                        and (absent, o[absent + "_id"]) not in resolved_presence
                        for o in original
                    ):
                        reasons.append("unresolved_inventoried_counterpart")
                    if (
                        f[absent]["presence"] != "absent"
                        or not f[absent]["region_visible"]
                    ):
                        reasons.append("opposing_region_absence_not_supported")
        if pending:
            # Partial reviews remain explicit; never silently publish raw labels as reviewed.
            reasons.append("not_yet_reviewed")
        if reasons:
            event.update(type="uncertain", trust="uncertain")
            event["uncertainties"] = sorted(set(event["uncertainties"] + reasons))
        status = (
            "not_reviewed"
            if pending
            else "uncertain"
            if event["type"] == "uncertain"
            else "confirmed"
            if len(original) == 1
            and all(
                event[k] == original[0][k] for k in ("before_id", "after_id", "type")
            )
            else "corrected"
        )
        event["review_status"] = status
        details = " ".join(f["explanation"] for f in applicable)
        event["explanation"] += " VLM review: " + (
            details or ", ".join(reasons) or "unresolved"
        )
        if reasons:
            event["explanation"] += " Uncertain: " + ", ".join(reasons) + "."
        if event["type"] in ("added", "removed"):
            event["basis"] = "visually_supported_presence"
            event["trust"] = "estimated"
            event["explanation"] = (
                "Estimated "
                + event["type"]
                + ": visual review cites the object and a visible opposing region. "
                + details
            )
    evaluated["counts"] = dict(
        sorted(Counter(e["type"] for e in evaluated["events"]).items())
    )
    evaluated["assumptions"][0] = (
        "Reviewed presence changes require visible opposing-region evidence; VLM visibility remains estimated."
    )
    return {
        "identities": sorted(identities, key=lambda x: x["shared_id"]),
        "changes": evaluated,
        "derived_observations": sorted(
            derived.values(), key=lambda x: x["object"]["id"]
        ),
        "coverage": {
            "total": len(originals),
            "reviewed": len(valid),
            "pending": len(originals) - len(valid),
            "uncertain": sum(e["type"] == "uncertain" for e in evaluated["events"]),
        },
    }
