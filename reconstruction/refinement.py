"""Validate VLM proposals and build provenance-preserving derived inventories."""

import copy
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .core import digest, identity, read_json, safe_path, write_json
from .object_geometry import CaptureGeometry, voxel_geometry
from .refinement_evidence import PROMPT, crop_image, font, prepare, schema
from .refinement_gemini import (
    Gemini,
    exclusive,
    project_cache_root,
    validate_config,
    validate_output,
)


def normalize_evidence(answer, records, inventory):
    """Recover misplaced references conservatively; a context citation is never a clean mask."""
    answer = copy.deepcopy(answer)
    for ent in answer["entities"]:
        context = []
        observations = {}
        for item in ent["observations"]:
            eid = item["evidence_id"]
            if eid in records and "observation_id" not in records[eid]:
                context.append(eid)
            else:
                observations[eid] = item
        for eid in ent["context_evidence"]:
            if eid in records and "observation_id" in records[eid]:
                source = records[eid]
                frame_context = next(
                    (
                        key
                        for key, rec in records.items()
                        if "observation_id" not in rec
                        and rec["capture"] == source["capture"]
                        and rec["frame_index"] == source["frame_index"]
                    ),
                    None,
                )
                if frame_context:
                    context.append(frame_context)
                else:
                    observations.setdefault(
                        eid, {"evidence_id": eid, "role": "uncertain"}
                    )
            else:
                context.append(eid)
        # A broad parent mask is not measured child geometry, even if Gemini calls it a "part".
        small = any(
            term in ent["label"].lower()
            for term in (
                "bottle",
                "flask",
                "plate",
                "spoon",
                "utensil",
                "container",
                "cooker",
                "pan",
                "bowl",
            )
        )
        if small:
            for eid, item in observations.items():
                rec = records.get(eid, {})
                if "observation_id" not in rec:
                    continue
                if re.search(
                    r"cabinet|shelf|table", rec.get("raw_label", ""), re.IGNORECASE
                ):
                    frame = inventory.captures[rec["capture"]]["frames"][
                        rec["frame_index"]
                    ]
                    box = rec["box"]
                    if (box[2] - box[0]) * (box[3] - box[1]) / (
                        frame["width"] * frame["height"]
                    ) > 0.08:
                        item["role"] = "uncertain"
        ent["observations"] = list(observations.values())
        ent["context_evidence"] = list(dict.fromkeys(context))
    return answer


def merge_inventory(previous, latest):
    """Reuse prior decisions for unreviewed evidence; newer mask classifications take priority."""
    result = copy.deepcopy(latest)
    entities = result["entities"]

    def canonical(label):
        label = re.sub(r"\s*\(instance candidate \d+\)$", "", label.lower()).strip()
        return {"spoon": "utensils", "cooking utensil": "utensils"}.get(label, label)

    for old in previous.get("entities", []):
        assigned = {o["evidence_id"] for ent in entities for o in ent["observations"]}
        remaining = [o for o in old["observations"] if o["evidence_id"] not in assigned]
        equal = next((e for e in entities if e["key"] == old["key"]), None)
        if equal is None:
            oldids = {o["evidence_id"] for o in old["observations"]}
            equal = next(
                (
                    e
                    for e in entities
                    if e["capture"] == old["capture"]
                    and canonical(e["label"]) == canonical(old["label"])
                    and oldids.intersection(o["evidence_id"] for o in e["observations"])
                ),
                None,
            )
        if equal is not None:
            equal["observations"].extend(copy.deepcopy(remaining))
        elif remaining or (
            not old["observations"]
            and old["context_evidence"]
            and not any(
                e["capture"] == old["capture"]
                and canonical(e["label"]) == canonical(old["label"])
                for e in entities
            )
        ):
            ent = copy.deepcopy(old)
            ent["observations"] = copy.deepcopy(remaining)
            ent["origin"] = "reused_inventory"
            entities.append(ent)
    keys = {e["key"] for e in entities}
    linked = {l[k] for l in result["links"] for k in ("before_key", "after_key")}
    for link in previous.get("links", []):
        if (
            link["before_key"] in keys
            and link["after_key"] in keys
            and not {link["before_key"], link["after_key"]}.intersection(linked)
        ):
            result["links"].append(
                {**link, "alternatives": [k for k in link["alternatives"] if k in keys]}
            )
    assigned = {o["evidence_id"] for ent in entities for o in ent["observations"]}
    result["unresolved_evidence"] = sorted(
        (
            set(previous.get("unresolved_evidence", []))
            | set(result["unresolved_evidence"])
        )
        - assigned
    )
    return result


def validate_answer(answer, records, response_schema=None):
    from scripts.compare_contact_sheets_gemini import validate_structure

    errors = validate_structure(answer, response_schema or schema())
    if errors:
        raise ValueError("Response schema invalid: " + "; ".join(errors[:8]))
    entities, used = {}, set()
    for ent in answer["entities"]:
        key, cap = ent["key"], ent["capture"]
        if not key or key in entities or not ent["label"].strip():
            raise ValueError("Duplicate/empty entity key or label")
        entities[key] = ent
        evidence = [o["evidence_id"] for o in ent["observations"]] + ent[
            "context_evidence"
        ]
        if not evidence or len(evidence) != len(set(evidence)):
            raise ValueError("Entity lacks unique evidence: " + key)
        for eid in evidence:
            if eid not in records or records[eid]["capture"] != cap:
                raise ValueError("Unknown or cross-capture evidence: " + eid)
        if any(
            "observation_id" not in records[o["evidence_id"]]
            for o in ent["observations"]
        ):
            raise ValueError("Context frame cannot masquerade as an observation")
        if any("observation_id" in records[e] for e in ent["context_evidence"]):
            raise ValueError("Observation must include an explicit mask role")
        used.update(e["evidence_id"] for e in ent["observations"])
    for eid in answer["unresolved_evidence"]:
        if eid not in records or "observation_id" not in records[eid]:
            raise ValueError("Invalid unresolved observation: " + eid)
    for link in answer["links"]:
        if (
            link["before_key"] not in entities
            or link["after_key"] not in entities
            or entities[link["before_key"]]["capture"] != "before"
            or entities[link["after_key"]]["capture"] != "after"
        ):
            raise ValueError("Identity link references invalid entities")
        if any(key not in entities for key in link["alternatives"]):
            raise ValueError("Identity alternative references an invalid entity")
    for region in answer["alignment_candidates"]:
        for cap in ("before", "after"):
            for eid in region[cap + "_evidence"]:
                if eid not in records or records[eid]["capture"] != cap:
                    raise ValueError("Alignment hint references invalid evidence")
    # Missing decisions are explicitly unresolved; missing rows never disappear.
    all_obs = {e for e, r in records.items() if "observation_id" in r}
    return sorted((all_obs - used) | set(answer["unresolved_evidence"]))


CATEGORY_TERMS = {
    "laptop": ("laptop", "notebook computer"),
    "drinking bottle": ("bottle", "flask", "thermos"),
    "oil bottle": ("oil bottle", "bottle of oil", "olive oil"),
    "pressure cooker": ("cooker",),
    "pan": ("pan", "skillet"),
    "plate": ("plate", "dish"),
    "utensils": ("utensil", "spoon", "fork", "knife", "spatula", "ladle"),
    "food container": ("container", "jar", "food box"),
}


def category_coverage(answer, records, categories):
    """Require explicit discovery decisions without asserting unseen presence."""
    entities = {e["key"]: e for e in answer["entities"]}
    reviews, gaps = {}, []
    for row in answer.get("category_reviews", []):
        key = (row["capture"], row["category"])
        if key in reviews:
            gaps.append(f"{key[0]}/{key[1]}: duplicate category review")
        reviews[key] = row
        if not row["evidence_ids"] or not row["reason"].strip():
            gaps.append(f"{key[0]}/{key[1]}: missing visual evidence/reason")
        for eid in row["evidence_ids"]:
            if eid not in records or records[eid]["capture"] != row["capture"]:
                raise ValueError("Category review references invalid evidence")
        for ek in row["entity_keys"]:
            if ek not in entities or entities[ek]["capture"] != row["capture"]:
                raise ValueError("Category review references invalid entity")
        if row["status"] == "visible":
            named = [entities[k] for k in row["entity_keys"]]
            terms = CATEGORY_TERMS.get(row["category"], (row["category"],))
            if not named or any(
                not any(t in e["label"].lower() for t in terms) for e in named
            ):
                gaps.append(
                    f"{key[0]}/{key[1]}: visible item lacks an individually named entity"
                )
            if (
                any(e["kind"] != "individual" for e in named)
                and row["category"] != "utensils"
            ):
                gaps.append(
                    f"{key[0]}/{key[1]}: distinct item incorrectly hidden in a collection"
                )
        elif row["status"] == "not_observed" and not any(
            "observation_id" not in records[e] for e in row["evidence_ids"]
        ):
            gaps.append(
                f"{key[0]}/{key[1]}: not-observed claim lacks context inspection"
            )
    for cap in ("before", "after"):
        for category in categories:
            if (cap, category) not in reviews:
                gaps.append(f"{cap}/{category}: category not reviewed")
    return sorted(set(gaps))


def empty_geometry():
    return {
        "points": np.empty((0, 3), np.float32),
        "colors": np.empty((0, 3), np.uint8),
        "representative_observation": np.empty(0, np.int64),
        "first_frame": np.empty(0, np.int64),
    }


def enforce_instances(answer, inventory, records):
    """Reject large-item collections and split known co-visible distinct instances.

    Descriptor affinity only propagates observations between these separate seeds;
    it cannot override a co-visibility conflict or establish a trusted identity.
    """
    answer = copy.deepcopy(answer)
    normalized, removed, replacements, notes = [], set(), {}, []
    large = re.compile(
        r"\b(chairs?|desks?|tables?|beds?|wardrobes?|cabinets?|shelfs?|shelves|bookcases?|doors?|bags?|backpacks?|suitcases?|luggage|laptops?|monitors?|televisions?|electronics)\b",
        re.IGNORECASE,
    )
    descriptor_cache = {}
    for ent in answer["entities"]:
        if ent["kind"] == "collection" and large.search(ent["label"]):
            if re.search(
                r"electronics|laptop|monitor|television", ent["label"], re.IGNORECASE
            ):
                replacements[ent["key"]] = []
                for i, item in enumerate(ent["observations"] or [None]):
                    child = {
                        **ent,
                        "key": ent["key"] + f"__review_{i + 1}",
                        "kind": "individual",
                        "confidence": "low",
                        "label": ent["label"] + f" item candidate {i + 1}",
                        "observations": [{**item, "role": "uncertain"}] if item else [],
                        "reason": "Broad electronics group requires individual visual recognition; retained for review. "
                        + ent["reason"],
                    }
                    normalized.append(child)
                    replacements[ent["key"]].append(child["key"])
                notes.append(
                    "Retained individual review candidates from invalid collection: "
                    + ent["label"]
                )
                continue
            removed.add(ent["key"])
            answer["unresolved_evidence"].extend(
                o["evidence_id"] for o in ent["observations"]
            )
            notes.append(
                "Rejected collection of distinct large objects: " + ent["label"]
            )
            continue
        cap = ent["capture"]
        data = inventory.captures[cap]
        observations = [
            o for o in ent["observations"] if o["role"] in ("whole", "part")
        ]
        if ent["kind"] == "collection" or len(observations) < 2:
            normalized.append(ent)
            continue
        conflicts, masks = set(), {}
        for i, item in enumerate(observations):
            rec = records[item["evidence_id"]]
            sameframe = [
                (j, prev)
                for j, prev in enumerate(observations[:i])
                if records[prev["evidence_id"]]["frame_index"] == rec["frame_index"]
            ]
            if not sameframe:
                continue
            for j, prev in sameframe:
                pair = []
                for source in (item, prev):
                    eid = source["evidence_id"]
                    if eid not in masks:
                        with Image.open(
                            safe_path(inventory.root, records[eid]["mask_path"])
                        ) as image:
                            masks[eid] = np.asarray(image) > 0
                    pair.append(masks[eid])
                fraction = np.count_nonzero(pair[0] & pair[1]) / max(
                    1, min(pair[0].sum(), pair[1].sum())
                )
                # Two disjoint part masks can legitimately be components of one object.
                if fraction < 0.05 and not (item["role"] == prev["role"] == "part"):
                    conflicts.add((j, i))
        if not conflicts:
            normalized.append(ent)
            continue
        if cap not in descriptor_cache:
            descriptor_cache[cap] = np.load(inventory.root / cap / "descriptors.npz")[
                "descriptors"
            ]
        vectors = descriptor_cache[cap][
            [
                data["index"][records[o["evidence_id"]]["observation_id"]]
                for o in observations
            ]
        ]
        vectors = vectors / np.maximum(
            np.linalg.norm(vectors, axis=1, keepdims=True), 1e-8
        )
        seeds = max(conflicts, key=lambda pair: (sum(pair), pair))
        groups = [[seeds[0]], [seeds[1]]]
        pending = [i for i in range(len(observations)) if i not in seeds]
        pending.sort(key=lambda i: -max(float(vectors[i] @ vectors[s]) for s in seeds))
        for i in pending:
            eligible = [
                (max(float(vectors[i] @ vectors[j]) for j in group), k)
                for k, group in enumerate(groups)
                if all(tuple(sorted((i, j))) not in conflicts for j in group)
            ]
            if eligible:
                groups[max(eligible)[1]].append(i)
            else:
                groups.append([i])
        extra = [o for o in ent["observations"] if o not in observations]
        answer["unresolved_evidence"].extend(o["evidence_id"] for o in extra)
        replacements[ent["key"]] = []
        for k, group in enumerate(groups):
            child = {
                **ent,
                "key": ent["key"] + f"__instance_{k + 1}",
                "label": ent["label"] + f" (instance candidate {k + 1})",
                "confidence": "low",
                "observations": [observations[i] for i in sorted(group)],
                "reason": "Local co-visibility check split a VLM group containing distinct objects. Cross-view assignment remains uncertain. "
                + ent["reason"],
            }
            normalized.append(child)
            replacements[ent["key"]].append(child["key"])
        notes.append(
            f"Split {ent['key']} into {len(groups)} tentative instances; no co-visible distinct masks were merged"
        )
    answer["entities"] = normalized
    links = []
    for link in answer["links"]:
        if link["before_key"] in removed or link["after_key"] in removed:
            continue
        bs = replacements.get(link["before_key"], [link["before_key"]])
        aa = replacements.get(link["after_key"], [link["after_key"]])
        if len(bs) * len(aa) == 1:
            links.append(
                {
                    **link,
                    "alternatives": [
                        x
                        for key in link["alternatives"]
                        if key not in removed
                        for x in replacements.get(key, [key])
                    ],
                }
            )
        else:
            # Alternatives remain visible; do not choose one pairing based on location.
            for b in bs:
                for a in aa:
                    links.append(
                        {
                            **link,
                            "before_key": b,
                            "after_key": a,
                            "confidence": "low",
                            "alternatives": [x for x in bs + aa if x not in (a, b)],
                            "reason": "VLM group required instance splitting; counterpart is ambiguous. "
                            + link["reason"],
                        }
                    )
    answer["links"] = links
    if "category_reviews" in answer:
        for row in answer["category_reviews"]:
            row["entity_keys"] = [
                new
                for key in row["entity_keys"]
                if key not in removed
                for new in replacements.get(key, [key])
            ]
    answer["unresolved_evidence"] = sorted(set(answer["unresolved_evidence"]))
    return answer, notes


def build(inventory, output, prepared, answer):
    output = Path(output)
    records = read_json(output / "evidence_index.json")
    response_schema = read_json(output / "response_schema.json")
    answer = normalize_evidence(answer, records, inventory)
    validate_answer(answer, records, response_schema)
    answer, instance_notes = enforce_instances(answer, inventory, records)
    unresolved = validate_answer(answer, records, response_schema)
    category_gaps = category_coverage(
        answer, records, prepared["config"].get("focus_categories", [])
    )
    write_json(output / "validated_answer.json", answer)
    cfg = prepared["config"]
    assigned = defaultdict(list)
    for ent in answer["entities"]:
        for ob in ent["observations"]:
            assigned[ob["evidence_id"]].append((ent["key"], ob["role"]))
    # Evidence assigned to competing entities is never trusted as clean geometry.
    disputed = {eid for eid, vals in assigned.items() if len(vals) > 1}
    repairs = []
    result = {
        "version": "refinement-v1",
        "prepared_identity": prepared["identity"],
        "binding": prepared["binding"],
        "synthetic": prepared["synthetic"],
        "status": "completed",
        "change_status": "not_evaluated",
        "captures": {},
        "links": [],
        "limitations": answer["limitations"],
        "validation_warnings": instance_notes,
        "category_review_gaps": category_gaps,
        "category_reviews": answer.get("category_reviews", []),
        "unresolved_evidence": unresolved,
        "artifacts": {},
    }
    keymap, all_objects = {}, {}
    for cap, data in inventory.captures.items():
        geom = CaptureGeometry(inventory.reconstruction, cap)
        voxel = geom.voxel * inventory.config["geometry"]["voxel_multiplier"]
        observations = data["observations"]
        entities = [e for e in answer["entities"] if e["capture"] == cap]
        builds = []
        for ent in entities:
            warnings, evidence_ids, accepted, candidates = [], [], [], []
            masks = {}
            for item in ent["observations"]:
                eid = item["evidence_id"]
                rec = records[eid]
                oid = rec["observation_id"]
                index = data["index"][oid]
                ob = observations[index]
                evidence_ids.append(oid)
                clean = (
                    item["role"] in ("whole", "part")
                    and eid not in disputed
                    and eid not in unresolved
                )
                if not clean:
                    warnings.append("mixed_or_ambiguous_mask")
                    repairs.append(
                        {
                            "capture": cap,
                            "entity_key": ent["key"],
                            "label": ent["label"],
                            "evidence_id": eid,
                            "observation_id": oid,
                            "frame_id": rec["frame_id"],
                            "timestamp_sec": rec["timestamp_sec"],
                            "box": rec["box"],
                            "mask_path": rec["mask_path"],
                            "reason": "Separate objects/uncertain mask; do not reuse the whole mask for each entity",
                        }
                    )
                    continue
                lo, hi = data["members"]["observation_offsets"][index : index + 2]
                rows = data["members"]["observation_indices"][lo:hi]
                if ob["status"] != "usable" or len(rows) < 3:
                    warnings.append("insufficient_depth_support")
                    continue
                with Image.open(safe_path(inventory.root, ob["mask_path"])) as im:
                    mask = np.asarray(im) > 0
                # Distinct co-visible instances cannot become duplicates. Nested parts can.
                for previous, oldmask in masks.get(ob["frame_index"], []):
                    fraction = np.count_nonzero(mask & oldmask) / max(
                        1, min(mask.sum(), oldmask.sum())
                    )
                    if fraction < 0.05:
                        warnings.append("distinct_same_frame_instances")
                masks.setdefault(ob["frame_index"], []).append((oid, mask))
                pts = geom.points(rows)
                candidates.append(
                    {
                        "rows": rows,
                        "oid": oid,
                        "center": np.median(pts, 0),
                        "extent": float(np.linalg.norm(np.ptp(pts, axis=0))),
                    }
                )
            if candidates:
                centers = np.array([x["center"] for x in candidates])
                spread = float(
                    np.max(np.linalg.norm(centers - np.median(centers, 0), axis=1))
                )
                extent = max(
                    float(np.median([x["extent"] for x in candidates])), voxel * 4
                )
                inconsistent = spread > cfg["geometry_max_spread_extents"] * extent
                if inconsistent:
                    warnings.append("inconsistent_multiview_geometry")
                if inconsistent or "distinct_same_frame_instances" in warnings:
                    candidates = [max(candidates, key=lambda x: len(x["rows"]))]
                accepted = candidates
            else:
                spread = None
            rows = (
                np.unique(np.concatenate([x["rows"] for x in accepted]))
                if accepted
                else np.empty(0, np.int64)
            )
            logical = identity(
                [
                    cap,
                    sorted(evidence_ids),
                    sorted(ent["context_evidence"]),
                    ent["label"],
                    ent["kind"],
                    ent["key"],
                ]
            )[:16]
            eid = cap + "_ref_" + logical
            if eid in all_objects:
                raise ValueError(
                    "Duplicate logical entity with identical evidence and label"
                )
            keymap[ent["key"]] = eid
            evidence_frames = {
                records[e["evidence_id"]]["frame_id"] for e in ent["observations"]
            }
            evidence_frames.update(
                records[e]["frame_id"] for e in ent["context_evidence"]
            )
            obj = {
                "id": eid,
                "capture": cap,
                "label": ent["label"].strip(),
                "kind": ent["kind"],
                "confidence": ent["confidence"],
                "description": ent["description"],
                "reason": ent["reason"],
                "observation_ids": sorted(set(evidence_ids)),
                "context_evidence": ent["context_evidence"],
                "raw_object_ids": sorted(
                    {
                        s
                        for e in ent["observations"]
                        for s in records[e["evidence_id"]]["raw_object_ids"]
                    }
                ),
                "frame_ids": sorted(evidence_frames),
                "supporting_frames": len(evidence_frames),
                "geometry_observation_ids": [x["oid"] for x in accepted],
                "coordinate_frame": inventory.manifest["captures"][cap]["binding"][
                    "coordinate_frame"
                ],
                "units": "relative",
                "view_centroid_max_deviation": spread,
                "quality": sorted(
                    set(warnings + ["uncalibrated_depth", "observed_surface_center"])
                ),
                "change_status": "not_evaluated",
                "identity_links": [],
                "identity_status": "uncertain"
                if ent["confidence"] == "low"
                else "provisional",
                "origin": ent.get("origin", "gemini"),
            }
            if ent["confidence"] == "low":
                obj["quality"].append("identity_requires_review")
            if not evidence_ids:
                repairs.append(
                    {
                        "capture": cap,
                        "entity_key": ent["key"],
                        "label": ent["label"],
                        "context_evidence": ent["context_evidence"],
                        "reason": "Visible object lacks its own segmentation mask",
                    }
                )
            builds.append([obj, rows])
            all_objects[eid] = obj
        # Shared source rows across different entities are contamination, not duplicated geometry.
        if builds:
            joined = np.concatenate([rows for _, rows in builds])
            values, counts = np.unique(joined, return_counts=True)
            shared = values[counts > 1]
        else:
            shared = np.empty(0, np.int64)
        geometries, member_rows = [], []
        for obj, rows in builds:
            if len(shared):
                keep = ~np.isin(rows, shared)
                if not keep.all():
                    obj["quality"].append("overlapping_entity_support_excluded")
                rows = rows[keep]
            geometry = (
                voxel_geometry(geom, rows, voxel)
                if len(rows) >= 3
                else empty_geometry()
            )
            points = geometry["points"]
            if len(points) < 3:
                geometry, rows = empty_geometry(), np.empty(0, np.int64)
                points = geometry["points"]
            obj.update(
                point_count=len(points),
                centroid=points.mean(0).tolist() if len(points) else None,
                bounds=[points.min(0).tolist(), points.max(0).tolist()]
                if len(points)
                else None,
                geometry_status="missing"
                if not len(points)
                else "inconsistent"
                if any(
                    w in obj["quality"]
                    for w in (
                        "inconsistent_multiview_geometry",
                        "distinct_same_frame_instances",
                    )
                )
                else "partial"
                if any(
                    w in obj["quality"]
                    for w in (
                        "mixed_or_ambiguous_mask",
                        "overlapping_entity_support_excluded",
                        "insufficient_depth_support",
                    )
                )
                else "supported",
            )
            if obj["geometry_status"] in ("missing", "inconsistent"):
                obj["movement_centroid_usable"] = False
            else:
                obj["movement_centroid_usable"] = (
                    obj["kind"] == "individual" and obj["confidence"] != "low"
                )
            # Every centroid still requires an accepted room alignment and uncertainty analysis.
            obj["geometry_trust"] = "observed_surface_only_unverified"
            geometries.append(geometry)
            member_rows.append(rows)
        directory = output / cap
        directory.mkdir(exist_ok=True)
        offsets = np.r_[0, np.cumsum([len(g["points"]) for g in geometries])].astype(
            np.int64
        )
        merged = {
            k: np.concatenate([g[k] for g in geometries]) if geometries else v
            for k, v in empty_geometry().items()
        }
        np.savez_compressed(directory / "geometry.npz", offsets=offsets, **merged)
        np.savez_compressed(
            directory / "membership.npz",
            indices=np.concatenate(member_rows)
            if member_rows
            else np.empty(0, np.int64),
            offsets=np.r_[0, np.cumsum([len(x) for x in member_rows])].astype(np.int64),
        )
        write_json(directory / "objects.json", {"objects": [o for o, _ in builds]})
        result["captures"][cap] = {
            "raw_objects": len(data["objects"]),
            "refined_objects": len(builds),
            "raw_unknown": sum(x["label"] == "unknown_object" for x in data["objects"]),
            "refined_unknown": sum(
                o["label"].lower() in ("unknown", "unknown_object", "unidentified")
                for o, _ in builds
            ),
            "individuals": sum(o["kind"] == "individual" for o, _ in builds),
            "collections": sum(o["kind"] == "collection" for o, _ in builds),
            "geometry_status": dict(Counter(o["geometry_status"] for o, _ in builds)),
        }
        del geom
    before_counts = Counter(l["before_key"] for l in answer["links"])
    after_counts = Counter(l["after_key"] for l in answer["links"])
    for link in answer["links"]:
        b, a = keymap[link["before_key"]], keymap[link["after_key"]]
        samekind = all_objects[b]["kind"] == all_objects[a]["kind"]
        confident = (
            link["confidence"] == "high"
            and all_objects[b]["confidence"] == "high"
            and all_objects[a]["confidence"] == "high"
        )
        unique = (
            before_counts[link["before_key"]] == after_counts[link["after_key"]] == 1
            and not link["alternatives"]
        )
        status = "proposed" if samekind and confident and unique else "ambiguous"
        if any(
            "distinct_same_frame_instances" in all_objects[x]["quality"] for x in (b, a)
        ):
            status = "ambiguous"
        record = {
            "id": "identity_" + identity([b, a])[:16],
            "before_id": b,
            "after_id": a,
            "confidence": link["confidence"],
            "status": status,
            "reason": link["reason"],
            "alternatives": [keymap[k] for k in link["alternatives"]],
            "change_status": "not_evaluated",
        }
        result["links"].append(record)
        for oid in (b, a):
            all_objects[oid]["identity_links"].append(record)
    for cap in inventory.captures:
        path = output / cap / "objects.json"
        objs = read_json(path)["objects"]
        write_json(path, {"objects": [all_objects[o["id"]] for o in objs]})
    for repair in repairs:
        repair["refined_id"] = keymap[repair["entity_key"]]
    write_json(output / "repair_tasks.json", repairs)
    write_json(
        output / "alignment_candidates.json",
        {
            "status": "proposals_require_geometric_validation",
            "regions": answer["alignment_candidates"],
        },
    )
    result["repair_tasks"] = len(repairs)
    result["unmatched_ids"] = sorted(
        oid for oid, obj in all_objects.items() if not obj["identity_links"]
    )
    result["status"] = (
        "completed_with_warnings"
        if repairs
        or category_gaps
        or unresolved
        or instance_notes
        or any(l["status"] == "ambiguous" for l in result["links"])
        else "completed"
    )
    if unresolved:
        result["validation_warnings"].append("Unresolved evidence remains")
    result["validation_warnings"].extend(category_gaps)
    result["request_receipts"] = [
        p.relative_to(output).as_posix() for p in (output / "requests").glob("*.json")
    ]
    for cap in inventory.captures:
        for file in ("objects.json", "geometry.npz", "membership.npz"):
            rel = f"{cap}/{file}"
            result["artifacts"][rel] = digest(output / rel)
    for rel in (
        "repair_tasks.json",
        "alignment_candidates.json",
        "evidence_index.json",
        "answer.json",
        "validated_answer.json",
    ):
        result["artifacts"][rel] = digest(output / rel)
    write_json(output / "refinement.json", result)
    render_report(inventory, output, result, records, all_objects)
    write_json(output / "refinement.json", result)
    return result


def render_report(inventory, output, result, records, objects):
    lines = [
        "# Refined room inventory and identity proposals",
        "",
        "Change classification: **not evaluated**. Coordinates remain capture-local; confidence is not calibrated.",
        "",
        "| Capture | Original objects | Refined objects | Original unknowns | Geometry status |",
        "|---|---:|---:|---:|---|",
    ]
    for cap, stats in result["captures"].items():
        lines.append(
            f"| {cap} | {stats['raw_objects']} | {stats['refined_objects']} | {stats['raw_unknown']} | {stats['geometry_status']} |"
        )
    lines += [
        "",
        f"{len(result['links'])} identity proposals; {result['repair_tasks']} repair tasks; {len(result['unresolved_evidence'])} unresolved observations.",
        "",
        "Refined counts are semantic inventory entries, not a proof of a complete physical inventory. Mixed masks and weak geometry remain explicit.",
        "",
        "## Identity proposals",
        "",
    ]
    if result["validation_warnings"]:
        lines += (
            ["## Local validation decisions", ""]
            + ["- " + s for s in result["validation_warnings"]]
            + [""]
        )
    receipts = [read_json(output / p) for p in result["request_receipts"]]
    lines += [
        f"Generation receipts: {len(receipts)}. Cache hits: {sum(bool(r.get('cache_hit')) for r in receipts)}.",
        "",
    ]
    if result.get("category_reviews"):
        lines += [
            "## Focus category review",
            "",
            "Not observed refers only to supplied views; it is not an absence/change classification.",
            "",
            "| Capture | Category | Visual review | Evidence |",
            "|---|---|---|---|",
        ]
        for row in result["category_reviews"]:
            lines.append(
                f"| {row['capture']} | {row['category']} | {row['status']} | {', '.join(row['evidence_ids'])} |"
            )
        lines += [""]
    for i, link in enumerate(result["links"]):
        b, a = objects[link["before_id"]], objects[link["after_id"]]
        page = Image.new("RGB", (960, 350), "#eeeee9")
        draw = ImageDraw.Draw(page)
        draw.text(
            (12, 8),
            f"{b['label']} / {a['label']} | {link['status']} | {link['confidence']}",
            font=font(18),
            fill="black",
        )
        evidence_links = []
        for j, obj in enumerate((b, a)):
            cap = obj["capture"]
            candidates = [
                inventory.captures[cap]["observations"][
                    inventory.captures[cap]["index"][oid]
                ]
                for oid in obj["observation_ids"]
            ]
            if candidates:
                ob = max(candidates, key=lambda o: o["mask_area"])
                crop = crop_image(inventory, cap, ob, 256)
                page.paste(crop, (j * 480 + (480 - crop.width) // 2, 40))
                fi = inventory.captures[cap]["frames"][ob["frame_index"]]
                caption = f"{cap} frame {fi['decoded_frame_index']}, {ob['timestamp_sec']:.2f}s"
                evidence_links.append(
                    f"[{caption}](<{(inventory.reconstruction / fi['image_path']).as_posix()}>)"
                )
                draw.text((j * 480 + 12, 306), caption, font=font(16), fill="black")
        rel = f"evidence/match_{i:03d}.jpg"
        page.save(output / rel, quality=90)
        link["evidence_sheet"] = rel
        lines += [
            f"### {b['label']}: {link['status']}",
            "",
            link["reason"],
            "",
            f"Before `{b['id']}` → After `{a['id']}`.",
            "",
            " · ".join(evidence_links),
            "",
            f"![Paired evidence]({rel})",
            "",
        ]
    lines += ["## Per-capture objects", ""]
    for obj in objects.values():
        lines += [
            f"### {obj['capture']}: {obj['label']} ({obj['kind']})",
            "",
            f"`{obj['id']}` · confidence {obj['confidence']} · geometry {obj['geometry_status']}",
            "",
            obj["reason"],
            "",
            "Original fragments: " + ", ".join(obj["raw_object_ids"]),
            "",
            "Quality: " + ", ".join(obj["quality"]),
            "",
        ]
    lines += ["## Limits", ""] + ["- " + x for x in result["limitations"]]
    lines += [
        "",
        "Identity proposals require review. Unmatched does not mean added/removed. Segmentation repairs and accepted alignment are required before trustworthy physical change measurements.",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    reconstruction,
    objects,
    output,
    config,
    credential_source=None,
    *,
    offline=False,
    followup=True,
    retry_failed=False,
    review_notes=None,
    baseline=None,
    answer_file=None,
):
    output = Path(output).resolve()
    offline = offline or bool(answer_file)
    validate_config(config)
    validate_output(reconstruction, objects, output)
    with exclusive(output / ".refinement.lock"):
        inventory, prepared = prepare(reconstruction, objects, output, config)
        records = read_json(output / "evidence_index.json")
        response_schema = read_json(output / "response_schema.json")
        notes = read_json(review_notes) if review_notes else None
        if notes:
            for check in notes.get("checks", []):
                for eid in check["evidence_ids"]:
                    if (
                        eid not in records
                        or records[eid]["capture"] != check["capture"]
                    ):
                        raise ValueError("Operator review references invalid evidence")
            notes_path = output / "operator_review_notes.json"
            if notes_path.exists() and read_json(notes_path) != notes:
                raise ValueError("Operator notes changed; use a new analysis output")
            write_json(notes_path, notes)
        answer_path = output / "answer.json"
        baseline_answer = None
        if baseline:
            from .refined_inventory import RefinedInventory

            checked = RefinedInventory(inventory, baseline)
            baseline_answer = read_json(Path(baseline) / "validated_answer.json")
            write_json(
                output / "baseline_binding.json",
                {
                    "refinement": str(Path(baseline).resolve()),
                    "manifest_sha256": digest(Path(baseline) / "refinement.json"),
                    "binding": checked.report["binding"],
                },
            )
        if answer_file:
            imported = read_json(answer_file)
            imported = imported.get("answer", imported)
            imported = normalize_evidence(imported, records, inventory)
            if answer_path.exists():
                prior = read_json(answer_path)
                imported = merge_inventory(prior, imported)
                write_json(output / "previous_answer.json", prior)
            if baseline_answer:
                imported = merge_inventory(baseline_answer, imported)
            validate_answer(imported, records, response_schema)
            write_json(answer_path, imported)
            write_json(
                output / "imported_answer_receipt.json",
                {
                    "source": str(Path(answer_file).resolve()),
                    "sha256": digest(Path(answer_file)),
                    "normalized": True,
                    "provider_calls": 0,
                },
            )
            if (output / "budget.json").exists() and read_json(output / "budget.json")[
                "generation_attempts"
            ] >= config["max_generation_attempts"]:
                write_json(
                    output / "followup_applied.json",
                    {"applied_from_saved_response": True, "provider_calls": 0},
                )
        client = (
            Gemini(
                output,
                config,
                credential_source,
                cache_root=project_cache_root(),
            )
            if not offline
            else None
        )
        if not offline and not answer_path.exists():
            answer = client.generate(
                PROMPT,
                (output / "user_prompt.txt").read_text(encoding="utf-8"),
                response_schema,
                [output / p for p in prepared["images"]],
                retry_failed=retry_failed,
            )
            answer = normalize_evidence(answer, records, inventory)
            validate_answer(answer, records, response_schema)
            write_json(answer_path, answer)
        elif not answer_path.exists():
            raise ValueError(
                "Offline rebuild needs a saved answer.json; no provider request was made"
            )
        answer = read_json(answer_path)
        if baseline_answer:
            answer = merge_inventory(baseline_answer, answer)
            write_json(answer_path, answer)
        preliminary = build(inventory, output, prepared, answer)
        if followup and not offline and not (output / "followup_applied.json").exists():
            problems = set(preliminary["unresolved_evidence"])
            discovery_review = bool(preliminary.get("category_review_gaps")) or any(
                r["status"] == "uncertain"
                for r in preliminary.get("category_reviews", [])
            )
            discovery_review |= bool(notes)
            for task in read_json(output / "repair_tasks.json"):
                if "evidence_id" in task:
                    problems.add(task["evidence_id"])
            ambiguous = {
                x
                for l in preliminary["links"]
                if l["status"] == "ambiguous"
                for x in (l["before_id"], l["after_id"])
            }
            for cap in inventory.captures:
                for obj in read_json(output / cap / "objects.json")["objects"]:
                    if obj["id"] in ambiguous or obj["confidence"] != "high":
                        problems.update(
                            eid
                            for eid, rec in records.items()
                            if rec.get("observation_id") in obj["observation_ids"]
                        )
            if problems or discovery_review:
                follow_images = []
                follow_images += [
                    output / p
                    for p in prepared["images"]
                    if "_detail_" in p or "_label_review" in p
                ]
                # A badly incomplete initial response needs the omitted pages, not just a few crops.
                broad_review = len(preliminary["unresolved_evidence"]) > max(
                    24, len(records) // 4
                )
                if broad_review:
                    pages = {
                        f"evidence/{records[e]['capture']}_crops_{int(e[1:]) // 12 * 12:04d}.jpg"
                        for e in problems
                        if "observation_id" in records[e]
                    }
                    follow_images += [
                        output / p
                        for p in prepared["images"]
                        if p in pages or "_context_" in p
                    ]
                # Largest masks first, bounded to 24 unresolved observations; all other ambiguities remain explicit.
                ranked = sorted(
                    problems,
                    key=lambda e: (
                        -(
                            records[e].get("box", [0, 0, 0, 0])[2]
                            - records[e].get("box", [0, 0, 0, 0])[0]
                        )
                        * (
                            records[e].get("box", [0, 0, 0, 0])[3]
                            - records[e].get("box", [0, 0, 0, 0])[1]
                        ),
                        e,
                    ),
                )[: 12 if broad_review else 24]
                for eid in ranked:
                    rec = records[eid]
                    if "observation_id" not in rec:
                        continue
                    ob = inventory.captures[rec["capture"]]["observations"][
                        inventory.captures[rec["capture"]]["index"][
                            rec["observation_id"]
                        ]
                    ]
                    crop = crop_image(inventory, rec["capture"], ob, 640)
                    page = Image.new(
                        "RGB", (crop.width, max(40, crop.height) + 32), "#eeeeeb"
                    )
                    page.paste(crop, (0, 32))
                    ImageDraw.Draw(page).text((5, 5), eid, font=font(18), fill="black")
                    p = output / "evidence" / ("followup_" + eid + ".jpg")
                    page.save(p, quality=90)
                    follow_images.append(p)
                prompt = (
                    "The first answer was incomplete. Every supplied observation must be assigned to a physical entity or explicitly listed in unresolved_evidence. "
                    "Inspect ALL supplied pages, not just the first categories. Explicitly inspect closed laptops, drinking bottles/metal flasks, oil bottles, pressure cookers, pots, pans, plates, spoons, utensils and food containers in the detail images. "
                    "Keep these individually named, including context-only entities when no mask exists. Correct monitor/television masks of laptops or windowsills and lamp masks of cookware by actual appearance. "
                    "Separate co-visible drinking bottles and metal flasks into their own entities. Keep every valid existing context-only discovery; do not discard it to shorten the answer. "
                    "Each category review MUST cite at least one supplied evidence ID, including not_observed reviews. Copy relevant IDs from an entity's context_evidence where appropriate; do not leave evidence_ids empty. "
                    "This is a COMPLETE inventory replacement, not just a focus-category list: retain the furniture, chair, bed, refrigerator, storage, bags and luggage supported by supplied evidence. "
                    "Inventory furniture and other distinct items too. Complete every requested focus category for each capture; do not use a generic electronics or kitchen-clutter group. "
                    "These category names are reminders of scope, not claims that an item exists. Describe only visible evidence. "
                    "Do not combine physically different items just because detector labels match: a cushion/headrest is not a backpack; a suitcase is not a refrigerator. "
                    "Review omitted and uncertain cases, return a COMPLETE replacement answer with all prior valid objects, and retain exact evidence references. "
                    "Unseen details stay unresolved. No addition/removal/movement classification.\n"
                    + json.dumps(
                        {
                            "review_evidence": sorted(problems),
                            "larger_crops": ranked,
                            "missing_decisions": preliminary["unresolved_evidence"],
                            "previous_answer": answer,
                            "category_review_gaps": preliminary.get(
                                "category_review_gaps", []
                            ),
                            "focus_categories": config.get("focus_categories", []),
                            "operator_review_notes": notes,
                            "evidence_legend": {
                                e: {
                                    k: v
                                    for k, v in r.items()
                                    if k
                                    in (
                                        "capture",
                                        "raw_label",
                                        "decoded_frame_index",
                                        "label",
                                    )
                                }
                                for e, r in records.items()
                            },
                        },
                        separators=(",", ":"),
                    )
                )
                (output / "followup_prompt.txt").write_text(prompt, encoding="utf-8")
                try:
                    revised = client.generate(
                        PROMPT,
                        prompt,
                        response_schema,
                        follow_images,
                        retry_failed=retry_failed,
                    )
                    revised = normalize_evidence(revised, records, inventory)
                    revised = merge_inventory(answer, revised)
                    validate_answer(revised, records, response_schema)
                    # Do not accept semantic deletions caused by a narrow follow-up view.
                    old_obs = {
                        o["evidence_id"]
                        for e in answer["entities"]
                        for o in e["observations"]
                    }
                    new_obs = {
                        o["evidence_id"]
                        for e in revised["entities"]
                        for o in e["observations"]
                    } | set(revised["unresolved_evidence"])
                    if not old_obs.issubset(new_obs):
                        raise ValueError(
                            "Follow-up silently dropped original observation decisions"
                        )
                    old_discoveries = [
                        e
                        for e in answer["entities"]
                        if not e["observations"] and e["context_evidence"]
                    ]
                    for ent in old_discoveries:
                        if not any(
                            e["key"] == ent["key"]
                            or (
                                e["capture"] == ent["capture"]
                                and e["label"].lower() == ent["label"].lower()
                            )
                            for e in revised["entities"]
                        ):
                            revised["entities"].append(copy.deepcopy(ent))
                    validate_answer(revised, records, response_schema)
                    write_json(output / "initial_answer.json", answer)
                    write_json(answer_path, revised)
                    write_json(
                        output / "followup_applied.json", {"review_evidence": ranked}
                    )
                    preliminary = build(inventory, output, prepared, revised)
                except Exception as exc:  # noqa: BLE001 -- retain the validated initial result on provider/build failure
                    write_json(
                        output / "followup_error.json",
                        {"error": str(exc), "initial_inventory_retained": True},
                    )
                    print(
                        "Targeted follow-up stopped; initial validated inventory retained: "
                        + str(exc),
                        flush=True,
                    )
        receipts = [read_json(p) for p in (output / "requests").glob("*.json")]
        preliminary["api_usage"] = {
            "generation_attempts_cumulative": read_json(output / "budget.json")[
                "generation_attempts"
            ]
            if (output / "budget.json").exists()
            else 0,
            "generation_calls_this_run": client.generation_calls if client else 0,
            "token_preflights_this_run": client.token_count_calls if client else 0,
            "cache_hits_this_run": client.cache_hits if client else 0,
            "receipt_input_tokens": sum(
                r.get("usage", {}).get("promptTokenCount", 0) for r in receipts
            ),
            "local_accounting_only": "Other applications can consume the same project quota",
        }
        write_json(output / "refinement.json", preliminary)
        return preliminary
