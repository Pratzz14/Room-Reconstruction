"""Source-bound review requests; classifications and distances are deliberately hidden."""

import json
from pathlib import Path

from PIL import Image, ImageDraw

from .core import digest, identity, safe_path

SYSTEM = """You review room object changes using supplied BEFORE and AFTER pictures.
Inventory labels and highlighted regions are hypotheses. Check that each named object
is actually depicted, rather than a supporting surface, part, or unrelated object.
Use unclear when the target cannot be identified; do not force a match from its label.
Update means location movement only, never rotation, contents, or appearance change.
Assess images independently; no proposed change labels or distances are supplied.
Review EVERY requested shared_id exactly once. Search the entire supplied room context
for moved objects, including inventory counterparts outside the selected object views.
Use appearance, shape and fixed structural landmarks, not proximity, to match identity.
Similar objects can be distinct; preserve alternative candidates explicitly.
presence: present requires a cited visible object region; absent requires a cited,
clearly visible corresponding region AND no counterpart in supplied room context.
Occlusion, unseen areas, crops, missing detections or projected boxes never prove absence.
region_visible means the corresponding location is clearly inspectable, not merely projected.
For absent, explain which fixed landmarks establish the corresponding region.
Only cite frame IDs supplied in this request. Boxes are [x0,y0,x1,y1] in 0..1000
normalized ORIGINAL FULL FRAME coordinates, including when citing a crop or montage tile.
object_id must be an inventory ID from that capture, or null for a newly found image-only
counterpart. New counterparts need a precise full-frame box. Do not invent 3D coordinates,
distance, depth, visibility measurements or confidence probabilities.
movement describes visual placement relative to fixed landmarks; use unclear when camera
framing or evidence prevents comparison. Briefly explain observations and limitations.
Image highlights identify proposed regions only; they do not prove object identity or absence.
"""


def schema():
    ref = {
        "type": "object",
        "properties": {
            "frame_id": {"type": "string"},
            "box": {
                "anyOf": [
                    {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    {"type": "null"},
                ]
            },
        },
        "required": ["frame_id", "box"],
        "additionalProperties": False,
    }
    side = {
        "type": "object",
        "properties": {
            "presence": {
                "type": "string",
                "enum": ["present", "absent", "occluded", "outside_frame", "unclear"],
            },
            "object_id": {"type": ["string", "null"]},
            "region_visible": {"type": "boolean"},
            "evidence": {"type": "array", "items": ref},
        },
        "required": ["presence", "object_id", "region_visible", "evidence"],
        "additionalProperties": False,
    }
    item = {
        "type": "object",
        "properties": {
            "shared_id": {"type": "string"},
            "before": side,
            "after": side,
            "identity": {"type": "string", "enum": ["same", "different", "ambiguous"]},
            "movement": {"type": "string", "enum": ["moved", "unchanged", "unclear"]},
            "alternatives": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "capture": {"type": "string", "enum": ["before", "after"]},
                        "object_id": {"type": "string"},
                    },
                    "required": ["capture", "object_id"],
                    "additionalProperties": False,
                },
            },
            "explanation": {"type": "string"},
        },
        "required": [
            "shared_id",
            "before",
            "after",
            "identity",
            "movement",
            "alternatives",
            "explanation",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"reviews": {"type": "array", "items": item}},
        "required": ["reviews"],
        "additionalProperties": False,
    }


def request_schema():
    """Bound future responses while retaining the original schema for old receipts."""
    result = schema()
    properties = result["properties"]["reviews"]["items"]["properties"]
    for capture in ("before", "after"):
        box = properties[capture]["properties"]["evidence"]["items"]["properties"][
            "box"
        ]
        box["description"] = (
            "[x0,y0,x1,y1] normalized ORIGINAL FULL-FRAME coordinates from 0 to 1000, never pixel coordinates or crop-local coordinates"
        )
        box["anyOf"][0]["items"].update(minimum=0, maximum=1000)
    return result


def prepare(inventory, report, report_root, output):
    """Index all saved room frames, not just frames attached to detected objects."""
    output = Path(output)
    assets = output / "assets"
    assets.mkdir(exist_ok=True)
    frames, sheets = {}, []
    for capture, data in inventory.captures.items():
        tiles = []
        for index, frame in enumerate(data["frames"]):
            path = safe_path(inventory.reconstruction, frame["image_path"])
            if digest(path) != frame["image_sha256"]:
                raise ValueError("Room context source image hash mismatch")
            frames[frame["frame_id"]] = {
                "capture": capture,
                "index": index,
                "image_path": frame["image_path"],
                "sha256": frame["image_sha256"],
                "width": frame["width"],
                "height": frame["height"],
            }
            with Image.open(path) as original:
                tile = original.convert("RGB")
                tile.thumbnail((384, 216))
            canvas = Image.new("RGB", (384, 244), "#172029")
            canvas.paste(tile, (0, 28))
            ImageDraw.Draw(canvas).text(
                (6, 8), f"{capture} frame {index}", fill="white"
            )
            tiles.append(canvas)
        for start in range(0, len(tiles), 9):
            canvas = Image.new("RGB", (1152, 732), "#172029")
            for i, tile in enumerate(tiles[start : start + 9]):
                canvas.paste(tile, ((i % 3) * 384, (i // 3) * 244))
            path = f"assets/context_{capture}_{start:04d}.jpg"
            canvas.save(output / path, quality=88)
            sheets.append(
                {
                    "path": path,
                    "capture": capture,
                    "frame_ids": [
                        f["frame_id"] for f in data["frames"][start : start + 9]
                    ],
                }
            )
    return {
        "frames": frames,
        "context_sheets": sheets,
        "catalog": {
            c: [
                {
                    "id": o["id"],
                    "label": o["label"],
                    "frame_ids": o.get("frame_ids", []),
                }
                for o in d["objects"]
            ]
            for c, d in inventory.captures.items()
        },
        "events": {
            e["shared_id"]: {
                "shared_id": e["shared_id"],
                "label": e["label"],
                "before_id": e["before_id"],
                "after_id": e["after_id"],
                "evidence": e["evidence"],
            }
            for e in report["events"]
        },
    }


def request(
    index,
    ids,
    root,
    *,
    followup=False,
    render=True,
    candidates=None,
    reconstruction=None,
):
    root = Path(root)
    paths = [root / x["path"] for x in index["context_sheets"]]
    legends = [
        {
            "image": x["path"],
            "kind": "room_context",
            "capture": x["capture"],
            "tiles_row_major": x["frame_ids"],
        }
        for x in index["context_sheets"]
    ]
    for sid in ids:
        for role, bundle in index["events"][sid]["evidence"].items():
            for view in bundle["views"] if followup else bundle["views"][:2]:
                for kind, relative in view["images"].items():
                    if not relative:
                        continue
                    source = safe_path(root / "evidence", relative)
                    name = "assets/" + identity([digest(source), kind]) + ".jpg"
                    path = root / name
                    if not path.exists():
                        if not render:
                            raise ValueError("Missing prepared review image: " + name)
                        with Image.open(source) as image:
                            rgb = image.convert("RGB")
                            rgb.thumbnail((1280, 1280))
                            rgb.save(path, quality=88)
                    if path not in paths:
                        paths.append(path)
                        legends.append(
                            {
                                "image": name,
                                "kind": kind,
                                "role": role,
                                "frame_id": view["frame_id"],
                                "region_box_original_pixels": view["region_box"],
                                "crop_box_original_pixels": view["crop_box"],
                                "width": view["width"],
                                "height": view["height"],
                                "limitations": view["limitations"],
                                "uses": [],
                            }
                        )
                    legends[paths.index(path)]["uses"].append(
                        {
                            "shared_id": sid,
                            "role": role,
                            "region_box_original_pixels": view["region_box"],
                            "source_id": view["source_id"],
                        }
                    )
    if candidates:
        for sid in ids:
            for capture, refs in candidates.get(sid, {}).items():
                for ref in refs:
                    frame = index["frames"][ref["frame_id"]]
                    if frame["capture"] != capture:
                        raise ValueError(
                            "Candidate evidence belongs to the wrong capture"
                        )
                    source = safe_path(reconstruction, frame["image_path"])
                    if digest(source) != frame["sha256"]:
                        raise ValueError("Candidate source image hash mismatch")
                    box = [
                        ref["box"][i]
                        * (frame["width"] if i % 2 == 0 else frame["height"])
                        / 1000
                        for i in range(4)
                    ]
                    padding = max(box[2] - box[0], box[3] - box[1]) * 0.2
                    crop = [
                        max(0, int(box[0] - padding)),
                        max(0, int(box[1] - padding)),
                        min(frame["width"], int(box[2] + padding + 1)),
                        min(frame["height"], int(box[3] + padding + 1)),
                    ]
                    for kind in ("candidate_full", "candidate_crop"):
                        name = (
                            "assets/" + identity([frame["sha256"], box, kind]) + ".jpg"
                        )
                        path = root / name
                        if not path.exists():
                            if not render:
                                raise ValueError(
                                    "Missing candidate confirmation picture"
                                )
                            with Image.open(source) as image:
                                rgb = image.convert("RGB")
                                if kind == "candidate_crop":
                                    rgb = rgb.crop(tuple(crop))
                                rgb.thumbnail((1280, 1280))
                                rgb.save(path, quality=90)
                        if path not in paths:
                            paths.append(path)
                            legends.append(
                                {
                                    "image": name,
                                    "kind": kind,
                                    "frame_id": ref["frame_id"],
                                    "capture": capture,
                                    "width": frame["width"],
                                    "height": frame["height"],
                                    "candidate_box_original_pixels": box,
                                    "crop_box_original_pixels": crop
                                    if kind == "candidate_crop"
                                    else None,
                                    "uses": [
                                        {
                                            "shared_id": sid,
                                            "role": "unverified_counterpart",
                                        }
                                    ],
                                }
                            )
    prompt = {
        "task": "Independent visual review"
        + (" with additional source views" if followup else ""),
        "requested": [
            {k: v for k, v in index["events"][sid].items() if k != "evidence"}
            for sid in ids
        ],
        "room_inventory": index["catalog"],
        "frame_index": index["frames"],
        "images_in_order": legends,
    }
    if candidates:
        prompt["candidate_checks"] = candidates
        prompt["candidate_check_instructions"] = (
            "These are unverified earlier suggestions, not confirmed objects. Inspect the new close-up and its full-frame context. A crop may contain empty space or an unrelated object. Confirm a new counterpart only if the named object is visibly identifiable in that candidate region and matches the other recording. For a presence change, verify that the source crop depicts the named object and the opposing crop shows the corresponding visible region. Search all room context for a counterpart before claiming absence. Otherwise use unclear; do not repeat the earlier suggestion. Continue using normalized original full-frame coordinates."
        )
    return json.dumps(prompt, sort_keys=True), paths


def has_extra(index, sid):
    return any(len(b["views"]) > 2 for b in index["events"][sid]["evidence"].values())


def candidate_checks(findings, ids):
    result = {}
    for sid in ids:
        row = findings.get(sid, {})
        if row.get("errors") or "finding" not in row:
            continue
        f = row["finding"]
        presence_change = any(f[c]["presence"] == "absent" for c in ("before", "after"))
        refs = {
            c: [r for r in f[c]["evidence"] if r["box"] is not None]
            for c in ("before", "after")
            if f[c]["presence"] in ("present", "absent")
            and (presence_change or f[c]["object_id"] is None)
        }
        if refs:
            result[sid] = refs
    return result


def candidate_confirmed(finding, candidates):
    """A new box must actually refer to a region shown in the confirmation crop."""
    for capture in ("before", "after"):
        side = finding.get(capture, {})
        if (
            side.get("presence") not in ("present", "absent")
            or capture not in candidates
        ):
            continue
        matched = False
        for ref in side.get("evidence", []):
            for candidate in candidates.get(capture, []):
                if (
                    ref.get("frame_id") != candidate["frame_id"]
                    or ref.get("box") is None
                ):
                    continue
                a, b = ref["box"], candidate["box"]
                overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
                    0, min(a[3], b[3]) - max(a[1], b[1])
                )
                union = (
                    (a[2] - a[0]) * (a[3] - a[1])
                    + (b[2] - b[0]) * (b[3] - b[1])
                    - overlap
                )
                matched |= union > 0 and overlap / union >= 0.5
        if not matched:
            return False
    return True
