"""Checked refinement adapter; original masks and source evidence stay immutable."""

from pathlib import Path

import numpy as np

from .core import digest, read_json, safe_path
from .object_contracts import ObjectInventory, check_indices, load_arrays, offsets
from .object_geometry import CaptureGeometry
from .refinement_evidence import binding


class RefinedInventory(ObjectInventory):
    def __init__(self, original, refinement):
        self.__dict__.update(original.__dict__)
        self.refinement_root = Path(refinement).resolve()
        if (self.refinement_root / ".refinement.lock").exists():
            raise ValueError("Refinement writer is active; wait for publication")
        report = read_json(self.refinement_root / "refinement.json")
        if report.get("version") != "refinement-v1" or report["binding"] != binding(
            original
        ):
            raise ValueError("Refinement belongs to different source artifacts")
        for relative, expected in report["artifacts"].items():
            if digest(safe_path(self.refinement_root, relative)) != expected:
                raise ValueError("Refinement artifact hash mismatch: " + relative)
        self.report = report
        self.context_index = read_json(self.refinement_root / "evidence_index.json")
        self.captures = {}
        allids = set()
        for cap, raw in original.captures.items():
            directory = self.refinement_root / cap
            objects = read_json(directory / "objects.json")["objects"]
            geometry, members = (
                load_arrays(directory / "geometry.npz"),
                load_arrays(directory / "membership.npz"),
            )
            offsets(
                geometry["offsets"],
                len(geometry["points"]),
                len(objects),
                "refined geometry",
            )
            offsets(
                members["offsets"],
                len(members["indices"]),
                len(objects),
                "refined membership",
            )
            if not np.isfinite(geometry["points"]).all():
                raise ValueError("Non-finite refined geometry")
            capture = CaptureGeometry(original.reconstruction, cap)
            for i, obj in enumerate(objects):
                if (
                    obj["id"] in allids
                    or obj["capture"] != cap
                    or obj["coordinate_frame"]
                    != report["binding"]["captures"][cap]["coordinate_frame"]
                ):
                    raise ValueError("Invalid refined identity or coordinate frame")
                allids.add(obj["id"])
                if any(
                    oid not in raw["index"]
                    for oid in obj["observation_ids"] + obj["geometry_observation_ids"]
                ):
                    raise ValueError("Refined object references unknown observations")
                lo, hi = members["offsets"][i : i + 2]
                rows = members["indices"][lo:hi]
                check_indices(
                    rows, len(capture.observations["depth"]), "refined source"
                )
                support = []
                for oid in obj["geometry_observation_ids"]:
                    oi = raw["index"][oid]
                    a, b = raw["members"]["observation_offsets"][oi : oi + 2]
                    support.append(raw["members"]["observation_indices"][a:b])
                allowed = (
                    np.unique(np.concatenate(support))
                    if support
                    else np.empty(0, np.int64)
                )
                if not np.isin(rows, allowed).all():
                    raise ValueError("Refined membership exceeds approved mask support")
                a, b = geometry["offsets"][i : i + 2]
                pts, rep = (
                    geometry["points"][a:b],
                    geometry["representative_observation"][a:b],
                )
                if len(pts) != obj["point_count"] or not np.isin(rep, rows).all():
                    raise ValueError("Refined geometry lost source provenance")
                if len(pts):
                    if not np.allclose(
                        pts.mean(0), obj["centroid"], atol=1e-6
                    ) or not np.allclose(
                        [pts.min(0), pts.max(0)], obj["bounds"], atol=1e-6
                    ):
                        raise ValueError("Refined centroid/bounds mismatch")
                    chosen = np.unique(
                        np.linspace(a, b - 1, min(1000, b - a)).astype(int)
                    )
                    sample = geometry["representative_observation"][chosen]
                    if not np.allclose(
                        geometry["points"][chosen], capture.points(sample), atol=1e-6
                    ) or not np.array_equal(
                        geometry["colors"][chosen], capture.colors(sample)
                    ):
                        raise ValueError("Refined points/colors differ from source")
                elif obj["centroid"] is not None or obj["bounds"] is not None:
                    raise ValueError(
                        "Evidence-only entity cannot have a measured centroid"
                    )
            self.captures[cap] = {
                **raw,
                "objects": objects,
                "geometry": geometry,
                "refined_members": members,
            }
        for link in report["links"]:
            if link["before_id"] not in allids or link["after_id"] not in allids:
                raise ValueError("Broken shared identity reference")

    def context_record(self, capture, identifier):
        eid = identifier.removeprefix("context:")
        rec = self.context_index.get(eid, {})
        if (
            rec.get("capture") != capture
            or "observation_id" in rec
            or "image_path" not in rec
        ):
            raise ValueError("Unknown context evidence")
        return {
            **rec,
            "id": identifier,
            "label": "Context frame (no object-specific mask)",
            "status": "context",
        }

    def listing(self, capture):
        result = super().listing(capture)
        records = self.context_index
        unresolved_ids = {
            records[e]["observation_id"]
            for e in self.report["unresolved_evidence"]
            if e in records
            and records[e]["capture"] == capture
            and "observation_id" in records[e]
        }
        existing = {o["id"] for o in result["unresolved"]}
        result["unresolved"] += [
            {
                **o,
                "status": "unresolved",
                "quality": o["quality"] + ["refinement_unresolved"],
            }
            for o in self.captures[capture]["observations"]
            if o["id"] in unresolved_ids and o["id"] not in existing
        ]
        result.update(
            status=self.report["status"],
            inventory_mode="refined",
            available_modes=["refined", "original"],
            refinement_summary=self.report["captures"].get(capture),
            links=self.report["links"],
            context_evidence=[
                self.context_record(capture, "context:" + eid)
                for eid, rec in self.context_index.items()
                if rec["capture"] == capture and "observation_id" not in rec
            ],
        )
        return result


class InventoryModes:
    def __init__(self, original, refinement):
        self.original = original
        self.refined = RefinedInventory(original, refinement)
        self.mode = "refined"

    def __getattr__(self, name):
        return getattr(self.refined if self.mode == "refined" else self.original, name)

    def use_mode(self, mode):
        if mode not in ("original", "refined"):
            raise ValueError("Unknown inventory mode")
        self.mode = mode

    def object(self, capture, identifier):
        reader = self.refined if self.mode == "refined" else self.original
        return reader.object(capture, identifier)

    def listing(self, capture):
        result = (self.refined if self.mode == "refined" else self.original).listing(
            capture
        )
        return {
            **result,
            "objects": [
                o
                for o in result["objects"]
                if o["label"].strip().lower()
                not in ("unknown_object", "unknown", "unidentified")
            ],
            "inventory_mode": self.mode,
            "available_modes": ["refined", "original"],
        }
