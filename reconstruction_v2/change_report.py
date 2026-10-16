"""CPU-only, immutable-source reports with portable, checked image evidence."""

import shutil
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from urllib.parse import quote, urlparse

from .alignment import apply_similarity
from .change_stage import usable_centroid
from .comparison import source_binding
from .comparison import verify as verify_comparison
from .core import atomic_replace, digest, identity, read_json, safe_path, write_json
from .refinement_gemini import exclusive
from .region_reference import project_region
from .report_evidence import DEFAULTS, LIMITATIONS, ROLES, EvidenceSelector, crop_box

VERSION = "change-report-v1"
CODE_FILES = (
    "change_report.py",
    "report_evidence.py",
    "report_render.py",
    "report.html",
    "region_reference.py",
)
WARNINGS = [
    "Measurements use relative units; real-room accuracy is not independently validated.",
    "Reliable means usable MVP object geometry, not independently validated physical change.",
    "Presence-only added/removed events are estimates without an opposing empty-region check.",
    "Whole-frame-cloud estimates are not object centroids or physical displacement.",
    "Projected/context regions and unvalidated depth do not establish visibility or absence.",
    "Unchanged means no position change above the existing threshold; rotation is not evaluated.",
]


def configuration(config=None):
    config = {
        "evidence": dict(DEFAULTS),
        "viewer_url": "http://127.0.0.1:8094/",
        **(config or {}),
    }
    if set(config) != {"evidence", "viewer_url"}:
        raise ValueError("Unknown report configuration")
    url = urlparse(config["viewer_url"])
    if (
        url.scheme not in ("http", "https")
        or not url.netloc
        or url.username
        or url.password
        or url.query
        or url.fragment
    ):
        raise ValueError(
            "viewer_url must be an HTTP(S) viewer URL without credentials, query or fragment"
        )
    config["viewer_url"] = config["viewer_url"].rstrip("/") + "/"
    config["evidence"] = {**DEFAULTS, **config["evidence"]}
    return config


def source_record(inventory, comparison, review=None):
    result = {
        "inventory": source_binding(inventory),
        "comparison_sha256": digest(comparison / "comparison.json"),
        "changes_sha256": digest(comparison / "changes.json"),
        "identities_sha256": digest(comparison / "identities.json"),
    }
    if review:
        result["review_sha256"] = digest(Path(review) / "review.json")
    return result


def event_record(event, proposal, objects, alignment, viewer_url):
    accepted = (
        alignment["quality"]["status"] == "accepted"
        and alignment.get("transform") is not None
    )
    source = {c: usable_centroid(obj) if obj else None for c, obj in objects.items()}
    common = {
        "before": source["before"] if accepted else None,
        "after": apply_similarity(source["after"], alignment["transform"])
        if accepted and source["after"] is not None
        else None,
    }
    vector = None
    if (
        event["basis"] == "object_centroids"
        and event["object_displacement"] is not None
        and all(v is not None for v in common.values())
    ):
        vector = (common["after"] - common["before"]).tolist()
    qualifications = []
    if event["basis"] == "presence_only":
        qualifications.append(
            "Estimated presence change; opposing absence has not been checked."
        )
    if event["basis"] == "frame_cloud_mean":
        qualifications.append(
            "Estimated whole-frame-cloud location; not an object centroid or displacement."
        )
    measurements = {
        "units": "relative",
        "coordinate_frame": "before_world" if accepted else None,
        "source_centroids": {
            c: v.tolist() if v is not None else None for c, v in source.items()
        },
        "source_coordinate_frames": {
            c: obj.get("coordinate_frame") if obj else None
            for c, obj in objects.items()
        },
        "centroids": {
            c: v.tolist() if v is not None else None for c, v in common.items()
        },
        "displacement_vector": vector,
        "displacement": event["object_displacement"] if vector is not None else None,
        "movement_threshold": event["threshold"],
        "basis": event["basis"],
        "frame_cloud_estimate": {
            "before_location": event["before_location"],
            "after_location": event["after_location"],
            "distance": event["frame_cloud_distance"],
            "support": event.get("frame_cloud_support", {}),
        }
        if event["basis"] == "frame_cloud_mean"
        else None,
    }
    result = {
        "event_id": event["shared_id"],
        "shared_id": event["shared_id"],
        **{
            k: event[k]
            for k in (
                "before_id",
                "after_id",
                "label",
                "type",
                "trust",
                "explanation",
                "uncertainties",
            )
        },
        "labels": {c: obj["label"] if obj else None for c, obj in objects.items()},
        "identity": proposal,
        "measurements": measurements,
        "qualifications": qualifications,
        "confidence": {
            "event_trust": event["trust"],
            "identity_status": proposal["status"],
            "identity_reasons": proposal.get("reasons", []),
            "objects": {
                c: {
                    "confidence": obj.get("confidence"),
                    "reason": obj.get("reason"),
                    "quality": obj.get("quality", []),
                }
                if obj
                else None
                for c, obj in objects.items()
            },
            "calibrated_probability": None,
        },
        "visualization": {
            "viewer_path": "/#event=" + quote(event["shared_id"], safe=""),
            "viewer_url": viewer_url + "#event=" + quote(event["shared_id"], safe=""),
            "object_ids": {c: obj["id"] if obj else None for c, obj in objects.items()},
            "interactive_requires_local_viewer": True,
        },
    }
    for key in (
        "review_status",
        "original_proposals",
        "review_findings",
        "source_event_ids",
    ):
        if key in event:
            result[key] = event[key]
    return result


def scene_summary(inventory, alignment, changes, events):
    counts = Counter(e["type"] for e in events)
    by_trust = {}
    for event in events:
        row = by_trust.setdefault(event["trust"], {})
        row[event["type"]] = row.get(event["type"], 0) + 1
    selected = {c: set() for c in ("before", "after")}
    roles = {}
    for role in ROLES:
        bundles = [e["evidence"][role] for e in events]
        roles[role] = {
            "events_with_views": sum(bool(b["views"]) for b in bundles),
            "events_without_views": sum(not b["views"] for b in bundles),
            "events_with_unknown_visibility": sum(
                "visibility_unknown" in b["limitations"] for b in bundles
            ),
            "limitation_counts": dict(
                sorted(Counter(x for b in bundles for x in b["limitations"]).items())
            ),
        }
        for bundle in bundles:
            selected[bundle["capture"]].update(v["frame_id"] for v in bundle["views"])
    return {
        "alignment": alignment,
        "units": "relative",
        "real_scene_accuracy": "not_independently_validated",
        "event_count": len(events),
        "change_counts": dict(sorted(counts.items())),
        "counts_by_trust": by_trust,
        "movement_threshold": changes["threshold"],
        "coverage": {
            "scope": "saved_input_frames_and_event_evidence_only",
            "available_frames": {
                c: len(d["frames"]) for c, d in inventory.captures.items()
            },
            "selected_frames": {c: len(s) for c, s in selected.items()},
            "roles": roles,
        },
    }


def build_report(inventory, comparison, output, config=None, review=None):
    comparison, output = Path(comparison).resolve(), Path(output).resolve()
    roots = [inventory.reconstruction, inventory.root, comparison]
    if review:
        roots.append(Path(review).resolve())
    if hasattr(inventory, "refinement_root"):
        roots.append(inventory.refinement_root)
    if any(
        output == r or output.is_relative_to(r) or r.is_relative_to(output)
        for r in roots
    ):
        raise ValueError("Report output must be separate from all source directories")
    config = configuration(config)
    compared = verify_comparison(inventory, comparison)
    if (
        compared["version"] != "comparison-v2"
        or not (comparison / "changes.json").exists()
    ):
        raise ValueError(
            "Report requires evaluated changes from compare.py run (comparison-v2)"
        )
    changes = read_json(comparison / "changes.json")
    proposals = {
        p["shared_id"]: p
        for p in read_json(comparison / "identities.json")["identities"]
    }
    sources = source_record(inventory, comparison, review)
    source_inventory = inventory
    reviewed = None
    if review:
        from .change_review import verify as verify_review
        from .review_analysis import AugmentedInventory

        reviewed = verify_review(review, inventory, comparison)
        changes = reviewed["changes"]
        proposals = {i["shared_id"]: i for i in reviewed["identities"]}
        inventory = AugmentedInventory(inventory, reviewed["derived_observations"])
    code = {name: digest(Path(__file__).with_name(name)) for name in CODE_FILES}
    selector = EvidenceSelector(inventory, compared["alignment"], config["evidence"])
    with exclusive(output.parent / ("." + output.name + ".report.lock")):
        if output.exists():
            raise FileExistsError(
                "Report already exists; choose a new output destination"
            )
        stage = Path(
            tempfile.mkdtemp(prefix="." + output.name + "-", dir=output.parent)
        )
        try:
            events = []
            for index, event in enumerate(
                sorted(changes["events"], key=lambda e: e["shared_id"])
            ):
                print(
                    f"Selecting evidence {index + 1}/{len(changes['events'])}: {event['label']}",
                    flush=True,
                )
                objects = {
                    c: inventory.object(c, event[c + "_id"])[0]
                    if event[c + "_id"]
                    else None
                    for c in ("before", "after")
                }
                record = event_record(
                    event,
                    proposals[event["shared_id"]],
                    objects,
                    compared["alignment"],
                    config["viewer_url"],
                )
                record["evidence"] = selector.event(objects["before"], objects["after"])
                for bundle in record["evidence"].values():
                    bundle["views"] = [
                        selector.export(v, stage) for v in bundle["views"]
                    ]
                events.append(record)
            report = {
                "schema_version": "change-report-v2" if reviewed else VERSION,
                "sources": sources,
                "config": config,
                "synthetic": compared.get("synthetic", False),
                "provider_calls": 0,
                "warnings": WARNINGS,
                "scene": scene_summary(
                    inventory, compared["alignment"], changes, events
                ),
                "events": events,
            }
            if reviewed:
                report.update(
                    review_source=str(Path(review).resolve()),
                    review_summary={
                        "coverage": reviewed["coverage"],
                        "usage": reviewed["usage"],
                        "status": reviewed["status"],
                    },
                )
                report["warnings"] = [
                    w for w in WARNINGS if "Presence-only" not in w
                ] + [
                    "VLM-reviewed visibility/presence remain estimates; original proposals are retained."
                ]
                if reviewed["status"] != "completed":
                    report["warnings"].append(
                        "VLM review is incomplete. "
                        + (reviewed.get("stopped_reason") or "Some review work is pending.")
                    )
            write_json(stage / "report.json", report)
            from .report_render import render_html, render_markdown

            (stage / "summary.md").write_text(render_markdown(report), encoding="utf-8")
            (stage / "report.html").write_text(render_html(report), encoding="utf-8")
            manifest = {
                "schema_version": report["schema_version"],
                "sources": sources,
                "config_identity": identity(config),
                "code": code,
                "artifacts": {
                    p.relative_to(stage).as_posix(): digest(p)
                    for p in sorted(stage.rglob("*"))
                    if p.is_file()
                },
            }
            write_json(stage / "manifest.json", manifest)
            verify_report(
                stage,
                source_inventory,
                comparison,
                comparison_verified=True,
                review=review,
            )
            if sources != source_record(source_inventory, comparison, review):
                raise ValueError("Report sources changed during generation")
            if any(
                digest(Path(__file__).with_name(name)) != sha
                for name, sha in code.items()
            ):
                raise ValueError(
                    "Report code changed during generation; regenerate from stable code"
                )
            atomic_replace(stage, output)
        finally:
            if stage.exists():
                # Only remove this call's unique, checked staging directory.
                if stage.parent != output.parent or not stage.name.startswith(
                    "." + output.name + "-"
                ):
                    raise ValueError("Unexpected report staging path")
                shutil.rmtree(stage)
    return report


def verify_report(
    output,
    inventory=None,
    comparison=None,
    *,
    comparison_verified=False,
    review=None,
    check_presentation=True,
):
    output = Path(output).resolve()
    manifest = read_json(output / "manifest.json")
    if manifest.get("schema_version") not in (VERSION, "change-report-v2"):
        raise ValueError("Unsupported report version")
    artifacts = manifest["artifacts"]
    if not {"report.json", "summary.md", "report.html"}.issubset(artifacts):
        raise ValueError("Report is missing required artifacts")
    for path, sha in artifacts.items():
        if path not in (
            "report.json",
            "summary.md",
            "report.html",
        ) and not path.startswith("assets/"):
            raise ValueError("Unexpected report artifact")
        if digest(safe_path(output, path)) != sha:
            raise ValueError("Report artifact hash mismatch: " + path)
    report = read_json(output / "report.json")
    if (
        report.get("schema_version") != manifest["schema_version"]
        or report["sources"] != manifest["sources"]
        or identity(report["config"]) != manifest["config_identity"]
    ):
        raise ValueError("Report and manifest disagree")
    configuration(report["config"])
    event_ids = [e["event_id"] for e in report["events"]]
    if event_ids != sorted(set(event_ids)):
        raise ValueError("Report event IDs must be unique and ordered")
    if (inventory is None) != (comparison is None):
        raise ValueError(
            "Provide both source inventory and comparison for bound verification"
        )
    originals, proposals = {}, {}
    selector = None
    reviewed = None
    if report["schema_version"] == "change-report-v2" and inventory is not None:
        from .change_review import verify as verify_review
        from .review_analysis import AugmentedInventory

        review = review or report["review_source"]
        reviewed = verify_review(review, inventory, comparison)
    if inventory is not None:
        comparison = Path(comparison).resolve()
        if not comparison_verified:
            verify_comparison(inventory, comparison)
        if report["sources"] != source_record(inventory, comparison, review):
            raise ValueError("Report belongs to different or stale source artifacts")
        originals = {
            e["shared_id"]: e for e in read_json(comparison / "changes.json")["events"]
        }
        proposals = {
            e["shared_id"]: e
            for e in read_json(comparison / "identities.json")["identities"]
        }
        if reviewed:
            originals = {e["shared_id"]: e for e in reviewed["changes"]["events"]}
            proposals = {i["shared_id"]: i for i in reviewed["identities"]}
            inventory = AugmentedInventory(inventory, reviewed["derived_observations"])
            if report["review_summary"] != {
                "coverage": reviewed["coverage"],
                "usage": reviewed["usage"],
                "status": reviewed["status"],
            }:
                raise ValueError("Report review coverage/usage disagree")
        if set(originals) != set(event_ids):
            raise ValueError("Report events do not cover source changes")
        selector = EvidenceSelector(
            inventory,
            read_json(comparison / "alignment.json"),
            report["config"]["evidence"],
        )
    for event in report["events"]:
        if event["event_id"] != event["shared_id"] or set(event["evidence"]) != set(
            ROLES
        ):
            raise ValueError("Invalid event identity/evidence roles")
        if inventory is not None:
            original = originals[event["event_id"]]
            objects = {
                c: inventory.object(c, original[c + "_id"])[0]
                if original[c + "_id"]
                else None
                for c in ("before", "after")
            }
            expected = event_record(
                original,
                proposals[event["event_id"]],
                objects,
                report["scene"]["alignment"],
                report["config"]["viewer_url"],
            )
            if any(event[k] != v for k, v in expected.items()):
                raise ValueError(
                    "Event measurements/classification disagree with source changes"
                )
        for role, bundle in event["evidence"].items():
            if bundle["capture"] != ROLES[role][1] or not set(
                bundle["limitations"]
            ).issubset(LIMITATIONS):
                raise ValueError("Invalid evidence role/limitations")
            views = bundle["views"]
            if len(views) > report["config"]["evidence"]["max_views"] or len(
                {v["frame_id"] for v in views}
            ) != len(views):
                raise ValueError("Invalid evidence selection count")
            for index, view in enumerate(views):
                if (
                    view["primary"] != (index == 0)
                    or view["capture"] != bundle["capture"]
                ):
                    raise ValueError("Invalid primary/capture reference")
                if view["crop_box"] != crop_box(
                    view["region_box"],
                    (view["width"], view["height"]),
                    view["crop_padding"],
                ):
                    raise ValueError("Invalid evidence crop coordinates")
                for path in view["images"].values():
                    if path is not None and (
                        not path.startswith("assets/") or path not in artifacts
                    ):
                        raise ValueError("Evidence references an unbound asset")
                if artifacts[view["images"]["full"]] != view["image_sha256"]:
                    raise ValueError("Full frame lost original source bytes")
                if view["region_box"] is None and (
                    view["images"]["highlight"] or view["images"]["crop"]
                ):
                    raise ValueError(
                        "Unlocated evidence must not have a highlight/crop"
                    )
                if inventory is not None:
                    frame = inventory.captures[view["capture"]]["frames"][
                        view["frame_index"]
                    ]
                    for key in (
                        "frame_id",
                        "image_sha256",
                        "source_sha256",
                        "decoded_frame_index",
                        "source_pts",
                        "time_base_numerator",
                        "time_base_denominator",
                        "source_timestamp_sec",
                        "elapsed_timestamp_sec",
                        "width",
                        "height",
                    ):
                        if view.get(key) != frame.get(key):
                            raise ValueError(
                                "Evidence provenance disagrees with source frame"
                            )
                    source_capture = ROLES[role][0]
                    obj = objects[source_capture]
                    allowed = (
                        set(
                            obj.get("observation_ids", [])
                            + ["context:" + x for x in obj.get("context_evidence", [])]
                        )
                        if obj
                        else set()
                    )
                    if view["source_id"] not in allowed:
                        raise ValueError("Evidence does not support the event object")
                    record = selector.inspection.evidence_record(
                        source_capture, view["source_id"]
                    )
                    if source_capture == view["capture"]:
                        if record["frame_index"] != view["frame_index"]:
                            raise ValueError("Object evidence uses an unrelated frame")
                        expected_box, _ = selector.mask_box(record, frame)
                        if (
                            crop_box(
                                expected_box,
                                (frame["width"], frame["height"]),
                                view["crop_padding"],
                            )
                            is None
                        ):
                            expected_box = None
                        if view["region_box"] != expected_box:
                            raise ValueError(
                                "Object region disagrees with source mask/context"
                            )
                        if (
                            view.get("source_mask_sha256") is not None
                            and digest(safe_path(inventory.root, record["mask_path"]))
                            != view["source_mask_sha256"]
                        ):
                            raise ValueError("Source mask binding disagrees")
                    else:
                        if selector.alignment["quality"]["status"] != "accepted":
                            raise ValueError(
                                "Cross-capture evidence requires accepted alignment"
                            )
                        points = selector.regions.region_points(
                            source_capture, obj, record
                        )
                        expected_box, _ = project_region(
                            points,
                            selector.regions.cameras_for(view["capture"])[
                                view["frame_index"]
                            ],
                            selector.regions.transforms[view["capture"]][
                                view["frame_index"]
                            ],
                            frame,
                        )
                        if expected_box != view["region_box"]:
                            raise ValueError(
                                "Projected region disagrees with source geometry/camera"
                            )
    if inventory is not None:
        changes = (
            reviewed["changes"]
            if reviewed
            else read_json(Path(comparison) / "changes.json")
        )
        expected_scene = scene_summary(
            inventory,
            read_json(Path(comparison) / "alignment.json"),
            changes,
            report["events"],
        )
        if report["scene"] != expected_scene:
            raise ValueError("Report scene/coverage counts disagree")
    from .report_render import render_html, render_markdown

    if check_presentation and (
        (output / "summary.md").read_text(encoding="utf-8") != render_markdown(report)
        or (output / "report.html").read_text(encoding="utf-8") != render_html(report)
    ):
        raise ValueError("Readable exports disagree with canonical JSON")
    return report


def refresh_report(
    source, output, inventory, comparison, *, viewer_url=None, review=None
):
    """Publish current presentation from checked saved evidence; no selection rerun."""
    source, output = Path(source).resolve(), Path(output).resolve()
    roots = [
        source,
        inventory.reconstruction,
        inventory.root,
        Path(comparison).resolve(),
    ]
    if hasattr(inventory, "refinement_root"):
        roots.append(inventory.refinement_root)
    if review:
        roots.append(Path(review).resolve())
    if any(
        output == r or output.is_relative_to(r) or r.is_relative_to(output)
        for r in roots
    ):
        raise ValueError("Refreshed report output must be separate from all sources")
    # Historical exports are hash checked, then their canonical records, source
    # bindings, measurements, and image regions are verified before re-rendering.
    report = verify_report(
        source, inventory, comparison, review=review, check_presentation=False
    )
    original_manifest = read_json(source / "manifest.json")
    source_manifest_sha = digest(source / "manifest.json")
    if viewer_url is not None:
        report["config"] = configuration({**report["config"], "viewer_url": viewer_url})
        for event in report["events"]:
            event["visualization"]["viewer_url"] = (
                report["config"]["viewer_url"]
                + "#event="
                + quote(event["event_id"], safe="")
            )
    from .report_render import render_html, render_markdown

    presentation_code = {
        name: digest(Path(__file__).with_name(name))
        for name in ("report_render.py", "report.html")
    }
    with exclusive(output.parent / ("." + output.name + ".report.lock")):
        if output.exists():
            raise FileExistsError(
                "Report already exists; choose a new output destination"
            )
        stage = Path(
            tempfile.mkdtemp(prefix="." + output.name + "-", dir=output.parent)
        )
        try:
            for relative in original_manifest["artifacts"]:
                destination = safe_path(stage, relative)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(safe_path(source, relative), destination)
            write_json(stage / "report.json", report)
            (stage / "summary.md").write_text(render_markdown(report), encoding="utf-8")
            (stage / "report.html").write_text(render_html(report), encoding="utf-8")
            manifest = {
                **original_manifest,
                "config_identity": identity(report["config"]),
                "presentation_source_manifest_sha256": source_manifest_sha,
                "presentation_code": presentation_code,
                "artifacts": {
                    **original_manifest["artifacts"],
                    **{
                        name: digest(stage / name)
                        for name in ("report.json", "summary.md", "report.html")
                    },
                },
            }
            write_json(stage / "manifest.json", manifest)
            verify_report(
                stage, inventory, comparison, review=review, comparison_verified=True
            )
            if digest(source / "manifest.json") != source_manifest_sha or any(
                digest(Path(__file__).with_name(name)) != sha
                for name, sha in presentation_code.items()
            ):
                raise ValueError("Source report or presentation changed during refresh")
            atomic_replace(stage, output)
        finally:
            if stage.exists():
                if stage.parent != output.parent or not stage.name.startswith(
                    "." + output.name + "-"
                ):
                    raise ValueError("Unexpected report staging path")
                shutil.rmtree(stage)
    return report


def pack_report(output, archive, inventory=None, comparison=None, review=None):
    output, archive = Path(output).resolve(), Path(archive).resolve()
    if archive.is_relative_to(output):
        raise ValueError("Archive must be outside the report directory")
    with exclusive(archive.parent / ("." + archive.name + ".lock")):
        if archive.exists():
            raise FileExistsError("Archive already exists; choose a new destination")
        verify_report(output, inventory, comparison, review=review)
        manifest = read_json(output / "manifest.json")
        temporary = archive.with_suffix(archive.suffix + ".tmp")
        try:
            with zipfile.ZipFile(
                temporary, "w", compression=zipfile.ZIP_DEFLATED
            ) as zipped:
                for relative in ["manifest.json", *sorted(manifest["artifacts"])]:
                    zipped.write(safe_path(output, relative), relative)
            with zipfile.ZipFile(temporary) as zipped:
                if zipped.testzip() is not None:
                    raise ValueError("Report ZIP failed integrity verification")
            atomic_replace(temporary, archive)
        finally:
            temporary.unlink(missing_ok=True)
    return {"archive": str(archive), "sha256": digest(archive)}
