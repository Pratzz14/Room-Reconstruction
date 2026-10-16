"""Resumable, source-bound Gemini review and replayable canonical results."""

from collections import Counter
from pathlib import Path

from .change_report import build_report, refresh_report, source_record, verify_report
from .comparison import verify as verify_comparison
from .core import digest, identity, read_json, safe_path, write_json
from .refinement_gemini import Gemini, RequestTooLarge, exclusive, project_cache_root
from .review_analysis import (
    AugmentedInventory,
    reconcile,
    recover_supplied_pixel_boxes,
    validate_finding,
)
from .review_config import configuration
from .review_evidence import (
    SYSTEM,
    candidate_checks,
    candidate_confirmed,
    has_extra,
    prepare,
    request,
    request_schema,
    schema,
)

VERSION = "change-review-v1"


def apply_local_validation(analysis, validation, index):
    """A source-cited local audit can reject a claim, never confirm or relocate it."""
    if validation is None:
        return analysis
    if (
        set(validation) != {"schema_version", "rejections"}
        or validation["schema_version"] != "local-visual-validation-v1"
    ):
        raise ValueError("Invalid local visual validation")
    rejected = {}
    for row in validation["rejections"]:
        if set(row) != {"shared_id", "frame_id", "image_sha256", "reason"}:
            raise ValueError("Invalid visual validation reference")
        sid, frame = row["shared_id"], index["frames"].get(row["frame_id"])
        if (
            sid not in index["events"]
            or sid in rejected
            or not frame
            or row["image_sha256"] != frame["sha256"]
            or not isinstance(row["reason"], str)
            or not row["reason"].strip()
        ):
            raise ValueError("Unsupported local visual validation reference")
        rejected[sid] = row
    for event in analysis["changes"]["events"]:
        rows = [rejected[sid] for sid in event["source_event_ids"] if sid in rejected]
        if rows:
            event.update(type="uncertain", trust="uncertain", review_status="uncertain")
            event["uncertainties"] = sorted(
                set(event["uncertainties"]) | {"local_visual_contradiction"}
            )
            event["explanation"] += " Local source-picture validation: " + " ".join(
                row["reason"] for row in rows
            )
            event["local_visual_validation"] = rows
    analysis["changes"]["counts"] = dict(
        sorted(Counter(e["type"] for e in analysis["changes"]["events"]).items())
    )
    analysis["coverage"]["uncertain"] = analysis["changes"]["counts"].get(
        "uncertain", 0
    )
    analysis["local_visual_validation"] = validation
    return analysis


def evaluated_output(output, inventory, comparison, steps, index):
    path = output / "local_visual_validation.json"
    return apply_local_validation(
        evaluate(inventory, comparison, steps, index),
        read_json(path) if path.exists() else None,
        index,
    )


def needs_followup(event):
    reasons = set(event["uncertainties"])
    return (
        event["type"] == "uncertain"
        and "alignment_not_accepted" not in reasons
        and bool(reasons - {"uncertain_capture_identity"})
    )


def _inputs(inventory, comparison, config):
    return {
        "sources": source_record(inventory, comparison),
        "model": config["model"],
        "project_scope": config["project_scope"],
        "batch_size": config["batch_size"],
        "prompt_sha256": identity([SYSTEM, schema()]),
    }


def _load_findings(steps, index):
    findings = {}
    for step in steps:
        pixel_ids = step.get("recover_supplied_pixel_boxes", [])
        if (
            not isinstance(pixel_ids, list)
            or len(pixel_ids) != len(set(pixel_ids))
            or set(pixel_ids) - set(step["ids"])
        ):
            raise ValueError("Invalid legacy coordinate recovery selection")
        answer = step["answer"]
        rows = answer.get("reviews", []) if isinstance(answer, dict) else []
        malformed = not isinstance(rows, list) or any(
            not isinstance(r, dict) or r.get("shared_id") not in step["ids"]
            for r in rows
        )
        if malformed:
            rows = []
        for sid in step["ids"]:
            candidates = [
                r for r in rows if isinstance(r, dict) and r.get("shared_id") == sid
            ]
            if len(candidates) == 1:
                finding, repairs = candidates[0], []
                if sid in pixel_ids:
                    finding, repairs = recover_supplied_pixel_boxes(finding, sid, index)
                findings[sid] = validate_finding(finding, sid, index)
                if repairs:
                    findings[sid]["coordinate_repairs"] = repairs
                findings[sid]["localized_followup"] = bool(
                    not findings[sid]["errors"]
                    and step["followup"]
                    and sid in step.get("candidate_checks", {})
                    and candidate_confirmed(finding, step["candidate_checks"][sid])
                )
            else:
                findings[sid] = {
                    "finding": {"shared_id": sid},
                    "errors": ["missing_or_duplicate_review"],
                }
    return findings


def evaluate(inventory, comparison, steps, index):
    if hasattr(inventory, "after_transform"):
        raise ValueError(
            "Review requires capture-local source geometry; refusing a second alignment transform"
        )
    findings = _load_findings(steps, index)
    return reconcile(
        inventory,
        read_json(comparison / "changes.json"),
        read_json(comparison / "identities.json")["identities"],
        read_json(comparison / "identities.json")["matching"],
        read_json(comparison / "alignment.json"),
        findings,
        index,
    )


def publish(output, inventory, comparison, config, steps, index, *, stopped=None):
    analysis = evaluated_output(output, inventory, comparison, steps, index)
    usage = {
        "generation_calls": 0,
        "token_count_calls": 0,
        "cache_hits": 0,
        "input_tokens": 0,
    }
    receipts = []
    for path in sorted((output / "requests").glob("*.json")):
        receipt = read_json(path)
        receipts.append(
            {"path": path.relative_to(output).as_posix(), "sha256": digest(path)}
        )
        usage["cache_hits"] += int(receipt.get("cache_hit", False))
        if not receipt.get("cache_hit", False):
            usage["generation_calls"] += int(
                receipt.get("generation_sent", receipt.get("status") == "completed")
            )
            usage["input_tokens"] += (
                receipt.get("input_tokens", 0)
                if receipt.get("generation_sent", receipt.get("status") == "completed")
                else 0
            )
    if (output / "budget.json").exists():
        usage["generation_calls"] = read_json(output / "budget.json")[
            "generation_attempts"
        ]
        usage["token_count_calls"] = read_json(output / "budget.json").get(
            "token_count_attempts", 0
        )
        usage["input_tokens"] = read_json(output / "budget.json").get(
            "generation_input_tokens", usage["input_tokens"]
        )
    report = {
        "schema_version": VERSION,
        "inputs": _inputs(inventory, comparison, config),
        "config": config,
        "status": "completed"
        if not analysis["coverage"]["pending"] and not stopped
        else "partial",
        "stopped_reason": stopped,
        "usage": usage,
        "receipts": receipts,
        **analysis,
    }
    write_json(output / "steps.json", steps)
    write_json(output / "review.json", report)
    lines = [
        "# Reviewed room changes",
        "",
        f"Status: {report['status']}",
        f"Coverage: {analysis['coverage']['reviewed']}/{analysis['coverage']['total']}",
        f"Counts: {analysis['changes']['counts']}",
        f"Usage: {usage}",
        "",
        "VLM findings and visibility are estimates. Distances retain their original geometry/frame-cloud basis.",
        "",
    ]
    for event in analysis["changes"]["events"]:
        lines += [
            f"## {event['label']} — {event['type']}",
            f"Review: {event['review_status']}",
            "Original proposal: "
            + ", ".join(e["type"] for e in event["original_proposals"]),
            event["explanation"],
            "",
        ]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(
        output / "manifest.json",
        {
            "schema_version": VERSION,
            "inputs": report["inputs"],
            "artifacts": {
                p.relative_to(output).as_posix(): digest(p)
                for p in sorted(output.rglob("*"))
                if p.is_file()
                and p != output / "manifest.json"
                and not p.name.endswith((".lock", ".tmp"))
            },
        },
    )
    return report


def run(
    inventory,
    comparison,
    output,
    config=None,
    *,
    prepare_only=False,
    answers=None,
    retry_failed=False,
    client=None,
    evidence_report=None,
    credential_source=None,
    batch_limit=None,
    execution_config=None,
):
    comparison, output = Path(comparison).resolve(), Path(output).resolve()
    config = configuration(config)
    provider_config = configuration(execution_config) if execution_config else config
    if any(
        provider_config[k] != config[k]
        for k in ("project_scope", "batch_size", "max_output_tokens")
    ):
        raise ValueError(
            "Continuation must preserve project scope and request settings"
        )
    size = config["batch_size"] if batch_limit is None else batch_limit
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or not 1 <= size <= config["batch_size"]
    ):
        raise ValueError(
            "Execution batch limit must be between one and the configured batch size"
        )
    sources = [inventory.root, inventory.reconstruction, comparison]
    if hasattr(inventory, "refinement_root"):
        sources.append(inventory.refinement_root)
    if evidence_report:
        sources.append(Path(evidence_report).resolve())
    if any(
        output == p or output.is_relative_to(p) or p.is_relative_to(output)
        for p in sources
    ):
        raise ValueError("Review output must be separate from every source directory")
    print("Verifying comparison inputs...", flush=True)
    verify_comparison(inventory, comparison)
    if not (comparison / "changes.json").exists():
        raise ValueError("Review requires comparison-v2 changes")
    output.mkdir(parents=True, exist_ok=True)
    with exclusive(output / ".review.lock"):
        inputs = _inputs(inventory, comparison, config)
        evidence = None
        if (output / "inputs.json").exists():
            if read_json(output / "inputs.json") != inputs:
                raise ValueError(
                    "Review sources/model/prompt changed; use a separate output directory"
                )
            if (output / "manifest.json").exists():
                print("Checking saved review and evidence...", flush=True)
                verify(
                    output,
                    inventory,
                    comparison,
                    allow_writer=True,
                    resume=True,
                    comparison_verified=True,
                )
                evidence = read_json(output / "evidence" / "report.json")
        else:
            if any(p.name != ".review.lock" for p in output.iterdir()):
                raise ValueError("Nonempty output is not a review workspace")
            write_json(output / "inputs.json", inputs)
        if not (output / "evidence" / "manifest.json").exists():
            if evidence_report:
                if (
                    read_json(Path(evidence_report) / "report.json").get(
                        "schema_version"
                    )
                    != "change-report-v1"
                ):
                    raise ValueError(
                        "Review evidence must come from an original, unreviewed report"
                    )
                evidence = refresh_report(
                    evidence_report, output / "evidence", inventory, comparison
                )
            else:
                evidence = build_report(inventory, comparison, output / "evidence")
        if evidence is None:
            evidence = verify_report(
                output / "evidence",
                inventory,
                comparison,
                comparison_verified=True,
                check_presentation=False,
            )
        if not (output / "evidence_index.json").exists():
            write_json(
                output / "evidence_index.json",
                prepare(inventory, evidence, output / "evidence", output),
            )
        index = read_json(output / "evidence_index.json")
        steps = (
            read_json(output / "steps.json") if (output / "steps.json").exists() else []
        )
        # Preparation is useful without credentials, quota claims or network access.
        if prepare_only:
            previous = (
                read_json(output / "review.json")
                if (output / "review.json").exists()
                else {}
            )
            return publish(
                output,
                inventory,
                comparison,
                config,
                steps,
                index,
                stopped=previous.get("stopped_reason"),
            )
        ids = sorted(index["events"])
        reviewed = {
            sid for step in steps if not step["followup"] for sid in step["ids"]
        }
        pending = [sid for sid in ids if sid not in reviewed]
        queue = [(pending[i : i + size], False) for i in range(0, len(pending), size)]
        offline = None
        if answers is not None:
            offline = read_json(answers)
            if not isinstance(offline, dict) or not isinstance(
                offline.get("reviews"), list
            ):
                raise ValueError("Offline answers require a reviews array")
            known = [
                r.get("shared_id") for r in offline["reviews"] if isinstance(r, dict)
            ]
            if len(known) != len(set(known)) or set(known) - set(ids):
                raise ValueError("Offline answers contain duplicate/unknown identities")
        provider = (
            None
            if offline is not None
            else client
            or Gemini(
                output,
                provider_config,
                credential_source,
                cache_root=project_cache_root(),
            )
        )
        stopped = None
        print("Review evidence ready; preparing visual-review batches...", flush=True)
        # Persist an initial snapshot before anything can fail or consume quota.
        publish(output, inventory, comparison, config, steps, index)
        try:
            while True:
                if not queue:
                    state = evaluated_output(
                        output, inventory, comparison, steps, index
                    )
                    done = {sid for s in steps if s["followup"] for sid in s["ids"]}
                    uncertain = {
                        sid
                        for e in state["changes"]["events"]
                        if needs_followup(e)
                        for sid in e["source_event_ids"]
                    }
                    followups = (
                        [
                            sid
                            for sid in sorted(uncertain - done)
                            if sid in reviewed
                            and (
                                has_extra(index, sid)
                                or sid
                                in candidate_checks(_load_findings(steps, index), [sid])
                            )
                        ]
                        if offline is None
                        else []
                    )
                    queue = [
                        (followups[i : i + size], True)
                        for i in range(0, len(followups), size)
                    ]
                    if not queue:
                        break
                batch, followup = queue.pop(0)
                checks = (
                    candidate_checks(_load_findings(steps, index), batch)
                    if followup
                    else None
                )
                prompt, images = request(
                    index,
                    batch,
                    output,
                    followup=followup,
                    candidates=checks,
                    reconstruction=inventory.reconstruction,
                )
                try:
                    answer = (
                        {
                            "reviews": [
                                r
                                for r in offline["reviews"]
                                if r.get("shared_id") in batch
                            ]
                        }
                        if offline is not None
                        else provider.generate(
                            SYSTEM,
                            prompt,
                            request_schema(),
                            images,
                            retry_failed=retry_failed,
                        )
                    )
                except RequestTooLarge:
                    if len(batch) == 1:
                        raise
                    middle = len(batch) // 2
                    queue[0:0] = [
                        (batch[:middle], followup),
                        (batch[middle:], followup),
                    ]
                    continue
                steps.append(
                    {
                        "ids": batch,
                        "followup": followup,
                        "answer": answer,
                        "origin": "offline"
                        if offline is not None
                        else "gemini"
                        if isinstance(provider, Gemini)
                        else "injected",
                        "prompt_sha256": identity(prompt),
                        "max_output_tokens": config["max_output_tokens"],
                        "response_schema": request_schema(),
                        "provider_config": provider_config,
                        "candidate_checks": checks or {},
                        "images": [
                            {
                                "path": p.relative_to(output).as_posix(),
                                "sha256": digest(p),
                            }
                            for p in images
                        ],
                    }
                )
                reviewed.update(batch)
                publish(output, inventory, comparison, config, steps, index)
                print(
                    f"Visual review saved: {len(reviewed)}/{len(ids)} identities"
                    + (" (targeted follow-up)" if followup else ""),
                    flush=True,
                )
        except (ValueError, OSError) as exc:
            stopped = provider._clean(str(exc)) if provider else str(exc)
        result = publish(
            output, inventory, comparison, config, steps, index, stopped=stopped
        )
        if source_record(inventory, comparison) != inputs["sources"]:
            raise ValueError("Review sources changed during processing")
        return result


def verify(
    output,
    inventory,
    comparison,
    *,
    allow_writer=False,
    resume=False,
    comparison_verified=False,
):
    output, comparison = Path(output).resolve(), Path(comparison).resolve()
    if not allow_writer and (output / ".review.lock").exists():
        raise ValueError("Review writer active")
    report = read_json(output / "review.json")
    config = configuration(report["config"])
    if report.get("schema_version") != VERSION or report["inputs"] != _inputs(
        inventory, comparison, config
    ):
        raise ValueError("Review belongs to stale/different sources or prompt")
    if not comparison_verified:
        verify_comparison(inventory, comparison)
    verify_report(
        output / "evidence",
        inventory,
        comparison,
        comparison_verified=True,
        check_presentation=False,
    )
    manifest = read_json(output / "manifest.json")
    if manifest["inputs"] != report["inputs"]:
        raise ValueError("Review manifest inputs disagree")
    required = {
        "review.json",
        "steps.json",
        "inputs.json",
        "evidence_index.json",
        "summary.md",
    }
    if not required.issubset(manifest["artifacts"]):
        raise ValueError("Review missing required artifacts")
    for path, sha in manifest["artifacts"].items():
        if resume and (
            path.startswith("requests/")
            or path in ("budget.json", "steps.json", "review.json", "summary.md")
        ):
            continue
        if digest(safe_path(output, path)) != sha:
            raise ValueError("Review artifact hash mismatch: " + path)
    index = read_json(output / "evidence_index.json")
    # Check the semantic index against independently verified source metadata.
    base = read_json(output / "evidence" / "report.json")
    expected_events = {
        e["shared_id"]: {
            k: e[k] for k in ("shared_id", "label", "before_id", "after_id", "evidence")
        }
        for e in base["events"]
    }
    if index["events"] != expected_events:
        raise ValueError("Review evidence index disagrees with bound report")
    frames = {
        f["frame_id"]: {
            "capture": c,
            "index": i,
            "image_path": f["image_path"],
            "sha256": f["image_sha256"],
            "width": f["width"],
            "height": f["height"],
        }
        for c, d in inventory.captures.items()
        for i, f in enumerate(d["frames"])
    }
    if index["frames"] != frames or index["catalog"] != {
        c: [
            {"id": o["id"], "label": o["label"], "frame_ids": o.get("frame_ids", [])}
            for o in d["objects"]
        ]
        for c, d in inventory.captures.items()
    }:
        raise ValueError("Review room index disagrees with source inventory")
    steps = read_json(output / "steps.json")
    seen = set()
    for step in steps:
        if (
            type(step["followup"]) is not bool
            or not step["ids"]
            or len(step["ids"]) > config["batch_size"]
        ):
            raise ValueError("Invalid review batch")
        for sid in step["ids"]:
            key = (sid, step["followup"])
            if (
                sid not in index["events"]
                or key in seen
                or step["followup"]
                and (sid, False) not in seen
            ):
                raise ValueError("Invalid duplicate/unknown review or follow-up")
            seen.add(key)
        prompt, images = request(
            index,
            step["ids"],
            output,
            followup=step["followup"],
            render=False,
            candidates=step.get("candidate_checks"),
            reconstruction=inventory.reconstruction,
        )
        if identity(prompt) != step["prompt_sha256"] or step["images"] != [
            {"path": p.relative_to(output).as_posix(), "sha256": digest(p)}
            for p in images
        ]:
            raise ValueError("Review request does not match saved evidence")
        if step["origin"] not in ("offline", "gemini", "injected"):
            raise ValueError("Unknown review response origin")
        if step["origin"] == "gemini":
            provider_config = configuration(step.get("provider_config", config))
            if any(
                provider_config[k] != config[k]
                for k in ("project_scope", "batch_size", "max_output_tokens")
            ):
                raise ValueError("Gemini continuation changed bound project/settings")
            response_schema = step.get("response_schema", schema())
            if response_schema not in (schema(), request_schema()):
                raise ValueError("Unsupported saved Gemini review response schema")
            request_id = identity(
                [
                    provider_config["model"],
                    SYSTEM,
                    prompt,
                    response_schema,
                    [(str(p.name), digest(p)) for p in images],
                    step.get("max_output_tokens", config["max_output_tokens"]),
                ]
            )
            receipt = read_json(output / "requests" / (request_id + ".json"))
            if (
                receipt.get("status") != "completed"
                or receipt.get("model") != provider_config["model"]
                or receipt.get("answer") != step["answer"]
            ):
                raise ValueError("Gemini review disagrees with its generation receipt")
    expected = evaluated_output(output, inventory, comparison, steps, index)
    if resume:
        return report
    if any(report[k] != v for k, v in expected.items()):
        raise ValueError(
            "Reviewed results disagree with replayed source locations/findings"
        )
    if report["status"] == "completed" and (
        report["coverage"]["pending"] or report["stopped_reason"]
    ):
        raise ValueError("Incomplete review marked complete")
    for c, data in inventory.captures.items():
        actual = [i[c + "_id"] for i in report["identities"] if i[c + "_id"]]
        expected_ids = {o["id"] for o in data["objects"]} | {
            x["object"]["id"]
            for x in report["derived_observations"]
            if x["object"]["capture"] == c
        }
        if len(actual) != len(set(actual)) or set(actual) != expected_ids:
            raise ValueError("Review must cover every capture object exactly once")
    return report


def apply_to_display(inventory, review):
    """ComparisonInventory has already transformed after geometry exactly once."""
    if hasattr(inventory, "review"):
        raise ValueError("Review has already been applied to this display inventory")
    adapted = AugmentedInventory(inventory, review["derived_observations"])
    adapted.review = review
    adapted.review_identities = review["identities"]
    adapted.comparison = {
        **inventory.comparison,
        "change_counts": review["changes"]["counts"],
        "review_coverage": review["coverage"],
        "review_status": review["status"],
    }
    for data in adapted.captures.values():
        for obj in data["objects"]:
            event = next(
                e
                for e in review["changes"]["events"]
                if obj["id"] in (e["before_id"], e["after_id"])
            )
            obj.update(
                change=event,
                change_status=event["type"],
                shared_identity=event["shared_id"],
            )
            link = next(
                i for i in review["identities"] if i["shared_id"] == event["shared_id"]
            )
            obj["identity_links"] = (
                [
                    {
                        **link,
                        "id": link["shared_id"],
                        "matching_status": link["status"],
                        "status": "ambiguous"
                        if link["status"] == "ambiguous"
                        else "proposed",
                    }
                ]
                if link["before_id"] and link["after_id"]
                else []
            )
    return adapted
