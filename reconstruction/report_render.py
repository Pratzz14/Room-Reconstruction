"""Readable exports rendered only from the canonical report record."""

import html
import json
from pathlib import Path

from .report_evidence import ROLES


def pretty(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def comparison_views(event):
    """Pair the existing capture images without inventing an object observation."""
    pairs = {}
    for capture, fallback in (
        ("before", "new_location_before"),
        ("after", "old_location_after"),
    ):
        direct = "object_" + capture
        role = direct if event["evidence"][direct]["views"] else fallback
        bundle = event["evidence"][role]
        pairs[capture] = {
            "role": role,
            "reference_only": role != direct,
            "views": bundle["views"],
            "limitations": bundle["limitations"],
        }
    return pairs


def render_markdown(report):
    scene = report["scene"]
    rows = [
        "# Room change report",
        "",
        f"Alignment: **{scene['alignment']['quality']['status']}** · relative units",
        "",
        f"Change counts: {pretty(scene['change_counts'])}",
        "",
        f"Counts by trust: {pretty(scene['counts_by_trust'])}",
        "",
        "## Limitations",
        "",
        *["> " + w for w in report["warnings"]],
        "",
        "## Saved-input coverage",
        "",
        "```json",
        json.dumps(scene["coverage"], indent=2),
        "```",
        "",
        "## Alignment diagnostics",
        "",
        "```json",
        json.dumps(scene["alignment"]["quality"], indent=2),
        "```",
        "",
    ]
    if report.get("review_summary"):
        rows += [
            "VLM review coverage and usage: " + pretty(report["review_summary"]),
            "",
        ]

    def text(value):
        return (
            str(value)
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("|", "\\|")
            .replace("\n", " ")
        )

    for event in report["events"]:
        rows += [
            f"## {text(event['label'])} · {event['event_id']}",
            "",
            f"**{event['type']} · {event['trust']}** — {text(event['explanation'])}",
            "",
            *["> " + x for x in event["qualifications"]],
            "",
            f"Objects: Before `{event['before_id']}` · After `{event['after_id']}`",
            "",
            f"[Inspect in 3D]({event['visualization']['viewer_url']}) — requires the local viewer.",
            "",
            "Measurements:",
            "",
            "```json",
            json.dumps(event["measurements"], indent=2),
            "```",
            "",
            "Confidence and identity:",
            "",
            "```json",
            json.dumps(
                {"confidence": event["confidence"], "identity": event["identity"]},
                indent=2,
            ),
            "```",
            "",
        ]
        if event.get("review_status"):
            rows += [
                "Review: " + event["review_status"],
                "Original proposal: "
                + ", ".join(e["type"] for e in event["original_proposals"]),
                "",
            ]
        rows += ["### Images used for comparison", ""]
        for capture, pair in comparison_views(event).items():
            rows += [
                f"**{capture.title()}** · {'Location reference; no direct object view' if pair['reference_only'] else 'Object observation/context'}",
                "",
            ]
            if pair["views"]:
                view = pair["views"][0]
                rows += [
                    f"![{capture.title()} comparison image]({view['images']['highlight'] or view['images']['full']})",
                    "",
                    f"`{view['frame_id']}` · {view['elapsed_timestamp_sec']:.3f}s · source role: `{pair['role']}`",
                    "",
                ]
            else:
                rows += ["No corresponding saved comparison image is available.", ""]
        for role, bundle in event["evidence"].items():
            rows += [
                f"### {role.replace('_', ' ').title()}",
                "",
                "Limitations: " + ", ".join(bundle["limitations"]),
                "",
            ]
            for view in bundle["views"]:
                rows += [
                    f"{'Primary' if view['primary'] else 'Alternative'} · `{view['frame_id']}` · {view['elapsed_timestamp_sec']:.3f}s · {view['method']}",
                    "",
                ]
                rows += [
                    f"[{kind}]({path})" for kind, path in view["images"].items() if path
                ]
                rows += [""]
    return "\n".join(rows) + "\n"


def render_html(report):
    esc = lambda value: html.escape(str(value), quote=True)
    scene = report["scene"]
    counts = (
        " · ".join(f"{n} {kind}" for kind, n in scene["change_counts"].items())
        or "No events"
    )
    events = []
    for event in report["events"]:
        roles = []
        paired = []

        def gallery(
            role,
            bundle,
            title,
            *,
            comparison=False,
            reference_only=False,
            label=event["label"],
        ):
            views = []
            options = []
            for i, view in enumerate(bundle["views"]):
                options.append(
                    f'<option value="{i}">{"Primary" if i == 0 else "Alternative"} · {view["elapsed_timestamp_sec"]:.3f}s · frame {view["decoded_frame_index"]}</option>'
                )
                images = view["images"]
                default = images["highlight"] or images["full"]
                views.append(f'''<figure class="view" {"hidden" if i else ""} data-images="{esc(pretty(images))}">
                  <a class="enlarge" href="{esc(default)}" target="_blank" rel="noopener"><img src="{esc(default)}" alt="{esc(label)} — {esc(title)}" loading="lazy"></a>
                  <figcaption><code>{esc(view["frame_id"])}</code><br>{view["elapsed_timestamp_sec"]:.3f}s · {esc(view["method"])}<br>{esc(", ".join(view["limitations"]))}</figcaption>
                  <details><summary>Frame provenance and ranking</summary><pre>{esc(json.dumps({k: v for k, v in view.items() if k != "images"}, indent=2))}</pre></details>
                </figure>''')
            controls = (
                f'''<div class="controls"><select class="frame-choice" aria-label="{esc(title)} frame">{"".join(options)}</select>
              <select class="image-choice" aria-label="{esc(title)} image"><option value="highlight">Highlight</option><option value="full">Full frame</option><option value="crop">Close-up</option></select></div>'''
                if views
                else ""
            )
            note = (
                (
                    "Location reference — no direct object observation in this capture. Inspect this photo to compare the location; object presence or absence remains unverified."
                    if reference_only
                    else "Saved object observation/context. Compare the original full frame, highlight, or close-up."
                )
                if comparison
                else ", ".join(bundle["limitations"])
            )
            missing = (
                "No corresponding saved comparison image is available."
                if comparison
                else "No saved image for this evidence role."
            )
            return f'''<section class="role{" comparison-side" if comparison else ""}" data-evidence-role="{esc(role)}"><h3>{esc(title)}</h3><p class="{"reference-note" if reference_only else "muted" if comparison else "limitations"}">{esc(note)}</p>
              {controls}{"".join(views) if views else '<p class="missing">' + missing + "</p>"}</section>'''

        for capture, pair in comparison_views(event).items():
            paired.append(
                gallery(
                    pair["role"],
                    pair,
                    capture.title(),
                    comparison=True,
                    reference_only=pair["reference_only"],
                )
            )
        for role in ROLES:
            roles.append(
                gallery(role, event["evidence"][role], role.replace("_", " ").title())
            )
        measurement = event["measurements"]
        delta = (
            "Unavailable"
            if measurement["displacement"] is None
            else f"{measurement['displacement']:.6g} relative units"
        )
        review_text = ""
        if event.get("review_status"):
            review_text = (
                '<p class="qualification">Review: '
                + esc(event["review_status"])
                + " · Original proposal: "
                + esc(", ".join(e["type"] for e in event["original_proposals"]))
                + "</p>"
            )
        events.append(f'''<article id="{esc(event["event_id"])}" data-type="{esc(event["type"])}" data-trust="{esc(event["trust"])}" data-label="{esc(event["label"].lower())}">
          <div class="event-heading"><div><h2>{esc(event["label"])}</h2><p><strong>{esc(event["type"])} · {esc(event["trust"])}</strong> · <code>{esc(event["event_id"])}</code></p></div>
          <a class="viewer-link" data-path="{esc(event["visualization"]["viewer_path"])}" href="{esc(event["visualization"]["viewer_url"])}">Inspect in 3D ↗</a></div>
          {review_text}<p>{esc(event["explanation"])}</p>{"".join('<p class="qualification">' + esc(x) + "</p>" for x in event["qualifications"])}
          <p class="measurement">Object displacement: <strong>{esc(delta)}</strong> · Basis: {esc(measurement["basis"])}</p>
          <details><summary>Object IDs, centroids, confidence and identity alternatives</summary><pre>{esc(json.dumps({k: event[k] for k in ("before_id", "after_id", "labels", "measurements", "confidence", "identity", "uncertainties")}, indent=2))}</pre></details>
          <section class="paired-comparison" aria-label="Before and After comparison"><h3>Images used for comparison</h3><div class="evidence-grid">{"".join(paired)}</div></section>
          <details class="detailed-evidence"><summary>Detailed evidence by role and missing views</summary><div class="evidence-grid">{"".join(roles)}</div></details></article>''')
    template = Path(__file__).with_name("report.html").read_text(encoding="utf-8")
    substitutions = {
        "TITLE": "Room change report"
        + (" · synthetic fixture" if report.get("synthetic") else ""),
        "ALIGNMENT": esc(scene["alignment"]["quality"]["status"]),
        "COUNTS": esc(
            counts
            + (
                " · VLM reviewed "
                + str(report["review_summary"]["coverage"]["reviewed"])
                + "/"
                + str(report["review_summary"]["coverage"]["total"])
                + (
                    " · fresh "
                    + str(report["review_summary"]["pass"]["fresh_reviewed"])
                    + "/"
                    + str(report["review_summary"]["pass"]["total"])
                    if report["review_summary"].get("pass")
                    else ""
                )
                if report.get("review_summary")
                else ""
            )
        ),
        "TRUST": esc(pretty(scene["counts_by_trust"])),
        "WARNINGS": "".join("<li>" + esc(w) + "</li>" for w in report["warnings"]),
        "COVERAGE": esc(json.dumps(scene["coverage"], indent=2)),
        "AVAILABLE": esc(pretty(scene["coverage"]["available_frames"])),
        "SELECTED": esc(pretty(scene["coverage"]["selected_frames"])),
        "DIAGNOSTICS": esc(json.dumps(scene["alignment"]["quality"], indent=2)),
        "TYPES": "".join(
            f"<option>{esc(x)}</option>"
            for x in sorted({e["type"] for e in report["events"]})
        ),
        "TRUST_OPTIONS": "".join(
            f"<option>{esc(x)}</option>"
            for x in sorted({e["trust"] for e in report["events"]})
        ),
        "EVENTS": "".join(events),
    }
    # Replace only template tokens; source labels may themselves contain token-like text.
    import re

    return re.sub(r"@@([A-Z_]+)@@", lambda match: substitutions[match[1]], template)
