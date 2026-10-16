# Gemini review of saved room changes

The reviewer covers every shared identity, including unchanged and uncertain proposals. Update means movement only. No reconstruction, segmentation or detection is rerun. Raw inventories and comparisons remain unchanged.

## Prepare without credentials or network calls

Run from Room Reconstruction v2 with the existing Python environment:

```powershell
$py = '../Room Reconstruction/.venv/Scripts/python.exe'
& $py scripts/compare.py review --reconstruction outputs/reconstruction_colab --objects outputs/objects_colab --refinement outputs/objects_refined_discovery --comparison outputs/comparison_changes_v1 --output outputs/change_review_v1 --prepare-only
```

Preparation saves a verified evidence report, indexed room-wide contact sheets, and a partial `review.json`. Unreviewed objects are explicitly uncertain with `review_status: not_reviewed`; original classifications remain in `original_proposals`. Preparation does not claim that Gemini reviewed the pictures.

## Configure verified free-tier quotas

Copy `configs/change_review.json` to an ignored local file under `outputs/`. Use a supported review model (Gemini 3.1 Flash Lite or Gemini 3.5 Flash Lite) and the same project scope as the refinement workflow. In Google AI Studio, verify that this model is available on the project's free tier and record:

- `quota_verified: true` and `free_tier_verified: true`;
- `quota_verified_at`: Unix timestamp when you checked the allowance;
- `requests_per_minute`, `input_tokens_per_minute`, `requests_per_day`;
- `remaining_requests`: known remaining daily requests at that timestamp.

These are operator-supplied observations, not values inferred from the old refinement config. Refresh the snapshot after Pacific midnight. Do not put API keys in config files. Set `GEMINI_API_KEY` or `GOOGLE_API_KEY` in the process environment.

```powershell
& $py scripts/compare.py review --reconstruction outputs/reconstruction_colab --objects outputs/objects_colab --refinement outputs/objects_refined_discovery --comparison outputs/comparison_changes_v1 --output outputs/change_review_v1 --config outputs/change_review.local.json
```

The review stage has no fixed two-generation or ten-generation/day ceiling. It retains at least 65 seconds between generations, uses 90 percent of verified minute limits, counts preflight and failed requests, and shares the existing project cache/quota ledger under this checkout's `outputs/refinement_cache`. Existing refinement retains its own two-generation budget. Other applications using the same project can consume quota outside this ledger; a provider quota response stops the run.

Use `--batch-size-limit 1` to resume with smaller requests after inspected provider capacity failures, without changing source bindings or replaying completed batches.

To deliberately continue unfinished work with another supported model, pass `--execution-config outputs/alternate_model.local.json` containing that model's verified free-tier quotas. Keep the original `--config` and project scope. Each new batch saves its actual provider configuration; verification checks its model-specific receipt. Completed answers retain their original model and are reused. The shared project ledger is preserved; there is no automatic model fallback.

A failed or unknown provider outcome is never automatically retried, including in a different output directory sharing the cache. Inspect receipts before explicitly adding `--retry-failed`; prior failed/unknown receipts are retained under `requests/attempts/`. Update an expired/exhausted quota snapshot before resuming. Completed batches are reused. Unresolved identities with a third selected view receive at most one targeted follow-up, grouped into batches of up to four identities to share room context. Large multi-identity requests split deterministically; an oversized single-identity request stops without discarding evidence.

The optional `--evidence-report outputs/change_report_v2` reuses a verified original report without rerunning evidence selection. Historical HTML is hash checked, while canonical JSON and image provenance remain authoritative; a renderer update does not invalidate review evidence. A new review output is required when the review prompt changes.

The optional `--credential-source ../"Room Reconstruction"/scripts/test_gemini_vlm.py` reads an already configured local key as data, without executing or copying its script. Environment variables take precedence. No credential value is written to review artifacts.

AI Studio's 28-day **peak** usage is not today's usage. When an exact current-day count is unavailable, reserve at least the larger of the reported daily peak and the known local API request count; this is a conservative allowance, not a live provider quota query. Provider quota errors still stop the run.

## Verify, display and export

```powershell
& $py scripts/compare.py verify-review --reconstruction outputs/reconstruction_colab --objects outputs/objects_colab --refinement outputs/objects_refined_discovery --comparison outputs/comparison_changes_v1 --output outputs/change_review_v1
& $py scripts/reconstruct.py object-view --run outputs/reconstruction_colab --objects outputs/objects_colab --refinement outputs/objects_refined_discovery --comparison outputs/comparison_changes_v1 --review outputs/change_review_v1 --port 8116
& $py scripts/compare.py report --reconstruction outputs/reconstruction_colab --objects outputs/objects_colab --refinement outputs/objects_refined_discovery --comparison outputs/comparison_changes_v1 --review outputs/change_review_v1 --output outputs/change_report_reviewed_v1 --viewer-url http://127.0.0.1:8116/
```

Use the same `--review` for report verification/packing, or the saved review source location. A reviewed report automatically selects its corresponding review when opened in the viewer. A raw report cannot be combined with a reviewed display. Browsing never calls Gemini. Older comparison and report artifacts still open without a review.

## Decision and evidence rules

Gemini gets source pictures, crops, proposed regions and room-wide context, but no original classifications or distances. Findings must cite supplied capture-specific frames and normalized full-frame boxes. Appearance and fixed landmarks support identity across the whole room. Invalid references, conflicts, occlusion, unseen regions and ambiguous assignments stay uncertain. New requests enforce 0–1000 box bounds in the response schema, and each batch records its actual schema so earlier receipts remain verifiable. A supporting shelf/container cannot be silently reassigned as the item it supports. If a VLM presence claim abandons an already inventoried matched counterpart, that counterpart must be accounted for before publishing added/removed. Added/removed results require a targeted close-up confirmation of the source object and visible opposing region, and remain estimated. A coarse room view alone cannot confirm disappearance or replacement.

Global reconciliation preserves each source object exactly once, including unmatched competitors. A corrected match merges added/removed proposals into one identity. Newly located counterparts remain uncertain until a targeted follow-up inspects their proposed close-up and full-frame context. Confirmation must refer to the candidate region; a different unconfirmed box remains uncertain. This also qualifies as useful additional evidence when no third inventory view exists. Newly located counterparts are derived image-only observations with null centroids; they never inherit a parent object's geometry. Existing arithmetic means and movement threshold remain authoritative for numerical comparisons. Visual disagreement downgrades the result to uncertain rather than inventing a displacement.

`review.json` records original proposals, reviewed identities/events, capture IDs, evidence findings, derived observations, movement basis, uncertainty, coverage and API receipts. Verification replays saved findings against original geometry and rejects changed results even when artifact hashes have been updated. Review/report outputs are separate artifacts; regenerating a review makes reports bound to an older review stale.

Saved structured responses can be imported using `--answers path/to/answers.json`, without credentials or provider calls. The file uses the response schema returned by `reconstruction_v2.review_evidence.schema()` and must reference shared IDs in `evidence_index.json`. Offline imports are explicitly marked `origin: offline` in the saved steps. A completed workflow means every source identity received a review attempt; uncertain conclusions are retained, and this is not an independently measured accuracy score.

A source-picture audit can reject a model claim using an optional `local_visual_validation.json` inside the review directory. Its `local-visual-validation-v1` rejections cite a known shared identity, frame ID, matching image hash, and reason. Replay only downgrades affected events to uncertain; it cannot confirm changes or invent locations. Model answers and receipts remain intact. Re-publishing with `--prepare-only` preserves a previous provider stop reason.
