# Steps 8–9: saved-image evidence and change reports

Reports run on CPU against existing saved artifacts. They do not rerun VGGT, object models, matching, or classification, and make no provider calls. Original video files are not required. Use a comparison produced by `scripts/compare.py run` with `comparison-v2` and `changes.json`; the older identity-only comparison is not sufficient.

## Generate, verify, view, and share

Run in PowerShell. Each report and ZIP needs a new destination; existing results are preserved. Use the same source directories that produced the comparison, including its refinement when applicable.

```powershell
Set-Location 'C:\Users\prati\OneDrive\Desktop\Room Reconstruction v2'
$rdPython = 'C:\Users\prati\OneDrive\Desktop\Room Reconstruction\.venv\Scripts\python.exe'
$rdReconstruction = 'outputs/reconstruction_colab'
$rdObjects = 'outputs/objects_colab'
$rdRefinement = 'outputs/objects_refined_discovery'
$rdComparison = 'outputs/comparison_changes_v1'
$rdReport = 'outputs/change_report_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
$rdArchive = $rdReport + '.zip'

& $rdPython scripts/compare.py report `
    --reconstruction $rdReconstruction --objects $rdObjects `
    --refinement $rdRefinement --comparison $rdComparison `
    --output $rdReport --viewer-url 'http://127.0.0.1:8098/'
if ($LASTEXITCODE -ne 0) { throw 'Report generation failed.' }

& $rdPython scripts/compare.py verify-report `
    --reconstruction $rdReconstruction --objects $rdObjects `
    --refinement $rdRefinement --comparison $rdComparison --output $rdReport
if ($LASTEXITCODE -ne 0) { throw 'Report verification failed.' }

& $rdPython scripts/compare.py pack-report `
    --reconstruction $rdReconstruction --objects $rdObjects `
    --refinement $rdRefinement --comparison $rdComparison `
    --output $rdReport --archive $rdArchive
if ($LASTEXITCODE -ne 0) { throw 'Report packaging failed.' }

& $rdPython scripts/reconstruct.py object-view `
    --run $rdReconstruction --objects $rdObjects --refinement $rdRefinement `
    --comparison $rdComparison --report $rdReport --port 8098
```

Open `http://127.0.0.1:8098/report`. **Inspect in 3D** opens the same event in both room scenes at the viewer root. The viewer uses ports 8098–8100. Change `--port` and `--viewer-url` together if these ports are occupied. Reports served through the viewer automatically use its current origin for 3D links. Other existing viewer commands continue to work without `--report`.

For offline viewing, unzip the complete package and open `report.html`; keep `assets/` beside it. Search, filters, alternative frame selection, full-frame/highlight/close-up controls, and image enlargement work without a server. JSON and Markdown can also be opened directly. Interactive 3D still requires the original saved reconstruction/inventory/comparison and running local viewer; the portable report does not embed room geometry. Sharing only the HTML file omits its images.

Every event starts with **Images used for comparison**, showing Before and After together. If a direct object view is unavailable, the pair reuses the saved location-reference image for that capture and labels it explicitly. This does not establish object presence or absence. Expand **Detailed evidence by role and missing views** for the four underlying roles and their limitations. Saved highlights are shown by default: segmented object masks are tinted cyan, while context/projected regions retain their outlines. Full frame, close-up, and alternative frames remain selectable. If a frame has no saved highlight, the original full frame is shown.

To update an existing report's presentation without repeating image selection, export to a new directory:

```powershell
& $rdPython scripts/compare.py refresh-report `
    --reconstruction $rdReconstruction --objects $rdObjects `
    --refinement $rdRefinement --comparison $rdComparison `
    --source-report outputs/change_report_v2 --output outputs/change_report_paired `
    --viewer-url 'http://127.0.0.1:8120/'
```

Use `--report outputs/change_report_paired --port 8120` when launching the viewer. Refresh verifies source bindings, saved frames/regions and all existing artifact hashes, copies the saved evidence, and regenerates HTML/Markdown. It preserves classifications, trust, measurements and the original report, and records the new presentation's code hashes in the new manifest. If the source report used a review, pass the matching `--review` directory as well.

An optional `--config` JSON file accepts only:

```json
{
  "viewer_url": "http://127.0.0.1:8098/",
  "evidence": {
    "max_views": 3,
    "crop_padding": 0.1,
    "minimum_projection_coverage": 0.2
  }
}
```

The configuration file's viewer URL takes precedence over the CLI default. Reports support one to three views per role. Defaults choose the primary view and up to two alternative frames, preferring different camera poses.

## Evidence and trust

Four roles distinguish the object from its location: `object_before`, `object_after`, `old_location_after`, and `new_location_before`. Missing object identities have no direct object image. Opposing location images use saved geometry/camera projection only with accepted alignment. Unlocated regions have no fabricated highlight or crop. A useful existing context image can still be retained as a full frame.

Ranking records coverage, border clipping, regional sharpness, depth/geometry support, and a possible-occlusion penalty. Direct masks precede context-only observations. Alternatives use distinct frame IDs; ties use frame and observation IDs. An inherited shelf/container segmentation for an image-only refined object is explicitly context, not a mask of the named item. Outlines do not assert pixel-level object membership.

Depth checks reuse the object association `depth_margin_voxels` and saved valid-depth filter. At least eight distinct projected model pixels are required, and a majority of closer surfaces flags suspected occlusion. All visibility remains unverified; even a matching-depth projection does not establish empty space or absence. View limitations are `no_supporting_view`, `outside_frame`, `suspected_occlusion`, `visibility_unknown`, `alignment_unusable`, and `region_unavailable`.

Full frames retain the exact source PNG bytes and original resolution. Highlights also retain source dimensions. Close-ups contain original, untinted pixels with 10% padding per axis and bounds clipping. Each selected view records original pixel boxes, source image/mask hashes, frame/observation IDs, decoded frame index, original PTS/time base, source and elapsed timestamps, ranking details, and selection reasons. Assets are deduplicated by content/source and render settings.

Classification is preserved from `changes.json`. Presence-only additions/removals remain estimates without an empty-region check. Whole-frame-cloud locations/distances are kept separately from object centroids/displacement. Null geometry stays null. “Reliable” describes usable MVP geometry, not independently validated physical change. Units remain relative.

## Contract and integrity

`report.json` uses `schema_version: change-report-v1`. Its `scene` contains alignment diagnostics, type/trust counts, movement threshold, and coverage. `events` are ordered by stable shared identity (`event_id == shared_id`); each contains capture-specific IDs/labels, unchanged source identity alternatives, measurements, confidence/reasons, qualifications, four evidence bundles, and viewer URLs. Confidence contains source assessments and no invented calibrated probability. `frame_cloud_estimate` is separate from `source_centroids`, common-frame `centroids`, and object displacement.

Coverage counts available/selected saved frames and events with each evidence role, including missing and unknown views. It does not estimate the fraction of the physical room surveyed.

`summary.md` and `report.html` are generated from the same JSON. `manifest.json` binds source manifests, changes, identities, configuration, generating code, and every exported artifact. Generation uses a writer lock and staging directory, verifies references/hashes/measurements/regions, then publishes atomically. Changed or mixed upstream inputs require a new report. Verification detects stale sources, corrupted exports, invalid event references, unsafe paths, and classification/measurement disagreement. The web handler serves only manifest-listed report assets.

Generation/verification success establishes artifact consistency. Visual acceptance requires reviewing every event's primary object/location images, particularly image-only/context observations and uncertain identity matches. Record that review separately; successful export is not evidence that every upstream object label or change is correct.
