# Room Reconstruction v2 — What Changed in the Room?

An end-to-end spatial AI workflow for comparing two phone recordings of the same
indoor space. It reconstructs each capture in 3D, builds a defensible shared
coordinate frame, associates objects across Before/After recordings, and presents
the result through interactive inspection, synchronized replay, and a change
report.

This project was built for the [Humanoid Spatial AI and Navigation internship
challenge](https://jobs.ashbyhq.com/humanoid/72741997-0fbc-413c-88a8-a273272641d1).
It is deliberately an offline-first, inspectable research prototype rather than
a real-time robot stack: expensive model inference runs in Colab, while saved
artifacts can be verified, queried, replayed, and compared locally on CPU.

## Project at a glance

**Input:** two short RGB phone videos, `data/before.mp4` and `data/after.mp4`,
recorded from different viewpoints with overlapping views of a small room.

**Output:** independently reconstructed Before/After rooms, an accepted
After-to-Before alignment when the evidence supports it, object identities and
uncertainties, estimated moved/added/removed events, source-linked image
evidence, and interactive 3D visualizations.

The completed example uses 32 Before and 30 After frames. Each frame remains
traceable to its source video, timestamp, image coordinates, depth, confidence,
and supporting cloud observations. The system keeps uncertainty explicit:
coordinates are relative rather than metric-calibrated, visibility is not
assumed from a missing detection, and ambiguous identities are not forced into
change labels.

### Demos

The most useful way to evaluate the project is to run the saved artifacts and
inspect the evidence, not only the final labels. The commands below reuse the
completed run and do not require another GPU inference.

1. **Replay reconstruction:** show the source video beside the reconstructed
   room. Play, pause, seek, change speed, orbit, and watch saved geometry appear
   when its earliest supporting observation is reached.

   ```powershell
   & $rdPython scripts/reconstruct.py sync-view --run outputs/reconstruction_colab --port 8092
   ```

   Open <http://127.0.0.1:8092>. Before and After have independent timelines
   and coordinate frames; replay is a retrospective reveal of a completed
   reconstruction, not online SLAM.

2. **Select an object and visualize it:** inspect the object inventory against
   the full room, focus or isolate a selected object, view its centroid/bounds,
   follow its supporting frames, and open the original image and saved mask
   evidence.

   ```powershell
   & $rdPython scripts/reconstruct.py object-view `
       --run outputs/reconstruction_colab --objects outputs/objects_colab --port 8094
   ```

   Open <http://127.0.0.1:8094>. The viewer distinguishes reliable 3D objects
   from image-only detections and never invents geometry for the latter.

3. **Compare Before and After:** run the CPU-only alignment, global identity
   matching, and change classification, then inspect shared objects side by side.

   ```powershell
   & $rdPython scripts/compare.py run `
       --reconstruction outputs/reconstruction_colab `
       --objects outputs/objects_colab `
       --refinement outputs/objects_refined_discovery `
       --output outputs/comparison_changes_v1

   & $rdPython scripts/reconstruct.py object-view `
       --run outputs/reconstruction_colab `
       --objects outputs/objects_colab `
       --refinement outputs/objects_refined_discovery `
       --comparison outputs/comparison_changes_v1 --port 8098
   ```

   Open <http://127.0.0.1:8098>. Select a shared identity to highlight its
   Before/After instances and open source-frame evidence. The generated report
   provides a portable, image-based view with the same evidence contract; see
   [change reports](docs/change_reports.md).

## Technical architecture

```text
phone videos
  -> decode, timestamp and hash every frame
  -> select sharp, diverse, connected frames
  -> independently reconstruct Before and After with VGGT
  -> export depth, cameras, RGB point clouds and observation provenance
  -> detect/segment objects and build per-capture inventories
  -> align After into Before with held-out visual/geometric checks
  -> globally match identities and classify conservative change events
  -> serve synchronized replay, object inspection and evidence reports
```

- **Reconstruction:** PyTorch/VGGT inference in a pinned Colab GPU
  environment; OpenCV handles video decoding, frame selection, image checks,
  and feature extraction.
- **Geometry:** camera intrinsics/poses and depth are converted into colored
  point clouds. Every point retains observation membership and source-frame
  provenance. Display sampling never changes saved measurements.
- **Object understanding:** segmentation and association produce stable
  per-capture object records, masks, descriptors, geometry quality, and
  supporting frames. Gemini refinement is optional and bounded; it proposes
  labels/identities but does not silently overwrite raw masks or geometry.
- **Comparison:** a CPU-only Sim(3) alignment maps After into the Before frame.
  Held-out residuals, inlier support, multiple frame pairs, plane structure,
  and scale stability gate acceptance. Matching is global and one-to-one, with
  an explicit unmatched option and ambiguity retention.
- **Presentation:** lightweight local Python servers and browser JavaScript
  provide 3D inspection, synchronized video replay, side-by-side comparison,
  source-image evidence, and portable HTML/Markdown/JSON reports.

## Design choices and tradeoffs

1. **Separate reconstruction before registration.** Each recording is processed
   independently so a failed capture cannot corrupt the other one. Alignment is
   a separate, auditable stage because the two reconstructions begin with
   arbitrary origins, orientations, and relative scales.
2. **Provenance over opaque results.** Source hashes, timestamps, masks,
   observations, pixels, depth, and camera references travel with the artifacts.
   This makes a label debuggable: a reviewer can go from a change event to the
   exact image and 3D support behind it.
3. **Conservative semantics.** Semantic proposals are advisory. Ambiguous
   matches, occlusions, missing geometry, and image-only detections remain
   explicit rather than being converted into confident “moved”, “added”, or
   “removed” claims.
4. **Offline-first iteration.** GPU/provider calls are isolated from local
   viewing, comparison, verification, and report generation. This makes
   experiments repeatable, keeps credentials out of artifacts, and makes the
   user-facing demo fast to rerun.
5. **Replay instead of pretending to be online SLAM.** The replay reveals the
   saved full-capture reconstruction according to observation time. It is
   honest about what the system computes while still making temporal coverage
   and source-to-geometry correspondence easy to understand.

The main tradeoff is that this MVP targets large room-scale changes and relative
spatial reasoning. It does not provide metric calibration, a watertight mesh,
rotation-change estimation, guaranteed physical absence, or real-time
performance. These boundaries are intentional and are included in every report.

## Quick setup

The repository is Windows-friendly and uses PowerShell commands. Local viewing,
comparison, and report generation use the existing compatible environment; GPU
inference is run through the Colab notebook.

```powershell
Set-Location 'C:\Users\prati\OneDrive\Desktop\Room Reconstruction v2'
$rdPython = 'C:\Users\prati\OneDrive\Desktop\Room Reconstruction\.venv\Scripts\python.exe'

# Import a completed reconstruction archive downloaded from Drive.
& $rdPython scripts/reconstruct.py import `
    --archive "$env:USERPROFILE\Downloads\roomdelta3d_v2_reconstruction.zip" `
    --output outputs/reconstruction_colab
& $rdPython scripts/reconstruct.py verify --run outputs/reconstruction_colab

# Install only the optional local stages you intend to run.
& $rdPython -m pip install -r requirements-viewer.txt
& $rdPython -m pip install -r requirements-comparison.txt
```

For a fresh GPU run, open
[`notebooks/reconstruction_v2_colab.ipynb`](notebooks/reconstruction_v2_colab.ipynb)
in VS Code with the Colab extension (or browser Colab), mount Drive, upload the
reviewed bundle, run the 4/8/16-frame pilots, then run separate full Before and
After inference. The notebook pins the environment, verifies the model
checkpoint, records runtime identities and memory diagnostics, and writes the
final archive plus receipt to Drive. The detailed [GPU workflow](#run-the-real-gpu-workflow--vs-code-colab-extension--drive)
and [developer checks](#developer-checks) follow below.

For the reasoning behind frame selection, coordinate systems, attention
chunking, artifact contracts, and failure diagnosis, see the
[raw-video-to-3D technical guide](docs/raw_video_to_3d_room_reconstruction.md).

## Current status

The completed independent GPU reconstruction is imported at `outputs/reconstruction_colab`: 32 Before and 30 After frames, depth, cameras, colored points, and source-observation links. Both captures pass saved-artifact consistency verification. Real-room accuracy remains independently unvalidated. `outputs/reconstruction` is the earlier prepared-only input, not the completed run.

Steps 2 and 3 now have a completed object inference/association pipeline and
per-capture inspection controls. The imported `outputs/objects_colab` artifacts
are available for local review; CPU synthetic and browser validation remain
separate from visual acceptance of the real object model. See [the detailed
implementation plan](docs/steps_2_3_implementation_plan.md).

Synthetic analytic planes and RGB gradients can be regenerated for regression/browser QA with `scripts/create_test_fixture.py --output outputs/reconstruction_test_fixture`. Generated test fixtures and temporary test runtimes are removed after use; they are not room reconstruction evidence.

The Colab notebook is `notebooks/reconstruction_v2_colab.ipynb`. Its upload bundle is generated by `scripts/build_colab_bundle.py`; the current reviewed bundle is `outputs/roomdelta3d_v2_colab_reviewed_20261015.zip`.

## Object inventories and inspection — Steps 2 and 3

Use the dedicated `notebooks/objects_v2_colab.ipynb` with `outputs/roomdelta3d_v2_objects_input.zip`. The bundle contains the completed saved reconstruction and current object-processing source. It excludes original videos, model weights, and credentials. No VGGT rerun is required.

1. Upload the input ZIP, still compressed, to **My Drive / Room Reconstruction v2**.
2. Open the object notebook in browser Colab or the VS Code Colab extension and choose a GPU runtime.
3. Run import/setup. It creates an isolated environment, resolves immutable model revisions, hashes weights, and records runtime identities.
4. Run the three-frame-per-capture pilot and inspect mask overlays. Then run the full 32/30-frame inventory. The pilot is partial coverage and does not establish full acceptance.
5. Run export to save a timestamped object ZIP and receipt beside the input in Drive. Download that result to Windows Downloads. Runtime caches remain local to Colab; exporting is required to preserve the final result.
6. Import alongside the same reconstruction and open the local viewer:

```powershell
Set-Location 'C:\Users\prati\OneDrive\Desktop\Room Reconstruction v2'
$rdPython = 'C:\Users\prati\OneDrive\Desktop\Room Reconstruction\.venv\Scripts\python.exe'
$rdRun = 'outputs/reconstruction_colab'
$rdObjects = 'outputs/objects_colab'
$rdObjectArchive = Get-ChildItem "$env:USERPROFILE\Downloads\roomdelta3d_v2_objects_2*.zip" |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $rdObjectArchive) { throw 'Download the timestamped object result ZIP from Drive first.' }
& $rdPython scripts/objects.py import --archive $rdObjectArchive.FullName --reconstruction $rdRun --output $rdObjects
& $rdPython scripts/objects.py verify --reconstruction $rdRun --objects $rdObjects
& $rdPython scripts/reconstruct.py sync-view --run $rdRun --objects $rdObjects --before-video data/before.mp4 --after-video data/after.mp4 --port 8092
```

Open `http://127.0.0.1:8092`. Select Before/After, search by ID or label, select an object, then use **Focus object**, **Isolate object**, centroid and bounds controls. Selection pauses video and displays the full saved object. **Follow playback** reveals only accepted object support observed by that time. Supporting-frame selection seeks the video and shows the saved mask overlay and close-up; original images remain separately available.

For a separate **object list + full 3D room** page, launch the object-only viewer:

```powershell
Set-Location 'C:\Users\prati\OneDrive\Desktop\Room Reconstruction v2'
$rdPython = 'C:\Users\prati\OneDrive\Desktop\Room Reconstruction\.venv\Scripts\python.exe'
& $rdPython scripts/reconstruct.py object-view --run outputs/reconstruction_colab --objects outputs/objects_colab --port 8094
```

Open `http://127.0.0.1:8094`. The left sidebar contains the object list and inspection controls; the right pane shows the complete selected capture immediately. There is no video panel, playback bar, speed control, or timeline. Focus, isolate, centroid/bounds, camera path, ceiling clipping, and optional still-image evidence remain available. Original video files and video conversion are not required. This local command reuses the existing saved artifacts; no Colab rerun or re-import is needed. The scene uses port 8095, so both 8094 and 8095 must be available. The playback viewer remains a separate command at 8092.

The ceiling crop applies to room context. Selected geometry bypasses it by default, with an explicit indication; **Apply ceiling crop to selection** opts into the same room clipping plane. Capture switching and Clear remove selection/evidence and restore context preferences. Objects lacking reliable 3D geometry remain in the unresolved 2D evidence list, with focus/isolation disabled.

For the standard saved viewer, use `& $rdPython scripts/reconstruct.py view --run $rdRun --objects $rdObjects --port 8090`. Both interfaces use the same inventory validation and exact observation-based geometry. Viewing needs no model runtime or GPU. Restart an already running viewer to load new controls.

Object output has a separate manifest/config/model lock and per-capture masks, observations, descriptors, membership, geometry, association diagnostics, and source previews. IDs are stable for identical saved inputs within an artifact version. Before IDs and After IDs do not establish shared physical identities. Centroids describe observed surfaces in independent relative coordinate frames, not calibrated physical centers.

## Shared alignment and large-change comparison

Steps 8–9 add CPU-only evidence selection and structured JSON, Markdown, and portable HTML reports. The existing comparison viewer can serve the report and focus events from report links. See [report generation, verification, viewing, and sharing](docs/change_reports.md) for exact PowerShell commands, evidence limitations, and the versioned contract.

The comparison stage is now available as a CPU-only saved-artifact module. It expects an
after-to-before set of fixed-structure correspondences and never changes raw reconstruction
or object files. Estimate and inspect a transform with:

```powershell
& $rdPython scripts/compare.py align `
    --correspondences outputs/fixed_structure_correspondences.npz `
    --output outputs/alignment.json `
    --threshold 0.05 --max-residual 0.10
```

The correspondence archive must contain finite `source` (After) and `target` (Before)
`Nx3` arrays. The output records the Sim(3) direction, inliers, residuals, and an
`accepted`, `provisional`, or `rejected` quality status. Use a rejected transform only for
diagnostic overlays.

After a usable alignment has been reviewed, compare the saved object inventory:

```powershell
& $rdPython scripts/compare.py match `
    --reconstruction outputs/reconstruction_colab `
    --objects outputs/objects_colab `
    --transform outputs/alignment.json `
    --tolerance 0.15 --output outputs/change_report.json
```

Matching is global and one-to-one across the room. Position is a soft clue, so a moved
object can retain its identity. Weak or competing matches remain ambiguous; unmatched
objects are not called added or removed without opposing visibility evidence. Measurements
remain in relative units and use `before_world` as the comparison frame.

To regenerate the upload bundle after source changes:

```powershell
& $rdPython scripts/build_objects_bundle.py --reconstruction $rdRun --output outputs/roomdelta3d_v2_objects_input_updated.zip
```

Use the regenerated object notebook with its matching bundle name. The builder preserves source reconstruction files and requires a new ZIP destination. Model inference uses the isolated `requirements-objects.txt`; saved viewing keeps `requirements-viewer.txt`. The model adapter follows the official [Grounding DINO](https://huggingface.co/docs/transformers/v5.13.0/en/model_doc/grounding-dino) and [SAM 2](https://huggingface.co/docs/transformers/model_doc/sam2) processor interfaces; clean Colab installation, model quality, and memory feasibility require the live pilot.

## Run the real GPU workflow — VS Code Colab extension + Drive

1. Create `Room Reconstruction v2` in **My Drive** and upload `outputs/roomdelta3d_v2_colab_reviewed_20261015.zip` into it. Keep the ZIP compressed.
2. Open `notebooks/reconstruction_v2_colab.ipynb` in VS Code. Select **Select Kernel → Colab → New Colab Server** and choose a GPU server. Use the same Google account as the Drive upload.
3. Run the first cell and complete the Drive authorization prompt. Its default input is `/content/drive/MyDrive/Room Reconstruction v2/roomdelta3d_v2_colab_reviewed_20261015.zip`; edit `DRIVE_FOLDER` if your folder differs. It copies and verifies the bundle in local runtime storage.
4. Run setup: isolated pinned PyTorch/torchvision, pinned VGGT commit, and a SHA-256 verified checkpoint (about 5 GB).
5. Run 4/8/16-frame pilots. Inspect actual depth panels and GPU allocated/reserved memory. Pilots have their own coordinate frames and are never substituted for full geometry.
6. Run full Before and After inference separately, then verify.
7. Run the final save cell. It writes `roomdelta3d_v2_reconstruction.zip` and `roomdelta3d_v2_reconstruction.receipt.json` directly into `My Drive/Room Reconstruction v2/`, beside the input ZIP, checking the copied archive's SHA-256. Existing results are preserved. Download that ZIP from Drive to your Windows Downloads folder, then use the local commands below.

Drive mounting is supported by the official extension from v0.2.1. If the mount prompt fails, check/update the extension, or use **Ctrl+Shift+P → Colab: Mount Google Drive to Server...**, run its appended mount cell first, then return to the first reconstruction cell. See the [official extension guide](https://github.com/googlecolab/colab-vscode/wiki/User-Guide) and [known limitations](https://github.com/googlecolab/colab-vscode/wiki/Known-Issues-and-Workarounds). The notebook also runs in browser Colab with a GPU runtime.

Inference and pilots use `/content` local runtime storage for performance. Results persist in Drive only after the final save completes; this is not an automatic checkpoint backup. The same-runtime resume/hash checks still apply.

Setup/import/pilot/full/verification/pack output is explicitly streamed into the notebook and saved in `/content/roomdelta3d_v2_workspace/logs/<stage>.log`. If a stage fails, its error includes the last 60 output lines, rather than only a parent `CalledProcessError`. Share that stage output when debugging; the wrapper traceback alone does not identify the cause.

Failed pilot/full workers print their capture, frame count, recorded runtime error, and last 60 worker-log lines. Pilot reports also retain `worker_log`, `worker_log_tail`, and `worker_error`. For an older uploaded bundle, inspect `/content/roomdelta3d_v2_workspace/pilots/p04/logs/`, `p08/logs/`, or `p16/logs/` (files `before.log` and `after.log`) and `reconstruction/pilot_report.json`. A failed pilot blocks full inference; diagnosis determines whether a dependency, configuration, or memory change is appropriate.

Workers bound eligible noncausal, zero-dropout inference attention to 512 query rows per SDPA call. Every query still attends to all keys/values across the capture; frames, image resolution, and global geometry are retained. This limits the temporary attention matrix on GPUs using the math backend, at the cost of more kernel calls. Kernel choice and floating-point rounding may differ. Runtime records retain the chunk size and call/token statistics. Code hashes invalidate older completed-worker caches when this implementation changes. Passing the repaired 16-frame pilot still does not prove the full 32/30-frame selections fit.

Setup checks pip even when the virtual-environment Python already exists. It repairs a missing pip using `ensurepip`, or the official PyPA `get-pip.py` fallback if the runtime omits `ensurepip`, and pins pip to 25.2. The fallback download URL and SHA-256 are recorded in `pip_bootstrap.json`. No environment deletion is needed for this repair.

The notebook records actual Python, CUDA, driver, GPU, package versions, checkpoint identity, tensor-exact upstream preprocessing parity, timings and sampled process RSS. Colab's assigned GPU and system Python/driver cannot be fixed by the notebook; they are recorded. This wheel set targets Python 3.11–3.13 and CUDA 12.6 PyTorch wheels. A pilot passing at 16 frames does not guarantee a 32-frame run will fit.

The notebook/bundle excludes original videos, Gemini credentials, VLM reports, evaluation annotations, checkpoint weights and installed environments. The selected originals are embedded, with frame IDs, source timestamps and video hashes retained.

## Import and inspect locally — PowerShell

Use the existing compatible local environment without installing PyTorch or changing the original project:

```powershell
Set-Location 'C:\Users\prati\OneDrive\Desktop\Room Reconstruction v2'
$rdPython = 'C:\Users\prati\OneDrive\Desktop\Room Reconstruction\.venv\Scripts\python.exe'
$rdArchive = "$env:USERPROFILE\Downloads\roomdelta3d_v2_reconstruction.zip"
$rdRun = 'outputs/reconstruction_colab'
& $rdPython scripts/reconstruct.py import --archive $rdArchive --output $rdRun
& $rdPython scripts/reconstruct.py verify --run $rdRun
& $rdPython scripts/reconstruct.py view --run $rdRun --port 8090
```

Open `http://127.0.0.1:8090`. Select Before/After, toggle cameras, choose a frame, or enter a **saved cloud point ID** and click **Show point source**. The original image marks the fractional source pixel; the sidebar lists model pixel, frame/time, relative depth, confidence, source-video hash and number of supporting observations. All point IDs remain accessible even if rendering is sampled.

For the synchronized visualizer, use the completed run and its original source videos:

```powershell
Set-Location 'C:\Users\prati\OneDrive\Desktop\Room Reconstruction v2'
$rdPython = 'C:\Users\prati\OneDrive\Desktop\Room Reconstruction\.venv\Scripts\python.exe'
$rdRun = 'outputs/reconstruction_colab'
& $rdPython scripts/reconstruct.py sync-view --run $rdRun --port 8092
```

Open `http://127.0.0.1:8092`. The source video is on the left and the room is on the right. **The room is fitted automatically** from the complete capture's robust bounds and the actual 3D viewport size, before playback starts. Drag to orbit or scroll to zoom; **Fit room** restores the overview. Playback preserves your viewpoint. Switching capture and changing the ceiling crop refit the selected room; resizing refits an automatic overview and preserves an orbit you adjusted manually.

Play/pause, restart, the seek bar and 0.25×/0.5×/1×/2× speed controls share the video's clock. Points appear when playback reaches their earliest supporting saved observation; seeking backward hides later groups. The saved final positions and RGB are retained, so this is a retrospective reveal of the full-capture reconstruction. Geometry updates at selected reconstruction frames, while the video plays continuously. Before and After use independent timelines and coordinate frames. Display sampling is deterministic and capped by `viewer.max_points` (200,000 by default); cropping is applied before sampling. The blue frustum marks the latest reached camera and the orange path shows its previous positions. Hide the ceiling with the crop controls; a 100% crop explicitly displays an empty-room message.

On first launch the viewer prints conversion progress while it prepares H.264/AAC playback copies of the original HEVC videos. Copies and hash-checked receipts live under `outputs/sync_cache`, outside the verified run. Later launches reuse them. Original video hashes are checked once at startup; source videos and saved reconstruction evidence are not edited. Install local-only dependencies with `& $rdPython -m pip install -r requirements-viewer.txt` if needed. No GPU environment or Colab rerun is required.

If originals were relocated, append `--before-video 'D:\Videos\before.mp4' --after-video 'D:\Videos\after.mp4'`. Overrides must match the recorded video hashes. Both localhost ports 8092 and 8093 must be free; use another `--port` to change the pair. Ctrl+C stops both servers. A single browser page controls playback: opening/reloading another page transfers control, and the older page reports that it must reload. Camera orbit and zoom remain available in the 3D pane.

To remove the ceiling from the view, enable **Hide ceiling** and adjust **Ceiling crop (%)**, starting at 15. The default **Ceiling direction** is `-Y`, matching the viewer's up direction. If it cuts the wrong side, change that direction; the captures are not gravity-aligned. This is a manual plane cut, so nearby tall objects can also be hidden. Settings are independent for Before/After and last for the viewer session. Disable the checkbox to restore the full display. Saved point IDs, files and reconstruction evidence remain unchanged; sampling occurs after clipping so the visible room retains its display budget. Restart the local viewer to load this change; no Colab run is needed.

For a depth-only viewer:

```powershell
& $rdPython scripts/reconstruct.py preview --run $rdRun --port 8091
```

Open `http://127.0.0.1:8091/before/previews/depth_preview.html` or the corresponding `after` path. Each panel shows RGB, numeric-range-labeled relative Z depth and confidence. Display ranges are consistent across one capture and independent across captures; raw arrays are never normalized or altered by visualization. Gray marks invalid/filtered samples.

For a new local viewer environment, use `requirements-viewer.txt` (NumPy, Pillow, Viser and PyAV). GPU libraries are not required for saved-run viewing. The provided reconstruction environment lock is intended for the isolated Colab GPU environment, not for overwriting another project's environment.

## Contracts and artifacts

```text
outputs/reconstruction/
  run_manifest.json             # prepared/running/completed/failed per capture
  config.json                   # explicit filtering, voxel and pilot settings
  models.lock.json               # source/weight revisions and checkpoint SHA-256
  preparation_environment.json
  pilot_report.json              # generated only by real GPU pilots
  environment_lock.txt           # actual Colab pip freeze, generated there
  before/                        # after/ has the same layout
    frames/                      # selected original RGB PNGs
    frames.json                  # frame IDs, source PTS/time-base and image hashes
    model_images/                # exact model RGB input PNGs
    image_transforms.json        # rotation, resize, crop, pad and inverse maps
    source.json                  # source video identity
    padding_valid.npz
    depth.npz                    # raw float32 (frames,height,width)
    confidence.npz               # raw float32; uncalibrated backend scores
    valid.npz                    # final retained observation mask
    cameras.json
    cloud.ply                    # binary float32 XYZ + uint8 RGB
    cloud.npz
    observations.npz
    inference_environment.json
    quality_report.json
    depth_contact_sheet.png
    previews/                    # individual depth/confidence/panels, scales + HTML
```

Paths in records are relative to the saved run root. Dense array order is explicitly bound by `frame_ids`. Source frame timestamps are independent between videos; they are not matched observations of a common time.

Camera records declare **OpenCV +X right, +Y down, +Z forward**, camera-to-world poses and their world-to-camera inverses, intrinsics in model-image pixels, and optical-axis Z depth. Before and After use `before_world` and `after_world`. All units are **relative** with unvalidated scale. There is no gravity alignment or implicit shared scale.

The cloud retains the highest-confidence source observation in each relative-sized voxel (ties retain input order). It is a representative sample, not an averaged point/color. `observations.npz` stores every retained depth observation with frame index, model pixel index, fractional original pixel, depth, confidence and cloud-row membership. `representative_observation` binds each PLY row to the point/color source. CSR offsets/indices recover all observations supporting a cloud point. Points not retained by the confidence/padding/invalid-depth policy are absent from the cloud; raw depth/confidence remain available.

Default confidence filtering retains values at or above the capture-wide 25th percentile. That threshold may remove little when scores are tied; it is not calibrated reliability. Automatic voxel size is `0.005 × median retained Z` in relative units, recorded in the report.

Quality reports verify array/frame/camera associations, finite geometry, pose matrices, point observation membership, source RGB, original/model maps, and sampled stored-point/self reprojection. Cross-view depth disagreement is diagnostic and includes occlusion, nearest-pixel sampling and view differences. It does not establish truth. **Completed** means saved computational artifacts passed consistency checks, not an accurate physical room. Real-scene accuracy and visual review remain unvalidated/pending until inspection of camera paths, surfaces, depth and coverage.

## Resume and transport

The reconstruction runtime review and comparison with the successful v1 run are in [docs/reconstruction_review_2026-10-15.md](docs/reconstruction_review_2026-10-15.md). The reviewed upload is `outputs/roomdelta3d_v2_colab_reviewed_20261015.zip`, paired with the current notebook. Upload that filename to the configured Drive folder and use a fresh Colab runtime. The notebook verifies its expected project hashes before setup and each inference phase; local edits do not update an existing remote extraction.

```powershell
& $rdPython scripts/reconstruct.py pack --run $rdRun --archive outputs/reconstruction_results.zip
```

GPU `run --resume` checks model/config/source-code identities and every completed capture artifact. If only After is corrupt, Before remains reusable. Missing GPU libraries/checkpoints are irrelevant to inspecting or reusing already verified saved geometry. Changed prepared image/mask/metadata refuses inference rather than silently reusing mismatched evidence. Model/config changes require a new prepared run.

All reconstruction writers use an exclusive `.run.lock`. A stale lock from a dead runtime must be removed only after confirming the original writer is stopped. No active lock is packaged. ZIP imports reject escaping paths, duplicate/symlink members and content mismatches, then verify completed captures before publishing the new directory. Original input videos are not copied or edited.

GPU failures preserve prepared inputs and the other capture's successful files. Worker logs identify the failing capture; available GPU/RSS timing measurements are saved on inference failures. A forced timeout may prevent final worker measurements.

If the Colab pilot cell fails, run the notebook's **Read-only diagnostics** cell immediately above it on the same runtime. It reads `pilot_report.json` and the last 60 lines of each failed worker log from `/content/roomdelta3d_v2_workspace/pilots/pXX/logs/`. It also works with older bundles and when no report was written. This inspection does not launch inference. The pilot cell prints these diagnostics before propagating a runner failure; missing memory measurements are shown as unavailable.

The outer `Pilot failed` exception alone does not identify a CUDA memory failure or any other worker cause. Preserve the current runtime while inspecting its logs. Updating a local notebook or Drive ZIP does not replace code already extracted under `/content/roomdelta3d_v2_bundle/project`; a fresh runtime must extract the current bundle to use its runner fixes.

Pilot reports remain `running` until all configured sizes complete. Full inference requires a complete report bound to the current inputs, configuration, model lock and inference code. Changed code requires the pilots to run again; changed configuration/model inputs require a fresh workspace. Setup checks worker imports and a small CUDA attention operation before downloading weights. These checks do not establish full-model memory feasibility.

## Developer checks

```powershell
& $rdPython -m unittest tests.test_reconstruction -v
& $rdPython scripts/reconstruct.py prepare --output outputs/reconstruction_repeat
& $rdPython scripts/build_colab_bundle.py --run outputs/reconstruction_repeat --output outputs/colab_repeat.zip
```

Regression checks use known analytic coordinates, the pinned upstream PIL preprocessing source, malformed depth, source RGB/provenance, semantic corruption, isolated resume, concurrency and ZIP transport. They establish implementation behavior, not VGGT accuracy on these videos.

The upstream reference is vendored with its original license header only for parity validation: https://github.com/facebookresearch/vggt/blob/a288dd0f14786c93483e45524328726ab7b1b4ce/vggt/utils/load_fn.py .


## Gemini inventory refinement and shared identity proposals

For the current **object inspection step**, use the viewer command below without `--comparison`. The left list defaults to objects with saved 3D points; clicking an entry highlights it in the right-hand room view. Focus and Isolate are available when needed. Choose Before or After to inspect each capture independently. Camera paths start off. Alignment overlays and cross-capture identity controls are reserved for the later comparison step. Image-only detections have a separate filter and retain their image evidence.

Refinement uses the completed reconstructions and object inventories. It creates a separate derived inventory with canonical labels, individual items and small clutter collections, source IDs, geometry quality, image evidence and proposed Before/After identities. It does not align the captures or classify physical changes.

```powershell
Set-Location 'C:\Users\prati\OneDrive\Desktop\Room Reconstruction v2'
$rdPython = 'C:\Users\prati\OneDrive\Desktop\Room Reconstruction\.venv\Scripts\python.exe'
& $rdPython -m pip install -r requirements-refinement.txt

# Local evidence preparation only: no provider calls.
& $rdPython scripts/refine_objects.py prepare --reconstruction outputs/reconstruction_colab --objects outputs/objects_colab --output outputs/objects_refined_discovery

# One joint Before/After generation; at most one targeted follow-up.
& $rdPython scripts/refine_objects.py run --reconstruction outputs/reconstruction_colab --objects outputs/objects_colab --output outputs/objects_refined_discovery --baseline outputs/objects_refined

# Verify provenance and geometry, without model execution.
& $rdPython scripts/refine_objects.py verify --reconstruction outputs/reconstruction_colab --objects outputs/objects_colab --output outputs/objects_refined_discovery

# Object-only viewer, with Refined / Original selection.
& $rdPython scripts/reconstruct.py object-view --run outputs/reconstruction_colab --objects outputs/objects_colab --refinement outputs/objects_refined_discovery --port 8096
```

Open <http://127.0.0.1:8096/>. Ports 8096 and 8097 must both be available. Opening or operating the viewer makes no Gemini calls. Existing viewer commands without `--refinement` continue to inspect the original artifacts.

Unknown objects are excluded from the viewer object list; saved segmentation artifacts remain intact. **Image-only detections** automatically open their supporting photo, highlight its saved camera frustum, and color cloud points supported by that frame in amber. The cloud highlight uses saved observation-to-cloud provenance, including points also observed in other frames. It covers the full source frame, not an object mask. Large highlights are sampled for display, with shown/total counts. Choose a different supporting frame to update the photo, camera, and cloud highlight. Amber photo outlines indicate a context region (or the full frame), never an inferred object mask or 3D position. Actual segmented objects remain cyan and automatically open their supporting photo, without adding source-camera or full-frame cloud highlights. The supporting-frame selector updates the photo for these objects. Clear or switch captures to remove the source highlight. No playback or alignment is needed for this inspection flow.

The revised discovery workflow uses a new output directory so the earlier `outputs/objects_refined` snapshot remains inspectable. Reusing the completed output uses its saved answers; it does not regenerate them. For an explicitly offline rebuild, append `--offline` to the refinement `run` command. Use a new output directory when changing source inputs or configuration. `--no-followup` restricts a new analysis to the initial generation. A failed/unknown request stops; `--retry-failed` permits an explicit retry within the same persisted attempt budget, not additional unbounded attempts.

`--baseline` preserves prior inventory decisions for evidence the focused pass did not revisit; newer mask classifications take priority. The baseline must have the same reconstruction and raw-object binding. Omit it for a first refinement. Optional `--review-notes path/to/notes.json` supplies audited capture/evidence checks to the bounded follow-up. `--answer-file path/to/saved-response.json` imports an answer or sanitized request receipt and implies offline execution: conservative reference normalization and inventory rebuilding make zero provider calls. Original responses remain in `requests/`; imported-response provenance is recorded separately.

Credentials: `GEMINI_API_KEY` or `GOOGLE_API_KEY` takes precedence. For compatibility, the CLI can read the existing local Gemini smoke-test configuration without executing it, via `--credential-source`; no credential is copied into output artifacts. No model fallback is automatic.

`configs/refinement.json` defaults to `gemini-3.1-flash-lite`, 15 RPM / 250,000 input TPM / 500 RPD, a 125,000-token request ceiling, a 65-second minimum generation interval, two generation attempts per analysis, and a local daily generation cap of ten. Requests are serialized with a shared local project ledger under `outputs/refinement_cache`. Token-count preflight calls are separate API calls, cached as well. Daily accounting resets at Pacific midnight. These are local safeguards; other clients sharing the Google project can consume quota too. Quota errors never trigger automatic retry loops.

Evidence preparation covers every saved observation with numbered crop sheets and up to eight context frames per capture. Gemini returns labels, groups, mask roles and identity proposals. Local validation catches invalid references, missing decisions, co-visible distinct objects and inappropriate large-item collections. Distinct chair/backpack identities can coexist at the same location. A mixed mask is image evidence, not permission to assign the whole point set to both objects. Small collections cannot provide rigid-object movement measurements. Low-confidence instance splits and cross-capture alternatives remain uncertain.

Outputs in the refinement directory:

- `summary.md`: inventory counts, validation decisions, source-frame links and paired evidence sheets.
- `refinement.json`: reconstruction binding, shared identity proposals, unresolved evidence, quality counts and API usage.
- `answer.json` and `validated_answer.json`: original model proposals and locally checked/split decisions.
- `before/` and `after/`: canonical objects, accepted source membership and derived geometry. Measured centroids use approved source observations in each capture's relative coordinate frame.
- `evidence_index.json`: source IDs, frame IDs, timestamps and mask references. `evidence/` holds the actual request images and paired evidence.
- `repair_tasks.json`: mixed masks and visible items requiring their own segmentation. Repairs are queued for targeted segmentation; no GPU inference is performed by this workflow.
- `alignment_candidates.json`: visual hints requiring later geometric validation, not accepted registration anchors.
- `requests/`, `budget.json`, and saved prompts/schema: sanitized request receipts and the persistent generation attempt counter.

Real-pair results require review. A named entry is not automatically a correct or complete physical identity; inconsistent geometry and omitted observations are retained explicitly. Unmatched identities are not additions/removals. Geometry status and observed-surface centroid variation are preparation for the next alignment/visibility stage, not a calibrated movement tolerance. There is no rotation estimation or movement trajectory stage.

The discovery update explicitly reviews laptops, drinking bottles/flasks, oil bottles, pressure cookers, pans, plates, utensils and food containers in each capture. It adds up to six unmasked desk/storage detail views and a sheet of larger, diverse monitor/TV/lamp masks per capture. These search categories do not force an object to exist: each receives an evidence-linked visible, uncertain or not-observed decision. Missing category decisions and generic collections hiding distinct items trigger review warnings and the bounded follow-up.

A laptop mask incorrectly labelled monitor/television can be renamed from its appearance. A pan/lid labelled lamp must be visually rechecked. Invalid electronics collections retain individual image-only review candidates instead of silently dropping all their observations. Items discovered in unmasked detail views are named even without a dedicated mask, with zero invented 3D points and a targeted segmentation repair task. The detector vocabulary in `configs/objects.json` is expanded for future object-inference runs; existing saved masks are not modified or regenerated by the Gemini refinement script.

## Shared-frame alignment and global identity matching

The saved-run comparison connects the refined inventories to room alignment, identity matching and simple object-change classification. It uses local image features, saved depth, camera poses and segmentation masks; it performs no GPU inference and makes no VLM/provider calls.

```powershell
Set-Location 'C:\Users\prati\OneDrive\Desktop\Room Reconstruction v2'
$rdPython = 'C:\Users\prati\OneDrive\Desktop\Room Reconstruction\.venv\Scripts\python.exe'
& $rdPython -m pip install -r requirements-comparison.txt

& $rdPython scripts/compare.py run --reconstruction outputs/reconstruction_colab --objects outputs/objects_colab --refinement outputs/objects_refined_discovery --output outputs/comparison_changes_v1

& $rdPython scripts/compare.py verify --reconstruction outputs/reconstruction_colab --objects outputs/objects_colab --refinement outputs/objects_refined_discovery --output outputs/comparison_changes_v1

& $rdPython scripts/reconstruct.py object-view --run outputs/reconstruction_colab --objects outputs/objects_colab --refinement outputs/objects_refined_discovery --comparison outputs/comparison_changes_v1 --port 8098
```

Open <http://127.0.0.1:8098/>. **Before and After appear side by side**, using separate scene ports 8099 and 8100. Choose one shared identity from the left: its available Before/After instances highlight together, and floating source-frame panels open in the corresponding rooms. Objects observed in only one recording highlight only there. The missing side opens a **region reference** photo, selected by projecting the saved object/context geometry into the opposing camera views. A purple outline marks the estimated comparison area; use its frame selector to inspect other views. If alignment or region support is unavailable, all saved frames remain available for manual comparison without an inferred outline. Region references help visual inspection; they do not prove visibility/absence or change the saved classification. Cyan marks saved object geometry. Image-only objects retain null centroids and show their supporting full-frame cloud area in amber. Each source-frame panel has its own frame selector, close control and enlarged image view. **Show frames** restores closed panels; **Clear selection** clears both scenes. Ambiguous pairs remain labelled as possible matches. Each room supports independent orbit/zoom; **Fit rooms** restores both. The existing single-capture inventory viewer remains available without `--comparison`. There is no source-video playback.

Alignment uses reciprocal SIFT matches within corresponding fixed-instance masks or the room shell outside other detected objects. Object centroids and camera paths are not fitting landmarks. Whole selected frames in either capture are withheld from fitting. The gate checks held-out residuals, inlier support, multiple frame pairs, distributed geometry, separate wall/floor plane directions and scale stability. Shell planes are fitted from Before training frames only; they partition support and do not create image correspondences. The configured fixed-label assumptions and residual threshold are recorded. Named static instances cannot match across region IDs. Collinear landmarks and non-rigid/reflected transforms are rejected.

After-to-Before mapping is `p_before = scale * rotation @ p_after + translation`. Separate exports transform the full After cloud, object geometry/centroids/bounds, camera positions/orientations and optical-axis depth scale together. Point-row ordering, RGB, observation membership, representative IDs, pixels and supporting frame references remain tied to the original artifacts. Raw reconstruction and segmentation outputs are preserved.

`outputs/comparison_changes_v1` contains `alignment.json`, `identities.json`, `changes.json`, `comparison.json`, `summary.md`, source-linked `correspondences.npz` / `correspondence_index.json`, and aligned exports under `after/`. `verify` checks source bindings, hashes, coordinate transformations, camera/depth scale and identity coverage. For new comparisons it also recomputes the change results from original source geometry and frame memberships, checking distances, thresholds, evidence references and counts. Repeating `run` reuses bound local feature correspondences. Changed source/config bindings require a new output directory.

Identity matching searches the entire room using semantic compatibility, saved appearance descriptors, shape and supporting geometry. Accepted room alignment permits a small position cost; distance never vetoes an identity match. Every object has an unmatched option, even with equal inventory sizes. Competing global assignments and alternatives in both directions remain ambiguous. Existing VLM identity proposals are advisory. A matching label alone cannot establish an identity; an image-only proposal is distinguished from a descriptor-supported match.

On the supplied real pair, the initial connected run passed its engineering alignment checks: 4,182 feature correspondences, 80.3% held-out inliers, median held-out residual 0.0122 relative units, scale 1.0132, and 0.49% scale variation across subsamples. Matching retained 3 descriptor-supported pairs, 7 visual-only proposals, 9 ambiguous pairs and 7 unmatched entries. These are real-video diagnostics, separate from the analytic regression fixtures. The refrigerator remains ambiguous because its retained mask appearances disagree; this is not an unchanged/moved conclusion. The gaming-chair fragments also retain capture-identity ambiguity.

### Simple object-change rules

Changes require accepted alignment and an unambiguous identity. Objects with usable 3D geometry on both sides use their object centroids. If either side lacks usable object geometry, both sides use the arithmetic mean of the **unique cloud points supported by all of that object's frames**, including points also observed in other frames. These means use complete saved clouds, unaffected by display sampling or ceiling cropping. After locations transform into Before coordinates once. Image-only object centroids remain null; comparison locations are separate fields in `changes.json`.

The default movement threshold is 20% of the Before room extent (the diagonal between the cloud's per-axis fifth and ninety-fifth percentiles). Set `movement_extent_fraction` in `configs/comparison.json` to adjust it. A distance above the threshold is **moved**; at or below it is **unchanged**. Direct object-centroid comparisons use the **reliable** MVP evidence basis. Frame-cloud comparisons are **estimated**, and their `frame_cloud_distance` is distinct from `object_displacement`.

An unambiguous object observed only Before is **removed**; one observed only After is **added**. These are **estimated presence-only** results: the simplified rule deliberately does not check whether the opposing location was visible or empty. Competing identities (including their unmatched alternatives), rejected alignment and missing frame-cloud support remain **uncertain**. Every shared identity has one event.

**Limits:** whole-frame means describe visible room context, so camera framing can cause estimated movement even when an object stayed still. Missed detections can cause estimated additions/removals. The method targets large room-scale moves. Reliable geometry means usable object support for this MVP, not independent depth calibration, metric accuracy or verified physical changes. Rotation is not evaluated; unchanged means below the location threshold. No new Gemini/provider calls or GPU inference are made.

Existing `comparison-v1` outputs remain readable; the new `run` refuses to overwrite them. Use a new output directory, as above. Legacy `align` and `match` retain their conservative diagnostic behavior; use the connected `run` workflow for these simple change rules.
