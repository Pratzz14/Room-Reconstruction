"""Build a credential-free single-upload Colab bundle and executable notebook."""

import argparse
import hashlib
import json
import sys
import tempfile
import textwrap
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reconstruction_v2.archive import pack
from reconstruction_v2.core import digest, read_json, write_json


def project_files(project):
    return list((project / "reconstruction_v2").glob("*.py")) + [
        project / "scripts" / name for name in ("reconstruct.py", "setup_colab.py", "colab_run.py")] + [
        project / "configs/reconstruction.json", project / "configs/models.lock.json",
        project / "requirements-reconstruction.txt", project / "reference/vggt_load_fn.py"]


def notebook(bundle_name="roomdelta3d_v2_colab_ready.zip"):
    project = Path(__file__).resolve().parents[1]
    expected_files = {p.relative_to(project).as_posix(): digest(p) for p in project_files(project)}
    cells = []
    def cell(kind, source):
        head, tail = source.split("\n", 1) if "\n" in source else (source, "")
        source = (head.strip() + "\n" + textwrap.dedent(tail)).strip() + "\n"
        value = {"cell_type": kind, "id": hashlib.sha256(source.encode()).hexdigest()[:12], "metadata": {}, "source": source.splitlines(keepends=True)}
        if kind == "code":
            value.update({"execution_count": None, "outputs": []})
        cells.append(value)
    cell("markdown", """# Room Reconstruction v2 — independent VGGT reconstruction

    In VS Code, open this notebook and select **Select Kernel → Colab → New Colab Server**, choosing a GPU server.
    Put `__BUNDLE_NAME__` in **My Drive / Room Reconstruction v2** before running the first cell.
    The first cell mounts Drive, copies the ZIP to local runtime storage, and verifies its contents.
    It contains only the reconstruction code/configuration and the 32 Before + 30 After selected PNGs/model inputs.
    No videos, Gemini credentials, VLM prompts, evaluation labels or model checkpoints are included.

    Run cells in order. Pilot outputs are separate from full predictions. All coordinates remain relative, and successful inference does not establish real-scene accuracy.
    The setup uses an isolated Python environment with pinned PyTorch/torchvision and a content-verified VGGT checkpoint.
    """.replace("__BUNDLE_NAME__", bundle_name))
    cell("code", """import hashlib, json, shutil, subprocess, sys, zipfile
    from pathlib import Path, PurePosixPath
    from google.colab import drive
    from IPython.display import display, Image as DisplayImage

    EXPECTED_PROJECT_FILES = __EXPECTED_PROJECT_FILES__

    def verify_active_project():
        changed = []
        for relative, expected in EXPECTED_PROJECT_FILES.items():
            path = PROJECT / relative
            if not path.is_file():
                changed.append(relative)
                continue
            with path.open('rb') as handle:
                if hashlib.file_digest(handle, 'sha256').hexdigest() != expected:
                    changed.append(relative)
        if changed:
            raise RuntimeError('Notebook/bundle code mismatch: ' + ', '.join(changed) +
                               '. Upload the ZIP paired with this notebook and extract it in a fresh runtime. '
                               'Read saved diagnostics before leaving the current runtime.')
        print('Active project matches this notebook:', len(EXPECTED_PROJECT_FILES), 'verified files')

    # Edit this folder only if you put the ZIP elsewhere in My Drive.
    DRIVE_FOLDER = Path('/content/drive/MyDrive/Room Reconstruction v2')
    if not Path('/content/drive/MyDrive').is_dir():
        drive.mount('/content/drive')  # Complete the extension's Google authorization prompt.
    drive_bundle = DRIVE_FOLDER / __BUNDLE_NAME_LITERAL__
    if not drive_bundle.is_file():
        raise FileNotFoundError(f'Put the prepared ZIP here, or edit DRIVE_FOLDER: {drive_bundle}')
    BUNDLE_ROOT = Path('/content/roomdelta3d_v2_bundle')
    if BUNDLE_ROOT.exists():
        raise RuntimeError('Bundle directory already exists. Reuse the subsequent cells on this runtime, or start a fresh runtime.')
    bundle_path = Path('/content/roomdelta3d_v2_colab_ready.zip')
    shutil.copyfile(drive_bundle, bundle_path)
    BUNDLE_ROOT.mkdir()
    with zipfile.ZipFile(bundle_path) as bundle:
        seen = set()
        for info in bundle.infolist():
            name = info.filename
            parts = PurePosixPath(name)
            if name in seen or parts.is_absolute() or '..' in parts.parts or '\\\\' in name or ':' in name:
                raise ValueError('Unsafe/duplicate bundle member')
            seen.add(name)
            if ((info.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError('Symlink bundle member')
            if not info.is_dir():
                target = (BUNDLE_ROOT / name).resolve()
                if not target.is_relative_to(BUNDLE_ROOT):
                    raise ValueError('Member escapes bundle root')
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(info) as source, target.open('wb') as destination:
                    shutil.copyfileobj(source, destination)
    receipt = json.loads((BUNDLE_ROOT / 'bundle_receipt.json').read_text())
    actual = {p.relative_to(BUNDLE_ROOT).as_posix() for p in BUNDLE_ROOT.rglob('*') if p.is_file()}
    assert actual == set(receipt['files']) | {'bundle_receipt.json'}
    for name, expected in receipt['files'].items():
        with (BUNDLE_ROOT / name).open('rb') as handle:
            assert hashlib.file_digest(handle, 'sha256').hexdigest() == expected, name
    PROJECT = BUNDLE_ROOT / 'project'
    WORKSPACE = Path('/content/roomdelta3d_v2_workspace')
    RUN = WORKSPACE / 'reconstruction'
    verify_active_project()
    print('Verified Drive bundle:', drive_bundle, 'Files:', len(receipt['files']))
    """.replace("__EXPECTED_PROJECT_FILES__", repr(expected_files)).replace("__BUNDLE_NAME_LITERAL__", repr(bundle_name)))
    cell("markdown", """## Install pinned runtime and checkpoint
    The checkpoint download is approximately 5 GB. This cell records package versions and verifies the checkpoint SHA-256.
    The system Python/driver/GPU assignment is recorded rather than assumed reproducible by Colab.
    """)
    cell("code", """import os
    from collections import deque

    def run_logged(command, label):
        # Pipe child output explicitly so VS Code receives it through notebook stdout.
        log_directory = WORKSPACE / 'logs'
        log_directory.mkdir(parents=True, exist_ok=True)
        log_path = log_directory / f'{label}.log'
        tail = deque(maxlen=60)
        print(f'{label}: starting; log: {log_path}', flush=True)
        with log_path.open('a', encoding='utf-8') as log:
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, bufsize=1, env=dict(os.environ, PYTHONUNBUFFERED='1')) as process:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                    tail.append(line)
                returncode = process.wait()
        if returncode:
            raise RuntimeError(f'{label} failed (exit {returncode}). Log: {log_path}\\nLast output:\\n' + ''.join(tail))
        print(f'{label}: completed', flush=True)

    verify_active_project()
    run_logged([sys.executable, str(PROJECT / 'scripts/setup_colab.py'), '--workspace', str(WORKSPACE)], 'setup')
    PYTHON = str(WORKSPACE / 'venv/bin/python')
    CLI = str(PROJECT / 'scripts/reconstruct.py')
    if not RUN.exists():
        run_logged([PYTHON, CLI, 'import', '--archive', str(BUNDLE_ROOT / 'prepared_run.zip'), '--output', str(RUN)], 'import')
    print('Prepared run:', RUN)
    """)
    cell("markdown", """## Measured GPU pilots
    Run 4, 8, then 16 representative frames per capture in fresh, separate workers.
    Each worker checks the saved model tensors against the exact upstream loader before inference.
    Attention is evaluated in bounded query slices while retaining all frame keys/values. Runtime records include this policy and memory measurements.
    Inspect failures/logs instead of silently reducing frame count or resolution. A passed pilot does not guarantee the full run will fit.
    If a pilot has already failed in this runtime, run the next diagnostic cell to inspect its saved worker exceptions without rerunning inference.
    """)
    cell("code", """# Read-only diagnostics: safe to run after a failure, including with an older bundle.
    from collections import deque

    def show_pilot_diagnostics():
        def read_saved_json(path):
            try:
                return json.loads(path.read_text(encoding='utf-8'))
            except (OSError, ValueError) as error:
                print(f'Cannot read {path}: {error}')
                return {}

        report_path = RUN / 'pilot_report.json'
        report = read_saved_json(report_path) if report_path.is_file() else {}
        print('Saved pilot report:', report.get('status', 'not available'), report_path)
        pilots = report.get('pilots', [])
        if not pilots:
            # Workers may have failed before the parent wrote its report.
            pilots = [{'requested_frames': int(path.name[1:]), 'captures': {}}
                      for path in sorted((WORKSPACE / 'pilots').glob('p[0-9][0-9]'))]
        for pilot in pilots:
            count = pilot['requested_frames']
            target = WORKSPACE / 'pilots' / f'p{count:02d}'
            for capture in ('before', 'after'):
                measurement = pilot.get('captures', {}).get(capture, {})
                env_path = target / capture / 'inference_environment.json'
                env = measurement.get('environment') or (read_saved_json(env_path) if env_path.is_file() else {})
                status = measurement.get('status', env.get('status', 'unknown'))
                def gib(key):
                    value = env.get(key)
                    return round(value / 1024**3, 3) if value is not None else 'unavailable'
                print(count, capture, status, 'allocated GiB:', gib('cuda_peak_allocated_bytes'),
                      'reserved GiB:', gib('cuda_peak_reserved_bytes'), 'inference seconds:', env.get('inference_sec'))
                print('Stage:', env.get('stage', 'unknown'), 'attention:', env.get('attention', 'not recorded'))
                if status == 'completed':
                    continue
                print('Worker error:', env.get('error') or measurement.get('worker_error')
                      or measurement.get('error') or 'No structured worker error; inspect log below.')
                log_path = target / 'logs' / f'{capture}.log'
                print('Worker log:', log_path)
                try:
                    with log_path.open(encoding='utf-8', errors='replace') as handle:
                        tail = ''.join(deque(handle, maxlen=60))
                except OSError as error:
                    tail = measurement.get('worker_log_tail') or f'Worker log unavailable: {error}'
                print(tail or '(empty worker log)')

    show_pilot_diagnostics()
    """)
    cell("code", """verify_active_project()
    try:
        run_logged([PYTHON, str(PROJECT / 'scripts/colab_run.py'), '--run', str(RUN), '--workspace', str(WORKSPACE), '--phase', 'pilot'], 'pilot')
    finally:
        show_pilot_diagnostics()
    for name in ('before', 'after'):
        panels = sorted((WORKSPACE / 'pilots/p04' / name / 'previews').glob('*_panel.png'))
        if panels:
            print(name, 'actual 4-frame pilot prediction; independent pilot coordinate/scale frame')
            display(DisplayImage(filename=str(panels[0]), width=1100))
        else:
            print(name, '4-frame pilot preview unavailable; inspect worker diagnostics above.')
    """)
    cell("markdown", """## Full independent predictions
    Process all 32 Before and 30 After selected frames separately. Save depth/confidence/masks, cameras, colored PLY/numerical clouds, all point observations, and quality reports.
    `--resume` reuses only hash-compatible, structurally verified captures. Never use pilot geometry as full-run geometry.
    """)
    cell("code", """verify_active_project()
    run_logged([PYTHON, str(PROJECT / 'scripts/colab_run.py'), '--run', str(RUN), '--workspace', str(WORKSPACE), '--phase', 'full'], 'full')
    run_logged([PYTHON, CLI, 'verify', '--run', str(RUN)], 'verify')
    for name in ('before', 'after'):
        report = json.loads((RUN / name / 'quality_report.json').read_text())
        print(name, 'points:', report['cloud_points'], 'internal consistency:', report['internal_consistency'],
              'real-scene accuracy:', report['real_scene_accuracy'])
        panels = sorted((RUN / name / 'previews').glob('*_panel.png'))
        display(DisplayImage(filename=str(panels[0]), width=1100))
    """)
    cell("markdown", """## Save portable results to Drive
    The ZIP contains source/model images, depth arrays, camera records, cloud/provenance and reports; it excludes the checkpoint and environment installation.
    This cell copies the completed archive directly beside the input ZIP in Drive and verifies its SHA-256 after copying.
    Download that ZIP from Drive, import it locally using `scripts/reconstruct.py import`, then run `verify`, `preview`, or `view`.
    Drive persistence occurs after this cell finishes; interrupted inference stays in the current runtime until explicitly saved.
    """)
    cell("code", """RESULTS = WORKSPACE / 'roomdelta3d_v2_reconstruction.zip'
    if RESULTS.exists():
        raise RuntimeError('Results archive already exists; choose a new RESULTS filename to preserve it.')
    run_logged([PYTHON, CLI, 'pack', '--run', str(RUN), '--archive', str(RESULTS)], 'pack')
    if not Path('/content/drive/MyDrive').is_dir():
        raise RuntimeError('Drive is no longer mounted; remount it before copying the local results archive.')
    saved_directory = DRIVE_FOLDER
    saved_archive = saved_directory / RESULTS.name
    staging_archive = saved_archive.with_suffix('.zip.partial')
    receipt_path = saved_archive.with_suffix('.receipt.json')
    if any(path.exists() for path in (saved_archive, staging_archive, receipt_path)):
        raise RuntimeError(f'Drive result or receipt already exists; choose a new RESULTS filename to preserve it: {saved_archive}')
    shutil.copyfile(RESULTS, staging_archive)
    with RESULTS.open('rb') as handle:
        expected = hashlib.file_digest(handle, 'sha256').hexdigest()
    with staging_archive.open('rb') as handle:
        actual = hashlib.file_digest(handle, 'sha256').hexdigest()
    if actual != expected:
        raise RuntimeError('Drive copy failed SHA-256 verification; local results remain available.')
    staging_archive.rename(saved_archive)
    save_receipt = {'archive': str(saved_archive), 'sha256': actual, 'bytes': saved_archive.stat().st_size}
    receipt_path.write_text(json.dumps(save_receipt, indent=2) + '\\n')
    print('Saved verified results to Drive:', saved_archive)
    print('SHA-256:', actual)
    """)
    return {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
            "language_info": {"name": "python"}, "colab": {"name": "reconstruction_v2_colab.ipynb"}}, "cells": cells}


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=project / "outputs/reconstruction")
    parser.add_argument("--output", type=Path, default=project / "outputs/roomdelta3d_v2_colab_ready.zip")
    parser.add_argument("--notebook-only", action="store_true", help="Refresh notebook without rebuilding or replacing the prepared ZIP")
    args = parser.parse_args()
    if args.output.exists() and not args.notebook_only:
        parser.error("Choose a new --output path to preserve existing bundles")
    notebooks = project / "notebooks"
    notebooks.mkdir(exist_ok=True)
    nb = notebook(args.output.name)
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")
    write_json(notebooks / "reconstruction_v2_colab.ipynb", nb)
    if args.notebook_only:
        print(f"Updated notebook: {notebooks / 'reconstruction_v2_colab.ipynb'}")
        return
    allowed = project_files(project)
    scratch = project / "outputs/.bundle_work"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="colab-", dir=scratch) as temporary:
        temporary = Path(temporary)
        prepared_zip = temporary / "prepared_run.zip"
        pack(args.run, prepared_zip)
        names = {"project/" + p.relative_to(project).as_posix(): p for p in allowed}
        names["prepared_run.zip"] = prepared_zip
        receipt = {"files": {name: digest(path) for name, path in names.items()},
                   "model_lock_identity": digest(project / "configs/models.lock.json"), "frame_counts": {n: r["frame_count"] for n, r in read_json(args.run / "run_manifest.json")["captures"].items()}}
        write_json(temporary / "bundle_receipt.json", receipt)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        staging = args.output.with_suffix(".zip.tmp")
        with zipfile.ZipFile(staging, "w", compression=zipfile.ZIP_STORED) as bundle:
            for name, path in names.items():
                bundle.write(path, name)
            bundle.write(temporary / "bundle_receipt.json", "bundle_receipt.json")
        staging.replace(args.output)
    write_json(args.output.with_suffix(".receipt.json"), {"bundle": str(args.output.resolve()), "sha256": digest(args.output),
                                                        "bytes": args.output.stat().st_size, "included_files": list(names), "notebook": str(notebooks / "reconstruction_v2_colab.ipynb")})
    print(json.dumps({"bundle": str(args.output), "megabytes": round(args.output.stat().st_size / 1024**2, 2), "notebook": str(notebooks / "reconstruction_v2_colab.ipynb")}, indent=2))


if __name__ == "__main__":
    main()
