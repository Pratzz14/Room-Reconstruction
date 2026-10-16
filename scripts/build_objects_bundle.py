"""Build a self-contained object Colab bundle from an existing completed reconstruction."""

import argparse
import hashlib
import json
import sys
import tempfile
import textwrap
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reconstruction.core import (
    digest,
    read_json,
    safe_path,
    verify_capture,
    write_json,
)

PROJECT = Path(__file__).resolve().parents[1]


def notebook(bundle_name, output):
    cells = []

    def cell(kind, source):
        value = {
            "cell_type": kind,
            "metadata": {},
            "source": textwrap.dedent(source).strip() + "\n",
        }
        if kind == "code":
            value.update(execution_count=None, outputs=[])
        cells.append(value)

    cell(
        "markdown",
        """
        # Object inventories
        Use a GPU Colab runtime (browser or VS Code extension). Upload the object bundle to
        **My Drive / Room Reconstruction**. It contains saved reconstruction evidence,
        source code and settings; no original videos, credentials or checkpoint weights.
        Before and After remain independent. No VGGT rerun, alignment, change classification or VLM.
        Review the pilot mask overlays before running the full stage.
    """,
    )
    cell(
        "code",
        """
        from pathlib import Path
        import hashlib, json, os, shutil, subprocess, sys, zipfile, venv
        from google.colab import drive
        DRIVE = Path('/content/drive/MyDrive/Room Reconstruction')
        if not Path('/content/drive/MyDrive').is_dir():
            drive.mount('/content/drive')
        INPUT = DRIVE / BUNDLE_NAME
        WORK = Path('/content/roomdelta3d_objects')
        PROJECT = WORK / 'project'
        if not INPUT.is_file():
            raise FileNotFoundError(f'Upload the object bundle: {INPUT}')
        if not WORK.exists():
            # Bootstrap extraction: reject Windows and POSIX escaping/duplicate/symlink names.
            WORK.mkdir()
            with zipfile.ZipFile(INPUT) as archive:
                seen = set()
                for item in archive.infolist():
                    path = Path(item.filename)
                    target = (WORK / path).resolve()
                    key = item.filename.casefold().rstrip('/')
                    if (key in seen or not target.is_relative_to(WORK) or target == WORK
                        or '..' in path.parts or '\\\\' in item.filename or ':' in item.filename
                        or ((item.external_attr >> 16) & 0o170000) == 0o120000):
                        raise ValueError('Unsafe bundle member')
                    seen.add(key)
                    if not item.is_dir():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(item) as src, target.open('xb') as dst:
                            shutil.copyfileobj(src, dst)
        receipt = json.loads((WORK / 'archive_receipt.json').read_text())
        actual = {p.relative_to(WORK).as_posix() for p in WORK.rglob('*') if p.is_file()
                  and not p.is_relative_to(WORK / 'venv') and not p.is_relative_to(WORK / 'results')
                  and '__pycache__' not in p.parts and not p.is_relative_to(WORK / 'logs')
                  and not p.is_relative_to(WORK / 'reconstruction')}
        # Verify immutable input files on every rerun; runtime/output folders are separate.
        for name, expected in receipt['files'].items():
            with (WORK / name).open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                    raise ValueError(f'Bundle content changed: {name}')
        print('Verified object bundle:', INPUT, len(receipt['files']), 'files')
    """.replace("BUNDLE_NAME", repr(bundle_name)),
    )
    cell(
        "code",
        """
        from collections import deque
        def execute(args, name):
            logs = WORK / 'logs'; logs.mkdir(exist_ok=True)
            tail = deque(maxlen=60)
            with (logs / (name + '.log')).open('a') as log:
                with subprocess.Popen([str(x) for x in args], stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, bufsize=1,
                                      env=dict(os.environ, PYTHONUNBUFFERED='1')) as process:
                    for line in process.stdout:
                        print(line, end='', flush=True); log.write(line); log.flush(); tail.append(line)
                    code = process.wait()
            if code:
                raise RuntimeError(f'{name} failed, exit {code}:\\n' + ''.join(tail))
        ENV = WORK / 'venv'; PYTHON = ENV / 'bin/python'
        if not PYTHON.exists():
            venv.EnvBuilder(with_pip=False).create(ENV)
        sys.path.insert(0, str(PROJECT))
        from scripts.setup_colab import ensure_pip
        ensure_pip(PYTHON, WORK)
        execute([PYTHON, '-m', 'pip', 'install', '--upgrade', 'pip==25.2'], 'pip-setup')
        execute([PYTHON, '-m', 'pip', 'install', 'torch==2.8.0', 'torchvision==0.23.0',
                 '--index-url', 'https://download.pytorch.org/whl/cu126'], 'torch-setup')
        execute([PYTHON, '-m', 'pip', 'install', '-r', PROJECT / 'requirements-objects.txt'], 'objects-setup')
        execute([PYTHON, '-c', 'import torch; assert torch.cuda.is_available(), "Select a GPU runtime"; print(torch.__version__, torch.cuda.get_device_name())'], 'gpu-check')
        RUN = WORK / 'reconstruction'; RESULTS = WORK / 'results'; RESULTS.mkdir(exist_ok=True)
        CLI = PROJECT / 'scripts/objects.py'
        if not RUN.exists():
            execute([PYTHON, PROJECT / 'scripts/reconstruct.py', 'import', '--archive', WORK / 'reconstruction.zip', '--output', RUN], 'reconstruction-import')
        LOCK = RESULTS / 'models.resolved.json'
        if not LOCK.exists():
            execute([PYTHON, CLI, 'resolve-models', '--output', LOCK], 'resolve-models')
        print('Frozen model lock:', LOCK)
    """,
    )
    cell(
        "markdown",
        """
        ## Pilot: three selected frames per capture
        The pilot processes first/middle/last selected frames, not a new reconstruction.
        Review masks, labels, original pixels, geometry and support. It is intentionally
        partial coverage and does not pass full inventory acceptance.
    """,
    )
    cell(
        "code",
        """
        PILOT = RESULTS / 'objects_pilot'
        execute([PYTHON, CLI, 'run', '--reconstruction', RUN, '--output', PILOT,
                 '--lock', LOCK, '--frames-per-capture', '3', '--resume'], 'objects-pilot')
        execute([PYTHON, CLI, 'verify', '--reconstruction', RUN, '--objects', PILOT], 'pilot-verify')
        from IPython.display import display, Image
        for capture in ('before', 'after'):
            print(capture, 'pilot mask previews:')
            for image in sorted((PILOT / capture / 'previews').glob('*.jpg'))[:12]:
                display(Image(filename=str(image)))
    """,
    )
    cell(
        "markdown",
        """
        ## Full inventory
        Run after reviewing the pilot. All existing 32/30 selected frames are processed.
        Separate capture/stage caches allow compatible retries in this runtime.
        Notebook results remain in local runtime storage until the export cell succeeds.
    """,
    )
    cell(
        "code",
        """
        OBJECTS = RESULTS / 'objects_full'
        execute([PYTHON, CLI, 'run', '--reconstruction', RUN, '--output', OBJECTS,
                 '--lock', LOCK, '--resume'], 'objects-full')
        execute([PYTHON, CLI, 'verify', '--reconstruction', RUN, '--objects', OBJECTS], 'full-verify')
        print(json.loads((OBJECTS / 'manifest.json').read_text())['status'])
        print('Computational checks passed. Real masks/identities still require visual acceptance.')
    """,
    )
    cell(
        "code",
        """
        from datetime import datetime, timezone
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        ARCHIVE = RESULTS / f'roomdelta3d_objects_{stamp}.zip'
        execute([PYTHON, CLI, 'pack', '--reconstruction', RUN, '--objects', OBJECTS, '--archive', ARCHIVE], 'export')
        destination = DRIVE / ARCHIVE.name
        if destination.exists():
            raise FileExistsError(f'Preserve existing result: {destination}')
        temporary = destination.with_suffix('.zip.copying')
        shutil.copyfile(ARCHIVE, temporary)
        def sha(path):
            with path.open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()
        expected = sha(ARCHIVE)
        if sha(temporary) != expected:
            raise ValueError('Drive copy hash mismatch')
        temporary.rename(destination)
        destination.with_suffix('.receipt.json').write_text(json.dumps({'sha256': expected, 'bytes': destination.stat().st_size, 'real_review': 'pending'}, indent=2))
        print('Saved beside input bundle:', destination)
        print('Download, import alongside the SAME reconstruction, then inspect locally with --objects.')
    """,
    )
    document = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    write_json(output, document)
    return document


def build(reconstruction, output, notebook_path):
    reconstruction, output = Path(reconstruction).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(reconstruction):
        raise ValueError("Choose a new bundle path outside the reconstruction")
    manifest = read_json(reconstruction / "run_manifest.json")
    for capture in ("before", "after"):
        verify_capture(reconstruction, capture)
    output.parent.mkdir(parents=True, exist_ok=True)
    files = {"run_manifest.json", "config.json", "models.lock.json"}
    for record in manifest["captures"].values():
        files.update(record["artifacts"])
    # Build the reconstruction archive receipt in the ZIP, never in the source run.
    with tempfile.TemporaryDirectory(
        prefix="objects-bundle-", dir=output.parent
    ) as temporary:
        embedded = Path(temporary) / "reconstruction.zip"
        receipt = {
            "schema_version": "1.0",
            "files": {
                name: digest(safe_path(reconstruction, name)) for name in sorted(files)
            },
        }
        with zipfile.ZipFile(
            embedded, "w", zipfile.ZIP_DEFLATED, compresslevel=1
        ) as archive:
            for name in sorted(files):
                archive.write(safe_path(reconstruction, name), name)
            archive.writestr("archive_receipt.json", json.dumps(receipt))
        sources = [
            *sorted((PROJECT / "reconstruction").glob("*.py")),
            *sorted((PROJECT / "reconstruction").glob("*.html")),
            PROJECT / "scripts/objects.py",
            PROJECT / "scripts/setup_colab.py",
            PROJECT / "scripts/reconstruct.py",
            PROJECT / "configs/objects.json",
            PROJECT / "configs/objects.models.lock.json",
            PROJECT / "requirements-objects.txt",
        ]
        hashes = {
            "project/" + p.relative_to(PROJECT).as_posix(): digest(p) for p in sources
        }
        hashes["reconstruction.zip"] = digest(embedded)
        temporary_zip = output.with_suffix(".zip.tmp")
        with zipfile.ZipFile(temporary_zip, "w", zipfile.ZIP_STORED) as archive:
            for source in sources:
                archive.write(
                    source, "project/" + source.relative_to(PROJECT).as_posix()
                )
            archive.write(embedded, "reconstruction.zip")
            archive.writestr(
                "archive_receipt.json",
                json.dumps({"schema_version": "1.0", "files": hashes}, indent=2),
            )
        temporary_zip.replace(output)
    notebook(output.name, notebook_path)
    result = {
        "bundle": str(output),
        "sha256": digest(output),
        "bytes": output.stat().st_size,
        "source_files": len(sources),
        "notebook": str(notebook_path),
        "reconstruction_identity": hashlib.sha256(
            json.dumps(receipt, sort_keys=True).encode()
        ).hexdigest(),
    }
    write_json(output.with_suffix(".receipt.json"), result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconstruction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--notebook", type=Path, default=PROJECT / "notebooks/objects_colab.ipynb"
    )
    args = parser.parse_args()
    print(json.dumps(build(args.reconstruction, args.output, args.notebook), indent=2))
