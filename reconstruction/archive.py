"""Portable ZIP transport with safe member paths and content receipts."""

import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from .core import check_artifacts, digest, identity, read_json, safe_path, verify_capture, write_json


def pack(root, archive):
    root, archive = Path(root).resolve(), Path(archive).resolve()
    if archive.exists() or archive.is_relative_to(root):
        raise ValueError("Archive must be a new path outside the run directory")
    if (root / ".run.lock").exists():
        raise ValueError("Cannot package a run while a reconstruction writer owns its lock")
    manifest = read_json(root / "run_manifest.json")
    if identity(read_json(root / "config.json")) != manifest["config_identity"] or identity(read_json(root / "models.lock.json")) != manifest["model_identity"]:
        raise ValueError("Run configuration/model identity mismatch")
    for capture, record in manifest["captures"].items():
        if record["status"] == "completed":
            verify_capture(root, capture)
        elif record["status"] == "prepared":
            check_artifacts(root, record)
    files = [p for p in sorted(root.rglob("*")) if p.is_file() and p.name != "archive_receipt.json"]
    if any(p.is_symlink() for p in files):
        raise ValueError("Symlink artifacts cannot be packaged")
    receipt = {"schema_version": "1.0", "files": {p.relative_to(root).as_posix(): digest(p) for p in files}}
    write_json(root / "archive_receipt.json", receipt)
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
        for path in files + [root / "archive_receipt.json"]:
            bundle.write(path, path.relative_to(root).as_posix())
    temporary.replace(archive)
    return {"path": str(archive), "sha256": digest(archive), "bytes": archive.stat().st_size}


def import_archive(archive, output):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Import requires a new destination")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".reconstruction-import-", dir=output.parent) as temporary:
        staging = Path(temporary)
        with zipfile.ZipFile(archive) as bundle:
            names = set()
            for info in bundle.infolist():
                name = info.filename
                posix = PurePosixPath(name)
                if name in names or posix.is_absolute() or ".." in posix.parts or "\\" in name or ":" in name:
                    raise ValueError("Duplicate or unsafe archive member")
                names.add(name)
                if ((info.external_attr >> 16) & 0o170000) == 0o120000:
                    raise ValueError("Symlink archive member")
                if not info.is_dir():
                    target = safe_path(staging, name)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.open(info) as source, target.open("wb") as destination:
                        shutil.copyfileobj(source, destination)
        receipt = read_json(staging / "archive_receipt.json")
        actual = {p.relative_to(staging).as_posix() for p in staging.rglob("*") if p.is_file()}
        if actual != set(receipt["files"]) | {"archive_receipt.json"}:
            raise ValueError("Archive receipt does not match member set")
        for path, expected in receipt["files"].items():
            if digest(safe_path(staging, path)) != expected:
                raise ValueError(f"Archive artifact hash mismatch: {path}")
        manifest = read_json(staging / "run_manifest.json")
        if identity(read_json(staging / "config.json")) != manifest["config_identity"] or identity(read_json(staging / "models.lock.json")) != manifest["model_identity"]:
            raise ValueError("Imported configuration/model identity mismatch")
        for name, record in manifest["captures"].items():
            if record["status"] == "completed":
                verify_capture(staging, name)
            elif record["status"] == "prepared":
                check_artifacts(staging, record)
        # Destination and staging are resolved siblings in the explicitly chosen parent.
        staging.rename(output)
    return read_json(output / "run_manifest.json")
