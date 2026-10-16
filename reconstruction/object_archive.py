"""Verified companion archives; never modify a reconstruction directory."""

import json
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from .core import digest, read_json, safe_path
from .object_contracts import ObjectInventory, atomic_move


def pack(reconstruction, root, archive):
    root, archive = Path(root).resolve(), Path(archive).resolve()
    inventory = ObjectInventory(reconstruction, root)
    if not inventory.captures:
        raise ValueError("No completed object capture to export")
    if archive.exists() or archive.is_relative_to(root):
        raise ValueError("Choose a new archive path outside the object directory")
    files = {"manifest.json", "config.json", "models.lock.json"}
    for name in inventory.captures:
        files.update(inventory.manifest["captures"][name]["artifacts"])
    files.update(p.relative_to(root).as_posix() for p in (root / "logs").glob("*.json"))
    receipt = {
        "schema_version": "1.0",
        "files": {name: digest(safe_path(root, name)) for name in sorted(files)},
    }
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix(".zip.tmp")
    with zipfile.ZipFile(
        temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=1
    ) as bundle:
        for name in sorted(files):
            bundle.write(safe_path(root, name), name)
        bundle.writestr("archive_receipt.json", json.dumps(receipt, indent=2))
    atomic_move(temporary, archive)
    return {
        "path": str(archive),
        "sha256": digest(archive),
        "bytes": archive.stat().st_size,
    }


def extract_verified(archive, staging):
    """Extract an allowlisted receipt archive into an empty caller-owned staging tree."""
    staging = Path(staging).resolve()
    with zipfile.ZipFile(archive) as bundle:
        names = set()
        for info in bundle.infolist():
            name = info.filename
            path = PurePosixPath(name)
            normalized = name.casefold().rstrip("/")
            if (
                normalized in names
                or path.is_absolute()
                or ".." in path.parts
                or "\\" in name
                or ":" in name
                or any(part.endswith((".", " ")) for part in path.parts)
                or not path.parts
                or ((info.external_attr >> 16) & 0o170000) == 0o120000
            ):
                raise ValueError("Unsafe, duplicate, or symlink archive member")
            names.add(normalized)
            target = safe_path(staging, name)
            if not info.is_dir():
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(info) as source, target.open("xb") as destination:
                    shutil.copyfileobj(source, destination)
    receipt = read_json(staging / "archive_receipt.json")
    actual = {
        p.relative_to(staging).as_posix() for p in staging.rglob("*") if p.is_file()
    }
    if actual != set(receipt["files"]) | {"archive_receipt.json"}:
        raise ValueError("Archive member/receipt mismatch")
    for name, expected in receipt["files"].items():
        if digest(safe_path(staging, name)) != expected:
            raise ValueError(f"Archive hash mismatch: {name}")


def import_archive(archive, reconstruction, output):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Import requires a new destination")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".objects-import-", dir=output.parent
    ) as directory:
        staging = Path(directory)
        extract_verified(archive, staging)
        inventory = ObjectInventory(reconstruction, staging)
        if not inventory.captures:
            raise ValueError("Archive has no completed object capture")
        atomic_move(staging, output)
    return read_json(output / "manifest.json")
