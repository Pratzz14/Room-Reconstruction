"""Colab transport validation without installing models or changing source evidence."""

import tempfile
import unittest
import zipfile
from pathlib import Path

from reconstruction import core
from reconstruction.archive import import_archive
from reconstruction.object_archive import extract_verified
from scripts.build_objects_bundle import build
from tests.fixtures import PROJECT, completed


class ObjectBundleTests(unittest.TestCase):
    def test_bundle_import_hashes_and_notebook_compile(self):
        scratch = PROJECT / "outputs/.test_work"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="object-bundle-", dir=scratch
        ) as temporary:
            base = Path(temporary)
            reconstruction = completed(base / "reconstruction")
            original = {
                p: core.digest(p) for p in reconstruction.rglob("*") if p.is_file()
            }
            result = build(reconstruction, base / "bundle.zip", base / "objects.ipynb")
            self.assertGreater(result["source_files"], 10)
            with zipfile.ZipFile(base / "bundle.zip") as archive:
                self.assertFalse(
                    any(
                        name.endswith((".mp4", ".safetensors"))
                        for name in archive.namelist()
                    )
                )
                self.assertIn("project/scripts/setup_colab.py", archive.namelist())
            unpacked = base / "bundle"
            unpacked.mkdir()
            extract_verified(base / "bundle.zip", unpacked)
            import_archive(unpacked / "reconstruction.zip", base / "imported")
            for capture in ("before", "after"):
                self.assertEqual(
                    core.verify_capture(base / "imported", capture)[
                        "internal_consistency"
                    ],
                    "passed",
                )
            document = core.read_json(base / "objects.ipynb")
            code_cells = [
                cell for cell in document["cells"] if cell["cell_type"] == "code"
            ]
            self.assertEqual(len(code_cells), 5)
            for i, cell in enumerate(code_cells):
                compile(cell["source"], f"cell_{i}", "exec")
                self.assertEqual(cell["outputs"], [])
            self.assertEqual(original, {p: core.digest(p) for p in original})


if __name__ == "__main__":
    unittest.main()
