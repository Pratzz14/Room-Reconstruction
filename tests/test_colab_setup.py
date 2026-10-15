import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import setup_colab


class ColabSetupTests(unittest.TestCase):
    def temporary_directory(self):
        scratch = Path(__file__).resolve().parents[1] / "outputs/.test_work"
        scratch.mkdir(parents=True, exist_ok=True)
        return tempfile.TemporaryDirectory(prefix="pip-repair-", dir=scratch)

    def test_existing_pip_skips_bootstrap(self):
        with patch.object(setup_colab, "has_pip", return_value=True), \
                patch.object(setup_colab.subprocess, "run") as run, \
                patch.object(setup_colab.urllib.request, "urlopen") as download:
            setup_colab.ensure_pip(Path("venv/bin/python"), Path("unused"))
        run.assert_not_called()
        download.assert_not_called()

    def test_real_pipless_environment_is_repaired_without_deletion(self):
        with self.temporary_directory() as temporary:
            workspace = Path(temporary)
            environment = workspace / "venv"
            subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(environment)], check=True)
            python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            marker = environment / "preserve-me.txt"
            marker.write_text("existing environment state")
            self.assertFalse(setup_colab.has_pip(python))
            # Child ensurepip uses its own tempfile module; keep wheel extraction in the workspace.
            with patch.dict(os.environ, {name: str(workspace) for name in ("TEMP", "TMP", "TMPDIR")}):
                setup_colab.ensure_pip(python, workspace)
            self.assertTrue(setup_colab.has_pip(python))
            self.assertEqual(marker.read_text(), "existing environment state")

    def test_missing_ensurepip_falls_back_to_recorded_pypa_bootstrap(self):
        with self.temporary_directory() as temporary:
            workspace = Path(temporary)
            python = workspace / "venv/bin/python"
            data = b"# synthetic bootstrap for mocked execution\n"
            response = subprocess.CompletedProcess([], 0, stdout="mock command output\n")
            with patch.object(setup_colab, "has_pip", side_effect=[False, False, True]), \
                    patch.object(setup_colab.subprocess, "run", return_value=response) as run, \
                    patch.object(setup_colab.urllib.request, "urlopen", return_value=io.BytesIO(data)) as download:
                setup_colab.ensure_pip(python, workspace)
            download.assert_called_once_with(setup_colab.PIP_BOOTSTRAP_URL, timeout=120)
            args = run.call_args_list[-1].args[0]
            self.assertEqual(args[0], str(python))
            self.assertIn(setup_colab.PIP_REQUIREMENT, args)
            receipt = json.loads((workspace / "pip_bootstrap.json").read_text())
            self.assertEqual(receipt["sha256"], hashlib.sha256(data).hexdigest())

    def test_bootstrap_must_actually_make_pip_available(self):
        with self.temporary_directory() as temporary:
            workspace = Path(temporary)
            response = subprocess.CompletedProcess([], 0, stdout="\n")
            with patch.object(setup_colab, "has_pip", return_value=False), \
                    patch.object(setup_colab.subprocess, "run", return_value=response), \
                    patch.object(setup_colab.urllib.request, "urlopen", return_value=io.BytesIO(b"mock")):
                with self.assertRaisesRegex(RuntimeError, "still cannot run pip"):
                    setup_colab.ensure_pip(workspace / "venv/bin/python", workspace)


if __name__ == "__main__":
    unittest.main()
