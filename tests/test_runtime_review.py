import ast
import contextlib
import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from reconstruction import core, inference
from scripts import colab_run
from scripts.build_colab_bundle import notebook
from tests.fixtures import prepared, PROJECT


class RuntimeReviewTests(unittest.TestCase):
    def temporary_directory(self):
        scratch = PROJECT / "outputs/.test_work"
        scratch.mkdir(parents=True, exist_ok=True)
        return tempfile.TemporaryDirectory(prefix="runtime-review-", dir=scratch)

    def test_preflight_failure_is_saved_and_replaces_previous_error(self):
        with self.temporary_directory() as tmp:
            root = prepared(Path(tmp) / "run")
            core.write_json(root / "before/inference_environment.json", {"error": "old OOM", "cuda_peak_allocated_bytes": 900})
            with self.assertRaises(FileNotFoundError):
                inference.worker(root, "before", Path(tmp) / "source", Path(tmp) / "missing-checkpoint")
            record = core.read_json(root / "before/inference_environment.json")
            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["stage"], "preflight")
            self.assertIn("FileNotFoundError", record["error"])
            self.assertNotIn("cuda_peak_allocated_bytes", record)

    def test_export_failure_retains_stage_and_traceback(self):
        with self.temporary_directory() as tmp:
            root = Path(tmp)
            def failed_export(root, capture, source, checkpoint, runtime):
                runtime["stage"] = "export"
                raise ValueError("bad predicted camera")
            with patch.object(inference, "_worker", side_effect=failed_export):
                with self.assertRaisesRegex(ValueError, "bad predicted camera"):
                    inference.worker(root, "before", root, root)
            record = core.read_json(root / "before/inference_environment.json")
            self.assertEqual(record["stage"], "export")
            self.assertIn("bad predicted camera", record["traceback"])

    def test_worker_launch_failure_is_failed_and_does_not_reuse_old_measurements(self):
        with self.temporary_directory() as tmp:
            root = prepared(Path(tmp) / "run")
            core.write_json(root / "before/inference_environment.json", {"error": "old error"})
            with patch.object(inference.subprocess, "run", side_effect=OSError("cannot launch")), \
                    contextlib.redirect_stdout(io.StringIO()):
                result = inference.run(root, root, root, captures=("before",))
            self.assertEqual(result["captures"]["before"]["status"], "failed")
            self.assertIn("cannot launch", result["captures"]["before"]["error"])
            self.assertNotIn("error", core.read_json(root / "before/inference_environment.json"))

    def test_stale_or_partial_pilots_cannot_authorize_full_run(self):
        with self.temporary_directory() as tmp:
            root = prepared(Path(tmp) / "run")
            pilots = [{"requested_frames": count, "status": "completed", "captures": {
                name: {"status": "completed"} for name in ("before", "after")}} for count in (4, 8, 16)]
            report = {"status": "passed", "run_identity": colab_run.pilot_identity(root), "pilots": pilots}
            colab_run.validate_pilot_report(root, report)
            with patch.object(colab_run, "code_identity", return_value="changed worker"):
                with self.assertRaisesRegex(RuntimeError, "rerun the pilot"):
                    colab_run.validate_pilot_report(root, report)
            report["pilots"] = pilots[:1]
            with self.assertRaises(RuntimeError):
                colab_run.validate_pilot_report(root, report)

    def test_interrupted_later_pilot_does_not_leave_passed_report(self):
        with self.temporary_directory() as tmp:
            workspace = Path(tmp)
            root = prepared(workspace / "run")
            core.write_json(workspace / "checkpoint.json", {"checkpoint": "unused"})
            manifest = {"status": "completed", "captures": {n: {"status": "completed", "frame_count": 4}
                                                            for n in ("before", "after")}}
            args = ["colab_run.py", "--run", str(root), "--workspace", str(workspace), "--phase", "pilot"]
            with patch("sys.argv", args), patch.object(colab_run, "pilot_subset"), \
                    patch.object(colab_run, "run", side_effect=[manifest, RuntimeError("interrupted")]), \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    colab_run.main()
            self.assertEqual(core.read_json(root / "pilot_report.json")["status"], "running")

    def test_pilot_directory_cannot_reuse_different_config(self):
        with self.temporary_directory() as tmp:
            root = prepared(Path(tmp) / "run")
            destination = Path(tmp) / "pilot"
            colab_run.pilot_subset(root, destination, 4)
            config = core.read_json(root / "config.json")
            config["inference"]["precision"] = "float32"
            core.write_json(root / "config.json", config)
            manifest = core.read_json(root / "run_manifest.json")
            manifest["config_identity"] = core.identity(config)
            core.write_json(root / "run_manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "another configuration"):
                colab_run.pilot_subset(root, destination, 4)

    def test_notebook_rejects_old_or_changed_worker_before_running(self):
        # Execute the actual generated verification function without importing Colab.
        source = next("".join(c["source"]) for c in notebook("reviewed.zip")["cells"] if c["cell_type"] == "code")
        parsed = ast.parse(source)
        nodes = [n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == "verify_active_project"]
        with self.temporary_directory() as tmp:
            root = Path(tmp)
            worker = root / "worker.py"
            worker.write_text("current worker")
            namespace = {"PROJECT": root, "hashlib": hashlib, "EXPECTED_PROJECT_FILES": {"worker.py": core.digest(worker)}}
            exec(compile(ast.Module(body=nodes, type_ignores=[]), "notebook-check", "exec"), namespace)
            with contextlib.redirect_stdout(io.StringIO()):
                namespace["verify_active_project"]()
            worker.write_text("old worker")
            with self.assertRaisesRegex(RuntimeError, "Notebook/bundle code mismatch"):
                namespace["verify_active_project"]()
        self.assertIn("drive_bundle = DRIVE_FOLDER / 'reviewed.zip'", source)


if __name__ == "__main__":
    unittest.main()
