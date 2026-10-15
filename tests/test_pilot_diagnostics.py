import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from scripts.colab_run import capture_measurement
from scripts.build_colab_bundle import notebook
from reconstruction_v2.core import write_json


class PilotDiagnosticsTests(unittest.TestCase):
    def temporary_directory(self):
        scratch = Path(__file__).resolve().parents[1] / "outputs/.test_work"
        scratch.mkdir(parents=True, exist_ok=True)
        return tempfile.TemporaryDirectory(prefix="worker-diagnostic-", dir=scratch)

    def test_failed_worker_retains_error_and_bounded_log_tail(self):
        with self.temporary_directory() as tmp:
            root = Path(tmp)
            (root / "logs").mkdir()
            (root / "before").mkdir()
            (root / "logs/before.log").write_text("".join(f"line {i}\n" for i in range(80)) + "CUDA out of memory\n")
            write_json(root / "before/inference_environment.json", {"error": "OutOfMemoryError: known failure"})
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = capture_measurement(root, "before", {"status": "failed", "frame_count": 16, "error": "GPU worker exited 1"})
            self.assertIn("OutOfMemoryError", output.getvalue())
            self.assertEqual(len(result["worker_log_tail"].splitlines()), 60)
            self.assertIn("CUDA out of memory", result["worker_log_tail"])

    def test_success_does_not_publish_an_old_failure_log(self):
        with self.temporary_directory() as tmp:
            result = capture_measurement(Path(tmp), "after", {"status": "completed", "frame_count": 8})
            self.assertNotIn("worker_log_tail", result)
            self.assertIsNone(result["error"])

    def notebook_cells(self):
        sources = ["".join(cell["source"]) for cell in notebook()["cells"] if cell["cell_type"] == "code"]
        diagnostics = next(source for source in sources if source.startswith("# Read-only diagnostics"))
        pilot = next(source for source in sources if source.startswith("verify_active_project()\ntry:\n"))
        return diagnostics, pilot

    def test_notebook_reads_old_worker_logs_and_preserves_runner_exception(self):
        with self.temporary_directory() as tmp:
            workspace = Path(tmp)
            run = workspace / "reconstruction"
            write_json(run / "pilot_report.json", {"status": "failed", "pilots": [
                {"requested_frames": 4, "captures": {
                    "before": {"status": "failed", "environment": None},
                    "after": {"status": "completed", "environment": {"cuda_peak_allocated_bytes": 1024**3}}}}]})
            log = workspace / "pilots/p04/logs/before.log"
            log.parent.mkdir(parents=True)
            log.write_text("".join(f"line {i}\n" for i in range(80)) + "ImportError: actual worker failure\n")
            original_error = RuntimeError("pilot wrapper failed")
            runner = Mock(side_effect=original_error)
            namespace = {"RUN": run, "WORKSPACE": workspace, "PROJECT": workspace,
                         "PYTHON": "unused", "json": json, "run_logged": runner, "verify_active_project": Mock()}
            diagnostics, pilot = self.notebook_cells()
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exec(diagnostics, namespace)
                with self.assertRaises(RuntimeError) as caught:
                    exec(pilot, namespace)
            self.assertIs(caught.exception, original_error)
            self.assertEqual(runner.call_count, 1)
            self.assertIn("ImportError: actual worker failure", output.getvalue())
            self.assertIn("allocated GiB: unavailable", output.getvalue())
            self.assertIn("allocated GiB: 1.0", output.getvalue())
            self.assertNotIn("line 0\n", output.getvalue())

    def test_notebook_finds_logs_without_a_report_and_tolerates_missing_logs(self):
        with self.temporary_directory() as tmp:
            workspace = Path(tmp)
            log = workspace / "pilots/p08/logs/before.log"
            log.parent.mkdir(parents=True)
            log.write_text("RuntimeError: failure before report\n")
            namespace = {"RUN": workspace / "reconstruction", "WORKSPACE": workspace, "json": json}
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exec(self.notebook_cells()[0], namespace)
            self.assertIn("failure before report", output.getvalue())
            self.assertIn("Worker log unavailable", output.getvalue())

    def test_notebook_success_does_not_index_missing_previews(self):
        with self.temporary_directory() as tmp:
            workspace = Path(tmp)
            namespace = {"RUN": workspace / "reconstruction", "WORKSPACE": workspace,
                         "PROJECT": workspace, "PYTHON": "unused", "json": json, "run_logged": Mock(),
                         "verify_active_project": Mock()}
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                diagnostics, pilot = self.notebook_cells()
                exec(diagnostics, namespace)
                exec(pilot, namespace)
            self.assertIn("preview unavailable", output.getvalue())


if __name__ == "__main__":
    unittest.main()
