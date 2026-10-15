"""Create an isolated pinned Colab environment and fetch the pinned checkpoint."""

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path


PIP_REQUIREMENT = "pip==25.2"
PIP_BOOTSTRAP_URL = "https://bootstrap.pypa.io/get-pip.py"


def has_pip(python):
    return subprocess.run([str(python), "-m", "pip", "--version"],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).returncode == 0


def ensure_pip(python, workspace):
    """Repair an existing pip-less environment without deleting installed artifacts."""
    if has_pip(python):
        return
    print("Virtual environment has no pip; trying ensurepip", flush=True)
    attempt = subprocess.run([str(python), "-m", "ensurepip", "--upgrade"],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(attempt.stdout, end="", flush=True)
    if has_pip(python):
        return
    print("ensurepip did not provide pip; using the official PyPA bootstrap in the virtual environment", flush=True)
    bootstrap = workspace / "get-pip.py"
    with urllib.request.urlopen(PIP_BOOTSTRAP_URL, timeout=120) as response:
        data = response.read(10 * 1024 * 1024 + 1)
    if len(data) > 10 * 1024 * 1024:
        raise RuntimeError("Unexpectedly large pip bootstrap download")
    bootstrap.write_bytes(data)
    (workspace / "pip_bootstrap.json").write_text(json.dumps({
        "url": PIP_BOOTSTRAP_URL, "sha256": hashlib.sha256(data).hexdigest(),
        "target_python": str(python), "pip_requirement": PIP_REQUIREMENT,
    }, indent=2) + "\n")
    subprocess.run([str(python), str(bootstrap), "--no-setuptools", "--no-wheel", PIP_REQUIREMENT], check=True)
    if not has_pip(python):
        raise RuntimeError("Pip bootstrap completed but the virtual environment still cannot run pip")


def runtime_preflight(python, source, lock_path, workspace):
    """Check the actual worker imports and a CUDA operation before fetching weights."""
    probe = """import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import torch, torchvision, numpy, psutil, safetensors
from PIL import Image
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
lock = json.loads(Path(sys.argv[2]).read_text())
if torch.__version__.split('+')[0] != lock['torch'] or torchvision.__version__.split('+')[0] != lock['torchvision']:
    raise RuntimeError('Installed PyTorch/torchvision differ from the model lock')
if not torch.cuda.is_available():
    raise RuntimeError('Select a Colab GPU runtime')
with torch.inference_mode():
    q = torch.zeros((1, 16, 32, 64), device='cuda', dtype=torch.float16)
    torch.nn.functional.scaled_dot_product_attention(q, q, q).float().cpu().numpy()
torch.cuda.synchronize()
result = {'status': 'passed', 'scope': 'worker imports and small CUDA attention operation; no model inference',
          'torch': torch.__version__, 'torchvision': torchvision.__version__, 'cuda': torch.version.cuda,
          'gpu': torch.cuda.get_device_name(0), 'gpu_total_bytes': torch.cuda.get_device_properties(0).total_memory,
          'gpu_free_bytes': torch.cuda.mem_get_info()[0]}
Path(sys.argv[3]).write_text(json.dumps(result, indent=2) + '\\n')
print(json.dumps(result, indent=2))
"""
    subprocess.run([str(python), "-c", probe, str(source), str(lock_path),
                    str(workspace / "runtime_preflight.json")], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    if not (3, 11) <= sys.version_info[:2] <= (3, 13):
        raise RuntimeError("This pinned wheel set requires Python 3.11 through 3.13")
    lock = json.loads((project / "configs/models.lock.json").read_text())
    environment = workspace / "venv"
    python = environment / "bin/python"
    if sys.platform != "linux":
        raise RuntimeError("Run setup_colab.py in a Linux Colab GPU runtime")
    if not python.is_file():
        # Redistributor Python builds may omit ensurepip; bootstrap pip explicitly below.
        subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(environment)], check=True)
    ensure_pip(python, workspace)
    subprocess.run([str(python), "-m", "pip", "install", "--upgrade", PIP_REQUIREMENT], check=True)
    subprocess.run([str(python), "-m", "pip", "install", f"torch=={lock['torch']}", f"torchvision=={lock['torchvision']}",
                    "--index-url", lock["torch_wheel_index"]], check=True)
    subprocess.run([str(python), "-m", "pip", "install", "-r", str(project / "requirements-reconstruction.txt")], check=True)
    source = workspace / "vggt"
    if not source.exists():
        subprocess.run(["git", "clone", lock["source_repository"], str(source)], check=True)
    current = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if current != lock["source_revision"]:
        if subprocess.run(["git", "-C", str(source), "diff", "--quiet"], check=False).returncode:
            raise RuntimeError("Source checkout has edits; use a new workspace")
        subprocess.run(["git", "-C", str(source), "checkout", "--detach", lock["source_revision"]], check=True)
    if subprocess.run(["git", "-C", str(source), "diff", "--quiet", "HEAD", "--", "vggt"], check=False).returncode:
        raise RuntimeError("VGGT source has edits; use a clean source checkout")
    runtime_preflight(python, source, project / "configs/models.lock.json", workspace)
    download = """import json,hashlib,sys
from pathlib import Path
from huggingface_hub import hf_hub_download
lock=json.loads(Path(sys.argv[1]).read_text())
path=Path(hf_hub_download(repo_id=lock['checkpoint_repository'],filename=lock['checkpoint_filename'],revision=lock['checkpoint_revision'],cache_dir=sys.argv[2]))
with path.open('rb') as handle: digest=hashlib.file_digest(handle,'sha256').hexdigest()
if digest!=lock['checkpoint_sha256'] or path.stat().st_size!=lock['checkpoint_bytes']: raise ValueError('Checkpoint identity mismatch')
Path(sys.argv[3]).write_text(json.dumps({'checkpoint':str(path.resolve()),'sha256':digest},indent=2)+'\\n')
print('Pinned checkpoint verified:',path)
"""
    subprocess.run([str(python), "-c", download, str(project / "configs/models.lock.json"), str(workspace / "weights_cache"), str(workspace / "checkpoint.json")], check=True)
    with (workspace / "pip_freeze.txt").open("w") as handle:
        subprocess.run([str(python), "-m", "pip", "freeze", "--all"], stdout=handle, check=True)
    print(json.dumps({"python": str(python), "source": str(source), "checkpoint_record": str(workspace / "checkpoint.json")}, indent=2))


if __name__ == "__main__":
    main()
