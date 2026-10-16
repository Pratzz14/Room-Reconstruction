"""Pinned VGGT GPU worker and independent capture orchestration."""

from __future__ import annotations

import contextlib
import json
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

from .core import (check_artifacts, code_identity, digest, environment, export_predictions,
                   identity, read_json, verify_capture, write_json)


@contextlib.contextmanager
def bounded_attention(torch, query_chunk_size=512):
    """Bound inference SDPA query rows, retaining every key/value and global view.

    VGGT uses unmasked, noncausal, zero-dropout attention at inference. Each query's
    softmax spans all keys, so independent query slices preserve that operation.
    This process-local wrapper is restored even if model inference fails.
    """
    if isinstance(query_chunk_size, bool) or not isinstance(query_chunk_size, int) or query_chunk_size < 1:
        raise ValueError("Attention query chunk size must be a positive integer")
    functional = torch.nn.functional
    original = functional.scaled_dot_product_attention
    stats = {"implementation": "bounded_query_sdpa", "query_chunk_size": query_chunk_size,
             "chunked_calls": 0, "native_calls": 0, "max_query_tokens": 0,
             "max_key_tokens": 0, "max_chunk_query_tokens": 0, "key_value_scope": "all_frames"}

    def attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
                  *, scale=None, enable_gqa=False):
        query_count, key_count = query.shape[-2], key.shape[-2]
        stats["max_query_tokens"] = max(stats["max_query_tokens"], query_count)
        stats["max_key_tokens"] = max(stats["max_key_tokens"], key_count)
        kwargs = {"attn_mask": attn_mask, "dropout_p": dropout_p, "is_causal": is_causal,
                  "scale": scale, "enable_gqa": enable_gqa}
        eligible = (not torch.is_grad_enabled() and attn_mask is None and dropout_p == 0.0
                    and not is_causal and not enable_gqa
                    and query.shape[:-2] == key.shape[:-2] == value.shape[:-2])
        if not eligible or query_count <= query_chunk_size:
            stats["native_calls"] += 1
            return original(query, key, value, **kwargs)
        stats["chunked_calls"] += 1
        stats["max_chunk_query_tokens"] = max(stats["max_chunk_query_tokens"], min(query_chunk_size, query_count))
        output = None
        for start in range(0, query_count, query_chunk_size):
            stop = min(start + query_chunk_size, query_count)
            part = original(query[..., start:stop, :], key, value, **kwargs)
            if output is None:
                output = part.new_empty((*part.shape[:-2], query_count, part.shape[-1]))
            output[..., start:stop, :] = part
            del part
        return output

    functional.scaled_dot_product_attention = attention
    try:
        yield stats
    finally:
        functional.scaled_dot_product_attention = original


def assert_run_configuration(root):
    manifest = read_json(root / "run_manifest.json")
    if identity(read_json(root / "config.json")) != manifest["config_identity"] or identity(read_json(root / "models.lock.json")) != manifest["model_identity"]:
        raise ValueError("Prepared configuration/model lock changed; prepare a new run")
    return manifest


def cache_compatible(root, capture):
    try:
        manifest = assert_run_configuration(root)
        record = manifest["captures"][capture]
        if record["status"] != "completed":
            return False
        expected = identity({"input": record["input_identity"], "config": manifest["config_identity"],
                             "model": manifest["model_identity"], "code": code_identity()})
        if record["prediction_identity"] != expected:
            return False
        verify_capture(root, capture)
        return True
    except (ValueError, KeyError, OSError):
        return False


def worker(root, capture, source, checkpoint):
    """One capture per process. No before/after tokens, poses or GPU state are shared."""
    root = Path(root).resolve()
    runtime = environment()
    runtime.update({"status": "running", "stage": "preflight", "capture": capture})
    write_json(root / capture / "inference_environment.json", runtime)
    try:
        return _worker(root, capture, source, checkpoint, runtime)
    except Exception as exc:
        # Include preflight/import and CPU export failures, not just model.forward.
        runtime.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc()})
        write_json(root / capture / "inference_environment.json", runtime)
        raise


def _worker(root, capture, source, checkpoint, runtime):
    root, source, checkpoint = Path(root).resolve(), Path(source).resolve(), Path(checkpoint).resolve()
    manifest = assert_run_configuration(root)
    record = manifest["captures"][capture]
    if record["status"] == "prepared":
        check_artifacts(root, record)
    config, lock = read_json(root / "config.json"), read_json(root / "models.lock.json")
    if digest(checkpoint) != lock["checkpoint_sha256"]:
        raise ValueError("Checkpoint content does not match pinned model identity")
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != lock["source_revision"]:
        raise ValueError("VGGT source checkout differs from pinned revision")
    if digest(source / "vggt/utils/load_fn.py") != lock["preprocessing_source_sha256"]:
        raise ValueError("VGGT preprocessing source differs from the pinned reference")
    if subprocess.run(["git", "-C", str(source), "diff", "--quiet", "HEAD", "--", "vggt"], check=False).returncode:
        raise ValueError("VGGT tracked model source has local changes")
    sys.path.insert(0, str(source))
    runtime["stage"] = "runtime_imports"
    import torch
    import torchvision
    from safetensors import safe_open
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU runtime is required; no silent CPU fallback")
    if torch.__version__.split("+")[0] != lock["torch"] or torchvision.__version__.split("+")[0] != lock["torchvision"]:
        raise RuntimeError("PyTorch/torchvision versions differ from the pinned Colab environment")
    frames = read_json(root / capture / "frames.json")["frames"]
    if any(int(f["orientation_counterclockwise_degrees"]) != 0 for f in frames):
        raise ValueError("The pinned upstream parity check expects the current zero-rotation source frames")
    paths = [str(root / f["image_path"]) for f in frames]
    runtime["stage"] = "preprocessing_parity"
    images = load_and_preprocess_images(paths, mode=config["preprocessing"]["mode"])
    saved = np.stack([np.array(Image.open(root / f["model_image_path"]).convert("RGB")) for f in frames])
    reference = torch.from_numpy(saved).permute(0, 3, 1, 2).float() / 255
    if not torch.equal(images, reference):
        raise ValueError("Prepared model images differ from the pinned upstream loader")
    runtime.update({"backend": "VGGT", "source_revision": revision, "checkpoint_sha256": lock["checkpoint_sha256"],
                    "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                    "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
                    "upstream_preprocessing_parity": "tensor_exact", "source_videos_decoded_in_worker": False,
                    "enabled_heads": ["camera", "depth"]})
    try:
        runtime["nvidia_smi"] = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        runtime["nvidia_smi"] = None
    import psutil
    stop = threading.Event()
    peak_rss = [0]
    def monitor():
        while not stop.wait(0.1):
            peak_rss[0] = max(peak_rss[0], psutil.Process().memory_info().rss)
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    start = time.perf_counter()
    try:
        runtime["stage"] = "model_loading"
        # Read only enabled parameters via mmap; avoid holding a second full state dict.
        model = VGGT(enable_point=False, enable_track=False)
        state = model.state_dict()
        with safe_open(str(checkpoint), framework="pt", device="cpu") as weights:
            keys = set(weights.keys())
            missing = set(state) - keys
            unexpected = [k for k in keys - set(state) if not k.startswith(("point_head.", "track_head."))]
            if missing or unexpected:
                raise ValueError(f"Checkpoint state mismatch: {len(missing)} missing, {len(unexpected)} unexpected tensors")
            with torch.no_grad():
                for name, target in state.items():
                    tensor = weights.get_tensor(name)
                    if tensor.shape != target.shape:
                        raise ValueError(f"Checkpoint tensor shape mismatch: {name}")
                    target.copy_(tensor)
        del state, tensor, target
        torch.cuda.reset_peak_memory_stats()
        model.eval().to("cuda")
        runtime["model_loading_sec"] = time.perf_counter() - start
        requested = config["inference"]["precision"]
        precision = ("bfloat16" if torch.cuda.is_bf16_supported() else "float16") if requested == "auto" else requested
        if precision not in ("bfloat16", "float16", "float32") or (precision == "bfloat16" and not torch.cuda.is_bf16_supported()):
            raise ValueError("Requested precision is unsupported on this GPU")
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[precision]
        images = images.to("cuda")
        runtime["precision"] = precision
        runtime["stage"] = "inference"
        torch.cuda.synchronize()
        inference_start = time.perf_counter()
        autocast = torch.autocast("cuda", dtype=dtype) if precision != "float32" else contextlib.nullcontext()
        print("Attention: query chunks of 512 tokens, all frame keys/values retained", flush=True)
        with torch.inference_mode(), autocast, bounded_attention(torch) as attention_stats:
            runtime["attention"] = attention_stats
            predictions = model(images)
        torch.cuda.synchronize()
        runtime.update({"inference_sec": time.perf_counter() - inference_start, "precision": precision,
                        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved()})
        runtime["stage"] = "camera_decode"
        extrinsics, intrinsics = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        c_from_w = extrinsics[0].float().cpu().numpy()
        pose = np.tile(np.eye(4), (len(frames), 1, 1))
        pose[:, :3, :] = c_from_w
        w_from_c = np.linalg.inv(pose)
        depth = predictions["depth"][0, ..., 0].float().cpu().numpy()
        confidence = predictions["depth_conf"][0].float().cpu().numpy()
        k = intrinsics[0].float().cpu().numpy()
        runtime["worker_total_sec_before_export"] = time.perf_counter() - start
        # CPU export does not need model weights or GPU predictions to stay resident.
        del model, predictions, images, extrinsics, intrinsics
        torch.cuda.empty_cache()
    except Exception as exc:
        runtime.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}", "worker_elapsed_sec": time.perf_counter() - start})
        raise
    finally:
        stop.set(); thread.join()
        runtime["rss_peak_sampled_bytes"] = peak_rss[0]
        runtime["rss_sampling_interval_sec"] = 0.1
        runtime["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        runtime["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        write_json(root / capture / "inference_environment.json", runtime)
    runtime["stage"] = "export"
    report = export_predictions(root, capture, depth, confidence, k, w_from_c, runtime)
    print(json.dumps({"capture": capture, "status": "completed", "points": report["cloud_points"], "gpu": runtime["gpu"]}), flush=True)


@contextlib.contextmanager
def run_lock(root):
    path = Path(root) / ".run.lock"
    try:
        handle = path.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise RuntimeError("Another run owns .run.lock. If a previous runtime died, confirm it is stopped before removing the stale lock.") from exc
    import os
    with handle:
        json.dump({"pid": os.getpid(), "started_epoch": time.time()}, handle)
    try:
        yield
    finally:
        path.unlink()


def run(root, source, checkpoint, *, resume=False, captures=("before", "after")):
    with run_lock(root):
        return _run(root, source, checkpoint, resume=resume, captures=captures)


def _run(root, source, checkpoint, *, resume=False, captures=("before", "after")):
    root = Path(root).resolve()
    assert_run_configuration(root)
    for capture in captures:
        if resume and cache_compatible(root, capture):
            print(f"{capture}: reusing verified saved reconstruction", flush=True)
            continue
        manifest = read_json(root / "run_manifest.json")
        record = manifest["captures"][capture]
        for path, expected in record["prepared_artifacts"].items():
            if digest(root / path) != expected:
                raise ValueError(f"Prepared evidence changed: {path}; prepare a new run")
        # Rebuild a corrupt capture only after verifying its immutable prepared inputs.
        frames = read_json(root / capture / "frames.json")["frames"]
        transforms = read_json(root / capture / "image_transforms.json")
        if identity({"frames": frames, "transforms": transforms}) != record["input_identity"]:
            raise ValueError("Prepared input metadata changed; prepare a new run")
        for f in frames:
            if digest(root / f["image_path"]) != f["image_sha256"] or digest(root / f["model_image_path"]) != f["model_image_sha256"]:
                raise ValueError("Prepared image changed; prepare a new run")
        record["status"] = "running"
        record.pop("error", None)
        manifest["status"] = "running"
        write_json(root / "run_manifest.json", manifest)
        logs = root / "logs"
        logs.mkdir(exist_ok=True)
        command = [sys.executable, str(Path(__file__).parents[1] / "scripts/reconstruct.py"), "worker", "--run", str(root),
                   "--capture", capture, "--source", str(source), "--checkpoint", str(checkpoint)]
        print(f"{capture}: fresh GPU worker ({len(frames)} frames)", flush=True)
        write_json(root / capture / "inference_environment.json",
                   {"status": "running", "stage": "worker_launch", "capture": capture})
        try:
            with (logs / f"{capture}.log").open("w", encoding="utf-8") as log:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                         timeout=read_json(root / "config.json")["inference"]["timeout_seconds"])
            if process.returncode:
                raise RuntimeError(f"GPU worker exited {process.returncode}; inspect logs/{capture}.log")
        except (RuntimeError, subprocess.TimeoutExpired, OSError) as exc:
            manifest = read_json(root / "run_manifest.json")
            manifest["captures"][capture].update({"status": "failed", "error": str(exc)})
            write_json(root / "run_manifest.json", manifest)
    manifest = read_json(root / "run_manifest.json")
    completed = [r["status"] == "completed" for r in manifest["captures"].values()]
    manifest["status"] = "completed" if all(completed) else "partially_completed" if any(completed) else "failed"
    write_json(root / "run_manifest.json", manifest)
    return manifest
