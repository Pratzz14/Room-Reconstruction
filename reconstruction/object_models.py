"""Optional GPU adapters. Importing saved artifacts never imports Torch."""

import gc
import importlib.metadata
import platform
import re
from pathlib import Path

import numpy as np

from .core import digest, read_json, write_json


def resolve_models(source, output):
    from huggingface_hub import HfApi, snapshot_download

    lock = read_json(source)
    resolved = {"schema_version": "1.0", "status": "resolved", "models": {}}
    for kind, spec in lock["models"].items():
        info = HfApi().model_info(spec["repository"], revision=spec["revision"])
        revision = info.sha
        folder = Path(
            snapshot_download(
                spec["repository"],
                revision=revision,
                allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
            )
        )
        files = {
            p.relative_to(folder).as_posix(): digest(p)
            for p in sorted(folder.rglob("*"))
            if p.is_file()
        }
        if not any(name.endswith(".safetensors") for name in files):
            raise ValueError(f"No model weights for {kind}")
        resolved["models"][kind] = {
            "repository": spec["repository"],
            "revision": revision,
            "files": files,
            "license": info.card_data.to_dict().get("license")
            if info.card_data
            else None,
        }
    write_json(output, resolved)
    return resolved


class ObjectModels:
    def __init__(self, config, lock):
        self.config, self.lock = config, lock
        self.model = self.processor = None
        self.kind = None
        self.runtime = {"python": platform.python_version(), "packages": {}}
        for name in (
            "torch",
            "torchvision",
            "transformers",
            "huggingface-hub",
            "numpy",
            "Pillow",
        ):
            try:
                self.runtime["packages"][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                raise RuntimeError(
                    "Run the object notebook's GPU setup cell before object inference"
                ) from None
        if lock.get("status") != "resolved" or any(
            not re.fullmatch(r"[0-9a-f]{40}", m["revision"]) or not m.get("files")
            for m in lock["models"].values()
        ):
            raise ValueError("Resolve and hash model revisions before object inference")

    def activate(self, kind):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import (
            AutoModel,
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
            Sam2Model,
        )

        self.close()
        spec = self.lock["models"][kind]
        folder = Path(
            snapshot_download(
                spec["repository"],
                revision=spec["revision"],
                allow_patterns=list(spec["files"]),
            )
        )
        if any(
            digest(folder / name) != expected
            for name, expected in spec["files"].items()
        ):
            raise ValueError(f"{kind}: checkpoint content mismatch")
        device = self.config["device"]
        self.device = (
            ("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else device
        )
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but unavailable; run the dedicated notebook on a GPU"
            )
        torch.manual_seed(self.config["seed"])
        if self.device == "cuda":
            torch.cuda.manual_seed_all(self.config["seed"])
            torch.cuda.reset_peak_memory_stats()
            self.runtime.update(
                gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda
            )
        cls = {
            "detector": AutoModelForZeroShotObjectDetection,
            "segmenter": Sam2Model,
            "appearance": AutoModel,
        }[kind]
        self.processor = AutoProcessor.from_pretrained(folder, local_files_only=True)
        self.model = (
            cls.from_pretrained(folder, local_files_only=True).eval().to(self.device)
        )
        self.kind = kind

    def detect(self, image):
        import torch

        d = self.config["detection"]
        inputs = self.processor(
            images=image, text=". ".join(d["vocabulary"]) + ".", return_tensors="pt"
        ).to(self.device)
        with torch.inference_mode():
            output = self.model(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            output,
            inputs.input_ids,
            threshold=d["box_threshold"],
            text_threshold=d["text_threshold"],
            target_sizes=[(image.height, image.width)],
        )[0]
        proposals = []
        for box, score, label in zip(
            result["boxes"],
            result["scores"],
            result.get("text_labels", result.get("labels")),
        ):
            raw = str(label).lower().strip(" .")
            normalized = d["aliases"].get(raw, raw)
            proposals.append(
                {
                    "box": box.detach().cpu().tolist(),
                    "score": float(score),
                    "raw_label": raw,
                    "label": normalized
                    if normalized in d["vocabulary"]
                    else "unknown_object",
                }
            )
        return sorted(proposals, key=lambda p: (-p["score"], p["label"], p["box"]))

    def segment(self, image, box):
        import torch

        inputs = self.processor(
            images=image, input_boxes=[[box]], return_tensors="pt"
        ).to(self.device)
        with torch.inference_mode():
            output = self.model(**inputs, multimask_output=True)
        masks = self.processor.post_process_masks(
            output.pred_masks.cpu(), inputs["original_sizes"].cpu(), binarize=False
        )[0]
        masks = masks.reshape(-1, image.height, image.width)
        scores = output.iou_scores.detach().float().cpu().numpy().reshape(-1)
        choice = int(np.argmax(scores))
        logits = masks[choice].float().numpy()
        probability = 1 / (1 + np.exp(-np.clip(logits, -80, 80)))
        return probability >= self.config["segmentation"]["mask_threshold"], float(
            scores[choice]
        )

    def embed(self, image):
        import torch

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            output = self.model(**inputs)
        descriptor = output.last_hidden_state[:, 0].detach().float().cpu().numpy()[0]
        return descriptor / max(float(np.linalg.norm(descriptor)), 1e-12)

    def close(self):
        if self.model is not None:
            import torch

            if self.device == "cuda":
                self.runtime[self.kind + "_peak_cuda_bytes"] = (
                    torch.cuda.max_memory_allocated()
                )
            self.model = self.processor = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
