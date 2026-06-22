"""ONNX model download and session management."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path


def _configure_nvidia_dll_paths() -> None:
    """Expose pip-installed CUDA/cuDNN DLLs without a system CUDA toolkit."""
    if sys.platform == "win32":
        nvidia_root = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
        if not nvidia_root.is_dir():
            return

        bins: list[str] = []
        for sub in sorted(nvidia_root.iterdir()):
            if not sub.is_dir() or sub.name.startswith("_"):
                continue
            bin_dir = sub / "bin"
            if bin_dir.is_dir():
                path_str = str(bin_dir)
                bins.append(path_str)
                os.add_dll_directory(path_str)

        if bins:
            os.environ["PATH"] = os.pathsep.join(bins) + os.pathsep + os.environ.get("PATH", "")


_configure_nvidia_dll_paths()

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

if hasattr(ort, "preload_dlls"):
    ort.preload_dlls()

from src.config import (
    HF_MODEL_FILENAME,
    HF_REPO_ID,
    INPUT_SIZE,
    LOCAL_MODEL_NAME,
    MODELS_DIR,
)

logger = logging.getLogger(__name__)


def ensure_model(models_dir: Path = MODELS_DIR) -> Path:
    """Download the REACT++ ONNX model on first launch if missing."""
    models_dir.mkdir(parents=True, exist_ok=True)
    local_path = models_dir / LOCAL_MODEL_NAME
    nested_path = models_dir / "yolo12m" / LOCAL_MODEL_NAME

    for candidate in (local_path, nested_path):
        if candidate.is_file():
            logger.info("Using cached model at %s", candidate)
            return candidate

    logger.info(
        "Downloading REACT++ model from Hugging Face (%s / %s)...",
        HF_REPO_ID,
        HF_MODEL_FILENAME,
    )
    downloaded = Path(
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_MODEL_FILENAME,
            repo_type="model",
            local_dir=str(models_dir),
        )
    )
    if downloaded.is_file():
        logger.info("Model ready at %s", downloaded)
        return downloaded

    raise FileNotFoundError(f"Model download failed; expected file under {models_dir}")


def create_session(onnx_path: Path) -> ort.InferenceSession:
    """Create an ONNX Runtime session with GPU-first provider fallback."""
    available = set(ort.get_available_providers())
    preferred = [
        (
            "CUDAExecutionProvider",
            {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "cudnn_conv_algo_search": "EXHAUSTIVE",
            },
        ),
        "DmlExecutionProvider",
    ]

    for provider in preferred:
        name = provider if isinstance(provider, str) else provider[0]
        if name not in available:
            continue
        providers = [provider] if isinstance(provider, str) else [provider]
        try:
            session = ort.InferenceSession(str(onnx_path), providers=providers)
            if session.get_providers()[0] == name:
                logger.info("ONNX Runtime providers: %s", session.get_providers())
                return session
        except Exception as exc:
            logger.warning("Could not initialize %s: %s", name, exc)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    logger.info("ONNX Runtime providers: %s", session.get_providers())
    if "CUDAExecutionProvider" in available or "DmlExecutionProvider" in available:
        logger.warning("GPU provider unavailable; running on CPU.")
    return session


def load_class_names(session: ort.InferenceSession) -> tuple[dict[int, str], dict[int, str]]:
    """Load object and relation class names from ONNX metadata."""
    meta = session.get_modelmeta().custom_metadata_map
    if "obj_classes" not in meta or "rel_classes" not in meta:
        raise ValueError(
            "ONNX model is missing embedded class metadata. "
            "Use a REACT++ export from maelic/REACTPlusPlus_PSG."
        )

    obj_list = json.loads(meta["obj_classes"])
    rel_list = json.loads(meta["rel_classes"])
    obj_classes = {i: name for i, name in enumerate(obj_list, start=1)}
    rel_classes = {i: name for i, name in enumerate(rel_list)}
    logger.info(
        "Loaded %d object classes and %d relation classes from metadata.",
        len(obj_classes),
        len(rel_classes),
    )
    return obj_classes, rel_classes


def preprocess_image(image: np.ndarray, size: int = INPUT_SIZE) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Letterbox resize, BGR->RGB, CHW, normalize — matches SGG-Benchmark export."""
    import cv2

    height, width = image.shape[:2]
    ratio = min(size / height, size / width)
    new_w = int(round(width * ratio))
    new_h = int(round(height * ratio))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    top = int(round((size - new_h) / 2 - 0.1))
    bottom = int(round((size - new_h) / 2 + 0.1))
    left = int(round((size - new_w) / 2 - 0.1))
    right = int(round((size - new_w) / 2 + 0.1))

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    tensor = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1)).astype(np.float32) / 255.0
    return tensor[None, ...], ratio, (left, top)
