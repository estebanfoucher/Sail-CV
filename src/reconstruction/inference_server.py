"""MASt3R-only stereo reconstruction inference server.

A lean FastAPI service that loads MASt3R once at startup and turns a stereo
image pair plus calibration into a colored 3D point cloud. No FastSAM.

Endpoints:
    GET  /health       -> model/device status
    POST /reconstruct  -> multipart {image1, image2, calibration[, subsample]}
                          returns JSON {timing, profile, stats, ply_base64}

Run:
    uvicorn inference_server:app --host 0.0.0.0 --port 7862
"""

import base64
import io
import json
import os
import time
from contextlib import contextmanager

import numpy as np
import PIL.Image
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from loguru import logger
from PIL.ImageOps import exif_transpose

from stereo.convert_calibration import convert_calibration_parameters
from stereo.image import preprocess_image, resize_image
from stereo.mast3r import MASt3RInferenceEngine
from stereo.triangulation import extract_colors_from_image, triangulate_points

MODEL_PATH = os.environ.get(
    "MAST3R_CHECKPOINT",
    "/app/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
)
DEFAULT_SUBSAMPLE = int(os.environ.get("SUBSAMPLE", "8"))
DEFAULT_TARGET_SIZE = int(os.environ.get("TARGET_SIZE", "512"))
DEFAULT_PATCH_SIZE = int(os.environ.get("PATCH_SIZE", "16"))
BOUND_DISTANCE = float(os.environ.get("BOUND_DISTANCE", "20"))

app = FastAPI(title="Sail-CV Reconstruction Inference Server")

_engine: MASt3RInferenceEngine | None = None
_device: str = "cpu"


def get_engine() -> MASt3RInferenceEngine:
    global _engine, _device
    if _engine is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading MASt3R on {_device} from {MODEL_PATH}")
        engine = MASt3RInferenceEngine(model_path=MODEL_PATH, device=_device)
        engine.load_model()
        _engine = engine
        logger.info("MASt3R model ready")
    return _engine


@app.on_event("startup")
def _startup() -> None:
    # Fail fast if the checkpoint is missing; load the model eagerly.
    if not os.path.exists(MODEL_PATH):
        logger.error(f"MASt3R checkpoint not found: {MODEL_PATH}")
        return
    try:
        get_engine()
    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to load MASt3R at startup: {e}")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if _engine is not None else "model_not_loaded",
        "device": _device,
        "model_loaded": _engine is not None,
        "checkpoint": MODEL_PATH,
        "checkpoint_exists": os.path.exists(MODEL_PATH),
    }


@contextmanager
def _stage(timing: dict, name: str):
    """Record wall-clock seconds for a named stage (CUDA-synced if available)."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    timing[name] = round(time.perf_counter() - start, 4)


def _build_ply_bytes(points: np.ndarray, colors: np.ndarray, bound_distance: float):
    """Build an ASCII PLY in memory, returning (bytes, kept_point_count)."""
    points = np.asarray(points).reshape(-1, 3)
    colors = np.asarray(colors).reshape(-1, 3)
    valid = np.isfinite(points).all(axis=1) & (
        np.linalg.norm(points, axis=1) < bound_distance
    )
    pts = points[valid]
    cols = colors[valid].astype(int)

    buf = io.StringIO()
    buf.write("ply\nformat ascii 1.0\n")
    buf.write(f"element vertex {len(pts)}\n")
    buf.write("property float x\nproperty float y\nproperty float z\n")
    buf.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
    buf.write("end_header\n")
    for p, c in zip(pts, cols, strict=False):
        buf.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]}\n")
    return buf.getvalue().encode("utf-8"), len(pts)


def _load_pil(upload: UploadFile) -> PIL.Image.Image:
    data = upload.file.read()
    if not data:
        raise HTTPException(status_code=400, detail=f"Empty image: {upload.filename}")
    img = PIL.Image.open(io.BytesIO(data))
    return exif_transpose(img).convert("RGB")


@app.post("/reconstruct")
def reconstruct(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
    calibration: str = Form(...),
    subsample: int = Form(DEFAULT_SUBSAMPLE),
    target_size: int = Form(DEFAULT_TARGET_SIZE),
    patch_size: int = Form(DEFAULT_PATCH_SIZE),
    bound_distance: float = Form(BOUND_DISTANCE),
):
    engine = get_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        calib_data = json.loads(calibration)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid calibration JSON: {e}")

    timing: dict = {}
    total_start = time.perf_counter()

    with _stage(timing, "load_images"):
        img1 = _load_pil(image1)
        img2 = _load_pil(image2)

    with _stage(timing, "convert_calibration"):
        try:
            calib_params = convert_calibration_parameters(
                calib_data, target_size=target_size, patch_size=patch_size
            )
        except (KeyError, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"Bad calibration: {e}")

    with _stage(timing, "preprocess"):
        images = [
            preprocess_image(img1, size=target_size, idx=0),
            preprocess_image(img2, size=target_size, idx=1),
        ]

    with _stage(timing, "inference"):
        output = engine.run_inference(images)

    with _stage(timing, "matching"):
        raw = engine.extract_raw_data(output, subsample=subsample)
        matches_im0 = raw["matches_im0"]
        matches_im1 = raw["matches_im1"]

    with _stage(timing, "triangulation"):
        point_cloud = triangulate_points(matches_im0, matches_im1, calib_params)

    with _stage(timing, "colorize"):
        img1_array = np.array(resize_image(img1, size=target_size))
        colors = extract_colors_from_image(matches_im0, img1_array)

    with _stage(timing, "build_ply"):
        ply_bytes, kept = _build_ply_bytes(point_cloud, colors, bound_distance)

    timing["total"] = round(time.perf_counter() - total_start, 4)

    profile = {
        "device": _device,
        "cuda": torch.cuda.is_available(),
        "subsample": subsample,
        "target_size": target_size,
        "patch_size": patch_size,
    }
    if torch.cuda.is_available():
        profile["gpu_name"] = torch.cuda.get_device_name(0)
        profile["gpu_mem_allocated_mb"] = round(
            torch.cuda.max_memory_allocated() / 1e6, 1
        )
        torch.cuda.reset_peak_memory_stats()

    stats = {
        "num_matches": int(len(matches_im0)),
        "points_triangulated": int(len(point_cloud)),
        "points_kept": int(kept),
        "points_dropped": int(len(point_cloud) - kept),
        "bound_distance": bound_distance,
        "mast3r_image_size": list(calib_params.get("image_size", [])),
    }

    logger.info(
        f"reconstruct: {stats['num_matches']} matches -> {kept} pts "
        f"in {timing['total']}s (infer {timing['inference']}s)"
    )

    return JSONResponse(
        {
            "timing": timing,
            "profile": profile,
            "stats": stats,
            "ply_base64": base64.b64encode(ply_bytes).decode("ascii"),
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "7862")),
        log_level="info",
    )
