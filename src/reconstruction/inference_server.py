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

from stereo.convert_calibration import (
    calculate_resize_and_crop_params,
    flatten_viewer_calibration,
)
from stereo.image import preprocess_image
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


def _parse_crop(crop_str: str, width: int, height: int):
    """Parse a UI crop spec into an integer pixel box (x0, y0, x1, y1) in the
    raw image, or None for the full frame.

    Spec JSON: {"cx", "cy", "w_frac", "h_frac"} where cx/cy are the crop center
    in raw pixels and w_frac/h_frac are the crop size as a fraction of the raw
    image width/height. Out-of-range values are clamped to the image.
    """
    if not crop_str:
        return None
    try:
        d = json.loads(crop_str)
    except (json.JSONDecodeError, TypeError):
        return None
    if not d:
        return None
    try:
        cx, cy = float(d["cx"]), float(d["cy"])
        wf, hf = float(d["w_frac"]), float(d["h_frac"])
    except (KeyError, TypeError, ValueError):
        return None
    cw = max(8.0, wf * width)
    ch = max(8.0, hf * height)
    x0 = max(0, min(int(round(cx - cw / 2)), width - 2))
    y0 = max(0, min(int(round(cy - ch / 2)), height - 2))
    x1 = max(x0 + 2, min(int(round(cx + cw / 2)), width))
    y1 = max(y0 + 2, min(int(round(cy + ch / 2)), height))
    if x0 == 0 and y0 == 0 and x1 == width and y1 == height:
        return None
    return (x0, y0, x1, y1)


_ROTATIONS = {
    90: PIL.Image.ROTATE_90,    # counter-clockwise
    180: PIL.Image.ROTATE_180,
    270: PIL.Image.ROTATE_270,
}


def _rotate_pil(img: PIL.Image.Image, rot: int) -> PIL.Image.Image:
    """Rotate by a multiple of 90° (CCW) using lossless transposes."""
    t = _ROTATIONS.get(rot % 360)
    return img.transpose(t) if t is not None else img


def _remap_to_raw(matches, box, raw_size, target_size, patch_size, rot=0):
    """Map MASt3R match coords (in the processed grid of the rotated crop) back
    to raw image pixel coords. Inverts, in order: MASt3R's resize+center-crop,
    the optional 90°/180°/270° rotation, then the crop offset.

    `box` is the (x0, y0, x1, y1) crop in raw coords, or None for the full frame.
    With box=None and rot=0 this reproduces the legacy coordinates exactly.
    """
    m = np.asarray(matches, dtype=np.float64).reshape(-1, 2)
    if box is None:
        x0, y0 = 0, 0
        crop_w, crop_h = raw_size
    else:
        x0, y0, x1, y1 = box
        crop_w, crop_h = x1 - x0, y1 - y0

    rot = rot % 360
    # Dimensions actually fed to MASt3R (rotation swaps W/H for 90/270).
    fed = (crop_h, crop_w) if rot in (90, 270) else (crop_w, crop_h)
    p = calculate_resize_and_crop_params(fed, target_size, patch_size)
    s = p["scale_factor"]
    ox, oy = p["crop_offset"]
    # Coords in the rotated-crop pixel grid.
    u = (m[:, 0] + ox) / s
    v = (m[:, 1] + oy) / s

    # Invert the rotation back into the un-rotated crop's pixel coords.
    if rot == 90:      # ROTATE_90 (CCW): xc = W-1-v, yc = u
        xc, yc = (crop_w - 1) - v, u
    elif rot == 270:   # ROTATE_270 (CW): xc = v, yc = H-1-u
        xc, yc = v, (crop_h - 1) - u
    elif rot == 180:
        xc, yc = (crop_w - 1) - u, (crop_h - 1) - v
    else:
        xc, yc = u, v

    raw = np.empty_like(m)
    raw[:, 0] = xc + x0
    raw[:, 1] = yc + y0
    return raw


@app.post("/reconstruct")
def reconstruct(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
    calibration: str = Form(...),
    subsample: int = Form(DEFAULT_SUBSAMPLE),
    target_size: int = Form(DEFAULT_TARGET_SIZE),
    patch_size: int = Form(DEFAULT_PATCH_SIZE),
    bound_distance: float = Form(BOUND_DISTANCE),
    crop1: str = Form(""),
    crop2: str = Form(""),
    rotate: int = Form(0),
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
    W1, H1 = img1.size
    W2, H2 = img2.size

    # Optional per-camera rectangle crop (raw pixels). Cropping focuses MASt3R
    # on the region of interest and discards peripheral (fish-eye) content.
    box1 = _parse_crop(crop1, W1, H1)
    box2 = _parse_crop(crop2, W2, H2)
    rot = rotate % 360
    if rot not in (0, 90, 180, 270):
        rot = 0

    try:
        flat_calib = flatten_viewer_calibration(calib_data)
    except (KeyError, ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=f"Bad calibration: {e}")

    with _stage(timing, "preprocess"):
        src1 = _rotate_pil(img1.crop(box1) if box1 else img1, rot)
        src2 = _rotate_pil(img2.crop(box2) if box2 else img2, rot)
        images = [
            preprocess_image(src1, size=target_size, idx=0),
            preprocess_image(src2, size=target_size, idx=1),
        ]

    with _stage(timing, "inference"):
        output = engine.run_inference(images)

    with _stage(timing, "matching"):
        raw = engine.extract_raw_data(output, subsample=subsample)
        matches_im0 = raw["matches_im0"]
        matches_im1 = raw["matches_im1"]

    # Map matches from the processed (cropped) grid back to raw image pixels so
    # we can triangulate with the unmodified raw-resolution intrinsics — no
    # on-the-fly cropped-camera calibration needed.
    matches_raw0 = _remap_to_raw(matches_im0, box1, (W1, H1), target_size, patch_size, rot)
    matches_raw1 = _remap_to_raw(matches_im1, box2, (W2, H2), target_size, patch_size, rot)

    with _stage(timing, "triangulation"):
        point_cloud = triangulate_points(matches_raw0, matches_raw1, flat_calib)

    with _stage(timing, "colorize"):
        img1_array = np.array(img1)
        colors = extract_colors_from_image(matches_raw0, img1_array)

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
        "raw_image_size": [int(W1), int(H1)],
        "crop1": list(box1) if box1 else None,
        "crop2": list(box2) if box2 else None,
        "rotate": rot,
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
