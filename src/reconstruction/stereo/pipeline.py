"""End-to-end reconstruction pipeline for one stereo pair.

Given an instantiated inference engine and a pair of PIL images, runs:
    preprocess → infer → match → composite → pointcloud

and returns everything as a single PipelineResult. No I/O — callers decide
where (and whether) to persist artefacts.

This is the seam shared by:
    - viewer/reconstruct_loop.py (live, RTSP-fed)
    - scripts/run_offline.py (offline, fixture-fed)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import PIL.Image
import torch
from loguru import logger

from stereo.composite import build_composite
from stereo.image import preprocess_image, resize_image
from stereo.matching import compute_match_confidence, select_top_matches
from stereo.pointcloud import build_pointcloud
from stereo.triangulation import triangulate_points, extract_colors_from_image


@dataclass
class PipelineResult:
    # Inputs (kept around for downstream rendering)
    img1: PIL.Image.Image
    img2: PIL.Image.Image

    # Selected matches (post-confidence + spatial spread)
    top_pts0: np.ndarray   # (N, 2) image coords in cam1
    top_pts1: np.ndarray   # (N, 2) image coords in cam2
    top_scores: np.ndarray # (N,) confidence per match
    num_matches_total: int # before top-N selection

    # Filtered + subsampled point cloud
    pts3d: np.ndarray      # (M, 3) float32 in metres
    colors: np.ndarray     # (M, 3) uint8

    # Pre-built side-by-side panel
    composite: PIL.Image.Image
    composite_w: int
    composite_h: int

    # Per-stage wall-clock in milliseconds
    timings_ms: dict[str, int] = field(default_factory=dict)

    # Full raw output of MASt3RInferenceEngine.extract_raw_data, for callers
    # that want to do their own analysis (e.g. evaluate alternate match
    # selection strategies offline). Kept opt-in to avoid pinning GPU memory.
    raw: dict[str, Any] | None = None


def run_pipeline(engine,
                 img1: PIL.Image.Image,
                 img2: PIL.Image.Image,
                 image_size: tuple[int, int],
                 subsample: int,
                 max_pointcloud_pts: int = 8000,
                 keep_raw: bool = False,
                 calibration_params: dict | None = None) -> PipelineResult:
    """Run one full reconstruction pass.

    Args:
        engine: instance of MASt3RInferenceEngine (or subclass) with model loaded.
        img1, img2: PIL RGB images at native capture resolution.
        image_size: (w, h) of the calibration's image space — used by the
            top-N grid selector. Typically read from the calibration JSON.
        subsample: stride passed to fast_reciprocal_NNs (1, 2, 4, 8, or 16).
        max_pointcloud_pts: cap on the returned point cloud size.
        keep_raw: if True, attach the raw inference dict to the result.
        calibration_params: optional stereo calibration dict. If provided, uses feature-based
            triangulation (geometrically constrained by cameras). If not provided, uses
            MASt3R's per-pixel depth predictions (self-calibrating but less accurate).

    Per-stage timings reported in PipelineResult.timings_ms (ms):
        preproc, infer, match, composite, pointcloud
    """
    timings: dict[str, int] = {}

    def lap(prev: float) -> tuple[int, float]:
        now = time.monotonic()
        return int((now - prev) * 1000), now

    img_w, img_h = image_size
    t = time.monotonic()

    # --- Preprocess ---
    img1_preprocessed = preprocess_image(img1, size=512, idx=0)
    img2_preprocessed = preprocess_image(img2, size=512, idx=1)
    images = [img1_preprocessed, img2_preprocessed]
    timings["preproc"], t = lap(t)

    # --- Inference (sync GPU before timing the next step) ---
    output = engine.run_inference(images)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    timings["infer"], t = lap(t)

    # --- Match extraction + confidence + top-N selection ---
    raw = engine.extract_raw_data(output, subsample=subsample)
    conf_scores = compute_match_confidence(raw)
    top_pts0, top_pts1, top_scores = select_top_matches(
        raw["matches_im0"], raw["matches_im1"], conf_scores, img_w, img_h
    )
    num_matches_total = int(raw.get("num_matches", len(conf_scores)))
    timings["match"], t = lap(t)

    # --- Composite (CPU, in-memory) ---
    composite, fw, fh = build_composite(img1, img2, top_pts0, top_pts1)
    timings["composite"], t = lap(t)

    # --- Point cloud (CPU, in-memory) ---
    if calibration_params is not None:
        logger.info("Using calibration-based triangulation (feature matches)")
        matches_im0 = raw["matches_im0"]
        matches_im1 = raw["matches_im1"]

        if len(matches_im0) > 0:
            pts3d = triangulate_points(matches_im0, matches_im1, calibration_params)
            img1_resized = resize_image(img1, size=512)
            img1_array = np.array(img1_resized)
            colors = extract_colors_from_image(matches_im0, img1_array)

            if len(pts3d) > max_pointcloud_pts:
                rng = np.random.default_rng()
                idx = rng.choice(len(pts3d), max_pointcloud_pts, replace=False)
                pts3d = pts3d[idx]
                colors = colors[idx]
        else:
            logger.warning("No matches found, returning empty point cloud")
            pts3d = np.empty((0, 3), dtype=np.float32)
            colors = np.empty((0, 3), dtype=np.uint8)
    else:
        logger.warning("No calibration provided, using MASt3R per-pixel depth predictions")
        pts3d, colors = build_pointcloud(raw, max_pts=max_pointcloud_pts)

    timings["pointcloud"], t = lap(t)

    return PipelineResult(
        img1=img1, img2=img2,
        top_pts0=top_pts0, top_pts1=top_pts1, top_scores=top_scores,
        num_matches_total=num_matches_total,
        pts3d=pts3d, colors=colors,
        composite=composite, composite_w=fw, composite_h=fh,
        timings_ms=timings,
        raw=raw if keep_raw else None,
    )


def make_engine(engine_name: str, checkpoint, device: str):
    """Factory for the two engine flavors. Imported lazily to keep the import
    graph lean (speedy_mast3r pulls in tensorrt)."""
    from stereo.mast3r import MASt3RInferenceEngine
    if engine_name == "speedy":
        from stereo.speedy_mast3r import SpeedyMASt3RInferenceEngine
        return SpeedyMASt3RInferenceEngine(model_path=checkpoint, device=device)
    if engine_name == "vanilla":
        return MASt3RInferenceEngine(model_path=checkpoint, device=device)
    raise ValueError(f"Unknown engine: {engine_name}")
