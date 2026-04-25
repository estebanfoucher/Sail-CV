"""Persistent reconstruction loop — loads model once, grabs live RTSP frames, loops.

Prints a JSON line to stdout after each frame:
    {"frame": N, "recon_ms": M, "num_matches": K}

Outputs per iteration (written to OUTPUT_DIR/live/):
    frame1.jpg, frame2.jpg, matches.json
"""

import argparse
import colorsys
import io
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import PIL.Image
import PIL.ImageDraw
import torch
from loguru import logger
from PIL.ImageOps import exif_transpose

sys.path.insert(0, "/app/src/reconstruction")
sys.path.insert(0, "/app/mast3r")
sys.path.insert(0, "/app/mast3r/dust3r")

from stereo.convert_calibration import convert_calibration_parameters
from stereo.image import preprocess_image, resize_image
from stereo.mast3r import MASt3RInferenceEngine

PROJECT_ROOT = Path("/app")
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"

CAMERAS = {
    "1": os.getenv("CAM1_URL", "rtsp://admin:123456@192.168.1.105/cam/realmonitor?channel=1&subtype=0"),
    "2": os.getenv("CAM2_URL", "rtsp://admin:123456@192.168.1.141/cam/realmonitor?channel=1&subtype=0"),
}

TOP_N = 20
GRID_COLS = 5
GRID_ROWS = 4  # 4x5 = 20 cells


def emit(obj: dict):
    print(json.dumps(obj), flush=True)


def grab_frame(rtsp_url: str) -> PIL.Image.Image:
    """Grab a single frame from an H.265 RTSP stream via ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-rtsp_transport", "tcp",
        "-i", rtsp_url,
        "-frames:v", "1",
        "-vf", "format=rgb24",
        "-f", "rawvideo", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=15)
    if not result.stdout:
        stderr = result.stderr.decode(errors="ignore")
        raise RuntimeError(f"ffmpeg returned no data. stderr: {stderr[-300:]}")
    import re
    # Pick last WxH match (avoids codec info lines, gets stream resolution)
    matches = re.findall(r"(\d{3,5})x(\d{3,5})", result.stderr.decode(errors="ignore"))
    if not matches:
        raise RuntimeError("Could not parse frame dimensions from ffmpeg stderr")
    w, h = int(matches[-1][0]), int(matches[-1][1])
    arr = np.frombuffer(result.stdout, dtype=np.uint8)
    expected = w * h * 3
    if arr.size != expected:
        raise RuntimeError(f"Raw frame size mismatch: got {arr.size}, expected {expected} ({w}x{h})")
    return PIL.Image.fromarray(arr.reshape(h, w, 3))


def select_top_matches(matches_im0, matches_im1, conf_scores, img_w, img_h, top_n=TOP_N):
    """
    Pick top_n matches mixing confidence + spatial spread.
    Strategy: divide image into a grid, pick best-confidence match per cell,
    then fill remaining slots from top-confidence leftovers.
    """
    pts0 = matches_im0.cpu().numpy() if hasattr(matches_im0, 'cpu') else np.asarray(matches_im0)
    pts1 = matches_im1.cpu().numpy() if hasattr(matches_im1, 'cpu') else np.asarray(matches_im1)
    scores = conf_scores  # (N,) float

    selected = []
    used = set()

    # --- Grid pass: best match per cell ---
    cell_w = img_w / GRID_COLS
    cell_h = img_h / GRID_ROWS
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            x_min, x_max = col * cell_w, (col + 1) * cell_w
            y_min, y_max = row * cell_h, (row + 1) * cell_h
            in_cell = np.where(
                (pts0[:, 0] >= x_min) & (pts0[:, 0] < x_max) &
                (pts0[:, 1] >= y_min) & (pts0[:, 1] < y_max)
            )[0]
            if len(in_cell) == 0:
                continue
            best = in_cell[np.argmax(scores[in_cell])]
            selected.append(best)
            used.add(best)

    # --- Fill remaining from top confidence ---
    if len(selected) < top_n:
        ranked = np.argsort(scores)[::-1]
        for idx in ranked:
            if idx not in used:
                selected.append(int(idx))
                used.add(int(idx))
            if len(selected) >= top_n:
                break

    selected = selected[:top_n]
    return pts0[selected], pts1[selected], scores[selected]


def build_composite(img1: PIL.Image.Image, img2: PIL.Image.Image,
                    pts0: np.ndarray, pts1: np.ndarray,
                    size: int = 512) -> PIL.Image.Image:
    """2×2 grid: top = plain frames, bottom = frames with match dots+lines."""
    scale = size / max(img1.width, img1.height)
    w, h = int(img1.width * scale), int(img1.height * scale)
    r1 = img1.resize((w, h), PIL.Image.LANCZOS)
    r2 = img2.resize((w, h), PIL.Image.LANCZOS)

    # Bottom row: draw matches
    bot = PIL.Image.new("RGB", (w * 2, h))
    bot.paste(r1, (0, 0))
    bot.paste(r2, (w, 0))
    draw = PIL.ImageDraw.Draw(bot)
    n = max(len(pts0), 1)
    dot_r = max(3, w // 80)
    for i, (p0, p1) in enumerate(zip(pts0, pts1)):
        r, g, b = colorsys.hsv_to_rgb(i / n, 1.0, 1.0)
        color = (int(r * 255), int(g * 255), int(b * 255))
        x0, y0 = int(p0[0]), int(p0[1])
        x1, y1 = int(p1[0]) + w, int(p1[1])
        draw.ellipse([x0 - dot_r, y0 - dot_r, x0 + dot_r, y0 + dot_r], fill=color)
        draw.ellipse([x1 - dot_r, y1 - dot_r, x1 + dot_r, y1 + dot_r], fill=color)
        draw.line([(x0, y0), (x1, y1)], fill=color, width=1)

    # Top row: plain
    top = PIL.Image.new("RGB", (w * 2, h))
    top.paste(r1, (0, 0))
    top.paste(r2, (w, 0))

    # Stack
    composite = PIL.Image.new("RGB", (w * 2, h * 2))
    composite.paste(top, (0, 0))
    composite.paste(bot, (0, h))
    return composite, w, h



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default=None, help="Use static scene fixture instead of live RTSP")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--subsample", type=int, default=8)
    parser.add_argument("--debug-frames", action="store_true", help="Skip inference, only grab+display frames")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / "output" / "live"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Debug-frames mode: skip NN entirely ---
    if args.debug_frames:
        logger.info("[debug-frames] Skipping model load, grabbing frames only")
        emit({"status": "ready"})
        frame = 0
        while True:
            frame += 1
            t0 = time.monotonic()
            try:
                try:
                    img1 = grab_frame(CAMERAS["1"])
                    img2 = grab_frame(CAMERAS["2"])
                    logger.info(f"[debug-frames] Frame {frame}: grabbed {img1.size} / {img2.size}, modes {img1.mode}/{img2.mode}")
                except Exception as e:
                    logger.warning(f"[debug-frames] RTSP failed ({e}), using fixture")
                    img1 = exif_transpose(PIL.Image.open(PROJECT_ROOT / "assets/reconstruction/scene_3/camera_1.png")).convert("RGB")
                    img2 = exif_transpose(PIL.Image.open(PROJECT_ROOT / "assets/reconstruction/scene_3/camera_2.png")).convert("RGB")

                # Save side-by-side composite (no matches)
                scale = 512 / max(img1.width, img1.height)
                w, h = int(img1.width * scale), int(img1.height * scale)
                r1 = img1.resize((w, h), PIL.Image.LANCZOS)
                r2 = img2.resize((w, h), PIL.Image.LANCZOS)
                comp = PIL.Image.new("RGB", (w * 2, h))
                comp.paste(r1, (0, 0))
                comp.paste(r2, (w, 0))
                comp.save(output_dir / "composite.jpg", quality=90)
                ms = int((time.monotonic() - t0) * 1000)
                logger.info(f"[debug-frames] Saved composite in {ms}ms")
                emit({"frame": frame, "recon_ms": ms, "num_matches": 0})
            except Exception as e:
                logger.error(f"[debug-frames] Error: {e}")
                emit({"error": str(e)})
                time.sleep(1)
        return

    # Calibration (canonical path, overwritten by calibration UI)
    calib_path = Path(os.getenv("CALIBRATION_PATH", PROJECT_ROOT / "output" / "calibration" / "calibration.json"))
    with open(calib_path) as f:
        calibration_data = json.load(f)
    calibration_params = convert_calibration_parameters(calibration_data)
    img_w = calibration_params["image_size"][0]
    img_h = calibration_params["image_size"][1]
    logger.info(f"Loaded calibration from {calib_path}")

    # Load static fixture images (fallback if no RTSP)
    fixture_dir = PROJECT_ROOT / "assets" / "reconstruction" / (args.scene or "scene_3")
    static_img1 = exif_transpose(PIL.Image.open(fixture_dir / "camera_1.png")).convert("RGB")
    static_img2 = exif_transpose(PIL.Image.open(fixture_dir / "camera_2.png")).convert("RGB")

    # Heartbeat: log a ping every 15s so the UI doesn't look frozen
    _stop_heartbeat = threading.Event()
    def _heartbeat():
        t0 = time.monotonic()
        while not _stop_heartbeat.wait(15):
            logger.info(f"[ping] still alive ({int(time.monotonic()-t0)}s elapsed)")
    threading.Thread(target=_heartbeat, daemon=True).start()

    # Load model once
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    engine = MASt3RInferenceEngine(model_path=args.checkpoint, device=device)
    engine.load_model()
    _stop_heartbeat.set()
    logger.info("Model loaded — starting loop")
    emit({"status": "ready"})

    frame = 0
    while True:
        frame += 1
        t0 = time.monotonic()
        try:
            # --- Grab frames ---
            try:
                img1 = grab_frame(CAMERAS["1"])
                img2 = grab_frame(CAMERAS["2"])
                logger.info(f"Frame {frame}: grabbed live frames")
            except Exception as e:
                logger.warning(f"RTSP grab failed ({e}), using fixture")
                img1, img2 = static_img1, static_img2

            # --- Preprocess ---
            images = [
                preprocess_image(img1, size=512, idx=0),
                preprocess_image(img2, size=512, idx=1),
            ]

            # --- Inference ---
            output = engine.run_inference(images)
            raw = engine.extract_raw_data(output, subsample=args.subsample)

            matches_im0 = raw["matches_im0"]  # pixel coords in img0
            matches_im1 = raw["matches_im1"]  # pixel coords in img1

            # Confidence: dot product of matched descriptors
            desc1 = raw["pred1"]["desc"].squeeze(0).detach()  # (H, W, D)
            desc2 = raw["pred2"]["desc"].squeeze(0).detach()
            H, W = desc1.shape[:2]
            pts0_np = matches_im0.cpu().numpy() if hasattr(matches_im0, 'cpu') else matches_im0
            pts1_np = matches_im1.cpu().numpy() if hasattr(matches_im1, 'cpu') else matches_im1
            pts0_idx = np.clip(pts0_np[:, 1], 0, H-1).astype(int) * W + np.clip(pts0_np[:, 0], 0, W-1).astype(int)
            pts1_idx = np.clip(pts1_np[:, 1], 0, H-1).astype(int) * W + np.clip(pts1_np[:, 0], 0, W-1).astype(int)
            d1 = desc1.reshape(-1, desc1.shape[-1])[pts0_idx]
            d2 = desc2.reshape(-1, desc2.shape[-1])[pts1_idx]
            conf_scores = (d1 * d2).sum(dim=-1).cpu().numpy().astype(float)

            # --- Select top 20 (grid + confidence) ---
            top_pts0, top_pts1, top_scores = select_top_matches(
                matches_im0, matches_im1, conf_scores, img_w, img_h
            )

            # --- Build and save 2×2 composite ---
            composite, fw, fh = build_composite(img1, img2, top_pts0, top_pts1)
            composite.save(output_dir / "composite.jpg", quality=85)

            # --- Save per-camera frames (downscaled) for camera-base point clouds ---
            for idx, im in enumerate((img1, img2), start=1):
                w_target = 256
                scale = w_target / im.width
                small = im.resize((w_target, int(im.height * scale)), PIL.Image.LANCZOS)
                small.save(output_dir / f"frame{idx}.jpg", quality=80)

            # --- Build and save point cloud (subsample to ~8k pts) ---
            pts3d_1 = raw["pts3d_1"]          # (H, W, 3)
            pts3d_2 = raw["pts3d_2"]          # (H, W, 3)
            col1    = raw["img1_colors"]       # (H, W, 3) in [-1, 1] or [0, 1] from view
            col2    = raw["img2_colors"]

            # Combine both clouds
            pts_all = np.concatenate([pts3d_1.reshape(-1, 3), pts3d_2.reshape(-1, 3)], axis=0)
            col_all = np.concatenate([col1.reshape(-1, 3),    col2.reshape(-1, 3)],    axis=0)

            # Normalize colors to [0, 1]
            col_min, col_max = col_all.min(), col_all.max()
            if col_max > col_min:
                col_all = (col_all - col_min) / (col_max - col_min)

            # Filter: drop points too far away (outliers)
            dist = np.linalg.norm(pts_all, axis=1)
            mask = (dist > 0.01) & (dist < 20.0) & np.isfinite(dist)
            pts_all = pts_all[mask]
            col_all = col_all[mask]

            # Subsample to at most 8000 points
            MAX_PTS = 8000
            if len(pts_all) > MAX_PTS:
                idx = np.random.choice(len(pts_all), MAX_PTS, replace=False)
                pts_all = pts_all[idx]
                col_all = col_all[idx]

            pc_data = {
                "pts":    pts_all.astype(np.float32).tolist(),
                "colors": (col_all * 255).astype(np.uint8).tolist(),
            }
            with open(output_dir / "pointcloud.json", "w") as f:
                json.dump(pc_data, f)

            recon_ms = int((time.monotonic() - t0) * 1000)
            logger.info(f"Frame {frame} done in {recon_ms}ms — {len(top_pts0)} matches, {len(pts_all)} pts")
            emit({"frame": frame, "recon_ms": recon_ms, "num_matches": len(top_pts0)})

        except Exception as e:
            logger.error(f"Frame {frame} failed: {e}")
            emit({"error": str(e)})
            time.sleep(2)


if __name__ == "__main__":
    main()
