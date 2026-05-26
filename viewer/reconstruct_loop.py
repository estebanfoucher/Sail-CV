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
from stereo.pipeline import run_pipeline
from stereo.mast3r import MASt3RInferenceEngine

PROJECT_ROOT = Path("/app")
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
DEFAULT_CALIB_DIR = Path("/tmp/sailcv_calibrations")


def get_calibration_path() -> Path:
    """Find calibration: env var > latest in /tmp/sailcv_calibrations > legacy location."""
    # 1. Check environment variable
    if "CALIBRATION_PATH" in os.environ:
        path = Path(os.environ["CALIBRATION_PATH"])
        if path.exists():
            return path

    # 2. Check latest in /tmp/sailcv_calibrations
    if DEFAULT_CALIB_DIR.exists():
        calib_files = sorted(DEFAULT_CALIB_DIR.glob("calibration_*.json"), reverse=True)
        if calib_files:
            return calib_files[0]

    # 3. Fall back to legacy location
    return PROJECT_ROOT / "output" / "calibration" / "calibration.json"

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
    """One-shot RTSP frame grab. Slow (ffmpeg startup + keyframe wait); use RTSPGrabber for the loop."""
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
    matches = re.findall(r"(\d{3,5})x(\d{3,5})", result.stderr.decode(errors="ignore"))
    if not matches:
        raise RuntimeError("Could not parse frame dimensions from ffmpeg stderr")
    w, h = int(matches[-1][0]), int(matches[-1][1])
    arr = np.frombuffer(result.stdout, dtype=np.uint8)
    expected = w * h * 3
    if arr.size != expected:
        raise RuntimeError(f"Raw frame size mismatch: got {arr.size}, expected {expected} ({w}x{h})")
    return PIL.Image.fromarray(arr.reshape(h, w, 3))


def _probe_size(rtsp_url: str) -> tuple[int, int]:
    """Use ffprobe to get the stream's WxH so the rawvideo reader knows chunk size."""
    cmd = [
        "ffprobe", "-v", "error",
        "-rtsp_transport", "tcp",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0:s=x",
        rtsp_url,
    ]
    out = subprocess.run(cmd, capture_output=True, timeout=15).stdout.decode().strip()
    w, h = out.split("x")
    return int(w), int(h)


class RTSPGrabber:
    """Long-lived ffmpeg subprocess piping rawvideo. Background thread holds the latest frame."""

    def __init__(self, rtsp_url: str, name: str = "", output_width: int = 1024, fps: int = 4):
        self.url = rtsp_url
        self.name = name
        self.fps = fps
        src_w, src_h = _probe_size(rtsp_url)
        # Downscale on the ffmpeg side to keep swscale (software) cheap.
        scale = output_width / src_w
        self.w = output_width
        self.h = (int(src_h * scale) // 2) * 2  # keep even for yuv->rgb
        self._frame_bytes = self.w * self.h * 3
        self._latest: bytes | None = None
        self._lock = threading.Lock()
        self._first_frame = threading.Event()
        self._stop = threading.Event()
        self._proc: subprocess.Popen | None = None
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"grabber-{name}")
        self._thread.start()
        logger.info(f"[grabber-{name}] started ({src_w}x{src_h} -> {self.w}x{self.h} @ {fps}fps)")

    def _spawn(self):
        # fps filter throttles decode rate (we only need ~latest frame, not every camera frame).
        # scale before format=rgb24 so swscale runs on the smaller image.
        vf = f"fps={self.fps},scale={self.w}:{self.h},format=rgb24"
        cmd = [
            "ffmpeg",
            "-loglevel", "warning",
            "-rtsp_transport", "tcp",
            "-i", self.url,
            "-vf", vf,
            "-f", "rawvideo",
            "-an",
            "-",
        ]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Drain stderr in a background thread so the pipe never blocks; log lines as they come.
        def _log_stderr(p):
            try:
                for line in iter(p.stderr.readline, b""):
                    s = line.decode(errors="ignore").rstrip()
                    if s:
                        logger.warning(f"[grabber-{self.name} ffmpeg] {s}")
            except Exception:
                pass
        threading.Thread(target=_log_stderr, args=(self._proc,), daemon=True).start()

    @staticmethod
    def _read_exact(stream, n: int) -> bytes | None:
        """Read exactly n bytes, accumulating across short reads. Returns None on EOF."""
        buf = bytearray()
        while len(buf) < n:
            chunk = stream.read(n - len(buf))
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf)

    def _run(self):
        while not self._stop.is_set():
            try:
                self._spawn()
                assert self._proc and self._proc.stdout
                while not self._stop.is_set():
                    buf = self._read_exact(self._proc.stdout, self._frame_bytes)
                    if buf is None:
                        break
                    with self._lock:
                        self._latest = buf
                    if not self._first_frame.is_set():
                        self._first_frame.set()
                logger.warning(f"[grabber-{self.name}] ffmpeg stream ended, restarting")
            except Exception as e:
                logger.warning(f"[grabber-{self.name}] error: {e}, restarting in 1s")
            finally:
                if self._proc:
                    try:
                        self._proc.kill()
                    except Exception:
                        pass
                    self._proc = None
            if not self._stop.is_set():
                time.sleep(1)

    def get_latest(self, timeout: float = 10.0) -> PIL.Image.Image:
        if not self._first_frame.wait(timeout):
            raise RuntimeError(f"[grabber-{self.name}] no frame within {timeout}s")
        with self._lock:
            buf = self._latest
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(self.h, self.w, 3)
        return PIL.Image.fromarray(arr)

    def close(self):
        self._stop.set()
        if self._proc:
            try:
                self._proc.kill()
            except Exception:
                pass


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
    parser.add_argument("--subsample", type=int, default=16)
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

    # Calibration (check env > /tmp/sailcv_calibrations > legacy location)
    calib_path = get_calibration_path()
    with open(calib_path) as f:
        calibration_data = json.load(f)
    calibration_params = convert_calibration_parameters(calibration_data)
    img_w = calibration_params["image_size"][0]
    img_h = calibration_params["image_size"][1]
    # Extract calibration name from path for logging
    calib_name = calib_path.stem.replace("calibration_", "") if "calibration_" in calib_path.stem else calib_path.stem
    logger.info(f"Loaded calibration '{calib_name}' from {calib_path}")

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

    # Persistent RTSP grabbers (long-lived ffmpeg, latest-frame reader)
    grabbers: dict[str, RTSPGrabber] = {}
    try:
        grabbers["1"] = RTSPGrabber(CAMERAS["1"], name="1")
        grabbers["2"] = RTSPGrabber(CAMERAS["2"], name="2")
    except Exception as e:
        logger.warning(f"Could not start RTSP grabbers ({e}); falling back to fixtures")

    emit({"status": "ready"})

    params_path = output_dir / "params.json"

    def read_subsample(default: int) -> int:
        try:
            if params_path.exists():
                with open(params_path) as f:
                    v = int(json.load(f).get("subsample", default))
                return max(1, min(16, v))
        except Exception:
            pass
        return default

    frame = 0
    while True:
        frame += 1
        t0 = time.monotonic()
        timings = {}

        def lap(prev: float) -> tuple[int, float]:
            now = time.monotonic()
            return int((now - prev) * 1000), now

        try:
            subsample = read_subsample(args.subsample)
            # --- Grab frames (latest from long-lived ffmpeg streams) ---
            t = t0
            try:
                if "1" not in grabbers or "2" not in grabbers:
                    raise RuntimeError("grabbers not initialized")
                img1 = grabbers["1"].get_latest()
                img2 = grabbers["2"].get_latest()
            except Exception as e:
                logger.warning(f"RTSP grab failed ({e}), using fixture")
                img1, img2 = static_img1, static_img2
            timings["grab"], t = lap(t)

            # --- Pure pipeline: preproc → infer → match → composite → pointcloud ---
            result = run_pipeline(
                engine, img1, img2,
                image_size=(img_w, img_h),
                subsample=subsample,
                calibration_params=calibration_params,
            )
            timings.update(result.timings_ms)
            t = time.monotonic()

            # --- Save composite ---
            result.composite.save(output_dir / "composite.jpg", quality=85)
            for idx, im in enumerate((result.img1, result.img2), start=1):
                w_target = 256
                scale = w_target / im.width
                small = im.resize((w_target, int(im.height * scale)), PIL.Image.LANCZOS)
                small.save(output_dir / f"frame{idx}.jpg", quality=80)
            timings["io"], t = lap(t)

            # --- Save point cloud ---
            pc_data = {
                "pts":    result.pts3d.astype(np.float32).tolist(),
                "colors": result.colors.astype(np.uint8).tolist(),
            }
            with open(output_dir / "pointcloud.json", "w") as f:
                json.dump(pc_data, f)

            recon_ms = int((time.monotonic() - t0) * 1000)
            logger.info(f"Frame {frame} done in {recon_ms}ms — subsample={subsample}, {len(result.top_pts0)} matches, {len(result.pts3d)} pts | {timings}")
            emit({"frame": frame, "recon_ms": recon_ms, "num_matches": len(top_pts0), "timings": timings})

        except Exception as e:
            logger.error(f"Frame {frame} failed: {e}")
            emit({"error": str(e)})
            time.sleep(2)


if __name__ == "__main__":
    main()
