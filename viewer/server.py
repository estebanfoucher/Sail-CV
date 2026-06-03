"""PLY point cloud viewer with continuous reconstruction loop."""
import json
import os
import queue
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import numpy as np

PROJECT_DIR = Path(os.getenv("PROJECT_DIR", Path(__file__).parent.parent)).resolve()
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_DIR / "output")).resolve()
STATIC_DIR = Path(__file__).parent / "static"
PORT = int(os.getenv("PORT", 7863))
DOCKER_IMAGE = os.getenv("DOCKER_IMAGE", "docker-sailcv-3d-reconstruction:latest")
CALIBRATION_PATH = Path(os.getenv("CALIBRATION_PATH", OUTPUT_DIR / "calibration" / "calibration.json")).resolve()
# MASt3R-only reconstruction inference server (separate CUDA Docker service).
INFERENCE_URL = os.getenv("INFERENCE_URL", "http://localhost:7862")

sys.path.insert(0, str(PROJECT_DIR / "src" / "reconstruction"))
from cameras.cameras import create_cameras_from_stereo_calibration

import cv2
from datetime import datetime


# --- Calibration management ---
DEFAULT_CALIB_DIR = Path("/tmp/sailcv_calibrations")
DEFAULT_CALIB_DIR.mkdir(exist_ok=True)

calibration_lock = threading.Lock()
current_calibration_name = "none"  # Display name of currently loaded calibration


def get_default_calibration_path() -> Path | None:
    """Find the most recent calibration in the default directory, or None."""
    calib_files = sorted(DEFAULT_CALIB_DIR.glob("calibration_*.json"), reverse=True)
    return calib_files[0] if calib_files else None


def get_active_calibration_path() -> Path | None:
    """Return the active calibration path (env override, or default)."""
    env_path = Path(os.getenv("CALIBRATION_PATH", ""))
    if env_path != Path("") and env_path.exists():
        return env_path.resolve()
    return get_default_calibration_path()


def update_current_calibration_name():
    """Update the display name of the current calibration."""
    global current_calibration_name
    path = get_active_calibration_path()
    if path is None:
        current_calibration_name = "none"
    else:
        # Extract timestamp from filename (calibration_YYYYMMDD_HHMMSS.json)
        name = path.stem
        if name.startswith("calibration_"):
            current_calibration_name = name.replace("calibration_", "")
        else:
            current_calibration_name = name


def _load_frame_rgb(path: Path, max_width: int = 128) -> np.ndarray | None:
    """Load JPEG, convert to RGB, downscale so width <= max_width."""
    if not path.exists():
        return None
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None
    h, w = bgr.shape[:2]
    if w > max_width:
        scale = max_width / w
        bgr = cv2.resize(bgr, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


PYRAMID_HEIGHT_M = 0.03  # frustum depth (apex → image plane) in metres


def _base_pointcloud(cam) -> tuple[list, list]:
    """Per-pixel point cloud on the camera's image-plane base (vectorized bilinear interp).

    Mirrors the math in cameras.py:get_pyramid_with_texture_coords (lines 207-243):
    for each pixel (col=j, row=i_flipped), uv ∈ [0,1] → bilinear blend of the 4
    image-plane corners → world-space point. Color sampled from the image.
    """
    img = cam.image  # RGB, HxWx3
    if img is None or img.size == 0 or img.shape[0] < 2 or img.shape[1] < 2:
        return [], []
    H, W = img.shape[:2]
    corners = cam.get_image_plane_corners(focal_length=PYRAMID_HEIGHT_M)  # (4, 3) world coords

    js = np.arange(W)
    rows = np.arange(H)
    u = (js / max(W - 1, 1))[None, :]              # (1, W)
    v = (rows / max(H - 1, 1))[:, None]             # (H, 1) row 0 (image top) -> v=0
    w0 = (1 - u) * (1 - v)
    w1 = u * (1 - v)
    w2 = u * v
    w3 = (1 - u) * v
    weights = np.stack([w0, w1, w2, w3], axis=-1)  # (H, W, 4)
    verts = weights @ corners                       # (H, W, 3)
    pts = verts.reshape(-1, 3).astype(np.float32).tolist()
    cols = img.reshape(-1, 3).astype(np.uint8).tolist()
    return pts, cols


def _flatten_calibration(calibration: dict) -> dict:
    """Convert nested session calibration to the flat keys the Camera factory expects."""
    if "camera_matrix1" in calibration:
        return calibration
    intr, extr = calibration["intrinsics"], calibration["extrinsics"]
    return {
        "camera_matrix1": intr["cam1"]["camera_matrix"],
        "camera_matrix2": intr["cam2"]["camera_matrix"],
        "rotation_matrix": extr["rotation_matrix"],
        "translation_vector": extr["translation_vector"],
        "image_size": calibration["image_size"],
    }


def build_frusta(calibration: dict, img1: np.ndarray | None, img2: np.ndarray | None) -> dict:
    """Build frusta + per-camera base point clouds from a (flat) stereo calibration."""
    dummy = np.zeros((1, 1, 3), dtype=np.uint8)
    cam1, cam2 = create_cameras_from_stereo_calibration(
        calibration,
        img1 if img1 is not None else dummy,
        img2 if img2 is not None else dummy,
        scale_factor=0.001,
    )
    out = []
    for cam, img in ((cam1, img1), (cam2, img2)):
        verts, edges = cam.get_pyramid_vertices(focal_length=PYRAMID_HEIGHT_M)
        base_pts, base_cols = _base_pointcloud(cam) if img is not None else ([], [])
        out.append({
            "name": cam.name,
            "vertices": verts.tolist(),
            "edges": edges,
            "base_points": base_pts,
            "base_colors": base_cols,
        })
    return {"cameras": out}


def compute_camera_frusta() -> dict | None:
    """Read calibration + latest frames, return frusta + per-camera base point clouds."""
    calib_path = get_active_calibration_path()
    if calib_path is None or not calib_path.exists():
        return None
    with open(calib_path) as f:
        calibration = json.load(f)

    live_dir = OUTPUT_DIR / "live"
    img1 = _load_frame_rgb(live_dir / "frame1.jpg")
    img2 = _load_frame_rgb(live_dir / "frame2.jpg")
    return build_frusta(calibration, img1, img2)

CAMERAS = {
    "1": os.getenv("CAM1_URL", "rtsp://192.168.1.34:554/stream1"),
    "2": os.getenv("CAM2_URL", "rtsp://192.168.1.214:554/stream1"),
}

# --- Shared state ---
state_lock = threading.Lock()
state = {
    "running": False,
    "scene": None,
    "frame_count": 0,
    "last_recon_ms": None,
}
stop_event = threading.Event()
sse_queues: list[queue.Queue] = []
sse_lock = threading.Lock()
log_queues: list[queue.Queue] = []
log_lock = threading.Lock()


def broadcast(event: dict):
    data = f"data: {json.dumps(event)}\n\n"
    with sse_lock:
        dead = []
        for q in sse_queues:
            try:
                q.put_nowait(data)
            except queue.Full:
                dead.append(q)
        for q in dead:
            sse_queues.remove(q)


def log(msg: str):
    """Print and stream a log line to all /logs SSE clients."""
    print(msg)
    data = f"data: {json.dumps({'line': msg})}\n\n"
    with log_lock:
        dead = []
        for q in log_queues:
            try:
                q.put_nowait(data)
            except queue.Full:
                dead.append(q)
        for q in dead:
            log_queues.remove(q)


CONTAINER_NAME      = "sailcv-recon"
CALIB_CONTAINER     = "sailcv-calib"

# --- Calibration state ---
calib_lock = threading.Lock()
calib_state = {
    "running": False,
    "status": None,   # "recording" | "calibrating" | "done" | "error"
    "reprojection_error": None,
    "error": None,
}


def reconstruction_loop(scene: str):
    """Run a single persistent Docker container that keeps the model in memory."""
    global state
    stop_event.clear()

    # Kill any leftover container from a previous run
    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True)

    # Determine calibration path to pass to container
    calib_path = get_active_calibration_path()
    calib_path_arg = str(calib_path) if calib_path else ""

    cmd = [
        "docker", "run", "--rm",
        "--name", CONTAINER_NAME,
        "--runtime=nvidia",
        "--memory=4g",
        "--shm-size=1g",
        "-w", "/app",
        "-v", f"{PROJECT_DIR}/viewer:/app/viewer:ro",
        "-v", f"{PROJECT_DIR}/assets/reconstruction:/app/assets/reconstruction:ro",
        "-v", f"{PROJECT_DIR}/checkpoints:/app/checkpoints:ro",
        "-v", f"{PROJECT_DIR}/src/reconstruction:/app/src/reconstruction:ro",
        "-v", f"{OUTPUT_DIR}:/app/output:rw",
        "-v", f"{DEFAULT_CALIB_DIR}:/tmp/sailcv_calibrations:ro",
        "-e", "DEVICE=cuda",
        "-e", "CUDA_MEMORY_FRACTION=0.6",
        *(["--env", f"CALIBRATION_PATH={calib_path_arg}"] if calib_path_arg else []),
        DOCKER_IMAGE,
        "python3", "viewer/reconstruct_loop.py",
        *(["--debug-frames"] if os.getenv("DEBUG_FRAMES") else []),
    ]
    log(f"[docker] Starting persistent container for scene '{scene}'")
    log(f"[docker] " + " ".join(cmd))

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Stream stderr (model logs) to log panel in background
    def stream_stderr():
        for line in proc.stderr:
            log(line.rstrip())
    threading.Thread(target=stream_stderr, daemon=True).start()

    # Read stdout — each line is a JSON status event
    for raw in proc.stdout:
        if stop_event.is_set():
            proc.terminate()
            break
        raw = raw.strip()
        if not raw:
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            log(f"[docker stdout] {raw}")
            continue

        if "status" in event:
            log(f"[loop] {event['status']}")
            continue

        if "error" in event:
            log(f"[loop] ERROR: {event['error']}")
            broadcast({"error": event["error"]})
            continue

        if "frame" in event:
            frame = event["frame"]
            ms = event["recon_ms"]
            timings = event.get("timings") or {}
            with state_lock:
                state["frame_count"] = frame
                state["last_recon_ms"] = ms
            log(f"[loop] Frame {frame} done in {ms}ms")
            broadcast({
                "frame": frame,
                "recon_ms": ms,
                "scene": scene,
                "num_matches": event.get("num_matches"),
                "timings": timings,
            })

    proc.wait()
    if proc.returncode not in (0, -15):
        log(f"[loop] Container exited with code {proc.returncode}")

    log("[loop] Stopped")
    with state_lock:
        state["running"] = False
        state["scene"] = None


CALIB_PAIRS_DIR = OUTPUT_DIR / "calib_session"
calib_pair_count = 0
calib_pair_lock  = threading.Lock()


# --- Stereo screenshot sessions ---
SCREENSHOT_DIR = OUTPUT_DIR / "screenshots"
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
screenshot_lock = threading.Lock()
active_session: str | None = None  # name of the active session folder

# Screenshot categories, each stored in its own subfolder per session.
SHOT_KINDS = ("normal", "intrinsic", "extrinsic")

# Rig board specs (committed config) used for per-session calibration.
CALIB_BOARDS_DIR = Path(__file__).parent / "calib_boards"
CHECKERBOARD_SPECS_PATH = CALIB_BOARDS_DIR / "checkerboard_specs.yml"
CHARUCO_SPECS_PATH = CALIB_BOARDS_DIR / "charuco_specs.yml"
CALIBRATION_FILENAME = "calibration.json"

# Known default camera intrinsics (committed). Used by the extrinsic-only
# calibration path: when the cameras are already calibrated, a new setup only
# needs the stereo extrinsics solved from Charuco pairs.
DEFAULT_INTRINSICS_DIR = PROJECT_DIR / "assets" / "reconstruction" / "intrinsics"
DEFAULT_INTRINSICS_CAM1 = DEFAULT_INTRINSICS_DIR / "intrinsics_1_1.json"
DEFAULT_INTRINSICS_CAM2 = DEFAULT_INTRINSICS_DIR / "intrinsics_1_2.json"

# Per-session calibration job state.
session_calib_lock = threading.Lock()
session_calib_state = {
    "running": False,
    "session": None,
    "status": None,   # "intrinsics" | "extrinsics" | "done" | "error"
    "error": None,
    "progress": None,  # {phase,label,current,total,found,detected}
}


def list_sessions() -> list[str]:
    """Session folder names, newest first."""
    return sorted(
        (d.name for d in SCREENSHOT_DIR.iterdir() if d.is_dir() and d.name.startswith("session_")),
        reverse=True,
    )


def list_pairs(session: str, kind: str) -> list[dict]:
    """Stereo pairs in a session/kind subfolder, grouped by timestamp, oldest first."""
    kind_dir = SCREENSHOT_DIR / session / kind
    if not kind_dir.is_dir():
        return []
    stamps = sorted(
        {f.name[len("shot_"):-len("_cam1.jpg")] for f in kind_dir.glob("shot_*_cam1.jpg")}
    )
    pairs = []
    for ts in stamps:
        cam1 = f"shot_{ts}_cam1.jpg"
        cam2 = f"shot_{ts}_cam2.jpg"
        if (kind_dir / cam1).exists() and (kind_dir / cam2).exists():
            pairs.append({"timestamp": ts, "cam1": cam1, "cam2": cam2})
    return pairs


def list_all_pairs(session: str) -> dict[str, list[dict]]:
    """All pairs in a session, grouped by kind."""
    return {kind: list_pairs(session, kind) for kind in SHOT_KINDS}


def _safe_segment(name: str) -> bool:
    return ".." not in name and "/" not in name and "\\" not in name


def capture_frame_jpeg(cam_id: str) -> bytes:
    """Grab one JPEG frame from camera via ffmpeg (runs on host, no Docker)."""
    url = CAMERAS[cam_id]
    cmd = [
        "ffmpeg", "-y",
        "-rtsp_transport", "tcp",
        "-i", url,
        "-frames:v", "1",
        "-f", "image2pipe", "-vcodec", "mjpeg", "-",
    ]
    r = subprocess.run(cmd, capture_output=True, timeout=15)
    if not r.stdout:
        raise RuntimeError(f"ffmpeg returned no data for cam {cam_id}: {r.stderr.decode(errors='ignore')[-200:]}")
    return r.stdout


def run_calibration_docker(square_mm: float):
    """Run calibrate_pairs.py in Docker on the collected image pairs."""
    subprocess.run(["docker", "rm", "-f", CALIB_CONTAINER], capture_output=True)
    cmd = [
        "docker", "run", "--rm",
        "--name", CALIB_CONTAINER,
        "--runtime=nvidia",
        "--memory=4g",
        "--shm-size=512m",
        "-w", "/app",
        "-v", f"{PROJECT_DIR}/viewer:/app/viewer:ro",
        "-v", f"{PROJECT_DIR}/src/reconstruction:/app/src/reconstruction:ro",
        "-v", f"{OUTPUT_DIR}:/app/output:rw",
        DOCKER_IMAGE,
        "python3", "viewer/calibrate_pairs.py",
        "--pairs-dir", "/app/output/calib_session",
        "--square-mm", str(square_mm),
    ]
    log(f"[calib] Running calibration on {calib_pair_count} pairs")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def stream_stderr():
        for line in proc.stderr:
            log(line.rstrip())
    threading.Thread(target=stream_stderr, daemon=True).start()

    for raw in proc.stdout:
        raw = raw.strip()
        if not raw:
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            log(f"[calib] {raw}")
            continue
        if "status" in event:
            with calib_lock:
                calib_state["status"] = event["status"]
                if event["status"] == "done":
                    calib_state["reprojection_error"] = event.get("reprojection_error")
                    # Update the current calibration name after successful calibration
                    update_current_calibration_name()
            rpe = event.get("reprojection_error")
            log(f"[calib] {event['status']}" + (f" — {rpe:.4f}px" if rpe else ""))
            broadcast({"calib_status": event["status"], "calibration_name": current_calibration_name, **{k: v for k, v in event.items() if k != "status"}})
        elif "error" in event:
            with calib_lock:
                calib_state["status"] = "error"
                calib_state["error"] = event["error"]
            log(f"[calib] ERROR: {event['error']}")
            broadcast({"calib_error": event["error"]})

    proc.wait()
    if proc.returncode not in (0, -15):
        log(f"[calib] Container exited with code {proc.returncode}")
    with calib_lock:
        calib_state["running"] = False


def session_calibration_path(session: str) -> Path:
    return SCREENSHOT_DIR / session / CALIBRATION_FILENAME


def load_session_calibration(session: str) -> dict | None:
    """Return the stored calibration summary for a session, or None."""
    f = session_calibration_path(session)
    if not f.exists():
        return None
    try:
        with open(f) as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


class _ImageFolderReader:
    """Minimal VideoReader stand-in: get_frames(indices) -> list[ndarray]."""

    def __init__(self, paths):
        self.paths = paths

    def get_frames(self, indices):
        return [cv2.imread(str(self.paths[i])) for i in indices]


def call_inference_server(img1_bytes: bytes, img2_bytes: bytes, calibration: dict,
                          subsample: int = 8, timeout: int = 300,
                          crop1: str = "", crop2: str = "", rotate: int = 0) -> dict:
    """POST a stereo pair + calibration to the reconstruction inference server.

    Returns the parsed JSON {timing, profile, stats, ply_base64}. Uses stdlib
    urllib so the Jetson host process needs no extra dependencies.
    """
    import io
    import uuid
    from urllib.request import Request, urlopen

    boundary = f"----sailcv{uuid.uuid4().hex}"
    crlf = b"\r\n"
    body = io.BytesIO()

    def _field(name, value):
        body.write(f"--{boundary}".encode() + crlf)
        body.write(f'Content-Disposition: form-data; name="{name}"'.encode() + crlf + crlf)
        body.write(str(value).encode() + crlf)

    def _file(name, filename, content):
        body.write(f"--{boundary}".encode() + crlf)
        body.write(
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"'.encode()
            + crlf
        )
        body.write(b"Content-Type: image/jpeg" + crlf + crlf)
        body.write(content + crlf)

    _file("image1", "cam1.jpg", img1_bytes)
    _file("image2", "cam2.jpg", img2_bytes)
    _field("calibration", json.dumps(calibration))
    _field("subsample", subsample)
    if crop1:
        _field("crop1", crop1)
    if crop2:
        _field("crop2", crop2)
    if rotate:
        _field("rotate", rotate)
    body.write(f"--{boundary}--".encode() + crlf)

    req = Request(
        f"{INFERENCE_URL}/reconstruct",
        data=body.getvalue(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class CameraStream:
    """Persistent ffmpeg RTSP reader. Keeps a single connection open and
    decodes a continuous MJPEG stream, retaining only the latest frame.

    Opening an RTSP connection + waiting for the first keyframe costs several
    seconds; doing it once (instead of per-frame) is what makes the live loop
    fast. After the first frame, get_latest() returns instantly.
    """

    def __init__(self, cam_id: str, url: str, fps: int = 10):
        self.cam_id = cam_id
        self.url = url
        self.fps = fps
        self.proc = None
        self.thread = None
        self.lock = threading.Lock()
        self.latest = None
        self.running = False

    def start(self):
        if self.running:
            return
        self.running = True
        cmd = [
            "ffmpeg", "-nostdin",
            "-rtsp_transport", "tcp",
            "-fflags", "nobuffer", "-flags", "low_delay",
            "-i", self.url,
            "-an",
            "-vf", f"fps={self.fps}",
            "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "5", "-",
        ]
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10 ** 8)
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        buf = b""
        SOI, EOI = b"\xff\xd8", b"\xff\xd9"
        try:
            while self.running:
                chunk = self.proc.stdout.read(65536)
                if not chunk:
                    break
                buf += chunk
                # Keep only the most recent complete JPEG in the buffer.
                while True:
                    start = buf.find(SOI)
                    if start < 0:
                        break
                    end = buf.find(EOI, start + 2)
                    if end < 0:
                        break
                    with self.lock:
                        self.latest = buf[start:end + 2]
                    buf = buf[end + 2:]
        finally:
            self.running = False

    def get_latest(self, timeout: float = 12.0) -> bytes:
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self.lock:
                if self.latest is not None:
                    return self.latest
            if not self.running:
                raise RuntimeError(f"cam {self.cam_id}: stream ended before a frame arrived")
            time.sleep(0.03)
        raise RuntimeError(f"cam {self.cam_id}: no frame within {timeout}s")

    def stop(self):
        self.running = False
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None
        with self.lock:
            self.latest = None


# --- Live reconstruction (inference-server based, no Docker) ---
live_recon_lock = threading.Lock()
live_recon_state = {"running": False, "frame": 0, "session": None}
live_recon_stop = threading.Event()


def live_reconstruction_loop(session: str, calibration: dict, subsample: int,
                             crop1: str = "", crop2: str = "", rotate: int = 0):
    """Continuously grab live camera frames, reconstruct via the inference
    server, and broadcast each new point cloud over SSE for the web viewer."""
    import base64
    live_recon_stop.clear()
    frame = 0
    scene = f"live_recon_{session}"
    scene_dir = OUTPUT_DIR / scene
    scene_dir.mkdir(parents=True, exist_ok=True)
    live_dir = OUTPUT_DIR / "live"
    live_dir.mkdir(parents=True, exist_ok=True)
    log(f"[live-recon] start session={session} subsample={subsample} scene={scene}")

    # Open one persistent RTSP stream per camera (connection cost paid once).
    streams = {c: CameraStream(c, CAMERAS[c]) for c in ("1", "2")}
    for s in streams.values():
        s.start()

    try:
      while not live_recon_stop.is_set():
        try:
            loop_t0 = time.time()
            results: dict = {}
            errors: dict = {}
            cap_ms: dict = {}

            def _grab(cam_id):
                t0 = time.time()
                try:
                    results[cam_id] = streams[cam_id].get_latest()
                except Exception as e:  # noqa: BLE001
                    errors[cam_id] = str(e)
                cap_ms[cam_id] = round((time.time() - t0) * 1000)

            threads = [threading.Thread(target=_grab, args=(c,)) for c in ("1", "2")]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            capture_ms = round((time.time() - loop_t0) * 1000)
            if errors:
                log(f"[live-recon] capture error: {errors}")
                broadcast({"live_recon_error": str(errors)})
                if live_recon_stop.wait(2):
                    break
                continue

            # Persist the grabbed frames so the web viewer can show the live feed.
            (live_dir / "frame1.jpg").write_bytes(results["1"])
            (live_dir / "frame2.jpg").write_bytes(results["2"])

            infer_t0 = time.time()
            result = call_inference_server(results["1"], results["2"], calibration,
                                           subsample=subsample, crop1=crop1, crop2=crop2,
                                           rotate=rotate)
            request_ms = round((time.time() - infer_t0) * 1000)
            (scene_dir / "point_cloud.ply").write_bytes(
                base64.b64decode(result["ply_base64"]))
            frame += 1
            loop_ms = round((time.time() - loop_t0) * 1000)
            with live_recon_lock:
                live_recon_state["frame"] = frame
            log(f"[live-recon] frame {frame}: loop={loop_ms}ms "
                f"capture={capture_ms}ms (cam1={cap_ms.get('1')}ms cam2={cap_ms.get('2')}ms) "
                f"request={request_ms}ms server_total={round((result.get('timing',{}).get('total') or 0)*1000)}ms")
            broadcast({
                "live_recon": "done",
                "scene": scene,
                "frame": frame,
                "stats": result.get("stats"),
                "timing": result.get("timing"),
                "loop_timing": {
                    "loop_ms": loop_ms,
                    "capture_ms": capture_ms,
                    "cam1_ms": cap_ms.get("1"),
                    "cam2_ms": cap_ms.get("2"),
                    "request_ms": request_ms,
                },
            })
        except Exception as e:  # noqa: BLE001
            log(f"[live-recon] error: {e}")
            broadcast({"live_recon_error": str(e)})
            if live_recon_stop.wait(2):
                break
    finally:
        for s in streams.values():
            s.stop()

    with live_recon_lock:
        live_recon_state["running"] = False
        live_recon_state["session"] = None
    log("[live-recon] stopped")


def _compute_session_intrinsics(intrinsic_dir: Path, cam: str, pattern_size, square_size, on_progress=None):
    """Calibrate one camera from a session's intrinsic checkerboard pairs.

    Processes one image at a time so callers can stream per-image progress.
    """
    from mv_utils.intrinsics_calibration import calibrate_camera, find_corners_in_images

    paths = sorted(intrinsic_dir.glob(f"shot_*_{cam}.jpg"))
    if len(paths) < 3:
        raise RuntimeError(f"{cam}: only {len(paths)} intrinsic shots (need >= 3)")
    reader = _ImageFolderReader(paths)
    obj_acc, img_acc = [], []
    for i in range(len(paths)):
        obj, img, ok = find_corners_in_images([i], reader, pattern_size, square_size)
        found = bool(ok)
        if found:
            obj_acc.extend(obj)
            img_acc.extend(img)
        if on_progress is not None:
            on_progress(cam, i + 1, len(paths), found, len(obj_acc))
    if len(obj_acc) < 3:
        raise RuntimeError(f"{cam}: corners found in only {len(obj_acc)}/{len(paths)} shots (need >= 3)")
    sample = cv2.imread(str(paths[0]))
    image_size = (sample.shape[1], sample.shape[0])
    K, dist, err = calibrate_camera(obj_acc, img_acc, image_size)
    return {
        "camera_matrix": K.tolist(),
        "dist_coeffs": dist.tolist(),
        "reprojection_error": float(err),
        "views_used": len(obj_acc),
        "views_total": len(paths),
    }, K, dist, image_size


def run_session_calibration(session: str):
    """Run intrinsic then extrinsic calibration on a session, write calibration.json."""
    import numpy as np
    import yaml

    def _set(status, error=None):
        with session_calib_lock:
            session_calib_state["status"] = status
            session_calib_state["error"] = error
            if status in ("intrinsics", "extrinsics", "starting"):
                session_calib_state["progress"] = None
        broadcast({"session_calib": status, "session": session, **({"error": error} if error else {})})
        log(f"[session-calib] {session}: {status}" + (f" — {error}" if error else ""))

    def _progress(phase, label, current, total, found, detected):
        p = {"phase": phase, "label": label, "current": current, "total": total,
             "found": found, "detected": detected}
        with session_calib_lock:
            session_calib_state["progress"] = p
        broadcast({"session_calib": "progress", "session": session, "progress": p})

    try:
        session_dir = SCREENSHOT_DIR / session
        intrinsic_dir = session_dir / "intrinsic"
        extrinsic_dir = session_dir / "extrinsic"

        cb = yaml.safe_load(CHECKERBOARD_SPECS_PATH.read_text())
        pattern_size = (cb["inner_corners_x"], cb["inner_corners_y"])
        square_size = cb["square_size_mm"]

        _set("intrinsics")

        def _intr_progress(cam, current, total, found, detected):
            _progress("intrinsics", f"{cam} corner detection", current, total, found, detected)

        intr1, K1, d1, image_size = _compute_session_intrinsics(
            intrinsic_dir, "cam1", pattern_size, square_size, _intr_progress)
        intr2, K2, d2, _ = _compute_session_intrinsics(
            intrinsic_dir, "cam2", pattern_size, square_size, _intr_progress)
        log(f"[session-calib] intrinsics cam1 err={intr1['reprojection_error']:.4f}px "
            f"({intr1['views_used']}/{intr1['views_total']}), "
            f"cam2 err={intr2['reprojection_error']:.4f}px ({intr2['views_used']}/{intr2['views_total']})")

        _set("extrinsics")
        from mv_utils.extrinsics_calibration import CharucoDetector, calibrate_stereo_many

        detector = CharucoDetector(config_path=str(CHARUCO_SPECS_PATH))
        cam1_paths = sorted(extrinsic_dir.glob("shot_*_cam1.jpg"))
        obj_list, ip1_list, ip2_list = [], [], []
        for idx, p1 in enumerate(cam1_paths):
            p2 = p1.with_name(p1.name.replace("_cam1.jpg", "_cam2.jpg"))
            found = False
            if p2.exists():
                p3d, q1, q2 = detector.get_correspondences(cv2.imread(str(p1)), cv2.imread(str(p2)))
                if p3d is not None:
                    obj_list.append(p3d)
                    ip1_list.append(q1)
                    ip2_list.append(q2)
                    found = True
            _progress("extrinsics", "charuco matching", idx + 1, len(cam1_paths), found, len(obj_list))
        try:
            detector.cleanup()
        except Exception:
            pass

        if len(obj_list) < 1:
            raise RuntimeError(f"no charuco correspondences from {len(cam1_paths)} extrinsic pairs")

        res = calibrate_stereo_many(obj_list, ip1_list, ip2_list, K1, d1, K2, d2, image_size)
        T = np.array(res["translation_vector"])
        baseline = float(np.linalg.norm(T))

        calibration = {
            "created": datetime.now().isoformat(timespec="seconds"),
            "session": session,
            "image_size": [int(image_size[0]), int(image_size[1])],
            "intrinsics": {"cam1": intr1, "cam2": intr2},
            "extrinsics": {
                "rotation_matrix": res["rotation_matrix"],
                "translation_vector": res["translation_vector"],
                "baseline_m": baseline,
                "reprojection_error": float(res["reprojection_error"]),
                "pairs_used": len(obj_list),
                "pairs_total": len(cam1_paths),
            },
        }
        with open(session_calibration_path(session), "w") as fh:
            json.dump(calibration, fh, indent=2)
        log(f"[session-calib] {session}: done — baseline={baseline:.4f}m "
            f"stereo_err={res['reprojection_error']:.4f} ({len(obj_list)}/{len(cam1_paths)} pairs)")
        with session_calib_lock:
            session_calib_state["status"] = "done"
        broadcast({"session_calib": "done", "session": session, "calibration": calibration})
    except Exception as e:
        _set("error", str(e))
    finally:
        with session_calib_lock:
            session_calib_state["running"] = False


def _load_default_intrinsics():
    """Load the committed per-camera default intrinsics (camera_matrix, dist_coeffs)."""
    def _read(path: Path):
        if not path.exists():
            raise RuntimeError(f"default intrinsics not found: {path}")
        with open(path) as fh:
            d = json.load(fh)
        K = np.array(d["camera_matrix"], dtype=np.float64)
        dist = np.array(d["dist_coeffs"], dtype=np.float64)
        return K, dist, d

    return _read(DEFAULT_INTRINSICS_CAM1), _read(DEFAULT_INTRINSICS_CAM2)


def run_session_extrinsic_calibration(session: str):
    """Extrinsic-only calibration: known default intrinsics + Charuco stereo solve.

    Skips per-camera intrinsic calibration entirely (the cameras are already
    calibrated). Only the stereo extrinsics (R, T) are solved from the session's
    Charuco extrinsic pairs, with cv2.CALIB_FIX_INTRINSIC holding K/dist fixed.
    Writes the same nested calibration.json schema the rest of the viewer reads.
    """
    def _set(status, error=None):
        with session_calib_lock:
            session_calib_state["status"] = status
            session_calib_state["error"] = error
            if status in ("extrinsics", "starting"):
                session_calib_state["progress"] = None
        broadcast({"session_calib": status, "session": session, **({"error": error} if error else {})})
        log(f"[session-extr] {session}: {status}" + (f" — {error}" if error else ""))

    def _progress(phase, label, current, total, found, detected):
        p = {"phase": phase, "label": label, "current": current, "total": total,
             "found": found, "detected": detected}
        with session_calib_lock:
            session_calib_state["progress"] = p
        broadcast({"session_calib": "progress", "session": session, "progress": p})

    try:
        extrinsic_dir = SCREENSHOT_DIR / session / "extrinsic"

        (K1, d1, raw1), (K2, d2, raw2) = _load_default_intrinsics()

        _set("extrinsics")
        from mv_utils.extrinsics_calibration import CharucoDetector, calibrate_stereo_many

        cam1_paths = sorted(extrinsic_dir.glob("shot_*_cam1.jpg"))
        if not cam1_paths:
            raise RuntimeError(f"no extrinsic shots in {extrinsic_dir}")

        sample = cv2.imread(str(cam1_paths[0]))
        if sample is None:
            raise RuntimeError(f"could not read {cam1_paths[0]}")
        image_size = (sample.shape[1], sample.shape[0])

        # Sanity: intrinsics are only valid at the resolution they were calibrated at.
        intr_w = raw1.get("image_size", [None, None])[0] if isinstance(raw1.get("image_size"), list) else None
        if intr_w is not None and intr_w != image_size[0]:
            log(f"[session-extr] WARNING: shot size {image_size} != intrinsics size {raw1.get('image_size')}")

        detector = CharucoDetector(config_path=str(CHARUCO_SPECS_PATH))
        obj_list, ip1_list, ip2_list = [], [], []
        for idx, p1 in enumerate(cam1_paths):
            p2 = p1.with_name(p1.name.replace("_cam1.jpg", "_cam2.jpg"))
            found = False
            if p2.exists():
                p3d, q1, q2 = detector.get_correspondences(cv2.imread(str(p1)), cv2.imread(str(p2)))
                if p3d is not None:
                    obj_list.append(p3d)
                    ip1_list.append(q1)
                    ip2_list.append(q2)
                    found = True
            _progress("extrinsics", "charuco matching", idx + 1, len(cam1_paths), found, len(obj_list))
        try:
            detector.cleanup()
        except Exception:
            pass

        if len(obj_list) < 1:
            raise RuntimeError(f"no charuco correspondences from {len(cam1_paths)} extrinsic pairs")

        res = calibrate_stereo_many(obj_list, ip1_list, ip2_list, K1, d1, K2, d2, image_size)
        T = np.array(res["translation_vector"])
        baseline = float(np.linalg.norm(T))

        intr1 = {"camera_matrix": K1.tolist(), "dist_coeffs": d1.tolist(),
                 "reprojection_error": raw1.get("reprojection_error"), "source": "default"}
        intr2 = {"camera_matrix": K2.tolist(), "dist_coeffs": d2.tolist(),
                 "reprojection_error": raw2.get("reprojection_error"), "source": "default"}

        calibration = {
            "created": datetime.now().isoformat(timespec="seconds"),
            "session": session,
            "image_size": [int(image_size[0]), int(image_size[1])],
            "intrinsics_source": "default",
            "intrinsics": {"cam1": intr1, "cam2": intr2},
            "extrinsics": {
                "rotation_matrix": res["rotation_matrix"],
                "translation_vector": res["translation_vector"],
                "baseline_m": baseline,
                "reprojection_error": float(res["reprojection_error"]),
                "pairs_used": len(obj_list),
                "pairs_total": len(cam1_paths),
            },
        }
        with open(session_calibration_path(session), "w") as fh:
            json.dump(calibration, fh, indent=2)
        log(f"[session-extr] {session}: done — baseline={baseline:.4f}m "
            f"stereo_err={res['reprojection_error']:.4f} ({len(obj_list)}/{len(cam1_paths)} pairs)")
        with session_calib_lock:
            session_calib_state["status"] = "done"
        broadcast({"session_calib": "done", "session": session, "calibration": calibration})
    except Exception as e:
        _set("error", str(e))
    finally:
        with session_calib_lock:
            session_calib_state["running"] = False


# --- Camera streaming ---

def mjpeg_frames(cam_id: str):
    """Yield raw JPEG bytes by decoding RTSP via ffmpeg."""
    url = CAMERAS[cam_id]
    cmd = [
        "ffmpeg",
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-probesize", "32",
        "-analyzeduration", "0",
        "-rtsp_transport", "udp",
        "-i", url,
        "-f", "image2pipe", "-vcodec", "mjpeg",
        "-q:v", "5", "-r", "15",
        "-flush_packets", "1",
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    buf = b""
    try:
        while True:
            chunk = proc.stdout.read(8192)
            if not chunk:
                break
            buf += chunk
            while True:
                start = buf.find(b"\xff\xd8")
                if start == -1:
                    buf = b""
                    break
                end = buf.find(b"\xff\xd9", start + 2)
                if end == -1:
                    buf = buf[start:]
                    break
                frame = buf[start:end + 2]
                buf = buf[end + 2:]
                yield frame
    finally:
        proc.kill()
        proc.wait()


# --- HTTP Handler ---

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"{self.address_string()} - {fmt % args}")

    def send_json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def send_bytes(self, code, content_type, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = unquote(self.path.split("?")[0])

        if path in ("/", "/index.html"):
            self.serve_file(STATIC_DIR / "index.html", "text/html")

        elif path == "/screenshots.html":
            self.serve_file(STATIC_DIR / "screenshots.html", "text/html")

        elif path in ("/3d-reconstruction", "/reconstruction.html"):
            self.serve_file(STATIC_DIR / "reconstruction.html", "text/html")

        elif path == "/scenes":
            scenes = []
            if OUTPUT_DIR.exists():
                scenes = sorted(
                    d.name for d in OUTPUT_DIR.iterdir()
                    if d.is_dir() and any(d.glob("*.ply"))
                )
            self.send_json(200, {"scenes": scenes})

        elif path.startswith("/ply/"):
            scene = path[5:]
            if ".." in scene or "/" in scene:
                self.send_bytes(400, "text/plain", b"Invalid scene")
                return
            scene_dir = OUTPUT_DIR / scene
            ply_files = list(scene_dir.glob("*.ply"))
            if not ply_files:
                self.send_bytes(404, "text/plain", b"No PLY found")
                return
            data = ply_files[0].read_bytes()
            self.send_bytes(200, "application/octet-stream", data)

        elif path == "/status":
            with state_lock:
                self.send_json(200, dict(state))

        elif path.startswith("/frame/"):
            cam_id = path[7:]
            if cam_id not in ("1", "2"):
                self.send_bytes(404, "text/plain", b"Unknown camera")
                return
            frame_file = OUTPUT_DIR / "live" / f"frame{cam_id}.jpg"
            if not frame_file.exists():
                self.send_bytes(404, "text/plain", b"No frame yet")
                return
            self.send_bytes(200, "image/jpeg", frame_file.read_bytes())

        elif path == "/calibrate/status":
            with calib_lock:
                s = dict(calib_state)
            s["pair_count"] = calib_pair_count
            self.send_json(200, s)

        elif path == "/calibrate/preview":
            # Serve last captured pair side-by-side as JPEG
            with calib_pair_lock:
                n = calib_pair_count
            if n == 0:
                self.send_bytes(404, "text/plain", b"No pairs yet")
                return
            pair_dir = CALIB_PAIRS_DIR / f"pair_{n:03d}"
            f1 = pair_dir / "cam1.jpg"
            f2 = pair_dir / "cam2.jpg"
            if not f1.exists() or not f2.exists():
                self.send_bytes(404, "text/plain", b"Pair not found")
                return
            import PIL.Image, io
            im1 = PIL.Image.open(f1).convert("RGB")
            im2 = PIL.Image.open(f2).convert("RGB")
            h = min(im1.height, im2.height)
            scale = h / im1.height
            w1 = int(im1.width * scale)
            r1 = im1.resize((w1, h), PIL.Image.LANCZOS)
            scale2 = h / im2.height
            w2 = int(im2.width * scale2)
            r2 = im2.resize((w2, h), PIL.Image.LANCZOS)
            side_by_side = PIL.Image.new("RGB", (w1 + w2, h))
            side_by_side.paste(r1, (0, 0))
            side_by_side.paste(r2, (w1, 0))
            buf = io.BytesIO()
            side_by_side.save(buf, format="JPEG", quality=85)
            self.send_bytes(200, "image/jpeg", buf.getvalue())

        elif path == "/pointcloud":
            f = OUTPUT_DIR / "live" / "pointcloud.json"
            if not f.exists():
                self.send_json(404, {"error": "No point cloud yet"})
                return
            self.send_bytes(200, "application/json", f.read_bytes())

        elif path == "/params":
            f = OUTPUT_DIR / "live" / "params.json"
            if f.exists():
                self.send_bytes(200, "application/json", f.read_bytes())
            else:
                self.send_json(200, {"subsample": 16})

        elif path == "/cameras":
            try:
                payload = compute_camera_frusta()
            except Exception as e:
                self.send_json(500, {"error": f"frustum compute failed: {e}"})
                return
            if payload is None:
                calib_path = get_active_calibration_path()
                self.send_json(404, {"error": f"No calibration available (checked: {calib_path})"})
                return
            self.send_json(200, payload)

        elif path == "/composite":
            f = OUTPUT_DIR / "live" / "composite.jpg"
            if not f.exists():
                self.send_bytes(404, "text/plain", b"No composite yet")
                return
            self.send_bytes(200, "image/jpeg", f.read_bytes())

        elif path == "/matches":
            matches_file = OUTPUT_DIR / "live" / "matches.json"
            if not matches_file.exists():
                self.send_json(404, {"error": "No matches yet"})
                return
            self.send_bytes(200, "application/json", matches_file.read_bytes())

        elif path.startswith("/stream/"):
            cam_id = path[8:]
            if cam_id not in CAMERAS:
                self.send_bytes(404, "text/plain", b"Unknown camera")
                return
            boundary = b"frame"
            self.send_response(200)
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try:
                for frame in mjpeg_frames(cam_id):
                    header = (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n"
                    )
                    self.wfile.write(header + frame + b"\r\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        elif path == "/sessions":
            with screenshot_lock:
                current = active_session
            self.send_json(200, {"sessions": list_sessions(), "active": current})

        elif path == "/session/pairs":
            from urllib.parse import parse_qs, urlparse
            qs = parse_qs(urlparse(self.path).query)
            session = qs.get("session", [None])[0]
            if session is None:
                with screenshot_lock:
                    session = active_session
            if session is None:
                self.send_json(200, {"session": None, "pairs": {k: [] for k in SHOT_KINDS}})
                return
            if not _safe_segment(session):
                self.send_json(400, {"error": "Invalid session"})
                return
            self.send_json(200, {"session": session, "pairs": list_all_pairs(session)})

        elif path == "/session/cameras":
            from urllib.parse import parse_qs, urlparse
            qs = parse_qs(urlparse(self.path).query)
            session = qs.get("session", [None])[0]
            if session is None or not _safe_segment(session):
                self.send_json(400, {"error": "Invalid session"})
                return
            calibration = load_session_calibration(session)
            if calibration is None:
                self.send_json(404, {"error": f"No calibration for session {session}"})
                return
            kind = qs.get("kind", ["normal"])[0]
            if kind not in SHOT_KINDS:
                self.send_json(400, {"error": f"Invalid kind '{kind}'"})
                return
            ts = qs.get("timestamp", [None])[0]
            live = qs.get("live", ["0"])[0] in ("1", "true")
            img1 = img2 = None
            if live:
                # Use the frames the live loop just grabbed (same images fed to
                # reconstruction), not the stored session pair.
                live_dir = OUTPUT_DIR / "live"
                img1 = _load_frame_rgb(live_dir / "frame1.jpg")
                img2 = _load_frame_rgb(live_dir / "frame2.jpg")
            else:
                pairs = list_pairs(session, kind)
                if pairs:
                    pair = next((p for p in pairs if p["timestamp"] == ts), pairs[-1]) if ts else pairs[-1]
                    kind_dir = SCREENSHOT_DIR / session / kind
                    img1 = _load_frame_rgb(kind_dir / pair["cam1"])
                    img2 = _load_frame_rgb(kind_dir / pair["cam2"])
            try:
                payload = build_frusta(_flatten_calibration(calibration), img1, img2)
            except Exception as e:
                self.send_json(500, {"error": f"frustum compute failed: {e}"})
                return
            self.send_json(200, payload)

        elif path == "/session/calibration":
            from urllib.parse import parse_qs, urlparse
            session = parse_qs(urlparse(self.path).query).get("session", [None])[0]
            if session is None or not _safe_segment(session):
                self.send_json(400, {"error": "Invalid session"})
                return
            with session_calib_lock:
                job = dict(session_calib_state)
            running = job["running"] and job["session"] == session
            self.send_json(200, {
                "session": session,
                "calibration": load_session_calibration(session),
                "running": running,
                "status": job["status"] if job["session"] == session else None,
                "error": job["error"] if job["session"] == session else None,
                "progress": job["progress"] if job["session"] == session else None,
            })

        elif path.startswith("/shot/"):
            rest = path[len("/shot/"):]
            parts = rest.split("/")
            if len(parts) != 3 or not all(_safe_segment(p) for p in parts):
                self.send_bytes(400, "text/plain", b"Invalid path")
                return
            session, kind, fname = parts
            if kind not in SHOT_KINDS:
                self.send_bytes(400, "text/plain", b"Invalid kind")
                return
            f = SCREENSHOT_DIR / session / kind / fname
            if not f.exists():
                self.send_bytes(404, "text/plain", b"Not found")
                return
            self.send_bytes(200, "image/jpeg", f.read_bytes())

        elif path == "/logs":
            q: queue.Queue = queue.Queue(maxsize=200)
            with log_lock:
                log_queues.append(q)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try:
                while True:
                    try:
                        msg = q.get(timeout=15)
                        self.wfile.write(msg.encode())
                        self.wfile.flush()
                    except queue.Empty:
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with log_lock:
                    if q in log_queues:
                        log_queues.remove(q)

        elif path == "/events":
            q: queue.Queue = queue.Queue(maxsize=32)
            with sse_lock:
                sse_queues.append(q)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try:
                while True:
                    try:
                        msg = q.get(timeout=15)
                        self.wfile.write(msg.encode())
                        self.wfile.flush()
                    except queue.Empty:
                        # keepalive
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with sse_lock:
                    if q in sse_queues:
                        sse_queues.remove(q)

        else:
            self.send_bytes(404, "text/plain", b"Not found")

    def do_POST(self):
        path = unquote(self.path.split("?")[0])

        if path.startswith("/start/"):
            scene = path[7:]
            if ".." in scene or "/" in scene:
                self.send_json(400, {"error": "Invalid scene"})
                return
            with state_lock:
                if state["running"]:
                    self.send_json(409, {"error": "Already running"})
                    return
                state["running"] = True
                state["scene"] = scene
                state["frame_count"] = 0
                state["last_recon_ms"] = None
            t = threading.Thread(target=reconstruction_loop, args=(scene,), daemon=True)
            t.start()
            self.send_json(200, {"started": scene})

        elif path == "/stop":
            stop_event.set()
            subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True)
            self.send_json(200, {"stopped": True})

        elif path == "/params":
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length)) if length else {}
                subsample = max(1, min(16, int(body.get("subsample", 16))))
            except Exception as e:
                self.send_json(400, {"error": f"bad payload: {e}"})
                return
            params_dir = OUTPUT_DIR / "live"
            params_dir.mkdir(parents=True, exist_ok=True)
            with open(params_dir / "params.json", "w") as f:
                json.dump({"subsample": subsample}, f)
            log(f"[params] subsample={subsample}")
            self.send_json(200, {"subsample": subsample})

        elif path == "/calibrate/capture":
            # Grab one frame from each camera simultaneously, save as a pair
            global calib_pair_count
            results = {}
            errors  = {}
            def _grab(cam_id):
                try:
                    results[cam_id] = capture_frame_jpeg(cam_id)
                except Exception as e:
                    errors[cam_id] = str(e)
            threads = [threading.Thread(target=_grab, args=(c,)) for c in ("1", "2")]
            for t in threads: t.start()
            for t in threads: t.join()
            if errors:
                self.send_json(500, {"error": str(errors)})
                return
            with calib_pair_lock:
                calib_pair_count += 1
                n = calib_pair_count
            pair_dir = CALIB_PAIRS_DIR / f"pair_{n:03d}"
            pair_dir.mkdir(parents=True, exist_ok=True)
            (pair_dir / "cam1.jpg").write_bytes(results["1"])
            (pair_dir / "cam2.jpg").write_bytes(results["2"])
            log(f"[calib] Captured pair {n}")
            broadcast({"calib_pair": n})
            self.send_json(200, {"pair": n})

        elif path == "/session/new":
            global active_session
            name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            (SCREENSHOT_DIR / name).mkdir(parents=True, exist_ok=True)
            with screenshot_lock:
                active_session = name
            log(f"[shot] New session {name}")
            broadcast({"session": name})
            self.send_json(200, {"session": name})

        elif path == "/session/calibrate":
            from urllib.parse import parse_qs, urlparse
            session = parse_qs(urlparse(self.path).query).get("session", [None])[0]
            if session is None:
                with screenshot_lock:
                    session = active_session
            if session is None or not _safe_segment(session):
                self.send_json(400, {"error": "Invalid session"})
                return
            if not (SCREENSHOT_DIR / session).is_dir():
                self.send_json(404, {"error": f"Session not found: {session}"})
                return
            with session_calib_lock:
                if session_calib_state["running"]:
                    self.send_json(409, {"error": f"Calibration already running for {session_calib_state['session']}"})
                    return
                session_calib_state.update(running=True, session=session, status="starting", error=None, progress=None)
            threading.Thread(target=run_session_calibration, args=(session,), daemon=True).start()
            self.send_json(200, {"started": True, "session": session})

        elif path == "/session/calibrate-extrinsic":
            # Extrinsic-only calibration using the committed default intrinsics.
            from urllib.parse import parse_qs, urlparse
            session = parse_qs(urlparse(self.path).query).get("session", [None])[0]
            if session is None:
                with screenshot_lock:
                    session = active_session
            if session is None or not _safe_segment(session):
                self.send_json(400, {"error": "Invalid session"})
                return
            if not (SCREENSHOT_DIR / session).is_dir():
                self.send_json(404, {"error": f"Session not found: {session}"})
                return
            if not (DEFAULT_INTRINSICS_CAM1.exists() and DEFAULT_INTRINSICS_CAM2.exists()):
                self.send_json(400, {"error": "Default camera intrinsics not found on server"})
                return
            with session_calib_lock:
                if session_calib_state["running"]:
                    self.send_json(409, {"error": f"Calibration already running for {session_calib_state['session']}"})
                    return
                session_calib_state.update(running=True, session=session, status="starting", error=None, progress=None)
            threading.Thread(target=run_session_extrinsic_calibration, args=(session,), daemon=True).start()
            self.send_json(200, {"started": True, "session": session})

        elif path == "/session/reconstruct":
            # Send a stereo pair + calibration to the inference server, store the PLY.
            import base64
            from urllib.error import URLError
            from urllib.parse import parse_qs, urlparse
            qs = parse_qs(urlparse(self.path).query)
            session = qs.get("session", [None])[0]
            if session is None or not _safe_segment(session):
                self.send_json(400, {"error": "Invalid session"})
                return
            calibration = load_session_calibration(session)
            if calibration is None:
                self.send_json(404, {"error": f"No calibration for session {session}"})
                return
            # Pick the requested pair, else the latest 'normal' pair.
            kind = qs.get("kind", ["normal"])[0]
            if kind not in SHOT_KINDS:
                self.send_json(400, {"error": f"Invalid kind '{kind}'"})
                return
            ts = qs.get("timestamp", [None])[0]
            pairs = list_pairs(session, kind)
            if not pairs:
                self.send_json(404, {"error": f"No {kind} pairs in session {session}"})
                return
            pair = next((p for p in pairs if p["timestamp"] == ts), pairs[-1]) if ts else pairs[-1]
            kind_dir = SCREENSHOT_DIR / session / kind
            img1 = (kind_dir / pair["cam1"]).read_bytes()
            img2 = (kind_dir / pair["cam2"]).read_bytes()
            subsample = int(qs.get("subsample", ["8"])[0])
            crop1 = qs.get("crop1", [""])[0]
            crop2 = qs.get("crop2", [""])[0]
            try:
                rotate = int(qs.get("rotate", ["0"])[0])
            except ValueError:
                rotate = 0
            try:
                result = call_inference_server(img1, img2, calibration, subsample=subsample,
                                               crop1=crop1, crop2=crop2, rotate=rotate)
            except URLError as e:
                self.send_json(502, {"error": f"Inference server unreachable at {INFERENCE_URL}: {e}"})
                return
            except Exception as e:
                self.send_json(500, {"error": f"Inference failed: {e}"})
                return
            scene = f"recon_{session}"
            scene_dir = OUTPUT_DIR / scene
            scene_dir.mkdir(parents=True, exist_ok=True)
            (scene_dir / "point_cloud.ply").write_bytes(
                base64.b64decode(result["ply_base64"]))
            response = {
                "session": session,
                "scene": scene,
                "pair": pair["timestamp"],
                "timing": result.get("timing"),
                "profile": result.get("profile"),
                "stats": result.get("stats"),
            }
            log(f"[reconstruct] {session} pair {pair['timestamp']}: "
                f"{result.get('stats', {}).get('points_kept')} pts "
                f"in {result.get('timing', {}).get('total')}s")
            broadcast({"reconstruct": "done", **response})
            self.send_json(200, response)

        elif path == "/live-reconstruct/start":
            from urllib.parse import parse_qs, urlparse
            qs = parse_qs(urlparse(self.path).query)
            session = qs.get("session", [None])[0]
            if session is None or not _safe_segment(session):
                self.send_json(400, {"error": "Invalid session"})
                return
            calibration = load_session_calibration(session)
            if calibration is None:
                self.send_json(404, {"error": f"No calibration for session {session}"})
                return
            try:
                subsample = max(1, min(16, int(qs.get("subsample", ["8"])[0])))
            except ValueError:
                subsample = 8
            crop1 = qs.get("crop1", [""])[0]
            crop2 = qs.get("crop2", [""])[0]
            try:
                rotate = int(qs.get("rotate", ["0"])[0])
            except ValueError:
                rotate = 0
            with live_recon_lock:
                if live_recon_state["running"]:
                    self.send_json(409, {"error": "Live reconstruction already running"})
                    return
                live_recon_state.update(running=True, session=session, frame=0)
            threading.Thread(
                target=live_reconstruction_loop,
                args=(session, calibration, subsample, crop1, crop2, rotate),
                daemon=True,
            ).start()
            self.send_json(200, {"started": True, "session": session,
                                 "subsample": subsample, "scene": f"live_recon_{session}"})

        elif path == "/live-reconstruct/stop":
            live_recon_stop.set()
            self.send_json(200, {"stopped": True})

        elif path == "/screenshot":
            # take_stereo_screenshot: grab both cameras simultaneously into the active session
            from urllib.parse import parse_qs, urlparse
            qs = parse_qs(urlparse(self.path).query)
            kind = qs.get("kind", ["normal"])[0]
            if kind not in SHOT_KINDS:
                self.send_json(400, {"error": f"Invalid kind '{kind}', expected one of {SHOT_KINDS}"})
                return
            requested = qs.get("session", [None])[0]
            if requested is not None and not _safe_segment(requested):
                self.send_json(400, {"error": "Invalid session"})
                return
            with screenshot_lock:
                if requested is not None:
                    session = requested
                else:
                    session = active_session
                    if session is None:
                        session = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                active_session = session
            kind_dir = SCREENSHOT_DIR / session / kind
            kind_dir.mkdir(parents=True, exist_ok=True)
            results, errors = {}, {}
            def _grab(cam_id):
                try:
                    results[cam_id] = capture_frame_jpeg(cam_id)
                except Exception as e:
                    errors[cam_id] = str(e)
            threads = [threading.Thread(target=_grab, args=(c,)) for c in ("1", "2")]
            for t in threads: t.start()
            for t in threads: t.join()
            if errors:
                self.send_json(500, {"error": str(errors)})
                return
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            (kind_dir / f"shot_{ts}_cam1.jpg").write_bytes(results["1"])
            (kind_dir / f"shot_{ts}_cam2.jpg").write_bytes(results["2"])
            log(f"[shot] Captured {kind} {ts} in {session}")
            broadcast({"screenshot": ts, "session": session, "kind": kind})
            self.send_json(200, {"session": session, "timestamp": ts, "kind": kind})

        elif path == "/calibrate/reset":
            import shutil
            if CALIB_PAIRS_DIR.exists():
                shutil.rmtree(CALIB_PAIRS_DIR)
            CALIB_PAIRS_DIR.mkdir(parents=True, exist_ok=True)
            with calib_pair_lock:
                calib_pair_count = 0
            log("[calib] Session reset")
            self.send_json(200, {"reset": True})

        elif path == "/calibrate/run":
            from urllib.parse import parse_qs, urlparse
            qs = parse_qs(urlparse(self.path).query)
            square_mm = float(qs.get("square_mm", ["30.78"])[0])
            with calib_lock:
                if calib_state["running"]:
                    self.send_json(409, {"error": "Calibration already running"})
                    return
                if calib_pair_count < 8:
                    self.send_json(400, {"error": f"Need at least 8 pairs, have {calib_pair_count}"})
                    return
                calib_state["running"] = True
                calib_state["status"] = "starting"
                calib_state["reprojection_error"] = None
                calib_state["error"] = None
            t = threading.Thread(target=run_calibration_docker, args=(square_mm,), daemon=True)
            t.start()
            self.send_json(200, {"started": True, "pairs": calib_pair_count})

        elif path == "/calibrate/stop":
            subprocess.run(["docker", "rm", "-f", CALIB_CONTAINER], capture_output=True)
            with calib_lock:
                calib_state["running"] = False
                calib_state["status"] = "stopped"
            self.send_json(200, {"stopped": True})

        elif path == "/calibrations":
            # List all available calibrations
            calib_files = sorted(DEFAULT_CALIB_DIR.glob("calibration_*.json"), reverse=True)
            calibrations = [
                {
                    "name": f.stem.replace("calibration_", ""),
                    "path": str(f),
                    "modified": f.stat().st_mtime,
                }
                for f in calib_files
            ]
            self.send_json(200, {"calibrations": calibrations, "current": current_calibration_name})

        elif path == "/calibration/current":
            # Get current calibration name
            self.send_json(200, {"current": current_calibration_name})

        elif path == "/calibration/select":
            # Select a calibration by name
            if self.command != "POST":
                self.send_json(405, {"error": "Method not allowed"})
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                calib_name = body.get("name")
                if not calib_name:
                    self.send_json(400, {"error": "Missing 'name' in request"})
                    return

                # Find the calibration file
                calib_file = DEFAULT_CALIB_DIR / f"calibration_{calib_name}.json"
                if not calib_file.exists():
                    self.send_json(404, {"error": f"Calibration not found: {calib_name}"})
                    return

                # Verify it's valid JSON
                with open(calib_file) as f:
                    json.load(f)

                with calibration_lock:
                    update_current_calibration_name()

                self.send_json(200, {"current": current_calibration_name, "selected": calib_name})
            except Exception as e:
                self.send_json(400, {"error": str(e)})

        else:
            self.send_bytes(404, "text/plain", b"Not found")

    def serve_file(self, path: Path, content_type: str):
        try:
            self.send_bytes(200, content_type, path.read_bytes())
        except FileNotFoundError:
            self.send_bytes(404, "text/plain", b"File not found")


if __name__ == "__main__":
    print(f"PLY viewer at http://0.0.0.0:{PORT}")
    print(f"Project dir: {PROJECT_DIR}")
    print(f"Output dir:  {OUTPUT_DIR}")
    update_current_calibration_name()
    print(f"Current calibration: {current_calibration_name}")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
