"""PLY point cloud viewer with continuous reconstruction loop."""
import json
import os
import queue
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

PROJECT_DIR = Path(os.getenv("PROJECT_DIR", Path(__file__).parent.parent)).resolve()
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_DIR / "output")).resolve()
STATIC_DIR = Path(__file__).parent / "static"
PORT = int(os.getenv("PORT", 7863))
DOCKER_IMAGE = os.getenv("DOCKER_IMAGE", "docker-sailcv-3d-reconstruction:latest")

CAMERAS = {
    "1": "rtsp://admin:123456@192.168.1.105/cam/realmonitor?channel=1&subtype=0",
    "2": "rtsp://admin:123456@192.168.1.141/cam/realmonitor?channel=1&subtype=0",
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
        "-e", "DEVICE=cuda",
        "-e", "CUDA_MEMORY_FRACTION=0.6",
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
            with state_lock:
                state["frame_count"] = frame
                state["last_recon_ms"] = ms
            log(f"[loop] Frame {frame} done in {ms}ms")
            broadcast({"frame": frame, "recon_ms": ms, "scene": scene})

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
            rpe = event.get("reprojection_error")
            log(f"[calib] {event['status']}" + (f" — {rpe:.4f}px" if rpe else ""))
            broadcast({"calib_status": event["status"], **{k: v for k, v in event.items() if k != "status"}})
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
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
