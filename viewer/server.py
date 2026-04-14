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


CONTAINER_NAME = "sailcv-recon"


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
