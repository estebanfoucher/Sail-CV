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

PROJECT_DIR = Path(os.getenv("PROJECT_DIR", Path(__file__).parent.parent))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_DIR / "output"))
STATIC_DIR = Path(__file__).parent / "static"
PORT = int(os.getenv("PORT", 7863))
DOCKER_IMAGE = os.getenv("DOCKER_IMAGE", "docker-sailcv-3d-reconstruction:latest")

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


def run_reconstruction(scene: str):
    """Run one reconstruction pass via Docker. Returns elapsed ms."""
    t0 = time.monotonic()
    result = subprocess.run(
        [
            "docker", "run", "--rm",
            "--runtime=nvidia",
            "-w", "/app",
            "-v", f"{PROJECT_DIR}/assets/reconstruction:/app/assets/reconstruction:ro",
            "-v", f"{PROJECT_DIR}/checkpoints:/app/checkpoints:ro",
            "-v", f"{PROJECT_DIR}/src/reconstruction:/app/src/reconstruction:ro",
            "-v", f"{OUTPUT_DIR}:/app/output:rw",
            "-e", "DEVICE=cuda",
            DOCKER_IMAGE,
            "python3", "src/reconstruction/reconstruct_pair.py", "--scene", scene,
        ],
        capture_output=True,
        text=True,
    )
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-500:] if result.stderr else "docker failed")
    return elapsed_ms


def reconstruction_loop(scene: str):
    global state
    stop_event.clear()
    while not stop_event.is_set():
        try:
            ms = run_reconstruction(scene)
            with state_lock:
                state["frame_count"] += 1
                state["last_recon_ms"] = ms
                frame = state["frame_count"]
            broadcast({"frame": frame, "recon_ms": ms, "scene": scene})
        except Exception as e:
            broadcast({"error": str(e)})
            time.sleep(2)

    with state_lock:
        state["running"] = False
        state["scene"] = None


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
