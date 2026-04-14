"""Lightweight PLY point cloud viewer — stdlib only, no dependencies."""
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import unquote

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/app/output"))
STATIC_DIR = Path(__file__).parent / "static"
PORT = int(os.getenv("PORT", 7863))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"{self.address_string()} - {fmt % args}")

    def send(self, code, content_type, body):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = unquote(self.path.split("?")[0])

        if path == "/" or path == "/index.html":
            self.serve_file(STATIC_DIR / "index.html", "text/html")

        elif path == "/scenes":
            scenes = []
            if OUTPUT_DIR.exists():
                scenes = sorted(
                    d.name for d in OUTPUT_DIR.iterdir()
                    if d.is_dir() and any(d.glob("*.ply"))
                )
            self.send(200, "application/json", json.dumps({"scenes": scenes}))

        elif path.startswith("/ply/"):
            scene = path[5:]
            if ".." in scene or "/" in scene:
                self.send(400, "text/plain", b"Invalid scene")
                return
            scene_dir = OUTPUT_DIR / scene
            ply_files = list(scene_dir.glob("*.ply"))
            if not ply_files:
                self.send(404, "text/plain", f"No PLY found for '{scene}'")
                return
            self.serve_file(ply_files[0], "application/octet-stream")

        else:
            self.send(404, "text/plain", b"Not found")

    def serve_file(self, path: Path, content_type: str):
        try:
            data = path.read_bytes()
            self.send(200, content_type, data)
        except FileNotFoundError:
            self.send(404, "text/plain", b"File not found")


if __name__ == "__main__":
    output_dir = OUTPUT_DIR
    output_dir_display = str(output_dir)
    print(f"PLY viewer running at http://0.0.0.0:{PORT}")
    print(f"Serving PLY files from: {output_dir_display}")
    HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
