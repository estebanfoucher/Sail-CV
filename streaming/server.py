import os
import subprocess
from flask import Flask, Response, send_from_directory

app = Flask(__name__)

GST = "/Library/Frameworks/GStreamer.framework/Commands/gst-launch-1.0"
GST_ENV = {**os.environ, "GST_PLUGIN_PATH": "/Library/Frameworks/GStreamer.framework/Libraries"}

CAMERAS = {
    "1": "rtsp://admin:123456@192.168.1.105/cam/realmonitor?channel=1&subtype=0",
    "2": "rtsp://admin:123456@192.168.1.141/cam/realmonitor?channel=1&subtype=0",
}


def gst_pipeline(rtsp_url: str) -> str:
    return (
        f"rtspsrc location={rtsp_url} latency=0 "
        "! rtph265depay ! avdec_h265 "
        "! videoconvert "
        "! queue max-size-buffers=1 leaky=downstream "
        "! jpegenc quality=80 "
        "! multipartmux boundary=frame "
        "! fdsink fd=1"
    )


def stream_camera(cam_id: str):
    url = CAMERAS[cam_id]
    proc = subprocess.Popen(
        [GST, "-q"] + gst_pipeline(url).split(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=GST_ENV,
    )
    try:
        while True:
            chunk = proc.stdout.read(4096)
            if not chunk:
                err = proc.stderr.read().decode(errors="replace")
                if err:
                    print(f"[gst cam {cam_id}] {err}", flush=True)
                break
            yield chunk
    finally:
        proc.kill()
        proc.wait()


@app.route("/stream/<cam_id>")
def stream(cam_id: str):
    if cam_id not in CAMERAS:
        return "Unknown camera", 404
    return Response(
        stream_camera(cam_id),
        content_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/")
def index():
    return send_from_directory(os.path.dirname(__file__), "index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, threaded=True)
