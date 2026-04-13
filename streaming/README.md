# Video Streaming Server

Low-latency dual-camera MJPEG streaming server using GStreamer + Flask.

## Setup

```bash
cd streaming
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
source .venv/bin/activate
python server.py
# open http://localhost:8080
```

## Architecture

```
POE Camera (H.265 RTSP)
  └─ rtspsrc latency=0
       └─ rtph265depay → avdec_h265
            └─ videoconvert
                 └─ queue (leaky, max 1 frame)   ← drops stale decoded frames
                      └─ jpegenc
                           └─ multipartmux
                                └─ fdsink → Flask → browser <img>
```

RTSP latency is eliminated via `latency=0` on `rtspsrc`. The leaky queue sits **after** decoding so only complete decoded frames are dropped — never RTP packets (which would corrupt the decoder).

## Cameras

| ID | IP              | Codec | Resolution |
|----|-----------------|-------|------------|
| 1  | 192.168.1.214   | H.265 | 640×480    |
| 2  | 192.168.1.34    | H.265 | 640×480    |

Credentials: `admin / 123456`

## Dependencies

- **GStreamer 1.26.5** — `/Library/Frameworks/GStreamer.framework/`
  Required plugins: `rtspsrc`, `rtph265depay`, `avdec_h265`, `videoconvert`, `jpegenc`, `multipartmux`, `fdsink`
- **Flask** ≥ 3.0
