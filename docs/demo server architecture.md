# Demo Server Architecture

Current state of the live stereo-reconstruction demo. Two RTSP cameras feed a
server (intended to run on the Jetson) that serves a web client and drives an
inference loop.

## Components

**Server — `viewer/server.py`**
Built on Python's stdlib `ThreadingHTTPServer`, listening on port `7863`. It:
- Serves the web client from `viewer/static/index.html`.
- Proxies live MJPEG previews from each camera (`GET /stream/<cam_id>`) via an
  on-demand ffmpeg pipe.
- Grabs single-frame screenshots from both cameras for the calibration workflow
  (`POST /calibrate/capture`, `/calibrate/run`).
- Starts/stops the reconstruction loop in a Docker container
  (`POST /start/<scene>`, `/stop`).
- Pushes real-time updates to the client over Server-Sent Events (`/events`,
  `/logs`) and serves per-frame artifacts (`/composite`, `/pointcloud`,
  `/cameras`, `/matches`).

**Inference loop — `viewer/reconstruct_loop.py`**
Launched by the server inside a GPU Docker container. It:
- Opens long-lived ffmpeg/RTSP grabbers (~4 fps, 1024px) for both cameras.
- Runs the MASt3R model per frame: match → triangulate → filter into a 3D point
  cloud.
- Writes artifacts to `OUTPUT_DIR/live/` (`frame1/2.jpg`, `composite.jpg`,
  `pointcloud.json`, `matches.json`).
- Emits one JSON status line per frame to stdout; the server parses it and
  rebroadcasts over SSE.

**Web client — `viewer/static/index.html`**
Three.js + OrbitControls. Renders the live point cloud and camera frusta in 3D,
shows the composite/match panel, streams logs, and exposes Start/Stop,
calibration, and tuning sliders.

## Data flow

```
RTSP cams → reconstruct_loop (Docker, MASt3R) → OUTPUT_DIR/live + stdout JSON
                                                      ↓
                              server.py (HTTP :7863, SSE) → web client (Three.js)
```

## Run

`python3 viewer/server.py`, then open `http://<jetson>:7863`. Camera IPs and
network setup are in [`hardware setup and specs.md`](hardware%20setup%20and%20specs.md).

## Calibration algorithm

### Intrinsics (per camera, independent)

Each camera is calibrated on its own — stereo sync is not needed. Reference
implementation: `SailCV-3D-reconstruction/src/mv_utils/intrinsics_calibration.py`.

1. **Collect views.** Capture many frames of a checkerboard held at varied
   distances, angles, and positions across the frame. In the demo, the
   `intrinsic` screenshot kind supplies these — each camera's image is used
   independently. Aim for ~15–30 good views.
2. **Detect corners.** Per frame: grayscale, then
   `cv2.findChessboardCorners(gray, (inner_corners_x, inner_corners_y))`,
   refined with `cv2.cornerSubPix`. Build a flat 3D object-point grid scaled by
   `square_size_mm`. Pattern dims come from `checkerboard_specs.yml`.
3. **Calibrate.** Accumulate object/image points over all frames where corners
   were found, then `cv2.calibrateCamera(object_points, image_points, image_size)`
   → `camera_matrix` (K), `dist_coeffs`, and mean reprojection error (target
   < ~0.5 px). Use the orientation-corrected frame size for `image_size`.
4. **Save** per camera as `intrinsics.json` (`camera_matrix`, `dist_coeffs`,
   `reprojection_error`); these feed the extrinsic stereo step.

## How-tos

### Launch the server on the Jetson from your Mac

Repo on the Jetson: `~/Workspace/Sail-CV`. SSH is password auth
(`estebanfoucher@192.168.1.100`, see hardware doc), so `sshpass` is handy:

```sh
JET=estebanfoucher@192.168.1.100

# Deploy changed files (back up first)
sshpass -p 'eaglesailvision' ssh $JET 'cd ~/Workspace/Sail-CV && cp viewer/server.py viewer/server.py.bak'
sshpass -p 'eaglesailvision' scp viewer/server.py $JET:~/Workspace/Sail-CV/viewer/server.py

# Launch detached, logging to a file
sshpass -p 'eaglesailvision' ssh $JET \
  'cd ~/Workspace/Sail-CV && nohup python3 -u viewer/server.py > /tmp/sailcv_server.log 2>&1 & echo "pid $!"'

# Verify (separate connection)
sshpass -p 'eaglesailvision' ssh $JET 'tail -5 /tmp/sailcv_server.log; ss -tlnp | grep 7863'
```

Then open `http://192.168.1.100:7863` (3D viewer) or `/screenshots.html`.

**Gotchas learned the hard way:**

- **Never `pkill -f "viewer/server.py"` over SSH.** The pattern also matches the
  SSH wrapper shell whose command line *contains* that string, so pkill kills
  its own session (SSH exits 255) and the python before it can start. Kill by
  port instead: `fuser -k 7863/tcp`, or match the binary only:
  `pkill -f "[p]ython3 -u viewer/server.py"`.
- **Always run python with `-u`.** Over a non-TTY SSH pipe, stdout is
  block-buffered, so a server that *is* running looks silent (empty log) until
  the buffer flushes or the process dies. `-u` forces line-buffering so startup
  lines appear immediately.
- **Backgrounding + an inline `sleep`/check in the same SSH command is
  flaky.** Launch in one connection, verify in a separate one.
- A bare `nohup … &` keeps the SSH channel open as long as python holds the
  redirected stdout fd; that's expected. The process survives disconnect via
  `nohup`. For a durable setup, prefer a `systemd` unit (not yet configured).

The Jetson clock may be wrong (NTP not synced), which skews the timestamp in
session/screenshot filenames — cosmetic only.
