"""
Interactive OpenCV tool to create a static telltale layout JSON.

This script is intentionally minimal: it loads a single frame (default at t=3s),
lets you click positions, prompts for id/name in the terminal, and exports a JSON
compatible with `Layout.from_json_dict` (tracking module).

Usage (from repo root):
  export PYTHONPATH="${PWD}/src/tracking"
  uv run python src/tracking/annotate_layout_opencv.py \
    --video assets/tracking/2Ce-CKKCtV4.mp4 \
    --time-sec 3 \
    --output output/tracking_layouts/2Ce-CKKCtV4_layout.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class LayoutPoint:
    id: str
    name: str
    x_px: int
    y_px: int


def _parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    default_video = project_root / "assets" / "tracking" / "2Ce-CKKCtV4.mp4"
    default_output = (
        project_root
        / "output"
        / "tracking_layouts"
        / f"{default_video.stem}_layout.json"
    )

    parser = argparse.ArgumentParser(
        description="Click to annotate telltale layout and export JSON"
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=default_video,
        help="Path to input video file",
    )
    parser.add_argument(
        "--time-sec",
        type=float,
        default=3.0,
        help="Time offset in seconds for the frame to annotate",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help="Frame index to annotate (overrides --time-sec if provided)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Output JSON path (default: output/tracking_layouts/<video_stem>_layout.json)",
    )
    parser.add_argument(
        "--direction",
        type=float,
        nargs=2,
        default=None,
        metavar=("DX", "DY"),
        help="Optional direction prior as a 2D vector (dx dy). Stored in JSON as [dx, dy].",
    )
    parser.add_argument(
        "--window-name",
        type=str,
        default="Layout Annotator",
        help="OpenCV window title",
    )
    parser.add_argument(
        "--headless-save-template",
        action="store_true",
        help=(
            "Do not open any UI. Save a template JSON with an empty layout to --output "
            "(useful to validate pipeline wiring)."
        ),
    )
    return parser.parse_args()


def _load_frame(
    video_path: Path, time_sec: float, frame: int | None
) -> tuple[object, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        cap.release()
        raise RuntimeError(f"Invalid FPS reported for video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    frame_idx = round(time_sec * fps) if frame is None else frame

    if total_frames > 0:
        frame_idx = max(0, min(frame_idx, total_frames - 1))
    else:
        frame_idx = max(0, frame_idx)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, img = cap.read()
    cap.release()
    if not ok or img is None:
        raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")

    h, w = img.shape[:2]
    return img, w, h


def _normalize_xy(x_px: int, y_px: int, width: int, height: int) -> tuple[float, float]:
    # Layout uses normalized coordinates (0..1) with origin at top-left.
    # Using /width and /height keeps values < 1.0 for valid pixel coordinates.
    x = float(x_px) / float(width)
    y = float(y_px) / float(height)
    # Clamp defensively.
    eps = 1e-9
    x = min(max(x, 0.0), 1.0 - eps)
    y = min(max(y, 0.0), 1.0 - eps)
    return x, y


def _layout_to_json_dict(
    points: list[LayoutPoint], width: int, height: int, direction: list[float] | None
) -> dict:
    layout_list: list[dict] = []
    for p in points:
        x, y = _normalize_xy(p.x_px, p.y_px, width, height)
        layout_list.append({"id": p.id, "name": p.name, "x": x, "y": y})

    payload: dict = {"layout": layout_list}
    if direction is not None:
        payload["direction"] = direction
    return payload


def _validate_layout_payload(payload: dict) -> None:
    if not isinstance(payload, dict):
        raise ValueError("Layout payload must be a JSON object")
    layout_list = payload.get("layout")
    if not isinstance(layout_list, list):
        raise ValueError("Layout payload must contain a 'layout' list")
    if len(layout_list) == 0:
        raise ValueError("Layout must contain at least one point")

    seen_ids: set[str] = set()
    for i, item in enumerate(layout_list):
        if not isinstance(item, dict):
            raise ValueError(f"layout[{i}] must be an object")
        for key in ("id", "name", "x", "y"):
            if key not in item:
                raise ValueError(f"layout[{i}] missing key '{key}'")
        pid = item["id"]
        if not isinstance(pid, str) or not pid.strip():
            raise ValueError(f"layout[{i}].id must be a non-empty string")
        if pid in seen_ids:
            raise ValueError(f"Duplicate id '{pid}'")
        seen_ids.add(pid)

        name = item["name"]
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"layout[{i}].name must be a non-empty string")

        x = item["x"]
        y = item["y"]
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise ValueError(f"layout[{i}].x/y must be numbers")
        if not (0.0 <= float(x) <= 1.0) or not (0.0 <= float(y) <= 1.0):
            raise ValueError(f"layout[{i}].x/y must be within [0, 1]")

    direction = payload.get("direction")
    if direction is not None and (
        not isinstance(direction, list)
        or len(direction) != 2
        or not all(isinstance(v, (int, float)) for v in direction)
    ):
        raise ValueError("direction must be a list of two numbers: [dx, dy]")


def _save_json(payload: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(payload, f, indent=2)

    # Re-load and validate what we wrote.
    with output_path.open() as f:
        written = json.load(f)
    _validate_layout_payload(written)


def _draw_overlay(
    base_bgr,
    points: list[LayoutPoint],
    width: int,
    height: int,
    pending_click: tuple[int, int] | None,
):
    img = base_bgr.copy()

    # Legend
    legend = [
        "Left click: add point (then enter id/name in terminal)",
        "u: undo | c: clear | s: save | q/esc: quit",
    ]
    y0 = 20
    for line in legend:
        cv2.putText(
            img,
            line,
            (10, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            line,
            (10, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y0 += 22

    for idx, p in enumerate(points):
        color = (0, 255, 255) if idx % 2 == 0 else (255, 0, 255)
        cv2.circle(img, (p.x_px, p.y_px), 6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.circle(img, (p.x_px, p.y_px), 6, color, 2, cv2.LINE_AA)
        label = f"{p.id}"
        cv2.putText(
            img,
            label,
            (min(p.x_px + 8, width - 10), max(p.y_px - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            label,
            (min(p.x_px + 8, width - 10), max(p.y_px - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    if pending_click is not None:
        x, y = pending_click
        cv2.drawMarker(
            img,
            (x, y),
            (0, 255, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=18,
            thickness=2,
            line_type=cv2.LINE_AA,
        )

    # Footer with normalized preview count
    footer = f"points={len(points)}  size={width}x{height}"
    cv2.putText(
        img,
        footer,
        (10, height - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        footer,
        (10, height - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return img


def main() -> None:
    args = _parse_args()

    video_path: Path = args.video
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_path: Path = args.output
    direction = None
    if args.direction is not None:
        direction = [float(args.direction[0]), float(args.direction[1])]

    if args.headless_save_template:
        payload = {"layout": []}
        if direction is not None:
            payload["direction"] = direction
        # Skip strict validation for empty template; just write it.
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote template layout JSON to {output_path}")
        return

    frame_bgr, width, height = _load_frame(video_path, args.time_sec, args.frame)

    points: list[LayoutPoint] = []
    pending_click: tuple[int, int] | None = None

    def on_mouse(event, x, y, _flags, _param) -> None:
        nonlocal pending_click
        if event == cv2.EVENT_LBUTTONDOWN and pending_click is None:
            pending_click = (int(x), int(y))

    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(args.window_name, on_mouse)

    print("")
    print("Layout annotation started.")
    print(f"  video:   {video_path}")
    print(f"  output:  {output_path}")
    if args.frame is None:
        print(f"  frame:   at t={args.time_sec:.3f}s")
    else:
        print(f"  frame:   index={args.frame}")
    print("")
    print("Controls:")
    print("  - left click: select a point, then enter id/name in terminal")
    print("  - u: undo last point")
    print("  - c: clear points")
    print("  - s: save JSON")
    print("  - q or ESC: quit")
    print("")

    while True:
        if pending_click is not None:
            x_px, y_px = pending_click
            print(f"\nClicked at px=({x_px}, {y_px})")
            point_id = input("Enter id (e.g. TL, TR, MB-R): ").strip()
            point_name = input("Enter name (free text): ").strip()

            if not point_id:
                print("Skipped: empty id")
            elif any(p.id == point_id for p in points):
                print(f"Skipped: duplicate id '{point_id}'")
            elif not point_name:
                print("Skipped: empty name")
            else:
                points.append(
                    LayoutPoint(id=point_id, name=point_name, x_px=x_px, y_px=y_px)
                )
                x_norm, y_norm = _normalize_xy(x_px, y_px, width, height)
                print(
                    f"Added {point_id} ({point_name}) at normalized x={x_norm:.4f}, y={y_norm:.4f}"
                )
            pending_click = None

        overlay = _draw_overlay(frame_bgr, points, width, height, pending_click)
        cv2.imshow(args.window_name, overlay)
        key = cv2.waitKey(20) & 0xFF

        if key in (27, ord("q")):  # ESC or q
            break
        if key == ord("u") and points:
            removed = points.pop()
            print(f"Undo: removed {removed.id} ({removed.name})")
        if key == ord("c") and points:
            points.clear()
            print("Cleared all points.")
        if key == ord("s"):
            payload = _layout_to_json_dict(points, width, height, direction)
            try:
                _validate_layout_payload(payload)
            except Exception as e:
                print(f"Cannot save: invalid layout payload: {e}")
                continue

            _save_json(payload, output_path)
            print(f"Saved layout JSON to {output_path}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
