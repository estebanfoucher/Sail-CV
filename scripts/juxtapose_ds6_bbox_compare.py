"""Place two replay clips side-by-side (same frame index) for A/B comparison.

Default inputs from ``replay_ds6_faker_bbox_video.py`` (ByteTrack mode)::

    <bbox-root>/<model>/C{i}_tracked_replay.mp4

Falls back to ``C{i}_bbox_only.mp4`` if the tracked file is missing.

Writes::

    <output-dir>/C{i}_bbox_compare.mp4   (width ≈ 2× source, same fps / frame count)

Requires ``ffmpeg`` on PATH. Optional labels use ``drawtext`` (needs a usable font on macOS).

Example::

    uv run python scripts/juxtapose_ds6_bbox_compare.py
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# macOS: common path for drawtext
_DEFAULT_FONT_MAC = Path(
    "/System/Library/Fonts/Supplemental/Arial.ttf"
)


def _ffmpeg_hstack(
    left: Path,
    right: Path,
    out: Path,
    *,
    label_left: str,
    label_right: str,
    fontfile: Path | None,
    crf: int,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)

    if fontfile and fontfile.is_file():
        # Escape single quotes for ffmpeg filter
        def esc(s: str) -> str:
            return s.replace("'", r"'\''")

        lf = esc(str(fontfile.resolve()))
        ll = esc(label_left)
        lr = esc(label_right)
        filt = (
            f"[0:v]drawtext=fontfile='{lf}':text='{ll}':fontsize=28:"
            f"fontcolor=white:borderw=2:bordercolor=black:x=16:y=16[v0];"
            f"[1:v]drawtext=fontfile='{lf}':text='{lr}':fontsize=28:"
            f"fontcolor=white:borderw=2:bordercolor=black:x=16:y=16[v1];"
            f"[v0][v1]hstack=inputs=2:shortest=1[v]"
        )
    else:
        filt = "[0:v][1:v]hstack=inputs=2:shortest=1[v]"

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(left),
        "-i",
        str(right),
        "-filter_complex",
        filt,
        "-map",
        "[v]",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(crf),
        str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed ({r.returncode}):\n{r.stderr or r.stdout}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bbox-root",
        type=Path,
        default=PROJECT_ROOT / "20260416_213654" / "bbox_replay",
    )
    parser.add_argument(
        "--left-model",
        default="custom_ulysse",
        help="Subfolder name (left half of output)",
    )
    parser.add_argument(
        "--right-model",
        default="after_finetune_fused_17ep_last",
        help="Subfolder name (right half of output)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <bbox-root>/juxtaposed",
    )
    parser.add_argument(
        "--label-left",
        default="custom_ulysse",
    )
    parser.add_argument(
        "--label-right",
        default="after_finetune_17ep (last.pt)",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Skip drawtext (no font required)",
    )
    parser.add_argument(
        "--font",
        type=Path,
        default=_DEFAULT_FONT_MAC,
        help="TTF for drawtext (ignored with --no-labels)",
    )
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument(
        "--only-video",
        type=str,
        default=None,
        help="Only process this tag, e.g. C4",
    )
    parser.add_argument(
        "--prefer-suffix",
        default="_tracked_replay.mp4",
        help="Primary filename suffix after tag, e.g. C1 + suffix",
    )
    parser.add_argument(
        "--fallback-suffix",
        default="_bbox_only.mp4",
        help="If primary file missing, try this suffix",
    )
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found on PATH; install ffmpeg and retry.", file=sys.stderr)
        sys.exit(1)

    root = args.bbox_root.resolve()
    left_dir = root / args.left_model
    right_dir = root / args.right_model
    out_dir = (args.output_dir or (root / "juxtaposed")).resolve()

    font = None if args.no_labels else args.font

    def resolve_clip(model_dir: Path, tag: str) -> Path | None:
        primary = model_dir / f"{tag}{args.prefer_suffix}"
        if primary.is_file():
            return primary
        fb = model_dir / f"{tag}{args.fallback_suffix}"
        if fb.is_file():
            return fb
        return None

    for i in range(1, 9):
        tag = f"C{i}"
        if args.only_video and tag != args.only_video:
            continue
        lp = resolve_clip(left_dir, tag)
        rp = resolve_clip(right_dir, tag)
        if lp is None or rp is None:
            print(
                f"Skip {tag}: missing clip under {left_dir.name} or {right_dir.name} "
                f"({args.prefer_suffix} / {args.fallback_suffix})"
            )
            continue
        outp = out_dir / f"{tag}_bbox_compare.mp4"
        print(f"{tag}: {lp.relative_to(root)} | {rp.relative_to(root)} -> {outp.name}")
        _ffmpeg_hstack(
            lp,
            rp,
            outp,
            label_left=args.label_left,
            label_right=args.label_right,
            fontfile=font,
            crf=args.crf,
        )

    print("Done. Outputs under:", out_dir)


if __name__ == "__main__":
    main()
