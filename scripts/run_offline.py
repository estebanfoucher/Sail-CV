"""Offline runner — exercise the reconstruction pipeline on saved fixtures.

Loads a stereo image pair from `assets/reconstruction/<scene>/`, runs the
shared pipeline (preprocess → infer → match → composite → pointcloud) N
times, and prints a per-stage timing summary. Useful for iterating on the
matching/inference algorithm without RTSP, the docker loop, or the UI.

Run inside the reconstruction docker image (it has torch + cuda + flash-attn):

    scripts/run_offline.sh \
        --scene scene_12 --engine vanilla --iters 5 --output-dir /tmp/out

…or directly if your environment already has the deps:

    PYTHONPATH=src/reconstruction:mast3r:mast3r/dust3r \\
        python3 scripts/run_offline.py --scene scene_12 --iters 5

Outputs (written to --output-dir):
    composite.jpg        — 2×2 panel with match overlays
    pointcloud.json      — same shape as the live viewer expects
    matches.json         — top-N matches with confidence
    timings.json         — per-iter, per-stage timings + summary
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import PIL.Image
import torch
from loguru import logger
from PIL.ImageOps import exif_transpose


def _bootstrap_paths():
    """Make stereo/dust3r/mast3r imports work whether we're inside the docker
    image (paths under /app) or running from a checkout (paths under repo root)."""
    here = Path(__file__).resolve().parent
    candidates = [
        Path("/app/src/reconstruction"), Path("/app/mast3r"), Path("/app/mast3r/dust3r"),
        here.parent / "src" / "reconstruction",
        here.parent / "mast3r",
        here.parent / "mast3r" / "dust3r",
    ]
    for p in candidates:
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))


_bootstrap_paths()

# Imports below depend on _bootstrap_paths() having run.
from stereo.convert_calibration import convert_calibration_parameters  # noqa: E402
from stereo.pipeline import make_engine, run_pipeline  # noqa: E402


def find_scene_pair(scene_dir: Path) -> tuple[Path, Path]:
    """Locate camera_{1,2} images in a scene dir. Tolerate .png and .jpg."""
    for ext in ("png", "jpg", "jpeg"):
        p1 = scene_dir / f"camera_1.{ext}"
        p2 = scene_dir / f"camera_2.{ext}"
        if p1.exists() and p2.exists():
            return p1, p2
    raise FileNotFoundError(f"No camera_1.{{png,jpg}} pair found in {scene_dir}")


def find_calibration(scene_dir: Path, project_root: Path,
                     explicit: Path | None) -> Path:
    """Pick a calibration JSON: explicit > scene-local > project default."""
    if explicit is not None:
        return explicit
    local = scene_dir / "calibration.json"
    if local.exists():
        return local
    fallback = project_root / "output" / "calibration" / "calibration.json"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"No calibration found. Tried {local} and {fallback}. Pass --calibration."
    )


def summarize(samples: list[float]) -> dict:
    if not samples:
        return {}
    return {
        "mean": round(statistics.mean(samples), 1),
        "std":  round(statistics.stdev(samples), 1) if len(samples) > 1 else 0.0,
        "min":  round(min(samples), 1),
        "max":  round(max(samples), 1),
    }


def resolve_scenes(args, scenes_root: Path) -> list[str]:
    """Expand --scenes / --scene into a concrete list of scene names.

    --scenes takes precedence; "all" auto-discovers any subdirectory with a
    valid camera_{1,2}.{png,jpg} pair. --scene is the single-scene fallback.
    """
    if args.scenes:
        if args.scenes.strip().lower() == "all":
            names = []
            for d in sorted(scenes_root.iterdir()):
                if not d.is_dir():
                    continue
                try:
                    find_scene_pair(d)
                    names.append(d.name)
                except FileNotFoundError:
                    continue
            return names
        return [s.strip() for s in args.scenes.split(",") if s.strip()]
    return [args.scene]


def run_one_scene(engine, scene_name: str, scenes_root: Path, project_root: Path,
                   args, output_root: Path) -> dict:
    """Run iters on a single scene with an already-loaded engine. Returns summary dict."""
    scene_dir = scenes_root / scene_name
    if not scene_dir.is_dir():
        logger.error(f"Scene not found: {scene_dir}")
        return {"scene": scene_name, "error": f"missing scene dir {scene_dir}"}

    try:
        img1_path, img2_path = find_scene_pair(scene_dir)
        calib_path = find_calibration(scene_dir, project_root, args.calibration)
    except FileNotFoundError as e:
        logger.error(str(e))
        return {"scene": scene_name, "error": str(e)}

    logger.info(f"=== {scene_name} ===")
    logger.info(f"Pair:        {img1_path.name} + {img2_path.name}")
    logger.info(f"Calibration: {calib_path}")

    img1 = exif_transpose(PIL.Image.open(img1_path)).convert("RGB")
    img2 = exif_transpose(PIL.Image.open(img2_path)).convert("RGB")
    logger.info(f"Images:      {img1.size}, {img2.size}")

    with open(calib_path) as f:
        calibration_data = json.load(f)
    calibration_params = convert_calibration_parameters(calibration_data)
    img_w, img_h = calibration_params["image_size"]

    per_iter: list[dict] = []
    last_result = None
    for i in range(args.iters):
        is_warmup = (i == 0)
        t0 = time.monotonic()
        result = run_pipeline(
            engine, img1, img2,
            image_size=(img_w, img_h),
            subsample=args.subsample,
            max_pointcloud_pts=args.max_pointcloud_pts,
            calibration_params=calibration_params,
        )
        total_ms = int((time.monotonic() - t0) * 1000)
        tag = "warmup" if is_warmup else f"iter {i}"
        logger.info(
            f"[{scene_name}/{tag}] total={total_ms}ms  "
            f"matches={len(result.top_pts0)}/{result.num_matches_total}  "
            f"pts={len(result.pts3d)}  | {result.timings_ms}"
        )
        if not is_warmup:
            per_iter.append({"total_ms": total_ms, **result.timings_ms})
        last_result = result

    summary: dict[str, dict] = {}
    if per_iter:
        all_keys = sorted({k for d in per_iter for k in d})
        for k in all_keys:
            summary[k] = summarize([d[k] for d in per_iter if k in d])

    out_dir = output_root / scene_name
    out_dir.mkdir(parents=True, exist_ok=True)
    if last_result is not None:
        last_result.composite.save(out_dir / "composite.jpg", quality=85)
        with open(out_dir / "pointcloud.json", "w") as f:
            json.dump({
                "pts":    last_result.pts3d.tolist(),
                "colors": last_result.colors.tolist(),
            }, f)
        with open(out_dir / "matches.json", "w") as f:
            json.dump({
                "pts0":   last_result.top_pts0.tolist(),
                "pts1":   last_result.top_pts1.tolist(),
                "scores": last_result.top_scores.tolist(),
                "num_matches_total": last_result.num_matches_total,
            }, f)
    payload = {
        "scene": scene_name,
        "engine": args.engine, "subsample": args.subsample, "iters": args.iters,
        "image_size": [img_w, img_h],
        "num_matches_total": last_result.num_matches_total if last_result else None,
        "per_iter": per_iter,
        "summary": summary,
    }
    with open(out_dir / "timings.json", "w") as f:
        json.dump(payload, f, indent=2)

    return payload


def print_cross_scene_table(per_scene: list[dict]):
    """Compact comparison across scenes — one row per scene, mean per stage."""
    rows = [r for r in per_scene if "summary" in r and r["summary"]]
    if not rows:
        return
    stages = ["preproc", "infer", "match", "composite", "pointcloud", "total_ms"]
    header = f"{'scene':<14} {'matches':>8}  " + "  ".join(f"{s:>10}" for s in stages)
    logger.info("--- cross-scene summary (mean ms, excludes warmup) ---")
    logger.info(header)
    for r in rows:
        cells = []
        for s in stages:
            m = r["summary"].get(s, {}).get("mean")
            cells.append(f"{m:>10.1f}" if m is not None else f"{'-':>10}")
        nm = r.get("num_matches_total")
        logger.info(f"{r['scene']:<14} {nm if nm is not None else '-':>8}  " + "  ".join(cells))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", default="scene_12",
                        help="Single scene under assets/reconstruction/. Ignored if --scenes is set.")
    parser.add_argument("--scenes", default=None,
                        help="Comma-separated list (e.g. 'scene_12,scene_13') or 'all' to auto-discover.")
    parser.add_argument("--engine", choices=["vanilla", "speedy"], default="vanilla")
    parser.add_argument("--calibration", type=Path, default=None,
                        help="Path to calibration.json. Defaults to scene-local then project default.")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("/app/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"))
    parser.add_argument("--iters", type=int, default=5,
                        help="Total iterations per scene. First is warm-up; remaining are timed.")
    parser.add_argument("--subsample", type=int, default=16, choices=[1, 2, 4, 8, 16])
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/sailcv_offline"),
                        help="Output root. Each scene writes to <output-dir>/<scene>/.")
    parser.add_argument("--max-pointcloud-pts", type=int, default=8000)
    parser.add_argument("--scenes-root", type=Path, default=None,
                        help="Override assets/reconstruction root (auto-detected by default).")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    project_root = Path("/app") if Path("/app/assets").exists() else here.parent
    scenes_root = args.scenes_root or (project_root / "assets" / "reconstruction")

    scenes = resolve_scenes(args, scenes_root)
    if not scenes:
        logger.error("No scenes resolved. Pass --scene <name> or --scenes <list|all>.")
        sys.exit(1)
    logger.info(f"Engine:  {args.engine}, subsample={args.subsample}, iters={args.iters}")
    logger.info(f"Scenes:  {', '.join(scenes)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device:  {device}")
    t_load = time.monotonic()
    engine = make_engine(args.engine, args.checkpoint, device)
    engine.load_model()
    logger.info(f"Model loaded in {time.monotonic() - t_load:.1f}s — running {len(scenes)} scene(s)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_scene: list[dict] = []
    t_run = time.monotonic()
    for name in scenes:
        per_scene.append(run_one_scene(engine, name, scenes_root, project_root, args, args.output_dir))
    logger.info(f"All scenes done in {time.monotonic() - t_run:.1f}s")

    print_cross_scene_table(per_scene)

    with open(args.output_dir / "summary.json", "w") as f:
        json.dump({"engine": args.engine, "subsample": args.subsample,
                   "iters": args.iters, "scenes": per_scene}, f, indent=2)
    logger.info(f"Wrote {args.output_dir}/summary.json + per-scene artefacts")


if __name__ == "__main__":
    main()
