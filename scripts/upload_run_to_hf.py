"""Upload a trained run zip to the Hugging Face model repo.

Usage (local machine, one-shot):

    huggingface-cli login                 # paste a WRITE token
    python scripts/upload_run_to_hf.py \
        --file finetune_rtdetr_fused_run_17_epoch.zip \
        --repo-id estefoucher/tell-tale-detector \
        --path-in-repo runs/finetune_rtdetr_fused_run_17_epoch.zip

The default values above match the asset we currently need. The upload uses
LFS automatically via ``huggingface_hub`` so the ~400 MB zip goes through fine.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, login


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a run zip to Hugging Face.")
    parser.add_argument(
        "--file",
        type=Path,
        default=Path("finetune_rtdetr_fused_run_17_epoch.zip"),
        help="Local file to upload.",
    )
    parser.add_argument(
        "--repo-id",
        default="estefoucher/tell-tale-detector",
        help="Target HF model repo id.",
    )
    parser.add_argument(
        "--path-in-repo",
        default="runs/finetune_rtdetr_fused_run_17_epoch.zip",
        help="Destination path inside the HF repo.",
    )
    parser.add_argument(
        "--repo-type",
        default="model",
        choices=["model", "dataset", "space"],
        help="HF repo type.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional HF access token (otherwise uses saved credentials).",
    )
    parser.add_argument(
        "--commit-message",
        default=None,
        help="Optional commit message.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.file.is_file():
        raise FileNotFoundError(f"Local file not found: {args.file.resolve()}")

    if args.token:
        login(token=args.token, add_to_git_credential=False)

    api = HfApi()

    commit_message = args.commit_message or (
        f"Upload {args.path_in_repo} ({args.file.stat().st_size / 1e6:.1f} MB)"
    )

    print(f"Uploading {args.file} -> {args.repo_id}:{args.path_in_repo}")
    print(f"Commit message: {commit_message}")

    commit_info = api.upload_file(
        path_or_fileobj=str(args.file),
        path_in_repo=args.path_in_repo,
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        commit_message=commit_message,
    )

    print("Upload done.")
    print("Commit:", commit_info)


if __name__ == "__main__":
    main()
