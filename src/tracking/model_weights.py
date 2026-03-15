"""Resolve tracking model weights from local path or Hugging Face Hub."""

from pathlib import Path

from loguru import logger

# Sail-CV tracking weights on Hugging Face (estefoucher/tell-tale-detector)
HF_REPO_ID = "estefoucher/tell-tale-detector"
HF_WEIGHTS_SUBFOLDER = "weights"

# Known weight filenames that can be fetched from HF if not present locally
HF_WEIGHT_FILES = frozenset(
    {
        "sailcv-yolo11n-cls224.pt",
        "sailcv-rtdetrl1088.pt",
        "sailcv-rtdetrl640.pt",
    }
)


def resolve_model_path(
    path: Path | str,
    *,
    project_root: Path | None = None,
) -> Path:
    """
    Resolve a model path: use local file if it exists, otherwise try to download
    from Hugging Face (estefoucher/tell-tale-detector, weights/).

    Args:
        path: Local path or filename (e.g. checkpoints/sailcv-yolo11n-cls224.pt
              or sailcv-yolo11n-cls224.pt).
        project_root: Optional project root for resolving relative paths.
                      If None, Path.cwd() is used for relative paths.

    Returns:
        Path to an existing file (local or cached from HF).

    Raises:
        FileNotFoundError: If the path does not exist locally and the file
            is not available on Hugging Face (or download fails).
    """
    p = Path(path) if isinstance(path, str) else path
    root = project_root or Path.cwd()

    if not p.is_absolute():
        p = root / p

    if p.exists():
        return p

    filename = p.name
    if filename in HF_WEIGHT_FILES:
        try:
            from huggingface_hub import hf_hub_download

            local_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=filename,
                subfolder=HF_WEIGHTS_SUBFOLDER,
                local_dir_use_symlinks=False,
            )
            logger.info(f"Downloaded {filename} from Hugging Face ({HF_REPO_ID})")
            return Path(local_path)
        except Exception as e:
            raise FileNotFoundError(
                f"Model not found at {p} and Hugging Face download failed: {e}"
            ) from e

    raise FileNotFoundError(
        f"Model not found: {p}. "
        f"For remote weights use a filename in {sorted(HF_WEIGHT_FILES)} "
        f"or place the file locally."
    )
