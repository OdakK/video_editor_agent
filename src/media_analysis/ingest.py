from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import cv2

from .schemas import MediaAsset


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mxf", ".mkv", ".avi", ".m4v"}


def _resolve_duration_and_fps(path: Path) -> tuple[float | None, float | None, bool]:
    """Best-effort probing of video metadata via OpenCV."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return None, None, False

    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = capture.get(cv2.CAP_PROP_FPS)
    # OpenCV's audio stream detection is unreliable across builds; default to True.
    has_audio = True
    capture.release()

    if frame_count > 0 and fps > 0:
        duration = frame_count / fps
    else:
        duration = None
    return duration, fps if fps > 0 else None, has_audio


def discover_media_assets(base_dir: Path | str) -> List[MediaAsset]:
    """Scan a directory tree and report all supported video rushes."""
    base_path = Path(base_dir).expanduser().resolve()
    if not base_path.exists():
        raise FileNotFoundError(f"Media directory not found: {base_path}")

    assets: List[MediaAsset] = []
    for file_path in _iter_media_files(base_path):
        duration, fps, has_audio = _resolve_duration_and_fps(file_path)
        asset_id = file_path.stem
        assets.append(
            MediaAsset(
                asset_id=asset_id,
                path=file_path,
                duration=duration,
                fps=fps,
                has_audio=has_audio,
            )
        )
    return assets


def _iter_media_files(base_path: Path) -> Iterable[Path]:
    for path in base_path.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            yield path
