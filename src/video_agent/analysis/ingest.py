"""Find the rushes in a folder and read their metadata."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

from ..probe import probe
from ..schemas import MediaAsset

LOGGER = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mxf", ".mkv", ".avi", ".m4v"}


def discover_media_assets(base_dir: Path | str) -> list[MediaAsset]:
    """Scan a directory tree and return every supported video file, sorted by path."""
    base_path = Path(base_dir).expanduser().resolve()
    if not base_path.exists():
        raise FileNotFoundError(f"Media directory not found: {base_path}")

    assets: list[MediaAsset] = []
    seen: dict[str, Path] = {}
    for file_path in sorted(_iter_media_files(base_path)):
        # Asset ids are file stems; two files with the same stem in different folders would collide.
        asset_id = file_path.stem
        if asset_id in seen:
            asset_id = file_path.relative_to(base_path).with_suffix("").as_posix().replace("/", "__")
        seen[asset_id] = file_path

        info = probe(file_path)
        if not info.has_audio:
            LOGGER.info("%s has no audio track: transcription will be skipped", file_path.name)
        assets.append(MediaAsset(asset_id=asset_id, path=file_path, duration=info.duration,
                                 fps=info.fps, has_audio=info.has_audio))
    return assets


def _iter_media_files(base_path: Path) -> Iterator[Path]:
    for path in base_path.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            yield path
