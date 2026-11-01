"""Read media metadata with ffprobe (duration, frame rate, resolution, audio track)."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class MediaInfo:
    duration: float | None = None
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    has_audio: bool = False


def _parse_rate(rate: str | None) -> float | None:
    """ffprobe gives frame rates as fractions, e.g. "30000/1001"."""
    if not rate or rate == "0/0":
        return None
    num, _, den = rate.partition("/")
    try:
        value = float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        return None
    return value or None


def parse_ffprobe(output: str) -> MediaInfo:
    data = json.loads(output or "{}")
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    duration = data.get("format", {}).get("duration") or video.get("duration")
    return MediaInfo(
        duration=float(duration) if duration else None,
        fps=_parse_rate(video.get("avg_frame_rate") or video.get("r_frame_rate")),
        width=video.get("width"),
        height=video.get("height"),
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
    )


@lru_cache(maxsize=256)
def probe(path: str | Path) -> MediaInfo:
    """Probe a file. Raises RuntimeError when ffprobe is missing or cannot read the file."""
    command = ["ffprobe", "-v", "error", "-show_entries",
               "format=duration:stream=codec_type,width,height,avg_frame_rate,r_frame_rate,duration",
               "-of", "json", str(path)]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe not found: install ffmpeg") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffprobe could not read {path}: {exc.stderr.strip()}") from exc
    return parse_ffprobe(result.stdout)
