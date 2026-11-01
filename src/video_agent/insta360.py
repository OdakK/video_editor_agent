"""Turn 360° camera files (Insta360 .insv) into flat views the editor can cut, with a proxy workflow.

A .insv file is an MP4 container holding the two fisheye lenses of the camera: either two square
video streams (one per lens, e.g. X3/X4 at full resolution), or one 2:1 stream with both circles
side by side (lower resolutions and the .lrv proxies). Older models write one file per lens.

Each lens covers about 200°, so it can be re-projected on its own into a clean flat view with
ffmpeg's v360 filter: no stitching, so no seam. A typical vlog clip gives two views: the scene in
front of the camera, and the person holding it (selfie).

Decoding 8K HEVC is the expensive part (about 1 minute per minute of footage for two full views).
So, like professional editing software, the import only makes light proxies: one decoded frame
every 2 seconds (the keyframes, which need no other frame to decode) at 540x960, plus the
untouched soundtrack. That is enough for the vision model and for transcription, at about
1 second per minute of footage. The final render goes back to the original .insv, at full
quality, for the seconds actually used in the edit.

Not handled: Insta360's gyro stabilization (FlowState) and free reframing to any direction.
"""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .schemas import Source360

EXTENSIONS_360 = {".insv", ".lrv"}
LENS_FOV = 200  # degrees covered by one Insta360 fisheye lens

ORIENTATIONS = {
    "vertical": {"w": 1080, "h": 1920, "h_fov": 75, "v_fov": 110},
    "horizontal": {"w": 1920, "h": 1080, "h_fov": 110, "v_fov": 70},
}
PROXY_SCALE = 2  # proxies are 2x smaller than the render: 540x960


@dataclass
class Lens:
    """How to get one fisheye circle out of the file."""

    stream: int        # video stream index
    crop: str | None   # ffmpeg crop for side-by-side circles, else None


def is_360(path: Path) -> bool:
    return path.suffix.lower() in EXTENSIONS_360


def find_lenses(path: Path) -> list[Lens]:
    """Locate the fisheye circles from the stream layout."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries",
                          "stream=index,width,height", "-of", "json", str(path)],
                         check=True, capture_output=True, text=True).stdout
    streams = json.loads(out).get("streams", [])
    square = [s for s in streams if s["width"] == s["height"]]
    if len(square) >= 2:  # one stream per lens
        return [Lens(i, None) for i in range(len(square[:2]))]
    if streams and streams[0]["width"] == 2 * streams[0]["height"]:  # both circles side by side
        return [Lens(0, "crop=iw/2:ih:0:0"), Lens(0, "crop=iw/2:ih:iw/2:0")]
    if square:  # a single lens (one file per lens on older cameras)
        return [Lens(0, None)]
    raise ValueError(f"{path.name}: unrecognized 360 layout {[(s['width'], s['height']) for s in streams]}")


def flat_filter(stream: int, crop: str | None, width: int, height: int, input_index: int = 0) -> str:
    """ffmpeg filter that re-projects one lens into a flat view (no output label)."""
    h_fov, v_fov = (75, 110) if height >= width else (110, 70)
    chain = f"{crop}," if crop else ""
    return (f"[{input_index}:v:{stream}]{chain}v360=input=fisheye:output=flat:ih_fov={LENS_FOV}:iv_fov={LENS_FOV}"
            f":h_fov={h_fov}:v_fov={v_fov}:w={width}:h={height}:interp=lanczos")


def hwaccel() -> list[str]:
    """Hardware HEVC decoding on macOS: about 3x faster on 8K files."""
    return ["-hwaccel", "videotoolbox"] if platform.system() == "Darwin" else []


def make_proxies(path: Path, out_dir: Path, base_name: str,
                 orientation: str = "vertical") -> list[tuple[Path, Source360]]:
    """Write one light proxy per lens (<base_name>_lens1.mp4, …) and return it with its full-quality source."""
    lenses = find_lenses(path)
    o = ORIENTATIONS[orientation]
    pw, ph = o["w"] // PROXY_SCALE, o["h"] // PROXY_SCALE
    has_audio = bool(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                                     "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout.strip())
    out_dir.mkdir(parents=True, exist_ok=True)
    filters, outputs, results = [], [], []
    for i, lens in enumerate(lenses, 1):
        filters.append(f"{flat_filter(lens.stream, lens.crop, pw, ph)},format=yuv420p[v{i}]")
        target = out_dir / f"{base_name}_lens{i}.mp4"
        outputs += ["-map", f"[v{i}]", *(["-map", "0:a:0", "-c:a", "copy"] if has_audio else []),
                    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", "-movflags", "+faststart", str(target)]
        results.append((target, Source360(path=path, stream=lens.stream, crop=lens.crop, width=o["w"], height=o["h"])))
    # Keyframes only: about 70x faster than decoding every frame of 8K HEVC.
    base = ["ffmpeg", "-y", "-v", "error", "-skip_frame", "nokey"]
    rest = ["-i", str(path), "-filter_complex", ";".join(filters), *outputs]
    if hwaccel() and subprocess.run([*base, *hwaccel(), *rest], capture_output=True, text=True).returncode == 0:
        return results
    result = subprocess.run([*base, *rest], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"360 conversion failed: {result.stderr[-800:]}")
    return results
