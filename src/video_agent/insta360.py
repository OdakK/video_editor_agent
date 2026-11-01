"""Turn 360° camera files (Insta360 .insv) into flat videos the editor can cut.

A .insv file is an MP4 container holding the two fisheye lenses of the camera: either two square
video streams (one per lens, e.g. X3/X4 at full resolution), or one 2:1 stream with both circles
side by side (lower resolutions and the .lrv proxies). Older models write one file per lens.

Each lens covers about 200°, so it can be re-projected on its own into a clean flat view with
ffmpeg's v360 filter: no stitching, so no seam. A typical vlog clip gives two views: the scene in
front of the camera, and the person holding it (selfie). The editor then picks from both.

Not handled: Insta360's gyro stabilization (FlowState) and horizon leveling, which need their
SDK; free reframing to any direction (yaw/pitch) during the edit.
"""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path

EXTENSIONS_360 = {".insv", ".lrv"}
LENS_FOV = 200  # degrees covered by one Insta360 fisheye lens

ORIENTATIONS = {
    "vertical": {"w": 1080, "h": 1920, "h_fov": 75, "v_fov": 110},
    "horizontal": {"w": 1920, "h": 1080, "h_fov": 110, "v_fov": 70},
}


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


def flatten(path: Path, out_dir: Path, base_name: str, orientation: str = "vertical") -> list[Path]:
    """Write one flat MP4 per lens: <base_name>_lens1.mp4, <base_name>_lens2.mp4. Returns their paths."""
    lenses = find_lenses(path)
    o = ORIENTATIONS[orientation]
    project = (f"v360=input=fisheye:output=flat:ih_fov={LENS_FOV}:iv_fov={LENS_FOV}:h_fov={o['h_fov']}"
               f":v_fov={o['v_fov']}:w={o['w']}:h={o['h']}:interp=lanczos")
    filters, outputs = [], []
    out_dir.mkdir(parents=True, exist_ok=True)
    has_audio = bool(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                                     "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout.strip())
    for i, lens in enumerate(lenses, 1):
        chain = f"{lens.crop}," if lens.crop else ""
        filters.append(f"[0:v:{lens.stream}]{chain}{project},format=yuv420p[v{i}]")
        target = out_dir / f"{base_name}_lens{i}.mp4"
        outputs += ["-map", f"[v{i}]", *(["-map", "0:a:0", "-c:a", "aac"] if has_audio else []),
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-movflags", "+faststart", str(target)]
    command = ["ffmpeg", "-y", "-v", "error", "-i", str(path), "-filter_complex", ";".join(filters), *outputs]
    if platform.system() == "Darwin":  # hardware HEVC decoding: about 3x faster on 8K files
        hw = command[:4] + ["-hwaccel", "videotoolbox"] + command[4:]
        if subprocess.run(hw, capture_output=True, text=True).returncode == 0:
            return [out_dir / f"{base_name}_lens{i}.mp4" for i in range(1, len(lenses) + 1)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"360 conversion failed: {result.stderr[-800:]}")
    return [out_dir / f"{base_name}_lens{i}.mp4" for i in range(1, len(lenses) + 1)]
