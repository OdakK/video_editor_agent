"""Synthetic demo footage, to try the agent and the UI without real rushes or ML models.

ffmpeg generates three short rushes with test patterns and tones. Their "analysis" (transcripts
and captions) is written by hand: it describes the story of a fictional travel video, not the
actual pixels. Everything else (store, search, timeline, rendering) is the real code.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from .analysis.merge import merge_analysis
from .schemas import (
    AnalysisBundle,
    AudioAnalysisResult,
    FrameCaption,
    MediaAsset,
    SceneSegment,
    TranscriptSegment,
    VideoAnalysisResult,
)

# (asset_id, seconds, tone in Hz, [(shot start, end, ffmpeg video source, caption)], [(start, end, speech)])
BEACH = "gradients=size=1280x720:rate=25:speed=0.02"
LIFE = "life=size=1280x720:rate=25:mold=10"

RUSHES = [
    ("beach", 12, 330, [
        (0, 6, f"{BEACH}:c0=0x0b5ea8:c1=0x6ec6f0", "a wide shot of a beach with waves"),
        (6, 12, f"{BEACH}:c0=0xf2c14e:c1=0xf78154", "a woman looking at the sunset"),
    ], [
        (0.5, 5.5, "Look at the waves, the sea is so calm this morning."),
        (6.5, 11.5, "The sky turns orange at sunset, it is the best moment of the day."),
    ]),
    ("interview", 14, 220, [
        (0, 7, "smptehdbars=size=1280x720:rate=25", "a man sitting in an office, facing the camera"),
        (7, 14, "testsrc2=size=1280x720:rate=25", "a close-up of a man talking"),
    ], [
        (0.5, 6.5, "I started the company in Lisbon three years ago, with two friends."),
        (7.5, 13.5, "Our first customer was a small surf school on the coast."),
    ]),
    ("city", 10, 440, [
        (0, 5, "mandelbrot=size=1280x720:rate=25", "a busy street at night with neon lights"),
        (5, 10, f"{LIFE}:life_color=0x2ec4b6:death_color=0x011627", "people crossing a square"),
    ], [
        (5.5, 9.5, "The city never sleeps, there is always someone in the square."),
    ]),
]


def build_demo(output_dir: Path | str = "demo") -> Path:
    """Generate the rushes, keyframes and analysis bundle. Returns the bundle path.

    An existing demo is reused as is, so rushes imported into it from the UI are kept.
    Delete the folder to start from a fresh demo.
    """
    out = Path(output_dir).resolve()
    if (out / "analysis_bundle.json").exists():
        return out / "analysis_bundle.json"
    rushes, keyframes = out / "rushes", out / "keyframes"
    rushes.mkdir(parents=True, exist_ok=True)
    keyframes.mkdir(parents=True, exist_ok=True)

    assets, audio, video = [], {}, {}
    for asset_id, seconds, tone, shots, speech in RUSHES:
        path = rushes / f"{asset_id}.mp4"
        if not path.exists():
            _render_rush(path, seconds, tone, shots)

        scenes = []
        for i, (start, end, _, caption) in enumerate(shots):
            shot_id = f"{asset_id}_shot_{i:03d}"
            keyframe = keyframes / f"{shot_id}.jpg"
            if not keyframe.exists():
                _run(["ffmpeg", "-y", "-v", "error", "-ss", str((start + end) / 2), "-i", str(path),
                      "-frames:v", "1", "-vf", "scale=480:-2", str(keyframe)])
            scenes.append(SceneSegment(shot_id=shot_id, start=start, end=end, keyframe_path=keyframe, captions=[
                FrameCaption(caption_id=f"{shot_id}_cap_00", timecode=(start + end) / 2, text=caption)]))

        assets.append(MediaAsset(asset_id=asset_id, path=path, duration=seconds, fps=25.0, has_audio=True))
        video[asset_id] = VideoAnalysisResult(asset_id=asset_id, scenes=scenes)
        audio[asset_id] = AudioAnalysisResult(asset_id=asset_id, language="en", segments=[
            TranscriptSegment(segment_id=f"{asset_id}_seg_{i:03d}", start=s, end=e, text=text)
            for i, (s, e, text) in enumerate(speech)])

    bundle: AnalysisBundle = merge_analysis(assets, audio, video)
    bundle_path = out / "analysis_bundle.json"
    bundle_path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
    return bundle_path


def _render_rush(path: Path, seconds: int, tone: int, shots: list) -> None:
    inputs, chain = [], []
    for i, (start, end, source, _) in enumerate(shots):
        inputs += ["-f", "lavfi", "-t", str(end - start), "-i", source]
        chain.append(f"[{i}:v]format=yuv420p,setsar=1[v{i}]")
    n = len(shots)
    graph = ";".join(chain) + ";" + "".join(f"[v{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[v]"
    _run(["ffmpeg", "-y", "-v", "error", *inputs, "-f", "lavfi", "-i", f"sine=frequency={tone}:duration={seconds}",
          "-filter_complex", graph, "-map", "[v]", "-map", f"{n}:a", "-c:v", "libx264", "-preset", "veryfast",
          "-c:a", "aac", "-shortest", str(path)])


def _run(command: list[str]) -> None:
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg not found: install it to build the demo") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg failed: {exc.stderr[-1500:]}") from exc
