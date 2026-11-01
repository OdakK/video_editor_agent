"""Non-destructive editing on an OpenTimelineIO timeline, and preview rendering with ffmpeg."""

from __future__ import annotations

import copy
import shlex
import subprocess
import threading
import uuid
from functools import wraps
from pathlib import Path
from typing import Any

import opentimelineio as otio

from .insta360 import flat_filter, hwaccel
from .probe import probe

DEFAULT_TRACK = "Video"
MIN_CLIP = 0.3  # seconds: shorter clips are almost always a mistake


def subclip(shot: dict[str, Any], start: float | None, end: float | None) -> dict[str, Any]:
    """The part of a shot between start and end (seconds in the source file), clamped to the shot."""
    lo = shot["start"] if start is None else max(shot["start"], start)
    hi = shot["end"] if end is None else min(shot["end"], end)
    if hi - lo < MIN_CLIP:
        raise ValueError(f"{shot['shot_id']}: {lo:.2f}-{hi:.2f}s is outside the shot ({shot['start']:.2f}-"
                         f"{shot['end']:.2f}s) or shorter than {MIN_CLIP}s.")
    return {**shot, "start": lo, "end": hi}

# One lock per timeline file: agents call tools in parallel (e.g. three insert_shot at once), and each
# edit is a load -> modify -> save cycle on the same .otio file, so concurrent edits would overwrite
# each other.
_locks: dict[Path, threading.RLock] = {}
_locks_guard = threading.Lock()


def _serialized(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        with _locks_guard:
            lock = _locks.setdefault(self.timeline_path, threading.RLock())
        with lock:
            return method(self, *args, **kwargs)
    return wrapper


class TimelineManager:
    """Load, edit and save one .otio file. Every edit returns the new state of the track."""

    def __init__(self, timeline_path: Path | str, default_name: str = "Agent rough cut") -> None:
        self.timeline_path = Path(timeline_path).expanduser().resolve()
        self.default_name = default_name

    # ------------------------------------------------------------------ editing

    @_serialized
    def initialize(self, name: str | None = None) -> list[dict[str, Any]]:
        """Create an empty timeline, replacing the current one."""
        timeline = self._empty_timeline(name or self.default_name)
        self._save(timeline)
        return self._summary(self._track(timeline))

    @_serialized
    def insert_shot(self, shot: dict[str, Any], timeline_time: float | None = None,
                    track_name: str = DEFAULT_TRACK, start: float | None = None,
                    end: float | None = None) -> list[dict[str, Any]]:
        """Insert a shot, or the part of it between `start` and `end` (seconds in the source file),
        before the clip playing at `timeline_time`, or at the end."""
        timeline = self._load()
        track = self._track(timeline, track_name)
        track.insert(self._insert_index(track, timeline_time), self._clip(subclip(shot, start, end)))
        self._save(timeline)
        return self._summary(track)

    @_serialized
    def replace_all(self, shots: list[dict[str, Any]], name: str | None = None) -> list[dict[str, Any]]:
        """Replace the whole edit with these shots, in order. Each may carry its own start and end."""
        timeline = self._empty_timeline(name or self.default_name)
        track = self._track(timeline)
        for shot in shots:
            track.append(self._clip(subclip(shot, shot.get("in"), shot.get("out"))))
        self._save(timeline)
        return self._summary(track)

    @_serialized
    def trim_clip(self, clip_id: str, source_start: float, source_end: float,
                  track_name: str = DEFAULT_TRACK) -> list[dict[str, Any]]:
        """Change which part of the source a clip plays (seconds in the source file)."""
        timeline = self._load()
        track = self._track(timeline, track_name)
        clip = track[self._index_of(track, clip_id)]
        available = clip.media_reference.available_range
        limit = available.duration.to_seconds() if available else source_end
        if not 0 <= source_start < source_end <= limit + 0.05:
            raise ValueError(f"Invalid range {source_start:.2f}-{source_end:.2f}s: the source lasts {limit:.2f}s.")
        if source_end - source_start < MIN_CLIP:
            raise ValueError(f"A clip must last at least {MIN_CLIP}s.")
        rate = clip.source_range.duration.rate
        clip.source_range = otio.opentime.TimeRange(
            otio.opentime.RationalTime.from_seconds(source_start, rate),
            otio.opentime.RationalTime.from_seconds(source_end - source_start, rate))
        self._save(timeline)
        return self._summary(track)

    @_serialized
    def split_at(self, timeline_time: float, track_name: str = DEFAULT_TRACK) -> list[dict[str, Any]]:
        """Cut the clip playing at `timeline_time` into two clips."""
        timeline = self._load()
        track = self._track(timeline, track_name)
        for index, segment in enumerate(self._summary(track)):
            if segment["timeline_start"] < timeline_time < segment["timeline_end"]:
                self._split_clip(track, index, timeline_time - segment["timeline_start"])
                self._save(timeline)
                return self._summary(track)
        raise ValueError(f"No clip is playing at {timeline_time:.2f}s on track {track_name}.")

    @_serialized
    def remove_clip(self, clip_id: str, track_name: str = DEFAULT_TRACK) -> list[dict[str, Any]]:
        timeline = self._load()
        track = self._track(timeline, track_name)
        del track[self._index_of(track, clip_id)]
        self._save(timeline)
        return self._summary(track)

    @_serialized
    def move_clip(self, clip_id: str, new_index: int, track_name: str = DEFAULT_TRACK) -> list[dict[str, Any]]:
        """Move a clip to a new position (0 = first) in the track."""
        timeline = self._load()
        track = self._track(timeline, track_name)
        index = self._index_of(track, clip_id)
        clip = track[index]
        del track[index]
        track.insert(max(0, min(new_index, len(track))), clip)
        self._save(timeline)
        return self._summary(track)

    @_serialized
    def remove_asset(self, asset_id: str, track_name: str = DEFAULT_TRACK) -> int:
        """Remove every clip that comes from a rush. Returns the number of clips removed."""
        timeline = self._load()
        track = self._track(timeline, track_name)
        indexes = [i for i, clip in enumerate(track) if clip.metadata.get("editing", {}).get("asset_id") == asset_id]
        for i in reversed(indexes):
            del track[i]
        if indexes:
            self._save(timeline)
        return len(indexes)

    @_serialized
    def describe(self, track_name: str = DEFAULT_TRACK) -> list[dict[str, Any]]:
        return self._summary(self._track(self._load(), track_name))

    # ------------------------------------------------------------------ rendering

    @_serialized
    def render_preview(self, output_path: Path | str, track_name: str = DEFAULT_TRACK,
                       include_audio: bool = True, ffmpeg: str = "ffmpeg") -> dict[str, Any]:
        """Render the track to MP4. Audio is dropped if any source has no audio track."""
        clips = self.describe(track_name)
        if not clips:
            raise ValueError("The timeline is empty: insert shots before rendering.")
        output = Path(output_path).expanduser().resolve()
        include_audio = include_audio and all(probe(c["asset_path"]).has_audio for c in clips)
        command = build_ffmpeg_command(clips, output, include_audio=include_audio, ffmpeg=ffmpeg)
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg not found: install it and make sure it is on PATH.") from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"ffmpeg failed: {exc.stderr[-2000:]}") from exc
        return {
            "output_path": str(output),
            "audio": include_audio,
            "duration": round(clips[-1]["timeline_end"], 3),
            "command": shlex.join(command),
        }

    # ------------------------------------------------------------------ internals

    def _load(self) -> otio.schema.Timeline:
        if self.timeline_path.exists():
            return otio.adapters.read_from_file(str(self.timeline_path))
        timeline = self._empty_timeline(self.default_name)
        self._save(timeline)
        return timeline

    def _save(self, timeline: otio.schema.Timeline) -> None:
        self.timeline_path.parent.mkdir(parents=True, exist_ok=True)
        otio.adapters.write_to_file(timeline, str(self.timeline_path))

    @staticmethod
    def _empty_timeline(name: str) -> otio.schema.Timeline:
        timeline = otio.schema.Timeline(name=name)
        timeline.tracks.append(otio.schema.Track(name=DEFAULT_TRACK, kind=otio.schema.TrackKind.Video))
        return timeline

    @staticmethod
    def _track(timeline: otio.schema.Timeline, name: str = DEFAULT_TRACK) -> otio.schema.Track:
        for track in timeline.tracks:
            if track.name == name:
                return track
        track = otio.schema.Track(name=name, kind=otio.schema.TrackKind.Video)
        timeline.tracks.append(track)
        return track

    @staticmethod
    def _clip(shot: dict[str, Any]) -> otio.schema.Clip:
        rate = float(shot.get("fps") or 24.0)
        duration = max(shot["end"] - shot["start"], 0.0)
        clip = otio.schema.Clip(name=shot["shot_id"])
        clip.metadata["editing"] = {
            "clip_id": uuid.uuid4().hex[:8],
            "shot_id": shot["shot_id"],
            "asset_id": shot["asset_id"],
            "asset_path": str(shot["asset_path"]),
            "captions": shot.get("captions", []),
            "source360": shot.get("source360"),  # full-quality original, used by the render
        }
        clip.media_reference = otio.schema.ExternalReference(
            target_url=str(shot["asset_path"]),
            available_range=otio.opentime.TimeRange(
                otio.opentime.RationalTime(0, rate),
                otio.opentime.RationalTime.from_seconds(shot.get("asset_duration") or shot["end"], rate)),
        )
        clip.source_range = otio.opentime.TimeRange(
            otio.opentime.RationalTime.from_seconds(shot["start"], rate),
            otio.opentime.RationalTime.from_seconds(duration, rate))
        return clip

    @staticmethod
    def _insert_index(track: otio.schema.Track, timeline_time: float | None) -> int:
        if timeline_time is None:
            return len(track)
        elapsed = 0.0
        for index, clip in enumerate(track):
            duration = clip.source_range.duration.to_seconds()
            if max(timeline_time, 0.0) < elapsed + duration:
                return index
            elapsed += duration
        return len(track)

    @staticmethod
    def _index_of(track: otio.schema.Track, clip_id: str) -> int:
        for index, clip in enumerate(track):
            if clip.metadata.get("editing", {}).get("clip_id") == clip_id:
                return index
        raise ValueError(f"No clip with id {clip_id}: call describe_timeline to get the current ids.")

    @staticmethod
    def _split_clip(track: otio.schema.Track, index: int, offset_seconds: float) -> None:
        clip = track[index]
        source = clip.source_range
        offset = otio.opentime.RationalTime.from_seconds(offset_seconds, source.duration.rate)
        first, second = copy.deepcopy(clip), copy.deepcopy(clip)
        first.source_range = otio.opentime.TimeRange(source.start_time, offset)
        second.source_range = otio.opentime.TimeRange(source.start_time + offset, source.duration - offset)
        for part, suffix in ((first, "A"), (second, "B")):
            part.name = f"{clip.name}_{suffix}"
            part.metadata["editing"] = {**clip.metadata.get("editing", {}), "clip_id": uuid.uuid4().hex[:8]}
        track[index] = first
        track.insert(index + 1, second)

    @staticmethod
    def _summary(track: otio.schema.Track) -> list[dict[str, Any]]:
        summary, elapsed = [], 0.0
        for index, clip in enumerate(track):
            meta = clip.metadata.get("editing", {})
            source = clip.source_range
            duration = source.duration.to_seconds()
            summary.append({
                "index": index,
                "clip_id": meta.get("clip_id"),
                "shot_id": meta.get("shot_id", clip.name),
                "asset_path": meta.get("asset_path") or clip.media_reference.target_url,
                "timeline_start": round(elapsed, 3),
                "timeline_end": round(elapsed + duration, 3),
                "source_start": round(source.start_time.to_seconds(), 3),
                "source_end": round((source.start_time + source.duration).to_seconds(), 3),
                "source360": _plain(meta.get("source360")),
            })
            elapsed += duration
        return summary


def _plain(value):
    """OTIO metadata values are C++ containers that die with their timeline: copy them to Python."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if hasattr(value, "items"):
        return {k: _plain(v) for k, v in value.items()}
    return [_plain(v) for v in value]


def build_ffmpeg_command(clips: list[dict[str, Any]], output: Path, *, include_audio: bool,
                         ffmpeg: str = "ffmpeg", size: tuple[int, int] | None = None) -> list[str]:
    """Trim each clip from its source, scale and pad to a common size, then concatenate.

    Clips from a 360 proxy are rendered from the original file instead, at full quality: the
    input is opened at the clip's start (-ss/-t) so only the seconds used are decoded."""
    if size is None:
        dims = []
        for c in clips:
            if c.get("source360"):
                dims.append((c["source360"]["width"], c["source360"]["height"]))
            else:
                info = probe(c["asset_path"])
                dims.append((info.width or 0, info.height or 0))
        size = (max(w for w, _ in dims) or 1920, max(h for _, h in dims) or 1080)
    width, height = size
    width, height = width - width % 2, height - height % 2  # libx264 needs even dimensions
    fit = (f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
           f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1")

    command, filters, streams = [ffmpeg, "-y"], [], []
    for i, clip in enumerate(clips):
        start, end = clip["source_start"], clip["source_end"]
        src = clip.get("source360")
        if src:
            command += [*hwaccel(), "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(src["path"])]
            filters.append(f"{flat_filter(src['stream'], src.get('crop'), src['width'], src['height'], i)},"
                           f"setpts=PTS-STARTPTS,{fit}[v{i}]")
            audio = f"[{i}:a:0]asetpts=PTS-STARTPTS[a{i}]"
        else:
            command += ["-i", clip["asset_path"]]
            filters.append(f"[{i}:v]trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS,{fit}[v{i}]")
            audio = f"[{i}:a]atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS[a{i}]"
        streams.append(f"[v{i}]")
        if include_audio:
            filters.append(audio)
            streams.append(f"[a{i}]")
    filters.append("".join(streams) + f"concat=n={len(clips)}:v=1:a={int(include_audio)}"
                   + ("[vout][aout]" if include_audio else "[vout]"))

    command += ["-filter_complex", ";".join(filters), "-map", "[vout]"]
    if include_audio:
        command += ["-map", "[aout]", "-c:a", "aac", "-b:a", "192k"]
    command += ["-c:v", "libx264", "-preset", "faster", "-crf", "23", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(output)]
    return command
