from __future__ import annotations

import copy
import uuid
import json
from pathlib import Path
from typing import Any, Optional
import shlex
import subprocess

import opentimelineio as otio


class TimelineManager:
    """Small helper around OTIO timelines with basic editing primitives."""

    def __init__(
        self,
        timeline_path: Path | str,
        *,
        default_name: str = "Agent Edit Timeline",
    ) -> None:
        self.timeline_path = Path(timeline_path).expanduser().resolve()
        self.default_name = default_name
        self._metadata_cache: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def initialize(self, name: Optional[str] = None) -> list[dict[str, Any]]:
        """Create a fresh empty timeline, overwriting any existing file."""
        timeline = self._create_empty_timeline(name or self.default_name)
        self._save_timeline(timeline)
        return self._summarize_track(self._get_track(timeline))

    def insert_shot(
        self,
        shot: dict[str, Any],
        *,
        timeline_time: float | None = None,
        track_name: str = "Video",
    ) -> list[dict[str, Any]]:
        """Insert a shot into the requested track."""
        timeline = self._load_or_initialize()
        track = self._get_track(timeline, track_name)

        clip = self._build_clip(shot)
        insert_index = self._compute_insert_index(track, timeline_time)
        track.insert(insert_index, clip)

        self._save_timeline(timeline)
        return self._summarize_track(track)

    def render_preview(
        self,
        output_path: Path | str,
        *,
        track_name: str = "Video",
        include_audio: bool = True,
        ffmpeg_binary: str = "ffmpeg",
    ) -> dict[str, Any]:
        """Render the current timeline to a video using ffmpeg."""
        timeline = self._load_or_initialize()
        track = self._get_track(timeline, track_name)
        clips = self._summarize_track(track)
        if not clips:
            raise ValueError("Timeline vide : insérer des plans avant de lancer un rendu.")

        resolved_output = Path(output_path).expanduser().resolve()
        include_audio = include_audio and self._timeline_has_audio(clips)

        command = self._build_ffmpeg_command(
            clips, resolved_output, include_audio=include_audio, ffmpeg_binary=ffmpeg_binary
        )
        try:
            completed = subprocess.run(
                command, check=True, capture_output=True, text=True
            )
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg introuvable sur le système.") from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Échec du rendu ffmpeg : {exc.stderr or exc.stdout}"
            ) from exc

        return {
            "message": (
                "Rendu terminé."
                if include_audio
                else "Rendu terminé (sans audio – piste audio absente sur certains rushes)."
            ),
            "output_path": str(resolved_output),
            "command": " ".join(shlex.quote(arg) for arg in command),
            "stdout": completed.stdout.strip(),
            "clips": clips,
        }

    def split_at_time(
        self, timeline_time: float, *, track_name: str = "Video"
    ) -> list[dict[str, Any]]:
        """Split the clip covering the given timeline timestamp."""
        if timeline_time < 0:
            raise ValueError("timeline_time must be positive.")

        timeline = self._load_or_initialize()
        track = self._get_track(timeline, track_name)
        segments = self._enumerate_segments(track)
        for index, segment in enumerate(segments):
            if timeline_time <= segment["timeline_start"]:
                continue
            if timeline_time >= segment["timeline_end"]:
                continue

            offset = timeline_time - segment["timeline_start"]
            self._split_clip(track, index, offset)
            self._save_timeline(timeline)
            return self._summarize_track(track)

        raise ValueError(
            f"Aucun plan ne couvre la position {timeline_time:.2f}s sur la piste {track_name}."
        )

    def describe(self, track_name: str = "Video") -> list[dict[str, Any]]:
        """Return a lightweight summary of the requested track."""
        timeline = self._load_or_initialize()
        track = self._get_track(timeline, track_name)
        return self._summarize_track(track)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_or_initialize(self) -> otio.schema.Timeline:
        if self.timeline_path.exists():
            return otio.adapters.read_from_file(str(self.timeline_path))
        timeline = self._create_empty_timeline(self.default_name)
        self._save_timeline(timeline)
        return timeline

    def _create_empty_timeline(self, name: str) -> otio.schema.Timeline:
        timeline = otio.schema.Timeline(name=name)
        video_track = otio.schema.Track(name="Video", kind=otio.schema.TrackKind.Video)
        timeline.tracks.append(video_track)
        return timeline

    def _save_timeline(self, timeline: otio.schema.Timeline) -> None:
        self.timeline_path.parent.mkdir(parents=True, exist_ok=True)
        otio.adapters.write_to_file(timeline, str(self.timeline_path))

    def _get_track(
        self, timeline: otio.schema.Timeline, track_name: str = "Video"
    ) -> otio.schema.Track:
        for track in timeline.tracks:
            if track.name == track_name:
                return track
        track = otio.schema.Track(name=track_name, kind=otio.schema.TrackKind.Video)
        timeline.tracks.append(track)
        return track

    def _build_clip(self, shot: dict[str, Any]) -> otio.schema.Clip:
        rate = float(shot.get("fps") or 24.0)
        duration = max(shot["end"] - shot["start"], 0.0)

        clip = otio.schema.Clip(name=shot["shot_id"])
        clip.metadata["editing"] = {
            "clip_id": self._generate_clip_id(),
            "shot_id": shot["shot_id"],
            "asset_id": shot["asset_id"],
            "captions": shot.get("captions", []),
            "tags": shot.get("tags", []),
            "asset_path": str(shot.get("asset_path") or ""),
        }

        clip.media_reference = otio.schema.ExternalReference(
            target_url=str(shot["asset_path"]),
            available_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime.from_seconds(0, rate),
                duration=otio.opentime.RationalTime.from_seconds(
                    shot.get("asset_duration") or duration, rate
                ),
            ),
        )
        clip.source_range = otio.opentime.TimeRange(
            start_time=otio.opentime.RationalTime.from_seconds(shot["start"], rate),
            duration=otio.opentime.RationalTime.from_seconds(duration, rate),
        )
        return clip

    def _compute_insert_index(
        self, track: otio.schema.Track, timeline_time: Optional[float]
    ) -> int:
        if timeline_time is None:
            return len(track)

        position = max(float(timeline_time), 0.0)
        accumulated = 0.0
        for idx, clip in enumerate(track):
            clip_duration = self._clip_duration_seconds(clip)
            if position <= accumulated:
                return idx
            if position < accumulated + clip_duration:
                return idx
            accumulated += clip_duration
        return len(track)

    def _clip_duration_seconds(self, clip: otio.schema.Clip) -> float:
        duration = self._clip_time_range(clip).duration
        return duration.to_seconds()

    def _clip_time_range(self, clip: otio.schema.Clip) -> otio.opentime.TimeRange:
        if clip.source_range is not None:
            return clip.source_range
        available = clip.available_range()
        if available is not None:
            return available
        raise ValueError("Clip missing timing information.")

    def _enumerate_segments(self, track: otio.schema.Track) -> list[dict[str, Any]]:
        segments: list[dict[str, Any]] = []
        cursor = 0.0
        for clip in track:
            duration = self._clip_duration_seconds(clip)
            segments.append(
                {
                    "clip": clip,
                    "timeline_start": cursor,
                    "timeline_end": cursor + duration,
                }
            )
            cursor += duration
        return segments

    def _split_clip(
        self, track: otio.schema.Track, index: int, offset_seconds: float
    ) -> None:
        clip = track[index]
        time_range = self._clip_time_range(clip)
        duration_seconds = time_range.duration.to_seconds()
        if offset_seconds <= 0 or offset_seconds >= duration_seconds:
            raise ValueError("Cut position must lie within the clip bounds.")

        rate = time_range.duration.rate or time_range.start_time.rate or 24.0
        offset = otio.opentime.RationalTime.from_seconds(offset_seconds, rate)

        first = copy.deepcopy(clip)
        second = copy.deepcopy(clip)

        first.source_range = otio.opentime.TimeRange(
            start_time=time_range.start_time,
            duration=offset,
        )
        second.source_range = otio.opentime.TimeRange(
            start_time=time_range.start_time + offset,
            duration=time_range.duration - offset,
        )

        first.metadata["editing"] = dict(clip.metadata.get("editing", {}))
        second.metadata["editing"] = dict(clip.metadata.get("editing", {}))
        first.metadata["editing"]["clip_id"] = self._generate_clip_id()
        second.metadata["editing"]["clip_id"] = self._generate_clip_id()

        if first.name == second.name:
            first.name = f"{clip.name}_A"
            second.name = f"{clip.name}_B"

        track[index] = first
        track.insert(index + 1, second)

    def _summarize_track(self, track: otio.schema.Track) -> list[dict[str, Any]]:
        summary: list[dict[str, Any]] = []
        cursor = 0.0
        for clip in track:
            duration = self._clip_duration_seconds(clip)
            display = clip.metadata.get("editing", {})
            time_range = self._clip_time_range(clip)
            asset_path = display.get("asset_path") or self._clip_media_path(clip)
            summary.append(
                {
                    "clip_id": display.get("clip_id"),
                    "shot_id": display.get("shot_id", clip.name),
                    "asset_id": display.get("asset_id"),
                    "asset_path": asset_path,
                    "timeline_start": cursor,
                    "timeline_end": cursor + duration,
                    "duration": duration,
                    "source_start": time_range.start_time.to_seconds(),
                    "source_end": (time_range.start_time + time_range.duration).to_seconds(),
                    "name": clip.name,
                }
            )
            cursor += duration
        return summary

    def _generate_clip_id(self) -> str:
        return uuid.uuid4().hex[:8]

    def _timeline_has_audio(self, clips: list[dict[str, Any]]) -> bool:
        """Check whether every asset referenced has at least one audio stream."""
        cache: dict[str, bool] = {}
        for clip in clips:
            asset_path = clip.get("asset_path")
            if not asset_path:
                return False
            if asset_path not in cache:
                cache[asset_path] = self._asset_has_audio(asset_path)
            if not cache[asset_path]:
                return False
        return True

    def _asset_has_audio(self, asset_path: str) -> bool:
        return self._get_asset_metadata(asset_path).get("has_audio", True)

    def _build_ffmpeg_command(
        self,
        clips: list[dict[str, Any]],
        output_path: Path,
        *,
        include_audio: bool,
        ffmpeg_binary: str,
    ) -> list[str]:
        cmd: list[str] = [ffmpeg_binary, "-y"]
        filter_parts: list[str] = []
        concat_inputs = []

        metadata_cache: dict[str, dict[str, Any]] = {}
        target_width = 0
        target_height = 0

        for clip in clips:
            asset_path = clip.get("asset_path")
            if not asset_path:
                raise ValueError("Impossible de récupérer le chemin média pour un clip.")
            metadata = metadata_cache.setdefault(
                asset_path, self._get_asset_metadata(asset_path)
            )
            width = metadata.get("width")
            height = metadata.get("height")
            if width:
                target_width = max(target_width, int(width))
            if height:
                target_height = max(target_height, int(height))

        if target_width <= 0 or target_height <= 0:
            target_width, target_height = 1920, 1080

        for idx, clip in enumerate(clips):
            asset_path = clip.get("asset_path")
            if not asset_path:
                raise ValueError("Impossible de récupérer le chemin média pour un clip.")
            cmd.extend(["-i", asset_path])
            start = clip["source_start"]
            end = clip["source_end"]
            scale_filter = (
                f"[{idx}:v]trim=start={start:.3f}:end={end:.3f},"
                f"setpts=PTS-STARTPTS,"
                f"scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,"
                f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2[v{idx}]"
            )
            filter_parts.append(
                scale_filter
            )
            if include_audio:
                filter_parts.append(
                    f"[{idx}:a]atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS[a{idx}]"
                )
                concat_inputs.append(f"[v{idx}][a{idx}]")
            else:
                concat_inputs.append(f"[v{idx}]")

        if include_audio:
            filter_parts.append(
                "".join(concat_inputs) + f"concat=n={len(clips)}:v=1:a=1[vout][aout]"
            )
        else:
            filter_parts.append(
                "".join(concat_inputs) + f"concat=n={len(clips)}:v=1:a=0[vout]"
            )

        filter_complex = ";".join(filter_parts)
        cmd.extend(["-filter_complex", filter_complex])
        cmd.extend(["-map", "[vout]"])
        if include_audio:
            cmd.extend(["-map", "[aout]"])
        cmd.extend(["-c:v", "libx264", "-preset", "faster", "-crf", "23"])
        if include_audio:
            cmd.extend(["-c:a", "aac", "-b:a", "192k"])
        cmd.extend(["-movflags", "+faststart", str(output_path)])
        return cmd

    def _clip_media_path(self, clip: otio.schema.Clip) -> Optional[str]:
        ref = clip.media_reference
        if isinstance(ref, otio.schema.ExternalReference) and ref.target_url:
            return str(Path(ref.target_url).expanduser())
        return None

    def _get_asset_metadata(self, asset_path: str) -> dict[str, Any]:
        resolved = str(Path(asset_path).expanduser())
        if resolved in self._metadata_cache:
            return self._metadata_cache[resolved]

        metadata = {"width": None, "height": None, "has_audio": True}
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "stream=codec_type,width,height",
                    "-of",
                    "json",
                    resolved,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            data = json.loads(result.stdout or "{}")
            width = metadata["width"]
            height = metadata["height"]
            has_audio = False
            for stream in data.get("streams", []):
                codec_type = stream.get("codec_type")
                if codec_type == "video":
                    if width is None and stream.get("width") is not None:
                        width = int(stream["width"])
                    if height is None and stream.get("height") is not None:
                        height = int(stream["height"])
                elif codec_type == "audio":
                    has_audio = True
            metadata = {"width": width, "height": height, "has_audio": has_audio}
        except (FileNotFoundError, subprocess.CalledProcessError, json.JSONDecodeError):
            metadata = {"width": None, "height": None, "has_audio": True}

        self._metadata_cache[resolved] = metadata
        return metadata
