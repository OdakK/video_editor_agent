"""The agent's tools: look at the rushes, then build, refine and render the edit."""

from __future__ import annotations

import functools
from pathlib import Path

from langchain_core.tools import BaseTool, ToolException, tool
from pydantic import BaseModel, Field

from .store import MediaStore
from .timeline import DEFAULT_TRACK, TimelineManager


class ClipSpec(BaseModel):
    shot_id: str
    start: float | None = Field(default=None, description="in point, seconds in the source file (default: shot start)")
    end: float | None = Field(default=None, description="out point, seconds in the source file (default: shot end)")


def _recoverable(func):
    """Turn expected errors (unknown id, empty timeline, ffmpeg failure) into tool errors that are
    sent back to the model as a message, so it can correct itself instead of crashing the run."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            raise ToolException(str(exc)) from exc
    return wrapper


def catalog_entry(shot: dict, detailed: bool = False) -> dict:
    """What the model sees of a shot: compact in the catalog, complete in get_shot."""
    entry = {"shot_id": shot["shot_id"], "rush": shot["asset_id"], "start": round(shot["start"], 2),
             "end": round(shot["end"], 2), "duration": round(shot["end"] - shot["start"], 2)}
    d = shot.get("description")
    if d:
        entry.update(summary=d["summary"], location=d["location"], shot_size=d["shot_size"], camera=d["camera"],
                     mood=d["mood"], issues=d["issues"],
                     moments=[f"{m['start']:.1f}-{m['end']:.1f}s [{m['quality']}] {m['description']}"
                              for m in d["moments"]])
        if detailed:
            entry.update(subjects=d["subjects"], tags=d["tags"])
    elif shot.get("captions"):
        entry["captions"] = shot["captions"]
    if shot.get("transcript"):
        entry["speech"] = shot["transcript"]
    return entry


def _timeline_view(clips: list[dict]) -> dict:
    return {"total_duration": clips[-1]["timeline_end"] if clips else 0.0,
            "clips": [{k: c[k] for k in ("index", "clip_id", "shot_id", "timeline_start", "timeline_end",
                                         "source_start", "source_end")} for c in clips]}


def replay_warnings(clips: list[dict]) -> list[str]:
    """Consecutive clips from the two lenses of one 360 clip should move forward in time: flag replays."""
    warnings = []
    for prev, cur in zip(clips, clips[1:], strict=False):
        a, b = prev["shot_id"].split("_lens"), cur["shot_id"].split("_lens")
        if len(a) == 2 and len(b) == 2 and a[0] == b[0] and a[1] != b[1] \
                and cur["source_start"] < prev["source_end"] - 0.1:
            warnings.append(f"Clip {cur['index']} replays {cur['source_start']:.1f}-{prev['source_end']:.1f}s already "
                            f"shown by clip {prev['index']} from the other lens (the sound repeats). Start it at "
                            f"{prev['source_end']:.1f}s unless a replay is intended.")
    return warnings


def build_tools(store: MediaStore, timeline: TimelineManager, output_dir: Path | str = "output") -> list[BaseTool]:
    output_dir = Path(output_dir)

    def shot_or_fail(shot_id: str) -> dict:
        shot = store.get_shot(shot_id)
        if shot is None:
            raise ValueError(f"Unknown shot_id {shot_id!r}: list or search shots first.")
        return shot

    @tool
    @_recoverable
    def list_shots(rush: str | None = None) -> list[dict]:
        """The catalog of shots: duration, what is seen (summary, location, shot size, camera, mood, issues),
        the moments with their quality and timecodes, and the speech. Start every edit here."""
        return [catalog_entry(s) for s in store.list_shots(asset_id=rush)]

    @tool
    @_recoverable
    def get_shot(shot_id: str) -> dict:
        """Everything about one shot, including each spoken sentence with its timecodes (cut between sentences)."""
        shot = shot_or_fail(shot_id)
        return {**catalog_entry(shot, detailed=True), "sentences": store.transcript_segments(shot_id)}

    @tool
    @_recoverable
    def semantic_search(query: str, limit: int = 5) -> list[dict]:
        """Find shots by meaning: what is said and what is seen. For descriptions like "the sunset over the city"."""
        return [r.as_dict() for r in store.semantic_search(query, limit)]

    @tool
    @_recoverable
    def keyword_search(query: str, limit: int = 5) -> list[dict]:
        """Find shots containing exact words (speech, descriptions, tags). FTS5 syntax: AND, OR, NOT, "phrase"."""
        return [r.as_dict() for r in store.keyword_search(query, limit)]

    @tool
    @_recoverable
    def build_timeline(clips: list[ClipSpec], name: str | None = None) -> dict:
        """Replace the whole edit with these clips, in this order. Each clip is a shot, optionally cut down
        with start and end (seconds in the source file). The best way to lay out a full edit in one step."""
        shots = []
        for spec in clips:
            shot = shot_or_fail(spec.shot_id)
            shots.append({**shot, "in": spec.start, "out": spec.end})
        clips = timeline.replace_all(shots, name)
        view = _timeline_view(clips)
        if warnings := replay_warnings(clips):
            view["warnings"] = warnings
        return view

    @tool
    @_recoverable
    def insert_shot(shot_id: str, start: float | None = None, end: float | None = None,
                    timeline_time: float | None = None) -> dict:
        """Add a shot, or the part of it between start and end (seconds in the source file), at the end of the
        timeline or before the clip playing at timeline_time."""
        return _timeline_view(timeline.insert_shot(shot_or_fail(shot_id), timeline_time, DEFAULT_TRACK, start, end))

    @tool
    @_recoverable
    def trim_clip(clip_id: str, source_start: float, source_end: float) -> dict:
        """Change which part of its source file a clip plays (seconds in the source file)."""
        return _timeline_view(timeline.trim_clip(clip_id, source_start, source_end))

    @tool
    @_recoverable
    def split_clip(timeline_time: float) -> dict:
        """Cut the clip playing at timeline_time (seconds in the timeline) into two clips."""
        return _timeline_view(timeline.split_at(timeline_time))

    @tool
    @_recoverable
    def remove_clip(clip_id: str) -> dict:
        """Remove a clip from the timeline (the source file is never touched)."""
        return _timeline_view(timeline.remove_clip(clip_id))

    @tool
    @_recoverable
    def move_clip(clip_id: str, new_index: int) -> dict:
        """Move a clip to a new position in the timeline (0 = first)."""
        return _timeline_view(timeline.move_clip(clip_id, new_index))

    @tool
    @_recoverable
    def describe_timeline() -> dict:
        """The current edit: total duration and clips in order, with timeline and source timecodes."""
        return _timeline_view(timeline.describe())

    @tool
    @_recoverable
    def render_preview(file_name: str = "preview.mp4", include_audio: bool = True) -> dict:
        """Render the timeline to an MP4 file with ffmpeg, in the output folder."""
        return timeline.render_preview(output_dir / Path(file_name).name, DEFAULT_TRACK, include_audio)

    tools = [list_shots, get_shot, semantic_search, keyword_search, build_timeline, insert_shot, trim_clip,
             split_clip, remove_clip, move_clip, describe_timeline, render_preview]
    for t in tools:
        t.handle_tool_error = True
    return tools
