"""The agent's tools: search the analyzed rushes, then edit and render the timeline."""

from __future__ import annotations

import functools

from langchain_core.tools import BaseTool, ToolException, tool

from .store import MediaStore
from .timeline import DEFAULT_TRACK, TimelineManager


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


def build_tools(store: MediaStore, timeline: TimelineManager) -> list[BaseTool]:
    @tool
    @_recoverable
    def semantic_search(query: str, limit: int = 5) -> list[dict]:
        """Find shots by meaning: what is said in them and what is seen (captions).
        Use it for descriptions like "the part where she talks about the weather"."""
        return [r.as_dict() for r in store.semantic_search(query, limit)]

    @tool
    @_recoverable
    def keyword_search(query: str, limit: int = 5) -> list[dict]:
        """Find shots containing exact words, in the transcript or the captions.
        Accepts SQLite FTS5 syntax: AND, OR, NOT, NEAR, "exact phrase"."""
        return [r.as_dict() for r in store.keyword_search(query, limit)]

    @tool
    @_recoverable
    def list_shots(asset_id: str | None = None, limit: int | None = None) -> list[dict]:
        """List analyzed shots with timecodes, transcript and captions, optionally for one rush."""
        return store.list_shots(asset_id=asset_id, limit=limit)

    @tool
    @_recoverable
    def shots_for_segment(segment_id: str) -> list[dict]:
        """Return the shot(s) that contain a given transcript segment."""
        return store.shots_for_segment(segment_id)

    @tool
    @_recoverable
    def initialize_timeline(name: str | None = None) -> list[dict]:
        """Start a new empty timeline. This replaces the current edit."""
        return timeline.initialize(name)

    @tool
    @_recoverable
    def insert_shot(shot_id: str, timeline_time: float | None = None, track: str = DEFAULT_TRACK) -> list[dict]:
        """Put a shot on the timeline: at the end, or before the clip playing at timeline_time (seconds)."""
        shot = store.get_shot(shot_id)
        if shot is None:
            raise ValueError(f"Unknown shot_id {shot_id!r}: search or list shots first.")
        return timeline.insert_shot(shot, timeline_time, track)

    @tool
    @_recoverable
    def split_clip(timeline_time: float, track: str = DEFAULT_TRACK) -> list[dict]:
        """Cut the clip playing at timeline_time (seconds) into two clips."""
        return timeline.split_at(timeline_time, track)

    @tool
    @_recoverable
    def remove_clip(clip_id: str, track: str = DEFAULT_TRACK) -> list[dict]:
        """Remove a clip from the timeline (the source file is never touched)."""
        return timeline.remove_clip(clip_id, track)

    @tool
    @_recoverable
    def move_clip(clip_id: str, new_index: int, track: str = DEFAULT_TRACK) -> list[dict]:
        """Move a clip to a new position in the timeline (0 = first)."""
        return timeline.move_clip(clip_id, new_index, track)

    @tool
    @_recoverable
    def describe_timeline(track: str = DEFAULT_TRACK) -> list[dict]:
        """Current edit: clips in order, with clip ids and timeline and source timecodes."""
        return timeline.describe(track)

    @tool
    @_recoverable
    def render_preview(output_path: str = "preview.mp4", include_audio: bool = True,
                       track: str = DEFAULT_TRACK) -> dict:
        """Render the timeline to an MP4 file with ffmpeg."""
        return timeline.render_preview(output_path, track, include_audio)

    tools = [semantic_search, keyword_search, list_shots, shots_for_segment, initialize_timeline,
             insert_shot, split_clip, remove_clip, move_clip, describe_timeline, render_preview]
    for t in tools:
        t.handle_tool_error = True
    return tools
