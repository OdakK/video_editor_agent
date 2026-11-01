"""Data model shared by the analysis pipeline, the media store and the agent."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class MediaAsset(BaseModel):
    """A rush available on disk."""

    asset_id: str
    path: Path
    duration: float | None = Field(default=None, description="Duration in seconds, if known.")
    fps: float | None = Field(default=None, description="Frames per second.")
    has_audio: bool = Field(default=True, description="Whether the file has an audio track.")


class WordTiming(BaseModel):
    """Word-level timing from speech recognition."""

    word: str
    start: float = Field(description="Start time in seconds.")
    end: float = Field(description="End time in seconds.")
    probability: float | None = None


class TranscriptSegment(BaseModel):
    """A transcribed sentence or phrase, with optional speaker."""

    segment_id: str
    start: float
    end: float
    text: str
    words: list[WordTiming] = Field(default_factory=list)
    speaker: str | None = None


class AudioAnalysisResult(BaseModel):
    """Speech recognition output for one asset."""

    asset_id: str
    language: str | None = None
    segments: list[TranscriptSegment] = Field(default_factory=list)


class FrameCaption(BaseModel):
    """Caption generated for a representative frame of a shot."""

    caption_id: str
    timecode: float = Field(description="Timestamp inside the asset (seconds).")
    text: str
    confidence: float | None = None
    image_path: Path | None = None


class SceneSegment(BaseModel):
    """A shot detected in the video track."""

    shot_id: str
    start: float
    end: float
    keyframe_path: Path | None = None
    captions: list[FrameCaption] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class VideoAnalysisResult(BaseModel):
    """Shots and descriptors for one asset."""

    asset_id: str
    scenes: list[SceneSegment] = Field(default_factory=list)


class TimelineAtom(BaseModel):
    """One shot with everything known about it: timecodes, what is said, what is seen."""

    asset_id: str
    shot_id: str
    start: float
    end: float
    transcript_segment_ids: list[str] = Field(default_factory=list)
    captions: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class AnalysisBundle(BaseModel):
    """Output of the analysis pipeline, loaded by the media store."""

    media_assets: list[MediaAsset]
    audio_index: dict[str, AudioAnalysisResult]
    video_index: dict[str, VideoAnalysisResult]
    timeline_atoms: list[TimelineAtom]
