from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, ConfigDict


class MediaAsset(BaseModel):
    """Represents a rush available on disk."""

    asset_id: str
    path: Path
    duration: Optional[float] = Field(
        default=None, description="Duration in seconds if known."
    )
    fps: Optional[float] = Field(default=None, description="Frames per second.")
    has_audio: bool = Field(
        default=True, description="Indicates whether an audio track is available."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


class WordTiming(BaseModel):
    """Word-level timing information extracted from ASR."""

    word: str
    start: float = Field(description="Start time in seconds.")
    end: float = Field(description="End time in seconds.")
    probability: Optional[float] = Field(default=None)


class TranscriptSegment(BaseModel):
    """Segment-level transcription with optional speaker metadata."""

    segment_id: str
    start: float
    end: float
    text: str
    words: List[WordTiming] = Field(default_factory=list)
    speaker: Optional[str] = None


class AudioAnalysisResult(BaseModel):
    """Aggregated ASR output for a media asset."""

    asset_id: str
    language: Optional[str] = None
    segments: List[TranscriptSegment] = Field(default_factory=list)


class FrameCaption(BaseModel):
    """Caption or tag generated for a representative video frame."""

    caption_id: str
    timecode: float = Field(description="Timestamp inside the asset (seconds).")
    text: str
    confidence: Optional[float] = None
    image_path: Optional[Path] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class SceneSegment(BaseModel):
    """Scene/shot boundaries detected in the video track."""

    shot_id: str
    start: float
    end: float
    keyframe_path: Optional[Path] = None
    captions: List[FrameCaption] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class VideoAnalysisResult(BaseModel):
    """Collection of scene segments and descriptors for a media asset."""

    asset_id: str
    scenes: List[SceneSegment] = Field(default_factory=list)


class TimelineAtom(BaseModel):
    """Normalized unit merging audio and visual annotations."""

    asset_id: str
    shot_id: str
    start: float
    end: float
    transcript_segment_ids: List[str] = Field(default_factory=list)
    captions: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)


class AnalysisBundle(BaseModel):
    """Final data structure persisted for the editing agent."""

    media_assets: List[MediaAsset]
    audio_index: dict[str, AudioAnalysisResult]
    video_index: dict[str, VideoAnalysisResult]
    timeline_atoms: List[TimelineAtom]
