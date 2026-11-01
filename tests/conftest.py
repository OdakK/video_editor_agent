"""Shared fixtures: a small hand-made analysis bundle and a deterministic encoder.

The real pipeline needs Whisper, BLIP and sentence-transformers. The tests replace their output
with a fixed bundle and the encoder with HashingEncoder (bag of words), so they run in seconds, offline.
"""

import pytest

from video_agent.analysis.merge import merge_analysis
from video_agent.schemas import (
    AudioAnalysisResult,
    FrameCaption,
    MediaAsset,
    SceneSegment,
    TranscriptSegment,
    VideoAnalysisResult,
)
from video_agent.store import HashingEncoder, MediaStore


def make_bundle(path_a="/rushes/beach.mp4", path_b="/rushes/interview.mp4"):
    assets = [
        MediaAsset(asset_id="beach", path=path_a, duration=10.0, fps=25.0, has_audio=True),
        MediaAsset(asset_id="interview", path=path_b, duration=8.0, fps=25.0, has_audio=True),
    ]
    audio = {
        "beach": AudioAnalysisResult(asset_id="beach", segments=[
            TranscriptSegment(segment_id="beach_seg_000", start=0.5, end=4.0,
                              text="Look at the waves, the sea is so calm today."),
            TranscriptSegment(segment_id="beach_seg_001", start=6.0, end=9.5,
                              text="The sky is completely blue, not a single cloud."),
        ]),
        "interview": AudioAnalysisResult(asset_id="interview", segments=[
            TranscriptSegment(segment_id="interview_seg_000", start=0.0, end=7.5,
                              text="I started the company in Lisbon three years ago."),
        ]),
    }
    video = {
        "beach": VideoAnalysisResult(asset_id="beach", scenes=[
            SceneSegment(shot_id="beach_shot_000", start=0.0, end=5.0, captions=[
                FrameCaption(caption_id="c0", timecode=2.5, text="a wide shot of a beach with waves")]),
            SceneSegment(shot_id="beach_shot_001", start=5.0, end=10.0, captions=[
                FrameCaption(caption_id="c1", timecode=7.5, text="a woman looking at the sky")]),
        ]),
        "interview": VideoAnalysisResult(asset_id="interview", scenes=[
            SceneSegment(shot_id="interview_shot_000", start=0.0, end=8.0, captions=[
                FrameCaption(caption_id="c2", timecode=4.0, text="a man sitting in an office")]),
        ]),
    }
    return merge_analysis(assets, audio, video)


@pytest.fixture
def bundle():
    return make_bundle()


@pytest.fixture
def encoder():
    return HashingEncoder()


@pytest.fixture
def store(tmp_path, bundle, encoder):
    s = MediaStore(tmp_path / "store.db", encoder=encoder)
    s.ingest_bundle(bundle)
    return s
