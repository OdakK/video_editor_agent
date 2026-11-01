"""Align transcript segments with the shots they overlap ("timeline atoms")."""

from __future__ import annotations

from collections.abc import Iterable

from ..schemas import (
    AnalysisBundle,
    AudioAnalysisResult,
    MediaAsset,
    TimelineAtom,
    TranscriptSegment,
    VideoAnalysisResult,
)


def merge_analysis(
    media_assets: Iterable[MediaAsset],
    audio_index: dict[str, AudioAnalysisResult],
    video_index: dict[str, VideoAnalysisResult],
) -> AnalysisBundle:
    media_assets = list(media_assets)
    atoms: list[TimelineAtom] = []
    for asset in media_assets:
        video = video_index.get(asset.asset_id)
        if video is None:
            continue
        audio = audio_index.get(asset.asset_id)
        segments = audio.segments if audio else []
        for scene in video.scenes:
            atoms.append(TimelineAtom(
                asset_id=asset.asset_id,
                shot_id=scene.shot_id,
                start=scene.start,
                end=scene.end,
                transcript_segment_ids=overlapping_segments(segments, scene.start, scene.end),
                captions=[caption.text for caption in scene.captions],
                tags=sorted(set(scene.tags)),
            ))
    return AnalysisBundle(media_assets=media_assets, audio_index=audio_index,
                          video_index=video_index, timeline_atoms=atoms)


def overlapping_segments(segments: Iterable[TranscriptSegment], start: float, end: float) -> list[str]:
    """Ids of the segments that overlap [start, end). Touching boundaries do not count."""
    return [s.segment_id for s in segments if s.start < end and start < s.end]
