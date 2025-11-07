from __future__ import annotations

from typing import Dict, Iterable, List

from .schemas import (
    AnalysisBundle,
    AudioAnalysisResult,
    MediaAsset,
    SceneSegment,
    TimelineAtom,
    TranscriptSegment,
    VideoAnalysisResult,
)


def merge_analysis(
    media_assets: Iterable[MediaAsset],
    audio_index: Dict[str, AudioAnalysisResult],
    video_index: Dict[str, VideoAnalysisResult],
) -> AnalysisBundle:
    media_assets = list(media_assets)
    timeline_atoms: List[TimelineAtom] = []

    for asset in media_assets:
        audio = audio_index.get(asset.asset_id)
        video = video_index.get(asset.asset_id)
        if video is None:
            continue

        audio_segments = audio.segments if audio else []
        for scene in video.scenes:
            transcript_ids = _collect_overlapping_segments(
                audio_segments, scene.start, scene.end
            )
            captions = [caption.text for caption in scene.captions]
            tags = set(scene.tags)

            timeline_atoms.append(
                TimelineAtom(
                    asset_id=asset.asset_id,
                    shot_id=scene.shot_id,
                    start=scene.start,
                    end=scene.end,
                    transcript_segment_ids=transcript_ids,
                    captions=captions,
                    tags=sorted(tags),
                )
            )

    return AnalysisBundle(
        media_assets=media_assets,
        audio_index=audio_index,
        video_index=video_index,
        timeline_atoms=timeline_atoms,
    )


def _collect_overlapping_segments(
    segments: Iterable[TranscriptSegment], start: float, end: float
) -> List[str]:
    matches: List[str] = []
    for segment in segments:
        if _overlaps(segment.start, segment.end, start, end):
            matches.append(segment.segment_id)
    return matches


def _overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return (a_start <= b_end) and (b_start <= a_end)
