import json
from types import SimpleNamespace

from video_agent.analysis.audio import build_segments
from video_agent.analysis.merge import overlapping_segments
from video_agent.probe import parse_ffprobe
from video_agent.schemas import TranscriptSegment


def seg(i, start, end):
    return TranscriptSegment(segment_id=f"s{i}", start=start, end=end, text="")


def test_overlap_ignores_segments_that_only_touch_the_shot():
    segments = [seg(0, 0, 5), seg(1, 4, 6), seg(2, 5, 7), seg(3, 10, 12)]
    # Shot [5, 10): s0 ends exactly at 5, so it belongs to the previous shot only.
    assert overlapping_segments(segments, 5, 10) == ["s1", "s2"]


def test_merge_attaches_transcript_and_captions_to_each_shot(bundle):
    atoms = {a.shot_id: a for a in bundle.timeline_atoms}
    assert atoms["beach_shot_000"].transcript_segment_ids == ["beach_seg_000"]
    assert atoms["beach_shot_001"].transcript_segment_ids == ["beach_seg_001"]
    assert atoms["beach_shot_001"].captions == ["a woman looking at the sky"]


def test_build_segments_from_whisper_output():
    word = SimpleNamespace(word=" hello ", start=0.1, end=0.4, probability=0.9)
    raw = [SimpleNamespace(start=0.0, end=1.0, text=" hello world ", words=[word]),
           SimpleNamespace(start=1.0, end=2.0, text="no words", words=None)]
    segments = build_segments("rush", raw)
    assert [s.segment_id for s in segments] == ["rush_seg_000", "rush_seg_001"]
    assert segments[0].text == "hello world"
    assert segments[0].words[0].word == "hello"
    assert segments[1].words == []


def test_parse_ffprobe():
    output = json.dumps({
        "format": {"duration": "12.48"},
        "streams": [
            {"codec_type": "video", "width": 1920, "height": 1080, "avg_frame_rate": "30000/1001"},
            {"codec_type": "audio"},
        ],
    })
    info = parse_ffprobe(output)
    assert info.duration == 12.48
    assert round(info.fps, 3) == 29.97
    assert (info.width, info.height, info.has_audio) == (1920, 1080, True)
    assert parse_ffprobe(json.dumps({"streams": [{"codec_type": "video"}]})).has_audio is False
