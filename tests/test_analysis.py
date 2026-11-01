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


def test_vision_samples_frames_inside_the_shot():
    from video_agent.analysis.vision import sample_times

    times = sample_times(5.0, 11.0, 6)
    assert len(times) == 5 and all(5.0 < t < 11.0 for t in times)
    assert len(sample_times(0.0, 1.0, 6)) == 2  # always at least two frames


def test_long_shots_are_described_window_by_window():
    """A 10-minute 360 walk has no cut: describe it in 20 s windows so the moments stay precise."""
    from video_agent.analysis.vision import VisionDescriber
    from video_agent.schemas import MediaAsset, Moment, SceneSegment, ShotDescription

    class FakeDescriber(VisionDescriber):
        def __init__(self):
            super().__init__(model=object())
            self.windows = []

        def _describe_window(self, asset, start, end):
            self.windows.append((start, end))
            return ShotDescription(summary=f"part at {start:.0f}s", shot_size="wide", camera="handheld",
                                   tags=["forest", "Forest", "path"],
                                   moments=[Moment(start=start, end=end, description="walk", quality="usable")])

    describer = FakeDescriber()
    asset = MediaAsset(asset_id="walk", path="/walk.mp4", duration=95.0)
    d = describer.describe(asset, SceneSegment(shot_id="walk_shot_000", start=0.0, end=95.0))
    assert sorted(describer.windows) == [(0.0, 20.0), (20.0, 40.0), (40.0, 60.0), (60.0, 80.0), (80.0, 95.0)]
    assert [(m.start, m.end) for m in d.moments][0] == (0.0, 20.0) and len(d.moments) == 5
    assert d.summary.startswith("part at 0s Then: part at 20s") and d.tags == ["forest", "path"]

    short = describer.describe(asset, SceneSegment(shot_id="s", start=0.0, end=25.0))
    assert short.summary == "part at 0s"  # short shots: one call
