import shutil
import subprocess
from pathlib import Path

import pytest

from video_agent.probe import probe
from video_agent.store import MediaStore
from video_agent.timeline import TimelineManager, build_ffmpeg_command

from .conftest import make_bundle


@pytest.fixture
def timeline(tmp_path):
    return TimelineManager(tmp_path / "edit.otio")


def shots(store, *shot_ids):
    return [store.get_shot(i) for i in shot_ids]


def test_insert_append_and_insert_before(timeline, store):
    first, second, third = shots(store, "beach_shot_000", "beach_shot_001", "interview_shot_000")
    timeline.insert_shot(first)
    timeline.insert_shot(second)
    clips = timeline.insert_shot(third, timeline_time=2.0)  # 2s is inside the first clip
    assert [c["shot_id"] for c in clips] == ["interview_shot_000", "beach_shot_000", "beach_shot_001"]
    assert [c["timeline_start"] for c in clips] == [0.0, 8.0, 13.0]


def test_split_keeps_source_timecodes(timeline, store):
    timeline.insert_shot(store.get_shot("beach_shot_001"))  # source 5s -> 10s
    clips = timeline.split_at(1.5)
    assert [(c["source_start"], c["source_end"]) for c in clips] == [(5.0, 6.5), (6.5, 10.0)]
    assert clips[0]["clip_id"] != clips[1]["clip_id"]
    with pytest.raises(ValueError):
        timeline.split_at(99)


def test_remove_and_move(timeline, store):
    for shot in shots(store, "beach_shot_000", "beach_shot_001", "interview_shot_000"):
        clips = timeline.insert_shot(shot)
    clips = timeline.move_clip(clips[2]["clip_id"], 0)
    assert [c["shot_id"] for c in clips] == ["interview_shot_000", "beach_shot_000", "beach_shot_001"]
    clips = timeline.remove_clip(clips[1]["clip_id"])
    assert [c["shot_id"] for c in clips] == ["interview_shot_000", "beach_shot_001"]
    with pytest.raises(ValueError, match="No clip"):
        timeline.remove_clip("missing")


def test_parallel_edits_are_not_lost(timeline, store):
    """Agents call tools in parallel: concurrent inserts on the same file must all land."""
    from concurrent.futures import ThreadPoolExecutor

    shot = store.get_shot("beach_shot_000")
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: timeline.insert_shot(shot), range(16)))
    assert len(timeline.describe()) == 16


def test_timeline_is_saved_as_otio(timeline, store):
    timeline.insert_shot(store.get_shot("beach_shot_000"))
    reloaded = TimelineManager(timeline.timeline_path)
    assert [c["shot_id"] for c in reloaded.describe()] == ["beach_shot_000"]


def test_ffmpeg_command(store):
    clips = [{"asset_path": "/a.mp4", "source_start": 1.0, "source_end": 3.5},
             {"asset_path": "/b.mp4", "source_start": 0.0, "source_end": 2.0}]
    command = build_ffmpeg_command(clips, Path("/out.mp4"), include_audio=True, size=(1281, 720))
    graph = command[command.index("-filter_complex") + 1]
    assert command.count("-i") == 2
    assert "trim=start=1.000:end=3.500" in graph
    assert "scale=1280:720" in graph  # odd width rounded down for libx264
    assert graph.endswith("concat=n=2:v=1:a=1[vout][aout]")
    silent = build_ffmpeg_command(clips, Path("/out.mp4"), include_audio=False, size=(640, 360))
    assert "[aout]" not in " ".join(silent)


@pytest.mark.ffmpeg
@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_render_real_files(tmp_path, encoder):
    """End to end with real media: generate two test clips, edit them and render a preview."""
    def make_clip(name, size, seconds, audio=True):
        path = tmp_path / name
        cmd = ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc=size={size}:rate=25:d={seconds}"]
        if audio:
            cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"]
        subprocess.run(cmd + ["-pix_fmt", "yuv420p", "-shortest", str(path)], check=True)
        return path

    beach = make_clip("beach.mp4", "640x360", 10)
    interview = make_clip("interview.mp4", "320x240", 8)
    store = MediaStore(tmp_path / "store.db", encoder=encoder)
    store.ingest_bundle(make_bundle(beach, interview))
    timeline = TimelineManager(tmp_path / "edit.otio")
    timeline.insert_shot(store.get_shot("beach_shot_001"))      # 5 s
    timeline.insert_shot(store.get_shot("interview_shot_000"))  # 8 s
    timeline.split_at(2.0)
    timeline.remove_clip(timeline.describe()[0]["clip_id"])     # drop the first 2 s

    result = timeline.render_preview(tmp_path / "preview.mp4")
    info = probe(result["output_path"])
    assert result["audio"] is True
    assert (info.width, info.height) == (640, 360)
    assert info.duration == pytest.approx(11.0, abs=0.2)
