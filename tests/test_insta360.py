"""360 files: synthetic .insv files in both layouts Insta360 cameras write."""

import shutil
import subprocess
import time

import pytest

from video_agent.insta360 import find_lenses, make_proxies
from video_agent.library import Library
from video_agent.probe import probe
from video_agent.store import HashingEncoder, MediaStore
from video_agent.timeline import TimelineManager, build_ffmpeg_command

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


def make_insv(path, layout="two_streams", seconds=1):
    """two_streams: one square stream per lens (X3/X4 full resolution). side_by_side: both circles in a
    2:1 stream (lower resolutions, .lrv proxies)."""
    src = f"testsrc=size=256x256:rate=24:d={seconds}"
    tone = ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"]
    if layout == "two_streams":
        cmd = ["-f", "lavfi", "-i", src, "-f", "lavfi", "-i", src, *tone, "-map", "0:v", "-map", "1:v", "-map", "2:a"]
    else:
        cmd = ["-f", "lavfi", "-i", src.replace("256x256", "512x256"), *tone, "-map", "0:v", "-map", "1:a"]
    # One keyframe per second, like a real camera (Insta360 writes one every 2 s).
    subprocess.run(["ffmpeg", "-y", "-v", "error", *cmd, "-c:v", "libx264", "-g", "24", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-f", "mp4", str(path)], check=True)
    return path


@pytest.mark.parametrize("layout", ["two_streams", "side_by_side"])
def test_each_lens_gets_a_light_proxy_and_keeps_its_full_quality_source(tmp_path, layout):
    insv = make_insv(tmp_path / "VID_test_00_001.insv", layout, seconds=4)
    assert len(find_lenses(insv)) == 2
    views = make_proxies(insv, tmp_path / "out", "trip")
    assert [p.name for p, _ in views] == ["trip_lens1.mp4", "trip_lens2.mp4"]
    for proxy, source in views:
        info = probe(proxy)
        assert (info.width, info.height, info.has_audio) == (540, 960, True)  # proxy: half the render size
        assert info.duration == pytest.approx(4.0, abs=0.2)  # the full soundtrack is kept
        assert source.path == insv and (source.width, source.height) == (1080, 1920)
    sources = [s for _, s in views]
    if layout == "two_streams":
        assert [(s.stream, s.crop) for s in sources] == [(0, None), (1, None)]
    else:
        assert [s.crop for s in sources] == ["crop=iw/2:ih:0:0", "crop=iw/2:ih:iw/2:0"]


def test_import_a_folder_of_360_files_in_place(tmp_path):
    shoot = tmp_path / "Sweden" / "DAY_1"
    shoot.mkdir(parents=True)
    insv = make_insv(shoot / "VID_20260810_120023_00_358.insv")
    make_insv(shoot / "LRV_20260810_120023_01_358.lrv", "side_by_side")  # proxy of the same clip: skipped
    (shoot / "notes.txt").write_text("not a video")

    work = tmp_path / "work"
    work.mkdir()
    library = Library(work / "analysis_bundle.json", MediaStore(work / "store.db", encoder=HashingEncoder()),
                      mode="basic", vision=None)
    jobs = library.import_path(tmp_path / "Sweden")
    assert [j.file_name for j in jobs] == ["VID_20260810_120023_00_358.insv"]

    deadline = time.time() + 60
    while any(j["status"] in ("queued", "analyzing") for j in library.list_jobs()):
        assert time.time() < deadline
        time.sleep(0.2)
    job = library.list_jobs()[0]
    assert job["status"] == "done", job["error"]
    assets = sorted(library.bundle.media_assets, key=lambda a: a.asset_id)
    assert [a.asset_id for a in assets] == ["VID_20260810_120023_00_358_lens1", "VID_20260810_120023_00_358_lens2"]
    assert all(a.source360.path == insv for a in assets)  # the render will read the original
    assert all(a.path.parent == work / "rushes" / "proxies" for a in assets)
    # A 360 view is one continuous take: one shot, no shot detection.
    assert all(len(library.bundle.video_index[a.asset_id].scenes) == 1 for a in assets)
    assert insv.exists()  # imported in place: the original is untouched

    # Importing the same clip again gets new ids instead of overwriting the views.
    again = library.import_path(insv)[0]
    assert again.asset_id == "VID_20260810_120023_00_358_2"


def test_unknown_layout_is_rejected(tmp_path):
    odd = tmp_path / "odd.insv"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:d=1", "-pix_fmt",
                    "yuv420p", "-f", "mp4", str(odd)], check=True)
    with pytest.raises(ValueError, match="unrecognized 360 layout"):
        find_lenses(odd)


def test_replaying_the_same_seconds_from_the_other_lens_is_flagged():
    from video_agent.tools import replay_warnings

    def clip(i, lens, start, end):
        return {"index": i, "shot_id": f"VID_1_lens{lens}_shot_000", "source_start": start, "source_end": end}

    forward = [clip(0, 1, 1.0, 2.8), clip(1, 2, 2.8, 4.5), clip(2, 1, 4.5, 6.0)]
    assert replay_warnings(forward) == []
    replay = [clip(0, 1, 1.0, 2.8), clip(1, 2, 1.0, 2.8)]
    assert "replays 1.0-2.8s" in replay_warnings(replay)[0]
    same_lens = [clip(0, 1, 1.0, 2.8), clip(1, 1, 1.0, 2.8)]  # a deliberate repeat of one angle is not flagged
    assert replay_warnings(same_lens) == []


def test_render_reads_the_360_original_for_the_used_seconds_only(tmp_path):
    insv = make_insv(tmp_path / "VID_1_00_001.insv", seconds=6)
    work = tmp_path / "work"
    work.mkdir()
    store = MediaStore(work / "store.db", encoder=HashingEncoder())
    library = Library(work / "analysis_bundle.json", store, mode="basic", vision=None)
    library.import_path(insv)
    deadline = time.time() + 60
    while any(j["status"] in ("queued", "analyzing") for j in library.list_jobs()):
        assert time.time() < deadline
        time.sleep(0.2)

    timeline = TimelineManager(work / "timeline.otio")
    timeline.insert_shot(store.get_shot("VID_1_00_001_lens1_shot_000"), start=1.0, end=3.0)
    timeline.insert_shot(store.get_shot("VID_1_00_001_lens2_shot_000"), start=3.0, end=4.5)
    clips = timeline.describe()
    command = build_ffmpeg_command(clips, tmp_path / "out.mp4", include_audio=True)
    assert command.count(str(insv)) == 2 and "-ss" in command  # seeks into the original, not the proxy
    assert not any("proxies" in arg for arg in command)

    result = timeline.render_preview(tmp_path / "out.mp4")
    info = probe(result["output_path"])
    assert (info.width, info.height) == (1080, 1920)  # full quality, not the 540x960 proxy
    assert info.duration == pytest.approx(3.5, abs=0.2)
