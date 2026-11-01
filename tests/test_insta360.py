"""360 files: synthetic .insv files in both layouts Insta360 cameras write."""

import shutil
import subprocess
import time

import pytest

from video_agent.insta360 import find_lenses, flatten
from video_agent.library import Library
from video_agent.probe import probe
from video_agent.store import HashingEncoder, MediaStore

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
    subprocess.run(["ffmpeg", "-y", "-v", "error", *cmd, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                    "-f", "mp4", str(path)], check=True)
    return path


@pytest.mark.parametrize("layout", ["two_streams", "side_by_side"])
def test_each_lens_becomes_a_flat_vertical_view(tmp_path, layout):
    insv = make_insv(tmp_path / "VID_test_00_001.insv", layout)
    assert len(find_lenses(insv)) == 2
    views = flatten(insv, tmp_path / "out", "trip")
    assert [v.name for v in views] == ["trip_lens1.mp4", "trip_lens2.mp4"]
    for view in views:
        info = probe(view)
        assert (info.width, info.height, info.has_audio) == (1080, 1920, True)
        assert info.duration == pytest.approx(1.0, abs=0.15)


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
    assets = sorted(a.asset_id for a in library.bundle.media_assets)
    assert assets == ["VID_20260810_120023_00_358_lens1", "VID_20260810_120023_00_358_lens2"]
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
