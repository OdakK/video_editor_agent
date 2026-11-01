"""Importing rushes from the UI: the library analyzes them in the background and re-indexes the store."""

import shutil
import subprocess
import time

import pytest
from fastapi.testclient import TestClient

from video_agent.library import Library
from video_agent.store import HashingEncoder, MediaStore
from video_agent.timeline import TimelineManager
from video_agent.ui.server import Session, create_app

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


def make_video(path, seconds=3):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc=size=320x240:rate=25:d={seconds}",
                    "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


def wait(library, timeout=30):
    deadline = time.time() + timeout
    while any(j["status"] in ("queued", "analyzing") for j in library.list_jobs()):
        assert time.time() < deadline, "import did not finish"
        time.sleep(0.1)


@pytest.fixture
def library(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    store = MediaStore(work / "store.db", encoder=HashingEncoder())
    return Library(work / "analysis_bundle.json", store, mode="basic", vision=None)


def test_import_adds_the_rush_to_the_bundle_and_the_store(tmp_path, library):
    with make_video(tmp_path / "clip.mp4").open("rb") as f:
        job = library.import_file("My clip.mp4", f)
    wait(library)

    assert library.jobs[job.id].status == "done"
    assert job.asset_id == "My_clip"
    shots = library.store.list_shots()
    assert [s["shot_id"] for s in shots] == ["My_clip_shot_000"]
    assert shots[0]["end"] == pytest.approx(3.0, abs=0.1)
    assert library.bundle_path.exists()  # persisted, so a restart keeps the import

    reloaded = Library(library.bundle_path, library.store, mode="basic", vision=None)
    assert [a.asset_id for a in reloaded.bundle.media_assets] == ["My_clip"]


def test_same_name_gets_a_unique_id(tmp_path, library):
    video = make_video(tmp_path / "clip.mp4")
    for _ in range(2):
        with video.open("rb") as f:
            library.import_file("clip.mp4", f)
    wait(library)
    assert sorted(a.asset_id for a in library.bundle.media_assets) == ["clip", "clip_2"]


def test_rejects_unsupported_and_broken_files(tmp_path, library):
    with pytest.raises(ValueError, match="unsupported format"):
        library.import_file("notes.txt", (tmp_path / "x").open("wb+"))
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")
    with broken.open("rb") as f:
        job = library.import_file("broken.mp4", f)
    wait(library)
    assert library.jobs[job.id].status == "error"
    assert library.store.list_shots() == []


def test_upload_endpoint(tmp_path, library):
    session = Session(library.store, TimelineManager(tmp_path / "work" / "timeline.otio"), "scripted", library)
    client = TestClient(create_app(session))
    video = make_video(tmp_path / "clip.mp4")
    response = client.post("/api/upload", files=[
        ("files", ("clip.mp4", video.read_bytes(), "video/mp4")),
        ("files", ("notes.txt", b"hello", "text/plain")),
    ]).json()
    assert [j["file_name"] for j in response["jobs"]] == ["clip.mp4"]
    assert "unsupported format" in response["errors"][0]
    wait(library)
    assert client.get("/api/jobs").json()["jobs"][0]["status"] == "done"
    assert [s["shot_id"] for s in client.get("/api/state").json()["shots"]] == ["clip_shot_000"]


def test_delete_rush_removes_shots_clips_and_imported_copy(tmp_path, library):
    session = Session(library.store, TimelineManager(tmp_path / "work" / "timeline.otio"), "scripted", library)
    client = TestClient(create_app(session))
    with make_video(tmp_path / "a.mp4").open("rb") as f:
        library.import_file("a.mp4", f)
    with make_video(tmp_path / "b.mp4").open("rb") as f:
        library.import_file("b.mp4", f)
    wait(library)
    client.post("/api/timeline/insert", json={"shot_id": "a_shot_000"})
    client.post("/api/timeline/insert", json={"shot_id": "b_shot_000"})
    imported = library.rushes_dir / "a.mp4"
    assert imported.exists()

    response = client.delete("/api/rushes/a").json()
    assert response["removed"] == {"shots": 1, "clips": 1}
    assert [s["shot_id"] for s in response["shots"]] == ["b_shot_000"]
    assert [c["shot_id"] for c in response["timeline"]] == ["b_shot_000"]
    assert not imported.exists()
    assert not (library.keyframes_dir / "a").exists()
    assert client.delete("/api/rushes/a").status_code == 404


def test_delete_never_touches_original_rushes(tmp_path, library):
    """Rushes analyzed with `video-agent analyze` live outside the library folder: keep them."""
    from video_agent.analysis.merge import merge_analysis
    from video_agent.schemas import MediaAsset, SceneSegment, VideoAnalysisResult

    original = make_video(tmp_path / "original.mp4")
    asset = MediaAsset(asset_id="original", path=original, duration=3.0, fps=25.0)
    video = VideoAnalysisResult(asset_id="original", scenes=[SceneSegment(shot_id="original_shot_000", start=0, end=3)])
    library.bundle = merge_analysis([asset], {}, {"original": video})
    library.store.ingest_bundle(library.bundle)

    assert library.remove_asset("original") == 1
    assert original.exists()
    assert library.store.list_shots() == []
