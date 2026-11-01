"""The rush library behind the UI: import new files, analyze them in the background, re-index.

Two analysis modes:
- full: the real pipeline (Whisper transcription, shot detection, BLIP captions), when the
  `analysis` extra is installed;
- basic: without it, each file becomes one shot with a keyframe and no transcript, so the
  UI still works for manual editing and keyword search on file names.
"""

from __future__ import annotations

import importlib.util
import itertools
import logging
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO

from .analysis.ingest import VIDEO_EXTENSIONS
from .analysis.merge import merge_analysis
from .probe import probe
from .schemas import (
    AnalysisBundle,
    AudioAnalysisResult,
    FrameCaption,
    MediaAsset,
    SceneSegment,
    VideoAnalysisResult,
)
from .store import MediaStore, load_analysis_bundle

LOGGER = logging.getLogger(__name__)


def has_analysis_extra() -> bool:
    return all(importlib.util.find_spec(m) for m in ("faster_whisper", "scenedetect", "transformers"))


@dataclass
class ImportJob:
    id: int
    file_name: str
    asset_id: str
    status: str = "queued"   # queued, analyzing, done, error
    step: str = "Waiting"
    error: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


class Library:
    def __init__(self, bundle_path: Path, store: MediaStore, mode: str | None = None,
                 whisper_model: str = "small") -> None:
        self.bundle_path = bundle_path.resolve()
        self.workdir = self.bundle_path.parent
        self.rushes_dir = self.workdir / "rushes"
        self.keyframes_dir = self.workdir / "keyframes"
        self.store = store
        self.mode = mode or ("full" if has_analysis_extra() else "basic")
        self.bundle = (load_analysis_bundle(self.bundle_path) if self.bundle_path.exists()
                       else AnalysisBundle(media_assets=[], audio_index={}, video_index={}, timeline_atoms=[]))
        self.jobs: dict[int, ImportJob] = {}
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        # One file at a time: the models are heavy, and Whisper is already multi-threaded.
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="import")
        self.whisper_model = whisper_model
        self._audio = self._video = None

    # ------------------------------------------------------------------ import

    def import_file(self, file_name: str, data: BinaryIO) -> ImportJob:
        """Save an uploaded file and queue its analysis."""
        name = Path(file_name).name
        if Path(name).suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"{name}: unsupported format (use {', '.join(sorted(VIDEO_EXTENSIONS))})")
        with self._lock:
            asset_id = self._unique_asset_id(Path(name).stem)
            job = ImportJob(id=next(self._ids), file_name=name, asset_id=asset_id)
            self.jobs[job.id] = job
        self.rushes_dir.mkdir(parents=True, exist_ok=True)
        path = self.rushes_dir / f"{asset_id}{Path(name).suffix.lower()}"
        with path.open("wb") as out:
            shutil.copyfileobj(data, out, length=1024 * 1024)
        self._worker.submit(self._analyze, job, path)
        return job

    def remove_asset(self, asset_id: str) -> int:
        """Remove a rush and its shots from the library. Returns the number of shots removed.

        Only files the library owns are deleted from disk: copies imported through the UI and
        generated keyframes. Original rushes analyzed elsewhere are never touched.
        """
        with self._lock:
            asset = next((a for a in self.bundle.media_assets if a.asset_id == asset_id), None)
            if asset is None:
                raise ValueError(f"Unknown rush {asset_id}")
            video = self.bundle.video_index.get(asset_id)
            removed = len(video.scenes) if video else 0
            bundle = merge_analysis([a for a in self.bundle.media_assets if a.asset_id != asset_id],
                                    {k: v for k, v in self.bundle.audio_index.items() if k != asset_id},
                                    {k: v for k, v in self.bundle.video_index.items() if k != asset_id})
            self.bundle_path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
            self.store.ingest_bundle(bundle)
            self.bundle = bundle

        if _inside(asset.path, self.rushes_dir):
            asset.path.unlink(missing_ok=True)
        keyframes = self.keyframes_dir / asset_id
        if _inside(keyframes, self.keyframes_dir) and keyframes.is_dir():
            shutil.rmtree(keyframes, ignore_errors=True)
        return removed

    def list_jobs(self) -> list[dict]:
        return [j.as_dict() for j in sorted(self.jobs.values(), key=lambda j: j.id)]

    def _unique_asset_id(self, stem: str) -> str:
        base = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem) or "rush"
        taken = {a.asset_id for a in self.bundle.media_assets} | {j.asset_id for j in self.jobs.values()}
        candidate, n = base, 2
        while candidate in taken:
            candidate, n = f"{base}_{n}", n + 1
        return candidate

    def _analyze(self, job: ImportJob, path: Path) -> None:
        job.status = "analyzing"
        try:
            job.step = "Reading the file"
            info = probe(path)
            if not info.duration:
                raise ValueError("no video stream found")
            asset = MediaAsset(asset_id=job.asset_id, path=path, duration=info.duration, fps=info.fps,
                               has_audio=info.has_audio)
            if self.mode == "full":
                audio, video = self._full_analysis(job, asset)
            else:
                audio, video = self._basic_analysis(job, asset)

            job.step = "Indexing"
            with self._lock:
                bundle = merge_analysis([*self.bundle.media_assets, asset],
                                        {**self.bundle.audio_index, asset.asset_id: audio},
                                        {**self.bundle.video_index, asset.asset_id: video})
                self.bundle_path.parent.mkdir(parents=True, exist_ok=True)
                self.bundle_path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
                self.store.ingest_bundle(bundle)
                self.bundle = bundle
            shots = len(video.scenes)
            job.status, job.step = "done", f"{shots} shot{'s' if shots > 1 else ''}"
        except Exception as exc:  # report any failure in the UI instead of losing the job
            LOGGER.exception("Import failed for %s", path)
            job.status, job.step, job.error = "error", "Failed", str(exc)

    def _full_analysis(self, job: ImportJob, asset: MediaAsset):
        from .analysis.audio import AudioAnalyzer
        from .analysis.video import VideoAnalyzer

        self._audio = self._audio or AudioAnalyzer(model_size=self.whisper_model)
        self._video = self._video or VideoAnalyzer()
        job.step = "Transcribing speech" if asset.has_audio else "No audio track"
        audio = self._audio.analyze(asset)
        job.step = "Detecting shots and captioning"
        video = self._video.analyze(asset, cache_dir=self.keyframes_dir / asset.asset_id)
        return audio, video

    def _basic_analysis(self, job: ImportJob, asset: MediaAsset):
        job.step = "Extracting a keyframe"
        keyframe = self.keyframes_dir / asset.asset_id / f"{asset.asset_id}_shot_000.jpg"
        keyframe.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(asset.duration / 2), "-i", str(asset.path),
                        "-frames:v", "1", "-vf", "scale=480:-2", str(keyframe)], check=True, capture_output=True)
        shot_id = f"{asset.asset_id}_shot_000"
        scene = SceneSegment(shot_id=shot_id, start=0.0, end=asset.duration, keyframe_path=keyframe, captions=[
            FrameCaption(caption_id=f"{shot_id}_cap_00", timecode=asset.duration / 2,
                         text=f"full rush {asset.path.name} (not analyzed: install the analysis extra)")])
        return (AudioAnalysisResult(asset_id=asset.asset_id),
                VideoAnalysisResult(asset_id=asset.asset_id, scenes=[scene]))


def _inside(path: Path, folder: Path) -> bool:
    try:
        return Path(path).resolve().is_relative_to(folder.resolve()) and Path(path).resolve() != folder.resolve()
    except OSError:
        return False
