"""The rush library behind the UI: import new files, analyze them in the background, re-index.

Two analysis modes:
- full: the real pipeline (Whisper transcription, shot detection, BLIP captions), when the
  `analysis` extra is installed;
- basic: without it, each file becomes one shot with a keyframe and no transcript.
In both modes, when an LLM API key is set, a vision model then describes every shot (place,
subjects, shot size, camera, issues, and the good and bad moments): see analysis/vision.py.

When the library opens in full mode, rushes that were imported in basic mode, or that have no
vision description yet, are re-analyzed in the background.
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
from .analysis.vision import VisionDescriber
from .insta360 import EXTENSIONS_360, flatten, is_360
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

SUPPORTED = VIDEO_EXTENSIONS | EXTENSIONS_360


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
                 whisper_model: str = "small", vision: VisionDescriber | None | str = "auto") -> None:
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
        if vision == "auto":  # use the vision model when an API key is configured
            vision = VisionDescriber() if VisionDescriber.available() else None
        self.vision = vision

    def refresh_stale(self) -> list[ImportJob]:
        """Queue re-analysis of rushes that are less analyzed than this library can do."""
        jobs = []
        for asset in self.bundle.media_assets:
            scenes = self.bundle.video_index.get(asset.asset_id)
            basic = asset.analysis == "basic" or any(
                c.text.startswith("full rush ") for s in (scenes.scenes if scenes else []) for c in s.captions)
            undescribed = self.vision and scenes and any(s.description is None for s in scenes.scenes)
            if (basic and self.mode == "full") or undescribed:
                jobs.append(self.reanalyze(asset.asset_id))
        return jobs

    def reanalyze(self, asset_id: str) -> ImportJob:
        asset = next((a for a in self.bundle.media_assets if a.asset_id == asset_id), None)
        if asset is None:
            raise ValueError(f"Unknown rush {asset_id}")
        with self._lock:
            job = ImportJob(id=next(self._ids), file_name=asset.path.name, asset_id=asset_id,
                            step="Waiting (re-analysis)")
            self.jobs[job.id] = job
        self._worker.submit(self._process, job, asset.path, False)
        return job

    # ------------------------------------------------------------------ import

    def import_file(self, file_name: str, data: BinaryIO) -> ImportJob:
        """Save an uploaded file and queue its analysis."""
        name = Path(file_name).name
        if Path(name).suffix.lower() not in SUPPORTED:
            raise ValueError(f"{name}: unsupported format (use {', '.join(sorted(SUPPORTED))})")
        job = self._new_job(name)
        self.rushes_dir.mkdir(parents=True, exist_ok=True)
        path = self.rushes_dir / f"{job.asset_id}{Path(name).suffix.lower()}"
        with path.open("wb") as out:
            shutil.copyfileobj(data, out, length=1024 * 1024)
        self._worker.submit(self._process, job, path, True)
        return job

    def import_path(self, path: Path | str) -> list[ImportJob]:
        """Import a file or every video in a folder, in place: nothing is copied or uploaded.

        Useful for large 360 files. In a folder, .lrv proxies are skipped when .insv originals exist.
        """
        path = Path(path).expanduser().resolve()
        if not path.exists():
            raise ValueError(f"{path} does not exist")
        files = [path] if path.is_file() else sorted(
            f for f in path.rglob("*") if f.is_file() and f.suffix.lower() in SUPPORTED and not f.name.startswith("."))
        if path.is_dir() and any(f.suffix.lower() == ".insv" for f in files):
            files = [f for f in files if f.suffix.lower() != ".lrv"]
        if not files:
            raise ValueError(f"No video found in {path}")
        jobs = []
        for f in files:
            if f.suffix.lower() not in SUPPORTED:
                raise ValueError(f"{f.name}: unsupported format")
            job = self._new_job(f.name)
            self._worker.submit(self._process, job, f, False)
            jobs.append(job)
        return jobs

    def _new_job(self, name: str) -> ImportJob:
        with self._lock:
            job = ImportJob(id=next(self._ids), file_name=name, asset_id=self._unique_asset_id(Path(name).stem))
            self.jobs[job.id] = job
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
        # A 360 file becomes <id>_lens1, <id>_lens2: treat their base id as taken too.
        taken |= {a.rsplit("_lens", 1)[0] for a in taken if "_lens" in a}
        candidate, n = base, 2
        while candidate in taken:
            candidate, n = f"{base}_{n}", n + 1
        return candidate

    def _process(self, job: ImportJob, path: Path, owned: bool) -> None:
        """Analyze one file. A 360 file is first turned into one flat view per lens."""
        job.status = "analyzing"
        try:
            if is_360(path):
                minutes = (probe(path).duration or 0) / 60
                job.step = f"Converting 360° to flat views ({minutes:.0f} min of footage)" if minutes >= 1 \
                    else "Converting 360° to flat views"
                views = flatten(path, self.rushes_dir, job.asset_id)
                if owned:  # the uploaded original is no longer needed: the views are the rushes
                    path.unlink(missing_ok=True)
                shots, audio = 0, None
                for i, view in enumerate(views, 1):
                    # The views share the soundtrack: transcribe it once and reuse it.
                    n, audio = self._analyze(job, view.stem, view, prefix=f"View {i}/{len(views)}: ", audio=audio)
                    shots += n
            else:
                shots, _ = self._analyze(job, job.asset_id, path)
            job.status, job.step = "done", f"{shots} shot{'s' if shots > 1 else ''}"
        except Exception as exc:  # report any failure in the UI instead of losing the job
            LOGGER.exception("Import failed for %s", path)
            job.status, job.step, job.error = "error", "Failed", str(exc)

    def _analyze(self, job: ImportJob, asset_id: str, path: Path, prefix: str = "",
                 audio: AudioAnalysisResult | None = None) -> tuple[int, AudioAnalysisResult]:
        """Analyze one flat video and add it to the library. Returns its number of shots and its
        transcript. Pass the transcript of another view of the same 360 clip to reuse it."""
        step = _Steps(job, prefix)
        step("Reading the file")
        info = probe(path)
        if not info.duration:
            raise ValueError("no video stream found")
        asset = MediaAsset(asset_id=asset_id, path=path, duration=info.duration, fps=info.fps,
                           has_audio=info.has_audio, analysis=self.mode)
        if self.mode == "full":
            audio, video = self._full_analysis(step, asset, audio)
        else:
            audio, video = self._basic_analysis(step, asset)
        if self.vision:
            for i, scene in enumerate(video.scenes, 1):
                step(f"Describing shot {i}/{len(video.scenes)}")
                try:
                    scene.description = self.vision.describe(asset, scene)
                except Exception as exc:  # the rush stays usable without the description
                    LOGGER.warning("Vision description failed for %s: %s", scene.shot_id, exc)

        step("Indexing")
        with self._lock:
            others = [a for a in self.bundle.media_assets if a.asset_id != asset.asset_id]  # re-analysis
            bundle = merge_analysis([*others, asset],
                                    {**self.bundle.audio_index, asset.asset_id: audio},
                                    {**self.bundle.video_index, asset.asset_id: video})
            self.bundle_path.parent.mkdir(parents=True, exist_ok=True)
            self.bundle_path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
            self.store.ingest_bundle(bundle)
            self.bundle = bundle
        return len(video.scenes), audio

    def _full_analysis(self, step: _Steps, asset: MediaAsset, audio: AudioAnalysisResult | None = None):
        from .analysis.audio import AudioAnalyzer
        from .analysis.video import VideoAnalyzer

        self._audio = self._audio or AudioAnalyzer(model_size=self.whisper_model)
        self._video = self._video or VideoAnalyzer()
        if audio is not None:  # same soundtrack as another view: copy the transcript under this asset's ids
            audio = AudioAnalysisResult(asset_id=asset.asset_id, language=audio.language, segments=[
                s.model_copy(update={"segment_id": f"{asset.asset_id}_seg_{i:03d}"})
                for i, s in enumerate(audio.segments)])
        else:
            step("Transcribing speech" if asset.has_audio else "No audio track")
            audio = self._audio.analyze(asset)
        step("Detecting shots and captioning")
        video = self._video.analyze(asset, cache_dir=self.keyframes_dir / asset.asset_id)
        return audio, video

    def _basic_analysis(self, step: _Steps, asset: MediaAsset):
        step("Extracting a keyframe")
        keyframe = self.keyframes_dir / asset.asset_id / f"{asset.asset_id}_shot_000.jpg"
        keyframe.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(asset.duration / 2), "-i", str(asset.path),
                        "-frames:v", "1", "-vf", "scale=480:-2", str(keyframe)], check=True, capture_output=True)
        shot_id = f"{asset.asset_id}_shot_000"
        scene = SceneSegment(shot_id=shot_id, start=0.0, end=asset.duration, keyframe_path=keyframe)
        if not self.vision:  # nothing else will describe it
            scene.captions = [FrameCaption(caption_id=f"{shot_id}_cap_00", timecode=asset.duration / 2,
                                           text=f"full rush {asset.path.name} (not analyzed)")]
        return (AudioAnalysisResult(asset_id=asset.asset_id),
                VideoAnalysisResult(asset_id=asset.asset_id, scenes=[scene]))


class _Steps:
    """Updates the job's visible step, prefixed with the current view for 360 files."""

    def __init__(self, job: ImportJob, prefix: str = "") -> None:
        self.job, self.prefix = job, prefix

    def __call__(self, text: str) -> None:
        self.job.step = f"{self.prefix}{text}"


def _inside(path: Path, folder: Path) -> bool:
    try:
        return Path(path).resolve().is_relative_to(folder.resolve()) and Path(path).resolve() != folder.resolve()
    except OSError:
        return False
