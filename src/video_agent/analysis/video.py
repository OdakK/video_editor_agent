"""Shot detection with PySceneDetect, one keyframe per shot, and BLIP captions."""

from __future__ import annotations

import logging
from pathlib import Path

from ..schemas import FrameCaption, MediaAsset, SceneSegment, VideoAnalysisResult

LOGGER = logging.getLogger(__name__)


class VideoAnalyzer:
    """Detect shots and caption their keyframes. Models are loaded on first use."""

    def __init__(
        self,
        threshold: float = 30.0,
        min_scene_len: int = 15,
        downscale_factor: int = 1,
        enable_captions: bool = True,
        caption_model: str = "Salesforce/blip-image-captioning-base",
        caption_device: str | int | None = None,
    ) -> None:
        self.threshold = threshold
        self.min_scene_len = max(min_scene_len, 2)
        self.downscale_factor = max(downscale_factor, 1)
        self.enable_captions = enable_captions
        self.caption_model = caption_model
        self.caption_device = caption_device
        self._captioner = None

    def analyze(self, asset: MediaAsset, cache_dir: Path) -> VideoAnalysisResult:
        LOGGER.info("Detecting shots in %s", asset.path.name)
        cache_dir.mkdir(parents=True, exist_ok=True)

        scenes = []
        for idx, (start, end) in enumerate(self._detect_scenes(asset)):
            shot_id = f"{asset.asset_id}_shot_{idx:03d}"
            keyframe = self._extract_keyframe(asset, (start + end) / 2, cache_dir / f"{shot_id}.jpg")
            captions = []
            if self.enable_captions and keyframe is not None:
                captions = self._caption(keyframe, (start + end) / 2, shot_id)
            scenes.append(SceneSegment(shot_id=shot_id, start=start, end=end,
                                       keyframe_path=keyframe, captions=captions))
        return VideoAnalysisResult(asset_id=asset.asset_id, scenes=scenes)

    def _detect_scenes(self, asset: MediaAsset) -> list[tuple[float, float]]:
        try:
            from scenedetect import SceneManager, open_video
            from scenedetect.detectors import ContentDetector
        except ImportError as exc:
            raise ImportError("Video analysis needs the analysis extra: uv sync --extra analysis") from exc

        video = open_video(str(asset.path))
        manager = SceneManager()
        manager.add_detector(ContentDetector(threshold=self.threshold, min_scene_len=self.min_scene_len))
        if self.downscale_factor > 1:  # otherwise PySceneDetect picks a factor from the resolution
            manager.auto_downscale = False
            manager.downscale = self.downscale_factor
        manager.detect_scenes(video)
        scene_list = manager.get_scene_list()

        if not scene_list:
            # A single continuous shot: no cut was detected.
            duration = asset.duration or video.duration.get_seconds()
            return [(0.0, duration)]
        return [(start.get_seconds(), end.get_seconds()) for start, end in scene_list]

    def _extract_keyframe(self, asset: MediaAsset, timestamp: float, path: Path) -> Path | None:
        import cv2

        capture = cv2.VideoCapture(str(asset.path))
        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
        success, frame = capture.read()
        capture.release()
        if not success or frame is None:
            LOGGER.warning("Could not read a keyframe from %s at %.2fs", asset.path.name, timestamp)
            return None
        cv2.imwrite(str(path), frame)
        return path

    def _caption(self, keyframe: Path, timestamp: float, shot_id: str) -> list[FrameCaption]:
        from PIL import Image

        with Image.open(keyframe) as image:
            outputs = self._ensure_captioner()(image)
        if not isinstance(outputs, list):
            outputs = [outputs]

        captions = []
        for idx, entry in enumerate(outputs):
            text = entry.get("generated_text") or entry.get("caption")
            if text:
                captions.append(FrameCaption(caption_id=f"{shot_id}_cap_{idx:02d}", timecode=timestamp,
                                             text=text.strip(), confidence=entry.get("score"),
                                             image_path=keyframe))
        return captions

    def _ensure_captioner(self):
        if self._captioner is None:
            from transformers import pipeline

            LOGGER.info("Loading caption model %s", self.caption_model)
            kwargs = {} if self.caption_device is None else {"device": self.caption_device}
            self._captioner = pipeline(task="image-to-text", model=self.caption_model, **kwargs)
        return self._captioner
