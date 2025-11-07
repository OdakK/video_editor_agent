from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
from PIL import Image
from scenedetect import SceneManager, StatsManager, VideoManager
from scenedetect.detectors import ContentDetector
from transformers import pipeline

from .schemas import FrameCaption, MediaAsset, SceneSegment, VideoAnalysisResult

LOGGER = logging.getLogger(__name__)


class VideoAnalyzer:
    """Perform scene detection and frame captioning."""

    def __init__(
        self,
        threshold: float = 30.0,
        min_scene_len: int = 15,
        downscale_factor: int = 1,
        enable_captions: bool = True,
        caption_model: str = "Salesforce/blip-image-captioning-base",
        caption_device: Optional[str | int] = None,
    ) -> None:
        self.threshold = threshold
        self.min_scene_len = max(min_scene_len, 2)
        self.downscale_factor = max(downscale_factor, 1)
        self.enable_captions = enable_captions
        self.caption_model = caption_model
        self.caption_device = caption_device
        self._caption_pipeline = None

    def analyze(self, asset: MediaAsset, cache_dir: Path) -> VideoAnalysisResult:
        LOGGER.info("Running scene detection for %s", asset.path)
        scene_boundaries = self._detect_scenes(asset)
        cache_dir.mkdir(parents=True, exist_ok=True)

        scenes: List[SceneSegment] = []
        for idx, (start_s, end_s) in enumerate(scene_boundaries):
            keyframe_path = self._extract_keyframe(asset, start_s, end_s, idx, cache_dir)
            captions = (
                self._generate_captions(keyframe_path, (start_s + end_s) / 2, idx)
                if self.enable_captions and keyframe_path is not None
                else []
            )

            scenes.append(
                SceneSegment(
                    shot_id=f"{asset.asset_id}_shot_{idx:03d}",
                    start=start_s,
                    end=end_s,
                    keyframe_path=keyframe_path,
                    captions=captions,
                    tags=[],
                )
            )

        return VideoAnalysisResult(asset_id=asset.asset_id, scenes=scenes)

    def _detect_scenes(self, asset: MediaAsset) -> List[Tuple[float, float]]:
        video_manager = VideoManager([str(asset.path)])
        stats_manager = StatsManager()
        scene_manager = SceneManager(stats_manager)
        scene_manager.add_detector(
            ContentDetector(threshold=self.threshold, min_scene_len=self.min_scene_len)
        )

        video_manager.set_downscale_factor(self.downscale_factor)
        video_manager.start()

        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list()

        video_manager.release()

        if not scene_list:
            duration = asset.duration or 0.0
            LOGGER.warning("No scene boundaries detected for %s", asset.path)
            return [(0.0, duration)]

        return [
            (start_time.get_seconds(), end_time.get_seconds())
            for start_time, end_time in scene_list
        ]

    def _extract_keyframe(
        self,
        asset: MediaAsset,
        start_s: float,
        end_s: float,
        idx: int,
        cache_dir: Path,
    ) -> Optional[Path]:
        timestamp = (start_s + end_s) / 2 if end_s > start_s else start_s
        capture = cv2.VideoCapture(str(asset.path))
        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
        success, frame = capture.read()
        capture.release()

        if not success or frame is None:
            LOGGER.warning(
                "Failed to grab keyframe for %s at %.2fs", asset.path, timestamp
            )
            return None

        keyframe_path = cache_dir / f"{asset.asset_id}_shot_{idx:03d}.jpg"
        cv2.imwrite(str(keyframe_path), frame)
        return keyframe_path

    def _generate_captions(
        self, keyframe_path: Path, timestamp: float, idx: int
    ) -> List[FrameCaption]:
        pipeline = self._ensure_caption_pipeline()
        with Image.open(keyframe_path) as img:
            results = pipeline(img)

        if isinstance(results, list):
            captions = results
        else:
            captions = [results]

        normalized = []
        for caption_idx, entry in enumerate(captions):
            text = entry.get("generated_text") or entry.get("caption")
            if not text:
                continue
            normalized.append(
                FrameCaption(
                    caption_id=f"cap_{idx:03d}_{caption_idx:02d}",
                    timecode=timestamp,
                    text=text,
                    confidence=entry.get("score"),
                    image_path=keyframe_path,
                )
            )
        return normalized

    def _ensure_caption_pipeline(self):
        if self._caption_pipeline is None:
            LOGGER.info("Loading caption model %s", self.caption_model)
            kwargs = {}
            if self.caption_device is not None:
                kwargs["device"] = self.caption_device

            self._caption_pipeline = pipeline(
                task="image-to-text", model=self.caption_model, **kwargs
            )
        return self._caption_pipeline
