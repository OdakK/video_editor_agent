from __future__ import annotations

import logging
from typing import Iterable, List, Optional

try:
    from faster_whisper import WhisperModel
except ImportError as exc:  # pragma: no cover - defensive import guard
    raise ImportError(
        "faster-whisper is required for audio analysis. "
        "Install the optional dependencies defined in pyproject.toml."
    ) from exc

from .schemas import AudioAnalysisResult, MediaAsset, TranscriptSegment, WordTiming

LOGGER = logging.getLogger(__name__)


class AudioAnalyzer:
    """Wrapper around faster-whisper with sane defaults."""

    def __init__(
        self,
        model_size: str = "medium",
        device: Optional[str] = None,
        compute_type: str = "float32",
        beam_size: int = 5,
        vad_filter: bool = True,
        language: Optional[str] = None,
        **model_kwargs,
    ) -> None:
        self.model_size = model_size
        self.device = device or "auto"
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.language = language
        self.model_kwargs = model_kwargs
        self._model: WhisperModel | None = None

    def analyze(self, asset: MediaAsset) -> AudioAnalysisResult:
        if not asset.has_audio:
            LOGGER.warning("Skipping audio analysis for %s (no audio track)", asset.path)
            return AudioAnalysisResult(asset_id=asset.asset_id, segments=[])

        model = self._ensure_model()
        LOGGER.info("Transcribing %s", asset.path)
        segments_iter, info = model.transcribe(
            str(asset.path),
            beam_size=self.beam_size,
            word_timestamps=True,
            vad_filter=self.vad_filter,
            language=self.language,
        )

        segment_models = _build_segments(asset.asset_id, segments_iter)

        return AudioAnalysisResult(
            asset_id=asset.asset_id,
            language=self.language or getattr(info, "language", None),
            segments=segment_models,
        )

    def _ensure_model(self) -> WhisperModel:
        if self._model is None:
            LOGGER.info(
                "Loading Whisper model '%s' (device=%s, compute=%s)",
                self.model_size,
                self.device,
                self.compute_type,
            )
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                **self.model_kwargs,
            )
        return self._model


def _build_segments(asset_id: str, segment_iterable: Iterable) -> List[TranscriptSegment]:
    segment_models: List[TranscriptSegment] = []
    for idx, segment in enumerate(segment_iterable):
        segment_id = f"{asset_id}_seg_{idx:03d}"
        words = [
            WordTiming(
                word=w.word.strip(),
                start=w.start,
                end=w.end,
                probability=getattr(w, "probability", None),
            )
            for w in getattr(segment, "words", [])
            if w.word.strip()
        ]
        segment_models.append(
            TranscriptSegment(
                segment_id=segment_id,
                start=segment.start,
                end=segment.end,
                text=segment.text.strip(),
                words=words,
            )
        )
    return segment_models
