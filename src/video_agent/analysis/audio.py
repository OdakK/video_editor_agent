"""Speech recognition with faster-whisper, with word-level timestamps."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from ..schemas import AudioAnalysisResult, MediaAsset, TranscriptSegment, WordTiming

LOGGER = logging.getLogger(__name__)


class AudioAnalyzer:
    """Wrapper around faster-whisper. The model is loaded on first use."""

    def __init__(
        self,
        model_size: str = "medium",
        device: str = "auto",
        compute_type: str = "default",  # lets CTranslate2 pick int8 on CPU, float16 on GPU
        beam_size: int = 5,
        vad_filter: bool = True,
        language: str | None = None,
        **model_kwargs: Any,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.language = language
        self.model_kwargs = model_kwargs
        self._model = None

    def analyze(self, asset: MediaAsset) -> AudioAnalysisResult:
        if not asset.has_audio:
            return AudioAnalysisResult(asset_id=asset.asset_id)

        LOGGER.info("Transcribing %s", asset.path.name)
        segments, info = self._ensure_model().transcribe(
            str(asset.path),
            beam_size=self.beam_size,
            word_timestamps=True,
            vad_filter=self.vad_filter,
            language=self.language,
        )
        return AudioAnalysisResult(
            asset_id=asset.asset_id,
            language=self.language or getattr(info, "language", None),
            segments=build_segments(asset.asset_id, segments),
        )

    def _ensure_model(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise ImportError("Audio analysis needs the analysis extra: uv sync --extra analysis") from exc
            LOGGER.info("Loading Whisper model %s (device=%s)", self.model_size, self.device)
            self._model = WhisperModel(self.model_size, device=self.device,
                                       compute_type=self.compute_type, **self.model_kwargs)
        return self._model


def build_segments(asset_id: str, segments: Iterable[Any]) -> list[TranscriptSegment]:
    """Convert faster-whisper segments into our schema (ids are stable: <asset>_seg_000)."""
    result = []
    for idx, segment in enumerate(segments):
        words = [
            WordTiming(word=w.word.strip(), start=w.start, end=w.end,
                       probability=getattr(w, "probability", None))
            for w in (getattr(segment, "words", None) or [])
            if w.word.strip()
        ]
        result.append(TranscriptSegment(segment_id=f"{asset_id}_seg_{idx:03d}", start=segment.start,
                                        end=segment.end, text=segment.text.strip(), words=words))
    return result
