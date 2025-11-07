from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
from typing import Dict, List, Optional

from langgraph.graph import END, StateGraph
from langchain_core.runnables import RunnableConfig
from typing_extensions import TypedDict

# Ensure sibling packages under src/ are importable when this module is loaded via path.
SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from media_analysis import (
    AnalysisBundle,
    AudioAnalysisResult,
    MediaAsset,
    VideoAnalysisResult,
)
from media_analysis.audio import AudioAnalyzer
from media_analysis.ingest import discover_media_assets
from media_analysis.merge import merge_analysis
from media_analysis.video import VideoAnalyzer

LOGGER = logging.getLogger(__name__)


class AnalysisState(TypedDict, total=False):
    """State propagated through the analysis workflow."""

    input_dir: str
    output_dir: str
    media_assets: List[MediaAsset]
    audio_index: Dict[str, AudioAnalysisResult]
    video_index: Dict[str, VideoAnalysisResult]
    analysis_bundle: AnalysisBundle
    artifacts: Dict[str, str]


def create_analysis_workflow(config: RunnableConfig) -> StateGraph:
    """Factory required by LangGraph CLI (expects a single RunnableConfig argument)."""
    configurable = {}
    if config is not None:
        configurable = config.get("configurable", {}) or {}

    audio_kwargs = configurable.get("audio_analyzer", {})
    video_kwargs = configurable.get("video_analyzer", {})

    audio_analyzer = AudioAnalyzer(**audio_kwargs)
    video_analyzer = VideoAnalyzer(**video_kwargs)

    return _build_graph(audio_analyzer, video_analyzer)


def build_analysis_workflow(
    audio_analyzer: Optional[AudioAnalyzer] = None,
    video_analyzer: Optional[VideoAnalyzer] = None,
) -> StateGraph:
    """Convenience wrapper for local execution/tests."""
    audio_analyzer = audio_analyzer or AudioAnalyzer()
    video_analyzer = video_analyzer or VideoAnalyzer()
    return _build_graph(audio_analyzer, video_analyzer)


def _build_graph(audio_analyzer: AudioAnalyzer, video_analyzer: VideoAnalyzer) -> StateGraph:
    graph = StateGraph(AnalysisState)

    graph.add_node("ingest", _make_ingest_node())
    graph.add_node("audio_analysis", _make_audio_node(audio_analyzer))
    graph.add_node("video_analysis", _make_video_node(video_analyzer))
    graph.add_node("merge", _make_merge_node())

    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "audio_analysis")
    graph.add_edge("audio_analysis", "video_analysis")
    graph.add_edge("video_analysis", "merge")
    graph.add_edge("merge", END)

    return graph


def _make_ingest_node():
    def ingest_node(state: AnalysisState):
        input_dir = state.get("input_dir")
        if not input_dir:
            raise ValueError("input_dir missing from state.")
        assets = discover_media_assets(input_dir)
        LOGGER.info("Discovered %d media assets", len(assets))
        return {"media_assets": assets}

    return ingest_node


def _make_audio_node(audio_analyzer: AudioAnalyzer):
    def audio_node(state: AnalysisState):
        assets = state.get("media_assets") or []
        index: Dict[str, AudioAnalysisResult] = {}
        for asset in assets:
            try:
                index[asset.asset_id] = audio_analyzer.analyze(asset)
            except Exception as exc:  # pragma: no cover - runtime safeguard
                LOGGER.exception("Audio analysis failed for %s", asset.path)
                raise exc
        return {"audio_index": index}

    return audio_node


def _make_video_node(video_analyzer: VideoAnalyzer):
    def video_node(state: AnalysisState):
        assets = state.get("media_assets") or []
        output_dir = Path(state.get("output_dir") or "./analysis-output").resolve()
        cache_dir = output_dir / "keyframes"
        index: Dict[str, VideoAnalysisResult] = {}
        for asset in assets:
            asset_cache = cache_dir / asset.asset_id
            try:
                index[asset.asset_id] = video_analyzer.analyze(
                    asset, cache_dir=asset_cache
                )
            except Exception as exc:  # pragma: no cover - runtime safeguard
                LOGGER.exception("Video analysis failed for %s", asset.path)
                raise exc
        return {"video_index": index, "artifacts": {"keyframes": str(cache_dir)}}

    return video_node


def _make_merge_node():
    def merge_node(state: AnalysisState):
        assets = state.get("media_assets") or []
        audio_index = state.get("audio_index") or {}
        video_index = state.get("video_index") or {}
        bundle = merge_analysis(assets, audio_index, video_index)

        output_dir = Path(state.get("output_dir") or "./analysis-output").resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = output_dir / "analysis_bundle.json"
        bundle_path.write_text(
            json.dumps(bundle.model_dump(mode="json"), indent=2), encoding="utf-8"
        )

        audio_path = output_dir / "audio_index.json"
        audio_path.write_text(
            json.dumps(
                {k: v.model_dump(mode="json") for k, v in bundle.audio_index.items()},
                indent=2,
            ),
            encoding="utf-8",
        )

        video_path = output_dir / "video_index.json"
        video_path.write_text(
            json.dumps(
                {k: v.model_dump(mode="json") for k, v in bundle.video_index.items()},
                indent=2,
            ),
            encoding="utf-8",
        )

        artifacts = state.get("artifacts") or {}
        artifacts.update(
            {
                "bundle": str(bundle_path),
                "audio_index": str(audio_path),
                "video_index": str(video_path),
            }
        )

        return {"analysis_bundle": bundle, "artifacts": artifacts}

    return merge_node
