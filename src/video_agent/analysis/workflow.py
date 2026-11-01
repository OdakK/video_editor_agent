"""The analysis pipeline as a LangGraph graph: ingest -> audio_analysis -> video_analysis -> describe -> merge.

`describe` asks a vision LLM to log each shot (see vision.py). It is skipped without an API key.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from ..schemas import AnalysisBundle, AudioAnalysisResult, MediaAsset, VideoAnalysisResult
from .audio import AudioAnalyzer
from .ingest import discover_media_assets
from .merge import merge_analysis
from .video import VideoAnalyzer
from .vision import VisionDescriber

LOGGER = logging.getLogger(__name__)


class AnalysisState(TypedDict, total=False):
    input_dir: str
    output_dir: str
    media_assets: list[MediaAsset]
    audio_index: dict[str, AudioAnalysisResult]
    video_index: dict[str, VideoAnalysisResult]
    analysis_bundle: AnalysisBundle
    artifacts: dict[str, str]


def build_analysis_workflow(audio: AudioAnalyzer | None = None, video: VideoAnalyzer | None = None,
                            vision: VisionDescriber | None | str = "auto"):
    """Return the compiled analysis graph. Pass analyzers to change models or thresholds."""
    audio = audio or AudioAnalyzer()
    video = video or VideoAnalyzer()
    if vision == "auto":
        vision = VisionDescriber() if VisionDescriber.available() else None

    def ingest(state: AnalysisState) -> dict:
        assets = discover_media_assets(state["input_dir"])
        LOGGER.info("Found %d rush(es)", len(assets))
        return {"media_assets": assets}

    def audio_analysis(state: AnalysisState) -> dict:
        return {"audio_index": {a.asset_id: audio.analyze(a) for a in state["media_assets"]}}

    def video_analysis(state: AnalysisState) -> dict:
        keyframes = _output_dir(state) / "keyframes"
        index = {a.asset_id: video.analyze(a, cache_dir=keyframes / a.asset_id) for a in state["media_assets"]}
        return {"video_index": index}

    def describe(state: AnalysisState) -> dict:
        if vision is None:
            LOGGER.info("No LLM API key: skipping the vision descriptions")
            return {}
        assets = {a.asset_id: a for a in state["media_assets"]}
        for asset_id, result in state["video_index"].items():
            for scene in result.scenes:
                LOGGER.info("Describing %s", scene.shot_id)
                try:
                    scene.description = vision.describe(assets[asset_id], scene)
                except Exception as exc:  # the shot stays usable without its description
                    LOGGER.warning("Vision description failed for %s: %s", scene.shot_id, exc)
        return {"video_index": state["video_index"]}

    def merge(state: AnalysisState) -> dict:
        bundle = merge_analysis(state["media_assets"], state.get("audio_index", {}), state["video_index"])
        out = _output_dir(state)
        out.mkdir(parents=True, exist_ok=True)
        bundle_path = out / "analysis_bundle.json"
        bundle_path.write_text(json.dumps(bundle.model_dump(mode="json"), indent=2), encoding="utf-8")
        return {"analysis_bundle": bundle,
                "artifacts": {"bundle": str(bundle_path), "keyframes": str(out / "keyframes")}}

    graph = StateGraph(AnalysisState)
    graph.add_node("ingest", ingest)
    graph.add_node("audio_analysis", audio_analysis)
    graph.add_node("video_analysis", video_analysis)
    graph.add_node("describe", describe)
    graph.add_node("merge", merge)
    graph.add_edge(START, "ingest")
    graph.add_edge("ingest", "audio_analysis")
    graph.add_edge("audio_analysis", "video_analysis")
    graph.add_edge("video_analysis", "describe")
    graph.add_edge("describe", "merge")
    graph.add_edge("merge", END)
    return graph.compile()


def make_graph(config: RunnableConfig):
    """Factory for `langgraph dev`: analyzer settings come from the run's configurable."""
    configurable = (config or {}).get("configurable", {}) or {}
    return build_analysis_workflow(AudioAnalyzer(**configurable.get("audio_analyzer", {})),
                                   VideoAnalyzer(**configurable.get("video_analyzer", {})))


def _output_dir(state: AnalysisState) -> Path:
    return Path(state.get("output_dir") or "output").resolve()
