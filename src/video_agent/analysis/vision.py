"""Describe shots with a vision LLM: what is seen, how it is filmed, and which moments are worth using.

BLIP gives one generic caption per keyframe ("a man with glasses"). An editor needs more: where we
are, the shot size and camera motion, what is unusable (blur, shake), and which seconds are the
good ones. A multimodal LLM looks at several frames of the shot, with their timestamps, and returns
that as structured data.
"""

from __future__ import annotations

import base64
import logging
import math
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ..schemas import MediaAsset, SceneSegment, ShotDescription

LOGGER = logging.getLogger(__name__)

PROMPT = """You are an assistant video editor logging rushes. You see frames sampled from ONE shot,
each labelled with its timestamp in seconds from the start of the file. Describe the shot for an
editor who has not seen it. Be concrete and factual: describe what is visible, and do not invent
places or names you cannot see. If text, signs or landmarks identify a place, say so."""


class _Moment(BaseModel):
    start: float = Field(description="seconds from the start of the file")
    end: float
    description: str = Field(description="what happens in this moment, in a few words")
    quality: Literal["great", "usable", "bad"] = Field(
        description="great: striking, sharp, well framed; usable: fine; bad: blurry, shaky, empty or accidental")


class _Description(BaseModel):
    summary: str = Field(description="one sentence: what the shot shows")
    location: str = Field(description="kind of place, plus its name if visible (sign, landmark); else 'unknown'")
    subjects: list[str] = Field(description="main subjects: people, objects, buildings…")
    shot_size: Literal["extreme wide", "wide", "medium", "close-up", "detail"]
    camera: Literal["static", "pan", "tilt", "handheld", "tracking", "drone", "mixed"]
    mood: str = Field(description="one or two words: calm, lively, festive, intimate…")
    issues: list[str] = Field(description="technical problems: blur, shake, overexposure, obstruction… or empty")
    moments: list[_Moment] = Field(description="2 to 5 consecutive moments covering the shot")
    tags: list[str] = Field(description="5 to 10 search keywords")


class VisionDescriber:
    """Calls a vision-capable chat model (VIDEO_AGENT_VISION_MODEL, else VIDEO_AGENT_MODEL)."""

    def __init__(self, model=None, frames_per_shot: int = 6, frame_width: int = 384) -> None:
        self._model = model
        self.frames_per_shot = frames_per_shot
        self.frame_width = frame_width

    @staticmethod
    def available() -> bool:
        return bool(os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))

    def _llm(self):
        if self._model is None:
            from langchain.chat_models import init_chat_model

            name = os.getenv("VIDEO_AGENT_VISION_MODEL") or os.getenv("VIDEO_AGENT_MODEL", "openai:gpt-6-luna")
            self._model = init_chat_model(name, temperature=0, max_retries=6)
        return self._model.with_structured_output(_Description)

    def describe(self, asset: MediaAsset, scene: SceneSegment, window: float = 20.0) -> ShotDescription:
        """Describe a shot. Long shots (a 10-minute walk filmed in one take) are described window by
        window, in parallel, so the moments stay precise all along the shot."""
        if scene.end - scene.start <= window * 1.5:
            return self._describe_window(asset, scene.start, scene.end)
        count = math.ceil((scene.end - scene.start) / window)
        edges = [scene.start + i * window for i in range(count)] + [scene.end]
        if len(edges) > 2 and edges[-1] - edges[-2] < window / 2:  # merge a short tail into the previous window
            edges.pop(-2)
        with ThreadPoolExecutor(max_workers=4) as pool:
            parts = list(pool.map(lambda w: self._describe_window(asset, *w), zip(edges, edges[1:], strict=False)))
        first = parts[0]
        return ShotDescription(
            summary=" Then: ".join(p.summary for p in parts[:3]) + (" …" if len(parts) > 3 else ""),
            location=first.location, subjects=_unique(x for p in parts for x in p.subjects),
            shot_size=first.shot_size, camera=first.camera, mood=first.mood,
            issues=_unique(x for p in parts for x in p.issues),
            moments=[m for p in parts for m in p.moments], tags=_unique(x for p in parts for x in p.tags)[:15])

    def _describe_window(self, asset: MediaAsset, start: float, end: float) -> ShotDescription:
        scene = SceneSegment(shot_id="window", start=start, end=end)
        times = sample_times(scene.start, scene.end, self.frames_per_shot)
        content: list[dict] = [{"type": "text", "text": f"Shot from {scene.start:.1f}s to {scene.end:.1f}s."}]
        with tempfile.TemporaryDirectory() as tmp:
            for i, t in enumerate(times):
                frame = Path(tmp) / f"{i}.jpg"
                subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.2f}", "-i", str(asset.path),
                                "-frames:v", "1", "-vf", f"scale={self.frame_width}:-2", "-q:v", "4", str(frame)],
                               check=True, capture_output=True)
                if not frame.exists():
                    continue
                data = base64.b64encode(frame.read_bytes()).decode()
                content.append({"type": "text", "text": f"t = {t:.1f}s"})
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}})
        result: _Description = self._llm().invoke([SystemMessage(PROMPT), HumanMessage(content=content)])
        moments = [m.model_dump() for m in result.moments]
        for m in moments:  # keep the model's timecodes inside the shot
            m["start"] = round(min(max(m["start"], scene.start), scene.end), 2)
            m["end"] = round(min(max(m["end"], m["start"]), scene.end), 2)
        return ShotDescription(**{**result.model_dump(), "moments": [m for m in moments if m["end"] > m["start"]]})


def _unique(items) -> list[str]:
    seen, out = set(), []
    for item in items:
        if item.lower() not in seen:
            seen.add(item.lower())
            out.append(item)
    return out


def sample_times(start: float, end: float, count: int) -> list[float]:
    """Evenly spaced timestamps inside [start, end), about one per 1.5 s, at least 2."""
    duration = max(end - start, 0.0)
    n = max(2, min(count, int(duration / 1.5) + 1))
    step = duration / n
    return [round(start + step * (i + 0.5), 2) for i in range(n)]
