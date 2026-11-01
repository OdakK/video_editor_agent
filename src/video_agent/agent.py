"""The editing agent: a tool-calling LLM over the media store and the timeline."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from .store import MediaStore, load_analysis_bundle
from .timeline import TimelineManager
from .tools import build_tools

load_dotenv()

SYSTEM_PROMPT = """You are an assistant video editor working on analyzed rushes.

- Find shots with the search tools before editing. Prefer semantic_search for descriptions and
  keyword_search for exact words or names.
- Always give shot ids and timecodes (in seconds) in your answers.
- After editing, check the result with describe_timeline before you report it.
- If what the user asks for is not in the rushes, say so instead of picking a random shot.
- Give file paths as plain text, not as links.
- Answer in the user's language."""


def default_paths() -> dict[str, Path]:
    return {
        "bundle": Path(os.getenv("MEDIA_BUNDLE_PATH", "output/analysis_bundle.json")),
        "db": Path(os.getenv("MEDIA_STORE_PATH", "output/media_store.db")),
        "timeline": Path(os.getenv("TIMELINE_PATH", "output/timeline.otio")),
    }


def load_model() -> BaseChatModel:
    """The model comes from VIDEO_AGENT_MODEL, as provider:model (default: OpenAI GPT-6 Luna)."""
    # Retries with backoff on rate limits (429) and transient errors.
    return init_chat_model(os.getenv("VIDEO_AGENT_MODEL", "openai:gpt-6-luna"), temperature=0, max_retries=8)


def create_editing_agent(bundle_path: Path | None = None, db_path: Path | None = None,
                         timeline_path: Path | None = None, model: BaseChatModel | None = None,
                         store: MediaStore | None = None):
    paths = default_paths()
    if store is None:
        bundle_path = Path(bundle_path or paths["bundle"]).expanduser()
        if not bundle_path.exists():
            raise FileNotFoundError(f"{bundle_path} not found: run `video-agent analyze` first.")
        store = MediaStore(db_path or paths["db"])
        store.ingest_bundle(load_analysis_bundle(bundle_path))
    return build_agent(store, TimelineManager(timeline_path or paths["timeline"]), model)


def build_agent(store: MediaStore, timeline: TimelineManager, model: BaseChatModel | None = None):
    """The agent over an existing store and timeline. Renders go next to the timeline file."""
    return create_agent(model=model or load_model(),
                        tools=build_tools(store, timeline, output_dir=timeline.timeline_path.parent),
                        system_prompt=SYSTEM_PROMPT)


def make_graph():
    """Factory for `langgraph dev` (see langgraph.json)."""
    return create_editing_agent()
