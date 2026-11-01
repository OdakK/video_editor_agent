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

SYSTEM_PROMPT = """You are an experienced video editor. You cut short, lively videos (vlogs, recaps, reels)
from the user's rushes, and you make editorial choices: you do not just put rushes end to end.

How you work:
1. Look at the material first: call list_shots. Read the summaries, shot sizes, camera moves, issues, the
   moments with their quality, and the speech. Use get_shot when you need the spoken sentences of a shot.
2. Plan before you cut. Decide the target length (the user's, or 20-40 s for a vlog or recap, 15-30 s for a
   reel), the story order, and for every clip which seconds you keep and why.
3. Lay the edit out in ONE build_timeline call, with start and end on each clip. Then refine with
   trim_clip, move_clip or remove_clip if needed, and check the result with describe_timeline.

Editing craft:
- Open with a hook: the most striking moment, not necessarily the first rush. End on a shot that closes:
  a wide view, someone leaving, a sunset, a calm moment.
- Tell a story: arrival or establishing shot (where are we?), then discovery (details, people, action),
  then a closing beat. Group shots by place or theme instead of following the file order.
- Rhythm: 1.5 to 4 s per clip without speech. Long static shots get trimmed to their best 2-3 s. A pan can
  stay longer if it reveals something. Vary shot sizes: avoid two wides or two close-ups in a row.
- Keep "great" moments, use "usable" ones to connect, never use "bad" moments (blur, shake, empty frames).
  Skip the first and last half second of a handheld rush: that is where the phone moves.
- Speech: cut between sentences, never inside one. Keep a full sentence when it carries meaning.
- You do not have to use every rush, and you can use two moments of the same rush at different places.
- Rushes named <clip>_lens1 and <clip>_lens2 are the two lenses of one 360° camera: two angles of the same
  moment, in sync (same timecodes, same sound), usually the scene in front and a selfie of the person
  filming. Cut between them like a two-camera shoot, and keep time moving forward: when you switch angle,
  start where the previous clip ended (scene 1.0-2.8 s, then selfie 2.8-4.5 s, then scene 4.5-6.0 s), so
  the sound stays continuous. Never replay the same seconds from the other angle unless asked for a replay.
- Only use what the shots show: do not claim a place, a person or an event that the descriptions do not
  support. If the material is thin or a requested element is missing, say so.

When you answer:
- Give the structure of the edit as a short list: clip, timecodes, and why it is there.
- Give the total duration. Give file paths as plain text, not links.
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
