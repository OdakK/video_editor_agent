"""Local web UI: shot library, timeline, preview player and a chat with the editing agent.

The chat streams the agent's steps (tool calls and results) as Server-Sent Events, so you see
the agent search, insert and render while it works. Manual edits from the UI (add, move,
remove a clip) go through the same TimelineManager as the agent's tools.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from collections.abc import Iterator
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from pydantic import BaseModel

from ..agent import build_agent
from ..store import MediaStore
from ..timeline import TimelineManager

STATIC = Path(__file__).parent / "static"


class Session:
    """One local editing session: store, timeline, agent and chat history."""

    def __init__(self, store: MediaStore, timeline: TimelineManager, model_name: str) -> None:
        self.store = store
        self.timeline = timeline
        self.model_name = model_name
        self.output_dir = timeline.timeline_path.parent
        self.history: list[AnyMessage] = []
        self.preview: Path | None = None
        self.lock = threading.Lock()  # one agent run at a time
        self._agent = None

    @property
    def agent(self):
        if self._agent is None:  # built on first message, so the UI opens even without an API key
            self._agent = build_agent(self.store, self.timeline)
        return self._agent


class ChatRequest(BaseModel):
    message: str


class InsertRequest(BaseModel):
    shot_id: str


class MoveRequest(BaseModel):
    clip_id: str
    new_index: int


class RemoveRequest(BaseModel):
    clip_id: str


def create_app(session: Session) -> FastAPI:
    app = FastAPI(title="Video Editor Agent")

    def state() -> dict:
        shots = [{**s, "keyframe_url": f"/media/keyframe/{s['shot_id']}" if s.get("keyframe_path") else None}
                 for s in session.store.list_shots()]
        for s in shots:
            s.pop("keyframe_path", None)
        return {"model": session.model_name, "shots": shots, "timeline": session.timeline.describe(),
                **_preview_urls(session), "conversation": _conversation(session.history)}

    @app.get("/api/state")
    def get_state() -> dict:
        return state()

    @app.post("/api/timeline/insert")
    def insert(req: InsertRequest) -> dict:
        shot = session.store.get_shot(req.shot_id)
        if shot is None:
            raise HTTPException(404, f"Unknown shot {req.shot_id}")
        session.timeline.insert_shot(shot)
        return state()

    @app.post("/api/timeline/move")
    def move(req: MoveRequest) -> dict:
        _edit(session.timeline.move_clip, req.clip_id, req.new_index)
        return state()

    @app.post("/api/timeline/remove")
    def remove(req: RemoveRequest) -> dict:
        _edit(session.timeline.remove_clip, req.clip_id)
        return state()

    @app.post("/api/timeline/render")
    def render() -> dict:
        result = _edit(session.timeline.render_preview, session.output_dir / "preview.mp4")
        _set_preview(session, Path(result["output_path"]))
        return state()

    @app.post("/api/reset")
    def reset() -> dict:
        session.timeline.initialize()
        session.history.clear()
        session.preview = None
        return state()

    @app.post("/api/chat")
    def chat(req: ChatRequest) -> StreamingResponse:
        if not req.message.strip():
            raise HTTPException(400, "Empty message")
        return StreamingResponse(_run_agent(session, req.message.strip()), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/media/keyframe/{shot_id}")
    def keyframe(shot_id: str) -> FileResponse:
        shot = session.store.get_shot(shot_id)
        if not shot or not shot.get("keyframe_path") or not Path(shot["keyframe_path"]).exists():
            raise HTTPException(404)
        return FileResponse(shot["keyframe_path"])

    @app.get("/media/preview")
    def preview() -> FileResponse:
        if not session.preview or not session.preview.exists():
            raise HTTPException(404)
        return FileResponse(session.preview, media_type="video/mp4")

    @app.get("/media/poster")
    def poster() -> FileResponse:
        path = session.preview.with_suffix(".jpg") if session.preview else None
        if not path or not path.exists():
            raise HTTPException(404)
        return FileResponse(path, media_type="image/jpeg")

    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return app


def _edit(func, *args):
    try:
        return func(*args)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(400, str(exc)) from exc


def _event(kind: str, **data) -> str:
    return f"data: {json.dumps({'type': kind, **data}, ensure_ascii=False, default=str)}\n\n"


def _run_agent(session: Session, message: str) -> Iterator[str]:
    if not session.lock.acquire(blocking=False):
        yield _event("error", message="The agent is already working on a request.")
        return
    try:
        session.history.append(HumanMessage(message))
        started = time.perf_counter()
        for update in session.agent.stream({"messages": session.history}, stream_mode="updates"):
            for node in update.values():
                for msg in (node or {}).get("messages", []):
                    session.history.append(msg)
                    yield from _describe(session, msg)
        yield _event("done", seconds=round(time.perf_counter() - started, 1), state=_state_after_run(session))
    except Exception as exc:  # surface model or credential errors in the chat instead of a broken stream
        yield _event("error", message=f"{type(exc).__name__}: {exc}")
    finally:
        session.lock.release()


def _describe(session: Session, msg: AnyMessage) -> Iterator[str]:
    if isinstance(msg, AIMessage):
        for call in msg.tool_calls or []:
            yield _event("tool_call", id=call["id"], name=call["name"], args=call["args"])
        text = msg.content if isinstance(msg.content, str) else "".join(
            p.get("text", "") for p in msg.content if isinstance(p, dict))
        if text.strip():
            yield _event("message", content=text)
    elif isinstance(msg, ToolMessage):
        ok = getattr(msg, "status", "success") != "error"
        if ok and msg.name == "render_preview":
            try:
                _set_preview(session, Path(json.loads(msg.content)["output_path"]))
            except (ValueError, KeyError, TypeError):
                pass
        yield _event("tool_result", id=msg.tool_call_id, name=msg.name, ok=ok, summary=_summarize(msg.content))


def _conversation(history: list[AnyMessage]) -> list[dict]:
    """User messages and the agent's text answers, to restore the chat after a page reload."""
    turns = []
    for msg in history:
        if isinstance(msg, HumanMessage):
            turns.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content.strip():
            turns.append({"role": "agent", "content": msg.content})
    return turns


def _summarize(content) -> str:
    text = content if isinstance(content, str) else json.dumps(content, default=str)
    return text if len(text) <= 300 else text[:300] + "…"


def _state_after_run(session: Session) -> dict:
    return {"timeline": session.timeline.describe(), **_preview_urls(session)}


def _set_preview(session: Session, path: Path) -> None:
    """Remember the latest render and extract a poster frame, shown before the video loads."""
    session.preview = path
    try:
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", "0.5", "-i", str(path), "-frames:v", "1",
                        "-vf", "scale=960:-2", str(path.with_suffix(".jpg"))], check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass  # the poster is a nicety: the video still plays without it


def _preview_urls(session: Session) -> dict:
    if not session.preview or not session.preview.exists():
        return {"preview_url": None, "poster_url": None}
    version = int(session.preview.stat().st_mtime)
    has_poster = session.preview.with_suffix(".jpg").exists()
    return {"preview_url": f"/media/preview?v={version}",
            "poster_url": f"/media/poster?v={version}" if has_poster else None}


def model_name() -> str:
    return os.getenv("VIDEO_AGENT_MODEL", "openai:gpt-6-luna").split(":", 1)[-1]
