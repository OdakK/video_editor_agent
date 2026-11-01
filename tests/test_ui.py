"""The web UI's API, with a scripted model: no API key, real store and timeline."""

import json
from itertools import cycle

from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from video_agent.agent import build_agent
from video_agent.timeline import TimelineManager
from video_agent.ui.server import Session, create_app


class ScriptedModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def client_for(tmp_path, store, *script):
    session = Session(store, TimelineManager(tmp_path / "timeline.otio"), "scripted")
    if script:
        session._agent = build_agent(store, session.timeline, ScriptedModel(messages=cycle(script)))
    return TestClient(create_app(session))


def events(response):
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


def test_state_lists_shots_and_serves_the_page(tmp_path, store):
    client = client_for(tmp_path, store)
    state = client.get("/api/state").json()
    assert [s["shot_id"] for s in state["shots"]][:2] == ["beach_shot_000", "beach_shot_001"]
    assert state["timeline"] == [] and state["preview_url"] is None
    assert "Video Editor Agent" in client.get("/").text


def test_manual_edits(tmp_path, store):
    client = client_for(tmp_path, store)
    client.post("/api/timeline/insert", json={"shot_id": "beach_shot_000"})
    clips = client.post("/api/timeline/insert", json={"shot_id": "interview_shot_000"}).json()["timeline"]
    clips = client.post("/api/timeline/move", json={"clip_id": clips[1]["clip_id"], "new_index": 0}).json()["timeline"]
    assert [c["shot_id"] for c in clips] == ["interview_shot_000", "beach_shot_000"]
    clips = client.post("/api/timeline/remove", json={"clip_id": clips[0]["clip_id"]}).json()["timeline"]
    assert [c["shot_id"] for c in clips] == ["beach_shot_000"]
    assert client.post("/api/timeline/insert", json={"shot_id": "nope"}).status_code == 404
    assert client.post("/api/timeline/remove", json={"clip_id": "nope"}).status_code == 400


def test_chat_streams_tool_steps_and_the_answer(tmp_path, store):
    client = client_for(
        tmp_path, store,
        AIMessage("", tool_calls=[{"name": "insert_shot", "args": {"shot_id": "beach_shot_001"}, "id": "c1"}]),
        AIMessage("Added the sunset shot."),
    )
    stream = events(client.post("/api/chat", json={"message": "Add the sunset"}))
    kinds = [e["type"] for e in stream]
    assert kinds == ["tool_call", "tool_result", "message", "done"]
    assert stream[1]["ok"] and stream[1]["name"] == "insert_shot"
    assert [c["shot_id"] for c in stream[-1]["state"]["timeline"]] == ["beach_shot_001"]
    # The conversation is kept for the next request and restored on reload.
    assert client.get("/api/state").json()["conversation"] == [
        {"role": "user", "content": "Add the sunset"}, {"role": "agent", "content": "Added the sunset shot."}]


def test_chat_reports_tool_errors(tmp_path, store):
    client = client_for(
        tmp_path, store,
        AIMessage("", tool_calls=[{"name": "insert_shot", "args": {"shot_id": "nope"}, "id": "c1"}]),
        AIMessage("That shot does not exist."),
    )
    result = next(e for e in events(client.post("/api/chat", json={"message": "x"})) if e["type"] == "tool_result")
    assert result["ok"] is False and "Unknown shot_id" in result["summary"]
