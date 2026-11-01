"""The agent with a scripted model: no API key, but real tools, store and timeline."""

from itertools import cycle

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage

from video_agent.agent import create_editing_agent


class ScriptedModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def call(name, args, call_id):
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": call_id}])


def test_agent_searches_then_edits_the_timeline(tmp_path, store):
    model = ScriptedModel(messages=cycle([
        call("keyword_search", {"query": "blue sky"}, "1"),
        call("insert_shot", {"shot_id": "beach_shot_001"}, "2"),
        AIMessage("Added beach_shot_001 (5.0s to 10.0s) to the timeline."),
    ]))
    agent = create_editing_agent(model=model, store=store, timeline_path=tmp_path / "edit.otio")
    result = agent.invoke({"messages": [{"role": "user", "content": "Add the blue sky shot"}]})

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert "beach_shot_001" in tool_messages[0].content
    assert "beach_shot_001" in tool_messages[1].content
    assert result["messages"][-1].content.startswith("Added beach_shot_001")


def test_tool_errors_go_back_to_the_model(tmp_path, store):
    model = ScriptedModel(messages=cycle([
        call("insert_shot", {"shot_id": "does_not_exist"}, "1"),
        AIMessage("That shot does not exist."),
    ]))
    agent = create_editing_agent(model=model, store=store, timeline_path=tmp_path / "edit.otio")
    result = agent.invoke({"messages": [{"role": "user", "content": "Insert shot does_not_exist"}]})

    error = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert "Unknown shot_id" in error.content
