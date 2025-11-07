"""Tooling utilities for the editing agent."""

from media_store import MediaStore

from .media_tools import build_media_store_tools
from .timeline_tools import build_timeline_tools
from timeline import TimelineManager


def build_agent_tools(
    store: MediaStore, timeline_manager: TimelineManager
):
    """Aggregate querying tools and timeline editing helpers."""
    tools = []
    tools.extend(build_media_store_tools(store))
    tools.extend(build_timeline_tools(store, timeline_manager))
    return tools


__all__ = [
    "build_media_store_tools",
    "build_timeline_tools",
    "build_agent_tools",
]
