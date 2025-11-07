from __future__ import annotations

from typing import List

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from media_store import MediaStore, SearchResult


class SemanticSearchInput(BaseModel):
    query: str = Field(description="Search phrase describing the desired content.")
    limit: int = Field(default=5, description="Maximum number of shots to return.")


class KeywordSearchInput(BaseModel):
    query: str = Field(
        description="Keyword expression using SQLite FTS syntax (AND, OR, NEAR, etc.)."
    )
    limit: int = Field(default=5, description="Maximum number of hits to return.")


class SegmentLookupInput(BaseModel):
    segment_id: str = Field(description="Identifier of the transcript segment.")


class ListShotsInput(BaseModel):
    asset_id: str | None = Field(
        default=None,
        description="Optional asset identifier to filter shots.",
    )
    limit: int | None = Field(
        default=None,
        description="Maximum number of shots to return (defaults to all).",
    )


def _serialize_results(results: List[SearchResult]) -> List[dict]:
    return [
        {
            "shot_id": r.shot_id,
            "asset_id": r.asset_id,
            "start": r.start,
            "end": r.end,
            "score": r.score,
            "text_snippet": r.text_snippet,
            "captions": r.captions,
        }
        for r in results
    ]


def build_media_store_tools(store: MediaStore) -> List[StructuredTool]:
    """Create LangChain tools that query the media store."""

    def semantic_tool(query: str, limit: int = 5) -> List[dict]:
        results = store.semantic_search(query=query, limit=limit)
        return _serialize_results(results)

    def keyword_tool(query: str, limit: int = 5) -> List[dict]:
        results = store.full_text_search(query=query, limit=limit)
        return _serialize_results(results)

    def segment_tool(segment_id: str) -> dict | None:
        return store.lookup_segment(segment_id)

    def list_shots_tool(asset_id: str | None = None, limit: int | None = None):
        return store.list_shots(asset_id=asset_id, limit=limit)

    return [
        StructuredTool.from_function(
            name="semantic_clip_search",
            func=semantic_tool,
            description=(
                "Search the media store for shots matching a natural-language query. "
                "Returns a list of candidate shots with timecodes and transcript snippets."
            ),
            args_schema=SemanticSearchInput,
        ),
        StructuredTool.from_function(
            name="keyword_clip_search",
            func=keyword_tool,
            description=(
                "Perform an exact keyword search over transcripts using SQLite FTS syntax. "
                "Useful when the user specifies precise words or logical operators."
            ),
            args_schema=KeywordSearchInput,
        ),
        StructuredTool.from_function(
            name="lookup_segment_context",
            func=segment_tool,
            description=(
                "Retrieve context for a transcript segment, including associated shot "
                "and timecodes."
            ),
            args_schema=SegmentLookupInput,
        ),
        StructuredTool.from_function(
            name="list_analyzed_shots",
            func=list_shots_tool,
            description=(
                "Return the catalog of analyzed shots with timecodes and metadata. "
                "Use when the user asks for all available shots or to browse rushes."
            ),
            args_schema=ListShotsInput,
        ),
    ]
