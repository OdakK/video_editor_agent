from __future__ import annotations

from typing import List

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from media_store import MediaStore
from timeline import TimelineManager


class InitializeTimelineInput(BaseModel):
    name: str | None = Field(
        default=None,
        description="Optional display name for the OTIO timeline.",
    )


class InsertShotInput(BaseModel):
    shot_id: str = Field(description="Shot identifier to insert into the timeline.")
    timeline_time: float | None = Field(
        default=None,
        description=(
            "Timeline position (seconds) where the shot should be inserted. "
            "Appends to the end when omitted."
        ),
    )
    track: str = Field(
        default="Video",
        description="Timeline track where the shot is placed.",
    )


class CutTimelineInput(BaseModel):
    timeline_time: float = Field(
        description="Time position (seconds) where the current shot should be split."
    )
    track: str = Field(
        default="Video",
        description="Target track containing the clip to split.",
    )


class DescribeTimelineInput(BaseModel):
    track: str = Field(
        default="Video",
        description="Timeline track to summarize.",
    )


class RenderTimelineInput(BaseModel):
    output_path: str = Field(
        description="Chemin du fichier vidéo de sortie (ex: preview.mp4)."
    )
    track: str = Field(
        default="Video",
        description="Timeline track to render.",
    )
    include_audio: bool = Field(
        default=True,
        description="Inclure l'audio si toutes les sources en possèdent.",
    )


def build_timeline_tools(
    store: MediaStore, manager: TimelineManager
) -> List[StructuredTool]:
    """Expose OTIO editing helpers as LangChain tools."""

    def initialize_timeline(name: str | None = None):
        clips = manager.initialize(name=name)
        return {
            "message": f"Timeline initialisée ({name or manager.default_name}).",
            "timeline_path": str(manager.timeline_path),
            "clips": clips,
        }

    def insert_shot(shot_id: str, timeline_time: float | None = None, track: str = "Video"):
        shot = store.get_shot_details(shot_id)
        if shot is None:
            raise ValueError(f"Impossible de trouver le plan '{shot_id}'.")
        clips = manager.insert_shot(
            shot,
            timeline_time=timeline_time,
            track_name=track,
        )
        return {
            "message": f"Plan {shot_id} inséré sur la piste {track}.",
            "timeline_path": str(manager.timeline_path),
            "clips": clips,
        }

    def cut_timeline(timeline_time: float, track: str = "Video"):
        clips = manager.split_at_time(timeline_time, track_name=track)
        return {
            "message": f"Cut appliqué à {timeline_time:.2f}s sur {track}.",
            "timeline_path": str(manager.timeline_path),
            "clips": clips,
        }

    def describe_timeline(track: str = "Video"):
        clips = manager.describe(track_name=track)
        return {
            "timeline_path": str(manager.timeline_path),
            "clips": clips,
        }

    def render_timeline(output_path: str, track: str = "Video", include_audio: bool = True):
        return manager.render_preview(
            output_path=output_path, track_name=track, include_audio=include_audio
        )

    return [
        StructuredTool.from_function(
            name="initialize_timeline",
            func=initialize_timeline,
            description="Crée une timeline OTIO vide (écrase l'ancienne si besoin).",
            args_schema=InitializeTimelineInput,
        ),
        StructuredTool.from_function(
            name="insert_shot_into_timeline",
            func=insert_shot,
            description=(
                "Place un plan (shot_id) sur la timeline. "
                "Fournir un temps en secondes pour l'insérer à un endroit précis."
            ),
            args_schema=InsertShotInput,
        ),
        StructuredTool.from_function(
            name="cut_timeline_clip",
            func=cut_timeline,
            description=(
                "Réalise un cut (split) du plan actuellement sur la timeline "
                "à la position temporelle donnée."
            ),
            args_schema=CutTimelineInput,
        ),
        StructuredTool.from_function(
            name="describe_timeline",
            func=describe_timeline,
            description="Retourne l'état actuel de la timeline (clips, timecodes).",
            args_schema=DescribeTimelineInput,
        ),
        StructuredTool.from_function(
            name="render_timeline_preview",
            func=render_timeline,
            description=(
                "Lance un rendu ffmpeg de la timeline actuelle au format MP4. "
                "Active ou non l'audio selon les rushes disponibles."
            ),
            args_schema=RenderTimelineInput,
        ),
    ]
