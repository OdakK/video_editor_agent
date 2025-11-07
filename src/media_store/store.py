from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from media_analysis import (
    AnalysisBundle,
    AudioAnalysisResult,
    SceneSegment,
    TranscriptSegment,
    VideoAnalysisResult,
)


# Simple dataclass to carry search results.
@dataclass
class SearchResult:
    shot_id: str
    asset_id: str
    start: float
    end: float
    score: float
    text_snippet: Optional[str] = None
    captions: Optional[str] = None


def load_analysis_bundle(path: Path | str) -> AnalysisBundle:
    """Load a serialized AnalysisBundle from disk."""
    bundle_path = Path(path).expanduser().resolve()
    with bundle_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return AnalysisBundle.model_validate(data)


class MediaStore:
    """SQLite + embeddings store built from an AnalysisBundle."""

    def __init__(
        self,
        db_path: Path | str,
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
    ) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.embedding_model_name = embedding_model
        self._encoder: SentenceTransformer | None = None

    def initialize(self, drop_existing: bool = False) -> None:
        if drop_existing and self.db_path.exists():
            self.db_path.unlink()
        conn = self._get_connection()
        try:
            self._create_schema(conn)
        finally:
            conn.close()

    def ingest_bundle(self, bundle: AnalysisBundle, overwrite: bool = False) -> None:
        conn = self._get_connection()
        try:
            if overwrite:
                self._clear_tables(conn)
            self._insert_media_assets(conn, bundle)
            self._insert_transcripts(conn, bundle.audio_index.values())
            self._insert_scenes(conn, bundle.video_index.values())
            self._insert_timeline_atoms(conn, bundle)
            conn.commit()
        finally:
            conn.close()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _create_schema(self, conn: sqlite3.Connection) -> None:
        cursor = conn.cursor()
        cursor.executescript(
            """
            PRAGMA journal_mode=WAL;

            CREATE TABLE IF NOT EXISTS media_asset (
                asset_id TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                duration REAL,
                fps REAL
            );

            CREATE TABLE IF NOT EXISTS transcript_segment (
                segment_id TEXT PRIMARY KEY,
                asset_id TEXT NOT NULL,
                start REAL NOT NULL,
                end REAL NOT NULL,
                speaker TEXT,
                text TEXT NOT NULL,
                FOREIGN KEY (asset_id) REFERENCES media_asset(asset_id)
            );

            CREATE TABLE IF NOT EXISTS scene_segment (
                shot_id TEXT PRIMARY KEY,
                asset_id TEXT NOT NULL,
                start REAL NOT NULL,
                end REAL NOT NULL,
                keyframe_path TEXT,
                captions TEXT,
                tags TEXT,
                FOREIGN KEY (asset_id) REFERENCES media_asset(asset_id)
            );

            CREATE TABLE IF NOT EXISTS timeline_atom (
                atom_id INTEGER PRIMARY KEY AUTOINCREMENT,
                asset_id TEXT NOT NULL,
                shot_id TEXT NOT NULL,
                start REAL NOT NULL,
                end REAL NOT NULL,
                transcript_segment_ids TEXT,
                captions TEXT,
                tags TEXT,
                FOREIGN KEY (asset_id) REFERENCES media_asset(asset_id),
                FOREIGN KEY (shot_id) REFERENCES scene_segment(shot_id)
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS transcript_fts USING fts5(
                segment_id UNINDEXED,
                asset_id,
                text,
                content=''
            );
            """
        )
        conn.commit()

    def _clear_tables(self, conn: sqlite3.Connection) -> None:
        cursor = conn.cursor()
        # Contentless FTS tables cannot be cleared with plain DELETE, so issue the special
        # delete-all command before wiping the relational tables.
        cursor.execute(
            "INSERT INTO transcript_fts(transcript_fts) VALUES ('delete-all');"
        )
        cursor.executescript(
            """
            DELETE FROM timeline_atom;
            DELETE FROM scene_segment;
            DELETE FROM transcript_segment;
            DELETE FROM media_asset;
            """
        )
        conn.commit()

    def _insert_media_assets(
        self, conn: sqlite3.Connection, bundle: AnalysisBundle
    ) -> None:
        cursor = conn.cursor()
        cursor.executemany(
            """
            INSERT OR REPLACE INTO media_asset (asset_id, path, duration, fps)
            VALUES (?, ?, ?, ?)
            """,
            [
                (
                    asset.asset_id,
                    str(asset.path),
                    asset.duration,
                    asset.fps,
                )
                for asset in bundle.media_assets
            ],
        )

    def _insert_transcripts(
        self, conn: sqlite3.Connection, audio_results: Iterable[AudioAnalysisResult]
    ) -> None:
        cursor = conn.cursor()
        rows = []
        fts_rows = []
        for audio in audio_results:
            for segment in audio.segments:
                rows.append(
                    (
                        segment.segment_id,
                        audio.asset_id,
                        segment.start,
                        segment.end,
                        segment.speaker,
                        segment.text,
                    )
                )
                fts_rows.append((segment.segment_id, audio.asset_id, segment.text))
        cursor.executemany(
            """
            INSERT OR REPLACE INTO transcript_segment
            (segment_id, asset_id, start, end, speaker, text)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        cursor.executemany(
            "INSERT INTO transcript_fts (segment_id, asset_id, text) VALUES (?, ?, ?)",
            fts_rows,
        )

    def _insert_scenes(
        self, conn: sqlite3.Connection, video_results: Iterable[VideoAnalysisResult]
    ) -> None:
        cursor = conn.cursor()
        rows = []
        for video in video_results:
            for scene in video.scenes:
                rows.append(
                    (
                        scene.shot_id,
                        video.asset_id,
                        scene.start,
                        scene.end,
                        str(scene.keyframe_path) if scene.keyframe_path else None,
                        json.dumps([cap.text for cap in scene.captions]),
                        json.dumps(scene.tags),
                    )
                )
        cursor.executemany(
            """
            INSERT OR REPLACE INTO scene_segment
            (shot_id, asset_id, start, end, keyframe_path, captions, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def _insert_timeline_atoms(self, conn: sqlite3.Connection, bundle: AnalysisBundle):
        cursor = conn.cursor()
        cursor.executemany(
            """
            INSERT INTO timeline_atom
            (asset_id, shot_id, start, end, transcript_segment_ids, captions, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    atom.asset_id,
                    atom.shot_id,
                    atom.start,
                    atom.end,
                    json.dumps(atom.transcript_segment_ids),
                    json.dumps(atom.captions),
                    json.dumps(atom.tags),
                )
                for atom in bundle.timeline_atoms
            ],
        )

    def semantic_search(self, query: str, limit: int = 5) -> list[SearchResult]:
        encoder = self._ensure_encoder()

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT segment_id, asset_id, text FROM transcript_segment
                """
            )
            rows = cursor.fetchall()
        finally:
            conn.close()

        if not rows:
            return []

        texts = [row[2] for row in rows]
        embeddings = encoder.encode(texts, convert_to_numpy=True)
        query_embedding = encoder.encode([query], convert_to_numpy=True)[0]

        scores = np.dot(embeddings, query_embedding) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        top_indices = np.argsort(scores)[::-1][:limit]
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            segment_id, asset_id, text = rows[idx]
            context = self.lookup_segment(segment_id)
            if context is None:
                continue
            results.append(
                SearchResult(
                    shot_id=context["shot_id"],
                    asset_id=asset_id,
                    start=context["start"],
                    end=context["end"],
                    score=score,
                    text_snippet=text,
                    captions=context["captions"],
                )
            )
        return results

    def lookup_segment(self, segment_id: str) -> Optional[dict]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT ta.shot_id, ta.start, ta.end, ss.captions
                FROM timeline_atom ta
                LEFT JOIN scene_segment ss ON ss.shot_id = ta.shot_id
                WHERE EXISTS (
                    SELECT 1
                    FROM json_tree(ta.transcript_segment_ids)
                    WHERE value = ?
                )
                ORDER BY ta.atom_id ASC
                LIMIT 1
                """,
                (segment_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            shot_id, start, end, captions_json = row
            captions = json.loads(captions_json) if captions_json else []
            return {
                "shot_id": shot_id,
                "start": start,
                "end": end,
                "captions": ", ".join(captions),
            }
        finally:
            conn.close()

    def list_shots(
        self, *, asset_id: Optional[str] = None, limit: Optional[int] = None
    ) -> list[dict]:
        """Return basic metadata for analyzed shots, optionally filtered."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            query = """
                SELECT shot_id, asset_id, start, end, keyframe_path, captions, tags
                FROM scene_segment
            """
            params: list = []
            conditions = []
            if asset_id:
                conditions.append("asset_id = ?")
                params.append(asset_id)
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            query += " ORDER BY asset_id, start"
            if limit is not None:
                query += " LIMIT ?"
                params.append(limit)
            cursor.execute(query, params)
            rows = cursor.fetchall()
        finally:
            conn.close()

        shot_list = []
        for (
            shot_id,
            resolved_asset_id,
            start,
            end,
            keyframe_path,
            captions_json,
            tags_json,
        ) in rows:
            shot_list.append(
                {
                    "shot_id": shot_id,
                    "asset_id": resolved_asset_id,
                    "start": start,
                    "end": end,
                    "keyframe_path": keyframe_path,
                    "captions": json.loads(captions_json) if captions_json else [],
                    "tags": json.loads(tags_json) if tags_json else [],
                }
            )
        return shot_list

    def get_shot_details(self, shot_id: str) -> Optional[dict]:
        """Fetch a single shot enriched with media asset metadata."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    ss.shot_id,
                    ss.asset_id,
                    ss.start,
                    ss.end,
                    ss.keyframe_path,
                    ss.captions,
                    ss.tags,
                    ma.path,
                    ma.duration,
                    ma.fps
                FROM scene_segment ss
                JOIN media_asset ma ON ma.asset_id = ss.asset_id
                WHERE ss.shot_id = ?
                """,
                (shot_id,),
            )
            row = cursor.fetchone()
        finally:
            conn.close()

        if not row:
            return None

        (
            shot_id,
            asset_id,
            start,
            end,
            keyframe_path,
            captions_json,
            tags_json,
            asset_path,
            asset_duration,
            fps,
        ) = row

        return {
            "shot_id": shot_id,
            "asset_id": asset_id,
            "start": start,
            "end": end,
            "keyframe_path": keyframe_path,
            "captions": json.loads(captions_json) if captions_json else [],
            "tags": json.loads(tags_json) if tags_json else [],
            "asset_path": asset_path,
            "asset_duration": asset_duration,
            "fps": fps or 24.0,
        }

    def full_text_search(self, query: str, limit: int = 5) -> list[SearchResult]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT segment_id, asset_id, text, rank FROM transcript_fts
                WHERE transcript_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (query, limit),
            )
            rows = cursor.fetchall()
        finally:
            conn.close()

        results = []
        for segment_id, asset_id, text, rank in rows:
            context = self.lookup_segment(segment_id)
            if context is None:
                continue
            results.append(
                SearchResult(
                    shot_id=context["shot_id"],
                    asset_id=asset_id,
                    start=context["start"],
                    end=context["end"],
                    score=float(-rank),
                    text_snippet=text,
                    captions=context["captions"],
                )
            )
        return results

    def _ensure_encoder(self) -> SentenceTransformer:
        if self._encoder is None:
            self._encoder = SentenceTransformer(self.embedding_model_name)
        return self._encoder
