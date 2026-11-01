"""SQLite media store built from an AnalysisBundle, with keyword and semantic search over shots.

- Keyword search: an FTS5 index over what is said (transcript) and what is seen (captions).
- Semantic search: one embedding per shot (its transcript + its captions), computed once and
  cached in the database, then compared to the query with cosine similarity.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .schemas import AnalysisBundle

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

SCHEMA = """
CREATE TABLE IF NOT EXISTS media_asset (
    asset_id TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    duration REAL,
    fps REAL,
    has_audio INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS shot (
    shot_id TEXT PRIMARY KEY,
    asset_id TEXT NOT NULL REFERENCES media_asset(asset_id),
    start REAL NOT NULL,
    end REAL NOT NULL,
    keyframe_path TEXT,
    transcript TEXT NOT NULL DEFAULT '',
    captions TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS transcript_segment (
    segment_id TEXT PRIMARY KEY,
    asset_id TEXT NOT NULL REFERENCES media_asset(asset_id),
    start REAL NOT NULL,
    end REAL NOT NULL,
    text TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS shot_segment (
    shot_id TEXT NOT NULL REFERENCES shot(shot_id),
    segment_id TEXT NOT NULL REFERENCES transcript_segment(segment_id),
    PRIMARY KEY (shot_id, segment_id)
);
CREATE TABLE IF NOT EXISTS shot_embedding (
    shot_id TEXT NOT NULL REFERENCES shot(shot_id),
    model TEXT NOT NULL,
    vector BLOB NOT NULL,
    PRIMARY KEY (shot_id, model)
);
CREATE VIRTUAL TABLE IF NOT EXISTS search_fts USING fts5(
    shot_id UNINDEXED,
    kind UNINDEXED,   -- speech or visual
    text
);
"""


class Encoder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEncoder:
    """Default encoder, loaded on first use (sentence-transformers is part of the analysis extra)."""

    def __init__(self, model: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self.model = model
        self._model = None

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError("Semantic search needs the analysis extra: uv sync --extra analysis") from exc
            self._model = SentenceTransformer(self.model)
        return self._model.encode(list(texts), convert_to_numpy=True)


@dataclass
class SearchResult:
    shot_id: str
    asset_id: str
    start: float
    end: float
    score: float
    matched: str          # transcript or caption text that matched
    captions: list[str]

    def as_dict(self) -> dict:
        return asdict(self)


def load_analysis_bundle(path: Path | str) -> AnalysisBundle:
    return AnalysisBundle.model_validate_json(Path(path).expanduser().read_text(encoding="utf-8"))


class MediaStore:
    def __init__(self, db_path: Path | str, encoder: Encoder | None = None,
                 encoder_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.encoder = encoder or SentenceTransformerEncoder(encoder_name)
        # Cached vectors are keyed by model name, so switching models never mixes embeddings.
        self.encoder_name = encoder_name if encoder is None else type(encoder).__name__
        with self._connect() as con:
            con.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        try:
            yield con
            con.commit()
        finally:
            con.close()

    # ------------------------------------------------------------------ ingestion

    def ingest_bundle(self, bundle: AnalysisBundle) -> None:
        """Replace the store content with the bundle (embeddings are recomputed on next search)."""
        segments = {s.segment_id: s for audio in bundle.audio_index.values() for s in audio.segments}
        scenes = {s.shot_id: s for video in bundle.video_index.values() for s in video.scenes}

        with self._connect() as con:
            for table in ("search_fts", "shot_embedding", "shot_segment", "transcript_segment",
                          "shot", "media_asset"):
                con.execute(f"DELETE FROM {table}")

            con.executemany("INSERT INTO media_asset VALUES (?, ?, ?, ?, ?)", [
                (a.asset_id, str(a.path), a.duration, a.fps, int(a.has_audio)) for a in bundle.media_assets])
            con.executemany("INSERT INTO transcript_segment VALUES (?, ?, ?, ?, ?)", [
                (s.segment_id, audio.asset_id, s.start, s.end, s.text)
                for audio in bundle.audio_index.values() for s in audio.segments])

            for atom in bundle.timeline_atoms:
                scene = scenes.get(atom.shot_id)
                transcript = " ".join(segments[i].text for i in atom.transcript_segment_ids if i in segments)
                con.execute("INSERT INTO shot VALUES (?, ?, ?, ?, ?, ?, ?)", (
                    atom.shot_id, atom.asset_id, atom.start, atom.end,
                    str(scene.keyframe_path) if scene and scene.keyframe_path else None,
                    transcript, json.dumps(atom.captions)))
                con.executemany("INSERT INTO shot_segment VALUES (?, ?)",
                                [(atom.shot_id, i) for i in atom.transcript_segment_ids if i in segments])
                con.executemany("INSERT INTO search_fts VALUES (?, 'speech', ?)",
                                [(atom.shot_id, segments[i].text)
                                 for i in atom.transcript_segment_ids if i in segments])
                con.executemany("INSERT INTO search_fts VALUES (?, 'visual', ?)",
                                [(atom.shot_id, c) for c in atom.captions])

    # ------------------------------------------------------------------ search

    def keyword_search(self, query: str, limit: int = 5) -> list[SearchResult]:
        """Exact search over transcripts and captions. Accepts FTS5 syntax (AND, OR, NEAR, "phrase")."""
        sql = """
            SELECT f.shot_id, f.text AS matched, -bm25(search_fts) AS score
            FROM search_fts f WHERE search_fts MATCH ? ORDER BY score DESC LIMIT ?
        """
        with self._connect() as con:
            try:
                rows = con.execute(sql, (query, limit * 3)).fetchall()
            except sqlite3.OperationalError:
                # Plain text with punctuation is not valid FTS5 syntax: search the words instead.
                words = re.findall(r"\w+", query)
                if not words:
                    return []
                rows = con.execute(sql, (" ".join(f'"{w}"' for w in words), limit * 3)).fetchall()
            return self._results(con, [(r["shot_id"], r["score"], r["matched"]) for r in rows], limit)

    def semantic_search(self, query: str, limit: int = 5) -> list[SearchResult]:
        """Shots whose transcript and captions are closest in meaning to the query."""
        shot_ids, vectors = self._shot_vectors()
        if not shot_ids:
            return []
        q = np.asarray(self.encoder.encode([query]), dtype=np.float32)[0]
        scores = vectors @ q / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(q) + 1e-9)
        order = np.argsort(scores)[::-1][:limit]
        with self._connect() as con:
            return self._results(con, [(shot_ids[i], float(scores[i]), None) for i in order], limit)

    def _shot_vectors(self) -> tuple[list[str], np.ndarray]:
        """Embeddings of every shot, computing and caching the missing ones."""
        with self._connect() as con:
            shots = con.execute("SELECT shot_id, transcript, captions FROM shot ORDER BY shot_id").fetchall()
            cached = {r["shot_id"]: np.frombuffer(r["vector"], dtype=np.float32) for r in con.execute(
                "SELECT shot_id, vector FROM shot_embedding WHERE model = ?", (self.encoder_name,))}

            missing = [s for s in shots if s["shot_id"] not in cached]
            if missing:
                new = np.asarray(self.encoder.encode([shot_document(s) for s in missing]), dtype=np.float32)
                for shot, vector in zip(missing, new, strict=True):
                    cached[shot["shot_id"]] = vector
                    con.execute("INSERT OR REPLACE INTO shot_embedding VALUES (?, ?, ?)",
                                (shot["shot_id"], self.encoder_name, vector.tobytes()))

        ids = [s["shot_id"] for s in shots]
        return ids, np.stack([cached[i] for i in ids]) if ids else np.empty((0, 0), dtype=np.float32)

    def _results(self, con, hits: list[tuple[str, float, str | None]], limit: int) -> list[SearchResult]:
        results, seen = [], set()
        for shot_id, score, matched in hits:
            if shot_id in seen:  # keep the best hit per shot
                continue
            seen.add(shot_id)
            row = con.execute("SELECT * FROM shot WHERE shot_id = ?", (shot_id,)).fetchone()
            captions = json.loads(row["captions"])
            results.append(SearchResult(shot_id=shot_id, asset_id=row["asset_id"], start=row["start"],
                                        end=row["end"], score=round(score, 4),
                                        matched=matched or row["transcript"] or "; ".join(captions),
                                        captions=captions))
            if len(results) == limit:
                break
        return results

    # ------------------------------------------------------------------ lookups

    def list_shots(self, asset_id: str | None = None, limit: int | None = None) -> list[dict]:
        sql = "SELECT shot_id, asset_id, start, end, transcript, captions FROM shot"
        params: list = []
        if asset_id:
            sql += " WHERE asset_id = ?"
            params.append(asset_id)
        sql += " ORDER BY asset_id, start"
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        with self._connect() as con:
            return [{**dict(r), "captions": json.loads(r["captions"])} for r in con.execute(sql, params)]

    def shots_for_segment(self, segment_id: str) -> list[dict]:
        with self._connect() as con:
            rows = con.execute("""
                SELECT s.shot_id, s.asset_id, s.start, s.end, t.text
                FROM shot_segment ss
                JOIN shot s ON s.shot_id = ss.shot_id
                JOIN transcript_segment t ON t.segment_id = ss.segment_id
                WHERE ss.segment_id = ? ORDER BY s.start
            """, (segment_id,)).fetchall()
        return [dict(r) for r in rows]

    def get_shot(self, shot_id: str) -> dict | None:
        """A shot with the metadata of its source file, as needed to place it on a timeline."""
        with self._connect() as con:
            row = con.execute("""
                SELECT s.*, m.path AS asset_path, m.duration AS asset_duration, m.fps, m.has_audio
                FROM shot s JOIN media_asset m ON m.asset_id = s.asset_id WHERE s.shot_id = ?
            """, (shot_id,)).fetchone()
        if row is None:
            return None
        return {**dict(row), "captions": json.loads(row["captions"]), "has_audio": bool(row["has_audio"])}


def shot_document(shot) -> str:
    """Text embedded for a shot: what is said, then what is seen."""
    captions = json.loads(shot["captions"])
    parts = [shot["transcript"]] if shot["transcript"] else []
    if captions:
        parts.append("Visual: " + "; ".join(captions))
    return " ".join(parts) or "(no speech, no caption)"
