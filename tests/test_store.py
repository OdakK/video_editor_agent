from video_agent.store import MediaStore


def ids(results):
    return [r.shot_id for r in results]


def test_keyword_search_in_transcripts(store):
    assert ids(store.keyword_search("Lisbon")) == ["interview_shot_000"]


def test_keyword_search_in_captions(store):
    assert ids(store.keyword_search("office")) == ["interview_shot_000"]


def test_keyword_search_accepts_fts_syntax_and_plain_text(store):
    assert ids(store.keyword_search("sky AND blue")) == ["beach_shot_001"]
    # Punctuation is invalid FTS5 syntax: the store falls back to searching the words.
    assert ids(store.keyword_search("sky, blue?")) == ["beach_shot_001"]
    assert store.keyword_search("?!") == []


def test_semantic_search_ranks_the_right_shot_first(store):
    assert ids(store.semantic_search("calm sea and waves", limit=1)) == ["beach_shot_000"]
    assert ids(store.semantic_search("a man in an office", limit=1)) == ["interview_shot_000"]


def test_embeddings_are_computed_once_and_cached(tmp_path, bundle, encoder):
    store = MediaStore(tmp_path / "db.sqlite", encoder=encoder)
    store.ingest_bundle(bundle)
    store.semantic_search("waves")
    store.semantic_search("sky")
    # 1 call to embed the 3 shots, then 1 call per query.
    assert encoder.calls == 3

    reopened = MediaStore(tmp_path / "db.sqlite", encoder=encoder)
    reopened.semantic_search("office")
    assert encoder.calls == 4  # the shot vectors come from the database


def test_reingesting_replaces_the_content(store, bundle):
    store.ingest_bundle(bundle)
    assert len(store.list_shots()) == 3
    assert len(store.keyword_search("Lisbon", limit=10)) == 1


def test_lookups(store):
    assert [s["shot_id"] for s in store.list_shots(asset_id="beach")] == ["beach_shot_000", "beach_shot_001"]
    assert store.shots_for_segment("beach_seg_001")[0]["shot_id"] == "beach_shot_001"
    shot = store.get_shot("beach_shot_001")
    assert shot["asset_path"] == "/rushes/beach.mp4"
    assert shot["fps"] == 25.0 and shot["has_audio"] is True
    assert store.get_shot("nope") is None


def test_vision_descriptions_are_stored_searchable_and_shown_to_the_agent(tmp_path, bundle, encoder):
    from video_agent.analysis.merge import merge_analysis
    from video_agent.schemas import Moment, ShotDescription
    from video_agent.tools import catalog_entry

    scene = bundle.video_index["beach"].scenes[0]
    scene.description = ShotDescription(
        summary="A drone flies over turquoise water toward a lighthouse.", location="coast, lighthouse visible",
        subjects=["lighthouse", "sea"], shot_size="extreme wide", camera="drone", mood="epic", issues=[],
        moments=[Moment(start=0.0, end=2.0, description="approach over the water", quality="great"),
                 Moment(start=2.0, end=5.0, description="lighthouse fills the frame", quality="usable")],
        tags=["drone", "lighthouse"])
    enriched = merge_analysis(bundle.media_assets, bundle.audio_index, bundle.video_index)
    store = MediaStore(tmp_path / "vision.db", encoder=encoder)
    store.ingest_bundle(enriched)

    assert ids(store.keyword_search("lighthouse")) == ["beach_shot_000"]
    assert ids(store.semantic_search("drone over the lighthouse", limit=1)) == ["beach_shot_000"]
    entry = catalog_entry(store.get_shot("beach_shot_000"))
    assert entry["shot_size"] == "extreme wide" and entry["camera"] == "drone"
    assert entry["moments"][0] == "0.0-2.0s [great] approach over the water"
    assert "drone" in store.get_shot("beach_shot_000")["tags"]


def test_old_databases_get_the_new_columns(tmp_path, bundle, encoder):
    import sqlite3

    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE shot (shot_id TEXT PRIMARY KEY, asset_id TEXT NOT NULL, start REAL NOT NULL, "
                "end REAL NOT NULL, keyframe_path TEXT, transcript TEXT NOT NULL DEFAULT '', "
                "captions TEXT NOT NULL DEFAULT '[]')")
    con.commit()
    con.close()
    store = MediaStore(path, encoder=encoder)
    store.ingest_bundle(bundle)
    assert len(store.list_shots()) == 3
