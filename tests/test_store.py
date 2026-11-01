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
