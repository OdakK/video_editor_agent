## Video Editor Agent

End-to-end LangGraph project that (1) analyzes local rushes, (2) builds a searchable
media store, and (3) exposes a conversational editing agent capable of assembling
an OpenTimelineIO sequence and rendering previews with ffmpeg.

### Quick Start

1. Copy environment defaults and fill in secrets:
   ```bash
   cp .env.example .env
   # edit .env with your LangSmith/Azure/OpenAI keys and custom paths if needed
   ```
2. Install dependencies (uv recommended):
   ```bash
   uv sync
   ```
3. Run the analysis once to generate `output/analysis_bundle.json` (or point
   `MEDIA_BUNDLE_PATH` to your own artifacts):
   ```bash
   python main.py --input-dir ./input_rushed --output-dir ./output
   ```
4. Launch the LangGraph dev server for LangSmith-powered tracing and the conversational editor:
   ```bash
   langgraph dev --allow-blocking
   ```
   This boots the LangGraph runtime locally (with LangSmith observability if your
   `LANGSMITH_API_KEY` is set) and provides the editing agent with the media-store
   search tools _plus_ the OTIO timeline actions described below.

### Features
- Whisper-based transcription with word-level timestamps (via `faster-whisper`).
- Scene detection through PySceneDetect with optional BLIP captioning of keyframes.
- Merge step that aligns audio segments with visual shots and writes a JSON bundle
  plus per-modality indexes for downstream tooling.
- CLI wrapper to run the full workflow on local media.

### Installation

Use [uv](https://github.com/astral-sh/uv) or pip/venv to install dependencies:

```bash
uv sync
# or
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

> **Note**  
> Captioning and transcription models are resource intensive. Running on CPU works
> but will be slower than GPU/Metal backends. Adjust `AudioAnalyzer` and
> `VideoAnalyzer` parameters if you have a preferred device.

### Running the Workflow

1. Place your rushes inside a directory (e.g. `./rushes`).
2. Execute the workflow:

```bash
python main.py --input-dir ./rushes --output-dir ./analysis-output
```

Generated artifacts:
- `analysis_bundle.json` – merged timeline atoms.
- `audio_index.json` – transcript segments grouped by asset.
- `video_index.json` – scene metadata (shots, keyframes, captions).
- `keyframes/` – extracted JPEG previews for each detected shot.

Customize detector thresholds or model sizes by constructing
`AudioAnalyzer`/`VideoAnalyzer` instances and passing them into
`create_analysis_workflow` before compiling:

```python
from workflows.analysis import build_analysis_workflow
from media_analysis.audio import AudioAnalyzer
from media_analysis.video import VideoAnalyzer

workflow = build_analysis_workflow(
    audio_analyzer=AudioAnalyzer(model_size="large-v3"),
    video_analyzer=VideoAnalyzer(threshold=28.0, enable_captions=False),
).compile()
workflow.invoke({"input_dir": "./rushes", "output_dir": "./analysis-output"})
```

### Next Steps
- Plug the exported bundle into a retrieval/indexing layer for the conversational
  editing agent.
- Extend `VideoAnalyzer` with object/face detection to populate semantic tags.
- Add diarization in the audio branch for speaker-aware editing operations.

## Media Store

The `media_store` package converts the analysis artifacts into a queryable SQLite
database enriched with text embeddings.

### Building the Store

```python
from pathlib import Path
from media_store import MediaStore, load_analysis_bundle

bundle = load_analysis_bundle(Path("analysis-output/analysis_bundle.json"))
store = MediaStore("media_store.db")
store.initialize(drop_existing=True)
store.ingest_bundle(bundle)
```

This creates:
- `media_asset`, `scene_segment`, `transcript_segment`, and `timeline_atom` tables.
- An FTS index for transcript keyword search.
- Embeddings (SentenceTransformers) enabling semantic queries.

### Query Examples

```python
# Semantic search on transcript content
results = store.semantic_search("parle du ciel bleu", limit=3)
for r in results:
    print(r.asset_id, r.shot_id, r.start, r.end, r.score, r.text_snippet)

# Exact keyword search (SQLite FTS)
fts_results = store.full_text_search("ciel AND bleu", limit=5)
```

These APIs will feed the future editing agent, which can retrieve relevant shots
based on natural-language requests. Further extensions may store embeddings for
captions/tags or expose a REST service for external consumers.


## Agent Tools (LangGraph)

When `langgraph dev --allow-blocking` is running, the editor agent can call the
following structured tools:

- `list_analyzed_shots` – enumerate every detected plan with asset/timecodes.
- `semantic_clip_search` – vector search over transcripts for fuzzy matches.
- `keyword_clip_search` – SQLite FTS queries (supports `AND`, `OR`, `NEAR`, ...).
- `lookup_segment_context` – map a transcript segment ID back to its shot/timecodes.
- Timeline-specific tools (`initialize_timeline`, `insert_shot_into_timeline`,
  `cut_timeline_clip`, `describe_timeline`, `render_timeline_preview`) described
  in the next section.

These cover the current “minimum viable editor”: browsing rushes, picking clips,
inserting them, cutting, and producing a preview render.


## Timeline Editing (OTIO)

The conversational agent can now manipulate an OpenTimelineIO timeline to sketch
rough edits directly from rush analysis results.

- The default timeline file is `timeline.otio` (override via `TIMELINE_PATH`).
- Available tools:
  - `initialize_timeline` – reset/create an empty timeline.
  - `insert_shot_into_timeline` – place a shot at a given time or append to the end.
  - `cut_timeline_clip` – split the clip that spans the requested timestamp.
  - `describe_timeline` – inspect the current sequence with clip/time metadata.
  - `render_timeline_preview` – render the current sequence to MP4 via ffmpeg.
- All edits are non-destructive OTIO operations; the resulting file can later be
  converted into EDL, OTIO JSON, or FFmpeg concat scripts for preview/export.
  Rendering automatically scales/pads shots to the largest source resolution and
  falls back to silent output if some rushes lack audio.

> **Prerequisite**  
> Install `ffmpeg`/`ffprobe` on your system to enable the render tool.
