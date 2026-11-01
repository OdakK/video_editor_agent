# 🎬 Video Editor Agent

[![CI](https://github.com/OdakK/video_editor_agent/actions/workflows/ci.yml/badge.svg)](https://github.com/OdakK/video_editor_agent/actions/workflows/ci.yml)

**Edit video by talking to an agent.** Drop your raw footage in a folder, let the pipeline watch and listen to it, then ask for a rough cut in plain language:

> *"Find the shot where she talks about the blue sky, put it at the start, then add the wide shot of the beach after it and render a preview."*

The agent searches your rushes by meaning or by keyword, places shots on an [OpenTimelineIO](https://opentimelineio.readthedocs.io/) timeline, cuts, moves and removes clips, and renders an MP4 preview with ffmpeg. Everything runs locally except the LLM call.

Built with **LangGraph**, **faster-whisper**, **PySceneDetect**, **BLIP**, **SentenceTransformers**, **SQLite FTS5** and **OpenTimelineIO**.

![The web UI: shot library, preview player, timeline and the agent's chat](docs/ui.jpg)

*The web UI on the demo footage (synthetic test patterns): one request, and the agent searched the shots, built the timeline and rendered the preview.*

---

## How it works

The project is split into two parts: an **offline analysis pipeline** (a LangGraph graph) that turns raw footage into structured, searchable data, and a **conversational agent** that edits a timeline with tools.

```mermaid
flowchart LR
    subgraph A["1 · Analysis pipeline (LangGraph)"]
        direction LR
        R[/"Raw rushes<br/>(.mp4, .mov, .mkv…)"/] --> I["ingest<br/>ffprobe metadata"]
        I --> AU["audio_analysis<br/>faster-whisper<br/>word timestamps"]
        AU --> V["video_analysis<br/>PySceneDetect shots<br/>+ BLIP keyframe captions"]
        V --> D["describe<br/>vision LLM: place, shot size,<br/>camera, issues, good moments"]
        D --> M["merge<br/>align transcript ↔ shots"]
        M --> B[("analysis_bundle.json")]
    end

    subgraph S["2 · Media store (SQLite)"]
        B --> FTS[("FTS5 index<br/>speech + captions")]
        B --> EMB[("shot embeddings<br/>cached")]
    end

    subgraph G["3 · Editing agent (LangChain create_agent)"]
        U(["🧑 You"]) <--> AG{{"LLM agent"}}
        AG -- search --> FTS
        AG -- search --> EMB
        AG -- edit --> T[("timeline.otio")]
        T -- render --> P[/"preview.mp4<br/>(ffmpeg)"/]
    end
```

1. **Ingest** finds every video file in the input folder and reads its duration, frame rate and audio track with ffprobe. Files without audio skip transcription.
2. **Audio analysis** transcribes each rush with faster-whisper (voice-activity filtering, word-level timestamps).
3. **Video analysis** detects shot boundaries with PySceneDetect, extracts a keyframe per shot and captions it with BLIP (optional).
4. **Describe** sends a few frames of each shot, with their timestamps, to a vision LLM (GPT-6 Luna by default). It returns what an editor logs: a summary, the place (with names read from signs and landmarks), subjects, shot size, camera move, mood, technical issues, and the shot split into moments rated *great*, *usable* or *bad*. Skipped without an API key.
5. **Merge** aligns transcript segments with the shots they overlap, producing "timeline atoms": one shot, its timecodes, what is said and what is seen.
6. **Media store** loads the bundle into SQLite: an FTS5 index over both speech and captions for exact queries, and one embedding per shot (transcript + captions) for semantic search. Embeddings are computed once and cached in the database.
7. **Editing agent** is a tool-calling LLM that combines search and timeline tools to answer requests and build the edit. Tool errors (unknown shot, empty timeline…) are sent back to the model so it can correct itself.

## Agent tools

| Tool | What it does |
|---|---|
| `list_shots` | The catalog the agent reads first: duration, summary, place, shot size, camera, issues, moments with their quality and timecodes, speech. |
| `get_shot` | One shot in full, with each spoken sentence and its timecodes, so cuts fall between sentences. |
| `semantic_search` / `keyword_search` | Find shots by meaning or by exact words, over speech, captions and descriptions. |
| `build_timeline` | Lay out a whole edit in one call: shots in order, each with its own in and out points. |
| `insert_shot` | Add a shot, or only part of it, at the end or at a given time. |
| `trim_clip` | Change which seconds of its source a clip plays. |
| `split_clip` / `remove_clip` / `move_clip` | Cut, remove or reorder clips. |
| `describe_timeline` | The current edit: total duration, clip ids, timeline and source timecodes. |
| `render_preview` | Render to MP4 with ffmpeg: scales and pads to the largest source, falls back to silent output if a rush has no audio. |

The system prompt gives the agent an editor's method rather than a list of commands: read the material,
plan (target length, story order, which seconds of each shot and why), lay the edit out in one
`build_timeline` call, then refine and check. And the craft: open on a hook, establish the place, group
by place or theme instead of file order, 1.5-4 s per clip without speech, vary shot sizes, never use a
moment rated *bad*, cut between sentences, and do not claim what the footage does not show.

All edits are non-destructive: the source files are never touched, and the `.otio` file can be opened in other OTIO-compatible tools or converted to EDL.

## Getting started

### Prerequisites

- Python 3.12 and [uv](https://github.com/astral-sh/uv)
- `ffmpeg` and `ffprobe` on your `PATH`
- An API key for the LLM: OpenAI GPT-6 Luna by default, or any provider supported by LangChain (`VIDEO_AGENT_MODEL=anthropic:...`, `azure_openai:...`), or OpenRouter (see `.env.example`)

### Install

```bash
git clone https://github.com/OdakK/video_editor_agent.git
cd video_editor_agent
uv sync --extra analysis      # Whisper, BLIP, PySceneDetect, sentence-transformers
cp .env.example .env          # then add your API key
```

### 1. Analyze your rushes

```bash
uv run video-agent analyze --input-dir ./rushes
```

Options: `--whisper-model small` (faster) or `large-v3` (more accurate), `--language fr`, `--threshold 27` (more sensitive shot detection), `--no-captions`.
This writes `output/analysis_bundle.json` and one JPEG keyframe per shot in `output/keyframes/`.

> Whisper and BLIP are heavy: CPU works but is slow, a GPU or Apple Silicon helps a lot.

### 2. Edit in the web UI

```bash
uv sync --extra analysis --extra ui
uv run video-agent ui              # opens http://127.0.0.1:8000
```

Shot library on the left (each shot shows its description, shot size, camera move and any issue), preview and timeline in the middle, the agent on the right. You can edit by hand
(add, move, remove clips, render) or ask the agent, and watch each tool call stream in while it works.
Manual and agent edits go through the same timeline code.

**Import rushes from the page**: click *Import* or drop video files on the shot library. Each file is
analyzed in the background (transcription, shot detection, captions), with its progress shown in the
library, then indexed and added to the existing analysis. The UI starts with an empty library if you
have not analyzed anything yet. Without the `analysis` extra, imports fall back to a basic mode: one
shot per file, with a keyframe but no transcript. Imports use Whisper `small` by default, for a
reasonable wait (`--whisper-model` to change it).

**360° cameras (Insta360 `.insv`)**: a `.insv` file holds the two fisheye lenses of the camera. Each lens is
re-projected on its own into a flat vertical view with ffmpeg's `v360` filter (no stitching, so no seam),
giving two synchronized rushes per clip: `<clip>_lens1` and `<clip>_lens2`, usually the scene in front and a
selfie of the person filming. The agent treats them as a two-camera shoot and cuts between them while keeping
time moving forward; `build_timeline` flags any clip that replays seconds already shown from the other lens.

It uses a **proxy workflow**, like professional editing software. Decoding 8K HEVC is the expensive part, so
the import only decodes the keyframes (one every 2 s) into light 540×960 proxies with the untouched
soundtrack: enough for transcription and the vision model. The final render reads the original `.insv` at
full quality, for the seconds actually used. A 360 recording is one continuous take, so shot detection is
skipped and the vision model splits it into moments. The soundtrack is transcribed once for both views.

| 1 minute of 8K 360 footage | Time | Size per view |
|---|---|---|
| Decode every frame, two 1080×1920 views (first version) | 54 s | 124 MB |
| Keyframes only, two 540×960 proxies | 0.7 s | 2.4 MB |
| Render 7 s of edit from the original, at 1080×1920 | 7.7 s | |

On real footage, a 17-minute clip is imported in about 5 minutes, most of it transcription and vision.
Use *Import from a folder* for these files: they are read in place instead of being uploaded (a 10-minute
clip weighs about 9 GB), and `.lrv` proxies are skipped. Not handled yet: Insta360's gyro stabilization, and
free reframing (yaw, pitch, roll), which helmet or chest mounts need.

**Delete a rush** with the bin icon on any of its shots: its shots leave the library and its clips
leave the timeline. Only files the UI owns are deleted from disk (imported copies and keyframes);
original rushes analyzed with `video-agent analyze` are never touched.

**No rushes at hand?** `uv run video-agent ui --demo` generates three synthetic rushes with ffmpeg
(test patterns and tones, with a hand-written transcript and captions) and opens them. It needs no
ML model, only ffmpeg and an API key for the LLM.

### 3. Or talk to the editor in the terminal

```bash
uv run video-agent chat -v      # -v prints every tool call
```

Or with LangGraph Studio, to watch the graph and each tool call:

```bash
uv sync --extra analysis --extra studio
uv run langgraph dev --allow-blocking
```

Try requests like:

- *"List all the shots from the first rush."*
- *"Find where someone says 'ciel bleu' and add that shot to the timeline."*
- *"Cut the clip at 12.5 seconds, remove the second half and describe the timeline."*
- *"Render a preview to preview.mp4."*

## Tests

```bash
uv sync && uv run pytest -v
```

The tests run in a few seconds, without models or API keys: the analysis output is a small hand-made bundle and the sentence encoder is replaced by a deterministic bag-of-words hash. They cover the transcript ↔ shot alignment, keyword and semantic search (including the embedding cache), every timeline operation, **concurrent edits** (agents call tools in parallel), the ffmpeg command, the agent loop and the web UI's API with a scripted model, rush import and deletion (including broken files, name clashes, and a check that original rushes are never deleted), and an **end-to-end render**: two test clips are generated with ffmpeg, edited, rendered, and the output duration and resolution are checked. CI runs them on every push.

## Project structure

```
src/video_agent/
├── analysis/
│   ├── workflow.py   # LangGraph pipeline: ingest → audio → video → merge
│   ├── ingest.py     # Media discovery
│   ├── audio.py      # faster-whisper transcription
│   ├── video.py      # PySceneDetect shots, keyframes, BLIP captions
│   └── merge.py      # Transcript ↔ shot alignment
├── schemas.py        # Pydantic models (MediaAsset, SceneSegment, TimelineAtom…)
├── probe.py          # ffprobe metadata
├── store.py          # SQLite store: FTS5 and cached embeddings
├── timeline.py       # OTIO editing and ffmpeg rendering
├── tools.py          # The agent's tools
├── agent.py          # Editing agent (also exposed to `langgraph dev`)
├── ui/               # FastAPI server + static page (no build step), chat streamed as SSE
├── library.py        # Rush import from the UI: background analysis and re-indexing
├── insta360.py       # 360° .insv files: one flat view per fisheye lens
├── demo.py           # Synthetic demo rushes
└── cli.py            # video-agent analyze | chat | ui
tests/                # Fast tests, no model or API key needed
```

## Design choices

- **Two stages, not one graph.** Analysis is slow and deterministic, and runs once per batch of rushes. Editing is fast, interactive and LLM-driven. Keeping them separate means you never re-transcribe footage to try a new edit.
- **Timeline atoms as the shared language.** Each shot carries its timecodes, overlapping transcript and captions, so the agent reasons about *shots* instead of raw frames or audio.
- **Keyword and semantic search, over speech and image.** FTS5 handles exact words and names; embeddings handle "the part where they talk about the weather". Both include the BLIP captions, so "the wide shot of the beach" works even when nobody speaks.
- **Heavy models are optional dependencies, loaded lazily.** The store, timeline and agent install and test without PyTorch, which keeps CI fast.
- **Long shots are logged in 20 s windows.** A 360 walk is one continuous take with no cut, so one description for the whole shot would give the agent a handful of vague moments for ten minutes of footage. Long shots are described window by window, four in parallel, and the moments are concatenated.
- **Logging before editing.** The first version only had one BLIP caption per shot, and no way to cut inside a rush: asked for a travel vlog, the agent put whole rushes end to end. What made the difference was not a bigger model but better material and better tools: a vision model that logs each shot like an assistant editor (place, shot size, good and bad moments with timecodes), in and out points on every clip, and a prompt that describes the editing craft. On the same request and the same rushes, the agent went from 6 full rushes in file order (42 s) to a 20 s edit with trimmed clips, a hook, a story order and a closing shot.
- **One lock per timeline file.** Models call tools in parallel (three `insert_shot` at once is common), and each edit is a load → modify → save cycle on the `.otio` file. Without the lock, parallel inserts overwrote each other: this was caught in a real session with the UI, and a test now covers it.
- **OTIO as the output format.** It's an open, non-destructive timeline format, so the rough cut isn't locked into this tool.

## Limitations and roadmap

This is a working prototype, not a finished editor.

- One video track; no transitions, titles or audio mixing yet.
- Semantic search is a brute-force cosine similarity over cached vectors: fine for hundreds of shots, a vector index would be the next step for large libraries.
- Next ideas: speaker diarization for speaker-aware edits, object and face detection for visual tags, and an evaluation set of editing requests with expected timelines.

## License

MIT
