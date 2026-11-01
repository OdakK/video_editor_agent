"""Command line.

    video-agent analyze --input-dir rushes/     # transcribe, detect shots, caption keyframes
    video-agent chat                            # talk to the editing agent
    video-agent ui [--demo]                     # web UI: shots, timeline, preview and chat
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path


def analyze(args: argparse.Namespace) -> None:
    from dotenv import load_dotenv

    from .analysis.audio import AudioAnalyzer
    from .analysis.video import VideoAnalyzer
    from .analysis.workflow import build_analysis_workflow

    load_dotenv()  # API key for the vision descriptions

    workflow = build_analysis_workflow(
        AudioAnalyzer(model_size=args.whisper_model, language=args.language),
        VideoAnalyzer(threshold=args.threshold, enable_captions=not args.no_captions),
    )
    state = workflow.invoke({"input_dir": args.input_dir, "output_dir": args.output_dir})
    bundle = state["analysis_bundle"]
    print(f"Analyzed {len(bundle.media_assets)} rush(es), {len(bundle.timeline_atoms)} shot(s).")
    for name, path in state["artifacts"].items():
        print(f"- {name}: {Path(path).resolve()}")


def chat(args: argparse.Namespace) -> None:
    from .agent import create_editing_agent

    agent = create_editing_agent(bundle_path=args.bundle, timeline_path=args.timeline)
    print("Editing agent ready. Type 'exit' to quit.")
    messages: list = []  # keep the history so follow-ups ("cut it at 3s") refer to earlier turns
    while True:
        try:
            query = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if query.lower() in {"exit", "quit"}:
            return
        if not query:
            continue
        messages.append({"role": "user", "content": query})
        result = agent.invoke({"messages": messages})
        messages = result["messages"]
        if args.verbose:
            for m in messages:
                for call in getattr(m, "tool_calls", None) or []:
                    print(f"  [tool] {call['name']}({call['args']})")
        print(f"\nagent> {messages[-1].content}")


def ui(args: argparse.Namespace) -> None:
    import uvicorn
    from dotenv import load_dotenv

    from .library import Library, has_analysis_extra
    from .store import HashingEncoder, MediaStore
    from .timeline import TimelineManager
    from .ui.server import Session, create_app, model_name

    load_dotenv()
    full = has_analysis_extra()
    if args.demo:
        from .demo import build_demo

        print("Building the demo footage with ffmpeg…")
        bundle_path = build_demo(args.demo_dir)
    else:
        # Start from an existing analysis, or from an empty library to fill by importing rushes.
        bundle_path = args.bundle or Path("output/analysis_bundle.json")
    if not full:
        print("The analysis extra is not installed: imported rushes get one shot each, without transcript.\n"
              "Install it for transcription, shot detection and captions: uv sync --extra analysis --extra ui")
    # The demo and the basic mode use the lightweight encoder, so nothing has to be downloaded.
    encoder = None if full and not args.demo else HashingEncoder()
    workdir = bundle_path.parent
    workdir.mkdir(parents=True, exist_ok=True)
    store = MediaStore(workdir / "media_store.db", encoder=encoder)
    library = Library(bundle_path, store, whisper_model=args.whisper_model)
    store.ingest_bundle(library.bundle)
    stale = library.refresh_stale()
    if stale:
        print(f"Re-analyzing {len(stale)} rush(es) in the background: {', '.join(j.asset_id for j in stale)}")
    if not library.vision:
        print("No LLM API key found: shots will not get vision descriptions.")
    session = Session(store, TimelineManager(workdir / "timeline.otio"), model_name(), library)
    print(f"Video Editor Agent UI: http://{args.host}:{args.port}")
    uvicorn.run(create_app(session), host=args.host, port=args.port, log_level="warning")


def main() -> None:
    parser = argparse.ArgumentParser(prog="video-agent", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("analyze", help="analyze a folder of rushes")
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", default="output")
    p.add_argument("--whisper-model", default="medium", help="tiny, base, small, medium, large-v3")
    p.add_argument("--language", default=None, help="force the transcription language, e.g. fr")
    p.add_argument("--threshold", type=float, default=30.0, help="shot detection sensitivity")
    p.add_argument("--no-captions", action="store_true", help="skip BLIP keyframe captions")
    p.set_defaults(func=analyze)

    p = sub.add_parser("chat", help="talk to the editing agent")
    p.add_argument("--bundle", type=Path, default=None, help="analysis_bundle.json")
    p.add_argument("--timeline", type=Path, default=None, help=".otio file to edit")
    p.add_argument("-v", "--verbose", action="store_true", help="print tool calls")
    p.set_defaults(func=chat)

    p = sub.add_parser("ui", help="open the web UI")
    p.add_argument("--bundle", type=Path, default=None, help="analysis_bundle.json (default: output/)")
    p.add_argument("--demo", action="store_true", help="generate synthetic demo rushes and open them")
    p.add_argument("--demo-dir", type=Path, default=Path("demo"))
    p.add_argument("--whisper-model", default="small", help="Whisper model for imported rushes")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=ui)

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
