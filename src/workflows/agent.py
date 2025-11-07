from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent  # /src
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from media_store import MediaStore, load_analysis_bundle
from timeline import TimelineManager
from tools import build_agent_tools

load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2023-06-01-preview")
DEFAULT_BUNDLE_PATH = Path(
    os.getenv("MEDIA_BUNDLE_PATH", "analysis-output/analysis_bundle.json")
)
DEFAULT_DB_PATH = Path(os.getenv("MEDIA_STORE_PATH", "media_store.db"))
DEFAULT_TIMELINE_PATH = Path(os.getenv("TIMELINE_PATH", "timeline.otio"))


def _create_llm() -> BaseChatModel:
    if AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT:
        return AzureChatOpenAI(
            azure_deployment=AZURE_OPENAI_DEPLOYMENT,
            api_version=AZURE_OPENAI_API_VERSION,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        return ChatOpenAI(
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

    raise RuntimeError(
        "No LLM credentials found. Configure Azure OpenAI or OpenAI environment variables."
    )


def create_media_query_agent(
    bundle_path: Path | str = DEFAULT_BUNDLE_PATH,
    *,
    db_path: Optional[Path | str] = DEFAULT_DB_PATH,
    timeline_path: Path | str = DEFAULT_TIMELINE_PATH,
    verbose: bool = False,
):
    """Create an agent capable of answering questions about analyzed rushes."""
    bundle_path = Path(bundle_path).expanduser().resolve()
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"Analysis bundle not found at {bundle_path}. "
            "Run the analysis workflow first or set MEDIA_BUNDLE_PATH."
        )

    db_path = Path(db_path).expanduser().resolve() if db_path else DEFAULT_DB_PATH

    bundle = load_analysis_bundle(bundle_path)
    store = MediaStore(db_path)
    store.initialize()
    store.ingest_bundle(bundle, overwrite=True)

    timeline_manager = TimelineManager(timeline_path)

    tools = build_agent_tools(store, timeline_manager)
    llm = _create_llm()

    system_prompt = (
        "Tu es un assistant monteur. Utilise les outils disponibles pour rechercher "
        "les plans pertinents à partir des rushes analysés. Donne toujours les IDs "
        "de shot et les timecodes lorsque tu réponds. Si l'information manque, "
        "explique ce qui te manque."
    )

    agent_chain = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
    )

    return agent_chain


def build_default_agent():
    """Build the default agent using project-level bundle and DB paths."""
    return create_media_query_agent(
        DEFAULT_BUNDLE_PATH, db_path=DEFAULT_DB_PATH, timeline_path=DEFAULT_TIMELINE_PATH
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Conversational agent for querying analyzed rushes."
    )
    parser.add_argument(
        "--bundle-path",
        default=str(DEFAULT_BUNDLE_PATH),
        help="Path to analysis_bundle.json produced by the workflow.",
    )
    parser.add_argument(
        "--db-path",
        default=str(DEFAULT_DB_PATH),
        help="SQLite file used to cache the media store.",
    )
    parser.add_argument(
        "--timeline-path",
        default=str(DEFAULT_TIMELINE_PATH),
        help="Path to the OTIO timeline file used for editing operations.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose LangChain logging.",
    )
    args = parser.parse_args()

    agent_executor = create_media_query_agent(
        bundle_path=args.bundle_path,
        db_path=args.db_path,
        timeline_path=args.timeline_path,
        verbose=args.verbose,
    )

    print("Agent prêt. Tape 'exit' pour quitter.")
    try:
        while True:
            try:
                query = input("> ").strip()
            except EOFError:
                break
            if not query:
                continue
            if query.lower() in {"exit", "quit"}:
                break

            result = agent_executor.invoke({"input": query})
            output = result.get("output") if isinstance(result, dict) else result
            print(output)
    except KeyboardInterrupt:
        pass


try:
    agent = build_default_agent()
except Exception:
    agent = None


if __name__ == "__main__":
    main()
