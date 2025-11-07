from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

# Ensure local src/ directory is importable when running without `pip install -e .`.
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from workflows.analysis import build_analysis_workflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the rush analysis workflow locally."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing the local rushes to analyse.",
    )
    parser.add_argument(
        "--output-dir",
        default="./analysis-output",
        help="Directory where analysis artifacts will be written.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ...).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    workflow = build_analysis_workflow().compile()
    final_state = workflow.invoke(
        {
            "input_dir": args.input_dir,
            "output_dir": args.output_dir,
        }
    )

    artifacts = final_state.get("artifacts", {})
    print("Analysis complete. Generated artifacts:")
    for name, path in artifacts.items():
        print(f"- {name}: {Path(path).resolve()}")


if __name__ == "__main__":
    main()
