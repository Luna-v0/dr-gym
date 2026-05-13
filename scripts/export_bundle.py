#!/usr/bin/env python3
"""Package a trained model into a DeepRacer physical-car upload tar.gz.

Usage:

    # From an SB3 zip (our training output), with metadata from app.py:
    uv run python scripts/export_bundle.py \\
        --model artifacts/<run>/final_model.zip \\
        --app app.py \\
        --output bundle.tar.gz

    # From an SB3 zip with a sibling <model>.model_metadata.json (default
    # layout for our checkpoint dirs — metadata source is auto-detected):
    uv run python scripts/export_bundle.py \\
        --model artifacts/<run>/final_model.zip \\
        --output bundle.tar.gz

    # From a pre-existing TF frozen-graph .pb:
    uv run python scripts/export_bundle.py \\
        --model my_model.pb \\
        --metadata my_model_metadata.json \\
        --output bundle.tar.gz

Bundle layout:

    bundle.tar.gz
    ├── model_metadata.json
    └── agent/
        └── agent.{pb,onnx}

Logic lives in ``gym_dr.export`` — this is a thin argparse wrapper around
``gym_dr.export.export_bundle``.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make the project root importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gym_dr.export import export_bundle  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, type=Path,
                        help="Path to .pb, .onnx, or SB3 .zip")
    parser.add_argument("--output", required=True, type=Path,
                        help="Path for the resulting .tar.gz")
    meta = parser.add_mutually_exclusive_group()
    meta.add_argument("--metadata", type=Path,
                      help="Explicit model_metadata.json to embed")
    meta.add_argument("--app", type=Path,
                      help="app.py (or any module exporting `experiment`/`base`) to render metadata from")
    parser.add_argument("--bundle-filename",
                        help="Override in-tar model filename (default: agent.pb or agent.onnx)")
    parser.add_argument("--opset", type=int, default=11,
                        help="ONNX opset for SB3 .zip exports (default 11)")
    parser.add_argument("--input-name", default="input",
                        help="ONNX input tensor name for non-dict obs (default 'input')")
    parser.add_argument("--output-name", default="action",
                        help="ONNX output tensor name (default 'action')")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    out = export_bundle(
        model_path=args.model,
        output_path=args.output,
        metadata_path=args.metadata,
        app_path=args.app,
        bundle_filename=args.bundle_filename,
        opset_version=args.opset,
        input_name=args.input_name,
        output_name=args.output_name,
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
