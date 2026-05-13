"""Internal helpers used by run_cpu_training.sh on the host side.

User-facing entrypoints live in experiment scripts under `experiments/` —
edit one and run `python experiments/<name>.py`. This module only exposes
the host-only `prepare-metadata` step and a debug `inspect` command.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from gym_dr.config import load_config


def _cmd_inspect(args: argparse.Namespace) -> int:
    from gym_dr.app import inspect as inspect_experiment

    cfg = load_config(args.experiment)
    inspect_experiment(cfg)
    return 0


def _cmd_prepare_metadata(args: argparse.Namespace) -> int:
    from gym_dr.action_space import write_model_metadata

    cfg = load_config(args.experiment)
    out_path = Path(args.output)
    write_model_metadata(out_path, cfg.action_space)
    print(f"Wrote {out_path} ({cfg.action_space.action_space_type})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gym-dr")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_inspect = sub.add_parser("inspect", help="print the resolved experiment")
    p_inspect.add_argument("experiment", help="path to an experiments/<name>.py")
    p_inspect.set_defaults(func=_cmd_inspect)

    p_prepare = sub.add_parser(
        "prepare-metadata",
        help="generate model_metadata.json from an experiment's action_space (host helper)",
    )
    p_prepare.add_argument("experiment", help="path to an experiments/<name>.py")
    p_prepare.add_argument("--output", default="model_metadata.json")
    p_prepare.set_defaults(func=_cmd_prepare_metadata)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
