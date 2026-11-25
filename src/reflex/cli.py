"""Small command router between reusable core operations and experiments."""

from __future__ import annotations

import argparse
import sys

CORE_COMMANDS = {
    "replay", "export-parquet", "state", "lens",
}
EXPERIMENT_COMMANDS = {
    "experiments", "benchmark", "demo-news", "ingest-bpi", "ingest-loghub",
    # Kept as familiar aliases; new examples use ``reflex experiments discovery``.
    "demo", "generate", "discover", "discovery-benchmark",
}


def build_parser() -> argparse.ArgumentParser:
    """Build concise top-level help without importing experiment modules."""
    parser = argparse.ArgumentParser(
        prog="reflex",
        description="REFLEX reusable core tools and opt-in experiment profiles",
        epilog="Use `reflex experiments --help` for research workflows. Dataset adapters and synthetic discovery are experiment commands.",
    )
    sub = parser.add_subparsers(dest="command")
    for name in sorted(CORE_COMMANDS):
        sub.add_parser(name, help="reusable event, SQL Lens, or state operation")
    for name in sorted(EXPERIMENT_COMMANDS):
        help_text = (
            "run experiment profiles"
            if name == "experiments"
            else "compatibility alias; prefer a command under `reflex experiments`"
        )
        sub.add_parser(name, help=help_text)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Route reusable commands to core and every research workflow to experiments."""
    command_args = sys.argv[1:] if argv is None else argv
    if not command_args:
        build_parser().print_help()
        return 0
    if command_args[0] in {"-h", "--help"}:
        build_parser().parse_args(command_args)
        return 0
    if command_args[0] in CORE_COMMANDS:
        from .core.cli import main as core_main

        return core_main(command_args)
    from .experiments.cli import main as experiments_main

    return experiments_main(command_args)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
