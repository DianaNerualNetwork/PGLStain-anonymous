"""Command-line launcher for PuzzleStain.

A thin wrapper over the training entrypoint. Forwards
``puzzlestain-run <config> [overrides...]`` to the Hydra entrypoint
:func:`puzzlestain.train.main`. On no/invalid arguments, prints available
config presets.

This is the system's launch logic, shipped as the ``puzzlestain-run`` console
script (see ``[project.scripts]`` in ``pyproject.toml``); it is equivalent to
``python -m puzzlestain.train experiment=<config> ...`` and also runnable as
``python -m puzzlestain.cli``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_CONFIG_DIR = Path(__file__).parent / "configs" / "experiment"


def _list_configs() -> list[str]:
    """Return available config preset names, sorted.

    Top-level presets are listed by stem (e.g. ``umdst``); presets inside
    sub-folders are listed with their relative path (e.g. ``mist/umdst``).
    """
    if not _CONFIG_DIR.is_dir():
        return []
    names: list[str] = []
    for p in sorted(_CONFIG_DIR.rglob("*.yaml")):
        rel = p.relative_to(_CONFIG_DIR)
        names.append(str(rel.with_suffix("")))
    return names


def _usage() -> str:
    """Build usage string, including available config presets."""
    configs = " ".join(_list_configs()) or "(none found)"
    return (
        "usage: puzzlestain-run <config> [hydra overrides...]\n"
        "       (equivalent to: python -m puzzlestain.train experiment=<config> ...)\n"
        f"configs: {configs}"
    )


def main() -> None:
    """Launch a PuzzleStain experiment from the command line.

    Reads ``sys.argv``: first positional is the config preset name, remaining
    tokens are Hydra overrides (e.g. ``training.lr=5e-5``). With no arguments,
    or ``-h`` / ``--help``, prints usage and config list and exits. An unknown
    preset name is rejected up front with the list of available presets —
    otherwise Hydra composes the default config and the run fails later with
    an unrelated-looking error (e.g. a missing ``data.dataroot``). Otherwise
    rewrites ``sys.argv`` to ``experiment=<name> <overrides>`` and hands off
    to the Hydra entrypoint :func:`puzzlestain.train.main`.
    """
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print(_usage(), file=sys.stderr)
        raise SystemExit(0 if argv else 2)

    config, *overrides = argv
    if config not in _list_configs():
        print(
            f"error: unknown experiment preset '{config}' "
            f"(no configs/experiment/{config}.yaml)\n\n{_usage()}",
            file=sys.stderr,
        )
        raise SystemExit(2)

    # lazy: keep --help / config listing torch-free (train.py eagerly imports
    # torch + model stack); HARD CONTRACT — do not hoist.
    from puzzlestain import train

    sys.argv = [sys.argv[0], f"experiment={config}", *overrides]
    train.main()


if __name__ == "__main__":
    main()
