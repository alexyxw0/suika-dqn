"""Find the environment clone, so PYTHONPATH does not have to be set.

The Gymnasium environment lives in a separate repository — it ships no licence,
so it is cloned and patched rather than vendored (see README.md). That leaves
every entry point needing `suika_rl` on the import path, and `export PYTHONPATH`
is easy to forget. When it is forgotten the failure is opaque: an ImportError
deep inside a worker thread, or a page that looks broken.

So: if `suika_rl` sits beside this file, or one directory up, use it. An
explicit PYTHONPATH still wins, because someone who set one meant it.
"""

from __future__ import annotations

import sys
from pathlib import Path

CLONE_NAME = "suika_rl"


def candidates():
    """Where a clone would plausibly be, nearest first."""
    here = Path(__file__).resolve().parent
    return [here / CLONE_NAME, here.parent / CLONE_NAME]


def ensure(quiet: bool = True) -> Path | None:
    """Put a local clone on sys.path if the environment is not importable.

    Returns the path used, or None if the import already worked or no clone was
    found. Never raises: a caller that genuinely has no environment should fail
    on its own import, with its own message, rather than here.
    """
    try:
        import suika_env  # noqa: F401
        return None
    except ImportError:
        pass

    for path in candidates():
        if (path / "suika_env" / "suika_browser_env.py").exists():
            sys.path.insert(0, str(path))
            if not quiet:
                print(f"  using the environment at {path}")
            return path
    return None


def instructions() -> str:
    """What to tell someone who has no clone at all."""
    return (
        "cannot import suika_env — the environment is not on the path.\n"
        "It is a separate clone; from the repo root:\n\n"
        "    git clone https://github.com/edwhu/suika_rl.git\n"
        "    git -C suika_rl apply env-fixes.patch\n\n"
        "That is enough — scripts find a clone sitting there on their own.\n"
        "See the Setup section of README.md."
    )
