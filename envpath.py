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


def _interpreters_with_deps():
    """Other interpreters on this machine that can already run this.

    Conda environments are the usual reason a script fails on `python` and
    works on a full path — the shell's `python` is whichever install came
    first, not the one the dependencies were installed into.
    """
    import subprocess
    found = []
    seen = {Path(sys.executable).resolve()}
    roots = [Path.home() / "miniconda3" / "envs", Path.home() / "anaconda3" / "envs",
             Path("/opt/miniconda3/envs"), Path("/opt/anaconda3/envs")]
    for root in roots:
        if not root.is_dir():
            continue
        for env in sorted(root.iterdir()):
            exe = env / "bin" / "python"
            if not exe.exists() or exe.resolve() in seen:
                continue
            seen.add(exe.resolve())
            try:
                ok = subprocess.run(
                    [str(exe), "-c", "import gymnasium, selenium"],
                    capture_output=True, timeout=25).returncode == 0
            except Exception:      # noqa: BLE001
                ok = False
            if ok:
                found.append(exe)
    return found


def diagnose(error: ImportError | None = None) -> str:
    """Say what is actually wrong, rather than assuming it is the path.

    Two very different failures reach the same `except ImportError`, and
    conflating them sends people to fix the wrong thing: there may be no clone
    of the environment, or there may be a clone that this interpreter cannot
    load because the dependencies were installed into a different one. The
    message used to claim the first in both cases.
    """
    clone = next((c for c in candidates()
                  if (c / "suika_env" / "suika_browser_env.py").exists()), None)

    if clone is None:
        return ("cannot find the environment — it is a separate clone.\n"
                "From the repo root:\n\n"
                "    git clone https://github.com/edwhu/suika_rl.git\n"
                "    git -C suika_rl apply env-fixes.patch\n\n"
                "Leave it there and scripts find it on their own.\n"
                "See the Setup section of README.md.")

    missing = ""
    if error is not None:
        text = str(error)
        if "No module named" in text:
            missing = text.split("No module named", 1)[1].strip().strip("'\"")

    lines = [f"found the environment at {clone}, but this interpreter cannot "
             f"load it."]
    if missing:
        lines.append(f"Missing package: {missing}")
    lines.append(f"Running under: {sys.executable}")

    others = _interpreters_with_deps()
    if others:
        lines.append("")
        lines.append("These interpreters already have what it needs:")
        for exe in others[:4]:
            lines.append(f"    {exe} scripts/dashboard.py")
    else:
        lines.append("")
        lines.append("Install the dependencies for this interpreter:")
        lines.append("    pip install -r requirements.txt")
        lines.append("    pip install -e suika_rl")
    return "\n".join(lines)


def instructions() -> str:
    """Kept for callers that have no exception to hand."""
    return diagnose(None)
