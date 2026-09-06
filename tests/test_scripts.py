"""Every entry point must at least import and parse its arguments.

This exists because it did not. Trimming `observation(obs, mode)` down to
`observation(obs)` left `scripts/eval_policy.py` calling it with two arguments,
and nothing noticed until a measurement run crashed forty minutes in — the
unit tests are pure numpy and never import the scripts, and the scripts need a
browser to do anything, so neither layer covered them.

Importing a module executes its top-level code and binds its functions;
`--help` drives `parse_args`. That is not much, but it is exactly the layer
where a rename or a signature change goes unnoticed.

Skips cleanly where TensorFlow or Selenium are absent, so the suite still runs
in an environment that has neither.
"""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# The check_* scripts take no arguments and run their verification immediately —
# one drives a browser, the other trains a network — so `--help` would not
# short-circuit them and they do not belong in a unit-test run. They are meant
# to be invoked directly.
VERIFIERS = {"check_seeding.py", "check_train_step.py"}
SCRIPTS = sorted(p.name for p in (ROOT / "scripts").glob("*.py")
                 if p.name not in VERIFIERS)
ENTRY_POINTS = ["train.py", "ppo.py"] + [f"scripts/{s}" for s in SCRIPTS]


def _run(path):
    return subprocess.run([sys.executable, str(ROOT / path), "--help"],
                          capture_output=True, text=True, timeout=300,
                          cwd=ROOT)


@pytest.mark.parametrize("path", ENTRY_POINTS)
def test_entry_point_imports_and_parses_arguments(path):
    done = _run(path)
    if done.returncode != 0:
        missing = ("No module named" in done.stderr
                   or "cannot import name" in done.stderr
                   and "agent" not in done.stderr)
        if missing:
            pytest.skip(f"dependency unavailable: {done.stderr.strip()[-120:]}")
    assert done.returncode == 0, (
        f"{path} --help failed:\n{done.stderr[-2000:]}")
    assert "usage:" in done.stdout.lower()
