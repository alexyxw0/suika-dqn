#!/usr/bin/env bash
# Verify env-fixes.patch against a clean upstream clone.
#
# `git apply --check` only tells you the hunks fit. It does not tell you the
# result is valid Python — a patch can apply perfectly and still splice two
# statements onto one line. This applies it for real and then compiles the
# result, which is the check that actually matters.
set -euo pipefail

UPSTREAM=${UPSTREAM:-https://github.com/edwhu/suika_rl.git}
# The import check needs an interpreter with gymnasium and selenium on it,
# which the system python3 is not. It was hardcoded, so the most useful
# check in this script could not actually run here.
PYTHON=${PYTHON:-python3}
PATCH=$(cd "$(dirname "$0")/.." && pwd)/env-fixes.patch
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

echo "cloning $UPSTREAM"
git clone -q "$UPSTREAM" "$WORK/suika_rl"
cd "$WORK/suika_rl"

echo "applying $(basename "$PATCH")"
git apply "$PATCH"

echo "compiling the result"
"$PYTHON" -m py_compile suika_env/suika_browser_env.py
command -v node >/dev/null && node --check suika_env/suika-game/index.js

# A patch that duplicates a method still compiles, still imports, and quietly
# uses whichever definition came last. Worth an explicit check — this caught
# two duplicated methods.
echo "checking for duplicate definitions"
"$PYTHON" - "$WORK/suika_rl/suika_env/suika_browser_env.py" <<'PYEOF'
import ast, sys
tree = ast.parse(open(sys.argv[1]).read())
bad = []
for node in ast.walk(tree):
    if isinstance(node, (ast.ClassDef, ast.Module)):
        names = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
        bad += [f"{getattr(node, 'name', 'module')}.{n}" for n in set(names) if names.count(n) > 1]
if bad:
    sys.exit("duplicate definitions after patching: " + ", ".join(sorted(bad)))
PYEOF

echo "importing the module"
PYTHONPATH="$WORK/suika_rl" "$PYTHON" -c "import suika_env.suika_browser_env" >/dev/null

echo "OK — patch applies, compiles, and imports"
