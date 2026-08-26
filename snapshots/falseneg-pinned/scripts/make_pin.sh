#!/bin/bash
# Build a read-only source pin for a training arm.
#
# WHY a pin at all: sbatch freezes the BATCH SCRIPT at submit time but NOT the python
# source. A job that sits PENDING for hours reads src/ and scripts/ at START time, so any
# edit landing in the working tree in between silently enters the run. The arms compare
# against each other, so that is not a nuisance, it is a corrupted control.
#
# WHY this script exists rather than a hand-rolled cp: both earlier pinning attempts died
# on the parts that are easy to forget, and both failed SILENTLY in the way that matters
# (the training ran fine and produced an empty ri_curve.json):
#
#   - a pin without third_party/ -> eval_checkpoints.py's PathoRobPaths.check() fails and
#     the FOLLOWER dies on its first line while training carries on for hours
#   - a pin without .venv/      -> eval_checkpoints.py's --python default, derived from
#     its own __file__, points at an interpreter that does not exist
#
# So all three of src/, third_party/ and .venv are REQUIRED here and verified to resolve
# before the script will report success. third_party and .venv are symlinks on purpose:
# they are large, immutable, and shared, and the thing that actually needs freezing is the
# python source.
#
#   scripts/make_pin.sh /admin/home/ryan.kim/waiv-snapshots/gridcmp3-offload-pinned

set -euo pipefail

REPO="${WAIV_REPO:-/admin/home/ryan.kim/waiv}"
DEST="${1:?usage: make_pin.sh <dest-snapshot-dir>}"

[ -e "$DEST" ] && { echo "refusing to overwrite existing pin: $DEST" >&2; exit 1; }

mkdir -p "$DEST"
cp -r "$REPO/src"     "$DEST/src"
cp -r "$REPO/scripts" "$DEST/scripts"
[ -d "$REPO/tests" ] && cp -r "$REPO/tests" "$DEST/tests"
cp "$REPO/pyproject.toml" "$DEST/pyproject.toml"
ln -s "$REPO/third_party" "$DEST/third_party"
ln -s "$REPO/.venv"       "$DEST/.venv"

# Drop caches: a stale __pycache__ copied out of the working tree can shadow the pinned
# source if timestamps land badly, which would defeat the entire point of pinning.
find "$DEST/src" "$DEST/scripts" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true

# Read-only LAST, so the copy above is not fighting its own permissions.
chmod -R a-w "$DEST/src" "$DEST/scripts" "$DEST/pyproject.toml"
[ -d "$DEST/tests" ] && chmod -R a-w "$DEST/tests"

# --- verify all three resolve, the way the JOB will use them ------------------------
[ -d "$DEST/src/waivphaet" ]          || { echo "PIN BROKEN: no src/waivphaet"          >&2; exit 3; }
[ -d "$DEST/third_party/PathoROB" ]   || { echo "PIN BROKEN: no third_party/PathoROB"   >&2; exit 3; }
[ -x "$DEST/.venv/bin/python" ]       || { echo "PIN BROKEN: no .venv/bin/python"       >&2; exit 3; }

# Not just "the paths exist" -- import THROUGH the pin and make PathoROB resolve, which is
# the exact call that killed the two dead followers.
PYTHONPATH="$DEST/src" "$DEST/.venv/bin/python" - "$DEST" <<'EOF'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]) / "src"))
import waivphaet
from waivphaet.eval.pathorob_adapter import PathoRobPaths
assert waivphaet.__file__.startswith(sys.argv[1]), \
    f"PYTHONPATH lost to the .venv copy: {waivphaet.__file__}"
PathoRobPaths(root=Path(sys.argv[1]) / "third_party" / "PathoROB").check()
print(f"[pin] waivphaet={waivphaet.__file__}")
print("[pin] PathoROB resolves: OK")
EOF

echo "[pin] built and verified -> $DEST"
