"""THUNDER data completeness check: every file referenced by data_splits/<ds>.json exists.

usage: python scripts/evals/thunder_datacheck.py [DATA_ROOT]  (default $THUNDER_DATA) [--out report.json] [--datasets a,b]
Path entries are either strings (relative to DATA_ROOT/<ds>) or [path, y0, y1, x0, x1] crops.
pannuke / patch_camelyon reference split-level array files (strings, not lists).
Exit 1 if anything is missing. Stdlib only (runs under any python).
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

args = [a for a in sys.argv[1:]]
out = None
only = None
if "--out" in args:
    i = args.index("--out"); out = args[i + 1]; del args[i:i + 2]
if "--datasets" in args:
    i = args.index("--datasets"); only = args[i + 1].split(","); del args[i:i + 2]
root = Path(args[0] if args else os.environ.get("THUNDER_DATA") or sys.exit("pass DATA_ROOT or set THUNDER_DATA"))


def leaves(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, list):
        if obj and isinstance(obj[0], str) and len(obj) == 5 and all(isinstance(v, int) for v in obj[1:]):
            yield obj[0]
        else:
            for v in obj:
                yield from leaves(v)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from leaves(v)


report = {}
bad = 0
for split_file in sorted((root / "data_splits").glob("*.json")):
    ds = split_file.stem
    if only and ds not in only:
        continue
    data = json.loads(split_file.read_text())
    paths = set()
    for key in ("train", "val", "test"):
        for field in ("images", "labels"):
            v = data.get(key, {}).get(field)
            if v is None:
                continue
            for p in leaves(v):
                if "/" in p or "." in p:
                    paths.add(p)
    paths = sorted(paths)
    with ThreadPoolExecutor(64) as pool:
        exists = list(pool.map(lambda p: os.path.exists(root / ds / p), paths))
    missing = [p for p, e in zip(paths, exists) if not e]
    report[ds] = {"referenced": len(paths), "missing": len(missing), "examples": missing[:5]}
    bad += len(missing)
    print(f"{ds:22s} referenced={len(paths):8d} missing={len(missing)}", flush=True)

if out:
    Path(out).write_text(json.dumps(report, indent=2) + "\n")
print("PASS" if bad == 0 else f"FAIL missing={bad}")
sys.exit(1 if bad else 0)
