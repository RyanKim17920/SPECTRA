#!/usr/bin/env python3
"""Check that an eval cell's ``BACKBONES`` table still agrees with the registry.

WHY A CHECKER AND NOT AN IMPORT.  The per-checkpoint eval cells (``$SPECTRA_CELLS/<name>/
model.py``) are executed OUTSIDE this repo, by four different vendored harnesses in four
different virtualenvs, none of which has ``waivphaet`` installed -- and each cell's
``run_manifest.py`` HASHES its own ``model.py`` to bind results to code.  Making a cell
import this package would therefore (a) not work where the cells actually run and
(b) invalidate every existing manifest the moment the import line was added, i.e. throw
away the provenance of results already on disk.

So the cells stay standalone copies, and :mod:`waivphaet.models.backbones` is the SOURCE:
this script proves a copy has not drifted from it.  Run it after editing the registry, or
before trusting a cell you did not write::

    ./.venv/bin/python scripts/check_backbone_registry.py                 # all cells
    ./.venv/bin/python scripts/check_backbone_registry.py <cell-dir> ...  # named cells

Exit status is 0 when every cell agrees, 1 on any mismatch, 2 when no cell could be read.

``weights_file`` is deliberately NOT compared: it names a file inside a local weight
directory, and this repo and the cells serve some backbones from DIFFERENT stores --
Virchow v1 is ``model.safetensors`` in the pinned hub snapshot we bind and
``pytorch_model.bin`` in the cells' converted copy.  Both are right about their own
directory, so an equality check there reports a difference that is not drift.

A field is compared only when BOTH sides state it.  That is not laxity: a cell for a
hub-served backbone deliberately carries ``mean``/``std`` resolved from the repo's own
``pretrained_cfg`` where the registry carries ``None`` ("no override, ask the repo"), and
those two are agreement, not conflict.  What the check does catch is the failure that
actually happened: a value stated in both places that stopped matching.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import CELLS  # noqa: E402
from waivphaet.models.backbones import BACKBONES  # noqa: E402

#: cell key -> (registry attribute, human name). Only these are compared.
FIELDS = [
    ("loader", "loader"),
    ("architecture", "architecture"),
    ("timm_kwargs", "timm_kwargs"),
    ("readout", "thunder_readout"),
    ("embed_dim", "embed_dim"),
    ("num_prefix_tokens", "num_prefix_tokens"),
    ("num_blocks", "num_blocks"),
    ("patch_size", "patch_size"),
    ("lora_target_suffixes", "lora_target_suffixes"),
]


def _literal(node: ast.AST):
    """``ast.literal_eval`` plus arithmetic, so ``2.66667 * 2`` parses like the cell means it.

    The cells write ``mlp_ratio`` as the model card writes it -- a product, not its value --
    and rewriting that to 5.33334 in either place to please a parser would be the exact
    kind of silent transcription this check exists to prevent.
    """
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Add, ast.Div, ast.Sub)):
        a, b = _literal(node.left), _literal(node.right)
        return {ast.Mult: lambda: a * b, ast.Add: lambda: a + b,
                ast.Div: lambda: a / b, ast.Sub: lambda: a - b}[type(node.op)]()
    return ast.literal_eval(node)


def cell_table(model_py: Path) -> dict[str, dict]:
    """The cell's ``BACKBONES`` dict, read off the UNEXECUTED source.

    Parsing rather than importing is deliberate: importing a cell pulls in timm, torch and
    the cell's own asset assertions, which fail on any machine that does not hold that
    checkpoint -- and this check must work on a laptop.
    """
    tree = ast.parse(model_py.read_text())
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "BACKBONES" for t in node.targets):
            continue
        out = {}
        for k, v in zip(node.value.keys, node.value.values):
            row = {}
            for kk, vv in zip(v.keys, v.values):
                try:
                    row[_literal(kk)] = _literal(vv)
                except (ValueError, TypeError, SyntaxError, KeyError):
                    continue    # a value we cannot read is not a mismatch
            out[_literal(k)] = row
        return out
    return {}


def norm_pair(row: dict):
    """``(mean, std)`` out of a cell row, or ``None`` if it does not state both."""
    if "mean" in row and "std" in row:
        return (tuple(row["mean"]), tuple(row["std"]))
    return None


def check_cell(model_py: Path) -> list[str]:
    table = cell_table(model_py)
    if not table:
        return [f"{model_py}: no BACKBONES table found"]
    bad = []
    for repo_id, row in table.items():
        bb = BACKBONES.get(repo_id)
        if bb is None:
            bad.append(f"{model_py.parent.name}: {repo_id} is in the cell but NOT in the "
                       f"registry -- add it to src/waivphaet/models/backbones.py")
            continue
        for cell_key, attr in FIELDS:
            if cell_key not in row:
                continue
            mine = getattr(bb, attr)
            if mine in (None, (), {}):
                continue
            theirs = row[cell_key]
            if isinstance(mine, tuple) and isinstance(theirs, (list, tuple)):
                theirs = tuple(theirs)
            if mine != theirs:
                bad.append(f"{model_py.parent.name}: {repo_id}.{cell_key}\n"
                           f"    registry: {mine!r}\n    cell:     {theirs!r}")
        theirs_norm = norm_pair(row)
        if bb.normalization and theirs_norm and bb.normalization != theirs_norm:
            bad.append(f"{model_py.parent.name}: {repo_id}.normalization\n"
                       f"    registry: {bb.normalization!r}\n    cell:     {theirs_norm!r}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cells", nargs="*", type=Path,
                    help=f"cell directories or model.py files (default: every cell under {CELLS})")
    args = ap.parse_args()

    targets: list[Path] = []
    for c in args.cells:
        targets.append(c if c.is_file() else c / "model.py")
    if not targets:
        if not CELLS.is_dir():
            print(f"no cell root at {CELLS}; set SPECTRA_CELLS or name cells explicitly",
                  file=sys.stderr)
            return 2
        targets = sorted(CELLS.glob("*/model.py"))
    targets = [t for t in targets if t.is_file()]
    if not targets:
        print("no cell model.py found", file=sys.stderr)
        return 2

    problems = []
    for t in targets:
        problems += check_cell(t)
    for p in problems:
        print("MISMATCH", p)
    print(f"checked {len(targets)} cells against {len(BACKBONES)} registry entries: "
          f"{len(problems)} mismatches")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
