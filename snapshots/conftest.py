"""Keep pytest out of the vendored snapshots.

Everything under ``snapshots/`` is FROZEN CODE kept for provenance: the exact bytes that
produced published results. It is not maintained, it is not on the import path of a normal
session, and it must never be collected as part of this repo's test suite.

Left to itself pytest does collect it, and the damage is not a few extra tests:

* ``snapshots/*/tests/test_invariants.py`` has the same basename as ``tests/test_invariants.py``
  and there is no ``__init__.py`` anywhere, so both want the module name ``test_invariants``.
  Whichever is imported first wins and the other is reported as ``import file mismatch`` --
  i.e. vendoring a second snapshot silently turns the REAL suite into a collection ERROR.
  The same collision exists for ``test_resume.py``, ``test_blocked_loss.py`` and
  ``scripts/smoke_test.py``.
* Even without the clash, those copies assert against their own snapshot's source, so a
  green result would say nothing about the working tree and a red one would be unfixable.

``"*"`` is matched against each candidate's basename, so this ignores files and
subdirectories alike -- the whole subtree, and any snapshot vendored here later.

To actually run a snapshot's own tests, point pytest inside it with its own rootdir:
``pytest --rootdir snapshots/falseneg-gated snapshots/falseneg-gated/tests``.
"""

collect_ignore_glob = ["*"]
