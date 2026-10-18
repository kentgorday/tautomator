"""Run directories for batch runs over a CSV of molecules.

A batch run keeps each molecule's output in a directory named for its
``unique_ID`` alone, so results stay attached to the molecule when the CSV is
regenerated and its rows move. Runs made before that carry a row-number prefix
(``003_serjeant2550``); `find_compound_dir` still finds them.
"""

from __future__ import annotations

import re
from pathlib import Path

_ROW_PREFIX = re.compile(r"^\d{3}_")


def compound_dir_name(unique_id: str) -> str:
    """Filesystem-safe directory name for a molecule's ``unique_ID``."""
    return "".join(c if c.isalnum() else "_" for c in unique_id)[:50]


def compound_id(dir_name: str) -> str:
    """The ID a run directory is named for, without any row-number prefix.

    An ID that itself begins with three digits and an underscore cannot be told
    apart from a prefixed one; go from ID to directory with `find_compound_dir`
    where that matters.
    """
    return _ROW_PREFIX.sub("", dir_name)


def find_compound_dir(output_root: Path, unique_id: str) -> Path:
    """Where a molecule's output lives under ``output_root``, or should be created.

    Prefers the directory named for the ID, then one from an older run that
    prefixed the row number, whatever that number was.
    """
    name = compound_dir_name(unique_id)
    plain = output_root / name
    if plain.is_dir():
        return plain
    prefixed = sorted(output_root.glob(f"[0-9][0-9][0-9]_{name}"))
    if len(prefixed) > 1:
        found = ", ".join(p.name for p in prefixed)
        raise ValueError(f"more than one run directory for {unique_id!r}: {found}")
    return prefixed[0] if prefixed else plain
