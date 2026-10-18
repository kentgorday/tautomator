"""Reading the pKa set's CSV: which transition each experimental pKa measures.

The `pkas` column mixes two label conventions, and which transition a value
refers to is carried by the *label*, never by the column order or by
`acid_base`:

* ``pKa{n}``  is the n-th proton lost starting from the neutral molecule, so
  ``pKa1`` is q=0 -> q=-1 and ``pKa2`` is q=-1 -> q=-2.
* ``pKaH{n}`` is the n-th proton lost starting from the fully protonated form,
  so ``pKaH1`` is q=+1 -> q=0 and ``pKaH2`` is q=+2 -> q=+1.

Two of the forty entries in the pKa set mix them -- ``perrin3294``
(``pKa1=9.208;pKaH1=2.186``) and ``perrin3287``
(``pKa1=10.640;pKaH1=1.952``) -- and for those, pairing values with transitions
by position assigns a q=0 -> q=-1 measurement to the q=+1 -> q=0 transition and
reports a number that means nothing. Parse the label.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_LABEL = re.compile(r"^pKa(H?)(\d+)$")


@dataclass(frozen=True)
class PkaValue:
    """One experimental pKa, with the charge-state transition it measures."""

    label: str
    value: float
    higher_charge: int
    """The protonated side of the equilibrium."""
    lower_charge: int
    """The deprotonated side; always ``higher_charge - 1``."""


def parse_pka_column(column: str) -> list[PkaValue]:
    """Parse a ``pkas`` cell into values tagged with their transitions.

    Raises on an unrecognised label rather than guessing, since a silent
    mis-assignment is indistinguishable from a bad prediction downstream.
    """
    out: list[PkaValue] = []
    for field in column.split(";"):
        field = field.strip()
        if not field:
            continue
        label, _, raw = field.partition("=")
        match = _LABEL.match(label.strip())
        if match is None or not raw:
            raise ValueError(f"unrecognised pKa label: {field!r}")
        protonated, index = match.group(1) == "H", int(match.group(2))
        if index < 1:
            raise ValueError(f"pKa index must start at 1: {field!r}")
        # pKaH{n}: +n -> +(n-1).   pKa{n}: -(n-1) -> -n.
        higher = index if protonated else -(index - 1)
        out.append(PkaValue(label.strip(), float(raw), higher, higher - 1))
    return out
