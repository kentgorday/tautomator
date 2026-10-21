"""The molecule and parameters the nonelectrostatic gradient and SCF tests share."""

from __future__ import annotations

import numpy as np

from tautomator.config import solvent_electron_density
from tautomator.nonelectrostatic import Params
from tautomator.types import Geometry

BOHR = 1.8897261246  # bohr per Angstrom
WATER = solvent_electron_density("water")

# Formate, off its symmetric geometry so that no gradient component vanishes.
FORMATE = Geometry(
    symbols=("C", "O", "O", "H"),
    coords=np.array(
        [[0.02, -0.03, 0.53], [1.15, -0.03, -0.04], [-1.11, -0.03, -0.04], [0.09, 0.01, 1.67]]
    ),
)
# Magnitudes of the order a fit should give: a few kcal/mol for formate.
PARAMS = Params(
    dispersion=-0.004,
    exchange=0.04,
    neg_coeff=-0.5,
    neg_exponent=2.5,
    pos_coeff=-0.2,
    pos_exponent=2.0,
)


def displaced(geom: Geometry, atom: int, axis: int, step: float) -> Geometry:
    """geom with one coordinate moved by step bohr."""
    coords = geom.coords.copy()
    coords[atom, axis] += step / BOHR
    return Geometry(geom.symbols, coords)
