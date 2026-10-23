"""Nonelectrostatic solvation terms, as an energy-only correction after a PCM SCF.

IEF-PCM is electrostatics only.  Following CMIRS (Pomogaeva & Chipman; equations
as collected in Herbert, arXiv:2203.06846, eqs. 4.18-4.24), the rest of the
solvation free energy is modelled as

    G = A * dispersion + B * exchange + short-range field term

with, for a cavity Omega, solute density rho and uniform solvent density rho_s:

  dispersion   int_{r in Omega} rho(r) I(r) / (w(r) (w(r) + sqrt(rho_s))) dr
               I(r) = int_{r' outside Omega} dr' / (|r - r'|^6 + delta^6)
               w(r) = sqrt(rho + (3C/4pi) |grad rho|^4 / rho^4)
  exchange     int_{r outside Omega} |grad rho(r)| dr

CMIRS's third term, FESR, is C |F_min|^g + D F_max^g: a power of the two
extrema of the normal field F on the cavity surface, which can describe one
hydrogen-bond acceptor site and one donor site.  It is replaced here by the
surface integral the extrema are a limit of,

  short range  sum_i A_i (c- |F_i|^n-  if F_i < 0  else  c+ F_i^n+)

so every strong-field site contributes.  F_i is the solute's own normal field,
or optionally the PCM apparent surface charge density recast as a field (see
docs/nonelectrostatic.md for why the field is the default).

The functions return the bare integrals; the linear coefficients (A, B, c-, c+)
and the exponents are applied by `Descriptors.energy`, so they can be fitted
without recomputing anything.  CMIRS's published parameters belong to an
isodensity cavity and do not transfer to the switching/Gaussian sphere cavity
used here.  Nothing is fitted yet.

The inputs (`Solute`) are backend-neutral; `pyscf_nonelectrostatic` fills one
from a converged PySCF mean field.  All quantities are in atomic units.
Scoring can keep each conformer's descriptors with its energies
(`save_descriptors`), which is what a fit that reweights the conformers needs.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from tautomator.dispersion_kernel import kernel_integral
from tautomator.types import Geometry

Array = NDArray[np.float64]
IntArray = NDArray[np.intp]

# Damping length of the dispersion kernel (bohr); CMIRS's fixed value.
DELTA = 7.0

# The gradient-correction constant of the VV10 local frequency.  Herbert's
# review says CMIRS takes it "without modification from VV10" (0.0093); the
# functional CMIRS is named after, VV09, used 0.0089.  The difference is
# absorbed by the fitted prefactor A.
VV_C = 0.0093

# Densities below this are treated as zero when forming |grad rho| / rho.
_RHO_FLOOR = 1e-14

# The exchange integrand's |grad rho| and the surface term's |F| are taken as
# x^2 / sqrt(x^2 + floor^2): smooth through x = 0, where the bare magnitude has a
# kink and its derivative has no limit, and off from |x| by floor^2 / (2|x|).
GRAD_FLOOR = 1e-14  # e / bohr^4
FIELD_FLOOR = 1e-8  # a.u.

FieldSource = Literal["field", "charge"]


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Surface:
    """A discretized PCM cavity surface and the converged apparent surface charges.

    The cavity is the union of the atomic spheres ``atom_radii``.  ``areas`` are
    the quadrature weights of the surface elements; on the switching/Gaussian
    discretization both drivers use, that is Lebedev weight x R^2 x switching
    function, so elements buried near a sphere intersection fade out smoothly.

    ``charge_scale`` is the factor the model's charges carry relative to those a
    conductor would put on the same surface, for a point charge at the centre of
    a sphere: (eps - 1)/eps for IEF-PCM and C-PCM (exact for IEF-PCM there),
    (eps - 1)/(eps + 1/2) for COSMO.  `charge_field` divides it out.
    """

    coords: Array  # (n_elements, 3), bohr
    normals: Array  # (n_elements, 3), outward unit normals
    areas: Array  # (n_elements,), bohr^2
    charges: Array  # (n_elements,), apparent surface charge q_i in e
    atom_coords: Array  # (n_atoms, 3), bohr
    atom_radii: Array  # (n_atoms,), bohr
    element_atoms: IntArray  # (n_elements,), the atom whose sphere each element is on
    eps: float  # dielectric constant the charges were solved with
    charge_scale: float  # the model's charges relative to a conductor's

    def __post_init__(self) -> None:
        n = len(self.areas)
        if self.coords.shape != (n, 3) or self.normals.shape != (n, 3):
            raise ValueError(
                f"coords and normals must have shape ({n}, 3), "
                f"got {self.coords.shape} and {self.normals.shape}"
            )
        if self.charges.shape != (n,) or self.element_atoms.shape != (n,):
            raise ValueError(
                f"charges and element_atoms must have shape ({n},), "
                f"got {self.charges.shape} and {self.element_atoms.shape}"
            )
        if self.atom_coords.shape != (len(self.atom_radii), 3):
            raise ValueError(
                f"atom_coords must have shape ({len(self.atom_radii)}, 3), "
                f"got {self.atom_coords.shape}"
            )


@dataclass(frozen=True)
class DensityGrid:
    """The solute's electron density and its gradient on a quadrature grid.

    The grids come from `cavity_quadrature.cavity_grid`, which covers either the
    inside or the outside of the cavity and ends exactly on its surface.
    """

    coords: Array  # (n_points, 3), bohr
    weights: Array  # (n_points,)
    rho: Array  # (n_points,), electrons / bohr^3
    grad: Array  # (n_points, 3)

    def __post_init__(self) -> None:
        n = len(self.weights)
        if self.coords.shape != (n, 3) or self.grad.shape != (n, 3) or self.rho.shape != (n,):
            raise ValueError(
                f"expected {n} points; got coords {self.coords.shape}, "
                f"rho {self.rho.shape}, grad {self.grad.shape}"
            )


@dataclass(frozen=True)
class Solute:
    """Everything the three terms need from one converged PCM calculation."""

    surface: Surface
    interior: DensityGrid  # the cavity, for the dispersion integral
    exterior: DensityGrid  # the rest of space, for the exchange integral
    # The solute's own electric field (nuclei + electrons, no reaction field)
    # projected on the outward normal at each surface element.  This is the
    # quantity whose extrema are FESR's F_min and F_max.
    normal_field: Array  # (n_elements,), a.u.

    def __post_init__(self) -> None:
        if self.normal_field.shape != self.surface.areas.shape:
            raise ValueError(
                f"normal_field must have shape {self.surface.areas.shape}, "
                f"got {self.normal_field.shape}"
            )


# ---------------------------------------------------------------------------
# The integrands
# ---------------------------------------------------------------------------


def dispersion_density(
    rho: Array, grad_sq: Array, solvent_rho: float, vv_c: float = VV_C
) -> tuple[Array, Array, Array]:
    """rho / (w (w + sqrt(rho_s))) and its derivatives in rho and |grad rho|^2.

    w = sqrt(rho + (3C/4pi) |grad rho|^4 / rho^4) is the VV local frequency (in
    units where the plasma term is just rho) and sqrt(rho_s) is the same for a
    uniform density.  Points with rho at or below _RHO_FLOOR get zeros.
    """
    rho = np.asarray(rho, dtype=np.float64)
    use = rho > _RHO_FLOOR
    r = np.where(use, rho, 1.0)
    s = np.where(use, grad_sq, 0.0)
    c = 3.0 * vv_c / (4.0 * np.pi)
    q = np.sqrt(solvent_rho)
    w = np.sqrt(r + c * s**2 / r**4)
    denom = w * (w + q)
    value = r / denom
    d_w = -r * (2.0 * w + q) / denom**2  # d value / d w
    d_rho = 1.0 / denom + d_w * (1.0 - 4.0 * c * s**2 / r**5) / (2.0 * w)
    d_grad_sq = d_w * c * s / (r**4 * w)
    zero = np.zeros_like(rho)
    return np.where(use, value, zero), np.where(use, d_rho, zero), np.where(use, d_grad_sq, zero)


def exchange_density(grad_sq: Array) -> tuple[Array, Array]:
    """|grad rho| (regularized, see GRAD_FLOOR) and its derivative in |grad rho|^2."""
    s = np.asarray(grad_sq, dtype=np.float64)
    shifted = s + GRAD_FLOOR**2
    root = np.sqrt(shifted)
    return s / root, (s + 2.0 * GRAD_FLOOR**2) / (2.0 * shifted * root)


def field_power(field: Array, exponent: float, sign: Literal[-1, 1]) -> tuple[Array, Array]:
    """|F|^exponent on the elements whose field has the given sign, 0 on the rest,
    and its derivative in F; |F| is regularized (see FIELD_FLOOR)."""
    field = np.asarray(field, dtype=np.float64)
    shifted = field**2 + FIELD_FLOOR**2
    root = np.sqrt(shifted)
    magnitude = field**2 / root
    d_magnitude = field * (field**2 + 2.0 * FIELD_FLOOR**2) / (shifted * root)
    mask = field * sign > 0
    value = np.where(mask, magnitude**exponent, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = exponent * magnitude ** (exponent - 1.0) * d_magnitude
    return value, np.where(mask, d, 0.0)


# ---------------------------------------------------------------------------
# The integrals
# ---------------------------------------------------------------------------


def interior_kernel_integral(points: Array, surface: Surface, delta: float = DELTA) -> Array:
    """J(r) = int_{r' in Omega} dr' / (|r - r'|^6 + delta^6) for each point r.

    The volume integral is done on the cavity surface.  1/(R^6 + delta^6) is the
    divergence of g(R) R_hat with g = arctan(R^3/delta^3) / (3 delta^3 R^2), so
    by the divergence theorem

        J(r) = sum_i A_i (R_i . n_i) arctan(t_i) / (3 delta^3 R_i^3),
        R_i = s_i - r,  t_i = R_i^3 / delta^3.

    g vanishes like R at R = 0, so unlike the usual double-layer kernel this has
    no singularity as r approaches the surface and holds for r inside or outside
    the cavity alike.  Written with arctan(t)/t it needs no division by R.  The
    pair sum is compiled (`dispersion_kernel`).
    """
    return kernel_integral(points, surface.coords, surface.normals, surface.areas, delta)


def whole_space_kernel_integral(delta: float = DELTA) -> float:
    """int dr' / (|r - r'|^6 + delta^6) over all of space: 2 pi^2 / (3 delta^3)."""
    return float(2.0 * np.pi**2 / (3.0 * delta**3))


def exterior_kernel_integral(points: Array, surface: Surface, delta: float = DELTA) -> Array:
    """I(r) = int_{r' outside Omega} dr' / (|r - r'|^6 + delta^6).

    All of space (`whole_space_kernel_integral`) less the cavity's share.
    """
    return whole_space_kernel_integral(delta) - interior_kernel_integral(points, surface, delta)


def dispersion_integral(
    interior: DensityGrid,
    surface: Surface,
    solvent_rho: float,
    delta: float = DELTA,
    vv_c: float = VV_C,
) -> float:
    """The CMIRS dispersion integral, without its prefactor A.

    The solute density inside the cavity interacts with uniform solvent outside
    it through the damped R^-6 kernel; w is the VV local frequency (in units
    where the plasma term is just rho) and sqrt(solvent_rho) is the same thing
    for a uniform density.  The density factor rho / (w (w + sqrt(rho_s)))
    saturates at 1 where rho >> rho_s, so the nuclear cusps do not show up in
    the integrand.  ``solvent_rho`` is the liquid's mean electron density in
    e / bohr^3 (`config.solvent_electron_density`).
    """
    factor, _, _ = dispersion_density(
        interior.rho, np.sum(interior.grad**2, axis=1), solvent_rho, vv_c
    )
    kernel = exterior_kernel_integral(interior.coords, surface, delta)
    return float(interior.weights @ (factor * kernel))


def exchange_integral(exterior: DensityGrid) -> float:
    """int |grad rho| over the region outside the cavity, without its prefactor B
    (|grad rho| regularized as in `exchange_density`)."""
    value, _ = exchange_density(np.sum(exterior.grad**2, axis=1))
    return float(exterior.weights @ value)


def electrons_outside(exterior: DensityGrid) -> float:
    """Number of solute electrons outside the cavity (the "escaped charge")."""
    return float(exterior.weights @ exterior.rho)


def charge_field(surface: Surface) -> Array:
    """The apparent surface charge density recast as a normal field.

    A conductor's surface charge density is sigma = -E_n / (4 pi), with E_n the
    total normal field just inside the cavity (the solute's plus the surface
    charges' own), and a dielectric model scales it by ``charge_scale``
    ((eps - 1)/eps at a true dielectric boundary).  Inverting that puts q_i / A_i
    on the same scale and sign convention as the solute's normal field, to which
    it reduces exactly for a charge at the centre of a sphere.
    """
    return -4.0 * np.pi / surface.charge_scale * surface.charges / surface.areas


def surface_power_integral(
    field: Array, areas: Array, exponent: float, sign: Literal[-1, 1]
) -> float:
    """sum_i A_i |F_i|^exponent over the elements whose field has the given sign.

    The power is on the field alone and the area enters once, so the sum
    converges to a surface integral as the discretization is refined.  |F| is
    regularized as in `field_power`.
    """
    value, _ = field_power(field, exponent, sign)
    return float(areas @ value)


# ---------------------------------------------------------------------------
# Parameters and descriptors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Params:
    """Fitted parameters of the nonelectrostatic model (atomic units)."""

    dispersion: float  # A
    exchange: float  # B
    neg_coeff: float  # c-, on elements with negative field (hydrogen-bond acceptors)
    neg_exponent: float  # n-
    pos_coeff: float  # c+, on elements with positive field (hydrogen-bond donors)
    pos_exponent: float  # n+


@dataclass(frozen=True)
class Energies:
    """The three contributions, in hartree."""

    dispersion: float
    exchange: float
    short_range: float

    @property
    def total(self) -> float:
        return self.dispersion + self.exchange + self.short_range


@dataclass(frozen=True)
class Descriptors:
    """Parameter-free integrals for one solute; all a fit needs to keep."""

    dispersion: float  # G_disp / A
    exchange: float  # G_exch / B
    areas: Array  # (n_elements,)
    normal_field: Array  # solute's normal field at each element
    charge_field: Array  # the same from the apparent surface charges
    electrons_outside: float
    element_atoms: IntArray  # the atom whose sphere each element is on

    def field(self, source: FieldSource = "field") -> Array:
        return self.normal_field if source == "field" else self.charge_field

    def field_extrema(self, source: FieldSource = "field") -> tuple[float, float]:
        """(F_min, F_max): the two numbers the original FESR term is built on."""
        f = self.field(source)
        return float(f.min()), float(f.max())

    def surface_integral(
        self, exponent: float, sign: Literal[-1, 1], source: FieldSource = "field"
    ) -> float:
        return surface_power_integral(self.field(source), self.areas, exponent, sign)

    def energy(self, params: Params, source: FieldSource = "field") -> Energies:
        short_range = params.neg_coeff * self.surface_integral(
            params.neg_exponent, -1, source
        ) + params.pos_coeff * self.surface_integral(params.pos_exponent, 1, source)
        return Energies(
            dispersion=params.dispersion * self.dispersion,
            exchange=params.exchange * self.exchange,
            short_range=short_range,
        )


def descriptors(
    solute: Solute, solvent_rho: float, delta: float = DELTA, vv_c: float = VV_C
) -> Descriptors:
    """Evaluate every parameter-free integral for one solute in one solvent.

    ``solvent_rho`` (e / bohr^3) enters the dispersion integral only; take it
    from `config.solvent_electron_density`.
    """
    return Descriptors(
        dispersion=dispersion_integral(solute.interior, solute.surface, solvent_rho, delta, vv_c),
        exchange=exchange_integral(solute.exterior),
        areas=solute.surface.areas,
        normal_field=solute.normal_field,
        charge_field=charge_field(solute.surface),
        electrons_outside=electrons_outside(solute.exterior),
        element_atoms=solute.surface.element_atoms,
    )


# ---------------------------------------------------------------------------
# Saved descriptors
# ---------------------------------------------------------------------------

# What scoring writes into a run's output directory when run_pipeline is asked for
# descriptors: this for the run's own solvent model, and `descriptor_file(model)` for
# each further model.
DESCRIPTOR_FILE = "nonelectrostatic.npz"
_FILE_VERSION = 1


def descriptor_file(solvent_model: str | None = None) -> str:
    """The file scoring saves a solvent model's descriptors to: DESCRIPTOR_FILE for the
    run's own model (None), nonelectrostatic_<model>.npz for a further one, the name
    normalized (upper case, no hyphens, then lower case): "IEFPCM/rho-DROP" ->
    nonelectrostatic_iefpcm_rhodrop.npz."""
    if solvent_model is None:
        return DESCRIPTOR_FILE
    slug = solvent_model.upper().replace("-", "").replace("/", "_").lower()
    return f"nonelectrostatic_{slug}.npz"


@dataclass(frozen=True)
class ScoredConformer:
    """A conformer scored in solution: its energies as scoring left them, and its descriptors.

    A fit adds `Descriptors.energy` to ``free_energy`` and reweights, the
    conformer standing for ``states`` states as in `ensemble.assign_weights`.
    """

    charge: int
    microstate: str  # Microstate.tautomer_id
    includes_enantiomer: bool  # Microstate.includes_enantiomer
    geometry: Geometry
    electronic_energy: float  # gas phase, at the scoring level (hartree)
    solvation_energy: float  # E(PCM) - E(gas), electrostatics only (hartree)
    rrho_correction: float | None  # hartree
    multiplicity: float  # Conformer.multiplicity
    descriptors: Descriptors

    @property
    def free_energy(self) -> float:
        """The conformer's free energy without the correction (hartree)."""
        return self.electronic_energy + self.solvation_energy + (self.rrho_correction or 0.0)

    @property
    def states(self) -> float:
        """How many physical states this conformer contributes."""
        return (2.0 if self.includes_enantiomer else 1.0) * self.multiplicity


def save_descriptors(
    path: Path, conformers: Sequence[ScoredConformer], settings: dict[str, Any]
) -> None:
    """Write scored conformers to one .npz file, for `load_descriptors`.

    The per-atom and per-element arrays of all conformers are concatenated,
    with each conformer's counts alongside, so the file holds plain arrays and
    loads without pickle.  ``settings`` (anything JSON can hold) records how the
    descriptors were computed.  The file is replaced in one step, so a reader
    finds the old file or the complete new one.
    """

    def joined(parts: list[NDArray[Any]], tail: tuple[int, ...], dtype: Any) -> NDArray[Any]:
        return np.concatenate(parts) if parts else np.empty((0, *tail), dtype=dtype)

    d = [c.descriptors for c in conformers]
    rrho = [np.nan if c.rrho_correction is None else c.rrho_correction for c in conformers]
    arrays: dict[str, Any] = {
        "settings": np.array(json.dumps({"version": _FILE_VERSION, **settings})),
        "charge": np.array([c.charge for c in conformers], dtype=np.int64),
        "microstate": np.array([c.microstate for c in conformers], dtype=str),
        "includes_enantiomer": np.array([c.includes_enantiomer for c in conformers], dtype=bool),
        "electronic_energy": np.array([c.electronic_energy for c in conformers], dtype=float),
        "solvation_energy": np.array([c.solvation_energy for c in conformers], dtype=float),
        "rrho_correction": np.array(rrho, dtype=float),
        "multiplicity": np.array([c.multiplicity for c in conformers], dtype=float),
        "dispersion": np.array([x.dispersion for x in d], dtype=float),
        "exchange": np.array([x.exchange for x in d], dtype=float),
        "electrons_outside": np.array([x.electrons_outside for x in d], dtype=float),
        "n_atoms": np.array([c.geometry.n_atoms for c in conformers], dtype=np.int64),
        "symbols": np.array([s for c in conformers for s in c.geometry.symbols], dtype=str),
        "coords": joined([c.geometry.coords for c in conformers], (3,), float),
        "n_elements": np.array([len(x.areas) for x in d], dtype=np.int64),
        "areas": joined([x.areas for x in d], (), float),
        "normal_field": joined([x.normal_field for x in d], (), float),
        "charge_field": joined([x.charge_field for x in d], (), float),
        "element_atoms": joined([x.element_atoms for x in d], (), np.intp),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as f:
        np.savez(f, **arrays)
    partial.replace(path)


def load_descriptors(path: Path) -> tuple[list[ScoredConformer], dict[str, Any]]:
    """Read a file `save_descriptors` wrote: (conformers, settings)."""
    with np.load(path, allow_pickle=False) as f:
        settings: dict[str, Any] = json.loads(str(f["settings"]))
        if settings.get("version") != _FILE_VERSION:
            raise ValueError(
                f"{path} is version {settings.get('version')}; this build reads {_FILE_VERSION}"
            )
        a = {key: f[key] for key in f.files}
    n = len(a["charge"])
    if n == 0:
        return [], settings

    def split(key: str, counts: str) -> list[NDArray[Any]]:
        return np.split(a[key], np.cumsum(a[counts])[:-1])

    symbols, coords = split("symbols", "n_atoms"), split("coords", "n_atoms")
    areas, normal, from_charges, atoms = (
        split(key, "n_elements")
        for key in ("areas", "normal_field", "charge_field", "element_atoms")
    )
    conformers = [
        ScoredConformer(
            charge=int(a["charge"][i]),
            microstate=str(a["microstate"][i]),
            includes_enantiomer=bool(a["includes_enantiomer"][i]),
            geometry=Geometry(tuple(str(s) for s in symbols[i]), coords[i]),
            electronic_energy=float(a["electronic_energy"][i]),
            solvation_energy=float(a["solvation_energy"][i]),
            rrho_correction=(
                None if np.isnan(a["rrho_correction"][i]) else float(a["rrho_correction"][i])
            ),
            multiplicity=float(a["multiplicity"][i]),
            descriptors=Descriptors(
                dispersion=float(a["dispersion"][i]),
                exchange=float(a["exchange"][i]),
                areas=areas[i],
                normal_field=normal[i],
                charge_field=from_charges[i],
                electrons_outside=float(a["electrons_outside"][i]),
                element_atoms=atoms[i],
            ),
        )
        for i in range(n)
    ]
    return conformers, settings
