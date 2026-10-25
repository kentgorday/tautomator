"""Refinement with the OrbMol-v2 machine-learned potential (orb-models).

OrbMol-v2 is a conservative graph network (forces and Hessians are exact
derivatives of its energy) trained on OMol25 + OPoly26: gas-phase
wB97M-V/def2-TZVPD total energies and forces.  It takes the total charge and
spin multiplicity as inputs and adds a long-range Coulomb term over learned
per-atom charges.  It has no implicit solvent, so it is a refinement-only
driver: config.load_config refuses it for scoring and refuses a refinement
solvent model.  Every energy here is a gas-phase energy in Hartree, on the same
absolute scale as wB97M-V/def2-TZVPD.

The functions mirror the psi4/pyscf runners so refinement.py can call any
driver the same way.  ``method`` selects the model (see MODELS); ``basis`` and
the solvent arguments exist only for that shared signature.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import uuid
import warnings
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tautomator.config import DEFAULT_MEMORY_GB
from tautomator.geometric_logging import geometric_quiet
from tautomator.types import Geometry

log = logging.getLogger(__name__)

HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANG = 0.529177210903


@dataclass(frozen=True)
class _Model:
    loader: str  # constructor name in orb_models.forcefield.pretrained
    url: str
    sha256: str


# Weights are fetched on first use and cached by cached_path under
# ~/.cache/cached_path, and read from there without a network call afterwards
# (see _weights_path); set TAUTOMATOR_ORBMOL_WEIGHTS to use a copy from elsewhere.
# Hugging Face serves the same file orb-models' default S3 URL does (identical
# sha256) but, in testing, ~1000x faster.  The checksum is verified on every
# load, so a changed or truncated file cannot silently change results.
MODELS: dict[str, _Model] = {
    "orbmol-v2": _Model(
        loader="orbmol_v2",
        url=(
            "https://huggingface.co/Orbital-Materials/orbmol-v2/resolve/main/"
            "orbmol-v2-teqabfhg-20260523.ckpt"
        ),
        sha256="3dbef70b5fdc9124392181b0d8686c79f1c02bf5fefcb878a4eaa51d03e4f5e8",
    ),
}

# orb-models' own default.  This was float64 on the theory that geomeTRIC's
# energy-based step control and the finite-difference Hessian both need a smooth
# surface; measured on 14 refinement inputs (4-18 atoms, across charge states),
# that costs 1.8x and buys nothing.  float32-high needed 3% more gradient calls
# to converge -- so the surface is not meaningfully rougher -- and against
# float64 gave: energies within 0.042 kcal/mol, geometries within 0.02 A (median
# 0.0000), frequencies within 3.4 cm^-1, no spurious imaginary modes (0 either
# way), and quasi-RRHO free energies within 0.056 kcal/mol, worth <=0.009 pKa on
# any measured deprotonation.  The largest deviations are on the flattest
# surfaces, so re-check this if a very flexible molecule starts failing to
# converge.  orb-models sets torch's *global* default dtype when it loads a
# model, so one precision per process is all that is safe regardless.
_PRECISION = "float32-high"

# Central-difference step for the Hessian (Angstrom).
_HESSIAN_STEP = 5e-3

# Steps per coordinate system before giving up.  The DFT drivers stop at 100
# because every step there is an SCF; a step here is one ~0.15 s gradient, so a
# ceiling that low is inherited frugality rather than a real budget.  500 costs
# at most ~75 s per phase on a 20-atom conformer -- far less than the wasted
# solvation single point and Hessian that an unconverged geometry goes on to
# buy, and less than the one geometry per ~400 that was being kept unconverged.
_MAXITER = 500


def _model_spec(method: str) -> _Model:
    spec = MODELS.get(method.lower())
    if spec is None:
        raise ValueError(f"Unknown OrbMol model {method!r} (known: {', '.join(MODELS)})")
    return spec


def _verified_weights(path: str, sha256: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != sha256:
        raise RuntimeError(
            f"OrbMol weights at {path} have sha256 {digest.hexdigest()}, expected "
            f"{sha256}; the file is corrupt or not the model this code was validated with."
        )
    return path


def _weights_path(spec: _Model) -> str:
    """Return a local path to the weights, going to the network only for a first download."""
    if override := os.environ.get("TAUTOMATOR_ORBMOL_WEIGHTS"):
        return override

    from cached_path import cached_path, find_latest_cached

    # cached_path() asks the server for the file's ETag on every call, cached
    # copy or not, so every process start depended on Hugging Face answering:
    # a 429 from its rate limiter failed the load with the weights on disk.
    # The sha256 pin already says exactly which file is wanted, so a cached copy
    # needs no revalidation.
    cached = find_latest_cached(spec.url)
    return str(cached if cached is not None else cached_path(spec.url))


@cache
def _load(method: str) -> tuple[Any, Any]:
    """Load a model on CPU, once per process.

    CPU only: orb-models 0.7.0 cannot target MPS (its architecture builder calls
    ``.cuda()`` for any non-CPU device, and the Warp neighbour list has no MPS
    backend), and at ~20 atoms CPU was faster than MPS with workarounds anyway.
    ``torch.compile`` is off: it buys nothing at this size.
    """
    spec = _model_spec(method)

    import warp

    # No initialisation banner on stdout: warp 1.18 logs it at LOG_INFO, and earlier
    # releases (1.17 on linux-64) print it unless config.quiet.  Set before importing
    # orb_models, which can initialise warp.
    if hasattr(warp, "LOG_WARNING"):
        warp.config.log_level = warp.LOG_WARNING
    else:
        setattr(warp.config, "quiet", True)  # noqa: B010 (not in 1.18's stubs)
    from orb_models.forcefield import pretrained

    weights = _verified_weights(_weights_path(spec), spec.sha256)
    log.info(f"Loading {method} from {weights}")
    with warnings.catch_warnings():
        # Expected: orb-models announces the global dtype it sets (see _PRECISION).
        warnings.filterwarnings("ignore", message="Setting global torch default dtype")
        model, adapter = getattr(pretrained, spec.loader)(
            weights_path=weights, device="cpu", precision=_PRECISION, compile=False
        )
    return model, adapter


def preload(method: str) -> None:
    """Load the model now, so that failing to get it stops the run.

    refine() wraps each conformer in a handler that records the failure and
    moves on -- right for an SCF that will not converge, wrong for a model that
    cannot be loaded, which is no conformer's fault.  Left to the first
    optimization, a failed load excluded every conformer in turn as
    "optimization_failed" and the run finished, exit 0, with an empty ensemble.
    """
    _load(method.lower())


def _energy_gradient(
    geom: Geometry, charge: int, method: str, threads: int = 1
) -> tuple[float, NDArray[np.float64]]:
    """Return (energy in Hartree, gradient in Hartree/Bohr with shape (n, 3))."""
    import torch
    from ase import Atoms

    torch.set_num_threads(threads)
    model, adapter = _load(method.lower())
    atoms = Atoms(symbols=list(geom.symbols), positions=np.asarray(geom.coords))
    atoms.info["charge"] = int(charge)
    atoms.info["spin"] = int(geom.multiplicity(charge))
    out = model.predict(adapter.from_ase_atoms(atoms, device="cpu"), split=False)
    energy_ev = float(out["energy"].detach().double().reshape(-1)[0])
    forces_ev_ang = out["grad_forces"].detach().double().cpu().numpy().reshape(-1, 3)
    gradient = -forces_ev_ang * BOHR_TO_ANG / HARTREE_TO_EV
    return energy_ev / HARTREE_TO_EV, gradient


def _refuse_solvent(cavity: str | None) -> None:
    if cavity is not None:
        raise ValueError(
            "OrbMol-v2 is a gas-phase potential with no implicit solvent; "
            f"cavity={cavity!r} cannot be honoured."
        )


def single_point(
    geom: Geometry,
    charge: int,
    method: str,
    basis: str = "",
    cavity: str | None = None,
    solvent: str | None = None,
    pcm_hydrogen_radius: float = 1.1,
    threads: int = 1,
    memory_gb: float = DEFAULT_MEMORY_GB,
) -> float:
    """Gas-phase energy in Hartree."""
    _refuse_solvent(cavity)
    return _energy_gradient(geom, charge, method, threads=threads)[0]


def optimize(
    geom: Geometry,
    charge: int,
    method: str,
    basis: str = "",
    cavity: str | None = None,
    solvent: str | None = None,
    pcm_hydrogen_radius: float = 1.1,
    threads: int = 1,
    memory_gb: float = DEFAULT_MEMORY_GB,
) -> tuple[Geometry, float, bool]:
    """Gas-phase geometry optimization with geomeTRIC.

    Same strategy and convergence criteria as the pyscf driver -- TRIC internal
    coordinates, then Cartesian coordinates from the last geometry -- but with a
    far higher step ceiling (see _MAXITER).  Returns (geometry, energy_hartree,
    converged); the geometry and energy are the last step's whether or not it
    converged.
    """
    _refuse_solvent(cavity)
    if geom.n_atoms == 1:
        # An atom has no internal coordinates: its minimum is itself.
        energy = single_point(geom, charge, method, basis, threads=threads, memory_gb=memory_gb)
        return geom, energy, True

    import geometric.engine
    import geometric.molecule
    import geometric.optimize
    from geometric.errors import GeomOptNotConvergedError

    class _Engine(geometric.engine.Engine):  # type: ignore[misc]
        def __init__(self, start: Geometry) -> None:
            m = geometric.molecule.Molecule()
            m.elem = list(start.symbols)
            m.xyzs = [np.asarray(start.coords, dtype=float)]
            super().__init__(m)
            self.last: tuple[NDArray[np.float64], float] | None = None

        def calc_new(self, coords: NDArray[np.float64], dirname: str) -> dict[str, Any]:
            xyz = coords.reshape(-1, 3) * BOHR_TO_ANG
            e, g = _energy_gradient(
                Geometry(symbols=geom.symbols, coords=xyz), charge, method, threads=threads
            )
            self.last = (xyz, e)
            return {"energy": e, "gradient": g.ravel()}

    def _run(start: Geometry, coordsys: str) -> tuple[Geometry, float, bool]:
        engine = _Engine(start)
        with tempfile.TemporaryDirectory(prefix="orbmol_opt_") as tmpdir, geometric_quiet():
            try:
                progress = geometric.optimize.run_optimizer(
                    customengine=engine,
                    input=os.path.join(tmpdir, str(uuid.uuid4())),
                    coordsys=coordsys,
                    maxiter=_MAXITER,
                )
                xyz = np.asarray(progress.xyzs[-1], dtype=float)
                energy = float(progress.qm_energies[-1])
                converged = True
            except GeomOptNotConvergedError:
                assert engine.last is not None
                xyz, energy = engine.last
                converged = False
        return Geometry(symbols=start.symbols, coords=xyz), energy, converged

    opt_geom, energy, converged = _run(geom, "tric")
    if converged:
        return opt_geom, energy, True
    log.info(
        f"Internal-coord opt did not converge in {_MAXITER} steps; "
        "retrying from last geometry with Cartesian coords"
    )
    return _run(opt_geom, "cart")


def frequencies(
    geom: Geometry,
    charge: int,
    method: str,
    basis: str = "",
    cavity: str | None = None,
    solvent: str | None = None,
    pcm_hydrogen_radius: float = 1.1,
    threads: int = 1,
    memory_gb: float = DEFAULT_MEMORY_GB,
) -> list[float]:
    """Harmonic frequencies in cm^-1 (imaginary ones negative), translations and
    rotations projected out.

    The Hessian is a central difference of the model's analytic gradients (6N
    gradient calls).  The harmonic analysis is PySCF's -- the one the pyscf
    driver uses -- so masses and projection match that driver exactly.  An empty
    list for an atom.
    """
    _refuse_solvent(cavity)
    if geom.n_atoms == 1:
        return []
    from pyscf import gto
    from pyscf.hessian.thermo import harmonic_analysis

    n = geom.n_atoms
    x0 = np.asarray(geom.coords, dtype=float)
    hess = np.zeros((3 * n, 3 * n))
    for k in range(3 * n):
        dx = np.zeros(3 * n)
        dx[k] = _HESSIAN_STEP
        grad = [
            _energy_gradient(
                Geometry(geom.symbols, x0 + sign * dx.reshape(-1, 3)),
                charge,
                method,
                threads=threads,
            )[1]
            for sign in (1.0, -1.0)
        ]
        hess[k] = (grad[0] - grad[1]).ravel() / (2 * _HESSIAN_STEP / BOHR_TO_ANG)
    hess = 0.5 * (hess + hess.T)

    mol = gto.M(
        atom=[(s, tuple(c)) for s, c in zip(geom.symbols, x0, strict=True)],
        basis="sto-3g",  # unused: harmonic_analysis only needs masses and coordinates
        charge=charge,
        spin=geom.n_electrons(charge) % 2,
        verbose=0,
    )
    hess_atoms = hess.reshape(n, 3, n, 3).transpose(0, 2, 1, 3)
    results = harmonic_analysis(mol, hess_atoms, imaginary_freq=False)
    return [float(f) for f in results["freq_wavenumber"]]
