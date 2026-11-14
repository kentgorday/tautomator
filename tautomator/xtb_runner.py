"""Wrappers around CREST/xtb for xTB geometry optimization, single-point
energy, and quasi-RRHO vibrational free-energy corrections.

CREST 2.12 drives the external ``xtb`` binary as a subprocess (the CREST 3.x
in-process rewrite produces degenerate single-conformer ensembles on macOS, so
we pin 2.12). That pin covers the conformer search only. Everything here --
optimization, single points and Hessians -- calls ``xtb`` directly.

Optimization used to route through CREST ``--mdopt`` to dodge a Fortran
format-string crash in conda-forge xtb 6.7.1 *build 2*
(grimme-lab/xtb#1332). Build 4 fixed it and pixi.toml already floors us above
that, so the workaround outlived its cause -- and it was not free. CREST accepts
an optimization level, reports success, and returns the input geometry
bit-identical whenever its own optimizer considers the structure converged, so
a CREST-optimized conformer could not be tightened at all: ``xtb --opt`` calls
it converged in one cycle at crude, loose, normal *and* tight, while ``vtight``
takes tens of cycles and moves it half an angstrom. Structures left at ``tight``
carry imaginary modes of a few hundred wavenumbers, which a numerical Hessian
then reports as real.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from tautomator.config import alpb_name
from tautomator.types import Geometry
from tautomator.xyz_io import read_multi_xyz, write_xyz


def optimize(
    geom: Geometry,
    charge: int = 0,
    gfn: int = 2,
    solvent: str | None = None,
    opt_level: str = "tight",
    work_dir: Path | None = None,
) -> tuple[Geometry, bool]:
    """Optimize a geometry with ``xtb --opt``.

    Returns ``(geometry, converged)``. A geometry that did not converge is
    returned anyway, carrying xtb's last step: it is still a better starting
    point than what went in, and discarding it would lose a conformer for a
    numerical outcome rather than a chemical one. Callers decide what to do
    with the flag; nothing here drops anything.

    ``opt_level`` matters more than it looks. CREST's conformer search leaves
    structures that xtb calls converged at every level up to ``tight``, so
    re-minimizing those needs ``vtight`` to move at all. Fresh embeddings are
    far enough from a minimum that ``tight`` does real work.
    """
    if geom.n_atoms == 1:
        # An atom has no internal coordinates, and xtb --opt stops on one.
        return geom, True
    cleanup = False
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="xtb_opt_"))
        cleanup = True

    try:
        input_xyz = work_dir / "input.xyz"
        write_xyz(geom, input_xyz)

        cmd = [
            "xtb",
            str(input_xyz),
            "--opt",
            opt_level,
            "--gfn",
            str(gfn),
            "--chrg",
            str(charge),
        ]
        if solvent is not None:
            cmd.extend(["--alpb", alpb_name(solvent)])

        result = subprocess.run(
            cmd,
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=3600,
        )

        opt_xyz = work_dir / "xtbopt.xyz"
        if not opt_xyz.exists():
            raise RuntimeError(
                f"xtb optimization produced no geometry (exit {result.returncode}):\n"
                f"{result.stderr[-2000:]}"
            )

        # xtb writes xtbopt.xyz even when it runs out of cycles, and says so in
        # the log rather than in the exit code.
        converged = "GEOMETRY OPTIMIZATION CONVERGED" in result.stdout
        conformers = read_multi_xyz(opt_xyz)
        if not conformers:
            raise RuntimeError("xtbopt.xyz is empty")
        return conformers[0].geometry, converged

    finally:
        if cleanup:
            import shutil

            shutil.rmtree(work_dir, ignore_errors=True)


def single_point(
    geom: Geometry,
    charge: int = 0,
    gfn: int = 2,
    solvent: str | None = None,
    work_dir: Path | None = None,
) -> float:
    """Run a single-point energy calculation via the standalone xtb binary.

    CREST 2.x has no single-point run mode (``--sp`` is silently ignored and a
    full conformer search runs instead), so this calls ``xtb`` directly, which
    is also the engine CREST drives internally. Returns the energy in Hartree.
    """
    cleanup = False
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="xtb_sp_"))
        cleanup = True

    try:
        input_xyz = work_dir / "input.xyz"
        write_xyz(geom, input_xyz)

        cmd = [
            "xtb",
            str(input_xyz),
            "--sp",
            "--gfn",
            str(gfn),
            "--chrg",
            str(charge),
        ]
        if solvent is not None:
            cmd.extend(["--alpb", alpb_name(solvent)])

        result = subprocess.run(
            cmd,
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"xtb single-point failed (exit {result.returncode}):\n{result.stderr[-2000:]}"
            )

        return _parse_energy(result.stdout)

    finally:
        if cleanup:
            import shutil

            shutil.rmtree(work_dir, ignore_errors=True)


def frequencies(
    geom: Geometry,
    charge: int = 0,
    gfn: int = 2,
    solvent: str | None = None,
    biased: bool = False,
    threads: int | None = None,
    work_dir: Path | None = None,
) -> list[float]:
    """Compute harmonic vibrational frequencies via the standalone xtb binary.

    Args:
        biased: If True, use ``--bhess`` (Spicher-Grimme single-point Hessian),
            appropriate for geometries that are *not* stationary points on the
            xTB surface (e.g. DFT-optimized geometries during refinement). If
            False, use the plain numerical Hessian ``--hess`` for xTB minima
            (e.g. CREST-optimized geometries during sampling).
        solvent: ALPB implicit-solvent name (e.g. "water"); xTB RRHO is always
            computed in implicit solvent in this workflow.

    Returns frequencies in cm**-1 (the 3N-6 vibrational modes, with translation
    and rotation already projected out by xtb; imaginary modes as negatives).
    An atom has none, but xtb does not know that: for a lone Cl- it writes its
    three translations as 0.17 cm**-1 modes, which quasi-RRHO would turn into
    -6.5 kcal/mol of vibrational free energy.  So an atom gets an empty list.
    """
    if geom.n_atoms == 1:
        return []
    return _parse_g98_frequencies(
        _run_hessian(geom, charge, gfn, solvent, biased, threads, work_dir, "g98.out")
    )


def normal_modes(
    geom: Geometry,
    charge: int = 0,
    gfn: int = 2,
    solvent: str | None = None,
    biased: bool = False,
    threads: int | None = None,
) -> tuple[list[float], np.ndarray]:
    """Frequencies (cm**-1) and Cartesian normal modes from an xtb Hessian.

    The modes are rebuilt from xtb's full-precision ``hessian`` file rather than
    read from g98.out, whose vectors are the eigenvectors of the mass-weighted
    Hessian rounded to two decimals: taken as displacements they move heavy
    atoms too far, light ones too little, and carry some rigid rotation.  See
    `_cartesian_modes`.  The frequencies agree with xtb's own to a fraction of a
    cm**-1; its table of atomic masses differs from RDKit's in the last digits.

    Returns the frequencies in ascending order, imaginary ones as negatives, and
    their modes as an array of shape (n_modes, n_atoms, 3), each of unit norm, in
    the input geometry's frame and atom order.
    """
    if geom.n_atoms == 1:
        return [], np.zeros((0, 1, 3))
    text = _run_hessian(geom, charge, gfn, solvent, biased, threads, None, "hessian")
    return _cartesian_modes(geom, _parse_hessian(text, geom.n_atoms))


def _run_hessian(
    geom: Geometry,
    charge: int,
    gfn: int,
    solvent: str | None,
    biased: bool,
    threads: int | None,
    work_dir: Path | None,
    output: str,
) -> str:
    """Run an xtb Hessian and return the text of one file it writes (``output``)."""
    cleanup = False
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="xtb_hess_"))
        cleanup = True

    try:
        input_xyz = work_dir / "input.xyz"
        write_xyz(geom, input_xyz)

        cmd = [
            "xtb",
            str(input_xyz),
            "--bhess" if biased else "--hess",
            "--gfn",
            str(gfn),
            "--chrg",
            str(charge),
        ]
        if solvent is not None:
            cmd.extend(["--alpb", alpb_name(solvent)])
        if threads is not None:
            cmd.extend(["--parallel", str(threads)])

        result = subprocess.run(
            cmd,
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"xtb Hessian failed (exit {result.returncode}):\n{result.stderr[-2000:]}"
            )

        written = work_dir / output
        if not written.exists():
            raise FileNotFoundError(f"xtb did not produce {output} in {work_dir}")
        return written.read_text()

    finally:
        if cleanup:
            import shutil

            shutil.rmtree(work_dir, ignore_errors=True)


def _parse_g98_frequencies(text: str) -> list[float]:
    """Parse vibrational frequencies (cm⁻¹) from xtb's Gaussian-98 output.

    g98.out lists only the real vibrational modes (translation/rotation are
    already projected out), three per ``Frequencies --`` line.
    """
    freqs: list[float] = []
    prefix = "Frequencies --"
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            freqs.extend(float(tok) for tok in stripped[len(prefix) :].split())
    if not freqs:
        raise RuntimeError("Could not parse any frequencies from xtb g98.out")
    return freqs


def _parse_hessian(text: str, n_atoms: int) -> np.ndarray:
    """The Cartesian Hessian of xtb's ``hessian`` file (Hartree/bohr**2), shape (3N, 3N)."""
    values: list[float] = []
    lines = iter(text.splitlines())
    for line in lines:
        if line.strip().startswith("$hessian"):
            break
    for line in lines:
        if line.strip().startswith("$"):
            break
        values.extend(float(tok) for tok in line.split())
    size = 3 * n_atoms
    if len(values) != size * size:
        raise RuntimeError(
            f"xtb hessian file holds {len(values)} values, not {size * size} for {n_atoms} atoms"
        )
    hessian = np.array(values).reshape(size, size)
    symmetric: np.ndarray = (hessian + hessian.T) / 2
    return symmetric


# sqrt(Hartree / (bohr**2 amu)) as a wavenumber: sqrt(Eh / (a0**2 u)) / (2 pi c).
_WAVENUMBER = 5140.487


def _cartesian_modes(geom: Geometry, hessian: np.ndarray) -> tuple[list[float], np.ndarray]:
    """Frequencies and Cartesian normal modes of a Cartesian Hessian.

    The Hessian is mass-weighted, the three translations and three rotations
    (two for a linear molecule) are projected out, and what is left is
    diagonalized; each eigenvector, divided atom by atom by the square root of
    the mass, is the displacement pattern of its mode.  So a mode moves light
    atoms further than heavy ones, and carries no momentum or angular momentum.
    """
    from rdkit.Chem import GetPeriodicTable

    table = GetPeriodicTable()
    n = geom.n_atoms
    mass = np.array([table.GetAtomicWeight(s) for s in geom.symbols])
    root = np.repeat(np.sqrt(mass), 3)
    x = np.asarray(geom.coords, dtype=np.float64)
    r = x - (mass[:, None] * x).sum(axis=0) / mass.sum()
    axes = np.eye(3)
    rigid = np.array(
        [np.tile(axes[k], (n, 1)).ravel() for k in range(3)]
        + [np.cross(axes[k], r).ravel() for k in range(3)]
    )
    _, s, vt = np.linalg.svd(rigid * root, full_matrices=True)
    vib = vt[int((s > 1e-6 * s[0]).sum()) :]  # orthonormal, mass-weighted, no rigid motion
    eigval, eigvec = np.linalg.eigh(vib @ (hessian / np.outer(root, root)) @ vib.T)
    modes = (vib.T @ eigvec).T / root
    modes /= np.linalg.norm(modes, axis=1, keepdims=True)
    freqs = np.sign(eigval) * np.sqrt(np.abs(eigval)) * _WAVENUMBER
    return [float(f) for f in freqs], modes.reshape(-1, n, 3)


def _parse_energy(stdout: str) -> float:
    """Parse total energy from xtb stdout."""
    match = re.search(r"TOTAL ENERGY\s+([-\d.]+)\s+Eh", stdout)
    if match is None:
        raise RuntimeError("Could not parse energy from xtb output")
    return float(match.group(1))
