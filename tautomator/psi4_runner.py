"""DFT calculations using Psi4 as a subprocess.

Runs Psi4 via its command-line interface, writing input files and parsing
output. Uses an (99, 590) integration grid for all DFT calculations
and a (50, 194) NLC grid for VV10-containing functionals, with Psi4's
recommended ROBUST grid pruning.

Implicit solvent comes from psi4-swig-pcm, installed in Psi4's own environment:
IEF-PCM on the same switching/Gaussian cavity as pyscf_runner's (ISWIG), with
analytical gradients, so Psi4 optimizes in solvent and its frequencies are finite
differences of analytical PCM gradients.  The same holds on MOIST's DROP cavities,
through psi4-swig-pcm's MOIST host; MOIST is installed in Psi4's environment too.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from tautomator.config import (
    AUX_DIFFUSE_S,
    AUXBASIS,
    CAVITIES,
    DEFAULT_MEMORY_GB,
    DROP_CAVITIES,
    DROP_COMPONENT,
    DROP_LSF,
    DROP_NLEB,
    DROP_PRUNE_LEVEL,
    PCM_METHOD,
    PCM_STORE_FRACTION,
    PCM_SWITCHING,
    solvent_dielectric,
)
from tautomator.types import Geometry
from tautomator.xyz_io import read_xyz

log = logging.getLogger(__name__)

# Functionals that include VV10 nonlocal correlation
_VV10_FUNCTIONALS = {"wb97m-v", "wb97m_v", "wb97x-v", "wb97x_v", "b97m-v", "b97m_v"}

# psi4-swig-pcm logs this when a cavity's nuclear surface charge strays from
# Gauss's law by more than 1% (C-PCM / IEF-PCM reach ~1e-3 on its surface).  It
# is a warning, not an error, there and here; a cavity it cannot use at all
# (S not positive-definite) raises inside Psi4 and fails the job.
_PCM_GAUSS_WARNING = "violates Gauss's law"


def _molecule_block(geom: Geometry, charge: int) -> str:
    """Build a Psi4 molecule block from a Geometry and charge."""
    mult = geom.multiplicity(charge)
    lines = [f"  {charge} {mult}"]
    for sym, (x, y, z) in zip(geom.symbols, geom.coords, strict=True):
        lines.append(f"  {sym}  {x: .10f}  {y: .10f}  {z: .10f}")
    lines.append("  units angstrom")
    lines.append("  no_reorient")
    lines.append("  no_com")
    return "\n".join(lines)


def _options_block(method: str, basis: str) -> str:
    """Build the Psi4 set block with DFT grid and other options."""
    method_lower = method.lower().replace("_", "-")

    opts = [
        f"  basis {basis}",
        "  dft_radial_points 99",
        "  dft_spherical_points 590",
        # ROBUST: Psi4's recommended region-based pruning (Bragg-Slater
        # radius zones).  Single keyword; Psi4 has no separate VV10 control.
        "  dft_pruning_scheme robust",
    ]

    # VV10 NLC grid
    if method_lower in _VV10_FUNCTIONALS:
        opts.extend(
            [
                "  dft_vv10_radial_points 50",
                "  dft_vv10_spherical_points 194",
            ]
        )

    # SCF convergence
    opts.append("  e_convergence 1e-8")
    opts.append("  d_convergence 1e-8")

    return "\n".join(opts)


# The fitting set of the elements in AUX_DIFFUSE_S, as Psi4 names it in its output.
_DF_BASIS_NAME = "def2-universal-jkfit-plus-s"


def _bse_basis_block(basis: str, geom: Geometry) -> str:
    """If `basis` should be loaded from basis-set-exchange, return a
    Psi4 input snippet that registers it via `basis_helper`. Otherwise "".

    Currently only vDZP is loaded from BSE: Psi4's bundled vdzp.gbs is
    missing fluorine (psi4/psi4#3205); the BSE copy carries the workaround
    (a vanishingly-small d-type ECP projector on F).
    """
    if basis.lower() != "vdzp":
        return ""
    import basis_set_exchange as bse

    elements = sorted(set(geom.symbols))
    psi4_str = bse.get_basis("Grimme vDZP", fmt="psi4", elements=elements)
    return f'\nbasis_helper("""\n{psi4_str}\n""", name="vDZP", set_option=True)\n'


def _df_basis_block(basis: str, geom: Geometry) -> str:
    """A Psi4 input snippet that gives the fitting basis its extra s primitives.

    "" unless `basis` is one Psi4 fits with AUXBASIS -- the def2 family and
    vDZP -- and `geom` holds an element of AUX_DIFFUSE_S.  Psi4 cannot add a
    function to a library set, so those elements get a set of their own --
    AUXBASIS as basis-set-exchange has it, which is what PySCF and Psi4 ship,
    plus the primitive -- and every other atom is assigned AUXBASIS by name,
    which is what Psi4 picks for it anyway.  Any other basis keeps the fit
    Psi4 picks for it, whatever the elements; config.AUXBASIS says why.
    """
    if not (basis.lower().startswith("def2") or basis.lower() == "vdzp"):
        return ""
    elements = sorted(AUX_DIFFUSE_S.keys() & set(geom.symbols))
    if not elements:
        return ""
    import basis_set_exchange as bse

    lines = [f"assign {AUXBASIS}"]
    lines += [f"assign {elem} {_DF_BASIS_NAME}" for elem in elements]
    lines += [f"[ {_DF_BASIS_NAME} ]", "spherical", "****"]
    for elem in elements:
        gbs = bse.get_basis(AUXBASIS, fmt="psi4", elements=[elem], header=False)
        # "spherical", then the element's shells between two "****" lines
        shells = gbs.split("****")[1].strip()
        lines += [shells, "S    1   1.00", f"      {AUX_DIFFUSE_S[elem]:.10f}  1.0000000", "****"]
    block = "\n".join(lines)
    return (
        f'\nbasis_helper("""\n{block}\n""", name="{_DF_BASIS_NAME}", '
        'key="df_basis_scf", set_option=True)\n'
    )


# What a job on a DROP cavity imports, beyond what every PCM job does; check_installed
# launches Psi4 on these lines alone.
# MOIST, and a psi4-swig-pcm whose MoistSettings builds the cavity on its own radii (the
# switching/Gaussian cavity's spheres); check_installed runs the same lines
_MOIST_IMPORTS = (
    "import dataclasses\n"
    "import moist\n"
    "from psi4_swig_pcm.moist_host import MoistSettings\n"
    'assert "radii" in {f.name for f in dataclasses.fields(MoistSettings)}, '
    '"psi4-swig-pcm predates MoistSettings.radii"\n'
)


def _solvation(
    cavity: str | None, solvent: str | None, pcm_hydrogen_radius: float, memory_gb: float
) -> tuple[str, float]:
    """A job's implicit-solvent input lines, and the memory (GB) left for Psi4 itself.

    A solvated job splits memory_gb as pyscf_runner's PCMs do: the stored surface
    integrals may take PCM_STORE_FRACTION of it (incore_mb), and Psi4 gets the rest
    (--memory).  Without a cavity there are no lines and Psi4 gets all of it.
    """
    if cavity is None or solvent is None:
        return "", memory_gb
    store_gb = memory_gb * PCM_STORE_FRACTION
    # Psi4 counts a GB as 1e9 bytes; psi4-swig-pcm's incore_mb is in MiB.
    incore_mb = int(store_gb * 1e9 / 2**20)
    lines = _pcm_block(cavity, solvent, pcm_hydrogen_radius, incore_mb=incore_mb)
    return lines, memory_gb - store_gb


def _pcm_block(
    cavity: str, solvent: str, pcm_hydrogen_radius: float = 1.1, *, incore_mb: int
) -> str:
    """Psi4 input lines that put psi4-swig-pcm's IEF-PCM on the cavity into every SCF of
    the job.

    psi4_swig_pcm.enable() makes Psi4's own energy(), gradient(), optimize() and
    frequency() include the PCM, with analytical gradients.  On ISWIG the model is
    pyscf_runner's: switching/Gaussian surface (PCM_SWITCHING) on Lebedev order 29,
    modified-Bondi radii x 1.2 with hydrogen at pcm_hydrogen_radius (Angstrom), and
    the same dielectric constant, so both drivers build the same cavity.  Its warnings go
    to stderr, where _run_psi4 looks for them.  incore_mb (MiB, outside Psi4's own
    memory) is the most the surface integrals of a cavity may take to be kept between
    SCF iterations; _solvation sizes it.

    A DROP cavity goes to psi4-swig-pcm's MOIST host instead,
    with the cavity, grid and pruning pyscf_runner gives MOIST, on the switching/Gaussian
    cavity's spheres (modified_bondi_radii, pyscf_runner's config.pcm_radii, as
    MoistSettings.radii, which psi4-swig-pcm puts into the cavity) rather than MOIST's own
    radii.  On rho-DROP pcm_hydrogen_radius is refused, as pyscf_runner
    refuses it: there the radii only place the isodensity surface's points.
    """
    eps = solvent_dielectric(solvent)
    preamble = (
        "\nimport logging\n"
        "import sys\n"
        "import psi4_swig_pcm\n"
        'logging.getLogger("psi4_swig_pcm").addHandler(logging.StreamHandler(sys.stderr))\n'
    )
    if cavity in DROP_CAVITIES:
        if cavity == "rho-DROP" and pcm_hydrogen_radius != 1.1:
            raise ValueError(
                f"pcm_hydrogen_radius = {pcm_hydrogen_radius} would change nothing on "
                f"{cavity!r}: the radii only place the isodensity surface's points"
            )
        moist_settings = (
            "from psi4_swig_pcm.cavity import modified_bondi_radii\n"
            "psi4_swig_pcm.enable(MoistSettings(\n"
            f"    cavity=moist.DROP(lsf=moist.{DROP_LSF[cavity]}(), "
            f"parameters=moist.DROPParameters(nleb={DROP_NLEB}, "
            f"wleb_prune_level={DROP_PRUNE_LEVEL})),\n"
            f"    components=(moist.{DROP_COMPONENT}({eps!r}),),\n"
            f"    incore_mb={incore_mb},\n"
            f"    radii=modified_bondi_radii(hydrogen={pcm_hydrogen_radius!r}),\n"
            "))\n"
        )
        return preamble + _MOIST_IMPORTS + moist_settings
    if cavity != "ISWIG":
        raise ValueError(f"Unknown cavity: {cavity!r} (has: {', '.join(CAVITIES)})")
    settings = (
        f"method={PCM_METHOD!r}, eps={eps!r}, "
        f"switching={PCM_SWITCHING!r}, "
        f"radii=modified_bondi_radii(hydrogen={pcm_hydrogen_radius!r}), "
        f"incore_mb={incore_mb}"
    )
    return preamble + (
        "from psi4_swig_pcm.cavity import modified_bondi_radii\n"
        f"psi4_swig_pcm.enable(psi4_swig_pcm.PCMSettings({settings}))\n"
    )


def single_point(
    geom: Geometry,
    charge: int,
    method: str,
    basis: str,
    cavity: str | None = None,
    solvent: str | None = None,
    pcm_hydrogen_radius: float = 1.1,
    threads: int = 1,
    memory_gb: float = DEFAULT_MEMORY_GB,
) -> float:
    """Run a single-point DFT energy calculation.

    Returns the total energy in Hartree.
    """
    mol_block = _molecule_block(geom, charge)
    opts_block = _options_block(method, basis)

    input_text = f"""\
molecule mol {{
{mol_block}
}}

set {{
{opts_block}
}}
"""
    input_text += _bse_basis_block(basis, geom)
    input_text += _df_basis_block(basis, geom)
    solvent_lines, psi4_memory_gb = _solvation(cavity, solvent, pcm_hydrogen_radius, memory_gb)
    input_text += solvent_lines

    input_text += f"""
E = energy('{method}')
psi4.print_out(f'\\n=== FINAL ENERGY: {{E:.12f}} ===\\n')
"""

    with _scratch_dir() as tmp:
        output = _run_psi4(input_text, Path(tmp), threads=threads, memory_gb=psi4_memory_gb)
        return _parse_final_energy(output)


def optimize(
    geom: Geometry,
    charge: int,
    method: str,
    basis: str,
    cavity: str | None = None,
    solvent: str | None = None,
    pcm_hydrogen_radius: float = 1.1,
    threads: int = 1,
    memory_gb: float = DEFAULT_MEMORY_GB,
) -> tuple[Geometry, float, bool]:
    """Run DFT geometry optimization.

    optking's dynamic_level=1 allows the optimizer to adaptively change
    parameters (coordinate system, trust radius, etc.) when progress stalls.
    With a cavity it minimizes in implicit solvent, on analytical PCM
    gradients.

    Returns (optimized_geometry, final_energy_hartree, converged). The
    geometry and energy are always the latest ones, even if optking did not
    fully converge.
    """
    if geom.n_atoms == 1:
        # An atom has no internal coordinates: its minimum is itself.
        energy = single_point(
            geom,
            charge,
            method,
            basis,
            cavity=cavity,
            solvent=solvent,
            pcm_hydrogen_radius=pcm_hydrogen_radius,
            threads=threads,
            memory_gb=memory_gb,
        )
        return geom, energy, True
    mol_block = _molecule_block(geom, charge)
    opts_block = _options_block(method, basis)

    input_text = f"""\
molecule mol {{
{mol_block}
}}

set {{
{opts_block}
  geom_maxiter 200
  dynamic_level 1
  g_convergence gau
}}
"""
    input_text += _bse_basis_block(basis, geom)
    input_text += _df_basis_block(basis, geom)
    solvent_lines, psi4_memory_gb = _solvation(cavity, solvent, pcm_hydrogen_radius, memory_gb)
    input_text += solvent_lines

    # Wrap optimize() so that a non-convergence exception still yields a
    # geometry and energy on disk (from the last optimizer step).  Two failure
    # modes are treated the same way: psi4's OptimizationConvergenceError
    # (geom_maxiter reached) and optking's "Maximum dynamic_level reached"
    # OptError (adaptive recovery exhausted).  Both leave a usable last-step
    # geometry in the active molecule; unlike OptimizationConvergenceError the
    # OptError carries no wfn, so we recover the energy with a single-point on
    # that geometry.  Any other failure (e.g. SCF non-convergence) propagates.
    input_text += f"""
from optking.exceptions import OptError
converged = True
try:
    E, wfn = optimize('{method}', return_wfn=True)
except psi4.OptimizationConvergenceError as exc:
    converged = False
    wfn = exc.wfn
    E = wfn.energy()
    psi4.print_out('\\n=== OPT NOT CONVERGED: using last geometry ===\\n')
except OptError as exc:
    if 'dynamic_level' not in str(exc):
        raise
    converged = False
    E, wfn = energy('{method}', return_wfn=True)
    psi4.print_out('\\n=== OPT NOT CONVERGED (dynamic_level): using last geometry ===\\n')
psi4.print_out(f'\\n=== FINAL ENERGY: {{E:.12f}} ===\\n')
psi4.print_out(f'\\n=== CONVERGED: {{1 if converged else 0}} ===\\n')
wfn.molecule().save_xyz_file('optimized.xyz', True)
"""

    with _scratch_dir() as tmp:
        work_dir = Path(tmp)
        output = _run_psi4(input_text, work_dir, threads=threads, memory_gb=psi4_memory_gb)
        energy = _parse_final_energy(output)
        converged = _parse_converged_flag(output)
        opt_geom = read_xyz(work_dir / "optimized.xyz")

    return opt_geom, energy, converged


def frequencies(
    geom: Geometry,
    charge: int,
    method: str,
    basis: str,
    cavity: str | None = None,
    solvent: str | None = None,
    pcm_hydrogen_radius: float = 1.1,
    threads: int = 1,
    memory_gb: float = DEFAULT_MEMORY_GB,
) -> list[float]:
    """Compute harmonic vibrational frequencies.

    Returns frequencies in cm⁻¹ (imaginary frequencies as negative values).  In
    implicit solvent the Hessian is finite differences of analytical PCM
    gradients (psi4-swig-pcm takes Psi4's analytical SCF Hessians out of play,
    since those would leave the PCM out), on MOIST's DROP cavities too.  An empty
    list for an atom.
    """
    if geom.n_atoms == 1:
        return []
    mol_block = _molecule_block(geom, charge)
    opts_block = _options_block(method, basis)

    input_text = f"""\
molecule mol {{
{mol_block}
}}

set {{
{opts_block}
}}
"""
    input_text += _bse_basis_block(basis, geom)
    input_text += _df_basis_block(basis, geom)
    solvent_lines, psi4_memory_gb = _solvation(cavity, solvent, pcm_hydrogen_radius, memory_gb)
    input_text += solvent_lines

    input_text += f"""
E, wfn = frequency('{method}', return_wfn=True)
freqs = wfn.frequencies().to_array()
with open('frequencies.dat', 'w') as f:
    for freq in freqs:
        f.write(f'{{freq:.6f}}\\n')
psi4.print_out(f'\\n=== FINAL ENERGY: {{E:.12f}} ===\\n')
"""

    with _scratch_dir() as tmp:
        work_dir = Path(tmp)
        _run_psi4(input_text, work_dir, threads=threads, memory_gb=psi4_memory_gb)
        freq_text = (work_dir / "frequencies.dat").read_text()

    freq_list: list[float] = []
    for line in freq_text.splitlines():
        stripped = line.strip()
        if stripped:
            freq_list.append(float(stripped))

    return freq_list


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


# The Psi4 environment (pixi.toml) of each environment tautomator runs in that does not
# take the psi4 one: the accelerate environment's Psi4 is on Accelerate too.
_PSI4_ENVIRONMENTS = {"accelerate": "psi4-accelerate"}


def psi4_environment() -> str:
    """The pixi environment whose Psi4 this interpreter launches."""
    return _PSI4_ENVIRONMENTS.get(Path(sys.prefix).name, "psi4")


def psi4_executable() -> Path:
    """The psi4 to launch: this project's Psi4 environment, else one on PATH.

    Psi4 is a separate program with a pixi environment of its own, installed
    beside the one this interpreter runs in: .pixi/envs/psi4 next to
    .pixi/envs/default, .pixi/envs/psi4-accelerate next to .pixi/envs/accelerate
    (psi4_environment).  That one comes first because it is the locked one; a
    psi4 on PATH is the fallback for installs that provide it some other way.
    The accelerate environment does not fall back to the psi4 one, whose Psi4 is
    on another BLAS.

    Raises if there is neither.  refine() and score() call this (through
    check_installed) when they pick the driver, so a missing Psi4 stops the run
    there instead of being recorded as every conformer's own failure.
    """
    env = psi4_environment()
    sibling = Path(sys.prefix).parent / env / "bin" / "psi4"
    if sibling.is_file():
        return sibling
    on_path = shutil.which("psi4")
    if on_path is None:
        raise FileNotFoundError(
            f"No psi4 executable: install its environment with `pixi install -e {env}`, "
            "or put psi4 on PATH."
        )
    return Path(on_path)


def check_installed(drop: bool = False) -> None:
    """Raise unless there is a Psi4 to launch, and with drop, one that can import MOIST.

    drop: the run is on a DROP cavity.  Its jobs import MOIST in Psi4's own
    interpreter, so the check launches that Psi4 once to import it (about half a
    second).  The Psi4 environments have MOIST, but one installed before they did, or
    a psi4 on PATH, may not.  refine() and score() call this when they pick the driver,
    so a missing Psi4 or MOIST stops the run there instead of being recorded as every
    conformer's own failure.
    """
    psi4 = psi4_executable()
    if not drop:
        return
    with _scratch_dir() as tmp:
        try:
            _run_psi4(_MOIST_IMPORTS, Path(tmp), timeout=300)
        except RuntimeError as err:
            errors = re.findall(r"\w+Error: .*", str(err))
            raise ImportError(
                f"The DROP cavities need MOIST and a psi4-swig-pcm that builds its "
                f"cavity on the PCM's radii, which {psi4} does not have "
                f"({errors[-1].strip() if errors else 'see the cause above'}). Install or "
                f"update the {psi4_environment()} environment, which has them: "
                f"`pixi install -e {psi4_environment()}` (it compiles MOIST)."
            ) from err


def _psi4_env(psi4: Path) -> dict[str, str]:
    """The environment to launch psi4 in: its own directory first on PATH.

    Psi4 looks its helper programs up on PATH (mctc-gcp for the gCP term of the
    3c composites, for one), and they are installed beside it, not in the
    environment this process runs in.
    """
    path = os.pathsep.join(filter(None, [str(psi4.parent), os.environ.get("PATH")]))
    return {**os.environ, "PATH": path}


def _scratch_dir() -> tempfile.TemporaryDirectory[str]:
    """Scratch directory for one Psi4 job, cleaned up on ``with`` exit.

    Every caller runs Psi4 inside one of these.  Leaking them is what filled
    the disk during the first pKa-set run: on the out-of-core DFJK path
    a single job writes several GB of three-index integrals, and nothing
    deleted the directories afterwards.

    Set TAUTOMATOR_KEEP_SCRATCH=1 to retain them for postmortem debugging.
    """
    keep = os.environ.get("TAUTOMATOR_KEEP_SCRATCH") == "1"
    return tempfile.TemporaryDirectory(prefix="psi4_", delete=not keep)


def _run_psi4(
    input_text: str,
    work_dir: Path,
    timeout: int = 86400,
    threads: int = 1,
    memory_gb: float = DEFAULT_MEMORY_GB,
) -> str:
    """Run psi4 inside work_dir and return the output text.

    work_dir owns the whole job: the input file, the output file, and — via
    ``-s`` — Psi4's binary scratch.  Callers create it with _scratch_dir() so
    that everything is removed together.  Any file Psi4 produces that the
    caller needs (optimized.xyz, frequencies.dat) must be read before the
    ``with`` block exits.

    memory_gb becomes ``--memory``, always passed so that Psi4 never silently
    falls back to its 500 MiB default (which starves the DFT collocation
    cache) and so a config means the same thing on both backends.  It is Psi4's
    own share: a solvated job passes what _solvation leaves after the PCM's.
    """
    input_path = work_dir / "input.dat"
    output_path = work_dir / "output.dat"
    input_path.write_text(input_text)

    psi4 = psi4_executable()
    cmd = [
        str(psi4),
        str(input_path),
        "-o",
        str(output_path),
        "-n",
        str(threads),
        "-s",
        str(work_dir),
        "--memory",
        f"{memory_gb}GB",
    ]

    log.info(f"Running Psi4 in {work_dir}")
    result = subprocess.run(
        cmd,
        cwd=work_dir,
        env=_psi4_env(psi4),
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    # Psi4 writes to the output file, not stdout
    output_text = ""
    if output_path.exists():
        output_text = output_path.read_text()

    if result.returncode != 0:
        raise RuntimeError(
            f"Psi4 failed (exit {result.returncode}):\n"
            f"stderr: {result.stderr[-2000:]}\n"
            f"output (last 2000 chars): {output_text[-2000:]}"
        )

    # psi4-swig-pcm's Gauss's-law check is a warning (see _PCM_GAUSS_WARNING):
    # pass it on rather than let it vanish with Psi4's stderr.
    if _PCM_GAUSS_WARNING in result.stderr:
        lines = [line for line in result.stderr.splitlines() if _PCM_GAUSS_WARNING in line]
        log.warning(f"  PCM cavity: {lines[-1].strip()}")

    return output_text


def _parse_final_energy(output: str) -> float:
    """Parse the energy from our sentinel line in Psi4 output."""
    match = re.search(r"=== FINAL ENERGY:\s+([-\d.]+)\s+===", output)
    if match is None:
        raise RuntimeError("Could not parse final energy from Psi4 output")
    return float(match.group(1))


def _parse_converged_flag(output: str) -> bool:
    """Parse the converged flag from our sentinel line in Psi4 output.

    If the flag is missing (older input templates), assume converged=True.
    """
    match = re.search(r"=== CONVERGED:\s+([01])\s+===", output)
    if match is None:
        return True
    return match.group(1) == "1"
