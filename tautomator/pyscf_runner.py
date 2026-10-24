"""DFT calculations using PySCF as a Python library.

Provides geometry optimization, single-point energy, and frequency
calculations. Uses an (99, 590) integration grid for all DFT calculations
and a (50, 194) NLC grid for VV10-containing functionals, with NWChem
region-based grid pruning.

Composite methods based on wB97X-V (wB97X-D4, wB97X-D4rev, wB97X-3c) are set
up by _set_method in whichever way this PySCF's dispersion dispatch accepts, and
their D4 parameters are pinned explicitly (see _D4_PARAMS). All three share the
reparameterized wB97X-V exchange-correlation functional (libxc 466), which is
distinct from the original wB97X (libxc 464).
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Iterable
from typing import Any

import numpy as np

from tautomator.config import (
    AUX_DIFFUSE_S,
    AUXBASIS,
    DEFAULT_MEMORY_GB,
    DROP_COMPONENTS,
    PCM_METHODS,
    PCM_SWITCHING,
    drop_solvent_model,
    pcm_radii,
    solvent_dielectric,
)
from tautomator.geometric_logging import geometric_quiet
from tautomator.types import Geometry

log = logging.getLogger(__name__)

# Angstrom <-> Bohr conversion
BOHR_TO_ANG = 0.529177210903
ANG_TO_BOHR = 1.0 / BOHR_TO_ANG

# Functionals that include VV10 nonlocal correlation
_VV10_XC = {"wb97m-v", "wb97x-v", "b97m-v"}

# ---------------------------------------------------------------------------
# wB97X-V + D4 composite methods
# ---------------------------------------------------------------------------
# PySCF natively maps "wb97x-d4" to libxc 464 (original wB97X, 2008) + D4
# (still its default in 2.14, with a FutureWarning that v2.16 will switch).
# That is not wB97X-D4 as defined by dftd4, which uses the reparameterized
# wB97X-V functional (libxc 466, 2013) with D4 dispersion instead of VV10.
# _set_method builds these composites itself rather than passing the name on.
#
# Mapping: user method name -> (XC functional, dftd4 param name)
# All use wB97X-V (466) as the XC functional with VV10 disabled.
_D4_COMPOSITES: dict[str, tuple[str, str]] = {
    "wb97x-d4": ("wb97x-v", "wb97x"),
    "wb97x-d4rev": ("wb97x-v", "wb97x-rev"),
    "wb97x-3c": ("wb97x-v", "wb97x-3c"),
}

# D4 rational-damping parameters for the composites, keyed by the dftd4 name
# in _D4_COMPOSITES and copied from dftd4 >= 4.0 parameters.toml (the
# "bj-eeq-atm" sets: all were fitted with the ATM three-body term, s9 = 1).
#
# They are set explicitly rather than looked up by name because PySCF uses the
# libdftd4 bundled inside pyscf-dispersion, and that library's name table is
# not stable across releases.  pyscf-dispersion 1.3.0 bundles libdftd4 3.7.0,
# where "wb97x" is the D4 fit for the *2008* wB97X functional (s8 = -0.0752);
# dftd4 PR #276 renamed that to "wb97x-2008" and gave "wb97x" to the fit that
# replaces VV10 in wB97X-V -- the one wB97X-D4 means here.  And with PySCF
# < 2.13.1, pyscf-dispersion 1.5 drops the ATM term these fits include.
# Pinning the numbers makes both independent of whichever libdftd4 is installed.
_D4_PARAMS: dict[str, dict[str, float]] = {
    # wB97X-D4: Najibi & Goerigk, J. Comput. Chem. 2020, 41, 2562
    # (doi:10.1002/jcc.26411)
    "wb97x": {"s6": 1.0, "s8": 0.5093, "a1": 0.0662, "a2": 5.4487, "s9": 1.0},
    # wB97X-D4rev and wB97X-3c: Mueller, Hansen & Grimme, J. Chem. Phys. 2023,
    # 158, 014103 (doi:10.1063/5.0133026)
    "wb97x-rev": {"s6": 1.0, "s8": 0.4485, "a1": 0.3306, "a2": 4.279, "s9": 1.0},
    "wb97x-3c": {"s6": 1.0, "s8": 0.0, "a1": 0.2464, "a2": 4.737, "s9": 1.0},
}

# Methods where the Hessian should "fall up" to wB97X-V (with VV10) for
# analytical second derivatives, avoiding expensive finite-difference D4
# Hessians.  The VV10 Hessian is analytical in PySCF.
_HESSIAN_FALLBACK: dict[str, str] = {
    "wb97x-d4": "wb97x-v",
    "wb97x-d4rev": "wb97x-v",
    "wb97x-3c": "wb97x-v",
}


def _d4_colon_syntax() -> bool:
    """Whether PySCF takes the dftd4 parameter name in ``mf.disp`` ("d4:<name>").

    PySCF 2.13.1 rewrote its dispersion dispatch (PR #3209).  Before it, the
    dftd4 name could only come from the XC string, which is what
    _register_d4_composites relies on.  From it on, XC_MAP is keyed on the
    parsed functional, so every wB97X-V composite registered that way would be
    sent to dftd4 as "wb97x"; the name has to be passed as ``mf.disp`` instead.
    ``parse_disp`` gained its ``disp`` argument in the same change, so probe
    that rather than compare version numbers.
    """
    from pyscf.scf import dispersion

    try:
        return bool(dispersion.parse_disp(None, "d4:wb97x-3c")[:2] == ("wb97x-3c", "d4"))
    except TypeError:  # parse_disp(dft_method) only: PySCF < 2.13.1
        return False


_D4_COLON_SYNTAX = _d4_colon_syntax()


def _register_d4_composites() -> None:
    """Register wB97X-V+D4 composite methods with PySCF's dispersion dispatch.

    Only for PySCF < 2.13.1 (see _d4_colon_syntax).  PySCF's libxc.parse_xc
    uses pyscf.scf.dispersion.parse_dft to extract the base XC code from method
    strings containing '-D3' or '-D4'.  We add entries to the internal
    _white_list and XC_MAP so that:
      - The XC functional evaluates as wB97X-V (libxc 466)
      - VV10 nonlocal correlation is disabled (nlc=False)
      - D4 dispersion is requested
      - The correct dftd4 parameter name is used for the D4 calculation

    Each composite is registered under an internal name with a '-d4' suffix
    (e.g. 'wb97x-3c-d4') so that libxc.parse_xc recognizes the '-D4' token
    and delegates to parse_dft.
    """
    from pyscf.scf import dispersion

    for _user_name, (_xc, d4_param) in _D4_COMPOSITES.items():
        internal = f"{d4_param}-d4"
        dispersion._white_list[internal] = (_xc, False, "d4")
        dispersion.XC_MAP[internal] = d4_param
    dispersion.parse_dft.cache_clear()
    dispersion.parse_disp.cache_clear()


if not _D4_COLON_SYNTAX:
    # Newer PySCF must not get these: "wb97x-d4" is a native name there (2.14).
    _register_d4_composites()


def _pin_d4_parameters() -> None:
    """Make PySCF's D4 model use the parameters in _D4_PARAMS.

    Energies, gradients and Hessians each construct
    ``pyscf.dispersion.dftd4.DFTD4Dispersion(mol, xc=<dftd4 name>, ...)``, looking
    the class up on the module at call time.  Replacing it with a subclass that
    overrides the named parameters covers all three paths.  Names not in
    _D4_PARAMS keep whatever the bundled libdftd4 defines.

    The wrapper's own ``set_param`` is not used: it calls
    ``dftd4_new_rational_damping`` without the leading error handle the C API
    requires (pyscf-dispersion 1.3.0 and 1.5.0 alike), so libdftd4 receives
    whatever is left in the pointer register -- sometimes NULL, which surfaces
    later as "Damping parameters are missing".
    """
    import ctypes

    from pyscf.dispersion import dftd4

    if getattr(dftd4.DFTD4Dispersion, "_tautomator_pinned_params", False):
        return  # idempotent: never double-wrap

    base = dftd4.DFTD4Dispersion
    lib = dftd4.libdftd4

    class DFTD4Dispersion(base):  # type: ignore[misc, valid-type]
        _tautomator_pinned_params = True
        _param: Any

        def __init__(self, mol: Any, xc: str, *args: Any, **kwargs: Any) -> None:
            # The base class still loads the named parameters (and fails on an
            # unknown name), so each key in _D4_PARAMS must exist in the bundled
            # libdftd4 -- all three do in 3.7.0 and 4.x, with different values.
            super().__init__(mol, xc, *args, **kwargs)
            p = _D4_PARAMS.get(xc.lower())
            if p is None:
                return
            err = lib.dftd4_new_error()
            try:
                param = lib.dftd4_new_rational_damping(
                    err,
                    ctypes.c_double(p["s6"]),
                    ctypes.c_double(p["s8"]),
                    ctypes.c_double(p["s9"]),
                    ctypes.c_double(p["a1"]),
                    ctypes.c_double(p["a2"]),
                    ctypes.c_double(16.0),  # alp: the ATM damping exponent, dftd4's default
                )
                dftd4.error_check(err)
            finally:
                lib.dftd4_delete_error(ctypes.byref(err))
            lib.dftd4_delete_param(ctypes.byref(self._param))
            self._param = param

    DFTD4Dispersion._tautomator_original = base
    dftd4.DFTD4Dispersion = DFTD4Dispersion


_pin_d4_parameters()


def _patch_einsum_path_for_numpy24() -> None:
    """Restore ``lib.einsum`` for contractions of three or more tensors.

    NumPy 2.4 dropped a field from ``einsum_path(..., einsum_call=True)``: each
    contraction became ``(inds, einsum_str, remaining)`` where it had been
    ``(inds, idx_rm, einsum_str, remaining)``. PySCF 2.11's
    ``numpy_helper.einsum`` unpacks ``contraction[:4]`` unconditionally, so any
    contraction over three or more operands raises ``ValueError: not enough
    values to unpack``. One- and two-operand contractions take earlier branches
    that never call ``_einsum_path``, which is why single-point energies are
    fine and density-fitted *gradients* are not -- the auxiliary term in
    ``pyscf/df/grad/rhf.py`` contracts ``'xpq,mp,nq->mnxp'``.

    Fixed upstream in 2.12.0 by commit 4e77c7e, four lines in one file. The
    locked osx-arm64 environment is still on 2.11, so the fix is applied here
    instead.

    This replaces ``numpy_helper._einsum_path``, not ``numpy_helper.einsum``.
    ``einsum`` reads ``_einsum_path`` from its module globals on every call, so
    one assignment reaches every call site -- including the handful of PySCF
    modules that bind ``einsum`` by name at import time and would keep a stale
    reference if the function itself were swapped. ``numpy.einsum_path`` is left
    alone, so nothing outside PySCF sees a modified NumPy.

    Gated on behaviour, not on version numbers: the probe runs a real
    three-operand contraction and patches only if it actually raises. On PySCF
    2.12+ or NumPy 2.3 and older it is a no-op. Only ``ValueError`` is caught,
    so an unrelated future breakage surfaces rather than being papered over.
    """
    import numpy
    from pyscf.lib import numpy_helper

    probe = (numpy.zeros((2, 2, 2)), numpy.zeros((2, 2)), numpy.zeros((2, 2)))
    try:
        numpy_helper.einsum("xpq,mp,nq->mnxp", *probe)
        return
    except ValueError:
        pass

    inner = numpy_helper._einsum_path
    if getattr(inner, "_tautomator_padded", False):
        return

    def padded(*args: Any, **kwargs: Any) -> Any:
        result = inner(*args, **kwargs)
        if not kwargs.get("einsum_call"):
            return result
        operands, contractions = result
        # Re-insert the dropped ``idx_rm`` slot. PySCF unpacks it and
        # ``remaining`` but reads neither, so a placeholder is exact.
        restored = [(c[0], None, c[1], c[2]) if len(c) == 3 else c for c in contractions]
        return operands, restored

    padded._tautomator_padded = True  # type: ignore[attr-defined]
    numpy_helper._einsum_path = padded


_patch_einsum_path_for_numpy24()


def _patch_pcm_ecp_cavity() -> None:
    """Correct PySCF's PCM cavity radii lookup for ECP atoms.

    ``pyscf.solvent.pcm.gen_surface`` indexes the inter-atomic switching-function
    vdW radii by ``mol.atom_charges()``, which an ECP *reduces* (e.g. oxygen
    8 -> 6).  ECP atoms therefore get the wrong element's radius (oxygen gets
    carbon's 1.70 A instead of 1.52).  The per-sphere radius already uses the
    true nuclear charge and is correct; only the switching radii are affected.

    Because vDZP places an ECP on every non-H atom, this corrupts the solvent
    cavity of every heavy atom in every PySCF implicit-solvent calculation
    (shifting solvation energies by ~0.26 kcal/mol on water).  We wrap
    ``gen_surface`` to expose true nuclear charges for the duration of the call.
    Every caller (PCM and SMD) reaches the function via ``pcm.gen_surface``, so a
    single reassignment covers them all.  gpu4pyscf has its own, already-correct
    ``gen_surface`` and is untouched.

    PySCF fixed this upstream in 2.13.0 (radii now come from ``mol.elements``),
    so the wrapper is installed only while ``gen_surface`` still reads
    ``mol.atom_charges()``.  If the source cannot be read, it is installed
    anyway: on a fixed PySCF it is a no-op.
    """
    import inspect

    import numpy
    import pyscf.solvent.pcm as pcm
    from pyscf import gto

    if getattr(pcm.gen_surface, "_ecp_cavity_patched", False):
        return  # idempotent: never double-wrap

    try:
        src = inspect.getsource(pcm.gen_surface)
    except (OSError, TypeError):
        src = None
    if src is not None and "mol.atom_charges()" not in src:
        return  # fixed upstream

    _orig_gen_surface = pcm.gen_surface

    def gen_surface(mol: Any, *args: Any, **kwargs: Any) -> Any:
        true_z = numpy.array([gto.charge(mol.atom_symbol(ia)) for ia in range(mol.natm)])
        had_attr = "atom_charges" in mol.__dict__
        saved = mol.__dict__.get("atom_charges")
        mol.atom_charges = lambda: true_z  # shadow the ECP-reduced method
        try:
            return _orig_gen_surface(mol, *args, **kwargs)
        finally:
            if had_attr:
                mol.atom_charges = saved
            else:
                del mol.atom_charges

    gen_surface._ecp_cavity_patched = True  # type: ignore[attr-defined]
    gen_surface._ecp_cavity_original = _orig_gen_surface  # type: ignore[attr-defined]
    pcm.gen_surface = gen_surface


_patch_pcm_ecp_cavity()


def _set_method(mf: Any, method: str) -> None:
    """Point a PySCF KS object at a user-facing method name.

    Most names go straight to ``mf.xc`` and PySCF parses them natively (e.g.
    "wB97X-D3BJ" dispatches D3BJ via pyscf-dispersion, "wB97M-V" enables VV10
    via libxc).  The D4 composites are wB97X-V with VV10 off and D4 on; how the
    dftd4 parameter name reaches PySCF depends on its version (see
    _d4_colon_syntax), while _pin_d4_parameters supplies the numbers either way.
    """
    key = method.lower().replace("_", "-")
    if key not in _D4_COMPOSITES:
        mf.xc = method
        return
    xc, d4_param = _D4_COMPOSITES[key]
    if _D4_COLON_SYNTAX:
        mf.xc = xc
        mf.nlc = False  # D4 replaces VV10
        mf.disp = f"d4:{d4_param}"
    else:
        mf.xc = f"{d4_param}-d4"  # the name _register_d4_composites registered


def _resolve_basis(
    basis: str, elements: list[str]
) -> tuple[str | dict[str, Any], dict[str, Any] | None]:
    """Resolve a basis set name, loading from basis-set-exchange if needed.

    PySCF doesn't include vDZP natively; we load it from BSE.  vDZP is an
    ECP-designed basis (heavier atoms have no core functions), so we must also
    load its per-element effective core potentials and apply them via
    ``mol.ecp`` — without them PySCF runs an all-electron SCF on a core-
    incapable basis and the energy is meaningless.

    Returns ``(basis, ecp)`` where ``ecp`` is ``None`` for built-in bases or a
    ``{element: ecp_data}`` dict for vDZP (elements with no core, e.g. H, are
    omitted).
    """
    if basis.lower() == "vdzp":
        import os
        import tempfile

        import basis_set_exchange as bse
        from pyscf.gto.basis import parse_nwchem

        nwchem_str = bse.get_basis("Grimme vDZP", fmt="nwchem")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".nw", delete=False) as f:
            f.write(nwchem_str)
            tmppath = f.name

        try:
            unique_elements = sorted(set(elements))
            basis_dict: dict[str, Any] = {}
            ecp_dict: dict[str, Any] = {}
            for elem in unique_elements:
                basis_dict[elem] = parse_nwchem.load(tmppath, elem)
                # vDZP replaces the core of heavier atoms with an ECP; light
                # atoms (H, He) have none, for which load_ecp returns empty.
                try:
                    ecp_data = parse_nwchem.load_ecp(tmppath, elem)
                except Exception:
                    ecp_data = None
                if ecp_data:
                    ecp_dict[elem] = ecp_data
            return basis_dict, (ecp_dict or None)
        finally:
            os.unlink(tmppath)

    # def2-family bases use effective core potentials for Z >= 37 (Rb onward).
    # PySCF does NOT auto-apply them from mol.basis, so an all-electron SCF on
    # a core-incapable basis would result (the same failure mode as vDZP).
    # Request the matching def2-ECP by name, but only for the heavy elements
    # that actually define one, so light-element calculations are untouched.
    if basis.lower().startswith("def2"):
        from pyscf import gto

        ecp_dict = {elem: basis for elem in sorted(set(elements)) if gto.charge(elem) >= 37}
        if ecp_dict:
            return basis, ecp_dict

    return basis, None


def _auxbasis(symbols: Iterable[str]) -> str | dict[str, Any]:
    """The fitting basis for a molecule of these elements, as density_fit takes it.

    AUXBASIS by name, unless an element is in AUX_DIFFUSE_S: then every element's
    shells are spelled out so that those can carry their extra s primitive.
    """
    elements = sorted(set(symbols))
    if AUX_DIFFUSE_S.keys().isdisjoint(elements):
        return AUXBASIS

    from pyscf import gto

    aux: dict[str, Any] = {}
    for elem in elements:
        shells = list(gto.basis.load(AUXBASIS, elem))
        if elem in AUX_DIFFUSE_S:
            shells.append([0, [AUX_DIFFUSE_S[elem], 1.0]])
        aux[elem] = shells
    return aux


def _build_mf(
    geom: Geometry,
    charge: int,
    method: str,
    basis: str,
    solvent_model: str | None = None,
    solvent: str | None = None,
    pcm_hydrogen_radius: float = 1.1,
    threads: int = 1,
    memory_gb: float = DEFAULT_MEMORY_GB,
) -> tuple[Any, Any]:
    """Build a PySCF Mole and mean-field object with the requested settings.

    Returns (mol, mf) where mf is ready for .kernel().

    memory_gb sets Mole.max_memory (PySCF's own units are MB).  Unlike Psi4's
    --memory this is a budget PySCF acts on -- it picks incore vs outcore
    algorithms from it and will try to allocate -- so overshooting real free
    memory risks an OOM rather than just slow I/O.

    It is not a cap on the process, and off Linux it barely limits anything.
    PySCF decides what fits by subtracting lib.current_memory() from
    max_memory, and current_memory() reads /proc, so on macOS it always
    returns 0: each buffer is checked against the whole budget as if nothing
    else were allocated.  At wB97M-V/def2-QZVPPD on a 10-heavy-atom molecule
    (966 AOs), both DF tensors (3.5 + 1.8 GB) stay in core at memory_gb=4 and
    the process peaks at 13-15 GB whether memory_gb is 4 or 8, with DF or
    conventional integrals alike.  On Linux the accounting is real, so the
    same job would likely move the larger DF tensor to disk instead.
    """
    import numba
    from pyscf import dft, gto, lib

    lib.num_threads(threads)
    # The nonelectrostatic correction's compiled kernels run on numba's own
    # threads, which would otherwise take every core.
    most = numba.config.NUMBA_NUM_THREADS  # type: ignore[attr-defined]  # set at import
    numba.set_num_threads(max(1, min(threads, most)))

    resolved_basis, resolved_ecp = _resolve_basis(basis, list(geom.symbols))

    mol = gto.Mole()
    mol.atom = [
        (sym, tuple(float(x) for x in coord))
        for sym, coord in zip(geom.symbols, geom.coords, strict=True)
    ]
    mol.basis = resolved_basis
    if resolved_ecp is not None:
        mol.ecp = resolved_ecp
    mol.charge = charge
    mol.spin = geom.n_electrons(charge) % 2  # 0 for singlet, 1 for doublet
    mol.verbose = 3
    mol.max_memory = memory_gb * 1024  # PySCF counts in MB
    mol.build()

    # Use RKS for closed-shell, UKS for open-shell
    mf = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)

    # Density fitting, before any solvent wrapper.  The order is not cosmetic:
    # PySCF raises "Gradients of solvent are not computed. Solvent must be
    # applied after density fit" if the wrapper is built around a conventional
    # mean field and density fitting is attached underneath it afterwards.
    mf = mf.density_fit(auxbasis=_auxbasis(geom.symbols))

    # XC functional, plus VV10/dispersion settings for the D4 composites.
    _set_method(mf, method)

    # Integration grid: (99, 590) for all DFT calculations, with NWChem
    # region-based pruning (PySCF's default scheme; the closest analog to
    # Psi4's ROBUST used in the Psi4 backend).
    mf.grids.atom_grid = (99, 590)
    mf.grids.prune = dft.gen_grid.nwchem_prune

    # VV10 NLC grid for functionals with active nonlocal correlation.
    # D4 composites have VV10 switched off, so they won't reach this branch.
    method_lower = method.lower().replace("_", "-")
    if method_lower in _VV10_XC:
        mf.nlcgrids.atom_grid = (50, 194)
        mf.nlcgrids.prune = dft.gen_grid.nwchem_prune

    # Implicit solvent.  Prefer the analytical PCM (with proper gradients)
    # over the experimental ddPCM/ddCOSMO domain-decomposed variants.
    if solvent_model is not None and solvent is not None:
        from pyscf import solvent as pyscf_solvent

        solvent_model_upper = solvent_model.upper().replace("-", "")
        drop = drop_solvent_model(solvent_model)
        if drop is not None:
            if drop[1] == "rho-DROP" and pcm_hydrogen_radius != 1.1:
                raise ValueError(
                    f"pcm_hydrogen_radius = {pcm_hydrogen_radius} would change nothing on "
                    f"{solvent_model!r}: the radii only place the isodensity surface's points"
                )
            mf = _attach_drop(
                mf, drop[0], drop[1], solvent_dielectric(solvent), threads, pcm_hydrogen_radius
            )
        elif solvent_model_upper == "SMD":
            mf = pyscf_solvent.SMD(mf)
            mf.with_solvent.solvent = solvent
        elif solvent_model_upper in PCM_METHODS:
            mf = pyscf_solvent.PCM(mf)
            mf.with_solvent.method = PCM_METHODS[solvent_model_upper]
            mf.with_solvent.eps = solvent_dielectric(solvent)
            # Cavity radii: the final sphere radii, the same table the DROP cavities
            # take (config.pcm_radii); with pcm_hydrogen_radius=1.10 it is PySCF's
            # default exactly (vdw_scale 1.2 x modified Bondi, no probe).
            ws = mf.with_solvent
            ws.radii_table = pcm_radii(pcm_hydrogen_radius)
            # Psi4 builds the same surface (psi4_runner._pcm_block).  A PySCF without the
            # attribute would ignore it and build SWIG's.
            if not hasattr(ws, "surface_discretization_method"):
                import pyscf

                raise NotImplementedError(
                    f"PySCF {pyscf.__version__} has no {PCM_SWITCHING} switching (2.13 and later)"
                )
            ws.surface_discretization_method = PCM_SWITCHING
            # The same model, several times faster; left as PySCF's own PCM on a PySCF
            # whose PCM internals FastPCM was not checked against.
            from tautomator.pyscf_fast_pcm import use_fast_pcm

            use_fast_pcm(mf)
        elif solvent_model_upper == "DDPCM":
            mf = pyscf_solvent.ddPCM(mf)
            mf.with_solvent.eps = solvent_dielectric(solvent)
        elif solvent_model_upper == "DDCOSMO":
            mf = pyscf_solvent.ddCOSMO(mf)
            mf.with_solvent.eps = solvent_dielectric(solvent)
        else:
            raise ValueError(f"Unknown PySCF solvent model: {solvent_model!r}")

    # SCF robustness: allow more iterations for difficult cases (especially
    # in implicit solvent).  Default is 50; bump to 100.
    mf.max_cycle = 100

    return mol, mf


def _attach_drop(
    mf: Any, pcm_method: str, cavity: str, eps: float, threads: int, hydrogen_radius: float
) -> Any:
    """``mf`` in a PCM model on one of MOIST's DROP cavities (tautomator.pyscf_moist), on
    the spheres of PySCF's cavity (config.pcm_radii, hydrogen at hydrogen_radius).

    After density fitting, like PySCF's own solvent wrappers: MOIST's wrapper rebuilds a
    density-fitted mean field without MoistHost.
    """
    try:
        import moist

        from tautomator.pyscf_moist import attach, drop_cavity
    except ImportError as err:
        raise ImportError(
            f"the {cavity} cavity needs MOIST, which the default pixi environment has "
            "(pixi install; the first install compiles it)"
        ) from err
    component = getattr(moist, DROP_COMPONENTS[pcm_method])(eps)
    context = moist.Context(nthreads=threads, verbosity=0)
    return attach(
        mf,
        cavity=drop_cavity(cavity, hydrogen_radius=hydrogen_radius),
        components=[component],
        context=context,
    )


def _run_scf_robust(mf: Any) -> float:
    """Run SCF; on non-convergence, retry with second-order SCF (.newton()).

    MOIST's PCM (a DROP solvent model) has no second-order SCF: there the retry is a
    level-shifted SCF from the last density instead.  The shift changes only the path, not
    the answer: it raises the virtual orbitals' energies, which leaves the occupied-virtual
    block (the orbital gradient) alone, and PySCF tests convergence on the unshifted Fock
    matrix, so a converged retry is a true SCF solution.  Its failure modes are harmless:
    a retry that does not converge raises, as before, and the conformer is excluded with a
    record; one that converged onto a higher-energy state would only give its conformer a
    higher energy, so a negligible Boltzmann weight, never a spurious minimum.  Checked on
    six ions, a zwitterion and a radical (wB97X-3c/vDZP in water, on both DROP cavities and
    on PySCF's PCM, each SCF stopped after 3 cycles): every retry reached the solution of a
    normal SCF to 7e-11 Ha, and on PySCF's PCM the level shift and Newton agree as closely,
    in 8-11 cycles against Newton's 3.  The retry covers the first SCF of a single point or
    an optimization only, not those geomeTRIC runs at each step (as with Newton).
    """
    energy = mf.kernel()
    if mf.converged:
        return float(energy)

    if hasattr(mf, "with_moist"):
        log.warning("SCF did not converge; retrying with a level shift (MOIST has no Newton)")
        dm = mf.make_rdm1()
        settings = mf.level_shift, mf.max_cycle
        mf.level_shift, mf.max_cycle = 0.25, 200
        try:
            energy = mf.kernel(dm0=dm)
        finally:
            mf.level_shift, mf.max_cycle = settings
        if not mf.converged:
            raise RuntimeError("PySCF SCF failed to converge, even with a level shift")
        return float(energy)

    log.warning("SCF did not converge; retrying with second-order SCF (Newton)")
    dm = mf.make_rdm1()
    mf_newton = mf.newton()
    mf_newton.max_cycle = 100
    energy = mf_newton.kernel(dm0=dm)
    if not mf_newton.converged:
        raise RuntimeError("PySCF SCF failed to converge even with Newton solver")
    # Copy converged density back so any subsequent operations see it
    mf.converged = True
    mf.e_tot = mf_newton.e_tot
    mf.mo_coeff = mf_newton.mo_coeff
    mf.mo_occ = mf_newton.mo_occ
    mf.mo_energy = mf_newton.mo_energy
    return float(energy)


def single_point(
    geom: Geometry,
    charge: int,
    method: str,
    basis: str,
    solvent_model: str | None = None,
    solvent: str | None = None,
    pcm_hydrogen_radius: float = 1.1,
    threads: int = 1,
    memory_gb: float = DEFAULT_MEMORY_GB,
) -> float:
    """Run a single-point DFT energy calculation.

    Returns the total energy in Hartree.
    """
    _mol, mf = _build_mf(
        geom,
        charge,
        method,
        basis,
        solvent_model,
        solvent,
        pcm_hydrogen_radius,
        threads,
        memory_gb,
    )
    return _run_scf_robust(mf)


def optimize(
    geom: Geometry,
    charge: int,
    method: str,
    basis: str,
    solvent_model: str | None = None,
    solvent: str | None = None,
    pcm_hydrogen_radius: float = 1.1,
    threads: int = 1,
    memory_gb: float = DEFAULT_MEMORY_GB,
) -> tuple[Geometry, float, bool]:
    """Run DFT geometry optimization.

    Strategy: first try the default TRIC internal coordinates (100 steps).
    If that doesn't converge, retry with Cartesian coordinates for another
    100 steps starting from the last internal-coords geometry.  This is
    analogous to Psi4 optking's dynamic_level mechanism.

    Returns (optimized_geometry, final_energy_hartree, converged).
    The geometry and energy are always the latest ones, regardless of
    convergence status.
    """
    if geom.n_atoms == 1:
        # An atom has no internal coordinates: its minimum is itself.
        energy = single_point(
            geom,
            charge,
            method,
            basis,
            solvent_model=solvent_model,
            solvent=solvent,
            pcm_hydrogen_radius=pcm_hydrogen_radius,
            threads=threads,
            memory_gb=memory_gb,
        )
        return geom, energy, True

    from pyscf.geomopt.geometric_solver import kernel as geom_kernel

    def _run(start_geom: Geometry, coordsys: str, maxsteps: int) -> tuple[Geometry, float, bool]:
        _mol, mf_local = _build_mf(
            start_geom,
            charge,
            method,
            basis,
            solvent_model,
            solvent,
            pcm_hydrogen_radius,
            threads,
            memory_gb,
        )
        _run_scf_robust(mf_local)
        # The optimizer works on its own copy (a scanner, with MOIST state of its own), so
        # this mean field's MOIST state, kept integrals and all, would only take memory
        # beside the copy's for the whole run.
        if hasattr(mf_local, "with_moist"):
            mf_local.with_moist.reset()
        # geom_kernel drives the optimization on a *copy* of the molecule and
        # never writes the final energy back to mf_local, so mf_local.e_tot is
        # stale (it holds the input-geometry SCF energy).  Capture the energy of
        # each step via geomeTRIC's callback; the last one corresponds to the
        # returned geometry, since convergence is checked after the final step.
        energies: list[float] = []
        with tempfile.TemporaryDirectory(prefix="pyscf_opt_") as tmpdir, geometric_quiet():
            conv, mol_opt = geom_kernel(
                mf_local,
                maxsteps=maxsteps,
                tmpdir=tmpdir,
                coordsys=coordsys,
                callback=lambda env: energies.append(float(env["energy"])),
            )
        coords_ang = np.asarray(mol_opt.atom_coords(), dtype=np.float64) * BOHR_TO_ANG
        out_geom = Geometry(symbols=tuple(mol_opt.elements), coords=coords_ang)
        # Fall back to mf_local.e_tot only if no step was evaluated, in which
        # case the geometry is unchanged and that energy is the right one.
        energy = energies[-1] if energies else float(mf_local.e_tot)
        return out_geom, energy, bool(conv)

    # First attempt: default TRIC (internal coordinates)
    opt_geom, energy, converged = _run(geom, coordsys="tric", maxsteps=100)
    if converged:
        return opt_geom, energy, True

    # Fallback: continue from the last geometry in Cartesian coordinates
    log.info(
        "Internal-coord opt did not converge in 100 steps; "
        "retrying from last geometry with Cartesian coords"
    )
    opt_geom, energy, converged = _run(opt_geom, coordsys="cart", maxsteps=100)
    return opt_geom, energy, converged


def frequencies(
    geom: Geometry,
    charge: int,
    method: str,
    basis: str,
    solvent_model: str | None = None,
    solvent: str | None = None,
    pcm_hydrogen_radius: float = 1.1,
    threads: int = 1,
    memory_gb: float = DEFAULT_MEMORY_GB,
) -> list[float]:
    """Compute harmonic vibrational frequencies.

    For D4-based methods (wB97X-D4, wB97X-D4rev, wB97X-3c), the Hessian
    "falls up" to wB97X-V (with analytical VV10) to avoid expensive
    finite-difference D4 Hessians.  The vibrational frequencies are
    insensitive to the choice of dispersion correction.

    Returns frequencies in cm⁻¹ (including imaginary frequencies as negative values);
    an empty list for an atom.
    """
    if geom.n_atoms == 1:
        return []
    if drop_solvent_model(solvent_model) is not None:
        raise NotImplementedError(
            f"no Hessian in solvent_model = {solvent_model!r}: MOIST's PCM has none"
        )
    method_lower = method.lower().replace("_", "-")
    hess_method = _HESSIAN_FALLBACK.get(method_lower, method)
    if hess_method != method:
        log.info(
            f"Hessian: falling up from {method} to {hess_method} for analytical second derivatives"
        )

    mol, mf = _build_mf(
        geom,
        charge,
        hess_method,
        basis,
        solvent_model,
        solvent,
        pcm_hydrogen_radius,
        threads,
        memory_gb,
    )
    _run_scf_robust(mf)

    hessian = mf.Hessian().kernel()

    # Use PySCF's built-in harmonic analysis, which projects out
    # translational and rotational modes before diagonalization.
    from pyscf.hessian.thermo import harmonic_analysis

    results = harmonic_analysis(mol, hessian, imaginary_freq=False)
    freqs_cm: Any = results["freq_wavenumber"]
    return [float(f) for f in freqs_cm]
