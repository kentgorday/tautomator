"""Tests for PySCF DFT runner, focusing on the D4 composite methods."""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys

import numpy as np
import pytest
from pyscf import dft, gto
from pyscf.data.radii import BOHR
from pyscf.dft import libxc
from pyscf.solvent import pcm as pyscf_pcm

from tautomator.config import PCM_METHODS, solvent_dielectric
from tautomator.pyscf_fast_pcm import FastPCM, unchecked_functions

# Import applies the D4 parameter pin, and the composite registration and PCM
# ECP-cavity patch where this PySCF needs them
from tautomator.pyscf_runner import (
    _D4_COLON_SYNTAX,
    _D4_COMPOSITES,
    _D4_PARAMS,
    _HESSIAN_FALLBACK,
    _build_mf,
    _resolve_basis,
    _set_method,
    frequencies,
    optimize,
    single_point,
)
from tautomator.types import Geometry

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
_WATER = "O 0 0 0; H 0 0 0.96; H 0 0.96 0"


@pytest.fixture()
def water_mol() -> gto.Mole:
    mol = gto.Mole()
    mol.atom = _WATER
    mol.basis = "def2-svp"
    mol.verbose = 0
    mol.build()
    return mol


def _ks(mol: gto.Mole, method: str) -> dft.rks.RKS:
    """An RKS object set up for ``method`` exactly as the runner does it."""
    mf = dft.RKS(mol)
    _set_method(mf, method)
    return mf


# ---------------------------------------------------------------------------
# _set_method tests
# ---------------------------------------------------------------------------
class TestSetMethod:
    @pytest.mark.parametrize("method", ["wb97x-v", "wb97m-v", "wb97x-d3bj", "b3lyp"])
    def test_non_composites_pass_through(self, method: str, water_mol: gto.Mole) -> None:
        assert _ks(water_mol, method).xc == method

    @pytest.mark.parametrize("method", list(_D4_COMPOSITES))
    def test_composite_route(self, method: str, water_mol: gto.Mole) -> None:
        # New PySCF takes the dftd4 name via mf.disp and must not see our
        # registered names ("wb97x-d4" is one of its own from 2.14); old PySCF
        # can only get it through the registered XC name.
        from pyscf.scf import dispersion

        d4_param = _D4_COMPOSITES[method][1]
        internal = f"{d4_param}-d4"
        mf = _ks(water_mol, method)
        if _D4_COLON_SYNTAX:
            assert (mf.xc, mf.nlc, mf.disp) == ("wb97x-v", False, f"d4:{d4_param}")
            assert internal not in dispersion.XC_MAP
        else:
            assert mf.xc == internal
            assert dispersion.XC_MAP[internal] == d4_param

    def test_route_detection_matches_known_boundary(self) -> None:
        # PR #3209 changed the dispatch in 2.13.1; the runner probes for it
        # rather than comparing versions, and this pins the two together.
        import pyscf

        version = tuple(int(x) for x in re.findall(r"\d+", pyscf.__version__)[:3])
        assert (version >= (2, 13, 1)) == _D4_COLON_SYNTAX


# ---------------------------------------------------------------------------
# _resolve_basis tests
# ---------------------------------------------------------------------------
class TestResolveBasis:
    def test_vdzp_loaded_from_bse(self) -> None:
        basis, _ecp = _resolve_basis("vDZP", ["H", "O"])
        assert isinstance(basis, dict)
        assert "H" in basis
        assert "O" in basis

    def test_vdzp_ecp_loaded_for_core_elements(self) -> None:
        # vDZP is ECP-designed: oxygen carries a core ECP, hydrogen does not.
        _basis, ecp = _resolve_basis("vDZP", ["H", "O"])
        assert ecp is not None
        assert "O" in ecp
        assert "H" not in ecp

    def test_vdzp_case_insensitive(self) -> None:
        basis, _ecp = _resolve_basis("vdzp", ["C", "H"])
        assert isinstance(basis, dict)

    def test_normal_basis_passes_through(self) -> None:
        basis, ecp = _resolve_basis("def2-svp", ["H", "O"])
        assert basis == "def2-svp"
        assert ecp is None

    def test_def2_no_ecp_for_light_elements(self) -> None:
        # def2 is all-electron through Z=36; no ECP for an organic element set.
        _basis, ecp = _resolve_basis("def2-QZVPPD", ["C", "H", "O", "Br"])
        assert ecp is None

    def test_def2_ecp_for_heavy_elements(self) -> None:
        # Z >= 37 (here iodine) needs the def2-ECP, applied by basis name.
        basis, ecp = _resolve_basis("def2-QZVPPD", ["I", "H"])
        assert ecp == {"I": "def2-QZVPPD"}
        assert basis == "def2-QZVPPD"


# ---------------------------------------------------------------------------
# PCM cavity correction for ECP atoms (monkeypatch of pyscf.solvent.pcm)
# ---------------------------------------------------------------------------
class TestPCMECPCavity:
    """The PCM cavity switching radii must use the true nuclear charge rather than
    the ECP-reduced ``mol.atom_charges()``: via the ``pcm.gen_surface`` patch on
    PySCF < 2.13.0, natively from 2.13.0 (where the patch is not installed)."""

    def _cavity_npts(self, basis: str) -> int:
        geom = Geometry(
            symbols=("O", "H", "H"),
            coords=np.array([[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]]),
        )
        _mol, mf = _build_mf(geom, 0, "pbe", basis, "IEFPCM", "water", 1)
        mf.with_solvent.build()
        return int(mf.with_solvent.surface["grid_coords"].shape[0])

    def test_ecp_basis_matches_all_electron_cavity(self) -> None:
        # Same geometry: an ECP basis (vDZP) must build the SAME PCM cavity as an
        # all-electron basis (def2-SVP), since the switching radii use true
        # nuclear charge in both.  Without the fix the vDZP (ECP) cavity differs
        # (oxygen's switching radius collapses to carbon's), changing the count.
        assert self._cavity_npts("vDZP") == self._cavity_npts("def2-svp")


# ---------------------------------------------------------------------------
# FastPCM (tautomator.pyscf_fast_pcm) for the PCM models
# ---------------------------------------------------------------------------
class TestFastPCM:
    """The PCM models get FastPCM where it was checked against this PySCF, with the same
    settings and results as PySCF's own PCM.  tests/test_pyscf_fast_pcm.py has the rest."""

    @staticmethod
    def _geom() -> Geometry:
        return Geometry(
            symbols=("O", "H", "H"),
            coords=np.array([[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]]),
        )

    @pytest.mark.parametrize("model", list(PCM_METHODS))
    def test_pcm_models_get_fast_pcm(self, model: str) -> None:
        _mol, mf = _build_mf(self._geom(), 0, "pbe", "def2-svp", model, "methanol", 1.2)
        ws = mf.with_solvent
        assert (type(ws) is FastPCM) == (not unchecked_functions())
        assert (ws.method, ws.eps) == (PCM_METHODS[model], solvent_dielectric("methanol"))
        assert ws.radii_table[1] == pytest.approx(ws.vdw_scale * 1.2 / BOHR + ws.r_probe / BOHR)

    def test_smd_keeps_pyscfs_class(self) -> None:
        _mol, mf = _build_mf(self._geom(), 0, "pbe", "def2-svp", "SMD", "water")
        assert not isinstance(mf.with_solvent, FastPCM)

    def test_matches_pyscfs_own_pcm_with_an_ecp_basis(self) -> None:
        """vDZP puts an ECP on oxygen.  Both runs build the same cavity, so its ECP radii
        are TestPCMECPCavity's to check, not this test's."""
        runs = []
        for fast in (True, False):
            _mol, mf = _build_mf(self._geom(), 0, "pbe", "vDZP", "IEFPCM", "water")
            if not fast:
                mf.with_solvent.__class__ = pyscf_pcm.PCM
                mf.with_solvent.reset()
            mf.conv_tol, mf.conv_tol_grad = 1e-11, 1e-7
            mf.kernel()
            runs.append((mf.e_tot, mf.with_solvent.e, mf.nuc_grad_method().kernel()))
        (e1, s1, g1), (e0, s0, g0) = runs
        assert abs(e1 - e0) < 1e-9 and abs(s1 - s0) < 1e-9
        np.testing.assert_allclose(g1, g0, atol=1e-7)


# ---------------------------------------------------------------------------
# D4 composite registration: correct XC functional
# ---------------------------------------------------------------------------
class TestD4CompositeXC:
    """All D4 composites must use wB97X-V (libxc 466), not wB97X (464)."""

    LIBXC_WB97X_V = 466
    LIBXC_WB97X = 464

    @pytest.mark.parametrize("method", list(_D4_COMPOSITES))
    def test_uses_wb97x_v_functional(self, method: str, water_mol: gto.Mole) -> None:
        """D4 composites must use the reparameterized wB97X-V (466)."""
        _, fac_list = libxc.parse_xc(_ks(water_mol, method).xc)
        func_id = int(fac_list[0][0])
        assert func_id == self.LIBXC_WB97X_V, (
            f"{method} uses libxc {func_id}, expected {self.LIBXC_WB97X_V} (wB97X-V)"
        )


# ---------------------------------------------------------------------------
# D4 composite registration: no VV10
# ---------------------------------------------------------------------------
class TestD4CompositeNoVV10:
    """D4 composites must NOT include VV10 nonlocal correlation."""

    @pytest.mark.parametrize("method", list(_D4_COMPOSITES))
    def test_no_vv10(self, method: str, water_mol: gto.Mole) -> None:
        mf = _ks(water_mol, method)
        assert not mf.do_nlc(), f"{method} has VV10 enabled"


# ---------------------------------------------------------------------------
# D4 composite registration: dispersion is applied
# ---------------------------------------------------------------------------
class TestD4CompositeDispersion:
    """D4 composites must have non-zero D4 dispersion energy."""

    @pytest.mark.parametrize("method", list(_D4_COMPOSITES))
    @pytest.mark.slow
    def test_d4_dispersion_applied(self, method: str, water_mol: gto.Mole) -> None:
        mf = _ks(water_mol, method)
        mf.grids.atom_grid = (50, 194)  # small grid for speed
        mf.verbose = 0
        mf.kernel()
        d4_energy = mf.scf_summary.get("dispersion", 0.0)
        assert abs(d4_energy) > 1e-8, f"{method} has no D4 dispersion"


# ---------------------------------------------------------------------------
# D4 parameters are distinct
# ---------------------------------------------------------------------------
class TestD4ParametersDistinct:
    """Each D4 composite must use different parameters."""

    @pytest.mark.slow
    def test_d4_energies_differ(self, water_mol: gto.Mole) -> None:
        d4_energies: dict[str, float] = {}
        for method in _D4_COMPOSITES:
            mf = _ks(water_mol, method)
            mf.grids.atom_grid = (50, 194)
            mf.verbose = 0
            mf.kernel()
            d4_energies[method] = float(mf.scf_summary.get("dispersion", 0.0))

        # All pairwise comparisons
        methods = list(d4_energies)
        for i, m1 in enumerate(methods):
            for m2 in methods[i + 1 :]:
                assert not np.isclose(d4_energies[m1], d4_energies[m2], atol=1e-8), (
                    f"{m1} and {m2} have same D4 energy: {d4_energies[m1]}"
                )


# ---------------------------------------------------------------------------
# D4 parameters match dftd4's own definitions
# ---------------------------------------------------------------------------
# dftd4-python's reference D4 for the water fixture, computed in a fresh process.
# Not in-process: pyscf-dispersion >= 1.5 bundles a libdftd4 with the same
# install name as conda's (libdftd4.4.dylib) but an older API, and once
# pyscf_runner has loaded it, dftd4-python's C extension binds to it and fails
# to import (missing dftd4_set_model_realspace_cutoff).
_D4_REFERENCE_SCRIPT = """
import json, sys
import numpy as np
from dftd4.interface import DampingParam, DispersionModel
numbers, positions, charge, names = json.load(sys.stdin)
model = DispersionModel(np.array(numbers), np.array(positions), charge=charge)
out = {}
for name in names:
    res = model.get_dispersion(DampingParam(method=name, atm=True), grad=True)
    out[name] = {"energy": float(res["energy"]), "gradient": np.asarray(res["gradient"]).tolist()}
json.dump(out, sys.stdout)
"""


@pytest.fixture(scope="module")
def d4_reference() -> dict[str, dict[str, np.ndarray]]:
    mol = gto.M(atom=_WATER, basis="def2-svp", verbose=0)
    names = [*sorted(_D4_PARAMS), "wb97x-2008"]
    payload = [mol.atom_charges().tolist(), mol.atom_coords().tolist(), mol.charge, names]
    proc = subprocess.run(
        [sys.executable, "-c", _D4_REFERENCE_SCRIPT],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=True,
    )
    raw = json.loads(proc.stdout)
    return {
        name: {"energy": np.asarray(v["energy"]), "gradient": np.asarray(v["gradient"])}
        for name, v in raw.items()
    }


class TestD4ParametersMatchDftd4:
    """PySCF's D4 for each composite must equal standalone dftd4-python's.

    PySCF evaluates D4 with the libdftd4 bundled in pyscf-dispersion, whose
    name table differs by release (3.7.0 maps "wb97x" to the wB97X-2008 fit),
    so _pin_d4_parameters sets the numbers explicitly.  These tests compare
    against the conda dftd4-python package, an independent build of the
    reference parameters, through the same dispersion hooks SCF energies and
    nuclear gradients use.  No SCF is needed: D4 depends only on geometry.
    """

    @pytest.mark.parametrize("d4_name", sorted(_D4_PARAMS))
    def test_param_table_matches_dftd4(self, d4_name: str) -> None:
        from dftd4.parameters import get_damping_param

        ref = get_damping_param(d4_name)
        for key, value in _D4_PARAMS[d4_name].items():
            assert value == pytest.approx(ref[key]), (d4_name, key, value, ref[key])

    @pytest.mark.parametrize("method", list(_D4_COMPOSITES))
    def test_energy_matches_dftd4(
        self, method: str, water_mol: gto.Mole, d4_reference: dict[str, dict[str, np.ndarray]]
    ) -> None:
        mf = _ks(water_mol, method)
        ref = d4_reference[_D4_COMPOSITES[method][1]]
        assert float(mf.get_dispersion()) == pytest.approx(float(ref["energy"]), rel=1e-10)

    @pytest.mark.parametrize("method", list(_D4_COMPOSITES))
    def test_gradient_matches_dftd4(
        self, method: str, water_mol: gto.Mole, d4_reference: dict[str, dict[str, np.ndarray]]
    ) -> None:
        mf = _ks(water_mol, method)
        ref = d4_reference[_D4_COMPOSITES[method][1]]
        grad = mf.nuc_grad_method().get_dispersion()
        np.testing.assert_allclose(grad, ref["gradient"], rtol=1e-9, atol=1e-12)

    @pytest.mark.parametrize("method", list(_D4_COMPOSITES))
    def test_solvated_energy_matches_dftd4(
        self, method: str, d4_reference: dict[str, dict[str, np.ndarray]]
    ) -> None:
        # The solvent wrapper _build_mf applies must carry the composite's
        # dispersion settings over (mf.disp and mf.nlc on the newer route).
        mol = gto.M(atom=_WATER, verbose=0)
        geom = Geometry(symbols=tuple(mol.elements), coords=mol.atom_coords(unit="Angstrom"))
        _mol, mf = _build_mf(geom, 0, method, "def2-svp", "IEFPCM", "water")
        ref = d4_reference[_D4_COMPOSITES[method][1]]
        assert not mf.do_nlc()
        assert float(mf.get_dispersion()) == pytest.approx(float(ref["energy"]), rel=1e-10)

    def test_wb97x_d4_is_not_the_2008_fit(
        self, water_mol: gto.Mole, d4_reference: dict[str, dict[str, np.ndarray]]
    ) -> None:
        # Regression: before the parameters were pinned, wB97X-D4 silently got
        # the D4 fit for the 2008 wB97X functional from the bundled libdftd4.
        mf = _ks(water_mol, "wb97x-d4")
        wrong = d4_reference["wb97x-2008"]
        assert not np.isclose(float(mf.get_dispersion()), float(wrong["energy"]), atol=1e-8)


# ---------------------------------------------------------------------------
# D4 dispersion in gradients
# ---------------------------------------------------------------------------
class TestD4Gradients:
    """D4 dispersion must be included in analytical gradients."""

    @pytest.mark.slow
    @pytest.mark.parametrize("method", list(_D4_COMPOSITES))
    def test_gradient_includes_d4(self, method: str, water_mol: gto.Mole) -> None:
        """Compare gradient with and without D4 to verify it contributes."""
        # With D4
        mf_d4 = _ks(water_mol, method)
        mf_d4.grids.atom_grid = (50, 194)
        mf_d4.verbose = 0
        mf_d4.kernel()
        grad_d4 = mf_d4.nuc_grad_method()
        grad_d4.verbose = 0
        g_with = grad_d4.kernel()

        # Without D4 (plain wb97x-v, no VV10, no dispersion)
        mf_bare = dft.RKS(water_mol)
        mf_bare.xc = "wb97x-v"
        mf_bare.nlc = 0
        mf_bare.grids.atom_grid = (50, 194)
        mf_bare.verbose = 0
        mf_bare.kernel()
        grad_bare = mf_bare.nuc_grad_method()
        grad_bare.verbose = 0
        g_without = grad_bare.kernel()

        # Gradients should differ (D4 contributes)
        assert not np.allclose(g_with, g_without, atol=1e-10), (
            f"{method} gradient unchanged by D4 dispersion"
        )


# ---------------------------------------------------------------------------
# Hessian fallback
# ---------------------------------------------------------------------------
class TestHessianFallback:
    @pytest.mark.parametrize("method", list(_HESSIAN_FALLBACK))
    def test_d4_methods_fall_up_to_vv10(self, method: str) -> None:
        assert _HESSIAN_FALLBACK[method] == "wb97x-v"

    def test_non_d4_methods_not_in_fallback(self) -> None:
        for method in ["wb97m-v", "wb97x-d3bj", "b3lyp"]:
            assert method not in _HESSIAN_FALLBACK


# ---------------------------------------------------------------------------
# optimize() returns the minimized energy, not a stale input-geometry energy
# ---------------------------------------------------------------------------
class TestOptimizeEnergy:
    """Regression guard: geomeTRIC drives the optimization on a copy of the
    molecule and never writes the final energy back to the mean-field object,
    so returning ``mf.e_tot`` yields the *input*-geometry energy (was ~25
    kcal/mol stale).  optimize() must return the energy of the geometry it
    returns.
    """

    @pytest.mark.slow
    def test_energy_matches_returned_geometry(self) -> None:
        # Distorted water so the optimizer has to move the atoms appreciably.
        start = Geometry(
            symbols=("O", "H", "H"),
            coords=np.array(
                [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [-0.3, 0.85, 0.0]],
                dtype=np.float64,
            ),
        )
        method, basis = "pbe", "sto-3g"

        e_input = single_point(start, 0, method, basis, threads=1)
        opt_geom, e_opt, converged = optimize(start, 0, method, basis, threads=1)
        e_at_opt = single_point(opt_geom, 0, method, basis, threads=1)

        assert converged
        # The returned energy is the one at the returned geometry ...
        assert e_opt == pytest.approx(e_at_opt, abs=1e-6)
        # ... and relaxation actually lowered it well below the input geometry,
        # so a stale mf.e_tot (== e_input) would be caught.
        assert e_input - e_opt > 1e-3
        assert e_opt < e_at_opt + 1e-6

    @pytest.mark.slow
    def test_optimize_leaves_logging_alone(self, caplog: pytest.LogCaptureFixture) -> None:
        # geomeTRIC reconfigures logging on every run (here from PySCF's
        # log.ini), which replaced the pipeline's handlers and printed every
        # optimization step into its log.
        start = Geometry(
            symbols=("O", "H", "H"),
            coords=np.array([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [-0.3, 0.85, 0.0]]),
        )
        root = logging.getLogger()
        with caplog.at_level(logging.INFO):
            handlers = list(root.handlers)
            optimize(start, 0, "pbe", "sto-3g", threads=1)
            assert root.handlers == handlers
            logging.getLogger("tautomator.refinement").info("still logging")
        assert "still logging" in caplog.text
        assert not [r for r in caplog.records if r.name.startswith("geometric")]


class TestEinsumPathShim:
    """PySCF 2.11's `lib.einsum` is broken under NumPy 2.4 for 3+ operands.

    NumPy 2.4 dropped a field from `einsum_path(..., einsum_call=True)`, and
    2.11 unpacks `contraction[:4]` unconditionally. Density-fitted gradients hit
    it through `'xpq,mp,nq->mnxp'` in the auxiliary term. Importing
    `tautomator.pyscf_runner` applies the fix; upstream's own fix is in 2.12.0
    (commit 4e77c7e), so this test keeps passing unchanged after an upgrade and
    the shim becomes removable without touching it.
    """

    @staticmethod
    def _cases() -> list[tuple[str, list[tuple[int, ...]]]]:
        return [
            ("xpq,mp,nq->mnxp", [(3, 4, 4), (2, 4), (2, 4)]),  # the DF-gradient one
            ("ab,bc,cd->ad", [(3, 3), (3, 3), (3, 3)]),
            ("ab,bc,cd,de->ae", [(3, 3), (3, 3), (3, 3), (3, 3)]),
        ]

    def test_multi_operand_einsum_matches_numpy(self) -> None:
        from pyscf import lib

        import tautomator.pyscf_runner  # noqa: F401  (import applies the shim)

        rng = np.random.default_rng(0)
        for subscripts, shapes in self._cases():
            tensors = [rng.random(shape) for shape in shapes]
            assert np.allclose(
                lib.einsum(subscripts, *tensors),
                np.einsum(subscripts, *tensors),
                atol=1e-12,
            ), subscripts

    def test_two_operand_contractions_are_untouched(self) -> None:
        """They take an earlier branch that never reaches `_einsum_path`."""
        from pyscf import lib

        import tautomator.pyscf_runner  # noqa: F401

        rng = np.random.default_rng(1)
        a, b = rng.random((4, 5)), rng.random((5, 6))
        assert np.allclose(lib.einsum("ij,jk->ik", a, b), a @ b)

    def test_numpy_einsum_path_is_not_modified_globally(self) -> None:
        """The patch is confined to a PySCF module-private name.

        Patching `numpy.einsum_path` itself would change behaviour for every
        other consumer in the process, which is not ours to do.
        """
        import numpy

        import tautomator.pyscf_runner  # noqa: F401

        assert numpy.einsum_path.__module__.startswith("numpy")

    def test_applying_it_twice_does_not_stack(self) -> None:
        from pyscf import lib
        from pyscf.lib import numpy_helper

        from tautomator.pyscf_runner import _patch_einsum_path_for_numpy24

        installed = numpy_helper._einsum_path
        rng = np.random.default_rng(2)
        tensors = [rng.random(s) for s in ((3, 4, 4), (2, 4), (2, 4))]
        expected = np.einsum("xpq,mp,nq->mnxp", *tensors)
        for _ in range(3):
            _patch_einsum_path_for_numpy24()
        # Re-applying must leave the installed function alone, not wrap it again.
        assert numpy_helper._einsum_path is installed
        assert np.allclose(lib.einsum("xpq,mp,nq->mnxp", *tensors), expected)


class TestDensityFittingIsOn:
    def test_the_mean_field_is_density_fitted(self) -> None:
        """Matches Psi4, which already runs DF with the same aux basis."""
        from tautomator.pyscf_runner import _build_mf
        from tautomator.types import Geometry

        geom = Geometry(
            symbols=("O", "H", "H"),
            coords=np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]),
        )
        _mol, mf = _build_mf(geom, 0, "PBE0", "def2-SVP", threads=1, memory_gb=1.0)
        assert getattr(mf, "with_df", None) is not None, "density fitting not attached"
        assert mf.with_df.auxbasis == "def2-universal-jkfit"

    def test_density_fitting_sits_under_the_solvent_wrapper(self) -> None:
        """Order is load-bearing: PySCF refuses solvent gradients otherwise."""
        from tautomator.pyscf_runner import _build_mf
        from tautomator.types import Geometry

        geom = Geometry(
            symbols=("O", "H", "H"),
            coords=np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]),
        )
        _mol, mf = _build_mf(
            geom, 0, "PBE0", "def2-SVP", "IEFPCM", "water", threads=1, memory_gb=1.0
        )
        assert hasattr(mf, "with_solvent"), "solvent wrapper missing"
        assert getattr(mf, "with_df", None) is not None, "density fitting not attached"
        # Either order leaves both attributes on the object, so check what the
        # order decides: PySCF refuses to build solvent gradients when density
        # fitting was attached outside the solvent wrapper.
        mf.nuc_grad_method()

    def test_short_fitting_sets_get_a_diffuse_s(self) -> None:
        """Only for AUX_DIFFUSE_S's elements; the others keep the library shells."""
        from tautomator.config import AUX_DIFFUSE_S, AUXBASIS
        from tautomator.pyscf_runner import _auxbasis

        aux = _auxbasis(("C", "S", "H", "H", "Br"))
        assert isinstance(aux, dict)
        assert set(aux) == {"C", "S", "H", "Br"}
        for elem in ("S", "Br"):
            assert aux[elem] == [*gto.basis.load(AUXBASIS, elem), [0, [AUX_DIFFUSE_S[elem], 1.0]]]
        for elem in ("C", "H"):
            assert aux[elem] == gto.basis.load(AUXBASIS, elem)

    def test_the_diffuse_exponents_follow_the_rule(self) -> None:
        """Each is the set's own smallest s exponent for that element, over 2.5."""
        from tautomator.config import AUX_DIFFUSE_S, AUXBASIS

        for elem, exponent in AUX_DIFFUSE_S.items():
            assert exponent == round(_smallest_s(AUXBASIS, elem) / 2.5, 3), elem

    def test_no_other_element_has_so_short_an_s_series(self) -> None:
        """Wherever def2-QZVPPD exists, a set that stops above 0.2 is one we extend.

        Hydrogen and the noble gases stop there too and are left alone: nothing
        here computes a hydride ion, and the noble gases form no anions.
        """
        from pyscf.lib.exceptions import BasisNotFoundError

        from tautomator.config import AUX_DIFFUSE_S, AUXBASIS

        short = set()
        for z in range(1, 87):
            elem = gto.elements.ELEMENTS[z]
            try:
                gto.basis.load("def2-qzvppd", elem)
            except BasisNotFoundError:  # the lanthanides
                continue
            if _smallest_s(AUXBASIS, elem) > 0.2:
                short.add(elem)
        assert short - {"H", "He", "Ar", "Kr", "Xe"} == set(AUX_DIFFUSE_S)

    @pytest.mark.slow
    def test_chloride_is_fitted_as_well_as_anything_else(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With the library set alone this is -0.27 kcal/mol."""
        from pyscf.scf import hf

        chloride = Geometry(symbols=("Cl",), coords=np.zeros((1, 3)))
        _mol, mf = _build_mf(chloride, -1, "PBE0", "def2-TZVPPD", threads=1, memory_gb=1.0)
        fitted = mf.kernel()

        monkeypatch.setattr(hf.SCF, "density_fit", lambda self, *args, **kwargs: self)
        _mol, mf = _build_mf(chloride, -1, "PBE0", "def2-TZVPPD", threads=1, memory_gb=1.0)
        assert getattr(mf, "with_df", None) is None
        conventional = mf.kernel()

        assert abs(fitted - conventional) * 627.5095 < 0.02


def _smallest_s(basis: str, elem: str) -> float:
    shells = gto.basis.load(basis, elem)
    return float(min(row[0] for shell in shells if shell[0] == 0 for row in shell[1:]))


class TestAtom:
    """An atom has no internal coordinates: geomeTRIC has nothing to optimize
    and there is no vibration, so both return at once rather than failing."""

    fluoride = Geometry(symbols=("F",), coords=np.zeros((1, 3)))

    def test_optimize_returns_the_atom_and_its_energy(self) -> None:
        geom, energy, converged = optimize(self.fluoride, -1, "pbe", "sto-3g", threads=1)
        assert converged
        assert geom is self.fluoride
        assert energy == pytest.approx(single_point(self.fluoride, -1, "pbe", "sto-3g"))

    def test_an_atom_has_no_frequencies(self) -> None:
        assert frequencies(self.fluoride, -1, "pbe", "sto-3g") == []


class TestThreads:
    def test_numba_takes_the_thread_count(self) -> None:
        """The nonelectrostatic kernels run on numba's threads: they follow `threads` too."""
        import numba
        from pyscf import lib

        atom = Geometry(("F",), np.zeros((1, 3)))
        most = numba.config.NUMBA_NUM_THREADS  # type: ignore[attr-defined]
        before = numba.get_num_threads(), lib.num_threads()
        try:
            _build_mf(atom, -1, "hf", "sto-3g", threads=1)
            assert numba.get_num_threads() == 1
            # More than numba's pool holds is capped, not an error
            _build_mf(atom, -1, "hf", "sto-3g", threads=most + 1)
            assert numba.get_num_threads() == most
        finally:
            numba.set_num_threads(before[0])
            lib.num_threads(before[1])
