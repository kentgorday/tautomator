"""FastPCM (tautomator.pyscf_fast_pcm) against PySCF's own PCM.

Each function FastPCM replaces is compared with PySCF's at the same input, then whole
SCF, gradient and Hessian runs.  To check another PySCF release, run this file in an
environment with tautomator's dependencies and that release; see the module docstring for
adding one to CHECKED.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from pyscf import __version__ as pyscf_version
from pyscf import dft, gto, scf
from pyscf.solvent import pcm
from pyscf.solvent.grad import pcm as pcm_grad

# Import applies the NumPy 2.4 einsum fix PySCF < 2.12 needs for density-fitted gradients
# (test_scf_and_gradient_match_stock), as in production
import tautomator.pyscf_runner  # noqa: F401
from tautomator import pyscf_fast_pcm
from tautomator.pyscf_fast_pcm import (
    CHECKED,
    REPLACED,
    FastPCM,
    fingerprint,
    installed_fingerprints,
    use_fast_pcm,
)

EPS = 78.39
MODELS = ["IEF-PCM", "C-PCM", "COSMO", "SS(V)PE"]

# name: (atoms in Angstrom, charge, spin, basis, cart)
MOLECULES: dict[str, tuple[str, int, int, str, bool]] = {
    "formate": (
        "C 0 0 0.52; O 1.13 0 -0.05; O -1.13 0 -0.05; H 0 0 1.66",
        -1,
        0,
        "def2-svp",
        False,
    ),
    "iodomethane": (  # iodine gets def2-SVP's ECP from ECPS
        "C 0 0 0; I 0 0 2.14; H 1.03 0 -0.36; H -0.515 0.892 -0.36; H -0.515 -0.892 -0.36",
        0,
        0,
        "def2-svp",
        False,
    ),
    "methyl": (
        "C 0 0 0; H 1.079 0 0; H -0.5395 0.9344 0; H -0.5395 -0.9344 0",
        0,
        1,
        "def2-svp",
        False,
    ),  # UHF
    "water": ("O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692", 0, 0, "6-31g*", False),
    "water_cart": ("O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692", 0, 0, "6-31g*", True),
}
# PySCF attaches a basis set's ECP only when asked.  vDZP, tautomator's default basis, puts
# one on every non-H atom, so at least one molecule here must carry one.
ECPS: dict[str, dict[str, str]] = {"iodomethane": {"I": "def2-svp"}}


def _stock(func: Any, *args: Any) -> Any:
    """Run a stock PCM function; skip where this PySCF's own PCM can't do the case."""
    try:
        return func(*args)
    except RuntimeError as err:
        if "Cartesian orbitals" in str(err):  # PySCF 2.10-2.12: no PCM with a Cartesian basis
            pytest.skip(f"PySCF {pyscf_version}'s own PCM: {err}")
        raise


def _mol(name: str) -> Any:
    atoms, charge, spin, basis, cart = MOLECULES[name]
    ecp = ECPS.get(name, {})
    mol = gto.M(atom=atoms, charge=charge, spin=spin, basis=basis, ecp=ecp, cart=cart, verbose=0)
    assert bool(mol.has_ecp()) == bool(ecp), f"{name}: ECP expected {ecp}, got {mol.has_ecp()}"
    return mol


def _solvents(mol: Any, model: str, max_memory: float | None = None) -> tuple[Any, Any]:
    """A stock PCM and a FastPCM with the same settings."""
    pair = (pcm.PCM(mol), FastPCM(mol))
    for ws in pair:
        ws.method, ws.eps = model, EPS
        if max_memory is not None:
            ws.max_memory = max_memory
    return pair


@pytest.fixture(scope="module")
def densities() -> dict[str, Any]:
    """A gas-phase SCF density per molecule (the comparisons need not be self-consistent)."""
    out = {}
    for name in MOLECULES:
        mol = _mol(name)
        mf = scf.UHF(mol) if mol.spin else scf.RHF(mol)
        mf.kernel()
        out[name] = mf.make_rdm1()
    return out


def _same(a: Any, b: Any) -> None:
    np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-11)


# ---------------------------------------------------------------------------
# The PySCF functions FastPCM replaces have not changed
# ---------------------------------------------------------------------------
class TestReplacedFunctions:
    @pytest.mark.parametrize("name", REPLACED)
    def test_replaced_functions_are_checked(self, name: str) -> None:
        found = installed_fingerprints()[name]
        assert found in CHECKED.get(name, {}), (
            f"PySCF {pyscf_version}: {name} has fingerprint {found!r}, which FastPCM was not "
            f"checked against, so use_fast_pcm keeps PySCF's own PCM.  If the rest of this "
            f"file passes with this PySCF, add it to CHECKED: {installed_fingerprints()}"
        )

    def test_fingerprint_ignores_comments_blank_lines_and_indentation(self) -> None:
        a = "def f(x):\n    y = x + 1  # one\n\n    return y\n"
        b = "    def f(x):\n        # a comment\n        y = x + 1\n        return y\n"
        assert fingerprint(a) == fingerprint(b)

    def test_fingerprint_sees_code_changes(self) -> None:
        a = "def f(x):\n    return x + 1\n"
        assert fingerprint(a) != fingerprint("def f(x):\n    return x + 2\n")
        assert fingerprint(a) != fingerprint("def f(x, y=None):\n    return x + 1\n")

    def test_unchecked_pyscf_keeps_its_own_pcm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(pyscf_fast_pcm, "CHECKED", {})
        pyscf_fast_pcm.unchecked_functions.cache_clear()
        try:
            mf = scf.RHF(_mol("formate")).PCM()
            assert not use_fast_pcm(mf)
            assert type(mf.with_solvent) is pcm.PCM
        finally:
            pyscf_fast_pcm.unchecked_functions.cache_clear()

    @pytest.mark.parametrize("model", ["SMD", "ddCOSMO"])  # SMD is a subclass of PCM
    def test_other_solvent_models_are_left_alone(self, model: str) -> None:
        rhf = scf.RHF(_mol("formate"))
        if not hasattr(rhf, model):
            pytest.skip(f"PySCF {pyscf_version} has no {model}")
        mf = getattr(rhf, model)()
        cls = type(mf.with_solvent)
        assert not use_fast_pcm(mf)
        assert type(mf.with_solvent) is cls

    def test_switch_keeps_the_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(pyscf_fast_pcm, "unchecked_functions", lambda: ())
        mf = scf.RHF(_mol("formate")).PCM()
        ws = mf.with_solvent
        ws.method, ws.eps, ws.radii_table = "C-PCM", 32.7, pcm.modified_Bondi * 1.1
        assert use_fast_pcm(mf)
        assert type(ws) is FastPCM
        assert (ws.method, ws.eps) == ("C-PCM", 32.7)
        np.testing.assert_array_equal(ws.radii_table, pcm.modified_Bondi * 1.1)
        assert use_fast_pcm(mf)  # a second call leaves it a FastPCM
        assert type(mf.with_solvent) is FastPCM


# ---------------------------------------------------------------------------
# Each replaced function gives PySCF's result at the same input
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("name", list(MOLECULES))
class TestSameInputSameOutput:
    def test_get_vind(self, name: str, model: str, densities: dict[str, Any]) -> None:
        stock, fast = _solvents(_mol(name), model)
        dm = densities[name]
        (e0, v0), (e1, v1) = _stock(stock._get_vind, dm), fast._get_vind(dm)
        assert "B" in fast._intermediates  # (k|mn) kept in core
        _same(e1, e0)
        _same(v1, v0)
        for key in ("q", "q_sym", "v_grids", "dm"):
            _same(fast._intermediates[key], stock._intermediates[key])

    def test_B_dot_x_any_density(self, name: str, model: str) -> None:
        """Response densities need not be symmetric; several at once."""
        mol = _mol(name)
        stock, fast = _solvents(mol, model)
        x = np.random.default_rng(0).standard_normal((2, 3, mol.nao, mol.nao))
        _same(fast._B_dot_x(x), _stock(stock._B_dot_x, x))

    def test_gradient(self, name: str, model: str, densities: dict[str, Any]) -> None:
        stock, fast = _solvents(_mol(name), model)
        dm = densities[name]
        dm = dm[0] + dm[1] if dm.ndim == 3 else dm  # as the gradient driver passes it
        _same(fast._grad_qv(dm), _stock(pcm_grad.grad_qv, stock, dm))
        expected = sum(
            f(stock, dm) for f in (pcm_grad.grad_qv, pcm_grad.grad_nuc, pcm_grad.grad_solver)
        )
        _same(fast.grad(dm), expected)

    def test_without_integrals_in_core(
        self, name: str, model: str, densities: dict[str, Any]
    ) -> None:
        stock, fast = _solvents(_mol(name), model, max_memory=0)
        dm = densities[name]
        (e0, v0), (e1, v1) = _stock(stock._get_vind, dm), fast._get_vind(dm)
        assert "B" not in fast._intermediates
        _same(e1, e0)
        _same(v1, v0)


# ---------------------------------------------------------------------------
# Whole calculations
# ---------------------------------------------------------------------------
def _scf(name: str, model: str, fast: bool, xc: str | None = None, switching: str = "SWIG") -> Any:
    mol = _mol(name)
    if xc is None:
        mf = scf.UHF(mol) if mol.spin else scf.RHF(mol)
    else:
        mf = (dft.UKS(mol) if mol.spin else dft.RKS(mol)).density_fit()
        mf.xc = xc
    mf = mf.PCM()
    mf.with_solvent.method, mf.with_solvent.eps = model, EPS
    if switching != "SWIG":
        mf.with_solvent.surface_discretization_method = switching
    if fast:
        mf.with_solvent.__class__ = FastPCM
    mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-8
    mf.kernel()
    assert mf.converged
    return mf


@pytest.mark.parametrize(
    ("name", "xc", "switching"),
    [("formate", "pbe", "SWIG"), ("methyl", None, "SWIG"), ("formate", "pbe", "ISWIG")],
)
def test_scf_and_gradient_match_stock(name: str, xc: str | None, switching: str) -> None:
    """ISWIG (PySCF 2.13 and later) only changes the surface PCM.build makes, and the
    cavity terms of the gradient FastPCM takes from PySCF."""
    if switching != "SWIG" and not hasattr(pcm.PCM(_mol(name)), "surface_discretization_method"):
        pytest.skip(f"PySCF {pyscf_version} has no {switching}")
    stock = _scf(name, "IEF-PCM", False, xc, switching)
    fast = _scf(name, "IEF-PCM", True, xc, switching)
    assert abs(fast.e_tot - stock.e_tot) < 1e-9
    assert abs(fast.with_solvent.e - stock.with_solvent.e) < 1e-9
    np.testing.assert_allclose(
        fast.nuc_grad_method().kernel(), stock.nuc_grad_method().kernel(), atol=1e-7
    )


def test_hessian_matches_stock() -> None:
    stock, fast = _scf("water", "IEF-PCM", False), _scf("water", "IEF-PCM", True)
    np.testing.assert_allclose(fast.Hessian().kernel(), stock.Hessian().kernel(), atol=1e-7)
