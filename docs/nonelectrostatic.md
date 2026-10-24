# The nonelectrostatic correction

IEF-PCM is electrostatics only. Against experiment, an electrostatics-only
continuum under-solvates anions by a median 15 kcal/mol and cations by 6 (see
the Roadmap in the README), and it leaves every logP below zero. This document
describes the correction meant to close that gap: what it computes, why it is
built the way it is, and what is still missing. Nothing is fitted yet.

The code:

- `tautomator/nonelectrostatic.py`: the integrands, the integrals, their inputs
  (`Solute`), the parameters (`Params`), the parameter-free `Descriptors` a fit
  keeps, and the file scoring saves them to (`save_descriptors`,
  `load_descriptors`).
- `tautomator/cavity_quadrature.py`: grids for the inside and the outside of
  the cavity, and the derivative of their weights.
- `tautomator/dispersion_kernel.py`: the dispersion kernel's surface sum and
  its derivatives, compiled with numba.
- `tautomator/pyscf_nonelectrostatic.py`: `solute_from_mf`, which reads the
  inputs off a converged PySCF PCM mean field;
  `single_point_with_descriptors`, the PCM single point scoring runs when it
  saves them; and `NonelectrostaticTerms`, the correction as a function of the
  density matrix: energy, potential dG/dD and explicit nuclear gradient.
- `tautomator/pyscf_nonelectrostatic_gradient.py`: the post-SCF correction's
  full nuclear gradient (Z-vector).
- `tautomator/pyscf_nonelectrostatic_scf.py`: the correction inside the SCF.
- psi4-swig-pcm's `nonelectrostatic.py` and `csrc/nonelectrostatic.cpp`: the
  same for Psi4, both modes (see Gradients below).
- On MOIST's DROP cavities (see "On MOIST's DROP cavities" below):
  `tautomator/surface_quadrature.py` (the volume integrals written on the
  surface), `tautomator/pyscf_nonelectrostatic_drop.py` (the terms, both
  models, descriptors), and psi4-swig-pcm's `nonelectrostatic_drop.py`.

---

## The model

The form follows CMIRS (Pomogaeva & Chipman; equations as collected in Herbert,
arXiv:2203.06846, eqs. 4.18–4.24):

```
G_nonelectrostatic = A * dispersion + B * exchange + short-range field term
```

| Term | Integral | Domain |
|---|---|---|
| dispersion | `rho I / (w (w + sqrt(rho_solvent)))`, with `I(r)` the damped `R^-6` interaction with uniform solvent | solute density inside the cavity, solvent outside |
| exchange | `\|grad rho\|` | outside the cavity |
| short range | `sum_i A_i (c- \|F_i\|^n- if F_i < 0 else c+ F_i^n+)` | the cavity surface |

The third term replaces CMIRS's FESR, `C |F_min|^g + D F_max^g`, which uses only
the two extrema of the normal field `F` on the surface and so describes one
hydrogen-bond acceptor site and one donor site. The surface integral is the
quantity those extrema are a limit of, and counts every strong-field site.

The code returns the parameter-free integrals (`nonelectrostatic.descriptors`);
the coefficients and exponents are applied afterwards (`Descriptors.energy`), so
a fit never recomputes them. CMIRS's published parameters belong to an
isodensity cavity and do not transfer to the switching/Gaussian sphere cavity:
fields on the sphere surface are larger, and CMIRS's FESR coefficients applied
to them give −144 kcal/mol for a glycine zwitterion.

```python
from tautomator import nonelectrostatic
from tautomator.config import solvent_electron_density
from tautomator.pyscf_nonelectrostatic import solute_from_mf
from tautomator.pyscf_runner import _build_mf

_mol, mf = _build_mf(geom, charge, "wB97M-V", "def2-QZVPPD", "IEFPCM", "water")
mf.kernel()
d = nonelectrostatic.descriptors(solute_from_mf(mf), solvent_electron_density("water"))

d.dispersion, d.exchange            # G_disp / A (dimensionless) and G_exch / B (e/bohr)
d.field_extrema()                   # (F_min, F_max), as FESR uses
d.surface_integral(3.2, -1)         # sum A |F|^3.2 over negative-field elements
d.energy(nonelectrostatic.Params(...)).total
```

The dispersion term needs the solvent's mean electron density, the liquid's
molecules per volume times electrons per molecule
(`config.solvent_electron_density`, from measured densities at 298.15 K):

| solvent | density (kg/m³) | ρ_s (e/bohr³) |
|---|---|---|
| water | 997.047 (IAPWS-95) | 0.0494 |
| 1-octanol (`octanol`) | 821.85 (median of eight measurements, 821.5–822.3) | 0.0417 |
| water-saturated 1-octanol (`woctanol`) | 831.7, with 4.9% water by mass | 0.0421 |
| n-hexadecane | 769.8 (Outcalt et al. 2010) | 0.0394 |

CMIRS uses 0.05 for water, the same number rounded; the prototype values below
were computed with it.

MNSol's octanol free energies are fitted in **water-saturated octanol**
(`woctanol`), not dry. MNSol derives most of them from the aqueous values and
log P(o/w) (the manual's worked example is phenol), and octanol–water partition
coefficients are measured between mutually saturated phases; logP, the reason
for fitting octanol at all, refers to the same phase. Water-saturated octanol
holds 4.9% water by mass and is denser than dry (831.7 kg/m³ at 298.15 K; J.
Chem. Eng. Data 57, 2221 (2012)), which puts its ρ_s 1.1% above dry octanol's.
No measured static permittivity of the wet phase turned up, so it keeps dry
octanol's 9.8629, as xtb's ALPB `woctanol` does (its fitted parameters are what
differ from `octanol`'s, and sampling and RRHO in `woctanol` use them). MNSol
does not say which of its octanol values came from dry octanol instead (gas to
octanol partitioning); the manual says most did not. The water side of logP,
octanol-saturated water, holds only a few millimolar octanol and is plain
water here.

At def2-QZVPPD the descriptors cost about a quarter of the PCM SCF they come
from (toluene: 26 s against 107 s, 8 threads).

---

## What the prototype established

These were measured in a standalone prototype before the code moved here, some
of them with SS(V)PE, which the pipeline no longer offers. The conditions are
given with each.

**The field belongs in the surface term, not the surface charge.** The apparent
surface charge density, recast as a field (`nonelectrostatic.charge_field`), is
exact for a charge at the centre of a sphere and tracks the solute's normal
field well on fully exposed elements. Near the creases where spheres meet it
does not:

| formate, HF/6-31G* | IEF-PCM | SS(V)PE |
|---|---|---|
| correlation with the normal field, fully exposed elements | 0.98 | 0.96 |
| correlation, all elements (area-weighted) | 0.96 | 0.41 |
| most negative value (normal field: −0.063) | −0.18 | −0.76 |

Part of that is physical — the continuum's surface charge really is singular
along a re-entrant crease — and for SS(V)PE part is the symmetrized operator,
whose charges are not local quantities there. Raised to a power near 3, those
few elements dominate: at HF/6-31+G*, `sum A |F|^3.2` over negative elements is
0.016 from the field and 0.33 from SS(V)PE's charges. Both are available
(`source="field"` or `"charge"`); the field is the default.

**The volume integrals need a grid that knows where the cavity is.** Cut off
with a mask on a DFT grid, the exchange integral converges only linearly in the
shell spacing and jumps as atoms move. The prototype's grid cut each ray from
each atom at the spheres it crosses and put a separate rule on every stretch.
Values below use CMIRS's water coefficients only to express the integrals in
kcal/mol (HF/6-31+G*, SS(V)PE):

| grid | formate dispersion | formate exchange | acetone dispersion | acetone exchange |
|---|---|---|---|---|
| masked DFT grid (99, 590), as the SCF uses | −3.895 | 11.790 | −6.178 | 15.839 |
| masked DFT grid (400, 2030) | −3.884 | 13.264 | −6.176 | 16.227 |
| cavity grid, Lebedev 29, 30 radial | −3.886 | 12.918 | −6.175 | 16.154 |
| **cavity grid, Lebedev 41, 40 radial (default)** | −3.887 | 12.943 | −6.177 | 16.017 |
| cavity grid, Lebedev 59, 80 radial | −3.887 | 12.934 | −6.177 | 16.045 |

The default is within 0.03 kcal/mol; the remaining error is angular, from the
crease in the surface each ray ends on.

That grid is accurate but not smooth: a ray turning tangent to a sphere adds or
removes a stretch, so the integrals have square-root kinks as atoms move (one
or two per 0.03 bohr of displacement on ethanol, 1e-6 to 1e-5 of the integral,
with the slope changing by its own size across them). No use for gradients.
`cavity_quadrature` now gives every atom a grid that moves rigidly with it,
Gauss-Legendre on [0, r_a] inside and on [r_a, infinity) outside (after
t = r_a + L (1 + x)/(1 - x), L = 0.5 bohr), and lets the other spheres in only
through a smooth (C^2) partition of unity: inside, an atom hands a point to
the others as it nears its own surface; outside, it gives up points as they
near another sphere, and has none inside one. The integrals are then C^2 in
the geometry, and the grid's derivative is the weights' (`partition_gradient`).
Against the finest grid on HF/6-31+G* densities (IEF-PCM, MMFF geometries):

| grid | formate dispersion | formate exchange | acetone dispersion | acetone exchange |
|---|---|---|---|---|
| ray-cut, Lebedev 41, 40 radial (before) | +3.9e-5 | −7.3e-4 | +1.5e-5 | −4.2e-4 |
| smooth, Lebedev 29, 30 radial | −3.3e-4 | +1.1e-3 | −3.6e-4 | −7.0e-4 |
| smooth, Lebedev 41, 40 radial | −4.5e-5 | −1.3e-3 | −2.2e-5 | +3.7e-4 |
| smooth, Lebedev 59, 60 radial | +2.6e-5 | +3.0e-4 | −6.4e-5 | +3.6e-4 |
| smooth, Lebedev 77, 60 radial | +4.5e-6 | +3.6e-4 | −1.7e-5 | −1.5e-4 |
| smooth, Lebedev 101, 80 radial | reference | reference | reference | reference |

So the change moved the descriptors by ~1e-4 (dispersion) and ~1e-3
(exchange), under 0.01 kcal/mol at fitted magnitudes: descriptors saved with
the earlier grid stay usable.

**How fine the grids need to be, in kcal/mol.** Relative 1e-3 is far tighter
than the fit needs, so the two grids were sized against the energy instead:
rough parameters (A = -0.0036, B = 0.0105, from the pilot), the density held
fixed, each grid varied alone against well-converged ones (inside 59/40,
outside 77/20), on the 25 pilot neutrals and 17 MNSol ions (wB97X-3c/vDZP;
the ions also at wB97M-V/def2-QZVPPD, with the same picture). A·D spans
-7.4 to -1.0 kcal/mol over the set, B·X 1.1 to 5.3:

| grid | max error | RMS |
|---|---|---|
| dispersion, inside 17/12 | 0.024 | 0.009 |
| **dispersion, inside 23/12 (default)** | 0.018 | 0.005 |
| dispersion, inside 29/12 | 0.005 | 0.002 |
| exchange, outside 17/12 | 0.062 | 0.022 |
| exchange, outside 29/12 | 0.031 | 0.009 |
| **exchange, outside 41/12 (default)** | 0.016 | 0.006 |
| exchange, outside 41/40 | 0.013 | 0.005 |
| **G with both defaults** | **0.013** | **0.005** |

(kcal/mol.) Radially, 12 nodes are enough in both regions (8 are not); the
exterior map length L hardly matters (0.25-2 bohr). Inside, the angular error
falls steadily with the order. Outside it is the creases between spheres: a
few 1e-2 kcal/mol that does not fall below order 41, whatever the radial
count. The defaults have ~10x fewer points inside and ~3x fewer outside than
41/40 did. With fewer radial nodes each node weighs more, so the integrals'
third derivative jumps more where a node crosses another sphere's surface (the
step is C^2): finite differences of the energy need steps of ~1e-5 bohr to
agree with the analytic gradient to 1e-8. The gradient itself is continuous.

**The integrals are smooth in the geometry; the extrema are not.** Stretching a
C–O bond of formate over ±0.02 Å in 21 steps, with surface elements appearing
and disappearing (706–710), the scatter about a quartic is 9e-7 kcal/mol for
dispersion, 6e-5 for exchange and 5e-8 relative for the surface integral. For
`F_min` it is 5e-4 relative: the extremum hops between surface elements, so the
original FESR term has a discontinuous derivative and the integral does not.

**Psi4 and PySCF agree.** psi4-swig-pcm reimplements the correction on Psi4's
basis functions and libint2; on the same calculation (formate, HF/6-31G*,
exact integrals, IEF-PCM) the three terms agree to 5e-10 relative and the
post-SCF and self-consistent energies to 2e-10 Eh.

---

## Fitting

The reference data are MNSol's absolute solvation free energies in water,
water-saturated 1-octanol and n-hexadecane; `scripts/select_mnsol_set.py` picks
the solutes and `tautomator/mnsol.py` turns the database's geometries into
SMILES. Octanol–water transfer free energies of solutes with no absolute value
in either solvent are held out as a logP test.

Each solute is run once per solvent and once in the gas phase
(`scripts/run_mnsol_set.py`), and `ΔG_solv = G_soln − G_gas` is taken between
the two ensembles. Every run holds the solute to the one species MNSol
measured (`[sampling] fixed_species`), and the gas-phase run is gas phase
throughout, sampling and RRHO included. Within the
conformer and RRHO model that is exact. A single ensemble with PCM single points
is not: gas-phase conformers miss the relaxation in solvent (ΔG too positive),
solution-phase conformers miss the relaxation in the gas phase (ΔG too
negative), and either misses conformers that exist in one phase only. It also
matches how the correction will be used, on conformers minimized in PCM.

The fit should keep each conformer's components and descriptors, not each
solute's final ΔG, and recompute the Boltzmann weights for every trial
parameter set: `G_soln(θ) = −kT ln Σ_j g_j exp(−β (G_j⁰ + ΔG_j(θ)))`. The
correction is applied after the SCF, so this costs no quantum chemistry.

### Saved descriptors

They are not part of the config schema: the fit is a one-time job, and
`scripts/run_mnsol_set.py` asks for them through `run_pipeline(...,
descriptors=[...])` (PySCF, with IEF-PCM, C-PCM or COSMO on PySCF's cavity or a
DROP cavity). Scoring then reads the descriptors off each conformer's solvated
single point and writes `<output_dir>/nonelectrostatic.npz`. The file holds
every conformer scored in solution, the ones scoring's energy window then drops
included, since the correction can bring one of them back into the window. The
ensemble.json beside it has only those the window kept.

Each further model in `descriptors` (run_mnsol_set.py's default: `IEFPCM` on
PySCF's cavity and `IEFPCM/rho-DROP`, beside its own `IEFPCM/SvdW-DROP`) adds
one more solvated single point per conformer at the same geometry, written to a
file of its own (`descriptor_file(model)`, e.g.
`nonelectrostatic_iefpcm_rhodrop.npz`) with the same gas-phase energy and RRHO,
that model's PCM energy and descriptors, and in its settings the model the
geometries came from (`geometry_solvent_model`) and any conformer that failed
there (`failed`). A fit can then choose its cavity without rerunning anything.

```python
from tautomator.nonelectrostatic import load_descriptors

conformers, settings = load_descriptors(run_dir / "nonelectrostatic.npz")
c = conformers[0]
c.free_energy                  # E_gas + ΔG_PCM + G_RRHO (hartree), no correction
c.states                       # g_j: multiplicity, times 2 for a collapsed enantiomer pair
c.descriptors.energy(params)   # ΔG_j(θ)
settings["solvent_rho"], settings["delta"]   # what the dispersion integral used
```

Each conformer has its geometry, its microstate and charge, its energy
components, the three integrals, and per surface element the area, both
normal fields and the atom whose sphere the element is on. That is about
50 kB per conformer, and on the order of 150 MB for the MNSol set. The
dispersion integral is stored at one damping length (`delta`, CMIRS's 7 bohr)
and one solvent density; changing either would mean rerunning the SCF.
Computing the descriptors adds about a quarter to scoring.

`settings["pcm_switching"]` names the PCM surface's switching function
(`config.PCM_SWITCHING`, ISWIG). Files without it were written on SWIG
surfaces: other elements and areas, so their surface terms differ a little from
ISWIG's, and their solvation energies by ~0.02 kcal/mol.

---

## Gradients, and the correction inside the SCF

Two models, both on both backends:

- **Post-SCF** (as CMIRS): G is evaluated on the converged PCM density, which
  does not see it. This is what the saved descriptors and the fit describe.
- **Self-consistent**: G and its potential V = dG/dD enter every SCF
  iteration, so the density relaxes with the correction. The energy is lower
  by the relaxation energy, second order in the correction.

The pieces are common: `NonelectrostaticTerms` gives G, V and G's explicit
nuclear gradient at fixed density matrix (basis functions, cavity grids,
surface elements and nuclei moving with their atoms). The exchange term's
|grad rho| and the surface term's |F| are regularized as x^2/sqrt(x^2 + floor^2)
(`GRAD_FLOOR`, `FIELD_FLOOR`), so that V and the gradient stay finite where they
pass through zero; the energy moves by ~1e-10.

**Post-SCF, PySCF** (`pyscf_nonelectrostatic_gradient.post_scf_gradient`): the
density's response comes from one Z-vector (CPHF) solve with the SCF's orbital
Hessian, the PCM's equilibrium response included, and the derivative Fock
matrix contracted with the response density. Its Coulomb, exchange and PCM
parts are quadratic forms, so they come exactly from PySCF's own SCF gradient
code at D +- tZ; the XC part is the kernel contraction on the SCF's grid, with
the grid's response. Closed shell (RHF/RKS), DF, LDA/GGA functionals including
(range-separated) hybrids, ECPs; meta-GGA and VV10 functionals are refused
(PySCF has no VV10 response kernel). Against finite differences of G on the SCF
density at displaced geometries (formate): 1e-7 Eh/bohr or better for HF, PBE,
wB97X and wB97X-3c/vDZP.

**Self-consistent, PySCF** (`pyscf_nonelectrostatic_scf.enable_self_consistent`):
the correction joins the PCM solvent object's energy, Fock contribution and
gradient, so `mf.kernel()`, `mf.nuc_grad_method()`, scanners and geomeTRIC
include it. Against central differences of energies at h = 4e-5 bohr
(`tests/test_pyscf_nonelectrostatic_scf.py`), gradients agree to 1e-7 Eh/bohr
for HF and 2e-7 for PBE with the grid response; PBE's residual is the
quadrature's h^2 error, and extrapolating from two steps leaves 1e-8.

**Psi4** (psi4-swig-pcm, `PCMSettings(nonelectrostatic=...)`): the same terms,
with the field integrals and their second derivatives from libint2. The
self-consistent mode adds G and V in the DDX slot; its gradient adds G's
explicit derivative. For the post-SCF gradient, Psi4 1.11 exports neither a JK
gradient for an arbitrary density, nor an XC-kernel derivative, nor ECP
derivative integrals, so the density's response is taken by finite field:
four extra SCFs with +-lambda V0 and +-2 lambda V0 on the Fock matrix, from the
previous orbitals, give it by Richardson extrapolation. Both modes agree with
energy differences to ~5e-8 Eh/bohr.

**Cost.** One geometry-optimization step (SCF from the previous density, then
the gradient) for toluene at wB97X-3c/vDZP in IEF-PCM water, 4 threads: 7.4 s
with the PCM alone; 14.0 s self-consistent (+90%: 0.27 s more per SCF
iteration, 4.4 s more in the gradient); 27.6 s post-SCF (+270%). The post-SCF
overhead is mostly the density's response, shared by all three terms: the
Z-vector solve, the XC-kernel contraction on the DFT grid with its grid
response, two DF-JK gradients for the polarization identity, and the PCM.

**What the relaxation is worth.** Self-consistent minus post-SCF energies, with
rough parameters: A and B fitted on the pilot's water and hexadecane neutrals
(with physical signs: A = -0.0036, B = 0.0105), c-, c+ set to give acetate
-15 and methylammonium -5 kcal/mol, exponent 2.5:

| solute | G | wB97X-3c/vDZP | wB97M-V/def2-TZVPPD |
|---|---|---|---|
| ethanol | -1 | -0.06 | -0.08 |
| acetic acid | -2 | -0.10 | -0.14 |
| acetate | -14.5 | **-0.97** | **-1.74** |
| methylammonium | -5 | -0.02 | -0.02 |

(kcal/mol.) Small for neutrals and the cation, but 1-2 kcal/mol for the anion,
more with diffuse functions: comparable to the accuracy a fit aims for. A
self-consistent model would have to be fitted self-consistently.

---

## On MOIST's DROP cavities

The same three terms on the surface MOIST builds for its PCM
(`solvent_model = "<model>/SvdW-DROP"` or `"<model>/rho-DROP"`; the README has
the cavities), both models, with analytical gradients, on PySCF and Psi4. Both
cavities are built on PySCF's cavity's spheres (`config.pcm_radii`), not on
MOIST's own radii; every number below is on them. A DROP surface is a set of
points s_i with outward normals n_i and areas a_i,
smooth in the geometry and, for rho-DROP, in the density. Everything is
written on those, and MOIST takes the derivatives with respect to them back to
the nuclei and the density.

**The volume integrals, by the divergence theorem** (`surface_quadrature`).
For a nucleus at c inside the cavity and any integrand q, the radial field
F(x) = e/r^2 int_r^inf q(c + t e) t^2 dt has div F = -q, so the outside
integral is sum_i a_i n_i . F(s_i): a ray from the nucleus through every
surface point, integrated from the surface outwards (Gauss-Legendre after
t = r + L (1 + x)/(1 - x)). Each atom takes its share of the integrand through
the Stratmann-Scuseria-Frisch partition, so no ray meets another nucleus's
cusp; the partition is exactly zero outside an atom's cell, and an atom's cell
factor never grows along a ray leaving its nucleus, so a surface point outside
the cell drops the pair exactly. The outside integrals (exchange, electrons
outside) are then a weighted point set; the inside one (dispersion) is an
all-space atom grid with the same partition less the outside. Checked on
acetate's rho-DROP surface against a brute-force integral over rho < 1e-3 on
a fine atom grid (the region rho-DROP bounds): 1e-5 relative at MOIST's nleb
1202, 1.3e-3 at its default 194. Two quite different partitions (SSF and
Becke) give the same integrals to 2e-5.

**Gradients through MOIST's reverse pass.** G depends on the cavity only
through (s_i, n_i, a_i), so dG/d(s, n, a) is the surface adjoint MOIST's own
models hand their cavity: its reverse pass contracts it into the nuclear
gradient (the anchor, switching and level-set field routes), and on rho-DROP
into density weights, conjugate to rho and its first two derivatives on the
grid, which give the Fock term (against d rho/dD) and the density's route in
the gradient (against d rho/dR at fixed D). MOIST 1.1.0a1 does not export the
two contractions; psi4-swig-pcm's `packages/moist` recipe adds them
(`patches/0002`, `moist_contract_cavity_surface_gradient` and `_density`,
validated against the cavity rebuilt at displaced nuclei). Energies and
descriptors work without them; gradients and the self-consistent Fock matrix
on rho-DROP refuse to run without them. Against finite differences (formate,
HF/def2-SVP): fixed-density gradients 3e-8, the potential 1e-8 relative, the
self-consistent and post-SCF total gradients 1e-8 to 9e-8 Eh/bohr, on both
cavities and both programs; Psi4 and PySCF agree to 1e-10 Eh.

**The models.** Self-consistent (`pyscf_nonelectrostatic_drop.
enable_self_consistent`; psi4-swig-pcm's `MoistSettings(nonelectrostatic=...)`
with `self_consistent=True`): G and V every iteration. On rho-DROP V includes
the surface's own response to the density. Post-SCF
(`pyscf_nonelectrostatic_drop.post_scf_gradient`): the Z-vector of the PCM
version, with MOIST's response in place of PySCF's PCM. On SvdW-DROP that is
exact, because the PCM's Fock matrix is affine in the density on a fixed
surface. On rho-DROP the surface follows the density and MOIST has no second
derivatives, so the PCM's response is differentiated numerically (central
differences at two steps, Richardson). On Psi4, the post-SCF response is the
finite field, as for psi4-swig-pcm's own PCM.

**MOIST's PCM solves its charges iteratively, to 1e-8 by default.** The SCF
density carries that noise. A variational energy does not see it; the
post-SCF G does, and its finite differences divide it by the step. The
finite-difference tests use MOIST's dense solver.

**Descriptors.** `drop_descriptors` (and scoring with descriptors on a DROP
model) gives the same `Descriptors`:
per surface point the area, the solute's normal field, the PCM's apparent
surface charge recast as a field (its correlation with the normal field is 0.97
on formate) and the atom whose sphere the point was projected from. The charges
are MOIST's apparent charges q (psi4-swig-pcm's `packages/moist` patch 0003
publishes them), as PySCF's cavity uses its own: not the potential adjoint
w_phi MOIST hands the host, which for IEF-PCM is the symmetrized adjoint and
differs from q: by 0.04% RMS (0.1% at most) on SvdW-DROP for formate, and on
PySCF's cavity, between its own q and q_sym, by 0.3-1.3% RMS (formate,
methylammonium, water). C-PCM and COSMO are stationary, so there the two are
equal. The file's settings name the charges (`charge_field`) and record the
cavity, its spheres (`drop_radii`: PySCF's cavity's, with `pcm_hydrogen_radius`;
a file without the key is from before, on MOIST's own CPCM radii), MOIST's grid
and pruning level (`drop_nleb`, `drop_prune_level`) and the quadrature
(`volume_rule`, `ray_rule`, `partition_a`).

**How fine the grids need to be, in kcal/mol.** The same 42 solutes as on
PySCF's cavity (25 neutrals, 17 ions; wB97X-3c/vDZP, density fixed), rough
parameters A = -0.0036 and B = 0.0105, and field coefficients giving acetate's
negative-field term -15 and methylammonium's positive-field term -5 kcal/mol
at n = 3 (G spans -40 to 0 kcal/mol on SvdW-DROP, -36 to 0 on rho-DROP). Each
grid varied alone against references (all-space (59, 40, 1.5), rays (40, 1.0),
MOIST's nleb 1202), max error / RMS:

| grid | SvdW-DROP | rho-DROP |
|---|---|---|
| all-space (17, 12, 1.5) | 0.092 / 0.040 | 0.093 / 0.040 |
| all-space (23, 12, 1.5) | 0.076 / 0.037 | 0.077 / 0.038 |
| **all-space (23, 16, 1.5) (default)** | 0.014 / 0.005 | 0.014 / 0.005 |
| all-space (35, 24, 1.5) | 0.013 / 0.003 | 0.013 / 0.003 |
| rays (8, 1.0) | 0.005 / 0.003 | 0.004 / 0.003 |
| **rays (12, 1.0) (default)** | 0.001 / 0.000 | 0.001 / 0.000 |
| MOIST's surface, nleb 194 (MOIST's default) | 0.065 / 0.021 | 0.062 / 0.022 |
| MOIST's surface, nleb 302 | 0.055 / 0.013 | 0.037 / 0.013 |
| MOIST's surface, nleb 590 | 0.011 / 0.005 | 0.020 / 0.006 |

(kcal/mol.  At n = 4 the same, but for MOIST's surface at nleb 194 on rho-DROP:
0.080 for methoxide.  The all-space grid's error is the dispersion integrand's
over all space, which hardly depends on the cavity.) The all-space grid needs
its 16 radial nodes; with them, MOIST's own surface is the largest error, as on PySCF's cavity, and
it falls with nleb, which also refines the PCM.

**MOIST's surface resolution**, which sets the PCM's own accuracy as much as
the correction's. The IEF-PCM energy on the same 42 solutes at a fixed density,
against nleb 1202, with the target of 0.1 kcal/mol or 1e-3 relative (worst ion
/ worst neutral, kcal/mol; solutes within the target; the correction's worst
error with the rough parameters above):

| nleb / pruning | points | SvdW-DROP | rho-DROP | within (SvdW / rho) | G worst |
|---|---|---|---|---|---|
| 194 / 4 (MOIST's default) | 1.00x | 0.25 / 0.08 | 0.25 / 0.08 | 26 / 27 | 0.065 / 0.062 |
| 302 / 4 | 1.45x | 0.13 / 0.035 | 0.12 / 0.03 | 36 / 37 | 0.055 / 0.037 |
| 350 / 4 | 1.63x | 0.13 / 0.034 | 0.12 / 0.03 | 39 / 40 | 0.044 / 0.044 |
| **434 / 4 (config.DROP_NLEB, DROP_PRUNE_LEVEL)** | 1.97x | 0.075 / 0.022 | 0.073 / 0.022 | 42 / 42 | 0.021 / 0.024 |
| 590 / 4 | 2.57x | 0.047 / 0.013 | 0.046 / 0.012 | 42 / 42 | 0.011 / 0.020 |

The worst cases are small anions (phenoxide, methoxide, the cyanomethyl
anion). Pruning level 4 is free for the energies: at 434 it drops 11-12% of
the points (15% at most) and moves the PCM energy by under 1e-6 kcal/mol and the
integrals by under 2e-6 relative (for the gradients, see below). Level 5 is as harmless up to nleb 302 but at 590 moves
the correction by up to 0.09 kcal/mol. 434 costs the PCM 2.0-2.3x its time at
194, and at def2-QZVPPD with `memory_gb = 12` the surface integrals of 73% of
the MNSol subset fit in core, against 96% at 194. With as many points as
PySCF's Lebedev order 29 (302 here), DROP's worst ion is 0.12-0.13 kcal/mol
against 0.19 on PySCF's cavity.

**The gradients converge more slowly than the energies.** Formate
(HF/def2-SVP, density fixed, the rough parameters above), the largest error of
a Cartesian gradient component against nleb 2030, in Eh/bohr:

| nleb | PCM (SvdW / rho) | correction (SvdW / rho) |
|---|---|---|
| 194 | 5.3e-4 / 6.2e-4 | 9.3e-5 / 5.6e-5 |
| **434** | 2.5e-4 / 2.9e-4 | 1.1e-4 / 1.1e-4 |
| 590 | 1.0e-4 / 1.2e-4 | 8.0e-5 / 7.3e-5 |
| 1202 | 4.3e-5 / 5.4e-5 | 7.4e-5 / 6.9e-5 |

(The largest components: PCM 8e-2, correction 6e-3 Eh/bohr.) The PCM's error
falls with nleb: at 194 it is above the GAU criterion both drivers' optimizers
converge to (maximum force 4.5e-4 Eh/bohr), at 434 about half of it, and 2030,
3074 and 5810 agree with each other to 1-3e-5. The correction's does not fall:
it is 0.5-2% of its largest component at every nleb, and 2030, 3074 and 5810
still differ from each other by 3-6e-5, on both cavities (and on SvdW-DROP
without pruning too). It is a floor of about 1e-4 Eh/bohr at these parameters (it scales with
the fitted ones), a quarter of GAU's maximum force and under the PCM's own error
at 434, so a finer grid buys the correction's gradient nothing. Most of it is the
field term's (1.6e-4 at 434 on SvdW-DROP); the exchange term's is the largest
relative to its size (20%), but that term is small here (2e-4 at most). On
MOIST's own radii, before, the floor was 2e-5. Pruning (level 4) adds features
about 1e-3 bohr wide to the correction's energy where a point's weight passes
through the pruning window: the analytic gradient follows them (7e-10 against
differences over 2.5e-4 bohr), but differences over 2e-3 bohr miss by 2e-6
(psi4-swig-pcm's formate test, which therefore steps 2.5e-4).

**What the relaxation is worth** on DROP, as on PySCF's cavity (rough
parameters, exponent 2.5, c- and c+ set on each cavity to give acetate -15 and
methylammonium -5 kcal/mol; wB97X-3c/vDZP), self-consistent minus post-SCF:

| solute | G (SvdW / rho) | SvdW-DROP | rho-DROP |
|---|---|---|---|
| ethanol | -1.27 / -1.28 | -0.048 | -0.049 |
| acetic acid | -1.97 / -1.86 | -0.076 | -0.098 |
| acetate | -15.0 / -15.3 | **-0.89** | **-0.80** |
| methylammonium | -5.5 / -5.0 | -0.016 | -0.074 |

(kcal/mol.) On rho-DROP the self-consistent density also moves the surface.

**Cost.** Toluene at wB97X-3c/vDZP (8 threads, nleb 434): on SvdW-DROP an SCF
iteration of the self-consistent correction adds 0.3 s (the normal field's map
onto the density matrix is kept with the surface), and its gradient 4.1 s,
beside the PCM's own 0.4 s per iteration and 3.7 s. On rho-DROP the surface
moves every iteration, so every iteration also needs dG/d(s, n, a) (the field's
gradient at the points and the density's Hessian on the rays): 4.4 s, beside
MOIST's own 1.5 s. The post-SCF gradient took 18 s on SvdW-DROP and 62 s on
rho-DROP (the numerical response). At nleb 194 these were 0.2 s, 2.3 s, about
3 s, 15 s and 39 s.

---

## Not done yet

- **Conformers outside the refinement window** have no scoring-level energy, so
  they have no descriptors either; refinement's 8 kcal/mol window decides
  which conformers the fit can reweight.
- **The pipeline does not use the correction yet**: no config switch puts it
  in refinement or scoring (there are no fitted parameters), and Psi4 scoring
  runs do not save descriptors.
- **On DROP cavities:** the rho-DROP post-SCF gradient's PCM response is
  numerical (MOIST has no second derivatives); the closed-shell restriction of
  the PySCF Z-vector applies as on PySCF's cavity.
- **Open-shell post-SCF gradient (PySCF).** The Z-vector is closed shell only;
  no MNSol solute is open shell. The self-consistent mode and psi4-swig-pcm
  take UHF/UKS but have no test for it.
- **An exact Psi4 post-SCF gradient** (a Z-vector in C++ against libint2 and
  libecpint), if the finite-field one is ever not good enough.
- **Speed.** At the refinement level a post-SCF step costs ~3.7x a PCM-only
  one, a self-consistent step ~1.9x. Most of the post-SCF cost is the shared
  response: one DF-JK call with both densities instead of the polarization
  identity's two, and AO screening on the XC and cavity grids, are the likely
  gains.
