# tautomator

Three-stage QM workflow for predicting macroscopic aqueous pKa values of small
organic molecules from a SMILES string. Built on RDKit + CREST/xTB +
PySCF/Psi4 (plus the OrbMol-v2 machine-learned potential as an optional
refinement driver). Python 3.13, managed with pixi.

The pipeline produces a fully-decomposed conformer **ensemble** (geometries +
energy components + Boltzmann weights) for every charge state in a requested
range. Going from that ensemble to a numeric macroscopic pKa is **not yet
implemented** — see [Status](#status).

## The pipeline

`tautomator.pipeline.run_pipeline(config)` orchestrates three stages. Between each
stage, conformers are filtered by an energy window *within* (never across)
each charge state.

| Stage | What it does | Tool | Sets |
|---|---|---|---|
| 1. Sampling | Enumerate tautomers, protonation states, stereoisomers, conformers | CREST + GFN2-xTB (ALPB water) | `electronic_energy`, `solvation_energy`, `rrho_correction` (xTB `--hess`), `multiplicity` |
| 2. Refinement | Geometry optimization + solvation + RRHO recompute on the refined geometry | PySCF, Psi4 or OrbMol-v2 (+ xtb) | overwrites electronic/solvation with refinement-level values; recomputes `rrho_correction` and `multiplicity` |
| 3. Scoring | High-level DFT single point | PySCF or Psi4 | overwrites electronic/solvation; leaves `rrho_correction` and `multiplicity` |

After scoring, `assign_weights` populates per-conformer Boltzmann weights and
the ensemble is serialized to `ensemble.json`.

RRHO (quasi-RRHO vibrational free energy) is always included. It is first
computed cheaply at sampling — an xTB Hessian (`xtb --hess`, ALPB water) on the
xTB geometry, folded into the sampling energy window so filtering ranks by free
energy. It is then recomputed once on the refined geometry, controlled by
`refinement.rrho`:

- `"xtb"` (default): GFN2 single-point biased Hessian (`xtb --bhess`, ALPB
  water) — the Spicher–Grimme SPH scheme. Cheap and robust on the non-stationary
  refined geometry; recommended.
- `"refinement"`: a Hessian from whatever optimized the geometry, at the
  refinement level — OrbMol-v2's own Hessian for an OrbMol refinement, the DFT
  backend's otherwise. More expensive and only as good as that Hessian.

Neither choice dominates, which is why the cheap default stands. Measured over
38 structures (v3 molecules 1–11, lowest conformer per charge state, at OrbMol
minima), the two error sources are the same size: moving the xTB Hessian from
gas to ALPB shifts ΔG_RRHO by 0.62 kcal/mol sd (0.44 pKa on measured
transitions), while swapping xTB for OrbMol in gas phase shifts it by 0.71
kcal/mol sd. Against a wB97X-3c gas Hessian on 10 small transitions, OrbMol is
the better *gas* Hessian by about 4x (sd 0.19 vs 0.81) — but a gas Hessian is
the wrong ensemble, and xTB is the only one of the two that can carry the
solvent. With DFT refinement in solvent (`solvation = "optimize"`),
`rrho = "refinement"` is a DFT Hessian in implicit solvent on either driver.
On the Psi4 driver it is allowed and warns loudly: Psi4 has no analytical
Hessian for these functionals, so each conformer costs ~6N gradient
calculations (analytical PCM gradients, through psi4-swig-pcm), usually more
than the optimization it follows.

Scoring never recomputes RRHO; the refinement value carries through.

Sampling and refinement each deduplicate conformers before spending Hessians on
them: a structure that is another one relabelled, or its mirror image, is
collapsed and the survivor carries the weight of both. Refinement is where this
matters most — DFT optimization relaxes a large fraction of distinct sampled
conformers onto the same minimum. Scoring does not deduplicate, since it does
not move geometries. See [docs/symmetry.md](docs/symmetry.md).

Microstates in the RDKit-first path are identified by `protomer_key`, not by
canonical SMILES. Two SMILES that describe one species with a delocalised charge
written on different atoms — a 4-substituted imidazolium, an enolate — are
otherwise computed twice and double-counted in the partition function. Radicals
and multi-component inputs are rejected at entry rather than approximated.

The same identity is read back off the geometry after sampling and after
refinement, because protons migrate between heavy atoms during minimisation —
often enough that about a quarter of conformers in the first pKa-set batch were
filed under a protomer they no longer matched. Those are re-filed under the
microstate they became, before deduplication, so they collapse against it rather
than buying a Hessian twice; a conformer that fragmented or became a species no
microstate describes is recorded in `excluded_conformers`. Every protomer of a
molecule is embedded in one heavy-atom order (`frame_atom_order`) so that
re-filing is a regrouping rather than an atom-correspondence problem. See
[docs/protomer-identity.md](docs/protomer-identity.md).

## Data model

Defined in `tautomator/types.py`. This is the most reused mental model in the
codebase; if you only read one file, read this one.

```
Ensemble
└── charge_states: dict[int, ChargeState]      # one per molecular charge
    └── microstates: list[Microstate]          # one per tautomer/protomer
        └── conformers: list[Conformer]        # one per 3D structure
```

A `Conformer.free_energy` is the sum of three optional components:

```
free_energy = electronic_energy + solvation_energy + rrho_correction
              (gas-phase DFT)    (implicit solvent) (quasi-RRHO G_vib)
```

Components are populated incrementally across stages. When a stage runs DFT
with implicit solvent, it does a second gas-phase single point on the same
geometry to keep the decomposition clean.

Two corrections affect partition-function weighting, at different levels:

- `Microstate.includes_enantiomer: bool` — true when the microstate represents
  a collapsed *configurational* enantiomer pair (R vs S), multiplying its
  weight by 2.
- `Conformer.multiplicity: float` — how many physical states this conformer
  stands for, divided by its own rotational symmetry number. Set by
  `ensemble.deduplicate_charge_state()` after every deduplication pass.

The two are disjoint by construction and never both fire on the same
reflection. See [docs/symmetry.md](docs/symmetry.md) for the full treatment —
what gets deduplicated, why the count of relabelled copies is not a weight, and
how to check a multiplicity by hand.

Boltzmann weights are normalized **across all conformers of all microstates
within a charge state**, since the macroscopic pKa depends on the full
partition function of each charge state.

`Microstate.excluded_conformers` holds conformers that left the ensemble with
their geometry and a `reason`, deliberately typed as a `Literal` so a typo is a
type error rather than a category that silently never matches. It is **not** a
complete record of every conformer that was dropped, and the asymmetry is on
purpose: it holds the ones that failed (`optimization_failed`,
`rrho_failed`, `solvation_failed`, `scoring_failed`, …), the ones whose
*identity* turned out wrong (`fragmented`, `proton_detached`, …), and the ones
the solvation pre-filter skipped (`gas_prefilter`) — but the energy-window
filters themselves drop conformers without a record. A conformer outside the
window has a complete, correct free energy and was ranked against its siblings
on it; there is nothing to diagnose, and at a 15 kcal/mol sampling window the
records would outnumber the ensemble. A conformer skipped by the pre-filter is
different in kind: it was dropped on a *cost* decision, using an energy missing
the solvation term, so the record is what lets a finished run be checked for
near misses rather than taken on faith.

## Two sampling approaches

Configured via `sampling.approach`:

- **`"rdkit_first"` (default).** Walk SMILES space first: BFS over SMARTS
  protonation/deprotonation rules → RDKit `TautomerEnumerator` → stereoisomer
  enumeration with enantiomer dedup → CREST conformer search per labeled
  microstate. Microstates carry an explicit-H SMILES, so SDF export gets real
  bond orders.
- **`"crest_first"`.** Walk physical structures first: a quick conformer
  prescreen → CREST `--tautomerize`/`--protonate`/`--deprotonate` walking
  outward from neutral charge → dedup tautomers by H-count-per-heavy-atom
  fingerprint → full conformer search. Microstates have no SMILES (SDF export
  uses `rdDetermineBonds`).

Approach 1 is the default; approach 2 is useful when you want CREST's
physics-based site detection rather than SMARTS rules.

## QM backends

`compute.driver` is the DFT backend for the whole run: all of scoring, and any
DFT in refinement. It is `"psi4"` or `"pyscf"` — never `"orbmol"`, which cannot
score. All three drivers implement the same three operations (`single_point`,
`optimize`, `frequencies`) with the same signature, so `refinement.py` and
`scoring.py` are backend-agnostic.

- **`"pyscf"` (default).** In-process Python library. Supports analytical PCM
  gradients (IEF-PCM, the default; C-PCM, COSMO) and SMD. Of the 3c composites
  it implements only wB97X-3c. The PCM models run through FastPCM
  (`tautomator/pyscf_fast_pcm.py`): PySCF's own PCM, cavity and results, with
  the per-geometry work kept between SCF iterations. A wB97X-3c/vDZP energy and
  gradient in IEF-PCM water costs 1.2-1.3x the gas-phase job, against 3.3-4.5x
  with PySCF's PCM. It replaces six PySCF functions and checks their source
  against the releases it was tested on (2.7.0-2.14.0); on any other version
  PySCF's own PCM runs.

  Most of that speedup comes from keeping the surface integrals (k|mn) in
  memory, which FastPCM does only when they fit in half of what
  `[compute] memory_gb` leaves after the memory the process already holds.
  They grow with the cube of molecule size: at wB97X-3c/vDZP they take 0.37 GB
  for acetaminophen (11 heavy atoms), 2.0 GB for lidocaine (17) and 12 GB for
  imatinib (37), and at def2-QZVPPD 10 GB for acetaminophen already. At the
  default `memory_gb = 12` every species of the pKa training set and the MNSol
  subset keeps them at vDZP, all but one at def2-TZVPPD, and 66% and 81% at
  def2-QZVPPD (16% and 41% at 4 GB); all of them at def2-QZVPPD would take
  40 GB. Without them the SCF runs at PySCF's speed
  and only the gradient stays faster: acetaminophen took 44 s with the
  integrals left out, against 17 s with them kept and 48 s with PySCF's PCM. A
  larger `memory_gb` raises the limit, if the machine has the memory. On macOS
  PySCF cannot measure the memory the process holds, so the whole half is
  available; on Linux it can, so the limit there is lower.

  The same PCM models also run on MOIST's DROP cavities
  (`tautomator/pyscf_moist.py`), in the experimental `moist` environment
  (`pixi install -e moist`, then `pixi run -e moist ...`): `solvent_model =
  "<model>/<cavity>"`, with model IEFPCM, CPCM or COSMO and cavity `SvdW-DROP`
  (a smooth solvent-excluded-like surface over the spheres of PySCF's cavity) or
  `rho-DROP` (the electron isodensity surface, rho_iso = 1e-3 e/bohr³, which
  follows the density through the SCF), e.g. `"IEFPCM/SvdW-DROP"`. The psi4
  driver runs them too (see below). Water at wB97X-3c/vDZP: −6.3 and −7.2
  kcal/mol, against −6.5 on PySCF's cavity. Both have analytical gradients, so
  refinement minimizes in them, and both take vDZP's ECPs, which MOIST's own
  PySCF code refuses. On PySCF they have no Hessian (`rrho = "refinement"` with
  `solvation = "optimize"` is refused; the psi4 driver has one). On either
  driver they have no nonelectrostatic descriptors yet. Both cavities are built
  on PySCF's cavity's spheres (modified Bondi × 1.2, `config.pcm_radii`), never
  on MOIST's own radius sets, which are up to 13% smaller on the p-block
  elements and a third off on the alkali metals, Cu and Zn (on MOIST's set,
  formate's SvdW-DROP solvation energy in water is 3.6 kcal/mol more negative,
  HF/def2-SVP). So `pcm_hydrogen_radius` sizes SvdW-DROP's hydrogens as it
  does PySCF's; rho-DROP refuses a value other than 1.1, as its surface is the
  density's contour and the radii only place its points (they move its
  solvation energies by under 0.04 kcal/mol). On PySCF, MOIST solves the PCM
  and tautomator's own code supplies its integrals, as FastPCM keeps them, and
  in bounded memory. An IEF-PCM energy and gradient on acetaminophen
  (wB97X/def2-SVP, 4 threads), against the gas-phase job:

  | | tautomator | MOIST's own PySCF code |
  |---|---|---|
  | SvdW-DROP, surface integrals kept | 1.25x | 2.9x |
  | SvdW-DROP, too big to keep | 1.7x | 2.9x, in one unbounded block |
  | rho-DROP | 3.0x | 6.6x |

  The surface integrals are kept, as with FastPCM, while they fit in half of
  what `[compute] memory_gb` leaves; otherwise they are recomputed in blocks
  twice per SCF iteration, as MOIST's own code does but in bounded memory
  (its one block for acetaminophen at def2-QZVPPD would be 13.7 GB). rho-DROP
  costs more whether or not they fit: its surface moves every SCF iteration,
  so every iteration also needs the derivatives of the potential at the moved
  points and with respect to their Gaussian widths, which a fixed surface
  needs only in the gradient.
- **`"psi4"`.** Subprocess driver, writes input decks and parses output. The
  `psi4` executable comes from a pixi environment of its own
  (`pixi install -e psi4`); tautomator never imports Psi4. Implicit solvent
  comes from [psi4-swig-pcm](https://github.com/kentgorday/psi4-swig-pcm),
  installed in that environment: PySCF's switching/Gaussian PCM (same surface,
  ISWIG switching, radii and models) with analytical gradients, so Psi4
  minimizes in solvent too, and its solvation energies match PySCF's to
  <1e-5 kcal/mol at the same level.

  A solvated job splits `[compute] memory_gb` as PySCF does: half may hold the
  surface integrals between SCF iterations (psi4-swig-pcm's `incore_mb`, kept
  only if they fit whole), and Psi4 gets the other half (`--memory`); a
  gas-phase job gives Psi4 all of it. Kept, the PCM cost phenol at
  wB97M-V/def2-QZVPPD 1.02x its gas-phase job, against 1.33x recomputed, and
  lidocaine and acetaminophen at wB97X-3c/vDZP 1.0x against 2.3x and 2.2x
  (8 threads). Psi4's own share is enough: phenol and lidocaine ran as fast at
  2 GB as at 4, and the pKa set's largest species at def2-QZVPPD as fast at
  6 GB as at 12 (that one measured on Accelerate's BLAS).

  The DROP solvent models (`"<model>/<cavity>"`, as for PySCF above) run
  through psi4-swig-pcm's MOIST host, with the same cavities, radii and
  pruning as on PySCF. The `psi4` environment has MOIST for them, which is why
  its first install compiles MOIST (see "Non-obvious design decisions");
  tautomator itself needs no MOIST then: run from the default environment. A
  run on a DROP model first checks that its Psi4 can import MOIST (a `psi4`
  environment installed before it had MOIST cannot), so a missing one stops
  the run instead of failing every conformer. Psi4 takes its frequencies as
  finite differences of the analytical gradients, so on this driver `rrho =
  "refinement"` works in MOIST's PCM too (at ~6N gradients per Hessian, as in
  any Psi4 PCM). `pcm_hydrogen_radius` works as on PySCF (on SvdW-DROP).
  Solvation energies match PySCF's on the same model to <1e-8 kcal/mol
  (formate, HF/def2-SVP, SvdW- and rho-DROP), and to ~3e-6 kcal/mol with
  vDZP's ECPs, whose integrals the two programs compute differently (their
  gas-phase energies differ by up to 2e-4 kcal/mol). A wB97X-3c/vDZP single
  point on acetaminophen (4 threads) costs 1.17x the gas-phase job on
  SvdW-DROP and 3.9x on rho-DROP, against 1.13x and 3.7x on PySCF.
  Offers all five 3c composites (wB97X-3c, r2SCAN-3c, B97-3c, PBEh-3c, HF-3c).
- **OrbMol-v2 (refinement only).** The machine-learned potential from
  orb-models (Apache-2.0 code and weights), trained on gas-phase
  wB97M-V/def2-TZVPD data (OMol25 + OPoly26), with total charge and spin as
  inputs. Selected by naming it as the refinement *method*
  (`[refinement] method = "OrbMol-v2"`); `compute.driver` stays psi4 or pyscf
  and still runs the solvation single point and scoring. There is no basis. It
  has **no implicit solvent**, so `solvation = "optimize"` is refused. Measured
  against wB97M-V (807 structures from pKa-set runs; 29 geometry pairs):
  conformer/tautomer energies within a charge state MAE 0.11 kcal/mol
  (wB97X-3c: 0.31), gas-phase deprotonation energies MAE 0.14 kcal/mol vs
  def2-TZVPD, and its minima are lower in wB97M-V energy than wB97X-3c's in
  29/29 cases. Dianions are weaker. CPU only, about 1 GB of RAM, seconds per
  optimization. See `examples/glycine_orbmol.toml`.

DFT integration grids are hardcoded: (99, 590) for the SCF grid, (50, 194)
for the VV10 NLC grid. These aren't user-tunable on purpose.

## Configuration

One TOML file, five sections (`[molecule]`, `[sampling]`, `[refinement]`,
`[scoring]`, `[compute]`), loaded by `config.load_config`. The config says
which *physics* goes into each energy term; the runners decide how to get it
(analytic vs finite-difference Hessians, grids, cavity construction). A key that
is not a setting of its section is an error, with the setting it most likely
meant or the section it belongs in, rather than ignored: a misspelled or
retired setting would otherwise leave the run on its default without a word.
The deprecated spellings are still accepted, with a warning.

A top-level `protocol` key fills in every level of theory and window at once;
anything written in the file wins over the preset. With gas-phase scoring
(`[scoring] solvent_model = "none"`), a preset's refinement is gas-phase too, as
it is without a preset, and `fast` drops its solvation pre-filter along with
the single point it skips. A gas-phase run uses no solvent anywhere: sampling
and the xtb RRHO are gas-phase as well, so its free energy can serve as the gas
end of a solvation free energy, and `[molecule] solvent` is unused.

`[sampling] fixed_species = true` takes the input SMILES as the only species:
no tautomers, no charge-separated protomers, and `charge_min = charge_max` at
its own charge. It is for reference data that fix the species, such as MNSol,
where adding forms the measurement did not see would change the answer; it
needs `approach = "rdkit_first"`.

`[scoring] nonelectrostatic_descriptors = true` also saves each solvated
conformer's nonelectrostatic descriptors, with its energies, to
`<output_dir>/nonelectrostatic.npz`, for fitting the correction (PySCF and an
analytical PCM model; see [docs/nonelectrostatic.md](docs/nonelectrostatic.md)).

| | `high_quality` | `fast` |
|---|---|---|
| sampling `ewin` | 10 | 15 |
| refinement | wB97X-3c/vDZP | OrbMol-v2 |
| refinement solvation | minimized in IEF-PCM | wB97X-3c/vDZP single point |
| refinement `ewin` | 8 | 6 |
| scoring | wB97M-V/def2-QZVPPD | wB97M-V/def2-TZVPPD |
| cost (40-molecule pKa set) | ~74 d | ~15–21 d |

`fast` agrees with `high_quality` to ~0.05 pKa at the scoring level and ≤0.01
pKa on the refinement window, on pKa-set molecules. That comparison, the
costs above and both refinement windows date from when `high_quality` refined
on gas-phase geometries (Psi4, with a solvated single point); they have not
been re-measured with solvated minimization. The basis step
QZVPPD → TZVPPD is nearly free (offset −0.20, scatter 0.08 kcal/mol over 22
transitions); `high_quality` keeps QZVPPD because that is what wB97M-V's
authors recommend. Dropping VV10 is *not* free — wB97X-D3BJ scatters 0.74
kcal/mol (0.39 pKa) — which is why neither protocol does.

**Solvation at refinement** (`refinement.solvation`) is the setting that
matters most for the window, because the refinement filter decides what scoring
ever sees:

- `"optimize"` — minimize in implicit solvent. The default on both DFT drivers.
- `"single_point"` — minimize in gas phase, then one solvated single point on
  that geometry. The default for OrbMol. For DFT this is one extra
  job, since the optimizer already produced the gas-phase energy; for OrbMol it
  is a DFT pair (solvated and gas) at `solvation_method`/`solvation_basis`,
  which default to wB97X-3c/vDZP.
- `"none"` — gas phase. Warns when scoring is solvated, because gas and
  solvated rankings genuinely differ: a 4 kcal/mol gas window cost 3.3 pKa in
  testing, and conformers inside a 6 kcal/mol *solvated* window sat up to 9.1
  kcal/mol above the gas-phase minimum.

Measured on gas-phase geometries with a solvated single point, refinement
ranked the way scoring does (Spearman 1.00 against a scoring-level reference)
and a 6 kcal/mol window kept 95% of the scored set at 0 pKa cost. Minimizing in
solvent puts refinement on the same physics as scoring, so it should do at
least as well, but the windows have not been re-tuned for it.

`refinement.solvation_prefilter_ewin` is an optional cost control for that
single point: conformers already this far above the charge state's gas-phase
minimum are skipped. It is off unless set, must exceed `ewin`, and warns if it
is within 4 kcal/mol of it. It matters for OrbMol, where optimization takes
seconds and the DFT pair dominates; for DFT refinement the single point is
~10% of one optimization at the same level, and the config says so.

The `3c` composites are a functional *and* the basis they were parameterized
with, so `config.load_config` supplies the basis and refuses any other pairing
(`wB97X-3c` with def2-TZVP is a different method, not wB97X-3c).

`load_config` warns when the scoring basis has no diffuse functions, whatever
the charge range. Scoring energies are compared across charge states, and the
deprotonations between them need diffuse functions even when no anion appears.
At wB97M-V in IEF-PCM water, def2-TZVPP put deprotonation energies above
def2-TZVPPD by 1.3–2.8 kcal/mol for neutral acids, 1.0–2.4 for cations becoming
zwitterions (66–91% of the same acid without the cation) and 0.7–0.9 for
ammonium ions; def2-SVP against def2-SVPD is 5–7 times that. The check reads
the basis *name* (`aug-`, `ma-def2-`, the calendar sets, Pople `+`, def2 `…D`),
so an augmented set it does not recognize warns too. vDZP is exempt: it has no
diffuse shells, but it came within 0.8 kcal/mol of def2-TZVPPD on the same
deprotonations, except two tetrazoles, which it put 1.6–1.9 lower. Refinement
bases are not checked: those energies only rank within a charge state, where
def2-TZVPP and def2-TZVPPD differed by at most 0.7 kcal/mol on a zwitterion
against its neutral tautomer.

`config.provenance()` records what each stage computed; `pipeline.py` writes it
into `ensemble.settings` so a checkpoint on disk says which level produced it.

## Quickstart

Needs pixi 0.71 or later (`requires-pixi` in `pixi.toml`): psi4-swig-pcm is
built from source with a backend older pixi cannot run.

```bash
pixi install            # the default environment: everything but Psi4
pixi install -e psi4    # only for compute.driver = "psi4" (compiles MOIST, for the DROP models)
pixi install -e moist   # experimental: default + MOIST's DROP cavities (compiles MOIST)
pixi run python scripts/run_pipeline.py examples/glycine_pyscf.toml
```

Outputs land in `compute.output_dir`:

```
output_dir/
├── sampling/ensemble.json     # post-stage-1 snapshot
├── refinement/ensemble.json   # post-stage-2 snapshot
└── ensemble.json              # final, with Boltzmann weights
```

Example configs in `examples/` cover both DFT backends, OrbMol-v2 refinement,
and both sampling approaches; each TOML is annotated.

For batch runs over a CSV of SMILES, see `scripts/run_pka_set.py`
(`--protocol`, `--driver`, `--resume`).

## Repo layout

Source under `tautomator/`. One-liner per module:

| Module | Role |
|---|---|
| `types.py` | Core dataclasses: `Geometry`, `Conformer`, `Microstate`, `ChargeState`, `Ensemble` |
| `config.py` | TOML loader; driver-specific defaults; cross-field validation |
| `pipeline.py` | Stage orchestration |
| `sampling.py` | Approach 1 + Approach 2 |
| `refinement.py`, `scoring.py` | Backend-agnostic stage runners |
| `pyscf_runner.py`, `psi4_runner.py`, `orbmol_runner.py` | Backend implementations (OrbMol: refinement only) |
| `geometric_logging.py` | Stops geomeTRIC reconfiguring the application's logging (PySCF and OrbMol optimizations) |
| `crest_runner.py`, `xtb_runner.py` | CREST/xTB subprocess wrappers |
| `charge_enumeration.py` | SMARTS-based BFS to enumerate charge states |
| `rdkit_utils.py` | SMILES ↔ 3D, tautomer enumeration, protomer identity, input validation |
| `protomer_geometry.py` | Protomer identity from coordinates; re-files conformers whose proton moved |
| `stereo.py` | Stereoisomer enumeration + enantiomer dedup via mirror-SMILES |
| `tautomer_dedup.py` | H-count-per-heavy-atom fingerprint (used by approach 2) |
| `thermo.py` | Quasi-RRHO free energy (Grimme 2012) |
| `conformer_symmetry.py` | Symmetry-aware conformer dedup and multiplicities, from coordinates alone |
| `ensemble.py` | Boltzmann weighting, energy-window filtering, JSON/SDF I/O |
| `xyz_io.py` | XYZ read/write, including CREST multi-frame ensembles |
| `nonelectrostatic.py` | Dispersion, exchange and surface-field integrals of the nonelectrostatic correction (unfitted; see [docs/nonelectrostatic.md](docs/nonelectrostatic.md)) |
| `cavity_quadrature.py` | Quadrature over the inside and the outside of the PCM cavity: rigid per-atom grids with a smooth partition, and its derivative |
| `pyscf_nonelectrostatic.py` | Reads the correction's inputs off a converged PySCF PCM mean field; the correction's energy, potential and explicit gradient for any density matrix |
| `pyscf_nonelectrostatic_gradient.py` | Analytic nuclear gradient of the post-SCF correction (Z-vector) |
| `pyscf_nonelectrostatic_scf.py` | The correction inside the PCM SCF (self-consistent mode) |
| `dispersion_kernel.py` | The dispersion kernel's surface sum and its derivatives (numba) |
| `mnsol.py` | MNSol geometries to checked RDKit structures and SMILES |

Scripts (`scripts/`):

- `run_pipeline.py` — single molecule.
- `run_pka_set.py` — batch over a CSV with per-molecule subdirectories
  and resume support.
- `select_pka_set.py` — the pKa evaluation set: diversity-based curation from
  the IUPAC Dissociation-Constants dataset. Nothing is fitted to it.
- `report_pka_set.py` — predicted against experimental pKa for a run over that
  set, with the transitions whose ion is also in the MNSol set reported apart.
- `select_mnsol_set.py` — MNSol solutes and reference free energies in water,
  water-saturated 1-octanol and n-hexadecane, for fitting the nonelectrostatic
  correction. Neither MNSol nor its output (`mnsol_set.csv`) is tracked, as
  for the IUPAC data: both are distributed under their own licenses.
- `run_mnsol_set.py` — runs that set: each solute in every medium its
  reference values need (water, `woctanol`, hexadecane, gas phase), one fixed
  species at its own charge, with per-run failure isolation and resume. Each
  solvent run saves its conformers' nonelectrostatic descriptors.
- `approach1_rdkit_first.py`, `approach2_crest_first.py` — sampling-only
  entry points (no DFT).

## Development

```bash
pixi run check          # ruff lint + format-check + mypy --strict
pixi run lint
pixi run format
pixi run typecheck
pytest                  # excludes external-tool tests by default
pytest -m slow          # tests that shell out to xtb/crest
```

Pre-commit hook runs ruff. Mypy is strict.

## Non-obvious design decisions

The set of things that look weird and aren't. Read before "fixing":

- **Psi4 has a pixi environment to itself** (`psi4`, beside `default`).
  tautomator only ever launches the `psi4` executable, so the two environments
  are solved separately and share no versions. In one environment they
  could not move: Psi4 after 1.10.0 is built against pybind11-abi 11 and
  PySCF's xcfun against abi 4, which held Psi4 at 1.10.0 and libxc at 7.0.
  `psi4_runner.psi4_executable` takes the `psi4` in the environment next to
  the running interpreter's (`.pixi/envs/psi4`) before any on `PATH`, so the
  locked one is not shadowed, and launches it with its own `bin` first on
  `PATH`: Psi4 finds `mctc-gcp` there, and r2SCAN-3c fails without it.
  `refine()` and `score()` look the executable up when they pick the driver,
  so a Psi4 that is not installed stops the run instead of failing every
  conformer in turn. Keep it that way round: don't `import psi4` in tautomator.
  Besides the ABI conflict, running Psi4 as a separate program is what keeps
  tautomator's license independent of Psi4's LGPL-3.0.
- **MOIST comes from psi4-swig-pcm's repository**, in an experimental `moist`
  environment: the default environment plus MOIST, in the same solve group, so
  both have the same versions. There is no Psi4 in it: MOIST has no conda-forge
  or PyPI package, and the rattler-build recipe that builds it lives in
  psi4-swig-pcm (`packages/moist`), so that both repositories build MOIST from
  one recipe. `pixi.lock` pins the commit, as for `psi4-swig-pcm` itself, and
  `pixi update moist` moves it. Change the recipe there, not here; its README
  ("The moist package") lists what it must keep for tautomator: the same
  Python; BLAS/LAPACK from the environment (MKL on linux-64, OpenBLAS on
  osx-arm64), by building against the netlib variants of `libblas`/`liblapack`;
  and MOIST's Fortran libraries linked statically into its extension and not
  installed, so they cannot clash with the `mctc-lib` xtb brings in. Its
  `tests/test_moist_build.py` checks the build. The `psi4` environment has
  MOIST too, for the DROP models on the psi4 driver, rather than a second Psi4
  environment of its own: every Psi4 job of a run then comes from one
  environment, so a solvation energy never subtracts energies from two
  installs that could have drifted apart, and a separate one would isolate
  nothing (it would have to share the `psi4` environment's solve group, so
  MOIST's dependencies would constrain that one anyway). The price is that
  the first `pixi install -e psi4` compiles MOIST, as the first `pixi install
  -e moist` does (about 9 minutes on 4 cores): once for both, as the two get
  the same build.
- **`xtb` is floored at `>=6.7.1` in `pixi.toml`.** Pressure on `mctc-lib`
  has made the solver fall back to the ancient `xtb` 6.4.1 (the last release
  with no `mctc-lib` dependency) before; that was `gcp-correction` 2.3.2
  (build 2), which now lives in the `psi4` environment. xtb 6.7.1 build >=4
  also fixes a Fortran format-string bug
  ([xtb#1332](https://github.com/grimme-lab/xtb/issues/1332)).
- **`crest` is pinned to `2.12.*`, not 3.x.** The CREST 3.x rewrite moved
  optimization/MD/MTD in-process, and on macOS its conformer search collapses
  every ensemble to a single structure — reproduced as n-pentane giving 1
  conformer on 3.0.2 vs 6 on 2.12, on both osx-arm64 and osx-64-under-Rosetta
  (so it's the 3.x source, not arch codegen), independent of threads/`--ewin`/
  TOML-vs-CLI; the 3.x optimizer itself is fine, the iMTD-GC pipeline is not.
  See [crest#407](https://github.com/crest-lab/crest/issues/407),
  [#411](https://github.com/crest-lab/crest/issues/411),
  [#412](https://github.com/crest-lab/crest/issues/412). 2.12 drives the
  external `xtb` binary and works; pinning it moves only `crest` (xtb and all
  DFT deps stay put). Consequence: `single_point()` and `frequencies()` call
  `xtb` directly, since CREST 2.x has no `--sp` run mode.
- **wB97X-D4 / wB97X-D4rev / wB97X-3c are built by `pyscf_runner.py::_set_method`,
  never passed to PySCF by name.** All three are the reparameterized wB97X-V
  exchange-correlation (libxc 466) with VV10 off and D4 on; PySCF's native
  `wb97x-d4` means the original 2008 wB97X (libxc 464) plus D4. How the dftd4
  parameter name reaches PySCF depends on its dispersion dispatch, which PySCF
  2.13.1 rewrote (PR #3209), so `_d4_colon_syntax` probes for it: on older PySCF,
  `_register_d4_composites` registers internal XC names in PySCF's dispatch
  tables; from 2.13.1 nothing is registered and the name goes in
  `mf.disp = "d4:<name>"` (registering there would be wrong — XC_MAP is keyed
  on the parsed functional, and 2.14 has a native `wb97x-d4` of its own). Both
  routes were verified (2.11, 2.13.1, 2.14); the tests build composites through
  `_set_method`, so they exercise whichever route the installed PySCF takes.
- **The D4 parameters of those composites are pinned** (`_D4_PARAMS`, installed
  by `_pin_d4_parameters`). PySCF evaluates D4 with the libdftd4 bundled inside
  pyscf-dispersion — not the conda `dftd4` Psi4 uses — and its parameter names
  move between releases: in pyscf-dispersion 1.3.0's copy, `wb97x` is the fit
  for the 2008 functional, which is what wB97X-D4 silently got before the pin.
  The pin sets the numbers (with the ATM three-body term) through the dftd4 C
  API; pyscf-dispersion's own `set_param` passes the wrong arguments. Don't
  bypass either.
- **OrbMol-v2 is loaded on CPU, in float32-high, from a checksummed file.**
  orb-models 0.7.0 cannot target MPS (it calls `.cuda()` for any non-CPU
  device, and its Warp neighbour list has no MPS backend), and at ~20 atoms CPU
  is faster anyway. The precision was float64, on the theory that geomeTRIC's
  step control and the finite-difference Hessian both need a smooth surface;
  measured, that costs 1.8x and buys nothing — 3% more gradient calls to
  converge, energies within 0.042 kcal/mol, geometries within 0.02 Å,
  frequencies within 3.4 cm⁻¹, no spurious imaginary modes either way, and
  quasi-RRHO within 0.056 kcal/mol (≤0.009 pKa on a measured deprotonation).
  orb-models sets torch's global default dtype, so it is one precision per
  process. The weights come from Hugging Face rather than orb-models' default
  S3 URL (same file, ~1000x faster here) and their sha256 is checked on every
  load; `TAUTOMATOR_ORBMOL_WEIGHTS` points at a local copy. Once downloaded they
  are read from the cache without asking the server again — the checksum
  already identifies the file, and a rate-limited revalidation request used to
  fail the load with the weights on disk. Refinement loads the model before
  its first conformer, so a model that cannot be loaded stops the run instead
  of being recorded as every conformer's optimization failure.
- **orb-models' dependencies are bent in `pixi.toml`.** It pins
  `dm-tree==0.1.8`, which has no py3.13 wheel and whose conda-forge builds
  conflict with current pytorch, so the pin is overridden (orb-models only uses
  `tree.map_structure`/`flatten`). torch comes from conda-forge and is held at
  2.8/2.9 because newer builds need pybind11-abi 11 while PySCF's xcfun needs
  abi 4; on linux-64 it is the MKL build, which moves libblas to MKL.
- **BLAS/LAPACK is conda-forge's default, not Apple's Accelerate.** On
  osx-arm64 that is OpenBLAS (its OpenMP build, the one pytorch accepts); on
  linux-64 the pytorch build moves it to MKL. numpy, scipy, PySCF, Psi4, xtb,
  torch and MOIST all link the generic `libblas`/`liblapack`, so `pixi.toml`
  names no variant. Accelerate (`libblas = { build = "*_newaccelerate" }` or
  `"*_accelerate"`) was faster — OrbMol-v2 gradients 3.7x, a PySCF
  wB97M-V/def2-TZVPPD Fock build 1.4x, a Psi4 wB97X-3c gradient job 1.1-1.5x,
  PySCF at def2-QZVPPD 1.05x (M4 Pro, 4 threads, a 20-atom molecule) — but its
  Cholesky factorization of the upper triangle (`dpotrf` with `uplo = 'U'`) is
  broken on macOS 27.0.1, in both of its LAPACK interfaces, which share the
  code. For orders of about 1450-5500 it reads and writes outside its matrix
  (Guard Malloc faults inside Accelerate's `libLAPACK.dylib`; lower-triangle
  `dpotrf`, `getrf`, `syevd` and the other routines checked are clean), and
  the heap it corrupts crashes Python later: a loop of 30 such factorizations
  segfaulted in half its runs. Through the newer interface it also fails, or
  returns a wrong factor, for orders 2048-3712. scipy 1.18's
  `cholesky(lower=True)` and `cho_factor(lower=True)` take that path, so
  psi4-swig-pcm's check of every cavity of more than about 12 atoms and PySCF's
  density fitting of molecules above about 30 atoms (metric, J/K and gradient)
  call it. conda-forge does not pair Accelerate's BLAS with another LAPACK.
  BLIS is not an option either (PySCF gets a NaN Fock matrix), nor pthreads
  OpenBLAS (pytorch excludes it).
- **D4 composite Hessians fall up to wB97X-V** (`_HESSIAN_FALLBACK`). PySCF
  has no analytical D4 Hessian, but VV10 *is* analytical. Vibrational
  frequencies are insensitive to the dispersion correction, so this is a free
  speedup.
- **Psi4's PCM is psi4-swig-pcm, not PCMSolver.** Psi4's built-in PCMSolver
  has no gradients (Psi4 then takes finite differences of energies), and its
  GePol cavities failed its own validity checks — Gauss' theorem on the nuclear
  surface charge, positive-definiteness of the single-layer operator — on 324 of
  941 conformers of the first pKa-set batch. Those checks are patched to
  warnings in the build used here, so Psi4 exited 0 with meaningless energies
  (one off by 22,000 kcal/mol), and keeping them out took a cavity probe and a
  tessera-area search before every job. psi4-swig-pcm uses PySCF's smooth
  switching/Gaussian surface instead, which has no tessellation to fail and
  showed no failure on any of the 941. It still refuses a cavity whose S matrix
  is not positive-definite (the job fails and the conformer is dropped), and
  `_run_psi4` passes on its warning when the nuclear surface charge misses
  Gauss' law by more than 1%. It is LGPL-3.0 like Psi4, so it lives in a
  repository of its own; psi4_runner's input decks import it, and
  `psi4_swig_pcm.enable()` routes Psi4's own `energy`, `optimize` and
  `frequency` through it (frequencies by finite differences of the analytical
  PCM gradient).
- **Geometry optimizer fallback (PySCF).** TRIC internal coordinates first
  (100 steps); on non-convergence, restart in Cartesian coordinates from the
  last geometry (another 100 steps). Analogous to Psi4 optking's
  `dynamic_level=1`.
- **Non-converged optimizations are kept**, with `refinement_converged=False`.
  The last-step geometry is usually good enough for conformer screening.
  Conformers that *raise* (e.g. SCF failure) are dropped with a warning.
- **Solvation is decomposed via a second gas-phase SP** at every stage that
  uses implicit solvent. This keeps `electronic_energy` and
  `solvation_energy` cleanly separated even though most QM codes return only
  the total.
- **Sampling solvent is hardcoded to ALPB water** and not user-exposed.

## Roadmap

tautomator started as a pKa tool, but most of what it does is not specific to
pKa. It turns a SMILES string into the right set of species — tautomers,
protomers, zwitterions, stereoisomers — and a deduplicated, symmetry-weighted
conformer ensemble for each, refined and scored with checks that every structure
is still the species it is filed under. That machinery is meant to serve two
kinds of work:

- **Solution properties from one run.** pKa, solvation free energy, logP and
  logD have the same shape: enumerate species, narrow the conformers, build a
  partition function per species set and solvent, read off the observable. The
  observables a run asks for will decide which charge states and solvents it
  computes. logD needs no calculation of its own: it is the neutral form's logP
  combined with the pKa populations.
- **Force-field parameterization.** Bespoke SMIRNOFF parameters, exported to
  Amber, GROMACS or OpenMM, for the protomer that dominates at the pH of
  interest, which the pKa machinery decides. It reuses the conformer pools and
  adds fragmentation by GFN2 bond orders, torsion scans, ML-potential sampling
  and NEB paths between neighbouring conformers (all rescored with DFT), RESP2
  charges with virtual sites, and valence fitting with smee/descent.

Both rest on **one solvation model on every engine**: the same smooth
switching/Gaussian cavity (ISWIG switching, `config.PCM_SWITCHING`) and IEF-PCM
in PySCF and in Psi4 (through
[psi4-swig-pcm](https://github.com/kentgorday/psi4-swig-pcm), LGPL-3,
installed only in the `psi4` environment).

IEF-PCM is electrostatics only, and that is not good enough for ions in water.
Water solvates anions and cations through strong charge-transfer and
hydrogen-bonding interactions that a dielectric continuum cannot represent, and
these do not cancel between protonation states: one side of every pKa
transition is an ion. Against experiment, an electrostatics-only continuum
under-solvates anions by a median 15 kcal/mol (about 11 pKa units) and cations
by 6, against about 1 for neutrals (Pomogaeva & Chipman's set of 372 solutes).
The same missing terms decide transfer between solvents: without them every
logP comes out below zero, n-hexane's included.

The missing part is a correction in the style of CMIRS: dispersion and
exchange integrals, plus a surface integral of the solute's field at the
cavity. The surface integral stands in for FESR's two field extrema and carries
the charge-transfer and hydrogen-bond term. The integrals are implemented
(`nonelectrostatic.py`; see [docs/nonelectrostatic.md](docs/nonelectrostatic.md))
but not fitted: they will be fitted on our own cavity, because CMIRS's
published parameters do not transfer to it.

Next, in order:

1. **IEF-PCM gradients in refinement on both backends** (done): psi4-swig-pcm
   for Psi4, and FastPCM for PySCF, whose stock PCM tripled the cost of an SCF.
   Solvated minimization works on Psi4 and stays cheap on PySCF while the
   surface integrals fit in memory (see the PySCF driver above).
2. **Run MNSol solutes** (≤10 heavy atoms, ≤2 rotatable bonds; 417 solutes,
   103 of them ions; `scripts/select_mnsol_set.py`) in water, water-saturated
   1-octanol (`woctanol`) and hexadecane, and in the gas phase.
3. **Fit the non-electrostatic correction** on that set.
4. **Evaluate pKa** on the IUPAC set with IEF-PCM plus the correction.
   Nothing is fitted to it. Transitions whose ion is also in MNSol are scored
   apart: MNSol derived those ions' solvation energies from their pKas.
5. **OrbMol-v2 with ALPB in refinement.** In the gas phase OrbMol turns
   zwitterions back into neutral forms. With an implicit solvent acting on its
   latent charges, its minima land within 0.06 Å of DFT+PCM on the molecules
   tested so far, and ALPB costs about a millisecond per geometry.
6. **logP and logD.**
7. **Force-field parameterization.**

## Status

What's implemented:

- Three-stage pipeline end-to-end, both sampling approaches, both QM backends.
- Energy decomposition, quasi-RRHO with frequency scaling, sigma_rot
  detection, enantiomer multiplicity.
- Batch runner with per-molecule failure isolation and resume.
- pKa evaluation-set selection from IUPAC dissociation constants.
- The nonelectrostatic correction (unfitted), saved per conformer at scoring,
  with analytic nuclear gradients both post-SCF and self-consistent, on PySCF
  and on Psi4 (psi4-swig-pcm), and the MNSol fitting-set selection.

What's not yet here:

- **No pKa step in the pipeline.** It stops at ensembles and
  Boltzmann-weighted free energies. pKas come from the direct cycle in
  `scripts/report_pka_set.py`, with the proton's free energy fixed, not fitted.

## License

Apache-2.0, except `tautomator/pyscf_moist.py`, which adapts MOIST's code and is under
MOIST's MPL-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).

That covers tautomator only. Its dependencies are installed separately (pixi
fetches them from conda-forge and PyPI) and keep their own licenses. Psi4, xtb
and CREST (LGPL-3.0) run as separate executables. Of the libraries tautomator
imports, most are Apache-2.0, BSD or MIT (PySCF, RDKit, PyTorch, orb-models,
geomeTRIC); ASE is LGPL-2.1, and pyscf-dispersion bundles the LGPL-3.0 dftd4
and s-dftd3 libraries.
