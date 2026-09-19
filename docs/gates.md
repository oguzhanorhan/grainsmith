# grainsmith QA gates (G1–G26)

QA gates are the **user interface of the physics constraints**: every gate is a
construction-time check that either *raises* (hard) or *records a WARN row*
(warn) in `summary.csv`. A hard gate aborts the run with an actionable message;
a warn gate completes the run but flags a quality concern. The CLI prints
`OK: N/N QA gates passed` or `WARN: k/N ... (failed: …)` at the end of
`generate`.

Each gate's threshold lives in `constants.py` (no bare numeric thresholds
elsewhere — a project mandate) or in the schema; this page names the constant so
the value and its justification are traceable. The default ranges below are the
shipped defaults; thresholds set per-run via the config are noted.

## Summary

| gate | check | kind | threshold (constant / field) |
|---|---|---|---|
| G1 | config schema + 29 cross-field rules | hard | pydantic `extra='forbid'` + `config/resolve.py` |
| G2 | spglib symmetry round-trip of the expanded crystal | hard | ITA number must round-trip (`symprec`) |
| G3 | tessellation volume sum = box volume | hard | flat: `VOL_REL_TOL = 1e-6`; voxel: 1 surface voxel layer |
| G4 | per-cell Euler characteristic V−E+F = 2 | hard | exact (combinatorial) |
| G5 | grain connectivity + no empty cells | hard (warn for imports; repair-and-report for `perturbed_distance`) | `voxel_import.strict_connectivity`; `perturbed_distance` → see G19 |
| G6 | warp bijectivity guards (clip + gradient) | hard (n/a for `perturbed_distance`) | `WARP_GRAD_MAX = 0.5`, `amplitude ≤ min_seed_distance/4` |
| G7 | post-overlap minimum interatomic distance | hard | overlap `cutoff` (default `0.85*d_nn`) |
| G8 | final atom count vs theoretical | warn > 2 %, **fail > 5 %** | hard-coded 2 % / 5 % bands |
| G9 | composition drift vs nominal stoichiometry | warn | warn > 1 % per species |
| G10 | LAMMPS data file re-parses to the same system | hard | exact count + bounds round-trip |
| G11 | re-measured grain-volume error vs SDOT target | hard | `size_distribution.vol_tol` (default `1e-3`) |
| G12 | MDF–target χ² after annealing | warn | `mdf_target.chi2_max` (default `0.5`) |
| G13 | back-estimated Hurst vs target (`perturbed_distance` only) | warn | `HURST_G13_TOL = 0.15` |
| G14 | single-crystal box/lattice commensurability | warn | `COMMENSURATE_TOL = 1e-8` |
| G15 | re-measured phase volume fractions | warn | granularity bound `V_max / V_box` |
| G16 | GB curvature sanity: fraction of degenerate level-set samples | warn | `CURV_G16_DROP_TOL = 0.05` |
| G17 | dopant composition + enrichment vs targets | warn | warn > 1 pp drift or E off by > 15 % |
| G18 | dopant min-distance re-verification | hard | any pair < min_distance fails |
| G19 | `perturbed_distance` G5-repair reassigned-voxel fraction | warn | `REASSIGNED_FRACTION_WARN_TOL = 0.02` |
| G20 | `perturbed_distance` box-counting roughness index (diagnostic; not a converged, large-effect-size fractal-dimension certificate below ~5.5 synthesis-band octaves) | diagnostic (never trips) | n/a (reported; no threshold — see below) |
| G21 | per-grain Gauss-Bonnet face-interior residual (`gb_curvature` only) | warn | `CURV_G21_GAUSS_BONNET_TOL = 4π` |
| G22 | volume-weighted ODF fidelity (re-measured drift + kernel-smoothed bias vs. a null) | warn | `mdf_target.odf_drift_max` (no default cap); kernel block memory-gated |
| G23 | angle-window vs true-CSL Sigma3 area fraction | diagnostic (never trips) | n/a — informational, no threshold |
| G24 | final (post-warp) per-grain volume fidelity vs. SDOT targets | warn | `WARP_VOLUME_G24_TOL = 0.10` |
| G25 | texture-component fidelity: configured weight vs. realised count/volume fraction, sampler-term and annealer-term TV, by `component_weight_basis` (`odf_components` only) | diagnostic (never trips) | n/a — informational, no threshold |
| G26 | atomistic ODF-weighting discretisation floor: tessellation-volume weight vs. atom-count weight | diagnostic (never trips) | n/a — informational, no threshold |

`d_nn` = ideal-crystal nearest-neighbour distance.

## G1 — Configuration schema + cross-field rules (hard)

**Checks.** The YAML is parsed (`yaml.safe_load` only) and validated against the
pydantic schema (`docs/config_reference.md`). Every model is `extra='forbid'`, so
a misspelt key is an error, not a silent no-op. Beyond single-field validation,
`config/resolve.py` applies 29 numbered cross-field rules (e.g. *exactly one of
`crystal` or `phases`*; *`vacuum` needs a non-periodic axis*; *`csl` requires a
cubic point group*; SDOT feasibility — smallest target volume above the
floor).
**Trips when.** Unknown key, wrong type, out-of-range value, or a violated
cross-field rule. **What to do.** Read the message — it names the field/rule and
the offending value. Cross-reference `docs/config_reference.md` for the field's
type, default and constraints.

## G2 — Space-group round-trip (hard)

**Checks.** The Wyckoff orbits are expanded to a full unit cell, then handed to
spglib; the detected ITA space-group number must equal the requested one
(`crystal/verify.py`). This proves the generated motif actually has the claimed
symmetry rather than trusting the input.
**Trips when.** Wrong Wyckoff coordinates, a free Wyckoff parameter left
unsubstituted, an incompatible `setting`, or coincident sites across distinct
Wyckoff positions (which would silently overlap atoms).
**What to do.** Verify the space group number, the Wyckoff `coords` (substitute
free parameters with numbers), and the `setting` (origin choice / R vs H). Use
`grainsmith info --sg <N>` to inspect the group.

**`crystal.cif` path.** A CIF-derived structure (`crystal.cif: {file: ...}`,
`crystal/cif.py`) runs through this SAME gate: spglib detects the file's true
space group at `symprec`, the symmetry-inequivalent Wyckoff sites are
expanded from spglib's own standardized cell, and the result is re-verified
right at config-resolution time. Note that the `requested_sg` shown in
`summary.csv` is NOT the CIF's own declared number — for the CIF path it is
set to the spglib-DETECTED group (a mismatch between the CIF header and
spglib's detection only WARNs; it is not a G2 failure).

## G3 — Volume sum (hard)

**Checks.** Σ grain volumes must equal the box volume. Flat geometry enforces
this at construction to `VOL_REL_TOL = 1e-6` (the Qhull pruning certificate);
curved/voxel geometry compares the voxelized grain-volume sum to the box within
one surface voxel layer (`gate_g3_voxel`).
**Trips when.** A tessellation bug or (voxel) too coarse a grid near the box
surface. **What to do.** For voxel geometry, raise `analysis.voxel_grid`.

## G4 — Euler characteristic (hard)

**Checks.** Every flat Voronoi cell is a closed convex polyhedron: V − E + F = 2.
Enforced combinatorially at construction (`flat.py`). **Trips when.** A
degenerate face/vertex from coincident seeds. **What to do.** Perturb seeds
(change `seed.value`) or `min_seed_distance`.

## G5 — Connectivity + empty cells (hard; WARN for imports; repair-and-report for `perturbed_distance`)

**Checks.** Each grain is non-empty and (for generated tessellations) connected.
For `voxel_import`, external microstructures may legitimately contain
disconnected or wrap-spanning grains, so G5 is **demoted to WARN** by default;
set `boundaries.voxel_import.strict_connectivity: true` to hard-fail.
**What to do (import WARN).** Decide whether fragmentation is physical; if not,
clean the label field or set strict mode.

**`perturbed_distance` path.** This method's level-set assignment rule can
locally fold space near a boundary (§5b, `docs/physics.md`), which can
produce voxel-scale fragmentation that a hard construction-time failure
would be too blunt an instrument for. Instead of raising, the constructor
**repairs** any disconnected fragment by iterative majority-vote
reassignment on the analysis voxel grid (converges or raises
`TessellationError` on non-convergence — repair itself is never silently
partial) and reports what fraction of voxels needed it as
`reassigned_fraction`, surfaced by gate **G19** below. G5 itself always
reports `passed=True` for this method (repair-and-report, not pass/fail);
G19 is the actual severity signal to watch.

## G6 — Warp bijectivity (hard; n/a for `perturbed_distance`)

**Checks.** The domain warp must not fold space: the displacement gradient norm
stays below `WARP_GRAD_MAX = 0.5` and the per-grain clip `A_clip` keeps cells
disjoint; the admissible amplitude is tied to `min_seed_distance/4`. Enforced
at construction (`warp.py`).
**Trips when.** `amplitude` too large for `correlation_length`.
**What to do.** Reduce `boundaries.curved.amplitude`, or increase
`correlation_length`.

**Gaussian spectrum only.** `warp` accepts `spectrum: gaussian` exclusively —
`spectrum: self_affine` is a hard `ConfigError` for this method (`resolve.py`
Rule 8a; also enforced at the `WarpTessellation.__init__` API level), not a
G6 variant: a coordinate diffeomorphism cannot produce a genuinely self-affine
boundary regardless of Hurst exponent, so the combination was removed
outright rather than given its own G6 trade-off (`docs/physics.md` §5b). Use
`boundaries.curved.method: perturbed_distance` for a self-affine boundary
instead — see that method's own guards below.

**`perturbed_distance` path.** G6 reports `n/a` for this method: there is no
coordinate diffeomorphism to check bijectivity of — the level-set rule folds
space by design (that is the whole point; see `docs/physics.md` §5b for why
a bijective map like `warp` cannot produce a genuinely self-affine boundary in
the first place). Two different guards take G6's place: a seed-containment
pre-flight check at config-resolve time (`amplitude ≤
PERTURBED_DISTANCE_SAFETY · min_seed_distance / (2 · ETA_CLIP)`, i.e.
`min_seed_distance/12`, `resolve.py` Rule 7b) and an **exact** post-hoc
seed-ownership check at construction (`ConfigError` if any grain's own seed
is captured by a competing grain's perturbation) — see `perturbed.py` and
the design note referenced in `docs/physics.md` §5b.

## G7 — Post-overlap minimum distance (hard)

**Checks.** A fresh PBC neighbour query on the FINAL structure proves no atom
pair is closer than the overlap `cutoff` (default `0.85*d_nn`); `measured` =
number of violating pairs. **Trips when.** Overlap removal disabled/too small a
cutoff, or a fill bug. **What to do.** Enable `boundaries.overlap_removal` or
raise its `cutoff`.

## G8 — Atom count (warn > 2 %, fail > 5 %)

**Checks.** |N_final − N_est| / N_est with N_est = ρ_atom·V_box − N_deleted.
WARN above 2 %, **hard-fails above 5 %**. **Trips when.** Unexpectedly large GB
deletions, a density/occupancy mismatch, or a fill bug. **What to do.** Inspect
the overlap deletion count in `summary.csv`; check occupancy and lattice
parameters.

## G9 — Composition drift (warn)

**Checks.** Per-species atom-fraction drift after deletions vs the nominal
stoichiometry; WARN above 1 % per species (never hard-fails). For alloys
(`occupancy` dicts) and multiphase the nominal is the phase/occupancy-weighted
mixture. **What to do.** If drift matters, avoid `midpoint_merge` (forbidden for
alloys anyway) and prefer `delete_shallower`; report the drift in the paper.

## G10 — LAMMPS round-trip (hard)

**Checks.** The written LAMMPS data file is re-parsed and must reproduce the atom
count and in-box coordinates. **Trips when.** A writer/precision bug.
**What to do.** File a bug — this gate guards output integrity.

## G11 — Grain-volume targeting (hard)

**Checks.** With `grains.size_distribution` set, the SDOT-fitted Laguerre cell
volumes are **re-measured** on the final tessellation; max relative error must be
≤ `vol_tol` (default `1e-3`). **Trips when.** The target spread is infeasible for
the seed set, or too few Newton iterations. **What to do.** Increase
`size_distribution.max_iter`, reduce `sigma_log`, or increase `grains.number`.
(measured: typical convergence in 3–6 iterations.)

## G12 — MDF target (warn)

**Checks.** After assignment annealing, the symmetric χ² distance between the
re-measured area-weighted misorientation distribution and the target must be
≤ `chi2_max` (default `0.5`, covering the multinomial noise floor
≈ n_bins/n_pairs). **Trips when.** Too few `annealing_steps`, or a target the
fixed orientation set cannot reach. **What to do.** Increase `annealing_steps`;
note that the attainable Σ3 area fraction is bounded by the twin content of that
fixed orientation set (the assignment annealing is **conserved-texture**: the
NUMBER-weighted orientation distribution is invariant by construction, the
VOLUME-weighted ODF is not — see `docs/physics.md` §4).

## G13 — Self-affine Hurst (warn; `perturbed_distance` only)

**Checks.** The Hurst exponent back-estimated from the generated per-color η
field(s) (radial PSD log-log fit) must be within `HURST_G13_TOL = 0.15` of the
target `hurst`. **Trips when.** The scale-free band is too narrow (a constant
finite-band/finite-size bias of ~0.1 is expected, measured — which is why
this gate is warn). **What to do.** Widen the band (`l_min`↓, `l_max`↑ within
`min(L)/2`) and use a finer voxel grid for more scale-free octaves.

**Owner.** `boundaries.curved.method: perturbed_distance` with `spectrum:
self_affine` is the only combination that triggers G13 — `warp` cannot accept
`spectrum: self_affine` at all (see G6 above / `docs/physics.md` §5b), so it
never populates this gate. G13 pairs with G20 below (both fire together,
whenever they fire, since they share one `PerturbedDistanceTessellation`
construction): G13 checks the synthesized FIELD's own spectral law; G20
checks the resulting BOUNDARY's own box-counting roughness index — a
direct read of the generated geometry that G13's field-only check cannot
provide, though (see G20 below) not itself a converged, large-effect-size
fractal-dimension certificate below ~5.5 synthesis-band octaves.

## G14 — Single-crystal commensurability (warn)

**Checks.** For `grains.number: 1`, the box must be an integer number of
(rotated) lattice repeats; the max residual strain must be ≤
`COMMENSURATE_TOL = 1e-8`. **Trips when.** The box is not a lattice multiple, or
the lattice is rotated off a commensurate orientation. **What to do.** Use a box
of integer lattice multiples and/or an identity orientation for a perfect
crystal; otherwise the periodic faces are self-boundary defects (which may be
intended).

## G15 — Phase volume fractions (warn)

**Checks.** For `phases:`, the achieved per-phase volume fractions are
**re-measured** and compared to the targets; the deviation floor is the
granularity bound `V_max / V_box` (moving one grain shifts a fraction by at most
that much). With ≥ 3 phases at extreme fractions and few grains the
every-phase-≥1-grain guard can push the deviation up to ~2× the bound (still a
WARN). **What to do.** Increase `grains.number` for finer
fraction resolution.

## G16 — GB curvature sanity (warn-only)

**Checks.** With `analysis.gb_curvature: true`, the fraction of boundary
level-set samples dropped as degenerate (gradient magnitude `|∇φ| < 0.1`,
`CURV_GRAD_MIN`) during central-difference curvature evaluation must be
≤ `CURV_G16_DROP_TOL = 0.05` (5 %). Flat geometries and single-crystal /
no-boundary runs compute no curvature samples and report `0/0` (trivially
ok). **Trips when.** The analysis-grid voxel spacing is too coarse relative
to the boundary curvature, or samples fall near a triple junction/void where
the level set is ill-conditioned. **What to do.** Raise `analysis.voxel_grid`
for a finer level-set sampling.

## G17 — Dopant composition + enrichment (warn)

**Checks.** With `doping:` active, each dopant's achieved atom fraction of the
final structure is compared to its nominal `concentration`; WARN above 1
percentage point of drift (same rule as G9, never hard-fails). When
`gb_segregation.enabled`, the achieved enrichment E = c_shell/c_bulk is also
compared to the target `enrichment`; WARN when off by more than 15 % relative,
or when it cannot be measured (`E=n/a`, e.g. an empty shell or bulk
population). **Trips when.** The Bernoulli placement noise pushes a small
dopant count off target, or the shell/bulk split is too small to measure E
reliably. **What to do.** Increase `grains.number` (more candidate sites) or
accept the statistical noise — G17 is diagnostic only, mirroring G9's honesty
about drift rather than silently absorbing it. **Rule of thumb.** The achieved
enrichment E_achieved = (d_s/c_s)/(d_b/c_b) is a ratio estimator over the
bulk dopant count d_b, so it is only statistically reliable when the expected
bulk dopant count E[d_b] is at least a few tens — e.g. E[d_b] ≈ 7 gives ~40 %
seed-to-seed spread in E_achieved and an upward Jensen bias from the 1/d_b
term; pick `concentration`, `shell_width`, and grain count so E[d_b] clears
that floor before trusting a single-seed G17 read. **Self-conflict pruning.**
For interstitial dopants, sites that collide with an already-accepted dopant
are pruned after the per-site Bernoulli draw rather than by re-solving for
the remaining candidates, which produces a small deterministic undershoot in
achieved concentration (the undershoot grows with `enrichment × concentration`).

## G18 — Dopant min-distance re-verification (hard)

**Checks.** An independent fresh PBC neighbour query on the FINAL structure
(G7-style) proves no dopant–atom pair is closer than that dopant's
`min_distance`; `measured` = number of violating pairs. **Trips when.** A
`min_distance` too large for the interstitial preset's site spacing, or (in
mixed-periodicity boxes) the rare corner-wrap case the placement stage's own
pruning does not catch (documented limitation in `atoms/doping.py`).
**What to do.** Lower `min_distance`, choose a different site preset, or lower
the dopant `concentration`.

## G19 — `perturbed_distance` reassigned-voxel fraction (warn-only)

**Checks.** The fraction of analysis-grid voxels the G5 repair pass (see
above) had to reassign to restore periodic connectivity,
`reassigned_fraction`; WARN above `REASSIGNED_FRACTION_WARN_TOL = 0.02` (2 %).
Unlike a hard gate, this never blocks the run — repair either converges (and
the structure is used as repaired) or the constructor itself raises
`TessellationError` on non-convergence, which never reaches this reporting
stage. **Trips when (WARN).** `amplitude`/`hurst` pushed the level-set
folding into pervasive topology-breaking territory — a large fraction of the
domain needed correction, which is worth attention even though the run
completed successfully. **What to do.** Reduce `boundaries.curved.amplitude`
or `hurst`, or increase `min_seed_distance` (fewer/more-separated grains
give the level-set rule more room before neighbouring perturbations start
folding into each other). **Observed range.** Validation runs spanning
12–60 grains and amplitudes from an aggressive prototype value up through
the seed-containment guard ceiling measured 0.001–0.16 % reassigned — the
single worst case still leaves roughly 13× headroom below the 2 % floor
(`constants.REASSIGNED_FRACTION_WARN_TOL` docstring).

## G20 — `perturbed_distance` box-counting roughness index (warn-only diagnostic)

**Checks.** A box-counting roughness-index fit `d_b_estimated` of the
generated grain boundaries — the G13 analogue for this method — following
the Braun et al. 2020 protocol (`docs/physics.md` §5b(d)) on up to 3 in-box
cross-sections of the 5 largest grains (by voxel count). **Fit window
restricted to the synthesis band itself,** `ε ∈ [l_min, l_max]`
(converted to each section's own pixel scale) — a box size finer than
`l_min` or coarser than `l_max` probes a scale the η field has no designed
spectral content at (pixelation noise on one side, finite-object-size
cutoff on the other), so it is excluded from the fit rather than averaged
in with the previous fixed-decade window. Reported, never hard-failed: a
box-counting fit from a finite, noisy, single-realization boundary on a
handful of sampled grains/sections is an *estimate* with a documented
resolution bias (§5b(d)), not a certificate — the same philosophy
`gate_g13_hurst` uses for this same method's own Hurst back-estimate (G13,
above — the two gates share one `PerturbedDistanceTessellation`
construction and always fire together). **Reports
`n/a`, never a fabricated number,** when no sampled (grain, section)
combination yields a usable fit (e.g. every sampled grain is too small
relative to the section resolution `SECTION_GRID`, or the band collapses at
that section's pixel scale).

**Roughness index, not a fractal-dimension certificate.** Box-counting's
ability to statistically separate the Hurst exponent's effect on boundary
roughness from finite-band/finite-size noise is a GRADIENT with the
synthesis band's octave count, not a sharp cliff at any one grain size.
This project's own independent validation campaign measured an effect
size indistinguishable from noise (η² ≈ 6%, p ≈ 0.14) at the ~1.7-octave
band `pdau_perturbed_self_affine.yaml` uses; a follow-up synthetic-field
Monte Carlo found the effect already statistically SIGNIFICANT (p ≪ 0.01)
by ~3.2 octaves, but only reaching a conventionally LARGE effect size
(η² > 80%) once the grain diameter grows to ≈50 nm (~5.5 octaves) —
significance and effect-size magnitude cross their respective thresholds
at different octave counts (docs/physics.md §5b(g) has the full
octave/η² table). `d_b_estimated` below the ~5.5-octave / large-effect
regime is honest evidence of *local* boundary roughness at the resolved
scales, not a converged, large-effect-size fractal dimension — the gate
name (G20) and its warn-only identity are unchanged, but do not read a
small-grain `d_b_estimated` as a fractal-dimension measurement comparable
to a large-grain one. **What it does NOT do.** G20 has no pass/fail threshold
and never triggers a WARN row by itself — `measured` is always
`d_b_estimated` itself (or NaN when `n/a`); read it against the target band
from your own experimental motivation (e.g. Braun 2020's D_b = 1.174 ±
0.004 for nanocrystalline Pd₉₀Au₁₀, itself only measured on grains ≳45 μm
under that paper's own minimum-grain-size protocol constraint, `d_min =
ε_max/0.4` with `ε_min ≥ 3s`) rather than against a gate tolerance. See
`docs/physics.md` §5b(e) for this backend's own honest calibration status
against that band, and `examples/` for a ≥50 nm/5.5-octave configuration
sized specifically to reach the statistically decisive regime.

## G21 — Per-grain Gauss-Bonnet face-interior residual (warn-only)

**Checks.** With `analysis.gb_curvature: true`, for every grain g the
face-interior Gauss-curvature integral `Σ_faces K·dA` (summed over that
grain's retained boundary samples, `analysis.curvature.
per_grain_gauss_bonnet`) is compared against `CURV_G21_GAUSS_BONNET_TOL
= 4π`. This runs immediately alongside G16 in the analysis stage,
sharing the same `CurvatureResult` — it never runs when `gb_curvature` is
off. **Why 4π, and why this is a coarse ceiling, not a zero-residual
test.** The Gauss-Bonnet theorem fixes `∮_S K dA = 4π·(1 − g)` exactly for
any closed orientable surface S of genus g; every grain under periodic
boundary conditions is such a surface. `analyze_curvature` deliberately
samples K only at boundary-FACE-INTERIOR points (its stencils are
undefined at a polyhedral grain's non-differentiable edges/vertices,
`analysis/curvature.py` §6.7); the DESIGN INTENT is that a Voronoi-like
grain's Gauss-Bonnet budget is dominated by exactly those excluded
edges/vertices, as angle-defect Dirac mass (a cube's eight corners alone
already sum to 4π), leaving a small face-interior remainder. **That
intent is NOT what calibration measures.** This gate's own validation
sweep (`constants.CURV_G21_GAUSS_BONNET_TOL` docstring has the full
numbers) found near-FLAT baselines (amplitude 0.01, six seeds) already
cluster their worst-grain ratio at 0.977–1.041× this tolerance — AT one
topological unit, not comfortably below it — and ordinary curved runs
routinely reach 5–14×. The residual does not shrink toward 0 as
`analysis.voxel_grid` is refined over the range tested (a 32→128 sweep
gave a non-monotonic sequence with no visible downward trend), and a
two-grain periodic bicrystal with one smooth boundary and NO triple
junctions still measured 8–13× — ruling out "it's only a many-grain/
many-triple-junction effect" as the sole explanation. A preliminary
attempt to separate an area effect from a resolution effect (holding
TRUE physical voxel spacing fixed while growing the box) did not show a
clean, reproducible area trend either — the seed-to-seed spread was
large enough to swamp it, so "grows with area" is not yet an established
finding. What is reproducible: in a representative near-flat case,
under 1% of retained samples account for about half of the worst
grain's total, a heavy-tailed pattern more consistent with leakage
concentrated at specific voxels than with generic uniform noise; the
precise mechanism remains uncharacterized. **What a WARN means (and does
not mean).** G16 asks "did we keep enough samples" (the dropped-sample
fraction); G21 asks "how much of the excluded edge/vertex curvature is
leaking into the kept, face-interior set". Given the calibration above,
**a WARN is routine on ordinary curved runs with the current
estimator — including near-flat ones** — it is a standing, coarse
caveat on the quantitative reliability of `gb_curvature.csv`'s H/K
columns for that run, not proof that something newly broke. G3/G4
(volume/topology) and G7 (interatomic distances) already certify the
atomistic output independently of this estimate, so **G21 never
hard-fails and a WARN here does not invalidate the structure.**
`measured` is the WORST (maximum `|Σ K·dA|`) grain; the message names
the violating grain count, the worst grain's id, its value, and the 4π
threshold. **What to do.** Treat a WARN as a caveat on curvature-column
reliability for that run rather than as an actionable defect by itself;
it is worth a second look only if the value is unusually large or grows
unexpectedly between otherwise-similar configurations. It is not a
signal to change `boundaries.curved` parameters, and raising
`analysis.voxel_grid` has NOT been observed to make the residual
converge toward zero in the range tested.

## G22 — Volume-weighted ODF fidelity (warn-only, never fails)

**Checks.** With `orientation.mdf_target` configured (single-phase only —
`mdf_target` cannot combine with `phases`), the FINAL structure's
volume-weighted orientation distribution is RE-MEASURED independently of
the assignment annealer's own bookkeeping (the same "never trust the
optimizer's own claim" discipline as G11/G12/G15), in the analysis stage
rather than inside the annealer. It always reports: the Kish effective
sample size `n_eff` of the grain-volume distribution, the assignment-
independent count-vs-volume TV gap on LABELLED orientation indices
(`orientation.odf.count_vs_volume_gap` — a property of grain sizes alone;
an upper bound after merging duplicate or symmetry-equivalent
orientations, not necessarily the physical quotient-space gap), and the re-measured atomic drift
`drift` against its configured cap (`mdf_target.odf_drift_max`, or
"no cap"). It also cross-checks the annealer's own `odf_drift_final`
claim against this re-measurement and notes any disagreement above
`1e-9`. Memory permitting (see below), it additionally builds a
symmetrized de la Vallée Poussin Gram matrix at `mdf_target.
odf_kernel_halfwidth_deg` and reports the kernel-smoothed discrepancy
`mmd_vs_count` (annealed vs. the count-weighted/uniform reference —
the SAME reference `null_mmd`'s null distribution uses),
`mmd_vs_initial` (annealed vs. pre-annealing, descriptive only, no null),
and `null_mmd`'s median / central-95% interval `[p2.5, p97.5]`, plus the
empirical quantile `mmd_vs_count` falls at as informational context
(`mdf_target.odf_null_samples` draws — see VALUE VS. QUANTILE below for
why the empirical quantile is reported but never decides the verdict).

The VP degree is the continuous half-width solution rounded to the nearest
positive integer, so the even quaternion-dot power and its symmetry average
are positive semidefinite. Fractional degrees do not have this property.
The message records requested and effective half-widths and `kappa`; for a
10-degree request, `kappa=91` gives about 9.99468 degrees. Effective widths
cannot exceed 90 degrees with this nonconstant integer-degree family.
The resulting MMD is a finite-bandwidth **pseudometric**: a zero diagnostic
alone does not prove equality of physical ODFs. Its norm is not numerically
bounded by `odf_drift_max`; the Markov-contraction bound is on the **TV of
smoothed densities**, not MMD. Integer-degree kernel normalization is tested
independently, including broad kernels where fractional powers previously
gave negative eigenvalues.

**Interpretation — read before reading the message.** The re-measured
`drift` is the CONTROL quantity: bounding it during annealing is what
LICENSES the "volume-weighted ODF preserved to within an explicit
epsilon" claim (a Markov-contraction argument — bounding the un-smoothed
drift bounds the kernel-smoothed drift at every half-width at once, see
`orientation/odf.py`). It is **not, by itself, evidence of bias.**
Measured directly: on a 60-grain lognormal test case, an *unconstrained*
anneal's drift sits at the 89th percentile of what random,
non-adversarial permutations of the same grain volumes produce against
the same orientation set (null median 0.427, p95 0.483) — comfortably
inside the bulk, not an outlier. Bias detection — "did the annealer
specifically concentrate volume onto a subset of orientations" — is what
`mmd_vs_count` vs. its own null is for; do not read a large `drift` alone
as a bias finding, or a small one as proof of its absence.

**The null test is TWO-SIDED, and this is load-bearing, not stylistic.**
`mmd_vs_count` is a DISTANCE (in kernel space) to the count-weighted
reference f_n; landing anomalously CLOSE to f_n is exactly as informative
as landing anomalously far — both mean the final assignment's search
happened to correlate orientation with grain size, just in opposite
directions. A **mid-range** quantile, not a small one, is the genuine
"nothing statistically resolvable" outcome. Worked example (SAME
60-grain orientation set / adjacency / volumes, annealed with no cap vs.
a tight `odf_drift_max=0.02`, reproduced exactly in
`tests/test_odf_gates.py`):

| run | drift | mmd_vs_count | null median / [p2.5, p97.5] | verdict (empirical quantile) |
|---|---|---|---|---|
| no cap | 0.4024 | 0.6329 | 0.6943 / [0.6389, 0.7555] | **WARN** — below p2.5 (1.6%) |
| cap 0.02 | 0.0197 | 0.6907 | 0.6916 / [0.6423, 0.7683] | ok — inside (47.7%) |

The uncapped run has the *larger* drift by 20× yet its `mmd_vs_count`
lands BELOW the interval's own lower bound — an upper-tail-only check
would have called this "ok" and flagged nothing, exactly backwards, since
this is the run whose search settled unusually close to uniform. The
capped run, despite its far smaller drift, sits comfortably inside the
interval: the boring, expected result. `mmd_vs_initial` is the more
directly interpretable "how far did the anneal move it" number here —
1.0359 uncapped vs. 0.1522 capped, tracking the drift ratio far better
than `mmd_vs_count` does.

**A second floor: watch `n_eff`.** It can collapse well below the grain
count on its own, independent of annealing — measured on a reference
end-to-end scenario (24 grains, `size_distribution.sigma_log=0.5`): a
seed landing at `n_eff=4.25` (of 24) is unremarkable at that spread. Read
`drift`/`mmd_vs_count` against that floor, not against zero — a
small `n_eff` means the volume-weighted ODF is effectively a measure over
a handful of grains before the annealer ever touches it.

**VALUE vs. QUANTILE — a real defect this fixes.** A run was observed
printing `mmd_vs_count=0.3439` against a stated interval lower bound of
`0.3494` while the SAME sentence asserted "inside the interval" — 0.3439
is plainly below 0.3494. This happened because an earlier version decided
the verdict from the *empirical* quantile (`mean(null <= mmd_vs_count)`,
quantized to `1/len(null)` — about 0.4% per sample at the default
`odf_null_samples=256`) while the message was phrased as a value-vs-
interval comparison; the two framings can disagree by one sample right at
a boundary. The verdict is now decided directly on `mmd_vs_count` against
`[p2.5, p97.5]` — the SAME comparison the message prints — so the
sentence and the verdict cannot disagree by construction. The empirical
quantile is still reported, but as *informational context only*. A run
whose value sits within roughly one null sample of either bound should be
read as **borderline**, not as a clean verdict either way; increase
`odf_null_samples` to sharpen the boundary if that matters for a specific
run.

**Trips when (WARN).** Either: the kernel block ran and `mmd_vs_count`
itself falls outside the central 95% interval — i.e. `mmd_vs_count <
null_p2_5 or mmd_vs_count > null_p97_5` (two-sided, see above); or a cap
was configured and `drift > odf_drift_max + 1e-9`. The `1e-9` slack is
required, not cosmetic — `atomic_drift`'s re-measurement sums per-class
volume masses in a different order than the annealer's own incremental
`DriftTracker` bookkeeping, so an exactly-satisfied cap can read a few
ULP high on independent re-measurement. **What to do.** For a drift-cap
WARN, loosen `odf_drift_max` (trades off against how far the histogram
can be pulled toward its target — see `MdfTargetConfig.odf_drift_max`'s
own docstring for the measured trade-off curve) or accept the drift. For
an `mmd_vs_count` WARN in EITHER tail, treat it as a genuine
orientation-size correlation signature in the final assignment —
tightening `odf_drift_max` is the direct lever, though a low-tail WARN
(unusually CLOSE to uniform) is not fixed by the same intuition as a
high-tail one and is worth inspecting the run before assuming any
specific remedy.

**Memory guard (§13).** The Gram matrix is an N×N float64 allocation
(`8·N²` bytes); this gate estimates that cost itself, BEFORE calling
`orientation.odf.symmetrized_gram`, and skips the whole kernel block —
reporting the atomic quantities above regardless, plus the skip reason
(N and the estimated size) — whenever the estimate exceeds a modest
share of the run's resolved §13 memory budget
(`tess.memory_limit_bytes`/`tess.memory_limit_source`, the same budget
every other §13 guard in the pipeline uses). A WARN-only diagnostic must
never abort an otherwise-successful run; no exception from this gate is
ever allowed to propagate. `measured` is `mmd_vs_count` when the kernel
block ran, else `drift`.

## G23 — Angle-window vs. true-CSL Sigma3 consistency (diagnostic, never trips)

**Checks.** For any single-phase run that reaches the MDF-histogram
block — independent of whether `orientation.mdf_target` is configured,
since the gap below is informative either way — reports, side by side,
two Sigma3 boundary-area fractions that are **not interchangeable**:
`sigma3_angle_window_area_fraction` (area-weighted fraction of
boundaries with `|misorientation_deg − 60| ≤ 15/√3`, computed from the
per-boundary arrays directly — matching the
validation campaign's Brandon-window computation (local validation tooling, not published with this repository)
exactly: a boundary with a non-finite (NaN) misorientation angle compares
False against the window bound, so its area stays in the total-area
denominator but never enters the in-window numerator — a conservative
dilution, in both tools alike) and
`sigma3_csl_area_fraction` (area-weighted fraction of boundaries whose
`csl_sigma` is exactly `"3"` — angle AND ⟨111⟩ axis, via the true CSL
classification in `analysis/boundaries.py`), plus their ratio when the
CSL value is available and non-zero. **`passed` is always `true` and the
message never carries a WARN — G23 exists to report, not to judge.**

**Why the gap is EXPECTED, not a defect.** When `mdf_target` is
configured, the annealing objective is the one-dimensional, area-weighted
marginal of the disorientation ANGLE only (`orientation/mdf.py`).
Placing a target in the Sigma3 Brandon angular window therefore enforces
only the angular condition that is *necessary* for Sigma3 — the ⟨111⟩
axis condition true CSL classification also requires never enters the
energy, so the annealer has no mechanism to close that gap even at full
convergence. **`sigma3_csl_area_fraction` requires `analysis.csl: true`
to be measured at all** — when it is `false` this is reported as **NOT
EVALUATED** (an empty `summary.csv` cell), never as `0.0`, which would
falsely claim no Sigma3 boundaries exist. **Campaign range** (MDF
validation campaign, case M2 — local validation tooling, not published with this repository): the angle-window
area fraction runs 0.24–0.44 while the true CSL Sigma3 area fraction
stays 0.02–0.06 — roughly an order of magnitude apart, by construction.
**What to do.** Nothing — this gate never trips. If a `csl_enriched`/
`sigma3_angle_enriched` target's TRUE Sigma3 content matters (not just
its angular window), enable `analysis.csl` and design the orientation
set with genuine twin-related pairs (e.g. two `odf_components` related by
60° ⟨111⟩ — see the shipped `cu_twin_odf_mdf.yaml` example), since the
angular target alone cannot manufacture the axis condition.

## G24 — Final (post-warp) per-grain volume fidelity (warn-only)

**Checks.** Closes a real hole left by G11: G11 is evaluated on the
UNWARPED SDOT power base (its own docstring already admits the realized,
warped cells deviate), but nothing previously measured that deviation.
G24 runs exactly when the geometry changed AFTER the volume fit —
`sdot_res is not None and sdot_res.tess is not tess` — so for flat +
`size_distribution` (where the power diagram *is* the final
tessellation, the SAME object G11 already checked) this gate correctly
never fires and can never double-report G11; only warp +
`size_distribution` reaches it. Compares the FINAL per-grain volumes
(measured in the analysis stage, the same array `grains.csv` reports)
against `sdot_res.target_volumes`; `measured` is the max relative error,
the message also reports the mean — and always states plainly that the
comparison is against the UNWARPED-base targets, so a reader cannot
mistake this for a second G11 reading.

**Trips when (WARN).** Max relative error exceeds `WARP_VOLUME_G24_TOL =
0.10`. This threshold is a deliberately loose SANITY bound, not a physics
claim, and is calibrated separately from (never reuses)
`size_distribution.vol_tol` (default `1e-3`) — that tolerance governs the
base-cell fit G11 checks, at a precision the warp displacement itself
blows through by (at minimum) two-plus orders of magnitude.

**Calibration.** Equal-volume targets, a 60 Å box, warp amplitude 0.6,
correlation length 10 Å: max relative POST-WARP volume error 0.0209 at 4
grains and 0.0412 at 8 grains, against an UNWARPED-base G11 error of only
1.5e-4 and 6.4e-5 respectively — G11 passes comfortably on the base while
the realized cells are already two-plus orders of magnitude further from
the targets. A wider sweep (70 Å box, seed 11, correlation length 10 Å)
varying both grain count and how broad the target size distribution is
(max relative error, mean in parentheses):

| N | size_distribution | amplitude | max (mean) error | verdict |
|---|---|---|---|---|
| 4 | equal | 0.6 | 0.0220 (0.0110) | ok |
| 8 | equal | 0.6 | 0.0389 (0.0208) | ok |
| 8 | lognormal σ=0.35 | 0.6 | 0.0465 (0.0217) | ok |
| 8 | lognormal σ=0.5 | 0.6 | **0.1088** (0.0309) | **WARN** |
| 24 | lognormal σ=0.5 | 0.4 | 0.0838 (0.0339) | ok |

Observed range across every configuration measured: 0.022–0.109. The
tolerance trips only at the extreme end (the broadest size spread
measured, at a substantial amplitude) — it is not a hair trigger on
ordinary warp + `size_distribution` combinations, but a BROAD grain-size
distribution combined with a LARGE warp amplitude CAN push the realized
error past it. **When that happens, the WARN is the gate correctly
reporting a real, physical effect — the warp is applied AFTER the volume
fit and nothing re-fits the targets to the warped geometry — not a false
alarm to tune away.** The competing explanation — that any of this is
merely voxel-grid discretisation noise rather than real warp displacement
— was checked and ruled out: voxel-vs-exact-polyhedral volumes on the
SAME (unwarped) flat tessellations differ by only 0.0024–0.0040 across
the configurations above, far below every measured value in the table.
`vol_tol` would therefore be the wrong threshold to reuse here: it sits
below that discretisation floor and would false-fail on essentially every
warped run. **What to do.** A WARN here flags a run whose combination of
warp amplitude and target size spread has pushed realized volumes outside
the calibrated range — reduce `boundaries.curved.amplitude`, narrow the
`size_distribution` spread, or accept that `size_distribution` targets
are only approximate once a domain warp of that severity is applied on
top of them; do NOT read it as a threshold miscalibration to loosen.

## G25 — Texture-component fidelity (diagnostic, never trips)

**Checks.** For `orientation.scheme: odf_components` runs, puts the
configured texture-component weights next to what the run actually
realised — both in the gate message and as `texture,component_<k>_*`
rows in `summary.csv` — tagged with the `orientation.component_weight_basis`
in effect (`"volume"` or `"count"`). Per component *k*:

* `w_cfg` — the configured `weight`, normalised;
* `f_count` — the realised grain-**count** fraction. A permutation of
  which grain carries which orientation cannot change a bincount, so this
  one number already covers both before and after annealing;
* `f_vol_pre` — the realised **volume** fraction right after
  `orientation.samplers.odf_components` assigns components, *before* any
  assignment annealing runs;
* `f_vol_post` — the realised volume fraction *after* annealing (equal to
  `f_vol_pre` when `mdf_target` is not configured);
* `mean_volume_A3` — the component's mean grain volume — the signature of
  the size-component correlation `component_weight_basis: "volume"`
  trades for volume fidelity (`docs/physics.md` §4's lemma). Measured on the
  **pre**-anneal labelling (see above), so `summary.csv` writes it as
  `texture,component_{k}_mean_volume_pre_A3` — the `_pre` suffix is part of
  the row key, unlike `f_vol_post`, whose row keeps the bare
  `component_{k}_volume_fraction` name.

It also reports three total-variation distances, each `0.5 · Σ_k |·|`
over components:

* `tv_cfg_vs_pre` — configured weight vs. realised pre-anneal volume
  fraction: the **sampler term** (`docs/physics.md` §4's two-term
  decomposition). Zero (up to floating point) under `"volume"` by
  construction (`orientation.odf.volume_balanced_partition`); set by the
  grain-size distribution under `"count"` (measured 24-grain,
  `sigma_log=0.5` case: 0.245, vs. `weight: 0.5/0.5` configured).
* `tv_pre_vs_post` — pre- vs. post-anneal volume fraction: the **annealer
  term**, MEASURED directly from the pre/post component labels, not
  inferred or bounded.
* `tv_cfg_vs_post` — configured weight vs. realised final volume
  fraction: the end-to-end gap, i.e. (up to the triangle inequality)
  `tv_cfg_vs_pre + tv_pre_vs_post`.

**Why this exists.** `weight` is realised as a grain-count fraction only
under `component_weight_basis: "count"`; the physically meaningful
quantity is the *volume* fraction, and with unequal grain volumes the two
differ (`f_count` vs. `f_vol_pre`). Reading the volume-weighted ODF as
"the configured weights" is exactly the misinterpretation this gate makes
impossible, for either basis.

**`tv_pre_vs_post` is a measurement, not a bound — it is *not*, in
general, capped by `mdf_target.odf_drift_max`.** `odf_drift_max` caps a
related but finer-grained quantity: the atomic, per-*orientation* drift
G22 re-measures. The two coincide only when the component labelling is
**constant on every orientation class** — no two grains with different
component labels ever carry bit-identical orientations. That condition
can fail even at `spread_deg: 0`: two zero-spread `odf_components` with
identical `euler_bunge_deg` are accepted by `resolve_config`, so two
bit-identical orientations can land in different components; swapping
those two grains during annealing leaves the atomic drift G22 reports at
exactly `0.0` while moving `tv_pre_vs_post` by their combined volume
fraction — up to 0.8 for two dominant, equal-volume grains split across
components. **`passed` is always `true` and the message never carries a
WARN — G25 exists to report, not to judge.** No universal threshold
exists: the expected sampler term is zero under `"volume"` and scales
with the grain-size distribution under `"count"`; the annealer term
scales with `odf_drift_max` when a cap is set, but this gate only ever
measures it, never bounds it.

**What to do.** Nothing — this gate never trips. If `tv_cfg_vs_post` is
larger than expected, read `tv_cfg_vs_pre` and `tv_pre_vs_post`
separately before reaching for a fix — they are independent terms with
independent levers: a nonzero `tv_cfg_vs_pre` under `"count"` is fixed by
switching to `component_weight_basis: "volume"` (or by increasing
`grains.number`, which shrinks the grain-size-driven gap under either
basis); a large `tv_pre_vs_post` points at the annealer instead — set or
tighten `mdf_target.odf_drift_max`. `mean_volume_A3` reports the
`"volume"` basis's honest cost, the size-component correlation
(`docs/physics.md` §4); it shrinks as `grains.number` grows.

## G26 — Atomistic ODF-weighting discretisation floor (diagnostic, never trips)

**Checks.** `odf_mtex.txt` writes `volumes / sum(volumes)` as its weight
column, gate **G22** re-measures drift on those same TESSELLATION
volumes, and `mdf_target.odf_drift_max` caps it — but the artefact
actually shipped to an MD user is ATOMISTIC, and the ODF a downstream
simulation experiences is weighted by the number of ATOMS in each grain,
not by its polyhedral/voxel volume. G26 puts those two weightings side by
side: `w_vol = volumes / sum(volumes)` (what `odf_mtex.txt` exports) vs.
`w_atom = n_atoms / sum(n_atoms)` (what the exported atomistic structure
actually realises), and reports

* `tv` — `0.5 · Σ|w_vol − w_atom|`, the total-variation distance between
  the two weightings (`summary.csv`: `odf,weighting_tv_volume_vs_atoms`);
* `max_dw` — the single largest per-grain weight discrepancy
  (`odf,weighting_max_dw`);
* the mean atoms per grain (`odf,atoms_per_grain_mean`), for context on
  where this run sits on the scaling curve below.

**This is a discretisation floor, not an error.** An atomistic structure
with *n* atoms in a grain cannot resolve a volume-weighted ODF finer than
this floor — no choice of `odf_drift_max` changes that, since the
annealer and G22 both operate on the continuous tessellation-volume
weighting, while the export is always discrete atom counts. Measured
(reference scenario: 24 grains, box 45 Å, `size_distribution` lognormal
`sigma_log 0.5`, two texture components at `weight 0.5`, seed 4242, cubic
Cu; same config, box size varied):

| atoms/grain | TV(volume vs atom-count) | max per-grain \|dw\| |
|---|---|---|
| 275  | 0.027418 | 0.009107 |
| 1092 | 0.018910 | 0.006912 |
| 3279 | 0.013133 | 0.004724 |
| 9191 | 0.008633 | 0.003126 |

Fitted exponent **−0.33**, i.e. the floor decays as `(atoms per
grain)^(-1/3)` — a **surface** effect: every atom within one interatomic
spacing of a grain boundary is only probabilistically assigned to one
side or the other, and that shell's atom count is a fixed fraction of the
grain's *surface*, which scales as `1/L` of its volume. This is **not**
Poisson counting noise, which would give the steeper `(atoms per
grain)^(-1/2)` — the fitted −0.33 rules that mechanism out. **At 275
atoms per grain the floor (0.027) exceeds a typical `odf_drift_max` of
0.02**: a drift cap tighter than this floor buys nothing that survives
into the exported structure — that is the reason G26 exists, and what
makes the `odf_drift_max` guarantee honest about what it does and does
not cover.

**NOT EVALUATED (never a false `0.0`)** whenever the atom counts driving
`w_atom` are absent or all zero — `analyze_grains` sets `n_atoms` to 0
per grain when the atom block is omitted, and a zero-sum weight vector
cannot be normalised; reporting `0.0` would falsely claim the two
weightings coincide exactly, when the atom-count side was simply never
measured. No `odf,*` rows are written in that case.

**Multiphase.** Different phases have different atom number densities (a
consequence of differing crystal structures and lattice parameters), so a
single global atom-count weighting mixed across phases would compare
unlike things — the gap would mostly read off the density *ratio*
between phases, not the within-phase discretisation floor this gate
exists to quantify. G26 therefore computes and reports the statistic
**separately within each phase's own grain subset**, the same
grain→phase-name pairing `pipeline.py` already uses to write one
`odf_mtex_<name>.txt` per phase — never a global mix. Each `odf,*` key is
suffixed `_<phase name>`, and a phase whose subset happens to carry no
atoms reports an empty cell for that phase's three keys only, other
phases unaffected.

**`passed` is always `true` and the message never carries a WARN — G26
exists to report, not to judge.** No exception can escape it: a
deliberately broken input degrades to an "ok: could not be evaluated"
message rather than aborting the run.

**What to do.** Nothing — this gate never trips. If the reported floor is
larger than expected relative to `odf_drift_max`, `atoms_per_grain_mean`
says where on the `(atoms/grain)^(-1/3)` curve this run sits — increasing
grain size (fewer, larger grains at fixed box volume, or a larger box at
fixed grain count) is the only lever that shrinks it; no config knob
tightens the floor itself, since it is a property of how many atoms
represent one grain, not of any targeting or annealing setting.
