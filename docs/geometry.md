# Grain geometry

grainsmith partitions the simulation box into grains using a **tessellation
backend**. This page is the single reference for every backend: what it does,
the exact config parameters that control it, the construction conditions it must
satisfy (the QA gates), and a worked config snippet. It is the page to read when
you are deciding *which geometry to use and how to tune it*.

The math behind each method is in [physics & conventions](physics.md); the field
descriptions are auto-generated in the [configuration
reference](config_reference.md); the QA gates are in [QA gates](gates.md). This
page ties them together per geometry.

## Choosing a backend

The top-level switch is `boundaries.geometry`, with one refinement
(`grains.size_distribution`) that upgrades the flat backend to a volume-fitted
diagram:

| You want | `boundaries.geometry` | Extra block | Backend |
|----------|----------------------|-------------|---------|
| Planar grain boundaries (default) | `flat` | — | flat Voronoi |
| Grains with **prescribed sizes** | `flat` | `grains.size_distribution` | power / Laguerre (SDOT) |
| **Curved** (wavy) boundaries, gaussian roughness | `curved` | `curved: {method: warp}` | domain-warped Voronoi (spectrum: gaussian ONLY — self_affine is a `ConfigError` here, see §4 below) |
| Johnson–Mehl / weighted (also **curved**) | `curved` | `curved: {method: additive_weights}` | Apollonius (additive weights) |
| Elongated / textured grains (also **curved**) | `curved` | `curved: {method: anisotropic}` | ellipsoidal-metric diagram |
| Genuinely **SELF-AFFINE** boundaries (box-counting `D_b` responds to the field) | `curved` | `curved: {method: perturbed_distance, spectrum: self_affine}` | level-set / perturbed-distance — the ONLY method that accepts `spectrum: self_affine` |
| An externally-produced microstructure | `voxel_import` | `voxel_import: {…}` | imported voxel label field |
| A single crystal (no boundaries) | `flat` + `grains.number: 1` | — | single-crystal fast path |

grainsmith's user-facing GB-morphology taxonomy is three-tiered: **flat**
(the first two rows above), **curved** (the next three — smoothly curved,
Voronoi-topology-preserving), and **self-affine** (`perturbed_distance` alone —
the only method whose as-built boundary genuinely responds to a self-affine
roughness spectrum; see `docs/physics.md` §5b for why the other three
curved methods cannot).

All backends implement the same `Tessellation` interface, so the atom fill,
overlap removal, and analysis behave identically regardless of geometry — only
the grain shapes change.

## 1. Flat — planar Voronoi (default)

**What it does.** Standard periodic Voronoi tessellation of the seed points:
every point of space belongs to the grain whose seed is nearest, giving
**planar** grain-boundary faces computed exactly with Qhull. This is the default
and the base for most other backends.

**Parameters.** None beyond the seeds themselves — the geometry is fully
determined by `grains.number` (or an explicit seed list) and the seeding mode.
Seed placement is controlled by `grains.seeding` (Poisson-disc vs Lloyd
relaxation) and the box periodicity `box.periodic`.

**Construction conditions (gates).**

- **G3 — volume closure.** The Voronoi cell volumes must sum to the box volume to
  a relative tolerance `VOL_REL_TOL = 1e-6` (`|Σ Vᵢ − V_box| / V_box ≤ 1e-6`; the
  tight bound is safe because Qhull volumes use exact-adaptive arithmetic). A
  failure means the periodic tessellation did not tile the torus.
- **G4 — Euler characteristic.** Each cell must be a topologically valid convex
  polyhedron (`V − E + F = 2`); degenerate vertices closer than `EULER_TOL ×
  max(L) = 1e-9 × max(L)` are merged first.
- **G5 — connectivity.** Each grain is a single connected region; discretization
  slivers below `G5_SPURIOUS_VOXEL_FRACTION = 0.5 %` of cell volume are ignored,
  so G5 fails only on macroscopic fragmentation.

```yaml
# examples/basics/b2_nial_flat.yaml — planar Voronoi, B2 NiAl
box: {lengths: [80.0, 80.0, 80.0]}
grains: {number: 16}
boundaries: {geometry: flat}      # (default — may be omitted)
```

## 2. Power / Laguerre — grains of prescribed size

**What it does.** When you add a `grains.size_distribution` block, the flat
backend is replaced by a **power (Laguerre) diagram**: each seed gets a scalar
*weight*, and cells are bounded by weighted (power-distance) bisectors instead of
perpendicular bisectors. The weights are fitted by **semi-discrete optimal
transport (SDOT)** — a damped-Newton solve on the Kantorovich dual — so that each
grain's volume matches a sampled target. This is how you prescribe a grain-size
distribution rather than accepting whatever Voronoi produces.

**Parameters** (`grains.size_distribution`):

| Field | Meaning | Default |
|-------|---------|---------|
| `type` | `lognormal` (prescribe the *shape*, scale fixed by `V_box/N`), `equal` (all `V_box/N`), or `volumes` (explicit list) | `lognormal` |
| `sigma_log` | log-normal shape parameter — std of `ln(d)`; larger = broader size spread | `0.35` |
| `volumes` | explicit relative-volume list (length = `grains.number`), used with `type: volumes` | — |
| `vol_tol` | **gate G11** threshold: max relative per-grain volume error on the fitted diagram | `1e-3` |
| `max_iter` | maximum damped-Newton iterations of the SDOT fit | `30` |
| `centroidal_iterations` | outer centroidal loop (Kuhn 2020): alternate volume-fitting with moving seeds to cell centroids → equiaxed grains **with** prescribed volumes | `0` (off) |

**Construction conditions.** The flat gates (G3–G5) still apply. Additionally:

- **G11 — volume accuracy.** After the SDOT fit, `max_i |V_i − V_iᵗᵃʳᵍᵉᵗ| /
  V_iᵗᵃʳᵍᵉᵗ ≤ vol_tol`. If the Newton solve does not reach `vol_tol` within
  `max_iter`, the gate fails — raise `max_iter` or relax `vol_tol`.

> **Scope note.** The SDOT volume guarantee holds on the *flat power base*. If
> you then wrap it in a curved warp (below), the realised per-grain volumes
> drift from the targets, and G11 measures only the unwarped base. Use
> prescribed sizes **or** strong warping, and read [physics.md](physics.md)
> before combining them.

```yaml
# examples/grain_geometry/cu_lognormal_sizes.yaml — log-normal grain sizes via SDOT
box: {lengths: [60.0, 60.0, 60.0]}
grains:
  number: 12
  size_distribution: {type: lognormal, sigma_log: 0.35, vol_tol: 1.0e-3}
boundaries: {geometry: flat}
```

## 3. Curved — domain-warped Voronoi (Gaussian spectrum)

**What it does.** The `curved` geometry with `method: warp` (the default curved
method) takes a flat (or power) base tessellation and **displaces space** by a
smooth, periodic, divergence-free Gaussian random field (GRF). Straight Voronoi
faces become gently curved surfaces, mimicking the wavy grain boundaries seen in
real microstructures — without changing grain topology or adjacency.

**Parameters** (`boundaries.curved`, `method: warp`, `spectrum: gaussian`):

| Field | Meaning | Default |
|-------|---------|---------|
| `method` | `warp` (domain-warped Voronoi via periodic GRF) | `warp` |
| `base` | base tessellation to warp: `flat`, `power`, `additive_weights`, `anisotropic` | `flat` |
| `amplitude` | warp-field RMS displacement per component, Å. **Hard cap:** `≤ min_seed_distance / 4` | `4.0` |
| `correlation_length` | Gaussian correlation length ℓ (Å) — the boundary "waviness wavelength"; larger ℓ = smoother, more gently curved boundaries | `15.0` |

**Construction conditions.**

- **G6 — bijectivity (seed containment).** The warp must remain invertible — no
  folded or self-intersecting boundaries. This is enforced two ways: the
  amplitude is hard-capped at `min_seed_distance / 4` (so a displaced point never
  crosses into a non-neighbouring cell), and the field gradient is bounded by
  `WARP_GRAD_MAX = 0.5` (`max‖∇u‖ < 0.5`). Requesting a larger `amplitude` than
  the cap raises at construction.

```yaml
# examples/grain_geometry/fcc_cu_curved_warp.yaml — curved boundaries, Gaussian spectrum
box: {lengths: [80.0, 80.0, 80.0]}
grains: {number: 12}
boundaries:
  geometry: curved
  curved: {method: warp, amplitude: 4.0, correlation_length: 15.0}
```

## 4. Self-affine spectrum — why `warp` cannot use it (the rough-boundary backend lives in §4b)

**History.** Earlier versions of grainsmith let `warp` take `spectrum:
self_affine` — a **band-limited power law** amplitude spectrum in place of
the single Gaussian scale, giving the displacement field a controllable
field fractal dimension `D = 3 − hurst`. **That combination has been
REMOVED, not deprecated:** `method: warp` + `spectrum: self_affine` is now
a hard `ConfigError`, raised both at config-resolve time
(`config/resolve.py` Rule 8a) and at the `WarpTessellation.__init__` API
level.

> **Why it was removed, not just discouraged.** `warp` is a coordinate
> diffeomorphism (`grain_of(x) = base.grain_of(x + u(x))`, bijective under
> the G6 guard), and a bijective map cannot change the box-counting
> dimension of the surface it displaces — see the Lipschitz argument in
> [physics.md §5b](physics.md#5b-self-affine-gb-morphology-why-warp-cannot-deliver-it-and-how-perturbed_distance-does).
> A `hurst` exponent on `warp`'s field only ever set a **roughness-color**
> knob (how the waviness is distributed across scales within the band), not
> a **fractal-dimension** knob — a 2D box-counting measurement of the
> actual grain-boundary contour stayed pinned near the flat-Voronoi value
> (`D_b ≈ 1.0`–`1.1`, depending on geometry) at *every* Hurst exponent,
> statistically indistinguishable across H. Since a self-affine spectrum on a
> non-self-affine boundary is scientifically hollow for its one stated purpose,
> grainsmith removes the combination outright rather than keeping a
> misleading knob alive with a caveat. **If you want a grain boundary whose
> box-counting dimension itself responds to a roughness parameter — e.g.
> reproducing Braun et al.'s measured D_b ≈ 1.17 nanocrystalline PdAu grain
> boundaries — use `method: perturbed_distance` (§4b below), the only
> method that still accepts `spectrum: self_affine`.**

`warp`'s own boundary waviness is still fully tunable — via `spectrum:
gaussian` (the only spectrum it accepts) and `correlation_length` (§3
above). Everything below this point — the self-affine spectrum's field
math, its parameters, the narrow-band trap, and a worked example — now
lives entirely in §4b, since `perturbed_distance` is its only consumer.

## 4b. Perturbed-distance — genuinely self-affine grain boundaries

**What it does.** `method: perturbed_distance` replaces the flat-Voronoi
membership test itself with a perturbed-distance argmin, using an
**independent per-grain scalar field** `η_i` rather than a single shared
displacement field:

$$
\mathrm{grain\_of}(x) = \operatorname*{argmin}_i \left[ d_{\mathrm{pbc}}(x, s_i) - \mathrm{amplitude} \cdot \eta_i(x) \right]
$$

Each `η_i` is drawn from the same `synthesize_grf` machinery `warp` uses for
its displacement field (`tessellation/warp.py`, shared verbatim) with either
a single Gaussian scale (`spectrum: gaussian`) or a **band-limited power
law** (`spectrum: self_affine`):

$$
\sqrt{\Phi(k)} \propto k^{-(3+2H)/2}, \qquad k \in \left[\frac{2\pi}{l_{\max}}, \frac{2\pi}{l_{\min}}\right]
$$

giving the field a controllable fractal dimension `D = 3 − hurst`. Because
this perturbs the ASSIGNMENT RULE rather than the coordinate frame, it is
**not** a coordinate diffeomorphism (§4's `warp` is; this is not) — it can
fold space near the boundary and produce a genuinely self-affine grain-boundary
contour whose box-counting dimension responds to the field's Hurst exponent
and amplitude, which `warp` can never deliver regardless of Hurst (§4; full
argument in
[physics.md §5b](physics.md#5b-self-affine-gb-morphology-why-warp-cannot-deliver-it-and-how-perturbed_distance-does)).
`perturbed_distance` is therefore the **only** method that accepts
`spectrum: self_affine` at all. This is the backend for reproducing an
experimentally measured section-perimeter fractal dimension, e.g. the
nanocrystalline Pd₉₀Au₁₀ abnormal-grain-growth boundaries of Braun et al.
2018/2020 (D_b = 1.174 ± 0.004).

**Parameters** (`boundaries.curved`, `method: perturbed_distance`) — the
*exact same* `spectrum` / `hurst` / `l_min` / `l_max` / `amplitude` schema
keys `warp` also declares, reused verbatim; there is no `base` or
`correlation_length` role for this method beyond what those keys already
cover:

| Field | Meaning | Default |
|-------|---------|---------|
| `spectrum` | `gaussian` (single scale) or `self_affine`: `√Φ(k) ∝ k^(−(3+2H)/2)` for `k ∈ [2π/l_max, 2π/l_min]` | `gaussian` |
| `hurst` | Hurst exponent `H ∈ (0, 1]` (self_affine only); field fractal dimension `D = 3 − H`. Typical metal GBs: 0.7–0.9 | `0.8` |
| `l_min` | shortest roughness wavelength (Å, self_affine only); band upper k-limit `2π/l_min` | `8.0` |
| `l_max` | longest roughness wavelength (Å, self_affine only); band lower k-limit `2π/l_max` | `60.0` |
| `amplitude_convention` | `total_rms` (default) or `reference_wavelength` (opt-in) — see below | `total_rms` |
| `reference_wavelength` | reference-octave anchor (Å, self_affine + `reference_wavelength` convention only); `null` means "use `l_max`" | `null` |

**`amplitude_convention` (opt-in, backward-compatible).** The
default `total_rms` fixes `amplitude` as the field's **total RMS over the
whole synthesis band** `[l_min, l_max]`. This has a counter-intuitive
consequence: because the band-integrated variance of a `k^(−(3+2H))`
spectrum is dominated by its longest wavelengths, WIDENING the band (more
octaves — smaller `l_min`, larger `l_max`) *redistributes* the same total
budget across more scales rather than adding to it. At fixed `amplitude`,
adding a coarse octave can therefore **reduce** short-wavelength (`l_min`
-scale) roughness — the opposite of the physically expected direction (see
[physics.md §5b](physics.md#5b-self-affine-gb-morphology-why-warp-cannot-deliver-it-and-how-perturbed_distance-does)
for the measured reversal in grain-boundary area this causes).

Setting `amplitude_convention: reference_wavelength` instead fixes the RMS
power contributed by **one reference octave**,
`[reference_wavelength/2, reference_wavelength]` (default anchor:
`l_max`), letting every other resolved octave accumulate its own
contribution independently on top. `amplitude` then has a stable physical
meaning — Å-RMS at a named wavelength — independent of how many octaves
the band spans.

**Caveat on the default anchor.** With the **default** anchor
(`reference_wavelength: null` → `l_max`, which MOVES as the band widens),
a sign-test found the small-scale roughness suppression above is
NOT fully corrected — the local-roughness trend still decreases as
octaves are added, just less steeply than under `total_rms`. Only an
explicit, **fixed absolute** `reference_wavelength` (anchored near
`l_min`, not moving with `l_max`) inverts the trend to the physically
expected monotonic increase. The default (moving) anchor is intended for
the common case where `l_max` itself represents a physical scale (e.g.
the grain radius) that should stay the roughness reference as other
settings change — not as a fix for a controlled band-width scan. See
[physics.md §5b(f)](physics.md#5b-self-affine-gb-morphology-why-warp-cannot-deliver-it-and-how-perturbed_distance-does)
for the measurement and the two anchor modes' different intended uses.

The two conventions are related by one exact,
invertible scale factor (`tessellation.warp.reference_shell_kappa`):

$$
\kappa = \frac{\sigma(\text{reference octave})}{\sigma(\text{whole band})} \quad (\kappa \le 1), \qquad
\mathrm{amplitude}_{\text{reference\_wavelength}} = \mathrm{amplitude}_{\text{total\_rms}} \cdot \kappa
$$

The seed-containment guard always binds the **realized total-RMS-
equivalent amplitude**, regardless of which convention supplied
`amplitude` — since `κ ≤ 1`, converting a `reference_wavelength` amplitude
back to its total-RMS equivalent (`÷ κ`) only ever makes the guard
*stricter*, never weaker. `reference_wavelength` requires `spectrum:
self_affine` and, when given explicitly, must lie within `[l_min,
l_max]`; it cannot be set under `amplitude_convention: total_rms` (a
silently-ignored field is treated as a config error, not a no-op). The
selected convention and (under `reference_wavelength`) the resolved
`reference_wavelength`/`κ`/equivalent total-RMS amplitude are reported in
`summary.csv` (`PerturbedDistanceTessellation.kappa` /
`.amplitude_total_rms` properties); `resolved_config.yaml` — a plain dump
of the config's OWN fields — carries `amplitude_convention` and
`reference_wavelength` as given, not the derived `κ` (which is computed
during tessellation construction, not resolved onto the config).

**Construction conditions (gates).** G6 does **not** apply (no
diffeomorphism to check); a different guard set takes its place:

- **Seed-containment guard (hard, pre-flight).** `amplitude ≤
  PERTURBED_DISTANCE_SAFETY · min_seed_distance / (2 · ETA_CLIP)`, i.e.
  `min_seed_distance / 12` — a *sufficient* (not tight) bound checked at
  config-resolve time, cheap insurance before paying for field synthesis.
- **Exact seed-ownership check (hard, post-hoc).** At construction, every
  grain's own seed must still evaluate to itself under the perturbed rule;
  a `ConfigError` if a competing grain's field captured it.
- **`l_min < l_max ≤ min(L)/2`** (self_affine only) — the longest wavelength
  must fit in the box.
- **`l_min ≥ 2·h_field`** (self_affine only) — the shortest wavelength must
  be resolvable on the field grid (Nyquist).
- **G5 → repair-and-report, not hard-fail.** Any voxel-scale fragmentation
  the folding causes is repaired by majority-vote reassignment rather than
  raising; **G19** (warn) reports the `reassigned_fraction` this took.
- **G13 — Hurst back-estimate (warn, self_affine only).** After synthesis,
  grainsmith re-estimates the Hurst exponent `Ĥ` from the generated
  per-color fields and warns if `|Ĥ − hurst| > HURST_G13_TOL = 0.15`.
- **G20 (warn, self_affine only) — box-counting roughness index, not a
  fractal-dimension certificate.** `d_b_estimated` is a box-counting fit
  of the actual generated boundary (Braun et al. 2020 protocol, 2D in-box
  cross-sections, fit window restricted to the synthesis band `[l_min,
  l_max]`) — the direct measurement G13 cannot provide, since G13 only
  characterizes the FIELD, not the resulting boundary. Box-counting's
  ability to statistically separate the Hurst exponent's effect from
  finite-band noise is a GRADIENT with octave count, not a sharp cliff:
  it is already significant (p ≪ 0.01) by ~3.2 octaves, but only reaches
  a conventionally LARGE effect size (η² > 80%) at ≈50 nm grain diameter
  (~5.5 octaves) — treat `d_b_estimated` below that regime as a
  local-roughness diagnostic, not a converged, large-effect-size fractal
  dimension comparable to Braun's D_b band — see
  [gates.md](gates.md) for the gate's exact framing, and
  [physics.md §5b](physics.md#5b-self-affine-gb-morphology-why-warp-cannot-deliver-it-and-how-perturbed_distance-does)
  for the estimator's documented resolution bias, the octave/decade
  budget this depends on, and this backend's honest calibration status
  against the Braun D_b band.

> **Convention caution.** grainsmith reports the **field** fractal dimension
> `D = 3 − H`. If you compare against a 2D box-counting dimension from an
> EBSD section, that is the **contour** dimension — a different, and for
> this method independently *measured* (via G20's `d_b_estimated`), number.
> Do not conflate them; see [physics.md](physics.md).

```yaml
# examples/self_affine_gb/pdau_perturbed_self_affine.yaml — genuinely self-affine GBs
box: {lengths: [200.0, 200.0, 200.0]}
grains: {number: 20}
boundaries:
  geometry: curved
  curved:
    method: perturbed_distance
    spectrum: self_affine
    amplitude: 3.6
    hurst: 0.7
    l_min: 12.0
    l_max: 40.0
```

**Narrow-band trap.** The Hurst exponent's effect on the field is a
within-band *tilt* of the power spectrum — it only becomes statistically
detectable when `l_max/l_min` spans a wide enough range for that tilt to
accumulate. This is a property of `synthesize_grf` itself (the function
`perturbed_distance` now exclusively consumes for self_affine fields,
since `warp` can no longer be configured with `spectrum: self_affine` at
all). At `l_max/l_min ≈ 2` (a factor-of-2 band, roughly what `l_min: 12,
l_max: 30`-scale settings give), same-seed fields synthesized at H = 0.5
and H = 0.9 are nearly indistinguishable — their scalar-field spatial
correlation exceeds 0.99 (measured directly on an actual
`WarpTessellation` object built while `spectrum: self_affine` was still
accepted, for this check: 0.995 on this exact narrow band, and the
resulting boundary's cross-section grain assignment differs on 0.0475 %
of sampled points between H = 0.5 and H = 0.9). Widening the band by an
order of magnitude (e.g. `l_min: 12, l_max: 150`, ≈ 1 decade) drops that
field correlation to 0.975 and raises the cross-section assignment
difference to 0.1125 % of sampled points (both measured the same way)
— still small.
**If you want the Hurst exponent's effect to be more pronounced** (for a
demonstration, a G13 back-estimation exercise, or a genuine
roughness-color study), budget **at least one decade** of `l_max/l_min`,
not a factor of ~2.

## 5. Additive weights & anisotropic

Two further `curved` methods target different microstructures. They warp nothing
— they change the *distance* used to assign points to grains.

**Additive weights** (`method: additive_weights`) — a Johnson–Mehl / Apollonius
diagram: each seed carries an additive radius, so cells meet on hyperbolic
(rather than planar) surfaces, modelling grains nucleated at different times.

| Field | Meaning | Default |
|-------|---------|---------|
| `weight_sigma` | weight spread σ_w (Å). **Constraint:** `σ_w ≤ min_seed_distance / 6` | `0.0` |

**Anisotropic** (`method: anisotropic`) — a per-grain ellipsoidal metric
(grain-boundary power-diagram family), producing **elongated** grains for
columnar or rolled textures.

| Field | Meaning | Default |
|-------|---------|---------|
| `aspect_ratio_range` | allowed `[min, max]` per-grain aspect ratios (both > 0, min ≤ max) | `[1.0, 1.0]` |

> **Scope note.** The anisotropic metric is a *Voronoi-class* backend (its
> per-grain metric is not volume-normalised), so it is **not** a volume-targeted
> anisotropic power diagram — it has **no volume handle at all**, and combining
> it with `grains.size_distribution` is refused as a config error (G1): volume
> targeting exists only for flat geometry or the `warp` method, whose base is
> the SDOT-fitted power diagram. To get curved boundaries *with* prescribed
> volumes, use `method: warp` over `base: power`
> (`examples/grain_geometry/cu_curved_sizes.yaml`). See
> [physics.md §3](physics.md#3-tessellation) for why only the power/SDOT
> backend targets volumes exactly.

```yaml
# examples/grain_geometry/fe_bcc_anisotropic.yaml — elongated grains
boundaries:
  geometry: curved
  curved: {method: anisotropic, aspect_ratio_range: [1.0, 3.0]}
```

```yaml
# examples/grain_geometry/al_equiaxed_lloyd_weights.yaml — Johnson–Mehl weights
boundaries:
  geometry: curved
  curved: {method: additive_weights, weight_sigma: 2.0}
```

## 6. Voxel import — externally-produced microstructures

**What it does.** `boundaries.geometry: voxel_import` bypasses seed-based
tessellation entirely and reads a **voxel label field** (one integer grain id per
voxel) produced elsewhere — e.g. a DREAM.3D synthetic microstructure or an
experimental reconstruction. grainsmith then fills each labelled region with an
oriented crystal exactly as for a computed tessellation.

**Parameters** (`boundaries.voxel_import`): the label-field path/dataset, the
voxel size, and how per-grain orientations are supplied (from the file or
sampled). See the [configuration reference](config_reference.md) for the full
`VoxelImportConfig` fields. HDF5 import requires the `[import]` extra (h5py).

**Construction conditions.** A voxel discretisation of a `(111)`-type plane
over-counts boundary voxels by a `√3` geometric factor; the analysis accounts for
this. The `analysis.voxel_grid` resolution is capped at
`VOXEL_GRID_MAX = 384³`.

```yaml
# examples/voxel_import/fe_voxel_import.yaml — import a label field
boundaries:
  geometry: voxel_import
  voxel_import: {path: microstructure.h5, ...}
```

## 7. Single crystal

Setting `grains.number: 1` with `geometry: flat` selects the single-crystal fast
path (`SingleCrystalTessellation`): no boundaries, one orientation, and a
commensurability check that reports the lattice/box misfit so you can pick a box
that tiles the crystal exactly.

```yaml
# examples/basics/cu_single_crystal.yaml
box: {lengths: [36.15, 36.15, 36.15]}
grains: {number: 1}
boundaries: {geometry: flat}
```

## Parameter → gate quick reference

| Parameter | Backend | Bound / gate |
|-----------|---------|--------------|
| `size_distribution.vol_tol` | power | G11: `max rel. volume error ≤ vol_tol` |
| `curved.amplitude` | warp | G6: `≤ min_seed_distance / 4` |
| `curved.spectrum` | warp | must be `gaussian` — `self_affine` is a `ConfigError` (Rule 8a) |
| `curved.hurst` | perturbed_distance + self_affine | G13 (warn): `|Ĥ − H| ≤ 0.15` |
| `curved.l_min` | perturbed_distance + self_affine | `≥ 2·h_field` (grid Nyquist), `< l_max` |
| `curved.l_max` | perturbed_distance + self_affine | `≤ min(L)/2` |
| `curved.weight_sigma` | additive weights | `≤ min_seed_distance / 6` |
| `curved.aspect_ratio_range` | anisotropic | both `> 0`, `min ≤ max` |
| warp field gradient | warp (gaussian spectrum only) | G6: `max‖∇u‖ < 0.5` |
| `curved.amplitude` | perturbed_distance | seed-containment guard: `≤ min_seed_distance / 12` (n/a for G6); under `amplitude_convention: reference_wavelength`, the REALIZED total-RMS-equivalent amplitude (`amplitude / κ`) is what is actually checked — Rule 29(d) |
| `curved.reference_wavelength` | perturbed_distance + self_affine + `amplitude_convention: reference_wavelength` | `∈ [l_min, l_max]` when given; must be unset under `amplitude_convention: total_rms` — Rule 29(a)–(c) |
| perturbed_distance connectivity | perturbed_distance | G19 (warn): `reassigned_fraction ≤ REASSIGNED_FRACTION_WARN_TOL = 0.02` |
| perturbed_distance box-counting roughness index | perturbed_distance | G20 (warn, reported only; NOT a converged, large-effect-size fractal-dimension certificate below ~5.5 synthesis-band octaves): `d_b_estimated`, fit restricted to `ε ∈ [l_min, l_max]` |
| `analysis.gb_curvature` | any (voxel-sampled on curved) | G16 (warn): `dropped_fraction ≤ CURV_G16_DROP_TOL = 0.05`; G21 (warn): per-grain `|Σ K·dA| ≤ CURV_G21_GAUSS_BONNET_TOL = 4π` |

See [gates.md](gates.md) for the full G1–G26 gate definitions and
[config_reference.md](config_reference.md) for every field with its type and
default.
