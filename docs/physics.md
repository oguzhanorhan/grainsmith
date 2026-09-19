# grainsmith physics & conventions

The conventions and formulae grainsmith implements, distilled from the code
docstrings. Citations are listed at the end and match
`constants.METHODS_BIBLIOGRAPHY`.

## 1. Orientations: quaternions and Bunge Euler angles

grainsmith uses **active, scalar-first unit quaternions** `q = (w, x, y, z)`. A
quaternion rotates a *crystal*-frame vector into the *lab* frame:

$$
v_{\mathrm{lab}} = R(q) \cdot v_{\mathrm{crystal}}
$$

where `R(q)` is the active rotation matrix. This is the orientation of the grain
(crystal → sample).

**Bunge Euler export.** The standard Bunge orientation matrix `g` maps
*sample → crystal* (the inverse direction), so

$$
g = R(q)^{\mathrm{T}} = \left[\mathrm{intrinsic\ ZXZ}(\varphi_1, \Phi, \varphi_2)\right]^{\mathrm{T}}
$$

`quat_to_bunge` extracts (φ1, Φ, φ2) from `R(q)` accordingly. This matches the
MTEX / EBSD convention: it was MTEX-round-trip-verified and pinned against the
textbook Bunge matrix and the ideal Brass component {011}⟨2-11⟩ → (35.26°,
45°, 0°). `grains.csv`, `odf_mtex.txt`, the EBSD-like section and the DREAM.3D
Euler reader all use this convention.

## 2. Disorientation, CSL, and boundary character

The **disorientation** between two grains is the misorientation of minimal angle
over the **double coset** of the point group (crystal symmetry applied on *both*
sides) — it is symmetry-invariant by construction (Grimmer / Brandon
conventions). `boundaries.csv` reports the disorientation angle and axis.

**CSL Σ** assignment uses a built-in **cubic** coincidence-site-lattice table; a
boundary is labelled ΣN when its disorientation lies within the **Brandon**
window Δθ ≤ 15°/√Σ of an exact CSL rotation [Brandon 1966]. This is a
distance between full rotation classes, not a test on the angle alone.
The implementation requires 24 proper cubic rotations; enabling
`analysis.csl` for an unsupported point group is an error. Disabled
classification leaves the CSV field blank. A blank classification with
CSL enabled means no match in the built-in finite table, not proof that
no higher-Sigma coincidence exists. Sigma3 classification alone also
does not establish a coherent twin plane or mechanical equilibrium.

**GB character** (tilt / twist / mixed) is the physical minimal-angle map between
the boundary-plane normal and the disorientation axis, with boundary-plane Miller
indices reported in both crystal frames.
Zero disorientation has no rotation axis: its axis indices are `(0,0,0)`,
axis deviation and character angle are `nan`, and character is `undefined`.
Geometrically defined boundary-plane indices and areas are retained.

## 3. Tessellation

**Flat Voronoi.** Planar Voronoi faces via Qhull. Periodic boundaries are handled
by *mirror seeds*; the wall-clipping inequality is **exact** (a d² half-space
test), and the `owns()` rule tiles the torus **exactly once** — atoms are filled
by tiling each grain's compact home cell, with no wrap-and-dedupe.

**Power / Laguerre diagram + SDOT.** To prescribe grain *volumes*, grainsmith fits
power-diagram weights by **semi-discrete optimal transport**: damped Newton on
the Kantorovich dual [Kitagawa–Mérigot–Thibert 2019], applied to polycrystals by
[Bourne et al. 2020, 2023]. With cell volumes V(w) and targets V*:

$$
F(w) = V(w) - V^*, \qquad J_{ij} = \frac{\partial V_i}{\partial w_j} = -\frac{A_{ij}}{2 d_{ij}} \ (i \neq j), \qquad
J_{ii} = +\sum_j \frac{A_{ij}}{2 d_{ij}}
$$

`A_ij` is the exact interface area and `d_ij` the seed distance — both read off
the polyhedral faces (which is why the power backend keeps exact geometry, not
voxel estimates). `J` is a graph Laplacian (rows sum to 0; gauge fixed by pinning
w₀ = 0). KMT damping backtracks the step so no cell ever vanishes — global
convergence, quadratic near the solution. An optional outer **centroidal** loop
[Kuhn et al. 2020] moves seeds to power-cell centroids for equiaxed grains *with*
prescribed volumes. Gate **G11** re-measures the achieved volumes
(measured: ≤ 1.2 × 10⁻⁴ relative in 3–6 Newton iterations).

**Curved boundaries.** grainsmith's user-facing GB-morphology taxonomy has
three tiers: **flat** (Voronoi/power, §3), **curved**, and **self-affine**.
Within `boundaries.geometry: curved`, three methods produce a *smoothly
curved* boundary: a periodic Gaussian random field warps the diagram
(`warp` method, **gaussian spectrum only** — see §5b for why) under
bijectivity guards (G6); `additive_weights` (Johnson–Mehl/Apollonius) and
`anisotropic` (ellipsoidal metric) reshape it by a different local metric.
Only a bijective `warp` preserves its base diagram's topology. Additive
weights and grain-dependent anisotropic metrics can change neighbours;
their adjacency is measured from their actual voxelized geometry.
The self-affine tier is a single method, `perturbed_distance` (§5/§5b): its
level-set folding is not a coordinate map, so it is the one curved method
whose as-built boundary genuinely responds to a self-affine roughness
spectrum, and the one place (along with `voxel_import`) genuinely
non-Voronoi shapes come from.

## 4. Texture (ODF) and boundary-angle statistics

**ODF components.** `orientation.scheme: odf_components` builds a discrete ODF
from weighted components: an Euler centre with isotropic spread (an SO(3) von
Mises–Fisher small-angle surrogate with the exact Haar sin²(θ/2) correction), a
fiber component, or one Haar-uniform random fraction. How a component's
`weight` is *realised* is `orientation.component_weight_basis`: `"count"`
draws each grain's component i.i.d. from `Categorical(weight)`, so `weight`
is a GRAIN-COUNT fraction; `"volume"` (the default) instead partitions
grains deterministically so `weight` is a VOLUME fraction — see below for
why that distinction matters for the ODF specifically.

**Disorientation-angle shaping.** `orientation.mdf_target` pulls the neighbour
misorientation-angle distribution toward a target by **simulated annealing that
permutes which grain gets which orientation** over a fixed orientation
multiset — Miodownik et al. 1999's own term for the swap-only variant is
**conserved texture** [Miodownik et al. 1999]. Because the set of orientations
never changes, only *who carries* each one, the NUMBER-weighted discrete
orientation distribution is invariant by construction. The ODF as
conventionally defined is **volume**-weighted, not number-weighted — f(g) by
volume fraction, the MDF f(Δg) by area fraction [Saylor et al. 2004] — and it
is **not** invariant in general: swapping distinct orientations on grains a and b moves
|V_a − V_b| / ΣV of its total-variation mass between the two orientations
swapped. `mdf_target.odf_drift_max` bounds that drift directly — a hard cap
enforced on every accepted swap — and because kernel smoothing is a
total-variation contraction, the same cap also bounds the drift of the
kernel-smoothed ODF at every half-width (see `orientation/odf.py`). The
default `odf_drift_max: null` only MEASURES drift; it does not preserve
the volume-weighted ODF. This one-dimensional angle histogram is not the
full distribution over misorientation space. Targets:
`haar_random` (deprecated alias `mackenzie`; the random-pair
disorientation-**angle** reference of the actual point group, equal to the
Mackenzie 1958 law only for the cubic proper point group), `sigma3_angle_enriched`
(deprecated alias `csl_enriched`; `(1-f)*haar_random + f*angle_window`, where
`f = sigma3_fraction` and the added distribution occupies the 60° Brandon
**angular** window — the ⟨111⟩ axis condition CSL
classification also requires is not enforced), or an explicit `histogram`. Gate
**G12** reports the final χ² distance; `mdf.csv` carries the achieved, reference
and target densities. `sigma3_fraction` is a mixture coefficient, NOT the
total target mass in the angular window: the Haar reference already has
mass there. Neither quantity is the true CSL Σ3 area fraction, measured
independently using both the angle and the axis.

**Component-fraction error budget: sampler + annealer.** For the
component-labelled volume-fraction vectors, the triangle inequality gives
**final error <= sampler error + annealing change**, not equality and not
statistical independence. The *sampler term* is the distance between each component's configured
`weight` and the volume fraction `odf_components` assigns it, before any
annealing runs at all. The *annealer term* is the assignment-annealing drift
in that component-labelled vector. This coarse component diagnostic does
not measure finite-sampling error within each angular component, and is
not the distance from a continuous target ODF to a discrete empirical
measure. G22 separately measures and, when configured, bounds drift on
exact orientation classes. Coincident or overlapping component labels
must not be mistaken for distinct physical orientations. Before this revision only
the annealer term was under any control — the sampler term was implicitly
assumed zero, which is false for `component_weight_basis: "count"`: drawing
each grain's component i.i.d. from `Categorical(weight)` realises `weight` as
a GRAIN-COUNT fraction. The realised component-volume error depends on
both grain sizes and the sampled component assignment. Measured on a 24-grain case
(`grains.size_distribution` lognormal, `sigma_log = 0.5`, two texture
components at `weight: 0.5` each): configured volume fractions 0.500/0.500
realised as 0.745/0.255 — a total-variation deviation of 0.245, **twelve
times** the 0.02 trust-region cap a typical `odf_drift_max` setting bounds
the annealer term to. The uncontrolled sampler term was the larger of the
two error sources, by more than an order of magnitude.

`orientation.component_weight_basis: "volume"` (the default from this
revision on) reduces the component-fraction sampling error, but cannot in
general eliminate finite-grain granularity, via
`orientation.odf.volume_balanced_partition`: sort grains by volume,
descending; repeatedly hand the next (largest remaining) grain to whichever
component currently has the largest unassigned volume deficit
(`weight * total_volume − volume_assigned_so_far`), and reduce that
component's deficit by the grain's volume. No RNG is involved, so the
assignment is deterministic and bit-reproducible. With two grains of
volumes 80 and 20 and target component fractions 0.5/0.5, for example,
the best whole-grain assignment still has a TV error of 0.3. Gate G25
reports this residual instead of certifying an exact match.

> **Lemma (no component overshoots by more than its smallest grain).** When
> grain *i* is assigned to component *c*, the sum of all components'
> deficits equals the volume not yet assigned, which is at least
> `volumes[i] > 0`; hence the *largest* deficit at that moment is at least
> `remaining / K > 0` for *K* components, so `deficit_c` is chosen `>
> −volumes[i]` after the assignment. A component's deficit changes again only
> when it receives another grain — which, by the descending processing
> order, is no larger than *i*. Therefore no component's realised volume
> ever exceeds its target by more than the smallest grain it received.

**Honest cost.** Demanding an exact volume match on a finite grain set with a
broad size distribution necessarily induces a correlation between grain size
and component: the lemma's mechanism deterministically routes the largest
remaining grain to whichever component is furthest short, so large grains
disproportionately land wherever the running deficit is largest. This is a
stated trade-off, not a bug — it shrinks as the grain count grows (the
lemma's bound is one grain's worth of granularity error, not a
whole-distribution one), and gate **G25** reports it directly: per
texture component, the configured weight, the realised grain-count and
volume fractions (pre- and post-annealing), and the mean grain volume, so
the size-component correlation is visible in the same table as the fidelity
numbers it is the cost of.

> **Honest bound (measured).** Annealing returns the lowest angle-histogram
> χ² encountered; individual accepted Metropolis moves may increase it,
> but the achievable Σ3 *area* fraction is bounded by the twin content of the
> fixed orientation set and the Voronoi adjacency graph — it does not track the
> nominal `sigma3_fraction` linearly. This is why G12 is a *warn* gate, and is
> a property of **conserved-texture** assignment [Miodownik et al. 1999] — the
> orientation multiset is fixed, only which grain carries which orientation
> changes — not a defect.

**The atomistic discretisation floor (gate G26).** Everything above — the
sampler term, the annealer term, `odf_drift_max`, gate **G22** — is
defined on **tessellation** grain volumes: `odf_mtex.txt` writes
`volumes / sum(volumes)` as its weight column, and that is the vector
`odf_drift_max` bounds. But the artefact actually shipped to an MD user
is **atomistic**. Atom-count weighting is a useful discrete texture proxy
within a phase when atomic volumes are equivalent; it is not a universal
replacement for the physical volume-weighted ODF in mixed-density,
strained or compositionally heterogeneous material. Those two weightings — `w_vol = volumes / sum(volumes)` and
`w_atom = n_atoms / sum(n_atoms)` — differ, and gate **G26** measures the
gap directly rather than leaving it unmeasured. Reference scenario (24
grains, box 45 Å, `size_distribution` lognormal `sigma_log 0.5`, two
texture components at `weight 0.5`, seed 4242, cubic Cu; same config, box
size varied):

| atoms/grain | TV(volume vs atom-count) | max per-grain \|dw\| |
|---|---|---|
| 275  | 0.027418 | 0.009107 |
| 1092 | 0.018910 | 0.006912 |
| 3279 | 0.013133 | 0.004724 |
| 9191 | 0.008633 | 0.003126 |

A fitted power law gives exponent **−0.33**: the gap decays as `(atoms
per grain)^(-1/3)`, i.e. as a **surface**, not a volume, quantity. The
mechanism is geometric, not statistical: every atom within one
interatomic spacing of a grain boundary is only probabilistically
assigned to one side or the other, and that boundary shell's atom count
is a fixed fraction of the grain's *surface area*, which scales as `1/L`
of its *volume* for a grain of characteristic size `L`. This is why the
exponent is `-1/3` and not the `-1/2` that pure Poisson counting noise
(atom placement treated as independent draws with no spatial structure)
would predict — the fitted exponent rules that simpler mechanism out.
**At 275 atoms per grain the floor (0.027) already exceeds a typical
`odf_drift_max` of 0.02**: tightening the trust region below this floor
buys no additional fidelity that survives into the exported atomistic
structure, since the annealer's own bound is defined on a weighting the
export does not use. This is a **discretisation floor**, not a defect in
the annealer or the sampler — it exists for any atomistic realisation of
a continuous volume-weighted target, and shrinks only with more atoms per
grain (larger grains, or more atoms at fixed grain count), never with a
tighter annealing setting. Gate **G26** reports it per run (globally for
single-phase, per phase for multiphase — atom number densities differ
across phases, so a global mix would read off the density ratio between
phases rather than the discretisation floor itself), so the
`odf_drift_max` guarantee can be read for what it actually covers.

`odf_mtex.txt` itself now carries both weightings, not just the one G26
reports on: whenever the run has a nonzero atom count, the file gets a
fifth column, `atom_fraction = n_atoms / sum(n_atoms)`, alongside `weight`
(tessellation-volume fraction), and its header spells out which is which
and why they differ. A texture specialist loading the file straight into
MTEX therefore sees both the geometric weighting grainsmith controls and
the realised, atom-count weighting the shipped structure actually carries,
instead of only the former. With no atom block, or an all-zero atom count,
the file is unchanged — the original four columns.

## 5. Self-affine boundary roughness

`spectrum: self_affine` (`boundaries.curved.method: perturbed_distance`
**only** — see §5b for why `warp` cannot accept it) replaces the
single-scale Gaussian field with a **band-limited power-law** amplitude
spectrum

$$
\sqrt{\Phi(k)} \propto k^{-(3+2H)/2}, \qquad k \in \left[\frac{2\pi}{l_{\max}}, \frac{2\pi}{l_{\min}}\right]
$$

so the 3D field has power P(k) ∝ k^(−(3+2H)) and its planar restriction has
surface PSD **C(q) ∝ q^(−2−2H)**, i.e. fractal dimension **D = 3 − H**
[Jacobs et al. 2017 conventions]. Self-affine grain-boundary morphology at
the nanoscale is experimentally established [Braun et al. 2018]; the
fractal-surface construction route for MD follows [Eder et al. 2017]. Gate
**G13** back-estimates H from the generated field(s) by a radial-PSD
log-log fit; because the scale-free band is finite, the estimate carries a
small constant offset (measured: ~0.1), so G13 is a *warn* gate,
tolerance `HURST_G13_TOL = 0.15`. This is a field-synthesis convention
shared verbatim by `warp` and `perturbed_distance` (`synthesize_grf`/
`estimate_hurst`, `tessellation/warp.py`) — what differs between the two
methods is not the spectrum math but what CONSUMES the field, which is
exactly §5b's subject and why only one of them can use it.

## 5b. Self-affine GB morphology: why `warp` cannot deliver it, and how `perturbed_distance` does

Section 5's `spectrum: self_affine` gives the *warp field* itself a self-affine
PSD, but that is not the same claim as the *grain boundary* being self-affine —
and, perhaps counter-intuitively, it never is under `warp`, at any Hurst
exponent. This section states the argument precisely and describes the
`boundaries.curved.method: perturbed_distance` backend that resolves it by
perturbing the assignment rule instead of the coordinate frame.

**(a) The diffeomorphism argument.** `warp` computes `grain_of(x) =
base.grain_of(x + u(x))` under the bijectivity guard G6 (`max‖∇u‖ <
WARP_GRAD_MAX = 0.5`, plus the seed-containment clip `‖u‖ ≤
min_seed_distance/4`). A field satisfying both bounds is, by construction, a
**bi-Lipschitz homeomorphism** of the periodic domain: `x ↦ x + u(x)` and its
inverse are each Lipschitz-continuous with a constant controlled by
`WARP_GRAD_MAX`. A bi-Lipschitz map distorts distances by at most a bounded
factor in either direction, and the box-counting dimension of a set is
invariant under any bi-Lipschitz reparametrization of the ambient space — a
covering of the image by `N(ε)` boxes of size `ε` pulls back to a covering of
the preimage by `N(ε)` boxes of a comparably-scaled size, so the two sets'
`log N(ε)/log(1/ε)` limits coincide. The flat-Voronoi bisector plane has
box-counting dimension 1 (a 2D section) / 2 (the full 3D surface); its image
under any admissible `u(x)` therefore *also* has dimension 1 / 2, **regardless
of the Hurst exponent driving `u`'s spectrum**. What the self-affine spectrum
changes is only the *local geometric roughness* of that dimension-preserving
surface — its curvature statistics, its visual waviness — not the
scale-invariant space-filling exponent that a box-counting measurement
reports. `hurst` on a `warp` field (back when `warp` could accept
`spectrum: self_affine` — see below) was real and useful (it sets how the
boundary looks and how a Hurst back-estimate characterizes it), but it was
always a **roughness-color** knob, not a **fractal-dimension** knob. A
direct numerical check makes this concrete: box-counting 2D sections of an
ACTUAL `WarpTessellation` object (its own trilinear field interpolation
and clip logic — not a hand-rolled stand-in) built while `WarpTessellation`
still accepted `spectrum: self_affine`, on a 460 Å / 10-grain box, run at
TWO amplitudes (both measured directly from the real class):
at `amplitude = 0.4 Å`, `l_min/l_max = 12/30 Å` (max‖∇u‖ up
to 0.469, under the G6 guard's 0.5 limit) — sub-pixel at this check's
200×200 section sampling (2.3 Å/px) — `D_b ≈ 1.017`–`1.018` at H = 0.5,
0.7 **and** 0.9 alike, indistinguishable from and in fact slightly BELOW
§5b(d)'s own `D_b ≈ 1.027` straight-edge pixelation floor: at this
amplitude the boundary is not just failing to show a Hurst-driven signal,
it is too small to register even the ordinary curvature bias §5b(d)
documents for a general curved section; and at `amplitude = 4.0 Å`,
`l_min/l_max = 100/225 Å` (max‖∇u‖ up to 0.494, the largest the REAL G6
guard admits on this wider band — 10× the smaller amplitude used above,
large enough to displace the boundary by ~1.7 section pixels)
— `D_b ≈ 1.032`–`1.034` at H = 0.5, 0.7 **and** 0.9 alike, still
statistically indistinguishable across H. The second, larger-amplitude
check is the one that actually discriminates "provably
dimension-invariant" from "too small to move the boundary at all" (the
first, at the file's own tiny amplitude, cannot rule out the latter on
its own); together they confirm the dimension-invariance argument holds
in the actual implementation across the amplitude range the G6 guard
admits, not just in the idealized proof. Because this numerical
hollowness makes the combination
scientifically indefensible for its one stated purpose (a genuinely
self-affine boundary), grainsmith REMOVES it outright: `WarpTessellation`
raises `ConfigError` for any spectrum other than `gaussian`, and
`config/resolve.py` Rule 8a rejects `method: warp` + `spectrum:
self_affine` before any other curved-boundary rule is even checked. §5's
`self_affine` spectrum is therefore `perturbed_distance`-only — see (b).

**(b) The level-set rule.** To get a grain boundary whose box-counting
dimension genuinely responds to a roughness spectrum, the assignment rule
itself must **fold** — i.e. be non-injective near the boundary, something a
diffeomorphism can never be. `perturbed_distance` replaces the flat-Voronoi
membership test with a perturbed-distance argmin:

$$
\mathrm{grain\_of}(x) = \operatorname*{argmin}_i \left[ d_{\mathrm{pbc}}(x, s_i) - A \cdot \eta_i(x) \right]
$$

where each grain draws an **independent** scalar field `η_i` from the same
`synthesize_grf` machinery Section 5 uses (`spectrum`/`hurst`/`l_min`/`l_max`
are the identical schema keys, reused verbatim — only the field is now
scalar-per-grain rather than a shared 3-component displacement). The zero
level set `d(x,s_i) − A η_i(x) = d(x,s_j) − A η_j(x)` is an implicit surface
whose roughness is inherited directly from `η_i − η_j`, with no bijectivity
constraint to preserve the flat bisector's topology — so its box-counting
dimension is free to move away from the Voronoi-facet value as `A` and the
field statistics change. Because there is no coordinate map to invert, **G6
does not apply** to this method; two different guards take its place: a
seed-containment pre-flight check (`A ≤ PERTURBED_DISTANCE_SAFETY ·
min_seed_distance / (2 · ETA_CLIP)`, i.e. `min_seed_distance/12`) and an exact
post-hoc seed-ownership check at construction (every grain's own seed must
still evaluate to itself under the perturbed rule). Any voxel-scale
fragmentation the folding can locally cause is handled by demoting G5 to a
repair-and-report gate for this method (majority-vote reassignment; severity
reported by **G19**) rather than the hard construction-time failure
`warp`/`weighted` use — see `docs/gates.md` for both gates' exact thresholds.

**(c) Experimental target.** The motivating measurement is Braun et al.'s
box-counting study of nanocrystalline Pd₉₀Au₁₀ [Braun et al. 2020]: annealed
samples exhibiting abnormal grain growth with dendritic, highly convoluted
boundaries gave a mean section-perimeter fractal dimension **D_b = 1.174 ±
0.004** (averaged over 57 grains), with individual grains ranging **1.10 to
1.24**, versus D_b ≈ 1.029 for conventionally cast, smooth boundaries — both
numbers verified directly against the source measurement [Braun et al. 2020].
This is the target `examples/advanced/adv_pdau_perturbed.yaml` — grainsmith's
flagship self-affine-GB example — is calibrated against; before this method
existed, an earlier `warp`-based version of that same flagship targeted the
same D_b band via the section-dimension convention **D_b = 2 − H** [Jacobs et
al. 2017 PSD conventions, consistent with Section 5's 3D surface convention D
= 3 − H] — a convention BRIDGE to the literature, not a measurement of that
file's own (non-self-affine, per (a)) boundary. `perturbed_distance` reaches the
same target by an entirely different, and genuinely measurable, mechanism
(amplitude-driven folding rather than H-driven roughness color) — see (e) for
its own honest calibration status.

**(d) Estimator bias.** `perturbed.py`'s `box_count_dimension` follows the
Braun et al. 2020 box-counting protocol on 2D in-box cross-sections (`log
N(ε)` vs. `log(1/ε)` linear fit; the G13 analogue, reported as **G20**,
warn-only). Box sizes are fit over `ε ∈ [l_min, l_max]` — the synthesis band
itself, converted to each section's own pixel scale — so the reported exponent
reflects only the scales the η field actually has designed spectral content
at, not pixelation noise below `l_min` or finite-object-size cutoff above
`l_max`. At this implementation's own section resolution
(`SECTION_GRID = 200` px), the estimator carries a positive bias for
smooth curved boundaries relative to dead-straight ones: a straight edge
box-counts to `D_b ≈ 1.027`, while a rasterized disk perimeter lands between
`D_b ≈ 1.07` and `D_b ≈ 1.15` **depending on the disk radius and the
perimeter-extraction stencil** (measured at radii 50–80 px on the same 200 px
grid). The bias is therefore not a single constant but an offset of order
**+0.04 to +0.12**, attributable purely to finite-pixel rasterization of a
curved perimeter, not to any genuine self-affine signal. This
sits alongside (not instead of) the geometric flat-Voronoi limit: a true `A =
0` perturbed-distance run box-counts to `D_b ≈ 1.07` (not the idealized 1.000)
for the same reason. Every `d_b_estimated` value this method reports should
be read as offset upward by roughly this order from the abstract
box-counting-of-a-mathematically-exact-curve value — a real property of a
finite-resolution voxel/section pipeline, stated rather than silently
corrected, in keeping with Section 7's estimator-bias philosophy.

**(e) Calibration status (honest).** A dedicated amplitude sweep (H ∈ {0.5,
0.7, 0.9}, 60 grains, 300 Å box, two field-seed repeats per cell) confirms
`d_b_estimated` rises **monotonically** with amplitude at every Hurst tested,
from the flat-Voronoi-like `A = 0` reference (`D_b ≈ 1.07`, this estimator's
own reading) up through the seed-containment safety ceiling `A_max =
min_seed_distance/12`. At that ceiling the measured `D_b` reaches only
**≈1.11–1.12** — inside Braun's reported per-grain *range* (1.10–1.24) but at
its lower edge, short of the 1.174 *mean*. The Hurst dependence at the
ceiling is real on ensemble average (lower H → slightly higher D_b) but
smaller and noisier than the amplitude effect, and the two are not cleanly
separable at the amplitudes this guard permits. Reaching the upper part of
the Braun band with a larger safety margin — a tighter per-axis containment
argument, or accepting a larger `reassigned_fraction` — is open calibration
work, not a claim this backend currently makes; `examples/self_affine_gb/pdau_perturbed_self_affine.yaml`
states its own measured `d_b_estimated`/`reassigned_fraction` explicitly
rather than presenting the method as a turnkey D_b = 1.174 generator.

This sweep's own box (300 Å, `l_min`/`l_max` narrower than one decade,
~1.5 octaves) sits well below the ≈5.5-octave band (g) below shows is
needed for a LARGE-effect-size (η² > 80%) Hurst response — so the
"amplitude effect dominates, Hurst effect is real but small and noisy"
pattern reported here is itself partly a decade-budget symptom, not only
an amplitude-ceiling one. `examples/self_affine_gb/pdau_perturbed_50nm_H05.yaml` is
sized specifically to reach that regime (≈50 nm grain diameter, 5.5
octaves) — see (g) below.

**(f) Amplitude reparametrization — the coarse-octave inversion, and its
fix.** An independent validation campaign (131 runs) found that
adding a coarse octave to the synthesis band (widening `l_max` at fixed
`l_min`) at fixed `amplitude` **decreased** the realized grain-boundary
area `S_V` by 16.1% — the opposite of the physically expected direction.
The root cause is in `synthesize_grf`'s normalization, not in the
level-set argument of (b): the default `amplitude_convention: total_rms`
fixes `amplitude` as the field's RMS over the **whole** band `[l_min,
l_max]`. For the `k^(−(3+2H))` target spectrum, the band-integrated
variance is dominated by its longest wavelengths, so widening the band
redistributes a fixed total-RMS budget across more scales rather than
adding to it — at fixed `amplitude`, a coarser `l_max` can starve the
`l_min`-scale content that actually drives local boundary roughness.

`amplitude_convention: reference_wavelength` (opt-in; default remains
`total_rms`) fixes instead the
RMS power contributed by **one reference octave**,
`[reference_wavelength/2, reference_wavelength]` (default anchor: `l_max`,
an anchor that MOVES with the band), leaving every other resolved octave
free to accumulate its own contribution on top. The two conventions are
related by one exact, invertible scale factor:

$$
\kappa = \frac{\sigma(\text{reference octave})}{\sigma(\text{whole band})} = \texttt{tessellation.warp.reference\_shell\_kappa(hurst, l\_min, l\_max, reference\_wavelength)}
$$

$$
\mathrm{amplitude}_{\text{reference\_wavelength}} = \mathrm{amplitude}_{\text{total\_rms}} \cdot \kappa \qquad (\kappa \le 1 \text{ always})
$$

A synthetic sign-test (fixed white-noise draw,
scanning `l_max` at fixed `l_min`) confirms the qualitative picture but
also its limit: the `total_rms` convention's local-roughness proxy
decreases monotonically as octaves are added (reproducing the campaign's
inversion), while `reference_wavelength` with a **fixed absolute**
`reference_wavelength` (anchored near `l_min`, not moving with `l_max`)
accumulates monotonically in the other direction. The **default**
`reference_wavelength` anchor (`= l_max`, moving with the band) does
*not* fully invert the trend on this diagnostic — it is intended for the
common case where `l_max` itself represents a physical scale (e.g. the
grain radius) that should stay the roughness reference as other settings
change, not for a controlled band-width scan at fixed physical scale. A
band-width scan that must accumulate variance monotonically in both
directions needs an explicit **fixed absolute** `reference_wavelength`
override, not the default.

**What the reparametrization does and does not fix.** It restores a
physically stable meaning for `amplitude` (Å-RMS at a named wavelength,
independent of how many octaves happen to be in band) and corrects the
coarse-octave sign inversion under a fixed absolute reference. It does
**not**, by itself, generally increase the Hurst exponent's statistical
separability in `d_b_estimated` — a feasibility study found
both conventions give similar effect sizes (η²) at matched grain size,
and in a majority of the grain-size points tested the ORIGINAL `total_rms`
convention's own η² was nominally as high or higher. What actually
controls whether `d_b_estimated` can resolve `hurst` is the **octave/decade
budget** of the synthesis band (see (g) below and G20's own docstring) —
this is a property of `[l_min, l_max]` and grid resolution, unaffected by
which amplitude convention scales the result. The seed-containment guard
binds the realized total-RMS-equivalent amplitude in either convention
(`config/resolve.py` Rule 29(d); `PerturbedDistanceTessellation.__init__`
converts `amplitude` back to its total-RMS equivalent, ÷ κ, before
checking it against `A_max`), so a convention switch can only make the
guard *stricter* (κ ≤ 1 ⇒ `amplitude/κ ≥ amplitude`), never weaker.

**(g) The decade/octave budget is the dominant limitation, not the
amplitude convention.** The feasibility study that produced (f)
above also swept grain diameter (hence, at fixed atomic-scale `l_min`,
the number of octaves the synthesis band spans) from ~10 nm to ~1000 nm
and measured the statistical separability of `d_b_estimated` by Hurst
exponent (effect size η², and its p-value) at each size, under BOTH
amplitude conventions. **This is a gradient, not a single cliff** — the
independent 131-run validation campaign's own ~1.7-octave band
(`pdau_perturbed_self_affine.yaml`) measured η² ≈ 6% at p ≈ 0.14 (statistically
indistinguishable from no effect), while the feasibility study's synthetic-
field Monte Carlo gives, at comparable amplitudes under both conventions:

| grain diameter | octaves | η² (`total_rms` / `reference_wavelength` convention) | worst-case p-value (either convention) |
|---|---|---|---|
| ~10 nm | 3.2 | 34% / 29% | 5×10⁻⁶ (significant) |
| ~20 nm | 4.2 | 66% / 59% | 1×10⁻¹⁴ |
| ~50 nm | 5.5 | 83% / 82% | ≈0 (below float64 precision) |
| ~100 nm | 6.5 | 90% / 91% | ≈0 (below float64 precision) |

(η² values from a direct Monte Carlo sweep over
Nyquist-safe grids; both conventions track closely at every size.) Two
different thresholds are worth keeping separate: the effect becomes
**statistically significant** (p ≪ 0.01) already by ~3.2 octaves, but only
reaches a conventionally **LARGE effect size** (η² > 80%, Cohen's convention)
at ≈50 nm grain diameter / ~5.5 octaves. Which threshold matters depends on
the goal — a config merely wanting a real (non-noise) Hurst signal in
`d_b_estimated` can get one well below 50 nm; a config wanting a
Braun-comparable, large-effect-size measurement needs the ~5.5-octave regime.
This mirrors Braun et al. 2020's own protocol constraint: their box-counting
range needs `ε_min ≥ 3s` and (in their study) `ε_max = 10·ε_min`, which
together fix a minimum analyzable equivalent grain diameter `d_min =
ε_max/0.4` (Braun et al. 2020, Eq. 2) — an experimental instantiation of the
same "you need enough decades in your box-counting range" constraint this
backend's synthesis band faces. The practical implication: a
`perturbed_distance` config aimed at *demonstrating* a large-effect-size,
Braun-comparable Hurst response in `d_b_estimated` (as opposed to a
merely-significant one, or just producing plausible boundary roughness) needs
a `l_max/l_min` ratio corresponding to ~5.5 octaves, which — since `l_min` is
bounded below by roughly twice the crystal's own nearest-neighbor distance —
in practice means a grain radius (hence `l_max`) at or near the upper end of
the molecular-dynamics-accessible size range (≳50 nm).

## 6. Multiphase polycrystals

`phases:` assigns each grain to a phase by a **deterministic greedy (LPT)
partition** of the measured pre-fill grain volumes, so the achieved VOLUME
fractions (the metallographic convention) match the targets as closely as the
grain-count granularity allows. Phases may differ in space group, lattice **and**
elements. Interphase boundaries carry a `NaN` misorientation (orientation
relationship is undefined across different lattices) but keep their habit-plane
Miller indices. Gate **G15** re-measures the fractions; the deviation floor is the
granularity bound V_max/V_box. Per-phase `mdf_<name>.csv` /
`odf_mtex_<name>.txt` are written.

**Multiphase outputs are geometric starting structures, not relaxed
interfaces.** grainsmith does **not** impose any crystallographic orientation
relationship (Kurdjumov–Sachs, Nishiyama–Wassermann, Burgers, …) across
phases: each phase is oriented independently, and the two lattices simply meet
on the Voronoi habit plane. Across a heterophase boundary the lattice
parameters and densities differ, so the as-built interface is generally
high-energy and non-equilibrium; the uniform `0.85·d_nn` overlap removal only
deletes geometrically coincident atoms, it does not build a coherent
interface. **Relax every multiphase configuration in MD (energy minimization +
equilibration) before using it** — see the companion LAMMPS benchmark
simulations (e.g. the Cu/Fe composite). Imposing interphase orientation
relationships is planned future work.

## 7. Estimator biases (stated, not silently corrected)

- **Voxel area/volume/length bias.** Flat geometry uses exact polyhedral
  estimators; voxel/curved geometry over-counts oblique boundary area by up to
  **√3** and triple-line length by the voxel staircase. The
  `estimator` field in `statistics.csv` records which was used; finer
  `analysis.voxel_grid` reduces the bias.
- **KS with fitted parameters.** The log-normal `lognormal_ks_p` uses the
  fitted μ̂, σ̂ and is therefore optimistic — the **Lilliefors** caveat
  [Lilliefors 1967].

## 8. Grain-boundary curvature

`analysis.gb_curvature` measures local mean (H, 1/Å) and Gaussian (K, 1/Å²)
curvature of each grain–grain interface from the level set

$$
\varphi = \mathrm{margin}_i - \mathrm{margin}_j
$$

where `margin` is `Tessellation.margin`, the signed distance to the nearest
boundary, positive *inside* its own grain (the same convention used
throughout the fill/overlap/boundary code). The unit normal
`n̂ = −∇φ/|∇φ|` points from grain `i` toward grain `j`, and

$$
H = \frac{\nabla\varphi^{\mathrm{T}} \cdot H_\varphi \cdot \nabla\varphi - |\nabla\varphi|^2 \cdot \mathrm{tr}\,H_\varphi}{2|\nabla\varphi|^3} \qquad \text{[Goldman 2005]}
$$

$$
K = \frac{\nabla\varphi^{\mathrm{T}} \cdot \mathrm{adj}(H_\varphi) \cdot \nabla\varphi}{|\nabla\varphi|^4}
$$

with `Hφ` the Hessian of φ and `adj(·)` the matrix adjugate. **Sign
convention.** `H > 0` ⇔ the center of curvature lies on the grain-`i` side ⇔
grain `i` is locally convex there — a spherical grain `i` of radius R embedded
in a matrix `j` reports `H = +1/R`, `K = +1/R²` (negate `H` for grain `j`'s
perspective; `K` is invariant under the normal flip).

**Sampling.** Derivatives are evaluated with 19-point central-difference
stencils (the 3×3×3 neighbourhood minus the 8 corners the central differences
never touch) using per-axis steps `h_vec` tied to the analysis voxel grid, at
the boundary sample points only — no full-grid φ is built. Samples whose
gradient magnitude falls below `CURV_GRAD_MIN` (`|∇φ| < 0.1`) are degenerate:
dropped, counted, and reported through gate **G16** (warn above 5 % dropped,
`CURV_G16_DROP_TOL`). The stencil spacing also sets the practical resolution
limit of the measurement, ≈ `1/(2h)`, where `h` is the voxel edge.

**Flat geometry is exact, not sampled.** `FlatTessellation.margin` is
piecewise planar (kinks only at face edges), so flat runs never reach the
level-set kernel above: the flat path writes literal zeros from the exact
face polygons instead — `H = K = 0` by construction, for every sample.

**Caveat (inherited estimator bias).** Curved/voxel geometry reuses the
voxel-face sample points of the boundary-area estimator (§7), so the same
staircase area-weighting bias applies to the *area* column
(`gb_curvature.csv`'s `area_A2`) used to weight the per-boundary H/K
summaries in `boundaries.csv` — the curvature *values* themselves are
computed from the level set directly and do not carry that bias, but their
area-weighted aggregates inherit it exactly as `statistics.csv`'s S_V does.

## 9. Dopant insertion

`doping:` adds substitutional (optionally GB-segregated) and interstitial
dopants to the finished host structure. **Shell/bulk mass balance.** For a
target dopant count N_t split across shell candidate sites N_s (within
`shell_width` of a boundary) and bulk candidate sites N_b, with enrichment
target E = c_shell/c_bulk,

$$
p_{\mathrm{bulk}} = \frac{N_t}{N_b + E \cdot N_s}, \qquad p_{\mathrm{shell}} = E \cdot p_{\mathrm{bulk}}
$$

are the per-site Bernoulli acceptance probabilities; a probability exceeding 1
is infeasible and raises `ConfigError` naming the feasible maximum (either the
enrichment or the concentration ceiling, whichever bound is violated).

**Joint final-total targeting.** Every dopant's `concentration` is a fraction
of the *final* structure, and substitution never changes the atom total, so
all dopants' targets are solved against the same estimated final total up
front: with `c_interstitial` the sum of all interstitial dopants'
concentrations and `n0` the pre-doping atom count,

$$
n_{\mathrm{final\_est}} = \frac{n_0}{1 - \sum c_{\mathrm{interstitial}}}
$$

solving every target sequentially against a running total would shortchange
earlier dopants by a relative factor of roughly the later dopants'
concentrations, which the joint solve avoids.

**Interstitial candidate sites** are generated with the ordinary per-grain
fill machinery: the interstitial sublattice is just an extra fractional basis
on that grain's own lattice and orientation, so candidates automatically
follow each grain's crystallography. The basis comes from one of two input
forms covering all 230 space groups:

* a **built-in preset** keyed by required space group (`fcc_octahedral` /
  `fcc_tetrahedral` → SG 225, `bcc_octahedral` / `bcc_tetrahedral` → SG 229,
  `hcp_octahedral` / `hcp_tetrahedral` → SG 194 with the motif on Wyckoff
  2c) — presets are stored as complete orbits and used as-is; or
* **explicit fractional coordinates** (`sites.coords`) for any structure.
  By default (`expand_orbit: true`) each listed coordinate is expanded to
  its full symmetry orbit under the host crystal's space group — the same
  spglib operations, in the same Hall setting, that expand the host's own
  `crystal.wyckoff_sites` — and deduplicated, so *one representative per
  orbit suffices* and the sublattice always respects the host's site
  symmetry (an incomplete orbit would be an unphysical, symmetry-breaking
  interstitial density). Listing an already-complete orbit is harmless:
  the expansion is idempotent. `expand_orbit: false` uses the coordinates
  verbatim as a deliberate symmetry-breaking escape hatch — you must then
  list every equivalent site yourself. The expanded site count is logged
  at config time and placement time (no silent site-count changes). With
  CIF input (`crystal.cif`) the expansion uses the spglib-*detected*
  space group, so interstitial sublattices follow the file's true
  symmetry even when the CIF header declares `P 1`.

Candidates are then rejected by a PBC-correct minimum-image
distance check against every previously placed atom (host and earlier
dopants), Bernoulli-selected at the shell/bulk probabilities above, and
finally pruned for self-conflicts among the accepted sites.

**Dopant–GB distance profile (proxigram).** For every dopant the stage
records the distance to the nearest grain boundary of each placed dopant
atom *and* of each candidate site (interstitial: min-distance survivors;
substitutional: eligible host atoms). `doping_profile.csv` bins both
populations by GB distance and reports the local dopant fraction

$$
c(d) = \frac{n_{\mathrm{dopant}}(d)}{n_{\mathrm{candidate}}(d)}
$$

on the candidate sublattice at distance d — the atom-probe-tomography
"proxigram" analogue. Normalising by the *candidate* histogram rather than
by bin volume removes the geometric artefact that shell bins contain more
volume than deep-bulk bins; for GB segregation with enrichment E the
profile should step from ≈ E·p_bulk inside the shell (d ≤ `shell_width`)
to ≈ p_bulk outside, and be flat at the nominal fraction when segregation
is off. The companion gnuplot script `doping_profile.plt` plots the dopant
count histogram and c(d) against the nominal fraction and shell edge, so
correct GB segregation is verifiable at a glance. Note the same
small-count caveat as G17: bins with few candidate sites give noisy
fractions (empty bins write `nan`).

**Achieved-enrichment caveat.** The achieved enrichment
`E_achieved = (d_s/c_s)/(d_b/c_b)` reported in `summary.csv` and checked by
gate **G17** is a ratio estimator over the bulk dopant count `d_b`, so it is
only statistically reliable when the *expected* bulk dopant count is at least
a few tens — a small expected count gives large seed-to-seed spread and an
upward Jensen-inequality bias from the `1/d_b` term (mirrors the rule of
thumb in `docs/gates.md` §G17). Gate **G18** independently re-verifies every
interstitial dopant's `min_distance` on the final structure (G7-style fresh
PBC neighbour query) and hard-fails on any violation.

## References

1. D. P. Bourne, P. J. J. Kok, S. M. Roper, W. D. T. Spanjer, *Laguerre
   tessellations … grains of given volumes*, Phil. Mag. 100 (2020) 2677.
2. D. P. Bourne, M. Pearce, S. M. Roper, *… periodic semi-discrete optimal
   transport*, Mech. Res. Commun. 127 (2023) 104023.
3. J. Kitagawa, Q. Mérigot, B. Thibert, *Convergence of a Newton algorithm for
   semi-discrete optimal transport*, JEMS 21 (2019) 2603.
4. M. Kuhn et al., *Centroidal Laguerre … prescribed volume fractions*, CMAME 369
   (2020) 113175.
5. J. K. Mackenzie, *Second paper on … misorientation*, Biometrika 45 (1958) 229.
6. D. G. Brandon, *The structure of high-angle grain boundaries*, Acta Metall. 14
   (1966) 1479.
7. T. D. B. Jacobs, T. Junge, L. Pastewka, *Quantitative characterization of
   surface topography*, Surf. Topogr.: Metrol. Prop. 5 (2017) 013001.
8. C. Braun, J. M. Dake, C. E. Krill III, R. Birringer, *Abnormal grain growth
   mediated by fractal boundary migration at the nanoscale*, Sci. Rep. 8 (2018)
   1592.
9. C. Braun, R. A. Zeller, H. Menzel, J. Schmauch, C. E. Krill III,
   R. Birringer, *Orientation mapping linked to fractal analysis: a method
   for studying abnormal grain growth in nanocrystalline PdAu*, J. Appl.
   Phys. 128 (2020) 105102.
10. S. J. Eder, D. Bianchi, U. Cihak-Bayr, K. Gkagkas, *Methods for atomistic
    abrasion simulations … fractal surfaces*, Comput. Phys. Commun. 212 (2017) 100.
11. H. W. Lilliefors, *On the KS test for normality with mean and variance
    unknown*, J. Am. Stat. Assoc. 62 (1967) 399.
12. R. Quey, P. R. Dawson, F. Barbe, *Large-scale 3D random polycrystals …*,
    CMAME 200 (2011) 1729 (Neper — state-of-practice comparison).
13. R. N. Goldman, *Curvature formulas for implicit curves and surfaces*,
    Computer Aided Geometric Design 22, 632–658 (2005).
14. M. Miodownik, A. W. Godfrey, E. A. Holm, D. A. Hughes, *On boundary
    misorientation distribution functions and how to incorporate them into
    three-dimensional models of microstructural evolution*, Acta Mater. 47
    (1999) 2661–2668.
15. D. M. Saylor, J. Fridy, B. S. El-Dasher, K. Y. Jung, A. D. Rollett,
    *Statistically representative three-dimensional microstructures based on
    orthogonal observation sections*, Metall. Mater. Trans. A 35 (2004)
    1969–1979.
