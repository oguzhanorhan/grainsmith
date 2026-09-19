"""Physical and numerical constants for grainsmith.

Every tolerance and magic number lives here with a one-line justification.
No other module may introduce a bare numeric literal for a physical/numerical
threshold — import from this module instead.
"""
import math
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Geometry / deduplication tolerances
# ---------------------------------------------------------------------------

ZERO_VECTOR_TOL: float = 1e-12
"""Minimum Euclidean norm for a user-supplied direction/axis vector
(rotation axis, fiber crystal axis, hkl/uvw) before it is normalized.

A zero (or numerically-zero) vector cannot be normalized — dividing by its
norm yields NaN that would silently poison every downstream orientation.
Inputs with ``‖v‖ < ZERO_VECTOR_TOL`` are rejected as a clean ConfigError
at the conversion chokepoint."""

IDENTITY_ANGLE_TOL: float = 1e-10
"""Rotation angle (degrees) below which a quaternion is treated as the
identity; the axis is then undefined and ``quat_to_axis_angle`` returns the
+z axis with angle 0 (orientation/quaternion.py)."""

PARALLEL_DOT_TOL: float = 1e-10
"""|v1·v2 − (±1)| below which two unit vectors are taken as exactly
(anti)parallel when constructing the rotation between them (samplers)."""

CELL_PARAM_TOL: float = 1e-12
"""Equality tolerance for lattice parameters/angles when classifying the
crystal family for METHODS.md (a == b, angle == 90°). Float64 roundoff on
validated cell parameters is far below this (io/methods.py)."""

VERIFY_SYMPREC: float = 1e-4
"""spglib symmetry-search tolerance for the gate-G2 space-group round-trip
(crystal/verify.py): loose enough for ideal-lattice coordinate roundoff,
tight enough to separate neighbouring settings."""

FRAC_DEDUPE_TOL: float = 1e-5
"""Fractional-coordinate deduplication tolerance (§6.2).

Used when collapsing symmetry-generated Wyckoff orbit positions that are
numerically identical (minimum-image in fractional space), within a single
crystallographic conventional cell (edge O(1-100) Å).  For a 10 Å cell edge,
1e-5 fractional ≈ 1e-4 Å — far above the ~1e-15 Å float64 rounding noise, and
far below any real inter-orbit separation.
"""

EULER_TOL: float = 1e-9
"""Relative box-size tolerance for vertex deduplication (§9 gate G4).

Voronoi vertices that are closer than EULER_TOL × max(L_x, L_y, L_z) are
treated as identical when computing the per-cell Euler characteristic
V − E + F = 2.  This removes degenerate vertices from Qhull output.
"""

VOL_REL_TOL: float = 1e-6
"""Volume-sum relative tolerance for gate G3 (flat geometry).

|Σ V_i − V_box| / V_box must be below this value.  Tighter than 1e-5
because Qhull volumes are exact-adaptive arithmetic; residuals are
typically < 1e-14.
"""

OWNS_TIE_TOL: float = 1e-9
"""Relative (× max box length) distance tolerance for compact-cell
ownership ties (§6.8 deterministic tie-break).

Two replica seeds whose distances to a query point differ by less than
OWNS_TIE_TOL × max(L) are treated as exactly tied (the lattice point sits
on a cell boundary, e.g. a commensurate lattice plane on a Voronoi face);
the tie is resolved to the lowest replica index so every torus point is
owned by exactly one image.  Float64 noise is ~1e-12 × L, true competitor
separations are ≥ O(seed spacing) — 1e-9 cleanly separates the two.
"""

# ---------------------------------------------------------------------------
# Overlap removal
# ---------------------------------------------------------------------------

OVERLAP_DEFAULT_FACTOR: float = 0.85
"""Default overlap cutoff as a fraction of d_nn (§6.9).

Atoms from different grains closer than 0.85 × d_nn are candidates for
removal.  Matches Atomsk's common 1.5 Å choice when d_nn ≈ 1.76 Å (FCC Cu).
"""

# ---------------------------------------------------------------------------
# Curved-boundary / warp guards
# ---------------------------------------------------------------------------

G5_SPURIOUS_VOXEL_FRACTION: float = 0.005
"""Connectivity gate G5: discretization-sliver filter (§6.7).

Even an exactly convex flat-Voronoi cell can voxel-split under
6-connectivity where a wedge-shaped corner is locally thinner than the
voxel spacing, leaving O(1)-voxel satellites.  Components smaller than this
fraction of the grain's voxel count (absolute floor: 2 voxels) are treated
as slivers; G5 fails only on macroscopic fragments, which are orders of
magnitude larger than slivers for any admissible warp amplitude.
"""

HURST_G13_TOL: float = 0.15
"""Gate G13 (warn) threshold: |Ĥ − H| of the back-estimated Hurst
exponent vs the configured one.

Ĥ comes from a log-log fit of the radially averaged PSD over the finite
band [2π/l_max, 2π/l_min] on a finite grid: shell discreteness near the
band edges and the per-mode χ² scatter of a single realization bias the
slope by up to ~0.1 for typical one-decade bands — 0.15 separates that
documented bias from a genuinely wrong spectrum.  Warn-only.
"""

WARP_GRAD_MAX: float = 0.5
"""Maximum allowed ‖∇u‖ for the warp field to be bijective (§6.6 guard 2).

A Gaussian-smoothed displacement field with max gradient < 0.5 guarantees
injectivity of x ↦ x + u(x) (by the Banach fixed-point argument on R³).
"""

# ---------------------------------------------------------------------------
# GB character classification
# ---------------------------------------------------------------------------

CHAR_TWIST_DEG: float = 15.0
"""GB twist threshold in degrees (§6.10).

A boundary is classified 'twist' when the angle ψ between the
misorientation axis and the boundary normal satisfies ψ < CHAR_TWIST_DEG.
Brandon (1966) uses 15° as the standard CSL angular tolerance prefactor.
"""

CHAR_TILT_DEG: float = 75.0
"""GB tilt threshold in degrees (§6.10).

A boundary with ψ > CHAR_TILT_DEG is classified 'tilt'; between
CHAR_TWIST_DEG and CHAR_TILT_DEG it is 'mixed'.
"""

CSL_BRANDON_FACTOR: float = 15.0
"""Brandon criterion prefactor (§6.10).

The angular tolerance for a grain boundary to be classified as Σn CSL
is Δθ ≤ CSL_BRANDON_FACTOR × Σ^(−1/2) degrees.
Reference: D. G. Brandon, Acta Metall. 14 (1966) 1479.
"""

# ---------------------------------------------------------------------------
# MDF reference / targeting
# ---------------------------------------------------------------------------

MDF_REFERENCE_SAMPLES: int = 200_000
"""Monte-Carlo sample count for the Haar-random disorientation-angle reference
(an estimate of the Mackenzie law for the 24-rotation group of cubic m-3m).

The misorientation of a Haar-random grain pair equals a single Haar-random
rotation, so one reduced rotation per sample suffices.  With 40 bins the
per-bin relative sampling noise is ≈ √(40/200000) ≈ 1.4 % — well below the
G12 warn threshold; the angle reduction is one-sided-orbit vectorized
(24 candidates for cubic), so the cost is well under a second.
"""

MDF_REFERENCE_SEED: int = 1958
"""Fixed internal seed of the Haar-random disorientation-angle reference.

The reference is a property of the POINT GROUP, not of the run — it must
be the same deterministic curve for every run seed, so it deliberately
does NOT come from the run's RNG bundle.
"""

MDF_THETA_MAX_ROUND_DEG: float = 0.5
"""Rounding quantum (degrees) for the empirical maximum disorientation
angle of the point group.

θ_max is estimated as the maximum over the MC reference samples and
rounded UP to this quantum (cubic: about 62.8° → 63.0°). This is not a
proof of fundamental-zone coverage for every group. Run pairs that exceed the last
edge (residual tail mass ≲ 1e-5) are clipped into the last bin.
"""

# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

POWER_PLANE_TOL: float = 1e-8
"""Relative (× max box length) tolerance for matching halfspace-intersection
vertices to their defining planes (power/Laguerre backend).

Qhull vertex residuals are ~1e-12 × scale; true distances between distinct
parallel planes are bounded below by the SDOT volume tolerance over the
face area — 1e-8 separates the two by orders of magnitude either way.
"""

RNG_MAX_ATTEMPTS_FACTOR: int = 5000
"""RSA maximum number of placement attempts per grain (§6.4).

Total attempt budget = RNG_MAX_ATTEMPTS_FACTOR × N_grains.  Exceeding this
raises TessellationError advising a smaller min_seed_distance or fewer grains.
"""

SINGLE_WINDOW_TOL: float = 1e-12
"""Single-crystal owns() window shift, RELATIVE to the box length:
the fundamental window is [−tol·L, L − tol·L) per axis.

A commensurate lattice puts whole atom planes EXACTLY on the window edge
0 ≡ L; with float64 the two lifts of such a plane evaluate to ~±L·2e−16
and ~L ∓ L·2e−16, so a raw [0, L) test can exclude BOTH (one negative,
one == L) — dropping the whole plane.  Shifting the window by tol·L keeps
its length exactly L (still one lift per torus point) while moving the
edge ~4 orders of magnitude away from the roundoff scale; 1e−12·L is in
turn ~10 orders below any interatomic distance, so no physical ambiguity
exists.  This is the window analogue of the flat backend's OWNS_TIE_TOL
deterministic tie-break.
"""

TRICLINIC_TILT_BOUND_TOL: float = 1e-9
"""Absolute Å slack applied to the LAMMPS restricted-triclinic tilt bound
(``box.cells``): a reduced tilt factor must satisfy ``|xy|, |xz| <=
box.lengths[0]/2`` and ``|yz| <= box.lengths[1]/2`` (Howto_triclinic).
``reduce_triclinic_tilts`` (crystal/cell.py) always lands the reduced tilt
inside ``[-edge/2, edge/2]`` up to float64 rounding of the ``round()`` in the
reduction (~1e-13 relative for well-conditioned cells); this constant is the
comparison slack so that exact-half-edge cases are not spuriously rejected
by roundoff.
"""

COMMENSURATE_TOL: float = 1e-8
"""Gate G14 (single crystal): maximum per-axis relative misfit
strain ε_i = ‖(R·A)(m − round m)‖ / L_i below which the rotated lattice is
considered COMMENSURATE with the periodic box.

A genuinely commensurate setup (box = integer lattice multiples, exact
orientation) leaves only float64 roundoff through one matrix inverse and
one matmul — ~1e-13 relative even for poorly conditioned cells.  Any
physical misfit is many orders of magnitude larger (one lattice parameter
mismatch across a 100 Å box is ~1e-2).  1e-8 sits safely between the two.
"""

# ---------------------------------------------------------------------------
# Publication statistics
# ---------------------------------------------------------------------------

LAGB_MAX_DEG: float = 15.0
"""Low-angle grain-boundary (LAGB) cutoff in degrees
(statistics.csv ``lagb_area_fraction``).

15° is the conventional LAGB/HAGB transition: the Read–Shockley
dislocation-wall description of a boundary breaks down around 15° (core
overlap), and Brandon (1966) adopts the same 15° as the CSL angular
prefactor — consistent with CHAR_TWIST_DEG / CSL_BRANDON_FACTOR above.
"""

LOGNORMAL_SIGMA_MIN: float = 1e-9
"""Degeneracy threshold for the statistics.csv log-normal fit.

σ̂ (the sample std of ln d) below this value means the diameters are
identical up to float64 roundoff (ln of equal values scatters at ~1e-16;
`type: equal` volume targets land here) — the KS goodness-of-fit against
a near-delta distribution would be numerically meaningless, so it is
reported as NaN.  Any physical grain-size spread has σ ≥ O(1e-3), seven
orders of magnitude above the threshold.
"""

SECTION_GRID: int = 200
"""Pixels along the shorter in-plane edge of the EBSD-like section export
(``analysis.section``; the longer edge scales proportionally).

200² ≈ 4·10⁴ pixels matches the order of a routine EBSD map and resolves
grains down to ~1/40 of the box edge; a 2D slice costs O(n²) memory/time,
so this is orders of magnitude below every §13 budget.  Fixed (not derived
from the analysis voxel grid) so the slice resolution is independent of
the 3D analysis settings.

Also reused as the section resolution
``PerturbedDistanceTessellation.d_b_estimate`` (gate G20) samples its own
in-box cross-sections at — this is what sets the Å/pixel scale ``h`` that
G20's fit window ``[l_min, l_max]`` (the self_affine synthesis band) gets
converted to pixel units against. A coarser SECTION_GRID would shrink the
number of pixel-scale box sizes available inside a given [l_min, l_max]
band and, at the smallest MD-typical grain sizes, could collapse an
already-narrow (a few octaves at most) band to too few usable ``eps``
steps for ``box_count_dimension`` to fit at all (reported honestly as
"n/a", never fabricated — see G20's own docstring and docs/physics.md
§5b for why that band width, not this resolution, is the dominant
limitation on what G20
can resolve).
"""

class BibEntry(NamedTuple):
    """One METHODS_BIBLIOGRAPHY entry.

    ``author_year`` is the self-contained in-text label that
    ``io/methods.py``'s ``cite()`` emits, and the ONLY field that reaches
    the generated METHODS.md.  Format: one author ``Surname Year``; two
    ``Surname1 & Surname2 Year``; three or more ``Surname1 et al. Year``
    — e.g. ``"Bourne et al. 2020"``, rendered in text as ``(Bourne et al.
    2020)``.

    ``reference`` is the full reference string, kept as the
    maintainer-facing record of WHICH work each label denotes, reconciled
    field-by-field against the manuscript bibliography (local tooling, not
    published with this repository).  No code path renders
    it; do not mistake it for dead data (without it the labels are
    ambiguous — there are two Braun et al. and two Bourne et al. entries).
    """

    reference: str
    author_year: str


METHODS_BIBLIOGRAPHY: dict[str, BibEntry] = {
    "laguerre_volumes": BibEntry(
        "D. P. Bourne, P. J. J. Kok, S. M. Roper, W. D. T. Spanjer, "
        "Laguerre tessellations and polycrystalline microstructures: a fast "
        "algorithm for generating grains of given volumes, Philos. Mag. 100 "
        "(2020) 2677. arXiv:1912.07188",
        "Bourne et al. 2020"),
    "sdot_periodic": BibEntry(
        "D. P. Bourne, M. Pearce, S. M. Roper, Geometric modelling of "
        "polycrystalline materials: Laguerre tessellations and periodic "
        "semi-discrete optimal transport, Mech. Res. Commun. 127 (2023) "
        "104023. arXiv:2207.12036",
        "Bourne et al. 2023"),
    "centroidal_laguerre": BibEntry(
        "J. Kuhn, M. Schneider, P. Sonnweber-Ribic, T. Böhlke, Fast "
        "methods for computing centroidal Laguerre tessellations for "
        "prescribed volume fractions with applications to microstructure "
        "generation of polycrystalline materials, Comput. Methods Appl. "
        "Mech. Eng. 369 (2020) 113175.",
        "Kuhn et al. 2020"),
    "sdot_newton": BibEntry(
        "J. Kitagawa, Q. Mérigot, B. Thibert, Convergence of a Newton "
        "algorithm for semi-discrete optimal transport, J. Eur. Math. Soc. "
        "21 (2019) 2603.",
        "Kitagawa et al. 2019"),
    "fractal_gb": BibEntry(
        "C. Braun, J. M. Dake, C. E. Krill III, R. Birringer, Abnormal "
        "grain growth mediated by fractal boundary migration at the "
        "nanoscale, Sci. Rep. 8 (2018) 1592.",
        "Braun et al. 2018"),
    "braun2020fractal": BibEntry(
        "C. Braun, R. A. Zeller, H. Menzel, J. Schmauch, C. E. Krill III, "
        "R. Birringer, Orientation mapping linked to fractal analysis: a "
        "method for studying abnormal grain growth in nanocrystalline "
        "PdAu, J. Appl. Phys. 128 (2020) 105102.",
        "Braun et al. 2020"),
    "selfaffine_psd": BibEntry(
        "T. D. B. Jacobs, T. Junge, L. Pastewka, Quantitative "
        "characterization of surface topography using spectral analysis, "
        "Surf. Topogr.: Metrol. Prop. 5 (2017) 013001.",
        "Jacobs et al. 2017"),
    "fractal_substrates": BibEntry(
        "S. J. Eder, D. Bianchi, U. Cihak-Bayr, K. Gkagkas, Methods for "
        "atomistic abrasion simulations of laterally periodic polycrystalline "
        "substrates with fractal surfaces, Comput. Phys. Commun. 212 (2017) "
        "100–112.",
        "Eder et al. 2017"),
    "mackenzie": BibEntry(
        "J. K. Mackenzie, Second paper on statistics associated with the "
        "random disorientation of cubes, Biometrika 45 (1958) 229.",
        "Mackenzie 1958"),
    "lilliefors": BibEntry(
        "H. W. Lilliefors, On the Kolmogorov–Smirnov test for normality with "
        "mean and variance unknown, J. Am. Stat. Assoc. 62 (318) (1967) "
        "399–402.",
        "Lilliefors 1967"),
    "brandon": BibEntry(
        "D. G. Brandon, The structure of high-angle grain boundaries, "
        "Acta Metall. 14 (1966) 1479.",
        "Brandon 1966"),
    "neper": BibEntry(
        "R. Quey, P. R. Dawson, F. Barbe, Large-scale 3D random "
        "polycrystals for the finite element method: generation, meshing "
        "and remeshing, Comput. Methods Appl. Mech. Eng. 200 (2011) 1729 "
        "(Neper).",
        "Quey et al. 2011"),
    "spglib": BibEntry(
        "A. Togo, I. Tanaka, Spglib: a software library for crystal "
        "symmetry search, arXiv:1808.01590 (2018).",
        "Togo & Tanaka 2018"),
    "goldman2005": BibEntry(
        "R. Goldman, Curvature formulas for implicit curves and surfaces, "
        "Comput. Aided Geom. Des. 22 (2005) 632.",
        "Goldman 2005"),
    "miodownik_bmd": BibEntry(
        "M. Miodownik, A. W. Godfrey, E. A. Holm, D. A. Hughes, On "
        "boundary misorientation distribution functions and how to "
        "incorporate them into three-dimensional models of "
        "microstructural evolution, Acta Mater. 47 (1999) 2661-2668.",
        "Miodownik et al. 1999"),
    "saylor_3d": BibEntry(
        "D. M. Saylor, J. Fridy, B. S. El-Dasher, K. Y. Jung, A. D. "
        "Rollett, Statistically representative three-dimensional "
        "microstructures based on orthogonal observation sections, "
        "Metall. Mater. Trans. A 35 (2004) 1969-1979.",
        "Saylor et al. 2004"),
    "csl_table": BibEntry(
        "V. Randle, O. Engler, Introduction to Texture Analysis: "
        "Macrotexture, Microtexture and Orientation Mapping, Gordon & "
        "Breach, 2000.",
        "Randle & Engler 2000"),
}
"""METHODS.md bibliography — keep in sync with io/methods.py.

io/methods.py assembles the auto-generated methods paragraph; each in-text
citation is a self-contained author-year label — e.g. ``(Bourne et al.
2020)`` — read off ``BibEntry.author_year`` by ``cite()``, with no numbered
reference list.  Keys are stable identifiers used by the sentence builders
(including spglib, which every run cites through the gate G2 round-trip).

Each value is a :class:`BibEntry` pairing that in-text label with the full
reference string.  Only ``author_year`` is rendered into METHODS.md; the
``reference`` field is deliberately retained even though no code path emits
it — it is the maintainer-facing record of exactly WHICH work each label
denotes, reconciled field-by-field against the manuscript's own
bibliography (local tooling, not published with this repository).  Do not drop it as dead data: without it the labels
are ambiguous (two Braun et al. entries, two Bourne et al. entries), and it
is what keeps this table honest against the paper.
"""

# ---------------------------------------------------------------------------
# Memory / chunking mandates (§13)
# ---------------------------------------------------------------------------

MEMORY_HARD_LIMIT_BYTES: float = 16e9
"""§13 mandate DEFAULT: any stage whose estimated peak allocation exceeds
this many bytes raises a hard error advising a smaller problem, instead of
OOM-killing the process. Applies to the voxel grid (voxel.py), the
voxel_import EDT margin (voxel_import.py), and the direct
lattice-enumeration grid (fill.py).

This is only the DEFAULT: the effective, per-run limit is
``min(config.runtime.memory_limit_gb, detected physical RAM)`` -- a HARD
CLAMP computed once per run by ``memory.resolve_memory_budget`` (the
configured/default GB value can never exceed what the machine actually
has). The pipeline threads that resolved value to CONSTRUCTION time for
every curved tessellation backend (their guarded allocation runs inside
``__init__``, before the pipeline gets another chance at the instance --
see ``pipeline._stage_tessellation``) and stamps it onto
``Tessellation.memory_limit_bytes`` (tessellation/base.py) as a
belt-and-braces safety net for everything else (``pipeline.
_apply_runtime_limits``), in both cases before any of the three guard
sites above can run. This module constant remains the single source of
truth for the DEFAULT value — every direct construction that bypasses the
pipeline (tests, analysis/grains.py) keeps working unchanged via the class
attribute default, unclamped by RAM.

This is a *per-allocation* guard. The cumulative resident set of the driver
process can exceed any single allocation (measured peaking near 13.1 GB on
a large run, back when this default was lower); that is covered separately
by the best-effort total-RSS monitor in memory.py (soft WARN only)."""

MEMORY_SOFT_WARN_FRACTION: float = 0.9
"""Soft total-RSS WARN ceiling as a fraction of physical RAM, used by the
optional psutil monitor (memory.py) when no explicit ``--max-rss`` is given.
Crossing it logs a WARN (never raises, never changes any output byte) so a
batch job is forewarned before the OS OOM-killer steps in."""

MEMORY_MONITOR_POLL_SECONDS: float = 0.1
"""Background sampling interval (s) for the optional total-RSS monitor
(memory.py). Small enough to catch transient peaks (parallel write/fill),
large enough to be negligible overhead; reading RSS does not affect outputs."""

FILL_POOL_RAM_FRACTION: float = 0.7
"""Fraction of CURRENTLY AVAILABLE RAM the per-grain fill pool may commit
to concurrent lattice grids (§13, ``atoms.fill.fill_grains``).

The §13 per-allocation guard bounds ONE grain's grid; with ``--jobs N`` up
to N grains fill at once, so the aggregate is N× that and nothing used to
bound it. ``fill_grains`` now prices every grain's grid in the driver
(``estimate_fill_grid_bytes``) and clamps the worker count so

    n_workers × max(grid bytes)  ≤  FILL_POOL_RAM_FRACTION × available RAM

Clamping is output-SAFE: ``fill_grains`` returns blocks in grain order and
each grain draws its own rng stream, so the result is bit-identical for
every worker count (the guarantee in this module's docstring).

The fraction is < 1 because the workers are not the only claimant: the
driver accumulates every finished ``AtomBlock`` (``list(pool.map(...))``)
while later grains are still filling, and the tessellation's voxel/field
arrays stay resident throughout. 0.7 leaves that headroom without being so
conservative that a well-sized run loses its parallelism."""

FILL_CHUNK: int = 2_000_000
"""Maximum number of candidate atom positions per fill chunk (§13 mandate).

Inside fill_grain the per-chunk expansion (candidate lattice points × basis)
is processed in chunks of this size to bound the peak working set; at float64
(24 bytes per (x,y,z) triple) this is ~48 MB per chunk — comfortably below
even the smallest sane ``runtime.memory_limit_gb`` setting. The bounding-box
grid T itself is materialized once and is guarded against
``tess.memory_limit_bytes`` before allocation (see tessellation/base.py)."""

VOXEL_GRID_MAX: int = 384
"""Maximum voxel grid size per axis (§6.7).

Auto-resolution is clamped to this value with a user warning; above 384³
the voxel array exceeds ~57 M int32 entries (~228 MB) which approaches
the memory budget for typical HPC nodes.
"""

AMU_PER_A3_TO_G_PER_CM3: float = 1.66053906892
"""Unit conversion: 1 amu/Å³ = 1.66053906892 g/cm³.

Equals the 2022 CODATA atomic mass constant (1.66053906892e-24 g) divided
by 1 Å³ = 1e-24 cm³.  Used for the crystal density row in summary.csv.
"""

# ---------------------------------------------------------------------------
# Atomic masses (amu)  —  IUPAC 2021 standard atomic weights
# ---------------------------------------------------------------------------

ATOMIC_MASSES: dict[str, float] = {
    "H":  1.008,
    "He": 4.0026,
    "Li": 6.94,   # IUPAC 2021 conventional value (interval [6.938, 6.997])
    "Be": 9.0122,
    "B":  10.81,
    "C":  12.011,
    "N":  14.007,
    "O":  15.999,
    "F":  18.998,
    "Ne": 20.180,
    "Na": 22.990,
    "Mg": 24.305,
    "Al": 26.982,
    "Si": 28.085,
    "P":  30.974,
    "S":  32.06,
    "Cl": 35.45,
    "Ar": 39.948,
    "K":  39.098,
    "Ca": 40.078,
    "Ti": 47.867,
    "V":  50.942,
    "Cr": 51.996,
    "Mn": 54.938,
    "Fe": 55.845,
    "Co": 58.933,
    "Ni": 58.693,
    "Cu": 63.546,
    "Zn": 65.38,
    "Zr": 91.224,
    "Nb": 92.906,
    "Mo": 95.95,
    "Ru": 101.07,
    "Pd": 106.42,
    "Ag": 107.868,
    "Sn": 118.711,
    "Ta": 180.948,
    "W":  183.84,
    "Re": 186.207,
    "Os": 190.23,
    "Ir": 192.217,
    "Pt": 195.084,
    "Au": 196.967,
    "Pb": 207.2,
    "Bi": 208.980,
}
"""Element symbol → standard atomic weight in amu (IUPAC 2021).

Used by the LAMMPS writer for the Masses section and by density calculations
in summary.csv.  Extend as needed; do not put masses inline in io/lammps.py.
"""

# ---------------------------------------------------------------------------
# Cubic CSL table  —  Sigma → (misorientation_angle_deg, [h, k, l] axis)
# Keys use string labels to accommodate a/b variants (e.g. "13a", "13b").
# Reference angles/axes from standard CSL tables (Randle & Engler, 2000).
# ---------------------------------------------------------------------------

CSL_TABLE: dict[str, tuple[float, list[int]]] = {
    "3":   (60.00,  [1, 1, 1]),
    "5":   (36.87,  [1, 0, 0]),
    "7":   (38.21,  [1, 1, 1]),
    "9":   (38.94,  [1, 1, 0]),
    "11":  (50.48,  [1, 1, 0]),
    "13a": (22.62,  [1, 0, 0]),
    "13b": (27.80,  [1, 1, 1]),
    "15":  (48.19,  [2, 1, 0]),
    "17a": (28.07,  [1, 0, 0]),
    "17b": (61.93,  [2, 2, 1]),
    "19a": (26.53,  [1, 1, 0]),
    "19b": (46.83,  [1, 1, 1]),
    "21a": (21.79,  [1, 1, 1]),
    "21b": (44.42,  [2, 1, 1]),
    "23":  (40.45,  [3, 1, 1]),
    "25a": (16.26,  [1, 0, 0]),
    "25b": (51.68,  [3, 3, 1]),
    "27a": (31.59,  [1, 1, 0]),
    "27b": (35.43,  [2, 1, 0]),
    "29a": (43.60,  [1, 0, 0]),
}
"""Cubic CSL table: Sigma label → (disorientation_angle_deg, rotation_axis [h,k,l]).

Used by analysis/boundaries.py to assign CSL Σ values under the Brandon
criterion (Δθ ≤ CSL_BRANDON_FACTOR × Σ^−½).  String keys handle the a/b
variants that share the same Σ integer (e.g. Σ13a and Σ13b).
"""

# ---------------------------------------------------------------------------
# GB curvature analysis
# ---------------------------------------------------------------------------

CURV_GRAD_MIN: float = 0.1
"""|∇φ| below this marks a degenerate curvature sample (φ = margin_i −
margin_j has |∇φ| ≈ 2 on a clean boundary); dropped and counted for G16."""

CURV_G16_DROP_TOL: float = 0.05
"""G16 (warn-only): dropped-sample fraction above this triggers WARN."""

CURV_G21_GAUSS_BONNET_TOL: float = 4.0 * math.pi
"""G21 (warn-only): per-grain |∮ K dA| above this triggers WARN.

The Gauss-Bonnet theorem fixes ∮_S K dA = 2π·χ(S) = 4π·(1 − g) EXACTLY
for any closed orientable surface S of genus g (sphere: g=0 ⇒ 4π); a
topological grain in a periodic polycrystal is such a closed 2-manifold.
The level-set curvature estimator (analysis/curvature.py) samples K only
at FACE-INTERIOR points, deliberately excluding the edges/vertices where
the polyhedral surface is not twice-differentiable (§6.7); a Voronoi-
like grain's angle-defect Dirac mass at those excluded edges/vertices (a
cube's 8 corners alone already sum to 4π) is meant to be the dominant
share of the total budget, leaving a small face-interior-only remainder.

CALIBRATION (this gate's own validation sweep — READ BEFORE trusting the
"should be small" framing above): that expectation does NOT hold at the
noise/leakage floor of the current estimator. Six near-FLAT
perturbed_distance baselines (amplitude 0.01, effectively planar) landed
their worst-grain ratio at 0.977-1.041× this tolerance — i.e. clustered
AT, not below, one full topological unit, not in "low single digits"
and not near 0. Ordinary curved configurations push well past that:
worst-grain ratios of 5-14× have been measured directly from a 131-run
benchmark campaign already on disk. A dedicated 2-grain periodic
bicrystal with ONE smooth boundary and NO triple junctions (ruling out
triple-junction sampling as the sole mechanism) measured 8-13× across
voxel_grid = 32/64/128 — i.e. this is not a many-grain-topology
artefact. Refining analysis.voxel_grid does not drive the residual
toward 0: a 32→128 sweep on one fixed geometry gave ratios
[1.53, 4.93, 2.57, 2.53, 2.11] (non-monotonic, no visible trend toward
0, but also not literally "flat" — a factor-of-3 swing is present
within the sweep). An attempt to separate an area effect from a
resolution effect by holding TRUE physical voxel spacing fixed while
growing the box (dx=0.625 A/voxel, 3 seeds/box size) did NOT show a
clean, reproducible area trend either: mean ratios of 9.4/7.5/10.5 for
L=30/40/60 A, with PER-SEED spread (roughly 5x-17x) large enough to
swamp any systematic area dependence at this sample size — so "grows
with area" should be read as unconfirmed, not established, pending a
larger-N sweep. What IS reproducible: in a representative near-flat
case, under 1% of the retained face-interior samples account for about
half of the worst grain's total |Σ K dA| (heavy-tailed, not spread
uniformly across samples) — consistent with a leakage mechanism
concentrated near specific (likely edge-adjacent) voxels rather than
generic, uniformly-distributed round-off. The precise mechanism is not
yet isolated; the calibration above establishes only that it does not
vanish, does not converge with resolution in the range tested, and is
present even in the simplest possible curved geometry.

Net effect: this bound is a coarse, order-of-magnitude CEILING, not a
tight zero-residual check, and it sits close enough to the estimator's
own baseline floor (~1x4pi on NEAR-FLAT configurations) that a WARN on
an ordinary curved run is a ROUTINE, EXPECTED consequence of the current
estimator's behavior — not evidence that something newly broke. Treat a
WARN as a standing, coarse reliability caveat on that run's
gb_curvature.csv H/K columns, worth a second look only if it is
unusually large or changes sharply between otherwise-similar
configurations (see gate_g21_gauss_bonnet in qa.py for the message this
produces). This diagnoses the SAMPLING/ESTIMATOR, not the generated
geometry: G7 (interatomic distances) and G3/G4 (volume/topology)
already certify the atomistic structure independently of this estimate,
which is why G21 is warn-only and never blocks a run."""

# ---------------------------------------------------------------------------
# perturbed_distance (level-set self-affine GB) tessellation
# ---------------------------------------------------------------------------

ETA_CLIP: float = 3.0
"""Hard sup-norm clip applied to every unit-RMS per-grain scalar field η_i
after synthesis (tessellation/perturbed.py), in units of its own RMS.

A stationary unit-RMS Gaussian random field has an UNBOUNDED supremum in
principle; empirically, over the Hurst/band/grid combinations exercised
during design (H ∈ [0.1, 1.0], grid 60³–300³, one-decade to ~1.5-decade
bands), the realized sup-norm ranged 4.0–5.7 — consistent with the
standard Gaussian extreme-value estimate √(2 ln N_eff) for N_eff
effectively independent grid cells. Rather than lean on that empirical
(unbounded-in-principle) tail, η is explicitly clipped to ±ETA_CLIP
post-synthesis so that ``max|η_i(x)| ≤ ETA_CLIP`` is a GUARANTEE, not an
estimate — the seed-containment guard (below) and the candidate-search
radius in perturbed.py both depend on this being a hard bound. 3.0 (three
RMS) clips a negligible tail (< 0.3% of a Gaussian's mass per sample, and
the field is spectrally correlated so genuine excursions are rarer still)
while keeping the bound tight enough that PERTURBED_DISTANCE_SAFETY below
does not have to absorb a large empirical slop.
"""

REASSIGNED_FRACTION_WARN_TOL: float = 0.02
"""G19 (warn-only): perturbed_distance G5-repair reassigned-voxel-fraction
above this triggers WARN (tessellation/perturbed.py, qa.py
gate_g19_reassigned_fraction). Measured across this method's validation
runs (12–60 grains; amplitudes from a deliberately aggressive prototype
value up through the final seed-containment guard ceiling A_max):
0.001–0.16% reassigned — the single worst case observed (0.156%, a
pre-final-module prototype run) still leaves ~13x headroom below this
2% floor. A run crossing 2% indicates amplitude/hurst pushing the
level-set into pervasive topology-breaking territory, worth a user's
attention even though repair itself always succeeds or raises (never
silently returns a badly-fragmented structure)."""

PERTURBED_DISTANCE_SAFETY: float = 0.5
"""Safety factor C in the perturbed_distance seed-containment guard
A_max = C · min_seed_distance / (2 · ETA_CLIP) (tessellation/perturbed.py,
config/resolve.py).

Derivation: grain i's own seed s_i must not be captured by a competing
grain j — i.e. d(s_i, s_j) − A·η_j(s_i) must not undercut
0 − A·η_i(s_i) = −A·η_i(s_i) for the true (unperturbed) distance
d(s_i,s_i)=0. Since |η_i|, |η_j| ≤ ETA_CLIP everywhere, the perturbation
term on either side of that comparison is bounded by A·ETA_CLIP, so a
sufficient condition for every grain's own seed to remain self-assigned
is 2·A·ETA_CLIP < min_seed_distance (the smallest possible d(s_i,s_j)).
C=0.5 halves that exact threshold — the same one-binary-doubling margin
role WARP_GRAD_MAX's "< 0.5" plays for the warp bijectivity guard — so
A_max = 0.5·min_seed_distance/(2·3.0) = min_seed_distance/12.

This bound is DELIBERATELY only a sufficient (not tight) pre-flight
check: unlike warp's bijectivity guard, which is the ONLY thing standing
between a configuration and a self-intersecting map, perturbed_distance
has two cheap POST-HOC checks that catch any residual pathology exactly
— (a) an exact seed-ownership check (ConfigError if any seed is not
self-assigned) and (b) the G5 repair-and-report connectivity pass (its
reassigned_fraction is the real-world severity signal). A permissive
a-priori bound is therefore appropriate: its job is to catch obviously
misconfigured runs before paying for field synthesis, not to be the sole
correctness barrier.

``boundaries.curved.amplitude_convention: reference_wavelength``
reinterprets the config's ``amplitude`` field as a
reference-octave RMS rather than a total-band RMS (see
``tessellation.warp.reference_shell_kappa``), but A_max above is
unchanged and still binds the REALIZED total-RMS-equivalent amplitude —
config/resolve.py Rule 29(d) and
``PerturbedDistanceTessellation.__init__`` both convert the given
amplitude back to its total-RMS equivalent (÷ κ, κ ≤ 1) BEFORE comparing
against A_max, so switching convention can only make this check
stricter, never weaker.
"""

# ---------------------------------------------------------------------------
# Volume-weighted ODF diagnostics (orientation/odf.py)
# ---------------------------------------------------------------------------

ODF_KERNEL_HALFWIDTH_DEG: float = 10.0
"""Default de la Vallee Poussin kernel half-width (degrees) for the
volume-weighted ODF diagnostics (orientation/odf.py).

10 degrees matches the smoothing scale conventionally used in the
EBSD/MTEX literature to recover a smooth orientation distribution
function from a few hundred to a few thousand discrete grain
orientations: narrow enough to resolve texture components a few degrees
apart, wide enough that a modest orientation set does not read as a
spiky sum of near-delta functions. This is only a DEFAULT — every
function in orientation/odf.py takes the half-width (or its derived
kappa) as an explicit argument, so a caller may always request a
different smoothing scale."""

ODF_NULL_SAMPLES: int = 256
"""Default sample count for the ODF kernel-discrepancy null distribution
(``orientation.odf.null_mmd``).

The statistic of interest is an upper-tail quantile (the 95th percentile
used to flag an assignment as atypically volume-biased), whose sampling
standard error scales as sqrt(q(1-q)/n); at q=0.95, n=256 this is
sqrt(0.95*0.05/256) ≈ 0.014 — precise enough to place an observed MMD
against the tail reliably — while remaining cheap enough (one
Gram-matrix quadratic form per sample) to run inside a QA gate on every
pipeline invocation rather than being reserved for offline validation."""

ODF_NULL_SEED: int = 90501
"""Fixed internal seed of the ODF kernel-discrepancy null distribution
(``orientation.odf.null_mmd``), analogous to MDF_REFERENCE_SEED above:
the null answers "what does the kernel-smoothed volume-vs-count gap look
like under a NEUTRAL, non-adversarial assignment of these grain volumes
to this orientation set", which is a property of the CODE, not of the
run — it must be the same deterministic reference for every run seed, so
it deliberately does not come from the run's own RNG bundle. The digit
string itself carries no meaning beyond being fixed (an ODF-themed
mnemonic, not a date or physical constant)."""

ODF_KERNEL_MAX_ELEMS: int = 4_000_000
"""Row-chunk sizing target for the symmetrized Gram matrix build
(``orientation.odf.symmetrized_gram``, §13 mandate).

The Gram build's transient candidate array is (row-chunk) x (N * Nsym)
float64 — rows are processed in chunks small enough that this transient
stays at or below ODF_KERNEL_MAX_ELEMS elements (~32 MB at 8 bytes each),
a negligible add-on next to the O(N^2) output matrix itself for the
orientation-set sizes (hundreds to low thousands of grains) this
diagnostic targets, regardless of how large N or the point group's
Nsym grows. The §13 hard ceiling (MEMORY_HARD_LIMIT_BYTES, or the
caller-supplied limit) is the actual guard that raises; this constant
only controls how the allocation is CHUNKED within whatever budget that
check allows.

NOT bitwise-invariant to this value, and therefore part of the numerical
contract, not a free performance knob: changing the row-chunk count
changes how many rows BLAS multiplies together per call
(``q[lo:hi] @ qs.T``), and different row counts can make the BLAS
library pick different internal blocking/vectorization, which changes
summation order at the ULP level. Measured directly: forcing a much
smaller chunk than the default changes roughly 40% of a Gram matrix's
entries, by up to ~4.6e-14 relative -- utterly harmless in magnitude
(it cannot move a gate decision) but real, and in the same category as
the BLAS build identity ``provenance._blas_record`` already records as
part of what a byte-reproducible run depends on. Do not change this
constant casually across a comparison you expect to be exactly
reproducible; the module's own tests only ever assert agreement to
``rtol=1e-10`` across chunk sizes, never bitwise equality."""

ODF_QUOTIENT_TOL_DEG: float = 1e-3
"""Angular tolerance (degrees) for merging symmetry-equivalent orientations
into quotient-space (SO(3)/Sym) classes in
``orientation.odf.symmetry_classes`` -- the MEASUREMENT-side merge reported
by gate G22, never the annealer's control loop (which stays on
``orientation_classes``' exact-equality merge; see orientation/odf.py's
"SYMMETRY-QUOTIENT CLASSES" section).

The window is chosen to sit orders of magnitude between the two scales it
must separate: a genuine q' = q (x) S pair survives floating-point
round-trip at ~1e-13 degrees of separation (ULP-scale quaternion
arithmetic), while two generically distinct orientations are degrees
apart.  1e-3 deg is therefore tight enough that no physically meaningful
near-degeneracy is merged by accident, and loose enough that every exact
symmetry duplicate is merged despite float noise.  Note the tolerance
cannot be taken much SMALLER than this: cos(radians(tol)/2) must stay
distinguishable from 1.0 in float64, which fails below ~1e-5 deg."""

# ---------------------------------------------------------------------------
# Gate G24: final (post-warp) per-grain volume fidelity
# ---------------------------------------------------------------------------

WARP_VOLUME_G24_TOL: float = 0.10
"""G24 (warn-only, qa.gate_g24_final_volume_targets): tolerance on the max
relative error between the FINAL (post-warp) per-grain volumes and the
``grains.size_distribution`` targets, for the ``sdot_res.tess is not tess``
case (warp on a volume-fitted power base) -- i.e. the realized cells G11
itself never re-measures, because G11 is evaluated on the UNWARPED base
(``pipeline.py``, alongside the G11 call site).

This is a deliberately loose SANITY bound, not a physics claim, and it must
NOT be confused with (or set equal to) ``size_distribution.vol_tol``
(default ``1e-3``): that tolerance governs the base-cell fit (G11) at a
precision the warp displacement itself blows through by (at minimum)
two-plus orders of magnitude. Calibration measured directly, correlation
length 10 A throughout:

* equal-volume targets, a 60 A box, warp amplitude 0.6: max relative
  POST-WARP volume error 0.0209 at 4 grains, 0.0412 at 8 grains -- while
  the UNWARPED base (what G11 actually measures) sat at 1.5e-4 and 6.4e-5
  respectively, i.e. G11 PASSES comfortably on the base while the
  realized, warped cells are already two-plus orders of magnitude further
  from the targets than G11's own pass certifies;
* a WIDER sweep, a 70 A box, seed 11, also amplitude 0.6 unless noted,
  varying both grain count and how broad the target size distribution
  itself is (max relative error, mean in parentheses, verdict against the
  0.10 tolerance):

    N=4,  equal,               amplitude 0.6 -> 0.0220 (0.0110)  ok
    N=8,  equal,                amplitude 0.6 -> 0.0389 (0.0208)  ok
    N=8,  lognormal sigma=0.35, amplitude 0.6 -> 0.0465 (0.0217)  ok
    N=8,  lognormal sigma=0.5,  amplitude 0.6 -> 0.1088 (0.0309)  WARN
    N=24, lognormal sigma=0.5,  amplitude 0.4 -> 0.0838 (0.0339)  ok

  Observed range across every configuration measured: 0.022-0.109. The
  tolerance trips only at the extreme end (8 grains, the BROADEST size
  spread measured, still at a substantial amplitude) -- it is not a hair
  trigger on ordinary warp+size_distribution combinations, but a BROAD
  grain-size distribution combined with a LARGE warp amplitude CAN push
  the realized error past it. When that happens the WARN is the gate
  correctly reporting a real, physical effect -- the warp is applied
  AFTER the volume fit and nothing re-fits the targets to the warped
  geometry -- not a false alarm to be tuned away by loosening the
  tolerance;
* the competing explanation -- that any of this is merely voxel-grid
  discretisation noise, not real warp displacement -- was checked and
  ruled out: voxel-vs-exact-polyhedral volumes on the SAME (unwarped) flat
  tessellations differ by only 0.0024-0.0040 across the configurations
  above, far below every measured value in the table.

``vol_tol`` (1e-3) would therefore be the WRONG threshold to reuse here: it
sits below the discretisation floor itself and would flag every warped run
as a false failure regardless of how mild the warp actually is. 0.10 KEEPS
the correct behaviour observed above -- passing the bulk of ordinary
configurations while still flagging the genuinely extreme combination of a
broad size distribution and a large amplitude -- rather than being loosened
to paper over that one real WARN."""
