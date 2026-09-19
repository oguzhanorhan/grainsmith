"""Dopant insertion stage.

Adds substitutional (optionally GB-segregated) and interstitial dopants
to the finished host structure, deterministically: every stochastic
choice draws from the per-grain ``doping`` RNG stream family in
dopant-then-grain-then-site order, so outputs are bit-identical for any
``--jobs`` value.

Interstitial candidate sites are generated with the ordinary per-grain
fill machinery (``fill_grain``): the interstitial sublattice is just an
extra fractional basis on the same grain lattice + orientation, with
``store_margin=True`` providing the distance-to-GB used for the
shell/bulk split.  Substitutional doping never moves atoms — it
replaces species in place, so the grain-sort invariant survives.

Shell/bulk mass balance: with target dopant count N_t, shell/bulk
candidate counts N_s/N_b and enrichment E = c_shell/c_bulk,
    p_bulk = N_t / (N_b + E·N_s),   p_shell = E·p_bulk;
p > 1 is infeasible and raises ConfigError with the feasible maximum.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from grainsmith.atoms.fill import AtomBlock, fill_grain
from grainsmith.atoms.overlap import _pbc_pairs, _wrap_positions
from grainsmith.errors import ConfigError

if TYPE_CHECKING:
    from grainsmith.config.schema import DopantConfig

log = logging.getLogger(__name__)

# --- interstitial preset tables ------------------------------------------
# name -> (required ITA space-group number, (M, 3) fractional coords).
# hcp assumes the host motif on Wyckoff 2c ((1/3,2/3,1/4),(2/3,1/3,3/4));
# the tetrahedral z values are the ideal-packing positions (z = 5/8
# family) — a documented approximation for non-ideal c/a.
INTERSTITIAL_PRESETS: dict[str, tuple[int, np.ndarray]] = {
    "fcc_octahedral": (225, np.array([
        [0.5, 0.5, 0.5], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5],
    ])),
    "fcc_tetrahedral": (225, np.array([
        [0.25, 0.25, 0.25], [0.25, 0.25, 0.75], [0.25, 0.75, 0.25],
        [0.75, 0.25, 0.25], [0.25, 0.75, 0.75], [0.75, 0.25, 0.75],
        [0.75, 0.75, 0.25], [0.75, 0.75, 0.75],
    ])),
    "bcc_octahedral": (229, np.array([
        [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5],
    ])),
    "bcc_tetrahedral": (229, np.array([
        [0.5, 0.25, 0.0], [0.5, 0.75, 0.0], [0.25, 0.5, 0.0],
        [0.75, 0.5, 0.0], [0.0, 0.5, 0.25], [0.0, 0.5, 0.75],
        [0.0, 0.25, 0.5], [0.0, 0.75, 0.5], [0.25, 0.0, 0.5],
        [0.75, 0.0, 0.5], [0.5, 0.0, 0.25], [0.5, 0.0, 0.75],
    ])),
    "hcp_octahedral": (194, np.array([
        [0.0, 0.0, 0.0], [0.0, 0.0, 0.5],
    ])),
    "hcp_tetrahedral": (194, np.array([
        [1.0 / 3.0, 2.0 / 3.0, 0.625], [1.0 / 3.0, 2.0 / 3.0, 0.875],
        [2.0 / 3.0, 1.0 / 3.0, 0.125], [2.0 / 3.0, 1.0 / 3.0, 0.375],
    ])),
}


def resolve_sites(sites, sg_number: int, what: str,
                  sg_setting: str | None = None) -> np.ndarray:
    """Fractional interstitial coords from a preset name or coords model.

    Explicit coordinates (SitesCoordsConfig) are orbit-expanded by
    default: each listed site is mapped through every symmetry
    operation of the host space group (same Hall setting as the host
    crystal — *sg_setting* must match the value used to build the
    host basis) and deduplicated at FRAC_DEDUPE_TOL, exactly like the
    host's own ``crystal.wyckoff_sites``.  ``expand_orbit: false``
    uses the coordinates verbatim (deliberate symmetry breaking).
    Presets are stored as complete orbits already and are never
    re-expanded.

    Config-time Rule 26 already validated compatibility; this re-checks
    defensively so the function is safe to call standalone.
    """
    if isinstance(sites, str):
        need, frac = INTERSTITIAL_PRESETS[sites]
        if sg_number != need:
            raise ConfigError(
                f"{what}: preset '{sites}' requires space group {need}, "
                f"got {sg_number}."
            )
        return np.array(frac, dtype=np.float64)
    coords = np.array(sites.coords, dtype=np.float64).reshape(-1, 3)
    if not getattr(sites, "expand_orbit", True):
        return coords
    from grainsmith.constants import FRAC_DEDUPE_TOL
    from grainsmith.crystal.spacegroup import (
        _dedupe_frac,
        _frac_minimage,
        hall_from_international,
        symmetry_ops,
    )
    hall = hall_from_international(sg_number, sg_setting)
    rots, trans = symmetry_ops(hall)
    # Unlike the host path (expand_wyckoff), coincidences ACROSS listed
    # sites are deduplicated rather than rejected: every interstitial
    # site hosts the same dopant species, so a user who lists a full
    # orbit (the only valid form before expand_orbit existed) or two
    # representatives of the same orbit gets the identical sublattice,
    # not an error — the expansion is idempotent by construction.
    images = [_frac_minimage(R @ c + t)
              for c in coords for R, t in zip(rots, trans, strict=True)]
    frac = _dedupe_frac(np.asarray(images, dtype=np.float64),
                        tol=FRAC_DEDUPE_TOL)
    log.info(
        "%s: interstitial orbit expansion (SG %d%s): %d listed site(s) "
        "-> %d site(s)/cell", what, sg_number,
        f", setting={sg_setting!r}" if sg_setting else "",
        len(coords), len(frac))
    return frac


# --- substitutional placement --------------------------------------------


@dataclass
class DopantGrainStats:
    """Per-(dopant, grain) placement bookkeeping (doping.csv source)."""
    n_candidate_sites: int = 0
    n_rejected_min_distance: int = 0
    n_dopant: int = 0
    n_dopant_shell: int = 0
    n_dopant_bulk: int = 0
    n_candidate_shell: int = 0
    n_candidate_bulk: int = 0


@dataclass
class DopingProfileData:
    """Per-dopant GB-distance raw data (doping_profile.csv source).

    dopant_margins : distance-to-nearest-GB (Å) of every PLACED dopant
        atom of this dopant config.
    candidate_margins : same for every candidate site that survived the
        min-distance filter (interstitial) / every eligible host atom
        (substitutional) — the denominator population, so
        c(d) = hist(dopant) / hist(candidate) is the local dopant
        fraction on the candidate sublattice at distance d (the
        quantity whose shell/bulk ratio G17 checks as enrichment).
    """
    element: str
    mode: str
    dopant_margins: np.ndarray
    candidate_margins: np.ndarray
    nominal_concentration: float
    shell_width: float          # 0.0 when segregation disabled
    enrichment: float           # 1.0 when segregation disabled


@dataclass
class DopingReportRow:
    """One doping.csv row: a (grain, dopant) pair.

    fraction_shell/bulk = achieved dopant fraction among that grain's
    shell/bulk candidate sites (NaN when the region has no candidates).
    """
    grain_id: int
    element: str
    mode: str
    n_candidate_sites: int
    n_rejected_min_distance: int
    n_dopant: int
    n_dopant_shell: int
    n_dopant_bulk: int
    fraction_shell: float
    fraction_bulk: float


def _solve_probabilities(n_target: int, n_shell: int, n_bulk: int,
                         enrichment: float, what: str
                         ) -> tuple[float, float]:
    """(p_bulk, p_shell) from n_target = p_b·n_bulk + E·p_b·n_shell.

    A probability is only meaningful (and only checked) when its
    population is non-empty; the infeasibility hint names whichever
    probability actually violates the bound and always reports the
    absolute placement ceiling (n_shell + n_bulk at p = 1), which is
    the maximum feasible dopant count / concentration.
    """
    denom = n_bulk + enrichment * n_shell
    if n_target > 0 and denom <= 0:
        raise ConfigError(
            f"{what}: no candidate sites available for the requested "
            "dopant. Lower min_distance or the concentration."
        )
    p_b = n_target / denom if denom > 0 else 0.0
    p_s = enrichment * p_b
    bad_b = n_bulk > 0 and p_b > 1.0
    bad_s = n_shell > 0 and p_s > 1.0
    if bad_b or bad_s:
        if bad_s and not bad_b and enrichment >= 1.0 and n_target > n_shell:
            e_max = n_bulk / (n_target - n_shell)
            hint = (f"lower the enrichment (maximum feasible here "
                    f"≈ {e_max:.2f}) or the concentration")
        elif bad_b and not bad_s and enrichment < 1.0:
            hint = "raise the enrichment toward 1 or lower the concentration"
        else:
            hint = "lower the concentration"
        which = f"p_shell={p_s:.3f}" if bad_s else f"p_bulk={p_b:.3f}"
        raise ConfigError(
            f"{what}: infeasible target — "
            f"{which} "
            f"exceeds 1 ({n_target} dopants requested over {n_shell} "
            f"shell / {n_bulk} bulk candidates at enrichment "
            f"{enrichment:g}; at most {n_shell + n_bulk} are placeable "
            f"at p=1). {hint}."
        )
    return p_b, p_s


def apply_substitutional(atoms, margins, protected, cfg, streams,
                         n_grains: int,
                         n_target: int
                         ) -> tuple[list[DopantGrainStats],
                                    DopingProfileData]:
    """Replace host species in place.

    Uses explicit per-grain masks (not the searchsorted fast path), so
    the block does NOT need to be grain-sorted — a later substitutional
    dopant applied after an interstitial one (which appends atoms at the
    end) still classifies correctly.

    margins : (N,) distance-to-GB in Å (Tessellation.margin; ≥ 0 inside).
    protected : (N,) bool — atoms placed by earlier dopants; never
        replaced, updated in place with this dopant's placements.
    n_target : dopant count solved by run_doping against the estimated
        FINAL total (all dopants' concentrations are fractions of the
        final structure, spec).
    """
    seg = cfg.gb_segregation
    width = seg.shell_width if seg.enabled else 0.0
    enrich = seg.enrichment if seg.enabled else 1.0
    cand = ~protected
    if cfg.host is not None:
        cand &= atoms.species == cfg.host
    shell = cand & (margins <= width)
    p_b, p_s = _solve_probabilities(
        n_target, int(shell.sum()), int((cand & ~shell).sum()), enrich,
        f"doping ({cfg.element}, substitutional)")
    stats = [DopantGrainStats() for _ in range(n_grains)]
    hits: list[np.ndarray] = []
    for g in range(n_grains):
        idx = np.flatnonzero(cand & (atoms.grain == g))
        st = stats[g]
        st.n_candidate_sites = len(idx)
        st.n_candidate_shell = int(shell[idx].sum())
        st.n_candidate_bulk = st.n_candidate_sites - st.n_candidate_shell
        if len(idx) == 0:
            continue
        r = streams[g].random(len(idx))
        hit = idx[r < np.where(shell[idx], p_s, p_b)]
        atoms.species[hit] = cfg.element
        protected[hit] = True
        hits.append(hit)
        st.n_dopant = len(hit)
        st.n_dopant_shell = int(shell[hit].sum())
        st.n_dopant_bulk = st.n_dopant - st.n_dopant_shell
    hit_all = (np.concatenate(hits) if hits
               else np.empty(0, dtype=np.int64))
    profile = DopingProfileData(
        element=cfg.element, mode="substitutional",
        dopant_margins=margins[hit_all].copy(),
        candidate_margins=margins[cand].copy(),
        nominal_concentration=float(cfg.concentration),
        shell_width=float(width), enrichment=float(enrich))
    return stats, profile


# --- interstitial placement ------------------------------------------------


def _ghost_pad(points, r, periodic, box_lengths):
    """Cumulatively ghost-pad points across periodic faces within r.

    Cumulative per-axis padding also covers corner ghosts (a point near
    two periodic faces at once).  Shared by _min_distance_ok (k=1
    queries) and run_doping's G18 re-verification (k=2 queries).
    """
    L = np.asarray(box_lengths, dtype=np.float64)
    ghosts = points
    for ax in range(3):
        if periodic[ax]:
            lo = ghosts[ghosts[:, ax] < r].copy()
            lo[:, ax] += L[ax]
            hi = ghosts[ghosts[:, ax] > L[ax] - r].copy()
            hi[:, ax] -= L[ax]
            ghosts = np.vstack([ghosts, lo, hi])
    return ghosts


def _min_distance_ok(candidates, existing, r: float, periodic,
                     box_lengths) -> np.ndarray:
    """True where no existing point lies within r of the candidate.

    PBC-correct: all-periodic boxes use a boxsize KDTree; mixed
    periodicity ghost-pads the EXISTING set cumulatively per periodic
    axis (cumulative padding also covers corner ghosts, which matters
    when a candidate sits near two periodic faces at once).
    """
    from scipy.spatial import KDTree

    L = np.asarray(box_lengths, dtype=np.float64)
    candidates = np.asarray(candidates, dtype=np.float64).reshape(-1, 3)
    if len(existing) == 0 or len(candidates) == 0:
        return np.ones(len(candidates), dtype=bool)
    cand = _wrap_positions(candidates, periodic, L)
    ex = _wrap_positions(np.asarray(existing, dtype=np.float64), periodic,
                         L)
    if all(periodic):
        d, _ = KDTree(ex, boxsize=L).query(
            cand, k=1, distance_upper_bound=r)
        return ~(d < r)
    ghosts = _ghost_pad(ex, r, periodic, L)
    d, _ = KDTree(ghosts).query(cand, k=1, distance_upper_bound=r)
    return ~(d < r)


def place_interstitial(existing, tess, cfg, frac_sites, A, quats, streams,
                       n_grains: int, box_lengths, periodic,
                       n_target: int
                       ) -> tuple[AtomBlock, list[DopantGrainStats],
                                  DopingProfileData]:
    """Interstitial dopant block for one dopant config (mutates nothing).

    Candidate sites come from fill_grain on the interstitial fractional
    basis (same lattice + orientation as the host, home-cell filtered,
    store_margin=True).  Rejection order: (1) distance to every
    (positions, base_r) set in *existing* at radius
    max(base_r, cfg.min_distance) — passing earlier interstitial blocks
    with THEIR min_distance guarantees a later dopant can never violate
    an earlier dopant's stricter radius (which would otherwise surface
    only as a late G18 hard failure); (2) Bernoulli selection at the
    shell/bulk probabilities for the caller-supplied n_target;
    (3) deterministic self-conflict pruning (higher index of each
    too-close accepted pair is dropped and counted as rejected).
    """
    seg = cfg.gb_segregation
    width = seg.shell_width if seg.enabled else 0.0
    enrich = seg.enrichment if seg.enabled else 1.0
    stats = [DopantGrainStats() for _ in range(n_grains)]

    blocks = []
    for g in range(n_grains):
        b = fill_grain(
            g, tess, frac_sites,
            [cfg.element] * len(frac_sites),
            [None] * len(frac_sites),
            A, quats[g], box_lengths, periodic,
            rng=streams[g], store_margin=True)
        blocks.append(b)
        stats[g].n_candidate_sites = len(b)
    cand = AtomBlock.concatenate(blocks)
    assert cand.gb_margin is not None  # store_margin=True above

    ok = np.ones(len(cand), dtype=bool)
    for pos_k, base_r in existing:
        r_eff = max(float(base_r), float(cfg.min_distance))
        ok &= _min_distance_ok(cand.pos, pos_k, r_eff, periodic,
                               box_lengths)
    for g in range(n_grains):
        rejected = int(np.count_nonzero(~ok[cand.grain == g]))
        stats[g].n_rejected_min_distance = rejected

    margins = cand.gb_margin[ok]
    pos = cand.pos[ok]
    grain = cand.grain[ok]
    shell = margins <= width
    for g in range(n_grains):
        in_g = grain == g
        stats[g].n_candidate_shell = int((shell & in_g).sum())
        stats[g].n_candidate_bulk = int((~shell & in_g).sum())

    p_b, p_s = _solve_probabilities(
        n_target, int(shell.sum()), int((~shell).sum()), enrich,
        f"doping ({cfg.element}, interstitial)")

    keep = np.zeros(len(pos), dtype=bool)
    for g in range(n_grains):
        idx = np.flatnonzero(grain == g)
        if len(idx) == 0:
            continue
        r = streams[g].random(len(idx))
        keep[idx[r < np.where(shell[idx], p_s, p_b)]] = True

    sel = np.flatnonzero(keep)
    # deterministic self-conflict pruning among the accepted sites.
    # KNOWN LIMIT: _pbc_pairs ghost-pads per axis from the original
    # set (non-cumulative), so in MIXED-periodicity boxes a pair that is
    # only close via a simultaneous two-axis corner wrap can slip past
    # pruning here; G18's cumulative-ghost re-verification then hard-
    # fails the run (safe, but late).  Fully periodic boxes are exact.
    if len(sel) > 1:
        wrapped = _wrap_positions(pos[sel].copy(), periodic,
                                  np.asarray(box_lengths, float))
        drop = set()
        for i_a, i_b in _pbc_pairs(wrapped, cfg.min_distance, periodic,
                                   np.asarray(box_lengths, float)):
            hi = max(i_a, i_b)
            if min(i_a, i_b) not in drop:
                drop.add(hi)
        if drop:
            drop_idx = sel[sorted(drop)]
            for g in range(n_grains):
                stats[g].n_rejected_min_distance += int(
                    np.count_nonzero(grain[drop_idx] == g))
            keep[drop_idx] = False
            sel = np.flatnonzero(keep)

    for g in range(n_grains):
        in_g = grain[sel] == g
        st = stats[g]
        st.n_dopant = int(in_g.sum())
        st.n_dopant_shell = int((shell[sel] & (grain[sel] == g)).sum())
        st.n_dopant_bulk = st.n_dopant - st.n_dopant_shell

    seg2 = cfg.gb_segregation
    profile = DopingProfileData(
        element=cfg.element, mode="interstitial",
        dopant_margins=margins[sel].copy(),
        candidate_margins=margins.copy(),
        nominal_concentration=float(cfg.concentration),
        shell_width=float(width),
        enrichment=float(seg2.enrichment if seg2.enabled else 1.0))
    block = AtomBlock(
        pos=pos[sel],
        species=np.full(len(sel), cfg.element, dtype="U2"),
        grain=grain[sel].astype(np.int32),
        gb_margin=margins[sel],
    )
    return block, stats, profile


# --- stage orchestrator ----------------------------------------------------


@dataclass
class DopingResult:
    """Everything the pipeline needs after the stage ran."""
    rows: list[DopingReportRow]
    gate_rows: list[dict]
    n_violations: int
    violation_detail: str
    summary_rows: list[tuple[str, str, object]]
    profiles: list[DopingProfileData]


def _host_margins(atoms, tess, n_grains: int) -> np.ndarray:
    """Distance-to-GB per atom (grain-sorted block; _attach_margins
    pattern, computed unconditionally — analysis.per_atom_margin only
    controls the exported column, not this internal classification)."""
    margins = np.empty(len(atoms), dtype=np.float64)
    bounds = np.searchsorted(atoms.grain, np.arange(n_grains + 1))
    for i in range(n_grains):
        s, e = int(bounds[i]), int(bounds[i + 1])
        if e > s:
            margins[s:e] = tess.margin(atoms.pos[s:e], i)
    return margins


def run_doping(doping_cfg, atoms, tess, rng_bundle, crystal, quats,
               n_grains: int, box_lengths,
               periodic, sg_setting: str | None = None
               ) -> tuple[AtomBlock, DopingResult]:
    """Apply every dopant in config order; return (new atoms, result).

    *sg_setting* is the host crystal's space-group setting string
    (config.crystal.space_group.setting) — it selects the SAME Hall
    setting for interstitial orbit expansion that built the host basis
    (a mismatched setting would place orbit images in a rotated/shifted
    frame relative to the host atoms).

    The returned AtomBlock is NOT grain-sorted when interstitials were
    added — the caller re-runs _sort_by_grain.
    """
    sg_number = int(crystal.dataset["number"])
    streams = rng_bundle.doping_streams(n_grains)
    protected = np.zeros(len(atoms), dtype=bool)
    margins = _host_margins(atoms, tess, n_grains)
    rows: list[DopingReportRow] = []
    gate_rows: list[dict] = []
    summary: list[tuple[str, str, object]] = []
    per_dopant_stats: list[tuple[DopantConfig, list[DopantGrainStats]]] = []
    interstitial_blocks: list[tuple[str, AtomBlock, float]] = []
    profiles: list[DopingProfileData] = []

    # Every dopant's `concentration` is a fraction of the FINAL
    # structure (spec).  Substitution never changes the total, so the
    # final total is n0 plus all interstitial insertions — solve every
    # target against the SAME estimated final total up front (a
    # per-dopant sequential denominator shortchanges earlier dopants by
    # a relative factor ≈ the later dopants' concentration).
    n0 = len(atoms)
    c_interstitial = sum(d.concentration for d in doping_cfg.dopants
                         if d.mode == "interstitial")
    if c_interstitial >= 1.0:
        raise ConfigError(
            "doping: interstitial concentrations sum to "
            f"{c_interstitial:g} >= 1 — impossible by definition "
            "(fractions of the final structure).")
    n_final_est = n0 / (1.0 - c_interstitial)
    targets = [int(round(d.concentration * n_final_est))
               for d in doping_cfg.dopants]
    # existing point sets candidates must clear: (positions, base_r).
    # Earlier interstitial blocks carry THEIR min_distance so later
    # dopants respect it (G18 can then never fail from dopant ordering).
    existing: list[tuple[np.ndarray, float]] = [(atoms.pos, 0.0)]

    for d, n_target in zip(doping_cfg.dopants, targets, strict=True):
        if d.mode == "substitutional":
            stats, profile = apply_substitutional(
                atoms, margins, protected, d, streams, n_grains,
                n_target=n_target)
        else:
            frac = resolve_sites(d.sites, sg_number,
                                 f"doping ({d.element})",
                                 sg_setting=sg_setting)
            block, stats, profile = place_interstitial(
                existing, tess, d, frac, crystal.A, quats, streams,
                n_grains, box_lengths, periodic, n_target=n_target)
            interstitial_blocks.append(
                (d.element, block, float(d.min_distance)))
            existing.append((block.pos, float(d.min_distance)))
            # atoms.gb_margin is intentionally NOT carried through this
            # concatenate (host margin is None here; the stage tracks
            # its own `margins` array, and _attach_margins recomputes
            # the field later when analysis.per_atom_margin is on).
            atoms = AtomBlock.concatenate([atoms, block])
            protected = np.concatenate(
                [protected, np.ones(len(block), dtype=bool)])
            margins = np.concatenate([margins, block.gb_margin])
            # NOTE: atoms is no longer grain-sorted from here on; all
            # later dopants use explicit grain masks, not searchsorted.

        for g in range(n_grains):
            st = stats[g]
            rows.append(DopingReportRow(
                grain_id=g, element=d.element, mode=d.mode,
                n_candidate_sites=st.n_candidate_sites,
                n_rejected_min_distance=st.n_rejected_min_distance,
                n_dopant=st.n_dopant,
                n_dopant_shell=st.n_dopant_shell,
                n_dopant_bulk=st.n_dopant_bulk,
                fraction_shell=(st.n_dopant_shell / st.n_candidate_shell
                                if st.n_candidate_shell else
                                float("nan")),
                fraction_bulk=(st.n_dopant_bulk / st.n_candidate_bulk
                               if st.n_candidate_bulk else float("nan")),
            ))
        per_dopant_stats.append((d, stats))
        profiles.append(profile)

    # Gate/summary rows AFTER all dopants: 'achieved' is defined against
    # the FINAL structure (nominal = fraction of the final total, spec),
    # so earlier dopants' fractions must use the final denominator.
    n_total = len(atoms)
    # Keys carry the dopant's LIST INDEX, not just its element symbol.
    # doping.dopants is an ordered list with no uniqueness constraint on
    # `element` -- and same-element passes are a documented use ("dopants
    # applied sequentially in list order; later dopants see earlier ones as
    # atoms", DopingConfig), e.g. a bulk pass plus a GB-enriched pass of the
    # same species. Keying on the symbol alone made two such dopants collide
    # on the same (section, key) in summary.csv, so a consumer reading it into
    # a dict silently kept only the last one -- and the run still reported OK
    # with every gate passing. The element stays IN the key (it is what a
    # reader looks for) and also gets its own explicit row per dopant.
    for k, (d, stats) in enumerate(per_dopant_stats):
        prefix = f"dopant{k}_{d.element}"
        seg = d.gb_segregation
        n_d = sum(st.n_dopant for st in stats)
        achieved = n_d / n_total if n_total else 0.0
        cs = sum(st.n_candidate_shell for st in stats)
        cb = sum(st.n_candidate_bulk for st in stats)
        ds = sum(st.n_dopant_shell for st in stats)
        db = sum(st.n_dopant_bulk for st in stats)
        if seg.enabled and cs and cb and db:
            e_achieved = (ds / cs) / (db / cb)
        elif seg.enabled:
            e_achieved = float("nan")
        else:
            e_achieved = None
        gate_rows.append({
            "element": d.element,
            "nominal": d.concentration,
            "achieved": achieved,
            "enrichment_target": seg.enrichment if seg.enabled else None,
            "enrichment_achieved": e_achieved,
        })
        summary.append(("doping", f"{prefix}_element", d.element))
        summary.append(("doping", f"{prefix}_nominal_fraction",
                        d.concentration))
        summary.append(("doping", f"{prefix}_achieved_fraction",
                        achieved))
        if seg.enabled:
            summary.append(("doping", f"{prefix}_enrichment_target",
                            seg.enrichment))
            summary.append(("doping", f"{prefix}_enrichment_achieved",
                            e_achieved))
        summary.append((
            "doping", f"{prefix}_rejected_min_distance",
            sum(st.n_rejected_min_distance for st in stats)))

    # G18: independent re-verification of every interstitial dopant's
    # min_distance (fresh query, G7-style).  Each block atom is itself a
    # member of the final atom set, so it always finds itself at d ≈ 0
    # in neighbor slot 0 — the nearest OTHER atom is slot 1 of a k=2
    # query, which is exactly what min_distance constrains.
    from scipy.spatial import KDTree

    n_violations = 0
    details = []
    L_arr = np.asarray(box_lengths, dtype=np.float64)
    allw = _wrap_positions(atoms.pos.copy(), periodic, L_arr)
    for elem, block, r_min in interstitial_blocks:
        if len(block) == 0:
            details.append(f"{elem}: 0 sites")
            continue
        bw = _wrap_positions(block.pos.copy(), periodic, L_arr)
        if all(periodic):
            d2, _ = KDTree(allw, boxsize=L_arr).query(
                bw, k=2, distance_upper_bound=r_min)
        else:
            ghosts = _ghost_pad(allw, r_min, periodic, L_arr)
            d2, _ = KDTree(ghosts).query(bw, k=2,
                                         distance_upper_bound=r_min)
        viol = int(np.count_nonzero(d2[:, 1] < r_min))
        n_violations += viol
        details.append(f"{elem}: {viol} violation(s)")
    detail = "; ".join(details) if details else "no interstitial dopants"

    return atoms, DopingResult(rows=rows, gate_rows=gate_rows,
                               n_violations=n_violations,
                               violation_detail=detail,
                               summary_rows=summary,
                               profiles=profiles)
