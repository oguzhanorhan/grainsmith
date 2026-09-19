"""GB overlap removal + ledger (§6.9)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import islice, product

import numpy as np
from scipy.spatial import KDTree

from grainsmith.atoms.fill import AtomBlock
from grainsmith.errors import ConfigError, FillError
from grainsmith.tessellation.base import Tessellation

_CUTOFF_EXPR_RE = re.compile(
    r"^\s*([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\s*\*\s*d_nn\s*$"
)


def resolve_cutoff(spec: float | str, d_nn: float | None = None) -> float:
    """Resolve the overlap cutoff (§6.9): absolute Å, or a
    ``"<factor>*d_nn"`` expression evaluated against the ideal crystal's
    minimum interatomic distance (see fill._compute_d_nn).

    Raises ConfigError for malformed expressions, non-positive results, or
    a d_nn expression without a finite positive d_nn.
    """
    if isinstance(spec, int | float):
        value = float(spec)
    elif isinstance(spec, str):
        m = _CUTOFF_EXPR_RE.match(spec)
        if m:
            if d_nn is None or not np.isfinite(d_nn) or d_nn <= 0.0:
                raise ConfigError(
                    f"Cutoff expression {spec!r} needs a finite positive "
                    f"d_nn (got {d_nn!r})."
                )
            value = float(m.group(1)) * d_nn
        else:
            try:
                value = float(spec)
            except ValueError:
                raise ConfigError(
                    f"Invalid overlap cutoff {spec!r}: use an absolute "
                    "value in Å or an expression like '0.85*d_nn'."
                ) from None
    else:
        raise ConfigError(f"Invalid overlap cutoff type: {type(spec).__name__}")

    if not np.isfinite(value) or value <= 0.0:
        raise ConfigError(f"Overlap cutoff must be positive and finite, got {value!r}.")
    return value


@dataclass
class OverlapLedger:
    """Record of atoms deleted during overlap removal."""
    deletions_by_species: dict[str, int] = field(default_factory=dict)
    deletions_by_pair: dict[tuple[int, int], int] = field(default_factory=dict)
    total_deleted: int = 0

    def record(self, species: str, grain_i: int, grain_j: int) -> None:
        self.deletions_by_species[species] = self.deletions_by_species.get(species, 0) + 1
        pair = (min(grain_i, grain_j), max(grain_i, grain_j))
        self.deletions_by_pair[pair] = self.deletions_by_pair.get(pair, 0) + 1
        self.total_deleted += 1


def _wrap_positions(pos: np.ndarray, periodic: list[bool],
                    L: np.ndarray) -> np.ndarray:
    """Wrap to [0, L) on periodic axes (neighbor searches need canonical
    coordinates; atom storage is unwrapped per §1)."""
    w = pos.copy()
    for ax in range(3):
        if periodic[ax]:
            w[:, ax] = w[:, ax] % L[ax]
            # x % L returns exactly L for tiny negative x — KDTree(boxsize)
            # requires strictly < L.
            col = w[:, ax]
            col[col >= L[ax]] = 0.0
    return w


def _wrap_positions_general(pos: np.ndarray, periodic: list[bool],
                            H: np.ndarray, Hinv: np.ndarray) -> np.ndarray:
    """Wrap to the fractional unit cell [0, 1) on periodic axes for a
    GENERAL (triclinic, ``box.cells``) box matrix ``H`` (columns =
    box vectors); reduces to :func:`_wrap_positions` exactly when ``H`` is
    diagonal (fractional wrap = Cartesian wrap / L)."""
    frac = (Hinv @ pos.T).T
    for ax in range(3):
        if periodic[ax]:
            frac[:, ax] = frac[:, ax] % 1.0
            col = frac[:, ax]
            col[col >= 1.0] = 0.0  # same >= L guard as _wrap_positions
    return (H @ frac.T).T


def _discover_pairs(
    all_pts: np.ndarray,
    cutoff: float,
    workers: int = 1,
) -> np.ndarray:
    """All index pairs (i < j) of *all_pts* at distance <= cutoff, in
    CANONICAL (lexicographically sorted) order.

    Two discovery routes, bit-identical pair SETS:

    * ``workers == 1`` -- KDTree.query_pairs, the serial route.
    * ``workers != 1`` -- two-pass ``KDTree.query_ball_point`` discovery
      (count neighbours per point, then materialize lists only for
      points with an in-range neighbour), thread-parallel via
      ``workers``.  The ball query runs the same squared-distance
      comparison in the same cKDTree C machinery as ``query_pairs``,
      so the pair set matches even at float64-boundary distances.
      Parallelism changes only the DISCOVERY SPEED; the returned pair
      set is identical.

    Either way the result is sorted canonically before return, so the
    order downstream deletion logic sees is a pure function of the atom
    coordinates -- never of Python set-hash iteration order nor of the
    worker count.
    """
    tree = KDTree(all_pts)
    return _pairs_from_tree(tree, all_pts, cutoff, workers)


# Row-chunk size for the parallel (ball-query) discovery route: bounds the
# per-chunk transient of the counting pass (the int64 count vector plus
# scipy's internal per-query buffers) REGARDLESS of the total point count
# n.  This matters for _pbc_pairs_general (triclinic), which feeds a
# 27x-ghost-replicated point set through the same query -- an unchunked
# pass over 27 x 18M ghost points is an OOM risk on commodity machines.
# The second (materialization) pass only touches "hot" points (those with
# an in-range neighbour -- on real structures the thin GB shell), so its
# transient is small without chunking.
_KNN_CHUNK_ROWS = 2_000_000


def _pairs_from_tree(
    tree: KDTree,
    pts: np.ndarray,
    cutoff: float,
    workers: int = 1,
) -> np.ndarray:
    """Canonically-ordered pairs (i < j) within cutoff for a built tree."""
    n = len(pts)
    if n < 2:
        return np.empty((0, 2), dtype=np.intp)
    if workers == 1:
        pairs = tree.query_pairs(cutoff, output_type="ndarray")
        if len(pairs) == 0:
            return np.empty((0, 2), dtype=np.intp)
        pairs = np.sort(pairs.astype(np.intp), axis=1)
    else:
        # Parallel discovery via query_ball_point(workers=...): scipy's
        # ball query runs the SAME squared-distance comparison
        # (d^2 <= cutoff^2) in the same cKDTree C machinery as
        # query_pairs, so the two routes return the EXACT same pair set
        # -- including pairs at float64-boundary distances.  (A k-NN
        # query route was rejected here: it filters on sqrt-space
        # distances, and d^2 <= r^2 vs sqrt(d^2) <= r genuinely disagree
        # on ~1/4 of exact-boundary float64 cases -- e.g. squared
        # distance 3 against cutoff sqrt(3) -- which would break the
        # jobs-invariance guarantee on degenerate geometries.)
        #
        # TWO-PASS structure (perf, not semantics): pass 1 counts
        # neighbours per point (return_length=True -- pure C, no Python
        # list materialization; overlap cutoffs sit below d_nn, so on
        # real structures only the thin GB shell has any neighbour at
        # all); pass 2 materializes neighbour lists ONLY for points with
        # count >= 2 (count includes the point itself).  The union over
        # "hot" points loses no pair: every pair (i, j) makes BOTH i and
        # j hot.  Counting is row-chunked (_KNN_CHUNK_ROWS) so the
        # transient stays bounded on the 27-image triclinic ghost path.
        hot_chunks: list[np.ndarray] = []
        for start in range(0, n, _KNN_CHUNK_ROWS):
            stop = min(start + _KNN_CHUNK_ROWS, n)
            cnt = tree.query_ball_point(
                pts[start:stop], r=cutoff, workers=workers,
                return_length=True)
            hot_chunks.append(start + np.where(cnt >= 2)[0].astype(np.intp))
        hot = (np.concatenate(hot_chunks) if hot_chunks
               else np.empty(0, dtype=np.intp))
        if len(hot) == 0:
            pairs = np.empty((0, 2), dtype=np.intp)
        else:
            neigh = tree.query_ball_point(
                pts[hot], r=cutoff, workers=workers, return_sorted=False)
            lens = np.fromiter((len(x) for x in neigh),
                               dtype=np.intp, count=len(hot))
            ii = np.repeat(hot, lens)
            jj = np.fromiter((j for js in neigh for j in js),
                             dtype=np.intp, count=int(lens.sum()))
            keep = jj > ii
            pairs = np.stack([ii[keep], jj[keep]], axis=1)
    # canonical lexicographic order (j fast axis, i slow axis)
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    return pairs[order]


def _pbc_pairs_general(
    pos: np.ndarray,
    cutoff: float,
    periodic: list[bool],
    H: np.ndarray,
    Hinv: np.ndarray,
    workers: int = 1,
) -> list[tuple[int, int]]:
    """Find all pairs within *cutoff*, respecting PBC, for a GENERAL
    (triclinic) box matrix ``H``.

    The legacy 27-image search compares images against each other, covering
    relative shifts through +/-2. It is complete for wrapped points when
    cutoff * ||row(Hinv)|| <= 1 on every periodic axis. More skewed cells
    use a reciprocal-norm-bounded search instead. Fractional component
    rounding alone is not a minimum-image algorithm for tilted cells.
    """
    reciprocal_bounds = cutoff * np.linalg.norm(Hinv, axis=1)
    if np.any(reciprocal_bounds[np.asarray(periodic, dtype=bool)] > 1.0):
        return _pbc_pairs_general_bounded(pos, cutoff, periodic, H, workers)
    n = len(pos)
    frac = (Hinv @ pos.T).T
    shift_range = [(-1, 0, 1) if periodic[ax] else (0,) for ax in range(3)]
    ghosts = []
    origin: list[int] = []
    for si in shift_range[0]:
        for sj in shift_range[1]:
            for sk in shift_range[2]:
                shifted = frac + np.array([si, sj, sk], dtype=np.float64)
                ghosts.append((H @ shifted.T).T)
                origin.extend(range(n))
    all_pts = np.concatenate(ghosts, axis=0)
    raw_pairs = _discover_pairs(all_pts, cutoff, workers=workers)
    origin_arr = np.asarray(origin, dtype=np.intp)
    pairs: set[tuple[int, int]] = set()
    for a, b in raw_pairs:
        oa, ob = int(origin_arr[a]), int(origin_arr[b])
        if oa != ob:
            pairs.add((min(oa, ob), max(oa, ob)))
    return sorted(pairs)


def _pbc_pairs_general_bounded(
    pos: np.ndarray, cutoff: float, periodic: list[bool],
    matrix: np.ndarray, workers: int,
) -> list[tuple[int, int]]:
    """Complete image search from |fractional residual| <= cutoff*||Hinv row||.

    Niggli reduction is only an efficiency aid for fully periodic cells;
    the reciprocal bounds remain complete if reduction fails. One shifted
    point block is queried at a time, not a full replicated atom cloud.
    """
    from grainsmith.atoms.fill import _search_lattice

    if len(pos) < 2:
        return []
    search_matrix = _search_lattice(matrix) if all(periodic) else matrix
    inverse = np.linalg.inv(search_matrix)
    wrapped = _wrap_positions_general(pos, periodic, search_matrix, inverse)
    fractional = (inverse @ wrapped.T).T
    extent = np.ptp(fractional, axis=0) + cutoff * np.linalg.norm(inverse, axis=1)
    extent = np.nextafter(extent, np.inf)
    ranges = [range(-int(np.floor(bound)), int(np.floor(bound)) + 1)
              if enabled else range(1)
              for bound, enabled in zip(extent, periodic, strict=True)]
    tree = KDTree(wrapped)
    pairs = {(int(pair[0]), int(pair[1]))
             for pair in _pairs_from_tree(tree, wrapped, cutoff, workers)}
    for shift in product(*ranges):
        first_nonzero = next((value for value in shift if value), 0)
        if first_nonzero <= 0:
            continue
        translation = search_matrix @ np.asarray(shift, dtype=np.float64)
        for start in range(0, len(wrapped), _KNN_CHUNK_ROWS):
            neighbors = tree.query_ball_point(
                wrapped[start:start + _KNN_CHUNK_ROWS] + translation,
                cutoff, workers=workers, return_sorted=True)
            for offset, indices in enumerate(neighbors):
                origin = start + offset
                pairs.update((min(origin, target), max(origin, target))
                             for target in indices if origin != target)
    return sorted(pairs)


def _pbc_pairs(
    pos: np.ndarray,
    cutoff: float,
    periodic: list[bool],
    box_lengths: np.ndarray,
    workers: int = 1,
) -> list[tuple[int, int]]:
    """Find all pairs within cutoff, respecting PBC.

    `pos` must already be wrapped to [0, L) on periodic axes.
    If all axes periodic: use KDTree with boxsize.
    Mixed periodicity: ghost-pad across periodic faces, search plain tree.
    """
    L = box_lengths
    all_periodic = all(periodic)

    if all_periodic:
        tree = KDTree(pos, boxsize=L)
        return [(int(a), int(b))
                for a, b in _pairs_from_tree(tree, pos, cutoff, workers)]

    # Mixed: ghost padding
    n_orig = len(pos)
    if n_orig == 0:
        # Defensive early return: with zero input rows, `ghosts` below
        # stays an empty Python list, and np.array([]) on it produces
        # shape (0,) rather than (0, 3) -- which would reach
        # _discover_pairs / KDTree with the wrong dimensionality. Callers
        # (_pairs_with_shell) already guard the shell subset from ever
        # going this thin, but this keeps the invariant true for any
        # caller of this function directly.
        return []
    ghosts = list(pos)
    ghost_origin = list(range(n_orig))  # maps ghost index → original index

    for ax in range(3):
        if periodic[ax]:
            mask_lo = pos[:, ax] < cutoff
            mask_hi = pos[:, ax] > L[ax] - cutoff
            for idx in np.where(mask_lo)[0]:
                g = pos[idx].copy()
                g[ax] += L[ax]
                ghosts.append(g)
                ghost_origin.append(int(idx))
            for idx in np.where(mask_hi)[0]:
                g = pos[idx].copy()
                g[ax] -= L[ax]
                ghosts.append(g)
                ghost_origin.append(int(idx))

    all_pts = np.array(ghosts, dtype=np.float64)
    raw_pairs = _discover_pairs(all_pts, cutoff, workers=workers)

    pairs: set[tuple[int, int]] = set()
    for a, b in raw_pairs:
        oa = ghost_origin[a]
        ob = ghost_origin[b]
        if oa != ob:
            pairs.add((min(oa, ob), max(oa, ob)))
    return sorted(pairs)


def _min_image_vec(dr: np.ndarray, periodic: list[bool],
                   L: np.ndarray) -> np.ndarray:
    out = dr.copy()
    for ax in range(3):
        if periodic[ax]:
            out[ax] -= np.round(out[ax] / L[ax]) * L[ax]
    return out


def _min_image_vec_general(dr: np.ndarray, periodic: list[bool],
                           H: np.ndarray, Hinv: np.ndarray) -> np.ndarray:
    """Shortest periodic displacement, with reciprocal-norm search bounds.

    A rounded fractional displacement supplies an initial upper bound R.
    Any better image satisfies |fractional[axis]-shift[axis]| <=
    R*||row(Hinv)||, so the finite integer search cannot miss it. The
    diagonal-cell fast path retains the original component rounding.
    """
    frac = Hinv @ dr
    for ax in range(3):
        if periodic[ax]:
            frac[ax] -= np.round(frac[ax])
    best = H @ frac
    if np.array_equal(H, np.diag(np.diag(H))) or not any(periodic):
        return best

    from grainsmith.atoms.fill import _search_lattice

    matrix = _search_lattice(H) if all(periodic) else H
    inverse = np.linalg.inv(matrix)
    fractional = inverse @ dr
    for axis, enabled in enumerate(periodic):
        if enabled:
            fractional[axis] -= np.round(fractional[axis])
    residual = matrix @ fractional
    best_squared = float(best @ best)
    radius = np.sqrt(best_squared) * np.linalg.norm(inverse, axis=1)
    lower = np.nextafter(fractional - radius, -np.inf)
    upper = np.nextafter(fractional + radius, np.inf)
    ranges = [range(int(np.ceil(lo)), int(np.floor(hi)) + 1)
              if enabled else range(1)
              for lo, hi, enabled in zip(lower, upper, periodic, strict=True)]
    shifts = iter(product(*ranges))
    while batch := list(islice(shifts, 50_000)):
        candidates = residual - np.asarray(batch, dtype=np.float64) @ matrix.T
        squared = np.einsum("ij,ij->i", candidates, candidates)
        winner = int(np.argmin(squared))
        if squared[winner] < best_squared:
            best = candidates[winner].copy()
            best_squared = float(squared[winner])
    return best


def remove_overlaps(
    atoms: AtomBlock,
    tess: Tessellation,
    cutoff: float,
    policy: str,
    periodic: list[bool],
    box_lengths: np.ndarray,
    cell_matrix: np.ndarray | None = None,
    jobs: int = 1,
    _force_full_requery: bool = False,
    _force_shell: bool = False,
) -> tuple[AtomBlock, OverlapLedger]:
    """
    Remove GB atom pairs that are closer than cutoff (§6.9): inter-grain
    pairs, and same-grain pairs across a periodic SELF-IMAGE boundary
    (close through the min-image wrap only).  Same-grain pairs that are
    close at the unwrapped distance indicate a crystal-build bug and raise
    FillError.

    Policies
    --------
    delete_shallower : delete the atom with the smaller signed boundary
        margin (tie → higher grain id loses).  NaN margins are an internal
        error and raise.
    keep_lower_id    : the atom from the higher-numbered grain loses.
    midpoint_merge   : elemental systems ONLY (raises ConfigError for
        multi-species blocks); the pair is replaced by one atom at the
        min-image midpoint, assigned to the lower grain id.

    Iterates until no pairs remain and finishes with a fresh neighbor query
    proving min inter-atomic distance ≥ cutoff (§6.9 post-condition; the
    pipeline-level G7 gate re-runs this proof on the final structure).

    For ``delete_shallower`` and ``keep_lower_id`` this is provably a
    SINGLE-PASS algorithm: every pair the first pass finds ends that pass
    with >= 1 endpoint deleted (directly, or via the deferred same-grain
    check, which otherwise raises FillError), deletion creates no new
    pairs, and no atom moves under these policies -- so a second full
    neighbor query is a re-derivation of the empty set, not new
    information.  Passes after the first therefore reuse the first pass's
    pair list, filtered against the *deleted* mask (O(n_pairs), no tree),
    instead of rebuilding a KDTree over all survivors.  ``midpoint_merge``
    moves survivors off-lattice and can create new pairs, so it always
    re-queries every pass.

    Every genuine full-population query on UNMOVED positions (pass 0, and
    any ``delete_shallower``/``keep_lower_id`` re-query) is additionally
    pre-filtered to the certified GB shell (``tess.gb_shell_lower_bound``
    -- see ``_pairs_with_shell`` below) when the backend provides one: the
    shell is a certified superset of every true pair endpoint, so pair
    discovery restricted to it finds the IDENTICAL pair set from a smaller
    tree.  Backends without a certified bound (``gb_shell_lower_bound``
    missing entirely, or returning ``None``) are unaffected -- the full
    atom population is searched, unfiltered.  A ``midpoint_merge``
    re-query on moved positions always searches the full population
    instead (see ``_pairs_with_shell``'s docstring for why the shell bound
    does not extend to moved atoms).  The §6.9 post-condition re-query
    below and the pipeline-level G7 gate deliberately do NOT use the shell
    filter, so they remain the independent full-N proof.

    The shell filter itself is only applied when ``jobs > 1`` (``workers >
    1`` below): the k-NN query it costs is scipy thread-parallel, so at
    ``workers == 1`` it is pure NEW serial work (~1.1 s per 1e6 atoms,
    measured) that only pays for itself once the discovery pass it shrinks
    is itself thread-parallel.  At ``jobs <= 1`` pair discovery therefore
    takes the unchanged full-N route, mirroring the existing jobs-dependent
    serial (``query_pairs``) vs threaded (``query_ball_point``) route
    choice in ``_discover_pairs``.  This is output-invariant, not merely a
    speed choice: the filtered and unfiltered routes are PROVEN to find the
    same pair set (``_pairs_with_shell``'s docstring), and the bitwise
    equality tests exercise both routes explicitly, forcing the filter on
    at ``workers == 1`` via ``_force_shell`` so the equality proof covers
    the serial case too.

    Deliberately-accepted limitation: the shell's exactness proof bounds
    distance to a DIFFERENT replica's boundary only, so a duplicate-
    lattice-point crystal-build bug that also sits far from every grain
    boundary (large margin) can escape the shell-filtered search entirely
    -- ``remove_overlaps`` does not raise ``FillError`` for such a case
    when the shell filter is active.  ``qa.gate_g7_min_distance`` (a
    SEPARATE, always full-N, never shell-filtered fresh query the pipeline
    runs after overlap removal) remains the independent, unconditional
    catcher of both an interior build bug and a buggy shell bound -- see
    tests/test_overlap.py's
    ``test_overlap_shell_duplicate_bug_deep_interior_known_gap``, which
    pins this behaviour AND asserts G7 catches it on that exact structure.

    Parameters
    ----------
    cell_matrix : (3,3), optional
        General box matrix (columns = box vectors) for a triclinic
        (``box.cells``) single-crystal box.  ``None`` (default) uses the
        orthogonal box ``diag(box_lengths)`` and the fast ``KDTree(boxsize=L)``
        path.  A plain per-axis min-image is WRONG in a tilted cell (the
        nearest periodic image is not found by wrapping each Cartesian axis
        independently), so a
        triclinic *cell_matrix* routes every neighbor search and
        minimum-image computation through the general (reciprocal-bounded,
        fractional-coordinate) path instead.
    _force_full_requery : bool, internal test hook, default False
        Disable the single-pass shortcut above and re-query on every pass
        instead, for equivalence testing against the fast path.
    _force_shell : bool, internal test hook, default False
        Apply the certified GB-shell pre-filter even at ``workers == 1``
        (normally skipped there -- see above), so byte-equality tests can
        prove the filtered and unfiltered routes agree in the serial case
        too, not only at ``jobs > 1``.

    Returns
    -------
    (AtomBlock, OverlapLedger)
    """
    if policy not in ("delete_shallower", "keep_lower_id", "midpoint_merge"):
        raise ConfigError(f"Unknown overlap policy: {policy!r}")

    ledger = OverlapLedger()
    pos = atoms.pos.copy()
    species = atoms.species.copy()
    grain = atoms.grain.copy()
    margin = atoms.gb_margin.copy() if atoms.gb_margin is not None else None

    if policy == "midpoint_merge" and len(np.unique(species)) > 1:
        raise ConfigError(
            "midpoint_merge is restricted to elemental (single-species) "
            "systems (§6.9): merging atoms of different species would "
            "silently change the composition. Use delete_shallower or "
            "keep_lower_id."
        )

    deleted = np.zeros(len(pos), dtype=bool)
    # midpoint_merge moves survivors OFF the lattice; such atoms may then
    # legitimately sit < cutoff from their own grain's lattice neighbors,
    # so the intra-grain build-error check must exempt them.
    moved = np.zeros(len(pos), dtype=bool)
    L = np.asarray(box_lengths, dtype=np.float64)
    # jobs parallelizes only the neighbour-search DISCOVERY (scipy thread
    # workers); the pair set and its canonical processing order are
    # identical for every jobs value (see _discover_pairs) -- the pair SET
    # is worker-count invariant by construction, so workers == jobs is a
    # pure measurement-scaling knob, not a correctness knob. jobs == 1
    # keeps the serial query_pairs route; jobs > 1 uses exactly that many
    # scipy threads for the ball-query route, so overlap's jobs=2..N rows
    # scale with --jobs instead of all measuring the same thread count.
    # jobs <= 1 (including non-positive values) maps to workers=1 -- scipy's
    # query_ball_point rejects workers=0 outright, and workers=-1 means
    # "all cores", neither of which is what a non-positive jobs should mean
    # here -- so only jobs > 1 is passed through verbatim; jobs <= 1 always
    # takes the serial query_pairs route.
    workers = jobs if jobs > 1 else 1
    triclinic = cell_matrix is not None
    if triclinic:
        H = np.asarray(cell_matrix, dtype=np.float64)
        Hinv = np.linalg.inv(H)

        def _wrap(p):
            return _wrap_positions_general(p, periodic, H, Hinv)

        def _pairs(p):
            return _pbc_pairs_general(p, cutoff, periodic, H, Hinv,
                                      workers=workers)

        def _minimg(dr):
            return _min_image_vec_general(dr, periodic, H, Hinv)
    else:
        def _wrap(p):
            return _wrap_positions(p, periodic, L)

        def _pairs(p):
            return _pbc_pairs(p, cutoff, periodic, L, workers=workers)

        def _minimg(dr):
            return _min_image_vec(dr, periodic, L)

    def _pairs_with_shell(
        pos_act_w: np.ndarray, grain_act: np.ndarray, moved_act: np.ndarray,
    ) -> list[tuple[int, int]]:
        """Full pair discovery, pre-filtered to the certified GB shell when
        *tess* provides one (``Tessellation.gb_shell_lower_bound``).

        Mechanically equal to ``_pairs(pos_act_w)``: the shell mask is a
        certified SUPERSET of every true pair endpoint (see
        ``gb_shell_lower_bound``'s docstring for the proof), so pair
        discovery restricted to it finds the identical pair set -- only
        the tree it is built over shrinks. ``_pairs`` always returns pairs
        in canonical lexicographic order (local indices), and
        ``np.flatnonzero`` of a boolean mask is strictly increasing, so
        mapping shell-local pairs back through it preserves that same
        canonical order; the explicit re-lexsort below is a cheap defensive
        re-assertion of that invariant, not a behaviour change.

        *moved_act* (``moved[act_idx]``, all-False except under
        ``midpoint_merge``) makes this function BYPASS the shell entirely
        and search the full population instead. The shell's exactness
        proof requires *x* and *y* to be owned by DIFFERENT replicas
        (``m_rep(x) <= |x-y|`` follows from ``d_k(y) <= d_h(y)`` under
        that ownership split); it also requires *grain_act* to be each
        atom's TRUE owning grain, an interior point of that grain's cell.
        Both hold for every freshly-filled atom but neither is guaranteed
        for a ``midpoint_merge`` survivor ``M`` relocated to an off-lattice
        GB midpoint and reassigned to ``min(gi, gj)``. Forcing only ``M``
        into the shell is not enough: an unmoved surviving atom ``Z`` of
        the same grain can land within cutoff of ``M`` while sharing
        ``M``'s owner replica, so ``m_rep(Z)`` is not bounded by cutoff --
        ``Z`` can be excluded from the shell, and the ``(Z, M)`` pair,
        which needs BOTH endpoints in the subset tree, goes undetected.
        Only the FIRST merge generation is provably safe at any cutoff
        below ``d_nn`` (``|Z-M| >= |Z-A| >= d_nn`` from the face-plane
        algebra of the originating pair ``A``); later generations move the
        survivor further off-lattice and that inequality no longer holds.
        Searching the full population whenever any input atom has moved
        is therefore unconditional rather than resting on a multi-
        generation bound, and the ``midpoint_merge`` re-query already runs
        on a much smaller surviving population, so the cost is small.

        KNOWN, DELIBERATELY-ACCEPTED LIMITATION: the shell's exactness
        proof bounds distance to a DIFFERENT replica's boundary only.  It
        says nothing about SAME-replica (same-grain) proximity, so a
        duplicate-lattice-point crystal-build bug that ALSO happens
        to sit far from every grain boundary (large margin) can escape
        this search entirely -- measured: a full-N same-grain-only
        supplementary search to close that gap costs as much as the
        unfiltered baseline itself (0.5-0.6 s at 1.37M atoms, and
        ``KDTree.query_pairs`` has no ``workers`` knob to shrink that with
        --jobs), which would erase this pre-filter's entire saving.
        ``qa.gate_g7_min_distance`` (a SEPARATE, always full-N, never
        shell-filtered fresh query the pipeline runs after overlap
        removal) remains the independent, unconditional catcher of both
        an interior build bug and a buggy shell bound -- see
        tests/test_overlap.py's
        ``test_overlap_shell_duplicate_bug_deep_interior_known_gap`` for
        this behaviour pinned and documented as a test, not a silent gap.
        """
        if moved_act.any():
            return _pairs(pos_act_w)
        if workers <= 1:
            # Serial route: the k-NN shell query itself is a scipy
            # thread-parallel operation (see gb_shell_lower_bound /
            # _shell_knn.certified_knn_shell_mask), so at workers == 1 it
            # is ~1.1 s of pure NEW serial cost per 1e6 atoms that only
            # pays for itself once the discovery it shrinks is itself
            # thread-parallel. The filtered and unfiltered routes are
            # provably the same pair set (this function's own docstring,
            # and the byte-equality tests in test_overlap.py, which force
            # the filter on at workers=1 via _force_shell to prove this
            # for the serial case too) -- so skipping the filter here is
            # a pure perf knob, output-invariant like the serial/threaded
            # discovery route choice in _discover_pairs itself.
            if not _force_shell:
                return _pairs(pos_act_w)
        # tess.gb_shell_lower_bound is an OPTIONAL hook (Tessellation's
        # base implementation already returns None, but some lightweight
        # test doubles duck-type only the required `.margin()` contract
        # and provide no such attribute at all) -- getattr rather than a
        # direct call keeps objects without that attribute working unchanged.
        bound = getattr(tess, "gb_shell_lower_bound", None)
        shell = (None if bound is None
                 else bound(pos_act_w, grain_act, cutoff, workers))
        if shell is None:
            return _pairs(pos_act_w)
        if shell.all():
            return _pairs(pos_act_w)
        shell_idx = np.flatnonzero(shell)
        if len(shell_idx) < 2:
            # Fewer than two shell atoms can never yield a pair (_pairs
            # filters oa != ob, so a lone atom cannot pair with its own
            # ghost) -- and calling _pairs on a (0, 3) input hits scipy's
            # `ValueError: data must be of shape (n, m)` in the mixed/
            # non-fully-periodic ghost-padding branch (np.array([]) on an
            # empty Python ghost list produces shape (0,), not (0, 3)).
            return []
        sub_pairs = _pairs(pos_act_w[shell_idx])
        if not sub_pairs:
            return sub_pairs
        arr = np.asarray(sub_pairs, dtype=np.intp).reshape(-1, 2)
        mapped = shell_idx[arr]
        order = np.lexsort((mapped[:, 1], mapped[:, 0]))
        mapped = mapped[order]
        return [(int(a), int(b)) for a, b in mapped]

    # delete_shallower margin memo: an atom's (position, grain) never
    # changes under the delete policies, so margins are computed ONCE,
    # batched per grain.  Per-pair scalar margin() calls in arbitrary
    # grain order made EDT-field-backed backends (voxel_import)
    # recompute whole distance fields per pair — 100× slowdowns.
    atom_margin = np.empty(len(pos), dtype=np.float64)
    margin_known = np.zeros(len(pos), dtype=bool)

    def _memoize_margins(idx_arr: np.ndarray) -> None:
        need = idx_arr[~margin_known[idx_arr]]
        if len(need) == 0:
            return
        for g in np.unique(grain[need]):
            sel = need[grain[need] == g]
            atom_margin[sel] = tess.margin(pos[sel], int(g))
        margin_known[need] = True

    MAX_PASSES = 100  # typical convergence is ≤ ~3 passes (§6.9)
    # Track convergence so the §6.9 post-condition (a fresh full PBC
    # query) can be skipped when it is provably redundant — see the post-loop
    # note.  This is bit-identical: it changes no deletion, it only avoids one
    # O(N) re-query.
    # The neighbour search is parallelisable BECAUSE the pair
    # ORDER is made deterministic: _discover_pairs returns pairs in
    # canonical lexicographic order (a pure function of the coordinates), so
    # the deletion sequence does not depend on set-hash iteration order, and
    # the discovery route (serial query_pairs vs parallel k-NN) provably
    # returns the same pair set.  Outputs therefore do not vary with the
    # Python hash seed and cannot vary with the worker count.  NOTE: at a
    # margin TIE, the deterministic canonical-order choice can differ from
    # a hash-order tie-break by ~0.01-0.2% of deleted atoms (one-time);
    # G7 re-proves min-distance on every run.
    # Single-pass invariant: pass 0's pair list, in ORIGINAL atom indices
    # (shape (n_pairs, 2)), kept so pass >= 1 can filter it against
    # `deleted` instead of re-querying (see the single-pass note in the
    # docstring). Only populated for policy != "midpoint_merge"; stays
    # None otherwise.
    pass0_pairs_global: np.ndarray | None = None
    converged = False
    for pass_idx in range(MAX_PASSES):
        act_idx = np.where(~deleted)[0]
        if len(act_idx) == 0:
            converged = True
            break

        use_filter = (
            not _force_full_requery
            and policy != "midpoint_merge"
            and pass0_pairs_global is not None
        )
        # pairs_arr, when set below, is raw_pairs already converted to an
        # intp (n_pairs, 2) array -- computed at most once per pass and
        # reused for both the pass0_pairs_global cache and the
        # delete_shallower margin memoization below, instead of converting
        # the same list-of-tuples twice.
        pairs_arr: np.ndarray | None = None
        if use_filter:
            # `use_filter` already includes `pass0_pairs_global is not None`
            # (see its construction above); mypy cannot narrow through a bool.
            assert pass0_pairs_global is not None
            # O(n_pairs) subset filter of pass 0's pair set on the CURRENT
            # deleted mask. Mechanically equal to a fresh re-query here: no
            # atom moves under the delete policies, so no pair can exist
            # among survivors that pass 0's full-population query did not
            # already find, and (see the pending_bug resolution below)
            # every pass-0 pair has >= 1 endpoint deleted by the end of
            # pass 0 -- so this is proven to always come back empty.
            alive = (~deleted[pass0_pairs_global[:, 0]]
                     & ~deleted[pass0_pairs_global[:, 1]])
            if alive.any():
                # Defensive only: the filter above is proven empty for
                # every pass >= 1 under the delete policies. If it were
                # ever non-empty, the survivors here are ORIGINAL atom
                # indices, not indices into this pass's `act_idx` (which
                # the per-pair loop below requires) -- so fall back to a
                # real, correctly-indexed re-query rather than risk
                # processing a mismatched index space. Also refresh the
                # cached pass0_pairs_global from this re-query (use_filter
                # already guarantees policy != "midpoint_merge" here), so
                # any subsequent pass filters against this FRESH pair set
                # instead of the stale pass-0 one.
                pos_act = _wrap(pos[act_idx])
                raw_pairs = _pairs_with_shell(
                    pos_act, grain[act_idx], moved[act_idx])
                pairs_arr = np.asarray(raw_pairs, dtype=np.intp).reshape(-1, 2)
                pass0_pairs_global = (act_idx[pairs_arr] if pairs_arr.size
                                      else np.empty((0, 2), dtype=np.intp))
            else:
                raw_pairs = []
        else:
            pos_act = _wrap(pos[act_idx])
            raw_pairs = _pairs_with_shell(
                pos_act, grain[act_idx], moved[act_idx])
            if pass_idx == 0 and policy != "midpoint_merge":
                pairs_arr = np.asarray(raw_pairs, dtype=np.intp).reshape(-1, 2)
                pass0_pairs_global = (act_idx[pairs_arr] if pairs_arr.size
                                      else np.empty((0, 2), dtype=np.intp))

        if not raw_pairs:
            # No pair among the surviving atoms: the structure is clean, and
            # act_idx == keep with identical wrapped positions, so the
            # post-condition below would re-prove exactly this empty result.
            converged = True
            break

        if policy == "delete_shallower":  # raw_pairs is non-empty here
            if pairs_arr is None:
                pairs_arr = np.asarray(raw_pairs, dtype=np.intp).reshape(-1, 2)
            _memoize_margins(np.unique(act_idx[pairs_arr.ravel()]))

        changed = False
        # Same-grain "crystal-build-error" verdicts are DEFERRED to the
        # end of this pass rather than raised on first sight: whether a
        # same-grain pair is a genuine duplicate-lattice-point bug, or
        # merely a transient artifact of an atom that a DIFFERENT (inter-
        # grain or self-image) pair is about to remove/move anyway, is
        # undecidable mid-pass.  Deciding immediately makes the verdict a
        # function of PAIR VISITATION ORDER: a hash-order iteration could
        # happen to resolve the unrelated conflicting pair BEFORE reaching
        # this one, while the canonical lexicographic order can visit it
        # FIRST instead, surfacing a false positive. Deferring the
        # verdict to pass-end restores order-independence
        # (regression: test_post_condition_no_residual_pairs).
        pending_bug: list[tuple[int, int, int, float]] = []
        for ia, ib in raw_pairs:
            oi = int(act_idx[ia])
            oj = int(act_idx[ib])
            if deleted[oi] or deleted[oj]:
                continue
            gi = int(grain[oi])
            gj = int(grain[oj])

            # Same-grain pairs: distinguish a crystal-build bug from a
            # legitimate SELF-IMAGE grain boundary.  A grain whose lattice
            # is incommensurate with the periodic box forms a GB with its
            # own periodic image (flat.py: neighbor_id == grain_id is a
            # valid GB face); such pairs are close only through the
            # min-image wrap.  A genuine build error (duplicate lattice
            # points) is close at the UNWRAPPED distance — positions are
            # stored unwrapped in the grain's compact home cell (§1).
            if gi == gj:
                d_direct = float(np.linalg.norm(pos[oi] - pos[oj]))
                if (d_direct < cutoff
                        and not (moved[oi] or moved[oj])):
                    # Two atoms of one grain closer than cutoff at the
                    # unwrapped distance are impossible for a single rigid
                    # lattice (min distance = d_nn) UNLESS this pass is
                    # about to remove/move one of them via an unrelated
                    # conflict -- defer the verdict (see note above)
                    # instead of raising immediately.
                    pending_bug.append((oi, oj, gi, d_direct))
                    continue
                # else: self-image GB contact, resolved per policy below.
                # Grain-id tie-breaks degenerate; both policies then
                # deterministically remove the higher atom index.

            if policy == "delete_shallower":
                mi = float(atom_margin[oi])
                mj = float(atom_margin[oj])
                if np.isnan(mi) or np.isnan(mj):
                    raise FillError(
                        f"NaN margin for atoms {oi}/{oj} (grains {gi}/{gj}) — "
                        "tessellation margin() is corrupt (internal error)."
                    )
                if mi < mj:
                    to_del = oi
                elif mj < mi:
                    to_del = oj
                else:
                    to_del = oi if gi > gj else oj
            elif policy == "keep_lower_id":
                # gi == gj (self-image GB): delete the higher atom index.
                to_del = oj if gi <= gj else oi
            else:  # midpoint_merge
                # Min-image midpoint: a plain average puts a PBC-straddling
                # pair on the wrong side of the box.
                dr = _minimg(pos[oj] - pos[oi])
                mid = pos[oi] + 0.5 * dr
                g_keep = min(gi, gj)
                survivor = oi if gi == g_keep else oj
                victim = oj if survivor == oi else oi
                deleted[victim] = True
                pos[survivor] = mid
                moved[survivor] = True
                grain[survivor] = g_keep
                ledger.record(str(species[victim]), gi, gj)
                changed = True
                continue

            deleted[to_del] = True
            ledger.record(str(species[to_del]), gi, gj)
            changed = True

        # Resolve deferred same-grain verdicts now that every pair in this
        # pass has been processed (see the note above the pending_bug
        # collection): a deferred pair is a genuine crystal-build error
        # only if BOTH atoms are still alive and unmoved after the full
        # pass -- i.e. no unrelated conflict explains their survival at
        # sub-cutoff separation. This is a pure function of end-of-pass
        # state, independent of the order raw_pairs was visited in.
        for oi, oj, gi, d_direct in pending_bug:
            if deleted[oi] or deleted[oj] or moved[oi] or moved[oj]:
                continue
            raise FillError(
                f"Intra-grain atom overlap at unwrapped distance "
                f"{d_direct:.4g} Å < cutoff (grain {gi}). This "
                "indicates a crystal-build error."
            )

        if not changed:
            break
    else:
        raise FillError(
            f"Overlap removal did not converge within {MAX_PASSES} passes "
            "(pathological overlap cluster — check cutoff and tessellation)."
        )

    keep = ~deleted

    # Post-condition (§6.9): a FRESH query must prove no remaining pair.
    # When the loop converged normally, the final pass already ESTABLISHED
    # this -- either by a literal query that found none (act_idx == keep,
    # identical wrapped positions), or, for the single-pass delete-policy
    # shortcut above, by the proven-equivalent deleted-mask filter -- so
    # re-querying here is redundant either way.  The fresh query is kept
    # only for the non-converged exit, which is in practice UNREACHABLE (a
    # non-empty pair set always deletes >= 1 atom, so the loop can only leave via
    # converged=True); it is retained as a cheap defensive fallback, and the
    # pipeline-level G7 gate independently re-proves min-distance regardless.
    if not converged:
        final_pos = _wrap(pos[keep])
        if len(final_pos) > 1:
            residual = _pairs(final_pos)
            if residual:
                raise FillError(
                    f"Overlap removal post-condition failed: {len(residual)} "
                    f"pair(s) closer than cutoff={cutoff:.4g} Å remain "
                    "(internal logic error)."
                )

    result = AtomBlock(
        pos=pos[keep],
        species=species[keep],
        grain=grain[keep],
        gb_margin=margin[keep] if margin is not None else None,
    )
    return result, ledger
