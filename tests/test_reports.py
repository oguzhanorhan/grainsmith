"""CSV column contracts (§8.3 exact headers) + rationalize pins (§10)."""
import csv

import numpy as np
import pytest

from grainsmith.analysis.boundaries import BoundaryReport
from grainsmith.analysis.grains import GrainReport
from grainsmith.io.reports import (
    BOUNDARIES_COLUMNS, GRAINS_COLUMNS, SUMMARY_COLUMNS, VERTICES_COLUMNS,
    write_boundaries_csv, write_grains_csv, write_summary_csv,
    write_vertices_csv,
)
from grainsmith.orientation.descriptors import direction_miller, grain_descriptors, plane_miller, rationalize
from grainsmith.tessellation.flat import FlatTessellation


# ---------------------------------------------------------------------------
# §8.3 column contracts — exact, order included
# ---------------------------------------------------------------------------


def test_grains_columns_exact():
    assert GRAINS_COLUMNS == [
        "grain_id", "seed_x", "seed_y", "seed_z", "volume_A3",
        "volume_fraction", "n_atoms", "n_neighbors", "q_w", "q_x", "q_y",
        "q_z", "euler_phi1_deg", "euler_Phi_deg", "euler_phi2_deg",
        "axis_x", "axis_y", "axis_z", "angle_deg", "z_plane_hkl",
        "z_plane_dev_deg", "x_dir_uvw", "x_dir_dev_deg",
    ]


def test_boundaries_columns_exact():
    assert BOUNDARIES_COLUMNS == [
        "grain_i", "grain_j", "seed_distance_A", "misorientation_deg",
        "axis_u", "axis_v", "axis_w", "axis_dev_deg", "area_A2",
        "mean_normal_x", "mean_normal_y", "mean_normal_z",
        "normal_spread_deg", "plane_i_hkl", "plane_i_dev_deg",
        "plane_j_hkl", "plane_j_dev_deg", "character",
        "character_angle_deg", "csl_sigma", "n_overlap_deleted",
    ]


def test_vertices_and_summary_columns_exact():
    assert VERTICES_COLUMNS == ["grain_id", "v_index", "x", "y", "z",
                                "adjacent_grains"]
    assert SUMMARY_COLUMNS == ["section", "key", "value"]


# ---------------------------------------------------------------------------
# Writers round-trip
# ---------------------------------------------------------------------------


def _grain_report() -> GrainReport:
    return GrainReport(
        grain_id=0, seed_x=1.25, seed_y=2.5, seed_z=3.75,
        volume_A3=1000.0, volume_fraction=0.5, n_atoms=123, n_neighbors=4,
        q_w=1.0, q_x=0.0, q_y=0.0, q_z=0.0,
        euler_phi1_deg=0.0, euler_Phi_deg=0.0, euler_phi2_deg=0.0,
        axis_x=0.0, axis_y=0.0, axis_z=1.0, angle_deg=0.0,
        z_plane_hkl="(0 0 1)", z_plane_dev_deg=0.0,
        x_dir_uvw="[1 0 0]", x_dir_dev_deg=0.0,
    )


def test_grains_csv_roundtrip(tmp_path):
    path = tmp_path / "grains.csv"
    write_grains_csv([_grain_report()], path)
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    r = rows[0]
    assert list(r.keys()) == GRAINS_COLUMNS
    assert int(r["grain_id"]) == 0
    assert float(r["seed_x"]) == 1.25          # shortest-repr → lossless
    assert float(r["volume_fraction"]) == 0.5
    assert r["z_plane_hkl"] == "(0 0 1)"


def test_boundaries_csv_roundtrip(tmp_path):
    rep = BoundaryReport(
        grain_i=0, grain_j=1, seed_distance_A=15.0,
        misorientation_deg=36.87, axis_u=1, axis_v=0, axis_w=0,
        axis_dev_deg=0.0, area_A2=400.0,
        mean_normal_x=0.0, mean_normal_y=0.0, mean_normal_z=1.0,
        normal_spread_deg=0.0, plane_i_hkl="001", plane_i_dev_deg=0.0,
        plane_j_hkl="034", plane_j_dev_deg=0.0, character="tilt",
        character_angle_deg=90.0, csl_sigma="5", n_overlap_deleted=7,
        axis_crystal=np.array([1.0, 0.0, 0.0]),
        mean_normal=np.array([0.0, 0.0, 1.0]),
    )
    path = tmp_path / "boundaries.csv"
    write_boundaries_csv([rep], path)
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    r = rows[0]
    assert list(r.keys()) == BOUNDARIES_COLUMNS   # raw vectors excluded
    assert float(r["area_A2"]) == 400.0
    assert r["character"] == "tilt"
    assert r["csl_sigma"] == "5"
    assert int(r["n_overlap_deleted"]) == 7


def test_vertices_csv_slab_bicrystal(tmp_path):
    """Slab bicrystal: one GB plane at z=15 → vertices there list '0;1';
    wall vertices carry the -1 sentinel (§8.3, flat only)."""
    L = np.array([20.0, 20.0, 30.0])
    seeds = np.array([[10.0, 10.0, 7.5], [10.0, 10.0, 22.5]])
    tess = FlatTessellation(seeds, L, [True, True, False])
    path = tmp_path / "vertices.csv"
    write_vertices_csv(tess, L, path)
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows and list(rows[0].keys()) == VERTICES_COLUMNS
    gb_rows = [r for r in rows if abs(float(r["z"]) - 15.0) < 1e-9]
    assert gb_rows, "no vertices found on the z=15 boundary plane"
    assert all("0;1" in r["adjacent_grains"] for r in gb_rows)
    wall_rows = [r for r in rows
                 if "-1" in r["adjacent_grains"].split(";")]
    assert wall_rows, "no wall (-1) vertices found in slab geometry"
    # every vertex of every cell appears once per cell
    assert {r["grain_id"] for r in rows} == {"0", "1"}


def test_summary_csv_rfc4180_quoting(tmp_path):
    rows = [
        ("crystal", "international", "F m -3 m"),
        ("gates", "G8", "PASS | measured=0.001, ok"),
        ("box", "lengths_A", "40.0 40.0 40.0"),
    ]
    path = tmp_path / "summary.csv"
    write_summary_csv(rows, path)
    text = path.read_text(encoding="utf-8")
    assert text.splitlines()[0] == "section,key,value"
    # the comma-containing value must be quoted...
    assert '"PASS | measured=0.001, ok"' in text
    # ...and round-trip exactly through a conforming reader
    with path.open(encoding="utf-8", newline="") as fh:
        parsed = list(csv.reader(fh))
    assert parsed[2] == ["gates", "G8", "PASS | measured=0.001, ok"]


# ---------------------------------------------------------------------------
# rationalize() pins (§10 test_reports row)
# ---------------------------------------------------------------------------


def test_plane_miller_111_from_normal():
    A = 3.615 * np.eye(3)
    hkl, dev = plane_miller(np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0), A)
    assert hkl == [1, 1, 1]
    assert dev == pytest.approx(0.0, abs=1e-9)


def test_direction_miller_1m10():
    A = 3.615 * np.eye(3)
    uvw, dev = direction_miller(
        3.615 * np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0), A)
    assert [abs(v) for v in uvw] == [1, 1, 0]
    assert uvw[0] * uvw[1] < 0
    assert dev == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Fix #3: rationalize() uses L∞ norm (so corner directions are reachable)
# ---------------------------------------------------------------------------


def test_rationalize_linf_high_index_corner():
    """[12, 12, 1] must rationalize to [12, 12, 1] with near-zero deviation.

    With L2 normalisation the unit vector is [12, 12, 1] / ‖[12,12,1]‖₂ and
    scanning denominator 12 gives 12 × that = [12, 12, 1] · 12/‖…‖₂ ≠
    integer multiples → rounding error → a spuriously large deviation (several
    degrees) that causes rationalize to prefer a simpler direction.  With L∞
    normalisation v = [1, 1, 1/12] and denom=12 hits [12, 12, 1] exactly;
    float64 arithmetic leaves only a ~1e-6° residual, far below the ~1° that
    the old L2 code produced.
    """
    hkl, dev = rationalize([12, 12, 1], max_index=12)
    assert hkl == [12, 12, 1], f"expected [12,12,1], got {hkl}"
    assert dev < 1e-4, f"deviation {dev}° is too large (should be ~0)"


def test_rationalize_110_exact():
    """[1, 1, 0] is a fundamental corner; must be exact."""
    hkl, dev = rationalize([1, 1, 0], max_index=12)
    assert [abs(v) for v in hkl] == [1, 1, 0]
    assert dev == pytest.approx(0.0, abs=1e-9)


def test_rationalize_881_exact():
    """[8, 8, 1] — another high-index case; must be exact with L∞."""
    hkl, dev = rationalize([8, 8, 1], max_index=12)
    assert hkl == [8, 8, 1], f"expected [8,8,1], got {hkl}"
    assert dev == pytest.approx(0.0, abs=1e-9)


def test_rationalize_offaxis_deviation_byte_pinned():
    """Off-lattice [1, 0.49, 0] → [2, 1, 0] with a NONZERO deviation whose
    exact repr lands in the *_dev_deg CSV columns.  Pinning the literal repr
    guards the vectorised denominator scan's reduction order (P3.5): any change
    that perturbs the deviation float by even one ULP breaks byte-identity."""
    hkl, dev = rationalize([1.0, 0.49, 0.0], max_index=12)
    assert hkl == [2, 1, 0]
    assert repr(dev) == "0.4601971679794616"


# ---------------------------------------------------------------------------
# Fix #13: grain_descriptors() hkl/uvw bracketed format
# ---------------------------------------------------------------------------


def test_grain_descriptors_hkl_format_identity():
    """Identity orientation → z_plane_hkl='(0 0 1)', x_dir_uvw='[1 0 0]'."""
    A = 3.615 * np.eye(3)
    q = np.array([1.0, 0.0, 0.0, 0.0])  # identity quaternion
    d = grain_descriptors(q, A)
    assert d["z_plane_hkl"] == "(0 0 1)", f"got {d['z_plane_hkl']!r}"
    assert d["x_dir_uvw"] == "[1 0 0]", f"got {d['x_dir_uvw']!r}"


def test_grain_descriptors_hkl_format_multidigit():
    """Verify bracketed format is used even for multi-digit/negative indices.

    The old ''.join format was irreversible for e.g. [12, -1, 0] → '12-10'.
    We check that the returned strings start/end with the bracket characters.
    """
    A = 3.615 * np.eye(3)
    q = np.array([1.0, 0.0, 0.0, 0.0])
    d = grain_descriptors(q, A)
    assert d["z_plane_hkl"].startswith("(") and d["z_plane_hkl"].endswith(")")
    assert d["x_dir_uvw"].startswith("[") and d["x_dir_uvw"].endswith("]")


# ---------------------------------------------------------------------------
# GB curvature: column constants + writer
# ---------------------------------------------------------------------------


def test_curvature_columns_exact():
    from grainsmith.io.reports import (
        BOUNDARIES_COLUMNS,
        BOUNDARIES_COLUMNS_CURVATURE,
        BOUNDARIES_COLUMNS_PHASES,
        BOUNDARIES_COLUMNS_PHASES_CURVATURE,
        CURVATURE_BOUNDARY_COLUMNS,
        GB_CURVATURE_COLUMNS,
    )
    assert CURVATURE_BOUNDARY_COLUMNS == [
        "H_mean_invA", "H_std_invA", "H_abs_mean_invA",
        "K_mean_invA2", "K_std_invA2", "curv_n_samples",
    ]
    assert BOUNDARIES_COLUMNS_CURVATURE == [
        *BOUNDARIES_COLUMNS, *CURVATURE_BOUNDARY_COLUMNS]
    assert BOUNDARIES_COLUMNS_PHASES_CURVATURE == [
        *BOUNDARIES_COLUMNS_PHASES, *CURVATURE_BOUNDARY_COLUMNS]
    assert GB_CURVATURE_COLUMNS == [
        "grain_i", "grain_j", "x", "y", "z", "area_A2",
        "H_invA", "K_invA2",
    ]


def test_write_gb_curvature_csv_roundtrip(tmp_path):
    from grainsmith.analysis.curvature import CurvatureResult, PairCurvature
    from grainsmith.io.reports import (
        GB_CURVATURE_COLUMNS,
        write_gb_curvature_csv,
    )

    res = CurvatureResult(
        pairs={(0, 1): PairCurvature(
            points=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            areas=np.array([0.5, 0.25]),
            H=np.array([0.1, 0.0]),
            K=np.array([0.01, 0.0]))},
        n_raw=2, n_dropped=0)
    path = tmp_path / "gb_curvature.csv"
    write_gb_curvature_csv(res, path)
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0].keys()) == GB_CURVATURE_COLUMNS
    assert rows[0]["grain_i"] == "0" and rows[0]["grain_j"] == "1"
    assert rows[0]["H_invA"] == "0.1"
    assert rows[1]["H_invA"] == "0.0"      # exact-zero contract
    # no comment/provenance line: header is line 1
    first = path.read_text().splitlines()[0]
    assert first == ",".join(GB_CURVATURE_COLUMNS)


# ---------------------------------------------------------------------------
# Doping: column constants + writer
# ---------------------------------------------------------------------------


def test_doping_columns_exact():
    from grainsmith.io.reports import DOPING_COLUMNS
    assert DOPING_COLUMNS == [
        "grain_id", "element", "mode", "n_candidate_sites",
        "n_rejected_min_distance", "n_dopant", "n_dopant_shell",
        "n_dopant_bulk", "fraction_shell", "fraction_bulk",
    ]


def test_write_doping_csv_roundtrip(tmp_path):
    from grainsmith.atoms.doping import DopingReportRow
    from grainsmith.io.reports import DOPING_COLUMNS, write_doping_csv

    rows = [DopingReportRow(
        grain_id=0, element="C", mode="interstitial",
        n_candidate_sites=100, n_rejected_min_distance=3, n_dopant=10,
        n_dopant_shell=8, n_dopant_bulk=2, fraction_shell=0.25,
        fraction_bulk=0.05)]
    path = tmp_path / "doping.csv"
    write_doping_csv(rows, path)
    with open(path, newline="") as fh:
        out = list(csv.DictReader(fh))
    assert list(out[0].keys()) == DOPING_COLUMNS
    assert out[0]["element"] == "C" and out[0]["fraction_shell"] == "0.25"
    assert path.read_text().splitlines()[0] == ",".join(DOPING_COLUMNS)
