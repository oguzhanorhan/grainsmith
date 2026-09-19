"""Golden reproducibility hashes for the grainsmith output set (R4).

Two independent reproducibility checks live here:

``--double-run``
    Runs the golden config TWICE in the SAME process/environment and
    byte-compares the two output directories (whole-directory, MANIFEST.txt
    included, run.log excluded). This is always valid — it needs no recorded
    fixture and cannot go stale, because it never compares across machines.

``--check`` / ``--write``
    Runs the golden config ONCE and compares its file digests against a
    committed fixture (``tests/data/golden/golden_hashes.json``). This check
    IS environment-sensitive: a different BLAS build or CPU ISA changes the
    last bit of a measurable fraction of atom coordinates (see
    ``grainsmith.provenance._blas_record``'s docstring), so the digests
    recorded on one machine will not, in general, reproduce on another. The
    ``fingerprint`` block exists exactly to detect that up front and degrade
    to a warning (``STALE-ENV``) instead of a hard failure.

Usage::

    python -m tools.golden_hashes --check [--out PATH]
    python -m tools.golden_hashes --write
    python -m tools.golden_hashes --double-run

Bootstrap (first time, or after intentionally changing the golden config):

  1. Merge/run with ``tests/data/golden/golden_hashes.json`` ABSENT. ``--check``
     reports ``BOOTSTRAP``, exits 0 (green), and (in CI) uploads the freshly
     computed record as an artifact (``--out``).
  2. Download that artifact, rename it to
     ``tests/data/golden/golden_hashes.json``, commit it.
  3. The next run compares against it. The check is now armed.

Update (after an intentional numpy/scipy/spglib bump, a runner-image change,
or a deliberate output-format change):

  1. Update ``constraints-repro.txt`` to the new versions.
  2. Re-run the ``golden-hash`` CI job (``workflow_dispatch``). It reports
     ``STALE-ENV`` (or ``REGRESSION`` if the environment is unchanged and the
     change was deliberate), stays green in the ``STALE-ENV`` case, and
     uploads a fresh record.
  3. Review the artifact's ``files`` diff against the committed one — THIS IS
     THE REVIEW STEP; do not skip it. A digest that moved without an
     explanation is the finding this whole mechanism exists to surface.
  4. Commit the artifact as the new ``tests/data/golden/golden_hashes.json``.

**Never** run ``--write`` on a developer laptop and commit the result: a
laptop's conda/generic-BLAS numpy does not match CI's PyPI-wheel OpenBLAS, so
the committed golden would immediately report ``STALE-ENV`` on CI and the
check would sit permanently disarmed. ``--write`` prints a warning to this
effect when the fingerprint differs from the committed one.

The ``::notice::`` / ``::warning::`` / ``::error::`` line prefixes are GitHub
Actions workflow commands; they are emitted unconditionally (not sniffed
behind ``$GITHUB_ACTIONS``) because they are harmless plain text anywhere
else this tool runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GOLDEN_DIR = REPO / "tests" / "data" / "golden"
GOLDEN_CONFIG = GOLDEN_DIR / "golden_config.yaml"
GOLDEN_HASHES = GOLDEN_DIR / "golden_hashes.json"

GOLDEN_SCHEMA = "grainsmith/golden-hashes/v1"

GOLDEN_EXCLUDE: frozenset[str] = frozenset({
    "run.log",             # wall clock by design; MANIFEST.txt dashes it out
    "MANIFEST.txt",        # hashes the three files below, so it inherits them
    "summary.csv",         # carries the `environment` section (§1.5)
    "microstructure.json",  # ditto
})
"""Files excluded from the golden digest set, each for a stated reason.

Everything ELSE in the output directory is hashed — a whitelist would silently
lose coverage the day a new writer is added, whereas this exclusion list makes
adding an output a deliberate, reviewed golden update.

summary.csv and microstructure.json are excluded not because they are
nondeterministic on one machine (with SOURCE_DATE_EPOCH set they are perfectly
stable) but because they intentionally embed the OS release, BLAS build string
and numba state — so a GitHub runner image refresh would change them with no
change in the science. Their determinism is covered instead by the double-run
check (§3.4), which compares them byte-for-byte within one environment.
"""

FINGERPRINT_KEYS: tuple[str, ...] = (
    "python_implementation", "python_version", "os", "machine",
    "blas_name", "blas_version", "blas_detection",
    "numpy", "scipy", "spglib",
)
"""The REDUCED environment key set that gates --check's pass/fail decision.

Deliberately omits, from grainsmith.provenance.ENVIRONMENT_KEYS: os_release,
platform, libc, processor, blas_openblas_config, numba, jobs_requested,
jobs_effective — a GitHub runner kernel bump changes os_release on a schedule
nobody controls and cannot, by itself, change a float. Adds numpy/scipy/spglib
(from the `versions` section, not `environment`) because a version bump of any
of the three CAN change output bytes entirely, not just the last one:
numpy.random.Generator gives no cross-version bit-stream guarantee,
scipy.spatial.transform.Rotation.random gives none either, and spglib's
symmetry-operation ORDER defines the basis atom order (see
constraints-repro.txt). `environment` (the full record) is still kept
alongside `fingerprint` in the golden JSON for diagnosis when a check fails.
"""

_BOOTSTRAP_INSTRUCTIONS = """\
golden hash set is not bootstrapped yet.

To arm this check:
  1. Download the artifact this CI run uploaded (the record printed above).
  2. Rename it to tests/data/golden/golden_hashes.json and commit it.
  3. The next run of this check compares against the committed file.

Never run `--write` on a developer laptop and commit the result -- a laptop's
BLAS build will not match CI's, and the check would sit permanently disarmed
(STALE-ENV forever). See this module's docstring ("Bootstrap" / "Update")."""


# ---------------------------------------------------------------------------
# Running the golden config
# ---------------------------------------------------------------------------


def run_golden(workdir: Path):
    """Run the golden config inside *workdir* (chdir'd) and return the outdir.

    Sets nothing itself except the chdir: the CALLER pins SOURCE_DATE_EPOCH,
    GRAINSMITH_NO_NUMBA and the thread-count variables, so the pinning is
    visible in the workflow/test rather than hidden in a helper.
    """
    import os

    from grainsmith.config.resolve import load_config
    from grainsmith.pipeline import run

    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    prev_cwd = Path.cwd()
    os.chdir(workdir)
    try:
        # golden_config.yaml's output.directory is deliberately relative
        # ("./out") so it resolves against THIS chdir, not against wherever
        # the caller's process started (see the config's own header comment).
        config = load_config(GOLDEN_CONFIG)
        result = run(config)
    finally:
        os.chdir(prev_cwd)
    return result.outdir


def collect(outdir: Path) -> dict[str, str]:
    """Sorted {posix relpath: sha256} for every non-excluded output file."""
    outdir = Path(outdir)
    out: dict[str, str] = {}
    for p in sorted(outdir.rglob("*")):
        if not p.is_file() or p.name.endswith(".tmp"):
            continue
        if p.name in GOLDEN_EXCLUDE:
            continue
        h = hashlib.sha256()
        with p.open("rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        out[p.relative_to(outdir).as_posix()] = h.hexdigest()
    return dict(sorted(out.items()))


def _parse_summary_sections(summary_path: Path) -> dict[str, dict[str, str]]:
    """{"versions": {...}, "environment": {...}} read back from summary.csv --
    reusing what the run ACTUALLY recorded rather than re-probing, so the
    golden record can never disagree with the shipped output it describes."""
    import csv

    sections: dict[str, dict[str, str]] = {"versions": {}, "environment": {}}
    with summary_path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            section = row.get("section")
            if section in sections:
                sections[section][row["key"]] = row["value"]
    return sections


def build_record() -> dict:
    """Run the golden config once (in a scratch temp dir) and build the full
    §3.3 golden-hashes record."""
    import grainsmith
    from grainsmith.provenance import source_date_epoch
    from grainsmith.verify import parse_manifest

    with tempfile.TemporaryDirectory(prefix="grainsmith-golden-") as td:
        outdir = run_golden(Path(td))
        manifest = parse_manifest(outdir / "MANIFEST.txt")
        sections = _parse_summary_sections(outdir / "summary.csv")
        files = collect(outdir)

    environment = sections["environment"]
    versions = sections["versions"]
    fingerprint = {k: environment[k] for k in FINGERPRINT_KEYS
                  if k in environment}
    for key in ("numpy", "scipy", "spglib"):
        if key in versions:
            fingerprint[key] = versions[key]

    return {
        "schema": GOLDEN_SCHEMA,
        "grainsmith_version": grainsmith.__version__,
        "source_date_epoch": source_date_epoch(),
        "config": GOLDEN_CONFIG.relative_to(REPO).as_posix(),
        "config_sha256": manifest.fields.get("config_sha256"),
        "provenance_sha256": manifest.fields.get("provenance_sha256"),
        "fingerprint": fingerprint,
        "environment": environment,
        "files": files,
    }


# ---------------------------------------------------------------------------
# Comparison (§3.4/§3.5) -- pure, no I/O, unit-tested directly (§5.3 R3)
# ---------------------------------------------------------------------------


def compare(golden: dict, fresh: dict) -> tuple[str, list[str]]:
    """Return (status, lines) where status is one of
    "MATCH" | "BOOTSTRAP" | "STALE-ENV" | "REGRESSION".

    fingerprint is checked FIRST and gates everything else: a fingerprint
    mismatch means byte-identity was never promised between these two
    environments, so file-digest differences downstream are expected noise,
    not a regression -- reporting them as REGRESSION would make the check
    permanently, uninformatively red on every runner-image refresh.
    """
    if not golden:
        return "BOOTSTRAP", []

    golden_fp = golden.get("fingerprint") or {}
    fresh_fp = fresh.get("fingerprint") or {}
    fp_diff = sorted(
        f"{key}: golden={golden_fp.get(key)!r} fresh={fresh_fp.get(key)!r}"
        for key in sorted(set(golden_fp) | set(fresh_fp))
        if golden_fp.get(key) != fresh_fp.get(key)
    )
    if fp_diff:
        return "STALE-ENV", fp_diff

    golden_files = golden.get("files") or {}
    fresh_files = fresh.get("files") or {}
    lines: list[str] = []
    for path in sorted(set(golden_files) | set(fresh_files)):
        g = golden_files.get(path)
        f = fresh_files.get(path)
        if g != f:
            lines.append(f"{path}: expected {g!r} != actual {f!r}")

    # Not part of the STALE-ENV gate (they are not in `fingerprint`), but a
    # changed provenance_sha256/config_sha256/grainsmith_version with
    # otherwise-unchanged files is exactly the kind of silent drift this
    # tool exists to catch (e.g. the version binding broke) -- so it is
    # still reported as a REGRESSION line rather than passed over quietly.
    for key in ("grainsmith_version", "config_sha256", "provenance_sha256"):
        if golden.get(key) != fresh.get(key):
            lines.append(
                f"{key}: expected {golden.get(key)!r} != actual "
                f"{fresh.get(key)!r}")

    if lines:
        return "REGRESSION", sorted(lines)
    return "MATCH", []


def double_run(tmp_root: Path) -> list[str]:
    """Run the golden config twice into sibling directories and return the
    list of files whose bytes differ, excluding run.log. Empty list ==
    deterministic.

    Normalization is limited to the TWO measured-at-runtime row families that
    summary.csv carries by design, and to the single manifest line that hashes
    it. With SOURCE_DATE_EPOCH set, ``summary.csv`` still records this
    execution's real wall-clock stage timings (``timings,*``) and peak driver
    RSS (``memory,*``) -- see pipeline.py's ``_summary_rows`` and
    ``tests/test_end_to_end.py::test_byte_identical_rerun_source_date_epoch``,
    whose contract this mirrors exactly: "fixed timestamps preserve scientific
    bytes, not real measured timings". Those rows cannot repeat, and
    MANIFEST.txt hashes summary.csv, so its ``summary.csv`` line cannot repeat
    either.

    Everything else is still compared byte-for-byte, MANIFEST.txt's other
    ~16 digests and microstructure.json included -- deliberately NOT by
    routing through :data:`GOLDEN_EXCLUDE`, which also excludes
    microstructure.json and the whole manifest and would gut this check.

    Also runs ``verify_outdir`` on BOTH output directories: a verify failure
    on either run is appended to the returned list too, so a caller checking
    "list is empty" catches that failure as well, with no separate workflow
    step and no subprocess (this is what gives the pytest test in §5.3
    identical coverage to the CI step).
    """
    from grainsmith.verify import verify_outdir

    tmp_root = Path(tmp_root)
    outdir_a = run_golden(tmp_root / "run_a")
    outdir_b = run_golden(tmp_root / "run_b")

    def _files(outdir: Path) -> set[str]:
        return {p.relative_to(outdir).as_posix() for p in outdir.rglob("*")
               if p.is_file() and p.name != "run.log"}

    def _comparable(path: Path) -> bytes:
        """File bytes with only the unrepeatable measurements removed."""
        data = path.read_bytes()
        if path.name == "summary.csv":
            return b"\n".join(
                line for line in data.split(b"\n")
                if not line.startswith((b"timings,", b"memory,")))
        if path.name == "MANIFEST.txt":
            # Drops exactly the entry for summary.csv (whose digest moves with
            # the rows stripped above); every other digest still compared.
            return b"\n".join(
                line for line in data.split(b"\n")
                if not line.endswith(b"  summary.csv"))
        return data

    diffs: list[str] = []
    for rel in sorted(_files(outdir_a) | _files(outdir_b)):
        pa, pb = outdir_a / rel, outdir_b / rel
        if not pa.is_file() or not pb.is_file():
            diffs.append(f"{rel}: present in only one of the two runs")
            continue
        if _comparable(pa) != _comparable(pb):
            diffs.append(rel)

    for label, outdir in (("run_a", outdir_a), ("run_b", outdir_b)):
        report = verify_outdir(outdir)
        if report.exit_code != 0:
            diffs.append(
                f"{label} ({outdir}): grainsmith verify failed "
                f"({report.n_fail} failed, {report.n_warn} warning(s))")

    return sorted(diffs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _write_json(path: Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, sort_keys=True)
        fh.write("\n")


def _cmd_check(out: Path | None) -> int:
    fresh = build_record()
    if out is not None:
        _write_json(out, fresh)

    golden: dict = {}
    if GOLDEN_HASHES.is_file():
        golden = json.loads(GOLDEN_HASHES.read_text(encoding="utf-8"))
    status, lines = compare(golden, fresh)

    if status == "BOOTSTRAP":
        print(json.dumps(fresh, indent=2, sort_keys=True))
        print()
        print(_BOOTSTRAP_INSTRUCTIONS)
        print(f"::notice::{_BOOTSTRAP_INSTRUCTIONS.splitlines()[0]}")
        return 0
    if status == "STALE-ENV":
        for line in lines:
            print(line)
        print("byte-identity across environments is not guaranteed; "
              "re-bootstrap to re-arm this check")
        print("::warning::golden-hash fingerprint differs from the "
              "committed environment -- see the diff above")
        return 0
    if status == "REGRESSION":
        for line in lines:
            print(line)
        print(f"::error::golden hashes regressed ({len(lines)} mismatch(es))")
        return 1
    print(f"golden hashes match ({len(fresh['files'])} files)")
    return 0


def _cmd_write() -> int:
    fresh = build_record()
    golden: dict = {}
    if GOLDEN_HASHES.is_file():
        golden = json.loads(GOLDEN_HASHES.read_text(encoding="utf-8"))
    status, lines = compare(golden, fresh)

    if status == "STALE-ENV":
        print("WARNING: this machine's fingerprint differs from the "
              "committed golden set. Writing anyway, but CI will very "
              "likely report STALE-ENV against this file -- see "
              "tools/golden_hashes.py's module docstring ('Never run "
              "--write on a developer laptop').")
        for line in lines:
            print(line)

    _write_json(GOLDEN_HASHES, fresh)
    print(f"wrote {GOLDEN_HASHES} ({status})")
    return 0


def _cmd_double_run() -> int:
    with tempfile.TemporaryDirectory(prefix="grainsmith-doublerun-") as td:
        diffs = double_run(Path(td))
    if diffs:
        for d in diffs:
            print(d)
        print(f"::error::double-run produced {len(diffs)} differing file(s) "
              "-- this must never happen with SOURCE_DATE_EPOCH set")
        return 1
    print("double-run: byte-identical (0 differing files)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Golden reproducibility hashes for grainsmith outputs.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true",
                       help="Compare a fresh golden run against the "
                            "committed golden_hashes.json.")
    group.add_argument("--write", action="store_true",
                       help="Overwrite tests/data/golden/golden_hashes.json "
                            "with a fresh record.")
    group.add_argument("--double-run", action="store_true",
                       help="Run the golden config twice and byte-compare "
                            "the two output directories.")
    parser.add_argument("--out", type=Path, default=None, metavar="PATH",
                        help="--check only: also write the freshly computed "
                             "record to PATH (for CI artifact upload).")
    args = parser.parse_args(argv)

    # All three modes compare bytes that contain the run's UTC timestamp, so
    # an unset SOURCE_DATE_EPOCH turns every one of them into a false result:
    # --double-run reports a timing-dependent "determinism failure" (two runs
    # that straddle a second boundary), and --check reports REGRESSION against
    # a golden recorded with the clock frozen. Fail loudly instead -- the
    # caller pins the variable (see run_golden's docstring); this only refuses
    # to produce a meaningless answer when it did not.
    from grainsmith.provenance import source_date_epoch
    if source_date_epoch() is None:
        parser.error(
            "SOURCE_DATE_EPOCH must be set for reproducibility checks "
            "(the CI jobs set it to 1700000000); without a frozen clock "
            "every provenance header differs between runs and the result "
            "is meaningless. Re-run as: "
            "SOURCE_DATE_EPOCH=1700000000 python -m tools.golden_hashes ...")

    if args.out is not None and not args.check:
        parser.error("--out is only valid with --check")

    if args.check:
        return _cmd_check(args.out)
    if args.write:
        return _cmd_write()
    return _cmd_double_run()


if __name__ == "__main__":
    raise SystemExit(main())
