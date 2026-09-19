"""``grainsmith verify``: audit an output directory against its own
MANIFEST.txt, from shipped files alone (§5, R2).

Pure stdlib + :mod:`grainsmith.provenance`.  Deliberately does NOT import
``numpy``, ``scipy``, ``spglib``, or :mod:`grainsmith.pipeline` — verifying an
archived directory (possibly on a machine with no scientific stack installed,
or produced by a grainsmith version that predates the current one) must work
in a minimal environment and must be fast even on a multi-GB LAMMPS dump,
since every file digest is streamed rather than loaded whole.

Two entry points:

``verify_outdir(outdir, ...)``
    Runs the fixed check sequence C1-C7 (§5.2) and returns a
    :class:`VerifyReport`.
``render_report(report, quiet=False)``
    Turns a report into the exact §5.3 text layout — kept separate from
    ``verify_outdir`` so rendering is unit-testable without capturing stdout.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import grainsmith
from grainsmith.errors import ManifestError
from grainsmith.provenance import provenance_sha256

_CONFIG_FILENAME = "resolved_config.yaml"
_SUMMARY_FILENAME = "summary.csv"
_MICROSTRUCTURE_FILENAME = "microstructure.json"

_FIELD_RE = re.compile(r"^#\s*([A-Za-z0-9_]+)\s*=\s*(.*)$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


# ---------------------------------------------------------------------------
# Manifest parsing (§5.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ManifestEntry:
    """One MANIFEST.txt data line: ``sha256  size_bytes  path``."""
    sha256: str | None      # None for a dashed (unhashed) entry, e.g. run.log
    size: int | None        # None for a dashed entry
    path: str               # POSIX-relative to the manifest's directory


@dataclass(frozen=True)
class Manifest:
    """A parsed MANIFEST.txt."""
    header: str                    # the provenance line, "#" and space stripped
    version: str | None            # parsed out of `header`
    timestamp_iso: str | None
    seed: str | None
    fields: dict[str, str]         # the "# <key> = <value>" comment lines
    entries: list[ManifestEntry]   # in file order (already sorted by the writer)


def parse_manifest(path: Path) -> Manifest:
    """Parse *path* as a grainsmith MANIFEST.txt.

    Strict by design (§5.1): a malformed manifest is a *cannot-verify*
    (``ManifestError``, ``grainsmith verify`` exit code 2), never a check
    failure — a check failure means the manifest is fine but what it
    describes doesn't match reality.
    """
    if not path.is_file():
        raise ManifestError(f"MANIFEST.txt not found: {path}")

    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    # The writer always terminates the file with "\n" (pipeline._write_manifest
    # joins lines with "\n" and appends one), which leaves one empty trailing
    # element after split("\n") — drop it so it is never mistaken for a data
    # line requiring "sha256 size path".
    if lines and lines[-1] == "":
        lines = lines[:-1]
    if not lines:
        raise ManifestError(f"MANIFEST.txt is empty: {path}")

    first = lines[0]
    if not first.startswith("# grainsmith "):
        raise ManifestError(
            f"MANIFEST.txt line 1 is not a grainsmith provenance header: "
            f"{first!r}")
    header = first[2:]  # strip the leading "# "
    parts = header.split(" | ")

    version: str | None = None
    tok0 = parts[0].split()
    if len(tok0) >= 2:
        version = tok0[1]
    timestamp_iso = parts[1] if len(parts) >= 2 else None
    seed: str | None = None
    for p in parts[2:]:
        if p.startswith("seed="):
            seed = p[len("seed="):]

    fields: dict[str, str] = {}
    entries: list[ManifestEntry] = []
    for lineno, line in enumerate(lines[1:], start=2):
        if not line:
            continue
        if line.startswith("#"):
            m = _FIELD_RE.match(line)
            if m:
                fields[m.group(1)] = m.group(2)
            # else: the column legend ("# sha256  size_bytes  path ...") or
            # any other comment — ignored, forward compatible.
            continue
        # Two literal spaces separate the columns (see _write_manifest), but
        # maxsplit=2 (not a bare split()) collapses that correctly AND lets a
        # filename containing a space survive as the third token.
        tokens = line.split(maxsplit=2)
        if len(tokens) != 3:
            raise ManifestError(
                f"MANIFEST.txt line {lineno}: expected "
                f"'sha256  size_bytes  path', got {line!r}")
        digest, size_s, rel = tokens
        if digest == "-" * 64 and size_s == "-":
            entries.append(ManifestEntry(sha256=None, size=None, path=rel))
            continue
        if not _HEX64_RE.match(digest):
            raise ManifestError(
                f"MANIFEST.txt line {lineno}: invalid sha256 digest "
                f"{digest!r}")
        try:
            size = int(size_s)
        except ValueError as exc:
            raise ManifestError(
                f"MANIFEST.txt line {lineno}: invalid size {size_s!r}"
            ) from exc
        if size < 0:
            raise ManifestError(
                f"MANIFEST.txt line {lineno}: negative size {size}")
        entries.append(ManifestEntry(sha256=digest, size=size, path=rel))

    if not entries:
        raise ManifestError(f"MANIFEST.txt has no file entries: {path}")
    for required in ("config_sha256", "provenance_sha256"):
        if required not in fields:
            raise ManifestError(
                f"MANIFEST.txt is missing the '# {required} = ...' field")

    return Manifest(header=header, version=version, timestamp_iso=timestamp_iso,
                    seed=seed, fields=fields, entries=entries)


# ---------------------------------------------------------------------------
# Checks (§5.2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckResult:
    """One row of the §5.3 report."""
    name: str
    status: str   # "OK" | "WARN" | "FAIL" | "SKIP"
    summary: str
    details: list[str] = field(default_factory=list)


def _strip_hash_header(raw: bytes) -> bytes:
    """Reconstructs exactly the bytes ``config.resolve._config_yaml`` produced.

    ``dump_resolved`` writes ``header + raw_str`` with no separator, and
    ``atomic_writer`` pins ``newline="\\n"``, so dropping the leading
    contiguous ``#``-prefixed lines and rejoining with ``b"\\n"`` recovers the
    exact hashed payload — this is what makes the config digest recomputable
    from the shipped file alone (§0, §5.2-C4). ``splitlines()`` is
    deliberately NOT used: it would eat the line terminator and change the
    hashed bytes.
    """
    lines = raw.split(b"\n")
    i = 0
    while i < len(lines) and lines[i].startswith(b"#"):
        i += 1
    return b"\n".join(lines[i:])


def _check_manifest(manifest: Manifest) -> CheckResult:
    return CheckResult("manifest", "OK", f"{len(manifest.entries)} entries")


def _check_file_digests(outdir: Path, manifest: Manifest) -> CheckResult:
    """C2: stream every non-dashed entry in 1 MiB blocks — outputs are
    multi-GB, so no file is ever read whole (mirrors
    ``pipeline._write_manifest``)."""
    problems: list[str] = []
    n_ok = 0
    dashed: list[str] = []
    for e in manifest.entries:
        if e.sha256 is None:
            dashed.append(e.path)
            continue
        full = outdir / e.path
        if not full.is_file():
            problems.append(f"{e.path}: file is missing")
            continue
        actual_size = full.stat().st_size
        if actual_size != e.size:
            problems.append(
                f"{e.path}: size mismatch (manifest {e.size}, "
                f"actual {actual_size})")
            continue
        h = hashlib.sha256()
        with full.open("rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        if h.hexdigest() != e.sha256:
            problems.append(f"{e.path}: sha256 mismatch")
            continue
        n_ok += 1

    if problems:
        problems.sort()
        details = list(problems[:10])
        if len(problems) > 10:
            details.append(f"... and {len(problems) - 10} more")
        return CheckResult(
            "file-digests", "FAIL",
            f"{len(problems)} of {len(manifest.entries)} entries failed "
            "verification", details)

    summary = f"{n_ok} verified"
    if dashed:
        summary += (f", {len(dashed)} without a recorded digest "
                   f"({', '.join(sorted(dashed))})")
    return CheckResult("file-digests", "OK", summary)


def _check_extra_files(outdir: Path, manifest: Manifest) -> CheckResult:
    """C3: stale files in a reused output directory are, by design, not
    attributed to the run (pipeline._write_manifest only lists files THIS
    run produced) — so extras are always a WARN here. ``--strict`` promotes
    ALL warnings to failures uniformly at the report level
    (``VerifyReport.exit_code``), not per-check, so every WARN-capable check
    (this one, C5, C6) is promoted the same way."""
    listed = {e.path for e in manifest.entries}
    extra: list[str] = []
    for p in outdir.rglob("*"):
        if not p.is_file() or p.name.endswith(".tmp"):
            continue
        rel = p.relative_to(outdir).as_posix()
        if rel == "MANIFEST.txt" or rel in listed:
            continue
        extra.append(rel)

    if not extra:
        return CheckResult("extra-files", "OK", "no files outside MANIFEST.txt")

    extra.sort()
    n = len(extra)
    summary = f"{n} file{'s' if n != 1 else ''} not listed in MANIFEST.txt"
    return CheckResult("extra-files", "WARN", summary, extra)


def _check_config_roundtrip(
    outdir: Path, manifest: Manifest,
) -> tuple[CheckResult, str | None]:
    """C4. Returns (result, recomputed_digest) — the digest is reused by C5's
    comparison (b) whenever it was actually computed (even on a mismatch:
    C5 must independently detect the same tamper, see spec V4)."""
    path = outdir / _CONFIG_FILENAME
    listed = any(e.path == _CONFIG_FILENAME for e in manifest.entries)
    exists = path.is_file()
    expected = manifest.fields["config_sha256"]

    if not exists and not listed:
        return CheckResult(
            "config-roundtrip", "SKIP",
            f"{_CONFIG_FILENAME} is absent and not listed in MANIFEST.txt "
            "— nothing to verify"), None
    if not exists:
        return CheckResult(
            "config-roundtrip", "FAIL",
            f"{_CONFIG_FILENAME} is listed in MANIFEST.txt but is missing "
            "from the output directory"), None
    if not listed:
        return CheckResult(
            "config-roundtrip", "WARN",
            f"{_CONFIG_FILENAME} is present but not listed in MANIFEST.txt "
            "— its digest cannot be cross-checked against the manifest"), None

    try:
        raw = path.read_bytes()
    except OSError as exc:
        return CheckResult(
            "config-roundtrip", "FAIL",
            f"{_CONFIG_FILENAME} could not be read: {exc}"), None

    digest = hashlib.sha256(_strip_hash_header(raw)).hexdigest()
    if digest != expected:
        return CheckResult(
            "config-roundtrip", "FAIL",
            f"{_CONFIG_FILENAME} body does not reproduce the recorded "
            "config sha256"), digest
    return CheckResult(
        "config-roundtrip", "OK",
        f"{_CONFIG_FILENAME} reproduces the recorded config sha256"), digest


def _check_version_binding(manifest: Manifest,
                           roundtrip_digest: str | None) -> CheckResult:
    """C5: recompute provenance_sha256(version, cfg) twice — (a) with the
    manifest's own recorded config_sha256 (catches an edited version string),
    (b) with C4's freshly recomputed digest (catches an edited config).  This
    is requirement R1: the recorded version cannot be changed without
    invalidating the digest."""
    version = manifest.version or ""
    expected = manifest.fields["provenance_sha256"]
    cfg_a = manifest.fields["config_sha256"]

    computed = {"a": provenance_sha256(version, cfg_a)}
    if roundtrip_digest is not None:
        computed["b"] = provenance_sha256(version, roundtrip_digest)

    mismatched = sorted(label for label, val in computed.items()
                        if val != expected)
    if mismatched:
        return CheckResult(
            "version-binding", "FAIL",
            f"provenance sha256 does not match the recorded digest "
            f"(comparison {', '.join(mismatched)} differs) — the recorded "
            f'version "{version}" is not cryptographically bound to this '
            "output's config")
    if len(computed) < 2:
        return CheckResult(
            "version-binding", "WARN",
            "only comparison (a) could be checked — config-roundtrip did "
            "not produce a digest to cross-check (see the config-roundtrip "
            "check above for why)")
    return CheckResult(
        "version-binding", "OK",
        f'provenance sha256 matches — "{version}" is bound to this output')


def _version_from_config_header(path: Path) -> str | None:
    try:
        first = path.read_text(encoding="utf-8").split("\n", 1)[0]
    except OSError:
        return None
    if not first.startswith("# grainsmith "):
        return None
    tok0 = first[2:].split(" | ")[0].split()
    return tok0[1] if len(tok0) >= 2 else None


def _version_from_summary_csv(path: Path) -> tuple[str | None, str | None]:
    """Returns (version, error) — csv module, "section,key,value" columns
    (§5.2-C6). A present-but-unparseable file is a FAIL, never swallowed."""
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                if (row.get("section") == "meta"
                        and row.get("key") == "grainsmith_version"):
                    return row.get("value"), None
    except (OSError, csv.Error) as exc:
        return None, f"could not be parsed ({exc})"
    return None, "has no meta,grainsmith_version row"


def _version_from_microstructure_json(path: Path) -> tuple[str | None, str | None]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"could not be parsed ({exc})"
    version = doc.get("provenance", {}).get("grainsmith_version")
    if version is None:
        return None, "has no provenance.grainsmith_version key"
    return version, None


def _check_version_consistency(outdir: Path, manifest: Manifest) -> CheckResult:
    """C6: MANIFEST.txt, resolved_config.yaml, summary.csv and
    microstructure.json each carry their own copy of the recorded version;
    every source that is actually present must agree."""
    sources: list[tuple[str, str]] = []
    failures: list[str] = []

    if manifest.version is not None:
        sources.append(("MANIFEST.txt", manifest.version))
    else:
        failures.append("MANIFEST.txt: header line 1 carries no version token")

    cfg_path = outdir / _CONFIG_FILENAME
    if cfg_path.is_file():
        v = _version_from_config_header(cfg_path)
        if v is not None:
            sources.append((_CONFIG_FILENAME, v))
        else:
            failures.append(
                f"{_CONFIG_FILENAME}: header line 1 is not a grainsmith "
                "provenance line")

    summary_path = outdir / _SUMMARY_FILENAME
    if summary_path.is_file():
        v, err = _version_from_summary_csv(summary_path)
        if err is not None:
            failures.append(f"{_SUMMARY_FILENAME}: {err}")
        else:
            sources.append((_SUMMARY_FILENAME, v))  # type: ignore[arg-type]

    micro_path = outdir / _MICROSTRUCTURE_FILENAME
    if micro_path.is_file():
        v, err = _version_from_microstructure_json(micro_path)
        if err is not None:
            failures.append(f"{_MICROSTRUCTURE_FILENAME}: {err}")
        else:
            sources.append((_MICROSTRUCTURE_FILENAME, v))  # type: ignore[arg-type]

    if failures:
        return CheckResult(
            "version-consistency", "FAIL",
            "could not determine the recorded version from every present "
            "source", sorted(failures))

    versions = {v for _, v in sources}
    names = ", ".join(name for name, _ in sources)
    if len(versions) > 1:
        detail = sorted(f"{name}: {v}" for name, v in sources)
        return CheckResult(
            "version-consistency", "FAIL",
            "recorded grainsmith version disagrees between sources", detail)
    if len(sources) < 2:
        return CheckResult(
            "version-consistency", "WARN",
            f"only one source present ({names}) — nothing to cross-check")
    version = next(iter(versions))
    return CheckResult("version-consistency", "OK", f"{version} in {names}")


def _check_expected_version(manifest: Manifest, expect_version: str | None,
                            running_version: str) -> CheckResult:
    """C7: verifying an archived run with a newer grainsmith is legitimate,
    so a bare version mismatch is never a failure — only an explicit
    ``--expect-version`` turns it into one."""
    recorded = manifest.version or "?"
    if expect_version is None:
        cmp = "==" if recorded == running_version else "!="
        return CheckResult(
            "expected-version", "SKIP",
            f"no --expect-version given (recorded {recorded} {cmp} "
            f"running {running_version})")
    if recorded == expect_version:
        return CheckResult(
            "expected-version", "OK",
            f"recorded version {recorded} matches --expect-version "
            f"{expect_version}")
    return CheckResult(
        "expected-version", "FAIL",
        f"recorded version {recorded} != --expect-version {expect_version}")


# ---------------------------------------------------------------------------
# Report + rendering (§5.3, §5.5)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VerifyReport:
    """The result of :func:`verify_outdir`."""
    outdir: Path
    manifest: Manifest
    checks: list[CheckResult]
    running_version: str
    strict: bool = False

    @property
    def n_ok(self) -> int:
        return sum(1 for c in self.checks if c.status == "OK")

    @property
    def n_warn(self) -> int:
        return sum(1 for c in self.checks if c.status == "WARN")

    @property
    def n_fail(self) -> int:
        return sum(1 for c in self.checks if c.status == "FAIL")

    @property
    def exit_code(self) -> int:
        # `--strict` treats ALL warnings as failures (cli.py's own help text:
        # "Treat warnings as failures") -- applied uniformly here at the
        # report level rather than per-check, so every WARN-capable check
        # (extra-files C3, version-binding C5, version-consistency C6) is
        # promoted the same way, not just the one that happens to fire most.
        return 1 if self.n_fail or (self.strict and self.n_warn) else 0


def verify_outdir(outdir: Path, *, strict: bool = False,
                  expect_version: str | None = None) -> VerifyReport:
    """Verify a grainsmith output directory from its shipped files alone."""
    outdir = Path(outdir)
    manifest = parse_manifest(outdir / "MANIFEST.txt")
    running_version = grainsmith.__version__

    c4, roundtrip_digest = _check_config_roundtrip(outdir, manifest)
    checks = [
        _check_manifest(manifest),
        _check_file_digests(outdir, manifest),
        _check_extra_files(outdir, manifest),
        c4,
        _check_version_binding(manifest, roundtrip_digest),
        _check_version_consistency(outdir, manifest),
        _check_expected_version(manifest, expect_version, running_version),
    ]
    return VerifyReport(outdir=outdir, manifest=manifest, checks=checks,
                        running_version=running_version, strict=strict)


_LABEL_WIDTH = 19


def render_report(report: VerifyReport, quiet: bool = False) -> str:
    """Render *report* to the exact §5.3 text layout.

    Deterministic and diffable between two verify runs of the same
    directory: no wall-clock, no elapsed time, no absolute paths other than
    ``report.outdir`` itself.
    """
    word = "VERIFIED" if report.exit_code == 0 else "FAILED"
    plural = "s" if report.n_warn != 1 else ""
    # A failing exit code has two distinct causes that must read differently:
    # a real check FAIL, or (only under --strict) warnings promoted to a
    # failing exit code with zero actual FAILs. Without this branch the
    # promoted-only case renders "FAILED: ... 0 failed", which is a
    # self-contradiction — the exit code says failure but the count says
    # none did.
    if report.exit_code != 0 and report.n_fail == 0:
        tail = f"{report.n_warn} warning{plural} treated as failure (--strict)"
    else:
        tail = f"{report.n_warn} warning{plural}, {report.n_fail} failed"
    final_line = (f"{word}: {report.outdir} — {report.n_ok} checks OK, "
                 f"{tail}")
    if quiet:
        return final_line

    m = report.manifest
    lines = [
        f"grainsmith verify: {report.outdir}",
        f"  {'recorded version':<{_LABEL_WIDTH}}: {m.version}",
        f"  {'running version':<{_LABEL_WIDTH}}: {report.running_version}",
        f"  {'timestamp (UTC)':<{_LABEL_WIDTH}}: {m.timestamp_iso}",
        f"  {'seed':<{_LABEL_WIDTH}}: {m.seed}",
        f"  {'config sha256':<{_LABEL_WIDTH}}: "
        f"{m.fields.get('config_sha256', '')}",
        f"  {'provenance sha256':<{_LABEL_WIDTH}}: "
        f"{m.fields.get('provenance_sha256', '')}",
        f"  {'payload schema':<{_LABEL_WIDTH}}: "
        f"{m.fields.get('provenance_payload_schema', '')}",
        "",
    ]
    for c in report.checks:
        tag = f"[{c.status}]".ljust(6)
        lines.append(f"{tag} {c.name:<{_LABEL_WIDTH}}: {c.summary}")
        for d in sorted(c.details):
            lines.append(" " * 9 + d)
    lines.append("")
    lines.append(final_line)
    return "\n".join(lines)
