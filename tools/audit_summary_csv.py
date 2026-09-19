"""Report what an ARCHIVED output directory's summary.csv is missing.

Run ``python -m tools.audit_summary_csv DIRECTORY ...``.

READ-ONLY BY DESIGN. This tool opens files and prints a report. It writes
nothing, creates no backups, and never touches MANIFEST.txt.

Why it is read-only
-------------------
Its predecessor rewrote ``summary.csv`` in place, re-signed that file's
MANIFEST.txt entry, and dropped ``*.before-summary-completion`` sidecars next
to the outputs. That had three consequences worth stating, because they are
the reason this rewrite exists:

1. ``grainsmith verify --strict`` FAILED on every directory it touched -- not
   on a digest, but on ``extra-files``: the un-manifested sidecars.
2. Re-signing the manifest entry made ``grainsmith verify`` report VERIFIED
   for a ``summary.csv`` that no grainsmith run ever wrote. A provenance chain
   whose signature an out-of-band script can re-issue is not a provenance
   chain.
3. It only ever verified 3 of the ~17 manifest entries before rewriting, and
   specifically NOT ``run.log`` -- the one input it scraped new facts from.

Against output from the current pipeline the predecessor could add exactly two
rows (``memory,measurement_source`` and ``meta,summary_completion_source``),
and both are assertions about recovery from an archived log that are false of
a live run. The pipeline's own ``_summary_rows`` is a strict superset
otherwise, so there is nothing left to "complete" -- only something to report.

Scope of the report
-------------------
This compares the archive's ``summary.csv`` against the rows that can be
DERIVED from files sitting in the same directory (``resolved_config.yaml``,
``statistics.csv``, ``run.log``). That is a LOWER BOUND on what a current-code
run would have written: rows computed from the atoms themselves (the
``crystal`` lattice block, ``atoms``, ``composition``, per-gate measurements)
cannot be reconstructed from an archive at all. To get a genuinely complete
summary, re-run the config -- the report says so when it finds gaps.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from grainsmith.io.reports import SUMMARY_COLUMNS, summary_mapping_rows


def _read_rows(path: Path) -> dict[tuple[str, str], Any]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != SUMMARY_COLUMNS:
            raise ValueError(f"Unexpected CSV columns in {path}")
        rows: dict[tuple[str, str], Any] = {}
        duplicates: list[tuple[str, str]] = []
        for row in reader:
            key = row["section"], row["key"]
            if key in rows:
                duplicates.append(key)
            rows[key] = row["value"]
    if duplicates:
        # Reported, not raised: a duplicate key is exactly the kind of defect
        # this tool exists to surface, and refusing to read would hide the
        # rest of the report.
        rows["__duplicates__", ""] = duplicates  # type: ignore[assignment]
    return rows


def _log_rows(text: str) -> dict[tuple[str, str], Any]:
    """Rows recoverable from an archived run.log.

    Last-match-wins on every pattern: a log holding several concatenated runs
    yields the LAST one's numbers, which is why nothing here may be trusted
    for anything but a human-read report.
    """
    rows: dict[tuple[str, str], Any] = {}
    timing_records = re.findall(r"stage timings \(s\): ([^\n]+)", text)
    if timing_records:
        for entry in timing_records[-1].split(","):
            stage, separator, value = entry.strip().partition("=")
            if not separator:
                continue
            try:
                parsed = float(value)
            except ValueError:
                continue
            if math.isfinite(parsed) and parsed >= 0.0:
                rows["timings", f"{stage}_s"] = value
    limits = re.findall(
        r"Memory guard [^\n]*?: ([0-9.eE+-]+) GB per allocation([^\n]*)", text)
    if limits:
        amount, description = limits[-1]
        rows["memory", "allocation_limit_effective_gb"] = amount
        rows["memory", "allocation_limit_source"] = (
            "ram" if "clamped" in description else "config")
    peaks = re.findall(
        r"(ok|WARN): peak driver RSS ([0-9.eE+-]+) GB vs "
        r"(system-RAM auto-ceiling|--max-rss) ([0-9.eE+-]+) GB", text)
    if peaks:
        status, peak, source, ceiling = peaks[-1]
        rows["memory", "monitor_available"] = True
        rows["memory", "peak_driver_rss_gb"] = peak
        rows["memory", "rss_warn_limit_gb"] = ceiling
        rows["memory", "rss_warn_limit_source"] = (
            "system_ram_auto" if source.startswith("system") else source)
        rows["memory", "rss_warn_tripped"] = status == "WARN"
    return rows


@dataclass
class AuditReport:
    """What one archived directory is missing, and how trustworthy it is."""

    directory: Path
    n_present: int = 0
    missing: list[tuple[str, str]] = field(default_factory=list)
    duplicates: list[tuple[str, str]] = field(default_factory=list)
    integrity: list[str] = field(default_factory=list)
    sidecars: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not (self.missing or self.duplicates
                    or self.integrity or self.sidecars)


def audit(directory: Path) -> AuditReport:
    """Inspect one output directory. Opens files; writes none."""
    directory = Path(directory)
    report = AuditReport(directory=directory)

    config_path = directory / "resolved_config.yaml"
    if not config_path.is_file():
        # A report tool should say what is wrong, not traceback.
        report.integrity.append(
            "resolved_config.yaml: absent — not a grainsmith output directory "
            "(or the run never reached the write stage)")
        return report
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    summary_name = (config.get("output", {}).get("csv", {})
                    .get("summary", "summary.csv"))
    summary_path = directory / summary_name

    # Un-manifested leftovers -- the reason `verify --strict` fails on the
    # directories the predecessor tool touched.
    report.sidecars = sorted(
        p.name for p in directory.iterdir()
        if p.is_file() and p.name.endswith(".before-summary-completion"))

    # Integrity: check EVERY hashed manifest entry, not a sample of three.
    manifest_path = directory / "MANIFEST.txt"
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            fields = line.split(maxsplit=2)
            if len(fields) != 3 or fields[0].startswith("#"):
                continue
            digest, size, name = fields
            if set(digest) == {"-"}:
                # Dashed entry = deliberately unhashed, because the file is
                # still being written when the manifest is built (run.log).
                # verify.py treats these as sha256=None; so must this.
                continue
            target = directory / name
            if not target.is_file():
                report.integrity.append(f"{name}: listed but missing")
                continue
            recorded = target.read_bytes()
            if (digest != hashlib.sha256(recorded).hexdigest()
                    or size != str(len(recorded))):
                report.integrity.append(f"{name}: digest/size mismatch")
    else:
        report.integrity.append("MANIFEST.txt: absent")

    present = _read_rows(summary_path)
    report.duplicates = list(present.pop(("__duplicates__", ""), []) or [])
    report.n_present = len(present)

    expected: dict[tuple[str, str], Any] = {}
    for section, key, value in summary_mapping_rows("config", config):
        expected[section, key] = value
    stats_path = directory / "statistics.csv"
    if stats_path.exists():
        for (section, key), value in _read_rows(stats_path).items():
            if section == "__duplicates__":
                continue
            expected[f"statistics:{section}", key] = value
    log_path = directory / "run.log"
    if log_path.exists():
        expected.update(_log_rows(log_path.read_text(encoding="utf-8")))
    grids = re.findall(r"Voxel grid \((\d+), (\d+), (\d+)\):",
                       log_path.read_text(encoding="utf-8")
                       if log_path.exists() else "")
    if grids:
        shape = tuple(int(v) for v in grids[-1])
        expected["analysis", "voxel_grid_shape"] = " ".join(map(str, shape))
        expected["analysis", "n_voxels"] = math.prod(shape)

    report.missing = sorted(k for k in expected if k not in present)
    return report


def render(report: AuditReport) -> str:
    lines = [f"{report.directory}: {report.n_present} summary rows"]
    if report.sidecars:
        lines.append(
            f"  [FAIL] {len(report.sidecars)} un-manifested sidecar file(s) — "
            f"these make `grainsmith verify --strict` fail on this directory:")
        lines += [f"           {name}" for name in report.sidecars]
    if report.integrity:
        lines.append(f"  [FAIL] {len(report.integrity)} manifest problem(s):")
        lines += [f"           {item}" for item in report.integrity]
    if report.duplicates:
        lines.append(
            f"  [FAIL] {len(report.duplicates)} duplicate (section,key) row(s) — "
            f"a reader keying by (section,key) silently keeps only the last:")
        lines += [f"           {s},{k}" for s, k in report.duplicates]
    if report.missing:
        lines.append(
            f"  [WARN] {len(report.missing)} derivable row(s) absent "
            f"(lower bound — rows computed from the atoms cannot be derived "
            f"from an archive at all):")
        lines += [f"           {s},{k}" for s, k in report.missing]
        lines.append("  -> re-run the config to get a genuinely complete "
                     "summary.csv; this tool will not forge one.")
    if report.clean:
        lines.append("  [OK]   nothing missing that this archive can prove")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    args = parser.parse_args(argv)
    worst = 0
    for directory in args.directories:
        report = audit(directory)
        print(render(report))
        if not report.clean:
            worst = 1
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
