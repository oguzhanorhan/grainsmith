#!/usr/bin/env python3
"""Run one family of grainsmith examples and check what it produced.

The script has two halves:

  1. RUN     — you pick a family (menu, or as an argument) and every
               example in it is run with a plain
               ``grainsmith generate <config> --jobs N`` call.
  2. REPORT  — for each example: did the run write the outputs its YAML
               declares, did every QA gate pass, and does the measured
               atom count match the one claimed in examples/README.md.

Nothing here reimplements grainsmith. Every run goes through the CLI and
every number in the report is read back from that run's own summary.csv,
so the report cannot disagree with the engine.

    python examples/run_examples.py                 # menu
    python examples/run_examples.py basics          # one family, --jobs 10
    python examples/run_examples.py texture -j 6    # six workers
    python examples/run_examples.py --list          # what families exist
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parent
README = EXAMPLES / "README.md"

# An example whose DECLARED atom count is at or above this is validated
# (schema + crystal, gate G1/G2) but never generated: at 10^7 atoms a run
# needs hardware you have sized for it on purpose.  --force overrides.
VALIDATE_ONLY_ATOMS = 1e7
# Generating is still legal above these, but it is no longer a coffee-break
# job, so the family asks before it starts: one file this big, or this much
# in the family as a whole.
CONFIRM_FILE = 2e6
CONFIRM_TOTAL = 5e6

SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
# A README table cell that is ONLY an atom count, e.g. "2.7×10⁵" or
# "5.3×10⁷ (not generated)".  Numbers inside prose are deliberately not
# matched — only a whole cell counts as a declaration.  The exponent class
# is spelled out because ¹²³ are Latin-1 characters and do not fall in the
# U+2070..U+2079 superscript block the other digits come from.
ATOM_CELL = re.compile(
    r"^~?\*{0,2}([\d.]+)\s*[×x]\s*10([⁰¹²³⁴⁵⁶⁷⁸⁹]+)\*{0,2}"
    r"(?:\s*\((?:not generated|validate-only)\))?$"
)


# --------------------------------------------------------------------------
# 0. what we are running against
# --------------------------------------------------------------------------

def find_cli() -> str:
    exe = shutil.which("grainsmith")
    if exe:
        return exe
    sys.exit(
        "error: the 'grainsmith' command is not on PATH.\n"
        "       activate the environment it is installed in, e.g.\n"
        "         source .venv/bin/activate\n"
        "       or install the package:  pip install -e .[perf]"
    )


def print_banner(cli: str) -> str:
    """Print, and return, the grainsmith version these examples will run on."""
    version = subprocess.run(
        [cli, "--version"], capture_output=True, text=True
    ).stdout.strip() or "unknown"
    try:  # best effort: which source tree the CLI actually imports
        import grainsmith
        where = Path(grainsmith.__file__).parent
        pkg = f"{grainsmith.__version__}  ({where})"
    except Exception:  # not importable from THIS interpreter — the CLI still is
        pkg = "(not importable from this interpreter)"
    print("=" * 74)
    print(f"  {version}")
    print(f"  package        : {pkg}")
    print(f"  examples tree  : {EXAMPLES}")
    print("=" * 74)
    return version


# --------------------------------------------------------------------------
# 1. families and what the README claims about them
# --------------------------------------------------------------------------

def discover_families() -> dict[str, list[Path]]:
    """Group folder -> its YAMLs.  advanced/ holds two tiers, so it splits."""
    fam: dict[str, list[Path]] = {}
    for d in sorted(p for p in EXAMPLES.iterdir() if p.is_dir()):
        if d.name.startswith((".", "_")) or d.name == "assets":
            continue
        files = sorted(d.glob("*.yaml"))
        if not files:
            continue
        if d.name == "advanced":  # adv_* (~10^6) and huge_* (~10^8)
            fam["advanced"] = [f for f in files if not f.name.startswith("huge_")]
            fam["huge"] = [f for f in files if f.name.startswith("huge_")]
        else:
            fam[d.name] = files
    return {k: v for k, v in fam.items() if v}


def declared_atoms() -> dict[str, float]:
    """{example filename: atom count claimed in examples/README.md}."""
    claims: dict[str, float] = {}
    for line in README.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        name = re.match(r"`([^`]+\.yaml)`", cells[0])
        if not name:
            continue
        for cell in reversed(cells[1:]):  # the atom column is on the right
            m = ATOM_CELL.match(cell)
            if m:
                claims[name.group(1)] = float(m.group(1)) * 10 ** int(
                    m.group(2).translate(SUPERSCRIPT))
                break
    return claims


def human(n: float) -> str:
    return f"{n:.1e}".replace("e+0", "e").replace("e+", "e")


# --------------------------------------------------------------------------
# 2. run one example
# --------------------------------------------------------------------------

def load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError:
        sys.exit("error: PyYAML is required (it ships with grainsmith).")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def run_one(cli: str, cfg_path: Path, workdir: Path, jobs: int,
            validate_only: bool) -> dict:
    """Run (or validate) one example.  Returns a result record."""
    workdir.mkdir(parents=True, exist_ok=True)
    cmd = ([cli, "validate", str(cfg_path)] if validate_only else
           [cli, "generate", str(cfg_path), "--jobs", str(jobs)])
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
    return {
        "config": cfg_path,
        "mode": "validate" if validate_only else "generate",
        "returncode": proc.returncode,
        "seconds": time.perf_counter() - t0,
        "tail": (proc.stdout + proc.stderr).strip().splitlines()[-1:],
        "outdir": workdir / Path(load_yaml(cfg_path)["output"]["directory"]).name,
    }


# --------------------------------------------------------------------------
# 3. report: expected outputs, gates, declared vs measured
# --------------------------------------------------------------------------

def read_summary(outdir: Path) -> dict[str, str]:
    """summary.csv is long format: section,key,value -> {'section.key': value}."""
    path = outdir / "summary.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    return {f"{r[0]}.{r[1]}": r[2] for r in rows[1:] if len(r) >= 3}


def expected_outputs(cfg: dict, summary: dict[str, str]) -> list[tuple[str, bool]]:
    """[(filename or glob, required)] — exactly what this YAML declares.

    `required` is False for a file the config asks for but the resolved
    backend cannot write (vertices.csv exists only for the flat/power
    tessellations — the curved, perturbed, voxel and single-crystal
    backends have no polyhedral vertices to write).
    """
    out = cfg.get("output", {}) or {}
    analysis = cfg.get("analysis", {}) or {}
    want: list[tuple[str, bool]] = [
        (out.get("lammps", {}).get("filename", "polycrystal.data"), True),
        ("MANIFEST.txt", True),
        ("resolved_config.yaml", True),
        ("run.log", True),
    ]
    csvs = out.get("csv", {}) or {}
    for key in ("summary", "grains", "boundaries"):
        if csvs.get(key):
            want.append((csvs[key], True))
    if csvs.get("vertices"):
        want.append((csvs["vertices"], summary.get("tessellation.is_flat") == "True"))
    if (out.get("xyz", {}) or {}).get("enabled"):
        want.append((out["xyz"].get("filename", "polycrystal.extxyz"), True))
    if (out.get("gnuplot", {}) or {}).get("enabled"):
        want.append(("view.plt", True))
    if (out.get("mesh", {}) or {}).get("enabled"):
        want.append((f"*.{out['mesh'].get('format', 'ply')}", True))
    if out.get("methods_snippet", True):
        want.append(("METHODS.md", True))
    if analysis.get("statistics", True):
        want += [("statistics.csv", True), ("microstructure.json", True)]
    if analysis.get("gb_curvature"):
        want.append(("gb_curvature.csv", True))
    if analysis.get("section"):
        want.append(("slice_*.csv", True))
    if cfg.get("doping"):
        want += [("doping.csv", True), ("doping_profile.csv", True)]
    phases = [p.get("name") for p in (cfg.get("phases") or []) if p.get("name")]
    if phases:  # per-phase texture files replace the single-phase pair
        for name in phases:
            want += [(f"mdf_{name}.csv", True), (f"odf_mtex_{name}.txt", True)]
    else:
        want += [("mdf.csv", True), ("odf_mtex.txt", True)]
    return want


def check_outputs(outdir: Path, want: list[tuple[str, bool]]) -> tuple[int, int, list[str]]:
    """-> (found, required, missing names)."""
    found = required = 0
    missing = []
    for pattern, is_required in want:
        hit = bool(list(outdir.glob(pattern))) if "*" in pattern else (outdir / pattern).exists()
        if is_required:
            required += 1
            if hit:
                found += 1
            else:
                missing.append(pattern)
    return found, required, missing


def gate_tally(summary: dict[str, str]) -> tuple[int, int, int]:
    passed = warned = failed = 0
    for key, value in summary.items():
        if not key.startswith("gates."):
            continue
        status = value.split("|")[0].strip().upper()
        passed += status == "PASS"
        warned += status == "WARN"
        failed += status == "FAIL"
    return passed, warned, failed


def report(results: list[dict], claims: dict[str, float], version: str) -> int:
    print()
    print("REPORT — declared (README/YAML) vs measured (this run's summary.csv)")
    print("-" * 104)
    print(f"{'example':<42}{'atoms':>12}{'vs README':>11}"
          f"{'gates P/W/F':>14}{'outputs':>10}{'time':>9}")
    print("-" * 104)
    problems = 0
    for r in results:
        name = r["config"].name
        label = f"{r['config'].parent.name}/{name}"
        if len(label) > 41:  # keep the columns aligned for long names
            label = label[:38] + "..."
        if r["mode"] == "validate":
            note = "validate-only (too large to generate here)"
            gates = "G1/G2 ok" if r["returncode"] == 0 else "G1/G2 FAIL"
            print(f"{label:<42}{'—':>12}{'—':>11}{gates:>14}"
                  f"{'—':>10}{r['seconds']:>8.1f}s")
            if r["returncode"] != 0:
                problems += 1
            print(f"{'':<42}{note}")
            continue
        summary = read_summary(r["outdir"])
        cfg = load_yaml(r["config"])
        want = expected_outputs(cfg, summary)
        found, required, missing = check_outputs(r["outdir"], want)
        p, w, f = gate_tally(summary)
        measured = summary.get("atoms.n_final")
        measured_n = int(measured) if measured and measured.isdigit() else None
        claim = claims.get(name)
        if measured_n and claim:
            dev = 100.0 * (measured_n - claim) / claim
            # README figures are rounded to two significant digits, so a few
            # percent is agreement; a real mismatch is an order-of-magnitude
            # or size-of-box error.
            delta = f"{dev:+.0f}%" + ("!" if abs(dev) > 15 else "")
        else:
            delta = "n/a"
        ok = (r["returncode"] == 0 and not missing and f == 0
              and summary.get("meta.status") == "complete"
              and not delta.endswith("!"))
        problems += not ok
        print(f"{label:<42}{(f'{measured_n:,}' if measured_n else '—'):>12}{delta:>11}"
              f"{f'{p}/{w}/{f}':>14}{f'{found}/{required}':>10}{r['seconds']:>8.1f}s")
        if r["returncode"] != 0:
            print(f"{'':<42}FAILED: {' '.join(r['tail']) or 'see run.log'}")
        if missing:
            print(f"{'':<42}MISSING: {', '.join(missing)}")
        if f:
            for key, value in summary.items():
                if key.startswith("gates.") and value.split("|")[0].strip() == "FAIL":
                    print(f"{'':<42}{key.split('.')[1]} FAIL: {value.split('|')[-1].strip()[:70]}")
        if delta.endswith("!"):
            print(f"{'':<42}ATOM COUNT off by {delta[:-1]} — README claims "
                  f"{human(claim)}, run wrote {measured_n:,}")
        run_version = summary.get("meta.grainsmith_version")
        if run_version and run_version not in version:
            print(f"{'':<42}VERSION: run recorded {run_version}, banner said {version}")
    print("-" * 104)
    verdict = "all OK" if problems == 0 else f"{problems} of {len(results)} need attention"
    print(f"{len(results)} example(s) — {verdict}")
    print("atoms = summary.csv atoms,n_final · gates = PASS/WARN/FAIL tally · "
          "outputs = declared files present")
    return 1 if problems else 0


# --------------------------------------------------------------------------
# 4. selection
# --------------------------------------------------------------------------

def family_size(files: list[Path], claims: dict[str, float]) -> tuple[float, float]:
    """(total, largest) DECLARED atom count of a family, from the README."""
    sizes = [claims.get(f.name, 0.0) for f in files]
    return sum(sizes), max(sizes, default=0.0)


def needs_confirmation(files: list[Path], claims: dict[str, float]) -> bool:
    total, biggest = family_size(files, claims)
    generated = [c for f in files if (c := claims.get(f.name, 0.0)) < VALIDATE_ONLY_ATOMS]
    return biggest >= CONFIRM_FILE or sum(generated) >= CONFIRM_TOTAL


def family_line(name: str, files: list[Path], claims: dict[str, float]) -> str:
    total, biggest = family_size(files, claims)
    if biggest >= VALIDATE_ONLY_ATOMS:
        tag = "cluster scale — the largest files are validate-only"
    elif needs_confirmation(files, claims):
        tag = "minutes each, GBs of output — asks to confirm"
    else:
        tag = "laptop, seconds"
    return (f"{name:<20}{len(files):>2} file(s)  ~{human(total):>7} atoms total   {tag}")


def choose_family(fam: dict[str, list[Path]], claims: dict[str, float]) -> str:
    names = list(fam)
    print("\nWhich example family?\n")
    for i, name in enumerate(names, 1):
        print(f"  {i:>2}) {family_line(name, fam[name], claims)}")
    print("   q) quit")
    while True:
        answer = input("\nfamily (number or name): ").strip()
        if answer.lower() in ("q", "quit", "exit", ""):
            sys.exit(0)
        if answer in fam:
            return answer
        if answer.isdigit() and 1 <= int(answer) <= len(names):
            return names[int(answer) - 1]
        print("not a family — enter a number from the list, a name, or q")


def confirm(family: str, files: list[Path], claims: dict[str, float],
            force: bool, assume_yes: bool) -> list[tuple[Path, bool]]:
    """-> [(config, validate_only)], after warning about the heavy ones."""
    plan = []
    for f in files:
        claim = claims.get(f.name, 0.0)
        plan.append((f, claim >= VALIDATE_ONLY_ATOMS and not force))
    generated = [(f, claims.get(f.name, 0.0)) for f, vo in plan if not vo]
    skipped = [(f, claims.get(f.name, 0.0)) for f, vo in plan if vo]
    if skipped:
        print(f"\nvalidate-only (>= {human(VALIDATE_ONLY_ATOMS)} atoms declared; "
              f"pass --force to generate anyway):")
        for f, n in skipped:
            print(f"  {f.name:<44} ~{human(n)} atoms")
    if generated and needs_confirmation(files, claims) and not assume_yes:
        total = sum(n for _, n in generated)
        print(f"\nfamily '{family}' will GENERATE {len(generated)} model(s), "
              f"~{human(total)} atoms in total:")
        for f, n in generated:
            print(f"  {f.name:<44} ~{human(n)} atoms")
        print("  (each writes its own out_* directory; budget disk and RAM.")
        print("   advanced/README.md notes the two weighted-tessellation files")
        print("   want a capped worker count, e.g. --jobs 6)")
        if input("\ntype 'yes' to continue: ").strip().lower() != "yes":
            sys.exit("aborted")
    return plan


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run one family of grainsmith examples and check its outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="with no FAMILY, a menu is shown.")
    ap.add_argument("family", nargs="?", help="e.g. basics, texture, doping")
    ap.add_argument("-j", "--jobs", type=int, default=10, metavar="N",
                    help="worker processes per run (default: 10; 0 = all cores)")
    ap.add_argument("--out", default="example_runs", metavar="DIR",
                    help="where the out_* directories go (default: ./example_runs)")
    ap.add_argument("--list", action="store_true", help="list families and exit")
    ap.add_argument("--force", action="store_true",
                    help="generate even the >= 10^7-atom examples")
    ap.add_argument("--yes", action="store_true", help="skip the size confirmation")
    args = ap.parse_args()

    cli = find_cli()
    version = print_banner(cli)
    fam = discover_families()
    claims = declared_atoms()

    if args.list:
        print()
        for name in fam:
            print("  " + family_line(name, fam[name], claims))
        return 0

    family = args.family or choose_family(fam, claims)
    if family not in fam:
        sys.exit(f"error: unknown family {family!r} — known: {', '.join(fam)}")

    plan = confirm(family, fam[family], claims, args.force, args.yes)
    workdir = Path(args.out).resolve() / family
    print(f"\nrunning {len(plan)} example(s) from '{family}' with --jobs {args.jobs}")
    print(f"outputs -> {workdir}\n")

    results = []
    for i, (cfg_path, validate_only) in enumerate(plan, 1):
        verb = "validate" if validate_only else "generate"
        print(f"[{i}/{len(plan)}] {verb} {cfg_path.name} ... ", end="", flush=True)
        r = run_one(cli, cfg_path, workdir, args.jobs, validate_only)
        print(f"{'ok' if r['returncode'] == 0 else 'FAILED'}  ({r['seconds']:.1f} s)")
        results.append(r)

    return report(results, claims, version)


if __name__ == "__main__":
    sys.exit(main())
