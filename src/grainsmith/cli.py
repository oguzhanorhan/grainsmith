"""grainsmith command-line interface.

Subcommands
-----------
grainsmith generate <config.yaml>
    Run the full generation pipeline.

grainsmith validate <config.yaml>
    Run gate G1 (schema + cross-field validation) and a crystal dry build
    (gate G2).  Prints "OK" or an error description.

grainsmith verify <outdir> [--strict] [--expect-version X.Y.Z] [--quiet]
    Verify an output directory against its own MANIFEST.txt (§5, R2):
    recompute every digest, re-derive the config hash from the shipped
    resolved_config.yaml, and check that the recorded grainsmith version is
    the one cryptographically bound to the run.  Exit 0/1/2 — see
    grainsmith.verify.

grainsmith info --sg <number> [--setting <s>] [--site "x,y,z"]
    Print space-group information (number, international/Hall symbols,
    point group, Wyckoff orbit expansion).

grainsmith --version
    Print the grainsmith version and exit 0.
"""
from __future__ import annotations

import argparse
import sys

# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def _cmd_generate(args: argparse.Namespace) -> int:
    """Load config and run the full pipeline."""
    from pathlib import Path

    from grainsmith.config.resolve import load_config
    from grainsmith.errors import ConfigError, GrainsmithError
    from grainsmith.pipeline import run

    try:
        config = load_config(Path(args.config))
    except (ConfigError, OSError) as exc:
        # OSError covers FileNotFoundError, IsADirectoryError (a directory
        # passed as the config path), PermissionError, etc. — surface them as
        # the CLI's clean one-line diagnostic rather than a raw traceback.
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    try:
        result = run(config, jobs=args.jobs, max_rss_gb=args.max_rss)
    except GrainsmithError as exc:
        print(f"Generation failed: {exc}", file=sys.stderr)
        return 1

    gate_results = result.gates.results()
    n_gates = len(gate_results)
    n_passed = sum(1 for r in gate_results if r.passed)
    failed = [r.gate for r in gate_results if not r.passed]
    status = "OK" if not failed else "WARN"
    note = "" if not failed else f" [WARN gates: {', '.join(failed)}]"
    print(
        f"{status}: {len(result.atoms)} atoms, {result.tess.n_grains} grains, "
        f"{len(result.boundary_reports)} boundaries; "
        f"{n_passed}/{n_gates} QA gates passed{note}; "
        f"outputs in {result.outdir} ({result.timings['total']:.2f} s)."
    )

    # Total-RSS monitor feedback: report the observed peak; note if
    # --max-rss was requested but psutil is unavailable (monitor stays inert).
    if result.peak_rss_bytes is not None:
        print(f"Peak driver RSS: {result.peak_rss_bytes / 1e9:.2f} GB.")
    elif args.max_rss is not None:
        print("Note: --max-rss requested but the 'psutil' extra is not "
              "installed; total-RSS monitoring was skipped "
              "(pip install \".[monitor]\").", file=sys.stderr)
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    """Validate a config file: gate G1 (schema) + G2 (crystal)."""
    from pathlib import Path

    from grainsmith.config.resolve import load_config
    from grainsmith.errors import ConfigError, CrystalError
    from grainsmith.pipeline import _validate_crystal

    # --- Gate G1: schema + cross-field validation ---
    try:
        config = load_config(Path(args.config))
    except OSError as exc:
        # FileNotFoundError, IsADirectoryError, PermissionError, ...
        print(f"G1 FAIL: {exc}", file=sys.stderr)
        return 1
    except ConfigError as exc:
        print(f"G1 FAIL: {exc}", file=sys.stderr)
        return 1

    print("G1 OK: schema validation passed.")

    # --- Gate G2: crystal dry build (one per phase for multiphase configs) ---
    try:
        g2_results = _validate_crystal(config)
        for label, ds in g2_results:
            tag = "G2 OK" if len(g2_results) == 1 else f"G2 OK [{label}]"
            print(
                f"{tag}: SG {ds['number']} {ds['international']} "
                f"| point group {ds['pointgroup']} "
                f"| hall {ds['hall_number']} "
                f"| {len(ds['wyckoffs'])} sites expanded."
            )
    except (ConfigError, CrystalError) as exc:
        print(f"G2 FAIL: {exc}", file=sys.stderr)
        return 1

    print("OK")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    """Verify an output directory against its own MANIFEST.txt."""
    from pathlib import Path

    from grainsmith.errors import GrainsmithError
    from grainsmith.verify import render_report, verify_outdir

    outdir = Path(args.outdir)
    if not outdir.is_dir():
        print(f"grainsmith verify: not a directory: {outdir}", file=sys.stderr)
        return 2
    try:
        report = verify_outdir(outdir, strict=args.strict,
                               expect_version=args.expect_version)
    except GrainsmithError as exc:
        print(f"grainsmith verify: cannot verify {outdir}: {exc}",
              file=sys.stderr)
        return 2

    print(render_report(report, quiet=args.quiet))
    if report.exit_code != 0:
        # Mirror render_report's own FAILED-vs-warnings-promoted distinction
        # (verify.py) here: under --strict with zero actual FAILs, "0
        # check(s) failed" while exiting 1 would be self-contradictory.
        if report.n_fail:
            print(f"grainsmith verify: {report.n_fail} check(s) failed in "
                  f"{outdir}", file=sys.stderr)
        else:
            print(f"grainsmith verify: {report.n_warn} warning(s) treated "
                  f"as failure under --strict in {outdir}", file=sys.stderr)
    return report.exit_code


def _cmd_ui(args: argparse.Namespace) -> int:
    """Report that the interactive UI is not part of this release.

    ``grainsmith.ui`` exists in the source tree but is not wired up here —
    this release ships CLI-only. No import is attempted; the subcommand stays
    registered purely so ``grainsmith ui`` gives a clear message instead of
    "unknown command".
    """
    print(
        "The interactive grainsmith UI is not included in this release; "
        "it is planned for a future version.",
        file=sys.stderr,
    )
    return 1


def _cmd_info(args: argparse.Namespace) -> int:
    """Print space-group / Wyckoff information."""
    import spglib

    from grainsmith.crystal import (
        WyckoffSite,
        expand_wyckoff,
        hall_from_international,
        symmetry_ops,
    )
    from grainsmith.errors import ConfigError, CrystalError

    sg_number: int = args.sg
    setting: str | None = args.setting

    try:
        hall = hall_from_international(sg_number, setting)
    except ConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    rots, trans = symmetry_ops(hall)
    sg_info = spglib.get_spacegroup_type(hall)

    # spglib ≥ 2.x returns a SpaceGroupType object (attribute access;
    # dict-style access is deprecated); older versions return a dict.
    def _info(name: str, default: str = "?") -> str:
        if isinstance(sg_info, dict):
            return sg_info.get(name, default)
        return getattr(sg_info, name, default)

    international = _info("international_short", _info("international"))
    pointgroup = _info("pointgroup_international")

    print(f"Space group : {sg_number}  ({international})")
    print(f"Hall number : {hall}")
    print(f"Point group : {pointgroup}")
    print(f"Symmetry ops: {len(rots)}")

    if args.site is not None:
        try:
            coords = [float(x.strip()) for x in args.site.split(",")]
            if len(coords) != 3:
                raise ValueError("Expected exactly 3 comma-separated values.")
        except ValueError as exc:
            print(f"Error parsing --site: {exc}", file=sys.stderr)
            return 1

        sites = [WyckoffSite(element="X", coords=coords)]
        try:
            basis = expand_wyckoff(sites, rots, trans)
        except CrystalError as exc:
            print(f"Error expanding site: {exc}", file=sys.stderr)
            return 1

        print(f"\nOrbit of {args.site} ({basis.atoms_per_cell} positions):")
        for i, pos in enumerate(basis.frac):
            print(f"  [{i+1:3d}]  {pos[0]:.6f}  {pos[1]:.6f}  {pos[2]:.6f}")

    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grainsmith",
        description=(
            "A Generator of Polycrystalline Models for Atomistic Simulations "
            "with Statistical and Grain-Boundary Morphology Control"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    from grainsmith import __version__

    parser.add_argument(
        "--version",
        action="version",
        version=f"grainsmith {__version__}",
        help="Print the grainsmith version and exit.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # -- generate --
    p_gen = sub.add_parser(
        "generate",
        help="Generate a polycrystal from a YAML config file.",
    )
    p_gen.add_argument(
        "config",
        metavar="config.yaml",
        help="Path to the YAML configuration file.",
    )
    p_gen.add_argument(
        "--jobs",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Worker processes for the per-grain fill stage, the "
            "GB-curvature analysis stage, and the per-boundary-pair "
            "analysis stage (§13), and the thread count for the "
            "overlap-removal and G7 neighbour search. 1 = serial "
            "(default), 0 = all cores available to this process (the "
            "cgroup/affinity mask — e.g. a SLURM --cpus-per-task "
            "allocation), not necessarily the whole machine. Outputs are "
            "bit-identical for every value."
        ),
    )
    p_gen.add_argument(
        "--max-rss",
        type=float,
        default=None,
        metavar="GB",
        help=(
            "Soft total-RSS WARN ceiling in GB for the best-effort memory "
            "monitor (needs the optional 'psutil' extra). WARN only — never "
            "aborts and never changes outputs. Default: auto-ceiling at a "
            "fraction of physical RAM."
        ),
    )

    # -- validate --
    p_val = sub.add_parser(
        "validate",
        help=(
            "Validate a config file. "
            "Runs gate G1 (schema + cross-field) and G2 (crystal dry build)."
        ),
    )
    p_val.add_argument(
        "config",
        metavar="config.yaml",
        help="Path to the YAML configuration file.",
    )

    # -- verify --
    p_ver = sub.add_parser(
        "verify",
        help=("Verify an output directory: recompute every MANIFEST.txt "
              "digest, re-derive the config hash from the shipped "
              "resolved_config.yaml, and check that the recorded grainsmith "
              "version is the one cryptographically bound to the run."),
    )
    p_ver.add_argument(
        "outdir",
        metavar="OUTDIR",
        help="Output directory produced by `grainsmith generate`.",
    )
    p_ver.add_argument(
        "--strict",
        action="store_true",
        help=("Treat warnings as failures — notably files present in OUTDIR "
              "that MANIFEST.txt does not list (stale artefacts from an "
              "earlier run into the same directory)."),
    )
    p_ver.add_argument(
        "--expect-version",
        default=None,
        metavar="X.Y.Z",
        help=("Fail unless the recorded grainsmith version equals this "
              "string. Without it, a recorded-vs-running version difference "
              "is reported as a note, not a failure (verifying an archived "
              "run with a newer grainsmith is legitimate)."),
    )
    p_ver.add_argument(
        "--quiet",
        action="store_true",
        help="Print only the final VERIFIED/FAILED line.",
    )

    # -- ui --
    # Not included in this release (see _cmd_ui). Kept registered so `grainsmith
    # ui [...]` reports that clearly instead of "unknown command"; any trailing
    # tokens are still collected as "unknown" args in main() so a flag-like
    # passthrough (e.g. `--port 9999`) never trips argparse's own error path.
    sub.add_parser(
        "ui",
        help=(
            "Interactive UI (not included in this release; planned for a "
            "future version)."
        ),
    )

    # -- info --
    p_info = sub.add_parser(
        "info",
        help="Print space-group and Wyckoff information (P1+).",
    )
    p_info.add_argument(
        "--sg",
        type=int,
        required=True,
        metavar="NUMBER",
        help="International space-group number (1–230).",
    )
    p_info.add_argument(
        "--setting",
        default=None,
        metavar="SETTING",
        help="Optional setting string (e.g. '2', 'H', 'R').",
    )
    p_info.add_argument(
        "--site",
        default=None,
        metavar="x,y,z",
        help="Representative fractional coordinate to expand (e.g. '0.0,0.0,0.0').",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Console-script entry point (``grainsmith = "grainsmith.cli:main"``)."""
    parser = _build_parser()
    args, unknown = parser.parse_known_args()
    if args.command == "ui":
        # Not included in this release (_cmd_ui prints a message and returns
        # 1); swallow any trailing tokens here so `grainsmith ui --anything`
        # doesn't fail argument parsing before that message is reached.
        args.ui_args = unknown
    elif unknown:
        parser.error("unrecognized arguments: " + " ".join(unknown))

    dispatch = {
        "generate": _cmd_generate,
        "validate": _cmd_validate,
        "verify": _cmd_verify,
        "ui": _cmd_ui,
        "info": _cmd_info,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    sys.exit(handler(args))


if __name__ == "__main__":
    main()
