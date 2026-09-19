"""Generate the architecture import matrix and figure from Python ASTs.

Run ``python -m tools.gen_module_graph`` after changing imports, then rebuild
the HTML docs. ``--check`` verifies the table without requiring Matplotlib.
Lazy and TYPE_CHECKING imports are included; the optional UI is excluded.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections import Counter
from importlib.util import resolve_name
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "grainsmith"
DOCUMENT = ROOT / "docs" / "architecture.md"
FIGURE = ROOT / "docs" / "module_graph.png"
START = "<!-- module-dependencies:start -->"
END = "<!-- module-dependencies:end -->"
LAYERS = ("foundation", "crystal", "orientation", "tessellation", "atoms",
          "analysis", "io", "config", "orchestration")


def module_edges(source: Path = SOURCE) -> set[tuple[str, str]]:
    """Distinct declared module imports, including relative module imports."""
    modules = {}
    for path in sorted(source.rglob("*.py")):
        name = ".".join(path.relative_to(source.parent).with_suffix("").parts)
        modules[name.removesuffix(".__init__")] = path
    edges = set()
    for name, path in modules.items():
        package = name if path.name == "__init__.py" else name.rpartition(".")[0]
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    base = resolve_name("." * node.level + base, package)
                targets = [f"{base}.{alias.name}" if f"{base}.{alias.name}" in modules
                           else base for alias in node.names]
            edges.update((name, target) for target in targets if target in modules)
    return edges


def layer(module: str) -> str | None:
    parts = module.split(".")
    name = parts[1] if len(parts) > 1 else "__init__"
    if name == "ui":
        return None
    if name in LAYERS:
        return name
    if name in {"pipeline", "cli", "qa", "verify"}:
        return "orchestration"
    if len(parts) <= 2:
        return "foundation"
    raise ValueError(f"Unclassified scientific subpackage: {module}")


def dependency_counts(edges: set[tuple[str, str]]) -> Counter:
    counts: Counter = Counter()
    for importer, target in edges:
        owner, dependency = layer(importer), layer(target)
        if owner is not None and dependency is not None and owner != dependency:
            counts[owner, dependency] += 1
    return counts


def render_table(edges: set[tuple[str, str]] | None = None) -> str:
    counts = dependency_counts(module_edges() if edges is None else edges)
    lines = ["| imports from -> | " + " | ".join(LAYERS) + " |",
             "|" + "---|" * (len(LAYERS) + 1)]
    for owner in LAYERS:
        cells = [str(counts[owner, target]) if counts[owner, target] else "-"
                 for target in LAYERS]
        lines.append(f"| **{owner}** | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def dependency_digest(edges: set[tuple[str, str]]) -> str:
    payload = json.dumps(sorted(edges), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def render_figure(edges: set[tuple[str, str]], path: Path = FIGURE) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = dependency_counts(edges)
    values = [[counts[owner, target] for target in LAYERS] for owner in LAYERS]
    maximum = max(max(row) for row in values)
    figure, axes = plt.subplots(figsize=(11.5, 8.5), layout="constrained")
    axes.imshow(values, cmap="Blues", vmin=0, vmax=max(maximum, 1))
    axes.set_xticks(range(len(LAYERS)), LAYERS, rotation=35, ha="left")
    axes.set_yticks(range(len(LAYERS)), LAYERS)
    axes.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    axes.set_ylabel("Importing layer", labelpad=18)
    axes.set_xlabel("Imported layer", labelpad=12)
    axes.xaxis.set_label_position("top")
    axes.set_title("grainsmith module dependencies", loc="left", pad=95,
                   fontsize=18, fontweight="bold")
    for row, owner in enumerate(LAYERS):
        for column, target in enumerate(LAYERS):
            count = counts[owner, target]
            axes.text(column, row, str(count) if count else "-", ha="center",
                      va="center", fontsize=12,
                      color="white" if count > maximum / 2 else "#243746")
    figure.text(0.5, -0.025,
                "Distinct module imports, including lazy/type-only imports. "
                "Same-layer imports and optional UI omitted.",
                ha="center", va="top", fontsize=9)
    figure.savefig(path, dpi=180, bbox_inches="tight",
                   metadata={"DependencyDigest": dependency_digest(edges)})
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    edges = module_edges()
    text = DOCUMENT.read_text(encoding="utf-8")
    before, start, remainder = text.partition(START)
    current, end, after = remainder.partition(END)
    if not start or not end:
        raise ValueError("Architecture dependency table markers are missing.")
    generated = "\n" + render_table(edges) + "\n"
    if args.check:
        return int(current != generated)
    render_figure(edges)
    DOCUMENT.write_text(before + start + generated + end + after, encoding="utf-8")
    print(f"wrote {DOCUMENT} and {FIGURE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())