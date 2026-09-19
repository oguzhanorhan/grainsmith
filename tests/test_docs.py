"""Drift-proof documentation pins.

These tests make the docs *un-driftable* from the code:

  1. ``docs/config_reference.md`` is regenerated in memory and must equal the
     committed file (so a schema change without a regen fails CI), every
     RunConfig-reachable field appears, and every table row is well-formed;
  2. every QA-gate id referenced in ``qa.py`` appears in ``docs/gates.md``;
  3. every configurable + fixed output filename appears in ``docs/outputs.md``;
  4. every ``docs/html/*.html`` page is regenerated in memory (via
     ``tools/build_docs_html.render_pages``, pinned to the ``markdown``
     engine — the one that actually produced the committed files, since a
     ``pandoc``-rendered comparison is not a same-engine drift check) and
     must equal the committed file — an edit to ``docs/*.md`` without
     re-running ``python -m tools.build_docs_html`` fails here, the same
     drift-proofing principle as #1 but for the HTML mirror (md is still
     the source of truth). Content-equality, not mtime: ``render_pages``
     was confirmed byte-deterministic under a fixed engine (two consecutive
     ``build()`` calls with the same engine produce an identical output
     directory) before this test was written, so a content mismatch here
     is always a genuine source/artifact drift, never CI clock/filesystem-
     order noise or an engine swap.
"""
from __future__ import annotations

import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

from pydantic import BaseModel

from grainsmith.config.schema import (
    OutputCsvConfig,
    OutputLammpsConfig,
    OutputXyzConfig,
    RunConfig,
)

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"


def _load_generator():
    """Import tools/gen_config_reference.py (not an installed package)."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import tools.gen_config_reference as gen
    return gen


def _all_field_names(root: type[BaseModel]) -> set[str]:
    """Every field name of every BaseModel reachable from *root*."""
    from grainsmith.config.introspect import discover

    names: set[str] = set()
    for model in discover(root):
        names.update(model.model_fields.keys())
    return names


# --- 1. config_reference.md ------------------------------------------------

def test_config_reference_is_not_stale():
    """The committed config_reference.md must equal a fresh generation —
    a schema change without `python -m tools.gen_config_reference` fails here."""
    gen = _load_generator()
    committed = (DOCS / "config_reference.md").read_text(encoding="utf-8")
    assert committed == gen.render(), (
        "docs/config_reference.md is stale; run "
        "`python -m tools.gen_config_reference`."
    )


def test_config_reference_lists_every_field():
    text = (DOCS / "config_reference.md").read_text(encoding="utf-8")
    missing = [f for f in _all_field_names(RunConfig) if f"`{f}`" not in text]
    assert not missing, f"config_reference.md missing fields: {sorted(missing)}"


def test_config_reference_tables_well_formed():
    """Every field row must have exactly 5 columns (6 unescaped pipes)."""
    bad: list[str] = []
    for line in (DOCS / "config_reference.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("| `"):
            unescaped = len(re.findall(r"(?<!\\)\|", line))
            if unescaped != 6:
                bad.append(f"{unescaped} cols: {line[:60]}")
    assert not bad, "malformed table rows:\n" + "\n".join(bad)


# --- 2. gates.md -----------------------------------------------------------

def test_every_gate_id_documented():
    """No hardcoded upper bound on the gate number: an earlier
    ``1 <= n <= 18`` cap silently stopped verifying anything past G18 --
    G19/G20/G21, and then G22/G23/G24, were all added without this test
    ever catching a missing ``docs/gates.md`` section, because a hardcoded
    constant has no way to know a new gate exists. Every ``G<digits>``
    token found in ``qa.py``'s source IS a real gate id (verified: no
    unrelated numeric-after-G token — e.g. a space-group number — appears
    anywhere in the file), so the set of ids to check is derived entirely
    from what the source actually contains, with no number to go stale."""
    qa_src = (REPO / "src" / "grainsmith" / "qa.py").read_text(encoding="utf-8")
    gate_ids = {f"G{m}" for m in re.findall(r"\bG(\d{1,2})\b", qa_src)}
    assert gate_ids, "no gate ids found in qa.py — extraction broke"
    gates_doc = (DOCS / "gates.md").read_text(encoding="utf-8")
    missing = sorted(g for g in gate_ids if g not in gates_doc)
    assert not missing, f"gates.md missing gate ids: {missing}"


# --- 3. outputs.md ---------------------------------------------------------

def test_every_output_filename_documented():
    configurable = {
        OutputLammpsConfig().filename,
        OutputXyzConfig().filename,
        OutputCsvConfig().grains,
        OutputCsvConfig().boundaries,
        OutputCsvConfig().vertices,
        OutputCsvConfig().summary,
    }
    fixed = {
        "statistics.csv", "microstructure.json", "METHODS.md",
        "mdf.csv", "odf_mtex.txt", "MANIFEST.txt", "resolved_config.yaml",
        "view.plt", "gb_curvature.csv", "curvature_hist.plt", "doping.csv",
    }
    outputs_doc = (DOCS / "outputs.md").read_text(encoding="utf-8")
    missing = sorted(f for f in (configurable | fixed) if f not in outputs_doc)
    assert not missing, f"outputs.md missing filenames: {missing}"


# --- 4. docs/html/*.html ----------------------------------------------------

def _load_html_builder():
    """Import tools/build_docs_html.py (not an installed package) — same
    sys.path pattern _load_generator() above uses for gen_config_reference."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import tools.build_docs_html as bdh
    return bdh


def test_runresult_contract_has_every_field():
    from dataclasses import fields
    from grainsmith.pipeline import RunResult

    contract = (DOCS / "developer.md").read_text(encoding="utf-8")
    contract = contract.split("### `RunResult`", 1)[1].split("\n### ", 1)[0]
    for field in fields(RunResult):
        assert f"| `{field.name}` |" in contract, field.name
    assert f"All {len(fields(RunResult))} fields" in contract


def test_frontend_schema_fixture_is_current_when_present():
    import pytest

    _load_generator()
    import tools.gen_ui_schema_fixture as generator

    if not generator.FIXTURE.is_file():
        pytest.skip("Optional frontend source tree is not present")
    assert generator.FIXTURE.read_text(encoding="utf-8") == generator.render(), \
        "Run python -m tools.gen_ui_schema_fixture"


def test_architecture_import_table_is_current():
    _load_generator()
    import tools.gen_module_graph as graph

    document = (DOCS / "architecture.md").read_text(encoding="utf-8")
    table = document.split(graph.START, 1)[1].split(graph.END, 1)[0].strip()
    assert table == graph.render_table(), "Run python -m tools.gen_module_graph"


def test_architecture_figure_data_is_current():
    import pytest

    image_module = pytest.importorskip("PIL.Image")
    _load_generator()
    import tools.gen_module_graph as graph

    with image_module.open(DOCS / "module_graph.png") as image:
        assert image.info.get("DependencyDigest") == graph.dependency_digest(
            graph.module_edges()), "Run python -m tools.gen_module_graph"


def test_bundled_html_and_images_match_documentation():
    builder = _load_html_builder()
    for stem, _ in builder.PAGES:
        name = f"{stem}.html"
        assert (builder.STATIC_DOCS_DIR / name).read_bytes() == \
            (builder.HTML_DIR / name).read_bytes(), name
    for image in DOCS.glob("*.png"):
        for mirror in (builder.HTML_DIR, builder.STATIC_DOCS_DIR):
            assert (mirror / image.name).read_bytes() == image.read_bytes(), image.name


class _PageLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids: set[str] = set()
        self.links: list[str] = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if values.get("id"):
            self.ids.add(values["id"])
        if tag == "a" and values.get("href"):
            self.links.append(values["href"])


def test_local_documentation_links_resolve():
    import pytest

    builder = _load_html_builder()
    engine = builder.pick_engine(preferred="markdown")
    if engine is None:
        pytest.skip("Markdown renderer is not installed")
    pages = {}
    for stem, html in builder.render_pages(engine).items():
        parser = _PageLinks()
        parser.feed(html)
        pages[stem] = parser
    broken = []
    for stem, page in pages.items():
        for href in page.links:
            target = urlsplit(href)
            if target.scheme or target.netloc:
                continue
            path = Path(unquote(target.path))
            if path.parent != Path(".") or path.suffix not in ("", ".html"):
                continue
            target_stem = path.stem if target.path else stem
            if target_stem not in pages:
                broken.append(f"{stem}: {href}")
            elif target.fragment and unquote(target.fragment) not in pages[target_stem].ids:
                broken.append(f"{stem}: {href}")
    assert not broken, "Broken documentation links:\n" + "\n".join(broken)


def test_html_docs_are_not_stale():
    """Every committed docs/html/*.html must equal a fresh in-memory render
    of the corresponding docs/*.md — an md edit without re-running
    ``python -m tools.build_docs_html`` fails here.

    Content-hash/string equality, deliberately NOT mtime: mtimes are not
    CI-stable (checkouts, cache restores and rebases can all touch mtimes
    without touching content, which would make an mtime-based check flake
    independently of any real drift). ``render_pages`` produces its HTML
    purely from ``docs/*.md`` content plus static template code — no
    timestamps, no filesystem ordering — so repeat calls WITH THE SAME
    ENGINE are byte-identical (verified before this test was written: two
    consecutive ``build()`` calls produce an identical output directory).

    Pinned to the ``markdown`` engine specifically (``pick_engine(
    preferred="markdown")``), NOT ``pick_engine()``'s own preference-order
    fallback: the committed docs/html/*.html were produced by the
    pure-Python ``markdown`` package, and ``_render_body_pandoc`` (a
    ``pandoc --from gfm --to html5`` subprocess) is a genuinely different
    code path — confirmed by actually rendering all 10 pages both ways and
    diffing (measured directly, on pandoc 3.8): EVERY
    page differs (e.g. pandoc adds ``id=`` attributes to heading tags that
    the markdown engine's output does not carry), so a pandoc-rendered
    comparison against the committed markdown-engine fixture would flag
    drift on every page with none actually present. Falling back to
    ``pandoc`` here would make this test FAIL on a pandoc-only machine with
    zero actual drift, which is worse than not testing at all; pinning to
    the exact engine that produced the fixture is what makes "equal ⇒ no
    drift" hold in every environment, at the cost of skipping (see below)
    rather than verifying on a machine that only has ``pandoc``.

    Skips (does not fail) when the pure-Python ``markdown`` package is not
    installed, REGARDLESS of whether a system ``pandoc`` binary is present:
    `docs/*.md` is the source of truth and complete on its own
    (build_docs_html.py's own contract — see its module docstring), so an
    environment that cannot reproduce the committed HTML's own rendering
    path cannot be blamed for an apparent mismatch; that is a
    missing-tool condition, not a drift condition, and this test only pins
    drift within the one engine actually used to produce the fixture.
    """
    bdh = _load_html_builder()
    engine = bdh.pick_engine(preferred="markdown")
    if engine is None:
        import pytest
        pytest.skip(
            "the pure-Python 'markdown' package is not installed — the "
            "committed docs/html/*.html were built with it specifically, "
            "and a pandoc-rendered comparison would not be a same-engine "
            "drift check (see this test's docstring); docs/*.md remains "
            "the complete source of truth. Install 'markdown' to "
            "render/verify the HTML mirror."
        )
    fresh = bdh.render_pages(engine)
    assert fresh, "render_pages() produced no pages — PAGES list or docs/*.md missing?"
    stale = []
    for stem, html in fresh.items():
        committed_path = bdh.HTML_DIR / f"{stem}.html"
        if not committed_path.is_file():
            stale.append(f"{stem}.html: MISSING from docs/html/")
            continue
        committed = committed_path.read_text(encoding="utf-8")
        if committed != html:
            stale.append(f"{stem}.html: content differs from a fresh render")
    assert not stale, (
        "docs/html/ is stale; run `python -m tools.build_docs_html`:\n"
        + "\n".join(stale)
    )
