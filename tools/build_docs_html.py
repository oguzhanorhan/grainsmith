"""Render the Markdown docs to a self-contained, full-screen themed HTML site.

Markdown under ``docs/`` is the source of truth (the drift-proof tests pin it);
this script builds a browsable HTML mirror under ``docs/html/`` — a full-screen
site with a fixed sidebar (all pages + per-page section links), an in-page table
of contents, syntax-highlighted code, and prev/next navigation, in the spirit of
the LAMMPS / Quantum ESPRESSO manuals.

The same rendered pages are also mirrored into
``src/grainsmith/ui/static/docs/``, the copy the "grainsmith studio" backend
serves at ``/docs/`` (``ui/server.py``'s static mount). Both copies come out
of this one build so they cannot drift apart — there is no separate
render path for the studio's copy.

**No external binary is required.** The Markdown engine is the pure-Python
``markdown`` package (``pip install ".[docs]"``); if it is not installed
the script falls back to the optional system ``pandoc`` if present, and otherwise
prints an actionable message and exits 0 (Markdown under ``docs/`` is complete on
its own, so a missing engine never breaks a build). Output is deterministic (no
timestamps), so re-runs are byte-stable and CI-friendly.

**Display math.** Source pages write display equations as fenced ``$$ ... $$``
blocks (raw TeX). Before either Markdown engine runs, every such block is
extracted (by regex, engine-agnostically) and swapped for a unique placeholder
token; after the engine produces HTML, each token is replaced with
``<div class="math-display">\\[ ... \\]</div>``. The page template loads
MathJax 3 from a CDN (``cdn.jsdelivr.net``) to typeset those divs client-side —
the only network dependency in an otherwise fully self-contained site. Without
that CDN (offline, no JS, or blocked network) the raw, HTML-escaped
``\\[ TeX \\]`` renders as plain centered text via the ``.math-display`` CSS
fallback — legible, just not typeset.

Usage::

    python -m tools.build_docs_html            # write docs/html/ + sync ui/static/docs/
    python -m tools.build_docs_html --check     # report which engine is available
    python -m tools.build_docs_html --engine markdown   # force a specific engine
"""
from __future__ import annotations

import argparse
import html as _html
import re
import shutil
import subprocess
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent / "docs"
HTML_DIR = DOCS / "html"
STATIC_DOCS_DIR = (Path(__file__).resolve().parent.parent
                   / "src" / "grainsmith" / "ui" / "static" / "docs")

# Docs rendered, grouped for the sidebar. (group_label, [(stem, human title), ...])
NAV: list[tuple[str, list[tuple[str, str]]]] = [
    ("Getting started", [
        ("index", "Overview"),
        ("manual", "User manual"),
    ]),
    ("Concepts & physics", [
        ("architecture", "Architecture"),
        ("geometry", "Grain geometry"),
        ("physics", "Physics & conventions"),
    ]),
    ("Reference", [
        ("config_reference", "Configuration reference"),
        ("gates", "QA gates (G1–G26)"),
        ("outputs", "Outputs"),
    ]),
    ("Developer", [
        ("developer", "Developer guide"),
    ]),
    ("Help", [
        ("faq", "FAQ & troubleshooting"),
    ]),
]

# Flattened page order (drives prev/next and the build loop).
PAGES: list[tuple[str, str]] = [pg for _, pgs in NAV for pg in pgs]
TITLES: dict[str, str] = {stem: title for stem, title in PAGES}

# ---------------------------------------------------------------------------
# Engine discovery
# ---------------------------------------------------------------------------
def have_markdown() -> bool:
    try:
        import markdown  # noqa: F401
        return True
    except ImportError:
        return False


def find_pandoc() -> str | None:
    return shutil.which("pandoc")


def pick_engine(preferred: str | None = None) -> str | None:
    """Return 'markdown', 'pandoc', or None (in preference order)."""
    if preferred == "markdown":
        return "markdown" if have_markdown() else None
    if preferred == "pandoc":
        return "pandoc" if find_pandoc() else None
    if have_markdown():
        return "markdown"
    if find_pandoc():
        return "pandoc"
    return None


# ---------------------------------------------------------------------------
# Fenced code blocks: mask before math extraction so a bare $$ line inside a
# ``` (or ~~~) fence is never mistaken for display math.
# ---------------------------------------------------------------------------
_FENCE_RE = re.compile(r"^[ \t]*(`{3,}|~{3,})[^\n]*\r?\n.*?\r?\n[ \t]*\1[ \t]*$",
                       re.MULTILINE | re.DOTALL)


def _fence_token(index: int) -> str:
    return f"GSFENCE{index}GSFENCE"


def _mask_fenced_code(md_text: str) -> tuple[str, list[str]]:
    """Replace every fenced code block with a placeholder line. Returns
    (text_with_tokens, [fenced_block_text, ...])."""
    blocks: list[str] = []

    def _stash(m: re.Match) -> str:
        blocks.append(m.group(0))
        return _fence_token(len(blocks) - 1)

    return _FENCE_RE.sub(_stash, md_text), blocks


def _unmask_fenced_code(md_text: str, blocks: list[str]) -> str:
    for i, block in enumerate(blocks):
        md_text = md_text.replace(_fence_token(i), block)
    return md_text


# ---------------------------------------------------------------------------
# Display math: $$ ... $$ fences <-> engine-proof placeholder tokens
# ---------------------------------------------------------------------------
# A display block is a line containing only '$$', TeX lines, a line containing
# only '$$' (the MATH CONTRACT in docs/*.md). Extracted BEFORE the Markdown
# engine runs (so neither engine mangles the TeX) and swapped back in AFTER
# (so both engines produce identical math markup).
_MATH_BLOCK_RE = re.compile(r"^\$\$[ \t]*\r?\n(.*?\r?\n)^\$\$[ \t]*$", re.MULTILINE | re.DOTALL)


def _math_token(index: int) -> str:
    return f"GSMATH{index}GSMATH"


def _extract_math_blocks(md_text: str) -> tuple[str, list[str]]:
    """Replace every ``$$ ... $$`` display block with a unique placeholder
    token on its own line. Fenced code blocks (``` / ~~~) are masked first so
    a bare ``$$`` line inside one is never swallowed as math. Returns
    (text_with_tokens, [tex, ...])."""
    blocks: list[str] = []

    def _stash(m: re.Match) -> str:
        blocks.append(m.group(1).strip("\n"))
        return _math_token(len(blocks) - 1)

    masked_text, fence_blocks = _mask_fenced_code(md_text)
    math_text = _MATH_BLOCK_RE.sub(_stash, masked_text)
    return _unmask_fenced_code(math_text, fence_blocks), blocks


def _reinject_math_blocks(body_html: str, blocks: list[str]) -> str:
    """Substitute each placeholder token (wherever the engine landed it,
    typically inside a stray <p>) with the rendered MathJax display div."""
    for i, tex in enumerate(blocks):
        token = _math_token(i)
        div = f'<div class="math-display">\\[ {_html.escape(tex)} \\]</div>'
        wrapped = f"<p>{token}</p>"
        if wrapped in body_html:
            body_html = body_html.replace(wrapped, div)
        else:
            body_html = body_html.replace(token, div)
    return body_html


# ---------------------------------------------------------------------------
# Markdown → HTML body (+ heading anchors + per-page TOC)
# ---------------------------------------------------------------------------
_SLUG_RE = re.compile(r"[^\w\- ]+")


def _slug(text: str) -> str:
    return _SLUG_RE.sub("", text).strip().lower().replace(" ", "-")


def _render_body_markdown(md_text: str) -> tuple[str, list[tuple[int, str, str]]]:
    """Render with the ``markdown`` package. Returns (html_body, toc_entries)."""
    import markdown
    from markdown.extensions import Extension
    from markdown.treeprocessors import Treeprocessor

    md_text, math_blocks = _extract_math_blocks(md_text)

    toc: list[tuple[int, str, str]] = []
    seen: dict[str, int] = {}

    class _Collect(Treeprocessor):
        def run(self, root):
            for el in root.iter():
                if el.tag in ("h2", "h3"):
                    # ``el.text`` is None when the heading STARTS with inline
                    # code (e.g. "### `RunResult`" -> <h3><code>...) since the
                    # text then lives on the child, not the h3 itself; collect
                    # via itertext() so those headings still get an id/TOC entry.
                    txt = "".join(el.itertext())
                    if not txt:
                        continue
                    base = _slug(txt)
                    n = seen.get(base, 0)
                    seen[base] = n + 1
                    sid = base if n == 0 else f"{base}-{n}"
                    el.set("id", sid)
                    toc.append((int(el.tag[1]), txt, sid))
            return None

    class _AnchorExt(Extension):
        def extendMarkdown(self, md):
            md.treeprocessors.register(_Collect(md), "gs_collect", 5)

    body = markdown.markdown(
        md_text,
        extensions=["extra", "sane_lists", "admonition", _AnchorExt()],
        output_format="html5",
    )
    body = _reinject_math_blocks(body, math_blocks)
    return body, toc


def _render_body_pandoc(pandoc: str, md_text: str) -> tuple[str, list[tuple[int, str, str]]]:
    """Fallback: render a body fragment with pandoc; parse a TOC from headings."""
    md_text, math_blocks = _extract_math_blocks(md_text)
    out = subprocess.run(
        [pandoc, "--from", "gfm", "--to", "html5"],
        input=md_text, check=True, capture_output=True, text=True, encoding="utf-8",
    ).stdout
    toc: list[tuple[int, str, str]] = []
    seen: dict[str, int] = {}

    def _add_id(m: re.Match) -> str:
        level, attrs, inner = int(m.group(1)), m.group(2), m.group(3)
        if "id=" in attrs:
            sid = re.search(r'id="([^"]+)"', attrs).group(1)
        else:
            base = _slug(re.sub(r"<[^>]+>", "", inner))
            n = seen.get(base, 0)
            seen[base] = n + 1
            sid = base if n == 0 else f"{base}-{n}"
            attrs = f'{attrs} id="{sid}"'
        if level in (2, 3):
            toc.append((level, re.sub(r"<[^>]+>", "", inner), sid))
        return f"<h{level}{attrs}>{inner}</h{level}>"

    body = re.sub(r"<h([1-6])([^>]*)>(.*?)</h\1>", _add_id, out, flags=re.DOTALL)
    body = _reinject_math_blocks(body, math_blocks)
    return body, toc


# ---------------------------------------------------------------------------
# Link rewriting: foo.md[#x] -> foo.html[#x]
# ---------------------------------------------------------------------------
def _rewrite_md_links(html: str) -> str:
    for stem, _ in PAGES:
        html = html.replace(f'href="{stem}.md"', f'href="{stem}.html"')
        html = html.replace(f'href="{stem}.md#', f'href="{stem}.html#')
    return html


# ---------------------------------------------------------------------------
# Page assembly (theme lives in CSS_THEME below)
# ---------------------------------------------------------------------------
def _sidebar(active: str) -> str:
    out = ['<nav class="sidebar" aria-label="Documentation navigation">']
    out.append('<div class="brand"><a href="index.html">grainsmith</a>'
               '<span class="brand-sub">documentation</span></div>')
    for group, pages in NAV:
        out.append(f'<div class="nav-group"><div class="nav-group-title">'
                   f'{_html.escape(group)}</div><ul>')
        for stem, title in pages:
            cls = ' class="active"' if stem == active else ""
            out.append(f'<li{cls}><a href="{stem}.html">{_html.escape(title)}</a></li>')
        out.append("</ul></div>")
    out.append("</nav>")
    return "\n".join(out)


def _toc(entries: list[tuple[int, str, str]]) -> str:
    if not entries:
        return ""
    out = ['<aside class="toc" aria-label="On this page"><div class="toc-title">'
           'On this page</div><ul>']
    for level, text, sid in entries:
        out.append(f'<li class="toc-l{level}"><a href="#{sid}">{_html.escape(text)}</a></li>')
    out.append("</ul></aside>")
    return "\n".join(out)


def _prevnext(stem: str) -> str:
    stems = [s for s, _ in PAGES]
    i = stems.index(stem)
    parts = ['<div class="prevnext">']
    if i > 0:
        ps, pt = PAGES[i - 1]
        parts.append(f'<a class="prev" href="{ps}.html"><span>← Previous</span>'
                     f'<strong>{_html.escape(pt)}</strong></a>')
    else:
        parts.append('<span></span>')
    if i < len(PAGES) - 1:
        ns, nt = PAGES[i + 1]
        parts.append(f'<a class="next" href="{ns}.html"><span>Next →</span>'
                     f'<strong>{_html.escape(nt)}</strong></a>')
    else:
        parts.append('<span></span>')
    parts.append("</div>")
    return "\n".join(parts)


def _page(stem: str, title: str, body: str, toc: list[tuple[int, str, str]]) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{_html.escape(title)} · grainsmith</title>
<style>
{CSS_THEME}
</style>
<script>
window.MathJax = {{
  tex: {{ displayMath: [['\\\\[', '\\\\]']], processEscapes: true }},
  options: {{ enableMenu: false }}
}};
</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
</head>
<body>
<button class="menu-toggle" aria-label="Toggle navigation" onclick="document.body.classList.toggle('nav-open')">☰</button>
{_sidebar(stem)}
<main class="content">
<div class="content-inner">
<article>
{body}
{_prevnext(stem)}
</article>
{_toc(toc)}
</div>
</main>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Full-screen theme
# ---------------------------------------------------------------------------
CSS_THEME = r"""
:root{
  --sidebar-w: 300px; --toc-w: 240px;
  --accent:#b5651d; --accent-2:#1f6f8b;
  --ink:#1a1f24; --ink-soft:#4a545e; --line:#e3e7ea;
  --bg:#ffffff; --bg-side:#1c2530; --bg-side-2:#243040;
  --code-bg:#f5f7f9; --side-ink:#c9d3de; --side-ink-soft:#8a97a6;
}
*{box-sizing:border-box;}
html,body{margin:0;padding:0;}
body{
  font-family:Georgia,"Times New Roman",Times,serif;
  color:var(--ink); background:var(--bg); line-height:1.68; font-size:16px;
}
/* ---- full-screen layout: fixed sidebar + fluid content ---- */
.sidebar{
  position:fixed; top:0; left:0; width:var(--sidebar-w); height:100vh;
  overflow-y:auto; background:var(--bg-side); color:var(--side-ink);
  padding:22px 0 40px; z-index:20;
}
.brand{padding:6px 24px 20px;border-bottom:1px solid rgba(255,255,255,.08);margin-bottom:14px;}
.brand a{color:#fff;font-weight:700;font-size:1.35rem;text-decoration:none;letter-spacing:.5px;}
.brand-sub{display:block;color:var(--side-ink-soft);font-size:.8rem;margin-top:2px;}
.nav-group{margin:0 0 6px;}
.nav-group-title{
  padding:12px 24px 5px;font-size:.72rem;text-transform:uppercase;
  letter-spacing:.09em;color:var(--side-ink-soft);font-weight:700;
}
.nav-group ul{list-style:none;margin:0;padding:0;}
.nav-group li a{
  display:block;padding:6px 24px 6px 28px;color:var(--side-ink);
  text-decoration:none;font-size:.92rem;border-left:3px solid transparent;
}
.nav-group li a:hover{background:var(--bg-side-2);color:#fff;}
.nav-group li.active a{
  color:#fff;border-left-color:var(--accent);background:var(--bg-side-2);font-weight:600;
}
.content{margin-left:var(--sidebar-w);min-height:100vh;}
.content-inner{
  display:grid;grid-template-columns:minmax(0,1fr) var(--toc-w);
  gap:48px;max-width:1500px;margin:0 auto;padding:52px 64px 96px;
}
article{min-width:0;}
/* ---- in-page TOC (sticky, right rail) ---- */
.toc{position:sticky;top:36px;align-self:start;font-size:.86rem;
  border-left:1px solid var(--line);padding-left:18px;max-height:calc(100vh-72px);overflow-y:auto;}
.toc-title{font-weight:700;font-size:.72rem;text-transform:uppercase;letter-spacing:.08em;
  color:var(--ink-soft);margin-bottom:10px;}
.toc ul{list-style:none;margin:0;padding:0;}
.toc li{margin:3px 0;}
.toc li a{color:var(--ink-soft);text-decoration:none;display:block;line-height:1.4;}
.toc li a:hover{color:var(--accent);}
.toc-l3{padding-left:14px;font-size:.82rem;}
/* ---- typography ---- */
article h1{font-size:2.05rem;line-height:1.2;margin:.1em 0 .7em;font-weight:750;letter-spacing:-.01em;}
article h2{font-size:1.5rem;margin:1.9em 0 .6em;padding-bottom:.28em;
  border-bottom:2px solid var(--line);font-weight:700;}
article h3{font-size:1.18rem;margin:1.6em 0 .5em;font-weight:650;}
article h4{font-size:1.02rem;margin:1.3em 0 .4em;color:var(--ink-soft);text-transform:uppercase;letter-spacing:.03em;}
article p{margin:.85em 0;}
article a{color:var(--accent-2);text-decoration:none;}
article a:hover{text-decoration:underline;}
article strong{font-weight:650;}
article img{max-width:100%;height:auto;display:block;margin:1.4em auto;
  border:1px solid var(--line);border-radius:8px;background:#fff;}
article ul,article ol{padding-left:1.5em;margin:.7em 0;}
article li{margin:.32em 0;}
article blockquote{
  margin:1.3em 0;padding:.7em 1.2em;border-left:4px solid var(--accent);
  background:#fbf6f0;color:var(--ink);border-radius:0 6px 6px 0;
}
article blockquote p{margin:.4em 0;}
/* ---- display math (MathJax typesets these; this is the no-JS/offline fallback) ---- */
.math-display{
  margin:1.5em 0;padding:.9em 1.2em;text-align:center;overflow-x:auto;
  font-family:"Cambria Math","STIX Two Math",Cambria,"Times New Roman",serif;
  font-size:1.05rem;color:var(--ink);
}
/* ---- code ---- */
code{font-family:"SF Mono",SFMono-Regular,ui-monospace,Menlo,Consolas,monospace;
  font-size:.88em;background:var(--code-bg);padding:.12em .38em;border-radius:4px;
  color:#b03a2e;border:1px solid var(--line);}
pre{background:var(--code-bg);border:1px solid var(--line);border-radius:8px;
  padding:16px 18px;overflow-x:auto;margin:1.2em 0;line-height:1.5;}
pre code{background:none;border:none;padding:0;color:var(--ink);font-size:.86rem;}
/* ---- tables (wide, striped, scrollable) ---- */
article table{border-collapse:collapse;width:100%;margin:1.3em 0;font-size:.9rem;display:block;overflow-x:auto;}
article thead th{background:#2c3844;color:#fff;text-align:left;padding:9px 12px;font-weight:600;white-space:nowrap;}
article tbody td{padding:8px 12px;border-bottom:1px solid var(--line);vertical-align:top;}
article tbody tr:nth-child(even){background:#f8fafb;}
article tbody code{white-space:nowrap;}
/* ---- prev/next ---- */
.prevnext{display:flex;justify-content:space-between;gap:16px;margin-top:3.5em;
  padding-top:1.6em;border-top:1px solid var(--line);}
.prevnext a{flex:1;max-width:48%;padding:14px 18px;border:1px solid var(--line);
  border-radius:8px;text-decoration:none;color:var(--ink);background:#fcfdfe;}
.prevnext a:hover{border-color:var(--accent);background:#fbf6f0;}
.prevnext .next{text-align:right;}
.prevnext span{display:block;font-size:.78rem;color:var(--ink-soft);margin-bottom:3px;}
.prevnext strong{font-size:.98rem;color:var(--accent-2);font-weight:600;}
/* ---- mobile ---- */
.menu-toggle{display:none;position:fixed;top:12px;left:12px;z-index:30;
  background:var(--bg-side);color:#fff;border:none;border-radius:6px;
  width:42px;height:42px;font-size:1.3rem;cursor:pointer;}
@media (max-width:1080px){
  .content-inner{grid-template-columns:minmax(0,1fr);}
  .toc{display:none;}
}
@media (max-width:820px){
  .menu-toggle{display:block;}
  .sidebar{transform:translateX(-100%);transition:transform .2s ease;}
  body.nav-open .sidebar{transform:translateX(0);}
  .content{margin-left:0;}
  .content-inner{padding:64px 22px 72px;}
}
"""


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def render_pages(engine: str) -> dict[str, str]:
    """Render every page in ``PAGES`` to an HTML string, in memory —
    no filesystem writes. Returns ``{stem: html}``.

    Pure function (mirrors ``tools/gen_config_reference.py``'s ``render()``
    split): this is what ``build()`` below writes to disk, and it is also
    what ``tests/test_docs.py::test_html_docs_are_not_stale`` calls to
    compare against the COMMITTED ``docs/html/*.html`` files without ever
    touching them — confirmed deterministic (byte-identical across repeat
    calls with the same source `docs/*.md`, verified by running `build()`
    twice and diffing the output directory), so the staleness test is
    CI-stable: no timestamp/mtime dependence anywhere in this function.
    """
    pandoc = find_pandoc() if engine == "pandoc" else None
    pages: dict[str, str] = {}
    for stem, title in PAGES:
        md = DOCS / f"{stem}.md"
        if not md.is_file():
            sys.stderr.write(f"skip: {md} missing\n")
            continue
        if engine == "markdown":
            body, toc = _render_body_markdown(md.read_text(encoding="utf-8"))
        else:
            body, toc = _render_body_pandoc(pandoc, md.read_text(encoding="utf-8"))
        body = _rewrite_md_links(body)
        pages[stem] = _page(stem, title, body, toc)
    return pages


def build(engine: str) -> list[Path]:
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    pages = render_pages(engine)
    written: list[Path] = []
    for stem, page in pages.items():
        out = HTML_DIR / f"{stem}.html"
        out.write_text(page, encoding="utf-8")
        written.append(out)
        print(f"wrote {out}")
    # copy figure assets referenced by the docs
    for img in DOCS.glob("*.png"):
        dst = HTML_DIR / img.name
        dst.write_bytes(img.read_bytes())

    # Mirror the same pages + figure assets into the studio's static bundle
    # (src/grainsmith/ui/static/docs/, served at /docs/ by ui/server.py's
    # StaticFiles mount) so the two copies are always the same build output.
    STATIC_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for stem, page in pages.items():
        (STATIC_DOCS_DIR / f"{stem}.html").write_text(page, encoding="utf-8")
        print(f"synced {STATIC_DOCS_DIR / f'{stem}.html'}")
    for img in DOCS.glob("*.png"):
        (STATIC_DOCS_DIR / img.name).write_bytes(img.read_bytes())
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the themed HTML doc site.")
    parser.add_argument("--check", action="store_true",
                        help="report which Markdown engine is available.")
    parser.add_argument("--engine", choices=["markdown", "pandoc"], default=None,
                        help="force a specific engine (default: markdown, then pandoc).")
    args = parser.parse_args(argv)

    engine = pick_engine(args.engine)
    if args.check:
        print(f"markdown package: {'available' if have_markdown() else 'NOT installed'}")
        print(f"pandoc binary:    {find_pandoc() or 'NOT FOUND'}")
        print(f"selected engine:  {engine or 'NONE'}")
        return 0 if engine else 1
    if engine is None:
        sys.stderr.write(
            "No Markdown engine found — HTML docs not built.\n"
            'Install the pure-Python engine:  pip install ".[docs]"\n'
            "  (or:  pip install markdown)\n"
            "or install the optional 'pandoc' binary. The Markdown docs under\n"
            "docs/ are the source of truth and are complete on their own.\n"
        )
        return 0

    written = build(engine)
    print(f"\nbuilt {len(written)} pages with engine '{engine}' → {HTML_DIR}"
         f" (synced to {STATIC_DOCS_DIR})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
