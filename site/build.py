"""Build the static website into _site/ (what GitHub Pages serves).

    python site/build.py            # writes _site/
    python -m http.server -d _site  # preview at http://localhost:8000

Copies the hand-written pages and assets from site/, fills in the shared
header and footer, copies the committed results (minus network checkpoints)
to _site/results/, and renders the research write-up from Markdown.
Needs the `markdown` package (pip install markdown).
"""

from __future__ import annotations

import argparse
import html
import re
import shutil
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
REPO_URL = "https://github.com/Sky1219/mnk-bench"
PORTFOLIO_URL = "https://sky1219.github.io"

NAV = [
    ("index.html", "Overview"),
    ("play.html", "Play"),
    ("arena.html", "Arena"),
    ("menace.html", "MENACE"),
    ("results/index.html", "Results"),
    ("research/targeted-selection.html", "Research"),
]

RESEARCH = [
    # (markdown source, output page, title)
    ("docs/research/targeted-selection.md", "research/targeted-selection.html",
     "Targeted labelling with submodular information measures"),
]

SKIP_SUFFIXES = {".pt"}  # model checkpoints: large and useless in a browser


# On narrow screens the nav scrolls sideways with no scrollbar: mark which edges hide more links
# (style.css fades them) and bring the current page's link into view.
NAV_SCRIPT = (
    "(()=>{const n=document.currentScript.previousElementSibling,"
    "f=()=>{n.classList.toggle('more-l',n.scrollLeft>2);"
    "n.classList.toggle('more-r',n.scrollLeft+n.clientWidth<n.scrollWidth-2)},"
    "a=n.querySelector('[aria-current]');"
    "if(a&&a.offsetLeft+a.offsetWidth>n.clientWidth)n.scrollLeft=a.offsetLeft+a.offsetWidth-n.clientWidth+24;"
    "n.addEventListener('scroll',f,{passive:true});addEventListener('resize',f);f()})()"
)


def header(page: str) -> str:
    prefix = "../" * page.count("/")
    current = ' aria-current="page"'
    links = "".join(
        f'<a href="{prefix}{href}"{current if href == page else ""}>{label}</a>' for href, label in NAV
    )
    logo = "".join(f'<i class="{c}"></i>' for c in "x.o.x.o.x")
    return (
        '<header class="site"><div class="wrap">'
        f'<a class="brand" href="{prefix}index.html"><span class="logo" aria-hidden="true">{logo}</span>'
        '<span class="name">mnk-bench</span></a>'
        f'<nav class="main" aria-label="Site">{links}<a href="{REPO_URL}">GitHub</a></nav>'
        f"<script>{NAV_SCRIPT}</script>"
        "</div></header>"
    )


def footer(page: str) -> str:
    return (
        '<footer class="site"><div class="wrap">'
        f'mnk-bench, built by <a href="{PORTFOLIO_URL}">Akash Dasarraju</a>. '
        f'Source, data and the Python package on <a href="{REPO_URL}">GitHub</a>. '
        "Every agent on this site runs locally in your browser."
        "</div></footer>"
    )


def fill(text: str, page: str) -> str:
    return text.replace("<!-- HEADER -->", header(page)).replace("<!-- FOOTER -->", footer(page))


def copy_results(out: Path) -> int:
    src, dst = ROOT / "docs" / "results", out / "results"
    count = 0
    for path in src.rglob("*"):
        if path.is_dir() or path.suffix in SKIP_SUFFIXES:
            continue
        target = dst / path.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        count += 1
    # A way back to the site from the generated report.
    index = dst / "index.html"
    if index.exists():
        back = (
            '<div style="max-width:1080px;margin:0 auto;padding:12px 16px 0;font-size:14px">'
            '<a href="../index.html" style="color:var(--accent);text-decoration:none">&larr; mnk-bench site</a>'
            ' &middot; <a href="../play.html" style="color:var(--accent);text-decoration:none">play the agents</a></div>'
        )
        report = index.read_text(encoding="utf-8")
        report = report.replace(
            "<head>", '<head><link rel="icon" href="../assets/favicon.svg" type="image/svg+xml">', 1
        )
        index.write_text(report.replace("<body>", "<body>" + back, 1), encoding="utf-8")
    return count


def fix_links(body: str) -> str:
    """Point repository-relative links at GitHub; keep links into results/ local."""

    def repl(m: re.Match) -> str:
        attr, url = m.group(1), m.group(2)
        if url.startswith("../../"):
            rest = url[len("../../"):]
            kind = "tree" if rest.endswith("/") else "blob"
            return f'{attr}="{REPO_URL}/{kind}/main/{rest}"'
        if url.startswith("../results/") and url.endswith("/"):
            return f'{attr}="{REPO_URL}/tree/main/docs/results/{url[len("../results/"):]}"'
        return m.group(0)

    return re.sub(r'(href|src)="([^"]+)"', repl, body)


def render_research(out: Path) -> None:
    template = (SITE / "templates" / "prose.html").read_text(encoding="utf-8")
    for src, page, title in RESEARCH:
        text = (ROOT / src).read_text(encoding="utf-8")
        body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists", "toc"])
        body = fix_links(body)
        source = f'<p class="small muted">Source: <a href="{REPO_URL}/blob/main/{src}">{src}</a></p>'
        doc = (
            template.replace("{{title}}", html.escape(title))
            .replace("{{prefix}}", "../" * page.count("/"))
            .replace("{{body}}", body + source)
        )
        target = out / page
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(fill(doc, page), encoding="utf-8")


def build(out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for page in sorted(SITE.glob("*.html")):
        (out / page.name).write_text(fill(page.read_text(encoding="utf-8"), page.name), encoding="utf-8")
    shutil.copytree(SITE / "assets", out / "assets")
    n = copy_results(out)
    render_research(out)
    (out / ".nojekyll").write_text("")
    print(f"built {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}: "
          f"{len(list(SITE.glob('*.html')))} pages, {n} result files, {len(RESEARCH)} research page(s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "_site")
    build(ap.parse_args().out.resolve())
