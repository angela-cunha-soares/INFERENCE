"""Audit manuscript for orphan refs/labels and unreferenced figures/tables.

Walks the manuscript main file and its \\input{...} chain, gathers every
\\label{...}, every \\ref/\\autoref/\\eqref{...}, every \\includegraphics{...}
and the \\input{...} chain of paper-tables files; reports:

* labels never \\ref'd
* \\ref's whose target is undefined
* image files in figures/paper/ not used by any \\includegraphics
* paper-table .tex files not \\input'd by any source
"""
from __future__ import annotations
import re, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
MAIN = ROOT / "manuscript" / "manuscript.tex"
TABLES_DIR = ROOT / "output" / "paper_tables"
FIGS_DIR = ROOT / "figures" / "paper"

LABEL = re.compile(r"\\label\{([^}]+)\}")
REF = re.compile(r"\\(?:ref|eqref|autoref)\{([^}]+)\}")
CITE = re.compile(r"\\cite[a-z]*\{([^}]+)\}")
INPUT = re.compile(r"\\input\{([^}]+)\}")
INCLUDE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")


def walk_inputs(path: pathlib.Path, base: pathlib.Path, seen: set) -> str:
    """Return concatenated text of file + recursive \\input chain."""
    if path in seen or not path.exists():
        return ""
    seen.add(path)
    text = path.read_text(encoding="utf-8")
    out = [text]
    for m in INPUT.finditer(text):
        target = (base / m.group(1)).resolve()
        if not target.suffix:
            target = target.with_suffix(".tex")
        # also try relative to ROOT for ../output/... patterns
        if not target.exists():
            alt = (path.parent / m.group(1)).resolve()
            if not alt.suffix:
                alt = alt.with_suffix(".tex")
            target = alt
        if target.exists():
            out.append(walk_inputs(target, target.parent, seen))
    return "\n".join(out)


def main() -> None:
    seen: set = set()
    full_text = walk_inputs(MAIN, MAIN.parent, seen)

    labels = set(LABEL.findall(full_text))
    refs = set(REF.findall(full_text))
    cites = set()
    for c in CITE.findall(full_text):
        for k in c.split(","):
            cites.add(k.strip())
    images = {pathlib.Path(p).name for p in INCLUDE.findall(full_text)}
    inputs = {pathlib.Path(p).name for p in INPUT.findall(full_text)}

    # bib keys
    bib = (ROOT / "manuscript" / "references.bib").read_text(encoding="utf-8")
    bib_keys = set(re.findall(r"^@\w+\{([^,]+),", bib, re.MULTILINE))

    # files on disk
    fig_files = {p.name for p in FIGS_DIR.glob("*.png")} | {p.name for p in FIGS_DIR.glob("*.pdf")}
    table_files = {p.name for p in TABLES_DIR.glob("*.tex")}

    print("=== ORPHAN \\ref (no matching \\label) ===")
    for r in sorted(refs - labels):
        print(" ", r)

    print("\n=== UNREFERENCED \\label (defined but never used) ===")
    for r in sorted(labels - refs):
        print(" ", r)

    print("\n=== ORPHAN \\cite (key not in references.bib) ===")
    for c in sorted(cites - bib_keys):
        print(" ", c)

    # Images in figures/paper not referenced
    img_used = {pathlib.Path(p).name.removesuffix(".png").removesuffix(".pdf")
                for p in INCLUDE.findall(full_text)}
    img_disk_stems = {p.removesuffix(".png").removesuffix(".pdf") for p in fig_files}

    print("\n=== FIGURES on disk NOT used by \\includegraphics ===")
    for s in sorted(img_disk_stems - img_used):
        print(" ", s)

    print("\n=== TABLES (paper_tables/*.tex) on disk NOT \\input'd ===")
    for t in sorted(table_files - inputs):
        print(" ", t)

    print("\n=== LABELS DEFINED (for reference) ===")
    print(f"  {len(labels)} labels, {len(refs)} refs, {len(cites)} cites, "
          f"{len(images)} \\includegraphics, {len(inputs)} \\input")


if __name__ == "__main__":
    main()
