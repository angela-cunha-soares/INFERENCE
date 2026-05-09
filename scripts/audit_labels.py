"""Audit \\label and \\ref consistency across the LaTeX build set.

Reads only the files that are part of the build:
  - manuscript/manuscript.tex
  - the appendix files it \\input{}s
  - the output/paper_tables/*.tex it \\input{}s

Lists labels defined but not cited, and citations without a definition.
"""
from __future__ import annotations

import re
from pathlib import Path

LABEL_RE = re.compile(r"\\label\{((?:eq|fig|tab|app|sec):[^}]+)\}")
REF_RE = re.compile(r"\\(?:ref|eqref|autoref)\{((?:eq|fig|tab|app|sec):[^}]+)\}")
INPUT_RE = re.compile(r"\\input\{([^}]+)\}")

ROOT = Path(".")


def gather(start: str, base: Path = ROOT / "manuscript") -> list[Path]:
    """Recursively collect all files reachable through \\input{} from `start`."""
    visited: set[Path] = set()
    queue: list[Path] = [base / start]
    while queue:
        f = queue.pop().with_suffix("")
        for ext in (".tex", ""):
            p = Path(str(f) + ext)
            if p.exists() and p not in visited:
                visited.add(p)
                txt = p.read_text(encoding="utf-8", errors="replace")
                for inc in INPUT_RE.findall(txt):
                    candidate = (p.parent / inc).resolve()
                    queue.append(candidate)
                break
    return sorted(visited)


def main() -> int:
    files = gather("manuscript.tex")
    labels: set[str] = set()
    refs: set[str] = set()
    for p in files:
        txt = p.read_text(encoding="utf-8", errors="replace")
        labels.update(LABEL_RE.findall(txt))
        refs.update(REF_RE.findall(txt))

    print(f"Build set: {len(files)} files")
    for p in files:
        print(f"  {p}")

    print("\nLabels defined but NOT cited (excluding section anchors):")
    orphans = sorted(
        k for k in (labels - refs) if not k.startswith("sec:")
    )
    for k in orphans:
        print(f"  - {k}")
    if not orphans:
        print("  (none)")

    print("\nReferences to UNDEFINED labels:")
    missing = sorted(refs - labels)
    for k in missing:
        print(f"  - {k}")
    if not missing:
        print("  (none)")

    print(f"\nTotals: {len(labels)} labels defined, {len(refs)} citations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
