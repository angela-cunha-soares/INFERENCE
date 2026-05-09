"""Quick check for orphan \\ref{...} and unused \\label{...} in the manuscript."""
from __future__ import annotations
import re, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
TEX = [
    ROOT / "manuscript" / "manuscript.tex",
    ROOT / "manuscript" / "appendix_inmet_validation.tex",
    ROOT / "manuscript" / "appendix_dag.tex",
]
TABLES_DIR = ROOT / "output" / "paper_tables"

LABEL = re.compile(r"\\label\{([^}]+)\}")
REF = re.compile(r"\\(?:ref|eqref|autoref)\{([^}]+)\}")
CITE = re.compile(r"\\cite[a-z]*\{([^}]+)\}")

labels: set[str] = set()
refs: set[str] = set()
cites: set[str] = set()

for f in TEX:
    s = f.read_text(encoding="utf-8")
    labels |= set(LABEL.findall(s))
    refs |= set(REF.findall(s))
    for c in CITE.findall(s):
        for k in c.split(","):
            cites.add(k.strip())

for f in TABLES_DIR.glob("*.tex"):
    s = f.read_text(encoding="utf-8")
    labels |= set(LABEL.findall(s))
    refs |= set(REF.findall(s))
    for c in CITE.findall(s):
        for k in c.split(","):
            cites.add(k.strip())

print("=== ORPHAN \\ref ===")
for r in sorted(refs - labels):
    print(" ", r)
print("\n=== UNUSED \\label ===")
for r in sorted(labels - refs):
    print(" ", r)

# Check bib keys
bib = (ROOT / "manuscript" / "references.bib").read_text(encoding="utf-8")
bib_keys = set(re.findall(r"^@\w+\{([^,]+),", bib, re.MULTILINE))
print("\n=== ORPHAN \\cite (key not in references.bib) ===")
for c in sorted(cites - bib_keys):
    print(" ", c)
