#!/usr/bin/env python3
import sys
import nbformat as nbf
from pathlib import Path

if len(sys.argv) != 2:
    print("Usage: python clean.py NOTEBOOK_NAME_WITHOUT_SUFFIX")
    print("Example: python clean.py check_figs")
    sys.exit(1)

fname = sys.argv[1]

src = Path("bk") / f"{fname}.ipynb"
dst = Path(f"{fname}.ipynb")

if not src.exists():
    print(f"Error: source file not found: {src}")
    sys.exit(1)

nb = nbf.read(src, as_version=4)

for c in nb.cells:
    if c.get("cell_type") == "code":
        c["outputs"] = []
        c["execution_count"] = None
    c["metadata"] = {}

nb["metadata"] = {}

nbf.write(nb, dst)
print(f"Cleaned notebook written to: {dst}")
