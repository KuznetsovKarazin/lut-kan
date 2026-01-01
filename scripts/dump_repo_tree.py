from __future__ import annotations

import argparse
from pathlib import Path

EXCLUDE_DIRS = {".git", ".venv", "venv", "env", "__pycache__", "outputs", "model", "checkpoints"}

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Repository root")
    ap.add_argument("--out", default="repo_tree.txt", help="Output text file")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    lines = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if any(part in EXCLUDE_DIRS for part in rel.parts):
            continue
        if p.is_dir():
            continue
        lines.append(str(rel).replace("\\", "/"))

    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.out} ({len(lines)} files).")

if __name__ == "__main__":
    main()
