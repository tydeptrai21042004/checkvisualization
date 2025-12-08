#!/usr/bin/env python3
"""
dirviz.py – pretty directory-tree visualiser (no external dependencies)

Usage examples
──────────────
# Show the entire current folder
python dirviz.py

# Limit depth to 2 levels
python dirviz.py --depth 2

# Directories only, no files, respect .gitignore
python dirviz.py ~/Projects/my-repo --dirs-only

# Show file sizes, ignore node_modules and *.pyc
python dirviz.py --size --ignore "node_modules" --ignore "*.pyc"
"""

import argparse
import sys
import fnmatch
from pathlib import Path
from typing import List, Optional


def human_readable(size: int) -> str:
    """Convert a byte count into a human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if size < 1024:
            return f"{size:.0f}{unit}"
        size /= 1024
    return f"{size:.0f}EB"


def load_gitignore(root: Path) -> List[str]:
    """Load ignore patterns from .gitignore in the root directory."""
    gi = root / ".gitignore"
    if not gi.is_file():
        return []
    patterns = []
    for line in gi.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        patterns.append(line)
    return patterns


def should_ignore(path: Path, root: Path, patterns: List[str]) -> bool:
    """Determine if a path should be ignored based on glob patterns."""
    try:
        rel = str(path.relative_to(root))
    except ValueError:
        rel = path.name
    for pat in patterns:
        if fnmatch.fnmatch(path.name, pat) or fnmatch.fnmatch(rel, pat):
            return True
    return False


def walk(
    path: Path,
    prefix: str,
    depth: int,
    max_depth: Optional[int],
    show_files: bool,
    show_size: bool,
    ignores: List[str],
    root: Path,
) -> None:
    """Recursively print the directory tree under `path` with given options."""
    if max_depth is not None and depth > max_depth:
        return
    try:
        entries = sorted(
            [p for p in path.iterdir() if not should_ignore(p, root, ignores)],
            key=lambda p: (p.is_file(), p.name.lower()),
        )
    except PermissionError:
        print(prefix + "⛔ permission denied")
        return

    dirs = [e for e in entries if e.is_dir()]
    files = [e for e in entries if e.is_file()]
    total = len(dirs) + (len(files) if show_files else 0)

    for idx, d in enumerate(dirs):
        last = (idx == len(dirs) - 1) and not (show_files)
        connector = "└── " if last else "├── "
        print(prefix + connector + d.name)
        extension = "    " if last else "│   "
        walk(d, prefix + extension, depth + 1, max_depth, show_files, show_size, ignores, root)

    if show_files:
        for idx, f in enumerate(files):
            last = idx == len(files) - 1
            connector = "└── " if last else "├── "
            label = f.name
            if show_size:
                try:
                    size = human_readable(f.stat().st_size)
                except Exception:
                    size = "?"
                label += f" ({size})"
            print(prefix + connector + label)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise a directory tree in the console.")
    parser.add_argument(
        "path", nargs="?", default=".", help="Root directory (default: current dir)"
    )
    parser.add_argument(
        "--depth", type=int, help="Max recursion depth"
    )
    parser.add_argument(
        "--dirs-only", action="store_true", help="Show only directories"
    )
    parser.add_argument(
        "--size", action="store_true", help="Show file sizes"
    )
    parser.add_argument(
        "--no-gitignore", action="store_true", help="Ignore .gitignore rules"
    )
    parser.add_argument(
        "--ignore", action="append", default=[], help="Additional glob pattern(s) to ignore"
    )
    args = parser.parse_args()

    root = Path(args.path).expanduser().resolve()
    if not root.exists():
        sys.exit(f"❌ Path '{root}' does not exist.")

    # Always skip typical virtualenv / build bins
    DEFAULT_IGNORES = ["venv", ".venv", "bin"]

    ignores: List[str] = [] if args.no_gitignore else load_gitignore(root)
    # Order doesn't matter: any match will be ignored
    ignores = DEFAULT_IGNORES + ignores + args.ignore

    print(root)
    walk(
        root,
        prefix="",
        depth=0,
        max_depth=args.depth,
        show_files=not args.dirs_only,
        show_size=args.size,
        ignores=ignores,
        root=root,
    )


if __name__ == "__main__":
    main()
