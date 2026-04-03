#!/usr/bin/env python3
"""
Directory Mapper for .gitignore Planning
=========================================
Scans a folder and prints its full tree structure,
then suggests common .gitignore patterns based on what it finds.

Usage:
    python map_directory.py /path/to/your/folder
    python map_directory.py .                        # current directory
    python map_directory.py /path/to/folder -d 5     # limit depth to 5
    python map_directory.py /path/to/folder -o out.txt  # save to file
"""

import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict

# Known patterns that are typically gitignored
GITIGNORE_SUGGESTIONS = {
    # Node / JS
    "node_modules":        ("node_modules/", "Node.js dependencies"),
    "package-lock.json":   ("package-lock.json", "npm lock file (optional)"),
    ".next":               (".next/", "Next.js build output"),
    "dist":                ("dist/", "Build output"),
    "build":               ("build/", "Build output"),
    ".parcel-cache":       (".parcel-cache/", "Parcel cache"),
    # Python
    "__pycache__":         ("__pycache__/", "Python bytecode cache"),
    ".pyc":                ("*.pyc", "Python compiled files"),
    "venv":                ("venv/", "Python virtual environment"),
    "env":                 ("env/", "Python virtual environment"),
    ".venv":               (".venv/", "Python virtual environment"),
    ".egg-info":           ("*.egg-info/", "Python package metadata"),
    # Java / JVM
    "target":              ("target/", "Maven/Gradle build output"),
    ".gradle":             (".gradle/", "Gradle cache"),
    # Rust
    "target":              ("target/", "Rust/Cargo build output"),
    # General
    ".env":                (".env", "Environment variables (secrets!)"),
    ".env.local":          (".env.local", "Local environment variables"),
    ".DS_Store":           (".DS_Store", "macOS folder metadata"),
    "Thumbs.db":           ("Thumbs.db", "Windows thumbnail cache"),
    ".idea":               (".idea/", "JetBrains IDE config"),
    ".vscode":             (".vscode/", "VS Code config (optional)"),
    "*.log":               ("*.log", "Log files"),
    ".cache":              (".cache/", "Generic cache directory"),
    "coverage":            ("coverage/", "Test coverage reports"),
    ".nyc_output":         (".nyc_output/", "NYC coverage output"),
    # Compiled / binary
    "*.o":                 ("*.o", "Compiled object files"),
    "*.so":                ("*.so", "Shared libraries"),
    "*.dll":               ("*.dll", "Windows DLLs"),
    "*.exe":               ("*.exe", "Windows executables"),
}

def scan_directory(root_path, max_depth=None):
    """Walk the directory and return the tree structure + stats."""
    tree_lines = []
    file_counts = defaultdict(int)
    dir_names = set()
    file_names = set()
    total_files = 0
    total_dirs = 0
    large_files = []

    root = Path(root_path).resolve()

    for dirpath, dirnames, filenames in os.walk(root):
        rel = Path(dirpath).resolve().relative_to(root)
        depth = len(rel.parts)

        if max_depth is not None and depth > max_depth:
            dirnames.clear()
            continue

        indent = "│   " * depth
        dir_display = Path(dirpath).name + "/"
        if depth == 0:
            tree_lines.append(f"📁 {root.name}/")
        else:
            tree_lines.append(f"{indent[:-4]}├── 📁 {dir_display}")

        dir_names.add(Path(dirpath).name)
        total_dirs += 1

        # Sort for consistent output
        dirnames.sort()
        filenames.sort()

        for fname in filenames:
            file_indent = "│   " * (depth + 1)
            fpath = Path(dirpath) / fname
            try:
                size = fpath.stat().st_size
            except OSError:
                size = 0

            size_str = format_size(size)
            tree_lines.append(f"{file_indent[:-4]}├── {fname}  ({size_str})")

            ext = fpath.suffix.lower()
            file_counts[ext if ext else "(no extension)"] += 1
            file_names.add(fname)
            total_files += 1

            if size > 10 * 1024 * 1024:  # > 10 MB
                large_files.append((str(fpath.relative_to(root)), size))

    return tree_lines, file_counts, dir_names, file_names, total_files, total_dirs, large_files


def format_size(size_bytes):
    """Format bytes into a human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def suggest_gitignore(dir_names, file_names, file_counts):
    """Suggest .gitignore entries based on what was found."""
    suggestions = []
    seen_patterns = set()

    # Check directory names
    for name in dir_names:
        if name in GITIGNORE_SUGGESTIONS:
            pattern, desc = GITIGNORE_SUGGESTIONS[name]
            if pattern not in seen_patterns:
                suggestions.append((pattern, desc))
                seen_patterns.add(pattern)

    # Check file names
    for name in file_names:
        if name in GITIGNORE_SUGGESTIONS:
            pattern, desc = GITIGNORE_SUGGESTIONS[name]
            if pattern not in seen_patterns:
                suggestions.append((pattern, desc))
                seen_patterns.add(pattern)

    # Check extensions
    for ext in file_counts:
        key = f"*{ext}"
        if key in GITIGNORE_SUGGESTIONS:
            pattern, desc = GITIGNORE_SUGGESTIONS[key]
            if pattern not in seen_patterns:
                suggestions.append((pattern, desc))
                seen_patterns.add(pattern)

    return suggestions


def main():
    parser = argparse.ArgumentParser(description="Map a directory tree for .gitignore planning")
    parser.add_argument("path", help="Path to the directory to scan")
    parser.add_argument("-d", "--depth", type=int, default=None, help="Max depth to scan (default: unlimited)")
    parser.add_argument("-o", "--output", type=str, default=None, help="Save output to a file")
    args = parser.parse_args()

    root = Path(args.path)
    if not root.is_dir():
        print(f"Error: '{args.path}' is not a valid directory.")
        sys.exit(1)

    print(f"\n🔍 Scanning: {root.resolve()}\n")

    tree, file_counts, dir_names, file_names, total_files, total_dirs, large_files = scan_directory(
        root, max_depth=args.depth
    )

    output_lines = []

    # --- Tree ---
    output_lines.append("=" * 60)
    output_lines.append("  DIRECTORY TREE")
    output_lines.append("=" * 60)
    output_lines.extend(tree)

    # --- Summary ---
    output_lines.append("")
    output_lines.append("=" * 60)
    output_lines.append("  SUMMARY")
    output_lines.append("=" * 60)
    output_lines.append(f"  Total directories: {total_dirs}")
    output_lines.append(f"  Total files:       {total_files}")
    output_lines.append("")
    output_lines.append("  Files by extension:")
    for ext, count in sorted(file_counts.items(), key=lambda x: -x[1]):
        output_lines.append(f"    {ext:20s}  {count}")

    # --- Large files ---
    if large_files:
        output_lines.append("")
        output_lines.append("=" * 60)
        output_lines.append("  ⚠️  LARGE FILES (>10 MB) — consider .gitignore or Git LFS")
        output_lines.append("=" * 60)
        for fpath, size in large_files:
            output_lines.append(f"    {fpath}  ({format_size(size)})")

    # --- Gitignore suggestions ---
    suggestions = suggest_gitignore(dir_names, file_names, file_counts)
    if suggestions:
        output_lines.append("")
        output_lines.append("=" * 60)
        output_lines.append("  SUGGESTED .gitignore ENTRIES")
        output_lines.append("=" * 60)
        output_lines.append("")
        for pattern, desc in suggestions:
            output_lines.append(f"  {pattern:30s}  # {desc}")
        output_lines.append("")
        output_lines.append("  ── Ready-to-copy block ──")
        output_lines.append("")
        for pattern, _ in suggestions:
            output_lines.append(f"  {pattern}")

    full_output = "\n".join(output_lines)
    print(full_output)

    if args.output:
        with open(args.output, "w") as f:
            f.write(full_output)
        print(f"\n✅ Output saved to {args.output}")


if __name__ == "__main__":
    main()