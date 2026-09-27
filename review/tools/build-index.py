#!/usr/bin/env python3
"""Build review/INDEX.csv from tool output: symbols, dependencies, commits, metrics."""

import os
import sys
import csv
import json
import subprocess
import ast
from pathlib import Path
from collections import defaultdict
from datetime import datetime

REPO_ROOT = Path(__file__).parent.parent.parent
REVIEW_ROOT = REPO_ROOT / "review"
TOOLS_ROOT = REVIEW_ROOT / "tools"

def run_command(cmd, cwd=REPO_ROOT, capture=True):
    """Run command and return stdout or exit code."""
    try:
        result = subprocess.run(
            cmd, cwd=cwd, shell=True, capture_output=capture, text=True, timeout=30
        )
        return result.stdout.strip() if capture else result.returncode
    except subprocess.TimeoutExpired:
        print(f"⚠ Timeout: {cmd}", file=sys.stderr)
        return "" if capture else 1
    except Exception as e:
        print(f"⚠ Error running {cmd}: {e}", file=sys.stderr)
        return "" if capture else 1

def get_public_symbols(filepath):
    """Extract public (non-underscore) class/function names from Python file."""
    symbols = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(filepath))

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                name = node.name
                if not name.startswith("_"):
                    symbols.append(name)
    except Exception:
        pass

    return symbols[:5]  # Top 5 public symbols

def get_dependencies_and_imports(filepath):
    """Count imports and estimate dependencies."""
    imports = defaultdict(int)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("import ") or line.startswith("from "):
                # Extract module name
                if line.startswith("from "):
                    module = line.split()[1].split(".")[0]
                else:
                    module = line.split()[1].split(".")[0]
                imports[module] += 1
    except Exception:
        pass

    return dict(sorted(imports.items(), key=lambda x: -x[1])[:3])  # Top 3

def get_recent_commits(filepath):
    """Get count and date of most recent commit for file."""
    try:
        # Get commit count
        count_cmd = f"git log --oneline '{filepath}' 2>/dev/null | wc -l"
        count = run_command(count_cmd).strip()
        count = int(count) if count.isdigit() else 0

        # Get most recent commit date
        date_cmd = f"git log -1 --format=%ai '{filepath}' 2>/dev/null"
        date = run_command(date_cmd).split()[0] if run_command(date_cmd) else "unknown"

        return count, date
    except Exception:
        return 0, "unknown"

def get_cyclomatic_complexity(filepath):
    """Estimate cyclomatic complexity from control flow statements."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Count control flow keywords
        keywords = ["if ", "for ", "while ", "except ", "elif ", "else:", "and ", "or "]
        complexity = sum(content.count(kw) for kw in keywords)
        return complexity if complexity > 0 else 1
    except Exception:
        return 0

def get_file_lines(filepath):
    """Count lines of code (excluding blanks and comments)."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        code_lines = sum(
            1 for line in lines
            if line.strip() and not line.strip().startswith("#")
        )
        return len(lines), code_lines
    except Exception:
        return 0, 0

def extract_module_from_path(filepath, repo_root):
    """Extract module name from file path."""
    rel_path = Path(filepath).relative_to(repo_root)
    parts = rel_path.parts

    if parts[0] == "hybrid":
        if len(parts) > 1:
            return f"hybrid.{parts[1]}"
        return "hybrid"
    elif parts[0] == "routes":
        return "routes"
    elif parts[0] == "scripts":
        return "scripts"
    elif parts[0] == "tests":
        return "tests"
    elif parts[0] == "cloud":
        return "cloud"
    else:
        return parts[0]

def build_index():
    """Build comprehensive index CSV."""
    print("🔍 Building index...", file=sys.stderr)

    # Find all Python source files
    python_files = []
    for root, dirs, files in os.walk(REPO_ROOT):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in {
            ".git", "node_modules", ".venv", ".venv-a2", ".codebase-memory",
            "work", "assets", "__pycache__", ".pytest_cache"
        }]

        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                python_files.append(filepath)

    python_files = sorted(python_files)
    print(f"Found {len(python_files)} Python files", file=sys.stderr)

    rows = []
    for i, filepath in enumerate(python_files):
        if (i + 1) % 20 == 0:
            print(f"  Processing {i+1}/{len(python_files)}...", file=sys.stderr)

        try:
            rel_path = Path(filepath).relative_to(REPO_ROOT)
            module = extract_module_from_path(filepath, REPO_ROOT)
            total_lines, code_lines = get_file_lines(filepath)
            commits, last_commit = get_recent_commits(filepath)
            symbols = ", ".join(get_public_symbols(filepath)) if filepath.endswith(".py") else ""
            imports = get_dependencies_and_imports(filepath)
            complexity = get_cyclomatic_complexity(filepath)

            row = {
                "module": module,
                "path": str(rel_path),
                "loc": code_lines,
                "lines": total_lines,
                "commits": commits,
                "last_modified": last_commit,
                "public_symbols": symbols,
                "top_imports": "; ".join(f"{m}({c})" for m, c in imports.items()),
                "complexity_estimate": complexity,
            }
            rows.append(row)
        except Exception as e:
            print(f"⚠ Error processing {filepath}: {e}", file=sys.stderr)

    # Sort by module, then path
    rows = sorted(rows, key=lambda r: (r["module"], r["path"]))

    # Write CSV
    output_file = REVIEW_ROOT / "INDEX.csv"
    fieldnames = [
        "module", "path", "loc", "lines", "commits", "last_modified",
        "public_symbols", "top_imports", "complexity_estimate"
    ]

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Index written: {output_file}", file=sys.stderr)
    print(f"   Rows: {len(rows)}", file=sys.stderr)
    print(f"   Columns: {len(fieldnames)}", file=sys.stderr)

    # Print statistics
    print("\n📊 Index Statistics:", file=sys.stderr)
    total_loc = sum(r["loc"] for r in rows)
    total_commits = sum(r["commits"] for r in rows)
    print(f"   Total LOC: {total_loc}", file=sys.stderr)
    print(f"   Total commits: {total_commits}", file=sys.stderr)
    print(f"   Avg LOC/file: {total_loc // len(rows) if rows else 0}", file=sys.stderr)
    print(f"   Avg commits/file: {total_commits / len(rows):.1f}" if rows else "0", file=sys.stderr)

    # Module breakdown
    modules = defaultdict(int)
    for row in rows:
        modules[row["module"]] += 1
    print(f"\n   Modules:", file=sys.stderr)
    for module, count in sorted(modules.items(), key=lambda x: -x[1]):
        print(f"      {module}: {count} files", file=sys.stderr)

    return len(rows), output_file

if __name__ == "__main__":
    try:
        row_count, output_path = build_index()
        print(f"\n{output_path}|{row_count}")
    except Exception as e:
        print(f"❌ Fatal error: {e}", file=sys.stderr)
        sys.exit(1)
