#!/usr/bin/env python3
"""Bump version across all configuration files.

Usage:
    python scripts/bump_version.py 0.5.6

Updates:
    - app.py (APP_VERSION)
    - desktop/src-tauri/Cargo.toml
    - desktop/src-tauri/tauri.conf.json
"""
import sys
import re
from pathlib import Path

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <version>")
    print("Example: python scripts/bump_version.py 0.5.6")
    sys.exit(1)

new_version = sys.argv[1].lstrip('v')
if not re.match(r'^\d+\.\d+\.\d+$', new_version):
    print(f"Error: version must be in format X.Y.Z (got {new_version})")
    sys.exit(1)

repo_root = Path(__file__).parent.parent

def update_app_py(content):
    return re.sub(
        r'APP_VERSION = os\.environ\.get\("NAM_MIXER_VERSION", "v[\d.]+"\)',
        f'APP_VERSION = os.environ.get("NAM_MIXER_VERSION", "v{new_version}")',
        content
    )

def update_cargo_toml(content):
    """Update only [package] version, not dependency versions."""
    lines = content.split('\n')
    result = []
    in_package = False
    version_updated = False

    for line in lines:
        if line.strip() == '[package]':
            in_package = True
        elif line.startswith('['):
            in_package = False

        if in_package and line.strip().startswith('version = ') and not version_updated:
            line = f'version = "{new_version}"'
            version_updated = True

        result.append(line)

    return '\n'.join(result)

def update_tauri_conf(content):
    return re.sub(
        r'"version": "[\d.]+"',
        f'"version": "{new_version}"',
        content
    )

files = [
    (repo_root / "app.py", update_app_py),
    (repo_root / "desktop/src-tauri/Cargo.toml", update_cargo_toml),
    (repo_root / "desktop/src-tauri/tauri.conf.json", update_tauri_conf),
]

for file_path, updater in files:
    if not file_path.exists():
        print(f"Warning: {file_path} not found")
        continue

    content = file_path.read_text()
    new_content = updater(content)

    if new_content == content:
        print(f"Warning: no version found in {file_path}")
    else:
        file_path.write_text(new_content)
        print(f"Updated {file_path}")

print(f"\nVersion bumped to {new_version}")
print("Next steps:")
print("  1. Verify the changes: git diff")
print(f"  2. Commit: git add -A && git commit -m 'Bump version to v{new_version}'")
print(f"  3. Tag: git tag v{new_version}")
print("  4. Push: git push && git push --tags")
