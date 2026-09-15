#!/usr/bin/env python3
"""Prepare the Sphinx source tree under PETSC_ARCH/doc without modifying doc/."""

import json
from pathlib import Path
import shutil
import subprocess
import sys


def main(petsc_dir, petsc_arch='arch-docs'):
  petsc_dir = Path(petsc_dir).resolve()
  docs_dir = petsc_dir / petsc_arch / 'doc'
  source_root = docs_dir / 'source'
  source_root.mkdir(parents=True, exist_ok=True)

  # Preserve repository-relative paths used by source includes and downloads.
  for name in ['src', 'include', 'config']:
    link = source_root / name
    if not link.is_symlink(): link.symlink_to(petsc_dir / name, target_is_directory=True)

  # Include uncommitted sources, but not ignored output from older in-source builds.
  files = subprocess.check_output(
    ['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard', '--', 'doc'],
    cwd=petsc_dir,
  ).decode().split('\0')
  files = {name for name in files if name and (petsc_dir / name).is_file()}
  manifest = source_root / 'files.json'
  if manifest.exists():
    for name in set(json.loads(manifest.read_text())) - files:
      (source_root / name).unlink(missing_ok=True)
  for name in sorted(files):
    target = source_root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(petsc_dir / name, target)
  manifest.write_text(json.dumps(sorted(files)))

  # Keep downloaded repositories outside the source tree removed by make clean.
  for name in ['images', 'packages-docs']:
    cache = docs_dir / name
    if name == 'packages-docs': cache.mkdir(exist_ok=True)
    link = source_root / 'doc' / name
    if not link.is_symlink(): link.symlink_to(cache, target_is_directory=True)


if __name__ == '__main__':
  main(*sys.argv[1:])
