#!/usr/bin/env python3
"""
# Created: Mon Aug  7 14:41:01 2023 (-0400)
# @author: Jacob Faibussowitsch
"""
from __future__ import annotations

from typing import Any

def assert_py_versions_match(config_file: str, toml_data: dict[str, Any]) -> None:
  import re

  def tuplify_version_str(version_str: str) -> tuple[int, int, int]:
    assert isinstance(version_str, str)
    version = list(map(int, version_str.split('.')))
    while len(version) < 3:
      version.append(0)
    # type checkers complain:
    #
    # Incompatible return value type (got "Tuple[int, ...]", expected
    # "Tuple[int, int, int]")  [return-value]
    #
    # but we know that version is of length 3, so we can safely ignore this
    return tuple(version) # type: ignore[return-value]

  project_data     = toml_data['project']
  py_version_match = re.search(r'([\d.]+)', project_data['requires-python'])
  assert py_version_match is not None # pacify type checkers
  min_py_version = tuplify_version_str(py_version_match.group(1))

  def assert_version_match(tool_name: str, version_str: str) -> None:
    tup = tuplify_version_str(version_str)
    if tup != min_py_version:
      raise ValueError(
        f'{tool_name} minimum python version {tup} in {config_file} does not match the projects '
        f'minimum {min_py_version}. Likely you have bumped up the project version and forgotten '
        'to update this one to match!'
      )
    return

  assert_version_match('mypy', toml_data['tool']['mypy']['python_version'])
  assert_version_match('vermin', str(toml_data['vermin']['targets']))
  return

def assert_requirements_match(toml_file: str, toml_data: dict[str, Any], req_file: str, req_lines: list[str]) -> None:
  toml_reqs = toml_data['project']['dependencies']
  if len(toml_reqs) != len(req_lines):
    raise RuntimeError(f'Number of requirements don\'t match. {toml_file}::[project][dependencies] has {len(toml_reqs)}, {req_file} has {len(req_lines)}')
  assert len(toml_reqs) == len(req_lines), f''
  for toml_req, req_req in zip(toml_reqs, req_lines):
    assert toml_req == req_req, f'{toml_file}: {toml_req} != {req_file}: {req_req}'
  return

def load_toml_data(toml_path: str) -> dict[str, Any]:
  try:
    # since 3.11
    # novermin
    import tomllib # type: ignore[import]
  except (ModuleNotFoundError, ImportError):
    try:
      import tomli as tomllib # type: ignore[import]
    except (ModuleNotFoundError, ImportError):
      try:
        from pip._vendor import tomli as tomllib
      except (ModuleNotFoundError, ImportError) as mnfe:
        raise RuntimeError(
          f'No package installed to read the {toml_path} file! Install tomli via '
          'python3 -m pip install tomli'
        ) from mnfe

  with open(toml_path, 'rb') as fd:
    toml_data: dict[str, Any] = tomllib.load(fd)
  return toml_data

def load_requirements_data(req_path: str) -> list[str]:
  ret: list[str] = []
  with open(req_path) as fd:
    for line in fd.readlines():
      if lstrp := line.strip():
        if not lstrp.startswith('#'):
          ret.append(lstrp)
  return ret

def main() -> None:
  import os

  par_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
  # open ./../pyproject.toml
  toml_path = os.path.join(par_dir, 'pyproject.toml')
  toml_data = load_toml_data(toml_path)
  # open ./../requirements.txt
  req_path = os.path.join(par_dir, 'requirements.txt')
  req_data = load_requirements_data(req_path)
  assert_py_versions_match(toml_path, toml_data)
  assert_requirements_match(toml_path, toml_data, req_path, req_data)
  return

if __name__ == '__main__':
  try:
    main()
  except Exception as exc:
    import sys

    print('=' * 90)
    print('ERROR:', str(exc))
    print('=' * 90)
    sys.exit(1)
