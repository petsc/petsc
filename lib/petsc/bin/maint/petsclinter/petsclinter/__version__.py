#!/usr/bin/env python3
"""
# Created: Thu Dec  8 10:23:34 2022 (-0500)
# @author: Jacob Faibussowitsch
"""
from __future__ import annotations

import sys

def _read_versions() -> tuple[tuple[int, int, int], str, tuple[int, int, int]]:
  import os
  import re

  config_file = 'pyproject.toml'
  try:
    # since 3.11
    # novermin
    import tomllib # type: ignore[import]
  except (ModuleNotFoundError, ImportError):
    try:
      import tomli as tomllib # type: ignore[import]
    except (ModuleNotFoundError, ImportError):
      try:
        from pip._vendor import tomli as tomllib # type: ignore[import]
      except (ModuleNotFoundError, ImportError) as mnfe:
        raise RuntimeError(
          f'No package installed to read the {config_file} file! Install tomli via '
          'python3 -m pip install tomli'
        ) from mnfe

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

  # open ./../pyproject.toml
  toml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, config_file))
  with open(toml_path, 'rb') as fd:
    toml_data = tomllib.load(fd)

  project_data     = toml_data['project']
  version_str      = project_data['version']
  version_tup      = tuplify_version_str(version_str)
  py_version_match = re.search(r'([\d.]+)', project_data['requires-python'])
  assert py_version_match is not None # pacify type checkers
  min_py_version = tuplify_version_str(py_version_match.group(1))
  return version_tup, version_str, min_py_version

__version__, __version_str__, __MIN_PYTHON_VERSION__ = _read_versions()

class RedundantMinVersionCheckError(Exception):
  """
  Exception thrown when code checks for minimum python version which is less than the minimum
  requirement for petsclinter
  """
  pass

def py_version_lt(major: int, minor: int, sub_minor: int = 0) -> bool:
  r"""Determines if python version is less than a particular version.

  This should be used whenever back-porting some code as it will automatically raise an
  error if the version check is useless.

  Parameters
  ----------
  major :
    major version number, e.g. 3
  minor :
    minor version number
  sub_minor : optional
    sub-minor or patch version

  Returns
  -------
  ret :
    True if python version is less than `major`.`minor`.`sub_minor`, False otherwise

  Raises
  ------
  RedundantMinVersionCheckError
    If the given version is below petsclinter.__MIN_PYTHON_VERSION__ (and therefore the version check
    is pointless) this raises RedundantMinVersionCheckError. This should not be caught.
  """
  version = (major, minor, sub_minor)
  if version <= __MIN_PYTHON_VERSION__:
    raise RedundantMinVersionCheckError(
      f'Minimum required version {version_str()} already >= checked version {version}. '
      f'There is no need to include this version check!'
    )
  return sys.version_info < version

def version_tuple() -> tuple[int, int, int]:
  r"""Return the package version as a tuple

  Returns
  -------
  version :
    the package version as a tuple
  """
  return __version__

def version_str() -> str:
  r"""Return the package version as a string

  Returns
  -------
  version :
    the package version as a string
  """
  return __version_str__
