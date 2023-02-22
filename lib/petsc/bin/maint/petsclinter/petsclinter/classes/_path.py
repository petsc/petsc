#!/usr/bin/env python3
"""
# Created:
# @author: Jacob Faibussowitsch
"""
import pathlib

from .. import __version__

class Path(type(pathlib.Path())):
  """
  a basic pathlib.Path wrapper with some additional utility backported
  """
  # inheriting pathlib.Path:
  # https://stackoverflow.com/questions/29850801/subclass-pathlib-path-fails
  def append_suffix(self, suffix):
    """
    Create a path with SUFFIX appended, regardless of whether the current path has a suffix or not
    """
    suffix    = str(suffix)
    dotstring = '' if suffix.startswith('.') else '.'
    return self.with_suffix(f'{self.suffix}{dotstring}{suffix}')

  def append_name(self, name):
    """
    Create a path with NAME appended
    """
    return self.with_name(f'{self.stem}{name}')

  def unlink(self, missing_ok=False):
    """
    backport the missing_ok kwarg
    """
    if __version__.py_version_lt(3, 8):
      try:
        super().unlink()
      except FileNotFoundError as fnfe:
        if missing_ok:
          return
        raise
    else:
      super().unlink(missing_ok=missing_ok)
    return
