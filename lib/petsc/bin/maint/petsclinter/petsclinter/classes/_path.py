#!/usr/bin/env python3
"""
# Created:
# @author: Jacob Faibussowitsch
"""
from __future__ import annotations

from .._typing import *

import pathlib

from .. import __version__

# pathlib.Path() actually dynamically selects between pathlib.PosixPath and
# pathlib.WindowsPath (but this is static on each machine). For this reason, we need to
# use type(pathlib.Path()) to select the right type at runtime, but the type checkers
# don't like it because it is -- seemingly -- dynamic. So we arbitrarily pick PosixPath
# for type checkers.
#
# see https://stackoverflow.com/questions/29850801/subclass-pathlib-path-fails
if TYPE_CHECKING:
  _PathType = pathlib.PosixPath
else:
  _PathType = type(pathlib.Path())

class Path(_PathType):
  """
  a basic pathlib.Path wrapper with some additional utility backported
  """
  def append_suffix(self, suffix: str) -> Path:
    r"""Create a path with `suffix` appended, regardless of whether the current path has a suffix
    or not.

    Parameters
    ----------
    suffix:
      the suffix to append

    Returns
    -------
    path:
      the path with the suffix
    """
    suffix     = str(suffix)
    dotstring  = '' if suffix.startswith('.') else '.'
    path: Path = self.with_suffix(f'{self.suffix}{dotstring}{suffix}')
    return path

  def append_name(self, name: str) -> Path:
    r"""Create a path with `name` appended

    Parameters
    ----------
    name:
      the name to append

    Returns
    -------
    path:
      the path with the name
    """
    path: Path = self.with_name(f'{self.stem}{name}')
    return path
