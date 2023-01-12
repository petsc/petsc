#!/usr/bin/env python3
"""
# Created: Thu Dec  8 10:23:34 2022 (-0500)
# @author: Jacob Faibussowitsch
"""
import sys

__MIN_PYTHON_VERSION__ = (3, 6, 0)
__version__            = (1, 0, 0)

class RedundantMinVersionCheckError(Exception):
  """
  Exception thrown when code checks for minimum python version which is less than the minimum
  requirement for petsclinter
  """
  pass

def py_version_lt(major, minor, sub_minor=0):
  version = (major, minor, sub_minor)
  if version <= __MIN_PYTHON_VERSION__:
    raise RedundantMinVersionCheckError(
      f'Minimum required version {__MIN_PYTHON_VERSION__} already >= checked version {version}. '
      f'There is no need to include this version check!'
    )
  return sys.version_info < version
