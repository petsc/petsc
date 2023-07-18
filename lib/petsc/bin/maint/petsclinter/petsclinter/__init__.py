#!/usr/bin/env python3
"""
# Created: Mon Jun 20 15:12:00 2022 (-0400)
# @author: Jacob Faibussowitsch
"""
import sys

# synchronized print function, should be used everywhere
sync_print = print

from .__version__ import __MIN_PYTHON_VERSION__, version_tuple, version_str

if sys.version_info < __MIN_PYTHON_VERSION__:
  raise ImportError('Need python ' + version_str() + '+')

del __MIN_PYTHON_VERSION__

try:
  import clang.cindex # type: ignore[import]
except ModuleNotFoundError as mnfe:
  if mnfe.name == 'clang':
    raise RuntimeError('Must run e.g. \'python3 -m pip install clang\' to use linter') from mnfe
  raise # whatever it is they should know about it


def __lazy_import(name, package=None):
  """
  Lazily import the module NAME, it is loaded but not fully imported until an attribute of it is
  accessed
  """
  import importlib
  import importlib.util

  spec              = importlib.util.find_spec(name, package=package)
  loader            = spec.loader
  spec.loader       = importlib.util.LazyLoader(loader)
  module            = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  loader.exec_module(module)
  return module

from ._error import *

# order is important to avoid circular imports
from . import util
from . import checks
from . import _typing as typing
# if all else fails, try the lazy import
# util = __lazy_import('.util', package='petsclinter')
from .classes import *

# these should not be exposed
del __lazy_import
