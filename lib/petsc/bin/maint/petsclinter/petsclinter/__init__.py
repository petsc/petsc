#!/usr/bin/env python3
"""
# Created: Mon Jun 20 15:12:00 2022 (-0400)
# @author: Jacob Faibussowitsch
"""
import sys

from . import __version__

if sys.version_info < __version__.__MIN_PYTHON_VERSION__:
  raise ImportError('Need python ' + str(__version__.__MIN_PYTHON_VERSION__) + '+')

try:
  import clang.cindex
except ModuleNotFoundError as mnfe:
  if mnfe.name == 'clang':
    raise RuntimeError('Must run e.g. \'python3 -m pip install clang\' to use linter') from mnfe
  raise # whatever it is they should know about it

import pkgutil
import importlib

def __import_submodules(package, parent, recursive=True):
  """
  Import all submodules of a module, recursively, including subpackages

  :param package: package (name or actual module)
  :type package: str | module
  :rtype: dict[str, types.ModuleType]
  """
  if isinstance(package, str):
    package = importlib.import_module(package)

  if package is not parent and hasattr(package, '__export_symbols__'):
    for symbol in package.__export_symbols__:
      setattr(parent, symbol, getattr(package, symbol))
  results = {}
  for _, name, is_pkg in pkgutil.walk_packages(package.__path__):
    full_name = package.__name__ + '.' + name
    results[full_name] = importlib.import_module(full_name)
    if recursive and is_pkg:
      results.update(__import_submodules(full_name, parent, recursive=recursive))
  return results

def __build__all__(name, **kwargs):
  parent   = importlib.import_module(name)
  _modules = __import_submodules(name, parent, **kwargs)
  return list(_modules.keys())

def __lazy_import(name, package=None):
  """
  Lazily import the module NAME, it is loaded but not fully imported until an attribute of it is
  accessed
  """
  import importlib.util

  spec              = importlib.util.find_spec(name, package=package)
  loader            = spec.loader
  spec.loader       = importlib.util.LazyLoader(loader)
  module            = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  loader.exec_module(module)
  return module

from ._error import BaseError, ParsingError

# order is important to avoid circular imports
from . import util
from . import checks
from . import classes
# if all else fails, try the lazy import
# util = __lazy_import('.util', package='petsclinter')

__all__ = __build__all__(__name__)

# these should not be exposed
del __lazy_import
del __build__all__
del __import_submodules
