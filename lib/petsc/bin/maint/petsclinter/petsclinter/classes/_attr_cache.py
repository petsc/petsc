#!/usr/bin/env python3
"""
# Created: Wed Aug  2 11:30:15 2023 (-0400)
# @author: Jacob Faibussowitsch
"""
from __future__ import annotations

from .._typing import *

_T = TypeVar('_T')

class AttributeCache:
  r"""
  An attribute cache (i.e. a `dict`) to enable lazy-evaluated default values. See the `_get_cached()`
  method for a description and further motivation for the existence of this class.
  """
  __slots__ = ('_cache',)

  _cache: dict[str, Any]

  def __init__(self, init_cache: Optional[dict[str, Any]] = None) -> None:
    r"""Construct an `AttributeCache`

    Parameters
    ----------
    init_cache : optional
      an existing dict to seed the cache with, or none to start fresh

    Raises
    ------
    TypeError
      if `init_cache` is not None but not a `dict`
    """
    if init_cache is None:
      init_cache = {}
    elif not isinstance(init_cache, dict):
      raise TypeError(type(init_cache))
    self._cache = init_cache
    return

  def _get_cached(self, attr: str, func: Callable[..., _T], *args, **kwargs) -> _T:
    r"""Get a cached value

    Parameters
    ----------
    attr :
      the attribute name to retrieve
    func :
      a function to produce a default value in case `attr` is not in the cache
    *args :
      positional arguments to `func`, if any
    **kwargs :
      keyword arguments to `func`, if any

    Returns
    -------
    ret :
      the value

    Notes
    -----
    If `attr` does not exist in the cache, this routine inserts the result of `func(*args, **kwargs)`
    into the cache.

    If all this sounds very similar to `dict.setdefault()`, then that's because it is. Normally you
    would have some `dict` member and use `dict.setdefault()` to get a default value from it.

    But this has one glaring weakness. `dict.setdefault()` (being a function call) obviously needs to
    evaluate its arguments, i.e. the default value is computed whether or not the key already exists in
    the dict (i.e. cache). This is OK if your default value is trivial, but what if producing it is
    expensive?

    This is why this class exists. Instead of taking the default by value, it takes the function which
    computes it by parts, and only calls it if needed.
    """
    cache = self._cache
    if attr in cache:
      return TYPE_CAST(_T, cache[attr])
    ret = cache[attr] = func(*args, **kwargs)
    return ret
