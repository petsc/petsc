#!/usr/bin/env python3
"""
# Created: Tue Aug  8 09:23:03 2023 (-0400)
# @author: Jacob Faibussowitsch
"""
from __future__ import annotations

from .._typing import *

_T = TypeVar('_T')

class WeakList(List[_T]):
  r"""Adaptor class to make builtin lists weakly referenceable"""
  __slots__ = ('__weakref__',)
