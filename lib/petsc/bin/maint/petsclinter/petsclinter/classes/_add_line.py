#!/usr/bin/env python3
"""
# Created: Tue Aug  8 09:25:09 2023 (-0400)
# @author: Jacob Faibussowitsch
"""
from __future__ import annotations

import re

from .._typing import *

class Addline:
  diff_line_re: ClassVar[re.Pattern[str]] = re.compile(r'^@@ -([0-9,]+) \+([0-9,]+) @@')
  __slots__                               = ('offset',)

  offset: int

  def __init__(self, offset: int) -> None:
    r"""Construct an `Addline`

    Parameters
    ----------
    offset :
      the integer offset to add to the line number for the diff
    """
    self.offset = offset
    return

  def __call__(self, re_match: re.Match[str]) -> str:
    r"""Apply the offset to the regex match

    Parameters
    ----------
    re_match :
      the regex match object for the line

    Returns
    -------
    ret :
      the same matched string, but with the line numbers bumped up by `self.offset`
    """
    ll, lr  = re_match.group(1).split(',')
    rl, rr  = re_match.group(2).split(',')
    return f'@@ -{self.offset + int(ll)},{lr} +{self.offset + int(rl)},{rr} @@'
