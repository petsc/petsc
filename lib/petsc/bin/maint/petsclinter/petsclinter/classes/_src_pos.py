#!/usr/bin/env python3
"""
# Created: Mon Jun 20 19:13:15 2022 (-0400)
# @author: Jacob Faibussowitsch
"""
from __future__ import annotations

import weakref
import functools
import clang.cindex as clx # type: ignore[import]

from .._typing import *

from .            import _util
from ._path       import Path
from ._attr_cache import AttributeCache

class ClangFileNameCache(weakref.WeakKeyDictionary[clx.TranslationUnit, clx.File]):
  """
  It is for whatever reason stupidly expensive to create these file objects, and clang does it
  every time you access a tu's file. So we cache them here
  """
  def getname(self, tu: clx.TranslationUnit) -> clx.File:
    if tu in self:
      return self[tu]
    new_file = self[tu] = tu.get_file(tu.spelling)
    return new_file

@functools.total_ordering
class SourceLocation(AttributeCache):
  """
  A simple wrapper class to add comparison operators to clx.SourceLocations since they only
  implement __eq__()
  """
  __filecache = ClangFileNameCache()
  __slots__   = 'source_location', 'translation_unit', 'offset'

  source_location: clx.SourceLocation
  translation_unit: Optional[clx.TranslationUnit]
  offset: int

  def __init__(self, source_location: SourceLocationLike, tu: Optional[clx.TranslationUnit] = None) -> None:
    r"""Construct a `SourceLocation`

    Parameters
    ----------
    source_location :
      the source location to create from
    tu : optional
      the translation unit owning the source location if available
    """
    if isinstance(source_location, SourceLocation):
      if tu is None:
        tu = source_location.translation_unit
      else:
        assert source_location.translation_unit is None, 'Both input tu and input source_location have valid Translation Units'
      super().__init__(source_location._cache)
      source_location = source_location.source_location
    else:
      super().__init__()
    self.source_location  = source_location
    self.translation_unit = tu # store a reference to guard against GC
    # offset is __by_far__ the most accessed attribute so we cache it in a slot, normally
    # we could just defer to _get_cached() but doing so results in a roughly 400%
    # performance degredation!
    self.offset = source_location.offset
    return

  def __hash__(self) -> int:
    return hash((self.source_location.file, self.offset))

  def __str__(self) -> str:
    return f'{Path(str(self.file)).resolve()}:{self.line}:{self.column}'

  def __repr__(self) -> str:
    return f'<self: {object.__repr__(self)}, translation unit: {self.translation_unit}, clang source location: {self.source_location}>'

  def __getattr__(self, attr: str) -> Any:
    return super()._get_cached(attr, getattr, self.source_location, attr)

  def __eq__(self, other: object) -> bool:
    return self is other or self.source_location.__eq__(self.as_clang_source_location(other))

  def __lt__(self, other: SourceLocationLike) -> bool:
    return self.offset < other.offset

  def __ge__(self, other: SourceLocationLike) -> bool:
    return self.offset >= other.offset

  @classmethod
  def cast(cls, other: SourceLocationLike, tu: Optional[clx.TranslationUnit] = None) -> SourceLocation:
    r"""Cast `other` to `SourceLocation

    Parameters
    ----------
    other :
      the object to cast

    Returns
    -------
    loc :
      the `SourceLocation`
    tu : optional
      the translation unit if `other` does not have one already

    Notes
    -----
    If `other` is a `SourceLocation` then this routine returns `other` unchanged. If it is a
    `clang.cindex.SourceLocation`, then it creates a new `SourceLocation` from it.

    Raises
    ------
    NotImplementedError
      if `other` is not a `SourceLocation` or `clang.cindex.SourceLocation`
    """
    if isinstance(other, cls):
      return other
    if isinstance(other, clx.SourceLocation):
      return cls(other, tu=tu)
    raise NotImplementedError(type(other))

  @classmethod
  def get_filename_from_tu(cls, tu: clx.TranslationUnit) -> clx.File:
    r"""Get the filename from a translation unit

    Parameters
    ----------
    tu :
      the translation unit

    Returns
    -------
    file :
      the file

    Notes
    -----
    It is for whatever reason stupidly expensive to create these as clang does not cache them. So
    this acts as a cache
    """
    return cls.__filecache.getname(tu)

  @classmethod
  def from_position(cls, tu: clx.TranslationUnit, line: int, col: int) -> SourceLocation:
    r"""Construct a `SourceLocation` from a position

    Parameters
    ----------
    tu :
      the translation unit of the source location
    line :
      the line number of the location
    col :
      the column number of the location

    Returns
    -------
    loc :
      the `SourceLocation`
    """
    return cls(clx.SourceLocation.from_position(tu, cls.get_filename_from_tu(tu), line, col), tu=tu)

  @functools.lru_cache
  def _get_src(self, func: Callable[..., str], *args, **kwargs) -> str:
    tu = self.translation_unit
    assert tu is not None
    right = SourceLocation.from_position(tu, self.line, self.column + 1)
    return func(SourceRange.from_locations(self, right), *args, **kwargs)

  def raw(self, *args, **kwargs) -> str:
    r"""Get the raw source for a `SourceLocation`

    Parameters
    ----------
    *args: iterable
      the positional arguments to `petsclinter._util.get_raw_source_from_source_range()`
    **kwargs: dict
      the keyword arguments to `petsclinter._util.get_raw_source_from_source_range()`

    Returns
    -------
    ret:
      the formatted source of the `SourceLocation`
    """
    return self._get_src(SourceRange.raw, *args, **kwargs)

  def formatted(self, *args, **kwargs) -> str:
    r"""Get the formatted source for a `SourceLocation`

    Parameters
    ----------
    *args: iterable
      the positional arguments to `petsclinter._util.get_formatted_source_from_source_range()`
    **kwargs: dict
      the keyword arguments to `petsclinter._util.get_formatted_source_from_source_range()`

    Returns
    -------
    ret:
      the formatted source of the `SourceLocation`
    """
    return self._get_src(SourceRange.formatted, *args, **kwargs)

  @classmethod
  def as_clang_source_location(cls, other: SourceLocationLike) -> clx.SourceLocation:
    r"""Get `other` as a `clang.cindex.SourceLocation`

    Parameters
    ----------
    other :
      a source location

    Returns
    -------
    loc :
      the `clang.cindex.SourceLocation`

    Raises
    ------
    NotImplementedError
      if `other` is not a `SourceLocation` or `clang.cindex.SourceLocation`

    Notes
    -----
    If `other` is a `clang.cindex.SourceLocation` then this routine returns `other` unchanged.
    Otherwise it returns the stored source location.
    """
    if isinstance(other, cls):
      return other.source_location
    if isinstance(other, clx.SourceLocation):
      return other
    raise NotImplementedError(type(other))

@functools.total_ordering
class SourceRange(AttributeCache):
  """Like SourceLocation but for clx.SourceRanges"""
  __slots__ = 'source_range', 'translation_unit', '_end', '_start'

  source_range: clx.SourceRange
  translation_unit: Optional[clx.TranslationUnit]
  _end: Optional[SourceLocation]
  _start: Optional[SourceLocation]

  def __init__(self, source_range: SourceRangeLike, tu: Optional[clx.TranslationUnit] = None) -> None:
    r"""Construct a `SourceRange`

    Parameters
    ----------
    source_range:
      the source `SourceRange`
    tu: optional
      the translation unit

    Raises
    ------
    ValueError
      if both `tu` is not None and `source_range` is a `SourceRange` and also has a valid translation
      unit, since it is ambigious which one should be used in that situation

    Notes
    -----
    Maybe it's not a big deal to simply prefer `tu` over `source_range.translation_unit` if both are
    given, but I had not found a test case in the wild to debug this situation with, so for now it
    errors.
    """
    if isinstance(source_range, SourceRange):
      if tu is None:
        tu = source_range.translation_unit
      elif source_range.translation_unit is not None:
        raise ValueError(
          'Both input tu and input source_range have valid Translation Units, don\'t know which to use!'
        )

      super().__init__(source_range._cache)
      self.source_range = source_range.source_range
      self._start       = source_range._start
      self._end         = source_range._end
    else:
      super().__init__()
      self.source_range = source_range
      self._start       = None
      self._end         = None
    self.translation_unit = tu # store a reference to guard against GC
    return

  def __hash__(self) -> int:
    return hash((self.__start(), self.__end()))

  def __repr__(self) -> str:
    return f'<self:{object.__repr__(self)}, tu: {self.translation_unit}, source range: {self.source_range}>'

  def __getattr__(self, attr: str) -> Any:
    return super()._get_cached(attr, getattr, self.source_range, attr)

  def __eq__(self, other: SourceRangeLike) -> bool:
    return self is other or self.source_range.__eq__(self.as_clang_source_range(other))

  def __lt__(self, other: Union[SourceRangeLike, SourceLocationLike]) -> bool:
    # If all this nonsense seems like a micro-optimization, it kinda is but also kinda
    # isn't. For regular usage this is way overkill, but all this __start() and __end()
    # caching and skipping the cast saves roughly 20s in a 100s run when overlap() is
    # called over 3 million times!
    if isinstance(other, SourceRange):
      other = other.__start()
    elif isinstance(other, clx.SourceRange):
      other = other.start
    elif isinstance(other, (clx.SourceLocation, SourceLocation)):
      pass
    else:
      raise NotImplementedError(type(other))
    self_end = self.__end()
    if self_end == other:
      return self.__start() < other
    return self_end < other

  def __contains__(self, other: Union[SourceRangeLike, SourceLocationLike]) -> bool:
    def contains(loc: Union[SourceRange, SourceLocation]) -> bool:
      # reimplement clx.SourceRange.__contains__() as it has a bug
      return start <= loc <= self.__end()

    start = self.__start()
    if isinstance(other, type(self)):
      return contains(other.__start()) and contains(other.__end())
    cast = SourceLocation.cast
    if isinstance(other, clx.SourceRange):
      return contains(cast(other.start)) and contains(cast(other.end))
    if isinstance(other, SourceLocation):
      return contains(other)
    if isinstance(other, clx.SourceLocation):
      return contains(cast(other))
    raise ValueError(type(other))

  def __len__(self) -> int:
    return self.__end().offset - self.__start().offset

  def __getitem__(self, idx: int) -> str:
    return super()._get_cached(
      '__raw_src', _util.get_raw_source_from_source_range, self
    ).splitlines()[idx]

  def __start(self) -> SourceLocation:
    if self._start is None:
      self._start = SourceLocation.cast(self.start)
    return self._start

  def __end(self) -> SourceLocation:
    if self._end is None:
      self._end = SourceLocation.cast(self.end)
    return self._end

  @classmethod
  def cast(cls, other: SourceRangeLike, tu: Optional[clx.TranslationUnit] = None) -> SourceRange:
    r"""Cast `other` into a `SourceRange`

    Parameters
    ----------
    other :
      the object to cast
    tu :
      the translation unit to attach (if `other` is a `clang.cindex.SourceRange`)

    Returns
    -------
    loc :
      the `SourceRange`

    Notes
    -----
    If `other` is a `SourceRange` then this routine returns `other` unchanged. If it is a
    `clang.cindex.SourceRange`, then it creates a new `SourceRange` from it.

    Raises
    ------
    NotImplementedError
      if `other` is not a `SourceRange` or `clang.cindex.SourceRange`
    """
    if isinstance(other, cls):
      return other
    if isinstance(other, clx.SourceRange):
      return cls(other, tu=tu)
    raise NotImplementedError(type(other))

  @classmethod
  def from_locations(cls, left: SourceLocationLike, right: SourceLocationLike, tu: Optional[clx.TranslationUnit] = None) -> SourceRange:
    r"""Construct a `SourceRange` from locations

    Parameters
    ----------
    left :
      the leftmost bound of the range
    right :
      the rightmost bound of the range
    tu : optional
      the translation unit of the range

    Returns
    -------
    rng :
      the constructed `SourceRange`

    Notes
    -----
    `left.offset` must be <= `right.offset`
    """
    assert left.offset <= right.offset
    if tu is None:
      attr = 'translation_unit'
      tu   = getattr(left, attr, None)
      if tu is None:
        tu = getattr(right, attr, None)
    as_clang_sl = SourceLocation.as_clang_source_location
    return cls(clx.SourceRange.from_locations(as_clang_sl(left), as_clang_sl(right)), tu=tu)

  @classmethod
  def from_positions(cls, tu: clx.TranslationUnit, line_left: int, col_left: int, line_right: int, col_right: int) -> SourceRange:
    r"""Construct a `SourceRange` from positions

    Parameters
    ----------
    tu :
      the translation unit containing the range
    line_left :
      the line number of the low bound
    col_left :
      the column number of the low bound
    line_right :
      the line number of the upper bound
    col_right :
      the column number of the upper bound

    Returns
    -------
    rng :
      the constructed `SourceRange`
    """
    filename = SourceLocation.get_filename_from_tu(tu)
    from_pos = clx.SourceLocation.from_position
    begin    = from_pos(tu, filename, line_left, col_left)
    end      = from_pos(tu, filename, line_right, col_right)
    return cls(clx.SourceRange.from_locations(begin, end), tu=tu)

  @classmethod
  def as_clang_source_range(cls, other: SourceRangeLike) -> clx.SourceRange:
    r"""Retrieve the `clang.cindex.SourceRange` from a source range

    Parameters
    ----------
    other :
      a source range

    Returns
    -------
    loc :
      the `clang.cindex.SourceRange`

    Raises
    ------
    NotImplementedError
      if `other` is not a `SourceRange` or `clang.cindex.SourceRange`

    Notes
    -----
    If `other` is a `clang.cindex.SourceRange` then this routine returns `other` unchanged.
    Otherwise it returns the stored source range.
    """
    if isinstance(other, cls):
      return other.source_range
    if isinstance(other, clx.SourceRange):
      return other
    raise NotImplementedError(type(other))

  @classmethod
  def merge(cls, left: SourceRangeLike, right: SourceRangeLike, tu: Optional[clx.TranslationUnit] = None) -> SourceRange:
    r"""Create a merged `SourceRange` from two ranges

    Parameters
    ----------
    left :
      the left range
    right :
      the right range
    tu :
      the translation unit containing the ranges

    Returns
    -------
    merged :
      the merged range

    Notes
    -----
    Constructs a range from the set union of `left` and `right`. `left` and `right` may overlap, be
    disjoint, or one may be entirely contained within the other.
    """
    cast  = SourceLocation.cast
    start = min(cast(left.start), cast(right.start))
    end   = max(cast(left.end),   cast(right.end))
    return cls.from_locations(start, end, tu=tu)

  def merge_with(self, other: SourceRangeLike) -> SourceRange:
    r"""See `SourceRange.merge()`"""
    return self.merge(self, other, tu=self.translation_unit)

  def overlaps(self, other: SourceRangeLike) -> bool:
    r"""Asks and answers the question: does this range overlap with `other`?

    Parameters
    ----------
    other :
      the other range

    Returns
    -------
    result :
      True if `self` overlaps with `other`, False otherwise

    Notes
    -----
    Two ranges are considered overlapping if either end is contained within the other. Notably, this
    also includes 'touching' ranges too, i.e. the start of one equals the end of the other. E.g. the
    following ranges:

    x------x          range_1
           x--------x range_2

    are considered to be overlapping, i.e. range_1.overlaps(range_2) is True (and vice-versa).
    """
    end = self.__end()
    if isinstance(other, type(self)):
      return end >= other.__start() and other.__end() >= self.__start()
    cast = SourceLocation.cast
    return end >= cast(other.start) and cast(other.end) >= self.__start()

  def resized(self, lbegin: int = 0, lend: int = 0, cbegin: Union[int, None] = 0, cend: Union[int, None] = 0) -> SourceRange:
    r"""Return a resized `SourceRange`, if the `SourceRange` was resized it is a new object

    Parameters
    ----------
    lbegin : optional
      number of lines to increment or decrement self.start.lines by
    lend : optional
      number of lines to increment or decrement self.end.lines by
    cbegin : optional
      number of columns to increment or decrement self.start.colummn by, None for BOL
    cend : optional
      number of columns to increment or decrement self.end.colummn by, None for EOL

    Returns
    -------
    ret :
      the resized `SourceRange`
    """
    start = self.__start()
    if cbegin is None:
      cbegin = -start.column + 1
    if cend == 0 and lbegin + lend + cbegin == 0:
      return self # nothing to do

    end    = self.__end()
    endcol = -1 if cend is None else end.column + cend # -1 is EOL
    return self.from_positions(
      self.translation_unit, start.line + lbegin, start.column + cbegin, end.line + lend, endcol
    )

  @functools.lru_cache
  def raw(self, *args, **kwargs) -> str:
    r"""Get the raw source for a `SourceRange`

    Parameters
    ----------
    *args: iterable
      the positional arguments to `petsclinter._util.get_raw_source_from_source_range()`
    **kwargs: dict
      the keyword arguments to `petsclinter._util.get_raw_source_from_source_range()`

    Returns
    -------
    ret:
      the raw source of the `SourceRange`
    """
    return _util.get_raw_source_from_source_range(self, *args, **kwargs)

  @functools.lru_cache
  def formatted(self, *args, **kwargs) -> str:
    r"""Get the formatted source for a `SourceRange`

    Parameters
    ----------
    *args: iterable
      the positional arguments to `petsclinter._util.get_formatted_source_from_source_range()`
    **kwargs: dict
      the keyword arguments to `petsclinter._util.get_formatted_source_from_source_range()`

    Returns
    -------
    ret:
      the formatted source of the `SourceRange`
    """
    return _util.get_formatted_source_from_source_range(self, *args, **kwargs)

  def view(self, *args, **kwargs) -> None:
    r"""View a `SourceRange`"""
    kwargs.setdefault('num_context', 5)
    print(self.formatted(*args, **kwargs))
    return
