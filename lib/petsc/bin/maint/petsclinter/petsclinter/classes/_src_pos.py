#!/usr/bin/env python3
"""
# Created: Mon Jun 20 19:13:15 2022 (-0400)
# @author: Jacob Faibussowitsch
"""
import weakref
import functools
import clang.cindex as clx

from .      import _util
from ._path import Path

class ClangFileNameCache(weakref.WeakKeyDictionary):
  """
  It is for whatever reason stupidly expensive to create these file objects, and clang does it
  every time you access a tu's file. So we cache them here
  """
  def getname(self, tu):
    if tu in self:
      return self[tu]
    new_file = self[tu] = tu.get_file(tu.spelling)
    return new_file

class AttributeCache:
  __slots__ = ('_cache',)

  def __init__(self, init_cache=None):
    if init_cache is None:
      init_cache = {}
    else:
      assert isinstance(init_cache, dict)
    self._cache = init_cache
    return

  def get_cached(self, attr, func, *args, **kwargs):
    cache = self._cache
    if attr in cache:
      return cache[attr]
    new_val = cache[attr] = func(*args, **kwargs)
    return new_val

@functools.total_ordering
class SourceLocation(AttributeCache):
  """
  A simple wrapper class to add comparison operators to clx.SourceLocations since they only
  implement __eq__()
  """
  __filecache = ClangFileNameCache()
  __slots__   = 'source_location', 'translation_unit', 'offset'

  def __init__(self, source_location, tu=None):
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

  def __hash__(self):
    return hash(self.source_location.file) ^ hash(self.offset)

  def __str__(self):
    return f'{Path(str(self.file)).resolve()}:{self.line}:{self.column}'

  def __repr__(self):
    return f'<self: {object.__repr__(self)}, translation unit: {self.translation_unit}, clang source location: {self.source_location}>'

  def __getattr__(self, attr):
    return super().get_cached(attr, getattr, self.source_location, attr)

  def __eq__(self, other):
    return self is other or self.source_location.__eq__(self.as_clang_source_location(other))

  def __lt__(self, other):
    if not isinstance(other, (type(self), clx.SourceLocation)):
      return NotImplemented
    return self.offset < other.offset

  def __ge__(self, other):
    if not isinstance(other, (type(self), clx.SourceLocation)):
      return NotImplemented
    return self.offset >= other.offset

  @classmethod
  def cast(cls, other):
    if isinstance(other, cls):
      return other
    if isinstance(other, clx.SourceLocation):
      return cls(other)
    raise NotImplementedError(type(other))

  @classmethod
  def get_filename_from_tu(cls, tu):
    return cls.__filecache.getname(tu)

  @classmethod
  def from_position(cls, tu, line, col):
    return cls(clx.SourceLocation.from_position(tu, cls.get_filename_from_tu(tu), line, col), tu=tu)

  @classmethod
  def as_clang_source_location(cls, other):
    if isinstance(other, cls):
      return other.source_location
    if isinstance(other, clx.SourceLocation):
      return other
    raise NotImplementedError(type(other))

@functools.total_ordering
class SourceRange(AttributeCache):
  """Like SourceLocation but for clx.SourceRanges"""
  __slots__ = 'source_range', 'translation_unit', '_end', '_start'

  def __init__(self, source_range, tu=None):
    if isinstance(source_range, type(self)):
      if tu is None:
        tu = source_range.translation_unit
      else:
        assert source_range.translation_unit is None, 'Both input tu and input source_range have valid Translation Units'

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

  def __hash__(self):
    return hash(self.__start()) ^ hash(self.__end())

  def __repr__(self):
    return f'<self:{object.__repr__(self)}, tu: {self.translation_unit}, source range: {self.source_range}>'

  def __getattr__(self, attr):
    return super().get_cached(attr, getattr, self.source_range, attr)

  def __eq__(self, other):
    return self is other or self.source_range.__eq__(self.as_clang_source_range(other))

  def __lt__(self, other):
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
      return NotImplemented
    self_end = self.__end()
    if self_end == other:
      return self.__start() < other
    return self_end < other

  def __contains__(self, other):
    def contains(loc):
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

  def __len__(self):
    return self.__end().offset - self.__start().offset

  def __getitem__(self, idx):
    return super().get_cached(
      '__raw_src', _util.get_raw_source_from_source_range, self
    ).splitlines()[idx]

  def __start(self):
    if self._start is None:
      self._start = SourceLocation.cast(self.start)
    return self._start

  def __end(self):
    if self._end is None:
      self._end = SourceLocation.cast(self.end)
    return self._end

  @classmethod
  def cast(cls, other, **kwargs):
    if isinstance(other, cls):
      return other
    if isinstance(other, clx.SourceRange):
      return cls(other, **kwargs)
    raise NotImplementedError(type(other))

  @classmethod
  def from_locations(cls, left, right, tu=None):
    if tu is None:
      attr = 'translation_unit'
      tu   = getattr(left, attr, None)
      if tu is None:
        tu = getattr(right, attr, None)
    as_clang_sl = SourceLocation.as_clang_source_location
    assert left.offset <= right.offset
    return cls(clx.SourceRange.from_locations(as_clang_sl(left), as_clang_sl(right)), tu=tu)

  @classmethod
  def from_positions(cls, tu, line_left, col_left, line_right, col_right):
    filename = SourceLocation.get_filename_from_tu(tu)
    from_pos = clx.SourceLocation.from_position
    begin    = from_pos(tu, filename, line_left, col_left)
    end      = from_pos(tu, filename, line_right, col_right)
    return cls(clx.SourceRange.from_locations(begin, end), tu=tu)

  @classmethod
  def as_clang_source_range(cls, other):
    if isinstance(other, cls):
      return other.source_range
    if isinstance(other, clx.SourceRange):
      return other
    raise NotImplementedError(type(other))

  @classmethod
  def merge(cls, left, right, **kwargs):
    cast  = SourceLocation.cast
    start = min(cast(left.start), cast(right.start))
    end   = max(cast(left.end),   cast(right.end))
    return cls.from_locations(start, end, **kwargs)

  def merge_with(self, other):
    return self.merge(self, other, tu=self.translation_unit)

  def overlaps(self, other):
    end = self.__end()
    if isinstance(other, type(self)):
      return end >= other.__start() and other.__end() >= self.__start()
    cast = SourceLocation.cast
    return end >= cast(other.start) and cast(other.end) >= self.__start()

  def resized(self, lbegin=0, lend=0, cbegin=0, cend=0):
    """
    return a resized SourceRange, if the sourceRange was resized it is a new object

    lbegin - number of lines to increment or decrement self.start.lines by
    lend   - number of lines to increment or decrement self.end.lines by
    cbegin - number of columns to increment or decrement self.start.colummn by, None for BOL
    cend   - number of columns to increment or decrement self.end.colummn by, None for EOL
    """
    start = self.start
    if cbegin is None:
      cbegin = -start.column + 1
    if cend == 0 and lbegin + lend + cbegin == 0:
      return self # nothing to do

    end    = self.end
    endcol = -1 if cend is None else end.column + cend # -1 is EOL
    return self.from_positions(
      self.translation_unit, start.line + lbegin, start.column + cbegin, end.line + lend, endcol
    )


  @functools.lru_cache
  def raw(self, *args, **kwargs):
    return _util.get_raw_source_from_source_range(self, *args, **kwargs)

  @functools.lru_cache
  def formatted(self, *args, **kwargs):
    return _util.get_formatted_source_from_source_range(self, *args, **kwargs)

  def view(self, *args, **kwargs):
    kwargs.setdefault('num_context', 5)
    return print(self.formatted(*args, **kwargs))
