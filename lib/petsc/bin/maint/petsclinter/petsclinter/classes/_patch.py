#!/usr/bin/env python3
"""
# Created: Mon Jun 20 19:33:25 2022 (-0400)
# @author: Jacob Faibussowitsch
"""
from ._src_pos import SourceRange

class Delta:
  __slots__ = 'value', 'extent', 'offset'

  def __init__(self, value, extent, ctxlines):
    self.value  = str(value)
    self.extent = extent
    begin       = extent.start
    self.offset = begin.offset - begin.column - sum(
      map(len, extent.raw(num_context=ctxlines).splitlines(True)[:ctxlines])
    ) + 1
    return

  def deleter(self):
    return self.value == ''

  def apply(self, src, delta):
    extent        = self.extent
    offset        = delta - self.offset
    begin_offset  = extent.start.offset + offset
    end_offset    = extent.end.offset + offset
    new_src       = f'{src[:begin_offset]}{self.value}{src[end_offset:]}'
    new_delta     = delta + len(new_src) - len(src)
    return new_src, new_delta

  def is_deletion_superset_of(self, other):
    """
    determine if self's change deletes all of other's extent, in which case other is a pointless
    delta and can be discarded
    """
    return self.deleter() and other.extent in self.extent

class Patch:
  __global_counter = 0
  __slots__        = 'extent', 'ctxlines', 'deltas', 'weak_data', '_cache', 'id'

  def __init__(self, src_range, value, contextlines=2):
    self.extent    = SourceRange.cast(src_range)
    self.ctxlines  = contextlines
    self.deltas    = (Delta(value, self.extent, self.ctxlines),)
    self.weak_data = []
    self._cache    = {}
    self.id        = Patch.__global_counter
    Patch.__global_counter += 1
    return

  @classmethod
  def from_cursor(cls, cursor, value, **kwargs):
    return cls(cursor.extent, value, **kwargs)

  def _make_source(self):
    return self.extent.raw(num_context=self.ctxlines)

  def _get_cached(self, attr, func, *args, **kwargs):
    cache = self._cache
    if attr in cache:
      return cache[attr]
    new_val = cache[attr] = func(*args, **kwargs)
    return new_val

  def _contiguous_extent(self):
    """
    does my extent (which is the union of the extents of my all my deltas) have no holes?
    """
    cache_entry = 'contiguous'
    deltas      = self.deltas
    if len(deltas) == 1:
      return self._cache.setdefault(cache_entry, True)
    return self._get_cached(
      cache_entry, all, (p.extent.overlaps(c.extent) for p, c in zip(deltas[:-1], deltas[1:]))
    )


  def discard(self):
    """
    drops the error messages corresponding to this patch from the linter
    """
    for weak_list in self.weak_data:
      elist = weak_list[0]()
      if elist is not None:
        idx = elist[3].index(self.id)
        if idx < 0:
          reason = '-1 is the default value' if idx == -1 else 'unknown negative idx value'
          raise RuntimeError(f'bad weakref id {idx} for patch, {reason}')
        del elist[1][idx] # delete the error message
        del elist[2][idx] # delete the patch indicator
        del elist[3][idx] # mark the fact we have deleted ourselves
    return

  def attach(self, *args):
    """
    attach the list and index into the linter error list corresponding to this patch
    """
    self.weak_data.append(args)
    return

  def is_deletion_superset_of(self, other):
    """
    determine if any of self's deltas delete all of other's extent, in which case other is a
    pointless patch and its error messages can be discarded
    """
    assert not self is other
    oextent = other.extent
    if oextent in self.extent:
      deltas = self.deltas
      # first check if any one delta deletes all of other, then check if all deltas are deleters,
      return any(d.deleter() and oextent in d.extent for d in deltas) or \
        (self._contiguous_extent() and all(d.deleter() for d in deltas))
    return False

  @staticmethod
  def cull_deltas(deltas):
    deltas = tuple(deltas)
    return tuple([
      d_i for d_i in deltas if not any(
        d_j.is_deletion_superset_of(d_i) for d_j in deltas if d_j is not d_i
      )
    ])

  def merge(self, other):
    """
    Merge a patch with another patch. If either of the patches is a 'deletion superset' of the other
    the redundant patch is discarded from the linter.
    """
    if not isinstance(other, type(self)):
      raise ValueError(type(other))

    if self is other:
      return self

    if self.is_deletion_superset_of(other):
      other.discard()
      return self

    if other.is_deletion_superset_of(self):
      self.discard()
      return other

    assert self._make_source() == other._make_source(), 'Need to update offset calculation to handle arbitrary src'
    assert self.ctxlines == other.ctxlines, 'Need to update ctxlines to handle arbitrary src'

    self.extent = self.extent.merge_with(other.extent)
    # uncomment when it handles arbitrary source
    # self.src    = self._make_source()
    # fixes and ranges must be applied in order
    combined    = self.cull_deltas(self.deltas + other.deltas)
    argsort     = sorted(range(len(combined)), key=lambda x: combined[x].extent)
    self.deltas = tuple(combined[i] for i in argsort)
    self._cache = {}
    self.weak_data.extend(other.weak_data)
    return self

  def collapse(self):
    """
    Collapses a the list of fixes and produces into a modified output
    """
    # Fixes probably should not overwrite each other (for now), so we error out, but this
    # is arguably a completely valid case. I just have not seen an example of it that I
    # can use to debug with yet.
    cache_entry = 'fixed'
    cache       = self._cache
    if cache_entry in cache:
      return cache[cache_entry]

    idx_delta = 0
    src       = old_src = self._make_source()
    for delta in self.deltas:
      src, idx_delta = delta.apply(src, idx_delta)

    assert src != old_src, 'Patch did not seem to do anything!'
    cache[cache_entry] = src
    return src

  def view(self):
    import difflib
    import petsclinter as pl

    idx_delta = 0
    before    = self._make_source()
    for i, delta in enumerate(self.deltas):
      pl.sync_print('Delta:', i, f'({delta})')
      after, idx_delta = delta.apply(before, idx_delta)
      pl.sync_print(''.join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True), fromfile='Original', tofile='Modified'
      )))
      before = after
    return
