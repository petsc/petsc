#!/usr/bin/env python3
"""
# Created: Mon Jun 20 19:42:53 2022 (-0400)
# @author: Jacob Faibussowitsch
"""
from __future__ import annotations

import functools

from .._typing import *

from ..util._color import Color

def verbose_print(*args, **kwargs) -> bool:
  """filter predicate for show_ast: show all"""
  return True

def no_system_includes(cursor: CursorLike, level: IndentLevel, **kwargs) -> bool:
  """
  filter predicate for show_ast: filter out verbose stuff from system include files
  """
  return level != 1 or (
    cursor.location.file is not None and not cursor.location.file.name.startswith('/usr/include')
  )

def only_files(cursor: CursorLike, level: IndentLevel, **kwargs) -> bool:
  """
  filter predicate to only show ast defined in file
  """
  filename = kwargs['filename']
  return level != 1 or (cursor.location.file is not None and cursor.location.file.name == filename)

# A function show(level, *args) would have been simpler but less fun
# and you'd need a separate parameter for the AST walkers if you want it to be exchangeable.
class IndentLevel(int):
  """
  represent currently visited level of a tree
  """
  def view(self, *args) -> str:
    """
    pretty print an indented line
    """
    return '  '*self+' '.join(map(str, args))

  def __add__(self, inc: int) -> IndentLevel:
    """
    increase number of tabs and newlines
    """
    return IndentLevel(super().__add__(inc))

def check_valid_type(t: clx.Type) -> bool:
  import clang.cindex as clx # type: ignore[import]

  return not t.kind == clx.TypeKind.INVALID

def fully_qualify(t: clx.Type) -> list[str]:
  q = []
  if t.is_const_qualified(): q.append('const')
  if t.is_volatile_qualified(): q.append('volatile')
  if t.is_restrict_qualified(): q.append('restrict')
  return q

def view_type(t: clx.Type, level: IndentLevel, title: str) -> list[str]:
  """
  pretty print type AST
  """
  ret_list = [level.view(title, str(t.kind), ' '.join(fully_qualify(t)))]
  if check_valid_type(t.get_pointee()):
    ret_list.extend(view_type(t.get_pointee(), level + 1, 'points to:'))
  return ret_list

def view_ast_from_cursor(cursor: CursorLike, pred: Callable[..., bool] = verbose_print, level: IndentLevel = IndentLevel(), max_depth: int = -1, **kwargs) -> list[str]:
  """
  pretty print cursor AST
  """
  ret_list: list[str] = []
  if max_depth >= 0:
    if int(level) > max_depth:
      return ret_list
  if pred(cursor, level, **kwargs):
    ret_list.append(level.view(cursor.kind, cursor.spelling, cursor.displayname, cursor.location))
    if check_valid_type(cursor.type):
      ret_list.extend(view_type(cursor.type, level + 1, 'type:'))
      ret_list.extend(view_type(cursor.type.get_canonical(), level + 1, 'canonical type:'))
    for c in cursor.get_children():
      ret_list.extend(
        view_ast_from_cursor(c, pred=pred, level=level + 1, max_depth=max_depth, **kwargs)
      )
  return ret_list

# surprise, surprise, we end up reading the same files over and over again when
# constructing the error messages and diagnostics and hence we make about a 8x performance
# improvement by caching the files read
@functools.lru_cache
def read_file_lines_cached(*args, **kwargs) -> list[str]:
  with open(*args, **kwargs) as fd:
    ret: list[str] = fd.readlines()
  return ret

def get_raw_source_from_source_range(source_range: SourceRangeLike, num_before_context: int = 0, num_after_context: int = 0, num_context: int = 0, trim: bool = False, tight: bool = False) -> str:
  num_before_context   = num_before_context if num_before_context else num_context
  num_after_context    = num_after_context  if num_after_context  else num_context
  rstart, rend         = source_range.start, source_range.end
  line_begin, line_end = rstart.line, rend.line
  lobound              = max(1, line_begin - num_before_context)
  hibound              = line_end + num_after_context

  line_list = read_file_lines_cached(rstart.file.name, 'r')[lobound - 1:hibound]

  if tight:
    assert line_begin == line_end
    # index into line_list where our actual line starts if we have context
    loidx, hiidx = line_begin - lobound, hibound - line_begin
    cbegin, cend = rstart.column - 1, rend.column - 1
    if loidx == hiidx:
      # same line, then we need to do it in 1 step to keep the indexing correct
      line_list[loidx] = line_list[loidx][cbegin:cend]
    else:
      line_list[loidx] = line_list[loidx][cbegin:]
      line_list[hiidx] = line_list[hiidx][:cend]
  # Find number of spaces to remove from beginning of line based on lowest.
  # This keeps indentation between lines, but doesn't start the string halfway
  # across the screeen
  if trim:
    min_spaces = min(len(s) - len(s.lstrip()) for s in line_list if s.replace('\n', ''))
    return '\n'.join([s[min_spaces:].rstrip() for s in line_list])
  return ''.join(line_list)

def get_raw_source_from_cursor(cursor: CursorLike, **kwargs) -> str:
  return get_raw_source_from_source_range(cursor.extent, **kwargs)

def get_formatted_source_from_source_range(source_range: SourceRangeLike, num_before_context: int = 0, num_after_context: int = 0, num_context: int = 0, view: bool = False, highlight: bool = True, trim: bool = True) -> str:
  num_before_context   = num_before_context if num_before_context else num_context
  num_after_context    = num_after_context  if num_after_context  else num_context
  begin, end           = source_range.start, source_range.end
  line_begin, line_end = begin.line, end.line

  lo_bound  = max(1, line_begin - num_before_context)
  hi_bound  = line_end + num_after_context
  max_width = len(str(hi_bound))

  if highlight:
    symbol_begin  = begin.column - 1
    symbol_end    = end.column - 1
    begin_offset  = max(symbol_begin, 0)
    len_underline = max(abs(max(symbol_end, 1) - begin_offset), 1)
    underline     = begin_offset * ' ' + Color.bright_yellow() + len_underline * '^' + Color.reset()

  line_list = []
  raw_lines = read_file_lines_cached(begin.file.name, 'r')[lo_bound - 1:hi_bound]
  for line_file, line in enumerate(raw_lines, start=lo_bound):
    indicator = '>' if (line_begin <= line_file <= line_end) else ' '
    prefix    = f'{indicator} {line_file: <{max_width}}: '
    if highlight and (line_file == line_begin):
      line = f'{line[:symbol_begin]}{Color.bright_yellow()}{line[symbol_begin:symbol_end]}{Color.reset()}{line[symbol_end:]}'
      line_list.extend([
        (prefix, line),
        (' ' * len(prefix), underline)
      ])
    else:
      line_list.append((prefix, line))
  # Find number of spaces to remove from beginning of line based on lowest.
  # This keeps indentation between lines, but doesn't start the string halfway
  # across the screen
  if trim:
    try:
      min_spaces = min(len(s) - len(s.lstrip(' ')) for _, s in line_list if s.replace('\n', ''))
    except:
      min_spaces = 0
  else:
    min_spaces = 0
  src_str = '\n'.join(p + s[min_spaces:].rstrip() for p, s in line_list)
  if view:
    print(src_str)
  return src_str

def get_formatted_source_from_cursor(cursor: CursorLike, **kwargs) -> str:
  return get_formatted_source_from_source_range(cursor.extent, **kwargs)

def view_cursor_full(cursor: CursorLike, **kwargs) -> list[str]:
  ret = [
    f'Spelling:        {cursor.spelling}',
    f'Type:            {cursor.type.spelling}',
    f'Kind:            {cursor.kind}',
    f'Storage Class:   {cursor.storage_class}',
    f'Linkage:         {cursor.linkage}',
  ]
  try:
    ret.append('Arguments:       '+' '.join([a.displayname for a in cursor.get_arguments()]))
  except AttributeError:
    pass
  try:
    ret.append(f'Semantic Parent: {cursor.semantic_parent.displayname}')
  except AttributeError:
    pass
  try:
    ret.append(f'Lexical Parent:  {cursor.lexical_parent.displayname}')
  except AttributeError:
    pass
  ret.append('Children:          '+' '.join([c.spelling for c in cursor.get_children()]))
  ret.append(get_formatted_source_from_cursor(cursor, num_context=2))
  ret.append('------------------ AST View: ------------------')
  ret.extend(view_ast_from_cursor(cursor, **kwargs))
  return ret
