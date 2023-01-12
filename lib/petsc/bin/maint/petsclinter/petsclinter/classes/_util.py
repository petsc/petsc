#!/usr/bin/env python3
"""
# Created: Mon Jun 20 19:42:53 2022 (-0400)
# @author: Jacob Faibussowitsch
"""
import functools

def verbose_print(*args, **kwargs):
    """
    filter predicate for show_ast: show all
    """
    return True

def no_system_includes(cursor, level, **kwargs):
    """
    filter predicate for show_ast: filter out verbose stuff from system include files
    """
    return level != 1 or (
      cursor.location.file is not None and not cursor.location.file.name.startswith('/usr/include')
    )

def only_files(cursor, level, **kwargs):
  """
  filter predicate to only show ast defined in file
  """
  filename = kwargs['filename']
  return level != 1 or (cursor.location.file is not None and cursor.location.file.name == filename)

# A function show(level, *args) would have been simpler but less fun
# and you'd need a separate parameter for the AST walkers if you want it to be exchangeable.
class Level(int):
  """
  represent currently visited level of a tree
  """
  def view(self, *args):
    """
    pretty print an indented line
    """
    return '  '*self+' '.join(map(str, args))

  def __add__(self, inc):
    """
    increase number of tabs and newlines
    """
    return Level(super().__add__(inc))

def check_valid_type(t):
  import clang.cindex as clx

  return t.kind != clx.TypeKind.INVALID

def fully_qualify(t):
  q = set()
  if t.is_const_qualified(): q.add('const')
  if t.is_volatile_qualified(): q.add('volatile')
  if t.is_restrict_qualified(): q.add('restrict')
  return q

def view_type(t, level, title):
  """
  pretty print type AST
  """
  retList = [level.view(title, str(t.kind), ' '.join(fully_qualify(t)))]
  if check_valid_type(t.get_pointee()):
    retList.extend(view_type(t.get_pointee(), level + 1, 'points to:'))
  return retList

def view_ast_from_cursor(cursor, pred=verbose_print, level=Level(), max_depth=-1, **kwargs):
  """
  pretty print cursor AST
  """
  ret_list = []
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
def read_file_lines_cached(*args, **kwargs):
  with open(*args, **kwargs) as fd:
    return fd.readlines()

def get_raw_source_from_source_range(source_range, num_before_context=0, num_after_context=0, num_context=0, trim=False, tight=False):
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

def get_raw_source_from_cursor(cursor, **kwargs):
  return get_raw_source_from_source_range(cursor.extent, **kwargs)

def get_formatted_source_from_source_range(source_range, num_before_context=0, num_after_context=0, num_context=0, view=False, highlight=True, trim=True):
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
    underline     = begin_offset * ' ' + len_underline * '^'

  line_list = []
  raw_lines = read_file_lines_cached(begin.file.name, 'r')[lo_bound - 1:hi_bound]
  for line_file, line in enumerate(raw_lines, start=lo_bound):
    indicator = '>' if (line_begin <= line_file <= line_end) else ' '
    prefix    = f'{indicator} {line_file: <{max_width}}: '
    line_list.append((prefix, line))
    if highlight and (line_file == line_begin):
      line_list.append((' ' * len(prefix), underline))
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

def get_formatted_source_from_cursor(cursor, **kwargs):
  return get_formatted_source_from_source_range(cursor.extent, **kwargs)

def view_cursor_full(cursor, **kwargs):
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
