#!/usr/bin/env python3
"""
# Created: Thu Nov 17 11:50:52 2022 (-0500)
# @author: Jacob Faibussowitsch
"""
import re
import difflib
import clang.cindex as clx

from .._diag    import DiagnosticManager, Diagnostic
from .._src_pos import SourceRange
from .._cursor  import Cursor
from .._patch   import Patch
from .._path    import Path

from ._doc_section_base import (
  DescribableItem, SectionBase, ParameterList, Prose, VerbatimBlock, InlineList
)

from ...util._clang import clx_enum_type_kinds

"""
==========================================================================================
Derived Classes

==========================================================================================
"""
class DefaultSection(SectionBase):
  @classmethod
  def __diagnostic_prefix__(cls, *flags):
    return DiagnosticManager.flag_prefix(super())('', *flags)

  def __init__(self, *args, **kwargs):
    kwargs.setdefault('name', 'UNKNOWN')
    kwargs.setdefault('titles', ('__UNKNOWN_SECTION__',))
    super().__init__(*args, **kwargs)
    return

@DiagnosticManager.register(
  ('matching-symbol-name','Verify that description matches the symbol name'),
  ('missing-description','Verify that a synopsis has a description'),
  ('wrong-description-separator','Verify that synopsis uses the right description separator'),
  ('verbose-description','Verify that synopsis descriptions don\'t drone on and on'),
  ('macro-explicit-synopsis-missing','Verify that macro docstrings have an explicit synopsis section'),
  ('macro-explicit-synopsis-valid-header','Verify that macro docstrings with explicit synopses have the right header include')
)
class Synopsis(SectionBase):
  __header_include_finder = re.compile(r'\s*#\s*include\s*[<"](.*)[>"]')
  __sowing_include_finder = re.compile(
    __header_include_finder.pattern + r'\s*/\*\s*I\s*(["<].*[>"])\s*I\s*\*/.*'
  )

  @classmethod
  def __diagnostic_prefix__(cls, *flags):
    return DiagnosticManager.flag_prefix(super())('synopsis', *flags)

  def __init__(self, *args, **kwargs):
    kwargs.setdefault('name', 'synopsis')
    kwargs.setdefault('required', True)
    kwargs.setdefault('keywords', ('Synopsis', 'Not Collective'))
    super().__init__(*args, **kwargs)
    return

  @staticmethod
  def barren():
    return False # synoposis is never barren

  def setup(self, ds, *args, **kwargs):
    cursor_name = ds.cursor.name
    lo_name     = cursor_name.casefold()
    items       = [{'name' : (None, None), 'blurb' : [], 'synopsis' : []}]

    class Inspector:
      __slots__ = 'found_description', 'found_synopsis', 'is_enum', 'capturing'

      def __init__(self, cursor):
        self.found_description = False
        self.found_synopsis    = False
        self.is_enum           = cursor.type.kind in clx_enum_type_kinds
        self.capturing         = False
        return

      def __call__(self, *args, **kwargs):
        if not self.found_description:
          self.description(*args, **kwargs)
        if self.is_enum:
          self.enum(*args, **kwargs)
        elif not self.found_synopsis:
          self.synopsis(*args, **kwargs)
        return

      def description(self, loc, line, *args, **kwargs):
        """
        Look for the '<NAME> - description' block in a synopsis
        """
        startline = loc.start.line
        if self.capturing:
          assert self.capturing == 'description', 'Mixing blurb and synopsis capture?'
          item = line.strip()
          if item:
            items[0]['blurb'].append((ds.make_source_range(item, line, startline), item))
          else:
            self.capturing        = False
            self.found_description = True
        else:
          pre, dash, rest = line.partition('-')
          if dash:
            rest = rest.strip()
          elif lo_name in line.casefold():
            pre  = cursor_name
            rest = line.split(cursor_name, maxsplit=1)[1].strip()
          else:
            return
          assert len(items) == 1
          item = pre.strip()
          items[0]['name'] = (ds.make_source_range(item, line, startline), item)
          items[0]['blurb'].append((ds.make_source_range(rest, line, startline), rest))
          self.capturing = 'description' # now capture the rest of the blurb
        return

      def synopsis(self, loc, line, *args, **kwargs):
        """
        Look for the Synopsis: heading and block in a synopsis
        """
        lstrp = line.strip()
        if 'synopsis:' in lstrp.casefold():
          self.capturing = 'synopsis'
        if self.capturing == 'synopsis':
          # don't want to accidentally capture the blurb
          if not lstrp:
            # reached the end of the synopsis block
            self.found_synopsis = True
            self.capturing      = False
            return
          items[0]['synopsis'].append((ds.make_source_range(lstrp, line, loc.start.line), line))
        return

      def enum(self, loc, line, *args, **kwargs):
        lstr = line.lstrip()
        # check that '-' is in the line since some people like to use entire blocks of $'s
        # to describe a single enum value...
        if lstr.startswith('$') and '-' in lstr:
          assert len(items) # we should have already found the symbol name
          name = lstr[1:].split(maxsplit=1)[0].strip()
          items.append((ds.make_source_range(name, line, loc.start.line), line, 0))
        return

    inspector = Inspector(ds.cursor)
    super().setup(ds,*args, inspect_line=inspector, **kwargs)

    if inspector.is_enum:
      def check_enum_starts_with_dollar(self, ds, items):
        for key, opts in sorted(items.items()):
          if len(opts) < 1:
            raise RuntimeError(f'number of options {len(opts)} < 1, key: {key}, items: {items}')
          for opt in opts:
            self._check_opt_starts_with(ds, opt, 'Enum', '$')
        return items


      params      = ParameterList(name='enum params', prefixes=('$',))
      param_lines = items[1:]
      assert param_lines, 'No parameter lines in enum description!'
      params.consume(param_lines)
      params.setup(ds, *args, parameter_list_prefix_check=check_enum_starts_with_dollar, **kwargs)
      # shuffle the enum values up
      params.items = {k + 1 : v for k, v in params.items.items()}
      # reinsert the heading
      params.items[0] = items[0]
      self.items      = params
    else:
      self.items = tuple(items)
    return

  def _check_missing_description(self, docstring, cursor, symbol):
    """
    Ensure that a synopsis is present and properly formatted with Cursor - description
    """
    if symbol is None:
      diag = docstring.make_diagnostic(
        self.diags.missing_description, 'Docstring missing synopsis', self.extent, highlight=False
      ).add_note(f"Expected '{cursor.name} - a very useful description'")
      docstring.add_error_from_diagnostic(diag)
    return

  def _check_macro_synopsis(self, linter, cursor, docstring, explicit_synopsis):
    """
    Ensure that synopsese of macros exist and have proper prototypes
    """
    if not (len(explicit_synopsis) or docstring.Modifier.FLOATING in docstring.type_mod):
      # we are missing the synopsis section entirely
      with open(cursor.location.file.name) as fh:
        lines = [l.strip() for l in fh if l.lstrip().startswith('#') and 'include' in l and '/*' in l]
      lines = [l.group(2).strip() for l in filter(None, map(self.__sowing_include_finder.match, lines))]

      try:
        include_header = lines[0]
      except IndexError:
        include_header = '"some_header.h"'
      args = ', '.join(
        f'{c.derivedtypename} {c.name}' for c in linter.get_argument_cursors(cursor)
      )
      extent      = docstring._attr['sowing_char_range']
      macro_ident = docstring.make_source_range('M', extent[0], extent.start.line)
      diag        = docstring.make_diagnostic(
        self.diags.macro_explicit_synopsis_missing,
        f'Macro docstring missing an explicit synopsis {Diagnostic.FLAG_SUBST}',
        self.extent, highlight=False
      ).add_note(
        '\n'.join([
          'Expected:',
          '',
          '  Synopsis:',
          f'  #include {include_header}',
          f'  {cursor.result_type.spelling} {cursor.name}({args})'
        ])
      ).add_note(
        f'symbol marked as macro here, but it is ambiguous if this symbol is really meant to be a macro or not\n{macro_ident.formatted(num_context=2)}',
        location=macro_ident.start
      )
      linter.add_error_from_cursor(cursor, diag)
      return False # the code should not check the name

    # search the explicit docstring for the
    # #include <header.h>
    # line
    header_name, header_loc = None, None
    for loc, line in explicit_synopsis:
      stripped = line.strip()
      if not stripped or stripped.endswith(':') or stripped.casefold().startswith('synopsis'):
        continue
      found = self.__header_include_finder.match(stripped)
      if found:
        header_name = found.group(1)
        header_loc  = loc
        break

    if header_name is None:
      print(80*'=')
      docstring.extent.view()
      print('')
      print('Dont know how to handle no header name yet')
      print(80*'=')
      return False # don't know how to handle this

    # OK found it, now find the actual file. Clang unfortunately cannot help us here since
    # it does not pick up header that are in the precompiled header (which chances are,
    # this one is). So we search for it ourselves
    def find_header(directory):
      header_path = directory / header_name
      if header_path.exists():
        return header_path.resolve()
      return None

    header_path = None
    for flag_path in (Path(flag[2:]) for flag in linter.flags if flag.startswith('-I')):
      header_path = find_header(flag_path)
      if header_path is not None:
        break

    if header_path is None:
      header_path = find_header(Path(str(docstring.extent.start.file)).parent)

    assert header_path
    fn_name = self.items[0]['name'][1]
    decls   = [line for line in header_path.read_text().splitlines() if fn_name in line]
    if not decls:
      # the name was not in the header, so the docstring is wrong
      mess = f"Macro docstring explicit synopsis appears to have incorrect include line. Could not locate '{fn_name}()' in '{header_name}'. Are you sure that's where it lives?"
      diag = self.diags.macro_explicit_synopsis_valid_header
      docstring.add_error_from_source_range(diag,mess,header_loc)
      return False

    cursor_spelling = cursor.spelling
    if len(decls) > 1 and len(cursor_spelling) > len(fn_name):
      decls2 = [
        d.replace(cursor_spelling, '') for d in decls if fn_name in d.replace(cursor_spelling, '')
      ]
      # We removed the longer of the two names, and now don't have any matches. Maybe
      # that means it's not defined in this header?
      if not decls2:
        mess = f'Removing {cursor.spelling} from the decl list:\n{decls}\nhas emptied it. Maybe this means {fn_name} is not defined in {header_path}?'
        raise RuntimeError(mess)
      decls = decls2
    # the only remaining item should be the macro (or maybe function), note this
    # probably needs a lot more postprocessing
    assert len(decls) == 1
    return False

  def _check_symbol_matches_synopsis_name(self, docstring, cursor, loc, symbol):
    """
    Ensure that the name of the symbol matches that of the name in the custom synopsis (if provided)
    """
    if symbol != cursor.name:
      if len(difflib.get_close_matches(symbol, [cursor.name], n=1)):
        mess  = f"Docstring name '{symbol}' does not match symbol. Assuming you meant '{cursor.name}'"
        patch = Patch(loc, cursor.name)
      else:
        mess  = f"Docstring name '{symbol}' does not match symbol name '{cursor.name}'"
        patch = None
      docstring.add_error_from_source_range(self.diags.matching_symbol_name, mess, loc, patch=patch)
    return

  def _check_synopsis_description_separator(self, docstring, start_line):
    """
    Ensure that the synopsis uses the proper separator
    """
    for sloc, sline, _ in self.lines():
      if sloc.start.line == start_line:
        item = DescribableItem.cast(sline, sep='-').check(docstring, self, sloc, expected_sep='-')
        break
    return

  def _check_blurb_length(self, docstring, cursor, items):
    """
    Ensure the blurb is not too wordy
    """
    total_blurb = [line for _, line in items[0]['blurb']]
    word_count  = sum(len(l.split()) for l in total_blurb)
    char_count  = sum(map(len, total_blurb))

    max_char_count = 250
    max_word_count = 40
    if char_count > max_char_count and word_count > max_word_count:
      mess = f"Synopsis for '{cursor.name}' is too long (must be at most {max_char_count} characters or {max_word_count} words), consider moving it to Notes. If you can't explain it simply, then you don't understand it well enough!"
      docstring.add_error_from_source_range(
        self.diags.verbose_description, mess, self.extent, highlight=False
      )
    return

  def check(self, linter, cursor, docstring):
    super().check(linter, cursor, docstring)

    items = self.items
    if isinstance(items, tuple):
      pass # normal synopsis
    elif isinstance(items, ParameterList):
      items = items.items # enum synopsis
    else:
      raise ValueError(type(items))
    loc, symbol = items[0]['name']

    self._check_missing_description(docstring, cursor, symbol)
    if loc is None:
      # missing synopsis entirely
      return

    if docstring.Modifier.MACRO in docstring.type_mod:
      # chances are that if it is a macro then the name won't match
      self._check_macro_synopsis(linter, cursor, docstring, items[0]['synopsis'])
      return

    self._check_symbol_matches_synopsis_name(docstring, cursor, loc, symbol)
    self._check_synopsis_description_separator(docstring, loc.start.line)
    self._check_blurb_length(docstring, cursor, items)
    return

@DiagnosticManager.register(
  ('parameter-documentation','Verify that if a, b, c are documented then the function exactly has parameters a, b, and c and vice versa'),
  ('fortran-interface','Verify that functions needing a custom fortran interface have the correct sowing indentifiers'),
)
class FunctionParameterList(ParameterList):
  @classmethod
  def __diagnostic_prefix__(cls, *flags):
    return DiagnosticManager.flag_prefix(super())('func', *flags)

  def __init__(self, *args, **kwargs):
    kwargs.setdefault('titles', ('Input Parameter', 'Output Parameter', 'Calling sequence'))
    kwargs.setdefault('keywords', ('Input', 'Output'))
    super().__init__(*args, **kwargs)
    return

  def _check_fortran_interface(self, docstring, fnargs):
    """
    Ensure that functions which require a custom fortran interface are correctly tagged with 'C' sowing
    designator
    """
    from ...util._clang import clx_char_type_kinds, clx_function_type_kinds

    requires_c = []
    for arg in fnargs:
      canon = arg.type.get_canonical()
      kind  = canon.kind
      it    = 0
      while kind == clx.TypeKind.POINTER:
        if it >= 100:
          import petsclinter as pl
          # there is no chance that someone has a variable over 100 pointers deep, so
          # clearly something is wrong
          cursorview = '\n'.join(pl.classes._util.view_ast_from_cursor(arg))
          emess      = f'Ran for {it} iterations (>= 100) trying to get pointer type for\n{arg.error_view_from_cursor(arg)}\n{cursorview}'
          raise RuntimError(emess)
        canon = canon.get_pointee()
        kind  = canon.kind
        it   += 1

      if kind in clx_char_type_kinds:
        requires_c.append((arg, 'char pointer'))
      elif kind in clx_function_type_kinds:
        requires_c.append((arg, 'function pointer'))

    if len(requires_c):
      begin_sowing_range = docstring._attr['sowing_char_range']
      sowing_chars       = begin_sowing_range.raw(tight=True)
      if docstring.Modifier.C_FUNC not in docstring.type_mod:
        assert 'C' not in sowing_chars
        diag = docstring.make_diagnostic(
          self.diags.fortran_interface,
          f"Function requires custom fortran interface but missing 'C' from docstring header {Diagnostic.FLAG_SUBST}",
          begin_sowing_range, patch=Patch(begin_sowing_range, sowing_chars + 'C')
        )
        for reason_cursor, reason_type in requires_c:
          diag.add_note(
            f'due to {reason_type} {reason_cursor.get_formatted_blurb(nboth=1).rstrip()}',
            location=reason_cursor.extent.start
          )
        docstring.add_error_from_diagnostic(diag)
    return

  def _check_valid_param_list_from_cursor(self, linter, docstring, arg_cursors):
    """
    Ensure that the parameter list matches the documented values, and that their order is correct
    """
    if arg_cursors and not self:
      # none of the function arguments are documented
      diag = docstring.make_diagnostic(
        self.diags.parameter_documentation,
        f'Symbol parameters are all undocumented {Diagnostic.FLAG_SUBST}',
        docstring.extent, highlight=False
      ).add_note(
        docstring.make_error_message(
          'Parameters defined here',
          SourceRange.from_locations(arg_cursors[0].extent.start, arg_cursors[-1].extent.end)
        ),
        location=arg_cursors[0].extent.start
      )
      docstring.add_error_from_diagnostic(diag)
      return

    if not arg_cursors:
      # function has no arguments, so check there are no parameter docstrings, if so, we can
      # delete them
      if self and len(self.items.values()):
        mess = f"Found parameter docstring(s) but '{docstring.cursor.displayname}' has no parameters"
        docstring.add_error_from_source_range(
          self.diags.parameter_documentation, mess, self.extent,
          highlight=False, patch=Patch(self.extent, '')
        )
      return

    def get_recursive_cursor_list(cursor_list):
      new_cursor_list = []
      PARM_DECL_KIND  = clx.CursorKind.PARM_DECL
      for cursor in map(Cursor.cast, cursor_list):
        new_cursor_list.append(cursor)
        # Special handling of functions taking function pointer arguments. In this case we
        # should recursively descend and pick up the names of all the function parameters
        #
        # note the order, by appending cursor first we effectively do a depth-first search
        if cursor.type.get_pointee().kind == clx.TypeKind.FUNCTIONPROTO:
          new_cursor_list.extend(
            get_recursive_cursor_list(c for c in cursor.get_children() if c.kind == PARM_DECL_KIND)
          )
      return new_cursor_list

    arg_cursors = get_recursive_cursor_list(arg_cursors)
    arg_names   = [a.name for a in arg_cursors if a.name]
    arg_seen    = [False] * len(arg_names)
    not_found   = []

    def mark_name_as_seen(name):
      idx = 0
      while 1:
        # in case of multiple arguments of the same name, we need to loop until we
        # find an index that has not yet been found
        try:
          idx = arg_names.index(name, idx)
        except ValueError:
          return -1
        if not arg_seen[idx]:
          # argument exists and has not been found yet
          break
        # argument exists but has already been claimed
        idx += 1
      arg_seen[idx] = True
      return idx

    solitary_param_diag = self.diags.solitary_parameter
    for _, group in self.items.items():
      indices = []
      remove  = set()
      for i, (loc, descr_item, _) in enumerate(group):
        arg, sep = descr_item.arg, descr_item.sep
        if sep == ',' or ',' in arg:
          sub_args = list(map(str.strip, arg.split(',')))
          if len(sub_args) > 1:
            diag = docstring.make_diagnostic(
              solitary_param_diag,
              'Each parameter entry must be documented separately on its own line',
              docstring.make_source_range(arg, descr_item.text, loc.start.line)
            )
            if docstring.cursor.is_variadic_function():
              diag.add_note('variable argument lists should be documented in notes')
            linter.add_error_from_cursor(docstring.cursor, diag)
        elif sep == '=':
          sub_args = list(map(str.strip, arg.split(' = ')))
          if len(sub_args) > 1:
            sub_args = (sub_args[0],) # case of bad separator, only the first entry is valid
        else:
          sub_args = (arg,)

        for sub in sub_args:
          idx = mark_name_as_seen(sub)
          if idx == -1:
            # argument was not found at all
            not_found.append((sub, docstring.make_source_range(sub, descr_item.text, loc.start.line)))
            remove.add(i)
          else:
            indices.append(idx)
            DescribableItem.cast(descr_item, sep='-').check(docstring, self, loc, expected_sep='-')

      self.check_aligned_descriptions(docstring, [g for i, g in enumerate(group) if i not in remove])

    args_left = [name for seen, name in zip(arg_seen, arg_names) if not seen]
    if not_found:
      diag         = self.diags.parameter_documentation
      base_message = "Extra docstring parameter '{}' not found in symbol parameter list:\n{}"
      for i, (arg, loc) in enumerate(not_found):
        patch   = None
        message = base_message.format(arg, loc.formatted(num_context=2))
        try:
          if (len(args_left) == 1) and (i == len(not_found) - 1):
            # if we only have 1 arg left and 1 wasn't found, chances are they are meant to
            # be the same
            arg_match = args_left[0]
            if docstring.Modifier.MACRO not in docstring.type_mod:
              # furthermore, if this is not a macro then we can be certain that this is
              # indeed an error we can fix
              patch = Patch(loc, arg_match)
          else:
            arg_match = difflib.get_close_matches(arg, args_left, n=1)[0]
        except IndexError:
          pass
        else:
          match_cursor = [c for c in arg_cursors if c.name == arg_match][0]
          message      = f'{message}\n\nmaybe you meant {match_cursor.get_formatted_blurb()}'
          args_left.remove(arg_match)
          assert mark_name_as_seen(arg_match) != -1, f'{arg_match=} was not found in {arg_names=}'
        docstring.add_error_from_diagnostic(Diagnostic(diag, message, loc.start, patch=patch))

    undoc_param_diag = self.diags.parameter_documentation
    for arg in args_left:
      idx = mark_name_as_seen(arg)
      assert idx != -1, f'{arg=} was not found in {arg_names=}'
      diag = docstring.make_diagnostic(
        undoc_param_diag, f'Undocumented parameter \'{arg}\' not found in parameter section',
        self.extent, highlight=False
      ).add_note(
        docstring.make_error_message(
          f'Parameter \'{arg}\' defined here', arg_cursors[idx], num_context=1
        ),
        location=arg_cursors[idx].extent.start
      )
      docstring.add_error_from_diagnostic(diag)
    return

  def check(self, linter, cursor, docstring):
    super().check(linter, cursor, docstring)
    fnargs = linter.get_argument_cursors(cursor)

    self._check_fortran_interface(docstring, fnargs)
    self._check_valid_param_list_from_cursor(linter, docstring, fnargs)
    return

class OptionDatabaseKeys(ParameterList):
  @classmethod
  def __diagnostic_prefix__(cls, *flags):
    return DiagnosticManager.flag_prefix(super())('option-keys', *flags)

  def __init__(self, *args, **kwargs):
    kwargs.setdefault('name', 'options')
    kwargs.setdefault('titles', ('Options Database',))
    super().__init__(*args, **kwargs)
    return

  def _check_option_database_key_alignment(self, docstring):
    """
    Ensure that option database keys and their descriptions are properly aligned
    """
    for _, group in sorted(self.items.items()):
      self.check_aligned_descriptions(docstring, group)
    return

  def check(self, linter, cursor, docstring):
    super().check(linter, cursor, docstring)

    self._check_option_database_key_alignment(docstring)
    return

class Notes(Prose):
  @classmethod
  def __diagnostic_prefix__(cls, *flags):
    return DiagnosticManager.flag_prefix(super())('notes', *flags)

  def __init__(self, *args, **kwargs):
    kwargs.setdefault('name', 'notes')
    kwargs.setdefault('titles', ('Notes', 'Note'))
    super().__init__(*args, **kwargs)
    return

class DeveloperNotes(Prose):
  @classmethod
  def __diagnostic_prefix__(cls, *flags):
    return DiagnosticManager.flag_prefix(super())('dev-notes', *flags)

  def __init__(self, *args, **kwargs):
    kwargs.setdefault('name', 'developer notes')
    kwargs.setdefault('keywords', ('Developer', ))
    super().__init__(*args, **kwargs)

class References(Prose):
  @classmethod
  def __diagnostic_prefix__(cls, *flags):
    return DiagnosticManager.flag_prefix(super())('references', *flags)

  def __init__(self, *args, **kwargs):
    kwargs.setdefault('name', 'references')
    kwargs.setdefault('solitary', False)
    super().__init__(*args, **kwargs)
    return

class FortranNotes(Prose):
  @classmethod
  def __diagnostic_prefix__(cls, *flags):
    return DiagnosticManager.flag_prefix(super())('fortran-notes', *flags)

  def __init__(self, *args, **kwargs):
    kwargs.setdefault('name', 'fortran notes')
    kwargs.setdefault('keywords', ('Fortran', ))
    super().__init__(*args, **kwargs)
    return

class SourceCode(VerbatimBlock):
  @classmethod
  def __diagnostic_prefix__(cls, *flags):
    return DiagnosticManager.flag_prefix(super())('source-code', *flags)

  def __init__(self, *args, **kwargs):
    kwargs.setdefault('name', 'code')
    # kwargs.setdefault('titles', ('Example Usage', 'Example', 'Calling Sequence'))
    # kwargs.setdefault('keywords', ('Example', 'Usage', 'Sample Usage', 'Calling
    # Sequence'))
    kwargs.setdefault('titles', ('Example Usage', 'Example'))
    kwargs.setdefault('keywords', ('Example', 'Usage', 'Sample Usage'))
    super().__init__(*args, **kwargs)
    return

@DiagnosticManager.register(
  ('casefold', 'Verify that level subheadings are lower-case'),
  ('spelling', 'Verify that level subheadings are correctly spelled'),
)
class Level(InlineList):
  __slots__ = ('valid_levels',)

  def __init__(self, *args, **kwargs):
    kwargs.setdefault('name', 'level')
    kwargs.setdefault('required', True)
    super().__init__(*args, **kwargs)
    self.valid_levels = ('beginner', 'intermediate', 'advanced', 'developer', 'deprecated')
    return

  @classmethod
  def __diagnostic_prefix__(cls, *flags):
    return DiagnosticManager.flag_prefix(super())('level', *flags)

  def __do_check_valid_level_spelling(self, docstring, loc, level_name):
    if level_name in self.valid_levels:
      return # all good

    def make_sub_loc(loc, sub):
      return docstring.make_source_range(sub, loc.raw(), loc.start.line, offset=loc.start.column - 1)

    locase  = level_name.casefold()
    patch   = None
    sub_loc = loc
    if locase in self.valid_levels:
      diag  = self.diags.casefold
      mess  = f"Level subheading must be lowercase, expected '{locase}' found '{level_name}'"
      patch = Patch(loc, locase)
    else:
      diag      = self.diags.spelling
      lvl_match = difflib.get_close_matches(locase, self.valid_levels, n=1)
      if not lvl_match:
        sub_split = level_name.split(maxsplit=1)[0]
        if sub_split != level_name:
          sub_loc   = make_sub_loc(loc, sub_split)
          lvl_match = difflib.get_close_matches(sub_split.casefold(), self.valid_levels, n=1)

      if lvl_match:
        lvl_match = lvl_match[0]
        if lvl_match == 'deprecated':
          re_match = re.match(
            r'(\w+)\s*(\(\s*[sS][iI][nN][cC][eE]\s*\d+\.\d+[\.\d\s]*\))', level_name
          )
          if re_match:
            # given:
            #
            # deprecated (since MAJOR.MINOR[.PATCH])
            # ^^^^^^^^^^ ^^^^^^^^^^^^^^^^^^^^^^^^^^^
            #     |                     |
            # re_match[1]               |
            #                   re_match[2]
            #
            # check that invalid_name_match is properly formatted
            invalid_name_match = re_match[1]
            return self.__do_check_valid_level_spelling(
              docstring, make_sub_loc(loc, invalid_name_match), invalid_name_match
            )

        mess  = f"Unknown Level subheading '{level_name}', assuming you meant '{lvl_match}'"
        patch = Patch(loc, lvl_match)
      else:
        if 'level' not in loc.raw().casefold():
          return # TODO fix this with _check_level_heading_on_same_line()
        expected = ', or '.join([', '.join(self.valid_levels[:-1]), self.valid_levels[-1]])
        mess     = f"Unknown Level subheading '{level_name}', expected one of {expected}"
    docstring.add_error_from_source_range(diag, mess, sub_loc, patch=patch)
    return

  def _check_valid_level_spelling(self, docstring):
    """
    Ensure that the level values are both proper and properly spelled
    """
    for line_after_colon, sub_items in self.items:
      for loc, level_name in sub_items:
        self.__do_check_valid_level_spelling(docstring, loc, level_name)
    return

  def _check_level_heading_on_same_line(self):
    """
    Ensure that the level heading value is on the same line as Level:
    """
    return
    # TODO FIX ME, need to be able to handle the below
    for loc, line, verdict in self.lines():
      if line and ':' not in line:
        # if you get a "prevloc" and "prevline" not defined error here this means that we
        # are erroring out on the first trip round this loop and somehow have a
        # lone-standing 'beginner' or whatever without an explicit "Level:" line...
        errorMessage = f"Level values must be on the same line as the 'Level' heading, not on separate line:\n{prevloc.merge_with(loc).formatted(num_context=2, highlight=False)}"
        # This is a stupid hack to solve a multifaceted issue. Suppose you have
        # Level:
        # BLABLABLA
        # The first fix above does a tolower() transformation
        # Level:
        # blabla
        # while this fix would apply a join transformation
        # Level: BLABLA
        # See the issue already? Since we sort the transformations by line the second
        # transformation would actually end up going *first*, meaning that the lowercase
        # transformation is no longer valid for patch...

        # create a range starting at newline of previous line going until the first
        # non-space character on the next line
        delrange = SourceRange.from_positions(
          cursor.translation_unit,prevloc.end.line,-1,loc.start.line,len(line)-len(line.lstrip())
        )
        # given '  Level:\n  blabla'
        #                ^^^
        #                 |
        #              delrange
        # delete delrange from it to get '  Level: blabla'
        # TODO: make a real diagnostic here
        diag = Diagnostic(spellingDiag, errorMessage, patch=Patch(delrange, ''))
        linter.add_error_from_cursor(cursor, diag)
      prevloc  = loc
      prevline = line
    return

  def check(self, linter, cursor, docstring):
    super().check(linter, cursor, docstring)
    self._check_valid_level_spelling(docstring)
    self._check_level_heading_on_same_line()
    return

@DiagnosticManager.register(
  ('duplicate','Verify that there are no duplicate entries in seealso lists'),
  ('self-reference','Verify that seealso lists don\'t contain the current symbol name'),
  ('backticks','Verify that seealso list entries are all enclosed by \'`\''),
)
class SeeAlso(InlineList):
  def __init__(self, *args, **kwargs):
    kwargs.setdefault('name', 'seealso')
    kwargs.setdefault('required', True)
    kwargs.setdefault('titles', ('.seealso',))
    kwargs.setdefault('special_chars', '`')
    super().__init__(*args, **kwargs)
    return

  @classmethod
  def __diagnostic_prefix__(cls, *flags):
    return DiagnosticManager.flag_prefix(super())('seealso', *flags)

  @staticmethod
  def transform(text):
    return text.casefold()

  @staticmethod
  def __make_deletion_patch(loc, text, look_behind):
    """
    first(),    second(),      third

    Extend source range of 'second' so that deleting it yields

    first(), third
    """
    raw = loc.raw().rstrip('\n')
    col = loc.start.column - 1
    # str.partition won't work here since it returns the first instance of 'sep', which in
    # our case might be the first instance of the value rather than the duplicate we just
    # found
    post = raw[col + len(text):]
    # get the number of characters between us and next alphabetical character
    cend = len(post) - len(post.lstrip(', '))
    if look_behind:
      # look to remove comma and space the entry behind us
      pre    = raw[:col]
      cbegin = len(pre.rstrip(', ')) - len(pre) # note intentionally negative value
      assert cbegin < 0
    else:
      cbegin = 0
    return Patch(loc.resized(cbegin=cbegin, cend=cend), '')

  def _check_self_referential(self, cursor, docstring, items, last_loc):
    """
    Ensure that the seealso list does not contain the name of the cursors symbol, i.e. that the
    docstring is not self-referential
    """
    item_remain = []
    symbol_name = Cursor.get_name_from_cursor(cursor)
    for line_after_colon, sub_items in items:
      for loc, text in sub_items:
        if text.rstrip('()') == symbol_name:
          mess = f"Found self-referential seealso entry '{text}'; your documentation may be good but it's not *that* good"
          docstring.add_error_from_source_range(
            self.diags.self_reference, mess, loc,
            patch=self.__make_deletion_patch(loc, text, loc == last_loc)
          )
        else:
          item_remain.append((loc, text))
    return item_remain

  def _check_enclosed_by_special_chars(self, docstring, item_remain):
    """
    Ensure that every entry in the seealso list is enclosed in backticks
    """
    def enclosed_by(string, char):
      return string.startswith(char) and string.endswith(char)

    btick = self.special_chars
    assert btick == '`'
    for loc, text in item_remain:
      if not enclosed_by(text, btick):
        docstring.add_error_from_source_range(
          self.diags.backticks, f"seealso symbol '{text}' not enclosed with '{btick}'",
          loc, patch=Patch(loc, f'{btick}{text.replace(btick, "")}{btick}')
        )
    return

  def _check_duplicate_entries(self, linter, cursor, docstring, item_remain, last_loc):
    """
    Ensure that the seealso list has no duplicate entries
    """
    seen     = {}
    dup_diag = self.diags.duplicate
    for loc, text in item_remain:
      if text not in seen:
        seen[text] = (loc, text)
        continue

      assert text
      first_seen = seen[text][0]
      diag       = docstring.make_diagnostic(
        dup_diag, f"Seealso entry '{text}' is duplicate", loc,
        patch=self.__make_deletion_patch(loc, text, loc == last_loc)
      ).add_note(
        docstring.make_error_message('first instance found here', first_seen, num_context=1),
        location=first_seen.start
      )
      linter.add_error_from_cursor(cursor, diag)
    return

  def check(self, linter, cursor, docstring):
    super().check(linter, cursor, docstring)

    if self.barren() or not self:
      return # barren

    items       = self.items
    last_loc    = items[-1][1][-1][0]
    item_remain = self._check_self_referential(cursor, docstring, items, last_loc)
    self._check_enclosed_by_special_chars(docstring, item_remain)
    self._check_duplicate_entries(linter, cursor, docstring, item_remain, last_loc)
    return
