#!/usr/bin/env python3
"""
# Created: Thu Nov 17 11:50:52 2022 (-0500)
# @author: Jacob Faibussowitsch
"""
from __future__ import annotations

import re
import difflib
import clang.cindex as clx # type: ignore[import]

from ..._typing import *

from .._diag    import DiagnosticManager, Diagnostic
from .._src_pos import SourceRange
from .._cursor  import Cursor
from .._patch   import Patch
from .._path    import Path

from ._doc_section_base import SectionBase, Synopsis, ParameterList, Prose, VerbatimBlock, InlineList

from ...util._clang import clx_char_type_kinds, clx_function_type_kinds

"""
==========================================================================================
Derived Classes

==========================================================================================
"""
class DefaultSection(SectionBase):
  @classmethod
  def __diagnostic_prefix__(cls, *flags):
    return DiagnosticManager.flag_prefix(super())('', *flags)

  def __init__(self, *args, **kwargs) -> None:
    r"""Construct a `DefaultSection`

    Parameters
    ----------
    *args :
      additional positional arguments to `SectionBase.__init__()`
    **kwargs :
      additional keyword arguments to `SectionBase.__init__()`
    """
    kwargs.setdefault('name', 'UNKNOWN_SECTION')
    kwargs.setdefault('titles', ('__UNKNOWN_SECTION__',))
    super().__init__(*args, **kwargs)
    return

class FunctionSynopsis(Synopsis):
  SynopsisItemType: TypeAlias = List[Tuple[SourceRange, str]]
  ItemsType                   = TypedDict(
    'ItemsType',
    # We want to extend Synopsis.ItemsType, this is the only way I saw how. I tried doing
    # {'synopsis' : ..., ***typing.get_type_hints(Synopsis.ItemsType)}
    #
    # but mypy barfed:
    #
    # error: Invalid TypedDict() field name  [misc]
    # {'synopsis' : List[Synopsis.ItemsEntryType], **typing.get_type_hints(Synopsis.ItemsType)}
    {
      'synopsis' : SynopsisItemType,
      'name'     : Synopsis.NameItemType,
      'blurb'    : Synopsis.BlurbItemType
    }
  )
  items: ItemsType

  class Inspector(Synopsis.Inspector):
    __slots__ = ('synopsis_items', )

    synopsis_items: FunctionSynopsis.SynopsisItemType

    def __init__(self, cursor: Cursor) -> None:
      r"""Construct an `Inspecto` for a funciton synopsis

      Parameters
      ----------
      cursor :
        the cursor that this docstring belongs to
      """
      super().__init__(cursor)
      self.synopsis_items = []
      return

    def __call__(self, ds: PetscDocStringImpl, loc: SourceRange, line: str, verdict: Verdict) -> None:
      super().__call__(ds, loc, line, verdict)
      if self.found_synopsis:
        return

      lstrp = line.strip()
      if 'synopsis:' in lstrp.casefold():
        self.capturing = self.CaptureKind.SYNOPSIS
      if self.capturing == self.CaptureKind.SYNOPSIS:
        # don't want to accidentally capture the blurb
        if lstrp:
          self.synopsis_items.append((ds.make_source_range(lstrp, line, loc.start.line), line))
        else:
          # reached the end of the synopsis block
          self.found_synopsis = True
          self.capturing      = self.CaptureKind.NONE
      return

    def get_items(self, ds: PetscDocStringImpl) -> FunctionSynopsis.ItemsType:
      r"""Get the items from this `Inspector`

      Parameters
      ----------
      ds :
        the docstring (unused)

      Returns
      -------
      items :
        the items
      """
      return {
        'synopsis' : self.synopsis_items,
        'name'     : self.items['name'],
        'blurb'    : self.items['blurb']
      }

  def setup(self, ds: PetscDocStringImpl) -> None:
    r"""Set up a `FunctionSynopsis`

    Parameters
    ----------
    ds :
      the `PetscDocString` instance for this section
    """
    inspector = self.Inspector(ds.cursor)
    super()._do_setup(ds, inspector)
    self.items = inspector.get_items(ds)
    return

  def _check_macro_synopsis(self, linter: Linter, cursor: Cursor, docstring: PetscDocStringImpl, explicit_synopsis: SynopsisItemType) -> bool:
    r"""Ensure that synopsese of macros exist and have proper prototypes

    Parameters
    ----------
    linter :
      the `Linter` instance to log errors to
    cursor :
      the cursor this docstring section belongs to
    docstring :
      the docstring that owns this section
    explicit_synopsis :
      the list of source-range - text pairs of lines that make up the synopsis section

    Returns
    -------
    should_check :
      True if the section should continue to check that the synopsis name matches the symbol

    Notes
    -----
    If the synopsis is a macro type, then the name in the synopsis won't match the actual symbol type,
    so it is pointless to check it
    """
    if not (len(explicit_synopsis) or docstring.Modifier.FLOATING in docstring.type_mod):
      # we are missing the synopsis section entirely
      with open(cursor.get_file()) as fh:
        gen   = (l.strip() for l in fh if l.lstrip().startswith('#') and 'include' in l and '/*' in l)
        lines = [
          l.group(2).strip() for l in filter(None, map(self._sowing_include_finder.match, gen))
        ]

      try:
        include_header = lines[0]
      except IndexError:
        include_header = '"some_header.h"'
      args        = ', '.join(
        f'{c.derivedtypename} {c.name}' for c in linter.get_argument_cursors(cursor)
      )
      extent      = docstring._attr['sowing_char_range']
      macro_ident = docstring.make_source_range('M', extent[0], extent.start.line)
      docstring.add_diagnostic(
        docstring.make_diagnostic(
          Diagnostic.Kind.ERROR, self.diags.macro_explicit_synopsis_missing,
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
        ),
        cursor=cursor
      )
       # the code should not check the name
      return False

    # search the explicit docstring for the
    # #include <header.h>
    # line
    header_name = ''
    header_loc  = None
    for loc, line in explicit_synopsis:
      stripped = line.strip()
      if not stripped or stripped.endswith(':') or stripped.casefold().startswith('synopsis'):
        continue
      if found := self._header_include_finder.match(stripped):
        header_name = found.group(1)
        header_loc  = loc
        break

    if not header_name:
      print(80*'=')
      docstring.extent.view()
      print('')
      print('Dont know how to handle no header name yet')
      print(80*'=')
      return False # don't know how to handle this

    assert header_loc is not None
    # TODO cursor.get_declaration() now appears it might work!
    # decl = cursor.get_declaration()

    # OK found it, now find the actual file. Clang unfortunately cannot help us here since
    # it does not pick up header that are in the precompiled header (which chances are,
    # this one is). So we search for it ourselves
    def find_header(directory: Path) -> Optional[Path]:
      path = directory / header_name
      if path.exists():
        return path.resolve()
      return None

    header_path = None
    for flag_path in (Path(flag[2:]) for flag in linter.flags if flag.startswith('-I')):
      header_path = find_header(flag_path)
      if header_path is not None:
        break

    if header_path is None:
      header_path = find_header(Path(str(docstring.extent.start.file)).parent)

    assert header_path
    fn_name = self.items['name'][1]
    decls   = [line for line in header_path.read_text().splitlines() if fn_name in line]
    if not decls:
      # the name was not in the header, so the docstring is wrong
      mess = f"Macro docstring explicit synopsis appears to have incorrect include line. Could not locate '{fn_name}()' in '{header_name}'. Are you sure that's where it lives?"
      docstring.add_diagnostic_from_source_range(
        Diagnostic.Kind.ERROR, self.diags.macro_explicit_synopsis_valid_header, mess, header_loc
      )
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
    # TODO
    # assert len(decls) == 1
    return False

  def check(self, linter: Linter, cursor: Cursor, docstring: PetscDocStringImpl) -> None:
    r"""Perform all checks for this function synopsis

    Parameters
    ----------
    linter :
      the `Linter` instance to log any errors with
    cursor :
      the cursor to which the docstring this section belongs to belongs
    docstring :
      the docstring to which this section belongs
    """
    super().check(linter, cursor, docstring)

    items = self.items
    if items['name'][0] is None:
      # missing synopsis entirely
      docstring.add_diagnostic(
        docstring.make_diagnostic(
          Diagnostic.Kind.ERROR, self.diags.missing_description, 'Docstring missing synopsis',
          self.extent, highlight=False
        ).add_note(
          f"Expected '{cursor.name} - a very useful description'"
        )
      )
      return

    if docstring.Modifier.MACRO in docstring.type_mod:
      # chances are that if it is a macro then the name won't match
      self._check_macro_synopsis(linter, cursor, docstring, items['synopsis'])
      return

    self._syn_common_checks(linter, cursor, docstring)
    return

class EnumSynopsis(Synopsis):
  ItemsType = TypedDict(
    'ItemsType',
    {
      'enum_params' : ParameterList,
      'name'        : Synopsis.NameItemType,
      'blurb'       : Synopsis.BlurbItemType
    }
  )
  items: ItemsType

  class Inspector(Synopsis.Inspector):
    __slots__ = ('enum_params', )

    enum_params: List[Tuple[SourceRange, str, Verdict]]

    def __init__(self, cursor: Cursor) -> None:
      super().__init__(cursor)
      self.enum_params = []
      return

    def __call__(self, ds: PetscDocStringImpl, loc: SourceRange, line: str, verdict: Verdict) -> None:
      super().__call__(ds, loc, line, verdict)
      lstrp = line.lstrip()
      # check that '-' is in the line since some people like to use entire blocks of $'s
      # to describe a single enum value...
      if lstrp.startswith('$') and '-' in lstrp:
        from ._doc_str import Verdict # HACK?

        assert self.items['name'][1] # we should have already found the symbol name
        name = lstrp[1:].split(maxsplit=1)[0].strip()
        self.enum_params.append(
          (ds.make_source_range(name, line, loc.start.line), line, Verdict.NOT_HEADING)
        )
      return

    # HACK
    @staticmethod
    def _check_enum_starts_with_dollar(params: ParameterList, ds: PetscDocStringImpl, items: ParameterList.ItemsType) -> ParameterList.ItemsType:
      for key, opts in sorted(items.items()):
        if len(opts) < 1:
          raise RuntimeError(f'number of options {len(opts)} < 1, key: {key}, items: {items}')
        for opt in opts:
          params._check_opt_starts_with(ds, opt, 'Enum', '$')
      return items

    def get_items(self, ds: PetscDocStringImpl) -> EnumSynopsis.ItemsType:
      params = ParameterList(name='enum params', prefixes=('$',))
      assert self.enum_params, 'No parameter lines in enum description!'
      params.consume(self.enum_params)
      params.setup(ds, parameter_list_prefix_check=self._check_enum_starts_with_dollar)
      return {
        'enum_params' : params,
        'name'        : self.items['name'],
        'blurb'       : self.items['blurb']
      }

  def setup(self, ds: PetscDocStringImpl) -> None:
    r"""Set up an `EnumSynopsis`

    Parameters
    ----------
    ds :
      the `PetscDocString` instance for this section
    """
    inspector = self.Inspector(ds.cursor)
    super()._do_setup(ds, inspector)
    self.items = inspector.get_items(ds)
    return

  def check(self, linter: Linter, cursor: Cursor, docstring: PetscDocStringImpl) -> None:
    r"""Perform all checks for this enum synopsis

    Parameters
    ----------
    linter :
      the `Linter` instance to log any errors with
    cursor :
      the cursor to which the docstring this section belongs to belongs
    docstring :
      the docstring to which this section belongs
    """
    super().check(linter, cursor, docstring)
    if self.items['name'][0] is None:
      # missing synopsis entirely
      docstring.add_diagnostic(
        docstring.make_diagnostic(
          Diagnostic.Kind.ERROR, self.diags.missing_description, 'Docstring missing synopsis',
          self.extent, highlight=False
        ).add_note(
          f"Expected '{cursor.name} - a very useful description'"
        )
      )
    else:
      self._syn_common_checks(linter, cursor, docstring)
    return

@DiagnosticManager.register(
  ('parameter-documentation','Verify that if a, b, c are documented then the function exactly has parameters a, b, and c and vice versa'),
  ('fortran-interface','Verify that functions needing a custom Fortran interface have the correct sowing indentifiers'),
)
class FunctionParameterList(ParameterList):
  diags: DiagnosticMap # satisfy type checkers

  @classmethod
  def __diagnostic_prefix__(cls, *flags):
    return DiagnosticManager.flag_prefix(super())('func', *flags)

  def __init__(self, *args, **kwargs) -> None:
    r"""Construct a `FunctionParameterList`

    Parameters
    ----------
    *args :
      additional positional arguments to `SectionBase.__init__()`
    **kwargs :
      additional keyword arguments to `SectionBase.__init__()`
    """
    kwargs.setdefault(
      'titles', ('Input Parameter', 'Output Parameter', 'Calling sequence', 'Calling Sequence')
    )
    kwargs.setdefault('keywords', ('Input', 'Output', 'Calling sequence of', 'Calling Sequence Of'))
    super().__init__(*args, **kwargs)
    return

  @staticmethod
  def _get_deref_pointer_cursor_type(cursor: CursorLike) -> clx.Type:
    r"""Get the 'bottom' type of a muli-level pointer type, i.e. get double from
    const double *const ****volatile *const *ptr
    """
    canon_type = cursor.type.get_canonical()
    it         = 0
    while canon_type.kind == clx.TypeKind.POINTER:
      if it >= 100:
        import petsclinter as pl
        # there is no chance that someone has a variable over 100 pointers deep, so
        # clearly something is wrong
        cursorview = '\n'.join(pl.classes._util.view_ast_from_cursor(cursor))
        emess      = f'Ran for {it} iterations (>= 100) trying to get pointer type for\n{cursor.error_view_from_cursor(cursor)}\n{cursorview}'
        raise RuntimeError(emess)
      canon_type = canon_type.get_pointee()
      it        += 1
    return canon_type

  def _check_fortran_interface(self, docstring: PetscDocStringImpl, fnargs: tuple[Cursor, ...]) -> None:
    r"""Ensure that functions which require a custom Fortran interface are correctly tagged with 'C'
    sowing designator

    Parameters
    ----------
    docstring :
      the docstring this section belongs to
    fnargs :
      the set of cursors of the function arguments
    """
    requires_c: list[tuple[Cursor, str]] = []
    for arg in fnargs:
      kind = self._get_deref_pointer_cursor_type(arg).kind

      #if kind in clx_char_type_kinds:
      #  requires_c.append((arg, 'char pointer'))
      if kind in clx_function_type_kinds:
        requires_c.append((arg, 'function pointer'))

    if len(requires_c):
      begin_sowing_range = docstring._attr['sowing_char_range']
      sowing_chars       = begin_sowing_range.raw(tight=True)
      if docstring.Modifier.C_FUNC not in docstring.type_mod:
        assert 'C' not in sowing_chars
        diag = docstring.make_diagnostic(
          Diagnostic.Kind.ERROR, self.diags.fortran_interface,
          f"Function requires custom Fortran interface but missing 'C' from docstring header {Diagnostic.FLAG_SUBST}",
          begin_sowing_range, patch=Patch(begin_sowing_range, sowing_chars + 'C')
        )
        for reason_cursor, reason_type in requires_c:
          diag.add_note(
            f'due to {reason_type} {reason_cursor.get_formatted_blurb(num_context=1).rstrip()}',
            location=reason_cursor.extent.start
          )
        docstring.add_diagnostic(diag)
    return

  def _check_no_args_documented(self, docstring: PetscDocStringImpl, arg_cursors: tuple[Cursor, ...]) -> bool:
    r"""Check if no arguments were documented

    Parameters
    ----------
    docstring :
      the docstring this section belongs to
    arg_cursors :
      the set of argument cursors for the function cursor to check

    Returns
    -------
    ret :
      True (and logs the appropriate error) if no arguments were documented, False otherwise
    """
    if arg_cursors and not self:
      # none of the function arguments are documented
      docstring.add_diagnostic(
        docstring.make_diagnostic(
          Diagnostic.Kind.ERROR, self.diags.parameter_documentation,
          f'Symbol parameters are all undocumented {Diagnostic.FLAG_SUBST}',
          docstring.extent, highlight=False
        ).add_note(
          Diagnostic.make_message_from_formattable(
            'Parameters defined here',
            crange=SourceRange.from_locations(arg_cursors[0].extent.start, arg_cursors[-1].extent.end)
          ),
          location=arg_cursors[0].extent.start
        )
      )
      return True

    if not arg_cursors and self and len(self.items.values()):
      # function has no arguments, so check there are no parameter docstrings, if so, we can
      # delete them
      doc_cursor = docstring.cursor
      disp_name  = doc_cursor.displayname
      docstring.add_diagnostic(
        docstring.make_diagnostic(
          Diagnostic.Kind.ERROR, self.diags.parameter_documentation,
          f'Found parameter docstring(s) but \'{disp_name}\' has no parameters',
          self.extent,
          highlight=False, patch=Patch(self.extent, '')
        ).add_note(
          # can't use the Diagnostic.make_diagnostic_message() (with doc_cursor.extent),
          # since that prints the whole function. Cursor.formatted() has special code to
          # only print the function line for us, so use that instead
          f'\'{disp_name}\' defined here:\n{doc_cursor.formatted(num_context=2)}',
          location=doc_cursor.extent.start
        )
      )
      return True

    return False

  class ParamVisitor:
    def __init__(self, num_groups: int, arg_cursors: Iterable[Cursor]) -> None:
      r"""Construct a `ParamVisitor`

      Parameters
      ----------
      num_groups :
        the number of argument groups in the `FunctionParameterList` items
      arg_cursors :
        the full set of argument cursors (i.e. those retrieved by
        `FunctionParameterList._get_recursive_cursor_list()`)
      """
      self.num_groups = num_groups
      self.arg_names  = [a.name for a in arg_cursors if a.name]
      self.arg_seen   = [0] * len(self.arg_names)
      return

    def mark_as_seen(self, name: str) -> int:
      r"""Mark an argument name as 'seen' by this visitor

      Parameters
      ----------
      name :
        the name of the argument

      Returns
      -------
      arg_idx :
        the 0-based index of `name` into the argument list, or -1 if `name` was invalid

      Notes
      -----
      `name` is considered invalid if:
      - it does not match any of the argument names
      - it does match an argument name, but that name has already been seen enough times. For example,
        if `self.arg_names` contains 3 instances of `name`, the first 3 calls to
        `self.mark_as_seen(name)` will return the first 3 indices of `name`, while the 4th call will
        return -1
      """
      idx  = 0
      prev = -1
      while 1:
        # in case of multiple arguments of the same name, we need to loop until we
        # find an index that has not yet been found
        try:
          idx = self.arg_names.index(name, idx)
        except ValueError:
          idx = prev
          break
        count = self.arg_seen[idx]
        if 0 <= count <= self.num_groups:
          # arg_seen[idx] = 0 -> argument exists and has not been found yet
          # arg_seen[idx] <= num_groups -> argument is possibly in-out and is defined in
          # multiple groups
          if count == 0:
            # first time, use this arg
            break
          # save this to come back to
          prev = idx
        # argument exists but has already been claimed
        idx += 1
      if idx >= 0:
        self.arg_seen[idx] += 1
      return idx

  def _param_initial_traversal(self, docstring: PetscDocStringImpl, visitor: FunctionParameterList.ParamVisitor) -> list[tuple[str, SourceRange]]:
    r"""Perform the initial traversal of a parameter list, and return any arguments that were seemingly
    never found

    Parameters
    ----------
    docstring :
      the docstring this section belongs to
    visitor :
      the visitor to call on each argument

    Returns
    -------
    not_found :
      a list of names (and their source ranges) which were not found in the function arguments

    Notes
    -----
    The visitor should implement `mark_as_seen(name: str) -> int` which returns the 0-based index of
    `name` in the list of function arguments if it was found, and `-1` otherwise
    """
    not_found           = []
    solitary_param_diag = self.diags.solitary_parameter
    for group in self.items.values():
      remove = set()
      for i, (loc, descr_item, _) in enumerate(group):
        arg, sep = descr_item.arg, descr_item.sep
        if sep == ',' or ',' in arg:
          sub_args = tuple(map(str.strip, arg.split(',')))
          if len(sub_args) > 1:
            diag = docstring.make_diagnostic(
              Diagnostic.Kind.ERROR, solitary_param_diag,
              'Each parameter entry must be documented separately on its own line',
              docstring.make_source_range(arg, descr_item.text, loc.start.line)
            )
            if docstring.cursor.is_variadic_function():
              diag.add_note('variable argument lists should be documented in notes')
            docstring.add_diagnostic(diag)
        elif sep == '=':
          sub_args = tuple(map(str.strip, arg.split(' = ')))
          if len(sub_args) > 1:
            sub_args = (sub_args[0],) # case of bad separator, only the first entry is valid
        else:
          sub_args = (arg,)

        for sub in sub_args:
          idx = visitor.mark_as_seen(sub)
          if idx == -1 and sub == '...' and docstring.cursor.is_variadic_function():
            idx = 0 # variadic parameters don't get a cursor, so can't be picked up
          if idx == -1:
            # argument was not found at all
            not_found.append((sub, docstring.make_source_range(sub, descr_item.text, loc.start.line)))
            remove.add(i)
          else:
            descr_item.check(docstring, self, loc)
      self.check_aligned_descriptions(docstring, [g for i, g in enumerate(group) if i not in remove])
    return not_found

  def _check_docstring_param_is_in_symbol_list(self, docstring: PetscDocStringImpl, arg_cursors: Sequence[Cursor], not_found: list[tuple[str, SourceRange]], args_left: list[str], visitor: FunctionParameterList.ParamVisitor) -> list[str]:
    r"""Check that all documented parameters are actually in the symbol list.

    Parameters
    ----------
    docstring :
      the docstring to which this section belongs
    arg_cursors :
      the set of argument cursors for the function cursor
    note_found :
      a list of name - source range pairs of arguments in the docstring which were not found
    args_left :
      a list of function argument names which were not seen in the docstring
    visitor :
      the visitor to call on each argument

    Returns
    -------
    args_left :
      the pruned args_left, all remaining entries will be undocuments function argument names

    Notes
    -----
    This catches items that were documented, but don't actually exist in the argument list
    """
    param_doc_diag   = self.diags.parameter_documentation
    func_ptr_cursors = None
    for i, (arg, loc) in enumerate(not_found):
      patch = None
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
        # the difflib call failed
        note_loc = docstring.cursor.extent.start
        note     = Diagnostic.make_message_from_formattable(
          'Parameter list defined here', crange=docstring.cursor
        )
      else:
        match_cursor = [c for c in arg_cursors if c.name == arg_match][0]
        note_loc     = match_cursor.extent.start
        note         = Diagnostic.make_message_from_formattable(
          f'Maybe you meant {match_cursor.get_formatted_blurb()}'
        )
        args_left.remove(arg_match)
        idx = visitor.mark_as_seen(arg_match)
        assert idx != -1, f'{arg_match} was not found in arg_names'
      diag = docstring.make_diagnostic(
        Diagnostic.Kind.ERROR, param_doc_diag,
        f"Extra docstring parameter \'{arg}\' not found in symbol parameter list", loc,
        patch=patch
      ).add_note(
        note, location=note_loc
      )
      if func_ptr_cursors is None:
        # have not checked yet
        func_ptr_cursors = any(
          c for c in arg_cursors
          if self._get_deref_pointer_cursor_type(c).kind == clx.TypeKind.FUNCTIONPROTO
        )
      if func_ptr_cursors:
        diag.add_note(
          '\n'.join((
            'If you are trying to document a function-pointer parameter, then you must name the function pointer arguments in source and introduce a new section \'Calling Sequence of `<name of function pointer arg>\'. For example:',
            '',
            '/*@C',
            '  ...',
            '  Input Parameter:',
            '. func_ptr - A function pointer',
            '',
            '  Calling Sequence of `func_ptr`:',
            '+ foo - a very useful description >-----------------------x Note named parameters!',
            '- bar - a very useful description >-----------------------|-----------x',
            '  ...                                                     |           |',
            '@*/                                                      vvv         vvv',
            'PetscErrorCode MyFunction(PetscErrorCode (*func_ptr)(int foo, double bar))'
          ))
        )
      docstring.add_diagnostic(diag)
    return args_left

  def _get_recursive_cursor_list(self, cursor_list: Iterable[CursorLike]) -> list[Cursor]:
    r"""Traverse an arg list recursively to get all nested arg cursors

    Parameters
    ----------
    cursor_list :
      the initial list of arg cursors

    Returns
    -------
    cursor_list :
      the complete cursor list

    Notes
    -----
    This performs a depth-first search to return all cursors. So given a function
    ```
    PetscErrorCode Foo(int x, void (*bar)(int y, void (*baz)(double z)), int w)
    ```
    This returns in `[x_cursor, bar_cursor, y_cursor, baz_cursor, z_cursor, w_cursor]` in `cursor_list`
    """
    new_cursor_list = []
    PARM_DECL_KIND  = clx.CursorKind.PARM_DECL
    for cursor in map(Cursor.cast, cursor_list):
      new_cursor_list.append(cursor)
      # Special handling of functions taking function pointer arguments. In this case we
      # should recursively descend and pick up the names of all the function parameters
      #
      # note the order, by appending cursor first we effectively do a depth-first search
      if self._get_deref_pointer_cursor_type(cursor).kind == clx.TypeKind.FUNCTIONPROTO:
        new_cursor_list.extend(
          self._get_recursive_cursor_list(c for c in cursor.get_children() if c.kind == PARM_DECL_KIND)
        )
    return new_cursor_list

  def _check_valid_param_list_from_cursor(self, docstring: PetscDocStringImpl, arg_cursors: tuple[Cursor, ...]) -> None:
    r"""Ensure that the parameter list matches the documented values, and that their order is correct

    Parameters
    ----------
    docstring :
      the docstring to which this section belongs
    arg_cursors :
      the set of argument cursors for the function cursor
    """
    if self._check_no_args_documented(docstring, arg_cursors) or not self:
      return

    full_arg_cursors = self._get_recursive_cursor_list(arg_cursors)
    visitor          = self.ParamVisitor(max(self.items.keys(), default=0), full_arg_cursors)
    not_found        = self._param_initial_traversal(docstring, visitor)
    args_left        = self._check_docstring_param_is_in_symbol_list(
      docstring, full_arg_cursors, not_found,
      [name for seen, name in zip(visitor.arg_seen, visitor.arg_names) if not seen],
      visitor
    )

    for arg in args_left:
      idx = visitor.mark_as_seen(arg)
      assert idx >= 0
      if docstring.Modifier.MACRO in docstring.type_mod:
        # TODO
        # Blindly assume that macro docstrings are OK for now. Ultimately this function
        # should check against a parsed synopsis instead of the actual function arguments.
        continue
      docstring.add_diagnostic(
        docstring.make_diagnostic(
          Diagnostic.Kind.ERROR, self.diags.parameter_documentation,
          f'Undocumented parameter \'{arg}\' not found in parameter section',
          self.extent, highlight=False
        ).add_note(
          Diagnostic.make_message_from_formattable(
            f'Parameter \'{arg}\' defined here', crange=full_arg_cursors[idx], num_context=1
          ),
          location=full_arg_cursors[idx].extent.start
        )
      )
    return

  def check(self, linter: Linter, cursor: Cursor, docstring: PetscDocStringImpl) -> None:
    r"""Perform all checks for this function param list

    Parameters
    ----------
    linter :
      the `Linter` instance to log any errors with
    cursor :
      the cursor to which the docstring this section belongs to
    docstring :
      the docstring to which this section belongs
    """
    super().check(linter, cursor, docstring)
    fnargs = linter.get_argument_cursors(cursor)

    self._check_fortran_interface(docstring, fnargs)
    self._check_valid_param_list_from_cursor(docstring, fnargs)
    return

class OptionDatabaseKeys(ParameterList):
  diags: DiagnosticMap # satisfy type checkers

  @classmethod
  def __diagnostic_prefix__(cls, *flags):
    return DiagnosticManager.flag_prefix(super())('option-keys', *flags)

  def __init__(self, *args, **kwargs) -> None:
    r"""Construct an `OptionsDatabaseKeys`

    Parameters
    ----------
    *args :
      additional positional arguments to `SectionBase.__init__()`
    **kwargs :
      additional keyword arguments to `SectionBase.__init__()`
    """
    kwargs.setdefault('name', 'options')
    kwargs.setdefault('titles', ('Options Database',))
    super().__init__(*args, **kwargs)
    return

  def _check_option_database_key_alignment(self, docstring: PetscDocStringImpl) -> None:
    r"""Ensure that option database keys and their descriptions are properly aligned

    Parameters
    ----------
    docstring :
      the docstring to which this section belongs
    """
    for _, group in sorted(self.items.items()):
      self.check_aligned_descriptions(docstring, group)
    return

  def check(self, linter: Linter, cursor: Cursor, docstring: PetscDocStringImpl) -> None:
    r"""Perform all checks for this optionsdb list

    Parameters
    ----------
    linter :
      the `Linter` instance to log any errors with
    cursor :
      the cursor to which the docstring this section belongs to
    docstring :
      the docstring to which this section belongs
    """
    super().check(linter, cursor, docstring)

    self._check_option_database_key_alignment(docstring)
    return

class Notes(Prose):
  diags: DiagnosticMap # satisfy type checkers

  @classmethod
  def __diagnostic_prefix__(cls, *flags):
    return DiagnosticManager.flag_prefix(super())('notes', *flags)

  def __init__(self, *args, **kwargs) -> None:
    r"""Construct a `Notes`

    Parameters
    ----------
    *args :
      additional positional arguments to `SectionBase.__init__()`
    **kwargs :
      additional keyword arguments to `SectionBase.__init__()`
    """
    kwargs.setdefault('name', 'notes')
    kwargs.setdefault('titles', ('Notes', 'Note'))
    super().__init__(*args, **kwargs)
    return

class DeveloperNotes(Prose):
  diags: DiagnosticMap # satisfy type checkers

  @classmethod
  def __diagnostic_prefix__(cls, *flags):
    return DiagnosticManager.flag_prefix(super())('dev-notes', *flags)

  def __init__(self, *args, **kwargs) -> None:
    r"""Construct a `DeveloperNotes`

    Parameters
    ----------
    *args :
      additional positional arguments to `SectionBase.__init__()`
    **kwargs :
      additional keyword arguments to `SectionBase.__init__()`
    """
    kwargs.setdefault('name', 'developer notes')
    kwargs.setdefault('titles', ('Developer Notes', 'Developer Note'))
    super().__init__(*args, **kwargs)

class References(Prose):
  diags: DiagnosticMap # satisfy type checkers

  @classmethod
  def __diagnostic_prefix__(cls, *flags):
    return DiagnosticManager.flag_prefix(super())('references', *flags)

  def __init__(self, *args, **kwargs) -> None:
    r"""Construct a `References`

    Parameters
    ----------
    *args :
      additional positional arguments to `SectionBase.__init__()`
    **kwargs :
      additional keyword arguments to `SectionBase.__init__()`
    """
    kwargs.setdefault('name', 'references')
    kwargs.setdefault('solitary', False)
    super().__init__(*args, **kwargs)
    return

class FortranNotes(Prose):
  diags: DiagnosticMap # satisfy type checkers

  @classmethod
  def __diagnostic_prefix__(cls, *flags):
    return DiagnosticManager.flag_prefix(super())('fortran-notes', *flags)

  def __init__(self, *args, **kwargs) -> None:
    r"""Construct a `FortranNotes`

    Parameters
    ----------
    *args :
      additional positional arguments to `SectionBase.__init__()`
    **kwargs :
      additional keyword arguments to `SectionBase.__init__()`
    """
    kwargs.setdefault('name', 'fortran notes')
    kwargs.setdefault('titles', ('Fortran Notes', 'Fortran Note'))
    kwargs.setdefault('keywords', ('Fortran', ))
    super().__init__(*args, **kwargs)
    return

class SourceCode(VerbatimBlock):
  diags: DiagnosticMap # satisfy type checkers

  @classmethod
  def __diagnostic_prefix__(cls, *flags):
    return DiagnosticManager.flag_prefix(super())('source-code', *flags)

  def __init__(self, *args, **kwargs) -> None:
    r"""Construct a `SourceCode`

    Parameters
    ----------
    *args :
      additional positional arguments to `SectionBase.__init__()`
    **kwargs :
      additional keyword arguments to `SectionBase.__init__()`
    """
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

  valid_levels: tuple[str, ...]

  diags: DiagnosticMap # satisfy type checkers

  def __init__(self, *args, **kwargs) -> None:
    r"""Construct a `Level`

    Parameters
    ----------
    *args :
      additional positional arguments to `SectionBase.__init__()`
    **kwargs :
      additional keyword arguments to `SectionBase.__init__()`
    """
    kwargs.setdefault('name', 'level')
    kwargs.setdefault('required', True)
    super().__init__(*args, **kwargs)
    self.valid_levels = ('beginner', 'intermediate', 'advanced', 'developer', 'deprecated')
    return

  @classmethod
  def __diagnostic_prefix__(cls, *flags):
    return DiagnosticManager.flag_prefix(super())('level', *flags)

  def __do_check_valid_level_spelling(self, docstring: PetscDocStringImpl, loc: SourceRange, level_name: str) -> None:
    r"""Do the actual valid level spelling check

    Parameters
    ----------
    docstring :
      the docstring to which this section belongs
    loc :
      the location of the level item, i.e. the location of 'beginner'
    level_name :
      the string of the location
    """
    if level_name in self.valid_levels:
      return # all good

    def make_sub_loc(loc: SourceRange, substr: str) -> SourceRange:
      return docstring.make_source_range(substr, loc.raw(), loc.start.line, offset=loc.start.column - 1)

    locase  = level_name.casefold()
    patch   = None
    sub_loc = loc
    if locase in self.valid_levels:
      diag  = self.diags.casefold
      mess  = f"Level subheading must be lowercase, expected '{locase}' found '{level_name}'"
      patch = Patch(loc, locase)
    else:
      diag            = self.diags.spelling
      lvl_match_close = difflib.get_close_matches(locase, self.valid_levels, n=1)
      if not lvl_match_close:
        sub_split = level_name.split(maxsplit=1)[0]
        if sub_split != level_name:
          sub_loc         = make_sub_loc(loc, sub_split)
          lvl_match_close = difflib.get_close_matches(sub_split.casefold(), self.valid_levels, n=1)

      if lvl_match_close:
        lvl_match = lvl_match_close[0]
        if lvl_match == 'deprecated':
          if re_match := re.match(
              r'(\w+)\s*(\(\s*[sS][iI][nN][cC][eE]\s*\d+\.\d+[\.\d\s]*\))', level_name
          ):
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
    docstring.add_diagnostic_from_source_range(Diagnostic.Kind.ERROR, diag, mess, sub_loc, patch=patch)
    return

  def _check_valid_level_spelling(self, docstring: PetscDocStringImpl) -> None:
    r"""Ensure that the level values are both proper and properly spelled

    Parameters
    ----------
    docstring :
      the docstring to which this section belongs
    """
    for line_after_colon, sub_items in self.items:
      for loc, level_name in sub_items:
        self.__do_check_valid_level_spelling(docstring, loc, level_name)
    return

  def _check_level_heading_on_same_line(self) -> None:
    r"""Ensure that the level heading value is on the same line as Level:

    Notes
    -----
    TODO
    """
    return
    # TODO FIX ME, need to be able to handle the below
    # for loc, line, verdict in self.lines():
    #   if line and ':' not in line:
    #     # if you get a "prevloc" and "prevline" not defined error here this means that we
    #     # are erroring out on the first trip round this loop and somehow have a
    #     # lone-standing 'beginner' or whatever without an explicit "Level:" line...
    #     errorMessage = f"Level values must be on the same line as the 'Level' heading, not on separate line:\n{prevloc.merge_with(loc).formatted(num_context=2, highlight=False)}"
    #     # This is a stupid hack to solve a multifaceted issue. Suppose you have
    #     # Level:
    #     # BLABLABLA
    #     # The first fix above does a tolower() transformation
    #     # Level:
    #     # blabla
    #     # while this fix would apply a join transformation
    #     # Level: BLABLA
    #     # See the issue already? Since we sort the transformations by line the second
    #     # transformation would actually end up going *first*, meaning that the lowercase
    #     # transformation is no longer valid for patch...

    #     # create a range starting at newline of previous line going until the first
    #     # non-space character on the next line
    #     delrange = SourceRange.from_positions(
    #       cursor.translation_unit, prevloc.end.line, -1, loc.start.line, len(line) - len(line.lstrip())
    #     )
    #     # given '  Level:\n  blabla'
    #     #                ^^^
    #     #                 |
    #     #              delrange
    #     # delete delrange from it to get '  Level: blabla'
    #     # TODO: make a real diagnostic here
    #     diag = Diagnostic(Diagnostic.Kind.ERRROR, spellingDiag, errorMessage, patch=Patch(delrange, ''))
    #     linter.add_diagnostic_from_cursor(cursor, diag)
    #   prevloc  = loc
    #   prevline = line
    # return

  def check(self, linter: Linter, cursor: Cursor, docstring: PetscDocStringImpl) -> None:
    r"""Perform all checks for this level

    Parameters
    ----------
    linter :
      the `Linter` instance to log any errors with
    cursor :
      the cursor to which the docstring this section belongs to
    docstring :
      the docstring to which this section belongs
    """
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
  __slots__ = ('special_chars',)

  special_chars: str

  diags: DiagnosticMap # satisfy type checkers

  def __init__(self, *args, **kwargs) -> None:
    r"""Construct a `SeeAlso`

    Parameters
    ----------
    *args :
      additional positional arguments to `SectionBase.__init__()`
    **kwargs :
      additional keyword arguments to `SectionBase.__init__()`
    """
    kwargs.setdefault('name', 'seealso')
    kwargs.setdefault('required', True)
    kwargs.setdefault('titles', ('.seealso',))
    super().__init__(*args, **kwargs)
    self.special_chars = '`'
    return

  @classmethod
  def __diagnostic_prefix__(cls, *flags):
    return DiagnosticManager.flag_prefix(super())('seealso', *flags)

  @staticmethod
  def transform(text: str) -> str:
    return text.casefold()

  @staticmethod
  def __make_deletion_patch(loc: SourceRange, text: str, look_behind: bool) -> Patch:
    """Make a cohesive deletion patch

    Parameters
    ----------
    loc :
      the source range for the item to delete
    text :
      the text of the full line
    look_behind :
      should we remove the comma and space behind the location as well?

    Returns
    -------
    patch :
      the patch

    Notes
    -----
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

  def _check_self_referential(self, cursor: Cursor, docstring: PetscDocStringImpl, items: InlineList.ItemsType, last_loc: SourceRange) -> list[tuple[SourceRange, str]]:
    r"""Ensure that the seealso list does not contain the name of the cursors symbol, i.e. that the
    docstring is not self-referential

    Parameters
    ----------
    cursor :
      the cursor to which this docstring belongs
    docstring :
      the docstring to which this section belongs
    items :
      the inline list items
    last_loc :
      the location of the final entry in the list

    Returns
    -------
    item_remain :
      the list of items, with self-referential items removed
    """
    item_remain: list[tuple[SourceRange, str]] = []
    symbol_name = Cursor.get_name_from_cursor(cursor)
    for line_after_colon, sub_items in items:
      for loc, text in sub_items:
        if text.replace(self.special_chars, '').rstrip('()') == symbol_name:
          mess = f"Found self-referential seealso entry '{text}'; your documentation may be good but it's not *that* good"
          docstring.add_diagnostic_from_source_range(
            Diagnostic.Kind.ERROR, self.diags.self_reference, mess, loc,
            patch=self.__make_deletion_patch(loc, text, loc == last_loc)
          )
        else:
          item_remain.append((loc, text))
    return item_remain

  def _check_enclosed_by_special_chars(self, docstring: PetscDocStringImpl, item_remain: list[tuple[SourceRange, str]]) -> None:
    r"""Ensure that every entry in the seealso list is enclosed in backticks

    Parameters
    ----------
    docstring :
      the docstring to which this section belongs
    item_remain :
      the list of valid items to check
    """
    def enclosed_by(string: str, char: str) -> bool:
      return string.startswith(char) and string.endswith(char)

    chars = self.special_chars
    for loc, text in item_remain:
      if not enclosed_by(text, chars) and not re.search(r'\[.*\]\(\w+\)', text):
        docstring.add_diagnostic_from_source_range(
          Diagnostic.Kind.ERROR, self.diags.backticks,
          f"seealso symbol '{text}' not enclosed with '{chars}'",
          loc, patch=Patch(loc, f'{chars}{text.replace(chars, "")}{chars}')
        )
    return

  def _check_duplicate_entries(self, docstring: PetscDocStringImpl, item_remain: list[tuple[SourceRange, str]], last_loc: SourceRange) -> None:
    r"""Ensure that the seealso list has no duplicate entries

    Parameters
    ----------
    docstring :
      the docstring to which this section belongs
    item_remain :
      the list of valid items to check
    last_loc :
      the location of the final entry in the list

    Notes
    -----
    `last_loc` must be the original final location, even if `item_remain` does not contain it (i.e. it
    is an invalid entry)!
    """
    seen: dict[str, SourceRange] = {}
    for loc, text in item_remain:
      text_no_special = text.replace(self.special_chars, '')
      assert text_no_special
      if text_no_special in seen:
        first_seen = seen[text_no_special]
        docstring.add_diagnostic(
          docstring.make_diagnostic(
            Diagnostic.Kind.ERROR, self.diags.duplicate, f"Seealso entry '{text}' is duplicate", loc,
            patch=self.__make_deletion_patch(loc, text, loc == last_loc)
          ).add_note(
            Diagnostic.make_message_from_formattable(
              'first instance found here', crange=first_seen, num_context=1
            ),
            location=first_seen.start
          )
        )
      else:
        seen[text_no_special] = loc
    return

  def check(self, linter: Linter, cursor: Cursor, docstring: PetscDocStringImpl) -> None:
    r"""Perform all checks for this seealso list

    Parameters
    ----------
    linter :
      the `Linter` instance to log any errors with
    cursor :
      the cursor to which the docstring this section belongs to
    docstring :
      the docstring to which this section belongs
    """
    super().check(linter, cursor, docstring)

    if self.barren() or not self:
      return # barren

    items       = self.items
    last_loc    = items[-1][1][-1][0]
    item_remain = self._check_self_referential(cursor, docstring, items, last_loc)
    self._check_enclosed_by_special_chars(docstring, item_remain)
    self._check_duplicate_entries(docstring, item_remain, last_loc)
    return
