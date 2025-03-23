#!/usr/bin/env python3
"""
# Created: Mon Jun 20 16:40:24 2022 (-0400)
# @author: Jacob Faibussowitsch
"""
from __future__ import annotations

import weakref
import difflib
import textwrap
import datetime
import itertools
import collections
import clang.cindex as clx # type: ignore[import]

from .._typing import *

from ._diag      import DiagnosticManager, Diagnostic
from ._cursor    import Cursor
from ._src_pos   import SourceLocation, SourceRange
from ._patch     import Patch
from ._scope     import Scope
from ._weak_list import WeakList
from ._add_line  import Addline

from .._error import ParsingError, KnownUnhandleableCursorError

from ..util._clang   import clx_func_call_cursor_kinds, base_clang_options
from ..util._utility import subprocess_capture_output, subprocess_check_returncode

class DiagnosticsContainer:
  __slots__ = 'prefix', 'data'

  prefix: str
  data: dict[Path, dict[int, WeakListType]]

  def __init__(self, prefix: str) -> None:
    r"""Construct a `DiagnosticsContainer`

    Parameters
    ----------
    prefix :
      the prefix for this diagnostic container
    """
    self.prefix = prefix
    self.data   = {}
    return

  def add_diagnostic(self, cursor: Cursor, diagnostic: Diagnostic) -> None:
    r"""Add a `Diagnostic` to this container

    Parameters
    ----------
    cursor :
      the cursor about which the `diagnostic` is concerned
    diagnostic :
      the diagnostic detailing the error or warning
    """
    filename = cursor.get_file()
    data     = self.data
    # Of the various dict-key probing method, try-except is the fastest lookup method if a
    # key exists in a dict, but is by far the slowest in the cases where the key is
    # missing.
    #
    # But each linter instance is likely to only be used on a single file so filename is
    # very likely to be in the dict.
    try:
      file_local = data[filename]
    except KeyError:
      file_local = data[filename] = {}

    # note that cursors are _unlikely_ to be in the dict, and hence do the if-test
    cursor_id = cursor.hash
    if cursor_id not in file_local:
      file_local[cursor_id] = WeakList()

    patch      = diagnostic.patch
    have_patch = patch is not None
    # type checkers don't grok that have_patch implies patch != None
    patch_id             = TYPE_CAST(Patch, patch).id if have_patch else -1
    cursor_id_file_local = file_local[cursor_id]
    cursor_id_file_local.append((diagnostic.formatted_header(), have_patch, patch_id))

    if have_patch:
      assert patch is not None # to satisfy type checkers
      patch.attach(weakref.ref(cursor_id_file_local))
    return

  def split_and_condense(self) -> tuple[CondensedDiags, CondensedDiags]:
    r"""Split the diagnostics into resolved and unresolved and condense them per path

    Returns
    -------
    unresolved :
      a dict mapping a `Path` to a list of diagnostic messages for all unresolved diagnostics
      (i.e. those without a `Patch`)
    resolved :
      a dict mapping a `Path` to a list of diagnostic messages for all resolved diagnostics
      (i.e. those with a `Patch`)
    """
    unresolved: CondensedDiags = {p : [] for p in self.data.keys()}
    resolved: CondensedDiags   = {p : [] for p in self.data.keys()}
    for path, diags in self.data.items():
      resolved_list   = resolved[path]
      unresolved_list = unresolved[path]
      for err_list in diags.values():
        for err, have_patch, _ in err_list:
          if have_patch:
            resolved_list.append(err)
          else:
            unresolved_list.append(err)
      # remove any empty sets
      for d in (resolved, unresolved):
        if not d[path]:
          del d[path]
    return unresolved, resolved

  def view_last(self) -> None:
    r"""Print the last diagnostic added"""
    import petsclinter as pl

    for files in reversed(self.data):
      diags = self.data[files]
      last  = diags[next(reversed(diags))]
      pl.sync_print(last[-1][0])
      return

@DiagnosticManager.register(
  ('duplicate-function', 'Check for duplicate function-calls on the same execution path'),
  ('static-function-candidate', 'Check for functions that are used only within a single TU, and make them static'),
  ('parsing-error', 'Generic parsing errors')
)
class Linter:
  """
  Object to manage the collection and processing of errors during a lint run.
  """
  __slots__ = 'flags', 'clang_opts', 'verbose', 'index', 'errors', 'warnings', 'patches', 'werror'

  flags: list[str]
  clang_opts: CXTranslationUnit
  verbose: int
  index: clx.Index
  errors: DiagnosticsContainer
  warnings: DiagnosticsContainer
  patches: collections.defaultdict[Path, list[Patch]]
  werror: bool

  diags: DiagnosticMap # satisfy type checkers

  def __init__(self, compiler_flags: list[str], clang_options: Optional[CXTranslationUnit] = None, verbose: int = 0, werror: bool = False) -> None:
    r"""Construct a `Linter`

    Parameters
    ----------
    compiler_flags :
      the set of compiler flags to parse with
    clang_options : optional
      the set of clang options to pass to the `clang.cindex.Index.parse()` function, defaults to
      `petsclinter.util.base_clang_options`
    verbose : optional
      whether to print verbose output (and at what level)
    werror : optional
      whether to treat warnings as errors
    """
    if clang_options is None:
      clang_options = base_clang_options

    self.flags      = compiler_flags
    self.clang_opts = clang_options
    self.verbose    = verbose
    self.index      = clx.Index.create()
    self.werror     = werror
    self.clear()
    return

  def __str__(self) -> str:
    print_list = [
      f'Compiler Flags: {self.flags}',
      f'Clang Options:  {self.clang_opts}',
      f'Verbose:        {self.verbose}'
    ]
    for getter_func in (self.get_all_warnings, self.get_all_errors):
      for v in getter_func():
        for mess in v.values():
          print_list.append('\n'.join(mess))
    return '\n'.join(print_list)

  def _vprint(self, *args, **kwargs) -> None:
    r"""Print only if verbose"""
    if self.verbose:
      import petsclinter as pl

      pl.sync_print(*args, **kwargs)
    return

  def _check_duplicate_function_calls(self, processed_funcs: dict[str, list[tuple[Cursor, Scope]]]) -> None:
    r"""Check for duplicate instances of functions along the same execution path

    Parameters
    ----------
    processed_funcs :
      a dict mapping parent function names and the list of functions and their scopes

    Notes
    -----
    If two instances of a function have the same `Scope` then they are duplicate and an error is
    logged
    """
    dup_diag = self.diags.duplicate_function
    for function_list in processed_funcs.values():
      seen = {}
      for func, scope in function_list:
        combo: list[str] = [func.displayname]
        try:
          combo.extend(map(Cursor.get_raw_name_from_cursor, func.get_arguments()))
        except ParsingError:
          continue

        # convert to tuple so it is hashable
        combo_tup = tuple(combo)
        if combo_tup not in seen:
          seen[combo_tup] = (func, scope)
        elif scope >= seen[combo_tup][1]:
          # this combination has already been seen, i.e. this call is duplicate!!
          start      = func.extent.start
          startline  = start.line
          tu         = func.translation_unit
          end        = clx.SourceLocation.from_position(tu, tu.get_file(tu.spelling), startline, -1)
          patch      = Patch(SourceRange.from_locations(start, end), '')
          previous   = seen[combo_tup][0].formatted(
            num_before_context=2, num_after_context=startline - seen[combo_tup][0].extent.start.line
          )
          message    = f'Duplicate function found previous identical usage:\n{previous}'
          self.add_diagnostic_from_cursor(
            func, Diagnostic(Diagnostic.Kind.ERROR, dup_diag, message, start, patch=patch)
          )
    return

  @staticmethod
  def find_lintable_expressions(tu: clx.TranslationUnit, symbol_names: Container[str]) -> Generator[Union[tuple[clx.Cursor, clx.Cursor, Scope], clx.Cursor], None, None]:
    r"""Finds all lintable expressions in container symbol_names.

    Parameters
    ----------
    tu :
      the `clang.cindex.TranslationUnit` to search
    symbol_names :
      the names of the symbols to search for and lint

    Notes
    -----
    Note that if a particular expression is not 100% correctly defined (i.e. would the
    file actually compile) then it will not be picked up by clang AST.

    Function-like macros can be picked up, but it will be in the wrong 'order'. The AST is
    built as if you are about to compile it, so macros are handled before any real
    function definitions in the AST, making it impossible to map a macro invocation to
    its 'parent' function.
    """
    UNEXPOSED_DECL = clx.CursorKind.UNEXPOSED_DECL
    SWITCH_STMT    = clx.CursorKind.SWITCH_STMT
    CASE_STMT      = clx.CursorKind.CASE_STMT
    COMPOUND_STMT  = clx.CursorKind.COMPOUND_STMT
    CALL_EXPR      = clx.CursorKind.CALL_EXPR

    def walk_scope_switch(parent: clx.Cursor, scope: Scope) -> Generator[tuple[clx.Cursor, clx.Cursor, Scope], None, None]:
      """
      Special treatment for switch-case since the AST setup for it is mind-boggingly stupid.
      The first node after a case statement is listed as the cases *child* whereas every other
      node (including the break!!) is the cases *sibling*
      """
      # in case we get here from a scope decrease within a case
      case_scope = scope
      for child in parent.get_children():
        child_kind = child.kind
        if child_kind == CASE_STMT:
          # create a new scope every time we encounter a case, this is now for all intents
          # and purposes the 'scope' going forward. We don't overwrite the original scope
          # since we still need each case scope to be the previous scopes sibling
          case_scope = scope.sub()
          yield from walk_scope(child, scope=case_scope)
        elif child_kind == CALL_EXPR:
          if child.spelling in symbol_names:
            yield (child, possible_parent, case_scope)
            # Cursors that indicate change of logical scope
        elif child_kind == COMPOUND_STMT:
          yield from walk_scope_switch(child, case_scope.sub())

    def walk_scope(parent: clx.Cursor, scope: Optional[Scope] = None) -> Generator[tuple[clx.Cursor, clx.Cursor, Scope], None, None]:
      """
      Walk the tree determining the scope of a node. here 'scope' refers not only
      to lexical scope but also to logical scope, see Scope object above
      """
      if scope is None:
        scope = Scope()

      for child in parent.get_children():
        child_kind = child.kind
        if child_kind == SWITCH_STMT:
          # switch-case statements require special treatment, we skip to the compound
          # statement
          switch_children = [c for c in child.get_children() if c.kind == COMPOUND_STMT]
          assert len(switch_children) == 1, "Switch statement has multiple '{' operators?"
          yield from walk_scope_switch(switch_children[0], scope.sub())
        elif child_kind == CALL_EXPR:
          if child.spelling in symbol_names:
            yield (child, possible_parent, scope)
        elif child_kind == COMPOUND_STMT:
          # scope has decreased
          yield from walk_scope(child, scope=scope.sub())
        else:
          # same scope
          yield from walk_scope(child, scope=scope)

    # normal lintable cursor kinds, the type of cursors we directly want to deal with
    lintable_kinds          = clx_func_call_cursor_kinds | {clx.CursorKind.ENUM_DECL}
    # "extended" lintable kinds.
    extended_lintable_kinds = lintable_kinds | {UNEXPOSED_DECL}

    cursor   = tu.cursor
    filename = tu.spelling
    for possible_parent in cursor.get_children():
      # getting filename is for some reason stupidly expensive, so we do this check first
      parent_kind = possible_parent.kind
      if parent_kind not in extended_lintable_kinds:
        continue
      try:
        if possible_parent.location.file.name != filename:
          continue
      except AttributeError:
        # possible_parent.location.file is None
        continue
      # Sometimes people declare their functions PETSC_EXTERN inline, which would normally
      # trip up the "lintable kinds" detection since the top-level cursor points to a
      # macro (i.e. unexposed decl). In this case we need to check the cursors 1 level
      # down for any lintable kinds.
      if parent_kind == UNEXPOSED_DECL:
        for sub_cursor in possible_parent.get_children():
          if sub_cursor.is_definition() and sub_cursor.kind in lintable_kinds:
            possible_parent = sub_cursor
            break
        else:
          continue
      # if we've gotten this far we have found something worth looking into, so first
      # yield the parent to process any documentation
      yield possible_parent
      if possible_parent.kind in clx_func_call_cursor_kinds:
        # then yield any children matching our function calls
        yield from walk_scope(possible_parent)

  @staticmethod
  def get_argument_cursors(func_cursor: CursorLike) -> tuple[Cursor, ...]:
    r"""Given a cursor representing a function, return a tuple of `Cursor`'s of its arguments

    Parameters
    ----------
    func_cursor :
      the function decl cursor

    Returns
    -------
    cursors :
      a tuple of `func_cursors` arguments
    """
    return tuple(Cursor(a, i) for i, a in enumerate(func_cursor.get_arguments(), start=1))

  def clear(self) -> None:
    r"""Resets the linter error, warning, and patch buffers.

    Notes
    -----
    Called automatically before parsing a file
    """
    self.errors   = DiagnosticsContainer("-" * 92)
    self.warnings = DiagnosticsContainer("%" * 92)
    # This can actually just be a straight list, since each linter object only ever
    # handles a single file, but use dict nonetheless
    self.patches  = collections.defaultdict(list)
    return

  def parse(self, filename: PathLike) -> Linter:
    r"""Parse a file for errors

    Parameters
    ----------
    filename :
      the path of the file to parse

    Returns
    -------
    self :
      the `Linter` instance
    """
    self.clear()
    self._vprint('Processing file     ', filename)
    tu = self.index.parse(str(filename), args=self.flags, options=self.clang_opts)
    if tu.diagnostics:
      self._vprint('\n'.join(map(str, tu.diagnostics)))
    self.process(tu)
    return self

  def parse_in_memory(self, src: str) -> clx.TranslationUnit:
    r"""Parse a particular source string in memory

    Parameters
    ----------
    src :
      the source string to parse

    Returns
    -------
    tu :
      the translation unit resulting from the parse

    Notes
    -----
    This lets you act as if `src` was some mini file somewhere on disk
    """
    fname = 'tempfile.cpp'
    return clx.TranslationUnit.from_source(
      fname, args=self.flags, unsaved_files=[(fname, src)], options=self.clang_opts
    )

  def _check_possible_static_function(self, func: Cursor) -> None:
    r"""Check that `func` could be make static

    Parameters
    ----------
    func :
      the function cursor to check

    Notes
    -----
    Determines whether `func` can be made static, and if so, adds the static qualifier to it. Currently
    the check is very basic, it only catches functions with are defined in a TU, and used absolutely
    nowhere else. As soon as it is defined in a header, or other file, this function bails immediately.

    We could try and figure out whether it belongs in that header, but that has many false positives.
    We would need to be able to:

    1. (reliably) distinguish between public and internal API
    2. if the function is internal API, (relibaly) determine whether it is used within the same mansec.
       If it is, we can make the decl PETSC_INTERN (if it isn't already). If it's used from multiple
       mansecs, we can make it PETSC_SINGLE_LIBRARY_INTERN (if it isn't already).

    But these are hard problems, which we leave for another time...
    """
    def cursor_is_public(cursor: CursorLike) -> bool:
      if cursor.storage_class == clx.StorageClass.EXTERN or cursor.spelling == 'main':
        return True

      for child in cursor.get_children():
        if child.kind == clx.CursorKind.VISIBILITY_ATTR and child.spelling in {'default', 'hidden'}:
          # The function cursor has a PETSC_INTERN or PETSC_EXTERN attached
          return True
      return False

    if func.kind != clx.CursorKind.FUNCTION_DECL or func.storage_class == clx.StorageClass.STATIC:
      # nothing to do
      return

    if cursor_is_public(func):
      return

    func_decl = func.get_declaration()
    if cursor_is_public(func_decl):
      # the cursor declaration has some kind of public api, be that extern, PETSC_EXTERN,
      # or whatever
      return

    lex_parent = func_decl.lexical_parent
    try:
      lex_parent_kind = lex_parent.kind
    except ValueError as ve:
      # Possible ValueError: Unknown template argument kind 300
      #
      # I think this is a bug in libclang. clx.CursorKind.TRANSLATION_UNIT is 350, I
      # think it used to be 300, and they haven't updated the python bindings?
      if 'unknown template argument kind 300' not in str(ve).casefold() and 'unknown template argument kind 350' not in str(ve).casefold():
        raise
      lex_parent_kind = clx.CursorKind.TRANSLATION_UNIT
    if lex_parent_kind == clx.CursorKind.CLASS_DECL:
      # we have a situation like
      #
      # class Foo <---- func_decl.lexical_parent
      # {
      #   friend void bar();
      #               ^^^---- func_decl
      # };
      #
      # void bar() { }
      #      ^^^------- func
      #
      # Note, I have *ONLY* seen this happen with friend functions, so let's assert that
      # that is the case here so we can potentially handle the other variants
      assert any('friend' in t.spelling for t in lex_parent.get_tokens())
      return

    origin_file = func.get_file()
    decl_file   = Cursor.get_file_from_cursor(func_decl).resolve()
    if origin_file != decl_file:
      # The function declaration is in some other file, presumably a header. This implies
      # the function is used elsewhere/is public API.
      return

    result_type     = func.result_type
    result_spelling = result_type.spelling
    if result_type.get_declaration().kind == clx.CursorKind.NO_DECL_FOUND:
      # The result type declaration (i.e. typedef x_type result_type) cannot be located!
      # This indicates 1 of 2 scenarios:
      #
      # 1. The type is built-in, e.g. int, or void, or double
      # 2. The type is actually completely uknown (likely due to linter not having the
      #    appropriate package installed). In this case, the type defaults to int, meaning
      #    that instead of searching for e.g. "BlopexInt PETSC_dpotrf_interface" the
      #    linter searches for "int PETSC_dpotrf_interface" which of course it will not
      #    find!

      # extract 'PETSC_EXTERN inline void' from 'PETSC_EXTERN inline void FooBar(...)'
      raw_result_spelling = func.raw().partition(func.spelling)[0]
      if result_spelling not in raw_result_spelling:
        # The type is likely unknown, i.e. it defaulted to int
        assert result_type.kind == clx.TypeKind.INT
        # Let's try and extract the type nonetheless
        raw_types = [
          t
          for t in raw_result_spelling.split()
            if t not in {'static', 'inline', 'extern', '\"C\"', '\"C++\"'} or
              not t.startswith(('PETSC_', 'SLEPC_'))
        ]
        if len(raw_types) > 1:
          # something we didn't handle
          return
        # we have narrowed the type down to just a single string, let's try it out
        result_spelling = raw_types[0]

    if result_spelling.endswith('*'):
      # if the result type is a pointer, it will sit flush against the function name, so
      # we should match for potentially 0 spaces, i.e.
      #
      # void *Foo()
      #
      # We don't match for exactly 0 spaces since someone may have disabled clang-format
      # and hence it's possible the pointer is not flush.
      result_type_spacing = ' *'
    else:
      # not a pointer, so result type must always be at least 1 space away from function
      # name
      result_type_spacing = ' +'
    # have to escape the pointers
    result_spelling    = result_spelling.replace('*', '\*')
    func_name_and_type = rf'{result_spelling}{result_type_spacing}{func.spelling} *\('
    # The absolute final check, we need to grep for the symbol across the code-base. This
    # is needed for cases when:
    #
    # // my_file.c
    # PetscErrorCode PetscFoo(PetscBar baz) <--- marked as a potential static candidate
    # {
    #   ...
    #
    # // my_file.h
    # #if PetscDefined(HAVE_FOO)
    # PETSC_EXTERN PetscErrorCode PetscFoo(PetscBar);
    # #endif
    #
    # In the case where the linter is not configured with PETSC_HAVE_FOO, the above checks
    # will fail to find the extern decl (since from the viewpoint of the compiler, it
    # literally does not exist) and hence produce a false positive. So the only way to
    # reliably determine is to search the text.
    #
    # The alternative is to emit the diagnostic anyway, knowing that there will be false
    # positives. To offset this, we could attach a note that says "if this is a false
    # positive, add PETSC_EXTERN/PETSC_INTERN to the definition as well".
    #
    # We chose not to do that because it makes the whole process more brittle, and
    # introduces otherwise unecessary changes just to placate the linter.
    ret = subprocess_capture_output([
      'git', '--no-pager', 'grep', '--color=never', '-r', '-l',
      '-P', # use Perl regex, which is -- for whatever reason -- over 6x faster
      '-e', func_name_and_type, '--',
      # The reason for all of this exclusion nonsense is because without it, this search
      # is __slooooow__. Running the test-lint job, even a naive search (search only
      # src and include) takes ~30s to complete. Adding these exclusions drops that to
      # just under 7s. Not great, but manageable.

      # magic git pathspecs see
      # https://git-scm.com/docs/gitglossary#Documentation/gitglossary.txt-aiddefpathspecapathspec

      # :/NAME means match NAME from the project root directory, i.e. PETSC_DIR
      ':/src',
      ':/include',
      # exclude all Fortran wrappers
      ':!**/ftn-*/**',
      # exclude docs, we don't want to match changelog mentions
      ':!**/docs/**',
      # exclude bindings (the symbol must have been declared extern somewhere else for it
      # to be usable from them anyways)
      ':!**/binding/**',
      # similarly, for the symbol to be usable from tests and tutorials it must be
      # extern somewhere else
      ':!**/tests/**',
      ':!**/tutorials/**',
      # ignore code we don't own
      ':!**/yaml/**',
      ':!**/perfstubs/**',
      # last but not least, don't search makefiles
      ':!*makefile'
    ], check=False)
    if (retcode := ret.returncode) == 0:
      # found matches
      str_origin = str(origin_file)
      for found_file in filter(len, map(str.strip, ret.stdout.splitlines())):
        # origin_file is:
        # '/full/path/to/petsc/src/sys/tests/linter/testStaticFunctionCandidates.cxx'
        # found_file is:
        # 'src/sys/tests/linter/testStaticFunctionCandidates.cxx'
        if not str_origin.endswith(found_file):
          # Some other file contains a match. We conservatively assume that the API is
          # somehow public and bail
          return
    elif retcode == 1:
      # no matches, the current file appears to be the only place this function is visible
      # from
      pass
    else:
      # something else went wrong, propagate the error
      subprocess_check_returncode(ret)

    start = SourceLocation.cast(func.extent.start, tu=func.translation_unit)
    self.add_diagnostic_from_cursor(
      func, Diagnostic(
        Diagnostic.Kind.ERROR, self.diags.static_function_candidate,
        Diagnostic.make_message_from_formattable(
          f'Function \'{func.name}()\' does not appear to be used anywhere outside of '
          f'{origin_file.name}, and can be made static',
          crange=func
        ),
        start,
        # Note the space in 'static '! start points to exactly the first letter of
        # the type, so we need to insert an extra space
        #
        #  ~~~ start
        # v
        # PetscErrorCode Foo(...)
        # {
        #    ...
        #
        patch=Patch(SourceRange.from_locations(start, start), 'static ')
      )
    )
    return

  def process(self, tu: clx.TranslationUnit) -> None:
    r"""Process a translation unit for errors

    Parameters
    ----------
    tu :
      the translation unit to process

    Notes
    -----
    This is the main entry point for the linter
    """
    from ..checks import _register

    # TODO: these check function maps need to be unified into a single dispatcher... it is
    # not very intuitive currently how to add "general" AST-matching checks.
    func_map        = _register.check_function_map
    docs_map        = _register.check_doc_map
    parsing_diag    = self.diags.parsing_error
    processed_funcs = collections.defaultdict(list)

    for results in self.find_lintable_expressions(tu, set(func_map.keys())):
      try:
        if isinstance(results, clx.Cursor):
          func = Cursor.cast(results)
          docs_map[results.kind](self, func)
        else:
          func, parent, scope = results
          func                = Cursor.cast(func)
          parent              = Cursor.cast(parent)
          func_map[func.spelling](self, func, parent)
          processed_funcs[parent.name].append((func, scope))
      except KnownUnhandleableCursorError:
        # ignored
        pass
      except ParsingError as pe:
        tu_cursor = Cursor.cast(tu.cursor)
        self.add_diagnostic_from_cursor(
          tu_cursor, Diagnostic(Diagnostic.Kind.WARNING, parsing_diag, str(pe), tu_cursor.extent.start)
        )
        # we don't want to check these
        continue
      self._check_possible_static_function(func)

    self._check_duplicate_function_calls(processed_funcs)
    return

  def add_diagnostic_from_cursor(self, cursor: Cursor, diagnostic: Diagnostic) -> None:
    r"""Given a cursor and a diagnostic, log the diagnostic with the linter

    Parameters
    ----------
    cursor :
      the cursor about which the `diagnostic` is concerned
    diagnostic :
      the diagnostic detailing the error or warning

    Raises
    ------
    TypeError
      if `cursor` is not a `Cursor`
    ValueError
      if the diagnostic kind is not handled
    """
    if not isinstance(cursor, Cursor):
      raise TypeError(type(cursor))
    if diagnostic.kind == Diagnostic.Kind.ERROR:
      container = self.errors
    elif diagnostic.kind == Diagnostic.Kind.WARNING:
      container = self.warnings
    else:
      raise ValueError(f'Unhandled diagnostic kind {diagnostic.kind}')

    if diagnostic.disabled():
      return

    container.add_diagnostic(cursor, diagnostic)
    if (patch := diagnostic.patch) is not None:
      self.patches[cursor.get_file()].append(patch)
    return

  def view_last_error(self) -> None:
    r"""Print the last error added, useful for debugging"""
    return self.errors.view_last()

  def view_last_warning(self) -> None:
    r"""Print the last warning added, useful for debugging"""
    return self.warnings.view_last()

  def get_all_errors(self) -> tuple[CondensedDiags, CondensedDiags]:
    r"""Return all errors collected so far

    Returns
    -------
    all_unresolved :
      a list of tuples of the path and message of unresolved errors (i.e. those without a `Patch`)
    all_resolved :
      a list of tuples of the path and message of resolved errors (i.e. those with a `Patch`)
    """
    return self.errors.split_and_condense()

  def get_all_warnings(self) -> tuple[CondensedDiags, CondensedDiags]:
    r"""Return all warnings collected so far

    Returns
    -------
    all_unresolved :
      a list of tuples of the path and message of unresolved warnings (i.e. those without a `Patch`)
    all_resolved :
      a list of tuples of the path and message of resolved warnings (i.e. those with a `Patch`)
      (should be empty!)
    """
    return self.warnings.split_and_condense()

  def coalesce_patches(self) -> list[PathDiffPair]:
    r"""Given a set of patches, collapse all patches and return the minimal set of diffs required

    Returns
    -------
    patches :
      the list of pairs of coalesced patches and their source files
    """
    def combine(filename: Path, patches: list[Patch]) -> PathDiffPair:
      fstr                   = str(filename)
      diffs: list[list[str]] = []
      for patch in patches:
        rn  = datetime.datetime.now().ctime()
        tmp = list(
          difflib.unified_diff(
            patch._make_source().splitlines(True), patch.collapse().splitlines(True),
            fromfile=fstr, tofile=fstr, fromfiledate=rn, tofiledate=rn, n=patch.ctxlines
          )
        )
        tmp[2] = Addline.diff_line_re.sub(Addline(patch.extent.start.line), tmp[2])
        # only the first diff should get the file heading
        diffs.append(tmp[2:] if diffs else tmp)
      return filename, ''.join(itertools.chain.from_iterable(diffs))

    def merge_patches(patch_list: list[Patch], patch: Patch) -> tuple[bool, Patch]:
      patch_extent       = patch.extent
      patch_extent_start = patch_extent.start.line
      for i, previous_patch in enumerate(patch_list):
        prev_patch_extent = previous_patch.extent
        if patch_extent_start == prev_patch_extent.start.line or patch_extent.overlaps(
            prev_patch_extent
        ):
          # this should now be the previous patch on the same line
          merged_patch = previous_patch.merge(patch)
          assert patch_list[i] == previous_patch
          del patch_list[i]
          return True, merged_patch
      return False, patch

    for patch_list in self.patches.values():
      # merge overlapping patches together before we collapse the actual patches
      # themselves
      new_list: list[Patch] = []
      for patch in sorted(patch_list, key=lambda x: x.extent.start.line):
        # we loop until we cannot merge the patch with any additional patches
        while 1:
          merged, patch = merge_patches(new_list, patch)
          if not merged:
            break
        new_list.append(patch)
      patch_list[:] = new_list

    return list(itertools.starmap(combine, self.patches.items()))

  def diagnostics(self) -> tuple[CondensedDiags, CondensedDiags, CondensedDiags, list[PathDiffPair]]:
    r"""Return the errors left (unfixed), fixed errors, warnings and avaiable patches. Automatically
    coalesces the patches

    Returns
    -------
    errors_left :
      the condensed set of filename - list of error-messages for errors that could not be patched
    errors_fixed :
      the condensed set of filename - list of error-messages for errors that are patchable
    warnings_left :
      the condensed set of filename - list of warning-messages for warnings that could not be patched
    patches :
      the list of patches corresponding to entries in `errors_fixed`

    Raises
    ------
    RuntimeError
      if there exist any fixable warnings

    Notes
    -----
    The linter technically also collects a `warnings_fixed` set, but these are not returned.
    As warnings indicate a failure of the linter to parse or understand some construct there is no
    reason for a warning to ever be fixable. These diagnostics should be errors instead.
    """
    # order is critical, coalesce_patches() will prune the patch and warning lists
    patches                       = self.coalesce_patches()
    errors_left, errors_fixed     = self.get_all_errors()
    warnings_left, warnings_fixed = self.get_all_warnings()
    if nfix := sum(map(len, warnings_fixed.values())):
      raise RuntimeError(
        f'Have {nfix} "fixable" warnings, this should not happen! '
        'If a warning has a fix then it should be an error instead!'
      )
    return errors_left, errors_fixed, warnings_left, patches
