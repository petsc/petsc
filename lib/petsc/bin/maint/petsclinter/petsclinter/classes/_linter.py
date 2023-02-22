#!/usr/bin/env python3
"""
# Created: Mon Jun 20 16:40:24 2022 (-0400)
# @author: Jacob Faibussowitsch
"""
import re
import weakref
import difflib
import datetime
import itertools
import collections
import clang.cindex as clx
import petsclinter  as pl

from ._diag    import DiagnosticManager, Diagnostic
from ._cursor  import Cursor
from ._src_pos import SourceRange
from ._patch   import Patch

class WeakList(list):
  """
  Adaptor class to make builtin lists weakly referenceable
  """
  __slots__ = ('__weakref__',)

class Scope:
  """
  Scope encompasses both the logical and lexical reach of a callsite, and is used to
  determine if two function calls may be occur in chronological order. Scopes may be
  approximated by incrementing or decrementing a counter every time a pair of '{}' are
  encountered however it is not that simple. In practice they behave almost identically
  to sets. Every relation between scopes may be formed by the following axioms.

  - Scope A is said to be greater than scope B if one is able to get to scope B from scope A
  e.g.:
  { // scope A
    { // scope B < scope A
      ...
    }
  }
  - Scope A is said to be equivalent to scope B if and only if they are the same object.
  e.g.:
  { // scope A and scope B
    ...
  }

  One notable exception are switch-case statements. Here every 'case' label acts as its
  own scope, regardless of whether a "break" is inserted i.e.:

  switch (cond) { // scope A
  case 1: // scope B begin
    ...
    break; // scope B end
  case 2: // scope C begin
    ...
  case 2:// scope C end, scope D begin
    ...
    break; // scope D end
  }

  Semantics here are weird, as:
  - scope B, C, D < scope A
  - scope B != scope C != scope D
  """
  __slots__ = 'gen', 'children'

  def __init__(self, super_scope=None):
    if super_scope:
      assert isinstance(super_scope, Scope)
      self.gen    = super_scope.gen + 1
    else:
      self.gen    = 0
    self.children = []
    return

  def __str__(self):
    return f'gen {self.gen} id {id(self)}'

  def __lt__(self, other):
    assert isinstance(other, Scope)
    return not self >= other

  def __gt__(self, other):
    assert isinstance(other, Scope)
    return self.is_child_of(other)

  def __le__(self, other):
    assert isinstance(other, Scope)
    return not self > other

  def __ge__(self, other):
    assert isinstance(other, Scope)
    return (self > other) or (self == other)

  def __eq__(self, other):
    if other is not None:
      assert isinstance(other, Scope)
      return id(self) == id(other)
    return False

  def __ne__(self, other):
    return not self == other

  def sub(self):
    """
    spawn sub-scope
    """
    child = Scope(self)
    self.children.append(child)
    return child

  def is_parent_of(self,other):
    """
    self is parent of other
    """
    if self == other:
      return False
    for child in self.children:
      if (other == child) or child.is_parent_of(other):
        return True
    return False

  def is_child_of(self,other):
    """
    self is child of other, or other is parent of self
    """
    return other.is_parent_of(self)

class Addline:
  __slots__ = ('offset',)

  def __init__(self, offset):
    self.offset = offset
    return

  def __call__(self, re_match):
    ll, lr  = re_match.group(1).split(',')
    rl, rr  = re_match.group(2).split(',')
    return f'@@ -{self.offset + int(ll)},{lr} +{self.offset + int(rl)},{rr} @@'

@DiagnosticManager.register(
  ('duplicate-function', 'Check for duplicate function-calls on the same execution path'),
)
class Linter:
  """
  Object to manage the collection and processing of errors during a lint run.
  """
  __slots__ = (
    'flags', 'clang_opts', 'verbose', 'werror', 'err_prefix', 'warn_prefix', 'index', 'errors',
    'warnings', 'patches'
  )

  def __init__(self, compiler_flags, clang_options=None, verbose=False, werror=False):
    if clang_options is None:
      clang_options = pl.util.base_clang_options

    self.flags       = compiler_flags
    self.clang_opts  = clang_options
    self.verbose     = verbose
    self.werror      = werror
    self.err_prefix  = f'{"-" * 92}'
    self.warn_prefix = f'{"%" * 92}'
    self.index       = clx.Index.create()
    self.clear()
    return

  def __str__(self):
    flag_str   = f'Compiler Flags: {self.flags}'
    clang_str  = f'Clang Options:  {self.clang_opts}'
    lock_str   = f'Lock:           {self.lock is not None}'
    show_str   = f'Verbose:        {self.verbose}'
    print_list = [flag_str, clang_str, lock_str, show_str]
    error_str  = self.get_all_errors()
    if error_str:
      print_list.append(error_str)
    warn_str = self.get_all_warnings(join_to_string=True)
    if warn_str:
      print_list.append(warn_str)
    return '\n'.join(print_list)

  def __enter__(self):
    return self

  def __exit__(self, exception_type, *args):
    if not exception_type:
      if self.verbose:
        pl.sync_print(self.get_all_warnings(join_to_string=True))
      pl.sync_print(self.get_all_errors())
    return

  def _check_duplicate_function_calls(self, processed_funcs):
    dup_diag = self.diags.duplicate_function
    for pname, function_list in processed_funcs.items():
      seen = {}
      for func, scope in function_list:
        combo = [func.displayname]
        try:
          combo.extend(map(Cursor.get_raw_name_from_cursor, func.get_arguments()))
        except pl.ParsingError:
          continue

        # convert to tuple so it is hashable
        combo = tuple(combo)
        if combo not in seen:
          seen[combo] = (func, scope)
        elif scope >= seen[combo][1]:
          start      = func.extent.start
          startline  = start.line
          tu         = func.translation_unit
          end        = clx.SourceLocation.from_position(tu, tu.get_file(tu.spelling), startline, -1)
          patch      = Patch(SourceRange.from_locations(start, end), '')
          previous   = seen[combo][0].formatted(
            nbefore=2, nafter=startline - seen[combo][0].extent.start.line
          )
          message    = f'Duplicate function found previous identical usage:\n{previous}'
          self.add_error_from_cursor(func, Diagnostic(dup_diag, message, start, patch=patch))
    return

  @staticmethod
  def find_lintable_expressions(tu, function_names):
    """
    Finds all lintable expressions in container function_names.

    Note that if a particular expression is not 100% correctly defined (i.e. would the
    file actually compile) then it will not be picked up by clang AST.

    Function-like macros can be picked up, but it will be in the wrong 'order'. The AST is
    built as if you are about to compile it, so macros are handled before any real
    function definitions in the AST, making it impossible to map a macro invocation to
    its 'parent' function.
    """
    def walk_scope_switch(parent, scope):
      """
      Special treatment for switch-case since the AST setup for it is mind-boggingly stupid.
      The first node after a case statement is listed as the cases *child* whereas every other
      node (including the break!!) is the cases *sibling*
      """
      CASE_KIND     = clx.CursorKind.CASE_STMT
      COMPOUND_KIND = clx.CursorKind.COMPOUND_STMT
      CALL_KIND     = clx.CursorKind.CALL_EXPR
      # in case we get here from a scope decrease within a case
      case_scope = scope
      for child in parent.get_children():
        child_kind = child.kind
        if child_kind == CASE_KIND:
          # create a new scope every time we encounter a case, this is now for all intents
          # and purposes the 'scope' going forward. We don't overwrite the original scope
          # since we still need each case scope to be the previous scopes sibling
          case_scope = scope.sub()
          yield from walk_scope(child, scope=case_scope)
        elif child_kind == CALL_KIND:
          if child.spelling in function_names:
            yield (child, possible_parent, case_scope)
            # Cursors that indicate change of logical scope
        elif child_kind == COMPOUND_KIND:
          yield from walk_scope_switch(child, case_scope.sub())

    def walk_scope(parent, scope=None):
      """
      Walk the tree determining the scope of a node. here 'scope' refers not only
      to lexical scope but also to logical scope, see Scope object above
      """
      if scope is None:
        scope = Scope()

      SWITCH_KIND   = clx.CursorKind.SWITCH_STMT
      COMPOUND_KIND = clx.CursorKind.COMPOUND_STMT
      CALL_KIND     = clx.CursorKind.CALL_EXPR
      for child in parent.get_children():
        child_kind = child.kind
        if child_kind == SWITCH_KIND:
          # switch-case statements require special treatment, we skip to the compound
          # statement
          switch_children = [c for c in child.get_children() if c.kind == COMPOUND_KIND]
          assert len(switch_children) == 1, "Switch statement has multiple '{' operators?"
          yield from walk_scope_switch(switch_children[0], scope.sub())
        elif child_kind == CALL_KIND:
          if child.spelling in function_names:
            yield (child, possible_parent, scope)
        elif child_kind == COMPOUND_KIND:
          # scope has decreased
          yield from walk_scope(child, scope=scope.sub())
        else:
          # same scope
          yield from walk_scope(child, scope=scope)

    # normal lintable cursor kinds, the type of cursors we directly want to deal with
    lintable_kinds          = pl.util.clx_func_call_cursor_kinds | {clx.CursorKind.ENUM_DECL}
    # "extended" lintable kinds.
    extended_lintable_kinds = lintable_kinds | {clx.CursorKind.UNEXPOSED_DECL}

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
      if parent_kind == clx.CursorKind.UNEXPOSED_DECL:
        for sub_cursor in possible_parent.get_children():
          if sub_cursor.is_definition() and sub_cursor.kind in lintable_kinds:
            possible_parent = sub_cursor
            break
        else:
          continue
      # if we've gotten this far we have found something worth looking into, so first
      # yield the parent to process any documentation
      yield possible_parent
      if possible_parent.kind in pl.util.clx_func_call_cursor_kinds:
        # then yield any children matching our function calls
        yield from walk_scope(possible_parent)

  @staticmethod
  def get_argument_cursors(func_cursor):
    """
    Given a cursor representing a function, return a tuple of Cursor's of its arguments
    """
    return tuple(Cursor(a, i) for i, a in enumerate(func_cursor.get_arguments(), start=1))


  def clear(self):
    """
    Resets the linter error, warning, and patch buffers.
    Called automatically before parsing a file
    """
    self.errors   = collections.OrderedDict()
    self.warnings = []
    # This can actually just be a straight list, since each linter object only ever
    # handles a single file, but use dict nonetheless
    self.patches  = {}
    return

  def parse(self, filename):
    """
    Parse a file for errors
    """
    self.clear()
    if self.verbose:
      pl.sync_print('Processing file     ', filename)
    tu = self.index.parse(str(filename), args=self.flags, options=self.clang_opts)
    if self.verbose and tu.diagnostics:
      pl.sync_print('\n'.join(map(str, tu.diagnostics)))
    self.process(tu)
    return self

  def parse_in_memory(self, src):
    fname = 'tempfile.cpp'
    return clx.TranslationUnit.from_source(
      fname, args=self.flags, unsaved_files=[(fname, src)], options=self.clang_opts
    )

  @DiagnosticManager.register(('parsing-error', 'Generic parsing errors'))
  def process(self, tu):
    """
    Process a translation unit for errors
    """
    func_map        = pl.checks._register.check_function_map
    docs_map        = pl.checks._register.check_doc_map
    parsing_diag    = self.process.diags.parsing_error
    processed_funcs = collections.defaultdict(list)

    for results in self.find_lintable_expressions(tu, set(func_map.keys())):
      try:
        if isinstance(results, clx.Cursor):
          docs_map[results.kind](self, results)
        else:
          func, parent, scope = results
          func_map[func.spelling](self, func, parent)
          processed_funcs[Cursor.get_name_from_cursor(parent)].append((Cursor.cast(func), scope))
      except pl.KnownUnhandleableCursorError as kuce:
        # ignored
        pass
      except pl.ParsingError as pe:
        tu_cursor = tu.cursor
        self.add_warning(tu_cursor.spelling, Diagnostic(parsing_diag, str(pe), tu_cursor.extent.start))
    self._check_duplicate_function_calls(processed_funcs)
    return

  def add_error_from_cursor(self, cursor, diagnostic):
    """
    Given a cursor attach a diagnostic error message to it, and optionally a fix
    """
    if diagnostic.disabled():
      return

    cursor   = Cursor.cast(cursor)
    filename = cursor.get_file()

    if filename not in self.errors:
      self.errors[filename] = collections.OrderedDict()

    errors    = self.errors[filename]
    cursor_id = cursor.hash
    if cursor_id not in errors:
      header = f'\nERROR {len(errors)}: {str(cursor)}\n'
      errors[cursor_id] = WeakList([header, [], [], []])

    patch            = diagnostic.patch
    have_patch       = patch is not None
    cursor_id_errors = errors[cursor_id]
    cursor_id_errors[1].append(diagnostic.format_message())
    cursor_id_errors[2].append(have_patch)
    cursor_id_errors[3].append(patch.id if have_patch else -1)

    if not have_patch:
      return # bail early

    patch.attach(weakref.ref(cursor_id_errors))
    patches = self.patches

    if filename not in patches:
      patches[filename] = []
    # check if this is a compound error, i.e. an additional error on the same line
    # in which case we need to combine with previous patch
    patches[filename].append(patch)
    return
    # OLD IMPLEMENTATION
    # patch_list = patches[filename]

    # def merge_patches(patch):
    #   patch_extent       = patch.extent
    #   patch_extent_start = patch_extent.start.line
    #   for i, previous_patch in enumerate(patch_list):
    #     prev_patch_extent = previous_patch.extent
    #     if patch_extent_start == prev_patch_extent.start.line or patch_extent.overlaps(
    #         prev_patch_extent
    #     ):
    #       # this should now be the previous patch on the same line
    #       merged_patch = previous_patch.merge(patch)
    #       assert patch_list[i] == previous_patch
    #       del patch_list[i]
    #       return True, merged_patch
    #   return False, patch


    # while 1:
    #   merged, patch = merge_patches(patch)
    #   if not merged:
    #     break

    # patch_list.append(patch) # didn't find any overlap, just append
    # return

  def view_last_error(self):
    """
    Print the last error added, useful for debugging
    """
    for files in reversed(self.errors):
      errors = self.errors[files]
      last   = errors[next(reversed(errors))]
      pl.sync_print(last[0], last[1][-1])
      break
    return

  def add_warning(self, filename, diag):
    """
    Add a generic warning given a filename
    """
    if self.werror:
      self.add_error_from_cursor(filename, diag)
      return

    if diag.disabled():
      return

    warn_msg = diag.format_message()
    try:
      if warn_msg in self.warnings[-1][1]:
        # we just had the exact same warning, we can ignore it. This happens very often
        # for warnings occurring deep within a macro
        return
    except IndexError:
      pass
    self.warnings.append((filename, f'\nWARNING {len(self.warnings)}: {warn_msg}'))
    return

  def add_warning_from_cursor(self, cursor, diag):
    """
    Given a cursor attach a diagnostic warning message to it
    """
    if self.werror:
      self.add_error_from_cursor(cursor, diag)
      return

    if diag.disabled():
      return

    cursor   = Cursor.cast(cursor)
    warn_str = f'\nWARNING {len(self.warnings)}: {str(cursor)}\n{diag.format_message()}'
    self.warnings.append((cursor.get_file(), warn_str))
    return

  def get_all_errors(self):
    """
    Return all errors collected so far in a tuple
    """
    def maybe_add_to_global_list(global_list, local_list, path):
      if local_list:
        global_list.append((
          path, '{prefix}\n{}\n{prefix}'.format('\n'.join(local_list)[1:], prefix=self.err_prefix)
        ))
      return

    def maybe_add_to_local_list(local_list, thing, mask, header):
      string = '\n\n'.join(itertools.compress(thing, mask))
      if string:
        local_list.append(f'{header}{string}')
      return


    all_unresolved, all_resolved = [], []
    for path, errors in self.errors.items():
      unresolved, resolved = [], []
      for header, errs, mask, _ in errors.values():
        maybe_add_to_local_list(resolved, errs, mask, header)
        maybe_add_to_local_list(unresolved, errs, [not m for m in mask], header)
      maybe_add_to_global_list(all_unresolved, unresolved, path)
      maybe_add_to_global_list(all_resolved, resolved, path)
    return all_unresolved, all_resolved

  def get_all_warnings(self, join_to_string=False):
    """
    Return all warnings collected so far, and optionally join them all as one string
    """
    if join_to_string:
      if self.warnings:
        return '\n'.join([
          self.warn_prefix, '\n'.join(s for _, s in self.warnings)[1:], self.warn_prefix
        ])
      return ''
    return self.warnings

  def coalesce_patches(self):
    """
    Given a set of patches, collapse all patches and return the minimal set of diffs required
    """
    diff_line_re = re.compile(r'^@@ -([0-9,]+) \+([0-9,]+) @@')

    def combine(filename, patches):
      fstr  = str(filename)
      diffs = []
      for patch in sorted(patches, key=lambda x: x.extent.start.line):
        rn  = datetime.datetime.now().ctime()
        tmp = list(
          difflib.unified_diff(
            patch._make_source().splitlines(True), patch.collapse().splitlines(True),
            fromfile=fstr, tofile=fstr, fromfiledate=rn, tofiledate=rn, n=patch.ctxlines
          )
        )
        tmp[2] = diff_line_re.sub(Addline(patch.extent.start.line), tmp[2])
        # only the first diff should get the file heading
        diffs.append(tmp[2:] if diffs else tmp)
      diffs = ''.join(itertools.chain.from_iterable(diffs))
      return filename, diffs

    def merge_patches(patch_list, patch):
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


    for _, patch_list in self.patches.items():
      # merge overlapping patches together before we collapse the actual patches
      # themselves
      new_list = []
      for patch in sorted(patch_list, key=lambda x: x.extent.start.line):
        # we loop until we cannot merge the patch with any additional patches
        while 1:
          merged, patch = merge_patches(new_list, patch)
          if not merged:
            break
        new_list.append(patch)
      patch_list[:] = new_list

    return list(itertools.starmap(combine, self.patches.items()))

  def diagnostics(self):
    """
    Return the errors left (unfixed), fixed errors, warnings and avaiable patches. Automatically
    coalesces the patches
    """
    # order is ciritical, coalesce_patches() will prune the patch and warning lists
    patches = self.coalesce_patches()
    errors_left, errors_fixed = self.get_all_errors()
    warnings = self.get_all_warnings()
    return errors_left, errors_fixed, warnings, patches
