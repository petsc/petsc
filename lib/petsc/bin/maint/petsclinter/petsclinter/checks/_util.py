#!/usr/bin/env python3
"""
# Created: Mon Jun 20 17:06:22 2022 (-0400)
# @author: Jacob Faibussowitsch
"""
from __future__ import annotations

import itertools
import clang.cindex as clx # type: ignore[import]

from .._typing import *
from .._error  import ParsingError, ClassidNotRegisteredError

from ..classes._diag    import DiagnosticManager, Diagnostic
from ..classes._cursor  import Cursor
from ..classes._patch   import Patch
from ..classes._src_pos import SourceRange, SourceLocation

from ..util._clang import (
  clx_scalar_type_kinds, clx_enum_type_kinds, clx_int_type_kinds, clx_conversion_cursor_kinds,
  clx_function_type_kinds, clx_var_token_kinds, clx_math_cursor_kinds, clx_pointer_type_kinds
)

# utilities for checking functions

# a dummy class so that type checkers don't complain about functions not having diags attribute
@DiagnosticManager.register(
  ('incompatible-function', 'Verify that the correct function was used for a type'),
  ('incompatible-type', 'Verify that a particular type matches the expected type'),
  ('incompatible-type-petscobject', 'Verify that a symbol is a PetscObject'),
  ('incompatible-classid', 'Verify that the given classid matches the PetscObject type'),
  ('matching-arg-num', 'Verify that the given argument number matches')
)
class _CodeDiags:
  diags: DiagnosticMap

def add_function_fix_to_bad_source(linter: Linter, obj: Cursor, func_cursor: Cursor, valid_func_name: str) -> None:
  r"""Shorthand for extracting a fix from a function cursor

  Parameters
  ----------
  linter :
    the linter instance
  obj :
    the cursor representing the object
  func_cursor :
    the cursor of the parent function
  valid_func_name :
    the name that should be used instead
  """
  call = [
    c for c in func_cursor.get_children() if c.type.get_pointee().kind == clx.TypeKind.FUNCTIONPROTO
  ]
  assert len(call) == 1
  mess = f'Incorrect use of {func_cursor.displayname}(), use {valid_func_name}() instead:\n{Cursor.get_formatted_source_from_cursor(func_cursor, num_context=2)}'
  linter.add_diagnostic_from_cursor(
    obj,
    Diagnostic(
      Diagnostic.Kind.ERROR, _CodeDiags.diags.incompatible_function, mess,
      func_cursor.extent.start, patch=Patch.from_cursor(call[0], valid_func_name)
    ).add_note(
      f'Due to {obj.get_formatted_blurb()}', location=obj.extent.start
    )
  )
  return

def convert_to_correct_PetscValidLogicalCollectiveXXX(linter: Linter, obj: Cursor, obj_type: clx.Type, func_cursor: Cursor, **kwargs) -> bool:
  r"""Try to glean the correct PetscValidLogicalCollectiveXXX from the type.

  Parameters
  ----------
  linter :
    the linter instance
  obj :
    the cursor for the object to check
  obj_type :
    the type of obj
  func_cursor :
    the cursor representing the parent function

  Notes
  -----
  Used as a failure hook in the validlogicalcollective checks.
  """
  valid_func_name = ''
  obj_type_kind   = obj_type.kind
  if obj_type_kind in clx_scalar_type_kinds:
    if 'PetscReal' in obj.derivedtypename:
      valid_func_name = 'PetscValidLogicalCollectiveReal'
    elif 'PetscScalar' in obj.derivedtypename:
      valid_func_name = 'PetscValidLogicalCollectiveScalar'
  elif obj_type_kind in clx_enum_type_kinds:
    if 'PetscBool' in obj.derivedtypename:
      valid_func_name = 'PetscValidLogicalCollectiveBool'
    else:
      valid_func_name = 'PetscValidLogicalCollectiveEnum'
  elif obj_type_kind in clx_int_type_kinds:
    if 'PetscInt' in obj.derivedtypename:
      valid_func_name = 'PetscValidLogicalCollectiveInt'
    elif 'PetscMPIInt' in obj.derivedtypename:
      valid_func_name = 'PetscValidLogicalCollectiveMPIInt'
  if valid_func_name:
    add_function_fix_to_bad_source(linter, obj, func_cursor, valid_func_name)
  return bool(valid_func_name)

def check_is_type_x_and_not_type_y(type_x: str, type_y: str, linter: Linter, obj: Cursor, obj_type: clx.Type, func_cursor: Optional[Cursor] = None, valid_func: str = '') -> bool:
  r"""Check that a cursor is at least some form of derived type X and not some form of type Y

  Parameters
  ----------
  type_x :
    the name of type "x"
  type_y :
    the name that `type_x` should not be
  linter :
    the linter instance
  obj :
    the object which is of `type_x`
  obj_type :
    the type of `obj`
  func_cursor : optional
    the cursor representing the parent function
  valid_func : optional
    the name of the valid function name

  Returns
  -------
  ret :
    True, as this routine always fixes the problem

  Notes
  -----
  i.e. for

  myInt **********x;

  you may check that 'x' is some form of 'myInt' instead of say 'PetscBool'
  """
  derived_name = obj.derivedtypename
  if type_x not in derived_name:
    if type_y in derived_name:
      assert func_cursor is not None
      add_function_fix_to_bad_source(linter, obj, func_cursor, valid_func)
    else:
      mess = f'Incorrect use of {valid_func}(), {valid_func}() should only be used for {type_x}'
      linter.add_diagnostic_from_cursor(
        obj,
        Diagnostic(
          Diagnostic.Kind.ERROR, _CodeDiags.diags.incompatible_type, mess,
          obj.extent.start
        )
      )
  return True

def check_is_PetscScalar_and_not_PetscReal(*args, **kwargs) -> bool:
  r"""Check that a cursor is a PetscScalar and not a PetscReal

  Parameters
  ----------
  *args :
    positional arguments to `check_is_type_x_and_not_type_y()`
  **kwargs :
    keyword arguments to `check_is_type_x_and_not_type_y()`

  Returns
  -------
  ret :
    the return value of `check_is_type_x_and_not_type_y()`
  """
  return check_is_type_x_and_not_type_y('PetscScalar', 'PetscReal', *args, **kwargs)

def check_is_PetscReal_and_not_PetscScalar(*args, **kwargs) -> bool:
  r"""Check that a cursor is a PetscReal and not a PetscScalar

  Parameters
  ----------
  *args :
    positional arguments to `check_is_type_x_and_not_type_y()`
  **kwargs :
    keyword arguments to `check_is_type_x_and_not_type_y()`

  Returns
  -------
  ret :
    the return value of `check_is_type_x_and_not_type_y()`
  """
  return check_is_type_x_and_not_type_y('PetscReal', 'PetscScalar', *args, **kwargs)

def check_is_not_type(typename: str, linter: Linter, obj: Cursor, func_cursor: Cursor, valid_func: str = '') -> bool:
  r"""Check a cursor is not of type `typename`

  Parameters
  ----------
  typename :
    the type that the cursor should not be
  linter :
    the linter instance
  obj :
    the cursor representing the object
  func_cursor : optional
    the cursor representing the parent function
  valid_func : optional
    the name of the valid function name

  Returns
  -------
  ret :
    True, since this routine always fixes the problem
  """
  if typename in obj.derivedtypename:
    add_function_fix_to_bad_source(linter, obj, func_cursor, valid_func)
  return True

def check_int_is_not_PetscBool(linter: Linter, obj: Cursor, *args, **kwargs) -> bool:
  r"""Check an int-like object is not a PetscBool

  Parameters
  ----------
  linter :
    the linter instance
  obj :
    the cursor representing the object
  *args :
    additional positional arguments to `check_is_not_type()`
  **kwargs :
    additional keyword arguments to `check_is_not_type()`

  Returns
  -------
  ret :
    the return value of `check_is_not_type()`
  """
  return check_is_not_type('PetscBool', linter, obj, **kwargs)

def check_MPIInt_is_not_PetscInt(linter: Linter, obj: Cursor, *args, **kwargs) -> bool:
  r"""Check a PetscMPIInt object is not a PetscBool

  Parameters
  ----------
  linter :
    the linter instance
  obj :
    the cursor representing the object
  *args :
    additional positional arguments to `check_is_not_type()`
  **kwargs :
    additional keyword arguments to `check_is_not_type()`

  Returns
  -------
  ret :
    the return value of `check_is_not_type()`
  """
  return check_is_not_type('PetscInt', linter, obj, **kwargs)

def check_is_PetscBool(linter: Linter, obj: Cursor, obj_type: clx.Type, func_cursor: Optional[Cursor] = None, valid_func: str = '') -> bool:
  r"""Check that a cursor is exactly a PetscBool

  Parameters
  ----------
  linter :
    the linter instance
  obj :
    the cursor representing the object
  obj_type :
    the type of obj
  func_cursor : optional
    the cursor representing the parent function
  valid_func_name : optional
    the name that should be used instead, unused
  """
  if ('PetscBool' not in obj.derivedtypename) and ('bool' not in obj.typename):
    assert func_cursor is not None
    func_name = func_cursor.displayname
    mess      = f'Incorrect use of {func_name}(), {func_name}() should only be used for PetscBool or bool:{Cursor.get_formatted_source_from_cursor(func_cursor, num_context=2)}'
    linter.add_diagnostic_from_cursor(
      obj,
      Diagnostic(
        Diagnostic.Kind.ERROR, _CodeDiags.diags.incompatible_function, mess, obj.extent.start
      )
    )
  return True

def check_is_petsc_object(linter: Linter, obj: Cursor) -> bool:
  r"""Check if `obj` is a PetscObject

  Parameters
  ----------
  linter :
    the linter instance
  obj :
    the cursor to check

  Returns
  -------
  is_valid :
    True if `obj` is a valid PetscObject, False otherwise.

  Raises
  ------
  ClassidNotRegisteredError
    if `obj` is a PetscObject that isn't registered in the classid_map.
  """
  from ._register import classid_map

  if obj.typename not in classid_map:
    # Raise exception here since this isn't a bad source, moreso a failure of this script
    # since it should know about all PETSc classes
    err_message = f"{obj}\nUnknown or invalid PETSc class '{obj.derivedtypename}'. If you are introducing a new class, you must register it with this linter! See lib/petsc/bin/maint/petsclinter/petsclinter/checks/_register.py and search for 'Adding new classes' for more information\n"
    raise ClassidNotRegisteredError(err_message)
  petsc_object_type = obj.type.get_canonical().get_pointee()
  # Must have a struct here, e.g. _p_Vec
  assert petsc_object_type.kind == clx.TypeKind.RECORD, 'Symbol does not appear to be a struct!'
  is_valid = True
  # PetscObject is of course a valid PetscObject, no need to check its members
  if petsc_object_type.spelling != '_p_PetscObject':
    struct_fields = [f for f in petsc_object_type.get_fields()]
    if len(struct_fields) >= 2:
      if Cursor.get_typename_from_cursor(struct_fields[0]) != '_p_PetscObject':
        is_valid = False
    else:
      is_valid = False

  if not is_valid:
    obj_decl = Cursor.cast(petsc_object_type.get_declaration())
    typename = obj_decl.typename
    diag     = Diagnostic(
      Diagnostic.Kind.ERROR, _CodeDiags.diags.incompatible_type_petscobject,
      f'Invalid type \'{typename}\', expected a PetscObject (or derived class):\n{obj_decl.formatted(num_before_context=1, num_after_context=2)}',
      obj_decl.extent.start
    )
    if typename.startswith('_p_'):
      if len(struct_fields) == 0:
        reason = 'cannot determine fields. Likely the header containing definition of the object is in a nonstandard place'
        # raise a warning instead of an error since this is a failure of the linter
        diag.kind = Diagnostic.Kind.WARNING
      else:
        reason = 'its definition is missing a PETSCHEADER as the first struct member'
      mess = f'Type \'{typename}\' is prefixed with \'_p_\' to indicate it is a PetscObject but {reason}. Either replace \'_p_\' with \'_n_\' to indicate it is not a PetscObject or add a PETSCHEADER declaration as the first member.'
      diag.add_note(mess).add_note(
        f'It is ambiguous whether \'{obj.name}\' is intended to be a PetscObject.'
      )
    linter.add_diagnostic_from_cursor(obj, diag)
  return is_valid

def check_matching_classid(linter: Linter, obj: Cursor, obj_classid: Cursor) -> None:
  r"""Does the classid match the particular PETSc type

  Parameters
  ----------
  linter :
    the linter instance
  obj :
    the cursor to check
  obj_classid :
    the cursor for the classid
  """
  from ._register import classid_map

  check_is_petsc_object(linter, obj)
  expected = classid_map[obj.typename]
  name     = obj_classid.name
  if name != expected:
    mess     = f"Classid doesn't match. Expected '{expected}' found '{name}':\n{obj_classid.formatted(num_context=2)}"
    diag = Diagnostic(
      Diagnostic.Kind.ERROR, _CodeDiags.diags.incompatible_classid, mess, obj_classid.extent.start,
      patch=Patch.from_cursor(obj_classid, expected)
    ).add_note(
      f'For {obj.get_formatted_blurb()}', location=obj.extent.start
    )
    linter.add_diagnostic_from_cursor(obj, diag)
  return

def _do_check_traceable_to_parent_args(obj: Cursor, parent_arg_names: tuple[str, ...], trace: list[CursorLike]) -> tuple[int, list[CursorLike]]:
  """
  The actual workhorse of `check_traceable_to_parent_args()`
  """

  potential_parents: list[CursorLike] = []
  if def_cursor := obj.get_definition():
    assert def_cursor.location != obj.location, 'Object has definition cursor, yet the cursor did not move. This should be handled!'
    if def_cursor.kind == clx.CursorKind.VAR_DECL:
      # found definition, so were in business
      # Parents here is an odd choice of words since on the very same line I loop
      # over children, but then again clangs AST has an odd semantic for parents/children
      convert_or_dereference_cursors = clx_conversion_cursor_kinds | {clx.CursorKind.UNARY_OPERATOR}
      for def_child in def_cursor.get_children():
        if def_child.kind in convert_or_dereference_cursors:
          decl_ref_gen = (
            child for child in def_child.walk_preorder() if child.kind == clx.CursorKind.DECL_REF_EXPR
          )
          # Weed out any self-references
          potential_parents_gen = (
            parent for parent in decl_ref_gen if parent.spelling != def_cursor.spelling
          )
          potential_parents.extend(potential_parents_gen)
    elif def_cursor.kind == clx.CursorKind.FIELD_DECL:
      # we have deduced that the original cursor may refer to a struct member
      # reference, so we go back and see if indeed this is the case
      for member_child in obj.get_children():
        if member_child.kind == clx.CursorKind.MEMBER_REF_EXPR:
          decl_ref_gen = (
            c for c in member_child.walk_preorder() if c.kind == clx.CursorKind.DECL_REF_EXPR
          )
          assert member_child.spelling == def_cursor.spelling, f'{member_child.spelling=}, {def_cursor.spelling=}'
          potential_parents_gen = (
            parent for parent in decl_ref_gen if parent.spelling != member_child.spelling
          )
          potential_parents.extend(potential_parents_gen)
  elif obj.kind in clx_conversion_cursor_kinds:
    curs = [
      Cursor(c, obj.argidx) for c in obj.walk_preorder() if c.kind == clx.CursorKind.DECL_REF_EXPR
    ]
    if len(curs) > 1:
      curs = [c for c in curs if c.displayname == obj.name]
    assert len(curs) == 1, f'Could not uniquely determine base cursor from conversion cursor {obj}'
    obj = curs[0]
    # for cases with casting + struct member reference:
    #
    # macro((type *)bar->baz, barIdx);
    #
    # the object "name" will (rightly) refer to 'baz', but since this is an inline
    # "definition" it doesn't show up in get_definition(), thus we check here
    potential_parents.append(obj)

  if not potential_parents:
    # this is the if-all-else-fails approach, first we search the __entire__ file for
    # references to the cursor. Once we have some matches we take the earliest one
    # as this one is in theory where the current cursor is instantiated. Then we select
    # the best match for the possible instantiating cursor and recursively call this
    # function. This section stops when the cursor definition is of type PARM_DECL (i.e.
    # defined as the function parameter to the parent function).
    all_possible_references = obj.find_cursor_references()
    # don't care about uses of object __after__ the macro, and don't want to pick up
    # the actual macro location either
    all_possible_references = [
      r for r in all_possible_references if r.location.line < obj.location.line
    ]
    # we just tried those and they didn't work, also more importantly weeds out the
    # instantiation line if this is an intermediate cursor in a recursive call to this
    # function
    decl_cursor_kinds = {clx.CursorKind.VAR_DECL, clx.CursorKind.FIELD_DECL}
    arg_refs          = [r for r in all_possible_references if r.kind not in decl_cursor_kinds]
    if not len(arg_refs):
      # it's not traceable to a function argument, so maybe its a global static variable
      if len([r for r in all_possible_references if r.storage_class == clx.StorageClass.STATIC]):
        # a global variable is not a function argumment, so this is unhandleable
        raise ParsingError('PETSC_CLANG_STATIC_ANALYZER_IGNORE')

    assert len(arg_refs), f'Could not determine the origin of cursor {obj}'
    # take the first, as this is the earliest
    first_ref = arg_refs[0]
    tu, line  = first_ref.translation_unit, first_ref.location.line
    src_len   = len(first_ref.raw())
    # Why the following song and dance? Because you cannot walk the AST backwards, and in
    # the case that the current cursor is in a function call we need to access our
    # co-arguments to the function, i.e. "adjacent" branches since they should link to (or
    # be) in the parent functions argument list. So we have to essentially reparse this
    # line to be able to start from the top.
    line_start         = SourceLocation.from_position(tu, line, 1)
    line_end           = SourceLocation.from_position(tu, line, src_len + 1)
    line_range         = SourceRange.from_locations(line_start, line_end).source_range
    token_group        = list(clx.TokenGroup.get_tokens(tu, line_range))
    function_prototype = [
      i for i, t in enumerate(token_group) if t.cursor.type.get_canonical().kind in clx_function_type_kinds
    ]
    if function_prototype:
      if len(function_prototype) > 1:
        # nested function calls likely from PetscCall(), so discard that
        function_prototype = [
          i for i in function_prototype if not token_group[i].spelling.startswith('PetscCall')
        ]
      assert len(function_prototype) == 1, 'Could not determine unique function prototype from {} for provenance of {}'.format(''.join([t.spelling for t in token_group]), obj)

      idx         = function_prototype[0]
      lambda_expr = lambda t: (t.spelling not in  {'(', ')'}) and t.kind in clx_var_token_kinds
      iterator    = (x.cursor for x in itertools.takewhile(lambda_expr, token_group[idx + 2:]))
    # we now have completely different cursor selected, so we recursively call this
    # function
    else:
      # not a function call, must be an assignment statement, meaning we should now
      # assert that the current obj is being assigned to
      assert Cursor.get_name_from_cursor(token_group[0].cursor) == obj.name
      # find the binary operator, it will contain the most comprehensive AST
      cursor_gen = token_group[[x.spelling for x in token_group].index('=')].cursor.walk_preorder()
      iterator   = (c for c in cursor_gen if c.kind == clx.CursorKind.DECL_REF_EXPR)

    alternate_cursor = (c for c in iterator if Cursor.get_name_from_cursor(c) != obj.name)
    potential_parents.extend(alternate_cursor)

  if not potential_parents:
    raise ParsingError
  # arguably at this point anything other than len(potential_parents) should be 1,
  # and anything else can be considered a failure of this routine (therefore a RTE)
  # as it should be able to detect the definition.
  assert len(potential_parents) == 1, 'Cannot determine a unique definition cursor for object'
  # If >1 cursor, probably a bug since we should have weeded something out
  parent = potential_parents[0]
  trace.append(parent)
  par_def = parent.get_definition()
  if par_def and par_def.kind == clx.CursorKind.PARM_DECL:
    name = Cursor.get_name_from_cursor(parent)
    try:
      loc = parent_arg_names.index(name)
    except ValueError as ve:
      # name isn't in the parent arguments, so we raise parsing error from it
      raise ParsingError from ve
  else:
    parent = Cursor(parent, obj.argidx)
    # deeper into the rabbit hole
    loc, trace = _do_check_traceable_to_parent_args(parent, parent_arg_names, trace)
  return loc, trace

def check_traceable_to_parent_args(obj: Cursor, parent_arg_names: tuple[str, ...]) -> tuple[int, list[CursorLike]]:
  r"""Try and see if the cursor can be linked to parent function arguments.

  Parameters
  ----------
  obj :
    the cursor to check
  parent_arg_names :
    the set of argument names in the parent function

  Returns
  -------
  idx :
    the index into the `parent_arg_names` to which `obj` has been traced
  trace :
    if `obj` is not directly in `parent_arg_names`, then the breadcrumb of cursors that lead from `obj`
    to `parent_arg_names[idx]`

  Raises
  ------
  ParsingError
    if `obj` cannot be traced back to `parent_arg_names`

  Notes
  -----
  Traces `foo` back to `bar` for example for these
  ```
  myFunction(barType bar)
  ...
  fooType foo = bar->baz;
  macro(foo, barIdx);
  /* or */
  macro(bar->baz, barIdx);
  /* or */
  initFooFromBar(bar,&foo);
  macro(foo, barIdx);
  ```
  """
  return _do_check_traceable_to_parent_args(obj, parent_arg_names, [])

def check_matching_arg_num(linter: Linter, obj: Cursor, idx_cursor: Cursor, parent_args: tuple[Cursor, ...]) -> None:
  r"""Is the Arg # correct w.r.t. the function arguments?

  Parameters
  ----------
  linter :
    the linter instance
  obj :
    the cursor to check
  idx :
    the cursor of the arg idx
  parent_args :
    the set of parent function argument cursors
  """
  diag_name = _CodeDiags.diags.matching_arg_num
  if idx_cursor.canonical.kind not in clx_math_cursor_kinds:
    # sometimes it is impossible to tell if the index is correct so this is a warning not
    # an error. For example in the case of a loop:
    # for (i = 0; i < n; ++i) PetscAssertPointer(arr+i, i);
    linter.add_diagnostic_from_cursor(
      idx_cursor, Diagnostic(
        Diagnostic.Kind.WARNING, diag_name,
        f"Index value is of unexpected type '{idx_cursor.canonical.kind}'", obj.extent.start
      )
    )
    return
  try:
    idx_num = int(idx_cursor.name)
  except ValueError:
    linter.add_diagnostic_from_cursor(
      idx_cursor, Diagnostic(
        Diagnostic.Kind.WARNING, diag_name,
        'Potential argument mismatch, could not determine integer value', obj.extent.start
      )
    )
    return
  trace: list[CursorLike] = []
  parent_arg_names        = tuple(s.name for s in parent_args)
  try:
    expected = parent_args[parent_arg_names.index(obj.name)]
  except ValueError:
    try:
      parent_idx, trace = check_traceable_to_parent_args(obj, parent_arg_names)
    except ParsingError as pe:
      # If the parent arguments don't contain the symbol and we couldn't determine a
      # definition then we cannot check for correct numbering, so we cannot do
      # anything here but emit a warning
      if 'PETSC_CLANG_STATIC_ANALYZER_IGNORE' in pe.args:
        return
      if len(parent_args):
        parent_func      = Cursor(parent_args[0].semantic_parent)
        parent_func_name = f'{parent_func.name}()'
        parent_func_src  = parent_func.formatted()
      else:
        # parent function has no arguments (very likely that "obj" is a global variable)
        parent_func_name = 'UNKNOWN FUNCTION'
        parent_func_src  = '  <could not determine parent function signature from arguments>'
      mess = f"Cannot determine index correctness, parent function '{parent_func_name}' seemingly does not contain the object:\n{parent_func_src}"
      linter.add_diagnostic_from_cursor(
        obj, Diagnostic(Diagnostic.Kind.WARNING, diag_name, mess, obj.extent.start)
      )
      return
    else:
      expected = parent_args[parent_idx]
  exp_idx = expected.argidx
  if idx_num != exp_idx:
    diag = Diagnostic(
      Diagnostic.Kind.ERROR, diag_name,
      f"Argument number doesn't match for '{obj.name}'. Expected '{exp_idx}', found '{idx_num}':\n{idx_cursor.formatted(num_context=2)}",
      idx_cursor.extent.start, patch=Patch.from_cursor(idx_cursor, str(exp_idx))
    ).add_note(
      f'\'{obj.name}\' is traceable to argument #{exp_idx} \'{expected.name}\' in enclosing function here:\n{expected.formatted(num_context=2)}',
      location=expected.extent.start
    )
    if trace:
      diag.add_note(f'starting with {obj.get_formatted_blurb().rstrip()}', location=obj.extent.start)
    for cursor in trace:
      diag.add_note(
        f'via {Cursor.get_formatted_blurb_from_cursor(cursor).rstrip()}', location=cursor.extent.start
      )
    linter.add_diagnostic_from_cursor(idx_cursor, diag)
  return

if TYPE_CHECKING:
  MatchingTypeCallback = Callable[[Linter, Cursor, Optional[Cursor], str], bool]

def check_matching_specific_type(
    linter: Linter, obj: Cursor, expected_type_kinds: Collection[clx.TypeKind], pointer: bool,
    unexpected_not_pointer_function: Optional[MatchingTypeCallback] = None,
    unexpected_pointer_function: Optional[MatchingTypeCallback] = None,
    success_function: Optional[MatchingTypeCallback] = None,
    failure_function: Optional[MatchingTypeCallback] = None,
    permissive: bool = False, pointer_depth: int = 1, **kwargs
) -> None:
  r"""Checks that obj is of a particular kind, for example char. Can optionally handle pointers too.

  Parameters
  ----------
  linter :
    the linter instance
  obj :
    the cursor to check
  expected_type_kinds :
    the base `clang.cindex.TypeKind` that you want `obj` to be, e.g. clx.TypeKind.ENUM for PetscBool
  pointer : optional
    should `obj` be a pointer to your type?
  unexpected_not_pointer_function : optional
    callback for when `pointer` is True, and `obj` matches the base type but IS NOT a pointer
  unexpected_pointer_function : optional
    callback for when `pointer` is False`, the object matches the base type but IS a pointer
  success_function : optional
    callback for when `obj` matches the type and pointer specification
  failure_function : optional
    callback for when `obj` does NOT match the base type
  permissive : optional
    allow type mismatch (e.g. when checking generic functions like PetscAssertPointer() which can
    accept anytype)
  pointer_depth : optional
    how many levels of pointer to remove (-1 for no limit)

  Raises
  ------
  RuntimeError
    if `success_function` returns False, indicating an unhandled error

  Notes
  -----
  The hooks must return True or False to indicate whether they handled the particular situation.
  This can mean either determining that the object was correct all along (and return True), or that a
  more helpful error message was logged and/or that a fix was created.

  Returning False indicates that the hook was not successful and that additional error messages should
  be logged
  """
  def always_false(*args: Any, **kwargs: Any) -> bool:
    return False

  def always_true(*args: Any, **kwargs: Any) -> bool:
    return True

  if unexpected_not_pointer_function is None:
    unexpected_not_pointer_function = always_false

  if unexpected_pointer_function is None:
    unexpected_pointer_function = always_false

  if success_function is None:
    success_function = always_true

  if failure_function is None:
    failure_function = always_false

  if pointer_depth < 0:
    pointer_depth = 100

  diag_name = _CodeDiags.diags.incompatible_type
  obj_type  = obj.canonical.type.get_canonical()
  if pointer:
    if obj_type.kind in expected_type_kinds and not permissive:
      # expecting a pointer to type, but obj is already that type
      handled = unexpected_not_pointer_function(linter, obj, obj_type, **kwargs)
      if not handled:
        mess = f'Object of clang type {obj_type.kind} is not a pointer. Expected pointer of one of the following types: {expected_type_kinds}'
        linter.add_diagnostic_from_cursor(
          obj, Diagnostic(Diagnostic.Kind.ERROR, diag_name, mess, obj.extent.start)
        )
      return

    # get rid of any nested pointer types
    def cycle_type(obj_type: clx.Type, get_obj_type: Callable[[clx.Type], clx.Type]) -> clx.Type:
      for _ in range(pointer_depth):
        tmp_type = get_obj_type(obj_type)
        if tmp_type == obj_type:
          break
        obj_type = tmp_type
      return obj_type

    if obj_type.kind == clx.TypeKind.INCOMPLETEARRAY:
      obj_type = cycle_type(obj_type, lambda otype: otype.element_type)
    elif obj_type.kind == clx.TypeKind.POINTER:
      obj_type = cycle_type(obj_type, lambda otype: otype.get_pointee())
  else:
    if obj_type.kind in clx_pointer_type_kinds:
      handled = unexpected_pointer_function(linter, obj, obj_type, **kwargs)
      if not handled:
        mess = f'Object of clang type {obj_type.kind} is a pointer when it should not be'
        linter.add_diagnostic_from_cursor(
          obj, Diagnostic(Diagnostic.Kind.ERROR, diag_name, mess, obj.extent.start)
        )
      return

  if permissive or obj_type.kind in expected_type_kinds:
    handled = success_function(linter, obj, obj_type, **kwargs)
    if not handled:
      error_message = "{}\nType checker successfully matched object of type {} to (one of) expected types:\n- {}\n\nBut user supplied on-successful-match hook '{}' returned non-truthy value '{}' indicating unhandled error!".format(obj, obj_type.kind, '\n- '.join(map(str, expected_type_kinds)), success_function, handled)
      raise RuntimeError(error_message)
  else:
    handled = failure_function(linter, obj, obj_type, **kwargs)
    if not handled:
      mess = f'Object of clang type {obj_type.kind} is not in expected types: {expected_type_kinds}'
      linter.add_diagnostic_from_cursor(
        obj, Diagnostic(Diagnostic.Kind.ERROR, diag_name, mess, obj.extent.start)
      )
  return
