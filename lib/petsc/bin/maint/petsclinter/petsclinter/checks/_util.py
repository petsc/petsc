#!/usr/bin/env python3
"""
# Created: Mon Jun 20 17:06:22 2022 (-0400)
# @author: Jacob Faibussowitsch
"""
import itertools
import clang.cindex as clx

from ..classes._diag    import DiagnosticManager, Diagnostic
from ..classes._cursor  import Cursor
from ..classes._patch   import Patch
from ..classes._src_pos import SourceRange, SourceLocation

from ..util._clang import *

# utilities for checking functions

@DiagnosticManager.register(
  ('incompatible-function', 'Verify that the correct function was used for a type')
)
def add_function_fix_to_bad_source(linter, obj, func_cursor, valid_func_name):
  """
  Shorthand for extracting a fix from a function cursor
  """
  call = [
    c for c in func_cursor.get_children() if c.type.get_pointee().kind == clx.TypeKind.FUNCTIONPROTO
  ]
  assert len(call) == 1
  mess = f'Incorrect use of {func_cursor.displayname}(), use {valid_func_name}() instead'
  diag = add_function_fix_to_bad_source.diags.incompatible_function
  linter.add_error_from_cursor(
    obj, Diagnostic(diag, mess, obj.extent.start, patch=Patch.from_cursor(call[0], valid_func_name))
  )
  return

def convert_to_correct_PetscValidLogicalCollectiveXXX(linter, obj, obj_type, func_cursor=None, **kwargs):
  """
  Try to glean the correct PetscValidLogicalCollectiveXXX from the type, used as a failure hook in the
  validlogicalcollective checks.
  """
  valid_func_name = None
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
    return True
  return False

def convert_to_correct_PetscValidXXXPointer(linter, obj, obj_type, func_cursor=None, **kwargs):
  """
  Try to glean the correct PetscValidLogicalXXXPointer from the type, used as a failure hook in the
  validpointer checks.
  """
  valid_func_name = None
  obj_type_kind   = obj_type.kind
  if obj_type_kind in {clx.TypeKind.RECORD, clx.TypeKind.VOID, clx.TypeKind.POINTER} | clx_array_type_kinds:
    # pointer to struct or void pointer, use PetscValidPointer() instead
    valid_func_name = 'PetscValidPointer'
  elif obj_type_kind in clx_char_type_kinds:
    valid_func_name = 'PetscValidCharPointer'
  elif obj_type_kind in clx_scalar_type_kinds:
    if 'PetscReal' in obj.derivedtypename:
      valid_func_name = 'PetscValidRealPointer'
    elif 'PetscScalar' in obj.derivedtypename:
      valid_func_name = 'PetscValidScalarPointer'
  elif obj_type_kind in clx_enum_type_kinds:
    if 'PetscBool' in obj.derivedtypename:
      valid_func_name = 'PetscValidBoolPointer'
  elif obj_type_kind in clx_int_type_kinds:
    if ('PetscInt' in obj.derivedtypename) or ('PetscMPIInt' in obj.derivedtypename):
      valid_func_name = 'PetscValidIntPointer'
  if valid_func_name:
    if valid_func_name != 'PetscValidPointer':
      count = 0
      while obj_type.kind == clx.TypeKind.INCOMPLETEARRAY or count > 100:
        count   += 1
        obj_type = obj_type.element_type
      while obj_type.kind == clx.TypeKind.POINTER or count > 100:
        count   += 1
        obj_type = obj_type.get_pointee()
      if count != 0:
        mess = f'\n{Cursor.error_view_from_cursor(obj)}\n\nExpected to select PetscValidPointer() for object of clang type {obj_type.kind} (a pointer of arrity > 1), chose {valid_func_name}() instead'
        raise RuntimeError(mess)
    add_function_fix_to_bad_source(linter, obj, func_cursor, valid_func_name)
    return True
  return False

@DiagnosticManager.register(
  ('incompatible-type', 'Verify that a particular type matches the expected type')
)
def check_is_type_x_and_not_type_y(typeX, typeY, linter, obj, obj_type, func_cursor=None, valid_func=None):
  """
  Check that a cursor is at least some form of derived type X and not some form of type Y
  i.e. for

  myInt **********x;

  you may check that 'x' is some form of 'myInt' instead of say 'PetscBool'
  """
  derived_name = obj.derivedtypename
  if typeX not in derived_name:
    if typeY in derived_name:
      add_function_fix_to_bad_source(linter, obj, func_cursor, valid_func)
    else:
      mess = f'Incorrect use of {funcName}(), {funcName}() should only be used for {typeX}'
      linter.add_error_from_cursor(
        obj, Diagnostic(checkIsTypeXandNotTypeY.diags.incompatible_type, mess, obj.extent.start)
      )
  return True

def check_is_PetscScalar_and_not_PetscReal(*args, **kwargs):
  return check_is_type_x_and_not_type_y('PetscScalar', 'PetscReal', *args, **kwargs)

def check_is_PetscReal_and_not_PetscScalar(*args, **kwargs):
  return check_is_type_x_and_not_type_y('PetscReal', 'PetscScalar', *args, **kwargs)

def check_is_not_type(typename, linter, obj, func_cursor=None, valid_func=None):
  if isinstance(typename, str):
    contains = typename in obj.derivedtypename
  elif isinstance(typename, (tuple, list)):
    contains = any(t in obj.derivedtypename for t in typename)
  else:
    raise ValueError(type(typename))
  if contains:
    add_function_fix_to_bad_source(linter, obj, func_cursor, valid_func)
  return True

def check_int_is_not_PetscBool(linter, obj, *args, **kwargs):
  return check_is_not_type('PetscBool', linter, obj, **kwargs)

def check_MPIInt_is_not_PetscInt(linter, obj, *args, **kwargs):
  return check_is_not_type('PetscInt', linter, obj, **kwargs)

@DiagnosticManager.register(
  ('incompatible-function', 'Verify that the correct function was used for a type')
)
def check_is_PetscBool(linter, obj, *args, func_cursor=None, **kwargs):
  if ('PetscBool' not in obj.derivedtypename) and ('bool' not in obj.typename):
    func_name = func_cursor.displayName
    mess      = f'Incorrect use of {func_name}(), {func_name}() should only be used for PetscBool or bool'
    linter.add_error_from_cursor(
      obj, Diagnostic(check_is_PetscBool.diags.incompatible_function, mess, obj.extent.start)
    )
  return True

@DiagnosticManager.register(('incompatible-type-petscobject', 'Verify that a symbol is a PetscObject'))
def check_is_petsc_object(linter, obj):
  """
  Returns True if obj is a valid PetscObject, otherwise False. Automatically adds the error to the
  linter. Raises RuntimeError if obj is a PetscObject that isn't registered in the classid_map.
  """
  from ._register import classid_map

  if obj.typename not in classid_map:
    # Raise exception here since this isn't a bad source, moreso a failure of this script
    # since it should know about all petsc classes
    err_message = f"{obj}\nUnknown or invalid PETSc class '{obj.derivedtypename}'. If you are introducing a new class, you must register it with this linter! See lib/petsc/bin/maint/petsclinter/petsclinter/checks/_register.py and search for 'Adding new classes' for more information\n"
    raise RuntimeError(err_message)
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
    warning  = False
    diag     = Diagnostic(
      check_is_petsc_object.diags.incompatible_type_petscobject,
      f'Invalid type \'{typename}\', expected a PetscObject (or derived class):\n{obj_decl.get_formatted_location_string()}\n{obj_decl.formatted(nbefore=1, nafter=2)}',
      obj_decl.extent.start
    )
    if typename.startswith('_p_'):
      if len(struct_fields) == 0:
        reason = 'cannot determine fields. Likely the header containing definition of the object is in a nonstandard place'
        # raise a warning instead of an error since this is a failure of the linter
        warning = True
      else:
        reason = 'its definition is missing a PETSCHEADER as the first struct member'
      mess = f'Type \'{typename}\' is prefixed with \'_p_\' to indicate it is a PetscObject but {reason}. Either replace \'_p_\' with \'_n_\' to indicate it is not a PetscObject or add a PETSCHEADER declaration as the first member.'
      diag.add_note(mess).add_note(
        f'It is ambiguous whether \'{obj.name}\' is intended to be a PetscObject.'
      )
    if warning:
      linter.add_warning_from_cursor(obj, diag)
    else:
      linter.add_error_from_cursor(obj, diag)
  return is_valid

@DiagnosticManager.register(
  ('incompatible-classid', 'Verify that the given classid matches the PetscObject type')
)
def check_matching_classid(linter, obj, obj_classid):
  """
  Does the classid match the particular PETSc type
  """
  from ._register import classid_map

  check_is_petsc_object(linter, obj)
  expected = classid_map[obj.typename]
  name     = obj_classid.name
  if name != expected:
    mess     = f"Classid doesn't match. Expected '{expected}' found '{name}'"
    diagname = check_matching_classid.diags.incompatible_classid
    diag     = Diagnostic(
      diagname, mess, obj.extent.start, patch=Patch.from_cursor(obj_classid, expected)
    )
    linter.add_error_from_cursor(obj, diag)
  return

def check_traceable_to_parent_args(obj, parent_arg_names, trace=None):
  """
  Try and see if the cursor can be linked to parent function arguments. If it can be successfully linked return the index of the matched object otherwise raises ParsingError.

  myFunction(barType bar)
  ...
  fooType foo = bar->baz;
  macro(foo, barIdx);
  /* or */
  macro(bar->baz, barIdx);
  /* or */
  initFooFromBar(bar,&foo);
  macro(foo, barIdx);
  """
  if trace is None:
    trace = []
  potential_parents = []
  def_cursor        = obj.get_definition()
  if def_cursor:
    assert def_cursor.location != obj.location, 'Object has definition cursor, yet the cursor did not move. This should be handled!'
    if def_cursor.kind == clx.CursorKind.VAR_DECL:
      # found definition, so were in business
      # Parents here is an odd choice of words since on the very same line I loop
      # over children, but then again clangs AST has an odd semantic for parents/children
      convert_or_dereference_cursors = clx_conversion_cursor_kinds | {clx.CursorKind.UNARY_OPERATOR}
      for def_child in def_cursor.get_children():
        if def_child.kind in convert_or_dereference_cursors:
          potential_parents_temp = [
            child for child in def_child.walk_preorder() if child.kind == clx.CursorKind.DECL_REF_EXPR
          ]
          # Weed out any self-references
          potential_parents_temp = [
            parent for parent in potential_parents_temp if parent.spelling != def_cursor.spelling
          ]
          potential_parents.extend(potential_parents_temp)
    elif def_cursor.kind == clx.CursorKind.FIELD_DECL:
      # we have deduced that the original cursor may refer to a struct member
      # reference, so we go back and see if indeed this is the case
      for member_child in obj.get_children():
        if member_child.kind == clx.CursorKind.MEMBER_REF_EXPR:
          potential_parents_temp = [
            c for c in member_child.walk_preorder() if c.kind == clx.CursorKind.DECL_REF_EXPR
          ]
          potential_parents_temp = [
            parent for parent in potential_parents_temp if parent.spelling != member_child.spelling
          ]
          potential_parents.extend(potential_parents_temp)
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
        raise pl.ParsingError('PETSC_CLANG_STATIC_ANALYZER_IGNORE')

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
      iterator    = map(lambda x: x.cursor, itertools.takewhile(lambda_expr, token_group[idx + 2:]))
    # we now have completely different cursor selected, so we recursively call this
    # function
    else:
      # not a function call, must be an assignment statement, meaning we should now
      # assert that the current obj is being assigned to
      assert Cursor.get_name_from_cursor(token_group[0].cursor) == obj.name
      # find the binary operator, it will contain the most comprehensive AST
      iterator = token_group[[x.spelling for x in token_group].index('=')].cursor.walk_preorder()
      iterator = [c for c in iterator if c.kind == clx.CursorKind.DECL_REF_EXPR]

    alternate_cursor = [c for c in iterator if Cursor.get_name_from_cursor(c) != obj.name]
    potential_parents.extend(alternate_cursor)
  if not potential_parents:
    raise pl.ParsingError
  # arguably at this point anything other than len(potential_parents) should be 1,
  # and anything else can be considered a failure of this routine (therefore a RTE)
  # as it should be able to detect the definition.
  assert len(potential_parents) == 1, 'Cannot determine a unique definition cursor for object'
  # If >1 cursor, probably a bug since we should have weeded something out
  parent = potential_parents[0]
  trace.append(parent)
  if parent.get_definition().kind == clx.CursorKind.PARM_DECL:
    name = Cursor.get_name_from_cursor(parent)
    try:
      loc = parent_arg_names.index(name)
    except ValueError as ve:
      # name isn't in the parent arguments, so we raise parsing error from it
      raise pl.ParsingError from ve
  else:
    parent = Cursor(parent, obj.argidx)
    # deeper into the rabbit hole
    loc, trace = check_traceable_to_parent_args(parent, parent_arg_names, trace=trace)
  return loc, trace

@DiagnosticManager.register(('matching-arg-num', 'Verify that the given argument number matches'))
def check_matching_arg_num(linter, obj, idx, parent_args):
  """
  Is the Arg # correct w.r.t. the function arguments
  """
  diag_name = check_matching_arg_num.diags.matching_arg_num
  if idx.canonical.kind not in clx_math_cursor_kinds:
    # sometimes it is impossible to tell if the index is correct so this is a warning not
    # an error. For example in the case of a loop:
    # for (i = 0; i < n; ++i) PetscValidIntPointer(arr+i, i);
    linter.add_warning_from_cursor(
      idx, Diagnostic(
        diag_name, f"Index value is of unexpected type '{idx.canonical.kind}'", obj.extent.start
      )
    )
    return
  try:
    idx_num = int(idx.name)
  except ValueError:
    linter.add_warning_from_cursor(
      idx, Diagnostic(
        diag_name, 'Potential argument mismatch, could not determine integer value', obj.extent.start
      )
    )
    return
  trace            = []
  parent_arg_names = tuple(s.name for s in parent_args)
  try:
    expected = parent_args[parent_arg_names.index(obj.name)]
  except ValueError:
    try:
      parent_idx, trace = check_traceable_to_parent_args(obj, parent_arg_names)
    except pl.ParsingError as pe:
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
      linter.add_warning_from_cursor(obj, Diagnostic(diag_name, mess, obj.extent.start))
      return
    else:
      expected = parent_args[parent_idx]
  exp_idx = expected.argidx
  if idx_num != exp_idx:
    diag = Diagnostic(
      diag_name,
      f"Argument number doesn't match for '{obj.name}'. Expected '{exp_idx}', found '{idx_num}'",
      idx.extent.start, patch=Patch.from_cursor(idx, exp_idx)
    ).add_note(
      f'\'{obj.name}\' is traceable to argument #{exp_idx} \'{expected.name}\' in enclosing function here:\n{expected.formatted(nboth=2)}',
      location=expected.extent.start
    )
    if trace:
      diag.add_note(f'starting with {obj.get_formatted_blurb().rstrip()}', location=obj.extent.start)
    for cursor in trace:
      diag.add_note(f'via {Cursor.get_formatted_blurb_from_cursor(cursor).rstrip()}', location=cursor.extent.start)
    linter.add_error_from_cursor(idx, diag)
  return

@DiagnosticManager.register(
  ('incompatible-type', 'Verify that a particular type matches the expected type')
)
def check_matching_specific_type(linter, obj, expected_type_kinds, pointer, unexpected_not_pointer_function=None, unexpected_pointer_function=None, success_function=None, failure_function=None, permissive=False, pointer_depth=1, **kwargs):
  """
  Checks that obj is of a particular kind, for example char. Can optionally handle pointers too.

  Nonstandard arguments:

  expected_type_kinds             - the base type that you want obj to be, e.g. clx.TypeKind.ENUM
                                    for PetscBool
  pointer                         - should obj be a pointer to your type?
  unexpected_not_pointer_function - pointer is TRUE, the object matches the base type but IS NOT
                                    a pointer
  unexpected_pointer_function     - pointer is FALSE, the object matches the base type but IS a
                                    pointer
  success_function                - the object matches the type and pointer specification
  failure_function                - the object does NOT match the base type
  permissive                      - allow type mismatch (e.g. when checking generic functions like
                                    PetscValidPointer() which can accept anytype)
  pointer_depth                   - how many levels of pointer to remove (-1 for no limit)

  The hooks must return whether they handled the failure, this can mean either determining
  that the object was correct all along, or that a more helpful error message was logged
  and/or that a fix was created.
  """
  def always_false(*args, **kwargs):
    return False

  def always_true(*args, **kwargs):
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

  diag_name = check_matching_specific_type.diags.incompatible_type
  obj_type  = obj.canonical.type.get_canonical()
  if pointer:
    if obj_type.kind in expected_type_kinds and not permissive:
      # expecting a pointer to type, but obj is already that type
      handled = unexpected_not_pointer_function(linter, obj, obj_type, **kwargs)
      if not handled:
        mess = f'Object of clang type {obj_type.kind} is not a pointer. Expected pointer of one of the following types: {expected_type_kinds}'
        linter.add_error_from_cursor(obj, Diagnostic(diag_name, mess, obj.extent.start))
      return

    # get rid of any nested pointer types
    def cycle_type(obj_type, get_obj_type):
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
    if obj_type.kind in clx_array_type_kinds or obj_type.kind == clx.TypeKind.POINTER:
      handled = unexpected_pointer_function(linter, obj, obj_type, **kwargs)
      if not handled:
        mess = f'Object of clang type {obj_type.kind} is a pointer when it should not be'
        linter.add_error_from_cursor(obj, Diagnostic(diag_name, mess, obj.extent.start))
      return

  if permissive or obj_type.kind in expected_type_kinds:
    handled = success_function(linter, obj, obj_type, **kwargs)
    if not handled:
      error_message = "{}\nType checker successfully matched object of type {} to (one of) expected types:\n- {}\n\nBut user supplied on-successful-match hook '{}' returned non-truthy value '{}' indicating unhandled error!".format(obj, obj_type.kind, '\n- '.join(map(str, expected_type_kinds)), success_function, handled, expected_type_kinds, obj_type.kind)
      raise RuntimeError(error_message)
  else:
    handled = failure_function(linter, obj, obj_type, **kwargs)
    if not handled:
      mess = f'Object of clang type {obj_type.kind} is not in expected types: {expected_type_kinds}'
      linter.add_error_from_cursor(obj, Diagnostic(diag_name, mess, obj.extent.start))
  return
