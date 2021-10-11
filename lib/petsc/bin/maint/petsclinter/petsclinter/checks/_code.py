#!/usr/bin/env python3
"""
# Created: Mon Jun 20 17:00:36 2022 (-0400)
# @author: Jacob Faibussowitsch
"""
import clang.cindex as clx

from ._util import *

"""Specific 'driver' function to test a particular function archetype"""
def check_obj_idx_generic(linter, func, parent):
  """
  For generic checks where the form is func(obj1, idx1,..., objN, idxN)
  """
  func_args   = linter.get_argument_cursors(func)
  parent_args = linter.get_argument_cursors(parent)

  for obj, idx in zip(func_args[::2], func_args[1::2]):
    check_matching_arg_num(linter, obj, idx, parent_args)
  return

def checkPetscValidHeaderSpecificType(linter, func, parent):
  """
  Specific check for PetscValidHeaderSpecificType(obj, classid, idx, type)
  """
  func_args   = linter.get_argument_cursors(func)
  parent_args = linter.get_argument_cursors(parent)

  # Don't need the type
  obj, classid, idx, _ = func_args
  check_matching_classid(linter, obj, classid)
  check_matching_arg_num(linter, obj, idx, parent_args)
  return

def checkPetscValidHeaderSpecific(linter, func, parent):
  """
  Specific check for PetscValidHeaderSpecific(obj, classid, idx)
  """
  func_args   = linter.get_argument_cursors(func)
  parent_args = linter.get_argument_cursors(parent)

  obj, classid, idx = func_args
  check_matching_classid(linter, obj, classid)
  check_matching_arg_num(linter, obj, idx, parent_args)
  return

def checkPetscValidHeader(linter, func, parent):
  """
  Specific check for PetscValidHeader(obj, idx)
  """
  func_args   = linter.get_argument_cursors(func)
  parent_args = linter.get_argument_cursors(parent)

  obj, idx = func_args
  check_is_petsc_object(linter, obj)
  check_matching_arg_num(linter, obj, idx, parent_args)
  return

def checkPetscValidPointerAndType(linter, func, parent, expected_types, **kwargs):
  """
  Generic check for PetscValidXXXPointer(obj, idx)
  """
  func_args   = linter.get_argument_cursors(func)
  parent_args = linter.get_argument_cursors(parent)

  obj, idx = func_args
  kwargs.setdefault('func_cursor', func)
  kwargs.setdefault('failure_function', convert_to_correct_PetscValidXXXPointer)
  check_matching_specific_type(linter, obj, expected_types, True, **kwargs)
  check_matching_arg_num(linter, obj, idx, parent_args)
  return

def checkPetscValidPointer(linter, func, parent):
  """
  Specific check for PetscValidPointer(obj, idx)
  """
  def try_to_convert_to_specific_PetscValidXXXPointer(linter, obj, obj_type, **kwargs):
    pointer_type_kinds = clx_array_type_kinds | {
      clx.TypeKind.RECORD, clx.TypeKind.VOID, clx.TypeKind.POINTER
    }
    if obj_type.kind not in pointer_type_kinds:
      convert_to_correct_PetscValidXXXPointer(linter, obj, obj_type, **kwargs)
    return True # PetscValidPointer is always good

  checkPetscValidPointerAndType(
    linter, func, parent, clx_array_type_kinds | {clx.TypeKind.POINTER},
    success_function=try_to_convert_to_specific_PetscValidXXXPointer, permissive=True
  )
  return

def checkPetscValidCharPointer(linter, func, parent):
  """
  Specific check for PetscValidCharPointer(obj, idx)
  """
  checkPetscValidPointerAndType(linter, func, parent, clx_char_type_kinds)
  return

def checkPetscValidIntPointer(linter, func, parent):
  """
  Specific check for PetscValidIntPointer(obj, idx)
  """
  checkPetscValidPointerAndType(
    linter, func, parent, clx_int_type_kinds,
    success_function=check_int_is_not_PetscBool, valid_func='PetscValidBoolPointer'
  )
  return

def checkPetscValidBoolPointer(linter, func, parent):
  """
  Specific check for PetscValidBoolPointer(obj, idx)
  """
  checkPetscValidPointerAndType(
    linter, func, parent, clx_bool_type_kinds, success_function=check_is_PetscBool
  )
  return

def checkPetscValidScalarPointer(linter, func, parent):
  """
  Specific check for PetscValidScalarPointer(obj, idx)
  """
  checkPetscValidPointerAndType(
    linter, func, parent, clx_scalar_type_kinds,
    success_function=check_is_PetscScalar_and_not_PetscReal, valid_func='PetscValidRealPointer'
  )
  return

def checkPetscValidRealPointer(linter, func, parent):
  """
  Specific check for PetscValidRealPointer(obj, idx)
  """
  checkPetscValidPointerAndType(
    linter, func, parent, clx_real_type_kinds,
    success_function=check_is_PetscReal_and_not_PetscScalar, valid_func='PetscValidScalarPointer'
  )
  return

def checkPetscValidLogicalCollective(linter, func, parent, expected_types, **kwargs):
  """
  Generic check for PetscValidLogicalCollectiveXXX(pobj, obj, idx)
  """
  func_args   = linter.get_argument_cursors(func)
  parent_args = linter.get_argument_cursors(parent)

  # dont need the petsc object, nothing to check there
  _, obj, idx = func_args
  kwargs.setdefault('func_cursor', func)
  kwargs.setdefault('failure_function', convert_to_correct_PetscValidLogicalCollectiveXXX)
  check_matching_specific_type(linter, obj, expected_types, False, **kwargs)
  check_matching_arg_num(linter, obj, idx, parent_args)
  return

def checkPetscValidLogicalCollectiveScalar(linter, func, parent):
  """
  Specific check for PetscValidLogicalCollectiveScalar(pobj, obj, idx)
  """
  checkPetscValidLogicalCollective(
    linter, func, parent, clx_scalar_type_kinds,
    success_function=check_is_PetscScalar_and_not_PetscReal,
    valid_func='PetscValidLogicalCollectiveReal'
  )
  return

def checkPetscValidLogicalCollectiveReal(linter, func, parent):
  """
  Specific check for PetscValidLogicalCollectiveReal(pobj, obj, idx)
  """
  checkPetscValidLogicalCollective(
    linter, func, parent, clx_real_type_kinds,
    success_function=check_is_PetscReal_and_not_PetscScalar,
    valid_func='PetscValidLogicalCollectiveScalar'
  )
  return

def checkPetscValidLogicalCollectiveInt(linter, func, parent):
  """
  Specific check for PetscValidLogicalCollectiveInt(pobj, obj, idx)
  """
  checkPetscValidLogicalCollective(
    linter, func, parent, clx_int_type_kinds,
    success_function=check_int_is_not_PetscBool, valid_func='PetscValidLogicalCollectiveBool'
  )
  return

def checkPetscValidLogicalCollectiveMPIInt(linter, func, parent):
  """
  Specific check for PetscValidLogicalCollectiveMPIInt(pobj, obj, idx)
  """
  checkPetscValidLogicalCollective(
    linter, func, parent, clx_mpiint_type_kinds,
    success_function=check_MPIInt_is_not_PetscInt, valid_func='PetscValidLogicalCollectiveInt'
  )
  return

def checkPetscValidLogicalCollectiveBool(linter, func, parent):
  """
  Specific check for PetscValidLogicalCollectiveBool(pobj, obj, idx)
  """
  checkPetscValidLogicalCollective(
    linter, func, parent, clx_bool_type_kinds, success_function=check_is_PetscBool
  )
  return

def checkPetscValidLogicalCollectiveEnum(linter, func, parent):
  """
  Specific check for PetscValidLogicalCollectiveEnum(pobj, obj, idx)
  """
  checkPetscValidLogicalCollective(linter, func, parent, clx_enum_type_kinds)
  return
