#!/usr/bin/env python3
"""
# Created: Mon Jun 20 17:00:36 2022 (-0400)
# @author: Jacob Faibussowitsch
"""
from ..util._clang import (
  clx_scalar_type_kinds, clx_real_type_kinds, clx_int_type_kinds, clx_mpiint_type_kinds,
  clx_bool_type_kinds, clx_enum_type_kinds
)

from ._util import (
  check_matching_arg_num, check_matching_classid, check_is_petsc_object, check_matching_specific_type,
  convert_to_correct_PetscValidLogicalCollectiveXXX,
  check_is_PetscScalar_and_not_PetscReal,
  check_is_PetscReal_and_not_PetscScalar,
  check_int_is_not_PetscBool,
  check_MPIInt_is_not_PetscInt,
  check_is_PetscBool
)

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
