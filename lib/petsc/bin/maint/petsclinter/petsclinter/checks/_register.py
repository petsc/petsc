#!/usr/bin/env python3
"""
# Created: Mon Jun 20 20:07:30 2022 (-0400)
# @author: Jacob Faibussowitsch
"""
from __future__ import annotations

from .._typing import *

import clang.cindex as clx # type: ignore[import]

_T = TypeVar('_T')
_U = TypeVar('_U')

classid_map: dict[str, str]                     = {}
check_function_map: dict[str, FunctionChecker]  = {}
check_doc_map: dict[clx.CursorKind, DocChecker] = {}

def filter_check_function_map(allowed_symbol_names: Collection[str]) -> None:
  r"""Remove checks from check_function_map

  Parameters
  ----------
  allowed_symbol_names :
    a set of symbol names

  Notes
  -----
  After this routine returns, the check function map will only contain functions whose symbol name
  was in `allowed_symbol_names`
  """
  if allowed_symbol_names:
    global check_function_map

    # note the list, this makes a copy of the keys allowing us to delete entries "in place"
    for key in list(check_function_map.keys()):
      if key not in allowed_symbol_names:
        del check_function_map[key]
  return

def __register_base(key: _T, value: _U, target_map: dict[_T, _U], exist_ok: bool) -> None:
  if key in target_map:
    if not exist_ok:
      raise RuntimeError(f'Key {key} already registered with check map')

  target_map[key] = value
  return

def register_classid(struct_name: str, classid_name: str, exist_ok: bool = False) -> None:
  r"""Register a classid to match struct names

  Parameters
  ----------
  struct_name :
    the fully qualified structure typename to match against, e.g. _p_PetscObject *
  classid_name :
    the string name of the classid variable, e.g. "KSP_CLASSID"
  exist_ok : optional
    is it ok if there already exists a classid for this struct?

  Raises
  ------
  RuntimeError
    if a classid has already been registered for `struct_name` and `exist_ok` is False
  """
  __register_base(struct_name, classid_name, classid_map, exist_ok)
  return

def register_symbol_check(name: str, function: FunctionChecker, exist_ok: bool = False) -> None:
  r"""Register a symbol checking function

  Parameters
  ----------
  name :
    the name of the symbol this checker will trigger on
  function :
    the function to check the symbol
  exist_ok : optional
    is it ok if there already exists a checker for this kind?

  Raises
  ------
  RuntimeError
    if a function has already been registered for `name` and `exist_ok` is False
  """
  __register_base(name, function, check_function_map, exist_ok)
  return

def register_doc_check(cursor_kind: clx.CursorKind, function: DocChecker, exist_ok: bool = False) -> None:
  r"""Register a docs-checking function

  Parameters
  ----------
  cursor_kind :
    the kind of cursor this docs checker will trigger on, (e.g. functions, enums, etc.)
  function :
    the function to check the docstring
  exist_ok : optional
    is it ok if there already exists a checker for this kind?

  Raises
  ------
  RuntimeError
    if a function has already been registered for `cursor_kind` and `exist_ok` is False
  """
  __register_base(cursor_kind, function, check_doc_map, exist_ok)
  return

def __register_all_classids() -> None:
  r"""
  Adding new classes
  ------------------

  You must register new instances of PETSc classes in the classid_map which expects its
  contents to be in the form:

  "CaseSensitiveNameOfPrivateStruct *" : "CaseSensitiveNameOfCorrespondingClassId",

  See below for examples.

  * please add your new class in alphabetical order and preserve the alignment! *

  The automated way to do it (in emacs) is to slap it in the first entry then highlight
  the contents (i.e. excluding "classid_map = {" and the closing "}") and do:

  1. M-x sort-fields RET
  2. M-x align-regexp RET : RET
  """
  default_classid_map = {
    "_p_AO *"                     : "AO_CLASSID",
    "_p_PetscBench *"                : "BM_CLASSID",
    "_p_Characteristic *"         : "CHARACTERISTIC_CLASSID",
    "_p_DM *"                     : "DM_CLASSID",
    "_p_DMAdaptor *"              : "DMADAPTOR_CLASSID",
    "_p_DMField *"                : "DMFIELD_CLASSID",
    "_p_DMKSP *"                  : "DMKSP_CLASSID",
    "_p_DMLabel *"                : "DMLABEL_CLASSID",
    "_p_DMPlexTransform *"        : "DMPLEXTRANSFORM_CLASSID",
    "_p_DMSwarmCellDM *"          : "DMSWARMCELLDM_CLASSID",
    "_p_DMSNES *"                 : "DMSNES_CLASSID",
    "_p_DMTS *"                   : "DMTS_CLASSID",
    "_p_IS *"                     : "IS_CLASSID",
    "_p_ISLocalToGlobalMapping *" : "IS_LTOGM_CLASSID",
    "_p_KSP *"                    : "KSP_CLASSID",
    "_p_KSPGuess *"               : "KSPGUESS_CLASSID",
    "_p_LineSearch *"             : "SNESLINESEARCH_CLASSID",
    "_p_Mat *"                    : "MAT_CLASSID",
    "_p_MatCoarsen *"             : "MAT_COARSEN_CLASSID",
    "_p_MatColoring *"            : "MAT_COLORING_CLASSID",
    "_p_MatFDColoring *"          : "MAT_FDCOLORING_CLASSID",
    "_p_MatMFFD *"                : "MATMFFD_CLASSID",
    "_p_MatNullSpace *"           : "MAT_NULLSPACE_CLASSID",
    "_p_MatPartitioning *"        : "MAT_PARTITIONING_CLASSID",
    "_p_MatTransposeColoring *"   : "MAT_TRANSPOSECOLORING_CLASSID",
    "_p_PC *"                     : "PC_CLASSID",
    "_p_PF *"                     : "PF_CLASSID",
    "_p_PetscContainer *"         : "PETSC_CONTAINER_CLASSID",
    "_p_PetscConvEst *"           : "PETSC_OBJECT_CLASSID",
    "_p_PetscDS *"                : "PETSCDS_CLASSID",
    "_p_PetscDraw *"              : "PETSC_DRAW_CLASSID",
    "_p_PetscDrawAxis *"          : "PETSC_DRAWAXIS_CLASSID",
    "_p_PetscDrawBar *"           : "PETSC_DRAWBAR_CLASSID",
    "_p_PetscDrawHG *"            : "PETSC_DRAWHG_CLASSID",
    "_p_PetscDrawLG *"            : "PETSC_DRAWLG_CLASSID",
    "_p_PetscDrawSP *"            : "PETSC_DRAWSP_CLASSID",
    "_p_PetscDualSpace *"         : "PETSCDUALSPACE_CLASSID",
    "_p_PetscFE *"                : "PETSCFE_CLASSID",
    "_p_PetscFV *"                : "PETSCFV_CLASSID",
    "_p_PetscLimiter *"           : "PETSCLIMITER_CLASSID",
    "_p_PetscLinterDummyObj *"    : "PETSC_LINTER_DUMMY_OBJECT",
    "_p_PetscLogHandler *"        : "PETSCLOGHANDLER_CLASSID",
    "_p_PetscObject *"            : "PETSC_OBJECT_CLASSID",
    "_p_PetscPartitioner *"       : "PETSCPARTITIONER_CLASSID",
    "_p_PetscQuadrature *"        : "PETSCQUADRATURE_CLASSID",
    "_p_PetscRandom *"            : "PETSC_RANDOM_CLASSID",
    "_p_PetscSF *"                : "PETSCSF_CLASSID",
    "_p_PetscSection *"           : "PETSC_SECTION_CLASSID",
    "_p_PetscSectionSym *"        : "PETSC_SECTION_SYM_CLASSID",
    "_p_PetscSpace *"             : "PETSCSPACE_CLASSID",
    "_p_PetscViewer *"            : "PETSC_VIEWER_CLASSID",
    "_p_PetscWeakForm *"          : "PETSCWEAKFORM_CLASSID",
    "_p_SNES *"                   : "SNES_CLASSID",
    "_p_TS *"                     : "TS_CLASSID",
    "_p_TSAdapt *"                : "TSADAPT_CLASSID",
    "_p_TSGLLEAdapt *"            : "TSGLLEADAPT_CLASSID",
    "_p_TSTrajectory *"           : "TSTRAJECTORY_CLASSID",
    "_p_Tao *"                    : "TAO_CLASSID",
    "_p_TaoLineSearch *"          : "TAOLINESEARCH_CLASSID",
    "_p_PetscRegressor *"         : "PETSCREGRESSOR_CLASSID",
    "_p_Vec *"                    : "VEC_CLASSID",
    "_p_VecTagger *"              : "VEC_TAGGER_CLASSID",
  }
  for key, value in default_classid_map.items():
    register_classid(key, value)
  return

def __register_all_symbol_checks() -> None:
  from . import _code

  default_checks = {
    "PetscValidHeaderSpecificType"       : _code.checkPetscValidHeaderSpecificType,
    "PetscValidHeaderSpecific"           : _code.checkPetscValidHeaderSpecific,
    "PetscValidHeader"                   : _code.checkPetscValidHeader,
    "PetscAssertPointer"                 : _code.check_obj_idx_generic,
    "PetscCheckSameType"                 : _code.check_obj_idx_generic,
    "PetscValidType"                     : _code.check_obj_idx_generic,
    "PetscCheckSameComm"                 : _code.check_obj_idx_generic,
    "PetscCheckSameTypeAndComm"          : _code.check_obj_idx_generic,
    "PetscValidLogicalCollectiveScalar"  : _code.checkPetscValidLogicalCollectiveScalar,
    "PetscValidLogicalCollectiveReal"    : _code.checkPetscValidLogicalCollectiveReal,
    "PetscValidLogicalCollectiveInt"     : _code.checkPetscValidLogicalCollectiveInt,
    "PetscValidLogicalCollectiveMPIInt"  : _code.checkPetscValidLogicalCollectiveMPIInt,
    "PetscValidLogicalCollectiveBool"    : _code.checkPetscValidLogicalCollectiveBool,
    "PetscValidLogicalCollectiveEnum"    : _code.checkPetscValidLogicalCollectiveEnum,
    "VecNestCheckCompatible2"            : _code.check_obj_idx_generic,
    "VecNestCheckCompatible3"            : _code.check_obj_idx_generic,
    "MatCheckPreallocated"               : _code.check_obj_idx_generic,
    "MatCheckProduct"                    : _code.check_obj_idx_generic,
    "MatCheckSameLocalSize"              : _code.check_obj_idx_generic,
    "MatCheckSameSize"                   : _code.check_obj_idx_generic,
    "PetscValidDevice"                   : _code.check_obj_idx_generic,
    "PetscCheckCompatibleDevices"        : _code.check_obj_idx_generic,
    "PetscValidDeviceContext"            : _code.check_obj_idx_generic,
    "PetscCheckCompatibleDeviceContexts" : _code.check_obj_idx_generic,
    "PetscSFCheckGraphSet"               : _code.check_obj_idx_generic,
  }
  for key, value in default_checks.items():
    register_symbol_check(key, value)
  return

def __register_all_doc_checks() -> None:
  from . import _docs

  default_checks = {
    clx.CursorKind.FUNCTION_DECL : _docs.check_petsc_function_docstring,
    clx.CursorKind.ENUM_DECL     : _docs.check_petsc_enum_docstring,
  }
  for key, value in default_checks.items():
    register_doc_check(key, value)
  return

def __register_all() -> None:
  __register_all_classids()
  __register_all_symbol_checks()
  __register_all_doc_checks()
  return

__register_all()
