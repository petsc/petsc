#!/usr/bin/env python3
"""
# Created: Mon Jun 20 20:07:30 2022 (-0400)
# @author: Jacob Faibussowitsch
"""
import clang.cindex as clx

classid_map        = {}
check_function_map = {}
check_doc_map      = {}

def filter_check_function_map(filter_checks):
  """
  Remove checks from check_function_map if they are not in filterChecks
  """
  if filter_checks:
    global check_function_map

    # note the list, this makes a copy of the keys allowing us to delete entries "in place"
    for key in list(check_function_map.keys()):
      if key not in filter_checks:
        del check_function_map[key]
  return

def __register_base(key, value, target_map, exist_ok):
  if not exist_ok:
    assert key not in target_map

  target_map[key] = value
  return

def register_classid(struct_name, classid_name, exist_ok=False):
  assert isinstance(struct_name, str)
  assert isinstance(classid_name, str)

  __register_base(struct_name, classid_name, classid_map, exist_ok)
  return

def register_symbol_check(name, function, exist_ok=False):
  assert isinstance(name, str)
  assert callable(function)

  __register_base(name, function, check_function_map, exist_ok)
  return

def register_doc_check(cursor_kind, function, exist_ok=False):
  assert isinstance(cursor_kind, clx.CursorKind)
  assert callable(function)

  __register_base(cursor_kind, function, check_doc_map, exist_ok)
  return

def __register_all_base(input_map, register):
  for key, value in input_map.items():
    register(key, value)
  return

def __register_all_classids():
  """
  Adding new classes
  ------------------

  You must register new instances of PETSc classes in the classid_map which expects its
  contents to be in the form:

  "CaseSensitiveNameOfPrivateStruct *" : "CaseSensitiveNameOfCorrespondingClassId",

  See below for examples.

  * please add your new class in alphabetical order and preserve the alignment! *

  The automated way to do it (in emacs) is to slap it in the first entry then highlight
  the the contents (i.e. excluding "classid_map = {" and the closing "}") and do:

  1. M-x sort-fields RET
  2. M-x align-regexp RET : RET
  """
  default_classid_map = {
    "_p_AO *"                     : "AO_CLASSID",
    "_p_Characteristic *"         : "CHARACTERISTIC_CLASSID",
    "_p_DM *"                     : "DM_CLASSID",
    "_p_DMAdaptor *"              : "DM_CLASSID",
    "_p_DMField *"                : "DMFIELD_CLASSID",
    "_p_DMKSP *"                  : "DMKSP_CLASSID",
    "_p_DMLabel *"                : "DMLABEL_CLASSID",
    "_p_DMPlexTransform *"        : "DMPLEXTRANSFORM_CLASSID",
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
    "_p_Vec *"                    : "VEC_CLASSID",
    "_p_VecTagger *"              : "VEC_TAGGER_CLASSID",
  }
  __register_all_base(default_classid_map, register_classid)
  return

def __register_all_symbol_checks():
  from . import _code

  default_checks = {
    "PetscValidHeaderSpecificType"       : _code.checkPetscValidHeaderSpecificType,
    "PetscValidHeaderSpecific"           : _code.checkPetscValidHeaderSpecific,
    "PetscValidHeader"                   : _code.checkPetscValidHeader,
    "PetscValidPointer"                  : _code.checkPetscValidPointer,
    "PetscValidCharPointer"              : _code.checkPetscValidCharPointer,
    "PetscValidIntPointer"               : _code.checkPetscValidIntPointer,
    "PetscValidBoolPointer"              : _code.checkPetscValidBoolPointer,
    "PetscValidScalarPointer"            : _code.checkPetscValidScalarPointer,
    "PetscValidRealPointer"              : _code.checkPetscValidRealPointer,
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
    "MatCheckProduect"                   : _code.check_obj_idx_generic,
    "MatCheckSameLocalSize"              : _code.check_obj_idx_generic,
    "MatCheckSameSize"                   : _code.check_obj_idx_generic,
    "PetscValidDevice"                   : _code.check_obj_idx_generic,
    "PetscCheckCompatibleDevices"        : _code.check_obj_idx_generic,
    "PetscValidDeviceContext"            : _code.check_obj_idx_generic,
    "PetscCheckCompatibleDeviceContexts" : _code.check_obj_idx_generic,
    "PetscSFCheckGraphSet"               : _code.check_obj_idx_generic,
  }
  __register_all_base(default_checks, register_symbol_check)
  return

def __register_all_doc_checks():
  from . import _docs

  default_checks = {
    clx.CursorKind.FUNCTION_DECL : _docs.check_petsc_function_docstring,
    clx.CursorKind.ENUM_DECL     : _docs.check_petsc_enum_docstring,
  }
  __register_all_base(default_checks, register_doc_check)
  return

def __register_all():
  __register_all_classids()
  __register_all_symbol_checks()
  __register_all_doc_checks()
  return

__register_all()
