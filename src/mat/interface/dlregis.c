/*$Id: dlregis.c,v 1.19 2001/03/23 23:24:34 balay Exp $*/

#include "petscconfig.h"
#include "petscmat.h"

#undef __FUNCT__  
#define __FUNCT__ "MatInitializePackage"
/*@C
  MatInitializePackage - This function initializes everything in the Mat package. It is called
  from PetscDLLibraryRegister() when using dynamic libraries, and on the first call to MatCreate()
  when using static libraries.

  Input Parameter:
  path - The dynamic library path, or PETSC_NULL

  Level: developer

.keywords: Mat, initialize, package
.seealso: PetscInitialize()
@*/
int MatInitializePackage(char *path) {
  static PetscTruth initialized = PETSC_FALSE;
  char              logList[256];
  char             *className;
  PetscTruth        opt;
  int               ierr;

  PetscFunctionBegin;
  if (initialized == PETSC_TRUE) PetscFunctionReturn(0);
  initialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscLogClassRegister(&MAT_COOKIE,              "Matrix");                                       CHKERRQ(ierr);
  ierr = PetscLogClassRegister(&MAT_FDCOLORING_COOKIE,   "Matrix FD Coloring");                           CHKERRQ(ierr);
  ierr = PetscLogClassRegister(&MAT_PARTITIONING_COOKIE, "Matrix Partitioning");                          CHKERRQ(ierr);
  ierr = PetscLogClassRegister(&MAT_NULLSPACE_COOKIE,    "Matrix Null Space");                            CHKERRQ(ierr);
  /* Register Constructors and Serializers */
  ierr = MatRegisterAll(path);                                                                            CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister(&MAT_Mult,                     "MatMult",          PETSC_NULL, MAT_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&MAT_MultMatrixFree,           "MatMultMatrixFre", PETSC_NULL, MAT_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&MAT_MultMultiple,             "MatMultMultiple",  PETSC_NULL, MAT_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&MAT_MultConstrained,          "MatMultConstr",    PETSC_NULL, MAT_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&MAT_MultAdd,                  "MatMultAdd",       PETSC_NULL, MAT_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&MAT_MultTranspose,            "MatMultTranspose", PETSC_NULL, MAT_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&MAT_MultTransposeConstrained, "MatMultTrConstr",  PETSC_NULL, MAT_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&MAT_MultTransposeAdd,         "MatMultTrAdd",     PETSC_NULL, MAT_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&MAT_Solve,                    "MatSolve",         PETSC_NULL, MAT_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&MAT_SolveMultiple,            "MatSolveMultiple", PETSC_NULL, MAT_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&MAT_SolveAdd,                 "MatSolveAdd",      PETSC_NULL, MAT_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&MAT_SolveTranspose,           "MatSolveTranspos", PETSC_NULL, MAT_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&MAT_SolveTransposeAdd,        "MatSolveTrAdd",    PETSC_NULL, MAT_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&MAT_Relax,                    "MatRelax",         PETSC_NULL, MAT_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&MAT_ForwardSolve,             "MatForwardSolve",  PETSC_NULL, MAT_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&MAT_BackwardSolve,            "MatBackwardSolve", PETSC_NULL, MAT_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&MAT_LUFactor,                 "MatLUFactor",      PETSC_NULL, MAT_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&MAT_LUFactorSymbolic,         "MatLUFactorSym",   PETSC_NULL, MAT_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&MAT_LUFactorNumeric,          "MatLUFactorNum",   PETSC_NULL, MAT_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&MAT_CholeskyFactor,           "MatCholeskyFctr",  PETSC_NULL, MAT_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&MAT_CholeskyFactorSymbolic,   "MatCholFctrSym",   PETSC_NULL, MAT_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&MAT_CholeskyFactorNumeric,    "MatCholFctrNum",   PETSC_NULL, MAT_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&MAT_ILUFactor,                "MatILUFactor",     PETSC_NULL, MAT_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&MAT_ILUFactorSymbolic,        "MatILUFactorSym",  PETSC_NULL, MAT_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&MAT_ICCFactorSymbolic,        "MatICCFactorSym",  PETSC_NULL, MAT_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&MAT_Copy,                     "MatCopy",          PETSC_NULL, MAT_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&MAT_Convert,                  "MatConvert",       PETSC_NULL, MAT_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&MAT_Scale,                    "MatScale",         PETSC_NULL, MAT_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&MAT_AssemblyBegin,            "MatAssemblyBegin", PETSC_NULL, MAT_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&MAT_AssemblyEnd,              "MatAssemblyEnd",   PETSC_NULL, MAT_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&MAT_SetValues,                "MatSetValues",     PETSC_NULL, MAT_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&MAT_GetValues,                "MatGetValues",     PETSC_NULL, MAT_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&MAT_GetRow,                   "MatGetRow",        PETSC_NULL, MAT_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&MAT_GetSubMatrices,           "MatGetSubMatrice", PETSC_NULL, MAT_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&MAT_GetColoring,              "MatGetColoring",   PETSC_NULL, MAT_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&MAT_GetOrdering,              "MatGetOrdering",   PETSC_NULL, MAT_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&MAT_IncreaseOverlap,          "MatIncreaseOvrlp", PETSC_NULL, MAT_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&MAT_Partitioning,             "MatPartitioning",  PETSC_NULL, MAT_PARTITIONING_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&MAT_ZeroEntries,              "MatZeroEntries",   PETSC_NULL, MAT_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&MAT_Load,                     "MatLoad",          PETSC_NULL, MAT_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&MAT_View,                     "MatView",          PETSC_NULL, MAT_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&MAT_AXPY,                     "MatAXPY",          PETSC_NULL, MAT_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&MAT_FDColoringCreate,         "MatFDColorCreate", PETSC_NULL, MAT_FDCOLORING_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&MAT_FDColoringApply,          "MatFDColorApply",  PETSC_NULL, MAT_FDCOLORING_COOKIE);CHKERRQ(ierr);
  /* Process info exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-log_info_exclude", logList, 256, &opt);                      CHKERRQ(ierr);
  if (opt == PETSC_TRUE) {
    ierr = PetscStrstr(logList, "mat", &className);                                                       CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogInfoDeactivateClass(MAT_COOKIE);                                                     CHKERRQ(ierr);
    }
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-log_summary_exclude", logList, 256, &opt);                   CHKERRQ(ierr);
  if (opt == PETSC_TRUE) {
    ierr = PetscStrstr(logList, "mat", &className);                                                       CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(MAT_COOKIE);                                                    CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#ifdef PETSC_USE_DYNAMIC_LIBRARIES
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PetscDLLibraryRegister"
/*
  PetscDLLibraryRegister - This function is called when the dynamic library it is in is opened.

  This one registers all the TS methods that are in the basic PETSc Matrix library.

  Input Parameter:
  path - library path
 */
int PetscDLLibraryRegister(char *path)
{
  int ierr;

  ierr = PetscInitializeNoArguments(); if (ierr) return 1;

  PetscFunctionBegin;
  /*
      If we got here then PETSc was properly loaded
  */
  ierr = MatInitializePackage(path);                                                                      CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

/* --------------------------------------------------------------------------*/
static char *contents = "PETSc Matrix library. \n";

#include "src/sys/src/utils/dlregis.h"

#endif /* PETSC_USE_DYNAMIC_LIBRARIES */
