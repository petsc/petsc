/*$Id: dlregis.c,v 1.19 2001/03/23 23:24:34 balay Exp $*/

#include "petscvec.h"
#include "petscpf.h"

#undef __FUNCT__  
#define __FUNCT__ "VecInitializePackage"
/*@C
  VecInitializePackage - This function initializes everything in the Vec package. It is called
  from PetscDLLibraryRegister() when using dynamic libraries, and on the first call to VecCreate()
  when using static libraries.

  Input Parameter:
  path - The dynamic library path, or PETSC_NULL

  Level: developer

.keywords: Vec, initialize, package
.seealso: PetscInitialize()
@*/
int VecInitializePackage(char *path) {
  static PetscTruth initialized = PETSC_FALSE;
  char              logList[256];
  char             *className;
  PetscTruth        opt;
  int               ierr;

  PetscFunctionBegin;
  if (initialized == PETSC_TRUE) PetscFunctionReturn(0);
  initialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscLogClassRegister(&IS_COOKIE,          "Index Set");                                         CHKERRQ(ierr);
  ierr = PetscLogClassRegister(&MAP_COOKIE,         "Map");                                               CHKERRQ(ierr);
  ierr = PetscLogClassRegister(&VEC_COOKIE,         "Vec");                                               CHKERRQ(ierr);
  ierr = PetscLogClassRegister(&VEC_SCATTER_COOKIE, "Vec Scatter");                                       CHKERRQ(ierr);
  ierr = PetscLogClassRegister(&PF_COOKIE,          "PointFunction");                                     CHKERRQ(ierr);
  /* Register Constructors and Serializers */
  ierr = PetscMapRegisterAll(path);                                                                       CHKERRQ(ierr);
  ierr = VecRegisterAll(path);                                                                            CHKERRQ(ierr);
  ierr = PFRegisterAll(path);                                                                             CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister(&VEC_View,                "VecView",          VEC_COOKIE);                 CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&VEC_Max,                 "VecMax",           VEC_COOKIE);                 CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&VEC_Min,                 "VecMin",           VEC_COOKIE);                 CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&VEC_DotBarrier,          "VecDotBarrier",    VEC_COOKIE);                 CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&VEC_Dot,                 "VecDot",           VEC_COOKIE);                 CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&VEC_MDotBarrier,         "VecMDotBarrier",   VEC_COOKIE);                 CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&VEC_MDot,                "VecMDot",          VEC_COOKIE);                 CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&VEC_TDot,                "VecTDot",          VEC_COOKIE);                 CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&VEC_MTDot,               "VecMTDot",         VEC_COOKIE);                 CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&VEC_NormBarrier,         "VecNormBarrier",   VEC_COOKIE);                 CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&VEC_Norm,                "VecNorm",          VEC_COOKIE);                 CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&VEC_Scale,               "VecScale",         VEC_COOKIE);                 CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&VEC_Copy,                "VecCopy",          VEC_COOKIE);                 CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&VEC_Set,                 "VecSet",           VEC_COOKIE);                 CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&VEC_AXPY,                "VecAXPY",          VEC_COOKIE);                 CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&VEC_AYPX,                "VecAYPX",          VEC_COOKIE);                 CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&VEC_WAXPY,               "VecWAXPY",         VEC_COOKIE);                 CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&VEC_MAXPY,               "VecMAXPY",         VEC_COOKIE);                 CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&VEC_Swap,                "VecSwap",          VEC_COOKIE);                 CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&VEC_AssemblyBegin,       "VecAssemblyBegin", VEC_COOKIE);                 CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&VEC_AssemblyEnd,         "VecAssemblyEnd",   VEC_COOKIE);                 CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&VEC_PointwiseMult,       "VecPointwiseMult", VEC_COOKIE);                 CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&VEC_SetValues,           "VecSetValues",     VEC_COOKIE);                 CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&VEC_Load,                "VecLoad",          VEC_COOKIE);                 CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&VEC_ScatterBarrier,      "VecScatterBarrie", VEC_COOKIE);                 CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&VEC_ScatterBegin,        "VecScatterBegin",  VEC_COOKIE);                 CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&VEC_ScatterEnd,          "VecScatterEnd",    VEC_COOKIE);                 CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&VEC_SetRandom,           "VecSetRandom",     VEC_COOKIE);                 CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&VEC_ReduceArithmetic,    "VecReduceArith",   VEC_COOKIE);                 CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&VEC_ReduceBarrier,       "VecReduceBarrier", VEC_COOKIE);                 CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&VEC_ReduceCommunication, "VecReduceComm",    VEC_COOKIE);                 CHKERRQ(ierr);
  /* Turn off high traffic events by default */
  ierr = PetscLogEventDeactivate(VEC_DotBarrier);                                                         CHKERRQ(ierr);
  ierr = PetscLogEventDeactivate(VEC_MDotBarrier);                                                        CHKERRQ(ierr);
  ierr = PetscLogEventDeactivate(VEC_NormBarrier);                                                        CHKERRQ(ierr);
  ierr = PetscLogEventDeactivate(VEC_SetValues);                                                          CHKERRQ(ierr);
  ierr = PetscLogEventDeactivate(VEC_ScatterBarrier);                                                     CHKERRQ(ierr);
  ierr = PetscLogEventDeactivate(VEC_ReduceBarrier);                                                      CHKERRQ(ierr);
  /* Process info exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-log_info_exclude", logList, 256, &opt);                      CHKERRQ(ierr);
  if (opt == PETSC_TRUE) {
    ierr = PetscStrstr(logList, "is", &className);                                                        CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogInfoDeactivateClass(IS_COOKIE);                                                      CHKERRQ(ierr);
    }
    ierr = PetscStrstr(logList, "map", &className);                                                       CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogInfoDeactivateClass(MAP_COOKIE);                                                     CHKERRQ(ierr);
    }
    ierr = PetscStrstr(logList, "vec", &className);                                                       CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogInfoDeactivateClass(VEC_COOKIE);                                                     CHKERRQ(ierr);
    }
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-log_summary_exclude", logList, 256, &opt);                   CHKERRQ(ierr);
  if (opt == PETSC_TRUE) {
    ierr = PetscStrstr(logList, "is", &className);                                                        CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(IS_COOKIE);                                                     CHKERRQ(ierr);
    }
    ierr = PetscStrstr(logList, "map", &className);                                                       CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(MAP_COOKIE);                                                    CHKERRQ(ierr);
    }
    ierr = PetscStrstr(logList, "vec", &className);                                                       CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(VEC_COOKIE);                                                    CHKERRQ(ierr);
    }
  }
  /* Special processing */
  ierr = PetscOptionsHasName(PETSC_NULL, "-log_sync", &opt);                                              CHKERRQ(ierr);
  if (opt == PETSC_TRUE) {
    ierr = PetscLogEventActivate(VEC_ScatterBarrier);                                                     CHKERRQ(ierr);
    ierr = PetscLogEventActivate(VEC_NormBarrier);                                                        CHKERRQ(ierr);
    ierr = PetscLogEventActivate(VEC_DotBarrier);                                                         CHKERRQ(ierr);
    ierr = PetscLogEventActivate(VEC_MDotBarrier);                                                        CHKERRQ(ierr);
    ierr = PetscLogEventActivate(VEC_ReduceBarrier);                                                      CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#ifdef PETSC_USE_DYNAMIC_LIBRARIES
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PetscDLLibraryRegister"
/*
  PetscDLLibraryRegister - This function is called when the dynamic library it is in is opened.

  This one registers all the methods that are in the basic PETSc Vec library.

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
  ierr = VecInitializePackage(path);                                                                      CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

/* --------------------------------------------------------------------------*/
static char *contents = "PETSc Vector library. \n";
static char *authors  = PETSC_AUTHOR_INFO;

#include "src/sys/src/utils/dlregis.h"

#endif /* PETSC_USE_DYNAMIC_LIBRARIES */
