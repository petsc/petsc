#define PETSCVEC_DLL

#include "petscvec.h"
#include "petscpf.h"

static PetscTruth ISPackageInitialized = PETSC_FALSE;
#undef __FUNCT__  
#define __FUNCT__ "ISFinalizePackage"
/*@C
  ISFinalizePackage - This function destroys everything in the IS package. It is
  called from PetscFinalize().

  Level: developer

.keywords: Petsc, destroy, package
.seealso: PetscFinalize()
@*/
PetscErrorCode PETSC_DLLEXPORT ISFinalizePackage(void) 
{
  PetscFunctionBegin;
  ISPackageInitialized = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISInitializePackage"
/*@C
      ISInitializePackage - This function initializes everything in the IS package. It is called
  from PetscDLLibraryRegister() when using dynamic libraries, and on the first call to ISCreateXXXX()
  when using static libraries.

  Input Parameter:
. path - The dynamic library path, or PETSC_NULL

  Level: developer

.keywords: Vec, initialize, package
.seealso: PetscInitialize()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT ISInitializePackage(const char path[]) 
{
  char              logList[256];
  char              *className;
  PetscTruth        opt;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (ISPackageInitialized) PetscFunctionReturn(0);
  ISPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscCookieRegister("Index Set",&IS_COOKIE);CHKERRQ(ierr);
  ierr = PetscCookieRegister("IS L to G Mapping",&IS_LTOGM_COOKIE);CHKERRQ(ierr);

  /* Process info exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-info_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "is", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscInfoDeactivateClass(IS_COOKIE);CHKERRQ(ierr);
      ierr = PetscInfoDeactivateClass(IS_LTOGM_COOKIE);CHKERRQ(ierr);
    }
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-log_summary_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "is", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(IS_COOKIE);CHKERRQ(ierr);
      ierr = PetscLogEventDeactivateClass(IS_LTOGM_COOKIE);CHKERRQ(ierr);
    }
  }
  ierr = PetscRegisterFinalize(ISFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

extern MPI_Op PetscSplitReduction_Op;
extern MPI_Op VecMax_Local_Op;
extern MPI_Op VecMin_Local_Op;

EXTERN_C_BEGIN
extern void PETSCVEC_DLLEXPORT MPIAPI VecMax_Local(void*,void*,PetscMPIInt*,MPI_Datatype*);
extern void PETSCVEC_DLLEXPORT MPIAPI VecMin_Local(void*,void*,PetscMPIInt*,MPI_Datatype*);
extern void PETSCVEC_DLLEXPORT MPIAPI PetscSplitReduction_Local(void*,void*,PetscMPIInt*,MPI_Datatype*);
EXTERN_C_END

const char *NormTypes[] = {"1","2","FROBENIUS","INFINITY","1_AND_2","NormType","NORM_",0};
PetscInt   NormIds[7];  /* map from NormType to IDs used to cache Normvalues */

static PetscTruth VecPackageInitialized = PETSC_FALSE;

#undef __FUNCT__  
#define __FUNCT__ "VecInitializePackage"
/*@C
  VecInitializePackage - This function initializes everything in the Vec package. It is called
  from PetscDLLibraryRegister() when using dynamic libraries, and on the first call to VecCreate()
  when using static libraries.

  Input Parameter:
. path - The dynamic library path, or PETSC_NULL

  Level: developer

.keywords: Vec, initialize, package
.seealso: PetscInitialize()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecInitializePackage(const char path[]) 
{
  char              logList[256];
  char              *className;
  PetscTruth        opt;
  PetscErrorCode    ierr;
  PetscInt          i;

  PetscFunctionBegin;
  if (VecPackageInitialized) PetscFunctionReturn(0);
  VecPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscCookieRegister("Vec",&VEC_COOKIE);CHKERRQ(ierr);
  ierr = PetscCookieRegister("Vec Scatter",&VEC_SCATTER_COOKIE);CHKERRQ(ierr);
  /* Register Constructors */
  ierr = VecRegisterAll(path);CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister("VecView",          VEC_COOKIE,&VEC_View);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecMax",           VEC_COOKIE,&VEC_Max);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecMin",           VEC_COOKIE,&VEC_Min);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecDotBarrier",    VEC_COOKIE,&VEC_DotBarrier);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecDot",           VEC_COOKIE,&VEC_Dot);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecDotNormBarr",   VEC_COOKIE,&VEC_DotNormBarrier);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecDotNorm2",      VEC_COOKIE,&VEC_DotNorm);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecMDotBarrier",   VEC_COOKIE,&VEC_MDotBarrier);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecMDot",          VEC_COOKIE,&VEC_MDot);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecTDot",          VEC_COOKIE,&VEC_TDot);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecMTDot",         VEC_COOKIE,&VEC_MTDot);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecNormBarrier",   VEC_COOKIE,&VEC_NormBarrier);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecNorm",          VEC_COOKIE,&VEC_Norm);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecScale",         VEC_COOKIE,&VEC_Scale);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecCopy",          VEC_COOKIE,&VEC_Copy);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecSet",           VEC_COOKIE,&VEC_Set);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecAXPY",          VEC_COOKIE,&VEC_AXPY);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecAYPX",          VEC_COOKIE,&VEC_AYPX);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecAXPBYCZ",       VEC_COOKIE,&VEC_AXPBYPCZ);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecWAXPY",         VEC_COOKIE,&VEC_WAXPY);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecMAXPY",         VEC_COOKIE,&VEC_MAXPY);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecSwap",          VEC_COOKIE,&VEC_Swap);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecOps",           VEC_COOKIE,&VEC_Ops);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecAssemblyBegin", VEC_COOKIE,&VEC_AssemblyBegin);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecAssemblyEnd",   VEC_COOKIE,&VEC_AssemblyEnd);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecPointwiseMult", VEC_COOKIE,&VEC_PointwiseMult);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecSetValues",     VEC_COOKIE,&VEC_SetValues);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecLoad",          VEC_COOKIE,&VEC_Load);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecScatterBarrie", VEC_COOKIE,&VEC_ScatterBarrier);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecScatterBegin",  VEC_COOKIE,&VEC_ScatterBegin);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecScatterEnd",    VEC_COOKIE,&VEC_ScatterEnd);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecSetRandom",     VEC_COOKIE,&VEC_SetRandom);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecReduceArith",   VEC_COOKIE,&VEC_ReduceArithmetic);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecReduceBarrier", VEC_COOKIE,&VEC_ReduceBarrier);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecReduceComm",    VEC_COOKIE,&VEC_ReduceCommunication);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecNormalize",     VEC_COOKIE,&VEC_Normalize);CHKERRQ(ierr);
  /* Turn off high traffic events by default */
  ierr = PetscLogEventSetActiveAll(VEC_DotBarrier, PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscLogEventSetActiveAll(VEC_DotNormBarrier, PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscLogEventSetActiveAll(VEC_MDotBarrier, PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscLogEventSetActiveAll(VEC_NormBarrier, PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscLogEventSetActiveAll(VEC_SetValues, PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscLogEventSetActiveAll(VEC_ScatterBarrier, PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscLogEventSetActiveAll(VEC_ReduceBarrier, PETSC_FALSE);CHKERRQ(ierr);
  /* Process info exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-info_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "vec", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscInfoDeactivateClass(VEC_COOKIE);CHKERRQ(ierr);
    }
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-log_summary_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "vec", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(VEC_COOKIE);CHKERRQ(ierr);
    }
  }
  /* Special processing */
  opt = PETSC_FALSE;
  ierr = PetscOptionsGetTruth(PETSC_NULL, "-log_sync", &opt,PETSC_NULL);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscLogEventSetActiveAll(VEC_ScatterBarrier, PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscLogEventSetActiveAll(VEC_NormBarrier, PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscLogEventSetActiveAll(VEC_DotBarrier, PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscLogEventSetActiveAll(VEC_DotNormBarrier, PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscLogEventSetActiveAll(VEC_MDotBarrier, PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscLogEventSetActiveAll(VEC_ReduceBarrier, PETSC_TRUE);CHKERRQ(ierr);
  }

  /*
    Create the special MPI reduction operation that may be used by VecNorm/DotBegin()
  */
  ierr = MPI_Op_create(PetscSplitReduction_Local,1,&PetscSplitReduction_Op);CHKERRQ(ierr);
  ierr = MPI_Op_create(VecMax_Local,2,&VecMax_Local_Op);CHKERRQ(ierr);
  ierr = MPI_Op_create(VecMin_Local,2,&VecMin_Local_Op);CHKERRQ(ierr);

  /* Register the different norm types for cached norms */
  for (i=0; i<4; i++) {
    ierr = PetscObjectComposedDataRegister(NormIds+i);CHKERRQ(ierr);
  }
  
  /* Register finalization routine */
  ierr = PetscRegisterFinalize(VecFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecFinalizePackage"
/*@C
  VecFinalizePackage - This function finalizes everything in the Vec package. It is called
  from PetscFinalize().

  Level: developer

.keywords: Vec, initialize, package
.seealso: PetscInitialize()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecFinalizePackage(void) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MPI_Op_free(&PetscSplitReduction_Op);CHKERRQ(ierr);
  ierr = MPI_Op_free(&VecMax_Local_Op);CHKERRQ(ierr);
  ierr = MPI_Op_free(&VecMin_Local_Op);CHKERRQ(ierr);
  VecPackageInitialized = PETSC_FALSE;
  VecList               = PETSC_NULL;
  VecRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#ifdef PETSC_USE_DYNAMIC_LIBRARIES
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PetscDLLibraryRegister_petscvec"
/*
  PetscDLLibraryRegister - This function is called when the dynamic library it is in is opened.

  This one registers all the methods that are in the basic PETSc Vec library.

  Input Parameter:
  path - library path
 */
PetscErrorCode PETSCVEC_DLLEXPORT PetscDLLibraryRegister_petscvec(const char path[])
{
  PetscErrorCode ierr;

  ierr = PetscInitializeNoArguments(); if (ierr) return 1;

  PetscFunctionBegin;
  /*
      If we got here then PETSc was properly loaded
  */
  ierr = ISInitializePackage(path);CHKERRQ(ierr);
  ierr = VecInitializePackage(path);CHKERRQ(ierr);
  ierr = PFInitializePackage(path);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#endif /* PETSC_USE_DYNAMIC_LIBRARIES */
