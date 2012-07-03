
#include <petscvec.h>
#include <petscpf.h>

static PetscBool  ISPackageInitialized = PETSC_FALSE;
#undef __FUNCT__  
#define __FUNCT__ "ISFinalizePackage"
/*@C
  ISFinalizePackage - This function destroys everything in the IS package. It is
  called from PetscFinalize().

  Level: developer

.keywords: Petsc, destroy, package
.seealso: PetscFinalize()
@*/
PetscErrorCode  ISFinalizePackage(void)
{
  PetscFunctionBegin;
  ISPackageInitialized = PETSC_FALSE;
  ISList               = PETSC_NULL;
  ISRegisterAllCalled  = PETSC_FALSE;
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
PetscErrorCode  ISInitializePackage(const char path[]) 
{
  char              logList[256];
  char              *className;
  PetscBool         opt;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (ISPackageInitialized) PetscFunctionReturn(0);
  ISPackageInitialized = PETSC_TRUE;
  /* Register Constructors */
  ierr = ISRegisterAll(path);CHKERRQ(ierr);
  /* Register Classes */
  ierr = PetscClassIdRegister("Index Set",&IS_CLASSID);CHKERRQ(ierr);
  ierr = PetscClassIdRegister("IS L to G Mapping",&IS_LTOGM_CLASSID);CHKERRQ(ierr);

  /* Process info exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-info_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "is", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscInfoDeactivateClass(IS_CLASSID);CHKERRQ(ierr);
      ierr = PetscInfoDeactivateClass(IS_LTOGM_CLASSID);CHKERRQ(ierr);
    }
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-log_summary_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "is", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(IS_CLASSID);CHKERRQ(ierr);
      ierr = PetscLogEventDeactivateClass(IS_LTOGM_CLASSID);CHKERRQ(ierr);
    }
  }
  ierr = PetscRegisterFinalize(ISFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

extern MPI_Op PetscSplitReduction_Op;
extern MPI_Op VecMax_Local_Op;
extern MPI_Op VecMin_Local_Op;

EXTERN_C_BEGIN
extern void  MPIAPI VecMax_Local(void*,void*,PetscMPIInt*,MPI_Datatype*);
extern void  MPIAPI VecMin_Local(void*,void*,PetscMPIInt*,MPI_Datatype*);
extern void  MPIAPI PetscSplitReduction_Local(void*,void*,PetscMPIInt*,MPI_Datatype*);
EXTERN_C_END

const char *NormTypes[] = {"1","2","FROBENIUS","INFINITY","1_AND_2","NormType","NORM_",0};
PetscInt   NormIds[7];  /* map from NormType to IDs used to cache Normvalues */

static PetscBool  VecPackageInitialized = PETSC_FALSE;

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
PetscErrorCode  VecInitializePackage(const char path[]) 
{
  char              logList[256];
  char              *className;
  PetscBool         opt;
  PetscErrorCode    ierr;
  PetscInt          i;

  PetscFunctionBegin;
  if (VecPackageInitialized) PetscFunctionReturn(0);
  VecPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscClassIdRegister("Vector",&VEC_CLASSID);CHKERRQ(ierr);
  ierr = PetscClassIdRegister("Vector Scatter",&VEC_SCATTER_CLASSID);CHKERRQ(ierr);
  /* Register Constructors */
  ierr = VecRegisterAll(path);CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister("VecView",          VEC_CLASSID,&VEC_View);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecMax",           VEC_CLASSID,&VEC_Max);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecMin",           VEC_CLASSID,&VEC_Min);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecDotBarrier",    VEC_CLASSID,&VEC_DotBarrier);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecDot",           VEC_CLASSID,&VEC_Dot);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecDotNormBarr",   VEC_CLASSID,&VEC_DotNormBarrier);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecDotNorm2",      VEC_CLASSID,&VEC_DotNorm);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecMDotBarrier",   VEC_CLASSID,&VEC_MDotBarrier);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecMDot",          VEC_CLASSID,&VEC_MDot);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecTDot",          VEC_CLASSID,&VEC_TDot);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecMTDot",         VEC_CLASSID,&VEC_MTDot);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecNormBarrier",   VEC_CLASSID,&VEC_NormBarrier);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecNorm",          VEC_CLASSID,&VEC_Norm);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecScale",         VEC_CLASSID,&VEC_Scale);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecCopy",          VEC_CLASSID,&VEC_Copy);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecSet",           VEC_CLASSID,&VEC_Set);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecAXPY",          VEC_CLASSID,&VEC_AXPY);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecAYPX",          VEC_CLASSID,&VEC_AYPX);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecAXPBYCZ",       VEC_CLASSID,&VEC_AXPBYPCZ);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecWAXPY",         VEC_CLASSID,&VEC_WAXPY);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecMAXPY",         VEC_CLASSID,&VEC_MAXPY);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecSwap",          VEC_CLASSID,&VEC_Swap);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecOps",           VEC_CLASSID,&VEC_Ops);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecAssemblyBegin", VEC_CLASSID,&VEC_AssemblyBegin);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecAssemblyEnd",   VEC_CLASSID,&VEC_AssemblyEnd);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecPointwiseMult", VEC_CLASSID,&VEC_PointwiseMult);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecSetValues",     VEC_CLASSID,&VEC_SetValues);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecLoad",          VEC_CLASSID,&VEC_Load);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecScatterBarrie", VEC_CLASSID,&VEC_ScatterBarrier);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecScatterBegin",  VEC_CLASSID,&VEC_ScatterBegin);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecScatterEnd",    VEC_CLASSID,&VEC_ScatterEnd);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecSetRandom",     VEC_CLASSID,&VEC_SetRandom);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecReduceArith",   VEC_CLASSID,&VEC_ReduceArithmetic);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecReduceBarrier", VEC_CLASSID,&VEC_ReduceBarrier);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecReduceComm",    VEC_CLASSID,&VEC_ReduceCommunication);CHKERRQ(ierr); /* must follow barrier */
  ierr = PetscLogEventRegister("VecReduceBegin",   VEC_CLASSID,&VEC_ReduceBegin);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecReduceEnd",     VEC_CLASSID,&VEC_ReduceEnd);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecNormalize",     VEC_CLASSID,&VEC_Normalize);CHKERRQ(ierr);
#if defined(PETSC_HAVE_CUSP)
  ierr = PetscLogEventRegister("VecCUSPCopyTo",     VEC_CLASSID,&VEC_CUSPCopyToGPU);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecCUSPCopyFrom",   VEC_CLASSID,&VEC_CUSPCopyFromGPU);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecCopyToSome",     VEC_CLASSID,&VEC_CUSPCopyToGPUSome);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecCopyFromSome",   VEC_CLASSID,&VEC_CUSPCopyFromGPUSome);CHKERRQ(ierr);
#endif
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
      ierr = PetscInfoDeactivateClass(VEC_CLASSID);CHKERRQ(ierr);
    }
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-log_summary_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "vec", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(VEC_CLASSID);CHKERRQ(ierr);
    }
  }
  /* Special processing */
  opt = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PETSC_NULL, "-log_sync", &opt,PETSC_NULL);CHKERRQ(ierr);
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
PetscErrorCode  VecFinalizePackage(void) {
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
PetscErrorCode  PetscDLLibraryRegister_petscvec(const char path[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = ISInitializePackage(path);CHKERRQ(ierr);
  ierr = VecInitializePackage(path);CHKERRQ(ierr);
  ierr = PFInitializePackage(path);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#endif /* PETSC_USE_DYNAMIC_LIBRARIES */
