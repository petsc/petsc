
#include <petsc/private/vecimpl.h>
#include <petsc/private/isimpl.h>
#include <petscpf.h>
#include <petscsf.h>
#include <petscsection.h>
#include <petscao.h>

static PetscBool         ISPackageInitialized = PETSC_FALSE;
extern PetscFunctionList ISLocalToGlobalMappingList;
const char       *ISInfos[] = {"SORTED", "UNIQUE", "PERMUTATION", "INTERVAL", "IDENTITY", "ISInfo", "IS_",NULL};

/*@C
  ISFinalizePackage - This function destroys everything in the IS package. It is
  called from PetscFinalize().

  Level: developer

.seealso: `PetscFinalize()`
@*/
PetscErrorCode  ISFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListDestroy(&ISList));
  PetscCall(PetscFunctionListDestroy(&ISLocalToGlobalMappingList));
  PetscCall(PetscFunctionListDestroy(&PetscSectionSymList));
  ISPackageInitialized                    = PETSC_FALSE;
  ISRegisterAllCalled                     = PETSC_FALSE;
  ISLocalToGlobalMappingRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
      ISInitializePackage - This function initializes everything in the IS package. It is called
  from PetscDLLibraryRegister_petscvec() when using dynamic libraries, and on the first call to ISCreateXXXX()
  when using shared or static libraries.

  Level: developer

.seealso: `PetscInitialize()`
@*/
PetscErrorCode  ISInitializePackage(void)
{
  char           logList[256];
  PetscBool      opt,pkg;

  PetscFunctionBegin;
  if (ISPackageInitialized) PetscFunctionReturn(0);
  ISPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  PetscCall(PetscClassIdRegister("Index Set",&IS_CLASSID));
  PetscCall(PetscClassIdRegister("IS L to G Mapping",&IS_LTOGM_CLASSID));
  PetscCall(PetscClassIdRegister("Section",&PETSC_SECTION_CLASSID));
  PetscCall(PetscClassIdRegister("Section Symmetry",&PETSC_SECTION_SYM_CLASSID));
  /* Register Constructors */
  PetscCall(ISRegisterAll());
  PetscCall(ISLocalToGlobalMappingRegisterAll());
  /* Register Events */
  PetscCall(PetscLogEventRegister("ISView",IS_CLASSID,&IS_View));
  PetscCall(PetscLogEventRegister("ISLoad",IS_CLASSID,&IS_Load));
  /* Process Info */
  {
    PetscClassId  classids[4];

    classids[0] = IS_CLASSID;
    classids[1] = IS_LTOGM_CLASSID;
    classids[2] = PETSC_SECTION_CLASSID;
    classids[3] = PETSC_SECTION_SYM_CLASSID;
    PetscCall(PetscInfoProcessClass("is", 2, classids));
    PetscCall(PetscInfoProcessClass("section", 2, &classids[2]));
  }
  /* Process summary exclusions */
  PetscCall(PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt));
  if (opt) {
    PetscCall(PetscStrInList("is",logList,',',&pkg));
    if (pkg) PetscCall(PetscLogEventExcludeClass(IS_CLASSID));
    if (pkg) PetscCall(PetscLogEventExcludeClass(IS_LTOGM_CLASSID));
    PetscCall(PetscStrInList("section",logList,',',&pkg));
    if (pkg) PetscCall(PetscLogEventExcludeClass(PETSC_SECTION_CLASSID));
    if (pkg) PetscCall(PetscLogEventExcludeClass(PETSC_SECTION_SYM_CLASSID));
  }
  /* Register package finalizer */
  PetscCall(PetscRegisterFinalize(ISFinalizePackage));
  PetscFunctionReturn(0);
}

extern MPI_Op PetscSplitReduction_Op;

/*
       These two functions are the MPI reduction operation used for max and min with index
   A call to MPI_Op_create() converts the function Vec[Max,Min]_Local() to the MPI operator Vec[Max,Min]_Local_Op.

*/
MPI_Op MPIU_MAXLOC = 0;
MPI_Op MPIU_MINLOC = 0;

static void MPIAPI MPIU_MaxIndex_Local(void *in,void *out,PetscMPIInt *cnt,MPI_Datatype *datatype)
{
  struct PetscRealInt { PetscReal v; PetscInt i; };
  struct PetscRealInt *xin = (struct PetscRealInt*)in;
  struct PetscRealInt *xout = (struct PetscRealInt*)out;
  int                 c;

  PetscFunctionBegin;
  if (*datatype != MPIU_REAL_INT) {
    (*PetscErrorPrintf)("Can only handle MPIU_REAL_INT data types");
    PETSCABORT(MPI_COMM_SELF,PETSC_ERR_ARG_WRONG);
  }
  for (c = 0; c < *cnt; c++) {
    if (xin[c].v > xout[c].v) {
      xout[c].v = xin[c].v;
      xout[c].i = xin[c].i;
    } else if (xin[c].v == xout[c].v) {
      xout[c].i = PetscMin(xin[c].i,xout[c].i);
    }
  }
  PetscFunctionReturnVoid(); /* cannot return a value */
}

static void MPIAPI MPIU_MinIndex_Local(void *in,void *out,PetscMPIInt *cnt,MPI_Datatype *datatype)
{
  struct PetscRealInt { PetscReal v; PetscInt i; };
  struct PetscRealInt *xin = (struct PetscRealInt*)in;
  struct PetscRealInt *xout = (struct PetscRealInt*)out;
  int                 c;

  PetscFunctionBegin;
  if (*datatype != MPIU_REAL_INT) {
    (*PetscErrorPrintf)("Can only handle MPIU_REAL_INT data types");
    PETSCABORT(MPI_COMM_SELF,PETSC_ERR_ARG_WRONG);
  }
  for (c = 0; c < *cnt; c++) {
    if (xin[c].v < xout[c].v) {
      xout[c].v = xin[c].v;
      xout[c].i = xin[c].i;
    } else if (xin[c].v == xout[c].v) {
      xout[c].i = PetscMin(xin[c].i,xout[c].i);
    }
  }
  PetscFunctionReturnVoid(); /* cannot return a value */
}

PETSC_EXTERN void MPIAPI PetscSplitReduction_Local(void*,void*,PetscMPIInt*,MPI_Datatype*);

const char *const NormTypes[] = {"1","2","FROBENIUS","INFINITY","1_AND_2","NormType","NORM_",NULL};
PetscInt          NormIds[7];  /* map from NormType to IDs used to cache Normvalues */

static PetscBool  VecPackageInitialized = PETSC_FALSE;

/*@C
  VecInitializePackage - This function initializes everything in the Vec package. It is called
  from PetscDLLibraryRegister_petscvec() when using dynamic libraries, and on the first call to VecCreate()
  when using shared or static libraries.

  Level: developer

.seealso: `PetscInitialize()`
@*/
PetscErrorCode  VecInitializePackage(void)
{
  char           logList[256];
  PetscBool      opt,pkg;
  PetscInt       i;

  PetscFunctionBegin;
  if (VecPackageInitialized) PetscFunctionReturn(0);
  VecPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  PetscCall(PetscClassIdRegister("Vector",&VEC_CLASSID));
  /* Register Constructors */
  PetscCall(VecRegisterAll());
  /* Register Events */
  PetscCall(PetscLogEventRegister("VecView",          VEC_CLASSID,&VEC_View));
  PetscCall(PetscLogEventRegister("VecMax",           VEC_CLASSID,&VEC_Max));
  PetscCall(PetscLogEventRegister("VecMin",           VEC_CLASSID,&VEC_Min));
  PetscCall(PetscLogEventRegister("VecDot",           VEC_CLASSID,&VEC_Dot));
  PetscCall(PetscLogEventRegister("VecDotNorm2",      VEC_CLASSID,&VEC_DotNorm2));
  PetscCall(PetscLogEventRegister("VecMDot",          VEC_CLASSID,&VEC_MDot));
  PetscCall(PetscLogEventRegister("VecTDot",          VEC_CLASSID,&VEC_TDot));
  PetscCall(PetscLogEventRegister("VecMTDot",         VEC_CLASSID,&VEC_MTDot));
  PetscCall(PetscLogEventRegister("VecNorm",          VEC_CLASSID,&VEC_Norm));
  PetscCall(PetscLogEventRegister("VecScale",         VEC_CLASSID,&VEC_Scale));
  PetscCall(PetscLogEventRegister("VecCopy",          VEC_CLASSID,&VEC_Copy));
  PetscCall(PetscLogEventRegister("VecSet",           VEC_CLASSID,&VEC_Set));
  PetscCall(PetscLogEventRegister("VecAXPY",          VEC_CLASSID,&VEC_AXPY));
  PetscCall(PetscLogEventRegister("VecAYPX",          VEC_CLASSID,&VEC_AYPX));
  PetscCall(PetscLogEventRegister("VecAXPBYCZ",       VEC_CLASSID,&VEC_AXPBYPCZ));
  PetscCall(PetscLogEventRegister("VecWAXPY",         VEC_CLASSID,&VEC_WAXPY));
  PetscCall(PetscLogEventRegister("VecMAXPY",         VEC_CLASSID,&VEC_MAXPY));
  PetscCall(PetscLogEventRegister("VecSwap",          VEC_CLASSID,&VEC_Swap));
  PetscCall(PetscLogEventRegister("VecOps",           VEC_CLASSID,&VEC_Ops));
  PetscCall(PetscLogEventRegister("VecAssemblyBegin", VEC_CLASSID,&VEC_AssemblyBegin));
  PetscCall(PetscLogEventRegister("VecAssemblyEnd",   VEC_CLASSID,&VEC_AssemblyEnd));
  PetscCall(PetscLogEventRegister("VecPointwiseMult", VEC_CLASSID,&VEC_PointwiseMult));
  PetscCall(PetscLogEventRegister("VecSetValues",     VEC_CLASSID,&VEC_SetValues));
  PetscCall(PetscLogEventRegister("VecLoad",          VEC_CLASSID,&VEC_Load));
  PetscCall(PetscLogEventRegister("VecScatterBegin",  VEC_CLASSID,&VEC_ScatterBegin));
  PetscCall(PetscLogEventRegister("VecScatterEnd  ",  VEC_CLASSID,&VEC_ScatterEnd));
  PetscCall(PetscLogEventRegister("VecSetRandom",     VEC_CLASSID,&VEC_SetRandom));
  PetscCall(PetscLogEventRegister("VecReduceArith",   VEC_CLASSID,&VEC_ReduceArithmetic));
  PetscCall(PetscLogEventRegister("VecReduceComm",    VEC_CLASSID,&VEC_ReduceCommunication));
  PetscCall(PetscLogEventRegister("VecReduceBegin",   VEC_CLASSID,&VEC_ReduceBegin));
  PetscCall(PetscLogEventRegister("VecReduceEnd",     VEC_CLASSID,&VEC_ReduceEnd));
  PetscCall(PetscLogEventRegister("VecNormalize",     VEC_CLASSID,&VEC_Normalize));
#if defined(PETSC_HAVE_VIENNACL)
  PetscCall(PetscLogEventRegister("VecVCLCopyTo",     VEC_CLASSID,&VEC_ViennaCLCopyToGPU));
  PetscCall(PetscLogEventRegister("VecVCLCopyFrom",   VEC_CLASSID,&VEC_ViennaCLCopyFromGPU));
#endif
#if defined(PETSC_HAVE_CUDA)
  PetscCall(PetscLogEventRegister("VecCUDACopyTo",    VEC_CLASSID,&VEC_CUDACopyToGPU));
  PetscCall(PetscLogEventRegister("VecCUDACopyFrom",  VEC_CLASSID,&VEC_CUDACopyFromGPU));
  PetscCall(PetscLogEventRegister("VecCopyToSome",    VEC_CLASSID,&VEC_CUDACopyToGPUSome));
  PetscCall(PetscLogEventRegister("VecCopyFromSome",  VEC_CLASSID,&VEC_CUDACopyFromGPUSome));
#endif
#if defined(PETSC_HAVE_HIP)
  PetscCall(PetscLogEventRegister("VecHIPCopyTo",    VEC_CLASSID,&VEC_HIPCopyToGPU));
  PetscCall(PetscLogEventRegister("VecHIPCopyFrom",  VEC_CLASSID,&VEC_HIPCopyFromGPU));
  PetscCall(PetscLogEventRegister("VecCopyToSome",    VEC_CLASSID,&VEC_HIPCopyToGPUSome));
  PetscCall(PetscLogEventRegister("VecCopyFromSome",  VEC_CLASSID,&VEC_HIPCopyFromGPUSome));
#endif

  /* Mark non-collective events */
  PetscCall(PetscLogEventSetCollective(VEC_SetValues,           PETSC_FALSE));
#if defined(PETSC_HAVE_VIENNACL)
  PetscCall(PetscLogEventSetCollective(VEC_ViennaCLCopyToGPU,   PETSC_FALSE));
  PetscCall(PetscLogEventSetCollective(VEC_ViennaCLCopyFromGPU, PETSC_FALSE));
#endif
#if defined(PETSC_HAVE_CUDA)
  PetscCall(PetscLogEventSetCollective(VEC_CUDACopyToGPU,       PETSC_FALSE));
  PetscCall(PetscLogEventSetCollective(VEC_CUDACopyFromGPU,     PETSC_FALSE));
  PetscCall(PetscLogEventSetCollective(VEC_CUDACopyToGPUSome,   PETSC_FALSE));
  PetscCall(PetscLogEventSetCollective(VEC_CUDACopyFromGPUSome, PETSC_FALSE));
#endif
#if defined(PETSC_HAVE_HIP)
  PetscCall(PetscLogEventSetCollective(VEC_HIPCopyToGPU,       PETSC_FALSE));
  PetscCall(PetscLogEventSetCollective(VEC_HIPCopyFromGPU,     PETSC_FALSE));
  PetscCall(PetscLogEventSetCollective(VEC_HIPCopyToGPUSome,   PETSC_FALSE));
  PetscCall(PetscLogEventSetCollective(VEC_HIPCopyFromGPUSome, PETSC_FALSE));
#endif
  /* Turn off high traffic events by default */
  PetscCall(PetscLogEventSetActiveAll(VEC_SetValues, PETSC_FALSE));
  /* Process Info */
  {
    PetscClassId  classids[1];

    classids[0] = VEC_CLASSID;
    PetscCall(PetscInfoProcessClass("vec", 1, classids));
  }
  /* Process summary exclusions */
  PetscCall(PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt));
  if (opt) {
    PetscCall(PetscStrInList("vec",logList,',',&pkg));
    if (pkg) PetscCall(PetscLogEventExcludeClass(VEC_CLASSID));
    if (pkg) PetscCall(PetscLogEventExcludeClass(PETSCSF_CLASSID));
  }

  /*
    Create the special MPI reduction operation that may be used by VecNorm/DotBegin()
  */
  PetscCallMPI(MPI_Op_create(PetscSplitReduction_Local,1,&PetscSplitReduction_Op));
  PetscCallMPI(MPI_Op_create(MPIU_MaxIndex_Local,1,&MPIU_MAXLOC));
  PetscCallMPI(MPI_Op_create(MPIU_MinIndex_Local,1,&MPIU_MINLOC));

  /* Register the different norm types for cached norms */
  for (i=0; i<4; i++) {
    PetscCall(PetscObjectComposedDataRegister(NormIds+i));
  }

  /* Register package finalizer */
  PetscCall(PetscRegisterFinalize(VecFinalizePackage));
  PetscFunctionReturn(0);
}

/*@C
  VecFinalizePackage - This function finalizes everything in the Vec package. It is called
  from PetscFinalize().

  Level: developer

.seealso: `PetscInitialize()`
@*/
PetscErrorCode  VecFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListDestroy(&VecList));
  PetscCallMPI(MPI_Op_free(&PetscSplitReduction_Op));
  PetscCallMPI(MPI_Op_free(&MPIU_MAXLOC));
  PetscCallMPI(MPI_Op_free(&MPIU_MINLOC));
  if (Petsc_Reduction_keyval != MPI_KEYVAL_INVALID) {
    PetscCallMPI(MPI_Comm_free_keyval(&Petsc_Reduction_keyval));
  }
  VecPackageInitialized = PETSC_FALSE;
  VecRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_DYNAMIC_LIBRARIES)
/*
  PetscDLLibraryRegister - This function is called when the dynamic library it is in is opened.

  This one registers all the methods that are in the basic PETSc Vec library.

 */
PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_petscvec(void)
{
  PetscFunctionBegin;
  PetscCall(PetscSFInitializePackage());
  PetscCall(ISInitializePackage());
  PetscCall(AOInitializePackage());
  PetscCall(VecInitializePackage());
  PetscCall(PFInitializePackage());
  PetscFunctionReturn(0);
}

#endif /* PETSC_HAVE_DYNAMIC_LIBRARIES */
