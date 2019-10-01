
#include <petsc/private/vecimpl.h>
#include <petsc/private/isimpl.h>
#include <petscpf.h>
#include <petscsf.h>
#include <petscsection.h>
#include <petscao.h>

static PetscBool         ISPackageInitialized = PETSC_FALSE;
extern PetscFunctionList ISLocalToGlobalMappingList;

/*@C
  ISFinalizePackage - This function destroys everything in the IS package. It is
  called from PetscFinalize().

  Level: developer

.seealso: PetscFinalize()
@*/
PetscErrorCode  ISFinalizePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListDestroy(&ISList);CHKERRQ(ierr);
  ierr = PetscFunctionListDestroy(&ISLocalToGlobalMappingList);CHKERRQ(ierr);
  ierr = PetscFunctionListDestroy(&PetscSectionSymList);CHKERRQ(ierr);
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

.seealso: PetscInitialize()
@*/
PetscErrorCode  ISInitializePackage(void)
{
  char           logList[256];
  PetscBool      opt,pkg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ISPackageInitialized) PetscFunctionReturn(0);
  ISPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscClassIdRegister("Index Set",&IS_CLASSID);CHKERRQ(ierr);
  ierr = PetscClassIdRegister("IS L to G Mapping",&IS_LTOGM_CLASSID);CHKERRQ(ierr);
  ierr = PetscClassIdRegister("Section",&PETSC_SECTION_CLASSID);CHKERRQ(ierr);
  ierr = PetscClassIdRegister("Section Symmetry",&PETSC_SECTION_SYM_CLASSID);CHKERRQ(ierr);
  /* Register Constructors */
  ierr = ISRegisterAll();CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingRegisterAll();CHKERRQ(ierr);
  /* Process info exclusions */
  ierr = PetscOptionsGetString(NULL,NULL,"-info_exclude",logList,sizeof(logList),&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrInList("is",logList,',',&pkg);CHKERRQ(ierr);
    if (pkg) {ierr = PetscInfoDeactivateClass(IS_CLASSID);CHKERRQ(ierr);}
    if (pkg) {ierr = PetscInfoDeactivateClass(IS_LTOGM_CLASSID);CHKERRQ(ierr);}
    ierr = PetscStrInList("section",logList,',',&pkg);CHKERRQ(ierr);
    if (pkg) {ierr = PetscInfoDeactivateClass(PETSC_SECTION_CLASSID);CHKERRQ(ierr);}
    if (pkg) {ierr = PetscInfoDeactivateClass(PETSC_SECTION_SYM_CLASSID);CHKERRQ(ierr);}
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrInList("is",logList,',',&pkg);CHKERRQ(ierr);
    if (pkg) {ierr = PetscLogEventExcludeClass(IS_CLASSID);CHKERRQ(ierr);}
    if (pkg) {ierr = PetscLogEventExcludeClass(IS_LTOGM_CLASSID);CHKERRQ(ierr);}
    ierr = PetscStrInList("section",logList,',',&pkg);CHKERRQ(ierr);
    if (pkg) {ierr = PetscLogEventExcludeClass(PETSC_SECTION_CLASSID);CHKERRQ(ierr);}
    if (pkg) {ierr = PetscLogEventExcludeClass(PETSC_SECTION_SYM_CLASSID);CHKERRQ(ierr);}
  }
  /* Register package finalizer */
  ierr = PetscRegisterFinalize(ISFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

extern MPI_Op PetscSplitReduction_Op;

/*
       These two functions are the MPI reduction operation used for max and min with index
   A call to MPI_Op_create() converts the function Vec[Max,Min]_Local() to the MPI operator Vec[Max,Min]_Local_Op.

*/
MPI_Op MPIU_MAXINDEX_OP = 0;
MPI_Op MPIU_MININDEX_OP = 0;

static void MPIAPI MPIU_MaxIndex_Local(void *in,void *out,PetscMPIInt *cnt,MPI_Datatype *datatype)
{
  PetscReal *xin = (PetscReal*)in,*xout = (PetscReal*)out;

  PetscFunctionBegin;
  if (*datatype != MPIU_REAL) {
    (*PetscErrorPrintf)("Can only handle MPIU_REAL data types");
    PETSCABORT(MPI_COMM_SELF,PETSC_ERR_ARG_WRONG);
  }
  if (xin[0] > xout[0]) {
    xout[0] = xin[0];
    xout[1] = xin[1];
  } else if (xin[0] == xout[0]) {
    xout[1] = PetscMin(xin[1],xout[1]);
  }
  PetscFunctionReturnVoid(); /* cannot return a value */
}

static void MPIAPI MPIU_MinIndex_Local(void *in,void *out,PetscMPIInt *cnt,MPI_Datatype *datatype)
{
  PetscReal *xin = (PetscReal*)in,*xout = (PetscReal*)out;

  PetscFunctionBegin;
  if (*datatype != MPIU_REAL) {
    (*PetscErrorPrintf)("Can only handle MPIU_REAL data types");
    PETSCABORT(MPI_COMM_SELF,PETSC_ERR_ARG_WRONG);
  }
  if (xin[0] < xout[0]) {
    xout[0] = xin[0];
    xout[1] = xin[1];
  } else if (xin[0] == xout[0]) {
    xout[1] = PetscMin(xin[1],xout[1]);
  }
  PetscFunctionReturnVoid();
}

PETSC_EXTERN void MPIAPI PetscSplitReduction_Local(void*,void*,PetscMPIInt*,MPI_Datatype*);

const char *const NormTypes[] = {"1","2","FROBENIUS","INFINITY","1_AND_2","NormType","NORM_",0};
PetscInt          NormIds[7];  /* map from NormType to IDs used to cache Normvalues */

static PetscBool  VecPackageInitialized = PETSC_FALSE;

/*@C
  VecInitializePackage - This function initializes everything in the Vec package. It is called
  from PetscDLLibraryRegister_petscvec() when using dynamic libraries, and on the first call to VecCreate()
  when using shared or static libraries.

  Level: developer

.seealso: PetscInitialize()
@*/
PetscErrorCode  VecInitializePackage(void)
{
  char           logList[256];
  PetscBool      opt,pkg;
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  if (VecPackageInitialized) PetscFunctionReturn(0);
  VecPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscClassIdRegister("Vector",&VEC_CLASSID);CHKERRQ(ierr);
  /* Register Constructors */
  ierr = VecRegisterAll();CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister("VecView",          VEC_CLASSID,&VEC_View);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecMax",           VEC_CLASSID,&VEC_Max);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecMin",           VEC_CLASSID,&VEC_Min);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecDot",           VEC_CLASSID,&VEC_Dot);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecDotNorm2",      VEC_CLASSID,&VEC_DotNorm2);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecMDot",          VEC_CLASSID,&VEC_MDot);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecTDot",          VEC_CLASSID,&VEC_TDot);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecMTDot",         VEC_CLASSID,&VEC_MTDot);CHKERRQ(ierr);
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
  ierr = PetscLogEventRegister("VecScatterBegin",  VEC_CLASSID,&VEC_ScatterBegin);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecScatterEnd  ",  VEC_CLASSID,&VEC_ScatterEnd);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecSetRandom",     VEC_CLASSID,&VEC_SetRandom);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecReduceArith",   VEC_CLASSID,&VEC_ReduceArithmetic);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecReduceComm",    VEC_CLASSID,&VEC_ReduceCommunication);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecReduceBegin",   VEC_CLASSID,&VEC_ReduceBegin);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecReduceEnd",     VEC_CLASSID,&VEC_ReduceEnd);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecNormalize",     VEC_CLASSID,&VEC_Normalize);CHKERRQ(ierr);
#if defined(PETSC_HAVE_VIENNACL)
  ierr = PetscLogEventRegister("VecVCLCopyTo",     VEC_CLASSID,&VEC_ViennaCLCopyToGPU);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecVCLCopyFrom",   VEC_CLASSID,&VEC_ViennaCLCopyFromGPU);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_CUDA)
  ierr = PetscLogEventRegister("VecCUDACopyTo",    VEC_CLASSID,&VEC_CUDACopyToGPU);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecCUDACopyFrom",  VEC_CLASSID,&VEC_CUDACopyFromGPU);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecCopyToSome",    VEC_CLASSID,&VEC_CUDACopyToGPUSome);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("VecCopyFromSome",  VEC_CLASSID,&VEC_CUDACopyFromGPUSome);CHKERRQ(ierr);
#endif

  /* Mark non-collective events */
  ierr = PetscLogEventSetCollective(VEC_SetValues,           PETSC_FALSE);CHKERRQ(ierr);
#if defined(PETSC_HAVE_VIENNACL)
  ierr = PetscLogEventSetCollective(VEC_ViennaCLCopyToGPU,   PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscLogEventSetCollective(VEC_ViennaCLCopyFromGPU, PETSC_FALSE);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_CUDA)
  ierr = PetscLogEventSetCollective(VEC_CUDACopyToGPU,       PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscLogEventSetCollective(VEC_CUDACopyFromGPU,     PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscLogEventSetCollective(VEC_CUDACopyToGPUSome,   PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscLogEventSetCollective(VEC_CUDACopyFromGPUSome, PETSC_FALSE);CHKERRQ(ierr);
#endif
  /* Turn off high traffic events by default */
  ierr = PetscLogEventSetActiveAll(VEC_SetValues, PETSC_FALSE);CHKERRQ(ierr);

  /* Process info exclusions */
  ierr = PetscOptionsGetString(NULL,NULL,"-info_exclude",logList,sizeof(logList),&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrInList("vec",logList,',',&pkg);CHKERRQ(ierr);
    if (pkg) {ierr = PetscInfoDeactivateClass(VEC_CLASSID);CHKERRQ(ierr);}
    if (pkg) {ierr = PetscInfoDeactivateClass(VEC_SCATTER_CLASSID);CHKERRQ(ierr);}
  }

  /* Process summary exclusions */
  ierr = PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrInList("vec",logList,',',&pkg);CHKERRQ(ierr);
    if (pkg) {ierr = PetscLogEventExcludeClass(VEC_CLASSID);CHKERRQ(ierr);}
    if (pkg) {ierr = PetscLogEventExcludeClass(VEC_SCATTER_CLASSID);CHKERRQ(ierr);}
  }

  /*
    Create the special MPI reduction operation that may be used by VecNorm/DotBegin()
  */
  ierr = MPI_Op_create(PetscSplitReduction_Local,1,&PetscSplitReduction_Op);CHKERRQ(ierr);
  ierr = MPI_Op_create(MPIU_MaxIndex_Local,2,&MPIU_MAXINDEX_OP);CHKERRQ(ierr);
  ierr = MPI_Op_create(MPIU_MinIndex_Local,2,&MPIU_MININDEX_OP);CHKERRQ(ierr);

  /* Register the different norm types for cached norms */
  for (i=0; i<4; i++) {
    ierr = PetscObjectComposedDataRegister(NormIds+i);CHKERRQ(ierr);
  }

  /* Register package finalizer */
  ierr = PetscRegisterFinalize(VecFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  VecFinalizePackage - This function finalizes everything in the Vec package. It is called
  from PetscFinalize().

  Level: developer

.seealso: PetscInitialize()
@*/
PetscErrorCode  VecFinalizePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListDestroy(&VecList);CHKERRQ(ierr);
  ierr = PetscFunctionListDestroy(&VecScatterList);CHKERRQ(ierr);
  ierr = MPI_Op_free(&PetscSplitReduction_Op);CHKERRQ(ierr);
  ierr = MPI_Op_free(&MPIU_MAXINDEX_OP);CHKERRQ(ierr);
  ierr = MPI_Op_free(&MPIU_MININDEX_OP);CHKERRQ(ierr);
  if (Petsc_Reduction_keyval != MPI_KEYVAL_INVALID) {
    ierr = MPI_Comm_free_keyval(&Petsc_Reduction_keyval);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSFInitializePackage();CHKERRQ(ierr);
  ierr = ISInitializePackage();CHKERRQ(ierr);
  ierr = AOInitializePackage();CHKERRQ(ierr);
  ierr = VecInitializePackage();CHKERRQ(ierr);
  ierr = PFInitializePackage();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#endif /* PETSC_HAVE_DYNAMIC_LIBRARIES */
