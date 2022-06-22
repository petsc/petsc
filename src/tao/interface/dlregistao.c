#include <petsc/private/taoimpl.h>

static PetscBool TaoPackageInitialized = PETSC_FALSE;

/*@C
  TaoFinalizePackage - This function destroys everything in the PETSc/TAO
  interface to the Tao package. It is called from PetscFinalize().

  Level: developer
@*/
PetscErrorCode TaoFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListDestroy(&TaoList));
  TaoPackageInitialized = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  TaoInitializePackage - This function sets up PETSc to use the Tao
  package.  When using static or shared libraries, this function is called from the
  first entry to TaoCreate(); when using shared or static libraries, it is called
  from PetscDLLibraryRegister_tao()

  Level: developer

.seealso: `TaoCreate()`
@*/
PetscErrorCode TaoInitializePackage(void)
{
  char           logList[256];
  PetscBool      opt,pkg;

  PetscFunctionBegin;
  if (TaoPackageInitialized) PetscFunctionReturn(0);
  TaoPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  PetscCall(PetscClassIdRegister("Tao",&TAO_CLASSID));
  /* Register Constructors */
  PetscCall(TaoRegisterAll());
  /* Register Events */
  PetscCall(PetscLogEventRegister("TaoSolve",         TAO_CLASSID,&TAO_Solve));
  PetscCall(PetscLogEventRegister("TaoObjectiveEval", TAO_CLASSID,&TAO_ObjectiveEval));
  PetscCall(PetscLogEventRegister("TaoGradientEval",  TAO_CLASSID,&TAO_GradientEval));
  PetscCall(PetscLogEventRegister("TaoObjGradEval",   TAO_CLASSID,&TAO_ObjGradEval));
  PetscCall(PetscLogEventRegister("TaoHessianEval",   TAO_CLASSID,&TAO_HessianEval));
  PetscCall(PetscLogEventRegister("TaoConstrEval",    TAO_CLASSID,&TAO_ConstraintsEval));
  PetscCall(PetscLogEventRegister("TaoJacobianEval",  TAO_CLASSID,&TAO_JacobianEval));
  /* Process Info */
  {
    PetscClassId  classids[1];

    classids[0] = TAO_CLASSID;
    PetscCall(PetscInfoProcessClass("tao", 1, classids));
  }
  /* Process summary exclusions */
  PetscCall(PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt));
  if (opt) {
    PetscCall(PetscStrInList("tao",logList,',',&pkg));
    if (pkg) PetscCall(PetscLogEventExcludeClass(TAO_CLASSID));
  }
  /* Register package finalizer */
  PetscCall(PetscRegisterFinalize(TaoFinalizePackage));
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_DYNAMIC_LIBRARIES)
/*
  PetscDLLibraryRegister - this function is called when the dynamic library it
  is in is opened.

  This registers all of the Tao methods that are in the libtao
  library.

  Input Parameter:
. path - library path
*/
PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_petsctao(void)
{
  PetscFunctionBegin;
  PetscCall(TaoInitializePackage());
  PetscCall(TaoLineSearchInitializePackage());
  PetscFunctionReturn(0);
}
#endif /* PETSC_HAVE_DYNAMIC_LIBRARIES */
