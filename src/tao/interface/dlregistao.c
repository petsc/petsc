#define TAO_DLL

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
  CHKERRQ(PetscFunctionListDestroy(&TaoList));
  TaoPackageInitialized = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  TaoInitializePackage - This function sets up PETSc to use the Tao
  package.  When using static or shared libraries, this function is called from the
  first entry to TaoCreate(); when using shared or static libraries, it is called
  from PetscDLLibraryRegister_tao()

  Level: developer

.seealso: TaoCreate()
@*/
PetscErrorCode TaoInitializePackage(void)
{
  char           logList[256];
  PetscBool      opt,pkg;

  PetscFunctionBegin;
  if (TaoPackageInitialized) PetscFunctionReturn(0);
  TaoPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  CHKERRQ(PetscClassIdRegister("Tao",&TAO_CLASSID));
  /* Register Constructors */
  CHKERRQ(TaoRegisterAll());
  /* Register Events */
  CHKERRQ(PetscLogEventRegister("TaoSolve",         TAO_CLASSID,&TAO_Solve));
  CHKERRQ(PetscLogEventRegister("TaoObjectiveEval", TAO_CLASSID,&TAO_ObjectiveEval));
  CHKERRQ(PetscLogEventRegister("TaoGradientEval",  TAO_CLASSID,&TAO_GradientEval));
  CHKERRQ(PetscLogEventRegister("TaoObjGradEval",   TAO_CLASSID,&TAO_ObjGradEval));
  CHKERRQ(PetscLogEventRegister("TaoHessianEval",   TAO_CLASSID,&TAO_HessianEval));
  CHKERRQ(PetscLogEventRegister("TaoConstrEval",    TAO_CLASSID,&TAO_ConstraintsEval));
  CHKERRQ(PetscLogEventRegister("TaoJacobianEval",  TAO_CLASSID,&TAO_JacobianEval));
  /* Process Info */
  {
    PetscClassId  classids[1];

    classids[0] = TAO_CLASSID;
    CHKERRQ(PetscInfoProcessClass("tao", 1, classids));
  }
  /* Process summary exclusions */
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt));
  if (opt) {
    CHKERRQ(PetscStrInList("tao",logList,',',&pkg));
    if (pkg) CHKERRQ(PetscLogEventExcludeClass(TAO_CLASSID));
  }
  /* Register package finalizer */
  CHKERRQ(PetscRegisterFinalize(TaoFinalizePackage));
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
  CHKERRQ(TaoInitializePackage());
  CHKERRQ(TaoLineSearchInitializePackage());
  PetscFunctionReturn(0);
}
#endif /* PETSC_HAVE_DYNAMIC_LIBRARIES */
