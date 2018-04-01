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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListDestroy(&TaoList);CHKERRQ(ierr);
  TaoPackageInitialized = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  TaoInitializePackage - This function sets up PETSc to use the Tao
  package.  When using static libraries, this function is called from the
  first entry to TaoCreate(); when using shared libraries, it is called
  from PetscDLLibraryRegister()

  Level: developer

.seealso: TaoCreate()
@*/
PetscErrorCode TaoInitializePackage(void)
{
  char           logList[256];
  PetscBool      opt,pkg;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  if (TaoPackageInitialized) PetscFunctionReturn(0);
  TaoPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscClassIdRegister("Tao",&TAO_CLASSID);CHKERRQ(ierr);
  /* Register Constructors */
  ierr = TaoRegisterAll();CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister("TaoSolve",TAO_CLASSID,&Tao_Solve);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TaoObjectiveEval",TAO_CLASSID,&Tao_ObjectiveEval);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TaoGradientEval",TAO_CLASSID,&Tao_GradientEval);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TaoHessianEval",TAO_CLASSID,&Tao_HessianEval);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TaoConstraintsEval",TAO_CLASSID,&Tao_ConstraintsEval);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TaoJacobianEval",TAO_CLASSID,&Tao_JacobianEval);CHKERRQ(ierr);
  /* Process info exclusions */
  ierr = PetscOptionsGetString(NULL,NULL,"-info_exclude",logList,sizeof(logList),&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrInList("tao",logList,',',&pkg);CHKERRQ(ierr);
    if (pkg) {ierr = PetscInfoDeactivateClass(TAO_CLASSID);CHKERRQ(ierr);}
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrInList("tao",logList,',',&pkg);CHKERRQ(ierr);
    if (pkg) {ierr = PetscLogEventDeactivateClass(TAO_CLASSID);CHKERRQ(ierr);}
  }
  /* Register package finalizer */
  ierr = PetscRegisterFinalize(TaoFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#ifdef PETSC_USE_DYNAMIC_LIBRARIES
/*
  PetscDLLibraryRegister - this function is called when the dynamic library it
  is in is opened.

  This registers all of the Tao methods that are in the libtao
  library.

  Input Parameter:
. path - library path
*/

PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_tao(void)
{
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = TaoInitializePackage();CHKERRQ(ierr);
    ierr = TaoLineSearchInitializePackage();CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#endif /* PETSC_USE_DYNAMIC_LIBRARIES */
