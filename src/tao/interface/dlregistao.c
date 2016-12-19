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
  PetscErrorCode ierr;

  PetscFunctionBegin;

  if (TaoPackageInitialized) PetscFunctionReturn(0);
  TaoPackageInitialized = PETSC_TRUE;

  ierr = PetscClassIdRegister("Tao",&TAO_CLASSID);CHKERRQ(ierr);

  /* Tell PETSc what solvers are available */
  ierr = TaoRegisterAll();CHKERRQ(ierr);

  /* Tell PETSc what events are associated with Tao */
  ierr = PetscLogEventRegister("TaoSolve",TAO_CLASSID,&Tao_Solve);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TaoObjectiveEval",TAO_CLASSID,&Tao_ObjectiveEval);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TaoGradientEval",TAO_CLASSID,&Tao_GradientEval);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TaoHessianEval",TAO_CLASSID,&Tao_HessianEval);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TaoConstraintsEval",TAO_CLASSID,&Tao_ConstraintsEval);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TaoJacobianEval",TAO_CLASSID,&Tao_JacobianEval);CHKERRQ(ierr);

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
