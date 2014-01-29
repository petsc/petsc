#define TAOSOLVER_DLL

#include "tao-private/taosolver_impl.h"
#include "tao-private/taodm_impl.h"

static PetscBool TaoPackageInitialized = PETSC_FALSE;

#undef __FUNCT__
#define __FUNCT__ "TaoFinalizePackage"
/*@C
  TaoFinalizePackage - This function destroys everything in the PETSc/TAO
  interface to the TaoSolver package. It is called from PetscFinalize().

  Level: developer
@*/
PetscErrorCode TaoFinalizePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListDestroy(&TaoSolverList);CHKERRQ(ierr);
  TaoPackageInitialized = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoInitializePackage"
/*@C
  TaoInitializePackage - This function sets up PETSc to use the TaoSolver
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

  ierr = PetscClassIdRegister("TaoSolver",&TAOSOLVER_CLASSID);CHKERRQ(ierr);

  /* Tell PETSc what solvers are available */
  ierr = TaoSolverRegisterAll();CHKERRQ(ierr);

  /* Tell PETSc what events are associated with TaoSolver */
  ierr = PetscLogEventRegister("TaoSolve",TAOSOLVER_CLASSID,&TaoSolver_Solve);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TaoObjectiveEval",TAOSOLVER_CLASSID,&TaoSolver_ObjectiveEval);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TaoGradientEval",TAOSOLVER_CLASSID,&TaoSolver_GradientEval);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TaoHessianEval",TAOSOLVER_CLASSID,&TaoSolver_HessianEval);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TaoConstraintsEval",TAOSOLVER_CLASSID,&TaoSolver_ConstraintsEval);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TaoJacobianEval",TAOSOLVER_CLASSID,&TaoSolver_JacobianEval);CHKERRQ(ierr);

  ierr = PetscRegisterFinalize(TaoFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#ifdef PETSC_USE_DYNAMIC_LIBRARIES
#undef __FUNCT__
#define __FUNCT__ "PetscDLLibraryRegister_tao"
/*
  PetscDLLibraryRegister - this function is called when the dynamic library it
  is in is opened.

  This registers all of the TaoSolver methods that are in the libtaosolver
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
