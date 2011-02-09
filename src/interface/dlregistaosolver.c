#define TAOSOLVER_DLL

#include "include/private/taosolver_impl.h"
#include "include/private/taodm_impl.h"

static PetscBool TaoSolverPackageInitialized = PETSC_FALSE;

#undef __FUNCT__
#define __FUNCT__ "TaoSolverInitializePackage"
/*@C
  TaoSolverInitializePackage - This function sets up PETSc to use the TaoSolver 
  package.  When using static libraries, this function is called from the
  first entry to TaoSolverCreate(); when using shared libraries, it is called
  from PetscDLLibraryRegister()

  Input parameter:
. path - The dynamic library path or PETSC_NULL

  Level: developer

.seealso: TaoSolverCreate()
@*/
PetscErrorCode TaoSolverInitializePackage(const char path[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;

  if (TaoSolverPackageInitialized) PetscFunctionReturn(0);
  TaoSolverPackageInitialized = PETSC_TRUE;

  ierr = PetscClassIdRegister("TaoSolver",&TAOSOLVER_CLASSID); CHKERRQ(ierr);
  
  /* Tell PETSc what solvers are available */
  ierr = TaoSolverRegisterAll(path); CHKERRQ(ierr);

  /* Tell PETSc what events are associated with TaoSolver */
  ierr = PetscLogEventRegister("TaoSolverSolve",TAOSOLVER_CLASSID,&TaoSolver_Solve); CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TaoSolverObjectiveEval",TAOSOLVER_CLASSID,&TaoSolver_ObjectiveEval); CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TaoSolverGradientEval",TAOSOLVER_CLASSID,&TaoSolver_GradientEval); CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TaoSolverHessianEval",TAOSOLVER_CLASSID,&TaoSolver_HessianEval); CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TaoSolverJacobianEval",TAOSOLVER_CLASSID,&TaoSolver_JacobianEval); CHKERRQ(ierr);


  PetscFunctionReturn(0);
}

#ifdef PETSC_USE_DYNAMIC_LIBRARIES
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PetscDLLibraryRegister_taosolver"
/*
  PetscDLLibraryRegister - this function is called when the dynamic library it
  is in is opened.

  This registers all of the TaoSolver methods that are in the libtaosolver
  library.

  Input Parameter:
. path - library path
*/

PetscErrorCode PetscDLLibraryRegister_taosolver(const char path[])
{
    PetscErrorCode ierr;

    ierr = PetscInitializeNoArguments();
    if (ierr)
	return 1;
    PetscFunctionBegin;
    ierr = TaoSolverInitializePackage(path); CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
EXTERN_C_END
#endif /* PETSC_USE_DYNAMIC_LIBRARIES */
