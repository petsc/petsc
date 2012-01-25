#define TAOSOLVER_DLL

#include "include/private/taosolver_impl.h"
#include "include/private/taodm_impl.h"

static PetscBool TaoPackageInitialized = PETSC_FALSE;

#undef __FUNCT__
#define __FUNCT__ "TaoInitializePackage"
/*@C
  TaoInitializePackage - This function sets up PETSc to use the TaoSolver 
  package.  When using static libraries, this function is called from the
  first entry to TaoCreate(); when using shared libraries, it is called
  from PetscDLLibraryRegister()

  Input parameter:
. path - The dynamic library path or PETSC_NULL

  Level: developer

.seealso: TaoCreate()
@*/
PetscErrorCode TaoInitializePackage(const char path[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;

  if (TaoPackageInitialized) PetscFunctionReturn(0);
  TaoPackageInitialized = PETSC_TRUE;

  ierr = PetscClassIdRegister("TaoSolver",&TAOSOLVER_CLASSID); CHKERRQ(ierr);
  
  /* Tell PETSc what solvers are available */
  ierr = TaoSolverRegisterAll(path); CHKERRQ(ierr);

  /* Tell PETSc what events are associated with TaoSolver */
  ierr = PetscLogEventRegister("TaoSolve",TAOSOLVER_CLASSID,&TaoSolver_Solve); CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TaoObjectiveEval",TAOSOLVER_CLASSID,&TaoSolver_ObjectiveEval); CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TaoGradientEval",TAOSOLVER_CLASSID,&TaoSolver_GradientEval); CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TaoHessianEval",TAOSOLVER_CLASSID,&TaoSolver_HessianEval); CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TaoConstraintsEval",TAOSOLVER_CLASSID,&TaoSolver_ConstraintsEval); CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TaoJacobianEval",TAOSOLVER_CLASSID,&TaoSolver_JacobianEval); CHKERRQ(ierr);


  PetscFunctionReturn(0);
}

#ifdef PETSC_USE_DYNAMIC_LIBRARIES
EXTERN_C_BEGIN
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

PetscErrorCode PetscDLLibraryRegister_tao(const char path[])
{
    PetscErrorCode ierr;

    ierr = PetscInitializeNoArguments();
    if (ierr)
	return 1;
    PetscFunctionBegin;
    ierr = TaoInitializePackage(path); CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
EXTERN_C_END
#endif /* PETSC_USE_DYNAMIC_LIBRARIES */
