
#include <petsc/private/snesimpl.h>
#include <petsc/private/linesearchimpl.h>

static PetscBool SNESPackageInitialized = PETSC_FALSE;

/*@C
  SNESFinalizePackage - This function destroys everything in the Petsc interface to the SNES package. It is
  called from PetscFinalize().

  Level: developer

.keywords: Petsc, destroy, package, mathematica
.seealso: PetscFinalize()
@*/
PetscErrorCode  SNESFinalizePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListDestroy(&SNESList);CHKERRQ(ierr);
  ierr = PetscFunctionListDestroy(&SNESLineSearchList);CHKERRQ(ierr);
  SNESPackageInitialized = PETSC_FALSE;
  SNESRegisterAllCalled  = PETSC_FALSE;
  SNESLineSearchRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  SNESInitializePackage - This function initializes everything in the SNES package. It is called
  from PetscDLLibraryRegister() when using dynamic libraries, and on the first call to SNESCreate()
  when using static libraries.

  Level: developer

.keywords: SNES, initialize, package
.seealso: PetscInitialize()
@*/
PetscErrorCode  SNESInitializePackage(void)
{
  char           logList[256];
  char           *className;
  PetscBool      opt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (SNESPackageInitialized) PetscFunctionReturn(0);
  SNESPackageInitialized = PETSC_TRUE;
  /* Initialize subpackages */
  ierr = SNESMSInitializePackage();CHKERRQ(ierr);
  /* Register Classes */
  ierr = PetscClassIdRegister("SNES",&SNES_CLASSID);CHKERRQ(ierr);
  ierr = PetscClassIdRegister("SNESLineSearch",&SNESLINESEARCH_CLASSID);CHKERRQ(ierr);
  ierr = PetscClassIdRegister("DMSNES",&DMSNES_CLASSID);CHKERRQ(ierr);
  /* Register Constructors */
  ierr = SNESRegisterAll();CHKERRQ(ierr);
  ierr = SNESLineSearchRegisterAll();CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister("SNESSolve",            SNES_CLASSID,&SNES_Solve);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SNESFunctionEval",     SNES_CLASSID,&SNES_FunctionEval);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SNESObjectiveEval",    SNES_CLASSID,&SNES_ObjectiveEval);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SNESNGSEval",          SNES_CLASSID,&SNES_NGSEval);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SNESNGSFuncEval",      SNES_CLASSID,&SNES_NGSFuncEval);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SNESJacobianEval",     SNES_CLASSID,&SNES_JacobianEval);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SNESLineSearch",       SNESLINESEARCH_CLASSID,&SNESLINESEARCH_Apply);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SNESNPCSolve",         SNES_CLASSID,&SNES_NPCSolve);CHKERRQ(ierr);
  /* Process info exclusions */
  ierr = PetscOptionsGetString(NULL,NULL, "-info_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "snes", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscInfoDeactivateClass(SNES_CLASSID);CHKERRQ(ierr);
    }
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(NULL,NULL, "-log_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "snes", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(SNES_CLASSID);CHKERRQ(ierr);
    }
  }
  ierr = PetscRegisterFinalize(SNESFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_DYNAMIC_LIBRARIES)
/*
  PetscDLLibraryRegister - This function is called when the dynamic library it is in is opened.

  This registers all of the SNES methods that are in the basic PETSc libpetscsnes library.

 */
PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_petscsnes(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESInitializePackage();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#endif /* PETSC_HAVE_DYNAMIC_LIBRARIES */
