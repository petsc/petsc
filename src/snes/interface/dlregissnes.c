
#include <petsc-private/snesimpl.h>
#include <petsc-private/linesearchimpl.h>

static PetscBool  SNESPackageInitialized = PETSC_FALSE;

#undef __FUNCT__
#define __FUNCT__ "SNESFinalizePackage"
/*@C
  SNESFinalizePackage - This function destroys everything in the Petsc interface to the SNES package. It is
  called from PetscFinalize().

  Level: developer

.keywords: Petsc, destroy, package, mathematica
.seealso: PetscFinalize()
@*/
PetscErrorCode  SNESFinalizePackage(void)
{
  PetscFunctionBegin;
  SNESPackageInitialized = PETSC_FALSE;
  SNESRegisterAllCalled  = PETSC_FALSE;
  SNESList               = PETSC_NULL;
  SNESLineSearchList     = PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESInitializePackage"
/*@C
  SNESInitializePackage - This function initializes everything in the SNES package. It is called
  from PetscDLLibraryRegister() when using dynamic libraries, and on the first call to SNESCreate()
  when using static libraries.

  Input Parameter:
  path - The dynamic library path, or PETSC_NULL

  Level: developer

.keywords: SNES, initialize, package
.seealso: PetscInitialize()
@*/
PetscErrorCode  SNESInitializePackage(const char path[])
{
  char              logList[256];
  char              *className;
  PetscBool         opt;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (SNESPackageInitialized) PetscFunctionReturn(0);
  SNESPackageInitialized = PETSC_TRUE;
  /* Initialize subpackages */
  ierr = SNESMSInitializePackage(path);CHKERRQ(ierr);
  /* Register Classes */
  ierr = PetscClassIdRegister("SNES",&SNES_CLASSID);CHKERRQ(ierr);
  ierr = PetscClassIdRegister("SNESLineSearch",&SNESLINESEARCH_CLASSID);CHKERRQ(ierr);
  /* Register Constructors */
  ierr = SNESRegisterAll(path);CHKERRQ(ierr);
  ierr = SNESLineSearchRegisterAll(path);CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister("SNESSolve",            SNES_CLASSID,&SNES_Solve);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SNESFunctionEval",     SNES_CLASSID,&SNES_FunctionEval);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SNESGSEval",           SNES_CLASSID,&SNES_GSEval);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SNESJacobianEval",     SNES_CLASSID,&SNES_JacobianEval);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SNESLineSearch",       SNESLINESEARCH_CLASSID,&SNESLineSearch_Apply);CHKERRQ(ierr);
  /* Process info exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-info_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "snes", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscInfoDeactivateClass(SNES_CLASSID);CHKERRQ(ierr);
    }
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-log_summary_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "snes", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(SNES_CLASSID);CHKERRQ(ierr);
    }
  }
  ierr = PetscRegisterFinalize(SNESFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#ifdef PETSC_USE_DYNAMIC_LIBRARIES
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PetscDLLibraryRegister_petscsnes"
/*
  PetscDLLibraryRegister - This function is called when the dynamic library it is in is opened.

  This registers all of the SNES methods that are in the basic PETSc libpetscsnes library.

  Input Parameter:
  path - library path

 */
PetscErrorCode  PetscDLLibraryRegister_petscsnes(const char path[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESInitializePackage(path);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#endif /* PETSC_USE_DYNAMIC_LIBRARIES */
