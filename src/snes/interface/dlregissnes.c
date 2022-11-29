
#include <petsc/private/snesimpl.h>
#include <petsc/private/linesearchimpl.h>

static PetscBool SNESPackageInitialized = PETSC_FALSE;

/*@C
  SNESFinalizePackage - This function destroys everything in the Petsc interface to the SNES package. It is
  called from PetscFinalize().

  Level: developer

.seealso: `PetscFinalize()`
@*/
PetscErrorCode SNESFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListDestroy(&SNESList));
  PetscCall(PetscFunctionListDestroy(&SNESLineSearchList));
  SNESPackageInitialized          = PETSC_FALSE;
  SNESRegisterAllCalled           = PETSC_FALSE;
  SNESLineSearchRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  SNESInitializePackage - This function initializes everything in the SNES package. It is called
  from PetscDLLibraryRegister_petscsnes() when using dynamic libraries, and on the first call to SNESCreate()
  when using shared or static libraries.

  Level: developer

.seealso: `PetscInitialize()`
@*/
PetscErrorCode SNESInitializePackage(void)
{
  char      logList[256];
  PetscBool opt, pkg, cls;

  PetscFunctionBegin;
  if (SNESPackageInitialized) PetscFunctionReturn(0);
  SNESPackageInitialized = PETSC_TRUE;
  /* Initialize subpackages */
  PetscCall(SNESMSInitializePackage());
  /* Register Classes */
  PetscCall(PetscClassIdRegister("SNES", &SNES_CLASSID));
  PetscCall(PetscClassIdRegister("DMSNES", &DMSNES_CLASSID));
  PetscCall(PetscClassIdRegister("SNESLineSearch", &SNESLINESEARCH_CLASSID));
  /* Register Constructors */
  PetscCall(SNESRegisterAll());
  PetscCall(SNESLineSearchRegisterAll());
  /* Register Events */
  PetscCall(PetscLogEventRegister("SNESSolve", SNES_CLASSID, &SNES_Solve));
  PetscCall(PetscLogEventRegister("SNESSetUp", SNES_CLASSID, &SNES_Setup));
  PetscCall(PetscLogEventRegister("SNESFunctionEval", SNES_CLASSID, &SNES_FunctionEval));
  PetscCall(PetscLogEventRegister("SNESObjectiveEval", SNES_CLASSID, &SNES_ObjectiveEval));
  PetscCall(PetscLogEventRegister("SNESNGSEval", SNES_CLASSID, &SNES_NGSEval));
  PetscCall(PetscLogEventRegister("SNESNGSFuncEval", SNES_CLASSID, &SNES_NGSFuncEval));
  PetscCall(PetscLogEventRegister("SNESJacobianEval", SNES_CLASSID, &SNES_JacobianEval));
  PetscCall(PetscLogEventRegister("SNESNPCSolve", SNES_CLASSID, &SNES_NPCSolve));
  PetscCall(PetscLogEventRegister("SNESLineSearch", SNESLINESEARCH_CLASSID, &SNESLINESEARCH_Apply));
  /* Process Info */
  {
    PetscClassId classids[3];

    classids[0] = SNES_CLASSID;
    classids[1] = DMSNES_CLASSID;
    classids[2] = SNESLINESEARCH_CLASSID;
    PetscCall(PetscInfoProcessClass("snes", 1, classids));
    PetscCall(PetscInfoProcessClass("dm", 1, &classids[1]));
    PetscCall(PetscInfoProcessClass("sneslinesearch", 1, &classids[2]));
  }
  /* Process summary exclusions */
  PetscCall(PetscOptionsGetString(NULL, NULL, "-log_exclude", logList, sizeof(logList), &opt));
  if (opt) {
    PetscCall(PetscStrInList("snes", logList, ',', &pkg));
    if (pkg) PetscCall(PetscLogEventExcludeClass(SNES_CLASSID));
    PetscCall(PetscStrInList("dm", logList, ',', &cls));
    if (pkg || cls) PetscCall(PetscLogEventExcludeClass(DMSNES_CLASSID));
    PetscCall(PetscStrInList("sneslinesearch", logList, ',', &cls));
    if (pkg || cls) PetscCall(PetscLogEventExcludeClass(SNESLINESEARCH_CLASSID));
  }
  /* Register package finalizer */
  PetscCall(PetscRegisterFinalize(SNESFinalizePackage));
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_DYNAMIC_LIBRARIES)
/*
  PetscDLLibraryRegister - This function is called when the dynamic library it is in is opened.

  This registers all of the SNES methods that are in the basic PETSc libpetscsnes library.

 */
PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_petscsnes(void)
{
  PetscFunctionBegin;
  PetscCall(SNESInitializePackage());
  PetscFunctionReturn(0);
}

#endif /* PETSC_HAVE_DYNAMIC_LIBRARIES */
