
#include <petsc/private/snesimpl.h>
#include <petsc/private/linesearchimpl.h>

static PetscBool SNESPackageInitialized = PETSC_FALSE;

/*@C
  SNESFinalizePackage - This function destroys everything in the Petsc interface to the SNES package. It is
  called from PetscFinalize().

  Level: developer

.seealso: PetscFinalize()
@*/
PetscErrorCode  SNESFinalizePackage(void)
{
  PetscFunctionBegin;
  CHKERRQ(PetscFunctionListDestroy(&SNESList));
  CHKERRQ(PetscFunctionListDestroy(&SNESLineSearchList));
  SNESPackageInitialized = PETSC_FALSE;
  SNESRegisterAllCalled  = PETSC_FALSE;
  SNESLineSearchRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  SNESInitializePackage - This function initializes everything in the SNES package. It is called
  from PetscDLLibraryRegister_petscsnes() when using dynamic libraries, and on the first call to SNESCreate()
  when using shared or static libraries.

  Level: developer

.seealso: PetscInitialize()
@*/
PetscErrorCode  SNESInitializePackage(void)
{
  char           logList[256];
  PetscBool      opt,pkg,cls;

  PetscFunctionBegin;
  if (SNESPackageInitialized) PetscFunctionReturn(0);
  SNESPackageInitialized = PETSC_TRUE;
  /* Initialize subpackages */
  CHKERRQ(SNESMSInitializePackage());
  /* Register Classes */
  CHKERRQ(PetscClassIdRegister("SNES",&SNES_CLASSID));
  CHKERRQ(PetscClassIdRegister("DMSNES",&DMSNES_CLASSID));
  CHKERRQ(PetscClassIdRegister("SNESLineSearch",&SNESLINESEARCH_CLASSID));
  /* Register Constructors */
  CHKERRQ(SNESRegisterAll());
  CHKERRQ(SNESLineSearchRegisterAll());
  /* Register Events */
  CHKERRQ(PetscLogEventRegister("SNESSolve",            SNES_CLASSID,&SNES_Solve));
  CHKERRQ(PetscLogEventRegister("SNESSetUp",            SNES_CLASSID,&SNES_Setup));
  CHKERRQ(PetscLogEventRegister("SNESFunctionEval",     SNES_CLASSID,&SNES_FunctionEval));
  CHKERRQ(PetscLogEventRegister("SNESObjectiveEval",    SNES_CLASSID,&SNES_ObjectiveEval));
  CHKERRQ(PetscLogEventRegister("SNESNGSEval",          SNES_CLASSID,&SNES_NGSEval));
  CHKERRQ(PetscLogEventRegister("SNESNGSFuncEval",      SNES_CLASSID,&SNES_NGSFuncEval));
  CHKERRQ(PetscLogEventRegister("SNESJacobianEval",     SNES_CLASSID,&SNES_JacobianEval));
  CHKERRQ(PetscLogEventRegister("SNESNPCSolve",         SNES_CLASSID,&SNES_NPCSolve));
  CHKERRQ(PetscLogEventRegister("SNESLineSearch",       SNESLINESEARCH_CLASSID,&SNESLINESEARCH_Apply));
  /* Process Info */
  {
    PetscClassId  classids[3];

    classids[0] = SNES_CLASSID;
    classids[1] = DMSNES_CLASSID;
    classids[2] = SNESLINESEARCH_CLASSID;
    CHKERRQ(PetscInfoProcessClass("snes", 1, classids));
    CHKERRQ(PetscInfoProcessClass("dm", 1, &classids[1]));
    CHKERRQ(PetscInfoProcessClass("sneslinesearch", 1, &classids[2]));
  }
  /* Process summary exclusions */
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt));
  if (opt) {
    CHKERRQ(PetscStrInList("snes",logList,',',&pkg));
    if (pkg) CHKERRQ(PetscLogEventExcludeClass(SNES_CLASSID));
    CHKERRQ(PetscStrInList("dm",logList,',',&cls));
    if (pkg || cls) CHKERRQ(PetscLogEventExcludeClass(DMSNES_CLASSID));
    CHKERRQ(PetscStrInList("sneslinesearch",logList,',',&cls));
    if (pkg || cls) CHKERRQ(PetscLogEventExcludeClass(SNESLINESEARCH_CLASSID));
  }
  /* Register package finalizer */
  CHKERRQ(PetscRegisterFinalize(SNESFinalizePackage));
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
  CHKERRQ(SNESInitializePackage());
  PetscFunctionReturn(0);
}

#endif /* PETSC_HAVE_DYNAMIC_LIBRARIES */
