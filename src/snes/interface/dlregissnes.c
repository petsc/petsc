
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
  from PetscDLLibraryRegister_petscsnes() when using dynamic libraries, and on the first call to SNESCreate()
  when using shared or static libraries.

  Level: developer

.seealso: PetscInitialize()
@*/
PetscErrorCode  SNESInitializePackage(void)
{
  char           logList[256];
  PetscBool      opt,pkg,cls;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (SNESPackageInitialized) PetscFunctionReturn(0);
  SNESPackageInitialized = PETSC_TRUE;
  /* Initialize subpackages */
  ierr = SNESMSInitializePackage();CHKERRQ(ierr);
  /* Register Classes */
  ierr = PetscClassIdRegister("SNES",&SNES_CLASSID);CHKERRQ(ierr);
  ierr = PetscClassIdRegister("DMSNES",&DMSNES_CLASSID);CHKERRQ(ierr);
  ierr = PetscClassIdRegister("SNESLineSearch",&SNESLINESEARCH_CLASSID);CHKERRQ(ierr);
  /* Register Constructors */
  ierr = SNESRegisterAll();CHKERRQ(ierr);
  ierr = SNESLineSearchRegisterAll();CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister("SNESSolve",            SNES_CLASSID,&SNES_Solve);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SNESSetUp",            SNES_CLASSID,&SNES_Setup);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SNESFunctionEval",     SNES_CLASSID,&SNES_FunctionEval);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SNESObjectiveEval",    SNES_CLASSID,&SNES_ObjectiveEval);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SNESNGSEval",          SNES_CLASSID,&SNES_NGSEval);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SNESNGSFuncEval",      SNES_CLASSID,&SNES_NGSFuncEval);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SNESJacobianEval",     SNES_CLASSID,&SNES_JacobianEval);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SNESNPCSolve",         SNES_CLASSID,&SNES_NPCSolve);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SNESLineSearch",       SNESLINESEARCH_CLASSID,&SNESLINESEARCH_Apply);CHKERRQ(ierr);
  /* Process Info */
  {
    PetscClassId  classids[3];

    classids[0] = SNES_CLASSID;
    classids[1] = DMSNES_CLASSID;
    classids[2] = SNESLINESEARCH_CLASSID;
    ierr = PetscInfoProcessClass("snes", 1, classids);CHKERRQ(ierr);
    ierr = PetscInfoProcessClass("dm", 1, &classids[1]);CHKERRQ(ierr);
    ierr = PetscInfoProcessClass("sneslinesearch", 1, &classids[2]);CHKERRQ(ierr);
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrInList("snes",logList,',',&pkg);CHKERRQ(ierr);
    if (pkg) {ierr = PetscLogEventExcludeClass(SNES_CLASSID);CHKERRQ(ierr);}
    ierr = PetscStrInList("dm",logList,',',&cls);CHKERRQ(ierr);
    if (pkg || cls) {ierr = PetscLogEventExcludeClass(DMSNES_CLASSID);CHKERRQ(ierr);}
    ierr = PetscStrInList("sneslinesearch",logList,',',&cls);CHKERRQ(ierr);
    if (pkg || cls) {ierr = PetscLogEventExcludeClass(SNESLINESEARCH_CLASSID);CHKERRQ(ierr);}
  }
  /* Register package finalizer */
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
