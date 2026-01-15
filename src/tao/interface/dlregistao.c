#include <petsc/private/taoimpl.h>

static PetscBool TaoPackageInitialized = PETSC_FALSE;

/*@C
  TaoFinalizePackage - This function destroys everything in the PETSc/Tao
  interface to the Tao package. It is called from `PetscFinalize()`.

  Level: developer

.seealso: `TaoInitializePackage()`, `PetscFinalize()`, `TaoRegister()`, `TaoRegisterAll()`
@*/
PetscErrorCode TaoFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListDestroy(&TaoList));
  PetscCall(PetscFunctionListDestroy(&TaoTermList));
  TaoPackageInitialized = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoInitializePackage - This function sets up PETSc to use the Tao
  package.  When using static or shared libraries, this function is called from the
  first entry to `TaoCreate()`; when using shared or static libraries, it is called
  from PetscDLLibraryRegister_tao()

  Level: developer

  Note:
  This function never needs to be called by PETSc users.

.seealso: `TaoCreate()`, `TaoFinalizePackage()`, `TaoRegister()`, `TaoRegisterAll()`
@*/
PetscErrorCode TaoInitializePackage(void)
{
  char      logList[256];
  PetscBool opt, pkg;

  PetscFunctionBegin;
  if (TaoPackageInitialized) PetscFunctionReturn(PETSC_SUCCESS);
  TaoPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  PetscCall(PetscClassIdRegister("Tao", &TAO_CLASSID));
  PetscCall(PetscClassIdRegister("TaoTerm", &TAOTERM_CLASSID));
  /* Register Constructors */
  PetscCall(TaoRegisterAll());
  PetscCall(TaoTermRegisterAll());
  /* Register Events */
  PetscCall(PetscLogEventRegister("TaoSolve", TAO_CLASSID, &TAO_Solve));
  PetscCall(PetscLogEventRegister("TaoTermObjEval", TAOTERM_CLASSID, &TAOTERM_ObjectiveEval));
  PetscCall(PetscLogEventRegister("TaoTermGradEval", TAOTERM_CLASSID, &TAOTERM_GradientEval));
  PetscCall(PetscLogEventRegister("TaoTermObjGrad", TAOTERM_CLASSID, &TAOTERM_ObjGradEval));
  PetscCall(PetscLogEventRegister("TaoTermHessEval", TAOTERM_CLASSID, &TAOTERM_HessianEval));
  PetscCall(PetscLogEventRegister("TaoResidualEval", TAO_CLASSID, &TAO_ResidualEval));
  PetscCall(PetscLogEventRegister("TaoConstrEval", TAO_CLASSID, &TAO_ConstraintsEval));
  PetscCall(PetscLogEventRegister("TaoJacobianEval", TAO_CLASSID, &TAO_JacobianEval));
  /* Process Info */
  {
    PetscClassId classids[2];

    classids[0] = TAO_CLASSID;
    classids[1] = TAOTERM_CLASSID;
    PetscCall(PetscInfoProcessClass("tao", 1, classids));
    PetscCall(PetscInfoProcessClass("taoterm", 1, &classids[1]));
  }
  /* Process summary exclusions */
  PetscCall(PetscOptionsGetString(NULL, NULL, "-log_exclude", logList, sizeof(logList), &opt));
  if (opt) {
    PetscCall(PetscStrInList("tao", logList, ',', &pkg));
    if (pkg) PetscCall(PetscLogEventExcludeClass(TAO_CLASSID));
    PetscCall(PetscStrInList("taoterm", logList, ',', &pkg));
    if (pkg) PetscCall(PetscLogEventExcludeClass(TAOTERM_CLASSID));
  }
  /* Register package finalizer */
  PetscCall(PetscRegisterFinalize(TaoFinalizePackage));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if defined(PETSC_HAVE_DYNAMIC_LIBRARIES)
/*
  PetscDLLibraryRegister - this function is called when the dynamic library it
  is in is opened.

  This registers all of the Tao methods that are in the libtao
  library.

  Input Parameter:
. path - library path
*/
PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_petsctao(void)
{
  PetscFunctionBegin;
  PetscCall(TaoInitializePackage());
  PetscCall(TaoLineSearchInitializePackage());
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif /* PETSC_HAVE_DYNAMIC_LIBRARIES */
