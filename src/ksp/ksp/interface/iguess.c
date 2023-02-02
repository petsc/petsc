#include <petsc/private/kspimpl.h> /*I "petscksp.h"  I*/

PetscFunctionList KSPGuessList = NULL;
static PetscBool  KSPGuessRegisterAllCalled;

/*
  KSPGuessRegister -  Adds a method for initial guess computation in Krylov subspace solver package.

   Not Collective

   Input Parameters:
+  name_solver - name of a new user-defined solver
-  routine_create - routine to create method context

   Notes:
   KSPGuessRegister() may be called multiple times to add several user-defined solvers.

   Sample usage:
.vb
   KSPGuessRegister("my_initial_guess",MyInitialGuessCreate);
.ve

   Then, it can be chosen with the procedural interface via
$     KSPSetGuessType(ksp,"my_initial_guess")
   or at runtime via the option
$     -ksp_guess_type my_initial_guess

   Level: developer

.seealso: [](chapter_ksp), `KSPGuess`, `KSPGuessRegisterAll()`
@*/
PetscErrorCode KSPGuessRegister(const char sname[], PetscErrorCode (*function)(KSPGuess))
{
  PetscFunctionBegin;
  PetscCall(KSPInitializePackage());
  PetscCall(PetscFunctionListAdd(&KSPGuessList, sname, function));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  KSPGuessRegisterAll - Registers all `KSPGuess` implementations in the `KSP` package.

  Not Collective

  Level: developer

.seealso: [](chapter_ksp), `KSPGuess`, `KSPRegisterAll()`, `KSPInitializePackage()`
@*/
PetscErrorCode KSPGuessRegisterAll(void)
{
  PetscFunctionBegin;
  if (KSPGuessRegisterAllCalled) PetscFunctionReturn(PETSC_SUCCESS);
  KSPGuessRegisterAllCalled = PETSC_TRUE;
  PetscCall(KSPGuessRegister(KSPGUESSFISCHER, KSPGuessCreate_Fischer));
  PetscCall(KSPGuessRegister(KSPGUESSPOD, KSPGuessCreate_POD));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    KSPGuessSetFromOptions - Sets the options for a `KSPGuess` from the options database

    Collective

    Input Parameter:
.    guess - `KSPGuess` object

   Level: developer

.seealso: [](chapter_ksp), `KSPGuess`, `KSPGetGuess()`, `KSPSetGuessType()`, `KSPGuessType`
@*/
PetscErrorCode KSPGuessSetFromOptions(KSPGuess guess)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(guess, KSPGUESS_CLASSID, 1);
  PetscTryTypeMethod(guess, setfromoptions);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    KSPGuessSetTolerance - Sets the relative tolerance used in either eigenvalue (POD) or singular value (Fischer type 3) calculations.
    Ignored by the first and second Fischer types.

    Collective

    Input Parameter:
.    guess - `KSPGuess` object

   Level: developer

.seealso: [](chapter_ksp), `KSPGuess`, `KSPGuessType`, `KSPGuessSetFromOptions()`
@*/
PetscErrorCode KSPGuessSetTolerance(KSPGuess guess, PetscReal tol)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(guess, KSPGUESS_CLASSID, 1);
  PetscTryTypeMethod(guess, settolerance, tol);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   KSPGuessDestroy - Destroys `KSPGuess` context.

   Collective

   Input Parameter:
.  guess - initial guess object

   Level: developer

.seealso: [](chapter_ksp), `KSPGuessCreate()`, `KSPGuess`, `KSPGuessType`
@*/
PetscErrorCode KSPGuessDestroy(KSPGuess *guess)
{
  PetscFunctionBegin;
  if (!*guess) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific((*guess), KSPGUESS_CLASSID, 1);
  if (--((PetscObject)(*guess))->refct > 0) {
    *guess = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscTryTypeMethod((*guess), destroy);
  PetscCall(MatDestroy(&(*guess)->A));
  PetscCall(PetscHeaderDestroy(guess));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   KSPGuessView - View the `KSPGuess` object

   Logically Collective

   Input Parameters:
+  guess  - the initial guess object for the Krylov method
-  viewer - the viewer object

  Level: developer

.seealso: [](chapter_ksp), `KSP`, `KSPGuess`, `KSPGuessType`, `KSPGuessRegister()`, `KSPGuessCreate()`, `PetscViewer`
@*/
PetscErrorCode KSPGuessView(KSPGuess guess, PetscViewer view)
{
  PetscBool ascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(guess, KSPGUESS_CLASSID, 1);
  if (!view) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)guess), &view));
  PetscValidHeaderSpecific(view, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(guess, 1, view, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)view, PETSCVIEWERASCII, &ascii));
  if (ascii) {
    PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)guess, view));
    PetscCall(PetscViewerASCIIPushTab(view));
    PetscTryTypeMethod(guess, view, view);
    PetscCall(PetscViewerASCIIPopTab(view));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   KSPGuessCreate - Creates the default `KSPGuess` context.

   Collective

   Input Parameter:
.  comm - MPI communicator

   Output Parameter:
.  guess - location to put the `KSPGuess` context

   Level: developer

.seealso: [](chapter_ksp), `KSPSolve()`, `KSPGuessDestroy()`, `KSPGuess`, `KSPGuessType`, `KSP`
@*/
PetscErrorCode KSPGuessCreate(MPI_Comm comm, KSPGuess *guess)
{
  KSPGuess tguess;

  PetscFunctionBegin;
  PetscValidPointer(guess, 2);
  *guess = NULL;
  PetscCall(KSPInitializePackage());
  PetscCall(PetscHeaderCreate(tguess, KSPGUESS_CLASSID, "KSPGuess", "Initial guess for Krylov Method", "KSPGuess", comm, KSPGuessDestroy, KSPGuessView));
  tguess->omatstate = -1;
  *guess            = tguess;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   KSPGuessSetType - Sets the type of a `KSPGuess`

   Logically Collective

   Input Parameters:
+  guess - the initial guess object for the Krylov method
-  type  - a known `KSPGuessType`

   Options Database Key:
.  -ksp_guess_type  <method> - Sets the method; use -help for a list of available methods

  Level: developer

.seealso: [](chapter_ksp), `KSP`, `KSPGuess`, `KSPGuessType`, `KSPGuessRegister()`, `KSPGuessCreate()`
@*/
PetscErrorCode KSPGuessSetType(KSPGuess guess, KSPGuessType type)
{
  PetscBool match;
  PetscErrorCode (*r)(KSPGuess);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(guess, KSPGUESS_CLASSID, 1);
  PetscValidCharPointer(type, 2);

  PetscCall(PetscObjectTypeCompare((PetscObject)guess, type, &match));
  if (match) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscFunctionListFind(KSPGuessList, type, &r));
  PetscCheck(r, PetscObjectComm((PetscObject)guess), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unable to find requested KSPGuess type %s", type);
  PetscTryTypeMethod(guess, destroy);
  guess->ops->destroy = NULL;

  PetscCall(PetscMemzero(guess->ops, sizeof(struct _KSPGuessOps)));
  PetscCall(PetscObjectChangeTypeName((PetscObject)guess, type));
  PetscCall((*r)(guess));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   KSPGuessGetType - Gets the `KSPGuessType` as a string from the `KSPGuess` object.

   Not Collective

   Input Parameter:
.  guess - the initial guess context

   Output Parameter:
.  name - type of `KSPGuess` method

   Level: developer

.seealso: [](chapter_ksp), `KSPGuess`, `KSPGuessSetType()`
@*/
PetscErrorCode KSPGuessGetType(KSPGuess guess, KSPGuessType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(guess, KSPGUESS_CLASSID, 1);
  PetscValidPointer(type, 2);
  *type = ((PetscObject)guess)->type_name;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    KSPGuessUpdate - Updates the guess object with the current solution and rhs vector

   Collective

   Input Parameters:
+  guess - the initial guess context
.  rhs   - the corresponding rhs
-  sol   - the computed solution

   Level: developer

.seealso: [](chapter_ksp), `KSPGuessCreate()`, `KSPGuess`
@*/
PetscErrorCode KSPGuessUpdate(KSPGuess guess, Vec rhs, Vec sol)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(guess, KSPGUESS_CLASSID, 1);
  PetscValidHeaderSpecific(rhs, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(sol, VEC_CLASSID, 3);
  PetscTryTypeMethod(guess, update, rhs, sol);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    KSPGuessFormGuess - Form the initial guess

   Collective

   Input Parameters:
+  guess - the initial guess context
.  rhs   - the current rhs vector
-  sol   - the initial guess vector

   Level: developer

.seealso: [](chapter_ksp), `KSPGuessCreate()`, `KSPGuess`
@*/
PetscErrorCode KSPGuessFormGuess(KSPGuess guess, Vec rhs, Vec sol)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(guess, KSPGUESS_CLASSID, 1);
  PetscValidHeaderSpecific(rhs, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(sol, VEC_CLASSID, 3);
  PetscTryTypeMethod(guess, formguess, rhs, sol);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    KSPGuessSetUp - Setup the initial guess object

   Collective

   Input Parameter:
-  guess - the initial guess context

   Level: developer

.seealso: [](chapter_ksp), `KSPGuessCreate()`, `KSPGuess`
@*/
PetscErrorCode KSPGuessSetUp(KSPGuess guess)
{
  PetscObjectState matstate;
  PetscInt         oM = 0, oN = 0, M, N;
  Mat              omat = NULL;
  PC               pc;
  PetscBool        reuse;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(guess, KSPGUESS_CLASSID, 1);
  if (guess->A) {
    omat = guess->A;
    PetscCall(MatGetSize(guess->A, &oM, &oN));
  }
  PetscCall(KSPGetOperators(guess->ksp, &guess->A, NULL));
  PetscCall(KSPGetPC(guess->ksp, &pc));
  PetscCall(PCGetReusePreconditioner(pc, &reuse));
  PetscCall(PetscObjectReference((PetscObject)guess->A));
  PetscCall(MatGetSize(guess->A, &M, &N));
  PetscCall(PetscObjectStateGet((PetscObject)guess->A, &matstate));
  if (M != oM || N != oN) {
    PetscCall(PetscInfo(guess, "Resetting KSPGuess since matrix sizes have changed (%" PetscInt_FMT " != %" PetscInt_FMT ", %" PetscInt_FMT " != %" PetscInt_FMT ")\n", oM, M, oN, N));
  } else if (!reuse && (omat != guess->A || guess->omatstate != matstate)) {
    PetscCall(PetscInfo(guess, "Resetting KSPGuess since %s has changed\n", omat != guess->A ? "matrix" : "matrix state"));
    PetscTryTypeMethod(guess, reset);
  } else if (reuse) {
    PetscCall(PetscInfo(guess, "Not resettting KSPGuess since reuse preconditioner has been specified\n"));
  } else {
    PetscCall(PetscInfo(guess, "KSPGuess status unchanged\n"));
  }
  PetscTryTypeMethod(guess, setup);
  guess->omatstate = matstate;
  PetscCall(MatDestroy(&omat));
  PetscFunctionReturn(PETSC_SUCCESS);
}
