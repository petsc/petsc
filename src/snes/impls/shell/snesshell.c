#include <petsc/private/snesimpl.h> /*I   "petscsnes.h"   I*/

typedef struct {
  PetscErrorCode (*solve)(SNES, Vec);
  void *ctx;
} SNES_Shell;

/*@C
  SNESShellSetSolve - Sets routine to apply as solver to a `SNESSHELL` `SNES` object

  Logically Collective

  Input Parameters:
+ snes  - the `SNES` nonlinear solver context
- solve - the application-provided solver routine

  Calling sequence of `apply`:
+ snes - the preconditioner, get the application context with `SNESShellGetContext()` provided with `SNESShellSetContext()`
- xout - solution vector

  Level: advanced

.seealso: [](ch_snes), `SNES`, `SNESSHELL`, `SNESShellSetContext()`, `SNESShellGetContext()`
@*/
PetscErrorCode SNESShellSetSolve(SNES snes, PetscErrorCode (*solve)(SNES snes, Vec xout))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscTryMethod(snes, "SNESShellSetSolve_C", (SNES, PetscErrorCode (*)(SNES, Vec)), (snes, solve));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESDestroy_Shell(SNES snes)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(snes->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SNESShellGetContext - Returns the user-provided context associated with a `SNESSHELL`

  Not Collective

  Input Parameter:
. snes - should have been created with `SNESSetType`(snes,`SNESSHELL`);

  Output Parameter:
. ctx - the user provided context

  Level: advanced

.seealso: [](ch_snes), `SNES`, `SNESSHELL`, `SNESCreateShell()`, `SNESShellSetContext()`
@*/
PetscErrorCode SNESShellGetContext(SNES snes, void *ctx)
{
  PetscBool flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscAssertPointer(ctx, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)snes, SNESSHELL, &flg));
  if (!flg) *(void **)ctx = NULL;
  else *(void **)ctx = ((SNES_Shell *)snes->data)->ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SNESShellSetContext - sets the context for a `SNESSHELL`

  Logically Collective

  Input Parameters:
+ snes - the `SNESSHELL`
- ctx  - the context

  Level: advanced

.seealso: [](ch_snes), `SNES`, `SNESSHELL`, `SNESCreateShell()`, `SNESShellGetContext()`
@*/
PetscErrorCode SNESShellSetContext(SNES snes, void *ctx)
{
  SNES_Shell *shell = (SNES_Shell *)snes->data;
  PetscBool   flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscCall(PetscObjectTypeCompare((PetscObject)snes, SNESSHELL, &flg));
  if (flg) shell->ctx = ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESSolve_Shell(SNES snes)
{
  SNES_Shell *shell = (SNES_Shell *)snes->data;

  PetscFunctionBegin;
  PetscCheck(shell->solve, PetscObjectComm((PetscObject)snes), PETSC_ERR_ARG_WRONGSTATE, "Must call SNESShellSetSolve() first");
  snes->reason = SNES_CONVERGED_ITS;
  PetscCall((*shell->solve)(snes, snes->vec_sol));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESShellSetSolve_Shell(SNES snes, PetscErrorCode (*solve)(SNES, Vec))
{
  SNES_Shell *shell = (SNES_Shell *)snes->data;

  PetscFunctionBegin;
  shell->solve = solve;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  SNESSHELL - a user provided nonlinear solver

   Level: advanced

.seealso: [](ch_snes), `SNESCreate()`, `SNES`, `SNESSetType()`, `SNESType`, `SNESShellGetContext()`, `SNESShellSetContext()`, `SNESShellSetSolve()`
M*/

PETSC_EXTERN PetscErrorCode SNESCreate_Shell(SNES snes)
{
  SNES_Shell *shell;

  PetscFunctionBegin;
  snes->ops->destroy = SNESDestroy_Shell;
  snes->ops->solve   = SNESSolve_Shell;

  snes->usesksp = PETSC_FALSE;
  snes->usesnpc = PETSC_FALSE;

  snes->alwayscomputesfinalresidual = PETSC_FALSE;

  PetscCall(SNESParametersInitialize(snes));

  PetscCall(PetscNew(&shell));
  snes->data = (void *)shell;
  PetscCall(PetscObjectComposeFunction((PetscObject)snes, "SNESShellSetSolve_C", SNESShellSetSolve_Shell));
  PetscFunctionReturn(PETSC_SUCCESS);
}
