
#include <petsc/private/snesimpl.h> /*I "petscsnes.h"  I*/

/*@
   SNESApplyNPC - Calls `SNESSolve()` on preconditioner for the `SNES`

   Collective

   Input Parameters:
+  snes - the `SNES` context
.  x - input vector
-  f - optional; the function evaluation on `x`

   Output Parameter:
.  y - function vector, as set by `SNESSetFunction()`

   Level: developer

   Note:
   `SNESComputeFunction()` should be called on `x` before `SNESApplyNPC()` is called, as it is
   with `SNESComuteJacobian()`.

.seealso: `SNESGetNPC()`, `SNESSetNPC()`, `SNESComputeFunction()`
@*/
PetscErrorCode SNESApplyNPC(SNES snes, Vec x, Vec f, Vec y)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 4);
  PetscCheckSameComm(snes, 1, x, 2);
  PetscCheckSameComm(snes, 1, y, 4);
  PetscCall(VecValidValues_Internal(x, 2, PETSC_TRUE));
  if (snes->npc) {
    if (f) PetscCall(SNESSetInitialFunction(snes->npc, f));
    PetscCall(VecCopy(x, y));
    PetscCall(PetscLogEventBegin(SNES_NPCSolve, snes->npc, x, y, 0));
    PetscCall(SNESSolve(snes->npc, snes->vec_rhs, y));
    PetscCall(PetscLogEventEnd(SNES_NPCSolve, snes->npc, x, y, 0));
    PetscCall(VecAYPX(y, -1.0, x));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SNESComputeFunctionDefaultNPC(SNES snes, Vec X, Vec F)
{
  /* This is to be used as an argument to SNESMF -- NOT as a "function" */
  SNESConvergedReason reason;

  PetscFunctionBegin;
  if (snes->npc) {
    PetscCall(SNESApplyNPC(snes, X, NULL, F));
    PetscCall(SNESGetConvergedReason(snes->npc, &reason));
    if (reason < 0 && reason != SNES_DIVERGED_MAX_IT) PetscCall(SNESSetFunctionDomainError(snes));
  } else {
    PetscCall(SNESComputeFunction(snes, X, F));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   SNESGetNPCFunction - Gets the current function value and its norm from a nonlinear preconditioner after `SNESSolve()` has been called on that `SNES`

   Collective

   Input Parameter:
.  snes - the `SNES` context

   Output Parameters:
+  F - function vector
-  fnorm - the norm of `F`

   Level: developer

.seealso: `SNESGetNPC()`, `SNESSetNPC()`, `SNESComputeFunction()`, `SNESApplyNPC()`, `SNESSolve()`
@*/
PetscErrorCode SNESGetNPCFunction(SNES snes, Vec F, PetscReal *fnorm)
{
  PCSide           npcside;
  SNESFunctionType functype;
  SNESNormSchedule normschedule;
  Vec              FPC, XPC;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  if (fnorm) PetscValidRealPointer(fnorm, 3);
  PetscCheck(snes->npc, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "No nonlinear preconditioner set");
  PetscCall(SNESGetNPCSide(snes->npc, &npcside));
  PetscCall(SNESGetFunctionType(snes->npc, &functype));
  PetscCall(SNESGetNormSchedule(snes->npc, &normschedule));

  /* check if the function is valid based upon how the inner solver is preconditioned */
  if (normschedule != SNES_NORM_NONE && normschedule != SNES_NORM_INITIAL_ONLY && (npcside == PC_RIGHT || functype == SNES_FUNCTION_UNPRECONDITIONED)) {
    PetscCall(SNESGetFunction(snes->npc, &FPC, NULL, NULL));
    PetscCheck(FPC, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Nonlinear preconditioner has no function");
    if (fnorm) PetscCall(VecNorm(FPC, NORM_2, fnorm));
    PetscCall(VecCopy(FPC, F));
  } else {
    PetscCall(SNESGetSolution(snes->npc, &XPC));
    PetscCheck(XPC, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Nonlinear preconditioner has no solution");
    PetscCall(SNESComputeFunction(snes->npc, XPC, F));
    if (fnorm) PetscCall(VecNorm(F, NORM_2, fnorm));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
