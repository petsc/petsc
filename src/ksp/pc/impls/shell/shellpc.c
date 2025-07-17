/*
   This provides a simple shell for Fortran (and C programmers) to
  create their own preconditioner without writing much interface code.
*/

#include <petsc/private/pcimpl.h> /*I "petscpc.h" I*/

typedef struct {
  void *ctx; /* user provided contexts for preconditioner */

  PetscErrorCode (*destroy)(PC);
  PetscErrorCode (*setup)(PC);
  PetscErrorCode (*apply)(PC, Vec, Vec);
  PetscErrorCode (*matapply)(PC, Mat, Mat);
  PetscErrorCode (*applysymmetricleft)(PC, Vec, Vec);
  PetscErrorCode (*applysymmetricright)(PC, Vec, Vec);
  PetscErrorCode (*applyBA)(PC, PCSide, Vec, Vec, Vec);
  PetscErrorCode (*presolve)(PC, KSP, Vec, Vec);
  PetscErrorCode (*postsolve)(PC, KSP, Vec, Vec);
  PetscErrorCode (*view)(PC, PetscViewer);
  PetscErrorCode (*applytranspose)(PC, Vec, Vec);
  PetscErrorCode (*matapplytranspose)(PC, Mat, Mat);
  PetscErrorCode (*applyrich)(PC, Vec, Vec, Vec, PetscReal, PetscReal, PetscReal, PetscInt, PetscBool, PetscInt *, PCRichardsonConvergedReason *);

  char *name;
} PC_Shell;

/*@C
  PCShellGetContext - Returns the user-provided context associated with a shell `PC` that was provided with `PCShellSetContext()`

  Not Collective

  Input Parameter:
. pc - of type `PCSHELL`

  Output Parameter:
. ctx - the user provided context

  Level: advanced

  Note:
  This routine is intended for use within the various user-provided routines set with, for example, `PCShellSetApply()`

  Fortran Note:
  To use this from Fortran you must write a Fortran interface definition for this
  function that tells Fortran the Fortran derived data type that you are passing in as the `ctx` argument.

.seealso: [](ch_ksp), `PC`, `PCSHELL`, `PCShellSetContext()`, `PCShellSetApply()`, `PCShellSetDestroy()`
@*/
PetscErrorCode PCShellGetContext(PC pc, void *ctx)
{
  PetscBool flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscAssertPointer(ctx, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)pc, PCSHELL, &flg));
  if (!flg) *(void **)ctx = NULL;
  else *(void **)ctx = ((PC_Shell *)pc->data)->ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCShellSetContext - sets the context for a shell `PC` that can be accessed with `PCShellGetContext()`

  Logically Collective

  Input Parameters:
+ pc  - the `PC` of type `PCSHELL`
- ctx - the context

  Level: advanced

  Notes:
  This routine is intended for use within the various user-provided routines set with, for example, `PCShellSetApply()`

  One should also provide a routine to destroy the context when `pc` is destroyed with `PCShellSetDestroy()`

  Fortran Notes:
  To use this from Fortran you must write a Fortran interface definition for this
  function that tells Fortran the Fortran derived data type that you are passing in as the `ctx` argument.

.seealso: [](ch_ksp), `PC`, `PCShellGetContext()`, `PCSHELL`, `PCShellSetApply()`, `PCShellSetDestroy()`
@*/
PetscErrorCode PCShellSetContext(PC pc, void *ctx)
{
  PC_Shell *shell = (PC_Shell *)pc->data;
  PetscBool flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscCall(PetscObjectTypeCompare((PetscObject)pc, PCSHELL, &flg));
  if (flg) shell->ctx = ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetUp_Shell(PC pc)
{
  PC_Shell *shell = (PC_Shell *)pc->data;

  PetscFunctionBegin;
  PetscCheck(shell->setup, PetscObjectComm((PetscObject)pc), PETSC_ERR_USER, "No setup() routine provided to Shell PC");
  PetscCallBack("PCSHELL callback setup", (*shell->setup)(pc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApply_Shell(PC pc, Vec x, Vec y)
{
  PC_Shell        *shell = (PC_Shell *)pc->data;
  PetscObjectState instate, outstate;

  PetscFunctionBegin;
  PetscCheck(shell->apply, PetscObjectComm((PetscObject)pc), PETSC_ERR_USER, "No apply() routine provided to Shell PC");
  PetscCall(PetscObjectStateGet((PetscObject)y, &instate));
  PetscCallBack("PCSHELL callback apply", (*shell->apply)(pc, x, y));
  PetscCall(PetscObjectStateGet((PetscObject)y, &outstate));
  /* increase the state of the output vector if the user did not update its state themself as should have been done */
  if (instate == outstate) PetscCall(PetscObjectStateIncrease((PetscObject)y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCMatApply_Shell(PC pc, Mat X, Mat Y)
{
  PC_Shell        *shell = (PC_Shell *)pc->data;
  PetscObjectState instate, outstate;

  PetscFunctionBegin;
  PetscCheck(shell->matapply, PetscObjectComm((PetscObject)pc), PETSC_ERR_USER, "No apply() routine provided to Shell PC");
  PetscCall(PetscObjectStateGet((PetscObject)Y, &instate));
  PetscCallBack("PCSHELL callback apply", (*shell->matapply)(pc, X, Y));
  PetscCall(PetscObjectStateGet((PetscObject)Y, &outstate));
  /* increase the state of the output vector if the user did not update its state themself as should have been done */
  if (instate == outstate) PetscCall(PetscObjectStateIncrease((PetscObject)Y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApplySymmetricLeft_Shell(PC pc, Vec x, Vec y)
{
  PC_Shell *shell = (PC_Shell *)pc->data;

  PetscFunctionBegin;
  PetscCheck(shell->applysymmetricleft, PetscObjectComm((PetscObject)pc), PETSC_ERR_USER, "No apply() routine provided to Shell PC");
  PetscCallBack("PCSHELL callback apply symmetric left", (*shell->applysymmetricleft)(pc, x, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApplySymmetricRight_Shell(PC pc, Vec x, Vec y)
{
  PC_Shell *shell = (PC_Shell *)pc->data;

  PetscFunctionBegin;
  PetscCheck(shell->applysymmetricright, PetscObjectComm((PetscObject)pc), PETSC_ERR_USER, "No apply() routine provided to Shell PC");
  PetscCallBack("PCSHELL callback apply symmetric right", (*shell->applysymmetricright)(pc, x, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApplyBA_Shell(PC pc, PCSide side, Vec x, Vec y, Vec w)
{
  PC_Shell        *shell = (PC_Shell *)pc->data;
  PetscObjectState instate, outstate;

  PetscFunctionBegin;
  PetscCheck(shell->applyBA, PetscObjectComm((PetscObject)pc), PETSC_ERR_USER, "No applyBA() routine provided to Shell PC");
  PetscCall(PetscObjectStateGet((PetscObject)w, &instate));
  PetscCallBack("PCSHELL callback applyBA", (*shell->applyBA)(pc, side, x, y, w));
  PetscCall(PetscObjectStateGet((PetscObject)w, &outstate));
  /* increase the state of the output vector if the user did not update its state themself as should have been done */
  if (instate == outstate) PetscCall(PetscObjectStateIncrease((PetscObject)w));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCPreSolveChangeRHS_Shell(PC pc, PetscBool *change)
{
  PetscFunctionBegin;
  *change = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCPreSolve_Shell(PC pc, KSP ksp, Vec b, Vec x)
{
  PC_Shell *shell = (PC_Shell *)pc->data;

  PetscFunctionBegin;
  PetscCheck(shell->presolve, PetscObjectComm((PetscObject)pc), PETSC_ERR_USER, "No presolve() routine provided to Shell PC");
  PetscCallBack("PCSHELL callback presolve", (*shell->presolve)(pc, ksp, b, x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCPostSolve_Shell(PC pc, KSP ksp, Vec b, Vec x)
{
  PC_Shell *shell = (PC_Shell *)pc->data;

  PetscFunctionBegin;
  PetscCheck(shell->postsolve, PetscObjectComm((PetscObject)pc), PETSC_ERR_USER, "No postsolve() routine provided to Shell PC");
  PetscCallBack("PCSHELL callback postsolve()", (*shell->postsolve)(pc, ksp, b, x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApplyTranspose_Shell(PC pc, Vec x, Vec y)
{
  PC_Shell        *shell = (PC_Shell *)pc->data;
  PetscObjectState instate, outstate;

  PetscFunctionBegin;
  PetscCheck(shell->applytranspose, PetscObjectComm((PetscObject)pc), PETSC_ERR_USER, "No applytranspose() routine provided to Shell PC");
  PetscCall(PetscObjectStateGet((PetscObject)y, &instate));
  PetscCallBack("PCSHELL callback applytranspose", (*shell->applytranspose)(pc, x, y));
  PetscCall(PetscObjectStateGet((PetscObject)y, &outstate));
  /* increase the state of the output vector if the user did not update its state themself as should have been done */
  if (instate == outstate) PetscCall(PetscObjectStateIncrease((PetscObject)y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCMatApplyTranspose_Shell(PC pc, Mat x, Mat y)
{
  PC_Shell        *shell = (PC_Shell *)pc->data;
  PetscObjectState instate, outstate;

  PetscFunctionBegin;
  PetscCheck(shell->matapplytranspose, PetscObjectComm((PetscObject)pc), PETSC_ERR_USER, "No matapplytranspose() routine provided to Shell PC");
  PetscCall(PetscObjectStateGet((PetscObject)y, &instate));
  PetscCallBack("PCSHELL callback matapplytranspose", (*shell->matapplytranspose)(pc, x, y));
  PetscCall(PetscObjectStateGet((PetscObject)y, &outstate));
  /* increase the state of the output matrix if the user did not update its state themself as should have been done */
  if (instate == outstate) PetscCall(PetscObjectStateIncrease((PetscObject)y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApplyRichardson_Shell(PC pc, Vec x, Vec y, Vec w, PetscReal rtol, PetscReal abstol, PetscReal dtol, PetscInt it, PetscBool guesszero, PetscInt *outits, PCRichardsonConvergedReason *reason)
{
  PC_Shell        *shell = (PC_Shell *)pc->data;
  PetscObjectState instate, outstate;

  PetscFunctionBegin;
  PetscCheck(shell->applyrich, PetscObjectComm((PetscObject)pc), PETSC_ERR_USER, "No applyrichardson() routine provided to Shell PC");
  PetscCall(PetscObjectStateGet((PetscObject)y, &instate));
  PetscCallBack("PCSHELL callback applyrichardson", (*shell->applyrich)(pc, x, y, w, rtol, abstol, dtol, it, guesszero, outits, reason));
  PetscCall(PetscObjectStateGet((PetscObject)y, &outstate));
  /* increase the state of the output vector since the user did not update its state themself as should have been done */
  if (instate == outstate) PetscCall(PetscObjectStateIncrease((PetscObject)y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCDestroy_Shell(PC pc)
{
  PC_Shell *shell = (PC_Shell *)pc->data;

  PetscFunctionBegin;
  PetscCall(PetscFree(shell->name));
  if (shell->destroy) PetscCallBack("PCSHELL callback destroy", (*shell->destroy)(pc));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCShellSetDestroy_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCShellSetSetUp_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCShellSetApply_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCShellSetMatApply_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCShellSetApplySymmetricLeft_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCShellSetApplySymmetricRight_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCShellSetApplyBA_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCShellSetPreSolve_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCShellSetPostSolve_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCShellSetView_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCShellSetApplyTranspose_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCShellSetMatApplyTranspose_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCShellSetName_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCShellGetName_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCShellSetApplyRichardson_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCPreSolveChangeRHS_C", NULL));
  PetscCall(PetscFree(pc->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCView_Shell(PC pc, PetscViewer viewer)
{
  PC_Shell *shell = (PC_Shell *)pc->data;
  PetscBool isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    if (shell->name) PetscCall(PetscViewerASCIIPrintf(viewer, "  %s\n", shell->name));
    else PetscCall(PetscViewerASCIIPrintf(viewer, "  no name\n"));
  }
  if (shell->view) {
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall((*shell->view)(pc, viewer));
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCShellSetDestroy_Shell(PC pc, PetscErrorCode (*destroy)(PC))
{
  PC_Shell *shell = (PC_Shell *)pc->data;

  PetscFunctionBegin;
  shell->destroy = destroy;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCShellSetSetUp_Shell(PC pc, PetscErrorCode (*setup)(PC))
{
  PC_Shell *shell = (PC_Shell *)pc->data;

  PetscFunctionBegin;
  shell->setup = setup;
  if (setup) pc->ops->setup = PCSetUp_Shell;
  else pc->ops->setup = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCShellSetApply_Shell(PC pc, PetscErrorCode (*apply)(PC, Vec, Vec))
{
  PC_Shell *shell = (PC_Shell *)pc->data;

  PetscFunctionBegin;
  shell->apply = apply;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCShellSetMatApply_Shell(PC pc, PetscErrorCode (*matapply)(PC, Mat, Mat))
{
  PC_Shell *shell = (PC_Shell *)pc->data;

  PetscFunctionBegin;
  shell->matapply = matapply;
  if (matapply) pc->ops->matapply = PCMatApply_Shell;
  else pc->ops->matapply = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCShellSetApplySymmetricLeft_Shell(PC pc, PetscErrorCode (*apply)(PC, Vec, Vec))
{
  PC_Shell *shell = (PC_Shell *)pc->data;

  PetscFunctionBegin;
  shell->applysymmetricleft = apply;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCShellSetApplySymmetricRight_Shell(PC pc, PetscErrorCode (*apply)(PC, Vec, Vec))
{
  PC_Shell *shell = (PC_Shell *)pc->data;

  PetscFunctionBegin;
  shell->applysymmetricright = apply;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCShellSetApplyBA_Shell(PC pc, PetscErrorCode (*applyBA)(PC, PCSide, Vec, Vec, Vec))
{
  PC_Shell *shell = (PC_Shell *)pc->data;

  PetscFunctionBegin;
  shell->applyBA = applyBA;
  if (applyBA) pc->ops->applyBA = PCApplyBA_Shell;
  else pc->ops->applyBA = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCShellSetPreSolve_Shell(PC pc, PCShellPSolveFn *presolve)
{
  PC_Shell *shell = (PC_Shell *)pc->data;

  PetscFunctionBegin;
  shell->presolve = presolve;
  if (presolve) {
    pc->ops->presolve = PCPreSolve_Shell;
    PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCPreSolveChangeRHS_C", PCPreSolveChangeRHS_Shell));
  } else {
    pc->ops->presolve = NULL;
    PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCPreSolveChangeRHS_C", NULL));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCShellSetPostSolve_Shell(PC pc, PCShellPSolveFn *postsolve)
{
  PC_Shell *shell = (PC_Shell *)pc->data;

  PetscFunctionBegin;
  shell->postsolve = postsolve;
  if (postsolve) pc->ops->postsolve = PCPostSolve_Shell;
  else pc->ops->postsolve = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCShellSetView_Shell(PC pc, PetscErrorCode (*view)(PC, PetscViewer))
{
  PC_Shell *shell = (PC_Shell *)pc->data;

  PetscFunctionBegin;
  shell->view = view;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCShellSetApplyTranspose_Shell(PC pc, PetscErrorCode (*applytranspose)(PC, Vec, Vec))
{
  PC_Shell *shell = (PC_Shell *)pc->data;

  PetscFunctionBegin;
  shell->applytranspose = applytranspose;
  if (applytranspose) pc->ops->applytranspose = PCApplyTranspose_Shell;
  else pc->ops->applytranspose = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCShellSetMatApplyTranspose_Shell(PC pc, PetscErrorCode (*matapplytranspose)(PC, Mat, Mat))
{
  PC_Shell *shell = (PC_Shell *)pc->data;

  PetscFunctionBegin;
  shell->matapplytranspose = matapplytranspose;
  if (matapplytranspose) pc->ops->matapplytranspose = PCMatApplyTranspose_Shell;
  else pc->ops->matapplytranspose = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCShellSetApplyRichardson_Shell(PC pc, PetscErrorCode (*applyrich)(PC, Vec, Vec, Vec, PetscReal, PetscReal, PetscReal, PetscInt, PetscBool, PetscInt *, PCRichardsonConvergedReason *))
{
  PC_Shell *shell = (PC_Shell *)pc->data;

  PetscFunctionBegin;
  shell->applyrich = applyrich;
  if (applyrich) pc->ops->applyrichardson = PCApplyRichardson_Shell;
  else pc->ops->applyrichardson = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCShellSetName_Shell(PC pc, const char name[])
{
  PC_Shell *shell = (PC_Shell *)pc->data;

  PetscFunctionBegin;
  PetscCall(PetscFree(shell->name));
  PetscCall(PetscStrallocpy(name, &shell->name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCShellGetName_Shell(PC pc, const char *name[])
{
  PC_Shell *shell = (PC_Shell *)pc->data;

  PetscFunctionBegin;
  *name = shell->name;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PCShellSetDestroy - Sets routine to use to destroy the user-provided application context that was provided with `PCShellSetContext()`

  Logically Collective

  Input Parameters:
+ pc      - the preconditioner context
- destroy - the application-provided destroy routine

  Calling sequence of `destroy`:
. pc - the preconditioner

  Level: intermediate

.seealso: [](ch_ksp), `PCSHELL`, `PCShellSetApply()`, `PCShellSetContext()`, `PCShellGetContext()`
@*/
PetscErrorCode PCShellSetDestroy(PC pc, PetscErrorCode (*destroy)(PC pc))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscTryMethod(pc, "PCShellSetDestroy_C", (PC, PetscErrorCode (*)(PC)), (pc, destroy));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PCShellSetSetUp - Sets routine to use to "setup" the preconditioner whenever the
  matrix operator is changed.

  Logically Collective

  Input Parameters:
+ pc    - the preconditioner context
- setup - the application-provided setup routine

  Calling sequence of `setup`:
. pc - the preconditioner

  Level: intermediate

  Note:
  You can get the `PCSHELL` context set with `PCShellSetContext()` using `PCShellGetContext()` if needed by `setup`.

.seealso: [](ch_ksp), `PCSHELL`, `PCShellSetApplyRichardson()`, `PCShellSetApply()`, `PCShellSetContext()`, , `PCShellGetContext()`
@*/
PetscErrorCode PCShellSetSetUp(PC pc, PetscErrorCode (*setup)(PC pc))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscTryMethod(pc, "PCShellSetSetUp_C", (PC, PetscErrorCode (*)(PC)), (pc, setup));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PCShellSetView - Sets routine to use as viewer of a `PCSHELL` shell preconditioner

  Logically Collective

  Input Parameters:
+ pc   - the preconditioner context
- view - the application-provided view routine

  Calling sequence of `view`:
+ pc - the preconditioner
- v  - viewer

  Level: advanced

  Note:
  You can get the `PCSHELL` context set with `PCShellSetContext()` using `PCShellGetContext()` if needed by `view`.

.seealso: [](ch_ksp), `PC`, `PCSHELL`, `PCShellSetApplyRichardson()`, `PCShellSetSetUp()`, `PCShellSetApplyTranspose()`, `PCShellSetContext()`, `PCShellGetContext()`
@*/
PetscErrorCode PCShellSetView(PC pc, PetscErrorCode (*view)(PC pc, PetscViewer v))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscTryMethod(pc, "PCShellSetView_C", (PC, PetscErrorCode (*)(PC, PetscViewer)), (pc, view));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PCShellSetApply - Sets routine to use as preconditioner.

  Logically Collective

  Input Parameters:
+ pc    - the preconditioner context
- apply - the application-provided preconditioning routine

  Calling sequence of `apply`:
+ pc   - the preconditioner, get the application context with `PCShellGetContext()`
. xin  - input vector
- xout - output vector

  Level: intermediate

  Note:
  You can get the `PCSHELL` context set with `PCShellSetContext()` using `PCShellGetContext()` if needed by `apply`.

.seealso: [](ch_ksp), `PCSHELL`, `PCShellSetApplyRichardson()`, `PCShellSetSetUp()`, `PCShellSetApplyTranspose()`, `PCShellSetContext()`, `PCShellSetApplyBA()`, `PCShellSetApplySymmetricRight()`, `PCShellSetApplySymmetricLeft()`, `PCShellGetContext()`
@*/
PetscErrorCode PCShellSetApply(PC pc, PetscErrorCode (*apply)(PC pc, Vec xin, Vec xout))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscTryMethod(pc, "PCShellSetApply_C", (PC, PetscErrorCode (*)(PC, Vec, Vec)), (pc, apply));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PCShellSetMatApply - Sets routine to use as preconditioner on a block of vectors.

  Logically Collective

  Input Parameters:
+ pc       - the preconditioner context
- matapply - the application-provided preconditioning routine

  Calling sequence of `matapply`:
+ pc   - the preconditioner
. Xin  - input block of vectors represented as a dense `Mat`
- Xout - output block of vectors represented as a dense `Mat`

  Level: advanced

  Note:
  You can get the `PCSHELL` context set with `PCShellSetContext()` using `PCShellGetContext()` if needed by `matapply`.

.seealso: [](ch_ksp), `PCSHELL`, `PCShellSetApply()`, `PCShellSetContext()`, `PCShellGetContext()`
@*/
PetscErrorCode PCShellSetMatApply(PC pc, PetscErrorCode (*matapply)(PC pc, Mat Xin, Mat Xout))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscTryMethod(pc, "PCShellSetMatApply_C", (PC, PetscErrorCode (*)(PC, Mat, Mat)), (pc, matapply));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PCShellSetApplySymmetricLeft - Sets routine to use as left preconditioner (when the `PC_SYMMETRIC` is used).

  Logically Collective

  Input Parameters:
+ pc    - the preconditioner context
- apply - the application-provided left preconditioning routine

  Calling sequence of `apply`:
+ pc   - the preconditioner
. xin  - input vector
- xout - output vector

  Level: advanced

  Note:
  You can get the `PCSHELL` context set with `PCShellSetContext()` using `PCShellGetContext()` if needed by `apply`.

.seealso: [](ch_ksp), `PCSHELL`, `PCShellSetApply()`, `PCShellSetSetUp()`, `PCShellSetApplyTranspose()`, `PCShellSetContext()`
@*/
PetscErrorCode PCShellSetApplySymmetricLeft(PC pc, PetscErrorCode (*apply)(PC pc, Vec xin, Vec xout))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscTryMethod(pc, "PCShellSetApplySymmetricLeft_C", (PC, PetscErrorCode (*)(PC, Vec, Vec)), (pc, apply));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PCShellSetApplySymmetricRight - Sets routine to use as right preconditioner (when the `PC_SYMMETRIC` is used).

  Logically Collective

  Input Parameters:
+ pc    - the preconditioner context
- apply - the application-provided right preconditioning routine

  Calling sequence of `apply`:
+ pc   - the preconditioner
. xin  - input vector
- xout - output vector

  Level: advanced

  Note:
  You can get the `PCSHELL` context set with `PCShellSetContext()` using `PCShellGetContext()` if needed by `apply`.

.seealso: [](ch_ksp), `PCSHELL`, `PCShellSetApply()`, `PCShellSetApplySymmetricLeft()`, `PCShellSetSetUp()`, `PCShellSetApplyTranspose()`, `PCShellSetContext()`, `PCShellGetContext()`
@*/
PetscErrorCode PCShellSetApplySymmetricRight(PC pc, PetscErrorCode (*apply)(PC pc, Vec xin, Vec xout))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscTryMethod(pc, "PCShellSetApplySymmetricRight_C", (PC, PetscErrorCode (*)(PC, Vec, Vec)), (pc, apply));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PCShellSetApplyBA - Sets routine to use as the preconditioner times the operator.

  Logically Collective

  Input Parameters:
+ pc      - the preconditioner context
- applyBA - the application-provided BA routine

  Calling sequence of `applyBA`:
+ pc   - the preconditioner
. side - `PC_LEFT`, `PC_RIGHT`, or `PC_SYMMETRIC`
. xin  - input vector
. xout - output vector
- w    - work vector

  Level: intermediate

  Note:
  You can get the `PCSHELL` context set with `PCShellSetContext()` using `PCShellGetContext()` if needed by `applyBA`.

.seealso: [](ch_ksp), `PCSHELL`, `PCShellSetApplyRichardson()`, `PCShellSetSetUp()`, `PCShellSetApplyTranspose()`, `PCShellSetContext()`, `PCShellSetApply()`, `PCShellGetContext()`, `PCSide`
@*/
PetscErrorCode PCShellSetApplyBA(PC pc, PetscErrorCode (*applyBA)(PC pc, PCSide side, Vec xin, Vec xout, Vec w))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscTryMethod(pc, "PCShellSetApplyBA_C", (PC, PetscErrorCode (*)(PC, PCSide, Vec, Vec, Vec)), (pc, applyBA));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PCShellSetApplyTranspose - Sets routine to use as preconditioner transpose.

  Logically Collective

  Input Parameters:
+ pc             - the preconditioner context
- applytranspose - the application-provided preconditioning transpose routine

  Calling sequence of `applytranspose`:
+ pc   - the preconditioner
. xin  - input vector
- xout - output vector

  Level: intermediate

  Note:
  You can get the `PCSHELL` context set with `PCShellSetContext()` using `PCShellGetContext()` if needed by `applytranspose`.

.seealso: [](ch_ksp), `PCSHELL`, `PCShellSetApplyRichardson()`, `PCShellSetSetUp()`, `PCShellSetApply()`, `PCShellSetContext()`, `PCShellSetApplyBA()`, `PCShellGetContext()`
@*/
PetscErrorCode PCShellSetApplyTranspose(PC pc, PetscErrorCode (*applytranspose)(PC pc, Vec xin, Vec xout))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscTryMethod(pc, "PCShellSetApplyTranspose_C", (PC, PetscErrorCode (*)(PC, Vec, Vec)), (pc, applytranspose));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PCShellSetMatApplyTranspose - Sets routine to use as preconditioner transpose.

  Logically Collective

  Input Parameters:
+ pc                - the preconditioner context
- matapplytranspose - the application-provided preconditioning transpose routine

  Calling sequence of `matapplytranspose`:
+ pc   - the preconditioner
. xin  - input matrix
- xout - output matrix

  Level: intermediate

  Note:
  You can get the `PCSHELL` context set with `PCShellSetContext()` using `PCShellGetContext()` if needed by `matapplytranspose`.

.seealso: [](ch_ksp), `PCSHELL`, `PCShellSetApplyRichardson()`, `PCShellSetSetUp()`, `PCShellSetApply()`, `PCShellSetContext()`, `PCShellSetApplyBA()`, `PCShellGetContext()`
@*/
PetscErrorCode PCShellSetMatApplyTranspose(PC pc, PetscErrorCode (*matapplytranspose)(PC pc, Mat xin, Mat xout))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscTryMethod(pc, "PCShellSetMatApplyTranspose_C", (PC, PetscErrorCode (*)(PC, Mat, Mat)), (pc, matapplytranspose));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PCShellSetPreSolve - Sets routine to apply to the operators/vectors before a `KSPSolve()` is
  applied. This usually does something like scale the linear system in some application
  specific way.

  Logically Collective

  Input Parameters:
+ pc       - the preconditioner context
- presolve - the application-provided presolve routine, see `PCShellPSolveFn`

  Level: advanced

  Note:
  You can get the `PCSHELL` context set with `PCShellSetContext()` using `PCShellGetContext()` if needed by `presolve`.

.seealso: [](ch_ksp), `PCSHELL`, `PCShellPSolveFn`, `PCShellSetApplyRichardson()`, `PCShellSetSetUp()`, `PCShellSetApplyTranspose()`, `PCShellSetPostSolve()`, `PCShellSetContext()`, `PCShellGetContext()`
@*/
PetscErrorCode PCShellSetPreSolve(PC pc, PCShellPSolveFn *presolve)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscTryMethod(pc, "PCShellSetPreSolve_C", (PC, PCShellPSolveFn *), (pc, presolve));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PCShellSetPostSolve - Sets routine to apply to the operators/vectors after a `KSPSolve()` is
  applied. This usually does something like scale the linear system in some application
  specific way.

  Logically Collective

  Input Parameters:
+ pc        - the preconditioner context
- postsolve - the application-provided postsolve routine, see `PCShellPSolveFn`

  Level: advanced

  Note:
  You can get the `PCSHELL` context set with `PCShellSetContext()` using `PCShellGetContext()` if needed by `postsolve`.

.seealso: [](ch_ksp), `PCSHELL`, `PCShellPSolveFn`, `PCShellSetApplyRichardson()`, `PCShellSetSetUp()`, `PCShellSetApplyTranspose()`, `PCShellSetPreSolve()`, `PCShellSetContext()`, `PCShellGetContext()`
@*/
PetscErrorCode PCShellSetPostSolve(PC pc, PCShellPSolveFn *postsolve)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscTryMethod(pc, "PCShellSetPostSolve_C", (PC, PCShellPSolveFn *), (pc, postsolve));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCShellSetName - Sets an optional name to associate with a `PCSHELL`
  preconditioner.

  Not Collective

  Input Parameters:
+ pc   - the preconditioner context
- name - character string describing shell preconditioner

  Level: intermediate

  Note:
  This is separate from the name you can provide with `PetscObjectSetName()`

.seealso: [](ch_ksp), `PCSHELL`, `PCShellGetName()`, `PetscObjectSetName()`, `PetscObjectGetName()`
@*/
PetscErrorCode PCShellSetName(PC pc, const char name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscTryMethod(pc, "PCShellSetName_C", (PC, const char[]), (pc, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCShellGetName - Gets an optional name that the user has set for a `PCSHELL` with `PCShellSetName()`
  preconditioner.

  Not Collective

  Input Parameter:
. pc - the preconditioner context

  Output Parameter:
. name - character string describing shell preconditioner (you should not free this)

  Level: intermediate

.seealso: [](ch_ksp), `PCSHELL`, `PCShellSetName()`, `PetscObjectSetName()`, `PetscObjectGetName()`
@*/
PetscErrorCode PCShellGetName(PC pc, const char *name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscAssertPointer(name, 2);
  PetscUseMethod(pc, "PCShellGetName_C", (PC, const char *[]), (pc, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PCShellSetApplyRichardson - Sets routine to use as preconditioner
  in Richardson iteration.

  Logically Collective

  Input Parameters:
+ pc    - the preconditioner context
- apply - the application-provided preconditioning routine

  Calling sequence of `apply`:
+ pc               - the preconditioner
. b                - right-hand side
. x                - current iterate
. r                - work space
. rtol             - relative tolerance of residual norm to stop at
. abstol           - absolute tolerance of residual norm to stop at
. dtol             - if residual norm increases by this factor than return
. maxits           - number of iterations to run
. zeroinitialguess - `PETSC_TRUE` if `x` is known to be initially zero
. its              - returns the number of iterations used
- reason           - returns the reason the iteration has converged

  Level: advanced

  Notes:
  You can get the `PCSHELL` context set with `PCShellSetContext()` using `PCShellGetContext()` if needed by `apply`.

  This is used when one can provide code for multiple steps of Richardson's method that is more efficient than computing a single step,
  recomputing the residual via $ r = b - A x $, and then computing the next step. SOR is an algorithm for which this is true.

.seealso: [](ch_ksp), `PCSHELL`, `PCShellSetApply()`, `PCShellSetContext()`, `PCRichardsonConvergedReason()`, `PCShellGetContext()`, `KSPRICHARDSON`
@*/
PetscErrorCode PCShellSetApplyRichardson(PC pc, PetscErrorCode (*apply)(PC pc, Vec b, Vec x, Vec r, PetscReal rtol, PetscReal abstol, PetscReal dtol, PetscInt maxits, PetscBool zeroinitialguess, PetscInt *its, PCRichardsonConvergedReason *reason))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscTryMethod(pc, "PCShellSetApplyRichardson_C", (PC, PetscErrorCode (*)(PC, Vec, Vec, Vec, PetscReal, PetscReal, PetscReal, PetscInt, PetscBool, PetscInt *, PCRichardsonConvergedReason *)), (pc, apply));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  PCSHELL - Creates a new preconditioner class for use with a users
            own private data storage format and preconditioner application code

  Level: advanced

  Usage:
.vb
  extern PetscErrorCode apply(PC,Vec,Vec);
  extern PetscErrorCode applyba(PC,PCSide,Vec,Vec,Vec);
  extern PetscErrorCode applytranspose(PC,Vec,Vec);
  extern PetscErrorCode setup(PC);
  extern PetscErrorCode destroy(PC);

  PCCreate(comm,&pc);
  PCSetType(pc,PCSHELL);
  PCShellSetContext(pc,ctx)
  PCShellSetApply(pc,apply);
  PCShellSetApplyBA(pc,applyba);               (optional)
  PCShellSetApplyTranspose(pc,applytranspose); (optional)
  PCShellSetSetUp(pc,setup);                   (optional)
  PCShellSetDestroy(pc,destroy);               (optional)
.ve

  Notes:
  Information required for the preconditioner and its internal datastructures can be set with `PCShellSetContext()` and then accessed
  with `PCShellGetContext()` inside the routines provided above.

  When using `MATSHELL`, where the explicit entries of matrix are not available to build the preconditioner, `PCSHELL` can be used
  to construct a custom preconditioner for the `MATSHELL`, assuming the user knows enough about their problem to provide a
  custom preconditioner.

.seealso: [](ch_ksp), `PCCreate()`, `PCSetType()`, `PCType`, `PC`,
          `MATSHELL`, `PCShellSetSetUp()`, `PCShellSetApply()`, `PCShellSetView()`, `PCShellSetDestroy()`, `PCShellSetPostSolve()`,
          `PCShellSetApplyTranspose()`, `PCShellSetName()`, `PCShellSetApplyRichardson()`, `PCShellSetPreSolve()`, `PCShellSetView()`,
          `PCShellGetName()`, `PCShellSetContext()`, `PCShellGetContext()`, `PCShellSetApplyBA()`, `MATSHELL`, `PCShellSetMatApply()`,
M*/

PETSC_EXTERN PetscErrorCode PCCreate_Shell(PC pc)
{
  PC_Shell *shell;

  PetscFunctionBegin;
  PetscCall(PetscNew(&shell));
  pc->data = (void *)shell;

  pc->ops->destroy             = PCDestroy_Shell;
  pc->ops->view                = PCView_Shell;
  pc->ops->apply               = PCApply_Shell;
  pc->ops->applysymmetricleft  = PCApplySymmetricLeft_Shell;
  pc->ops->applysymmetricright = PCApplySymmetricRight_Shell;
  pc->ops->matapply            = NULL;
  pc->ops->applytranspose      = NULL;
  pc->ops->applyrichardson     = NULL;
  pc->ops->setup               = NULL;
  pc->ops->presolve            = NULL;
  pc->ops->postsolve           = NULL;

  shell->apply               = NULL;
  shell->applytranspose      = NULL;
  shell->name                = NULL;
  shell->applyrich           = NULL;
  shell->presolve            = NULL;
  shell->postsolve           = NULL;
  shell->ctx                 = NULL;
  shell->setup               = NULL;
  shell->view                = NULL;
  shell->destroy             = NULL;
  shell->applysymmetricleft  = NULL;
  shell->applysymmetricright = NULL;

  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCShellSetDestroy_C", PCShellSetDestroy_Shell));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCShellSetSetUp_C", PCShellSetSetUp_Shell));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCShellSetApply_C", PCShellSetApply_Shell));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCShellSetMatApply_C", PCShellSetMatApply_Shell));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCShellSetApplySymmetricLeft_C", PCShellSetApplySymmetricLeft_Shell));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCShellSetApplySymmetricRight_C", PCShellSetApplySymmetricRight_Shell));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCShellSetApplyBA_C", PCShellSetApplyBA_Shell));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCShellSetPreSolve_C", PCShellSetPreSolve_Shell));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCShellSetPostSolve_C", PCShellSetPostSolve_Shell));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCShellSetView_C", PCShellSetView_Shell));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCShellSetApplyTranspose_C", PCShellSetApplyTranspose_Shell));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCShellSetMatApplyTranspose_C", PCShellSetMatApplyTranspose_Shell));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCShellSetName_C", PCShellSetName_Shell));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCShellGetName_C", PCShellGetName_Shell));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCShellSetApplyRichardson_C", PCShellSetApplyRichardson_Shell));
  PetscFunctionReturn(PETSC_SUCCESS);
}
