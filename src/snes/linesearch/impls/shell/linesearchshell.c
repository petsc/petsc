#include <petsc/private/linesearchimpl.h>
#include <petsc/private/snesimpl.h>

typedef struct {
  SNESLineSearchShellApplyFn *func;
  void                       *ctx;
} SNESLineSearch_Shell;

// PetscClangLinter pragma disable: -fdoc-param-list-func-parameter-documentation
/*@C
  SNESLineSearchShellSetApply - Sets the apply function for the `SNESLINESEARCHSHELL` implementation.

  Not Collective

  Input Parameters:
+ linesearch - `SNESLineSearch` context
. func       - function implementing the linesearch shell, see `SNESLineSearchShellApplyFn` for calling sequence
- ctx        - context for func

  Usage\:
.vb
  PetscErrorCode shellfunc(SNESLineSearch linesearch,void * ctx)
  {
     Vec  X,Y,F,W,G;
     SNES snes;

     PetscFunctionBegin;
     PetscCall(SNESLineSearchGetSNES(linesearch,&snes));
     PetscCall(SNESLineSearchSetReason(linesearch,SNES_LINESEARCH_SUCCEEDED));
     PetscCall(SNESLineSearchGetVecs(linesearch,&X,&F,&Y,&W,&G));
     // determine lambda using W and G as work vecs..
     PetscCall(VecAXPY(X,-lambda,Y));
     PetscCall(SNESComputeFunction(snes,X,F));
     PetscCall(SNESLineSearchComputeNorms(linesearch));
     PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(SNESGetLineSearch(snes, &linesearch));
  PetscCall(SNESLineSearchSetType(linesearch, SNESLINESEARCHSHELL));
  PetscCall(SNESLineSearchShellSetApply(linesearch, shellfunc, NULL));
.ve

  Level: advanced

.seealso: [](ch_snes), `SNESLineSearchShellGetApply()`, `SNESLINESEARCHSHELL`, `SNESLineSearchType`, `SNESLineSearch`,
          `SNESLineSearchShellApplyFn`
@*/
PetscErrorCode SNESLineSearchShellSetApply(SNESLineSearch linesearch, SNESLineSearchShellApplyFn *func, void *ctx)
{
  PetscBool             flg;
  SNESLineSearch_Shell *shell = (SNESLineSearch_Shell *)linesearch->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch, SNESLINESEARCH_CLASSID, 1);
  PetscCall(PetscObjectTypeCompare((PetscObject)linesearch, SNESLINESEARCHSHELL, &flg));
  if (flg) {
    shell->ctx  = ctx;
    shell->func = func;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  SNESLineSearchShellGetApply - Gets the apply function and context for the `SNESLINESEARCHSHELL`

  Not Collective

  Input Parameter:
. linesearch - the line search object

  Output Parameters:
+ func - the user function; can be `NULL` if it is not needed, see `SNESLineSearchShellApplyFn` for calling sequence
- ctx  - the user function context; can be `NULL` if it is not needed

  Level: advanced

.seealso: [](ch_snes), `SNESLineSearchShellSetApply()`, `SNESLINESEARCHSHELL`, `SNESLineSearchType`, `SNESLineSearch`,
          `SNESLineSearchShellApplyFn`
@*/
PetscErrorCode SNESLineSearchShellGetApply(SNESLineSearch linesearch, SNESLineSearchShellApplyFn **func, void **ctx)
{
  PetscBool             flg;
  SNESLineSearch_Shell *shell = (SNESLineSearch_Shell *)linesearch->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch, SNESLINESEARCH_CLASSID, 1);
  if (func) PetscAssertPointer(func, 2);
  if (ctx) PetscAssertPointer(ctx, 3);
  PetscCall(PetscObjectTypeCompare((PetscObject)linesearch, SNESLINESEARCHSHELL, &flg));
  if (flg) {
    if (func) *func = shell->func;
    if (ctx) *ctx = shell->ctx;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESLineSearchApply_Shell(SNESLineSearch linesearch)
{
  SNESLineSearch_Shell *shell = (SNESLineSearch_Shell *)linesearch->data;

  PetscFunctionBegin;
  /* apply the user function */
  PetscCheck(shell->func, PetscObjectComm((PetscObject)linesearch), PETSC_ERR_USER, "SNESLineSearchShell needs to have a shell function set with SNESLineSearchShellSetApply()");
  PetscCall((*shell->func)(linesearch, shell->ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESLineSearchDestroy_Shell(SNESLineSearch linesearch)
{
  SNESLineSearch_Shell *shell = (SNESLineSearch_Shell *)linesearch->data;

  PetscFunctionBegin;
  PetscCall(PetscFree(shell));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  SNESLINESEARCHSHELL - Provides an API for a user-provided line search routine.

  Any of the other line searches may serve as a guide to how this is to be done.  There is also a basic
  template in the documentation for `SNESLineSearchShellSetApply()`.

  Level: advanced

.seealso: [](ch_snes), `SNESLineSearch`, `SNES`, `SNESLineSearchCreate()`, `SNESLineSearchSetType()`, `SNESLineSearchShellSetApply()`,
          `SNESLineSearchShellApplyFn`
M*/

PETSC_EXTERN PetscErrorCode SNESLineSearchCreate_Shell(SNESLineSearch linesearch)
{
  SNESLineSearch_Shell *shell;

  PetscFunctionBegin;
  linesearch->ops->apply          = SNESLineSearchApply_Shell;
  linesearch->ops->destroy        = SNESLineSearchDestroy_Shell;
  linesearch->ops->setfromoptions = NULL;
  linesearch->ops->reset          = NULL;
  linesearch->ops->view           = NULL;
  linesearch->ops->setup          = NULL;

  PetscCall(PetscNew(&shell));

  linesearch->data = (void *)shell;
  PetscFunctionReturn(PETSC_SUCCESS);
}
