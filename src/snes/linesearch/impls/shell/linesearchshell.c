#include <petsc/private/linesearchimpl.h>
#include <petsc/private/snesimpl.h>

typedef struct {
  SNESLineSearchUserFunc func;
  void                  *ctx;
} SNESLineSearch_Shell;

/*@C
   SNESLineSearchShellSetUserFunc - Sets the user function for the `SNESLINESEARCHSHELL` implementation.

   Not Collective

   Input Parameters:
+  linesearch - `SNESLineSearch` context
.  func - function implementing the linesearch shell.
-  ctx - context for func

   Calling sequence of `func`:
$  PetscErrorCode func(SNESLinesearch, void *ctx)
+  linesearch - the linesearch instance
-  ctx - the above mentioned context

   Usage:
.vb
  PetscErrorCode shellfunc(SNESLineSearch linesearch,void * ctx)
  {
     Vec  X,Y,F,W,G;
     SNES snes;
     PetscFunctionBegin;
     PetscCall(SNESLineSearchGetSNES(linesearch,&snes));
     PetscCall(SNESLineSearchSetReason(linesearch,SNES_LINESEARCH_SUCCEEDED));
     PetscCall(SNESLineSearchGetVecs(linesearch,&X,&F,&Y,&W,&G));
     .. determine lambda using W and G as work vecs..
     PetscCall(VecAXPY(X,-lambda,Y));
     PetscCall(SNESComputeFunction(snes,X,F));
     PetscCall(SNESLineSearchComputeNorms(linesearch));
     PetscFunctionReturn(PETSC_SUCCESS);
  }

  ...

  PetscCall(SNESGetLineSearch(snes, &linesearch));
  PetscCall(SNESLineSearchSetType(linesearch, SNESLINESEARCHSHELL));
  PetscCall(SNESLineSearchShellSetUserFunc(linesearch, shellfunc, NULL));
.ve

   Level: advanced

.seealso: `SNESLineSearchShellGetUserFunc()`, `SNESLINESEARCHSHELL`, `SNESLineSearchType`, `SNESLineSearch`
@*/
PetscErrorCode SNESLineSearchShellSetUserFunc(SNESLineSearch linesearch, SNESLineSearchUserFunc func, void *ctx)
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
   SNESLineSearchShellGetUserFunc - Gets the user function and context for the  `SNESLINESEARCHSHELL`

   Not Collective

   Input Parameter:
.   linesearch - the line search object

   Output Parameters:
+    func  - the user function; can be NULL if you do not want it
-    ctx   - the user function context; can be NULL if you do not want it

   Level: advanced

.seealso: `SNESLineSearchShellSetUserFunc()`, `SNESLINESEARCHSHELL`, `SNESLineSearchType`, `SNESLineSearch`
@*/
PetscErrorCode SNESLineSearchShellGetUserFunc(SNESLineSearch linesearch, SNESLineSearchUserFunc *func, void **ctx)
{
  PetscBool             flg;
  SNESLineSearch_Shell *shell = (SNESLineSearch_Shell *)linesearch->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch, SNESLINESEARCH_CLASSID, 1);
  if (func) PetscValidPointer(func, 2);
  if (ctx) PetscValidPointer(ctx, 3);
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
  if (shell->func) {
    PetscCall((*shell->func)(linesearch, shell->ctx));
  } else SETERRQ(PetscObjectComm((PetscObject)linesearch), PETSC_ERR_USER, "SNESLineSearchShell needs to have a shell function set with SNESLineSearchShellSetUserFunc");
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
   SNESLINESEARCHSHELL - Provides context for a user-provided line search routine.

The user routine has one argument, the SNESLineSearch context.  The user uses the interface to
extract line search parameters and set them accordingly when the computation is finished.

Any of the other line searches may serve as a guide to how this is to be done.  There is also a basic
template in the documentation for SNESLineSearchShellSetUserFunc().

Level: advanced

.seealso: `SNESLineSearch`, `SNES`, `SNESLineSearchCreate()`, `SNESLineSearchSetType()`
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
