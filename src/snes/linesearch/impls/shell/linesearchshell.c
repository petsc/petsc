#include <petsc/private/linesearchimpl.h>
#include <petsc/private/snesimpl.h>

typedef struct {
  SNESLineSearchUserFunc func;
  void                   *ctx;
} SNESLineSearch_Shell;

/*@C
   SNESLineSearchShellSetUserFunc - Sets the user function for the SNESLineSearch Shell implementation.

   Not Collective

   Input Parameters:
+  linesearch - SNESLineSearch context
.  func - function implementing the linesearch shell.
-  ctx - context for func

   Calling sequence of func:
+  linesearch - the linesearch instance
-  ctx - the above mentioned context

   Usage:

$  PetscErrorCode shellfunc(SNESLineSearch linesearch,void * ctx)
$  {
$     Vec  X,Y,F,W,G;
$     SNES snes;
$     PetscFunctionBegin;
$     ierr = SNESLineSearchGetSNES(linesearch,&snes);CHKERRQ(ierr);
$     ierr = SNESLineSearchSetReason(linesearch,SNES_LINESEARCH_SUCCEEDED);CHKERRQ(ierr);
$     ierr = SNESLineSearchGetVecs(linesearch,&X,&F,&Y,&W,&G);CHKERRQ(ierr);
$     .. determine lambda using W and G as work vecs..
$     ierr = VecAXPY(X,-lambda,Y);CHKERRQ(ierr);
$     ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);
$     ierr = SNESLineSearchComputeNorms(linesearch);CHKERRQ(ierr);
$     PetscFunctionReturn(0);
$  }
$
$  ...
$
$  ierr = SNESGetLineSearch(snes, &linesearch);CHKERRQ(ierr);
$  ierr = SNESLineSearchSetType(linesearch, SNESLINESEARCHSHELL);CHKERRQ(ierr);
$  ierr = SNESLineSearchShellSetUserFunc(linesearch, shellfunc, NULL);CHKERRQ(ierr);

   Level: advanced

   .seealso: SNESLineSearchShellGetUserFunc(), SNESLINESEARCHSHELL
@*/
PetscErrorCode SNESLineSearchShellSetUserFunc(SNESLineSearch linesearch, SNESLineSearchUserFunc func, void *ctx)
{
  PetscErrorCode       ierr;
  PetscBool            flg;
  SNESLineSearch_Shell *shell = (SNESLineSearch_Shell*)linesearch->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch, SNESLINESEARCH_CLASSID, 1);
  ierr = PetscObjectTypeCompare((PetscObject)linesearch,SNESLINESEARCHSHELL,&flg);CHKERRQ(ierr);
  if (flg) {
    shell->ctx  = ctx;
    shell->func = func;
  }
  PetscFunctionReturn(0);
}

/*@C
   SNESLineSearchShellGetUserFunc - Gets the user function and context for the shell implementation.

   Not Collective

   Input Parameter:
.     linesearch - the line search object

   Output Parameters:
+    func  - the user function; can be NULL if you do not want it
-    ctx   - the user function context; can be NULL if you do not want it

   Level: advanced

   .seealso: SNESLineSearchShellSetUserFunc()
@*/
PetscErrorCode SNESLineSearchShellGetUserFunc(SNESLineSearch linesearch, SNESLineSearchUserFunc *func, void **ctx)
{
  PetscErrorCode       ierr;
  PetscBool            flg;
  SNESLineSearch_Shell *shell = (SNESLineSearch_Shell*)linesearch->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch, SNESLINESEARCH_CLASSID, 1);
  if (func) PetscValidPointer(func,2);
  if (ctx)  PetscValidPointer(ctx,3);
  ierr = PetscObjectTypeCompare((PetscObject)linesearch,SNESLINESEARCHSHELL,&flg);CHKERRQ(ierr);
  if (flg) {
    if (func) *func = shell->func;
    if (ctx) *ctx  = shell->ctx;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode  SNESLineSearchApply_Shell(SNESLineSearch linesearch)
{
  SNESLineSearch_Shell *shell = (SNESLineSearch_Shell*)linesearch->data;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  /* apply the user function */
  if (shell->func) {
    ierr = (*shell->func)(linesearch, shell->ctx);CHKERRQ(ierr);
  } else SETERRQ(PetscObjectComm((PetscObject)linesearch), PETSC_ERR_USER, "SNESLineSearchShell needs to have a shell function set with SNESLineSearchShellSetUserFunc");
  PetscFunctionReturn(0);
}

static PetscErrorCode  SNESLineSearchDestroy_Shell(SNESLineSearch linesearch)
{
  SNESLineSearch_Shell *shell = (SNESLineSearch_Shell*)linesearch->data;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  ierr = PetscFree(shell);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
   SNESLINESEARCHSHELL - Provides context for a user-provided line search routine.

The user routine has one argument, the SNESLineSearch context.  The user uses the interface to
extract line search parameters and set them accordingly when the computation is finished.

Any of the other line searches may serve as a guide to how this is to be done.  There is also a basic
template in the documentation for SNESLineSearchShellSetUserFunc().

Level: advanced

M*/
PETSC_EXTERN PetscErrorCode SNESLineSearchCreate_Shell(SNESLineSearch linesearch)
{

  SNESLineSearch_Shell *shell;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  linesearch->ops->apply          = SNESLineSearchApply_Shell;
  linesearch->ops->destroy        = SNESLineSearchDestroy_Shell;
  linesearch->ops->setfromoptions = NULL;
  linesearch->ops->reset          = NULL;
  linesearch->ops->view           = NULL;
  linesearch->ops->setup          = NULL;

  ierr = PetscNewLog(linesearch,&shell);CHKERRQ(ierr);

  linesearch->data = (void*) shell;
  PetscFunctionReturn(0);
}
