#include <petsc-private/linesearchimpl.h>
#include <petsc-private/snesimpl.h>


typedef struct {
  SNESLineSearchUserFunc func;
  void               *ctx;
} SNESLineSearch_Shell;

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchShellSetUserFunc"
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

$  PetscErrorCode shellfunc(SNESLineSearch linesearch, void * ctx) {
$     Vec  X, Y, F, W, G;
$     SNES snes;
$     PetscFunctionBegin;
$     ierr = SNESLineSearchGetSNES(linesearch, &snes);CHKERRQ(ierr);
$     ierr = SNESLineSearchSetSuccess(linesearch, PETSC_TRUE);CHKERRQ(ierr);
$     ierr = SNESLineSearchGetVecs(linesearch, X, Y, F, W, G);CHKERRQ(ierr);
$     .. determine lambda using W and G as work vecs..
$     ierr = VecAXPY(X, -lambda, Y);CHKERRQ(ierr);
$     ierr = SNESComputeFunction(snes, X, F);CHKERRQ(ierr);
$     ierr = SNESLineSearchComputeNorms(linesearch);CHKERRQ(ierr);
$     PetscFunctionReturn(0);
$  }
$
$  ...
$
$  ierr = SNESGetSNESLineSearch(snes, &linesearch);CHKERRQ(ierr);
$  ierr = SNESLineSearchSetType(linesearch, SNESLINESEARCHSHELL);CHKERRQ(ierr);
$  ierr = SNESLineSearchShellSetUserFunc(linesearch, shellfunc, PETSC_NULL);CHKERRQ(ierr);

   Level: advanced

   .keywords: SNESLineSearch, Shell, user, function, set

   .seealso: SNESLineSearchShellGetUserFunc(), SNESLINESEARCHSHELL
@*/
PetscErrorCode SNESLineSearchShellSetUserFunc(SNESLineSearch linesearch, SNESLineSearchUserFunc func, void *ctx) {

  PetscErrorCode   ierr;
  PetscBool        flg;
  SNESLineSearch_Shell *shell = (SNESLineSearch_Shell *)linesearch->data;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch, SNESLINESEARCH_CLASSID, 1);
  ierr = PetscObjectTypeCompare((PetscObject)linesearch,SNESLINESEARCHSHELL,&flg);CHKERRQ(ierr);
  if (flg) {
    shell->ctx = ctx;
    shell->func = func;
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchShellGetUserFunc"
/*@C
   SNESLineSearchShellGetUserFunc - Gets the user function and context for the shell implementation.

   Not Collective

   Level: advanced

   .keywords: SNESLineSearch, get, Shell, user, function

   .seealso: SNESLineSearchShellSetUserFunc()
@*/
PetscErrorCode SNESLineSearchShellGetUserFunc(SNESLineSearch linesearch, SNESLineSearchUserFunc *func, void **ctx) {

  PetscErrorCode   ierr;
  PetscBool        flg;
  SNESLineSearch_Shell *shell = (SNESLineSearch_Shell *)linesearch->data;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch, SNESLINESEARCH_CLASSID, 1);
  if (func) PetscValidPointer(func,2);
  if (ctx)  PetscValidPointer(ctx,3);
  ierr = PetscObjectTypeCompare((PetscObject)linesearch,SNESLINESEARCHSHELL,&flg);CHKERRQ(ierr);
  if (flg) {
    *ctx  = shell->ctx;
    *func = shell->func;
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchApply_Shell"
static PetscErrorCode  SNESLineSearchApply_Shell(SNESLineSearch linesearch)
{
  SNESLineSearch_Shell *shell = (SNESLineSearch_Shell *)linesearch->data;
  PetscErrorCode   ierr;

  PetscFunctionBegin;

  /* apply the user function */
  if (shell->func) {
    ierr = (*shell->func)(linesearch, shell->ctx);CHKERRQ(ierr);
  } else {
    SETERRQ(((PetscObject)linesearch)->comm, PETSC_ERR_USER, "SNESLineSearchShell needs to have a shell function set with SNESLineSearchShellSetUserFunc");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchDestroy_Shell"
static PetscErrorCode  SNESLineSearchDestroy_Shell(SNESLineSearch linesearch)
{
  SNESLineSearch_Shell *shell = (SNESLineSearch_Shell *)linesearch->data;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscFree(shell);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchCreate_Shell"
/*MC

SNESLINESEARCHSHELL - Provides context for a user-provided line search routine.

The user routine has one argument, the SNESLineSearch context.  The user uses the interface to
extract line search parameters and set them accordingly when the computation is finished.

Any of the other line searches may serve as a guide to how this is to be done.  There is also a basic
template in the documentation for SNESLineSearchShellSetUserFunc().

Level: advanced

M*/
PETSC_EXTERN_C PetscErrorCode SNESLineSearchCreate_Shell(SNESLineSearch linesearch)
{

  SNESLineSearch_Shell     *shell;
  PetscErrorCode       ierr;

  PetscFunctionBegin;

  linesearch->ops->apply          = SNESLineSearchApply_Shell;
  linesearch->ops->destroy        = SNESLineSearchDestroy_Shell;
  linesearch->ops->setfromoptions = PETSC_NULL;
  linesearch->ops->reset          = PETSC_NULL;
  linesearch->ops->view           = PETSC_NULL;
  linesearch->ops->setup          = PETSC_NULL;

  ierr = PetscNewLog(linesearch, SNESLineSearch_Shell, &shell);CHKERRQ(ierr);
  linesearch->data = (void*) shell;
  PetscFunctionReturn(0);
}
