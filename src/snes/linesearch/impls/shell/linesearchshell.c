#include <private/linesearchimpl.h>
#include <private/snesimpl.h>

/*MC

PetscLineSearchShell - Provides context for a user-provided line search routine.

The user routine has one argument, the PetscLineSearch context.  The user uses the interface to
extract line search parameters and set them accordingly when the computation is finished.

Any of the other line searches may serve as a guide to how this is to be done.

Level: advanced

 M*/

typedef struct {
  PetscLineSearchUserFunc func;
  void               *ctx;
} PetscLineSearch_Shell;

#undef __FUNCT__
#define __FUNCT__ "PetscLineSearchShellSetUserFunc"
/*@C
   PetscLineSearchShellSetUserFunc - Sets the user function for the PetscLineSearch Shell implementation.

   Not Collective

   Level: advanced

   .keywords: PetscLineSearch, PetscLineSearchShell, Shell

   .seealso: PetscLineSearchShellGetUserFunc()
@*/

PetscErrorCode PetscLineSearchShellSetUserFunc(PetscLineSearch linesearch, PetscLineSearchUserFunc func, void *ctx) {

  PetscErrorCode   ierr;
  PetscBool        flg;
  PetscLineSearch_Shell *shell = (PetscLineSearch_Shell *)linesearch->data;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch, PETSCLINESEARCH_CLASSID, 1);
  ierr = PetscTypeCompare((PetscObject)linesearch,PETSCLINESEARCHSHELL,&flg);CHKERRQ(ierr);
  if (flg) {
    shell->ctx = ctx;
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PetscLineSearchShellGetUserFunc"
/*@C
   PetscLineSearchShellGetUserFunc - Gets the user function and context for the shell implementation.

   Not Collective

   Level: advanced

   .keywords: PetscLineSearch, PetscLineSearchShell, Shell

   .seealso: PetscLineSearchShellSetUserFunc()
@*/
PetscErrorCode PetscLineSearchShellGetUserFunc(PetscLineSearch linesearch, PetscLineSearchUserFunc *func, void **ctx) {

  PetscErrorCode   ierr;
  PetscBool        flg;
  PetscLineSearch_Shell *shell = (PetscLineSearch_Shell *)linesearch->data;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch, PETSCLINESEARCH_CLASSID, 1);
  if (func) PetscValidPointer(func,2);
  if (ctx)  PetscValidPointer(ctx,3);
  ierr = PetscTypeCompare((PetscObject)linesearch,PETSCLINESEARCHSHELL,&flg);CHKERRQ(ierr);
  if (flg) {
    *ctx  = shell->ctx;
    *func = shell->func;
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PetscLineSearchApply_Shell"
PetscErrorCode  PetscLineSearchApply_Shell(PetscLineSearch linesearch)
{
  PetscLineSearch_Shell *shell = (PetscLineSearch_Shell *)linesearch->data;
  PetscErrorCode   ierr;

  PetscFunctionBegin;

  /* apply the user function */
  if (shell->func) {
    ierr = (*shell->func)(linesearch, shell->ctx);CHKERRQ(ierr);
  } else {
    SETERRQ(((PetscObject)linesearch)->comm, PETSC_ERR_USER, "PetscLineSearchShell needs to have a shell function set with PetscLineSearchShellSetUserFunc");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscLineSearchDestroy_Shell"
PetscErrorCode  PetscLineSearchDestroy_Shell(PetscLineSearch linesearch)
{
  PetscLineSearch_Shell *shell = (PetscLineSearch_Shell *)linesearch->data;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscFree(shell);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PetscLineSearchCreate_Shell"
PetscErrorCode PetscLineSearchCreate_Shell(PetscLineSearch linesearch)
{

  PetscLineSearch_Shell     *shell;
  PetscErrorCode       ierr;

  PetscFunctionBegin;

  linesearch->ops->apply          = PetscLineSearchApply_Shell;
  linesearch->ops->destroy        = PetscLineSearchDestroy_Shell;
  linesearch->ops->setfromoptions = PETSC_NULL;
  linesearch->ops->reset          = PETSC_NULL;
  linesearch->ops->view           = PETSC_NULL;
  linesearch->ops->setup          = PETSC_NULL;

  ierr = PetscNewLog(linesearch, PetscLineSearch_Shell, &shell);CHKERRQ(ierr);
  linesearch->data = (void*) shell;
  PetscFunctionReturn(0);
}
EXTERN_C_END
