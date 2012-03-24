#include <petsc-private/snesimpl.h>             /*I   "petscsnes.h"   I*/

typedef struct {PetscErrorCode (*solve)(SNES,Vec);void *ctx;} SNES_Shell;

#undef __FUNCT__  
#define __FUNCT__ "SNESShellSetSolve"
/*@C
   SNESShellSetSolve - Sets routine to apply as solver

   Logically Collective on SNES

   Input Parameters:
+  snes - the nonlinear solver context
-  apply - the application-provided solver routine

   Calling sequence of solve:
.vb
   PetscErrorCode apply (SNES snes,Vec xout)
.ve

+  snes - the preconditioner, get the application context with SNESShellGetContext()
-  xout - solution vector

   Notes: the function MUST return an error code of 0 on success and nonzero on failure.

   Level: advanced

.keywords: SNES, shell, set, apply, user-provided

.seealso: SNESSHELL, SNESShellSetContext(), SNESShellGetContext()
@*/
PetscErrorCode  SNESShellSetSolve(SNES snes,PetscErrorCode (*solve)(SNES,Vec))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  ierr = PetscTryMethod(snes,"SNESShellSetSolve_C",(SNES,PetscErrorCode (*)(SNES,Vec)),(snes,solve));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESReset_Shell"
PetscErrorCode SNESReset_Shell(SNES snes)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESDestroy_Shell"
PetscErrorCode SNESDestroy_Shell(SNES snes)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESReset_Shell(snes);CHKERRQ(ierr);
  ierr = PetscFree(snes->data);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSetUp_Shell"
PetscErrorCode SNESSetUp_Shell(SNES snes)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSetFromOptions_Shell"
PetscErrorCode SNESSetFromOptions_Shell(SNES snes)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = PetscOptionsHead("SNES Shell options");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESView_Shell"
PetscErrorCode SNESView_Shell(SNES snes, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESShellGetContext"
/*@C
    SNESShellGetContext - Returns the user-provided context associated with a shell SNES

    Not Collective

    Input Parameter:
.   snes - should have been created with SNESSetType(snes,SNESSHELL);

    Output Parameter:
.   ctx - the user provided context

    Level: advanced

    Notes:
    This routine is intended for use within various shell routines
    
.keywords: SNES, shell, get, context

.seealso: SNESCreateShell(), SNESShellSetContext()
@*/
PetscErrorCode  SNESShellGetContext(SNES snes,void **ctx)
{
  PetscErrorCode ierr;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidPointer(ctx,2); 
  ierr = PetscTypeCompare((PetscObject)snes,SNESSHELL,&flg);CHKERRQ(ierr);
  if (!flg) *ctx = 0; 
  else      *ctx = ((SNES_Shell*)(snes->data))->ctx; 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESShellSetContext"
/*@
    SNESShellSetContext - sets the context for a shell SNES

   Logically Collective on SNES

    Input Parameters:
+   snes - the shell SNES
-   ctx - the context

   Level: advanced

   Fortran Notes: The context can only be an integer or a PetscObject
      unfortunately it cannot be a Fortran array or derived type.


.seealso: SNESCreateShell(), SNESShellGetContext()
@*/
PetscErrorCode  SNESShellSetContext(SNES snes,void *ctx)
{
  SNES_Shell     *shell = (SNES_Shell*)snes->data;
  PetscErrorCode ierr;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  ierr = PetscTypeCompare((PetscObject)snes,SNESSHELL,&flg);CHKERRQ(ierr);
  if (flg) {
    shell->ctx = ctx;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSolve_Shell"
PetscErrorCode SNESSolve_Shell(SNES snes)
{
  SNES_Shell     *shell = (SNES_Shell *) snes->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!shell->solve) SETERRQ(((PetscObject)snes)->comm,PETSC_ERR_ARG_WRONGSTATE,"Must call SNESShellSetSolve() first");
  snes->reason = SNES_CONVERGED_ITS;
  ierr = (*shell->solve)(snes,snes->vec_sol);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SNESShellSetSolve_Shell"
PetscErrorCode  SNESShellSetSolve_Shell(SNES snes,PetscErrorCode (*solve)(SNES,Vec))
{
  SNES_Shell *shell = (SNES_Shell*)snes->data;

  PetscFunctionBegin;
  shell->solve = solve;
  PetscFunctionReturn(0);
}
EXTERN_C_END

/*MC
  SNESSHELL - a user provided nonlinear solver 

   Level: advanced

.seealso: SNESCreate(), SNES, SNESSetType(), SNESType (for list of available types)
M*/

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "SNESCreate_Shell"
PetscErrorCode SNESCreate_Shell(SNES snes)
{
  SNES_Shell     *shell;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  snes->ops->destroy        = SNESDestroy_Shell;
  snes->ops->setup          = SNESSetUp_Shell;
  snes->ops->setfromoptions = SNESSetFromOptions_Shell;
  snes->ops->view           = SNESView_Shell;
  snes->ops->solve          = SNESSolve_Shell;
  snes->ops->reset          = SNESReset_Shell;

  snes->usesksp             = PETSC_FALSE;
  snes->usespc              = PETSC_FALSE;

  ierr = PetscNewLog(snes, SNES_Shell, &shell);CHKERRQ(ierr);
  snes->data = (void*) shell;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)snes,"SNESShellSetSolve_C","SNESShellSetSolve_Shell",SNESShellSetSolve_Shell);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
