#include <private/linesearchimpl.h>
#include <private/snesimpl.h>

#undef __FUNCT__
#define __FUNCT__ "LineSearchApply_Basic"

/*@C
   SNESLineSearchBasic - This routine is not a line search at all;
   it simply uses the full step.  Thus, this routine is intended
   to serve as a template and is not recommended for general use.

   Logically Collective on SNES and Vec

   Input Parameters:
+  snes - nonlinear context
.  lsctx - optional context for line search (not used here)
.  x - current iterate
.  f - residual evaluated at x
.  y - search direction
.  fnorm - 2-norm of f
-  xnorm - norm of x if known, otherwise 0

   Output Parameters:
+  g - residual evaluated at new iterate y
.  w - new iterate
.  gnorm - 2-norm of g
.  ynorm - 2-norm of search length
-  flag - PETSC_TRUE on success, PETSC_FALSE on failure

   Options Database Key:
.  -snes_ls basic - Activates SNESLineSearchNo()

   Level: advanced

.keywords: SNES, nonlinear, line search, cubic

.seealso: SNESLineSearchCubic(), SNESLineSearchQuadratic(),
          SNESLineSearchSet(), SNESLineSearchNoNorms()
@*/
PetscErrorCode  LineSearchApply_Basic(LineSearch linesearch)
{
  PetscBool      changed_y, changed_w;
  PetscErrorCode ierr;
  Vec            X = linesearch->vec_sol;
  Vec            F = linesearch->vec_func;
  Vec            Y = linesearch->vec_update;
  Vec            W = linesearch->vec_sol_new;
  SNES           snes = linesearch->snes;
  PetscReal      *gnorm = &linesearch->fnorm;
  PetscReal      *ynorm = &linesearch->ynorm;
  PetscReal      *xnorm = &linesearch->xnorm;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(SNES_LineSearch,linesearch,X,F,snes);CHKERRQ(ierr);

  /* precheck */
  ierr = LineSearchPreCheck(linesearch, &changed_y);CHKERRQ(ierr);

  /* update */
  ierr = VecWAXPY(W,-linesearch->damping,Y,X);CHKERRQ(ierr);

  /* postcheck */
  ierr = LineSearchPostCheck(linesearch, &changed_y, &changed_w);CHKERRQ(ierr);
  if (changed_y) {
    ierr = VecWAXPY(W,-linesearch->damping,Y,X);CHKERRQ(ierr);
  }
  ierr = SNESComputeFunction(snes,W,F);CHKERRQ(ierr);
  if (snes->domainerror) {
    linesearch->success = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
  if (linesearch->norms) {
    ierr = VecNormBegin(F,NORM_2,gnorm);CHKERRQ(ierr);
    ierr = VecNormBegin(X,NORM_2,xnorm);CHKERRQ(ierr);
    ierr = VecNormBegin(Y,NORM_2,ynorm);CHKERRQ(ierr);
    ierr = VecNormEnd(F,NORM_2,gnorm);CHKERRQ(ierr);
    ierr = VecNormEnd(X,NORM_2,xnorm);CHKERRQ(ierr);
    ierr = VecNormEnd(Y,NORM_2,ynorm);CHKERRQ(ierr);
    if (PetscIsInfOrNanReal(*gnorm)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"User provided compute function generated a Not-a-Number");
  }
  /* copy the solution over */
  ierr = VecCopy(W, X);CHKERRQ(ierr);

  ierr = PetscLogEventEnd(SNES_LineSearch,linesearch,X,F,snes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "LineSearchCreate_Basic"
PetscErrorCode LineSearchCreate_Basic(LineSearch linesearch)
{
  PetscFunctionBegin;
  linesearch->ops->apply          = LineSearchApply_Basic;
  linesearch->ops->destroy        = PETSC_NULL;
  linesearch->ops->setfromoptions = PETSC_NULL;
  linesearch->ops->reset          = PETSC_NULL;
  linesearch->ops->view           = PETSC_NULL;
  linesearch->ops->setup          = PETSC_NULL;
  PetscFunctionReturn(0);
}
EXTERN_C_END
