#include <private/linesearchimpl.h>
#include <private/snesimpl.h>

/*MC
   LineSearchBasic - This routine is not a line search at all;
   it simply uses the full step.  Thus, this routine is intended
   to serve as a template and is not recommended for general use.

   Level: advanced

.keywords: SNES, LineSearch, damping

.seealso: LineSearchCreate(), LineSearchSetType()
M*/

#undef __FUNCT__
#define __FUNCT__ "LineSearchApply_Basic"

PetscErrorCode  LineSearchApply_Basic(LineSearch linesearch)
{
  PetscBool      changed_y, changed_w;
  PetscErrorCode ierr;
  Vec            X, F, Y, W;
  SNES           snes;
  PetscReal      gnorm, xnorm, ynorm, lambda;
  PetscBool      domainerror;

  PetscFunctionBegin;

  ierr = LineSearchGetVecs(linesearch, &X, &F, &Y, &W, PETSC_NULL);CHKERRQ(ierr);
  ierr = LineSearchGetNorms(linesearch, &xnorm, &gnorm, &ynorm);CHKERRQ(ierr);
  ierr = LineSearchGetLambda(linesearch, &lambda);CHKERRQ(ierr);
  ierr = LineSearchGetSNES(linesearch, &snes);CHKERRQ(ierr);
  ierr = LineSearchSetSuccess(linesearch, PETSC_TRUE);CHKERRQ(ierr);

  /* precheck */
  ierr = LineSearchPreCheck(linesearch, &changed_y);CHKERRQ(ierr);

  /* update */
  ierr = VecWAXPY(W,-lambda,Y,X);CHKERRQ(ierr);

  /* postcheck */
  ierr = LineSearchPostCheck(linesearch, &changed_y, &changed_w);CHKERRQ(ierr);
  if (changed_y) {
    ierr = VecWAXPY(W,-lambda,Y,X);CHKERRQ(ierr);
  }
  ierr = SNESComputeFunction(snes,W,F);CHKERRQ(ierr);
  ierr = SNESGetFunctionDomainError(snes, &domainerror);CHKERRQ(ierr);
  if (domainerror) {
    ierr = LineSearchSetSuccess(linesearch, PETSC_FALSE);
    PetscFunctionReturn(0);
  }

  ierr = LineSearchComputeNorms(linesearch);CHKERRQ(ierr);

  /* copy the solution over */
  ierr = VecCopy(W, X);CHKERRQ(ierr);
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
