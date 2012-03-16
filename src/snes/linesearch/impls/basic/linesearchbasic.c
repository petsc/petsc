#include <private/linesearchimpl.h>
#include <private/snesimpl.h>

/*MC
   PetscLineSearchBasic - This routine is not a line search at all;
   it simply uses the full step.  Thus, this routine is intended
   to serve as a template and is not recommended for general use.

   Level: advanced

.keywords: SNES, PetscLineSearch, damping

.seealso: PetscLineSearchCreate(), PetscLineSearchSetType()
M*/

#undef __FUNCT__
#define __FUNCT__ "PetscLineSearchApply_Basic"

PetscErrorCode  PetscLineSearchApply_Basic(PetscLineSearch linesearch)
{
  PetscBool      changed_y, changed_w;
  PetscErrorCode ierr;
  Vec            X, F, Y, W;
  SNES           snes;
  PetscReal      gnorm, xnorm, ynorm, lambda;
  PetscBool      domainerror;

  PetscFunctionBegin;

  ierr = PetscLineSearchGetVecs(linesearch, &X, &F, &Y, &W, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscLineSearchGetNorms(linesearch, &xnorm, &gnorm, &ynorm);CHKERRQ(ierr);
  ierr = PetscLineSearchGetLambda(linesearch, &lambda);CHKERRQ(ierr);
  ierr = PetscLineSearchGetSNES(linesearch, &snes);CHKERRQ(ierr);
  ierr = PetscLineSearchSetSuccess(linesearch, PETSC_TRUE);CHKERRQ(ierr);

  /* precheck */
  ierr = PetscLineSearchPreCheck(linesearch, &changed_y);CHKERRQ(ierr);

  /* update */
  ierr = VecWAXPY(W,-lambda,Y,X);CHKERRQ(ierr);
  if (linesearch->ops->viproject) {
    ierr = (*linesearch->ops->viproject)(snes, W);CHKERRQ(ierr);
  }

  /* postcheck */
  ierr = PetscLineSearchPostCheck(linesearch, &changed_y, &changed_w);CHKERRQ(ierr);
  if (changed_y) {
    ierr = VecWAXPY(W,-lambda,Y,X);CHKERRQ(ierr);
    if (linesearch->ops->viproject) {
      ierr = (*linesearch->ops->viproject)(snes, W);CHKERRQ(ierr);
    }
  }
  ierr = SNESComputeFunction(snes,W,F);CHKERRQ(ierr);
  ierr = SNESGetFunctionDomainError(snes, &domainerror);CHKERRQ(ierr);
  if (domainerror) {
    ierr = PetscLineSearchSetSuccess(linesearch, PETSC_FALSE);
    PetscFunctionReturn(0);
  }

  ierr = VecNorm(Y, NORM_2, &linesearch->ynorm);CHKERRQ(ierr);
  ierr = VecNorm(W, NORM_2, &linesearch->xnorm);CHKERRQ(ierr);
  if (linesearch->ops->vinorm) {
    linesearch->fnorm = gnorm;
    ierr = (*linesearch->ops->vinorm)(snes, F, W, &linesearch->fnorm);CHKERRQ(ierr);
  } else {
    ierr = VecNorm(F,NORM_2,&linesearch->fnorm);CHKERRQ(ierr);
  }

  /*
  ierr = PetscLineSearchComputeNorms(linesearch);CHKERRQ(ierr);
   */

  /* copy the solution over */
  ierr = VecCopy(W, X);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PetscLineSearchCreate_Basic"
PetscErrorCode PetscLineSearchCreate_Basic(PetscLineSearch linesearch)
{
  PetscFunctionBegin;
  linesearch->ops->apply          = PetscLineSearchApply_Basic;
  linesearch->ops->destroy        = PETSC_NULL;
  linesearch->ops->setfromoptions = PETSC_NULL;
  linesearch->ops->reset          = PETSC_NULL;
  linesearch->ops->view           = PETSC_NULL;
  linesearch->ops->setup          = PETSC_NULL;
  PetscFunctionReturn(0);
}
EXTERN_C_END
