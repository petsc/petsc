#include <petsc-private/linesearchimpl.h>
#include <petsc-private/snesimpl.h>

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchApply_Basic"
static PetscErrorCode  SNESLineSearchApply_Basic(SNESLineSearch linesearch)
{
  PetscBool      changed_y, changed_w;
  PetscErrorCode ierr;
  Vec            X, F, Y, W;
  SNES           snes;
  PetscReal      gnorm, xnorm, ynorm, lambda;
  PetscBool      domainerror;

  PetscFunctionBegin;

  ierr = SNESLineSearchGetVecs(linesearch, &X, &F, &Y, &W, PETSC_NULL);CHKERRQ(ierr);
  ierr = SNESLineSearchGetNorms(linesearch, &xnorm, &gnorm, &ynorm);CHKERRQ(ierr);
  ierr = SNESLineSearchGetLambda(linesearch, &lambda);CHKERRQ(ierr);
  ierr = SNESLineSearchGetSNES(linesearch, &snes);CHKERRQ(ierr);
  ierr = SNESLineSearchSetSuccess(linesearch, PETSC_TRUE);CHKERRQ(ierr);

  /* precheck */
  ierr = SNESLineSearchPreCheck(linesearch,X,Y,&changed_y);CHKERRQ(ierr);

  /* update */
  ierr = VecWAXPY(W,-lambda,Y,X);CHKERRQ(ierr);
  if (linesearch->ops->viproject) {
    ierr = (*linesearch->ops->viproject)(snes, W);CHKERRQ(ierr);
  }

  /* postcheck */
  ierr = SNESLineSearchPostCheck(linesearch,X,Y,W,&changed_y,&changed_w);CHKERRQ(ierr);
  if (changed_y) {
    ierr = VecWAXPY(W,-lambda,Y,X);CHKERRQ(ierr);
    if (linesearch->ops->viproject) {
      ierr = (*linesearch->ops->viproject)(snes, W);CHKERRQ(ierr);
    }
  }
  if (linesearch->norms || snes->iter < snes->max_its-1) {
    ierr = SNESComputeFunction(snes,W,F);CHKERRQ(ierr);
    ierr = SNESGetFunctionDomainError(snes, &domainerror);CHKERRQ(ierr);
    if (domainerror) {
      ierr = SNESLineSearchSetSuccess(linesearch, PETSC_FALSE);
      PetscFunctionReturn(0);
    }
  }

  if (linesearch->norms) {
    if (!linesearch->ops->vinorm) VecNormBegin(F, NORM_2, &linesearch->fnorm);
    ierr = VecNormBegin(Y, NORM_2, &linesearch->ynorm);CHKERRQ(ierr);
    ierr = VecNormBegin(W, NORM_2, &linesearch->xnorm);CHKERRQ(ierr);
    if (!linesearch->ops->vinorm) VecNormEnd(F, NORM_2, &linesearch->fnorm);
    ierr = VecNormEnd(Y, NORM_2, &linesearch->ynorm);CHKERRQ(ierr);
    ierr = VecNormEnd(W, NORM_2, &linesearch->xnorm);CHKERRQ(ierr);

    if (linesearch->ops->vinorm) {
      linesearch->fnorm = gnorm;
      ierr = (*linesearch->ops->vinorm)(snes, F, W, &linesearch->fnorm);CHKERRQ(ierr);
    } else {
      ierr = VecNorm(F,NORM_2,&linesearch->fnorm);CHKERRQ(ierr);
    }
  }

  /* copy the solution over */
  ierr = VecCopy(W, X);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchCreate_Basic"
/*MC
   SNESLINESEARCHBASIC - This line search implementation is not a line
   search at all; it simply uses the full step.  Thus, this routine is intended
   for methods with well-scaled updates; i.e. Newton's method (SNESLS), on
   well-behaved problems.

   Options Database Keys:
+   -snes_linesearch_damping (1.0) damping parameter.
-   -snes_linesearch_norms (true) whether to compute norms or not.

   Notes:
   For methods with ill-scaled updates (SNESNRICHARDSON, SNESNCG), a small
   damping parameter may yield satisfactory but slow convergence despite
   the simplicity of the line search.

   Level: advanced

.keywords: SNES, SNESLineSearch, damping

.seealso: SNESLineSearchCreate(), SNESLineSearchSetType()
M*/
PETSC_EXTERN_C PetscErrorCode SNESLineSearchCreate_Basic(SNESLineSearch linesearch)
{
  PetscFunctionBegin;
  linesearch->ops->apply          = SNESLineSearchApply_Basic;
  linesearch->ops->destroy        = PETSC_NULL;
  linesearch->ops->setfromoptions = PETSC_NULL;
  linesearch->ops->reset          = PETSC_NULL;
  linesearch->ops->view           = PETSC_NULL;
  linesearch->ops->setup          = PETSC_NULL;
  PetscFunctionReturn(0);
}
