#include <petsc/private/linesearchimpl.h>
#include <petsc/private/snesimpl.h>

static PetscErrorCode  SNESLineSearchApply_Basic(SNESLineSearch linesearch)
{
  PetscBool      changed_y, changed_w;
  PetscErrorCode ierr;
  Vec            X, F, Y, W;
  SNES           snes;
  PetscReal      gnorm, xnorm, ynorm, lambda;
  PetscBool      domainerror;

  PetscFunctionBegin;
  ierr = SNESLineSearchGetVecs(linesearch, &X, &F, &Y, &W, NULL);CHKERRQ(ierr);
  ierr = SNESLineSearchGetNorms(linesearch, &xnorm, &gnorm, &ynorm);CHKERRQ(ierr);
  ierr = SNESLineSearchGetLambda(linesearch, &lambda);CHKERRQ(ierr);
  ierr = SNESLineSearchGetSNES(linesearch, &snes);CHKERRQ(ierr);
  ierr = SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_SUCCEEDED);CHKERRQ(ierr);

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
    ierr = (*linesearch->ops->snesfunc)(snes,W,F);CHKERRQ(ierr);
    ierr = SNESGetFunctionDomainError(snes, &domainerror);CHKERRQ(ierr);
    if (domainerror) {
      ierr = SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_FAILED_DOMAIN);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
  }

  if (linesearch->norms) {
    if (!linesearch->ops->vinorm) {ierr = VecNormBegin(F, NORM_2, &linesearch->fnorm);CHKERRQ(ierr);}
    ierr = VecNormBegin(Y, NORM_2, &linesearch->ynorm);CHKERRQ(ierr);
    ierr = VecNormBegin(W, NORM_2, &linesearch->xnorm);CHKERRQ(ierr);
    if (!linesearch->ops->vinorm) {ierr = VecNormEnd(F, NORM_2, &linesearch->fnorm);CHKERRQ(ierr);}
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

/*MC
   SNESLINESEARCHBASIC - This line search implementation is not a line
   search at all; it simply uses the full step.  Thus, this routine is intended
   for methods with well-scaled updates; i.e. Newton's method (SNESNEWTONLS), on
   well-behaved problems.

   Options Database Keys:
+   -snes_linesearch_damping <damping> - search vector is scaled by this amount, default is 1.0
-   -snes_linesearch_norms <flag> - whether to compute norms or not, default is true (SNESLineSearchSetComputeNorms())

   Notes:
   For methods with ill-scaled updates (SNESNRICHARDSON, SNESNCG), a small
   damping parameter may yield satisfactory but slow convergence despite
   the simplicity of the line search.

   Level: advanced

.seealso: SNESLineSearchCreate(), SNESLineSearchSetType(), SNESLineSearchSetDamping(), SNESLineSearchSetComputeNorms()
M*/
PETSC_EXTERN PetscErrorCode SNESLineSearchCreate_Basic(SNESLineSearch linesearch)
{
  PetscFunctionBegin;
  linesearch->ops->apply          = SNESLineSearchApply_Basic;
  linesearch->ops->destroy        = NULL;
  linesearch->ops->setfromoptions = NULL;
  linesearch->ops->reset          = NULL;
  linesearch->ops->view           = NULL;
  linesearch->ops->setup          = NULL;
  PetscFunctionReturn(0);
}
