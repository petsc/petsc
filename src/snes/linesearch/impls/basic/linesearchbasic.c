#include <petsc/private/linesearchimpl.h>
#include <petsc/private/snesimpl.h>

static PetscErrorCode  SNESLineSearchApply_Basic(SNESLineSearch linesearch)
{
  PetscBool      changed_y, changed_w;
  Vec            X, F, Y, W;
  SNES           snes;
  PetscReal      gnorm, xnorm, ynorm, lambda;
  PetscBool      domainerror;

  PetscFunctionBegin;
  CHKERRQ(SNESLineSearchGetVecs(linesearch, &X, &F, &Y, &W, NULL));
  CHKERRQ(SNESLineSearchGetNorms(linesearch, &xnorm, &gnorm, &ynorm));
  CHKERRQ(SNESLineSearchGetLambda(linesearch, &lambda));
  CHKERRQ(SNESLineSearchGetSNES(linesearch, &snes));
  CHKERRQ(SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_SUCCEEDED));

  /* precheck */
  CHKERRQ(SNESLineSearchPreCheck(linesearch,X,Y,&changed_y));

  /* update */
  CHKERRQ(VecWAXPY(W,-lambda,Y,X));
  if (linesearch->ops->viproject) {
    CHKERRQ((*linesearch->ops->viproject)(snes, W));
  }

  /* postcheck */
  CHKERRQ(SNESLineSearchPostCheck(linesearch,X,Y,W,&changed_y,&changed_w));
  if (changed_y) {
    CHKERRQ(VecWAXPY(W,-lambda,Y,X));
    if (linesearch->ops->viproject) {
      CHKERRQ((*linesearch->ops->viproject)(snes, W));
    }
  }
  if (linesearch->norms || snes->iter < snes->max_its-1) {
    CHKERRQ((*linesearch->ops->snesfunc)(snes,W,F));
    CHKERRQ(SNESGetFunctionDomainError(snes, &domainerror));
    if (domainerror) {
      CHKERRQ(SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_FAILED_DOMAIN));
      PetscFunctionReturn(0);
    }
  }

  if (linesearch->norms) {
    if (!linesearch->ops->vinorm) CHKERRQ(VecNormBegin(F, NORM_2, &linesearch->fnorm));
    CHKERRQ(VecNormBegin(Y, NORM_2, &linesearch->ynorm));
    CHKERRQ(VecNormBegin(W, NORM_2, &linesearch->xnorm));
    if (!linesearch->ops->vinorm) CHKERRQ(VecNormEnd(F, NORM_2, &linesearch->fnorm));
    CHKERRQ(VecNormEnd(Y, NORM_2, &linesearch->ynorm));
    CHKERRQ(VecNormEnd(W, NORM_2, &linesearch->xnorm));

    if (linesearch->ops->vinorm) {
      linesearch->fnorm = gnorm;

      CHKERRQ((*linesearch->ops->vinorm)(snes, F, W, &linesearch->fnorm));
    } else {
      CHKERRQ(VecNorm(F,NORM_2,&linesearch->fnorm));
    }
  }

  /* copy the solution over */
  CHKERRQ(VecCopy(W, X));
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
