#include <petsc/private/linesearchimpl.h>
#include <petsc/private/snesimpl.h>

static PetscErrorCode SNESLineSearchApply_None(SNESLineSearch linesearch)
{
  PetscBool changed_y, changed_w;
  Vec       X, F, Y, W;
  SNES      snes;
  PetscReal gnorm, xnorm, ynorm, lambda, fnorm = 0.0;

  PetscFunctionBegin;
  PetscCall(SNESLineSearchGetVecs(linesearch, &X, &F, &Y, &W, NULL));
  PetscCall(SNESLineSearchGetNorms(linesearch, &xnorm, &gnorm, &ynorm));
  /* SNESLineSearchSetUp() sets linesearch->lambda to linesearch->damping */
  PetscCall(SNESLineSearchGetLambda(linesearch, &lambda));
  PetscCall(SNESLineSearchGetSNES(linesearch, &snes));
  PetscCall(SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_SUCCEEDED));

  /* precheck */
  PetscCall(SNESLineSearchPreCheck(linesearch, X, Y, &changed_y));

  /* update */
  PetscCall(VecWAXPY(W, -lambda, Y, X));
  if (linesearch->ops->viproject) PetscCall((*linesearch->ops->viproject)(snes, W));

  /* postcheck */
  PetscCall(SNESLineSearchPostCheck(linesearch, X, Y, W, &changed_y, &changed_w));
  if (changed_y) {
    if (!changed_w) PetscCall(VecWAXPY(W, -lambda, Y, X));
    if (linesearch->ops->viproject) PetscCall((*linesearch->ops->viproject)(snes, W));
  }
  if (linesearch->norms || snes->iter < snes->max_its - 1) {
    PetscCall((*linesearch->ops->snesfunc)(snes, W, F));
    PetscCall(VecNorm(F, NORM_2, &fnorm));
    SNESLineSearchCheckFunctionDomainError(snes, linesearch, fnorm);
  }
  if (linesearch->norms) {
    PetscCall(VecNormBegin(Y, NORM_2, &linesearch->ynorm));
    PetscCall(VecNormBegin(W, NORM_2, &linesearch->xnorm));
    PetscCall(VecNormEnd(Y, NORM_2, &linesearch->ynorm));
    PetscCall(VecNormEnd(W, NORM_2, &linesearch->xnorm));

    if (linesearch->ops->vinorm) {
      linesearch->fnorm = gnorm;

      PetscCall((*linesearch->ops->vinorm)(snes, F, W, &linesearch->fnorm));
    } else linesearch->fnorm = fnorm;
  }

  /* copy the solution over */
  PetscCall(VecCopy(W, X));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESLineSearchView_None(SNESLineSearch linesearch, PetscViewer viewer)
{
  PetscBool isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii && linesearch->damping != 1.) PetscCall(PetscViewerASCIIPrintf(viewer, "  damping factor: %g\n", (double)linesearch->damping));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   SNESLINESEARCHNONE - This line search implementation is not a line
   search at all; it simply uses the full step $x_{k+1} = x_k - \lambda Y_k$.
   $\lambda=1$ by default, but it can be configured to be a constant damping factor using `-snes_linesearch_damping` or `SNESLineSearchSetDamping()`.
   Thus, this routine is intended for methods with well-scaled updates; i.e. Newton's method (`SNESNEWTONLS`), on
   well-behaved problems.

   Options Database Keys:
+  -snes_linesearch_damping damping    - step length is scaled by this factor, i.e. this is the $\lambda$ above
-  -snes_linesearch_norms (true|false) - whether to compute norms or not (`SNESLineSearchSetComputeNorms()`)

   Level: advanced

   Note:
   For methods with ill-scaled updates (`SNESNRICHARDSON`, `SNESNCG`), a small
   damping parameter may yield satisfactory, but slow convergence, despite
   the lack of the line search.

.seealso: [](ch_snes), `SNES`, `SNESLineSearch`, `SNESLineSearchType`, `SNESGetLineSearch()`, `SNESLineSearchCreate()`, `SNESLineSearchSetType()`, `SNESLineSearchSetDamping()`, `SNESLineSearchSetComputeNorms()`
M*/
PETSC_EXTERN PetscErrorCode SNESLineSearchCreate_None(SNESLineSearch linesearch)
{
  PetscFunctionBegin;
  linesearch->ops->apply          = SNESLineSearchApply_None;
  linesearch->ops->destroy        = NULL;
  linesearch->ops->setfromoptions = NULL;
  linesearch->ops->reset          = NULL;
  linesearch->ops->view           = SNESLineSearchView_None;
  linesearch->ops->setup          = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}
