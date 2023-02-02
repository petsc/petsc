#include <petsctaolinesearch.h>
#include <../src/tao/unconstrained/impls/cg/taocg.h>

#define CG_FletcherReeves   0
#define CG_PolakRibiere     1
#define CG_PolakRibierePlus 2
#define CG_HestenesStiefel  3
#define CG_DaiYuan          4
#define CG_Types            5

static const char *CG_Table[64] = {"fr", "pr", "prp", "hs", "dy"};

static PetscErrorCode TaoSolve_CG(Tao tao)
{
  TAO_CG                      *cgP       = (TAO_CG *)tao->data;
  TaoLineSearchConvergedReason ls_status = TAOLINESEARCH_CONTINUE_ITERATING;
  PetscReal                    step      = 1.0, f, gnorm, gnorm2, delta, gd, ginner, beta;
  PetscReal                    gd_old, gnorm2_old, f_old;

  PetscFunctionBegin;
  if (tao->XL || tao->XU || tao->ops->computebounds) PetscCall(PetscInfo(tao, "WARNING: Variable bounds have been set but will be ignored by cg algorithm\n"));

  /*  Check convergence criteria */
  PetscCall(TaoComputeObjectiveAndGradient(tao, tao->solution, &f, tao->gradient));
  PetscCall(VecNorm(tao->gradient, NORM_2, &gnorm));
  PetscCheck(!PetscIsInfOrNanReal(f) && !PetscIsInfOrNanReal(gnorm), PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "User provided compute function generated Inf or NaN");

  tao->reason = TAO_CONTINUE_ITERATING;
  PetscCall(TaoLogConvergenceHistory(tao, f, gnorm, 0.0, tao->ksp_its));
  PetscCall(TaoMonitor(tao, tao->niter, f, gnorm, 0.0, step));
  PetscUseTypeMethod(tao, convergencetest, tao->cnvP);
  if (tao->reason != TAO_CONTINUE_ITERATING) PetscFunctionReturn(PETSC_SUCCESS);

  /*  Set initial direction to -gradient */
  PetscCall(VecCopy(tao->gradient, tao->stepdirection));
  PetscCall(VecScale(tao->stepdirection, -1.0));
  gnorm2 = gnorm * gnorm;

  /*  Set initial scaling for the function */
  if (f != 0.0) {
    delta = 2.0 * PetscAbsScalar(f) / gnorm2;
    delta = PetscMax(delta, cgP->delta_min);
    delta = PetscMin(delta, cgP->delta_max);
  } else {
    delta = 2.0 / gnorm2;
    delta = PetscMax(delta, cgP->delta_min);
    delta = PetscMin(delta, cgP->delta_max);
  }
  /*  Set counter for gradient and reset steps */
  cgP->ngradsteps  = 0;
  cgP->nresetsteps = 0;

  while (1) {
    /* Call general purpose update function */
    PetscTryTypeMethod(tao, update, tao->niter, tao->user_update);

    /*  Save the current gradient information */
    f_old      = f;
    gnorm2_old = gnorm2;
    PetscCall(VecCopy(tao->solution, cgP->X_old));
    PetscCall(VecCopy(tao->gradient, cgP->G_old));
    PetscCall(VecDot(tao->gradient, tao->stepdirection, &gd));
    if ((gd >= 0) || PetscIsInfOrNanReal(gd)) {
      ++cgP->ngradsteps;
      if (f != 0.0) {
        delta = 2.0 * PetscAbsScalar(f) / gnorm2;
        delta = PetscMax(delta, cgP->delta_min);
        delta = PetscMin(delta, cgP->delta_max);
      } else {
        delta = 2.0 / gnorm2;
        delta = PetscMax(delta, cgP->delta_min);
        delta = PetscMin(delta, cgP->delta_max);
      }

      PetscCall(VecCopy(tao->gradient, tao->stepdirection));
      PetscCall(VecScale(tao->stepdirection, -1.0));
    }

    /*  Search direction for improving point */
    PetscCall(TaoLineSearchSetInitialStepLength(tao->linesearch, delta));
    PetscCall(TaoLineSearchApply(tao->linesearch, tao->solution, &f, tao->gradient, tao->stepdirection, &step, &ls_status));
    PetscCall(TaoAddLineSearchCounts(tao));
    if (ls_status != TAOLINESEARCH_SUCCESS && ls_status != TAOLINESEARCH_SUCCESS_USER) {
      /*  Linesearch failed */
      /*  Reset factors and use scaled gradient step */
      ++cgP->nresetsteps;
      f      = f_old;
      gnorm2 = gnorm2_old;
      PetscCall(VecCopy(cgP->X_old, tao->solution));
      PetscCall(VecCopy(cgP->G_old, tao->gradient));

      if (f != 0.0) {
        delta = 2.0 * PetscAbsScalar(f) / gnorm2;
        delta = PetscMax(delta, cgP->delta_min);
        delta = PetscMin(delta, cgP->delta_max);
      } else {
        delta = 2.0 / gnorm2;
        delta = PetscMax(delta, cgP->delta_min);
        delta = PetscMin(delta, cgP->delta_max);
      }

      PetscCall(VecCopy(tao->gradient, tao->stepdirection));
      PetscCall(VecScale(tao->stepdirection, -1.0));

      PetscCall(TaoLineSearchSetInitialStepLength(tao->linesearch, delta));
      PetscCall(TaoLineSearchApply(tao->linesearch, tao->solution, &f, tao->gradient, tao->stepdirection, &step, &ls_status));
      PetscCall(TaoAddLineSearchCounts(tao));

      if (ls_status != TAOLINESEARCH_SUCCESS && ls_status != TAOLINESEARCH_SUCCESS_USER) {
        /*  Linesearch failed again */
        /*  switch to unscaled gradient */
        f = f_old;
        PetscCall(VecCopy(cgP->X_old, tao->solution));
        PetscCall(VecCopy(cgP->G_old, tao->gradient));
        delta = 1.0;
        PetscCall(VecCopy(tao->solution, tao->stepdirection));
        PetscCall(VecScale(tao->stepdirection, -1.0));

        PetscCall(TaoLineSearchSetInitialStepLength(tao->linesearch, delta));
        PetscCall(TaoLineSearchApply(tao->linesearch, tao->solution, &f, tao->gradient, tao->stepdirection, &step, &ls_status));
        PetscCall(TaoAddLineSearchCounts(tao));
        if (ls_status != TAOLINESEARCH_SUCCESS && ls_status != TAOLINESEARCH_SUCCESS_USER) {
          /*  Line search failed for last time -- give up */
          f = f_old;
          PetscCall(VecCopy(cgP->X_old, tao->solution));
          PetscCall(VecCopy(cgP->G_old, tao->gradient));
          step        = 0.0;
          tao->reason = TAO_DIVERGED_LS_FAILURE;
        }
      }
    }

    /*  Check for bad value */
    PetscCall(VecNorm(tao->gradient, NORM_2, &gnorm));
    PetscCheck(!PetscIsInfOrNanReal(f) && !PetscIsInfOrNanReal(gnorm), PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "User-provided compute function generated Inf or NaN");

    /*  Check for termination */
    gnorm2 = gnorm * gnorm;
    tao->niter++;
    PetscCall(TaoLogConvergenceHistory(tao, f, gnorm, 0.0, tao->ksp_its));
    PetscCall(TaoMonitor(tao, tao->niter, f, gnorm, 0.0, step));
    PetscUseTypeMethod(tao, convergencetest, tao->cnvP);
    if (tao->reason != TAO_CONTINUE_ITERATING) break;

    /*  Check for restart condition */
    PetscCall(VecDot(tao->gradient, cgP->G_old, &ginner));
    if (PetscAbsScalar(ginner) >= cgP->eta * gnorm2) {
      /*  Gradients far from orthogonal; use steepest descent direction */
      beta = 0.0;
    } else {
      /*  Gradients close to orthogonal; use conjugate gradient formula */
      switch (cgP->cg_type) {
      case CG_FletcherReeves:
        beta = gnorm2 / gnorm2_old;
        break;

      case CG_PolakRibiere:
        beta = (gnorm2 - ginner) / gnorm2_old;
        break;

      case CG_PolakRibierePlus:
        beta = PetscMax((gnorm2 - ginner) / gnorm2_old, 0.0);
        break;

      case CG_HestenesStiefel:
        PetscCall(VecDot(tao->gradient, tao->stepdirection, &gd));
        PetscCall(VecDot(cgP->G_old, tao->stepdirection, &gd_old));
        beta = (gnorm2 - ginner) / (gd - gd_old);
        break;

      case CG_DaiYuan:
        PetscCall(VecDot(tao->gradient, tao->stepdirection, &gd));
        PetscCall(VecDot(cgP->G_old, tao->stepdirection, &gd_old));
        beta = gnorm2 / (gd - gd_old);
        break;

      default:
        beta = 0.0;
        break;
      }
    }

    /*  Compute the direction d=-g + beta*d */
    PetscCall(VecAXPBY(tao->stepdirection, -1.0, beta, tao->gradient));

    /*  update initial steplength choice */
    delta = 1.0;
    delta = PetscMax(delta, cgP->delta_min);
    delta = PetscMin(delta, cgP->delta_max);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoSetUp_CG(Tao tao)
{
  TAO_CG *cgP = (TAO_CG *)tao->data;

  PetscFunctionBegin;
  if (!tao->gradient) PetscCall(VecDuplicate(tao->solution, &tao->gradient));
  if (!tao->stepdirection) PetscCall(VecDuplicate(tao->solution, &tao->stepdirection));
  if (!cgP->X_old) PetscCall(VecDuplicate(tao->solution, &cgP->X_old));
  if (!cgP->G_old) PetscCall(VecDuplicate(tao->gradient, &cgP->G_old));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoDestroy_CG(Tao tao)
{
  TAO_CG *cgP = (TAO_CG *)tao->data;

  PetscFunctionBegin;
  if (tao->setupcalled) {
    PetscCall(VecDestroy(&cgP->X_old));
    PetscCall(VecDestroy(&cgP->G_old));
  }
  PetscCall(TaoLineSearchDestroy(&tao->linesearch));
  PetscCall(PetscFree(tao->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoSetFromOptions_CG(Tao tao, PetscOptionItems *PetscOptionsObject)
{
  TAO_CG *cgP = (TAO_CG *)tao->data;

  PetscFunctionBegin;
  PetscCall(TaoLineSearchSetFromOptions(tao->linesearch));
  PetscOptionsHeadBegin(PetscOptionsObject, "Nonlinear Conjugate Gradient method for unconstrained optimization");
  PetscCall(PetscOptionsReal("-tao_cg_eta", "restart tolerance", "", cgP->eta, &cgP->eta, NULL));
  PetscCall(PetscOptionsEList("-tao_cg_type", "cg formula", "", CG_Table, CG_Types, CG_Table[cgP->cg_type], &cgP->cg_type, NULL));
  PetscCall(PetscOptionsReal("-tao_cg_delta_min", "minimum delta value", "", cgP->delta_min, &cgP->delta_min, NULL));
  PetscCall(PetscOptionsReal("-tao_cg_delta_max", "maximum delta value", "", cgP->delta_max, &cgP->delta_max, NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoView_CG(Tao tao, PetscViewer viewer)
{
  PetscBool isascii;
  TAO_CG   *cgP = (TAO_CG *)tao->data;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer, "CG Type: %s\n", CG_Table[cgP->cg_type]));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Gradient steps: %" PetscInt_FMT "\n", cgP->ngradsteps));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Reset steps: %" PetscInt_FMT "\n", cgP->nresetsteps));
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
     TAOCG -   Nonlinear conjugate gradient method is an extension of the
nonlinear conjugate gradient solver for nonlinear optimization.

   Options Database Keys:
+      -tao_cg_eta <r> - restart tolerance
.      -tao_cg_type <taocg_type> - cg formula
.      -tao_cg_delta_min <r> - minimum delta value
-      -tao_cg_delta_max <r> - maximum delta value

  Notes:
     CG formulas are:
         "fr" - Fletcher-Reeves
         "pr" - Polak-Ribiere
         "prp" - Polak-Ribiere-Plus
         "hs" - Hestenes-Steifel
         "dy" - Dai-Yuan
  Level: beginner
M*/

PETSC_EXTERN PetscErrorCode TaoCreate_CG(Tao tao)
{
  TAO_CG     *cgP;
  const char *morethuente_type = TAOLINESEARCHMT;

  PetscFunctionBegin;
  tao->ops->setup          = TaoSetUp_CG;
  tao->ops->solve          = TaoSolve_CG;
  tao->ops->view           = TaoView_CG;
  tao->ops->setfromoptions = TaoSetFromOptions_CG;
  tao->ops->destroy        = TaoDestroy_CG;

  /* Override default settings (unless already changed) */
  if (!tao->max_it_changed) tao->max_it = 2000;
  if (!tao->max_funcs_changed) tao->max_funcs = 4000;

  /*  Note: nondefault values should be used for nonlinear conjugate gradient  */
  /*  method.  In particular, gtol should be less that 0.5; the value used in  */
  /*  Nocedal and Wright is 0.10.  We use the default values for the  */
  /*  linesearch because it seems to work better. */
  PetscCall(TaoLineSearchCreate(((PetscObject)tao)->comm, &tao->linesearch));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)tao->linesearch, (PetscObject)tao, 1));
  PetscCall(TaoLineSearchSetType(tao->linesearch, morethuente_type));
  PetscCall(TaoLineSearchUseTaoRoutines(tao->linesearch, tao));
  PetscCall(TaoLineSearchSetOptionsPrefix(tao->linesearch, tao->hdr.prefix));

  PetscCall(PetscNew(&cgP));
  tao->data      = (void *)cgP;
  cgP->eta       = 0.1;
  cgP->delta_min = 1e-7;
  cgP->delta_max = 100;
  cgP->cg_type   = CG_PolakRibierePlus;
  PetscFunctionReturn(PETSC_SUCCESS);
}
