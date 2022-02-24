#include <petsctaolinesearch.h>
#include <../src/tao/unconstrained/impls/cg/taocg.h>

#define CG_FletcherReeves       0
#define CG_PolakRibiere         1
#define CG_PolakRibierePlus     2
#define CG_HestenesStiefel      3
#define CG_DaiYuan              4
#define CG_Types                5

static const char *CG_Table[64] = {"fr", "pr", "prp", "hs", "dy"};

static PetscErrorCode TaoSolve_CG(Tao tao)
{
  TAO_CG                       *cgP = (TAO_CG*)tao->data;
  TaoLineSearchConvergedReason ls_status = TAOLINESEARCH_CONTINUE_ITERATING;
  PetscReal                    step=1.0,f,gnorm,gnorm2,delta,gd,ginner,beta;
  PetscReal                    gd_old,gnorm2_old,f_old;

  PetscFunctionBegin;
  if (tao->XL || tao->XU || tao->ops->computebounds) {
    CHKERRQ(PetscInfo(tao,"WARNING: Variable bounds have been set but will be ignored by cg algorithm\n"));
  }

  /*  Check convergence criteria */
  CHKERRQ(TaoComputeObjectiveAndGradient(tao, tao->solution, &f, tao->gradient));
  CHKERRQ(VecNorm(tao->gradient,NORM_2,&gnorm));
  PetscCheck(!PetscIsInfOrNanReal(f) && !PetscIsInfOrNanReal(gnorm),PetscObjectComm((PetscObject)tao),PETSC_ERR_USER, "User provided compute function generated Inf or NaN");

  tao->reason = TAO_CONTINUE_ITERATING;
  CHKERRQ(TaoLogConvergenceHistory(tao,f,gnorm,0.0,tao->ksp_its));
  CHKERRQ(TaoMonitor(tao,tao->niter,f,gnorm,0.0,step));
  CHKERRQ((*tao->ops->convergencetest)(tao,tao->cnvP));
  if (tao->reason != TAO_CONTINUE_ITERATING) PetscFunctionReturn(0);

  /*  Set initial direction to -gradient */
  CHKERRQ(VecCopy(tao->gradient, tao->stepdirection));
  CHKERRQ(VecScale(tao->stepdirection, -1.0));
  gnorm2 = gnorm*gnorm;

  /*  Set initial scaling for the function */
  if (f != 0.0) {
    delta = 2.0*PetscAbsScalar(f) / gnorm2;
    delta = PetscMax(delta,cgP->delta_min);
    delta = PetscMin(delta,cgP->delta_max);
  } else {
    delta = 2.0 / gnorm2;
    delta = PetscMax(delta,cgP->delta_min);
    delta = PetscMin(delta,cgP->delta_max);
  }
  /*  Set counter for gradient and reset steps */
  cgP->ngradsteps = 0;
  cgP->nresetsteps = 0;

  while (1) {
    /* Call general purpose update function */
    if (tao->ops->update) {
      CHKERRQ((*tao->ops->update)(tao, tao->niter, tao->user_update));
    }

    /*  Save the current gradient information */
    f_old = f;
    gnorm2_old = gnorm2;
    CHKERRQ(VecCopy(tao->solution, cgP->X_old));
    CHKERRQ(VecCopy(tao->gradient, cgP->G_old));
    CHKERRQ(VecDot(tao->gradient, tao->stepdirection, &gd));
    if ((gd >= 0) || PetscIsInfOrNanReal(gd)) {
      ++cgP->ngradsteps;
      if (f != 0.0) {
        delta = 2.0*PetscAbsScalar(f) / gnorm2;
        delta = PetscMax(delta,cgP->delta_min);
        delta = PetscMin(delta,cgP->delta_max);
      } else {
        delta = 2.0 / gnorm2;
        delta = PetscMax(delta,cgP->delta_min);
        delta = PetscMin(delta,cgP->delta_max);
      }

      CHKERRQ(VecCopy(tao->gradient, tao->stepdirection));
      CHKERRQ(VecScale(tao->stepdirection, -1.0));
    }

    /*  Search direction for improving point */
    CHKERRQ(TaoLineSearchSetInitialStepLength(tao->linesearch,delta));
    CHKERRQ(TaoLineSearchApply(tao->linesearch, tao->solution, &f, tao->gradient, tao->stepdirection, &step, &ls_status));
    CHKERRQ(TaoAddLineSearchCounts(tao));
    if (ls_status != TAOLINESEARCH_SUCCESS && ls_status != TAOLINESEARCH_SUCCESS_USER) {
      /*  Linesearch failed */
      /*  Reset factors and use scaled gradient step */
      ++cgP->nresetsteps;
      f = f_old;
      gnorm2 = gnorm2_old;
      CHKERRQ(VecCopy(cgP->X_old, tao->solution));
      CHKERRQ(VecCopy(cgP->G_old, tao->gradient));

      if (f != 0.0) {
        delta = 2.0*PetscAbsScalar(f) / gnorm2;
        delta = PetscMax(delta,cgP->delta_min);
        delta = PetscMin(delta,cgP->delta_max);
      } else {
        delta = 2.0 / gnorm2;
        delta = PetscMax(delta,cgP->delta_min);
        delta = PetscMin(delta,cgP->delta_max);
      }

      CHKERRQ(VecCopy(tao->gradient, tao->stepdirection));
      CHKERRQ(VecScale(tao->stepdirection, -1.0));

      CHKERRQ(TaoLineSearchSetInitialStepLength(tao->linesearch,delta));
      CHKERRQ(TaoLineSearchApply(tao->linesearch, tao->solution, &f, tao->gradient, tao->stepdirection, &step, &ls_status));
      CHKERRQ(TaoAddLineSearchCounts(tao));

      if (ls_status != TAOLINESEARCH_SUCCESS && ls_status != TAOLINESEARCH_SUCCESS_USER) {
        /*  Linesearch failed again */
        /*  switch to unscaled gradient */
        f = f_old;
        CHKERRQ(VecCopy(cgP->X_old, tao->solution));
        CHKERRQ(VecCopy(cgP->G_old, tao->gradient));
        delta = 1.0;
        CHKERRQ(VecCopy(tao->solution, tao->stepdirection));
        CHKERRQ(VecScale(tao->stepdirection, -1.0));

        CHKERRQ(TaoLineSearchSetInitialStepLength(tao->linesearch,delta));
        CHKERRQ(TaoLineSearchApply(tao->linesearch, tao->solution, &f, tao->gradient, tao->stepdirection, &step, &ls_status));
        CHKERRQ(TaoAddLineSearchCounts(tao));
        if (ls_status != TAOLINESEARCH_SUCCESS && ls_status != TAOLINESEARCH_SUCCESS_USER) {

          /*  Line search failed for last time -- give up */
          f = f_old;
          CHKERRQ(VecCopy(cgP->X_old, tao->solution));
          CHKERRQ(VecCopy(cgP->G_old, tao->gradient));
          step = 0.0;
          tao->reason = TAO_DIVERGED_LS_FAILURE;
        }
      }
    }

    /*  Check for bad value */
    CHKERRQ(VecNorm(tao->gradient,NORM_2,&gnorm));
    PetscCheck(!PetscIsInfOrNanReal(f) && !PetscIsInfOrNanReal(gnorm),PetscObjectComm((PetscObject)tao),PETSC_ERR_USER,"User-provided compute function generated Inf or NaN");

    /*  Check for termination */
    gnorm2 =gnorm * gnorm;
    tao->niter++;
    CHKERRQ(TaoLogConvergenceHistory(tao,f,gnorm,0.0,tao->ksp_its));
    CHKERRQ(TaoMonitor(tao,tao->niter,f,gnorm,0.0,step));
    CHKERRQ((*tao->ops->convergencetest)(tao,tao->cnvP));
    if (tao->reason != TAO_CONTINUE_ITERATING) {
      break;
    }

    /*  Check for restart condition */
    CHKERRQ(VecDot(tao->gradient, cgP->G_old, &ginner));
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
        beta = PetscMax((gnorm2-ginner)/gnorm2_old, 0.0);
        break;

      case CG_HestenesStiefel:
        CHKERRQ(VecDot(tao->gradient, tao->stepdirection, &gd));
        CHKERRQ(VecDot(cgP->G_old, tao->stepdirection, &gd_old));
        beta = (gnorm2 - ginner) / (gd - gd_old);
        break;

      case CG_DaiYuan:
        CHKERRQ(VecDot(tao->gradient, tao->stepdirection, &gd));
        CHKERRQ(VecDot(cgP->G_old, tao->stepdirection, &gd_old));
        beta = gnorm2 / (gd - gd_old);
        break;

      default:
        beta = 0.0;
        break;
      }
    }

    /*  Compute the direction d=-g + beta*d */
    CHKERRQ(VecAXPBY(tao->stepdirection, -1.0, beta, tao->gradient));

    /*  update initial steplength choice */
    delta = 1.0;
    delta = PetscMax(delta, cgP->delta_min);
    delta = PetscMin(delta, cgP->delta_max);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSetUp_CG(Tao tao)
{
  TAO_CG         *cgP = (TAO_CG*)tao->data;

  PetscFunctionBegin;
  if (!tao->gradient) CHKERRQ(VecDuplicate(tao->solution,&tao->gradient));
  if (!tao->stepdirection) CHKERRQ(VecDuplicate(tao->solution,&tao->stepdirection));
  if (!cgP->X_old) CHKERRQ(VecDuplicate(tao->solution,&cgP->X_old));
  if (!cgP->G_old) CHKERRQ(VecDuplicate(tao->gradient,&cgP->G_old));
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoDestroy_CG(Tao tao)
{
  TAO_CG         *cgP = (TAO_CG*) tao->data;

  PetscFunctionBegin;
  if (tao->setupcalled) {
    CHKERRQ(VecDestroy(&cgP->X_old));
    CHKERRQ(VecDestroy(&cgP->G_old));
  }
  CHKERRQ(TaoLineSearchDestroy(&tao->linesearch));
  CHKERRQ(PetscFree(tao->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSetFromOptions_CG(PetscOptionItems *PetscOptionsObject,Tao tao)
{
  TAO_CG         *cgP = (TAO_CG*)tao->data;

  PetscFunctionBegin;
  CHKERRQ(TaoLineSearchSetFromOptions(tao->linesearch));
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"Nonlinear Conjugate Gradient method for unconstrained optimization"));
  CHKERRQ(PetscOptionsReal("-tao_cg_eta","restart tolerance", "", cgP->eta,&cgP->eta,NULL));
  CHKERRQ(PetscOptionsEList("-tao_cg_type","cg formula", "", CG_Table, CG_Types, CG_Table[cgP->cg_type], &cgP->cg_type,NULL));
  CHKERRQ(PetscOptionsReal("-tao_cg_delta_min","minimum delta value", "", cgP->delta_min,&cgP->delta_min,NULL));
  CHKERRQ(PetscOptionsReal("-tao_cg_delta_max","maximum delta value", "", cgP->delta_max,&cgP->delta_max,NULL));
  CHKERRQ(PetscOptionsTail());
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoView_CG(Tao tao, PetscViewer viewer)
{
  PetscBool      isascii;
  TAO_CG         *cgP = (TAO_CG*)tao->data;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    CHKERRQ(PetscViewerASCIIPushTab(viewer));
    CHKERRQ(PetscViewerASCIIPrintf(viewer, "CG Type: %s\n", CG_Table[cgP->cg_type]));
    CHKERRQ(PetscViewerASCIIPrintf(viewer, "Gradient steps: %D\n", cgP->ngradsteps));
    CHKERRQ(PetscViewerASCIIPrintf(viewer, "Reset steps: %D\n", cgP->nresetsteps));
    CHKERRQ(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(0);
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
  TAO_CG         *cgP;
  const char     *morethuente_type = TAOLINESEARCHMT;

  PetscFunctionBegin;
  tao->ops->setup = TaoSetUp_CG;
  tao->ops->solve = TaoSolve_CG;
  tao->ops->view = TaoView_CG;
  tao->ops->setfromoptions = TaoSetFromOptions_CG;
  tao->ops->destroy = TaoDestroy_CG;

  /* Override default settings (unless already changed) */
  if (!tao->max_it_changed) tao->max_it = 2000;
  if (!tao->max_funcs_changed) tao->max_funcs = 4000;

  /*  Note: nondefault values should be used for nonlinear conjugate gradient  */
  /*  method.  In particular, gtol should be less that 0.5; the value used in  */
  /*  Nocedal and Wright is 0.10.  We use the default values for the  */
  /*  linesearch because it seems to work better. */
  CHKERRQ(TaoLineSearchCreate(((PetscObject)tao)->comm, &tao->linesearch));
  CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)tao->linesearch, (PetscObject)tao, 1));
  CHKERRQ(TaoLineSearchSetType(tao->linesearch, morethuente_type));
  CHKERRQ(TaoLineSearchUseTaoRoutines(tao->linesearch, tao));
  CHKERRQ(TaoLineSearchSetOptionsPrefix(tao->linesearch,tao->hdr.prefix));

  CHKERRQ(PetscNewLog(tao,&cgP));
  tao->data = (void*)cgP;
  cgP->eta = 0.1;
  cgP->delta_min = 1e-7;
  cgP->delta_max = 100;
  cgP->cg_type = CG_PolakRibierePlus;
  PetscFunctionReturn(0);
}
