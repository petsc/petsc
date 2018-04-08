#include <petsctaolinesearch.h>
#include <../src/tao/bound/impls/bncg/bncg.h>

#define CG_FletcherReeves       0
#define CG_PolakRibiere         1
#define CG_PolakRibierePlus     2
#define CG_HestenesStiefel      3
#define CG_DaiYuan              4
#define CG_Types                5

static const char *CG_Table[64] = {"fr", "pr", "prp", "hs", "dy"};

PetscErrorCode TaoBNCGResetStepForNewInactives(Tao tao, Vec step) 
{
  TAO_BNCG                     *cg = (TAO_BNCG*)tao->data;
  PetscErrorCode               ierr;
  const PetscScalar            *xl, *xo, *xn, *xu, *gn, *go;
  PetscInt                     size, i;
  PetscScalar                  *s;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(tao->solution, &size);CHKERRQ(ierr);
  ierr = VecGetArrayRead(cg->unprojected_gradient_old, &go);CHKERRQ(ierr);
  ierr = VecGetArrayRead(cg->unprojected_gradient, &gn);CHKERRQ(ierr);
  ierr = VecGetArrayRead(cg->X_old, &xo);CHKERRQ(ierr);
  ierr = VecGetArrayRead(tao->solution, &xn);CHKERRQ(ierr);
  ierr = VecGetArrayRead(tao->XL, &xl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(tao->XU, &xu);CHKERRQ(ierr);
  ierr = VecGetArray(step, &s);CHKERRQ(ierr);
  for (i=0; i<size; i++) {
    if (xl[i] == xu[i]) {
      s[i] = 0.0;
    } else {
      if (xl[i] > PETSC_NINFINITY) {
        if ((xn[i] == xl[i] && gn[i] < 0.0) && (xo[i] == xl[i] && go[i] >= 0.0)) {
          s[i] = -gn[i];
        }
      }
      if (xu[i] < PETSC_NINFINITY) {
        if ((xn[i] == xu[i] && gn[i] > 0.0) && (xo[i] == xu[i] && go[i] <= 0.0)) {
          s[i] = -gn[i];
        }
      }
    }
  }
  ierr = VecRestoreArrayRead(cg->unprojected_gradient_old, &go);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(cg->unprojected_gradient, &gn);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(cg->X_old, &xo);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(tao->solution, &xn);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(tao->XL, &xl);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(tao->XU, &xu);CHKERRQ(ierr);
  ierr = VecRestoreArray(step, &s);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSolve_BNCG(Tao tao)
{
  TAO_BNCG                     *cg = (TAO_BNCG*)tao->data;
  PetscErrorCode               ierr;
  TaoLineSearchConvergedReason ls_status = TAOLINESEARCH_CONTINUE_ITERATING;
  PetscReal                    step=1.0,f,gnorm,gnorm2,delta,gd,ginner,beta,dnorm;
  PetscReal                    gd_old,gnorm2_old,f_old;
  PetscBool                    cg_restart;

  PetscFunctionBegin;
  /*   Project the current point onto the feasible set */
  ierr = TaoComputeVariableBounds(tao);CHKERRQ(ierr);
  ierr = TaoLineSearchSetVariableBounds(tao->linesearch,tao->XL,tao->XU);CHKERRQ(ierr);
  
  /* Project the initial point onto the feasible region */
  ierr = VecMedian(tao->XL,tao->solution,tao->XU,tao->solution);CHKERRQ(ierr);

  /*  Compute the objective function and criteria */
  ierr = TaoComputeObjectiveAndGradient(tao, tao->solution, &f, cg->unprojected_gradient);CHKERRQ(ierr);
  ierr = VecNorm(cg->unprojected_gradient,NORM_2,&gnorm);CHKERRQ(ierr);
  if (PetscIsInfOrNanReal(f) || PetscIsInfOrNanReal(gnorm)) SETERRQ(PETSC_COMM_SELF,1, "User provided compute function generated Inf or NaN");

  /* Project the gradient and calculate the norm */
  ierr = VecBoundGradientProjection(cg->unprojected_gradient,tao->solution,tao->XL,tao->XU,tao->gradient);CHKERRQ(ierr);
  ierr = VecNorm(tao->gradient,NORM_2,&gnorm);CHKERRQ(ierr);
  gnorm2 = gnorm*gnorm;
  
  /* Convergence check */
  tao->reason = TAO_CONTINUE_ITERATING;
  ierr = TaoLogConvergenceHistory(tao, f, gnorm, 0.0, tao->ksp_its);CHKERRQ(ierr);
  ierr = TaoMonitor(tao, tao->niter, f, gnorm, 0.0, step);CHKERRQ(ierr);
  ierr = (*tao->ops->convergencetest)(tao,tao->cnvP);CHKERRQ(ierr);
  if (tao->reason != TAO_CONTINUE_ITERATING) PetscFunctionReturn(0);
  
  /* Start optimization iterations */
  f_old = f;
  gnorm2_old = gnorm2;
  ierr = VecCopy(tao->solution, cg->X_old);CHKERRQ(ierr);
  ierr = VecCopy(tao->gradient, cg->G_old);CHKERRQ(ierr);
  ierr = VecCopy(cg->unprojected_gradient, cg->unprojected_gradient_old);CHKERRQ(ierr);
  tao->niter = cg->ls_fails = cg->broken_ortho = cg->descent_error = 0;
  cg->resets = -1;
  while (tao->reason == TAO_CONTINUE_ITERATING) {
    /* Check restart conditions for using steepest descent */
    cg_restart = PETSC_FALSE;
    ierr = VecDot(tao->gradient, cg->G_old, &ginner);CHKERRQ(ierr);
    if (tao->niter == 0) {
      /* 1) First iteration */
      cg_restart = PETSC_TRUE;
    } else if (PetscAbsScalar(ginner) >= cg->eta * gnorm2) {
      /* 2) Gradients are far from orthogonal */
      cg_restart = PETSC_TRUE;
      cg->broken_ortho++;
    }
    
    /* Compute CG step */
    if (cg_restart) {
      beta = 0.0;
      cg->resets++;
    } else {
      switch (cg->cg_type) {
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
        ierr = VecDot(tao->gradient, tao->stepdirection, &gd);CHKERRQ(ierr);
        ierr = VecDot(cg->G_old, tao->stepdirection, &gd_old);CHKERRQ(ierr);
        beta = (gnorm2 - ginner) / (gd - gd_old);
        break;

      case CG_DaiYuan:
        ierr = VecDot(tao->gradient, tao->stepdirection, &gd);CHKERRQ(ierr);
        ierr = VecDot(cg->G_old, tao->stepdirection, &gd_old);CHKERRQ(ierr);
        beta = gnorm2 / (gd - gd_old);
        break;

      default:
        beta = 0.0;
        break;
      }
    }
    
    /*  Compute the direction d=-g + beta*d */
    ierr = VecAXPBY(tao->stepdirection, -1.0, beta, tao->gradient);CHKERRQ(ierr);
    ierr = TaoBNCGResetStepForNewInactives(tao, tao->stepdirection);CHKERRQ(ierr);
    
    /* Verify that this is a descent direction */
    ierr = VecDot(tao->gradient, tao->stepdirection, &gd);CHKERRQ(ierr);
    ierr = VecNorm(tao->stepdirection, NORM_2, &dnorm);
    if (gd > -cg->rho*PetscPowReal(dnorm, cg->pow)) {
      /* Not a descent direction, so we reset back to projected gradient descent */
      ierr = VecAXPBY(tao->stepdirection, -1.0, 0.0, tao->gradient);CHKERRQ(ierr);
      cg->resets++;
      cg->descent_error++;
    }
    
    /*  update initial steplength choice */
    delta = 1.0;
    delta = PetscMax(delta, cg->delta_min);
    delta = PetscMin(delta, cg->delta_max);
    
    /* Store solution and gradient info before it changes */
    ierr = VecCopy(tao->solution, cg->X_old);CHKERRQ(ierr);
    ierr = VecCopy(tao->gradient, cg->G_old);CHKERRQ(ierr);
    ierr = VecCopy(cg->unprojected_gradient, cg->unprojected_gradient_old);CHKERRQ(ierr);
    gnorm2_old = gnorm2;
    f_old = f;
    
    /* Perform bounded line search */
    ierr = TaoLineSearchSetInitialStepLength(tao->linesearch,delta);CHKERRQ(ierr);
    ierr = TaoLineSearchApply(tao->linesearch, tao->solution, &f, cg->unprojected_gradient, tao->stepdirection, &step, &ls_status);CHKERRQ(ierr);
    ierr = TaoAddLineSearchCounts(tao);CHKERRQ(ierr);
    
    /*  Check linesearch failure */
    if (ls_status != TAOLINESEARCH_SUCCESS && ls_status != TAOLINESEARCH_SUCCESS_USER) {
      cg->ls_fails++;
      /* Restore previous point */
      gnorm2 = gnorm2_old;
      f = f_old;
      ierr = VecCopy(cg->X_old, tao->solution);CHKERRQ(ierr);
      ierr = VecCopy(cg->G_old, tao->gradient);CHKERRQ(ierr);
      ierr = VecCopy(cg->unprojected_gradient_old, cg->unprojected_gradient);CHKERRQ(ierr);
      
      /* Fall back on the unscaled gradient step */
      delta = 1.0;
      ierr = VecCopy(tao->solution, tao->stepdirection);CHKERRQ(ierr);
      ierr = VecScale(tao->stepdirection, -1.0);CHKERRQ(ierr);
      
      ierr = TaoLineSearchSetInitialStepLength(tao->linesearch,delta);CHKERRQ(ierr);
      ierr = TaoLineSearchApply(tao->linesearch, tao->solution, &f, cg->unprojected_gradient, tao->stepdirection, &step, &ls_status);CHKERRQ(ierr);
      ierr = TaoAddLineSearchCounts(tao);CHKERRQ(ierr);
        
      if (ls_status != TAOLINESEARCH_SUCCESS && ls_status != TAOLINESEARCH_SUCCESS_USER){
        cg->ls_fails++;
        /* Restore previous point */
        gnorm2 = gnorm2_old;
        f = f_old;
        ierr = VecCopy(cg->X_old, tao->solution);CHKERRQ(ierr);
        ierr = VecCopy(cg->G_old, tao->gradient);CHKERRQ(ierr);
        ierr = VecCopy(cg->unprojected_gradient_old, cg->unprojected_gradient);CHKERRQ(ierr);
        
        /* Nothing left to do but fail out of the optimization */
        step = 0.0;
        tao->reason = TAO_DIVERGED_LS_FAILURE;
      }
    }

    /* Compute the projected gradient and its norm */
    ierr = VecBoundGradientProjection(cg->unprojected_gradient,tao->solution,tao->XL,tao->XU,tao->gradient);CHKERRQ(ierr);
    ierr = VecNorm(tao->gradient,NORM_2,&gnorm);CHKERRQ(ierr);
    gnorm2 = gnorm*gnorm;
    
    /* Convergence test */
    tao->niter++;
    ierr = TaoLogConvergenceHistory(tao, f, gnorm, 0.0, tao->ksp_its);CHKERRQ(ierr);
    ierr = TaoMonitor(tao, tao->niter, f, gnorm, 0.0, step);CHKERRQ(ierr);
    ierr = (*tao->ops->convergencetest)(tao,tao->cnvP);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSetUp_BNCG(Tao tao)
{
  TAO_BNCG         *cg = (TAO_BNCG*)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!tao->gradient) {ierr = VecDuplicate(tao->solution,&tao->gradient);CHKERRQ(ierr);}
  if (!tao->stepdirection) {ierr = VecDuplicate(tao->solution,&tao->stepdirection);CHKERRQ(ierr); }
  if (!cg->X_old) {ierr = VecDuplicate(tao->solution,&cg->X_old);CHKERRQ(ierr);}
  if (!cg->G_old) {ierr = VecDuplicate(tao->gradient,&cg->G_old);CHKERRQ(ierr); }
  if (!cg->unprojected_gradient) {ierr = VecDuplicate(tao->gradient,&cg->unprojected_gradient);CHKERRQ(ierr);}
  if (!cg->unprojected_gradient_old) {ierr = VecDuplicate(tao->gradient,&cg->unprojected_gradient_old);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoDestroy_BNCG(Tao tao)
{
  TAO_BNCG       *cg = (TAO_BNCG*) tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (tao->setupcalled) {
    ierr = VecDestroy(&cg->X_old);CHKERRQ(ierr);
    ierr = VecDestroy(&cg->G_old);CHKERRQ(ierr);
    ierr = VecDestroy(&cg->unprojected_gradient);CHKERRQ(ierr);
    ierr = VecDestroy(&cg->unprojected_gradient_old);CHKERRQ(ierr);
  }
  ierr = TaoLineSearchDestroy(&tao->linesearch);CHKERRQ(ierr);
  ierr = PetscFree(tao->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSetFromOptions_BNCG(PetscOptionItems *PetscOptionsObject,Tao tao)
 {
    TAO_BNCG       *cg = (TAO_BNCG*)tao->data;
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = TaoLineSearchSetFromOptions(tao->linesearch);CHKERRQ(ierr);
    ierr = PetscOptionsHead(PetscOptionsObject,"Nonlinear Conjugate Gradient method for unconstrained optimization");CHKERRQ(ierr);
    ierr = PetscOptionsReal("-tao_BNCG_eta","restart tolerance", "", cg->eta,&cg->eta,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-tao_BNCG_rho","descent direction tolerance", "", cg->rho,&cg->rho,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-tao_BNCG_pow","descent direction exponent", "", cg->pow,&cg->pow,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEList("-tao_BNCG_type","cg formula", "", CG_Table, CG_Types, CG_Table[cg->cg_type], &cg->cg_type,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-tao_BNCG_delta_min","minimum delta value", "", cg->delta_min,&cg->delta_min,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-tao_BNCG_delta_max","maximum delta value", "", cg->delta_max,&cg->delta_max,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsTail();CHKERRQ(ierr);
   PetscFunctionReturn(0);
}

static PetscErrorCode TaoView_BNCG(Tao tao, PetscViewer viewer)
{
  PetscBool      isascii;
  TAO_BNCG       *cg = (TAO_BNCG*)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "CG Type: %s\n", CG_Table[cg->cg_type]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "Resets: %i\n", cg->resets);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "  Broken ortho: %i\n", cg->broken_ortho);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "  Not a descent dir.: %i\n", cg->descent_error);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "Line search fails: %i\n", cg->ls_fails);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*MC
     TAOBNCG -   Bound-constrained Nonlinear Conjugate Gradient method.

   Options Database Keys:
+      -tao_BNCG_eta <r> - restart tolerance
.      -tao_BNCG_type <taocg_type> - cg formula
.      -tao_BNCG_delta_min <r> - minimum delta value
-      -tao_BNCG_delta_max <r> - maximum delta value

  Notes:
     CG formulas are:
         "fr" - Fletcher-Reeves
         "pr" - Polak-Ribiere
         "prp" - Polak-Ribiere-Plus
         "hs" - Hestenes-Steifel
         "dy" - Dai-Yuan
  Level: beginner
M*/


PETSC_EXTERN PetscErrorCode TaoCreate_BNCG(Tao tao)
{
  TAO_BNCG       *cg;
  const char     *morethuente_type = TAOLINESEARCHMT;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  tao->ops->setup = TaoSetUp_BNCG;
  tao->ops->solve = TaoSolve_BNCG;
  tao->ops->view = TaoView_BNCG;
  tao->ops->setfromoptions = TaoSetFromOptions_BNCG;
  tao->ops->destroy = TaoDestroy_BNCG;

  /* Override default settings (unless already changed) */
  if (!tao->max_it_changed) tao->max_it = 2000;
  if (!tao->max_funcs_changed) tao->max_funcs = 4000;

  /*  Note: nondefault values should be used for nonlinear conjugate gradient  */
  /*  method.  In particular, gtol should be less that 0.5; the value used in  */
  /*  Nocedal and Wright is 0.10.  We use the default values for the  */
  /*  linesearch because it seems to work better. */
  ierr = TaoLineSearchCreate(((PetscObject)tao)->comm, &tao->linesearch);CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject)tao->linesearch, (PetscObject)tao, 1);CHKERRQ(ierr);
  ierr = TaoLineSearchSetType(tao->linesearch, morethuente_type);CHKERRQ(ierr);
  ierr = TaoLineSearchUseTaoRoutines(tao->linesearch, tao);CHKERRQ(ierr);
  ierr = TaoLineSearchSetOptionsPrefix(tao->linesearch,tao->hdr.prefix);CHKERRQ(ierr);

  ierr = PetscNewLog(tao,&cg);CHKERRQ(ierr);
  tao->data = (void*)cg;
  cg->rho = 1e-4;
  cg->pow = 2.1;
  cg->eta = 0.5;
  cg->delta_min = 1e-7;
  cg->delta_max = 100;
  cg->cg_type = CG_DaiYuan;
  PetscFunctionReturn(0);
}
