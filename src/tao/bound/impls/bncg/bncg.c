#include <petsctaolinesearch.h>
#include <../src/tao/bound/impls/bncg/bncg.h>

#define CG_FletcherReeves       0
#define CG_PolakRibiere         1
#define CG_PolakRibierePlus     2
#define CG_HestenesStiefel      3
#define CG_DaiYuan              4
#define CG_Types                5

static const char *CG_Table[64] = {"fr", "pr", "prp", "hs", "dy"};

#define CG_AS_NONE       0
#define CG_AS_BERTSEKAS  1
#define CG_AS_SIZE       2

static const char *CG_AS_TYPE[64] = {"none", "bertsekas"};

PetscErrorCode TaoBNCGSetRecycleFlag(Tao tao, PetscBool recycle)
{
  TAO_BNCG                     *cg = (TAO_BNCG*)tao->data;

  PetscFunctionBegin;
  cg->recycle = recycle;
  PetscFunctionReturn(0);
}

PetscErrorCode TaoBNCGEstimateActiveSet(Tao tao, PetscInt asType)
{
  PetscErrorCode               ierr;
  TAO_BNCG                     *cg = (TAO_BNCG *)tao->data;

  PetscFunctionBegin;
  ierr = ISDestroy(&cg->inactive_old);CHKERRQ(ierr);
  if (cg->inactive_idx) {
    ierr = ISDuplicate(cg->inactive_idx, &cg->inactive_old);CHKERRQ(ierr);
    ierr = ISCopy(cg->inactive_idx, cg->inactive_old);CHKERRQ(ierr);
  }
  switch (asType) {
  case CG_AS_NONE:
    ierr = ISDestroy(&cg->inactive_idx);CHKERRQ(ierr);
    ierr = VecWhichInactive(tao->XL, tao->solution, cg->unprojected_gradient, tao->XU, PETSC_TRUE, &cg->inactive_idx);CHKERRQ(ierr);
    ierr = ISDestroy(&cg->active_idx);CHKERRQ(ierr);
    ierr = ISComplementVec(cg->inactive_idx, tao->solution, &cg->active_idx);CHKERRQ(ierr);
    break;

  case CG_AS_BERTSEKAS:
    /* Use gradient descent to estimate the active set */
    ierr = VecCopy(cg->unprojected_gradient, cg->W);CHKERRQ(ierr);
    ierr = VecScale(cg->W, -1.0);CHKERRQ(ierr);
    ierr = TaoEstimateActiveBounds(tao->solution, tao->XL, tao->XU, cg->unprojected_gradient, cg->W, cg->work, cg->as_step, &cg->as_tol, 
                                   &cg->active_lower, &cg->active_upper, &cg->active_fixed, &cg->active_idx, &cg->inactive_idx);CHKERRQ(ierr);
    break;
    
  default:
    break;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode TaoBNCGBoundStep(Tao tao, PetscInt asType, Vec step)
{
  PetscErrorCode               ierr;
  TAO_BNCG                     *cg = (TAO_BNCG *)tao->data;
  
  PetscFunctionBegin;
  switch (asType) {
  case CG_AS_NONE:
    ierr = VecISSet(step, cg->active_idx, 0.0);CHKERRQ(ierr);
    break;

  case CG_AS_BERTSEKAS:
    ierr = TaoBoundStep(tao->solution, tao->XL, tao->XU, cg->active_lower, cg->active_upper, cg->active_fixed, 1.0, step);CHKERRQ(ierr);
    break;

  default:
    break;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSolve_BNCG(Tao tao)
{
  TAO_BNCG                     *cg = (TAO_BNCG*)tao->data;
  PetscErrorCode               ierr;
  TaoLineSearchConvergedReason ls_status = TAOLINESEARCH_CONTINUE_ITERATING;
  PetscReal                    step=1.0,gnorm,gnorm2,gd,ginner,beta,dnorm;
  PetscReal                    gd_old,gnorm2_old,f_old,resnorm;
  PetscBool                    cg_restart;
  PetscInt                     nDiff;

  PetscFunctionBegin;
  /*   Project the current point onto the feasible set */
  ierr = TaoComputeVariableBounds(tao);CHKERRQ(ierr);
  ierr = TaoLineSearchSetVariableBounds(tao->linesearch,tao->XL,tao->XU);CHKERRQ(ierr);
  
  /* Project the initial point onto the feasible region */
  ierr = TaoBoundSolution(tao->solution, tao->XL,tao->XU, 0.0, &nDiff, tao->solution);CHKERRQ(ierr);

  if (nDiff > 0 || !cg->recycle) {
    /*  Solver is not being recycled so just compute the objective function and criteria */
    ierr = TaoComputeObjectiveAndGradient(tao, tao->solution, &cg->f, cg->unprojected_gradient);CHKERRQ(ierr);
  } else {
    /* We are recycling, so we have to compute ||g_old||^2 for use in the CG step calculation */
    ierr = VecDot(cg->G_old, cg->G_old, &gnorm2_old);CHKERRQ(ierr);
  }
  ierr = VecNorm(cg->unprojected_gradient,NORM_2,&gnorm);CHKERRQ(ierr);
  if (PetscIsInfOrNanReal(cg->f) || PetscIsInfOrNanReal(gnorm)) SETERRQ(PETSC_COMM_SELF,1, "User provided compute function generated Inf or NaN");
  
  /* Estimate the active set and compute the projected gradient */
  ierr = TaoBNCGEstimateActiveSet(tao, cg->as_type);CHKERRQ(ierr);

  /* Project the gradient and calculate the norm */
  ierr = VecCopy(cg->unprojected_gradient, tao->gradient);CHKERRQ(ierr);
  ierr = VecISSet(tao->gradient, cg->active_idx, 0.0);CHKERRQ(ierr);
  ierr = VecNorm(tao->gradient,NORM_2,&gnorm);CHKERRQ(ierr);
  gnorm2 = gnorm*gnorm;
  
  /* Convergence check */
  tao->niter = 0;
  tao->reason = TAO_CONTINUE_ITERATING;
  ierr = TaoLogConvergenceHistory(tao, cg->f, gnorm, 0.0, tao->ksp_its);CHKERRQ(ierr);
  ierr = TaoMonitor(tao, tao->niter, cg->f, gnorm, 0.0, step);CHKERRQ(ierr);
  ierr = (*tao->ops->convergencetest)(tao,tao->cnvP);CHKERRQ(ierr);
  if (tao->reason != TAO_CONTINUE_ITERATING) PetscFunctionReturn(0);
  
  /* Start optimization iterations */
  cg->ls_fails = cg->broken_ortho = cg->descent_error = 0;
  cg->resets = -1;
  while (tao->reason == TAO_CONTINUE_ITERATING) {
    ++tao->niter;
    
    /* Check restart conditions for using steepest descent */
    cg_restart = PETSC_FALSE;
    ierr = VecDot(tao->gradient, cg->G_old, &ginner);CHKERRQ(ierr);
    ierr = VecNorm(tao->stepdirection, NORM_2, &dnorm);CHKERRQ(ierr);
    if (tao->niter == 1 && !cg->recycle && dnorm != 0.0) {
      /* 1) First iteration, with recycle disabled, and a non-zero previous step */
      cg_restart = PETSC_TRUE;
    } else if (PetscAbsScalar(ginner) >= cg->eta * gnorm2) {
      /* 2) Gradients are far from orthogonal */
      cg_restart = PETSC_TRUE;
      ++cg->broken_ortho;
    }
    
    /* Compute CG step */
    if (cg_restart) {
      beta = 0.0;
      ++cg->resets;
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
    ierr = TaoBNCGBoundStep(tao, cg->as_type, tao->stepdirection);CHKERRQ(ierr);
    
    /* Figure out which previously active variables became inactive this iteration */
    ierr = ISDestroy(&cg->new_inactives);CHKERRQ(ierr);
    if (cg->inactive_idx && cg->inactive_old) {
      ierr = ISDifference(cg->inactive_idx, cg->inactive_old, &cg->new_inactives);CHKERRQ(ierr);
    }
    
    /* Selectively reset the CG step those freshly inactive variables */
    if (cg->new_inactives) {
      ierr = VecGetSubVector(tao->stepdirection, cg->new_inactives, &cg->inactive_step);CHKERRQ(ierr);
      ierr = VecGetSubVector(cg->unprojected_gradient, cg->new_inactives, &cg->inactive_grad);CHKERRQ(ierr);
      ierr = VecCopy(cg->inactive_grad, cg->inactive_step);CHKERRQ(ierr);
      ierr = VecScale(cg->inactive_step, -1.0);CHKERRQ(ierr);
      ierr = VecRestoreSubVector(tao->stepdirection, cg->new_inactives, &cg->inactive_step);CHKERRQ(ierr);
      ierr = VecRestoreSubVector(cg->unprojected_gradient, cg->new_inactives, &cg->inactive_grad);CHKERRQ(ierr);
    }
    
    /* Verify that this is a descent direction */
    ierr = VecDot(tao->gradient, tao->stepdirection, &gd);CHKERRQ(ierr);
    ierr = VecNorm(tao->stepdirection, NORM_2, &dnorm);
    if (gd > -cg->rho*PetscPowReal(dnorm, cg->pow)) {
      /* Not a descent direction, so we reset back to projected gradient descent */
      ierr = VecAXPBY(tao->stepdirection, -1.0, 0.0, tao->gradient);CHKERRQ(ierr);
      ++cg->resets;
      ++cg->descent_error;
    }
    
    /* Store solution and gradient info before it changes */
    ierr = VecCopy(tao->solution, cg->X_old);CHKERRQ(ierr);
    ierr = VecCopy(tao->gradient, cg->G_old);CHKERRQ(ierr);
    ierr = VecCopy(cg->unprojected_gradient, cg->unprojected_gradient_old);CHKERRQ(ierr);
    gnorm2_old = gnorm2;
    f_old = cg->f;
    
    /* Perform bounded line search */
    ierr = TaoLineSearchApply(tao->linesearch, tao->solution, &cg->f, cg->unprojected_gradient, tao->stepdirection, &step, &ls_status);CHKERRQ(ierr);
    ierr = TaoAddLineSearchCounts(tao);CHKERRQ(ierr);
    
    /*  Check linesearch failure */
    if (ls_status != TAOLINESEARCH_SUCCESS && ls_status != TAOLINESEARCH_SUCCESS_USER) {
      ++cg->ls_fails;
      /* Restore previous point */
      gnorm2 = gnorm2_old;
      cg->f = f_old;
      ierr = VecCopy(cg->X_old, tao->solution);CHKERRQ(ierr);
      ierr = VecCopy(cg->G_old, tao->gradient);CHKERRQ(ierr);
      ierr = VecCopy(cg->unprojected_gradient_old, cg->unprojected_gradient);CHKERRQ(ierr);
      
      /* Fall back on the gradient descent step */
      ierr = VecCopy(tao->gradient, tao->stepdirection);CHKERRQ(ierr);
      ierr = VecScale(tao->stepdirection, -1.0);CHKERRQ(ierr);
      ierr = TaoBNCGBoundStep(tao, cg->as_type, tao->stepdirection);CHKERRQ(ierr);
      ierr = TaoLineSearchApply(tao->linesearch, tao->solution, &cg->f, cg->unprojected_gradient, tao->stepdirection, &step, &ls_status);CHKERRQ(ierr);
      ierr = TaoAddLineSearchCounts(tao);CHKERRQ(ierr);
        
      if (ls_status != TAOLINESEARCH_SUCCESS && ls_status != TAOLINESEARCH_SUCCESS_USER){
        ++cg->ls_fails;
        /* Restore previous point */
        gnorm2 = gnorm2_old;
        cg->f = f_old;
        ierr = VecCopy(cg->X_old, tao->solution);CHKERRQ(ierr);
        ierr = VecCopy(cg->G_old, tao->gradient);CHKERRQ(ierr);
        ierr = VecCopy(cg->unprojected_gradient_old, cg->unprojected_gradient);CHKERRQ(ierr);
        
        /* Nothing left to do but fail out of the optimization */
        step = 0.0;
        tao->reason = TAO_DIVERGED_LS_FAILURE;
      }
    }
    
    if (tao->reason != TAO_DIVERGED_LS_FAILURE) {
      /* Estimate the active set at the new solution */
      ierr = TaoBNCGEstimateActiveSet(tao, cg->as_type);CHKERRQ(ierr);

      /* Compute the projected gradient and its norm */
      ierr = VecCopy(cg->unprojected_gradient, tao->gradient);CHKERRQ(ierr);
      ierr = VecISSet(tao->gradient, cg->active_idx, 0.0);CHKERRQ(ierr);
      ierr = VecNorm(tao->gradient,NORM_2,&gnorm);CHKERRQ(ierr);
      gnorm2 = gnorm*gnorm;
    }
    
    /* Convergence test */
    ierr = VecFischer(tao->solution, cg->unprojected_gradient, tao->XL, tao->XU, cg->W);CHKERRQ(ierr);
    ierr = VecNorm(cg->W, NORM_2, &resnorm);CHKERRQ(ierr);
    if (PetscIsInfOrNanReal(resnorm)) SETERRQ(PETSC_COMM_SELF,1, "User provided compute function generated Inf or NaN");
    ierr = TaoLogConvergenceHistory(tao, cg->f, resnorm, 0.0, tao->ksp_its);CHKERRQ(ierr);
    ierr = TaoMonitor(tao, tao->niter, cg->f, resnorm, 0.0, step);CHKERRQ(ierr);
    ierr = (*tao->ops->convergencetest)(tao,tao->cnvP);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSetUp_BNCG(Tao tao)
{
  TAO_BNCG         *cg = (TAO_BNCG*)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!tao->gradient) {
    ierr = VecDuplicate(tao->solution,&tao->gradient);CHKERRQ(ierr);
  }
  if (!tao->stepdirection) {
    ierr = VecDuplicate(tao->solution,&tao->stepdirection);CHKERRQ(ierr);
  }
  if (!cg->W) {
    ierr = VecDuplicate(tao->solution,&cg->W);CHKERRQ(ierr);
  }
  if (!cg->work) {
    ierr = VecDuplicate(tao->solution,&cg->work);CHKERRQ(ierr);
  }
  if (!cg->X_old) {
    ierr = VecDuplicate(tao->solution,&cg->X_old);CHKERRQ(ierr);
  }
  if (!cg->G_old) {
    ierr = VecDuplicate(tao->gradient,&cg->G_old);CHKERRQ(ierr);
  }
  if (!cg->unprojected_gradient) {
    ierr = VecDuplicate(tao->gradient,&cg->unprojected_gradient);CHKERRQ(ierr);
  }
  if (!cg->unprojected_gradient_old) {
    ierr = VecDuplicate(tao->gradient,&cg->unprojected_gradient_old);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoDestroy_BNCG(Tao tao)
{
  TAO_BNCG       *cg = (TAO_BNCG*) tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (tao->setupcalled) {
    ierr = VecDestroy(&cg->W);CHKERRQ(ierr);
    ierr = VecDestroy(&cg->work);CHKERRQ(ierr);
    ierr = VecDestroy(&cg->X_old);CHKERRQ(ierr);
    ierr = VecDestroy(&cg->G_old);CHKERRQ(ierr);
    ierr = VecDestroy(&cg->unprojected_gradient);CHKERRQ(ierr);
    ierr = VecDestroy(&cg->unprojected_gradient_old);CHKERRQ(ierr);
  }
  ierr = ISDestroy(&cg->active_lower);CHKERRQ(ierr);
  ierr = ISDestroy(&cg->active_upper);CHKERRQ(ierr);
  ierr = ISDestroy(&cg->active_fixed);CHKERRQ(ierr);
  ierr = ISDestroy(&cg->active_idx);CHKERRQ(ierr);
  ierr = ISDestroy(&cg->inactive_idx);CHKERRQ(ierr);
  ierr = ISDestroy(&cg->inactive_old);CHKERRQ(ierr);
  ierr = ISDestroy(&cg->new_inactives);CHKERRQ(ierr);
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
    ierr = PetscOptionsReal("-tao_bncg_eta","restart tolerance", "", cg->eta,&cg->eta,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-tao_bncg_rho","descent direction tolerance", "", cg->rho,&cg->rho,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-tao_bncg_pow","descent direction exponent", "", cg->pow,&cg->pow,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEList("-tao_bncg_type","cg formula", "", CG_Table, CG_Types, CG_Table[cg->cg_type], &cg->cg_type,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEList("-tao_bncg_as_type","active set estimation method", "", CG_AS_TYPE, CG_AS_SIZE, CG_AS_TYPE[cg->cg_type], &cg->cg_type,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-tao_bncg_recycle","enable recycling the existing solution and gradient at the start of a new solve","",cg->recycle,&cg->recycle,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-tao_bncg_as_tol", "initial tolerance used when estimating actively bounded variables","",cg->as_tol,&cg->as_tol,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-tao_bncg_as_step", "step length used when estimating actively bounded variables","",cg->as_step,&cg->as_step,NULL);CHKERRQ(ierr);
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
+      -tao_bncg_recycle - enable recycling the latest calculated gradient vector in subsequent TaoSolve() calls
.      -tao_bncg_eta <r> - restart tolerance
.      -tao_bncg_type <taocg_type> - cg formula
.      -tao_bncg_as_type <none,bertsekas> - active set estimation method
.      -tao_bncg_as_tol <r> - tolerance used in Bertsekas active-set estimation
.      -tao_bncg_as_step <r> - trial step length used in Bertsekas active-set estimation

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
  cg->as_step = 0.001;
  cg->as_tol = 0.001;
  cg->as_type = CG_AS_BERTSEKAS;
  cg->cg_type = CG_DaiYuan;
  cg->recycle = PETSC_FALSE;
  PetscFunctionReturn(0);
}
