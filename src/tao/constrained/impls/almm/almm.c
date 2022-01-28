#include <../src/tao/constrained/impls/almm/almm.h> /*I "petsctao.h" I*/ /*I "petscvec.h" I*/
#include <petsctao.h>
#include <petsc/private/petscimpl.h>
#include <petsc/private/vecimpl.h>

static PetscErrorCode TaoALMMCombinePrimal_Private(Tao,Vec,Vec,Vec);
static PetscErrorCode TaoALMMCombineDual_Private(Tao,Vec,Vec,Vec);
static PetscErrorCode TaoALMMSplitPrimal_Private(Tao,Vec,Vec,Vec);
static PetscErrorCode TaoALMMComputeOptimalityNorms_Private(Tao);
static PetscErrorCode TaoALMMComputeAugLagAndGradient_Private(Tao);
static PetscErrorCode TaoALMMComputePHRLagAndGradient_Private(Tao);

static PetscErrorCode TaoSolve_ALMM(Tao tao)
{
  TAO_ALMM           *auglag = (TAO_ALMM*)tao->data;
  TaoConvergedReason reason;
  PetscReal          updated;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  /* reset initial multiplier/slack guess */
  if (!tao->recycle) {
    if (tao->ineq_constrained) {
      ierr = VecZeroEntries(auglag->Ps);CHKERRQ(ierr);
      ierr = TaoALMMCombinePrimal_Private(tao, auglag->Px, auglag->Ps, auglag->P);CHKERRQ(ierr);
      ierr = VecZeroEntries(auglag->Yi);CHKERRQ(ierr);
    }
    if (tao->eq_constrained) {
      ierr = VecZeroEntries(auglag->Ye);CHKERRQ(ierr);
    }
  }

  /* compute initial nonlinear Lagrangian and its derivatives */
  ierr = (*auglag->sub_obj)(tao);CHKERRQ(ierr);
  ierr = TaoALMMComputeOptimalityNorms_Private(tao);CHKERRQ(ierr);
  /* print initial step and check convergence */
  ierr = PetscInfo(tao,"Solving with %s formulation\n",TaoALMMTypes[auglag->type]);CHKERRQ(ierr);
  ierr = TaoLogConvergenceHistory(tao, auglag->Lval, auglag->gnorm, auglag->cnorm, tao->ksp_its);CHKERRQ(ierr);
  ierr = TaoMonitor(tao, tao->niter, auglag->fval, auglag->gnorm, auglag->cnorm, 0.0);CHKERRQ(ierr);
  ierr = (*tao->ops->convergencetest)(tao, tao->cnvP);CHKERRQ(ierr);
  /* set initial penalty factor and inner solver tolerance */
  switch (auglag->type) {
    case TAO_ALMM_CLASSIC:
      auglag->mu = auglag->mu0;
      break;
    case TAO_ALMM_PHR:
      auglag->cenorm = 0.0;
      if (tao->eq_constrained) {
        ierr = VecDot(auglag->Ce, auglag->Ce, &auglag->cenorm);CHKERRQ(ierr);
      }
      auglag->cinorm = 0.0;
      if (tao->ineq_constrained) {
        ierr = VecCopy(auglag->Ci, auglag->Ciwork);CHKERRQ(ierr);
        ierr = VecScale(auglag->Ciwork, -1.0);CHKERRQ(ierr);
        ierr = VecPointwiseMax(auglag->Ciwork, auglag->Cizero, auglag->Ciwork);CHKERRQ(ierr);
        ierr = VecDot(auglag->Ciwork, auglag->Ciwork, &auglag->cinorm);CHKERRQ(ierr);
      }
      /* determine initial penalty factor based on the balance of constraint violation and objective function value */
      auglag->mu = PetscMax(1.e-6, PetscMin(10.0, 2.0*PetscAbsReal(auglag->fval)/(auglag->cenorm + auglag->cinorm)));
      break;
    default:
      break;
  }
  auglag->gtol = auglag->gtol0;
  ierr = PetscInfo(tao,"Initial penalty: %.2f\n",auglag->mu);CHKERRQ(ierr);

  /* start aug-lag outer loop */
  while (tao->reason == TAO_CONTINUE_ITERATING) {
    ++tao->niter;
    /* update subsolver tolerance */
    ierr = PetscInfo(tao,"Subsolver tolerance: ||G|| <= %e\n",auglag->gtol);CHKERRQ(ierr);
    ierr = TaoSetTolerances(auglag->subsolver, auglag->gtol, 0.0, 0.0);CHKERRQ(ierr);
    /* solve the bound-constrained or unconstrained subproblem */
    ierr = TaoSolve(auglag->subsolver);CHKERRQ(ierr);
    ierr = TaoGetConvergedReason(auglag->subsolver, &reason);CHKERRQ(ierr);
    tao->ksp_its += auglag->subsolver->ksp_its;
    if (reason != TAO_CONVERGED_GATOL) {
      ierr = PetscInfo(tao,"Subsolver failed to converge, reason: %s\n",TaoConvergedReasons[reason]);CHKERRQ(ierr);
    }
    /* evaluate solution and test convergence */
    ierr = (*auglag->sub_obj)(tao);CHKERRQ(ierr);
    ierr = TaoALMMComputeOptimalityNorms_Private(tao);CHKERRQ(ierr);
    /* decide whether to update multipliers or not */
    updated = 0.0;
    if (auglag->cnorm <= auglag->ytol) {
      ierr = PetscInfo(tao,"Multipliers updated: ||C|| <= %e\n",auglag->ytol);CHKERRQ(ierr);
      /* constraints are good, update multipliers and convergence tolerances */
      if (tao->eq_constrained) {
        ierr = VecAXPY(auglag->Ye, auglag->mu, auglag->Ce);CHKERRQ(ierr);
        ierr = VecSet(auglag->Cework, auglag->ye_max);CHKERRQ(ierr);
        ierr = VecPointwiseMin(auglag->Ye, auglag->Cework, auglag->Ye);CHKERRQ(ierr);
        ierr = VecSet(auglag->Cework, auglag->ye_min);CHKERRQ(ierr);
        ierr = VecPointwiseMax(auglag->Ye, auglag->Cework, auglag->Ye);CHKERRQ(ierr);
      }
      if (tao->ineq_constrained) {
        ierr = VecAXPY(auglag->Yi, auglag->mu, auglag->Ci);CHKERRQ(ierr);
        ierr = VecSet(auglag->Ciwork, auglag->yi_max);CHKERRQ(ierr);
        ierr = VecPointwiseMin(auglag->Yi, auglag->Ciwork, auglag->Yi);CHKERRQ(ierr);
        ierr = VecSet(auglag->Ciwork, auglag->yi_min);CHKERRQ(ierr);
        ierr = VecPointwiseMax(auglag->Yi, auglag->Ciwork, auglag->Yi);CHKERRQ(ierr);
      }
      /* tolerances are updated only for non-PHR methods */
      if (auglag->type != TAO_ALMM_PHR) {
        auglag->ytol = PetscMax(tao->catol, auglag->ytol/PetscPowReal(auglag->mu, auglag->mu_pow_good));
        auglag->gtol = PetscMax(tao->gatol, auglag->gtol/auglag->mu);
      }
      updated = 1.0;
    } else {
      /* constraints are bad, update penalty factor */
      auglag->mu = PetscMin(auglag->mu_max, auglag->mu_fac*auglag->mu);
      /* tolerances are reset only for non-PHR methods */
      if (auglag->type != TAO_ALMM_PHR) {
        auglag->ytol = PetscMax(tao->catol, 0.1/PetscPowReal(auglag->mu, auglag->mu_pow_bad));
        auglag->gtol = PetscMax(tao->gatol, 1.0/auglag->mu);
      }
      ierr = PetscInfo(tao,"Penalty increased: mu = %.2f\n",auglag->mu);CHKERRQ(ierr);
    }
    ierr = TaoLogConvergenceHistory(tao, auglag->fval, auglag->gnorm, auglag->cnorm, tao->ksp_its);CHKERRQ(ierr);
    ierr = TaoMonitor(tao, tao->niter, auglag->fval, auglag->gnorm, auglag->cnorm, updated);CHKERRQ(ierr);
    ierr = (*tao->ops->convergencetest)(tao, tao->cnvP);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode TaoView_ALMM(Tao tao,PetscViewer viewer)
{
  TAO_ALMM       *auglag = (TAO_ALMM*)tao->data;
  PetscBool      isascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  ierr = TaoView(auglag->subsolver,viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "ALMM Formulation Type: %s\n", TaoALMMTypes[auglag->type]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSetUp_ALMM(Tao tao)
{
  TAO_ALMM       *auglag = (TAO_ALMM*)tao->data;
  VecType        vec_type;
  Vec            SL, SU;
  PetscBool      is_cg = PETSC_FALSE, is_lmvm = PETSC_FALSE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscAssertFalse(tao->ineq_doublesided,PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_WRONGSTATE, "TAOALMM does not support double-sided inequality constraint definition. Please restructure your inequality constrainst to fit the form c(x) >= 0.");
  PetscAssertFalse(!tao->eq_constrained && !tao->ineq_constrained,PetscObjectComm((PetscObject)tao), PETSC_ERR_ORDER, "Equality and/or inequality constraints must be defined before solver setup.");
  ierr = TaoComputeVariableBounds(tao);CHKERRQ(ierr);
  /* alias base vectors and create extras */
  ierr = VecGetType(tao->solution, &vec_type);CHKERRQ(ierr);
  auglag->Px = tao->solution;
  if (!tao->gradient) { /* base gradient */
    ierr = VecDuplicate(tao->solution, &tao->gradient);CHKERRQ(ierr);
  }
  auglag->LgradX = tao->gradient;
  if (!auglag->Xwork) { /* opt var work vector */
    ierr = VecDuplicate(tao->solution, &auglag->Xwork);CHKERRQ(ierr);
  }
  if (tao->eq_constrained) {
    auglag->Ce = tao->constraints_equality;
    auglag->Ae = tao->jacobian_equality;
    if (!auglag->Ye) { /* equality multipliers */
      ierr = VecDuplicate(auglag->Ce, &auglag->Ye);CHKERRQ(ierr);
    }
    if (!auglag->Cework) {
      ierr = VecDuplicate(auglag->Ce, &auglag->Cework);CHKERRQ(ierr);
    }
  }
  if (tao->ineq_constrained) {
    auglag->Ci = tao->constraints_inequality;
    auglag->Ai = tao->jacobian_inequality;
    if (!auglag->Yi) { /* inequality multipliers */
      ierr = VecDuplicate(auglag->Ci, &auglag->Yi);CHKERRQ(ierr);
    }
    if (!auglag->Ciwork) {
      ierr = VecDuplicate(auglag->Ci, &auglag->Ciwork);CHKERRQ(ierr);
    }
    if (!auglag->Cizero) {
      ierr = VecDuplicate(auglag->Ci, &auglag->Cizero);CHKERRQ(ierr);
      ierr = VecZeroEntries(auglag->Cizero);CHKERRQ(ierr);
    }
    if (!auglag->Ps) { /* slack vars */
      ierr = VecDuplicate(auglag->Ci, &auglag->Ps);CHKERRQ(ierr);
    }
    if (!auglag->LgradS) { /* slack component of Lagrangian gradient */
      ierr = VecDuplicate(auglag->Ci, &auglag->LgradS);CHKERRQ(ierr);
    }
    /* create vector for combined primal space and the associated communication objects */
    if (!auglag->P) {
      ierr = PetscMalloc1(2, &auglag->Parr);CHKERRQ(ierr);
      auglag->Parr[0] = auglag->Px; auglag->Parr[1] = auglag->Ps;
      ierr = VecConcatenate(2, auglag->Parr, &auglag->P, &auglag->Pis);CHKERRQ(ierr);
      ierr = PetscMalloc1(2, &auglag->Pscatter);CHKERRQ(ierr);
      ierr = VecScatterCreate(auglag->P, auglag->Pis[0], auglag->Px, NULL, &auglag->Pscatter[0]);CHKERRQ(ierr);
      ierr = VecScatterCreate(auglag->P, auglag->Pis[1], auglag->Ps, NULL, &auglag->Pscatter[1]);CHKERRQ(ierr);
    }
    if (tao->eq_constrained) {
      /* create vector for combined dual space and the associated communication objects */
      if (!auglag->Y) {
        ierr = PetscMalloc1(2, &auglag->Yarr);CHKERRQ(ierr);
        auglag->Yarr[0] = auglag->Ye; auglag->Yarr[1] = auglag->Yi;
        ierr = VecConcatenate(2, auglag->Yarr, &auglag->Y, &auglag->Yis);CHKERRQ(ierr);
        ierr = PetscMalloc1(2, &auglag->Yscatter);CHKERRQ(ierr);
        ierr = VecScatterCreate(auglag->Y, auglag->Yis[0], auglag->Ye, NULL, &auglag->Yscatter[0]);CHKERRQ(ierr);
        ierr = VecScatterCreate(auglag->Y, auglag->Yis[1], auglag->Yi, NULL, &auglag->Yscatter[1]);CHKERRQ(ierr);
      }
      if (!auglag->C) {
        ierr = VecDuplicate(auglag->Y, &auglag->C);CHKERRQ(ierr);
      }
    } else {
      if (!auglag->C) {
        auglag->C = auglag->Ci;
      }
      if (!auglag->Y) {
        auglag->Y = auglag->Yi;
      }
    }
  } else {
    if (!auglag->P) {
      auglag->P = auglag->Px;
    }
    if (!auglag->G) {
      auglag->G = auglag->LgradX;
    }
    if (!auglag->C) {
      auglag->C = auglag->Ce;
    }
    if (!auglag->Y) {
      auglag->Y = auglag->Ye;
    }
  }
  /* initialize parameters */
  if (auglag->type == TAO_ALMM_PHR) {
    auglag->mu_fac = 10.0;
    auglag->yi_min = 0.0;
    auglag->ytol0 = 0.5;
    auglag->gtol0 = tao->gatol;
    if (tao->gatol_changed && tao->catol_changed) {
      ierr = PetscInfo(tao,"TAOALMM with PHR: different gradient and constraint tolerances are not supported, setting catol = gatol\n");CHKERRQ(ierr);
      tao->catol = tao->gatol;
    }
  }
  /* set the Lagrangian formulation type for the subsolver */
  switch (auglag->type) {
    case TAO_ALMM_CLASSIC:
      auglag->sub_obj = TaoALMMComputeAugLagAndGradient_Private;
      break;
    case TAO_ALMM_PHR:
      auglag->sub_obj = TaoALMMComputePHRLagAndGradient_Private;
      break;
    default:
      break;
  }
  /* set up the subsolver */
  ierr = TaoSetInitialVector(auglag->subsolver, auglag->P);CHKERRQ(ierr);
  ierr = TaoSetObjectiveRoutine(auglag->subsolver, TaoALMMSubsolverObjective_Private, (void*)auglag);CHKERRQ(ierr);
  ierr = TaoSetObjectiveAndGradientRoutine(auglag->subsolver, TaoALMMSubsolverObjectiveAndGradient_Private, (void*)auglag);CHKERRQ(ierr);
  if (tao->bounded) {
    /* make sure that the subsolver is a bound-constrained method */
    ierr = PetscObjectTypeCompare((PetscObject)auglag->subsolver, TAOCG, &is_cg);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)auglag->subsolver, TAOLMVM, &is_lmvm);CHKERRQ(ierr);
    if (is_cg) {
      ierr = TaoSetType(auglag->subsolver, TAOBNCG);CHKERRQ(ierr);
      ierr = PetscInfo(tao,"TAOCG detected for bound-constrained problem, switching to TAOBNCG instead.");CHKERRQ(ierr);
    }
    if (is_lmvm) {
      ierr = TaoSetType(auglag->subsolver, TAOBQNLS);CHKERRQ(ierr);
      ierr = PetscInfo(tao,"TAOLMVM detected for bound-constrained problem, switching to TAOBQNLS instead.");CHKERRQ(ierr);
    }
    /* create lower and upper bound clone vectors for subsolver */
    if (!auglag->PL) {
      ierr = VecDuplicate(auglag->P, &auglag->PL);CHKERRQ(ierr);
    }
    if (!auglag->PU) {
      ierr = VecDuplicate(auglag->P, &auglag->PU);CHKERRQ(ierr);
    }
    if (tao->ineq_constrained) {
      /* create lower and upper bounds for slack, set lower to 0 */
      ierr = VecDuplicate(auglag->Ci, &SL);CHKERRQ(ierr);
      ierr = VecSet(SL, 0.0);CHKERRQ(ierr);
      ierr = VecDuplicate(auglag->Ci, &SU);CHKERRQ(ierr);
      ierr = VecSet(SU, PETSC_INFINITY);CHKERRQ(ierr);
      /* combine opt var bounds with slack bounds */
      ierr = TaoALMMCombinePrimal_Private(tao, tao->XL, SL, auglag->PL);CHKERRQ(ierr);
      ierr = TaoALMMCombinePrimal_Private(tao, tao->XU, SU, auglag->PU);CHKERRQ(ierr);
      /* destroy work vectors */
      ierr = VecDestroy(&SL);CHKERRQ(ierr);
      ierr = VecDestroy(&SU);CHKERRQ(ierr);
    } else {
      /* no inequality constraints, just copy bounds into the subsolver */
      ierr = VecCopy(tao->XL, auglag->PL);CHKERRQ(ierr);
      ierr = VecCopy(tao->XU, auglag->PU);CHKERRQ(ierr);
    }
    ierr = TaoSetVariableBounds(auglag->subsolver, auglag->PL, auglag->PU);CHKERRQ(ierr);
  }
  ierr = TaoSetUp(auglag->subsolver);CHKERRQ(ierr);
  auglag->G = auglag->subsolver->gradient;

  PetscFunctionReturn(0);
}

static PetscErrorCode TaoDestroy_ALMM(Tao tao)
{
  TAO_ALMM       *auglag = (TAO_ALMM*)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TaoDestroy(&auglag->subsolver);CHKERRQ(ierr);
  if (tao->setupcalled) {
    ierr = VecDestroy(&auglag->Xwork);CHKERRQ(ierr);              /* opt work */
    if (tao->eq_constrained) {
      ierr = VecDestroy(&auglag->Ye);CHKERRQ(ierr);               /* equality multipliers */
      ierr = VecDestroy(&auglag->Cework);CHKERRQ(ierr);           /* equality work vector */
    }
    if (tao->ineq_constrained) {
      ierr = VecDestroy(&auglag->Ps);CHKERRQ(ierr);               /* slack vars */
      auglag->Parr[0] = NULL;                                     /* clear pointer to tao->solution, will be destroyed by TaoDestroy() shell */
      ierr = PetscFree(auglag->Parr);CHKERRQ(ierr);               /* array of primal vectors */
      ierr = VecDestroy(&auglag->LgradS);CHKERRQ(ierr);           /* slack grad */
      ierr = VecDestroy(&auglag->Cizero);CHKERRQ(ierr);           /* zero vector for pointwise max */
      ierr = VecDestroy(&auglag->Yi);CHKERRQ(ierr);               /* inequality multipliers */
      ierr = VecDestroy(&auglag->Ciwork);CHKERRQ(ierr);           /* inequality work vector */
      ierr = VecDestroy(&auglag->P);CHKERRQ(ierr);                /* combo primal */
      ierr = ISDestroy(&auglag->Pis[0]);CHKERRQ(ierr);            /* index set for X inside P */
      ierr = ISDestroy(&auglag->Pis[1]);CHKERRQ(ierr);            /* index set for S inside P */
      ierr = PetscFree(auglag->Pis);CHKERRQ(ierr);                /* array of P index sets */
      ierr = VecScatterDestroy(&auglag->Pscatter[0]);CHKERRQ(ierr);
      ierr = VecScatterDestroy(&auglag->Pscatter[1]);CHKERRQ(ierr);
      ierr = PetscFree(auglag->Pscatter);CHKERRQ(ierr);
      if (tao->eq_constrained) {
        ierr = VecDestroy(&auglag->Y);CHKERRQ(ierr);              /* combo multipliers */
        ierr = PetscFree(auglag->Yarr);CHKERRQ(ierr);             /* array of dual vectors */
        ierr = VecDestroy(&auglag->C);CHKERRQ(ierr);              /* combo constraints */
        ierr = ISDestroy(&auglag->Yis[0]);CHKERRQ(ierr);          /* index set for Ye inside Y */
        ierr = ISDestroy(&auglag->Yis[1]);CHKERRQ(ierr);          /* index set for Yi inside Y */
        ierr = PetscFree(auglag->Yis);CHKERRQ(ierr);
        ierr = VecScatterDestroy(&auglag->Yscatter[0]);CHKERRQ(ierr);
        ierr = VecScatterDestroy(&auglag->Yscatter[1]);CHKERRQ(ierr);
        ierr = PetscFree(auglag->Yscatter);CHKERRQ(ierr);
      }
    }
    if (tao->bounded) {
      ierr = VecDestroy(&auglag->PL);CHKERRQ(ierr);                /* lower bounds for subsolver */
      ierr = VecDestroy(&auglag->PU);CHKERRQ(ierr);                /* upper bounds for subsolver */
    }
  }
  ierr = PetscFree(tao->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSetFromOptions_ALMM(PetscOptionItems *PetscOptionsObject,Tao tao)
{
  TAO_ALMM       *auglag = (TAO_ALMM*)tao->data;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"Augmented Lagrangian multipler method solves problems with general constraints by converting them into a sequence of unconstrained problems.");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_almm_mu_init","initial penalty parameter","",auglag->mu0,&auglag->mu0,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_almm_mu_factor","increase factor for the penalty parameter","",auglag->mu_fac,&auglag->mu_fac,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_almm_mu_power_good","exponential for penalty parameter when multiplier update is accepted","",auglag->mu_pow_good,&auglag->mu_pow_good,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_almm_mu_power_bad","exponential for penalty parameter when multiplier update is rejected","",auglag->mu_pow_bad,&auglag->mu_pow_bad,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_almm_mu_max","maximum safeguard for penalty parameter updates","",auglag->mu_max,&auglag->mu_max,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_almm_ye_min","minimum safeguard for equality multiplier updates","",auglag->ye_min,&auglag->ye_min,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_almm_ye_max","maximum safeguard for equality multipliers updates","",auglag->ye_max,&auglag->ye_max,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_almm_yi_min","minimum safeguard for inequality multipliers updates","",auglag->yi_min,&auglag->yi_min,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_almm_yi_max","maximum safeguard for inequality multipliers updates","",auglag->yi_max,&auglag->yi_max,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-tao_almm_type","augmented Lagrangian formulation type for the subproblem","TaoALMMType",TaoALMMTypes,(PetscEnum)auglag->type,(PetscEnum*)&auglag->type,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  ierr = TaoSetOptionsPrefix(auglag->subsolver,((PetscObject)tao)->prefix);CHKERRQ(ierr);
  ierr = TaoAppendOptionsPrefix(auglag->subsolver,"tao_almm_subsolver_");CHKERRQ(ierr);
  ierr = TaoSetFromOptions(auglag->subsolver);CHKERRQ(ierr);
  for (i=0; i<tao->numbermonitors; i++) {
    ierr = PetscObjectReference((PetscObject)tao->monitorcontext[i]);CHKERRQ(ierr);
    ierr = TaoSetMonitor(auglag->subsolver, tao->monitor[i], tao->monitorcontext[i], tao->monitordestroy[i]);CHKERRQ(ierr);
    if (tao->monitor[i] == TaoMonitorDefault || tao->monitor[i] == TaoDefaultCMonitor || tao->monitor[i] == TaoDefaultGMonitor || tao->monitor[i] == TaoDefaultSMonitor) {
      auglag->info = PETSC_TRUE;
    }
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------- */

/*MC
  TaoALMM - Augmented Lagrangian multiplier method for solving nonlinear optimization problems with general constraints.

  Options Database Keys:
+ -tao_almm_mu_init <real>       - initial penalty parameter (default: 10.)
. -tao_almm_mu_factor <real>     - increase factor for the penalty parameter (default: 100.)
. -tao_almm_mu_max <real>        - maximum safeguard for penalty parameter updates (default: 1.e20)
. -tao_almm_mu_power_good <real> - exponential for penalty parameter when multiplier update is accepted (default: 0.9)
. -tao_almm_mu_power_bad <real>  - exponential for penalty parameter when multiplier update is rejected (default: 0.1)
. -tao_almm_ye_min <real>        - minimum safeguard for equality multiplier updates (default: -1.e20)
. -tao_almm_ye_max <real>        - maximum safeguard for equality multiplier updates (default: 1.e20)
. -tao_almm_yi_min <real>        - minimum safeguard for inequality multiplier updates (default: -1.e20)
. -tao_almm_yi_max <real>        - maximum safeguard for inequality multiplier updates (default: 1.e20)
- -tao_almm_type <classic,phr>   - change formulation of the augmented Lagrangian merit function for the subproblem (default: classic)

  Level: beginner

  Notes:
  This method converts a constrained problem into a sequence of unconstrained problems via the augmented
  Lagrangian merit function. Bound constraints are pushed down to the subproblem without any modifications.

  Two formulations are offered for the subproblem: canonical Hestenes-Powell augmented Lagrangian with slack
  variables for inequality constraints, and a slack-less Powell-Hestenes-Rockafellar (PHR) formulation utilizing a
  pointwise max() penalty on inequality constraints. The canonical augmented Lagrangian formulation typically
  converges faster for most problems. However, PHR may be desirable for problems featuring a large number
  of inequality constraints because it avoids inflating the size of the subproblem with slack variables.

  The subproblem is solved using a nested first-order TAO solver. The user can retrieve a pointer to
  the subsolver via TaoALMMGetSubsolver() or pass command line arguments to it using the
  "-tao_almm_subsolver_" prefix. Currently, TaoALMM does not support second-order methods for the
  subproblem. It is also highly recommended that the subsolver chosen by the user utilize a trust-region
  strategy for globalization (default: TAOBQNKTR) especially if the outer problem features bound constraints.

.vb
  while unconverged
    solve argmin_x L(x) s.t. l <= x <= u
    if ||c|| <= y_tol
      if ||c|| <= c_tol && ||Lgrad|| <= g_tol:
        problem converged, return solution
      else
        constraints sufficiently improved
        update multipliers and tighten tolerances
      endif
    else
      constraints did not improve
      update penalty and loosen tolerances
    endif
  endwhile
.ve

.seealso: TaoALMMGetType(), TaoALMMSetType(), TaoALMMSetSubsolver(), TaoALMMGetSubsolver(),
          TaoALMMGetMultipliers(), TaoALMMSetMultipliers(), TaoALMMGetPrimalIS(), TaoALMMGetDualIS()
M*/
PETSC_EXTERN PetscErrorCode TaoCreate_ALMM(Tao tao)
{
  TAO_ALMM       *auglag;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(tao, &auglag);CHKERRQ(ierr);

  tao->ops->destroy        = TaoDestroy_ALMM;
  tao->ops->setup          = TaoSetUp_ALMM;
  tao->ops->setfromoptions = TaoSetFromOptions_ALMM;
  tao->ops->view           = TaoView_ALMM;
  tao->ops->solve          = TaoSolve_ALMM;

  tao->gatol = 1.e-5;
  tao->grtol = 0.0;
  tao->gttol = 0.0;
  tao->catol = 1.e-5;
  tao->crtol = 0.0;

  tao->data           = (void*)auglag;
  auglag->parent      = tao;
  auglag->mu0         = 10.0;
  auglag->mu          = auglag->mu0;
  auglag->mu_fac      = 10.0;
  auglag->mu_max      = PETSC_INFINITY;
  auglag->mu_pow_good = 0.9;
  auglag->mu_pow_bad  = 0.1;
  auglag->ye_min      = PETSC_NINFINITY;
  auglag->ye_max      = PETSC_INFINITY;
  auglag->yi_min      = PETSC_NINFINITY;
  auglag->yi_max      = PETSC_INFINITY;
  auglag->ytol0       = 0.1/PetscPowReal(auglag->mu0, auglag->mu_pow_bad);
  auglag->ytol        = auglag->ytol0;
  auglag->gtol0       = 1.0/auglag->mu0;
  auglag->gtol        = auglag->gtol0;

  auglag->sub_obj     = TaoALMMComputeAugLagAndGradient_Private;
  auglag->type        = TAO_ALMM_CLASSIC;
  auglag->info        = PETSC_FALSE;

  ierr = TaoCreate(PetscObjectComm((PetscObject)tao),&auglag->subsolver);CHKERRQ(ierr);
  ierr = TaoSetType(auglag->subsolver, TAOBQNKTR);CHKERRQ(ierr);
  ierr = TaoSetTolerances(auglag->subsolver, auglag->gtol, 0.0, 0.0);CHKERRQ(ierr);
  ierr = TaoSetMaximumIterations(auglag->subsolver, 1000);CHKERRQ(ierr);
  ierr = TaoSetMaximumFunctionEvaluations(auglag->subsolver, 10000);CHKERRQ(ierr);
  ierr = TaoSetFunctionLowerBound(auglag->subsolver, PETSC_NINFINITY);CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject)auglag->subsolver,(PetscObject)tao,1);CHKERRQ(ierr);

  ierr = PetscObjectComposeFunction((PetscObject)tao, "TaoALMMGetType_C", TaoALMMGetType_Private);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)tao, "TaoALMMSetType_C", TaoALMMSetType_Private);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)tao, "TaoALMMGetSubsolver_C", TaoALMMGetSubsolver_Private);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)tao, "TaoALMMSetSubsolver_C", TaoALMMSetSubsolver_Private);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)tao, "TaoALMMGetMultipliers_C", TaoALMMGetMultipliers_Private);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)tao, "TaoALMMSetMultipliers_C", TaoALMMSetMultipliers_Private);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)tao, "TaoALMMGetPrimalIS_C", TaoALMMGetPrimalIS_Private);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)tao, "TaoALMMGetDualIS_C", TaoALMMGetDualIS_Private);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoALMMCombinePrimal_Private(Tao tao, Vec X, Vec S, Vec P)
{
  TAO_ALMM     *auglag = (TAO_ALMM*)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (tao->ineq_constrained) {
    ierr = VecScatterBegin(auglag->Pscatter[0], X, P, INSERT_VALUES, SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(auglag->Pscatter[0], X, P, INSERT_VALUES, SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterBegin(auglag->Pscatter[1], S, P, INSERT_VALUES, SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(auglag->Pscatter[1], S, P, INSERT_VALUES, SCATTER_REVERSE);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(X, P);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoALMMCombineDual_Private(Tao tao, Vec EQ, Vec IN, Vec Y)
{
  TAO_ALMM     *auglag = (TAO_ALMM*)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (tao->eq_constrained) {
    if (tao->ineq_constrained) {
      ierr = VecScatterBegin(auglag->Yscatter[0], EQ, Y, INSERT_VALUES, SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecScatterEnd(auglag->Yscatter[0], EQ, Y, INSERT_VALUES, SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecScatterBegin(auglag->Yscatter[1], IN, Y, INSERT_VALUES, SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecScatterEnd(auglag->Yscatter[1], IN, Y, INSERT_VALUES, SCATTER_REVERSE);CHKERRQ(ierr);
    } else {
      ierr = VecCopy(EQ, Y);CHKERRQ(ierr);
    }
  } else {
    ierr = VecCopy(IN, Y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoALMMSplitPrimal_Private(Tao tao, Vec P, Vec X, Vec S)
{
  TAO_ALMM     *auglag = (TAO_ALMM*)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (tao->ineq_constrained) {
    ierr = VecScatterBegin(auglag->Pscatter[0], P, X, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(auglag->Pscatter[0], P, X, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterBegin(auglag->Pscatter[1], P, S, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(auglag->Pscatter[1], P, S, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(P, X);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* this assumes that the latest constraints are stored in Ce and Ci, and also combined in C */
static PetscErrorCode TaoALMMComputeOptimalityNorms_Private(Tao tao)
{
  TAO_ALMM     *auglag = (TAO_ALMM*)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* if bounded, project the gradient */
  if (tao->bounded) {
    ierr = VecBoundGradientProjection(auglag->LgradX, auglag->Px, tao->XL, tao->XU, auglag->LgradX);CHKERRQ(ierr);
  }
  if (auglag->type == TAO_ALMM_PHR) {
    ierr = VecNorm(auglag->LgradX, NORM_INFINITY, &auglag->gnorm);CHKERRQ(ierr);
    auglag->cenorm = 0.0;
    if (tao->eq_constrained) {
      ierr = VecNorm(auglag->Ce, NORM_INFINITY, &auglag->cenorm);CHKERRQ(ierr);
    }
    auglag->cinorm = 0.0;
    if (tao->ineq_constrained) {
      ierr = VecCopy(auglag->Yi, auglag->Ciwork);CHKERRQ(ierr);
      ierr = VecScale(auglag->Ciwork, -1.0/auglag->mu);CHKERRQ(ierr);
      ierr = VecPointwiseMax(auglag->Ciwork, auglag->Ci, auglag->Ciwork);CHKERRQ(ierr);
      ierr = VecNorm(auglag->Ciwork, NORM_INFINITY, &auglag->cinorm);CHKERRQ(ierr);
    }
    auglag->cnorm_old = auglag->cnorm;
    auglag->cnorm = PetscMax(auglag->cenorm, auglag->cinorm);
    auglag->ytol = auglag->ytol0 * auglag->cnorm_old;
  } else {
    ierr = VecNorm(auglag->LgradX, NORM_2, &auglag->gnorm);CHKERRQ(ierr);
    ierr = VecNorm(auglag->C, NORM_2, &auglag->cnorm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoALMMEvaluateIterate_Private(Tao tao, Vec P)
{
  TAO_ALMM       *auglag = (TAO_ALMM*)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* split solution into primal and slack components */
  ierr = TaoALMMSplitPrimal_Private(tao, auglag->P, auglag->Px, auglag->Ps);CHKERRQ(ierr);

  /* compute f, df/dx and the constraints */
  ierr = TaoComputeObjectiveAndGradient(tao, auglag->Px, &auglag->fval, auglag->LgradX);CHKERRQ(ierr);
  if (tao->eq_constrained) {
    ierr = TaoComputeEqualityConstraints(tao, auglag->Px, auglag->Ce);CHKERRQ(ierr);
    ierr = TaoComputeJacobianEquality(tao, auglag->Px, auglag->Ae, auglag->Ae);CHKERRQ(ierr);
  }
  if (tao->ineq_constrained) {
    ierr = TaoComputeInequalityConstraints(tao, auglag->Px, auglag->Ci);CHKERRQ(ierr);
    ierr = TaoComputeJacobianInequality(tao, auglag->Px, auglag->Ai, auglag->Ai);CHKERRQ(ierr);
    switch (auglag->type) {
      case TAO_ALMM_CLASSIC:
        /* classic formulation converts inequality to equality constraints via slack variables */
        ierr = VecAXPY(auglag->Ci, -1.0, auglag->Ps);CHKERRQ(ierr);
        break;
      case TAO_ALMM_PHR:
        /* PHR is based on Ci <= 0 while TAO defines Ci >= 0 so we hit it with a negative sign */
        ierr = VecScale(auglag->Ci, -1.0);CHKERRQ(ierr);
        ierr = MatScale(auglag->Ai, -1.0);CHKERRQ(ierr);
        break;
      default:
        break;
    }
  }
  /* combine constraints into one vector */
  ierr = TaoALMMCombineDual_Private(tao, auglag->Ce, auglag->Ci, auglag->C);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
Lphr = f + 0.5*mu*[ (Ce + Ye/mu)^T (Ce + Ye/mu) + pmin(0, Ci + Yi/mu)^T pmin(0, Ci + Yi/mu)]

dLphr/dX = dF/dX + mu*[ (Ce + Ye/mu)^T Ae + pmin(0, Ci + Yi/mu)^T Ai]

dLphr/dS = 0
*/
static PetscErrorCode TaoALMMComputePHRLagAndGradient_Private(Tao tao)
{
  TAO_ALMM       *auglag = (TAO_ALMM*)tao->data;
  PetscReal      eq_norm=0.0, ineq_norm=0.0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TaoALMMEvaluateIterate_Private(tao, auglag->P);CHKERRQ(ierr);
  if (tao->eq_constrained) {
    /* Ce_work = mu*(Ce + Ye/mu) */
    ierr = VecWAXPY(auglag->Cework, 1.0/auglag->mu, auglag->Ye, auglag->Ce);CHKERRQ(ierr);
    ierr = VecDot(auglag->Cework, auglag->Cework, &eq_norm);CHKERRQ(ierr); /* contribution to scalar Lagrangian */
    ierr = VecScale(auglag->Cework, auglag->mu);CHKERRQ(ierr);
    /* dL/dX += mu*(Ce + Ye/mu)^T Ae */
    ierr = MatMultTransposeAdd(auglag->Ae, auglag->Cework, auglag->LgradX, auglag->LgradX);CHKERRQ(ierr);
  }
  if (tao->ineq_constrained) {
    /* Ci_work = mu * pmax(0, Ci + Yi/mu) where pmax() is pointwise max() */
    ierr = VecWAXPY(auglag->Ciwork, 1.0/auglag->mu, auglag->Yi, auglag->Ci);CHKERRQ(ierr);
    ierr = VecPointwiseMax(auglag->Ciwork, auglag->Cizero, auglag->Ciwork);CHKERRQ(ierr);
    ierr = VecDot(auglag->Ciwork, auglag->Ciwork, &ineq_norm);CHKERRQ(ierr); /* contribution to scalar Lagrangian */
    /* dL/dX += mu * pmax(0, Ci + Yi/mu)^T Ai */
    ierr = VecScale(auglag->Ciwork, auglag->mu);CHKERRQ(ierr);
    ierr = MatMultTransposeAdd(auglag->Ai, auglag->Ciwork, auglag->LgradX, auglag->LgradX);CHKERRQ(ierr);
    /* dL/dS = 0 because there are no slacks in PHR */
    ierr = VecZeroEntries(auglag->LgradS);CHKERRQ(ierr);
  }
  /* combine gradient together */
  ierr = TaoALMMCombinePrimal_Private(tao, auglag->LgradX, auglag->LgradS, auglag->G);CHKERRQ(ierr);
  /* compute L = f + 0.5 * mu * [(Ce + Ye/mu)^T (Ce + Ye/mu) + pmax(0, Ci + Yi/mu)^T pmax(0, Ci + Yi/mu)] */
  auglag->Lval = auglag->fval + 0.5*auglag->mu*(eq_norm + ineq_norm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
Lc = F + Ye^TCe + Yi^T(Ci - S) + 0.5*mu*[Ce^TCe + (Ci - S)^T(Ci - S)]

dLc/dX = dF/dX + Ye^TAe + Yi^TAi + 0.5*mu*[Ce^TAe + (Ci - S)^TAi]

dLc/dS = -[Yi + mu*(Ci - S)]
*/
static PetscErrorCode TaoALMMComputeAugLagAndGradient_Private(Tao tao)
{
  TAO_ALMM       *auglag = (TAO_ALMM*)tao->data;
  PetscReal      yeTce=0.0, yiTcims=0.0, ceTce=0.0, cimsTcims=0.0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TaoALMMEvaluateIterate_Private(tao, auglag->P);CHKERRQ(ierr);
  if (tao->eq_constrained) {
    /* compute scalar contributions */
    ierr = VecDot(auglag->Ye, auglag->Ce, &yeTce);CHKERRQ(ierr);
    ierr = VecDot(auglag->Ce, auglag->Ce, &ceTce);CHKERRQ(ierr);
    /* dL/dX += ye^T Ae */
    ierr = MatMultTransposeAdd(auglag->Ae, auglag->Ye, auglag->LgradX, auglag->LgradX);CHKERRQ(ierr);
    /* dL/dX += mu * ce^T Ae */
    ierr = MatMultTranspose(auglag->Ae, auglag->Ce, auglag->Xwork);CHKERRQ(ierr);
    ierr = VecAXPY(auglag->LgradX, auglag->mu, auglag->Xwork);CHKERRQ(ierr);
  }
  if (tao->ineq_constrained) {
    /* compute scalar contributions */
    ierr = VecDot(auglag->Yi, auglag->Ci, &yiTcims);CHKERRQ(ierr);
    ierr = VecDot(auglag->Ci, auglag->Ci, &cimsTcims);CHKERRQ(ierr);
    /* dL/dX += yi^T Ai */
    ierr = MatMultTransposeAdd(auglag->Ai, auglag->Yi, auglag->LgradX, auglag->LgradX);CHKERRQ(ierr);
    /* dL/dX += mu * (ci - s)^T Ai */
    ierr = MatMultTranspose(auglag->Ai, auglag->Ci, auglag->Xwork);CHKERRQ(ierr);
    ierr = VecAXPY(auglag->LgradX, auglag->mu, auglag->Xwork);CHKERRQ(ierr);
    /* dL/dS = -[yi + mu*(ci - s)] */
    ierr = VecWAXPY(auglag->LgradS, auglag->mu, auglag->Ci, auglag->Yi);CHKERRQ(ierr);
    ierr = VecScale(auglag->LgradS, -1.0);CHKERRQ(ierr);
  }
  /* combine gradient together */
  ierr = TaoALMMCombinePrimal_Private(tao, auglag->LgradX, auglag->LgradS, auglag->G);CHKERRQ(ierr);
  /* compute L = f + ye^T ce + yi^T (ci - s) + 0.5*mu*||ce||^2 + 0.5*mu*||ci - s||^2 */
  auglag->Lval = auglag->fval + yeTce + yiTcims + 0.5*auglag->mu*(ceTce + cimsTcims);
  PetscFunctionReturn(0);
}

PetscErrorCode TaoALMMSubsolverObjective_Private(Tao tao, Vec P, PetscReal *Lval, void *ctx)
{
  TAO_ALMM       *auglag = (TAO_ALMM*)ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCopy(P, auglag->P);CHKERRQ(ierr);
  ierr = (*auglag->sub_obj)(auglag->parent);CHKERRQ(ierr);
  *Lval = auglag->Lval;
  PetscFunctionReturn(0);
}

PetscErrorCode TaoALMMSubsolverObjectiveAndGradient_Private(Tao tao, Vec P, PetscReal *Lval, Vec G, void *ctx)
{
  TAO_ALMM       *auglag = (TAO_ALMM*)ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCopy(P, auglag->P);CHKERRQ(ierr);
  ierr = (*auglag->sub_obj)(auglag->parent);CHKERRQ(ierr);
  ierr = VecCopy(auglag->G, G);CHKERRQ(ierr);
  *Lval = auglag->Lval;
  PetscFunctionReturn(0);
}
