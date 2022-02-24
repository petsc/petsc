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

  PetscFunctionBegin;
  /* reset initial multiplier/slack guess */
  if (!tao->recycle) {
    if (tao->ineq_constrained) {
      CHKERRQ(VecZeroEntries(auglag->Ps));
      CHKERRQ(TaoALMMCombinePrimal_Private(tao, auglag->Px, auglag->Ps, auglag->P));
      CHKERRQ(VecZeroEntries(auglag->Yi));
    }
    if (tao->eq_constrained) {
      CHKERRQ(VecZeroEntries(auglag->Ye));
    }
  }

  /* compute initial nonlinear Lagrangian and its derivatives */
  CHKERRQ((*auglag->sub_obj)(tao));
  CHKERRQ(TaoALMMComputeOptimalityNorms_Private(tao));
  /* print initial step and check convergence */
  CHKERRQ(PetscInfo(tao,"Solving with %s formulation\n",TaoALMMTypes[auglag->type]));
  CHKERRQ(TaoLogConvergenceHistory(tao, auglag->Lval, auglag->gnorm, auglag->cnorm, tao->ksp_its));
  CHKERRQ(TaoMonitor(tao, tao->niter, auglag->fval, auglag->gnorm, auglag->cnorm, 0.0));
  CHKERRQ((*tao->ops->convergencetest)(tao, tao->cnvP));
  /* set initial penalty factor and inner solver tolerance */
  switch (auglag->type) {
    case TAO_ALMM_CLASSIC:
      auglag->mu = auglag->mu0;
      break;
    case TAO_ALMM_PHR:
      auglag->cenorm = 0.0;
      if (tao->eq_constrained) {
        CHKERRQ(VecDot(auglag->Ce, auglag->Ce, &auglag->cenorm));
      }
      auglag->cinorm = 0.0;
      if (tao->ineq_constrained) {
        CHKERRQ(VecCopy(auglag->Ci, auglag->Ciwork));
        CHKERRQ(VecScale(auglag->Ciwork, -1.0));
        CHKERRQ(VecPointwiseMax(auglag->Ciwork, auglag->Cizero, auglag->Ciwork));
        CHKERRQ(VecDot(auglag->Ciwork, auglag->Ciwork, &auglag->cinorm));
      }
      /* determine initial penalty factor based on the balance of constraint violation and objective function value */
      auglag->mu = PetscMax(1.e-6, PetscMin(10.0, 2.0*PetscAbsReal(auglag->fval)/(auglag->cenorm + auglag->cinorm)));
      break;
    default:
      break;
  }
  auglag->gtol = auglag->gtol0;
  CHKERRQ(PetscInfo(tao,"Initial penalty: %.2f\n",auglag->mu));

  /* start aug-lag outer loop */
  while (tao->reason == TAO_CONTINUE_ITERATING) {
    ++tao->niter;
    /* update subsolver tolerance */
    CHKERRQ(PetscInfo(tao,"Subsolver tolerance: ||G|| <= %e\n",auglag->gtol));
    CHKERRQ(TaoSetTolerances(auglag->subsolver, auglag->gtol, 0.0, 0.0));
    /* solve the bound-constrained or unconstrained subproblem */
    CHKERRQ(TaoSolve(auglag->subsolver));
    CHKERRQ(TaoGetConvergedReason(auglag->subsolver, &reason));
    tao->ksp_its += auglag->subsolver->ksp_its;
    if (reason != TAO_CONVERGED_GATOL) {
      CHKERRQ(PetscInfo(tao,"Subsolver failed to converge, reason: %s\n",TaoConvergedReasons[reason]));
    }
    /* evaluate solution and test convergence */
    CHKERRQ((*auglag->sub_obj)(tao));
    CHKERRQ(TaoALMMComputeOptimalityNorms_Private(tao));
    /* decide whether to update multipliers or not */
    updated = 0.0;
    if (auglag->cnorm <= auglag->ytol) {
      CHKERRQ(PetscInfo(tao,"Multipliers updated: ||C|| <= %e\n",auglag->ytol));
      /* constraints are good, update multipliers and convergence tolerances */
      if (tao->eq_constrained) {
        CHKERRQ(VecAXPY(auglag->Ye, auglag->mu, auglag->Ce));
        CHKERRQ(VecSet(auglag->Cework, auglag->ye_max));
        CHKERRQ(VecPointwiseMin(auglag->Ye, auglag->Cework, auglag->Ye));
        CHKERRQ(VecSet(auglag->Cework, auglag->ye_min));
        CHKERRQ(VecPointwiseMax(auglag->Ye, auglag->Cework, auglag->Ye));
      }
      if (tao->ineq_constrained) {
        CHKERRQ(VecAXPY(auglag->Yi, auglag->mu, auglag->Ci));
        CHKERRQ(VecSet(auglag->Ciwork, auglag->yi_max));
        CHKERRQ(VecPointwiseMin(auglag->Yi, auglag->Ciwork, auglag->Yi));
        CHKERRQ(VecSet(auglag->Ciwork, auglag->yi_min));
        CHKERRQ(VecPointwiseMax(auglag->Yi, auglag->Ciwork, auglag->Yi));
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
      CHKERRQ(PetscInfo(tao,"Penalty increased: mu = %.2f\n",auglag->mu));
    }
    CHKERRQ(TaoLogConvergenceHistory(tao, auglag->fval, auglag->gnorm, auglag->cnorm, tao->ksp_its));
    CHKERRQ(TaoMonitor(tao, tao->niter, auglag->fval, auglag->gnorm, auglag->cnorm, updated));
    CHKERRQ((*tao->ops->convergencetest)(tao, tao->cnvP));
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode TaoView_ALMM(Tao tao,PetscViewer viewer)
{
  TAO_ALMM       *auglag = (TAO_ALMM*)tao->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  CHKERRQ(PetscViewerASCIIPushTab(viewer));
  CHKERRQ(TaoView(auglag->subsolver,viewer));
  CHKERRQ(PetscViewerASCIIPopTab(viewer));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    CHKERRQ(PetscViewerASCIIPushTab(viewer));
    CHKERRQ(PetscViewerASCIIPrintf(viewer, "ALMM Formulation Type: %s\n", TaoALMMTypes[auglag->type]));
    CHKERRQ(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSetUp_ALMM(Tao tao)
{
  TAO_ALMM       *auglag = (TAO_ALMM*)tao->data;
  VecType        vec_type;
  Vec            SL, SU;
  PetscBool      is_cg = PETSC_FALSE, is_lmvm = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCheck(!tao->ineq_doublesided,PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_WRONGSTATE, "TAOALMM does not support double-sided inequality constraint definition. Please restructure your inequality constrainst to fit the form c(x) >= 0.");
  PetscCheck(tao->eq_constrained || tao->ineq_constrained,PetscObjectComm((PetscObject)tao), PETSC_ERR_ORDER, "Equality and/or inequality constraints must be defined before solver setup.");
  CHKERRQ(TaoComputeVariableBounds(tao));
  /* alias base vectors and create extras */
  CHKERRQ(VecGetType(tao->solution, &vec_type));
  auglag->Px = tao->solution;
  if (!tao->gradient) { /* base gradient */
    CHKERRQ(VecDuplicate(tao->solution, &tao->gradient));
  }
  auglag->LgradX = tao->gradient;
  if (!auglag->Xwork) { /* opt var work vector */
    CHKERRQ(VecDuplicate(tao->solution, &auglag->Xwork));
  }
  if (tao->eq_constrained) {
    auglag->Ce = tao->constraints_equality;
    auglag->Ae = tao->jacobian_equality;
    if (!auglag->Ye) { /* equality multipliers */
      CHKERRQ(VecDuplicate(auglag->Ce, &auglag->Ye));
    }
    if (!auglag->Cework) {
      CHKERRQ(VecDuplicate(auglag->Ce, &auglag->Cework));
    }
  }
  if (tao->ineq_constrained) {
    auglag->Ci = tao->constraints_inequality;
    auglag->Ai = tao->jacobian_inequality;
    if (!auglag->Yi) { /* inequality multipliers */
      CHKERRQ(VecDuplicate(auglag->Ci, &auglag->Yi));
    }
    if (!auglag->Ciwork) {
      CHKERRQ(VecDuplicate(auglag->Ci, &auglag->Ciwork));
    }
    if (!auglag->Cizero) {
      CHKERRQ(VecDuplicate(auglag->Ci, &auglag->Cizero));
      CHKERRQ(VecZeroEntries(auglag->Cizero));
    }
    if (!auglag->Ps) { /* slack vars */
      CHKERRQ(VecDuplicate(auglag->Ci, &auglag->Ps));
    }
    if (!auglag->LgradS) { /* slack component of Lagrangian gradient */
      CHKERRQ(VecDuplicate(auglag->Ci, &auglag->LgradS));
    }
    /* create vector for combined primal space and the associated communication objects */
    if (!auglag->P) {
      CHKERRQ(PetscMalloc1(2, &auglag->Parr));
      auglag->Parr[0] = auglag->Px; auglag->Parr[1] = auglag->Ps;
      CHKERRQ(VecConcatenate(2, auglag->Parr, &auglag->P, &auglag->Pis));
      CHKERRQ(PetscMalloc1(2, &auglag->Pscatter));
      CHKERRQ(VecScatterCreate(auglag->P, auglag->Pis[0], auglag->Px, NULL, &auglag->Pscatter[0]));
      CHKERRQ(VecScatterCreate(auglag->P, auglag->Pis[1], auglag->Ps, NULL, &auglag->Pscatter[1]));
    }
    if (tao->eq_constrained) {
      /* create vector for combined dual space and the associated communication objects */
      if (!auglag->Y) {
        CHKERRQ(PetscMalloc1(2, &auglag->Yarr));
        auglag->Yarr[0] = auglag->Ye; auglag->Yarr[1] = auglag->Yi;
        CHKERRQ(VecConcatenate(2, auglag->Yarr, &auglag->Y, &auglag->Yis));
        CHKERRQ(PetscMalloc1(2, &auglag->Yscatter));
        CHKERRQ(VecScatterCreate(auglag->Y, auglag->Yis[0], auglag->Ye, NULL, &auglag->Yscatter[0]));
        CHKERRQ(VecScatterCreate(auglag->Y, auglag->Yis[1], auglag->Yi, NULL, &auglag->Yscatter[1]));
      }
      if (!auglag->C) {
        CHKERRQ(VecDuplicate(auglag->Y, &auglag->C));
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
      CHKERRQ(PetscInfo(tao,"TAOALMM with PHR: different gradient and constraint tolerances are not supported, setting catol = gatol\n"));
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
  CHKERRQ(TaoSetSolution(auglag->subsolver, auglag->P));
  CHKERRQ(TaoSetObjective(auglag->subsolver, TaoALMMSubsolverObjective_Private, (void*)auglag));
  CHKERRQ(TaoSetObjectiveAndGradient(auglag->subsolver, NULL, TaoALMMSubsolverObjectiveAndGradient_Private, (void*)auglag));
  if (tao->bounded) {
    /* make sure that the subsolver is a bound-constrained method */
    CHKERRQ(PetscObjectTypeCompare((PetscObject)auglag->subsolver, TAOCG, &is_cg));
    CHKERRQ(PetscObjectTypeCompare((PetscObject)auglag->subsolver, TAOLMVM, &is_lmvm));
    if (is_cg) {
      CHKERRQ(TaoSetType(auglag->subsolver, TAOBNCG));
      CHKERRQ(PetscInfo(tao,"TAOCG detected for bound-constrained problem, switching to TAOBNCG instead."));
    }
    if (is_lmvm) {
      CHKERRQ(TaoSetType(auglag->subsolver, TAOBQNLS));
      CHKERRQ(PetscInfo(tao,"TAOLMVM detected for bound-constrained problem, switching to TAOBQNLS instead."));
    }
    /* create lower and upper bound clone vectors for subsolver */
    if (!auglag->PL) {
      CHKERRQ(VecDuplicate(auglag->P, &auglag->PL));
    }
    if (!auglag->PU) {
      CHKERRQ(VecDuplicate(auglag->P, &auglag->PU));
    }
    if (tao->ineq_constrained) {
      /* create lower and upper bounds for slack, set lower to 0 */
      CHKERRQ(VecDuplicate(auglag->Ci, &SL));
      CHKERRQ(VecSet(SL, 0.0));
      CHKERRQ(VecDuplicate(auglag->Ci, &SU));
      CHKERRQ(VecSet(SU, PETSC_INFINITY));
      /* combine opt var bounds with slack bounds */
      CHKERRQ(TaoALMMCombinePrimal_Private(tao, tao->XL, SL, auglag->PL));
      CHKERRQ(TaoALMMCombinePrimal_Private(tao, tao->XU, SU, auglag->PU));
      /* destroy work vectors */
      CHKERRQ(VecDestroy(&SL));
      CHKERRQ(VecDestroy(&SU));
    } else {
      /* no inequality constraints, just copy bounds into the subsolver */
      CHKERRQ(VecCopy(tao->XL, auglag->PL));
      CHKERRQ(VecCopy(tao->XU, auglag->PU));
    }
    CHKERRQ(TaoSetVariableBounds(auglag->subsolver, auglag->PL, auglag->PU));
  }
  CHKERRQ(TaoSetUp(auglag->subsolver));
  auglag->G = auglag->subsolver->gradient;

  PetscFunctionReturn(0);
}

static PetscErrorCode TaoDestroy_ALMM(Tao tao)
{
  TAO_ALMM       *auglag = (TAO_ALMM*)tao->data;

  PetscFunctionBegin;
  CHKERRQ(TaoDestroy(&auglag->subsolver));
  if (tao->setupcalled) {
    CHKERRQ(VecDestroy(&auglag->Xwork));              /* opt work */
    if (tao->eq_constrained) {
      CHKERRQ(VecDestroy(&auglag->Ye));               /* equality multipliers */
      CHKERRQ(VecDestroy(&auglag->Cework));           /* equality work vector */
    }
    if (tao->ineq_constrained) {
      CHKERRQ(VecDestroy(&auglag->Ps));               /* slack vars */
      auglag->Parr[0] = NULL;                                     /* clear pointer to tao->solution, will be destroyed by TaoDestroy() shell */
      CHKERRQ(PetscFree(auglag->Parr));               /* array of primal vectors */
      CHKERRQ(VecDestroy(&auglag->LgradS));           /* slack grad */
      CHKERRQ(VecDestroy(&auglag->Cizero));           /* zero vector for pointwise max */
      CHKERRQ(VecDestroy(&auglag->Yi));               /* inequality multipliers */
      CHKERRQ(VecDestroy(&auglag->Ciwork));           /* inequality work vector */
      CHKERRQ(VecDestroy(&auglag->P));                /* combo primal */
      CHKERRQ(ISDestroy(&auglag->Pis[0]));            /* index set for X inside P */
      CHKERRQ(ISDestroy(&auglag->Pis[1]));            /* index set for S inside P */
      CHKERRQ(PetscFree(auglag->Pis));                /* array of P index sets */
      CHKERRQ(VecScatterDestroy(&auglag->Pscatter[0]));
      CHKERRQ(VecScatterDestroy(&auglag->Pscatter[1]));
      CHKERRQ(PetscFree(auglag->Pscatter));
      if (tao->eq_constrained) {
        CHKERRQ(VecDestroy(&auglag->Y));              /* combo multipliers */
        CHKERRQ(PetscFree(auglag->Yarr));             /* array of dual vectors */
        CHKERRQ(VecDestroy(&auglag->C));              /* combo constraints */
        CHKERRQ(ISDestroy(&auglag->Yis[0]));          /* index set for Ye inside Y */
        CHKERRQ(ISDestroy(&auglag->Yis[1]));          /* index set for Yi inside Y */
        CHKERRQ(PetscFree(auglag->Yis));
        CHKERRQ(VecScatterDestroy(&auglag->Yscatter[0]));
        CHKERRQ(VecScatterDestroy(&auglag->Yscatter[1]));
        CHKERRQ(PetscFree(auglag->Yscatter));
      }
    }
    if (tao->bounded) {
      CHKERRQ(VecDestroy(&auglag->PL));                /* lower bounds for subsolver */
      CHKERRQ(VecDestroy(&auglag->PU));                /* upper bounds for subsolver */
    }
  }
  CHKERRQ(PetscFree(tao->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSetFromOptions_ALMM(PetscOptionItems *PetscOptionsObject,Tao tao)
{
  TAO_ALMM       *auglag = (TAO_ALMM*)tao->data;
  PetscInt       i;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"Augmented Lagrangian multipler method solves problems with general constraints by converting them into a sequence of unconstrained problems."));
  CHKERRQ(PetscOptionsReal("-tao_almm_mu_init","initial penalty parameter","",auglag->mu0,&auglag->mu0,NULL));
  CHKERRQ(PetscOptionsReal("-tao_almm_mu_factor","increase factor for the penalty parameter","",auglag->mu_fac,&auglag->mu_fac,NULL));
  CHKERRQ(PetscOptionsReal("-tao_almm_mu_power_good","exponential for penalty parameter when multiplier update is accepted","",auglag->mu_pow_good,&auglag->mu_pow_good,NULL));
  CHKERRQ(PetscOptionsReal("-tao_almm_mu_power_bad","exponential for penalty parameter when multiplier update is rejected","",auglag->mu_pow_bad,&auglag->mu_pow_bad,NULL));
  CHKERRQ(PetscOptionsReal("-tao_almm_mu_max","maximum safeguard for penalty parameter updates","",auglag->mu_max,&auglag->mu_max,NULL));
  CHKERRQ(PetscOptionsReal("-tao_almm_ye_min","minimum safeguard for equality multiplier updates","",auglag->ye_min,&auglag->ye_min,NULL));
  CHKERRQ(PetscOptionsReal("-tao_almm_ye_max","maximum safeguard for equality multipliers updates","",auglag->ye_max,&auglag->ye_max,NULL));
  CHKERRQ(PetscOptionsReal("-tao_almm_yi_min","minimum safeguard for inequality multipliers updates","",auglag->yi_min,&auglag->yi_min,NULL));
  CHKERRQ(PetscOptionsReal("-tao_almm_yi_max","maximum safeguard for inequality multipliers updates","",auglag->yi_max,&auglag->yi_max,NULL));
  CHKERRQ(PetscOptionsEnum("-tao_almm_type","augmented Lagrangian formulation type for the subproblem","TaoALMMType",TaoALMMTypes,(PetscEnum)auglag->type,(PetscEnum*)&auglag->type,NULL));
  CHKERRQ(PetscOptionsTail());
  CHKERRQ(TaoSetOptionsPrefix(auglag->subsolver,((PetscObject)tao)->prefix));
  CHKERRQ(TaoAppendOptionsPrefix(auglag->subsolver,"tao_almm_subsolver_"));
  CHKERRQ(TaoSetFromOptions(auglag->subsolver));
  for (i=0; i<tao->numbermonitors; i++) {
    CHKERRQ(PetscObjectReference((PetscObject)tao->monitorcontext[i]));
    CHKERRQ(TaoSetMonitor(auglag->subsolver, tao->monitor[i], tao->monitorcontext[i], tao->monitordestroy[i]));
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

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(tao, &auglag));

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

  CHKERRQ(TaoCreate(PetscObjectComm((PetscObject)tao),&auglag->subsolver));
  CHKERRQ(TaoSetType(auglag->subsolver, TAOBQNKTR));
  CHKERRQ(TaoSetTolerances(auglag->subsolver, auglag->gtol, 0.0, 0.0));
  CHKERRQ(TaoSetMaximumIterations(auglag->subsolver, 1000));
  CHKERRQ(TaoSetMaximumFunctionEvaluations(auglag->subsolver, 10000));
  CHKERRQ(TaoSetFunctionLowerBound(auglag->subsolver, PETSC_NINFINITY));
  CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)auglag->subsolver,(PetscObject)tao,1));

  CHKERRQ(PetscObjectComposeFunction((PetscObject)tao, "TaoALMMGetType_C", TaoALMMGetType_Private));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)tao, "TaoALMMSetType_C", TaoALMMSetType_Private));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)tao, "TaoALMMGetSubsolver_C", TaoALMMGetSubsolver_Private));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)tao, "TaoALMMSetSubsolver_C", TaoALMMSetSubsolver_Private));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)tao, "TaoALMMGetMultipliers_C", TaoALMMGetMultipliers_Private));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)tao, "TaoALMMSetMultipliers_C", TaoALMMSetMultipliers_Private));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)tao, "TaoALMMGetPrimalIS_C", TaoALMMGetPrimalIS_Private));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)tao, "TaoALMMGetDualIS_C", TaoALMMGetDualIS_Private));
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoALMMCombinePrimal_Private(Tao tao, Vec X, Vec S, Vec P)
{
  TAO_ALMM     *auglag = (TAO_ALMM*)tao->data;

  PetscFunctionBegin;
  if (tao->ineq_constrained) {
    CHKERRQ(VecScatterBegin(auglag->Pscatter[0], X, P, INSERT_VALUES, SCATTER_REVERSE));
    CHKERRQ(VecScatterEnd(auglag->Pscatter[0], X, P, INSERT_VALUES, SCATTER_REVERSE));
    CHKERRQ(VecScatterBegin(auglag->Pscatter[1], S, P, INSERT_VALUES, SCATTER_REVERSE));
    CHKERRQ(VecScatterEnd(auglag->Pscatter[1], S, P, INSERT_VALUES, SCATTER_REVERSE));
  } else {
    CHKERRQ(VecCopy(X, P));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoALMMCombineDual_Private(Tao tao, Vec EQ, Vec IN, Vec Y)
{
  TAO_ALMM     *auglag = (TAO_ALMM*)tao->data;

  PetscFunctionBegin;
  if (tao->eq_constrained) {
    if (tao->ineq_constrained) {
      CHKERRQ(VecScatterBegin(auglag->Yscatter[0], EQ, Y, INSERT_VALUES, SCATTER_REVERSE));
      CHKERRQ(VecScatterEnd(auglag->Yscatter[0], EQ, Y, INSERT_VALUES, SCATTER_REVERSE));
      CHKERRQ(VecScatterBegin(auglag->Yscatter[1], IN, Y, INSERT_VALUES, SCATTER_REVERSE));
      CHKERRQ(VecScatterEnd(auglag->Yscatter[1], IN, Y, INSERT_VALUES, SCATTER_REVERSE));
    } else {
      CHKERRQ(VecCopy(EQ, Y));
    }
  } else {
    CHKERRQ(VecCopy(IN, Y));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoALMMSplitPrimal_Private(Tao tao, Vec P, Vec X, Vec S)
{
  TAO_ALMM     *auglag = (TAO_ALMM*)tao->data;

  PetscFunctionBegin;
  if (tao->ineq_constrained) {
    CHKERRQ(VecScatterBegin(auglag->Pscatter[0], P, X, INSERT_VALUES, SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(auglag->Pscatter[0], P, X, INSERT_VALUES, SCATTER_FORWARD));
    CHKERRQ(VecScatterBegin(auglag->Pscatter[1], P, S, INSERT_VALUES, SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(auglag->Pscatter[1], P, S, INSERT_VALUES, SCATTER_FORWARD));
  } else {
    CHKERRQ(VecCopy(P, X));
  }
  PetscFunctionReturn(0);
}

/* this assumes that the latest constraints are stored in Ce and Ci, and also combined in C */
static PetscErrorCode TaoALMMComputeOptimalityNorms_Private(Tao tao)
{
  TAO_ALMM     *auglag = (TAO_ALMM*)tao->data;

  PetscFunctionBegin;
  /* if bounded, project the gradient */
  if (tao->bounded) {
    CHKERRQ(VecBoundGradientProjection(auglag->LgradX, auglag->Px, tao->XL, tao->XU, auglag->LgradX));
  }
  if (auglag->type == TAO_ALMM_PHR) {
    CHKERRQ(VecNorm(auglag->LgradX, NORM_INFINITY, &auglag->gnorm));
    auglag->cenorm = 0.0;
    if (tao->eq_constrained) {
      CHKERRQ(VecNorm(auglag->Ce, NORM_INFINITY, &auglag->cenorm));
    }
    auglag->cinorm = 0.0;
    if (tao->ineq_constrained) {
      CHKERRQ(VecCopy(auglag->Yi, auglag->Ciwork));
      CHKERRQ(VecScale(auglag->Ciwork, -1.0/auglag->mu));
      CHKERRQ(VecPointwiseMax(auglag->Ciwork, auglag->Ci, auglag->Ciwork));
      CHKERRQ(VecNorm(auglag->Ciwork, NORM_INFINITY, &auglag->cinorm));
    }
    auglag->cnorm_old = auglag->cnorm;
    auglag->cnorm = PetscMax(auglag->cenorm, auglag->cinorm);
    auglag->ytol = auglag->ytol0 * auglag->cnorm_old;
  } else {
    CHKERRQ(VecNorm(auglag->LgradX, NORM_2, &auglag->gnorm));
    CHKERRQ(VecNorm(auglag->C, NORM_2, &auglag->cnorm));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoALMMEvaluateIterate_Private(Tao tao, Vec P)
{
  TAO_ALMM       *auglag = (TAO_ALMM*)tao->data;

  PetscFunctionBegin;
  /* split solution into primal and slack components */
  CHKERRQ(TaoALMMSplitPrimal_Private(tao, auglag->P, auglag->Px, auglag->Ps));

  /* compute f, df/dx and the constraints */
  CHKERRQ(TaoComputeObjectiveAndGradient(tao, auglag->Px, &auglag->fval, auglag->LgradX));
  if (tao->eq_constrained) {
    CHKERRQ(TaoComputeEqualityConstraints(tao, auglag->Px, auglag->Ce));
    CHKERRQ(TaoComputeJacobianEquality(tao, auglag->Px, auglag->Ae, auglag->Ae));
  }
  if (tao->ineq_constrained) {
    CHKERRQ(TaoComputeInequalityConstraints(tao, auglag->Px, auglag->Ci));
    CHKERRQ(TaoComputeJacobianInequality(tao, auglag->Px, auglag->Ai, auglag->Ai));
    switch (auglag->type) {
      case TAO_ALMM_CLASSIC:
        /* classic formulation converts inequality to equality constraints via slack variables */
        CHKERRQ(VecAXPY(auglag->Ci, -1.0, auglag->Ps));
        break;
      case TAO_ALMM_PHR:
        /* PHR is based on Ci <= 0 while TAO defines Ci >= 0 so we hit it with a negative sign */
        CHKERRQ(VecScale(auglag->Ci, -1.0));
        CHKERRQ(MatScale(auglag->Ai, -1.0));
        break;
      default:
        break;
    }
  }
  /* combine constraints into one vector */
  CHKERRQ(TaoALMMCombineDual_Private(tao, auglag->Ce, auglag->Ci, auglag->C));
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

  PetscFunctionBegin;
  CHKERRQ(TaoALMMEvaluateIterate_Private(tao, auglag->P));
  if (tao->eq_constrained) {
    /* Ce_work = mu*(Ce + Ye/mu) */
    CHKERRQ(VecWAXPY(auglag->Cework, 1.0/auglag->mu, auglag->Ye, auglag->Ce));
    CHKERRQ(VecDot(auglag->Cework, auglag->Cework, &eq_norm)); /* contribution to scalar Lagrangian */
    CHKERRQ(VecScale(auglag->Cework, auglag->mu));
    /* dL/dX += mu*(Ce + Ye/mu)^T Ae */
    CHKERRQ(MatMultTransposeAdd(auglag->Ae, auglag->Cework, auglag->LgradX, auglag->LgradX));
  }
  if (tao->ineq_constrained) {
    /* Ci_work = mu * pmax(0, Ci + Yi/mu) where pmax() is pointwise max() */
    CHKERRQ(VecWAXPY(auglag->Ciwork, 1.0/auglag->mu, auglag->Yi, auglag->Ci));
    CHKERRQ(VecPointwiseMax(auglag->Ciwork, auglag->Cizero, auglag->Ciwork));
    CHKERRQ(VecDot(auglag->Ciwork, auglag->Ciwork, &ineq_norm)); /* contribution to scalar Lagrangian */
    /* dL/dX += mu * pmax(0, Ci + Yi/mu)^T Ai */
    CHKERRQ(VecScale(auglag->Ciwork, auglag->mu));
    CHKERRQ(MatMultTransposeAdd(auglag->Ai, auglag->Ciwork, auglag->LgradX, auglag->LgradX));
    /* dL/dS = 0 because there are no slacks in PHR */
    CHKERRQ(VecZeroEntries(auglag->LgradS));
  }
  /* combine gradient together */
  CHKERRQ(TaoALMMCombinePrimal_Private(tao, auglag->LgradX, auglag->LgradS, auglag->G));
  /* compute L = f + 0.5 * mu * [(Ce + Ye/mu)^T (Ce + Ye/mu) + pmax(0, Ci + Yi/mu)^T pmax(0, Ci + Yi/mu)] */
  auglag->Lval = auglag->fval + 0.5*auglag->mu*(eq_norm + ineq_norm);
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

  PetscFunctionBegin;
  CHKERRQ(TaoALMMEvaluateIterate_Private(tao, auglag->P));
  if (tao->eq_constrained) {
    /* compute scalar contributions */
    CHKERRQ(VecDot(auglag->Ye, auglag->Ce, &yeTce));
    CHKERRQ(VecDot(auglag->Ce, auglag->Ce, &ceTce));
    /* dL/dX += ye^T Ae */
    CHKERRQ(MatMultTransposeAdd(auglag->Ae, auglag->Ye, auglag->LgradX, auglag->LgradX));
    /* dL/dX += mu * ce^T Ae */
    CHKERRQ(MatMultTranspose(auglag->Ae, auglag->Ce, auglag->Xwork));
    CHKERRQ(VecAXPY(auglag->LgradX, auglag->mu, auglag->Xwork));
  }
  if (tao->ineq_constrained) {
    /* compute scalar contributions */
    CHKERRQ(VecDot(auglag->Yi, auglag->Ci, &yiTcims));
    CHKERRQ(VecDot(auglag->Ci, auglag->Ci, &cimsTcims));
    /* dL/dX += yi^T Ai */
    CHKERRQ(MatMultTransposeAdd(auglag->Ai, auglag->Yi, auglag->LgradX, auglag->LgradX));
    /* dL/dX += mu * (ci - s)^T Ai */
    CHKERRQ(MatMultTranspose(auglag->Ai, auglag->Ci, auglag->Xwork));
    CHKERRQ(VecAXPY(auglag->LgradX, auglag->mu, auglag->Xwork));
    /* dL/dS = -[yi + mu*(ci - s)] */
    CHKERRQ(VecWAXPY(auglag->LgradS, auglag->mu, auglag->Ci, auglag->Yi));
    CHKERRQ(VecScale(auglag->LgradS, -1.0));
  }
  /* combine gradient together */
  CHKERRQ(TaoALMMCombinePrimal_Private(tao, auglag->LgradX, auglag->LgradS, auglag->G));
  /* compute L = f + ye^T ce + yi^T (ci - s) + 0.5*mu*||ce||^2 + 0.5*mu*||ci - s||^2 */
  auglag->Lval = auglag->fval + yeTce + yiTcims + 0.5*auglag->mu*(ceTce + cimsTcims);
  PetscFunctionReturn(0);
}

PetscErrorCode TaoALMMSubsolverObjective_Private(Tao tao, Vec P, PetscReal *Lval, void *ctx)
{
  TAO_ALMM       *auglag = (TAO_ALMM*)ctx;

  PetscFunctionBegin;
  CHKERRQ(VecCopy(P, auglag->P));
  CHKERRQ((*auglag->sub_obj)(auglag->parent));
  *Lval = auglag->Lval;
  PetscFunctionReturn(0);
}

PetscErrorCode TaoALMMSubsolverObjectiveAndGradient_Private(Tao tao, Vec P, PetscReal *Lval, Vec G, void *ctx)
{
  TAO_ALMM       *auglag = (TAO_ALMM*)ctx;

  PetscFunctionBegin;
  CHKERRQ(VecCopy(P, auglag->P));
  CHKERRQ((*auglag->sub_obj)(auglag->parent));
  CHKERRQ(VecCopy(auglag->G, G));
  *Lval = auglag->Lval;
  PetscFunctionReturn(0);
}
