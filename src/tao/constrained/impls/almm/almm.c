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
      PetscCall(VecZeroEntries(auglag->Ps));
      PetscCall(TaoALMMCombinePrimal_Private(tao, auglag->Px, auglag->Ps, auglag->P));
      PetscCall(VecZeroEntries(auglag->Yi));
    }
    if (tao->eq_constrained) {
      PetscCall(VecZeroEntries(auglag->Ye));
    }
  }

  /* compute initial nonlinear Lagrangian and its derivatives */
  PetscCall((*auglag->sub_obj)(tao));
  PetscCall(TaoALMMComputeOptimalityNorms_Private(tao));
  /* print initial step and check convergence */
  PetscCall(PetscInfo(tao,"Solving with %s formulation\n",TaoALMMTypes[auglag->type]));
  PetscCall(TaoLogConvergenceHistory(tao, auglag->Lval, auglag->gnorm, auglag->cnorm, tao->ksp_its));
  PetscCall(TaoMonitor(tao, tao->niter, auglag->fval, auglag->gnorm, auglag->cnorm, 0.0));
  PetscCall((*tao->ops->convergencetest)(tao, tao->cnvP));
  /* set initial penalty factor and inner solver tolerance */
  switch (auglag->type) {
    case TAO_ALMM_CLASSIC:
      auglag->mu = auglag->mu0;
      break;
    case TAO_ALMM_PHR:
      auglag->cenorm = 0.0;
      if (tao->eq_constrained) {
        PetscCall(VecDot(auglag->Ce, auglag->Ce, &auglag->cenorm));
      }
      auglag->cinorm = 0.0;
      if (tao->ineq_constrained) {
        PetscCall(VecCopy(auglag->Ci, auglag->Ciwork));
        PetscCall(VecScale(auglag->Ciwork, -1.0));
        PetscCall(VecPointwiseMax(auglag->Ciwork, auglag->Cizero, auglag->Ciwork));
        PetscCall(VecDot(auglag->Ciwork, auglag->Ciwork, &auglag->cinorm));
      }
      /* determine initial penalty factor based on the balance of constraint violation and objective function value */
      auglag->mu = PetscMax(1.e-6, PetscMin(10.0, 2.0*PetscAbsReal(auglag->fval)/(auglag->cenorm + auglag->cinorm)));
      break;
    default:
      break;
  }
  auglag->gtol = auglag->gtol0;
  PetscCall(PetscInfo(tao,"Initial penalty: %.2f\n",auglag->mu));

  /* start aug-lag outer loop */
  while (tao->reason == TAO_CONTINUE_ITERATING) {
    ++tao->niter;
    /* update subsolver tolerance */
    PetscCall(PetscInfo(tao,"Subsolver tolerance: ||G|| <= %e\n",auglag->gtol));
    PetscCall(TaoSetTolerances(auglag->subsolver, auglag->gtol, 0.0, 0.0));
    /* solve the bound-constrained or unconstrained subproblem */
    PetscCall(TaoSolve(auglag->subsolver));
    PetscCall(TaoGetConvergedReason(auglag->subsolver, &reason));
    tao->ksp_its += auglag->subsolver->ksp_its;
    if (reason != TAO_CONVERGED_GATOL) {
      PetscCall(PetscInfo(tao,"Subsolver failed to converge, reason: %s\n",TaoConvergedReasons[reason]));
    }
    /* evaluate solution and test convergence */
    PetscCall((*auglag->sub_obj)(tao));
    PetscCall(TaoALMMComputeOptimalityNorms_Private(tao));
    /* decide whether to update multipliers or not */
    updated = 0.0;
    if (auglag->cnorm <= auglag->ytol) {
      PetscCall(PetscInfo(tao,"Multipliers updated: ||C|| <= %e\n",auglag->ytol));
      /* constraints are good, update multipliers and convergence tolerances */
      if (tao->eq_constrained) {
        PetscCall(VecAXPY(auglag->Ye, auglag->mu, auglag->Ce));
        PetscCall(VecSet(auglag->Cework, auglag->ye_max));
        PetscCall(VecPointwiseMin(auglag->Ye, auglag->Cework, auglag->Ye));
        PetscCall(VecSet(auglag->Cework, auglag->ye_min));
        PetscCall(VecPointwiseMax(auglag->Ye, auglag->Cework, auglag->Ye));
      }
      if (tao->ineq_constrained) {
        PetscCall(VecAXPY(auglag->Yi, auglag->mu, auglag->Ci));
        PetscCall(VecSet(auglag->Ciwork, auglag->yi_max));
        PetscCall(VecPointwiseMin(auglag->Yi, auglag->Ciwork, auglag->Yi));
        PetscCall(VecSet(auglag->Ciwork, auglag->yi_min));
        PetscCall(VecPointwiseMax(auglag->Yi, auglag->Ciwork, auglag->Yi));
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
      PetscCall(PetscInfo(tao,"Penalty increased: mu = %.2f\n",auglag->mu));
    }
    PetscCall(TaoLogConvergenceHistory(tao, auglag->fval, auglag->gnorm, auglag->cnorm, tao->ksp_its));
    PetscCall(TaoMonitor(tao, tao->niter, auglag->fval, auglag->gnorm, auglag->cnorm, updated));
    PetscCall((*tao->ops->convergencetest)(tao, tao->cnvP));
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode TaoView_ALMM(Tao tao,PetscViewer viewer)
{
  TAO_ALMM       *auglag = (TAO_ALMM*)tao->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  PetscCall(PetscViewerASCIIPushTab(viewer));
  PetscCall(TaoView(auglag->subsolver,viewer));
  PetscCall(PetscViewerASCIIPopTab(viewer));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer, "ALMM Formulation Type: %s\n", TaoALMMTypes[auglag->type]));
    PetscCall(PetscViewerASCIIPopTab(viewer));
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
  PetscCall(TaoComputeVariableBounds(tao));
  /* alias base vectors and create extras */
  PetscCall(VecGetType(tao->solution, &vec_type));
  auglag->Px = tao->solution;
  if (!tao->gradient) { /* base gradient */
    PetscCall(VecDuplicate(tao->solution, &tao->gradient));
  }
  auglag->LgradX = tao->gradient;
  if (!auglag->Xwork) { /* opt var work vector */
    PetscCall(VecDuplicate(tao->solution, &auglag->Xwork));
  }
  if (tao->eq_constrained) {
    auglag->Ce = tao->constraints_equality;
    auglag->Ae = tao->jacobian_equality;
    if (!auglag->Ye) { /* equality multipliers */
      PetscCall(VecDuplicate(auglag->Ce, &auglag->Ye));
    }
    if (!auglag->Cework) {
      PetscCall(VecDuplicate(auglag->Ce, &auglag->Cework));
    }
  }
  if (tao->ineq_constrained) {
    auglag->Ci = tao->constraints_inequality;
    auglag->Ai = tao->jacobian_inequality;
    if (!auglag->Yi) { /* inequality multipliers */
      PetscCall(VecDuplicate(auglag->Ci, &auglag->Yi));
    }
    if (!auglag->Ciwork) {
      PetscCall(VecDuplicate(auglag->Ci, &auglag->Ciwork));
    }
    if (!auglag->Cizero) {
      PetscCall(VecDuplicate(auglag->Ci, &auglag->Cizero));
      PetscCall(VecZeroEntries(auglag->Cizero));
    }
    if (!auglag->Ps) { /* slack vars */
      PetscCall(VecDuplicate(auglag->Ci, &auglag->Ps));
    }
    if (!auglag->LgradS) { /* slack component of Lagrangian gradient */
      PetscCall(VecDuplicate(auglag->Ci, &auglag->LgradS));
    }
    /* create vector for combined primal space and the associated communication objects */
    if (!auglag->P) {
      PetscCall(PetscMalloc1(2, &auglag->Parr));
      auglag->Parr[0] = auglag->Px; auglag->Parr[1] = auglag->Ps;
      PetscCall(VecConcatenate(2, auglag->Parr, &auglag->P, &auglag->Pis));
      PetscCall(PetscMalloc1(2, &auglag->Pscatter));
      PetscCall(VecScatterCreate(auglag->P, auglag->Pis[0], auglag->Px, NULL, &auglag->Pscatter[0]));
      PetscCall(VecScatterCreate(auglag->P, auglag->Pis[1], auglag->Ps, NULL, &auglag->Pscatter[1]));
    }
    if (tao->eq_constrained) {
      /* create vector for combined dual space and the associated communication objects */
      if (!auglag->Y) {
        PetscCall(PetscMalloc1(2, &auglag->Yarr));
        auglag->Yarr[0] = auglag->Ye; auglag->Yarr[1] = auglag->Yi;
        PetscCall(VecConcatenate(2, auglag->Yarr, &auglag->Y, &auglag->Yis));
        PetscCall(PetscMalloc1(2, &auglag->Yscatter));
        PetscCall(VecScatterCreate(auglag->Y, auglag->Yis[0], auglag->Ye, NULL, &auglag->Yscatter[0]));
        PetscCall(VecScatterCreate(auglag->Y, auglag->Yis[1], auglag->Yi, NULL, &auglag->Yscatter[1]));
      }
      if (!auglag->C) {
        PetscCall(VecDuplicate(auglag->Y, &auglag->C));
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
      PetscCall(PetscInfo(tao,"TAOALMM with PHR: different gradient and constraint tolerances are not supported, setting catol = gatol\n"));
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
  PetscCall(TaoSetSolution(auglag->subsolver, auglag->P));
  PetscCall(TaoSetObjective(auglag->subsolver, TaoALMMSubsolverObjective_Private, (void*)auglag));
  PetscCall(TaoSetObjectiveAndGradient(auglag->subsolver, NULL, TaoALMMSubsolverObjectiveAndGradient_Private, (void*)auglag));
  if (tao->bounded) {
    /* make sure that the subsolver is a bound-constrained method */
    PetscCall(PetscObjectTypeCompare((PetscObject)auglag->subsolver, TAOCG, &is_cg));
    PetscCall(PetscObjectTypeCompare((PetscObject)auglag->subsolver, TAOLMVM, &is_lmvm));
    if (is_cg) {
      PetscCall(TaoSetType(auglag->subsolver, TAOBNCG));
      PetscCall(PetscInfo(tao,"TAOCG detected for bound-constrained problem, switching to TAOBNCG instead."));
    }
    if (is_lmvm) {
      PetscCall(TaoSetType(auglag->subsolver, TAOBQNLS));
      PetscCall(PetscInfo(tao,"TAOLMVM detected for bound-constrained problem, switching to TAOBQNLS instead."));
    }
    /* create lower and upper bound clone vectors for subsolver */
    if (!auglag->PL) {
      PetscCall(VecDuplicate(auglag->P, &auglag->PL));
    }
    if (!auglag->PU) {
      PetscCall(VecDuplicate(auglag->P, &auglag->PU));
    }
    if (tao->ineq_constrained) {
      /* create lower and upper bounds for slack, set lower to 0 */
      PetscCall(VecDuplicate(auglag->Ci, &SL));
      PetscCall(VecSet(SL, 0.0));
      PetscCall(VecDuplicate(auglag->Ci, &SU));
      PetscCall(VecSet(SU, PETSC_INFINITY));
      /* combine opt var bounds with slack bounds */
      PetscCall(TaoALMMCombinePrimal_Private(tao, tao->XL, SL, auglag->PL));
      PetscCall(TaoALMMCombinePrimal_Private(tao, tao->XU, SU, auglag->PU));
      /* destroy work vectors */
      PetscCall(VecDestroy(&SL));
      PetscCall(VecDestroy(&SU));
    } else {
      /* no inequality constraints, just copy bounds into the subsolver */
      PetscCall(VecCopy(tao->XL, auglag->PL));
      PetscCall(VecCopy(tao->XU, auglag->PU));
    }
    PetscCall(TaoSetVariableBounds(auglag->subsolver, auglag->PL, auglag->PU));
  }
  PetscCall(TaoSetUp(auglag->subsolver));
  auglag->G = auglag->subsolver->gradient;

  PetscFunctionReturn(0);
}

static PetscErrorCode TaoDestroy_ALMM(Tao tao)
{
  TAO_ALMM       *auglag = (TAO_ALMM*)tao->data;

  PetscFunctionBegin;
  PetscCall(TaoDestroy(&auglag->subsolver));
  if (tao->setupcalled) {
    PetscCall(VecDestroy(&auglag->Xwork));              /* opt work */
    if (tao->eq_constrained) {
      PetscCall(VecDestroy(&auglag->Ye));               /* equality multipliers */
      PetscCall(VecDestroy(&auglag->Cework));           /* equality work vector */
    }
    if (tao->ineq_constrained) {
      PetscCall(VecDestroy(&auglag->Ps));               /* slack vars */
      auglag->Parr[0] = NULL;                                     /* clear pointer to tao->solution, will be destroyed by TaoDestroy() shell */
      PetscCall(PetscFree(auglag->Parr));               /* array of primal vectors */
      PetscCall(VecDestroy(&auglag->LgradS));           /* slack grad */
      PetscCall(VecDestroy(&auglag->Cizero));           /* zero vector for pointwise max */
      PetscCall(VecDestroy(&auglag->Yi));               /* inequality multipliers */
      PetscCall(VecDestroy(&auglag->Ciwork));           /* inequality work vector */
      PetscCall(VecDestroy(&auglag->P));                /* combo primal */
      PetscCall(ISDestroy(&auglag->Pis[0]));            /* index set for X inside P */
      PetscCall(ISDestroy(&auglag->Pis[1]));            /* index set for S inside P */
      PetscCall(PetscFree(auglag->Pis));                /* array of P index sets */
      PetscCall(VecScatterDestroy(&auglag->Pscatter[0]));
      PetscCall(VecScatterDestroy(&auglag->Pscatter[1]));
      PetscCall(PetscFree(auglag->Pscatter));
      if (tao->eq_constrained) {
        PetscCall(VecDestroy(&auglag->Y));              /* combo multipliers */
        PetscCall(PetscFree(auglag->Yarr));             /* array of dual vectors */
        PetscCall(VecDestroy(&auglag->C));              /* combo constraints */
        PetscCall(ISDestroy(&auglag->Yis[0]));          /* index set for Ye inside Y */
        PetscCall(ISDestroy(&auglag->Yis[1]));          /* index set for Yi inside Y */
        PetscCall(PetscFree(auglag->Yis));
        PetscCall(VecScatterDestroy(&auglag->Yscatter[0]));
        PetscCall(VecScatterDestroy(&auglag->Yscatter[1]));
        PetscCall(PetscFree(auglag->Yscatter));
      }
    }
    if (tao->bounded) {
      PetscCall(VecDestroy(&auglag->PL));                /* lower bounds for subsolver */
      PetscCall(VecDestroy(&auglag->PU));                /* upper bounds for subsolver */
    }
  }
  PetscCall(PetscFree(tao->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSetFromOptions_ALMM(PetscOptionItems *PetscOptionsObject,Tao tao)
{
  TAO_ALMM       *auglag = (TAO_ALMM*)tao->data;
  PetscInt       i;

  PetscFunctionBegin;
  PetscCall(PetscOptionsHead(PetscOptionsObject,"Augmented Lagrangian multiplier method solves problems with general constraints by converting them into a sequence of unconstrained problems."));
  PetscCall(PetscOptionsReal("-tao_almm_mu_init","initial penalty parameter","",auglag->mu0,&auglag->mu0,NULL));
  PetscCall(PetscOptionsReal("-tao_almm_mu_factor","increase factor for the penalty parameter","",auglag->mu_fac,&auglag->mu_fac,NULL));
  PetscCall(PetscOptionsReal("-tao_almm_mu_power_good","exponential for penalty parameter when multiplier update is accepted","",auglag->mu_pow_good,&auglag->mu_pow_good,NULL));
  PetscCall(PetscOptionsReal("-tao_almm_mu_power_bad","exponential for penalty parameter when multiplier update is rejected","",auglag->mu_pow_bad,&auglag->mu_pow_bad,NULL));
  PetscCall(PetscOptionsReal("-tao_almm_mu_max","maximum safeguard for penalty parameter updates","",auglag->mu_max,&auglag->mu_max,NULL));
  PetscCall(PetscOptionsReal("-tao_almm_ye_min","minimum safeguard for equality multiplier updates","",auglag->ye_min,&auglag->ye_min,NULL));
  PetscCall(PetscOptionsReal("-tao_almm_ye_max","maximum safeguard for equality multipliers updates","",auglag->ye_max,&auglag->ye_max,NULL));
  PetscCall(PetscOptionsReal("-tao_almm_yi_min","minimum safeguard for inequality multipliers updates","",auglag->yi_min,&auglag->yi_min,NULL));
  PetscCall(PetscOptionsReal("-tao_almm_yi_max","maximum safeguard for inequality multipliers updates","",auglag->yi_max,&auglag->yi_max,NULL));
  PetscCall(PetscOptionsEnum("-tao_almm_type","augmented Lagrangian formulation type for the subproblem","TaoALMMType",TaoALMMTypes,(PetscEnum)auglag->type,(PetscEnum*)&auglag->type,NULL));
  PetscCall(PetscOptionsTail());
  PetscCall(TaoSetOptionsPrefix(auglag->subsolver,((PetscObject)tao)->prefix));
  PetscCall(TaoAppendOptionsPrefix(auglag->subsolver,"tao_almm_subsolver_"));
  PetscCall(TaoSetFromOptions(auglag->subsolver));
  for (i=0; i<tao->numbermonitors; i++) {
    PetscCall(PetscObjectReference((PetscObject)tao->monitorcontext[i]));
    PetscCall(TaoSetMonitor(auglag->subsolver, tao->monitor[i], tao->monitorcontext[i], tao->monitordestroy[i]));
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
  PetscCall(PetscNewLog(tao, &auglag));

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

  PetscCall(TaoCreate(PetscObjectComm((PetscObject)tao),&auglag->subsolver));
  PetscCall(TaoSetType(auglag->subsolver, TAOBQNKTR));
  PetscCall(TaoSetTolerances(auglag->subsolver, auglag->gtol, 0.0, 0.0));
  PetscCall(TaoSetMaximumIterations(auglag->subsolver, 1000));
  PetscCall(TaoSetMaximumFunctionEvaluations(auglag->subsolver, 10000));
  PetscCall(TaoSetFunctionLowerBound(auglag->subsolver, PETSC_NINFINITY));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)auglag->subsolver,(PetscObject)tao,1));

  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoALMMGetType_C", TaoALMMGetType_Private));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoALMMSetType_C", TaoALMMSetType_Private));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoALMMGetSubsolver_C", TaoALMMGetSubsolver_Private));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoALMMSetSubsolver_C", TaoALMMSetSubsolver_Private));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoALMMGetMultipliers_C", TaoALMMGetMultipliers_Private));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoALMMSetMultipliers_C", TaoALMMSetMultipliers_Private));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoALMMGetPrimalIS_C", TaoALMMGetPrimalIS_Private));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoALMMGetDualIS_C", TaoALMMGetDualIS_Private));
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoALMMCombinePrimal_Private(Tao tao, Vec X, Vec S, Vec P)
{
  TAO_ALMM     *auglag = (TAO_ALMM*)tao->data;

  PetscFunctionBegin;
  if (tao->ineq_constrained) {
    PetscCall(VecScatterBegin(auglag->Pscatter[0], X, P, INSERT_VALUES, SCATTER_REVERSE));
    PetscCall(VecScatterEnd(auglag->Pscatter[0], X, P, INSERT_VALUES, SCATTER_REVERSE));
    PetscCall(VecScatterBegin(auglag->Pscatter[1], S, P, INSERT_VALUES, SCATTER_REVERSE));
    PetscCall(VecScatterEnd(auglag->Pscatter[1], S, P, INSERT_VALUES, SCATTER_REVERSE));
  } else {
    PetscCall(VecCopy(X, P));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoALMMCombineDual_Private(Tao tao, Vec EQ, Vec IN, Vec Y)
{
  TAO_ALMM     *auglag = (TAO_ALMM*)tao->data;

  PetscFunctionBegin;
  if (tao->eq_constrained) {
    if (tao->ineq_constrained) {
      PetscCall(VecScatterBegin(auglag->Yscatter[0], EQ, Y, INSERT_VALUES, SCATTER_REVERSE));
      PetscCall(VecScatterEnd(auglag->Yscatter[0], EQ, Y, INSERT_VALUES, SCATTER_REVERSE));
      PetscCall(VecScatterBegin(auglag->Yscatter[1], IN, Y, INSERT_VALUES, SCATTER_REVERSE));
      PetscCall(VecScatterEnd(auglag->Yscatter[1], IN, Y, INSERT_VALUES, SCATTER_REVERSE));
    } else {
      PetscCall(VecCopy(EQ, Y));
    }
  } else {
    PetscCall(VecCopy(IN, Y));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoALMMSplitPrimal_Private(Tao tao, Vec P, Vec X, Vec S)
{
  TAO_ALMM     *auglag = (TAO_ALMM*)tao->data;

  PetscFunctionBegin;
  if (tao->ineq_constrained) {
    PetscCall(VecScatterBegin(auglag->Pscatter[0], P, X, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(auglag->Pscatter[0], P, X, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterBegin(auglag->Pscatter[1], P, S, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(auglag->Pscatter[1], P, S, INSERT_VALUES, SCATTER_FORWARD));
  } else {
    PetscCall(VecCopy(P, X));
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
    PetscCall(VecBoundGradientProjection(auglag->LgradX, auglag->Px, tao->XL, tao->XU, auglag->LgradX));
  }
  if (auglag->type == TAO_ALMM_PHR) {
    PetscCall(VecNorm(auglag->LgradX, NORM_INFINITY, &auglag->gnorm));
    auglag->cenorm = 0.0;
    if (tao->eq_constrained) {
      PetscCall(VecNorm(auglag->Ce, NORM_INFINITY, &auglag->cenorm));
    }
    auglag->cinorm = 0.0;
    if (tao->ineq_constrained) {
      PetscCall(VecCopy(auglag->Yi, auglag->Ciwork));
      PetscCall(VecScale(auglag->Ciwork, -1.0/auglag->mu));
      PetscCall(VecPointwiseMax(auglag->Ciwork, auglag->Ci, auglag->Ciwork));
      PetscCall(VecNorm(auglag->Ciwork, NORM_INFINITY, &auglag->cinorm));
    }
    auglag->cnorm_old = auglag->cnorm;
    auglag->cnorm = PetscMax(auglag->cenorm, auglag->cinorm);
    auglag->ytol = auglag->ytol0 * auglag->cnorm_old;
  } else {
    PetscCall(VecNorm(auglag->LgradX, NORM_2, &auglag->gnorm));
    PetscCall(VecNorm(auglag->C, NORM_2, &auglag->cnorm));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoALMMEvaluateIterate_Private(Tao tao, Vec P)
{
  TAO_ALMM       *auglag = (TAO_ALMM*)tao->data;

  PetscFunctionBegin;
  /* split solution into primal and slack components */
  PetscCall(TaoALMMSplitPrimal_Private(tao, auglag->P, auglag->Px, auglag->Ps));

  /* compute f, df/dx and the constraints */
  PetscCall(TaoComputeObjectiveAndGradient(tao, auglag->Px, &auglag->fval, auglag->LgradX));
  if (tao->eq_constrained) {
    PetscCall(TaoComputeEqualityConstraints(tao, auglag->Px, auglag->Ce));
    PetscCall(TaoComputeJacobianEquality(tao, auglag->Px, auglag->Ae, auglag->Ae));
  }
  if (tao->ineq_constrained) {
    PetscCall(TaoComputeInequalityConstraints(tao, auglag->Px, auglag->Ci));
    PetscCall(TaoComputeJacobianInequality(tao, auglag->Px, auglag->Ai, auglag->Ai));
    switch (auglag->type) {
      case TAO_ALMM_CLASSIC:
        /* classic formulation converts inequality to equality constraints via slack variables */
        PetscCall(VecAXPY(auglag->Ci, -1.0, auglag->Ps));
        break;
      case TAO_ALMM_PHR:
        /* PHR is based on Ci <= 0 while TAO defines Ci >= 0 so we hit it with a negative sign */
        PetscCall(VecScale(auglag->Ci, -1.0));
        PetscCall(MatScale(auglag->Ai, -1.0));
        break;
      default:
        break;
    }
  }
  /* combine constraints into one vector */
  PetscCall(TaoALMMCombineDual_Private(tao, auglag->Ce, auglag->Ci, auglag->C));
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
  PetscCall(TaoALMMEvaluateIterate_Private(tao, auglag->P));
  if (tao->eq_constrained) {
    /* Ce_work = mu*(Ce + Ye/mu) */
    PetscCall(VecWAXPY(auglag->Cework, 1.0/auglag->mu, auglag->Ye, auglag->Ce));
    PetscCall(VecDot(auglag->Cework, auglag->Cework, &eq_norm)); /* contribution to scalar Lagrangian */
    PetscCall(VecScale(auglag->Cework, auglag->mu));
    /* dL/dX += mu*(Ce + Ye/mu)^T Ae */
    PetscCall(MatMultTransposeAdd(auglag->Ae, auglag->Cework, auglag->LgradX, auglag->LgradX));
  }
  if (tao->ineq_constrained) {
    /* Ci_work = mu * pmax(0, Ci + Yi/mu) where pmax() is pointwise max() */
    PetscCall(VecWAXPY(auglag->Ciwork, 1.0/auglag->mu, auglag->Yi, auglag->Ci));
    PetscCall(VecPointwiseMax(auglag->Ciwork, auglag->Cizero, auglag->Ciwork));
    PetscCall(VecDot(auglag->Ciwork, auglag->Ciwork, &ineq_norm)); /* contribution to scalar Lagrangian */
    /* dL/dX += mu * pmax(0, Ci + Yi/mu)^T Ai */
    PetscCall(VecScale(auglag->Ciwork, auglag->mu));
    PetscCall(MatMultTransposeAdd(auglag->Ai, auglag->Ciwork, auglag->LgradX, auglag->LgradX));
    /* dL/dS = 0 because there are no slacks in PHR */
    PetscCall(VecZeroEntries(auglag->LgradS));
  }
  /* combine gradient together */
  PetscCall(TaoALMMCombinePrimal_Private(tao, auglag->LgradX, auglag->LgradS, auglag->G));
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
  PetscCall(TaoALMMEvaluateIterate_Private(tao, auglag->P));
  if (tao->eq_constrained) {
    /* compute scalar contributions */
    PetscCall(VecDot(auglag->Ye, auglag->Ce, &yeTce));
    PetscCall(VecDot(auglag->Ce, auglag->Ce, &ceTce));
    /* dL/dX += ye^T Ae */
    PetscCall(MatMultTransposeAdd(auglag->Ae, auglag->Ye, auglag->LgradX, auglag->LgradX));
    /* dL/dX += mu * ce^T Ae */
    PetscCall(MatMultTranspose(auglag->Ae, auglag->Ce, auglag->Xwork));
    PetscCall(VecAXPY(auglag->LgradX, auglag->mu, auglag->Xwork));
  }
  if (tao->ineq_constrained) {
    /* compute scalar contributions */
    PetscCall(VecDot(auglag->Yi, auglag->Ci, &yiTcims));
    PetscCall(VecDot(auglag->Ci, auglag->Ci, &cimsTcims));
    /* dL/dX += yi^T Ai */
    PetscCall(MatMultTransposeAdd(auglag->Ai, auglag->Yi, auglag->LgradX, auglag->LgradX));
    /* dL/dX += mu * (ci - s)^T Ai */
    PetscCall(MatMultTranspose(auglag->Ai, auglag->Ci, auglag->Xwork));
    PetscCall(VecAXPY(auglag->LgradX, auglag->mu, auglag->Xwork));
    /* dL/dS = -[yi + mu*(ci - s)] */
    PetscCall(VecWAXPY(auglag->LgradS, auglag->mu, auglag->Ci, auglag->Yi));
    PetscCall(VecScale(auglag->LgradS, -1.0));
  }
  /* combine gradient together */
  PetscCall(TaoALMMCombinePrimal_Private(tao, auglag->LgradX, auglag->LgradS, auglag->G));
  /* compute L = f + ye^T ce + yi^T (ci - s) + 0.5*mu*||ce||^2 + 0.5*mu*||ci - s||^2 */
  auglag->Lval = auglag->fval + yeTce + yiTcims + 0.5*auglag->mu*(ceTce + cimsTcims);
  PetscFunctionReturn(0);
}

PetscErrorCode TaoALMMSubsolverObjective_Private(Tao tao, Vec P, PetscReal *Lval, void *ctx)
{
  TAO_ALMM       *auglag = (TAO_ALMM*)ctx;

  PetscFunctionBegin;
  PetscCall(VecCopy(P, auglag->P));
  PetscCall((*auglag->sub_obj)(auglag->parent));
  *Lval = auglag->Lval;
  PetscFunctionReturn(0);
}

PetscErrorCode TaoALMMSubsolverObjectiveAndGradient_Private(Tao tao, Vec P, PetscReal *Lval, Vec G, void *ctx)
{
  TAO_ALMM       *auglag = (TAO_ALMM*)ctx;

  PetscFunctionBegin;
  PetscCall(VecCopy(P, auglag->P));
  PetscCall((*auglag->sub_obj)(auglag->parent));
  PetscCall(VecCopy(auglag->G, G));
  *Lval = auglag->Lval;
  PetscFunctionReturn(0);
}
