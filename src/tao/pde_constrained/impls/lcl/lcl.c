#include <../src/tao/pde_constrained/impls/lcl/lcl.h>
static PetscErrorCode LCLComputeLagrangianAndGradient(TaoLineSearch,Vec,PetscReal*,Vec,void*);
static PetscErrorCode LCLComputeAugmentedLagrangianAndGradient(TaoLineSearch,Vec,PetscReal*,Vec,void*);
static PetscErrorCode LCLScatter(TAO_LCL*,Vec,Vec,Vec);
static PetscErrorCode LCLGather(TAO_LCL*,Vec,Vec,Vec);

static PetscErrorCode TaoDestroy_LCL(Tao tao)
{
  TAO_LCL        *lclP = (TAO_LCL*)tao->data;

  PetscFunctionBegin;
  if (tao->setupcalled) {
    CHKERRQ(MatDestroy(&lclP->R));
    CHKERRQ(VecDestroy(&lclP->lamda));
    CHKERRQ(VecDestroy(&lclP->lamda0));
    CHKERRQ(VecDestroy(&lclP->WL));
    CHKERRQ(VecDestroy(&lclP->W));
    CHKERRQ(VecDestroy(&lclP->X0));
    CHKERRQ(VecDestroy(&lclP->G0));
    CHKERRQ(VecDestroy(&lclP->GL));
    CHKERRQ(VecDestroy(&lclP->GAugL));
    CHKERRQ(VecDestroy(&lclP->dbar));
    CHKERRQ(VecDestroy(&lclP->U));
    CHKERRQ(VecDestroy(&lclP->U0));
    CHKERRQ(VecDestroy(&lclP->V));
    CHKERRQ(VecDestroy(&lclP->V0));
    CHKERRQ(VecDestroy(&lclP->V1));
    CHKERRQ(VecDestroy(&lclP->GU));
    CHKERRQ(VecDestroy(&lclP->GV));
    CHKERRQ(VecDestroy(&lclP->GU0));
    CHKERRQ(VecDestroy(&lclP->GV0));
    CHKERRQ(VecDestroy(&lclP->GL_U));
    CHKERRQ(VecDestroy(&lclP->GL_V));
    CHKERRQ(VecDestroy(&lclP->GAugL_U));
    CHKERRQ(VecDestroy(&lclP->GAugL_V));
    CHKERRQ(VecDestroy(&lclP->GL_U0));
    CHKERRQ(VecDestroy(&lclP->GL_V0));
    CHKERRQ(VecDestroy(&lclP->GAugL_U0));
    CHKERRQ(VecDestroy(&lclP->GAugL_V0));
    CHKERRQ(VecDestroy(&lclP->DU));
    CHKERRQ(VecDestroy(&lclP->DV));
    CHKERRQ(VecDestroy(&lclP->WU));
    CHKERRQ(VecDestroy(&lclP->WV));
    CHKERRQ(VecDestroy(&lclP->g1));
    CHKERRQ(VecDestroy(&lclP->g2));
    CHKERRQ(VecDestroy(&lclP->con1));

    CHKERRQ(VecDestroy(&lclP->r));
    CHKERRQ(VecDestroy(&lclP->s));

    CHKERRQ(ISDestroy(&tao->state_is));
    CHKERRQ(ISDestroy(&tao->design_is));

    CHKERRQ(VecScatterDestroy(&lclP->state_scatter));
    CHKERRQ(VecScatterDestroy(&lclP->design_scatter));
  }
  CHKERRQ(MatDestroy(&lclP->R));
  CHKERRQ(PetscFree(tao->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSetFromOptions_LCL(PetscOptionItems *PetscOptionsObject,Tao tao)
{
  TAO_LCL        *lclP = (TAO_LCL*)tao->data;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"Linearly-Constrained Augmented Lagrangian Method for PDE-constrained optimization"));
  CHKERRQ(PetscOptionsReal("-tao_lcl_eps1","epsilon 1 tolerance","",lclP->eps1,&lclP->eps1,NULL));
  CHKERRQ(PetscOptionsReal("-tao_lcl_eps2","epsilon 2 tolerance","",lclP->eps2,&lclP->eps2,NULL));
  CHKERRQ(PetscOptionsReal("-tao_lcl_rho0","init value for rho","",lclP->rho0,&lclP->rho0,NULL));
  CHKERRQ(PetscOptionsReal("-tao_lcl_rhomax","max value for rho","",lclP->rhomax,&lclP->rhomax,NULL));
  lclP->phase2_niter = 1;
  CHKERRQ(PetscOptionsInt("-tao_lcl_phase2_niter","Number of phase 2 iterations in LCL algorithm","",lclP->phase2_niter,&lclP->phase2_niter,NULL));
  lclP->verbose = PETSC_FALSE;
  CHKERRQ(PetscOptionsBool("-tao_lcl_verbose","Print verbose output","",lclP->verbose,&lclP->verbose,NULL));
  lclP->tau[0] = lclP->tau[1] = lclP->tau[2] = lclP->tau[3] = 1.0e-4;
  CHKERRQ(PetscOptionsReal("-tao_lcl_tola","Tolerance for first forward solve","",lclP->tau[0],&lclP->tau[0],NULL));
  CHKERRQ(PetscOptionsReal("-tao_lcl_tolb","Tolerance for first adjoint solve","",lclP->tau[1],&lclP->tau[1],NULL));
  CHKERRQ(PetscOptionsReal("-tao_lcl_tolc","Tolerance for second forward solve","",lclP->tau[2],&lclP->tau[2],NULL));
  CHKERRQ(PetscOptionsReal("-tao_lcl_told","Tolerance for second adjoint solve","",lclP->tau[3],&lclP->tau[3],NULL));
  CHKERRQ(PetscOptionsTail());
  CHKERRQ(TaoLineSearchSetFromOptions(tao->linesearch));
  CHKERRQ(MatSetFromOptions(lclP->R));
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoView_LCL(Tao tao, PetscViewer viewer)
{
  return 0;
}

static PetscErrorCode TaoSetup_LCL(Tao tao)
{
  TAO_LCL        *lclP = (TAO_LCL*)tao->data;
  PetscInt       lo, hi, nlocalstate, nlocaldesign;
  IS             is_state, is_design;

  PetscFunctionBegin;
  PetscCheck(tao->state_is,PetscObjectComm((PetscObject)tao),PETSC_ERR_ARG_WRONGSTATE,"LCL Solver requires an initial state index set -- use TaoSetStateIS()");
  CHKERRQ(VecDuplicate(tao->solution, &tao->gradient));
  CHKERRQ(VecDuplicate(tao->solution, &tao->stepdirection));
  CHKERRQ(VecDuplicate(tao->solution, &lclP->W));
  CHKERRQ(VecDuplicate(tao->solution, &lclP->X0));
  CHKERRQ(VecDuplicate(tao->solution, &lclP->G0));
  CHKERRQ(VecDuplicate(tao->solution, &lclP->GL));
  CHKERRQ(VecDuplicate(tao->solution, &lclP->GAugL));

  CHKERRQ(VecDuplicate(tao->constraints, &lclP->lamda));
  CHKERRQ(VecDuplicate(tao->constraints, &lclP->WL));
  CHKERRQ(VecDuplicate(tao->constraints, &lclP->lamda0));
  CHKERRQ(VecDuplicate(tao->constraints, &lclP->con1));

  CHKERRQ(VecSet(lclP->lamda,0.0));

  CHKERRQ(VecGetSize(tao->solution, &lclP->n));
  CHKERRQ(VecGetSize(tao->constraints, &lclP->m));

  CHKERRQ(VecCreate(((PetscObject)tao)->comm,&lclP->U));
  CHKERRQ(VecCreate(((PetscObject)tao)->comm,&lclP->V));
  CHKERRQ(ISGetLocalSize(tao->state_is,&nlocalstate));
  CHKERRQ(ISGetLocalSize(tao->design_is,&nlocaldesign));
  CHKERRQ(VecSetSizes(lclP->U,nlocalstate,lclP->m));
  CHKERRQ(VecSetSizes(lclP->V,nlocaldesign,lclP->n-lclP->m));
  CHKERRQ(VecSetType(lclP->U,((PetscObject)(tao->solution))->type_name));
  CHKERRQ(VecSetType(lclP->V,((PetscObject)(tao->solution))->type_name));
  CHKERRQ(VecSetFromOptions(lclP->U));
  CHKERRQ(VecSetFromOptions(lclP->V));
  CHKERRQ(VecDuplicate(lclP->U,&lclP->DU));
  CHKERRQ(VecDuplicate(lclP->U,&lclP->U0));
  CHKERRQ(VecDuplicate(lclP->U,&lclP->GU));
  CHKERRQ(VecDuplicate(lclP->U,&lclP->GU0));
  CHKERRQ(VecDuplicate(lclP->U,&lclP->GAugL_U));
  CHKERRQ(VecDuplicate(lclP->U,&lclP->GL_U));
  CHKERRQ(VecDuplicate(lclP->U,&lclP->GAugL_U0));
  CHKERRQ(VecDuplicate(lclP->U,&lclP->GL_U0));
  CHKERRQ(VecDuplicate(lclP->U,&lclP->WU));
  CHKERRQ(VecDuplicate(lclP->U,&lclP->r));
  CHKERRQ(VecDuplicate(lclP->V,&lclP->V0));
  CHKERRQ(VecDuplicate(lclP->V,&lclP->V1));
  CHKERRQ(VecDuplicate(lclP->V,&lclP->DV));
  CHKERRQ(VecDuplicate(lclP->V,&lclP->s));
  CHKERRQ(VecDuplicate(lclP->V,&lclP->GV));
  CHKERRQ(VecDuplicate(lclP->V,&lclP->GV0));
  CHKERRQ(VecDuplicate(lclP->V,&lclP->dbar));
  CHKERRQ(VecDuplicate(lclP->V,&lclP->GAugL_V));
  CHKERRQ(VecDuplicate(lclP->V,&lclP->GL_V));
  CHKERRQ(VecDuplicate(lclP->V,&lclP->GAugL_V0));
  CHKERRQ(VecDuplicate(lclP->V,&lclP->GL_V0));
  CHKERRQ(VecDuplicate(lclP->V,&lclP->WV));
  CHKERRQ(VecDuplicate(lclP->V,&lclP->g1));
  CHKERRQ(VecDuplicate(lclP->V,&lclP->g2));

  /* create scatters for state, design subvecs */
  CHKERRQ(VecGetOwnershipRange(lclP->U,&lo,&hi));
  CHKERRQ(ISCreateStride(((PetscObject)lclP->U)->comm,hi-lo,lo,1,&is_state));
  CHKERRQ(VecGetOwnershipRange(lclP->V,&lo,&hi));
  if (0) {
    PetscInt sizeU,sizeV;
    CHKERRQ(VecGetSize(lclP->U,&sizeU));
    CHKERRQ(VecGetSize(lclP->V,&sizeV));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"size(U)=%D, size(V)=%D\n",sizeU,sizeV));
  }
  CHKERRQ(ISCreateStride(((PetscObject)lclP->V)->comm,hi-lo,lo,1,&is_design));
  CHKERRQ(VecScatterCreate(tao->solution,tao->state_is,lclP->U,is_state,&lclP->state_scatter));
  CHKERRQ(VecScatterCreate(tao->solution,tao->design_is,lclP->V,is_design,&lclP->design_scatter));
  CHKERRQ(ISDestroy(&is_state));
  CHKERRQ(ISDestroy(&is_design));
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSolve_LCL(Tao tao)
{
  TAO_LCL                      *lclP = (TAO_LCL*)tao->data;
  PetscInt                     phase2_iter,nlocal,its;
  TaoLineSearchConvergedReason ls_reason = TAOLINESEARCH_CONTINUE_ITERATING;
  PetscReal                    step=1.0,f, descent, aldescent;
  PetscReal                    cnorm, mnorm;
  PetscReal                    adec,r2,rGL_U,rWU;
  PetscBool                    set,pset,flag,pflag,symmetric;

  PetscFunctionBegin;
  lclP->rho = lclP->rho0;
  CHKERRQ(VecGetLocalSize(lclP->U,&nlocal));
  CHKERRQ(VecGetLocalSize(lclP->V,&nlocal));
  CHKERRQ(MatSetSizes(lclP->R, nlocal, nlocal, lclP->n-lclP->m, lclP->n-lclP->m));
  CHKERRQ(MatLMVMAllocate(lclP->R,lclP->V,lclP->V));
  lclP->recompute_jacobian_flag = PETSC_TRUE;

  /* Scatter to U,V */
  CHKERRQ(LCLScatter(lclP,tao->solution,lclP->U,lclP->V));

  /* Evaluate Function, Gradient, Constraints, and Jacobian */
  CHKERRQ(TaoComputeObjectiveAndGradient(tao,tao->solution,&f,tao->gradient));
  CHKERRQ(TaoComputeJacobianState(tao,tao->solution,tao->jacobian_state,tao->jacobian_state_pre,tao->jacobian_state_inv));
  CHKERRQ(TaoComputeJacobianDesign(tao,tao->solution,tao->jacobian_design));
  CHKERRQ(TaoComputeConstraints(tao,tao->solution, tao->constraints));

  /* Scatter gradient to GU,GV */
  CHKERRQ(LCLScatter(lclP,tao->gradient,lclP->GU,lclP->GV));

  /* Evaluate Lagrangian function and gradient */
  /* p0 */
  CHKERRQ(VecSet(lclP->lamda,0.0)); /*  Initial guess in CG */
  CHKERRQ(MatIsSymmetricKnown(tao->jacobian_state,&set,&flag));
  if (tao->jacobian_state_pre) {
    CHKERRQ(MatIsSymmetricKnown(tao->jacobian_state_pre,&pset,&pflag));
  } else {
    pset = pflag = PETSC_TRUE;
  }
  if (set && pset && flag && pflag) symmetric = PETSC_TRUE;
  else symmetric = PETSC_FALSE;

  lclP->solve_type = LCL_ADJOINT2;
  if (tao->jacobian_state_inv) {
    if (symmetric) {
      CHKERRQ(MatMult(tao->jacobian_state_inv, lclP->GU, lclP->lamda)); } else {
      CHKERRQ(MatMultTranspose(tao->jacobian_state_inv, lclP->GU, lclP->lamda));
    }
  } else {
    CHKERRQ(KSPSetOperators(tao->ksp, tao->jacobian_state, tao->jacobian_state_pre));
    if (symmetric) {
      CHKERRQ(KSPSolve(tao->ksp, lclP->GU,  lclP->lamda));
    } else {
      CHKERRQ(KSPSolveTranspose(tao->ksp, lclP->GU,  lclP->lamda));
    }
    CHKERRQ(KSPGetIterationNumber(tao->ksp,&its));
    tao->ksp_its+=its;
    tao->ksp_tot_its+=its;
  }
  CHKERRQ(VecCopy(lclP->lamda,lclP->lamda0));
  CHKERRQ(LCLComputeAugmentedLagrangianAndGradient(tao->linesearch,tao->solution,&lclP->aug,lclP->GAugL,tao));

  CHKERRQ(LCLScatter(lclP,lclP->GL,lclP->GL_U,lclP->GL_V));
  CHKERRQ(LCLScatter(lclP,lclP->GAugL,lclP->GAugL_U,lclP->GAugL_V));

  /* Evaluate constraint norm */
  CHKERRQ(VecNorm(tao->constraints, NORM_2, &cnorm));
  CHKERRQ(VecNorm(lclP->GAugL, NORM_2, &mnorm));

  /* Monitor convergence */
  tao->reason = TAO_CONTINUE_ITERATING;
  CHKERRQ(TaoLogConvergenceHistory(tao,f,mnorm,cnorm,tao->ksp_its));
  CHKERRQ(TaoMonitor(tao,tao->niter,f,mnorm,cnorm,step));
  CHKERRQ((*tao->ops->convergencetest)(tao,tao->cnvP));

  while (tao->reason == TAO_CONTINUE_ITERATING) {
    /* Call general purpose update function */
    if (tao->ops->update) {
      CHKERRQ((*tao->ops->update)(tao, tao->niter, tao->user_update));
    }
    tao->ksp_its=0;
    /* Compute a descent direction for the linearly constrained subproblem
       minimize f(u+du, v+dv)
       s.t. A(u0,v0)du + B(u0,v0)dv = -g(u0,v0) */

    /* Store the points around the linearization */
    CHKERRQ(VecCopy(lclP->U, lclP->U0));
    CHKERRQ(VecCopy(lclP->V, lclP->V0));
    CHKERRQ(VecCopy(lclP->GU,lclP->GU0));
    CHKERRQ(VecCopy(lclP->GV,lclP->GV0));
    CHKERRQ(VecCopy(lclP->GAugL_U,lclP->GAugL_U0));
    CHKERRQ(VecCopy(lclP->GAugL_V,lclP->GAugL_V0));
    CHKERRQ(VecCopy(lclP->GL_U,lclP->GL_U0));
    CHKERRQ(VecCopy(lclP->GL_V,lclP->GL_V0));

    lclP->aug0 = lclP->aug;
    lclP->lgn0 = lclP->lgn;

    /* Given the design variables, we need to project the current iterate
       onto the linearized constraint.  We choose to fix the design variables
       and solve the linear system for the state variables.  The resulting
       point is the Newton direction */

    /* Solve r = A\con */
    lclP->solve_type = LCL_FORWARD1;
    CHKERRQ(VecSet(lclP->r,0.0)); /*  Initial guess in CG */

    if (tao->jacobian_state_inv) {
      CHKERRQ(MatMult(tao->jacobian_state_inv, tao->constraints, lclP->r));
    } else {
      CHKERRQ(KSPSetOperators(tao->ksp, tao->jacobian_state, tao->jacobian_state_pre));
      CHKERRQ(KSPSolve(tao->ksp, tao->constraints,  lclP->r));
      CHKERRQ(KSPGetIterationNumber(tao->ksp,&its));
      tao->ksp_its+=its;
      tao->ksp_tot_its+=tao->ksp_its;
    }

    /* Set design step direction dv to zero */
    CHKERRQ(VecSet(lclP->s, 0.0));

    /*
       Check sufficient descent for constraint merit function .5*||con||^2
       con' Ak r >= eps1 ||r||^(2+eps2)
    */

    /* Compute WU= Ak' * con */
    if (symmetric)  {
      CHKERRQ(MatMult(tao->jacobian_state,tao->constraints,lclP->WU));
    } else {
      CHKERRQ(MatMultTranspose(tao->jacobian_state,tao->constraints,lclP->WU));
    }
    /* Compute r * Ak' * con */
    CHKERRQ(VecDot(lclP->r,lclP->WU,&rWU));

    /* compute ||r||^(2+eps2) */
    CHKERRQ(VecNorm(lclP->r,NORM_2,&r2));
    r2 = PetscPowScalar(r2,2.0+lclP->eps2);
    adec = lclP->eps1 * r2;

    if (rWU < adec) {
      CHKERRQ(PetscInfo(tao,"Newton direction not descent for constraint, feasibility phase required\n"));
      if (lclP->verbose) {
        CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Newton direction not descent for constraint: %g -- using steepest descent\n",(double)descent));
      }

      CHKERRQ(PetscInfo(tao,"Using steepest descent direction instead.\n"));
      CHKERRQ(VecSet(lclP->r,0.0));
      CHKERRQ(VecAXPY(lclP->r,-1.0,lclP->WU));
      CHKERRQ(VecDot(lclP->r,lclP->r,&rWU));
      CHKERRQ(VecNorm(lclP->r,NORM_2,&r2));
      r2 = PetscPowScalar(r2,2.0+lclP->eps2);
      CHKERRQ(VecDot(lclP->r,lclP->GAugL_U,&descent));
      adec = lclP->eps1 * r2;
    }

    /*
       Check descent for aug. lagrangian
       r' (GUk - Ak'*yk - rho*Ak'*con) <= -eps1 ||r||^(2+eps2)
          GL_U = GUk - Ak'*yk
          WU   = Ak'*con
                                         adec=eps1||r||^(2+eps2)

       ==>
       Check r'GL_U - rho*r'WU <= adec
    */

    CHKERRQ(VecDot(lclP->r,lclP->GL_U,&rGL_U));
    aldescent =  rGL_U - lclP->rho*rWU;
    if (aldescent > -adec) {
      if (lclP->verbose) {
        CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Newton direction not descent for augmented Lagrangian: %g",(double)aldescent));
      }
      CHKERRQ(PetscInfo(tao,"Newton direction not descent for augmented Lagrangian: %g\n",(double)aldescent));
      lclP->rho =  (rGL_U - adec)/rWU;
      if (lclP->rho > lclP->rhomax) {
        lclP->rho = lclP->rhomax;
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"rho=%g > rhomax, case not implemented.  Increase rhomax (-tao_lcl_rhomax)",(double)lclP->rho);
      }
      if (lclP->verbose) {
        CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  Increasing penalty parameter to %g\n",(double)lclP->rho));
      }
      CHKERRQ(PetscInfo(tao,"  Increasing penalty parameter to %g\n",(double)lclP->rho));
    }

    CHKERRQ(LCLComputeAugmentedLagrangianAndGradient(tao->linesearch,tao->solution,&lclP->aug,lclP->GAugL,tao));
    CHKERRQ(LCLScatter(lclP,lclP->GL,lclP->GL_U,lclP->GL_V));
    CHKERRQ(LCLScatter(lclP,lclP->GAugL,lclP->GAugL_U,lclP->GAugL_V));

    /* We now minimize the augmented Lagrangian along the Newton direction */
    CHKERRQ(VecScale(lclP->r,-1.0));
    CHKERRQ(LCLGather(lclP, lclP->r,lclP->s,tao->stepdirection));
    CHKERRQ(VecScale(lclP->r,-1.0));
    CHKERRQ(LCLGather(lclP, lclP->GAugL_U0, lclP->GAugL_V0, lclP->GAugL));
    CHKERRQ(LCLGather(lclP, lclP->U0,lclP->V0,lclP->X0));

    lclP->recompute_jacobian_flag = PETSC_TRUE;

    CHKERRQ(TaoLineSearchSetInitialStepLength(tao->linesearch,1.0));
    CHKERRQ(TaoLineSearchSetType(tao->linesearch, TAOLINESEARCHMT));
    CHKERRQ(TaoLineSearchSetFromOptions(tao->linesearch));
    CHKERRQ(TaoLineSearchApply(tao->linesearch, tao->solution, &lclP->aug, lclP->GAugL, tao->stepdirection, &step, &ls_reason));
    if (lclP->verbose) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Steplength = %g\n",(double)step));
    }

    CHKERRQ(LCLScatter(lclP,tao->solution,lclP->U,lclP->V));
    CHKERRQ(TaoComputeObjectiveAndGradient(tao,tao->solution,&f,tao->gradient));
    CHKERRQ(LCLScatter(lclP,tao->gradient,lclP->GU,lclP->GV));

    CHKERRQ(LCLScatter(lclP,lclP->GAugL,lclP->GAugL_U,lclP->GAugL_V));

    /* Check convergence */
    CHKERRQ(VecNorm(lclP->GAugL, NORM_2, &mnorm));
    CHKERRQ(VecNorm(tao->constraints, NORM_2, &cnorm));
    CHKERRQ(TaoLogConvergenceHistory(tao,f,mnorm,cnorm,tao->ksp_its));
    CHKERRQ(TaoMonitor(tao,tao->niter,f,mnorm,cnorm,step));
    CHKERRQ((*tao->ops->convergencetest)(tao,tao->cnvP));
    if (tao->reason != TAO_CONTINUE_ITERATING) {
      break;
    }

    /* TODO: use a heuristic to choose how many iterations should be performed within phase 2 */
    for (phase2_iter=0; phase2_iter<lclP->phase2_niter; phase2_iter++) {
      /* We now minimize the objective function starting from the fraction of
         the Newton point accepted by applying one step of a reduced-space
         method.  The optimization problem is:

         minimize f(u+du, v+dv)
         s. t.    A(u0,v0)du + B(u0,v0)du = -alpha g(u0,v0)

         In particular, we have that
         du = -inv(A)*(Bdv + alpha g) */

      CHKERRQ(TaoComputeJacobianState(tao,lclP->X0,tao->jacobian_state,tao->jacobian_state_pre,tao->jacobian_state_inv));
      CHKERRQ(TaoComputeJacobianDesign(tao,lclP->X0,tao->jacobian_design));

      /* Store V and constraints */
      CHKERRQ(VecCopy(lclP->V, lclP->V1));
      CHKERRQ(VecCopy(tao->constraints,lclP->con1));

      /* Compute multipliers */
      /* p1 */
      CHKERRQ(VecSet(lclP->lamda,0.0)); /*  Initial guess in CG */
      lclP->solve_type = LCL_ADJOINT1;
      CHKERRQ(MatIsSymmetricKnown(tao->jacobian_state,&set,&flag));
      if (tao->jacobian_state_pre) {
        CHKERRQ(MatIsSymmetricKnown(tao->jacobian_state_pre,&pset,&pflag));
      } else {
        pset = pflag = PETSC_TRUE;
      }
      if (set && pset && flag && pflag) symmetric = PETSC_TRUE;
      else symmetric = PETSC_FALSE;

      if (tao->jacobian_state_inv) {
        if (symmetric) {
          CHKERRQ(MatMult(tao->jacobian_state_inv, lclP->GAugL_U, lclP->lamda));
        } else {
          CHKERRQ(MatMultTranspose(tao->jacobian_state_inv, lclP->GAugL_U, lclP->lamda));
        }
      } else {
        if (symmetric) {
          CHKERRQ(KSPSolve(tao->ksp, lclP->GAugL_U,  lclP->lamda));
        } else {
          CHKERRQ(KSPSolveTranspose(tao->ksp, lclP->GAugL_U,  lclP->lamda));
        }
        CHKERRQ(KSPGetIterationNumber(tao->ksp,&its));
        tao->ksp_its+=its;
        tao->ksp_tot_its+=its;
      }
      CHKERRQ(MatMultTranspose(tao->jacobian_design,lclP->lamda,lclP->g1));
      CHKERRQ(VecAXPY(lclP->g1,-1.0,lclP->GAugL_V));

      /* Compute the limited-memory quasi-newton direction */
      if (tao->niter > 0) {
        CHKERRQ(MatSolve(lclP->R,lclP->g1,lclP->s));
        CHKERRQ(VecDot(lclP->s,lclP->g1,&descent));
        if (descent <= 0) {
          if (lclP->verbose) {
            CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Reduced-space direction not descent: %g\n",(double)descent));
          }
          CHKERRQ(VecCopy(lclP->g1,lclP->s));
        }
      } else {
        CHKERRQ(VecCopy(lclP->g1,lclP->s));
      }
      CHKERRQ(VecScale(lclP->g1,-1.0));

      /* Recover the full space direction */
      CHKERRQ(MatMult(tao->jacobian_design,lclP->s,lclP->WU));
      /* CHKERRQ(VecSet(lclP->r,0.0)); */ /*  Initial guess in CG */
      lclP->solve_type = LCL_FORWARD2;
      if (tao->jacobian_state_inv) {
        CHKERRQ(MatMult(tao->jacobian_state_inv,lclP->WU,lclP->r));
      } else {
        CHKERRQ(KSPSolve(tao->ksp, lclP->WU, lclP->r));
        CHKERRQ(KSPGetIterationNumber(tao->ksp,&its));
        tao->ksp_its += its;
        tao->ksp_tot_its+=its;
      }

      /* We now minimize the augmented Lagrangian along the direction -r,s */
      CHKERRQ(VecScale(lclP->r, -1.0));
      CHKERRQ(LCLGather(lclP,lclP->r,lclP->s,tao->stepdirection));
      CHKERRQ(VecScale(lclP->r, -1.0));
      lclP->recompute_jacobian_flag = PETSC_TRUE;

      CHKERRQ(TaoLineSearchSetInitialStepLength(tao->linesearch,1.0));
      CHKERRQ(TaoLineSearchSetType(tao->linesearch,TAOLINESEARCHMT));
      CHKERRQ(TaoLineSearchSetFromOptions(tao->linesearch));
      CHKERRQ(TaoLineSearchApply(tao->linesearch, tao->solution, &lclP->aug, lclP->GAugL, tao->stepdirection,&step,&ls_reason));
      if (lclP->verbose) {
        CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Reduced-space steplength =  %g\n",(double)step));
      }

      CHKERRQ(LCLScatter(lclP,tao->solution,lclP->U,lclP->V));
      CHKERRQ(LCLScatter(lclP,lclP->GL,lclP->GL_U,lclP->GL_V));
      CHKERRQ(LCLScatter(lclP,lclP->GAugL,lclP->GAugL_U,lclP->GAugL_V));
      CHKERRQ(TaoComputeObjectiveAndGradient(tao,tao->solution,&f,tao->gradient));
      CHKERRQ(LCLScatter(lclP,tao->gradient,lclP->GU,lclP->GV));

      /* Compute the reduced gradient at the new point */

      CHKERRQ(TaoComputeJacobianState(tao,lclP->X0,tao->jacobian_state,tao->jacobian_state_pre,tao->jacobian_state_inv));
      CHKERRQ(TaoComputeJacobianDesign(tao,lclP->X0,tao->jacobian_design));

      /* p2 */
      /* Compute multipliers, using lamda-rho*con as an initial guess in PCG */
      if (phase2_iter==0) {
        CHKERRQ(VecWAXPY(lclP->lamda,-lclP->rho,lclP->con1,lclP->lamda0));
      } else {
        CHKERRQ(VecAXPY(lclP->lamda,-lclP->rho,tao->constraints));
      }

      CHKERRQ(MatIsSymmetricKnown(tao->jacobian_state,&set,&flag));
      if (tao->jacobian_state_pre) {
        CHKERRQ(MatIsSymmetricKnown(tao->jacobian_state_pre,&pset,&pflag));
      } else {
        pset = pflag = PETSC_TRUE;
      }
      if (set && pset && flag && pflag) symmetric = PETSC_TRUE;
      else symmetric = PETSC_FALSE;

      lclP->solve_type = LCL_ADJOINT2;
      if (tao->jacobian_state_inv) {
        if (symmetric) {
          CHKERRQ(MatMult(tao->jacobian_state_inv, lclP->GU, lclP->lamda));
        } else {
          CHKERRQ(MatMultTranspose(tao->jacobian_state_inv, lclP->GU, lclP->lamda));
        }
      } else {
        if (symmetric) {
          CHKERRQ(KSPSolve(tao->ksp, lclP->GU,  lclP->lamda));
        } else {
          CHKERRQ(KSPSolveTranspose(tao->ksp, lclP->GU,  lclP->lamda));
        }
        CHKERRQ(KSPGetIterationNumber(tao->ksp,&its));
        tao->ksp_its += its;
        tao->ksp_tot_its += its;
      }

      CHKERRQ(MatMultTranspose(tao->jacobian_design,lclP->lamda,lclP->g2));
      CHKERRQ(VecAXPY(lclP->g2,-1.0,lclP->GV));

      CHKERRQ(VecScale(lclP->g2,-1.0));

      /* Update the quasi-newton approximation */
      CHKERRQ(MatLMVMUpdate(lclP->R,lclP->V,lclP->g2));
      /* Use "-tao_ls_type gpcg -tao_ls_ftol 0 -tao_lmm_broyden_phi 0.0 -tao_lmm_scale_type scalar" to obtain agreement with Matlab code */

    }

    CHKERRQ(VecCopy(lclP->lamda,lclP->lamda0));

    /* Evaluate Function, Gradient, Constraints, and Jacobian */
    CHKERRQ(TaoComputeObjectiveAndGradient(tao,tao->solution,&f,tao->gradient));
    CHKERRQ(LCLScatter(lclP,tao->solution,lclP->U,lclP->V));
    CHKERRQ(LCLScatter(lclP,tao->gradient,lclP->GU,lclP->GV));

    CHKERRQ(TaoComputeJacobianState(tao,tao->solution,tao->jacobian_state,tao->jacobian_state_pre,tao->jacobian_state_inv));
    CHKERRQ(TaoComputeJacobianDesign(tao,tao->solution,tao->jacobian_design));
    CHKERRQ(TaoComputeConstraints(tao,tao->solution, tao->constraints));

    CHKERRQ(LCLComputeAugmentedLagrangianAndGradient(tao->linesearch,tao->solution,&lclP->aug,lclP->GAugL,tao));

    CHKERRQ(VecNorm(lclP->GAugL, NORM_2, &mnorm));

    /* Evaluate constraint norm */
    CHKERRQ(VecNorm(tao->constraints, NORM_2, &cnorm));

    /* Monitor convergence */
    tao->niter++;
    CHKERRQ(TaoLogConvergenceHistory(tao,f,mnorm,cnorm,tao->ksp_its));
    CHKERRQ(TaoMonitor(tao,tao->niter,f,mnorm,cnorm,step));
    CHKERRQ((*tao->ops->convergencetest)(tao,tao->cnvP));
  }
  PetscFunctionReturn(0);
}

/*MC
 TAOLCL - linearly constrained lagrangian method for pde-constrained optimization

+ -tao_lcl_eps1 - epsilon 1 tolerance
. -tao_lcl_eps2","epsilon 2 tolerance","",lclP->eps2,&lclP->eps2,NULL);CHKERRQ(ierr);
. -tao_lcl_rho0","init value for rho","",lclP->rho0,&lclP->rho0,NULL);CHKERRQ(ierr);
. -tao_lcl_rhomax","max value for rho","",lclP->rhomax,&lclP->rhomax,NULL);CHKERRQ(ierr);
. -tao_lcl_phase2_niter - Number of phase 2 iterations in LCL algorithm
. -tao_lcl_verbose - Print verbose output if True
. -tao_lcl_tola - Tolerance for first forward solve
. -tao_lcl_tolb - Tolerance for first adjoint solve
. -tao_lcl_tolc - Tolerance for second forward solve
- -tao_lcl_told - Tolerance for second adjoint solve

  Level: beginner
M*/
PETSC_EXTERN PetscErrorCode TaoCreate_LCL(Tao tao)
{
  TAO_LCL        *lclP;
  const char     *morethuente_type = TAOLINESEARCHMT;

  PetscFunctionBegin;
  tao->ops->setup = TaoSetup_LCL;
  tao->ops->solve = TaoSolve_LCL;
  tao->ops->view = TaoView_LCL;
  tao->ops->setfromoptions = TaoSetFromOptions_LCL;
  tao->ops->destroy = TaoDestroy_LCL;
  CHKERRQ(PetscNewLog(tao,&lclP));
  tao->data = (void*)lclP;

  /* Override default settings (unless already changed) */
  if (!tao->max_it_changed) tao->max_it = 200;
  if (!tao->catol_changed) tao->catol = 1.0e-4;
  if (!tao->gatol_changed) tao->gttol = 1.0e-4;
  if (!tao->grtol_changed) tao->gttol = 1.0e-4;
  if (!tao->gttol_changed) tao->gttol = 1.0e-4;
  lclP->rho0 = 1.0e-4;
  lclP->rhomax=1e5;
  lclP->eps1 = 1.0e-8;
  lclP->eps2 = 0.0;
  lclP->solve_type=2;
  lclP->tau[0] = lclP->tau[1] = lclP->tau[2] = lclP->tau[3] = 1.0e-4;
  CHKERRQ(TaoLineSearchCreate(((PetscObject)tao)->comm, &tao->linesearch));
  CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)tao->linesearch, (PetscObject)tao, 1));
  CHKERRQ(TaoLineSearchSetType(tao->linesearch, morethuente_type));
  CHKERRQ(TaoLineSearchSetOptionsPrefix(tao->linesearch,tao->hdr.prefix));

  CHKERRQ(TaoLineSearchSetObjectiveAndGradientRoutine(tao->linesearch,LCLComputeAugmentedLagrangianAndGradient, tao));
  CHKERRQ(KSPCreate(((PetscObject)tao)->comm,&tao->ksp));
  CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)tao->ksp, (PetscObject)tao, 1));
  CHKERRQ(KSPSetOptionsPrefix(tao->ksp, tao->hdr.prefix));
  CHKERRQ(KSPSetFromOptions(tao->ksp));

  CHKERRQ(MatCreate(((PetscObject)tao)->comm, &lclP->R));
  CHKERRQ(MatSetType(lclP->R, MATLMVMBFGS));
  PetscFunctionReturn(0);
}

static PetscErrorCode LCLComputeLagrangianAndGradient(TaoLineSearch ls, Vec X, PetscReal *f, Vec G, void *ptr)
{
  Tao            tao = (Tao)ptr;
  TAO_LCL        *lclP = (TAO_LCL*)tao->data;
  PetscBool      set,pset,flag,pflag,symmetric;
  PetscReal      cdotl;

  PetscFunctionBegin;
  CHKERRQ(TaoComputeObjectiveAndGradient(tao,X,f,G));
  CHKERRQ(LCLScatter(lclP,G,lclP->GU,lclP->GV));
  if (lclP->recompute_jacobian_flag) {
    CHKERRQ(TaoComputeJacobianState(tao,X,tao->jacobian_state,tao->jacobian_state_pre,tao->jacobian_state_inv));
    CHKERRQ(TaoComputeJacobianDesign(tao,X,tao->jacobian_design));
  }
  CHKERRQ(TaoComputeConstraints(tao,X, tao->constraints));
  CHKERRQ(MatIsSymmetricKnown(tao->jacobian_state,&set,&flag));
  if (tao->jacobian_state_pre) {
    CHKERRQ(MatIsSymmetricKnown(tao->jacobian_state_pre,&pset,&pflag));
  } else {
    pset = pflag = PETSC_TRUE;
  }
  if (set && pset && flag && pflag) symmetric = PETSC_TRUE;
  else symmetric = PETSC_FALSE;

  CHKERRQ(VecDot(lclP->lamda0, tao->constraints, &cdotl));
  lclP->lgn = *f - cdotl;

  /* Gradient of Lagrangian GL = G - J' * lamda */
  /*      WU = A' * WL
          WV = B' * WL */
  if (symmetric) {
    CHKERRQ(MatMult(tao->jacobian_state,lclP->lamda0,lclP->GL_U));
  } else {
    CHKERRQ(MatMultTranspose(tao->jacobian_state,lclP->lamda0,lclP->GL_U));
  }
  CHKERRQ(MatMultTranspose(tao->jacobian_design,lclP->lamda0,lclP->GL_V));
  CHKERRQ(VecScale(lclP->GL_U,-1.0));
  CHKERRQ(VecScale(lclP->GL_V,-1.0));
  CHKERRQ(VecAXPY(lclP->GL_U,1.0,lclP->GU));
  CHKERRQ(VecAXPY(lclP->GL_V,1.0,lclP->GV));
  CHKERRQ(LCLGather(lclP,lclP->GL_U,lclP->GL_V,lclP->GL));

  f[0] = lclP->lgn;
  CHKERRQ(VecCopy(lclP->GL,G));
  PetscFunctionReturn(0);
}

static PetscErrorCode LCLComputeAugmentedLagrangianAndGradient(TaoLineSearch ls, Vec X, PetscReal *f, Vec G, void *ptr)
{
  Tao            tao = (Tao)ptr;
  TAO_LCL        *lclP = (TAO_LCL*)tao->data;
  PetscReal      con2;
  PetscBool      flag,pflag,set,pset,symmetric;

  PetscFunctionBegin;
  CHKERRQ(LCLComputeLagrangianAndGradient(tao->linesearch,X,f,G,tao));
  CHKERRQ(LCLScatter(lclP,G,lclP->GL_U,lclP->GL_V));
  CHKERRQ(VecDot(tao->constraints,tao->constraints,&con2));
  lclP->aug = lclP->lgn + 0.5*lclP->rho*con2;

  /* Gradient of Aug. Lagrangian GAugL = GL + rho * J' c */
  /*      WU = A' * c
          WV = B' * c */
  CHKERRQ(MatIsSymmetricKnown(tao->jacobian_state,&set,&flag));
  if (tao->jacobian_state_pre) {
    CHKERRQ(MatIsSymmetricKnown(tao->jacobian_state_pre,&pset,&pflag));
  } else {
    pset = pflag = PETSC_TRUE;
  }
  if (set && pset && flag && pflag) symmetric = PETSC_TRUE;
  else symmetric = PETSC_FALSE;

  if (symmetric) {
    CHKERRQ(MatMult(tao->jacobian_state,tao->constraints,lclP->GAugL_U));
  } else {
    CHKERRQ(MatMultTranspose(tao->jacobian_state,tao->constraints,lclP->GAugL_U));
  }

  CHKERRQ(MatMultTranspose(tao->jacobian_design,tao->constraints,lclP->GAugL_V));
  CHKERRQ(VecAYPX(lclP->GAugL_U,lclP->rho,lclP->GL_U));
  CHKERRQ(VecAYPX(lclP->GAugL_V,lclP->rho,lclP->GL_V));
  CHKERRQ(LCLGather(lclP,lclP->GAugL_U,lclP->GAugL_V,lclP->GAugL));

  f[0] = lclP->aug;
  CHKERRQ(VecCopy(lclP->GAugL,G));
  PetscFunctionReturn(0);
}

PetscErrorCode LCLGather(TAO_LCL *lclP, Vec u, Vec v, Vec x)
{
  PetscFunctionBegin;
  CHKERRQ(VecScatterBegin(lclP->state_scatter, u, x, INSERT_VALUES, SCATTER_REVERSE));
  CHKERRQ(VecScatterEnd(lclP->state_scatter, u, x, INSERT_VALUES, SCATTER_REVERSE));
  CHKERRQ(VecScatterBegin(lclP->design_scatter, v, x, INSERT_VALUES, SCATTER_REVERSE));
  CHKERRQ(VecScatterEnd(lclP->design_scatter, v, x, INSERT_VALUES, SCATTER_REVERSE));
  PetscFunctionReturn(0);

}
PetscErrorCode LCLScatter(TAO_LCL *lclP, Vec x, Vec u, Vec v)
{
  PetscFunctionBegin;
  CHKERRQ(VecScatterBegin(lclP->state_scatter, x, u, INSERT_VALUES, SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(lclP->state_scatter, x, u, INSERT_VALUES, SCATTER_FORWARD));
  CHKERRQ(VecScatterBegin(lclP->design_scatter, x, v, INSERT_VALUES, SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(lclP->design_scatter, x, v, INSERT_VALUES, SCATTER_FORWARD));
  PetscFunctionReturn(0);

}
