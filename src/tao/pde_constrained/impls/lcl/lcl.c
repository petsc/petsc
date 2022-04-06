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
    PetscCall(MatDestroy(&lclP->R));
    PetscCall(VecDestroy(&lclP->lamda));
    PetscCall(VecDestroy(&lclP->lamda0));
    PetscCall(VecDestroy(&lclP->WL));
    PetscCall(VecDestroy(&lclP->W));
    PetscCall(VecDestroy(&lclP->X0));
    PetscCall(VecDestroy(&lclP->G0));
    PetscCall(VecDestroy(&lclP->GL));
    PetscCall(VecDestroy(&lclP->GAugL));
    PetscCall(VecDestroy(&lclP->dbar));
    PetscCall(VecDestroy(&lclP->U));
    PetscCall(VecDestroy(&lclP->U0));
    PetscCall(VecDestroy(&lclP->V));
    PetscCall(VecDestroy(&lclP->V0));
    PetscCall(VecDestroy(&lclP->V1));
    PetscCall(VecDestroy(&lclP->GU));
    PetscCall(VecDestroy(&lclP->GV));
    PetscCall(VecDestroy(&lclP->GU0));
    PetscCall(VecDestroy(&lclP->GV0));
    PetscCall(VecDestroy(&lclP->GL_U));
    PetscCall(VecDestroy(&lclP->GL_V));
    PetscCall(VecDestroy(&lclP->GAugL_U));
    PetscCall(VecDestroy(&lclP->GAugL_V));
    PetscCall(VecDestroy(&lclP->GL_U0));
    PetscCall(VecDestroy(&lclP->GL_V0));
    PetscCall(VecDestroy(&lclP->GAugL_U0));
    PetscCall(VecDestroy(&lclP->GAugL_V0));
    PetscCall(VecDestroy(&lclP->DU));
    PetscCall(VecDestroy(&lclP->DV));
    PetscCall(VecDestroy(&lclP->WU));
    PetscCall(VecDestroy(&lclP->WV));
    PetscCall(VecDestroy(&lclP->g1));
    PetscCall(VecDestroy(&lclP->g2));
    PetscCall(VecDestroy(&lclP->con1));

    PetscCall(VecDestroy(&lclP->r));
    PetscCall(VecDestroy(&lclP->s));

    PetscCall(ISDestroy(&tao->state_is));
    PetscCall(ISDestroy(&tao->design_is));

    PetscCall(VecScatterDestroy(&lclP->state_scatter));
    PetscCall(VecScatterDestroy(&lclP->design_scatter));
  }
  PetscCall(MatDestroy(&lclP->R));
  PetscCall(PetscFree(tao->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSetFromOptions_LCL(PetscOptionItems *PetscOptionsObject,Tao tao)
{
  TAO_LCL        *lclP = (TAO_LCL*)tao->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject,"Linearly-Constrained Augmented Lagrangian Method for PDE-constrained optimization");
  PetscCall(PetscOptionsReal("-tao_lcl_eps1","epsilon 1 tolerance","",lclP->eps1,&lclP->eps1,NULL));
  PetscCall(PetscOptionsReal("-tao_lcl_eps2","epsilon 2 tolerance","",lclP->eps2,&lclP->eps2,NULL));
  PetscCall(PetscOptionsReal("-tao_lcl_rho0","init value for rho","",lclP->rho0,&lclP->rho0,NULL));
  PetscCall(PetscOptionsReal("-tao_lcl_rhomax","max value for rho","",lclP->rhomax,&lclP->rhomax,NULL));
  lclP->phase2_niter = 1;
  PetscCall(PetscOptionsInt("-tao_lcl_phase2_niter","Number of phase 2 iterations in LCL algorithm","",lclP->phase2_niter,&lclP->phase2_niter,NULL));
  lclP->verbose = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-tao_lcl_verbose","Print verbose output","",lclP->verbose,&lclP->verbose,NULL));
  lclP->tau[0] = lclP->tau[1] = lclP->tau[2] = lclP->tau[3] = 1.0e-4;
  PetscCall(PetscOptionsReal("-tao_lcl_tola","Tolerance for first forward solve","",lclP->tau[0],&lclP->tau[0],NULL));
  PetscCall(PetscOptionsReal("-tao_lcl_tolb","Tolerance for first adjoint solve","",lclP->tau[1],&lclP->tau[1],NULL));
  PetscCall(PetscOptionsReal("-tao_lcl_tolc","Tolerance for second forward solve","",lclP->tau[2],&lclP->tau[2],NULL));
  PetscCall(PetscOptionsReal("-tao_lcl_told","Tolerance for second adjoint solve","",lclP->tau[3],&lclP->tau[3],NULL));
  PetscOptionsHeadEnd();
  PetscCall(TaoLineSearchSetFromOptions(tao->linesearch));
  PetscCall(MatSetFromOptions(lclP->R));
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
  PetscCall(VecDuplicate(tao->solution, &tao->gradient));
  PetscCall(VecDuplicate(tao->solution, &tao->stepdirection));
  PetscCall(VecDuplicate(tao->solution, &lclP->W));
  PetscCall(VecDuplicate(tao->solution, &lclP->X0));
  PetscCall(VecDuplicate(tao->solution, &lclP->G0));
  PetscCall(VecDuplicate(tao->solution, &lclP->GL));
  PetscCall(VecDuplicate(tao->solution, &lclP->GAugL));

  PetscCall(VecDuplicate(tao->constraints, &lclP->lamda));
  PetscCall(VecDuplicate(tao->constraints, &lclP->WL));
  PetscCall(VecDuplicate(tao->constraints, &lclP->lamda0));
  PetscCall(VecDuplicate(tao->constraints, &lclP->con1));

  PetscCall(VecSet(lclP->lamda,0.0));

  PetscCall(VecGetSize(tao->solution, &lclP->n));
  PetscCall(VecGetSize(tao->constraints, &lclP->m));

  PetscCall(VecCreate(((PetscObject)tao)->comm,&lclP->U));
  PetscCall(VecCreate(((PetscObject)tao)->comm,&lclP->V));
  PetscCall(ISGetLocalSize(tao->state_is,&nlocalstate));
  PetscCall(ISGetLocalSize(tao->design_is,&nlocaldesign));
  PetscCall(VecSetSizes(lclP->U,nlocalstate,lclP->m));
  PetscCall(VecSetSizes(lclP->V,nlocaldesign,lclP->n-lclP->m));
  PetscCall(VecSetType(lclP->U,((PetscObject)(tao->solution))->type_name));
  PetscCall(VecSetType(lclP->V,((PetscObject)(tao->solution))->type_name));
  PetscCall(VecSetFromOptions(lclP->U));
  PetscCall(VecSetFromOptions(lclP->V));
  PetscCall(VecDuplicate(lclP->U,&lclP->DU));
  PetscCall(VecDuplicate(lclP->U,&lclP->U0));
  PetscCall(VecDuplicate(lclP->U,&lclP->GU));
  PetscCall(VecDuplicate(lclP->U,&lclP->GU0));
  PetscCall(VecDuplicate(lclP->U,&lclP->GAugL_U));
  PetscCall(VecDuplicate(lclP->U,&lclP->GL_U));
  PetscCall(VecDuplicate(lclP->U,&lclP->GAugL_U0));
  PetscCall(VecDuplicate(lclP->U,&lclP->GL_U0));
  PetscCall(VecDuplicate(lclP->U,&lclP->WU));
  PetscCall(VecDuplicate(lclP->U,&lclP->r));
  PetscCall(VecDuplicate(lclP->V,&lclP->V0));
  PetscCall(VecDuplicate(lclP->V,&lclP->V1));
  PetscCall(VecDuplicate(lclP->V,&lclP->DV));
  PetscCall(VecDuplicate(lclP->V,&lclP->s));
  PetscCall(VecDuplicate(lclP->V,&lclP->GV));
  PetscCall(VecDuplicate(lclP->V,&lclP->GV0));
  PetscCall(VecDuplicate(lclP->V,&lclP->dbar));
  PetscCall(VecDuplicate(lclP->V,&lclP->GAugL_V));
  PetscCall(VecDuplicate(lclP->V,&lclP->GL_V));
  PetscCall(VecDuplicate(lclP->V,&lclP->GAugL_V0));
  PetscCall(VecDuplicate(lclP->V,&lclP->GL_V0));
  PetscCall(VecDuplicate(lclP->V,&lclP->WV));
  PetscCall(VecDuplicate(lclP->V,&lclP->g1));
  PetscCall(VecDuplicate(lclP->V,&lclP->g2));

  /* create scatters for state, design subvecs */
  PetscCall(VecGetOwnershipRange(lclP->U,&lo,&hi));
  PetscCall(ISCreateStride(((PetscObject)lclP->U)->comm,hi-lo,lo,1,&is_state));
  PetscCall(VecGetOwnershipRange(lclP->V,&lo,&hi));
  if (0) {
    PetscInt sizeU,sizeV;
    PetscCall(VecGetSize(lclP->U,&sizeU));
    PetscCall(VecGetSize(lclP->V,&sizeV));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"size(U)=%" PetscInt_FMT ", size(V)=%" PetscInt_FMT "\n",sizeU,sizeV));
  }
  PetscCall(ISCreateStride(((PetscObject)lclP->V)->comm,hi-lo,lo,1,&is_design));
  PetscCall(VecScatterCreate(tao->solution,tao->state_is,lclP->U,is_state,&lclP->state_scatter));
  PetscCall(VecScatterCreate(tao->solution,tao->design_is,lclP->V,is_design,&lclP->design_scatter));
  PetscCall(ISDestroy(&is_state));
  PetscCall(ISDestroy(&is_design));
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
  PetscCall(VecGetLocalSize(lclP->U,&nlocal));
  PetscCall(VecGetLocalSize(lclP->V,&nlocal));
  PetscCall(MatSetSizes(lclP->R, nlocal, nlocal, lclP->n-lclP->m, lclP->n-lclP->m));
  PetscCall(MatLMVMAllocate(lclP->R,lclP->V,lclP->V));
  lclP->recompute_jacobian_flag = PETSC_TRUE;

  /* Scatter to U,V */
  PetscCall(LCLScatter(lclP,tao->solution,lclP->U,lclP->V));

  /* Evaluate Function, Gradient, Constraints, and Jacobian */
  PetscCall(TaoComputeObjectiveAndGradient(tao,tao->solution,&f,tao->gradient));
  PetscCall(TaoComputeJacobianState(tao,tao->solution,tao->jacobian_state,tao->jacobian_state_pre,tao->jacobian_state_inv));
  PetscCall(TaoComputeJacobianDesign(tao,tao->solution,tao->jacobian_design));
  PetscCall(TaoComputeConstraints(tao,tao->solution, tao->constraints));

  /* Scatter gradient to GU,GV */
  PetscCall(LCLScatter(lclP,tao->gradient,lclP->GU,lclP->GV));

  /* Evaluate Lagrangian function and gradient */
  /* p0 */
  PetscCall(VecSet(lclP->lamda,0.0)); /*  Initial guess in CG */
  PetscCall(MatIsSymmetricKnown(tao->jacobian_state,&set,&flag));
  if (tao->jacobian_state_pre) {
    PetscCall(MatIsSymmetricKnown(tao->jacobian_state_pre,&pset,&pflag));
  } else {
    pset = pflag = PETSC_TRUE;
  }
  if (set && pset && flag && pflag) symmetric = PETSC_TRUE;
  else symmetric = PETSC_FALSE;

  lclP->solve_type = LCL_ADJOINT2;
  if (tao->jacobian_state_inv) {
    if (symmetric) {
      PetscCall(MatMult(tao->jacobian_state_inv, lclP->GU, lclP->lamda)); } else {
      PetscCall(MatMultTranspose(tao->jacobian_state_inv, lclP->GU, lclP->lamda));
    }
  } else {
    PetscCall(KSPSetOperators(tao->ksp, tao->jacobian_state, tao->jacobian_state_pre));
    if (symmetric) {
      PetscCall(KSPSolve(tao->ksp, lclP->GU,  lclP->lamda));
    } else {
      PetscCall(KSPSolveTranspose(tao->ksp, lclP->GU,  lclP->lamda));
    }
    PetscCall(KSPGetIterationNumber(tao->ksp,&its));
    tao->ksp_its+=its;
    tao->ksp_tot_its+=its;
  }
  PetscCall(VecCopy(lclP->lamda,lclP->lamda0));
  PetscCall(LCLComputeAugmentedLagrangianAndGradient(tao->linesearch,tao->solution,&lclP->aug,lclP->GAugL,tao));

  PetscCall(LCLScatter(lclP,lclP->GL,lclP->GL_U,lclP->GL_V));
  PetscCall(LCLScatter(lclP,lclP->GAugL,lclP->GAugL_U,lclP->GAugL_V));

  /* Evaluate constraint norm */
  PetscCall(VecNorm(tao->constraints, NORM_2, &cnorm));
  PetscCall(VecNorm(lclP->GAugL, NORM_2, &mnorm));

  /* Monitor convergence */
  tao->reason = TAO_CONTINUE_ITERATING;
  PetscCall(TaoLogConvergenceHistory(tao,f,mnorm,cnorm,tao->ksp_its));
  PetscCall(TaoMonitor(tao,tao->niter,f,mnorm,cnorm,step));
  PetscCall((*tao->ops->convergencetest)(tao,tao->cnvP));

  while (tao->reason == TAO_CONTINUE_ITERATING) {
    /* Call general purpose update function */
    if (tao->ops->update) {
      PetscCall((*tao->ops->update)(tao, tao->niter, tao->user_update));
    }
    tao->ksp_its=0;
    /* Compute a descent direction for the linearly constrained subproblem
       minimize f(u+du, v+dv)
       s.t. A(u0,v0)du + B(u0,v0)dv = -g(u0,v0) */

    /* Store the points around the linearization */
    PetscCall(VecCopy(lclP->U, lclP->U0));
    PetscCall(VecCopy(lclP->V, lclP->V0));
    PetscCall(VecCopy(lclP->GU,lclP->GU0));
    PetscCall(VecCopy(lclP->GV,lclP->GV0));
    PetscCall(VecCopy(lclP->GAugL_U,lclP->GAugL_U0));
    PetscCall(VecCopy(lclP->GAugL_V,lclP->GAugL_V0));
    PetscCall(VecCopy(lclP->GL_U,lclP->GL_U0));
    PetscCall(VecCopy(lclP->GL_V,lclP->GL_V0));

    lclP->aug0 = lclP->aug;
    lclP->lgn0 = lclP->lgn;

    /* Given the design variables, we need to project the current iterate
       onto the linearized constraint.  We choose to fix the design variables
       and solve the linear system for the state variables.  The resulting
       point is the Newton direction */

    /* Solve r = A\con */
    lclP->solve_type = LCL_FORWARD1;
    PetscCall(VecSet(lclP->r,0.0)); /*  Initial guess in CG */

    if (tao->jacobian_state_inv) {
      PetscCall(MatMult(tao->jacobian_state_inv, tao->constraints, lclP->r));
    } else {
      PetscCall(KSPSetOperators(tao->ksp, tao->jacobian_state, tao->jacobian_state_pre));
      PetscCall(KSPSolve(tao->ksp, tao->constraints,  lclP->r));
      PetscCall(KSPGetIterationNumber(tao->ksp,&its));
      tao->ksp_its+=its;
      tao->ksp_tot_its+=tao->ksp_its;
    }

    /* Set design step direction dv to zero */
    PetscCall(VecSet(lclP->s, 0.0));

    /*
       Check sufficient descent for constraint merit function .5*||con||^2
       con' Ak r >= eps1 ||r||^(2+eps2)
    */

    /* Compute WU= Ak' * con */
    if (symmetric)  {
      PetscCall(MatMult(tao->jacobian_state,tao->constraints,lclP->WU));
    } else {
      PetscCall(MatMultTranspose(tao->jacobian_state,tao->constraints,lclP->WU));
    }
    /* Compute r * Ak' * con */
    PetscCall(VecDot(lclP->r,lclP->WU,&rWU));

    /* compute ||r||^(2+eps2) */
    PetscCall(VecNorm(lclP->r,NORM_2,&r2));
    r2 = PetscPowScalar(r2,2.0+lclP->eps2);
    adec = lclP->eps1 * r2;

    if (rWU < adec) {
      PetscCall(PetscInfo(tao,"Newton direction not descent for constraint, feasibility phase required\n"));
      if (lclP->verbose) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Newton direction not descent for constraint: %g -- using steepest descent\n",(double)descent));
      }

      PetscCall(PetscInfo(tao,"Using steepest descent direction instead.\n"));
      PetscCall(VecSet(lclP->r,0.0));
      PetscCall(VecAXPY(lclP->r,-1.0,lclP->WU));
      PetscCall(VecDot(lclP->r,lclP->r,&rWU));
      PetscCall(VecNorm(lclP->r,NORM_2,&r2));
      r2 = PetscPowScalar(r2,2.0+lclP->eps2);
      PetscCall(VecDot(lclP->r,lclP->GAugL_U,&descent));
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

    PetscCall(VecDot(lclP->r,lclP->GL_U,&rGL_U));
    aldescent =  rGL_U - lclP->rho*rWU;
    if (aldescent > -adec) {
      if (lclP->verbose) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD," Newton direction not descent for augmented Lagrangian: %g",(double)aldescent));
      }
      PetscCall(PetscInfo(tao,"Newton direction not descent for augmented Lagrangian: %g\n",(double)aldescent));
      lclP->rho =  (rGL_U - adec)/rWU;
      if (lclP->rho > lclP->rhomax) {
        lclP->rho = lclP->rhomax;
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"rho=%g > rhomax, case not implemented.  Increase rhomax (-tao_lcl_rhomax)",(double)lclP->rho);
      }
      if (lclP->verbose) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,"  Increasing penalty parameter to %g\n",(double)lclP->rho));
      }
      PetscCall(PetscInfo(tao,"  Increasing penalty parameter to %g\n",(double)lclP->rho));
    }

    PetscCall(LCLComputeAugmentedLagrangianAndGradient(tao->linesearch,tao->solution,&lclP->aug,lclP->GAugL,tao));
    PetscCall(LCLScatter(lclP,lclP->GL,lclP->GL_U,lclP->GL_V));
    PetscCall(LCLScatter(lclP,lclP->GAugL,lclP->GAugL_U,lclP->GAugL_V));

    /* We now minimize the augmented Lagrangian along the Newton direction */
    PetscCall(VecScale(lclP->r,-1.0));
    PetscCall(LCLGather(lclP, lclP->r,lclP->s,tao->stepdirection));
    PetscCall(VecScale(lclP->r,-1.0));
    PetscCall(LCLGather(lclP, lclP->GAugL_U0, lclP->GAugL_V0, lclP->GAugL));
    PetscCall(LCLGather(lclP, lclP->U0,lclP->V0,lclP->X0));

    lclP->recompute_jacobian_flag = PETSC_TRUE;

    PetscCall(TaoLineSearchSetInitialStepLength(tao->linesearch,1.0));
    PetscCall(TaoLineSearchSetType(tao->linesearch, TAOLINESEARCHMT));
    PetscCall(TaoLineSearchSetFromOptions(tao->linesearch));
    PetscCall(TaoLineSearchApply(tao->linesearch, tao->solution, &lclP->aug, lclP->GAugL, tao->stepdirection, &step, &ls_reason));
    if (lclP->verbose) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Steplength = %g\n",(double)step));
    }

    PetscCall(LCLScatter(lclP,tao->solution,lclP->U,lclP->V));
    PetscCall(TaoComputeObjectiveAndGradient(tao,tao->solution,&f,tao->gradient));
    PetscCall(LCLScatter(lclP,tao->gradient,lclP->GU,lclP->GV));

    PetscCall(LCLScatter(lclP,lclP->GAugL,lclP->GAugL_U,lclP->GAugL_V));

    /* Check convergence */
    PetscCall(VecNorm(lclP->GAugL, NORM_2, &mnorm));
    PetscCall(VecNorm(tao->constraints, NORM_2, &cnorm));
    PetscCall(TaoLogConvergenceHistory(tao,f,mnorm,cnorm,tao->ksp_its));
    PetscCall(TaoMonitor(tao,tao->niter,f,mnorm,cnorm,step));
    PetscCall((*tao->ops->convergencetest)(tao,tao->cnvP));
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

      PetscCall(TaoComputeJacobianState(tao,lclP->X0,tao->jacobian_state,tao->jacobian_state_pre,tao->jacobian_state_inv));
      PetscCall(TaoComputeJacobianDesign(tao,lclP->X0,tao->jacobian_design));

      /* Store V and constraints */
      PetscCall(VecCopy(lclP->V, lclP->V1));
      PetscCall(VecCopy(tao->constraints,lclP->con1));

      /* Compute multipliers */
      /* p1 */
      PetscCall(VecSet(lclP->lamda,0.0)); /*  Initial guess in CG */
      lclP->solve_type = LCL_ADJOINT1;
      PetscCall(MatIsSymmetricKnown(tao->jacobian_state,&set,&flag));
      if (tao->jacobian_state_pre) {
        PetscCall(MatIsSymmetricKnown(tao->jacobian_state_pre,&pset,&pflag));
      } else {
        pset = pflag = PETSC_TRUE;
      }
      if (set && pset && flag && pflag) symmetric = PETSC_TRUE;
      else symmetric = PETSC_FALSE;

      if (tao->jacobian_state_inv) {
        if (symmetric) {
          PetscCall(MatMult(tao->jacobian_state_inv, lclP->GAugL_U, lclP->lamda));
        } else {
          PetscCall(MatMultTranspose(tao->jacobian_state_inv, lclP->GAugL_U, lclP->lamda));
        }
      } else {
        if (symmetric) {
          PetscCall(KSPSolve(tao->ksp, lclP->GAugL_U,  lclP->lamda));
        } else {
          PetscCall(KSPSolveTranspose(tao->ksp, lclP->GAugL_U,  lclP->lamda));
        }
        PetscCall(KSPGetIterationNumber(tao->ksp,&its));
        tao->ksp_its+=its;
        tao->ksp_tot_its+=its;
      }
      PetscCall(MatMultTranspose(tao->jacobian_design,lclP->lamda,lclP->g1));
      PetscCall(VecAXPY(lclP->g1,-1.0,lclP->GAugL_V));

      /* Compute the limited-memory quasi-newton direction */
      if (tao->niter > 0) {
        PetscCall(MatSolve(lclP->R,lclP->g1,lclP->s));
        PetscCall(VecDot(lclP->s,lclP->g1,&descent));
        if (descent <= 0) {
          if (lclP->verbose) {
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Reduced-space direction not descent: %g\n",(double)descent));
          }
          PetscCall(VecCopy(lclP->g1,lclP->s));
        }
      } else {
        PetscCall(VecCopy(lclP->g1,lclP->s));
      }
      PetscCall(VecScale(lclP->g1,-1.0));

      /* Recover the full space direction */
      PetscCall(MatMult(tao->jacobian_design,lclP->s,lclP->WU));
      /* PetscCall(VecSet(lclP->r,0.0)); */ /*  Initial guess in CG */
      lclP->solve_type = LCL_FORWARD2;
      if (tao->jacobian_state_inv) {
        PetscCall(MatMult(tao->jacobian_state_inv,lclP->WU,lclP->r));
      } else {
        PetscCall(KSPSolve(tao->ksp, lclP->WU, lclP->r));
        PetscCall(KSPGetIterationNumber(tao->ksp,&its));
        tao->ksp_its += its;
        tao->ksp_tot_its+=its;
      }

      /* We now minimize the augmented Lagrangian along the direction -r,s */
      PetscCall(VecScale(lclP->r, -1.0));
      PetscCall(LCLGather(lclP,lclP->r,lclP->s,tao->stepdirection));
      PetscCall(VecScale(lclP->r, -1.0));
      lclP->recompute_jacobian_flag = PETSC_TRUE;

      PetscCall(TaoLineSearchSetInitialStepLength(tao->linesearch,1.0));
      PetscCall(TaoLineSearchSetType(tao->linesearch,TAOLINESEARCHMT));
      PetscCall(TaoLineSearchSetFromOptions(tao->linesearch));
      PetscCall(TaoLineSearchApply(tao->linesearch, tao->solution, &lclP->aug, lclP->GAugL, tao->stepdirection,&step,&ls_reason));
      if (lclP->verbose) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Reduced-space steplength =  %g\n",(double)step));
      }

      PetscCall(LCLScatter(lclP,tao->solution,lclP->U,lclP->V));
      PetscCall(LCLScatter(lclP,lclP->GL,lclP->GL_U,lclP->GL_V));
      PetscCall(LCLScatter(lclP,lclP->GAugL,lclP->GAugL_U,lclP->GAugL_V));
      PetscCall(TaoComputeObjectiveAndGradient(tao,tao->solution,&f,tao->gradient));
      PetscCall(LCLScatter(lclP,tao->gradient,lclP->GU,lclP->GV));

      /* Compute the reduced gradient at the new point */

      PetscCall(TaoComputeJacobianState(tao,lclP->X0,tao->jacobian_state,tao->jacobian_state_pre,tao->jacobian_state_inv));
      PetscCall(TaoComputeJacobianDesign(tao,lclP->X0,tao->jacobian_design));

      /* p2 */
      /* Compute multipliers, using lamda-rho*con as an initial guess in PCG */
      if (phase2_iter==0) {
        PetscCall(VecWAXPY(lclP->lamda,-lclP->rho,lclP->con1,lclP->lamda0));
      } else {
        PetscCall(VecAXPY(lclP->lamda,-lclP->rho,tao->constraints));
      }

      PetscCall(MatIsSymmetricKnown(tao->jacobian_state,&set,&flag));
      if (tao->jacobian_state_pre) {
        PetscCall(MatIsSymmetricKnown(tao->jacobian_state_pre,&pset,&pflag));
      } else {
        pset = pflag = PETSC_TRUE;
      }
      if (set && pset && flag && pflag) symmetric = PETSC_TRUE;
      else symmetric = PETSC_FALSE;

      lclP->solve_type = LCL_ADJOINT2;
      if (tao->jacobian_state_inv) {
        if (symmetric) {
          PetscCall(MatMult(tao->jacobian_state_inv, lclP->GU, lclP->lamda));
        } else {
          PetscCall(MatMultTranspose(tao->jacobian_state_inv, lclP->GU, lclP->lamda));
        }
      } else {
        if (symmetric) {
          PetscCall(KSPSolve(tao->ksp, lclP->GU,  lclP->lamda));
        } else {
          PetscCall(KSPSolveTranspose(tao->ksp, lclP->GU,  lclP->lamda));
        }
        PetscCall(KSPGetIterationNumber(tao->ksp,&its));
        tao->ksp_its += its;
        tao->ksp_tot_its += its;
      }

      PetscCall(MatMultTranspose(tao->jacobian_design,lclP->lamda,lclP->g2));
      PetscCall(VecAXPY(lclP->g2,-1.0,lclP->GV));

      PetscCall(VecScale(lclP->g2,-1.0));

      /* Update the quasi-newton approximation */
      PetscCall(MatLMVMUpdate(lclP->R,lclP->V,lclP->g2));
      /* Use "-tao_ls_type gpcg -tao_ls_ftol 0 -tao_lmm_broyden_phi 0.0 -tao_lmm_scale_type scalar" to obtain agreement with Matlab code */

    }

    PetscCall(VecCopy(lclP->lamda,lclP->lamda0));

    /* Evaluate Function, Gradient, Constraints, and Jacobian */
    PetscCall(TaoComputeObjectiveAndGradient(tao,tao->solution,&f,tao->gradient));
    PetscCall(LCLScatter(lclP,tao->solution,lclP->U,lclP->V));
    PetscCall(LCLScatter(lclP,tao->gradient,lclP->GU,lclP->GV));

    PetscCall(TaoComputeJacobianState(tao,tao->solution,tao->jacobian_state,tao->jacobian_state_pre,tao->jacobian_state_inv));
    PetscCall(TaoComputeJacobianDesign(tao,tao->solution,tao->jacobian_design));
    PetscCall(TaoComputeConstraints(tao,tao->solution, tao->constraints));

    PetscCall(LCLComputeAugmentedLagrangianAndGradient(tao->linesearch,tao->solution,&lclP->aug,lclP->GAugL,tao));

    PetscCall(VecNorm(lclP->GAugL, NORM_2, &mnorm));

    /* Evaluate constraint norm */
    PetscCall(VecNorm(tao->constraints, NORM_2, &cnorm));

    /* Monitor convergence */
    tao->niter++;
    PetscCall(TaoLogConvergenceHistory(tao,f,mnorm,cnorm,tao->ksp_its));
    PetscCall(TaoMonitor(tao,tao->niter,f,mnorm,cnorm,step));
    PetscCall((*tao->ops->convergencetest)(tao,tao->cnvP));
  }
  PetscFunctionReturn(0);
}

/*MC
 TAOLCL - linearly constrained lagrangian method for pde-constrained optimization

+ -tao_lcl_eps1 - epsilon 1 tolerance
. -tao_lcl_eps2","epsilon 2 tolerance","",lclP->eps2,&lclP->eps2,NULL);
. -tao_lcl_rho0","init value for rho","",lclP->rho0,&lclP->rho0,NULL);
. -tao_lcl_rhomax","max value for rho","",lclP->rhomax,&lclP->rhomax,NULL);
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
  PetscCall(PetscNewLog(tao,&lclP));
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
  PetscCall(TaoLineSearchCreate(((PetscObject)tao)->comm, &tao->linesearch));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)tao->linesearch, (PetscObject)tao, 1));
  PetscCall(TaoLineSearchSetType(tao->linesearch, morethuente_type));
  PetscCall(TaoLineSearchSetOptionsPrefix(tao->linesearch,tao->hdr.prefix));

  PetscCall(TaoLineSearchSetObjectiveAndGradientRoutine(tao->linesearch,LCLComputeAugmentedLagrangianAndGradient, tao));
  PetscCall(KSPCreate(((PetscObject)tao)->comm,&tao->ksp));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)tao->ksp, (PetscObject)tao, 1));
  PetscCall(KSPSetOptionsPrefix(tao->ksp, tao->hdr.prefix));
  PetscCall(KSPSetFromOptions(tao->ksp));

  PetscCall(MatCreate(((PetscObject)tao)->comm, &lclP->R));
  PetscCall(MatSetType(lclP->R, MATLMVMBFGS));
  PetscFunctionReturn(0);
}

static PetscErrorCode LCLComputeLagrangianAndGradient(TaoLineSearch ls, Vec X, PetscReal *f, Vec G, void *ptr)
{
  Tao            tao = (Tao)ptr;
  TAO_LCL        *lclP = (TAO_LCL*)tao->data;
  PetscBool      set,pset,flag,pflag,symmetric;
  PetscReal      cdotl;

  PetscFunctionBegin;
  PetscCall(TaoComputeObjectiveAndGradient(tao,X,f,G));
  PetscCall(LCLScatter(lclP,G,lclP->GU,lclP->GV));
  if (lclP->recompute_jacobian_flag) {
    PetscCall(TaoComputeJacobianState(tao,X,tao->jacobian_state,tao->jacobian_state_pre,tao->jacobian_state_inv));
    PetscCall(TaoComputeJacobianDesign(tao,X,tao->jacobian_design));
  }
  PetscCall(TaoComputeConstraints(tao,X, tao->constraints));
  PetscCall(MatIsSymmetricKnown(tao->jacobian_state,&set,&flag));
  if (tao->jacobian_state_pre) {
    PetscCall(MatIsSymmetricKnown(tao->jacobian_state_pre,&pset,&pflag));
  } else {
    pset = pflag = PETSC_TRUE;
  }
  if (set && pset && flag && pflag) symmetric = PETSC_TRUE;
  else symmetric = PETSC_FALSE;

  PetscCall(VecDot(lclP->lamda0, tao->constraints, &cdotl));
  lclP->lgn = *f - cdotl;

  /* Gradient of Lagrangian GL = G - J' * lamda */
  /*      WU = A' * WL
          WV = B' * WL */
  if (symmetric) {
    PetscCall(MatMult(tao->jacobian_state,lclP->lamda0,lclP->GL_U));
  } else {
    PetscCall(MatMultTranspose(tao->jacobian_state,lclP->lamda0,lclP->GL_U));
  }
  PetscCall(MatMultTranspose(tao->jacobian_design,lclP->lamda0,lclP->GL_V));
  PetscCall(VecScale(lclP->GL_U,-1.0));
  PetscCall(VecScale(lclP->GL_V,-1.0));
  PetscCall(VecAXPY(lclP->GL_U,1.0,lclP->GU));
  PetscCall(VecAXPY(lclP->GL_V,1.0,lclP->GV));
  PetscCall(LCLGather(lclP,lclP->GL_U,lclP->GL_V,lclP->GL));

  f[0] = lclP->lgn;
  PetscCall(VecCopy(lclP->GL,G));
  PetscFunctionReturn(0);
}

static PetscErrorCode LCLComputeAugmentedLagrangianAndGradient(TaoLineSearch ls, Vec X, PetscReal *f, Vec G, void *ptr)
{
  Tao            tao = (Tao)ptr;
  TAO_LCL        *lclP = (TAO_LCL*)tao->data;
  PetscReal      con2;
  PetscBool      flag,pflag,set,pset,symmetric;

  PetscFunctionBegin;
  PetscCall(LCLComputeLagrangianAndGradient(tao->linesearch,X,f,G,tao));
  PetscCall(LCLScatter(lclP,G,lclP->GL_U,lclP->GL_V));
  PetscCall(VecDot(tao->constraints,tao->constraints,&con2));
  lclP->aug = lclP->lgn + 0.5*lclP->rho*con2;

  /* Gradient of Aug. Lagrangian GAugL = GL + rho * J' c */
  /*      WU = A' * c
          WV = B' * c */
  PetscCall(MatIsSymmetricKnown(tao->jacobian_state,&set,&flag));
  if (tao->jacobian_state_pre) {
    PetscCall(MatIsSymmetricKnown(tao->jacobian_state_pre,&pset,&pflag));
  } else {
    pset = pflag = PETSC_TRUE;
  }
  if (set && pset && flag && pflag) symmetric = PETSC_TRUE;
  else symmetric = PETSC_FALSE;

  if (symmetric) {
    PetscCall(MatMult(tao->jacobian_state,tao->constraints,lclP->GAugL_U));
  } else {
    PetscCall(MatMultTranspose(tao->jacobian_state,tao->constraints,lclP->GAugL_U));
  }

  PetscCall(MatMultTranspose(tao->jacobian_design,tao->constraints,lclP->GAugL_V));
  PetscCall(VecAYPX(lclP->GAugL_U,lclP->rho,lclP->GL_U));
  PetscCall(VecAYPX(lclP->GAugL_V,lclP->rho,lclP->GL_V));
  PetscCall(LCLGather(lclP,lclP->GAugL_U,lclP->GAugL_V,lclP->GAugL));

  f[0] = lclP->aug;
  PetscCall(VecCopy(lclP->GAugL,G));
  PetscFunctionReturn(0);
}

PetscErrorCode LCLGather(TAO_LCL *lclP, Vec u, Vec v, Vec x)
{
  PetscFunctionBegin;
  PetscCall(VecScatterBegin(lclP->state_scatter, u, x, INSERT_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterEnd(lclP->state_scatter, u, x, INSERT_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterBegin(lclP->design_scatter, v, x, INSERT_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterEnd(lclP->design_scatter, v, x, INSERT_VALUES, SCATTER_REVERSE));
  PetscFunctionReturn(0);

}
PetscErrorCode LCLScatter(TAO_LCL *lclP, Vec x, Vec u, Vec v)
{
  PetscFunctionBegin;
  PetscCall(VecScatterBegin(lclP->state_scatter, x, u, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(lclP->state_scatter, x, u, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterBegin(lclP->design_scatter, x, v, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(lclP->design_scatter, x, v, INSERT_VALUES, SCATTER_FORWARD));
  PetscFunctionReturn(0);

}
