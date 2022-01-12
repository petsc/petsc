#include <../src/tao/pde_constrained/impls/lcl/lcl.h>
static PetscErrorCode LCLComputeLagrangianAndGradient(TaoLineSearch,Vec,PetscReal*,Vec,void*);
static PetscErrorCode LCLComputeAugmentedLagrangianAndGradient(TaoLineSearch,Vec,PetscReal*,Vec,void*);
static PetscErrorCode LCLScatter(TAO_LCL*,Vec,Vec,Vec);
static PetscErrorCode LCLGather(TAO_LCL*,Vec,Vec,Vec);

static PetscErrorCode TaoDestroy_LCL(Tao tao)
{
  TAO_LCL        *lclP = (TAO_LCL*)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (tao->setupcalled) {
    ierr = MatDestroy(&lclP->R);CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->lamda);CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->lamda0);CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->WL);CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->W);CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->X0);CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->G0);CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->GL);CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->GAugL);CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->dbar);CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->U);CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->U0);CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->V);CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->V0);CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->V1);CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->GU);CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->GV);CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->GU0);CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->GV0);CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->GL_U);CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->GL_V);CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->GAugL_U);CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->GAugL_V);CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->GL_U0);CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->GL_V0);CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->GAugL_U0);CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->GAugL_V0);CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->DU);CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->DV);CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->WU);CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->WV);CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->g1);CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->g2);CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->con1);CHKERRQ(ierr);

    ierr = VecDestroy(&lclP->r);CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->s);CHKERRQ(ierr);

    ierr = ISDestroy(&tao->state_is);CHKERRQ(ierr);
    ierr = ISDestroy(&tao->design_is);CHKERRQ(ierr);

    ierr = VecScatterDestroy(&lclP->state_scatter);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&lclP->design_scatter);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&lclP->R);CHKERRQ(ierr);
  ierr = PetscFree(tao->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSetFromOptions_LCL(PetscOptionItems *PetscOptionsObject,Tao tao)
{
  TAO_LCL        *lclP = (TAO_LCL*)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"Linearly-Constrained Augmented Lagrangian Method for PDE-constrained optimization");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_lcl_eps1","epsilon 1 tolerance","",lclP->eps1,&lclP->eps1,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_lcl_eps2","epsilon 2 tolerance","",lclP->eps2,&lclP->eps2,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_lcl_rho0","init value for rho","",lclP->rho0,&lclP->rho0,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_lcl_rhomax","max value for rho","",lclP->rhomax,&lclP->rhomax,NULL);CHKERRQ(ierr);
  lclP->phase2_niter = 1;
  ierr = PetscOptionsInt("-tao_lcl_phase2_niter","Number of phase 2 iterations in LCL algorithm","",lclP->phase2_niter,&lclP->phase2_niter,NULL);CHKERRQ(ierr);
  lclP->verbose = PETSC_FALSE;
  ierr = PetscOptionsBool("-tao_lcl_verbose","Print verbose output","",lclP->verbose,&lclP->verbose,NULL);CHKERRQ(ierr);
  lclP->tau[0] = lclP->tau[1] = lclP->tau[2] = lclP->tau[3] = 1.0e-4;
  ierr = PetscOptionsReal("-tao_lcl_tola","Tolerance for first forward solve","",lclP->tau[0],&lclP->tau[0],NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_lcl_tolb","Tolerance for first adjoint solve","",lclP->tau[1],&lclP->tau[1],NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_lcl_tolc","Tolerance for second forward solve","",lclP->tau[2],&lclP->tau[2],NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_lcl_told","Tolerance for second adjoint solve","",lclP->tau[3],&lclP->tau[3],NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  ierr = TaoLineSearchSetFromOptions(tao->linesearch);CHKERRQ(ierr);
  ierr = MatSetFromOptions(lclP->R);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  IS             is_state, is_design;

  PetscFunctionBegin;
  if (!tao->state_is) SETERRQ(PetscObjectComm((PetscObject)tao),PETSC_ERR_ARG_WRONGSTATE,"LCL Solver requires an initial state index set -- use TaoSetStateIS()");
  ierr = VecDuplicate(tao->solution, &tao->gradient);CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution, &tao->stepdirection);CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution, &lclP->W);CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution, &lclP->X0);CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution, &lclP->G0);CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution, &lclP->GL);CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution, &lclP->GAugL);CHKERRQ(ierr);

  ierr = VecDuplicate(tao->constraints, &lclP->lamda);CHKERRQ(ierr);
  ierr = VecDuplicate(tao->constraints, &lclP->WL);CHKERRQ(ierr);
  ierr = VecDuplicate(tao->constraints, &lclP->lamda0);CHKERRQ(ierr);
  ierr = VecDuplicate(tao->constraints, &lclP->con1);CHKERRQ(ierr);

  ierr = VecSet(lclP->lamda,0.0);CHKERRQ(ierr);

  ierr = VecGetSize(tao->solution, &lclP->n);CHKERRQ(ierr);
  ierr = VecGetSize(tao->constraints, &lclP->m);CHKERRQ(ierr);

  ierr = VecCreate(((PetscObject)tao)->comm,&lclP->U);CHKERRQ(ierr);
  ierr = VecCreate(((PetscObject)tao)->comm,&lclP->V);CHKERRQ(ierr);
  ierr = ISGetLocalSize(tao->state_is,&nlocalstate);CHKERRQ(ierr);
  ierr = ISGetLocalSize(tao->design_is,&nlocaldesign);CHKERRQ(ierr);
  ierr = VecSetSizes(lclP->U,nlocalstate,lclP->m);CHKERRQ(ierr);
  ierr = VecSetSizes(lclP->V,nlocaldesign,lclP->n-lclP->m);CHKERRQ(ierr);
  ierr = VecSetType(lclP->U,((PetscObject)(tao->solution))->type_name);CHKERRQ(ierr);
  ierr = VecSetType(lclP->V,((PetscObject)(tao->solution))->type_name);CHKERRQ(ierr);
  ierr = VecSetFromOptions(lclP->U);CHKERRQ(ierr);
  ierr = VecSetFromOptions(lclP->V);CHKERRQ(ierr);
  ierr = VecDuplicate(lclP->U,&lclP->DU);CHKERRQ(ierr);
  ierr = VecDuplicate(lclP->U,&lclP->U0);CHKERRQ(ierr);
  ierr = VecDuplicate(lclP->U,&lclP->GU);CHKERRQ(ierr);
  ierr = VecDuplicate(lclP->U,&lclP->GU0);CHKERRQ(ierr);
  ierr = VecDuplicate(lclP->U,&lclP->GAugL_U);CHKERRQ(ierr);
  ierr = VecDuplicate(lclP->U,&lclP->GL_U);CHKERRQ(ierr);
  ierr = VecDuplicate(lclP->U,&lclP->GAugL_U0);CHKERRQ(ierr);
  ierr = VecDuplicate(lclP->U,&lclP->GL_U0);CHKERRQ(ierr);
  ierr = VecDuplicate(lclP->U,&lclP->WU);CHKERRQ(ierr);
  ierr = VecDuplicate(lclP->U,&lclP->r);CHKERRQ(ierr);
  ierr = VecDuplicate(lclP->V,&lclP->V0);CHKERRQ(ierr);
  ierr = VecDuplicate(lclP->V,&lclP->V1);CHKERRQ(ierr);
  ierr = VecDuplicate(lclP->V,&lclP->DV);CHKERRQ(ierr);
  ierr = VecDuplicate(lclP->V,&lclP->s);CHKERRQ(ierr);
  ierr = VecDuplicate(lclP->V,&lclP->GV);CHKERRQ(ierr);
  ierr = VecDuplicate(lclP->V,&lclP->GV0);CHKERRQ(ierr);
  ierr = VecDuplicate(lclP->V,&lclP->dbar);CHKERRQ(ierr);
  ierr = VecDuplicate(lclP->V,&lclP->GAugL_V);CHKERRQ(ierr);
  ierr = VecDuplicate(lclP->V,&lclP->GL_V);CHKERRQ(ierr);
  ierr = VecDuplicate(lclP->V,&lclP->GAugL_V0);CHKERRQ(ierr);
  ierr = VecDuplicate(lclP->V,&lclP->GL_V0);CHKERRQ(ierr);
  ierr = VecDuplicate(lclP->V,&lclP->WV);CHKERRQ(ierr);
  ierr = VecDuplicate(lclP->V,&lclP->g1);CHKERRQ(ierr);
  ierr = VecDuplicate(lclP->V,&lclP->g2);CHKERRQ(ierr);

  /* create scatters for state, design subvecs */
  ierr = VecGetOwnershipRange(lclP->U,&lo,&hi);CHKERRQ(ierr);
  ierr = ISCreateStride(((PetscObject)lclP->U)->comm,hi-lo,lo,1,&is_state);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(lclP->V,&lo,&hi);CHKERRQ(ierr);
  if (0) {
    PetscInt sizeU,sizeV;
    ierr = VecGetSize(lclP->U,&sizeU);CHKERRQ(ierr);
    ierr = VecGetSize(lclP->V,&sizeV);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"size(U)=%D, size(V)=%D\n",sizeU,sizeV);CHKERRQ(ierr);
  }
  ierr = ISCreateStride(((PetscObject)lclP->V)->comm,hi-lo,lo,1,&is_design);CHKERRQ(ierr);
  ierr = VecScatterCreate(tao->solution,tao->state_is,lclP->U,is_state,&lclP->state_scatter);CHKERRQ(ierr);
  ierr = VecScatterCreate(tao->solution,tao->design_is,lclP->V,is_design,&lclP->design_scatter);CHKERRQ(ierr);
  ierr = ISDestroy(&is_state);CHKERRQ(ierr);
  ierr = ISDestroy(&is_design);CHKERRQ(ierr);
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
  PetscErrorCode               ierr;

  PetscFunctionBegin;
  lclP->rho = lclP->rho0;
  ierr = VecGetLocalSize(lclP->U,&nlocal);CHKERRQ(ierr);
  ierr = VecGetLocalSize(lclP->V,&nlocal);CHKERRQ(ierr);
  ierr = MatSetSizes(lclP->R, nlocal, nlocal, lclP->n-lclP->m, lclP->n-lclP->m);CHKERRQ(ierr);
  ierr = MatLMVMAllocate(lclP->R,lclP->V,lclP->V);CHKERRQ(ierr);
  lclP->recompute_jacobian_flag = PETSC_TRUE;

  /* Scatter to U,V */
  ierr = LCLScatter(lclP,tao->solution,lclP->U,lclP->V);CHKERRQ(ierr);

  /* Evaluate Function, Gradient, Constraints, and Jacobian */
  ierr = TaoComputeObjectiveAndGradient(tao,tao->solution,&f,tao->gradient);CHKERRQ(ierr);
  ierr = TaoComputeJacobianState(tao,tao->solution,tao->jacobian_state,tao->jacobian_state_pre,tao->jacobian_state_inv);CHKERRQ(ierr);
  ierr = TaoComputeJacobianDesign(tao,tao->solution,tao->jacobian_design);CHKERRQ(ierr);
  ierr = TaoComputeConstraints(tao,tao->solution, tao->constraints);CHKERRQ(ierr);

  /* Scatter gradient to GU,GV */
  ierr = LCLScatter(lclP,tao->gradient,lclP->GU,lclP->GV);CHKERRQ(ierr);

  /* Evaluate Lagrangian function and gradient */
  /* p0 */
  ierr = VecSet(lclP->lamda,0.0);CHKERRQ(ierr); /*  Initial guess in CG */
  ierr = MatIsSymmetricKnown(tao->jacobian_state,&set,&flag);CHKERRQ(ierr);
  if (tao->jacobian_state_pre) {
    ierr = MatIsSymmetricKnown(tao->jacobian_state_pre,&pset,&pflag);CHKERRQ(ierr);
  } else {
    pset = pflag = PETSC_TRUE;
  }
  if (set && pset && flag && pflag) symmetric = PETSC_TRUE;
  else symmetric = PETSC_FALSE;

  lclP->solve_type = LCL_ADJOINT2;
  if (tao->jacobian_state_inv) {
    if (symmetric) {
      ierr = MatMult(tao->jacobian_state_inv, lclP->GU, lclP->lamda);CHKERRQ(ierr); } else {
      ierr = MatMultTranspose(tao->jacobian_state_inv, lclP->GU, lclP->lamda);CHKERRQ(ierr);
    }
  } else {
    ierr = KSPSetOperators(tao->ksp, tao->jacobian_state, tao->jacobian_state_pre);CHKERRQ(ierr);
    if (symmetric) {
      ierr = KSPSolve(tao->ksp, lclP->GU,  lclP->lamda);CHKERRQ(ierr);
    } else {
      ierr = KSPSolveTranspose(tao->ksp, lclP->GU,  lclP->lamda);CHKERRQ(ierr);
    }
    ierr = KSPGetIterationNumber(tao->ksp,&its);CHKERRQ(ierr);
    tao->ksp_its+=its;
    tao->ksp_tot_its+=its;
  }
  ierr = VecCopy(lclP->lamda,lclP->lamda0);CHKERRQ(ierr);
  ierr = LCLComputeAugmentedLagrangianAndGradient(tao->linesearch,tao->solution,&lclP->aug,lclP->GAugL,tao);CHKERRQ(ierr);

  ierr = LCLScatter(lclP,lclP->GL,lclP->GL_U,lclP->GL_V);CHKERRQ(ierr);
  ierr = LCLScatter(lclP,lclP->GAugL,lclP->GAugL_U,lclP->GAugL_V);CHKERRQ(ierr);

  /* Evaluate constraint norm */
  ierr = VecNorm(tao->constraints, NORM_2, &cnorm);CHKERRQ(ierr);
  ierr = VecNorm(lclP->GAugL, NORM_2, &mnorm);CHKERRQ(ierr);

  /* Monitor convergence */
  tao->reason = TAO_CONTINUE_ITERATING;
  ierr = TaoLogConvergenceHistory(tao,f,mnorm,cnorm,tao->ksp_its);CHKERRQ(ierr);
  ierr = TaoMonitor(tao,tao->niter,f,mnorm,cnorm,step);CHKERRQ(ierr);
  ierr = (*tao->ops->convergencetest)(tao,tao->cnvP);CHKERRQ(ierr);

  while (tao->reason == TAO_CONTINUE_ITERATING) {
    /* Call general purpose update function */
    if (tao->ops->update) {
      ierr = (*tao->ops->update)(tao, tao->niter, tao->user_update);CHKERRQ(ierr);
    }
    tao->ksp_its=0;
    /* Compute a descent direction for the linearly constrained subproblem
       minimize f(u+du, v+dv)
       s.t. A(u0,v0)du + B(u0,v0)dv = -g(u0,v0) */

    /* Store the points around the linearization */
    ierr = VecCopy(lclP->U, lclP->U0);CHKERRQ(ierr);
    ierr = VecCopy(lclP->V, lclP->V0);CHKERRQ(ierr);
    ierr = VecCopy(lclP->GU,lclP->GU0);CHKERRQ(ierr);
    ierr = VecCopy(lclP->GV,lclP->GV0);CHKERRQ(ierr);
    ierr = VecCopy(lclP->GAugL_U,lclP->GAugL_U0);CHKERRQ(ierr);
    ierr = VecCopy(lclP->GAugL_V,lclP->GAugL_V0);CHKERRQ(ierr);
    ierr = VecCopy(lclP->GL_U,lclP->GL_U0);CHKERRQ(ierr);
    ierr = VecCopy(lclP->GL_V,lclP->GL_V0);CHKERRQ(ierr);

    lclP->aug0 = lclP->aug;
    lclP->lgn0 = lclP->lgn;

    /* Given the design variables, we need to project the current iterate
       onto the linearized constraint.  We choose to fix the design variables
       and solve the linear system for the state variables.  The resulting
       point is the Newton direction */

    /* Solve r = A\con */
    lclP->solve_type = LCL_FORWARD1;
    ierr = VecSet(lclP->r,0.0);CHKERRQ(ierr); /*  Initial guess in CG */

    if (tao->jacobian_state_inv) {
      ierr = MatMult(tao->jacobian_state_inv, tao->constraints, lclP->r);CHKERRQ(ierr);
    } else {
      ierr = KSPSetOperators(tao->ksp, tao->jacobian_state, tao->jacobian_state_pre);CHKERRQ(ierr);
      ierr = KSPSolve(tao->ksp, tao->constraints,  lclP->r);CHKERRQ(ierr);
      ierr = KSPGetIterationNumber(tao->ksp,&its);CHKERRQ(ierr);
      tao->ksp_its+=its;
      tao->ksp_tot_its+=tao->ksp_its;
    }

    /* Set design step direction dv to zero */
    ierr = VecSet(lclP->s, 0.0);CHKERRQ(ierr);

    /*
       Check sufficient descent for constraint merit function .5*||con||^2
       con' Ak r >= eps1 ||r||^(2+eps2)
    */

    /* Compute WU= Ak' * con */
    if (symmetric)  {
      ierr = MatMult(tao->jacobian_state,tao->constraints,lclP->WU);CHKERRQ(ierr);
    } else {
      ierr = MatMultTranspose(tao->jacobian_state,tao->constraints,lclP->WU);CHKERRQ(ierr);
    }
    /* Compute r * Ak' * con */
    ierr = VecDot(lclP->r,lclP->WU,&rWU);CHKERRQ(ierr);

    /* compute ||r||^(2+eps2) */
    ierr = VecNorm(lclP->r,NORM_2,&r2);CHKERRQ(ierr);
    r2 = PetscPowScalar(r2,2.0+lclP->eps2);
    adec = lclP->eps1 * r2;

    if (rWU < adec) {
      ierr = PetscInfo(tao,"Newton direction not descent for constraint, feasibility phase required\n");CHKERRQ(ierr);
      if (lclP->verbose) {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"Newton direction not descent for constraint: %g -- using steepest descent\n",(double)descent);CHKERRQ(ierr);
      }

      ierr = PetscInfo(tao,"Using steepest descent direction instead.\n");CHKERRQ(ierr);
      ierr = VecSet(lclP->r,0.0);CHKERRQ(ierr);
      ierr = VecAXPY(lclP->r,-1.0,lclP->WU);CHKERRQ(ierr);
      ierr = VecDot(lclP->r,lclP->r,&rWU);CHKERRQ(ierr);
      ierr = VecNorm(lclP->r,NORM_2,&r2);CHKERRQ(ierr);
      r2 = PetscPowScalar(r2,2.0+lclP->eps2);
      ierr = VecDot(lclP->r,lclP->GAugL_U,&descent);CHKERRQ(ierr);
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

    ierr = VecDot(lclP->r,lclP->GL_U,&rGL_U);CHKERRQ(ierr);
    aldescent =  rGL_U - lclP->rho*rWU;
    if (aldescent > -adec) {
      if (lclP->verbose) {
        ierr = PetscPrintf(PETSC_COMM_WORLD," Newton direction not descent for augmented Lagrangian: %g",(double)aldescent);CHKERRQ(ierr);
      }
      ierr = PetscInfo(tao,"Newton direction not descent for augmented Lagrangian: %g\n",(double)aldescent);CHKERRQ(ierr);
      lclP->rho =  (rGL_U - adec)/rWU;
      if (lclP->rho > lclP->rhomax) {
        lclP->rho = lclP->rhomax;
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"rho=%g > rhomax, case not implemented.  Increase rhomax (-tao_lcl_rhomax)",(double)lclP->rho);
      }
      if (lclP->verbose) {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"  Increasing penalty parameter to %g\n",(double)lclP->rho);CHKERRQ(ierr);
      }
      ierr = PetscInfo(tao,"  Increasing penalty parameter to %g\n",(double)lclP->rho);CHKERRQ(ierr);
    }

    ierr = LCLComputeAugmentedLagrangianAndGradient(tao->linesearch,tao->solution,&lclP->aug,lclP->GAugL,tao);CHKERRQ(ierr);
    ierr = LCLScatter(lclP,lclP->GL,lclP->GL_U,lclP->GL_V);CHKERRQ(ierr);
    ierr = LCLScatter(lclP,lclP->GAugL,lclP->GAugL_U,lclP->GAugL_V);CHKERRQ(ierr);

    /* We now minimize the augmented Lagrangian along the Newton direction */
    ierr = VecScale(lclP->r,-1.0);CHKERRQ(ierr);
    ierr = LCLGather(lclP, lclP->r,lclP->s,tao->stepdirection);CHKERRQ(ierr);
    ierr = VecScale(lclP->r,-1.0);CHKERRQ(ierr);
    ierr = LCLGather(lclP, lclP->GAugL_U0, lclP->GAugL_V0, lclP->GAugL);CHKERRQ(ierr);
    ierr = LCLGather(lclP, lclP->U0,lclP->V0,lclP->X0);CHKERRQ(ierr);

    lclP->recompute_jacobian_flag = PETSC_TRUE;

    ierr = TaoLineSearchSetInitialStepLength(tao->linesearch,1.0);CHKERRQ(ierr);
    ierr = TaoLineSearchSetType(tao->linesearch, TAOLINESEARCHMT);CHKERRQ(ierr);
    ierr = TaoLineSearchSetFromOptions(tao->linesearch);CHKERRQ(ierr);
    ierr = TaoLineSearchApply(tao->linesearch, tao->solution, &lclP->aug, lclP->GAugL, tao->stepdirection, &step, &ls_reason);CHKERRQ(ierr);
    if (lclP->verbose) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Steplength = %g\n",(double)step);CHKERRQ(ierr);
    }

    ierr = LCLScatter(lclP,tao->solution,lclP->U,lclP->V);CHKERRQ(ierr);
    ierr = TaoComputeObjectiveAndGradient(tao,tao->solution,&f,tao->gradient);CHKERRQ(ierr);
    ierr = LCLScatter(lclP,tao->gradient,lclP->GU,lclP->GV);CHKERRQ(ierr);

    ierr = LCLScatter(lclP,lclP->GAugL,lclP->GAugL_U,lclP->GAugL_V);CHKERRQ(ierr);

    /* Check convergence */
    ierr = VecNorm(lclP->GAugL, NORM_2, &mnorm);CHKERRQ(ierr);
    ierr = VecNorm(tao->constraints, NORM_2, &cnorm);CHKERRQ(ierr);
    ierr = TaoLogConvergenceHistory(tao,f,mnorm,cnorm,tao->ksp_its);CHKERRQ(ierr);
    ierr = TaoMonitor(tao,tao->niter,f,mnorm,cnorm,step);CHKERRQ(ierr);
    ierr = (*tao->ops->convergencetest)(tao,tao->cnvP);CHKERRQ(ierr);
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

      ierr = TaoComputeJacobianState(tao,lclP->X0,tao->jacobian_state,tao->jacobian_state_pre,tao->jacobian_state_inv);CHKERRQ(ierr);
      ierr = TaoComputeJacobianDesign(tao,lclP->X0,tao->jacobian_design);CHKERRQ(ierr);

      /* Store V and constraints */
      ierr = VecCopy(lclP->V, lclP->V1);CHKERRQ(ierr);
      ierr = VecCopy(tao->constraints,lclP->con1);CHKERRQ(ierr);

      /* Compute multipliers */
      /* p1 */
      ierr = VecSet(lclP->lamda,0.0);CHKERRQ(ierr); /*  Initial guess in CG */
      lclP->solve_type = LCL_ADJOINT1;
      ierr = MatIsSymmetricKnown(tao->jacobian_state,&set,&flag);CHKERRQ(ierr);
      if (tao->jacobian_state_pre) {
        ierr = MatIsSymmetricKnown(tao->jacobian_state_pre,&pset,&pflag);CHKERRQ(ierr);
      } else {
        pset = pflag = PETSC_TRUE;
      }
      if (set && pset && flag && pflag) symmetric = PETSC_TRUE;
      else symmetric = PETSC_FALSE;

      if (tao->jacobian_state_inv) {
        if (symmetric) {
          ierr = MatMult(tao->jacobian_state_inv, lclP->GAugL_U, lclP->lamda);CHKERRQ(ierr);
        } else {
          ierr = MatMultTranspose(tao->jacobian_state_inv, lclP->GAugL_U, lclP->lamda);CHKERRQ(ierr);
        }
      } else {
        if (symmetric) {
          ierr = KSPSolve(tao->ksp, lclP->GAugL_U,  lclP->lamda);CHKERRQ(ierr);
        } else {
          ierr = KSPSolveTranspose(tao->ksp, lclP->GAugL_U,  lclP->lamda);CHKERRQ(ierr);
        }
        ierr = KSPGetIterationNumber(tao->ksp,&its);CHKERRQ(ierr);
        tao->ksp_its+=its;
        tao->ksp_tot_its+=its;
      }
      ierr = MatMultTranspose(tao->jacobian_design,lclP->lamda,lclP->g1);CHKERRQ(ierr);
      ierr = VecAXPY(lclP->g1,-1.0,lclP->GAugL_V);CHKERRQ(ierr);

      /* Compute the limited-memory quasi-newton direction */
      if (tao->niter > 0) {
        ierr = MatSolve(lclP->R,lclP->g1,lclP->s);CHKERRQ(ierr);
        ierr = VecDot(lclP->s,lclP->g1,&descent);CHKERRQ(ierr);
        if (descent <= 0) {
          if (lclP->verbose) {
            ierr = PetscPrintf(PETSC_COMM_WORLD,"Reduced-space direction not descent: %g\n",(double)descent);CHKERRQ(ierr);
          }
          ierr = VecCopy(lclP->g1,lclP->s);CHKERRQ(ierr);
        }
      } else {
        ierr = VecCopy(lclP->g1,lclP->s);CHKERRQ(ierr);
      }
      ierr = VecScale(lclP->g1,-1.0);CHKERRQ(ierr);

      /* Recover the full space direction */
      ierr = MatMult(tao->jacobian_design,lclP->s,lclP->WU);CHKERRQ(ierr);
      /* ierr = VecSet(lclP->r,0.0);CHKERRQ(ierr); */ /*  Initial guess in CG */
      lclP->solve_type = LCL_FORWARD2;
      if (tao->jacobian_state_inv) {
        ierr = MatMult(tao->jacobian_state_inv,lclP->WU,lclP->r);CHKERRQ(ierr);
      } else {
        ierr = KSPSolve(tao->ksp, lclP->WU, lclP->r);CHKERRQ(ierr);
        ierr = KSPGetIterationNumber(tao->ksp,&its);CHKERRQ(ierr);
        tao->ksp_its += its;
        tao->ksp_tot_its+=its;
      }

      /* We now minimize the augmented Lagrangian along the direction -r,s */
      ierr = VecScale(lclP->r, -1.0);CHKERRQ(ierr);
      ierr = LCLGather(lclP,lclP->r,lclP->s,tao->stepdirection);CHKERRQ(ierr);
      ierr = VecScale(lclP->r, -1.0);CHKERRQ(ierr);
      lclP->recompute_jacobian_flag = PETSC_TRUE;

      ierr = TaoLineSearchSetInitialStepLength(tao->linesearch,1.0);CHKERRQ(ierr);
      ierr = TaoLineSearchSetType(tao->linesearch,TAOLINESEARCHMT);CHKERRQ(ierr);
      ierr = TaoLineSearchSetFromOptions(tao->linesearch);CHKERRQ(ierr);
      ierr = TaoLineSearchApply(tao->linesearch, tao->solution, &lclP->aug, lclP->GAugL, tao->stepdirection,&step,&ls_reason);CHKERRQ(ierr);
      if (lclP->verbose) {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"Reduced-space steplength =  %g\n",(double)step);CHKERRQ(ierr);
      }

      ierr = LCLScatter(lclP,tao->solution,lclP->U,lclP->V);CHKERRQ(ierr);
      ierr = LCLScatter(lclP,lclP->GL,lclP->GL_U,lclP->GL_V);CHKERRQ(ierr);
      ierr = LCLScatter(lclP,lclP->GAugL,lclP->GAugL_U,lclP->GAugL_V);CHKERRQ(ierr);
      ierr = TaoComputeObjectiveAndGradient(tao,tao->solution,&f,tao->gradient);CHKERRQ(ierr);
      ierr = LCLScatter(lclP,tao->gradient,lclP->GU,lclP->GV);CHKERRQ(ierr);

      /* Compute the reduced gradient at the new point */

      ierr = TaoComputeJacobianState(tao,lclP->X0,tao->jacobian_state,tao->jacobian_state_pre,tao->jacobian_state_inv);CHKERRQ(ierr);
      ierr = TaoComputeJacobianDesign(tao,lclP->X0,tao->jacobian_design);CHKERRQ(ierr);

      /* p2 */
      /* Compute multipliers, using lamda-rho*con as an initial guess in PCG */
      if (phase2_iter==0) {
        ierr = VecWAXPY(lclP->lamda,-lclP->rho,lclP->con1,lclP->lamda0);CHKERRQ(ierr);
      } else {
        ierr = VecAXPY(lclP->lamda,-lclP->rho,tao->constraints);CHKERRQ(ierr);
      }

      ierr = MatIsSymmetricKnown(tao->jacobian_state,&set,&flag);CHKERRQ(ierr);
      if (tao->jacobian_state_pre) {
        ierr = MatIsSymmetricKnown(tao->jacobian_state_pre,&pset,&pflag);CHKERRQ(ierr);
      } else {
        pset = pflag = PETSC_TRUE;
      }
      if (set && pset && flag && pflag) symmetric = PETSC_TRUE;
      else symmetric = PETSC_FALSE;

      lclP->solve_type = LCL_ADJOINT2;
      if (tao->jacobian_state_inv) {
        if (symmetric) {
          ierr = MatMult(tao->jacobian_state_inv, lclP->GU, lclP->lamda);CHKERRQ(ierr);
        } else {
          ierr = MatMultTranspose(tao->jacobian_state_inv, lclP->GU, lclP->lamda);CHKERRQ(ierr);
        }
      } else {
        if (symmetric) {
          ierr = KSPSolve(tao->ksp, lclP->GU,  lclP->lamda);CHKERRQ(ierr);
        } else {
          ierr = KSPSolveTranspose(tao->ksp, lclP->GU,  lclP->lamda);CHKERRQ(ierr);
        }
        ierr = KSPGetIterationNumber(tao->ksp,&its);CHKERRQ(ierr);
        tao->ksp_its += its;
        tao->ksp_tot_its += its;
      }

      ierr = MatMultTranspose(tao->jacobian_design,lclP->lamda,lclP->g2);CHKERRQ(ierr);
      ierr = VecAXPY(lclP->g2,-1.0,lclP->GV);CHKERRQ(ierr);

      ierr = VecScale(lclP->g2,-1.0);CHKERRQ(ierr);

      /* Update the quasi-newton approximation */
      ierr = MatLMVMUpdate(lclP->R,lclP->V,lclP->g2);CHKERRQ(ierr);
      /* Use "-tao_ls_type gpcg -tao_ls_ftol 0 -tao_lmm_broyden_phi 0.0 -tao_lmm_scale_type scalar" to obtain agreement with Matlab code */

    }

    ierr = VecCopy(lclP->lamda,lclP->lamda0);CHKERRQ(ierr);

    /* Evaluate Function, Gradient, Constraints, and Jacobian */
    ierr = TaoComputeObjectiveAndGradient(tao,tao->solution,&f,tao->gradient);CHKERRQ(ierr);
    ierr = LCLScatter(lclP,tao->solution,lclP->U,lclP->V);CHKERRQ(ierr);
    ierr = LCLScatter(lclP,tao->gradient,lclP->GU,lclP->GV);CHKERRQ(ierr);

    ierr = TaoComputeJacobianState(tao,tao->solution,tao->jacobian_state,tao->jacobian_state_pre,tao->jacobian_state_inv);CHKERRQ(ierr);
    ierr = TaoComputeJacobianDesign(tao,tao->solution,tao->jacobian_design);CHKERRQ(ierr);
    ierr = TaoComputeConstraints(tao,tao->solution, tao->constraints);CHKERRQ(ierr);

    ierr = LCLComputeAugmentedLagrangianAndGradient(tao->linesearch,tao->solution,&lclP->aug,lclP->GAugL,tao);CHKERRQ(ierr);

    ierr = VecNorm(lclP->GAugL, NORM_2, &mnorm);CHKERRQ(ierr);

    /* Evaluate constraint norm */
    ierr = VecNorm(tao->constraints, NORM_2, &cnorm);CHKERRQ(ierr);

    /* Monitor convergence */
    tao->niter++;
    ierr = TaoLogConvergenceHistory(tao,f,mnorm,cnorm,tao->ksp_its);CHKERRQ(ierr);
    ierr = TaoMonitor(tao,tao->niter,f,mnorm,cnorm,step);CHKERRQ(ierr);
    ierr = (*tao->ops->convergencetest)(tao,tao->cnvP);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  const char     *morethuente_type = TAOLINESEARCHMT;

  PetscFunctionBegin;
  tao->ops->setup = TaoSetup_LCL;
  tao->ops->solve = TaoSolve_LCL;
  tao->ops->view = TaoView_LCL;
  tao->ops->setfromoptions = TaoSetFromOptions_LCL;
  tao->ops->destroy = TaoDestroy_LCL;
  ierr = PetscNewLog(tao,&lclP);CHKERRQ(ierr);
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
  ierr = TaoLineSearchCreate(((PetscObject)tao)->comm, &tao->linesearch);CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject)tao->linesearch, (PetscObject)tao, 1);CHKERRQ(ierr);
  ierr = TaoLineSearchSetType(tao->linesearch, morethuente_type);CHKERRQ(ierr);
  ierr = TaoLineSearchSetOptionsPrefix(tao->linesearch,tao->hdr.prefix);CHKERRQ(ierr);

  ierr = TaoLineSearchSetObjectiveAndGradientRoutine(tao->linesearch,LCLComputeAugmentedLagrangianAndGradient, tao);CHKERRQ(ierr);
  ierr = KSPCreate(((PetscObject)tao)->comm,&tao->ksp);CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject)tao->ksp, (PetscObject)tao, 1);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(tao->ksp, tao->hdr.prefix);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(tao->ksp);CHKERRQ(ierr);

  ierr = MatCreate(((PetscObject)tao)->comm, &lclP->R);CHKERRQ(ierr);
  ierr = MatSetType(lclP->R, MATLMVMBFGS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode LCLComputeLagrangianAndGradient(TaoLineSearch ls, Vec X, PetscReal *f, Vec G, void *ptr)
{
  Tao            tao = (Tao)ptr;
  TAO_LCL        *lclP = (TAO_LCL*)tao->data;
  PetscBool      set,pset,flag,pflag,symmetric;
  PetscReal      cdotl;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TaoComputeObjectiveAndGradient(tao,X,f,G);CHKERRQ(ierr);
  ierr = LCLScatter(lclP,G,lclP->GU,lclP->GV);CHKERRQ(ierr);
  if (lclP->recompute_jacobian_flag) {
    ierr = TaoComputeJacobianState(tao,X,tao->jacobian_state,tao->jacobian_state_pre,tao->jacobian_state_inv);CHKERRQ(ierr);
    ierr = TaoComputeJacobianDesign(tao,X,tao->jacobian_design);CHKERRQ(ierr);
  }
  ierr = TaoComputeConstraints(tao,X, tao->constraints);CHKERRQ(ierr);
  ierr = MatIsSymmetricKnown(tao->jacobian_state,&set,&flag);CHKERRQ(ierr);
  if (tao->jacobian_state_pre) {
    ierr = MatIsSymmetricKnown(tao->jacobian_state_pre,&pset,&pflag);CHKERRQ(ierr);
  } else {
    pset = pflag = PETSC_TRUE;
  }
  if (set && pset && flag && pflag) symmetric = PETSC_TRUE;
  else symmetric = PETSC_FALSE;

  ierr = VecDot(lclP->lamda0, tao->constraints, &cdotl);CHKERRQ(ierr);
  lclP->lgn = *f - cdotl;

  /* Gradient of Lagrangian GL = G - J' * lamda */
  /*      WU = A' * WL
          WV = B' * WL */
  if (symmetric) {
    ierr = MatMult(tao->jacobian_state,lclP->lamda0,lclP->GL_U);CHKERRQ(ierr);
  } else {
    ierr = MatMultTranspose(tao->jacobian_state,lclP->lamda0,lclP->GL_U);CHKERRQ(ierr);
  }
  ierr = MatMultTranspose(tao->jacobian_design,lclP->lamda0,lclP->GL_V);CHKERRQ(ierr);
  ierr = VecScale(lclP->GL_U,-1.0);CHKERRQ(ierr);
  ierr = VecScale(lclP->GL_V,-1.0);CHKERRQ(ierr);
  ierr = VecAXPY(lclP->GL_U,1.0,lclP->GU);CHKERRQ(ierr);
  ierr = VecAXPY(lclP->GL_V,1.0,lclP->GV);CHKERRQ(ierr);
  ierr = LCLGather(lclP,lclP->GL_U,lclP->GL_V,lclP->GL);CHKERRQ(ierr);

  f[0] = lclP->lgn;
  ierr = VecCopy(lclP->GL,G);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode LCLComputeAugmentedLagrangianAndGradient(TaoLineSearch ls, Vec X, PetscReal *f, Vec G, void *ptr)
{
  Tao            tao = (Tao)ptr;
  TAO_LCL        *lclP = (TAO_LCL*)tao->data;
  PetscReal      con2;
  PetscBool      flag,pflag,set,pset,symmetric;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = LCLComputeLagrangianAndGradient(tao->linesearch,X,f,G,tao);CHKERRQ(ierr);
  ierr = LCLScatter(lclP,G,lclP->GL_U,lclP->GL_V);CHKERRQ(ierr);
  ierr = VecDot(tao->constraints,tao->constraints,&con2);CHKERRQ(ierr);
  lclP->aug = lclP->lgn + 0.5*lclP->rho*con2;

  /* Gradient of Aug. Lagrangian GAugL = GL + rho * J' c */
  /*      WU = A' * c
          WV = B' * c */
  ierr = MatIsSymmetricKnown(tao->jacobian_state,&set,&flag);CHKERRQ(ierr);
  if (tao->jacobian_state_pre) {
    ierr = MatIsSymmetricKnown(tao->jacobian_state_pre,&pset,&pflag);CHKERRQ(ierr);
  } else {
    pset = pflag = PETSC_TRUE;
  }
  if (set && pset && flag && pflag) symmetric = PETSC_TRUE;
  else symmetric = PETSC_FALSE;

  if (symmetric) {
    ierr = MatMult(tao->jacobian_state,tao->constraints,lclP->GAugL_U);CHKERRQ(ierr);
  } else {
    ierr = MatMultTranspose(tao->jacobian_state,tao->constraints,lclP->GAugL_U);CHKERRQ(ierr);
  }

  ierr = MatMultTranspose(tao->jacobian_design,tao->constraints,lclP->GAugL_V);CHKERRQ(ierr);
  ierr = VecAYPX(lclP->GAugL_U,lclP->rho,lclP->GL_U);CHKERRQ(ierr);
  ierr = VecAYPX(lclP->GAugL_V,lclP->rho,lclP->GL_V);CHKERRQ(ierr);
  ierr = LCLGather(lclP,lclP->GAugL_U,lclP->GAugL_V,lclP->GAugL);CHKERRQ(ierr);

  f[0] = lclP->aug;
  ierr = VecCopy(lclP->GAugL,G);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode LCLGather(TAO_LCL *lclP, Vec u, Vec v, Vec x)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecScatterBegin(lclP->state_scatter, u, x, INSERT_VALUES, SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(lclP->state_scatter, u, x, INSERT_VALUES, SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterBegin(lclP->design_scatter, v, x, INSERT_VALUES, SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(lclP->design_scatter, v, x, INSERT_VALUES, SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);

}
PetscErrorCode LCLScatter(TAO_LCL *lclP, Vec x, Vec u, Vec v)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecScatterBegin(lclP->state_scatter, x, u, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(lclP->state_scatter, x, u, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterBegin(lclP->design_scatter, x, v, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(lclP->design_scatter, x, v, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);

}
