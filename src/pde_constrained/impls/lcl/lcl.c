#include "lcl.h"
#include "src/matrix/lmvmmat.h"
#include "src/matrix/approxmat.h"
#include "src/matrix/submatfree.h"
static PetscErrorCode LCLComputeLagrangianAndGradient(TaoLineSearch,Vec,PetscReal*,Vec,void*);
static PetscErrorCode LCLComputeAugmentedLagrangianAndGradient(TaoLineSearch,Vec,PetscReal*,Vec,void*);
static PetscErrorCode LCLScatter(TAO_LCL*,Vec,Vec,Vec);
static PetscErrorCode LCLGather(TAO_LCL*,Vec,Vec,Vec);

#undef __FUNCT__
#define __FUNCT__ "TaoSolverDestroy_LCL"
static PetscErrorCode TaoSolverDestroy_LCL(TaoSolver tao)
{
  TAO_LCL *lclP = (TAO_LCL*)tao->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (tao->setupcalled) {
    ierr = MatDestroy(&lclP->R); CHKERRQ(ierr);
    
    ierr = VecDestroy(&lclP->lamda); CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->lamda0); CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->WL); CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->W); CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->X0); CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->G0); CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->GL); CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->GAugL); CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->dbar); CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->U); CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->U0); CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->V); CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->V0); CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->V1); CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->GU); CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->GV); CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->GU0); CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->GV0); CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->GL_U); CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->GL_V); CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->GAugL_U); CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->GAugL_V); CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->GL_U0); CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->GL_V0); CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->GAugL_U0); CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->GAugL_V0); CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->DU); CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->DV); CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->WU); CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->WV); CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->g1); CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->g2); CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->con1); CHKERRQ(ierr);

    ierr = VecDestroy(&lclP->r); CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->s); CHKERRQ(ierr);

    ierr = ISDestroy(&tao->state_is); CHKERRQ(ierr);
    ierr = ISDestroy(&tao->design_is); CHKERRQ(ierr);
    //ierr = ISDestroy(&lclP->UIM); CHKERRQ(ierr);

    ierr = VecScatterDestroy(&lclP->state_scatter); CHKERRQ(ierr);
    ierr = VecScatterDestroy(&lclP->design_scatter); CHKERRQ(ierr);
  }
  ierr = PetscFree(tao->data);
  tao->data = PETSC_NULL;

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoSolverSetFromOptions_LCL"
static PetscErrorCode TaoSolverSetFromOptions_LCL(TaoSolver tao)
{
  /*TAO_LCL *lclP = (TAO_LCL*)tao->data;*/
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = TaoLineSearchSetFromOptions(tao->linesearch); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoSolverView_LCL"
static PetscErrorCode TaoSolverView_LCL(TaoSolver tao, PetscViewer viewer)
{
  /*
  TAO_LCL *lclP = (TAO_LCL*)tao->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscFunctionReturn(0);
  */
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "TaoSolverSetup_LCL"
static PetscErrorCode TaoSolverSetup_LCL(TaoSolver tao)
{
  TAO_LCL *lclP = (TAO_LCL*)tao->data;
  PetscInt lo, hi;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  /* Check for state/design IS */
  if (!tao->state_is) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONGSTATE,"LCL Solver requires an initial state index set -- use TaoSolverSetStateIS()");
  }
  ierr = VecDuplicate(tao->solution, &tao->gradient); CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution, &tao->stepdirection); CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution, &lclP->W); CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution, &lclP->X0); CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution, &lclP->G0); CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution, &lclP->GL); CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution, &lclP->GAugL); CHKERRQ(ierr);
  
  ierr = VecDuplicate(tao->constraints, &lclP->lamda); CHKERRQ(ierr);
  //ierr = VecDuplicate(tao->constraints, &lclP->DL); CHKERRQ(ierr);
  ierr = VecDuplicate(tao->constraints, &lclP->WL); CHKERRQ(ierr);
  ierr = VecDuplicate(tao->constraints, &lclP->lamda0); CHKERRQ(ierr);
  ierr = VecDuplicate(tao->constraints, &lclP->con1); CHKERRQ(ierr);

  ierr = VecSet(lclP->lamda,0.0); CHKERRQ(ierr);

  ierr = VecGetSize(tao->solution, &lclP->n); CHKERRQ(ierr);
  ierr = VecGetSize(tao->constraints, &lclP->m); CHKERRQ(ierr);

  //ierr = VecGetOwnershipRange(tao->solution,&lo,&hi); CHKERRQ(ierr);
  //ierr = ISComplement(tao->state_is,lo,hi,&tao->design_is); CHKERRQ(ierr);
  //ierr = VecGetOwnershipRange(tao->constraints,&lo,&hi); CHKERRQ(ierr);
  //ierr = ISCreateStride(((PetscObject)tao)->comm,hi-lo,lo+lclP->n-lclP->m,1,&tao->design_is);

  //ierr = VecGetOwnershipRange(tao->constraints,&lo,&hi); CHKERRQ(ierr);
  //ierr = ISCreateStride(((PetscObject)tao)->comm,hi-lo,lo,1,&lclP->UIM);

  IS is_state, is_design;
  ierr = VecCreate(((PetscObject)tao)->comm,&lclP->U); CHKERRQ(ierr);
  ierr = VecCreate(((PetscObject)tao)->comm,&lclP->V); CHKERRQ(ierr);
  ierr = VecSetSizes(lclP->U,PETSC_DECIDE,lclP->m); CHKERRQ(ierr);
  ierr = VecSetSizes(lclP->V,PETSC_DECIDE,lclP->n-lclP->m); CHKERRQ(ierr);
  ierr = VecSetType(lclP->U,((PetscObject)(tao->solution))->type_name); CHKERRQ(ierr);
  ierr = VecSetType(lclP->V,((PetscObject)(tao->solution))->type_name); CHKERRQ(ierr);
  ierr = VecSetFromOptions(lclP->U); CHKERRQ(ierr);
  ierr = VecSetFromOptions(lclP->V); CHKERRQ(ierr);
  ierr = VecDuplicate(lclP->U,&lclP->DU); CHKERRQ(ierr);
  ierr = VecDuplicate(lclP->U,&lclP->U0); CHKERRQ(ierr);
  ierr = VecDuplicate(lclP->U,&lclP->GU); CHKERRQ(ierr);
  ierr = VecDuplicate(lclP->U,&lclP->GU0); CHKERRQ(ierr);
  ierr = VecDuplicate(lclP->U,&lclP->GAugL_U); CHKERRQ(ierr);
  ierr = VecDuplicate(lclP->U,&lclP->GL_U); CHKERRQ(ierr);
  ierr = VecDuplicate(lclP->U,&lclP->GAugL_U0); CHKERRQ(ierr);
  ierr = VecDuplicate(lclP->U,&lclP->GL_U0); CHKERRQ(ierr);
  ierr = VecDuplicate(lclP->U,&lclP->WU); CHKERRQ(ierr);
  ierr = VecDuplicate(lclP->U,&lclP->r); CHKERRQ(ierr);
  ierr = VecDuplicate(lclP->V,&lclP->V0); CHKERRQ(ierr);
  ierr = VecDuplicate(lclP->V,&lclP->V1); CHKERRQ(ierr);
  ierr = VecDuplicate(lclP->V,&lclP->DV); CHKERRQ(ierr);
  ierr = VecDuplicate(lclP->V,&lclP->s); CHKERRQ(ierr);
  ierr = VecDuplicate(lclP->V,&lclP->GV); CHKERRQ(ierr);
  ierr = VecDuplicate(lclP->V,&lclP->GV0); CHKERRQ(ierr);
  ierr = VecDuplicate(lclP->V,&lclP->dbar); CHKERRQ(ierr);
  ierr = VecDuplicate(lclP->V,&lclP->GAugL_V); CHKERRQ(ierr);
  ierr = VecDuplicate(lclP->V,&lclP->GL_V); CHKERRQ(ierr);
  ierr = VecDuplicate(lclP->V,&lclP->GAugL_V0); CHKERRQ(ierr);
  ierr = VecDuplicate(lclP->V,&lclP->GL_V0); CHKERRQ(ierr);
  ierr = VecDuplicate(lclP->V,&lclP->WV); CHKERRQ(ierr);
  ierr = VecDuplicate(lclP->V,&lclP->g1); CHKERRQ(ierr);
  ierr = VecDuplicate(lclP->V,&lclP->g2); CHKERRQ(ierr);
  

  /*ierr = MatDuplicate(tao->jacobian_state,MAT_SHARE_NONZERO_PATTERN,&lclP->jacobian_state0); CHKERRQ(ierr);
  if (tao->jacobian_state != tao->jacobian_state_pre) {
    ierr = MatDuplicate(tao->jacobian_state_pre,MAT_SHARE_NONZERO_PATTERN,&lclP->jacobian_state0_pre); CHKERRQ(ierr);
    }
  ierr = MatDuplicate(tao->jacobian_design,MAT_SHARE_NONZERO_PATTERN,&lclP->jacobian_design0); CHKERRQ(ierr);*/
  /*lclP->jacobian_design0 = tao->jacobian_design;
  lclP->jacobian_state0 = tao->jacobian_state;
  lclP->jacobian_state0_pre = tao->jacobian_state_pre;*/
  

  /* create scatters for state, design subvecs */
  ierr = VecGetOwnershipRange(lclP->U,&lo,&hi); CHKERRQ(ierr);
  ierr = ISCreateStride(((PetscObject)lclP->U)->comm,hi-lo,lo,1,&is_state); CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(lclP->V,&lo,&hi); CHKERRQ(ierr);
  ierr = ISCreateStride(((PetscObject)lclP->V)->comm,hi-lo,lo,1,&is_design); CHKERRQ(ierr);
  ierr = VecScatterCreate(tao->solution,tao->state_is,lclP->U,is_state,&lclP->state_scatter); CHKERRQ(ierr);
  ierr = VecScatterCreate(tao->solution,tao->design_is,lclP->V,is_design,&lclP->design_scatter); CHKERRQ(ierr);  //ierr = VecScatterCreate(tao->solution,tao->state_is,lclP->U,is_state,&lclP->state_scatter); CHKERRQ(ierr);
  ierr = ISDestroy(&is_state); CHKERRQ(ierr);
  ierr = ISDestroy(&is_design); CHKERRQ(ierr); 

  //ierr = VecGetLocalSize(lclP->U,&nlocal); CHKERRQ(ierr);
  //ierr = VecGetLocalSize(lclP->V,&nlocal); CHKERRQ(ierr);
  //ierr = MatCreateLMVM(((PetscObject)tao)->comm,nlocal,lclP->n - lclP->m,&lclP->R); CHKERRQ(ierr);
  //ierr = MatLMVMAllocateVectors(lclP->R,lclP->V); CHKERRQ(ierr);
  //lclP->rho = 1.0e-4;

  PetscBool flag;
  lclP->phase2_niter = 1;
  ierr = PetscOptionsInt("-tao_lcl_phase2_niter","Number of phase 2 iterations in LCL algorithm","",lclP->phase2_niter,&lclP->phase2_niter,&flag); CHKERRQ(ierr);
  lclP->verbose = PETSC_FALSE;
  ierr = PetscOptionsBool("-tao_lcl_verbose","Print verbose output","",lclP->verbose,&lclP->verbose,&flag); CHKERRQ(ierr);
  lclP->tola = 1e-4;
  ierr = PetscOptionsReal("-tao_lcl_tola","Tolerance for first forward solve","",lclP->tola,&lclP->tola,&flag); CHKERRQ(ierr);
  lclP->tolb = 1e-4;
  ierr = PetscOptionsReal("-tao_lcl_tolb","Tolerance for first adjoint solve","",lclP->tolb,&lclP->tolb,&flag); CHKERRQ(ierr);
  lclP->tolc = 1e-4;
  ierr = PetscOptionsReal("-tao_lcl_tolc","Tolerance for second forward solve","",lclP->tolc,&lclP->tolc,&flag); CHKERRQ(ierr);
  lclP->told = 1e-4;
  ierr = PetscOptionsReal("-tao_lcl_told","Tolerance for second adjoint solve","",lclP->told,&lclP->told,&flag); CHKERRQ(ierr);

  if (!tao->jacobian_state_inv) {
    ierr = KSPCreate(((PetscObject)tao)->comm,&tao->ksp); CHKERRQ(ierr);
    ierr = KSPSetFromOptions(tao->ksp); CHKERRQ(ierr);
  }


  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoSolverSolve_LCL"
static PetscErrorCode TaoSolverSolve_LCL(TaoSolver tao)
{
  TAO_LCL *lclP = (TAO_LCL*)tao->data;
  PetscInt iter=0,phase2_iter,nlocal;
  TaoSolverTerminationReason reason = TAO_CONTINUE_ITERATING;
  TaoLineSearchTerminationReason ls_reason = TAOLINESEARCH_CONTINUE_ITERATING;
  PetscReal step=1.0,f, descent,con2;
  PetscReal cnorm, mnorm;
  PetscReal adec,r2,rGL_U,rWU;
  
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = VecGetLocalSize(lclP->U,&nlocal); CHKERRQ(ierr);
  ierr = VecGetLocalSize(lclP->V,&nlocal); CHKERRQ(ierr);
  ierr = MatCreateLMVM(((PetscObject)tao)->comm,nlocal,lclP->n - lclP->m,&lclP->R); CHKERRQ(ierr);
  ierr = MatLMVMAllocateVectors(lclP->R,lclP->V); CHKERRQ(ierr);
  lclP->rho = 1.0e-4;
  lclP->recompute_jacobian_flag = PETSC_TRUE;
  
  /* Scatter to U,V */
  ierr = LCLScatter(lclP,tao->solution,lclP->U,lclP->V); CHKERRQ(ierr);
  
  /* Evaluate Function, Gradient, Constraints, and Jacobian */
  ierr = TaoSolverComputeObjectiveAndGradient(tao,tao->solution,&f,tao->gradient); CHKERRQ(ierr);
  //ierr = TaoSolverComputeConstraints(tao,tao->solution, tao->constraints); CHKERRQ(ierr);
  ierr = TaoSolverComputeJacobianState(tao,tao->solution, &tao->jacobian_state, &tao->jacobian_state_pre, &tao->jacobian_state_inv, &lclP->statematflag); CHKERRQ(ierr);
<<<<<<< local
  ierr = TaoSolverComputeJacobianDesign(tao,tao->solution, &tao->jacobian_design, &tao->jacobian_design_pre, &lclP->statematflag); CHKERRQ(ierr);
  ierr = TaoSolverComputeConstraints(tao,tao->solution, tao->constraints); CHKERRQ(ierr);
=======
  ierr = TaoSolverComputeJacobianDesign(tao,tao->solution, &tao->jacobian_design); CHKERRQ(ierr);
>>>>>>> other

  
  /* Scatter gradient to GU,GV */
  ierr = LCLScatter(lclP,tao->gradient,lclP->GU,lclP->GV); CHKERRQ(ierr);
  

  /* Evaluate Lagrangian function and gradient */
  /* p0 */
  ierr = VecSet(lclP->lamda,0.0); CHKERRQ(ierr); // Initial guess in CG
  if (tao->jacobian_state_inv) {
    ierr = MatMultTranspose(tao->jacobian_state_inv, lclP->GU, lclP->lamda); CHKERRQ(ierr);
  } else {
    ierr = KSPSetOperators(tao->ksp, tao->jacobian_state, tao->jacobian_state_pre, lclP->statematflag); CHKERRQ(ierr);
    ierr = KSPSolveTranspose(tao->ksp, lclP->GU,  lclP->lamda); CHKERRQ(ierr);
  }
    
  ierr = VecCopy(lclP->lamda,lclP->lamda0); CHKERRQ(ierr);

  //ierr = LCLComputeLagrangianAndGradient(tao->linesearch,tao->solution,&lclP->lgn,lclP->GL,tao); CHKERRQ(ierr);
  ierr = LCLComputeAugmentedLagrangianAndGradient(tao->linesearch,tao->solution,&lclP->aug,lclP->GAugL,tao); CHKERRQ(ierr);

  ierr = LCLScatter(lclP,lclP->GL,lclP->GL_U,lclP->GL_V); CHKERRQ(ierr);
  ierr = LCLScatter(lclP,lclP->GAugL,lclP->GAugL_U,lclP->GAugL_V); CHKERRQ(ierr);
  

  /* Evaluate constraint norm */
  ierr = VecNorm(tao->constraints, NORM_2, &cnorm); CHKERRQ(ierr); 
  ierr = VecNorm(lclP->GAugL, NORM_2, &mnorm); CHKERRQ(ierr);
  

  /* Monitor convergence */
  ierr = TaoSolverMonitor(tao, iter,f,mnorm,cnorm,step,&reason); CHKERRQ(ierr);

  while (reason == TAO_CONTINUE_ITERATING) {
    /* Compute a descent direction for the linearly constrained subproblem
       minimize f(u+du, v+dv)
       s.t. A(u0,v0)du + B(u0,v0)dv = -g(u0,v0) */

    /* Store the points around the linearization */
    ierr = VecCopy(lclP->U, lclP->U0); CHKERRQ(ierr);
    ierr = VecCopy(lclP->V, lclP->V0); CHKERRQ(ierr);
    ierr = VecCopy(lclP->GU,lclP->GU0); CHKERRQ(ierr);
    ierr = VecCopy(lclP->GV,lclP->GV0); CHKERRQ(ierr);
    ierr = VecCopy(lclP->GAugL_U,lclP->GAugL_U0); CHKERRQ(ierr);
    ierr = VecCopy(lclP->GAugL_V,lclP->GAugL_V0); CHKERRQ(ierr);
    ierr = VecCopy(lclP->GL_U,lclP->GL_U0); CHKERRQ(ierr);
    ierr = VecCopy(lclP->GL_V,lclP->GL_V0); CHKERRQ(ierr);

    lclP->aug0 = lclP->aug;
    lclP->lgn0 = lclP->lgn;

    /*lclP->jacobian_design0 = tao->jacobian_design;
    lclP->jacobian_state0 = tao->jacobian_state;
    lclP->jacobian_state0_pre = tao->jacobian_state_pre;
    lclP->jacobian_state_inv0 = tao->jacobian_state_inv;*/
    
    /* Given the design variables, we need to project the current iterate
       onto the linearized constraint.  We choose to fix the design variables
       and solve the linear system for the state variables.  The resulting
       point is the Newton direction */
    
    /* Solve r = A\con */
    ierr = VecSet(lclP->r,0.0); CHKERRQ(ierr); // Initial guess in CG
<<<<<<< local
    ierr = MatMult(tao->jacobian_state_inv, tao->constraints, lclP->r); CHKERRQ(ierr);
=======
    if (lclP->jacobian_state_inv0) {
      ierr = MatMult(lclP->jacobian_state_inv0, tao->constraints, lclP->r); CHKERRQ(ierr);
    } else {
      ierr = KSPSetOperators(tao->ksp, lclP->jacobian_state0, lclP->jacobian_state0_pre, lclP->statematflag); CHKERRQ(ierr);
      ierr = KSPSolve(tao->ksp, tao->constraints,  lclP->r); CHKERRQ(ierr);
    }
      
>>>>>>> other

    ierr = VecNorm(lclP->r,NORM_2,&cnorm); CHKERRQ(ierr);
    ierr = VecSet(lclP->s, 0.0); CHKERRQ(ierr);

    /* Make sure the Newton direction is a descent direction for the merit function */
    ierr = MatMultTranspose(tao->jacobian_state,tao->constraints,lclP->WU); CHKERRQ(ierr);
    
    ierr = VecDot(lclP->r,lclP->WU,&descent); CHKERRQ(ierr);
    if (descent <= 0) {
      ierr = PetscInfo1(tao,"Newton direction not descent: %g\n",descent); CHKERRQ(ierr);
      if (lclP->verbose) {
	ierr = PetscPrintf(PETSC_COMM_WORLD,"Newton direction not descent: %g\n",descent); CHKERRQ(ierr);
      }
      ierr = PetscInfo(tao,"Using steepest descent direction instead.\n"); CHKERRQ(ierr);
      ierr = VecSet(lclP->r,0.0); CHKERRQ(ierr);
      ierr = VecAXPY(lclP->r,-1.0,lclP->WU); CHKERRQ(ierr);
    }

    /* Check descent for aug. lagrangian */
    ierr = VecDot(lclP->r,lclP->r,&r2); CHKERRQ(ierr);
    ierr = VecDot(lclP->r,lclP->GAugL_U,&descent); CHKERRQ(ierr); 
    adec = 1e-8 * r2;
    if (descent <= adec) {
      ierr = PetscInfo1(tao,"Newton direction not descent for augmented Lagrangian: %g",descent); CHKERRQ(ierr);
      ierr = VecDot(tao->constraints,tao->constraints,&con2); CHKERRQ(ierr);
      ierr = VecDot(lclP->r,lclP->GL_U,&rGL_U);
      ierr = VecDot(lclP->r,lclP->WU,&rWU);
      lclP->rho = 2 * (adec - rGL_U) / rWU;
      ierr = PetscInfo1(tao,"  Increasing penalty parameter to %g",lclP->rho);
      if (lclP->verbose) {
	ierr = PetscPrintf(PETSC_COMM_WORLD,"  Increasing penalty parameter to %g\n",lclP->rho); CHKERRQ(ierr);
      }
    }
    //PetscPrintf(PETSC_COMM_WORLD,"rho = %10.5f\n",lclP->rho);

    //ierr = LCLComputeLagrangianAndGradient(tao->linesearch,tao->solution,&lclP->lgn,lclP->GL,tao); CHKERRQ(ierr);
    ierr = LCLComputeAugmentedLagrangianAndGradient(tao->linesearch,tao->solution,&lclP->aug,lclP->GAugL,tao); CHKERRQ(ierr);
    ierr = LCLScatter(lclP,lclP->GL,lclP->GL_U,lclP->GL_V); CHKERRQ(ierr);
    ierr = LCLScatter(lclP,lclP->GAugL,lclP->GAugL_U,lclP->GAugL_V); CHKERRQ(ierr);

    /* We now minimize the augmented Lagrangian along the Newton direction */
    ierr = VecScale(lclP->r,-1.0); CHKERRQ(ierr);
    ierr = LCLGather(lclP, lclP->r,lclP->s,tao->stepdirection); CHKERRQ(ierr);
    ierr = VecScale(lclP->r,-1.0); CHKERRQ(ierr);
    ierr = LCLGather(lclP, lclP->GAugL_U0, lclP->GAugL_V0, lclP->GAugL); CHKERRQ(ierr);
    ierr = LCLGather(lclP, lclP->U0,lclP->V0,lclP->X0); CHKERRQ(ierr);

    lclP->recompute_jacobian_flag = PETSC_TRUE;

    ierr = TaoLineSearchSetInitialStepLength(tao->linesearch,1.0); CHKERRQ(ierr);
    ierr = TaoLineSearchSetType(tao->linesearch, TAOLINESEARCH_MT); CHKERRQ(ierr);
    ierr = TaoLineSearchSetFromOptions(tao->linesearch); CHKERRQ(ierr);
    ierr = TaoLineSearchApply(tao->linesearch, tao->solution, &lclP->aug, lclP->GAugL, tao->stepdirection, &step, &ls_reason); CHKERRQ(ierr);
    if (lclP->verbose) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Steplength = %10.8f\n",step); CHKERRQ(ierr);
    }
    //TaoLineSearchView(tao->linesearch,PETSC_VIEWER_STDOUT_WORLD);
    
    ierr = LCLScatter(lclP,tao->solution,lclP->U,lclP->V); CHKERRQ(ierr);
    ierr = TaoSolverComputeObjectiveAndGradient(tao,tao->solution,&f,tao->gradient); CHKERRQ(ierr);
    ierr = LCLScatter(lclP,tao->gradient,lclP->GU,lclP->GV); CHKERRQ(ierr);

    ierr = LCLScatter(lclP,lclP->GAugL,lclP->GAugL_U,lclP->GAugL_V); CHKERRQ(ierr); 

    /* Check convergence */
    ierr = VecNorm(lclP->GAugL, NORM_2, &mnorm); CHKERRQ(ierr);
    ierr = VecNorm(tao->constraints, NORM_2, &cnorm); CHKERRQ(ierr);
    ierr = TaoSolverMonitor(tao, iter,f,mnorm,cnorm,step,&reason); CHKERRQ(ierr);
    if (reason != TAO_CONTINUE_ITERATING){
      break;
    }


    // TODO: use a heuristic to choose how many iterations should be performed within phase 2
    for (phase2_iter=0; phase2_iter<lclP->phase2_niter; phase2_iter++){
      /* We now minimize the objective function starting from the fraction of
	 the Newton point accepted by applying one step of a reduced-space
	 method.  The optimization problem is:
       
         minimize f(u+du, v+dv)
         s. t.    A(u0,v0)du + B(u0,v0)du = -alpha g(u0,v0)

	 In particular, we have that
	 du = -inv(A)*(Bdv + alpha g) */

      ierr = TaoSolverComputeJacobianState(tao,lclP->X0,&tao->jacobian_state,&tao->jacobian_state_pre,&tao->jacobian_state_inv,&lclP->statematflag); CHKERRQ(ierr);
      ierr = TaoSolverComputeJacobianDesign(tao,lclP->X0,&tao->jacobian_design,&tao->jacobian_design_pre,&lclP->designmatflag); CHKERRQ(ierr);

      /* Store V and constraints */
      ierr = VecCopy(lclP->V, lclP->V1); CHKERRQ(ierr);
      ierr = VecCopy(tao->constraints,lclP->con1); CHKERRQ(ierr);

      /* Compute multipliers */
      /* p1 */
      ierr = VecSet(lclP->lamda,0.0); CHKERRQ(ierr); // Initial guess in CG
<<<<<<< local
      ierr = MatMultTranspose(tao->jacobian_state_inv, lclP->GAugL_U, lclP->lamda); CHKERRQ(ierr);
      
=======
      if (lclP->jacobian_state_inv0) {
	ierr = MatMultTranspose(lclP->jacobian_state_inv0, lclP->GAugL_U, lclP->lamda); CHKERRQ(ierr);
      } else {
	ierr = KSPSolveTranspose(tao->ksp, lclP->GAugL_U,  lclP->lamda); CHKERRQ(ierr);
      }
>>>>>>> other

      ierr = MatMultTranspose(tao->jacobian_design,lclP->lamda,lclP->g1); CHKERRQ(ierr);
      ierr = VecAXPY(lclP->g1,-1.0,lclP->GAugL_V); CHKERRQ(ierr);
    
      /* Compute the limited-memory quasi-newton direction */
      if (iter > 0) {
	ierr = MatLMVMSolve(lclP->R,lclP->g1,lclP->s); CHKERRQ(ierr);
	ierr = VecDot(lclP->s,lclP->g1,&descent); CHKERRQ(ierr);
	if (descent < 0e-8) {
	  if (lclP->verbose) {
	    ierr = PetscPrintf(PETSC_COMM_WORLD,"Reduced-space direction not descent: %g\n",descent); CHKERRQ(ierr);
	  }
	  ierr = VecCopy(lclP->g1,lclP->s); CHKERRQ(ierr);
	}
      } else {
	ierr = VecCopy(lclP->g1,lclP->s); CHKERRQ(ierr);
      }
      ierr = VecScale(lclP->g1,-1.0); CHKERRQ(ierr);

      //MatView_LMVM(lclP->R,PETSC_VIEWER_STDOUT_WORLD);

      /* Recover the full space direction */ 
      ierr = MatMult(tao->jacobian_design,lclP->s,lclP->WU); CHKERRQ(ierr);
      ierr = VecSet(lclP->r,0.0); CHKERRQ(ierr); // Initial guess in CG
<<<<<<< local
      ierr = MatMult(tao->jacobian_state_inv,lclP->WU,lclP->r); CHKERRQ(ierr);
=======
      if (lclP->jacobian_state_inv0) {
	ierr = MatMult(lclP->jacobian_state_inv0,lclP->WU,lclP->r); CHKERRQ(ierr);
      } else {
//	ierr = KSPSetOperators(tao->ksp,lclP->jacobian_state0, lclP->jacobian_state0_pre,lslP->statematflag); CHKERRQ(ierr);
	ierr = KSPSolve(tao->ksp, lclP->WU, lclP->r); CHKERRQ(ierr);
      }
>>>>>>> other

      /* We now minimize the augmented Lagrangian along the direction -r,s */
      ierr = VecScale(lclP->r, -1.0); CHKERRQ(ierr);
      ierr = LCLGather(lclP,lclP->r,lclP->s,tao->stepdirection); CHKERRQ(ierr);
      ierr = VecScale(lclP->r, -1.0); CHKERRQ(ierr);
      lclP->recompute_jacobian_flag = PETSC_TRUE;

      ierr = TaoLineSearchSetInitialStepLength(tao->linesearch,1.0); CHKERRQ(ierr);
      ierr = TaoLineSearchSetType(tao->linesearch,TAOLINESEARCH_MT); CHKERRQ(ierr);
      ierr = TaoLineSearchSetFromOptions(tao->linesearch); CHKERRQ(ierr);
      ierr = TaoLineSearchApply(tao->linesearch, tao->solution, &lclP->aug, lclP->GAugL, tao->stepdirection,&step,&ls_reason); CHKERRQ(ierr);
      if (lclP->verbose){
	ierr = PetscPrintf(PETSC_COMM_WORLD,"Reduced-space steplength =  %10.8f\n",step); CHKERRQ(ierr);
      }
      //TaoLineSearchView(tao->linesearch,PETSC_VIEWER_STDOUT_WORLD);

      ierr = LCLScatter(lclP,tao->solution,lclP->U,lclP->V); CHKERRQ(ierr);
      ierr = LCLScatter(lclP,lclP->GL,lclP->GL_U,lclP->GL_V); CHKERRQ(ierr);
      ierr = LCLScatter(lclP,lclP->GAugL,lclP->GAugL_U,lclP->GAugL_V); CHKERRQ(ierr);
      ierr = TaoSolverComputeObjectiveAndGradient(tao,tao->solution,&f,tao->gradient); CHKERRQ(ierr);
      ierr = LCLScatter(lclP,tao->gradient,lclP->GU,lclP->GV); CHKERRQ(ierr);
      //ierr = TaoSolverComputeConstraints(tao,tao->solution, tao->constraints); CHKERRQ(ierr); // !!! 9-8-11

      /* TODO - check convergence? */
    
      /* Compute the reduced gradient at the new point */

      ierr = TaoSolverComputeJacobianState(tao,lclP->X0,&tao->jacobian_state,&tao->jacobian_state_pre,&tao->jacobian_state_inv,&lclP->statematflag); CHKERRQ(ierr);
      ierr = TaoSolverComputeJacobianDesign(tao,lclP->X0,&tao->jacobian_design,&tao->jacobian_design_pre,&lclP->designmatflag); CHKERRQ(ierr);

      /* p2 */
      /* Compute multipliers, using lamda-rho*con as an initial guess in PCG */
      if (phase2_iter==0){
	ierr = VecWAXPY(lclP->lamda,-lclP->rho,lclP->con1,lclP->lamda0); CHKERRQ(ierr);
      }
      else {
	ierr = VecAXPY(lclP->lamda,-lclP->rho,tao->constraints); CHKERRQ(ierr);
      }
<<<<<<< local
      ierr = MatMultTranspose(tao->jacobian_state_inv, lclP->GU, lclP->lamda); CHKERRQ(ierr);
=======
      if (lclP->jacobian_state_inv0) {
	ierr = MatMultTranspose(lclP->jacobian_state_inv0, lclP->GU, lclP->lamda); CHKERRQ(ierr);
      } else {
	ierr = KSPSolveTranspose(tao->ksp, lclP->GU,  lclP->lamda); CHKERRQ(ierr);
      }
	
>>>>>>> other
    
      ierr = MatMultTranspose(tao->jacobian_design,lclP->lamda,lclP->g2); CHKERRQ(ierr);  
      ierr = VecAXPY(lclP->g2,-1.0,lclP->GV); CHKERRQ(ierr);

      ierr = VecScale(lclP->g2,-1.0); CHKERRQ(ierr);

      /* Update the quasi-newton approximation */
      if (phase2_iter >= 0){
	ierr = MatLMVMSetPrev(lclP->R,lclP->V1,lclP->g1);
      }
      ierr = MatLMVMUpdate(lclP->R,lclP->V,lclP->g2); CHKERRQ(ierr);
      /* Use "-tao_ls_type gpcg -tao_ls_ftol 0 -tao_lmm_broyden_phi 0.0 -tao_lmm_scale_type scalar" to obtain agreement with Matlab code */

    }

    ierr = VecCopy(lclP->lamda,lclP->lamda0); CHKERRQ(ierr);
    
    /* Evaluate Function, Gradient, Constraints, and Jacobian */
    ierr = TaoSolverComputeObjectiveAndGradient(tao,tao->solution,&f,tao->gradient); CHKERRQ(ierr);
    ierr = LCLScatter(lclP,tao->solution,lclP->U,lclP->V); CHKERRQ(ierr);
    ierr = LCLScatter(lclP,tao->gradient,lclP->GU,lclP->GV); CHKERRQ(ierr);

    //ierr = TaoSolverComputeConstraints(tao,tao->solution, tao->constraints); CHKERRQ(ierr);
    ierr = TaoSolverComputeJacobianState(tao,tao->solution, &tao->jacobian_state, &tao->jacobian_state_pre, &tao->jacobian_state_inv, &lclP->statematflag); CHKERRQ(ierr);
<<<<<<< local
    ierr = TaoSolverComputeJacobianDesign(tao,tao->solution, &tao->jacobian_design, &tao->jacobian_design_pre, &lclP->designmatflag); CHKERRQ(ierr);
    ierr = TaoSolverComputeConstraints(tao,tao->solution, tao->constraints); CHKERRQ(ierr);
=======
    ierr = TaoSolverComputeJacobianDesign(tao,tao->solution, &tao->jacobian_design); CHKERRQ(ierr);
>>>>>>> other

    //ierr = LCLComputeLagrangianAndGradient(tao->linesearch,tao->solution,&lclP->lgn,lclP->GL,tao); CHKERRQ(ierr);
    ierr = LCLComputeAugmentedLagrangianAndGradient(tao->linesearch,tao->solution,&lclP->aug,lclP->GAugL,tao); CHKERRQ(ierr);

    ierr = VecNorm(lclP->GAugL, NORM_2, &mnorm); CHKERRQ(ierr);

    /* Evaluate constraint norm */
    ierr = VecNorm(tao->constraints, NORM_2, &cnorm); CHKERRQ(ierr);
  
    /* Monitor convergence */
    iter++;
    ierr = TaoSolverMonitor(tao, iter,f,mnorm,cnorm,step,&reason); CHKERRQ(ierr);
    
  }

   ierr = PetscPrintf(PETSC_COMM_WORLD,"Convergence in %i iterations\n",iter);
   ierr = MatDestroy(&lclP->R); CHKERRQ(ierr);

   PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "TaoSolverCreate_LCL"
PetscErrorCode TaoSolverCreate_LCL(TaoSolver tao)
{
  TAO_LCL *lclP;
  PetscErrorCode ierr;
  const char *morethuente_type = TAOLINESEARCH_MT;
  PetscFunctionBegin;
  
  tao->ops->setup = TaoSolverSetup_LCL;
  tao->ops->solve = TaoSolverSolve_LCL;
  tao->ops->view = TaoSolverView_LCL;
  tao->ops->setfromoptions = TaoSolverSetFromOptions_LCL;
  tao->ops->destroy = TaoSolverDestroy_LCL;
  

  ierr = PetscNewLog(tao,TAO_LCL,&lclP); CHKERRQ(ierr);
  tao->data = (void*)lclP;

  /*tao->max_its=200;
  tao->fatol=1e-4;
  tao->frtol=1e-4;
  tao->gatol=1e-4;
  tao->grtol=1e-4;*/

  tao->max_its=200;
  tao->fatol=1e-8;
  tao->frtol=1e-8;
  tao->catol=1e-4;
  tao->crtol=1e-4;
  tao->gttol=1e-4;
  tao->gatol=1e-4;
  tao->grtol=1e-4;
  

  //lclP->subset_type=LCL_SUBSETes_SUBMAT;

  ierr = TaoLineSearchCreate(((PetscObject)tao)->comm, &tao->linesearch); CHKERRQ(ierr);
  ierr = TaoLineSearchSetType(tao->linesearch, morethuente_type); CHKERRQ(ierr);

  ierr = TaoLineSearchSetObjectiveAndGradientRoutine(tao->linesearch,LCLComputeAugmentedLagrangianAndGradient, tao); CHKERRQ(ierr);
  
  PetscFunctionReturn(0);

}
EXTERN_C_END


#undef __FUNCT__
#define __FUNCT__ "LCLComputeLagrangianAndGradient"
static PetscErrorCode LCLComputeLagrangianAndGradient(TaoLineSearch ls, Vec X, PetscReal *f, Vec G, void *ptr)
{
  TaoSolver tao = (TaoSolver)ptr;
  TAO_LCL *lclP = (TAO_LCL*)tao->data;
  PetscReal cdotl;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TaoSolverComputeObjectiveAndGradient(tao,X,f,G); CHKERRQ(ierr);
  ierr = LCLScatter(lclP,G,lclP->GU,lclP->GV); CHKERRQ(ierr);
  if (lclP->recompute_jacobian_flag) {
    //ierr = TaoSolverComputeConstraints(tao,X, tao->constraints); CHKERRQ(ierr);
    ierr = TaoSolverComputeJacobianState(tao,X, &tao->jacobian_state, &tao->jacobian_state_pre, &tao->jacobian_state_inv, &lclP->statematflag); CHKERRQ(ierr);
    ierr = TaoSolverComputeJacobianDesign(tao,X, &tao->jacobian_design); CHKERRQ(ierr);
  }
  ierr = TaoSolverComputeConstraints(tao,X, tao->constraints); CHKERRQ(ierr);

  /* Keep the Lagrange multipliers fixed during the linesearch */
  /*ierr = KSPSetOperators(tao->ksp, tao->jacobian_state, tao->jacobian_state_pre, lclP->statematflag); CHKERRQ(ierr);
  ierr = KSPSolveTranspose(tao->ksp, lclP->GU,  lclP->lamda); CHKERRQ(ierr);
  ierr = LCLMonitorConvergence(tao->ksp); CHKERRQ(ierr);*/

  ierr = VecDot(lclP->lamda0, tao->constraints, &cdotl); CHKERRQ(ierr);
  lclP->lgn = *f - cdotl;

  /* Gradient of Lagrangian GL = G - J' * lamda */
  /*      WU = A' * WL
          WV = B' * WL */
  ierr = MatMultTranspose(tao->jacobian_state,lclP->lamda0,lclP->GL_U); CHKERRQ(ierr);
  ierr = MatMultTranspose(tao->jacobian_design,lclP->lamda0,lclP->GL_V); CHKERRQ(ierr);
  ierr = VecScale(lclP->GL_U,-1.0); CHKERRQ(ierr);
  ierr = VecScale(lclP->GL_V,-1.0); CHKERRQ(ierr);
  ierr = VecAXPY(lclP->GL_U,1.0,lclP->GU); CHKERRQ(ierr); 
  ierr = VecAXPY(lclP->GL_V,1.0,lclP->GV); CHKERRQ(ierr); 
  //ierr = LCLGather(lclP,lclP->GL_U,lclP->GL_V,G); CHKERRQ(ierr);
  ierr = LCLGather(lclP,lclP->GL_U,lclP->GL_V,lclP->GL); CHKERRQ(ierr);

  f[0] = lclP->lgn;
  ierr = VecCopy(lclP->GL,G);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "LCLComputeAugmentedLagrangianAndGradient"
static PetscErrorCode LCLComputeAugmentedLagrangianAndGradient(TaoLineSearch ls, Vec X, PetscReal *f, Vec G, void *ptr)
{
  TaoSolver tao = (TaoSolver)ptr;
  TAO_LCL *lclP = (TAO_LCL*)tao->data;
  PetscReal con2;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = LCLComputeLagrangianAndGradient(tao->linesearch,X,f,G,tao); CHKERRQ(ierr);
  ierr = LCLScatter(lclP,G,lclP->GL_U,lclP->GL_V); CHKERRQ(ierr);
  ierr = VecDot(tao->constraints,tao->constraints,&con2); CHKERRQ(ierr);
  lclP->aug = lclP->lgn + 0.5*lclP->rho*con2;
  
  /* Gradient of Aug. Lagrangian GAugL = GL + rho * J' c */
  /*      WU = A' * c
          WV = B' * c */
  ierr = MatMultTranspose(tao->jacobian_state,tao->constraints,lclP->GAugL_U); CHKERRQ(ierr);
  ierr = MatMultTranspose(tao->jacobian_design,tao->constraints,lclP->GAugL_V); CHKERRQ(ierr);
  ierr = VecAYPX(lclP->GAugL_U,lclP->rho,lclP->GL_U); CHKERRQ(ierr);
  ierr = VecAYPX(lclP->GAugL_V,lclP->rho,lclP->GL_V); CHKERRQ(ierr);
  ierr = LCLGather(lclP,lclP->GAugL_U,lclP->GAugL_V,lclP->GAugL); CHKERRQ(ierr);

  f[0] = lclP->aug;
  ierr = VecCopy(lclP->GAugL,G); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "LCLGather"
PetscErrorCode LCLGather(TAO_LCL *lclP, Vec u, Vec v, Vec x)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecScatterBegin(lclP->state_scatter, u, x, INSERT_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
  ierr = VecScatterEnd(lclP->state_scatter, u, x, INSERT_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
  ierr = VecScatterBegin(lclP->design_scatter, v, x, INSERT_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
  ierr = VecScatterEnd(lclP->design_scatter, v, x, INSERT_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
  PetscFunctionReturn(0);
  
}
#undef __FUNCT__
#define __FUNCT__ "LCLScatter"
PetscErrorCode LCLScatter(TAO_LCL *lclP, Vec x, Vec u, Vec v)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecScatterBegin(lclP->state_scatter, x, u, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(lclP->state_scatter, x, u, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterBegin(lclP->design_scatter, x, v, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(lclP->design_scatter, x, v, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  PetscFunctionReturn(0);
  
}
