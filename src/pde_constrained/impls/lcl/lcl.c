#include "lcl.h"
#include "src/matrix/lmvmmat.h"
#include "src/matrix/approxmat.h"
#include "src/matrix/submatfree.h"
static PetscErrorCode LCLComputeLagrangianAndGradient(TaoLineSearch,Vec,PetscReal*,Vec,void*);
static PetscErrorCode LCLComputeAugmentedLagrangianAndGradient(TaoLineSearch,Vec,PetscReal*,Vec,void*);
static PetscErrorCode LCLScatter(TAO_LCL*,Vec,Vec,Vec);
static PetscErrorCode LCLGather(TAO_LCL*,Vec,Vec,Vec);

#undef __FUNCT__
#define __FUNCT__ "TaoSolverLCLSetStateIS"
PetscErrorCode TaoSolverLCLSetStateIS(TaoSolver tao, IS is)
{
  TAO_LCL *lclP = (TAO_LCL*)tao->data;
  const char *type;
  PetscErrorCode ierr;
  PetscBool islcl;
  ierr = PetscObjectGetType((PetscObject)tao,&type); CHKERRQ(ierr);
  ierr = PetscStrcmp(type,"tao_lcl",&islcl); CHKERRQ(ierr);
  if (!islcl) {
    ierr = PetscInfo(tao,"Ignored for non-pde_constrained solvers"); CHKERRQ(ierr);
  }
  if (lclP->UIS) {
    ierr = PetscObjectDereference((PetscObject)lclP->UIS); CHKERRQ(ierr);
  }
  lclP->UIS = is;
  if (is) {
    ierr = PetscObjectReference((PetscObject)is); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

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

    ierr = VecDestroy(&lclP->r); CHKERRQ(ierr);
    ierr = VecDestroy(&lclP->s); CHKERRQ(ierr);

    ierr = ISDestroy(&lclP->UIS); CHKERRQ(ierr);
    ierr = ISDestroy(&lclP->UID); CHKERRQ(ierr);
    ierr = ISDestroy(&lclP->UIM); CHKERRQ(ierr);

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
  PetscInt lo, hi, nlocal;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  /* Check for state IS */
  if (!lclP->UIS) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONGSTATE,"LCL Solver requires an initial state index set -- use TaoSolverLCLSetStateIS()");
  }
  ierr = VecDuplicate(tao->solution, &tao->gradient); CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution, &tao->stepdirection); CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution, &lclP->W); CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution, &lclP->X0); CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution, &lclP->G0); CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution, &lclP->GL); CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution, &lclP->GAugL); CHKERRQ(ierr);
  
  ierr = VecDuplicate(tao->constraints, &lclP->lamda); CHKERRQ(ierr);
  ierr = VecDuplicate(tao->constraints, &lclP->DL); CHKERRQ(ierr);
  ierr = VecDuplicate(tao->constraints, &lclP->WL); CHKERRQ(ierr);

  ierr = VecSet(lclP->lamda,0.0); CHKERRQ(ierr);

  ierr = VecGetSize(tao->solution, &lclP->n); CHKERRQ(ierr);
  ierr = VecGetSize(tao->constraints, &lclP->m); CHKERRQ(ierr);

  ierr = VecGetOwnershipRange(tao->solution,&lo,&hi); CHKERRQ(ierr);
  ierr = ISComplement(lclP->UIS,lo,hi,&lclP->UID); CHKERRQ(ierr);

  ierr = VecGetOwnershipRange(tao->constraints,&lo,&hi); CHKERRQ(ierr);
  ierr = ISCreateStride(((PetscObject)tao)->comm,hi-lo,lo,1,&lclP->UIM);

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

/*  ierr = MatDuplicate(tao->jacobian_state,MAT_SHARE_NONZERO_PATTERN,&lclP->jacobian_state0); CHKERRQ(ierr);
  if (tao->jacobian_state != tao->jacobian_state_pre) {
    ierr = MatDuplicate(tao->jacobian_state_pre,MAT_SHARE_NONZERO_PATTERN,&lclP->jacobian_state0_pre); CHKERRQ(ierr);
  }
  ierr = MatDuplicate(tao->jacobian_design,MAT_SHARE_NONZERO_PATTERN,&lclP->jacobian_design0); CHKERRQ(ierr);*/
  lclP->jacobian_design0 = tao->jacobian_design;
  lclP->jacobian_state0 = tao->jacobian_state;
  lclP->jacobian_state0_pre = tao->jacobian_state_pre;
  

  /* create scatters for state, design subvecs */
  ierr = VecGetOwnershipRange(lclP->U,&lo,&hi); CHKERRQ(ierr);
  ierr = ISCreateStride(((PetscObject)lclP->U)->comm,hi-lo,lo,1,&is_state); CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(lclP->V,&lo,&hi); CHKERRQ(ierr);
  ierr = ISCreateStride(((PetscObject)lclP->V)->comm,hi-lo,lo,1,&is_design); CHKERRQ(ierr);
  ierr = VecScatterCreate(tao->solution,lclP->UIS,lclP->U,is_state,&lclP->state_scatter); CHKERRQ(ierr);
  ierr = VecScatterCreate(tao->solution,lclP->UID,lclP->V,is_design,&lclP->design_scatter); CHKERRQ(ierr);
  ierr = ISDestroy(&is_state); CHKERRQ(ierr);
  ierr = ISDestroy(&is_design); CHKERRQ(ierr);

  ierr = VecGetLocalSize(lclP->U,&nlocal); CHKERRQ(ierr);
  ierr = VecGetLocalSize(lclP->V,&nlocal); CHKERRQ(ierr);
  ierr = MatCreateLMVM(((PetscObject)tao)->comm,nlocal,lclP->n - lclP->m,&lclP->R); CHKERRQ(ierr);
  ierr = MatLMVMAllocateVectors(lclP->R,lclP->V); CHKERRQ(ierr);
  lclP->rho = 1.0e-4;

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoSolverSolve_LCL"
static PetscErrorCode TaoSolverSolve_LCL(TaoSolver tao)
{
  TAO_LCL *lclP = (TAO_LCL*)tao->data;
  PetscInt iter=0;
  TaoSolverTerminationReason reason = TAO_CONTINUE_ITERATING;
  TaoLineSearchTerminationReason ls_reason = TAOLINESEARCH_CONTINUE_ITERATING;
  PetscReal step=1.0,f, descent,augl,lgn,con2;
  PetscReal cnorm, mnorm;
  
  PetscErrorCode ierr;
  PetscFunctionBegin;
  
  
  
  /* Scatter to U,V */
  ierr = LCLScatter(lclP,tao->solution,lclP->U,lclP->V); CHKERRQ(ierr);
  
  /* Evaluate Function, Gradient, Constraints, and Jacobian */
  ierr = TaoSolverComputeObjectiveAndGradient(tao,tao->solution,&f,tao->gradient); CHKERRQ(ierr);
  ierr = TaoSolverComputeConstraints(tao,tao->solution, tao->constraints); CHKERRQ(ierr);
  ierr = TaoSolverComputeJacobianState(tao,tao->solution, &tao->jacobian_state, &tao->jacobian_state_pre, &lclP->statematflag); CHKERRQ(ierr);
  ierr = TaoSolverComputeJacobianDesign(tao,tao->solution, &tao->jacobian_design, &tao->jacobian_design_pre, &lclP->statematflag); CHKERRQ(ierr);
  
  /* Scatter gradient to GU,GV */
  ierr = LCLScatter(lclP,tao->gradient,lclP->GU,lclP->GV); CHKERRQ(ierr);
  

  /* Evaluate Lagrangian function and gradient */
 
  ierr = LCLComputeLagrangianAndGradient(tao->linesearch,tao->solution,&lgn,lclP->GL,tao); CHKERRQ(ierr);
  ierr = VecNorm(lclP->GL_U,NORM_2,&cnorm); CHKERRQ(ierr);
  ierr = LCLComputeAugmentedLagrangianAndGradient(tao->linesearch,tao->solution,&augl,lclP->GAugL,tao); CHKERRQ(ierr);
  

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
    /*ierr = MatCopy(tao->jacobian_state,lclP->jacobian_state0,SAME_NONZERO_PATTERN); CHKERRQ(ierr);
    if (tao->jacobian_state == tao->jacobian_state_pre) {
      ierr = PetscObjectDereference((PetscObject)lclP->jacobian_state0_pre); CHKERRQ(ierr);
      lclP->jacobian_state0_pre = lclP->jacobian_state0;
      ierr = PetscObjectReference((PetscObject)lclP->jacobian_state0_pre); CHKERRQ(ierr);
      }*/
    


    lclP->aug0 = lclP->aug;
    lclP->lgn0 = lclP->lgn;
    
    /* Given the design variables, we need to project the current iterate
       onto the linearized constraint.  We choose to fix the design variables
       and solve the linear system for the state variables.  The resulting
       point is the Newton direction */
    
    
    /* Solve r = A\con */
    ierr = KSPSetOperators(tao->ksp, tao->jacobian_state, tao->jacobian_state_pre, lclP->statematflag); CHKERRQ(ierr);
    ierr = KSPSolve(tao->ksp,  tao->constraints, lclP->r); CHKERRQ(ierr);
    ierr = VecNorm(lclP->r,NORM_2,&cnorm); CHKERRQ(ierr);
    ierr = VecSet(lclP->s, 0.0); CHKERRQ(ierr);

    /* Make sure the Newton direction is a descent direction for the merit function */
    ierr = MatMultTranspose(tao->jacobian_state,tao->constraints,lclP->WU); CHKERRQ(ierr);
    
    ierr = VecDot(lclP->r,lclP->WU,&descent); CHKERRQ(ierr);
    if (descent <= 0) {
      ierr = PetscInfo1(tao,"Newton direction not descent: %g",descent);
      reason = TAO_DIVERGED_LS_FAILURE;
      tao->reason = TAO_DIVERGED_LS_FAILURE;
      continue;
    }

    /* Check descent for aug. lagrangian */
    ierr = VecDot(lclP->r,lclP->GAugL_U,&descent); CHKERRQ(ierr);
    if (descent <= 0) {
      ierr = PetscInfo1(tao,"Newton direction not descent for augmented Lagrangian: %g",descent);
      ierr = VecDot(tao->constraints,tao->constraints,&con2); CHKERRQ(ierr);
      while (descent <= 0) {
	lclP->rho*=2;
	lclP->aug0 = lclP->lgn0 + 0.5*lclP->rho*con2;
	ierr = MatMultTranspose(tao->jacobian_state,tao->constraints,lclP->GAugL_U0); CHKERRQ(ierr);
	ierr = MatMultTranspose(tao->jacobian_design,tao->constraints,lclP->GAugL_V0); CHKERRQ(ierr);
	ierr = VecDot(lclP->GAugL_U0, lclP->r, &descent); CHKERRQ(ierr);
      }
      ierr = PetscInfo1(tao,"  Increasing penalty parameter to %g",lclP->rho);
    }


    /* We now minimize the augmented Lagrangian along the Newton direction */
    ierr = VecScale(lclP->r,-1.0); CHKERRQ(ierr);
    ierr = LCLGather(lclP, lclP->r,lclP->s,tao->stepdirection);
    ierr = VecScale(lclP->r,-1.0); CHKERRQ(ierr);
    ierr = LCLGather(lclP, lclP->GAugL_U0, lclP->GAugL_V0, lclP->GAugL); CHKERRQ(ierr);
    ierr = LCLGather(lclP, lclP->U0,lclP->V0,lclP->X0); CHKERRQ(ierr);
    
    lclP->recompute_jacobian_flag = PETSC_TRUE;

    ierr = TaoLineSearchSetInitialStepLength(tao->linesearch,1.0); CHKERRQ(ierr);
    ierr = TaoLineSearchApply(tao->linesearch, tao->solution, &lclP->aug, lclP->GAugL, tao->stepdirection, &step, &ls_reason); CHKERRQ(ierr);
    ierr = LCLScatter(lclP,tao->solution,lclP->U,lclP->V); CHKERRQ(ierr);
    ierr = TaoSolverComputeObjectiveAndGradient(tao,tao->solution,&f,tao->gradient); CHKERRQ(ierr);
    ierr = LCLScatter(lclP,tao->gradient,lclP->GU,lclP->GV); CHKERRQ(ierr);


    /* TODO - check convergence? */

    /* We now minimize the objective function starting from the fraction of
       the Newton point accepted by applying one step of a reduced-space
       method.  The optimization problem is:
       
         minimize f(u+du, v+dv)
         s. t.    A(u0,v0)du + B(u0,v0)du = -alpha g(u0,v0)

       In particular, we have that
       du = -inv(A)*(Bdv + alpha g) */

    /* Compute reduced gradient */
    ierr = VecCopy(lclP->V, lclP->V1); CHKERRQ(ierr);
    /* Store the points 
    ierr = VecCopy(lclP->U, lclP->U1); CHKERRQ(ierr);
    ierr = VecCopy(lclP->GU,lclP->GU1); CHKERRQ(ierr);
    ierr = VecCopy(lclP->GV,lclP->GV1); CHKERRQ(ierr);
    ierr = VecCopy(lclP->GAugL_U,lclP->GAugL_U1); CHKERRQ(ierr);
    ierr = VecCopy(lclP->GAugL_V,lclP->GAugL_V1); CHKERRQ(ierr);
    ierr = VecCopy(lclP->GL_U,lclP->GL_U1); CHKERRQ(ierr);
    ierr = VecCopy(lclP->GL_V,lclP->GL_V1); CHKERRQ(ierr);
    f1 = f;
    lclP->aug1 = lclP->aug;
    lclP->lgn1 = lclP->lgn;
    */


    ierr = TaoSolverComputeJacobianState(tao,lclP->X0,&tao->jacobian_state,&tao->jacobian_state_pre,&lclP->statematflag); CHKERRQ(ierr);
    ierr = TaoSolverComputeJacobianDesign(tao,lclP->X0,&tao->jacobian_design,&tao->jacobian_design_pre,&lclP->designmatflag); CHKERRQ(ierr);


    ierr = KSPSolveTranspose(tao->ksp, lclP->GAugL_V, lclP->lamda); CHKERRQ(ierr);
    ierr = MatMultTranspose(tao->jacobian_design,lclP->lamda,lclP->g1); CHKERRQ(ierr);
    ierr = VecAXPY(lclP->g1,-1.0,lclP->GAugL_V); CHKERRQ(ierr);

    
    /* Compute the limited-memory quasi-newton direction */
    if (iter > 0) {
      ierr = MatLMVMSolve(lclP->R,lclP->g1,lclP->s); CHKERRQ(ierr);
    } else {
      ierr = VecCopy(lclP->g1,lclP->s); CHKERRQ(ierr);
    }
    ierr = VecScale(lclP->g1,-1.0); CHKERRQ(ierr);

    /* Recover the full space direction */
    
    ierr = MatMult(tao->jacobian_design,lclP->s,lclP->WV); CHKERRQ(ierr);
    ierr = KSPSetOperators(tao->ksp,tao->jacobian_state,tao->jacobian_state_pre,lclP->statematflag); CHKERRQ(ierr);
    ierr = KSPSolveTranspose(tao->ksp,lclP->WV,lclP->r); CHKERRQ(ierr);


    /* We now minimize the augmented Lagrangian along the direction -r,s */
    ierr = VecScale(lclP->r, -1.0); CHKERRQ(ierr);
    ierr = LCLGather(lclP,lclP->r,lclP->s,tao->stepdirection); CHKERRQ(ierr);
    ierr = VecScale(lclP->r, -1.0); CHKERRQ(ierr);
    lclP->recompute_jacobian_flag = PETSC_TRUE;

    ierr = TaoLineSearchSetInitialStepLength(tao->linesearch,1.0); CHKERRQ(ierr);
    ierr = TaoLineSearchApply(tao->linesearch, tao->solution, &lclP->aug, lclP->GAugL, tao->stepdirection,&step,&ls_reason); CHKERRQ(ierr);
    ierr = LCLScatter(lclP,tao->solution,lclP->U,lclP->V); CHKERRQ(ierr);
    ierr = LCLScatter(lclP,lclP->GL,lclP->GL_U,lclP->GL_V); CHKERRQ(ierr);
    ierr = LCLScatter(lclP,lclP->GAugL,lclP->GAugL_U,lclP->GAugL_V); CHKERRQ(ierr);
    ierr = TaoSolverComputeObjectiveAndGradient(tao,tao->solution,&f,tao->gradient); CHKERRQ(ierr);
    ierr = LCLScatter(lclP,tao->gradient,lclP->GU,lclP->GV); CHKERRQ(ierr);

    /* TODO - check convergence? */
    
    /* Compute the reduced gradient at the new point */

    ierr = TaoSolverComputeJacobianState(tao,lclP->X0,&tao->jacobian_state,&tao->jacobian_state_pre,&lclP->statematflag); CHKERRQ(ierr);
    ierr = TaoSolverComputeJacobianDesign(tao,lclP->X0,&tao->jacobian_design,&tao->jacobian_design_pre,&lclP->designmatflag); CHKERRQ(ierr);

    ierr = KSPSolveTranspose(tao->ksp, lclP->GAugL_V, lclP->lamda); CHKERRQ(ierr);
    ierr = MatMultTranspose(tao->jacobian_design,lclP->lamda,lclP->g2); CHKERRQ(ierr);
    ierr = VecAXPY(lclP->g2,-1.0,lclP->GAugL_V); CHKERRQ(ierr);


    /* Update the quasi-newton approximation */
    ierr = MatLMVMSetPrev(lclP->R,lclP->V1,lclP->g1);
    ierr = MatLMVMUpdate(lclP->R,lclP->V,lclP->g2); CHKERRQ(ierr);
    
    
    
    /* Evaluate Function, Gradient, Constraints, and Jacobian */
    ierr = TaoSolverComputeObjectiveAndGradient(tao,tao->solution,&f,tao->gradient); CHKERRQ(ierr);
    ierr = LCLScatter(lclP,tao->solution,lclP->U,lclP->V); CHKERRQ(ierr);
    ierr = LCLScatter(lclP,tao->gradient,lclP->GU,lclP->GV); CHKERRQ(ierr);

    ierr = TaoSolverComputeConstraints(tao,tao->solution, tao->constraints); CHKERRQ(ierr);
    ierr = TaoSolverComputeJacobianState(tao,tao->solution, &tao->jacobian_state, &tao->jacobian_state_pre, &lclP->statematflag); CHKERRQ(ierr);
    ierr = TaoSolverComputeJacobianDesign(tao,tao->solution, &tao->jacobian_design, &tao->jacobian_design_pre, &lclP->designmatflag); CHKERRQ(ierr);
    ierr = LCLComputeAugmentedLagrangianAndGradient(tao->linesearch,tao->solution,&augl,lclP->GAugL,tao); CHKERRQ(ierr);

    
    ierr = VecNorm(lclP->GAugL, NORM_2, &mnorm); CHKERRQ(ierr);

    /* Evaluate constraint norm */
    ierr = VecNorm(tao->constraints, NORM_2, &cnorm); CHKERRQ(ierr);
  
    /* Monitor convergence */
    iter++;
    ierr = TaoSolverMonitor(tao, iter,f,mnorm,cnorm,step,&reason); CHKERRQ(ierr);
    
  }

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

  tao->max_its=200;
  tao->fatol=1e-4;
  tao->frtol=1e-4;
  tao->gatol=1e-4;
  tao->grtol=1e-4;
  

  //lclP->subset_type=LCL_SUBSETes_SUBMAT;

  ierr = TaoLineSearchCreate(((PetscObject)tao)->comm, &tao->linesearch); CHKERRQ(ierr);
  ierr = TaoLineSearchSetType(tao->linesearch, morethuente_type); CHKERRQ(ierr);

  ierr = TaoLineSearchUseTaoSolverRoutines(tao->linesearch,tao); CHKERRQ(ierr);

    //SetObjectiveAndGradient(tao->linesearch,LCLObjectiveAndGradient, tao); CHKERRQ(ierr);


  ierr = KSPCreate(((PetscObject)tao)->comm,&tao->ksp); CHKERRQ(ierr);

  ierr = KSPSetFromOptions(tao->ksp); CHKERRQ(ierr);
  
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
    ierr = TaoSolverComputeConstraints(tao,X, tao->constraints); CHKERRQ(ierr);
    ierr = TaoSolverComputeJacobianState(tao,X, &tao->jacobian_state, &tao->jacobian_state_pre, &lclP->statematflag); CHKERRQ(ierr);
    ierr = TaoSolverComputeJacobianDesign(tao,X, &tao->jacobian_design, &tao->jacobian_design_pre, &lclP->designmatflag); CHKERRQ(ierr);
  }
  ierr = KSPSetOperators(tao->ksp, tao->jacobian_state, tao->jacobian_state_pre, lclP->statematflag); CHKERRQ(ierr);
  ierr = KSPSolveTranspose(tao->ksp, lclP->GU,  lclP->lamda); CHKERRQ(ierr);
  ierr = VecDot(lclP->lamda, tao->constraints, &cdotl); CHKERRQ(ierr);
  lclP->lgn = *f - cdotl;
  
  /* Gradient of Lagrangian GL = G - J' * lamda */
  /*      WU = A' * WL
          WV = B' * WL */
  ierr = MatMultTranspose(tao->jacobian_state,lclP->lamda,lclP->GL_U); CHKERRQ(ierr);
  ierr = MatMultTranspose(tao->jacobian_design,lclP->lamda,lclP->GL_V); CHKERRQ(ierr);
  ierr = VecScale(lclP->GL_U,-1.0); CHKERRQ(ierr);
  ierr = VecScale(lclP->GL_V,-1.0); CHKERRQ(ierr);
  ierr = VecAXPY(lclP->GL_U,1.0,lclP->GU); CHKERRQ(ierr);
  ierr = VecAXPY(lclP->GL_V,1.0,lclP->GV); CHKERRQ(ierr);
  ierr = LCLGather(lclP,lclP->GL_U,lclP->GL_V,G); CHKERRQ(ierr);

  
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
