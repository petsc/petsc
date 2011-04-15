#include "rsqn.h"
#include "src/matrix/lmvmmat.h"
#include "src/matrix/submatfree.h"
/*static const char *RSQN_SUBSET[64] = {
  "submat","mask","matrixfree"
  };*/

#define RSQN_SUBSET_SUBMAT 0
#define RSQN_SUBSET_MASK 1
#define RSQN_SUBSET_MATRIXFREE 2
#define RSQN_SUBSET_TYPES 3

static PetscErrorCode RSQNObjectiveAndGradient(TaoLineSearch,Vec,PetscReal*,Vec,void*);

#undef __FUNCT__
#define __FUNCT__ "TaoSolverRSQNSetStateIS"
PetscErrorCode TaoSolverRSQNSetStateIS(TaoSolver tao, IS is)
{
  TAO_RSQN *rsqnP = (TAO_RSQN*)tao->data;
  const char *type;
  PetscErrorCode ierr;
  PetscBool isrsqn;
  ierr = PetscObjectGetType((PetscObject)tao,&type); CHKERRQ(ierr);
  ierr = PetscStrcmp(type,"tao_rsqn",&isrsqn); CHKERRQ(ierr);
  if (!isrsqn) {
    ierr = PetscInfo(tao,"Ignored for non-RSQN solvers"); CHKERRQ(ierr);
  }
  if (rsqnP->UIS) {
    ierr = PetscObjectDereference((PetscObject)rsqnP->UIS); CHKERRQ(ierr);
  }
  rsqnP->UIS = is;
  if (is) {
    ierr = PetscObjectReference((PetscObject)is); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoSolverDestroy_RSQN"
static PetscErrorCode TaoSolverDestroy_RSQN(TaoSolver tao)
{
  TAO_RSQN *rsqnP = (TAO_RSQN*)tao->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (tao->setupcalled) {
    ierr = VecDestroy(rsqnP->LM); CHKERRQ(ierr);
    ierr = VecDestroy(rsqnP->WL); CHKERRQ(ierr);
    ierr = VecDestroy(rsqnP->W); CHKERRQ(ierr);
    ierr = VecDestroy(rsqnP->GM); CHKERRQ(ierr);
    ierr = VecDestroy(rsqnP->GL); CHKERRQ(ierr);
    ierr = VecDestroy(rsqnP->Gr); CHKERRQ(ierr);
    ierr = VecDestroy(rsqnP->U); CHKERRQ(ierr);
    ierr = VecDestroy(rsqnP->V); CHKERRQ(ierr);
    ierr = VecDestroy(rsqnP->GU); CHKERRQ(ierr);
    ierr = VecDestroy(rsqnP->GV); CHKERRQ(ierr);
    ierr = VecDestroy(rsqnP->DU); CHKERRQ(ierr);
    ierr = VecDestroy(rsqnP->DV); CHKERRQ(ierr);
    ierr = VecDestroy(rsqnP->WU); CHKERRQ(ierr);
    ierr = VecDestroy(rsqnP->WV); CHKERRQ(ierr);

    ierr = MatDestroy(rsqnP->M); CHKERRQ(ierr);
    ierr = ISDestroy(rsqnP->UIS); CHKERRQ(ierr);
    ierr = ISDestroy(rsqnP->UID); CHKERRQ(ierr);
    ierr = ISDestroy(rsqnP->UIM); CHKERRQ(ierr);
  }
  ierr = PetscFree(tao->data);
  tao->data = PETSC_NULL;

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoSolverSetFromOptions_RSQN"
static PetscErrorCode TaoSolverSetFromOptions_RSQN(TaoSolver tao)
{
  /*TAO_RSQN *rsqnP = (TAO_RSQN*)tao->data;*/
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = TaoLineSearchSetFromOptions(tao->linesearch); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoSolverView_RSQN"
static PetscErrorCode TaoSolverView_RSQN(TaoSolver tao, PetscViewer viewer)
{
  /*
  TAO_RSQN *rsqnP = (TAO_RSQN*)tao->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscFunctionReturn(0);
  */
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "TaoSolverSetup_RSQN"
static PetscErrorCode TaoSolverSetup_RSQN(TaoSolver tao)
{
  TAO_RSQN *rsqnP = (TAO_RSQN*)tao->data;
  PetscInt lo, hi, nlocal;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  /* Check for state IS */
  if (!rsqnP->UIS) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONGSTATE,"RSQN Solver requires an initial state index set -- use TaoSolverRSQNSetStateIS()");
  }
  ierr = VecDuplicate(tao->solution, &tao->gradient); CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution, &tao->stepdirection); CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution, &rsqnP->W); CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution, &rsqnP->GM); CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution, &rsqnP->GL); CHKERRQ(ierr);
  ierr = VecDuplicate(tao->constraints, &rsqnP->LM); CHKERRQ(ierr);
  ierr = VecDuplicate(tao->constraints, &rsqnP->WL); CHKERRQ(ierr);
  ierr = VecSet(rsqnP->LM,0.0); CHKERRQ(ierr);

  ierr = VecGetSize(tao->solution, &rsqnP->n); CHKERRQ(ierr);
  ierr = VecGetSize(tao->constraints, &rsqnP->m); CHKERRQ(ierr);

  ierr = VecGetOwnershipRange(tao->solution,&lo,&hi); CHKERRQ(ierr);
  ierr = ISComplement(rsqnP->UIS,lo,hi,&rsqnP->UID); CHKERRQ(ierr);

  ierr = VecGetOwnershipRange(tao->constraints,&lo,&hi); CHKERRQ(ierr);
  ierr = ISCreateStride(((PetscObject)tao)->comm,hi-lo,lo,1,&rsqnP->UIM);

  if (rsqnP->subset_type == RSQN_SUBSET_SUBMAT) {
    IS is_state, is_design;
    ierr = VecCreate(((PetscObject)tao)->comm,&rsqnP->U); CHKERRQ(ierr);
    ierr = VecCreate(((PetscObject)tao)->comm,&rsqnP->V); CHKERRQ(ierr);
    ierr = VecSetSizes(rsqnP->U,PETSC_DECIDE,rsqnP->m); CHKERRQ(ierr);
    ierr = VecSetSizes(rsqnP->V,PETSC_DECIDE,rsqnP->n-rsqnP->m); CHKERRQ(ierr);
    ierr = VecSetType(rsqnP->U,((PetscObject)(tao->solution))->type_name); CHKERRQ(ierr);
    ierr = VecSetType(rsqnP->V,((PetscObject)(tao->solution))->type_name); CHKERRQ(ierr);
    ierr = VecSetFromOptions(rsqnP->U); CHKERRQ(ierr);
    ierr = VecSetFromOptions(rsqnP->V); CHKERRQ(ierr);
    ierr = VecDuplicate(rsqnP->U,&rsqnP->DU); CHKERRQ(ierr);
    ierr = VecDuplicate(rsqnP->U,&rsqnP->GU); CHKERRQ(ierr);
    ierr = VecDuplicate(rsqnP->U,&rsqnP->WU); CHKERRQ(ierr);
    ierr = VecDuplicate(rsqnP->V,&rsqnP->DV); CHKERRQ(ierr);
    ierr = VecDuplicate(rsqnP->V,&rsqnP->GV); CHKERRQ(ierr);
    ierr = VecDuplicate(rsqnP->V,&rsqnP->Gr); CHKERRQ(ierr);
    ierr = VecDuplicate(rsqnP->V,&rsqnP->WV); CHKERRQ(ierr);

    /* create scatters for state, design subvecs */
    ierr = VecGetOwnershipRange(rsqnP->U,&lo,&hi); CHKERRQ(ierr);
    ierr = ISCreateStride(((PetscObject)rsqnP->U)->comm,hi-lo,lo,1,&is_state); CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(rsqnP->V,&lo,&hi); CHKERRQ(ierr);
    ierr = ISCreateStride(((PetscObject)rsqnP->V)->comm,hi-lo,lo,1,&is_design); CHKERRQ(ierr);
    ierr = VecScatterCreate(tao->solution,rsqnP->UIS,rsqnP->U,is_state,&rsqnP->state_scatter); CHKERRQ(ierr);
    ierr = VecScatterCreate(tao->solution,rsqnP->UID,rsqnP->V,is_design,&rsqnP->design_scatter); CHKERRQ(ierr);
    ierr = ISDestroy(is_state); CHKERRQ(ierr);
    ierr = ISDestroy(is_design); CHKERRQ(ierr);
  } else if (rsqnP->subset_type == RSQN_SUBSET_MASK) {
    ierr = VecDuplicate(tao->solution, &rsqnP->U); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->solution, &rsqnP->V); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->solution, &rsqnP->DU); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->solution, &rsqnP->DV); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->solution, &rsqnP->GU); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->solution, &rsqnP->GV); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->solution, &rsqnP->Gr); CHKERRQ(ierr);
  }

  ierr = VecGetLocalSize(tao->solution,&nlocal); CHKERRQ(ierr);
  ierr = MatCreateLMVM(((PetscObject)tao)->comm,nlocal,rsqnP->n,&rsqnP->M); CHKERRQ(ierr);
  rsqnP->rho = 0.1;

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoSolverSolve_RSQN"
static PetscErrorCode TaoSolverSolve_RSQN(TaoSolver tao)
{
  TAO_RSQN *rsqnP = (TAO_RSQN*)tao->data;
  PetscInt iter=0;
  TaoSolverTerminationReason reason = TAO_CONTINUE_ITERATING;
  TaoLineSearchTerminationReason ls_reason = TAOLINESEARCH_CONTINUE_ITERATING;
  PetscReal step=1.0,f,fm;
  PetscReal cnorm, mnorm;
  
  PetscErrorCode ierr;
  PetscFunctionBegin;
  /* Estimate initial X */
  ierr = VecSet(tao->solution,1.0); CHKERRQ(ierr);
  
  /* Scatter to U,V */
  ierr = VecScatterBegin(rsqnP->state_scatter, tao->solution, rsqnP->U, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(rsqnP->state_scatter, tao->solution, rsqnP->U, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterBegin(rsqnP->design_scatter, tao->solution, rsqnP->V, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(rsqnP->design_scatter, tao->solution, rsqnP->V, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  
  /* Evaluate Function, Gradient, Constraints, and Jacobian */
  ierr = TaoSolverComputeObjectiveAndGradient(tao,tao->solution,&f,tao->gradient); CHKERRQ(ierr);
  ierr = TaoSolverComputeConstraints(tao,tao->solution, tao->constraints); CHKERRQ(ierr);
  ierr = TaoSolverComputeJacobianState(tao,tao->solution, &tao->jacobian_state, &tao->jacobian_state_pre, &rsqnP->statematflag); CHKERRQ(ierr);
  ierr = TaoSolverComputeJacobianDesign(tao,tao->solution, &tao->jacobian_design, &tao->jacobian_design_pre, &rsqnP->statematflag); CHKERRQ(ierr);
  
  /* Scatter gradient to GU,GV */
  ierr = VecScatterBegin(rsqnP->state_scatter, tao->gradient, rsqnP->GU, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(rsqnP->state_scatter, tao->gradient, rsqnP->GU, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterBegin(rsqnP->design_scatter, tao->gradient, rsqnP->GV, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(rsqnP->design_scatter, tao->gradient, rsqnP->GV, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  
  /* Scatter jacobian to JU, JV */
/*  ierr = MatGetSubMatrix(tao->jacobian, rsqnP->UIM,rsqnP->UIS,MAT_INITIAL_MATRIX,&rsqnP->JU); CHKERRQ(ierr);
  ierr = MatGetSubMatrix(tao->jacobian, rsqnP->UIM, rsqnP->UID,MAT_INITIAL_MATRIX,&rsqnP->JV); CHKERRQ(ierr);
  if (tao->jacobian != tao->jacobian_pre) {
    ierr = MatGetSubMatrix(tao->jacobian_pre, rsqnP->UIM, rsqnP->UIS, MAT_INITIAL_MATRIX,&rsqnP->Jpre_U); CHKERRQ(ierr);
  } else {
    rsqnP->Jpre_U = rsqnP->JU;
    ierr = PetscObjectReference((PetscObject)rsqnP->Jpre_U); CHKERRQ(ierr);
    }*/
  CHKMEMQ;
  void *ptr;
  ierr = MatShellGetContext(tao->jacobian_state,&ptr); CHKERRQ(ierr); //for debugging

  ierr = KSPSetOperators(tao->ksp, tao->jacobian_state, tao->jacobian_state_pre, rsqnP->statematflag); CHKERRQ(ierr);
  ierr = KSPSolveTranspose(tao->ksp,  rsqnP->GU, rsqnP->LM); CHKERRQ(ierr);
  CHKMEMQ;
  /* Evaluate Lagrangian gradient norm */

  ierr = MatMultTranspose(tao->jacobian_state,rsqnP->LM, rsqnP->WU); CHKERRQ(ierr);
  ierr = MatMultTranspose(tao->jacobian_design,rsqnP->LM, rsqnP->WV); CHKERRQ(ierr);

  CHKMEMQ;
  ierr = VecScatterBegin(rsqnP->state_scatter, rsqnP->WU, rsqnP->GL, INSERT_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
  ierr = VecScatterEnd(rsqnP->state_scatter, rsqnP->WU, rsqnP->GL, INSERT_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
  ierr = VecScatterBegin(rsqnP->design_scatter, rsqnP->WV, rsqnP->GL, INSERT_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
  ierr = VecScatterEnd(rsqnP->design_scatter, rsqnP->WV, rsqnP->GL, INSERT_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
  
  ierr = VecAYPX(rsqnP->GL, -1.0, tao->gradient); CHKERRQ(ierr);
  ierr = VecNorm(rsqnP->GL, NORM_2, &mnorm); CHKERRQ(ierr);
  
  /* Evaluate constraint norm */
  ierr = VecNorm(tao->constraints, NORM_2, &cnorm); CHKERRQ(ierr);
  
  /* Monitor convergence */
  ierr = TaoSolverMonitor(tao, iter,f,mnorm,cnorm,step,&reason); CHKERRQ(ierr);

  while (reason == TAO_CONTINUE_ITERATING) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"========================ITER %3d========================\n", iter);
    
    /* update reduced hessian */
    ierr = MatLMVMUpdate(rsqnP->M, rsqnP->V, rsqnP->GV); CHKERRQ(ierr);
    CHKMEMQ;
    /* compute reduced gradient */
    ierr = MatMultTranspose(tao->jacobian_design, rsqnP->LM, rsqnP->Gr); CHKERRQ(ierr);
    ierr = VecAYPX(rsqnP->Gr, -1.0, rsqnP->GV); CHKERRQ(ierr);
    
    /* Compute DV */
    ierr = MatLMVMSolve(rsqnP->M, rsqnP->Gr, rsqnP->DV); CHKERRQ(ierr);
    ierr = VecScale(rsqnP->DV, -1.0); CHKERRQ(ierr);

    CHKMEMQ;
    /* Compute DU */
    ierr = MatMult(tao->jacobian_design, rsqnP->DV, rsqnP->WU); CHKERRQ(ierr);
    ierr = VecAYPX(rsqnP->WU, 1.0, tao->constraints); CHKERRQ(ierr);
    ierr = KSPSetOperators(tao->ksp, tao->jacobian_state, tao->jacobian_state_pre, rsqnP->statematflag); CHKERRQ(ierr);
    ierr = KSPSolve(tao->ksp,  rsqnP->WU, rsqnP->DU); CHKERRQ(ierr);
    ierr = VecScale(rsqnP->DU, -1.0); CHKERRQ(ierr);

    CHKMEMQ;
    /* Assemble Big D */
    ierr = VecScatterBegin(rsqnP->state_scatter, rsqnP->DU, tao->stepdirection, INSERT_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
    ierr = VecScatterEnd(rsqnP->state_scatter, rsqnP->DU, tao->stepdirection, INSERT_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
    ierr = VecScatterBegin(rsqnP->design_scatter, rsqnP->DV, tao->stepdirection, INSERT_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
    ierr = VecScatterEnd(rsqnP->design_scatter, rsqnP->DV, tao->stepdirection, INSERT_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);

    /* Perform Line Search */
    ierr = TaoLineSearchComputeObjectiveAndGradient(tao->linesearch,tao->solution,&fm,rsqnP->GM); CHKERRQ(ierr);
    ierr = TaoLineSearchSetInitialStepLength(tao->linesearch,1.0);
    ierr = TaoLineSearchApply(tao->linesearch, tao->solution, &fm, rsqnP->GM, tao->stepdirection,&step, &ls_reason); CHKERRQ(ierr);
   
    while (ls_reason < 0) {
      ierr = PetscInfo1(tao,"line search failed, increasing rho to %f",rsqnP->rho); CHKERRQ(ierr);
      
      if (rsqnP->rho >= 1000) {
	ierr = PetscInfo(tao,"rho > 1000");
	tao->reason = TAO_DIVERGED_LS_FAILURE; 
	PetscFunctionReturn(0);
      }
      rsqnP->rho *= 2;
      /* Redo Line Search */
      ierr = TaoLineSearchComputeObjectiveAndGradient(tao->linesearch,tao->solution,&fm,rsqnP->GM); CHKERRQ(ierr);
      ierr = TaoLineSearchSetInitialStepLength(tao->linesearch,1.0);
      ierr = TaoLineSearchApply(tao->linesearch, tao->solution, &fm, rsqnP->GM, tao->stepdirection,&step, &ls_reason); CHKERRQ(ierr);
    }
      
	
    /* TODO check linesearch results, double rho and reapply if necessary */
    
    /* Scatter X to U,V */
    ierr = VecScatterBegin(rsqnP->state_scatter, tao->solution, rsqnP->U, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
    ierr = VecScatterEnd(rsqnP->state_scatter, tao->solution, rsqnP->U, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
    ierr = VecScatterBegin(rsqnP->design_scatter, tao->solution, rsqnP->V, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
    ierr = VecScatterEnd(rsqnP->design_scatter, tao->solution, rsqnP->V, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);

    /* Evaluate Function, Gradient, Constraints, and Jacobian */
    ierr = TaoSolverComputeObjectiveAndGradient(tao,tao->solution,&f,tao->gradient); CHKERRQ(ierr);
    ierr = TaoSolverComputeConstraints(tao,tao->solution, tao->constraints); CHKERRQ(ierr);
    ierr = TaoSolverComputeJacobianState(tao,tao->solution, &tao->jacobian_state, &tao->jacobian_state_pre, &rsqnP->statematflag); CHKERRQ(ierr);
    ierr = TaoSolverComputeJacobianDesign(tao,tao->solution, &tao->jacobian_design, &tao->jacobian_design_pre, &rsqnP->designmatflag); CHKERRQ(ierr);
  
    /* Scatter gradient to GU,GV */
    ierr = VecScatterBegin(rsqnP->state_scatter, tao->gradient, rsqnP->GU, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
    ierr = VecScatterEnd(rsqnP->state_scatter, tao->gradient, rsqnP->GU, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
    ierr = VecScatterBegin(rsqnP->design_scatter, tao->gradient, rsqnP->GV, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
    ierr = VecScatterEnd(rsqnP->design_scatter, tao->gradient, rsqnP->GV, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  
    /* Scatter jacobian to JU, JV */
/*
    ierr = MatGetSubMatrix(tao->jacobian, rsqnP->UIM,rsqnP->UIS,MAT_REUSE_MATRIX,&rsqnP->JU); CHKERRQ(ierr);
    ierr = MatGetSubMatrix(tao->jacobian, rsqnP->UIM, rsqnP->UID,MAT_REUSE_MATRIX,&rsqnP->JV); CHKERRQ(ierr);
    if (tao->jacobian != tao->jacobian_pre) {
      ierr = MatGetSubMatrix(tao->jacobian_pre, rsqnP->UIM, rsqnP->UIS, MAT_REUSE_MATRIX,&rsqnP->Jpre_U); CHKERRQ(ierr);
    } else {
      ierr = PetscObjectReference((PetscObject)rsqnP->Jpre_U); CHKERRQ(ierr);
    }
*/
    ierr = KSPSetOperators(tao->ksp, tao->jacobian_state, tao->jacobian_state_pre, rsqnP->statematflag); CHKERRQ(ierr);
    ierr = KSPSolveTranspose(tao->ksp,  rsqnP->GU, rsqnP->LM); CHKERRQ(ierr);

    /* Evaluate Lagrangian gradient norm */

    ierr = MatMultTranspose(tao->jacobian_state,rsqnP->LM, rsqnP->WU); CHKERRQ(ierr);
    ierr = MatMultTranspose(tao->jacobian_design,rsqnP->LM, rsqnP->WV); CHKERRQ(ierr);

    ierr = VecScatterBegin(rsqnP->state_scatter, rsqnP->WU, rsqnP->GL, INSERT_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
    ierr = VecScatterEnd(rsqnP->state_scatter, rsqnP->WU, rsqnP->GL, INSERT_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
    ierr = VecScatterBegin(rsqnP->design_scatter, rsqnP->WV, rsqnP->GL, INSERT_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
    ierr = VecScatterEnd(rsqnP->design_scatter, rsqnP->WV, rsqnP->GL, INSERT_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
    
    ierr = VecAYPX(rsqnP->GL, -1.0, tao->gradient); CHKERRQ(ierr);
    ierr = VecNorm(rsqnP->GL, NORM_2, &mnorm); CHKERRQ(ierr);
  
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
#define __FUNCT__ "TaoSolverCreate_RSQN"
PetscErrorCode TaoSolverCreate_RSQN(TaoSolver tao)
{
  TAO_RSQN *rsqnP;
  PetscErrorCode ierr;
  const char *morethuente_type = TAOLINESEARCH_MT;
  PetscFunctionBegin;
  
  tao->ops->setup = TaoSolverSetup_RSQN;
  tao->ops->solve = TaoSolverSolve_RSQN;
  tao->ops->view = TaoSolverView_RSQN;
  tao->ops->setfromoptions = TaoSolverSetFromOptions_RSQN;
  tao->ops->destroy = TaoSolverDestroy_RSQN;
  

  ierr = PetscNewLog(tao,TAO_RSQN,&rsqnP); CHKERRQ(ierr);
  tao->data = (void*)rsqnP;

  tao->max_its=200;
  tao->fatol=1e-4;
  tao->frtol=1e-4;
  tao->gatol=1e-4;
  tao->grtol=1e-4;
  

  rsqnP->subset_type=RSQN_SUBSET_SUBMAT;

  ierr = TaoLineSearchCreate(((PetscObject)tao)->comm, &tao->linesearch); CHKERRQ(ierr);
  ierr = TaoLineSearchSetType(tao->linesearch, morethuente_type); CHKERRQ(ierr);

  ierr = TaoLineSearchSetObjectiveAndGradient(tao->linesearch,RSQNObjectiveAndGradient, tao); CHKERRQ(ierr);


  ierr = KSPCreate(((PetscObject)tao)->comm,&tao->ksp); CHKERRQ(ierr);

  ierr = KSPSetFromOptions(tao->ksp); CHKERRQ(ierr);
  
  PetscFunctionReturn(0);

}
EXTERN_C_END


#undef __FUNCT__
#define __FUNCT__ "RSQNObjectiveAndGradient"
static PetscErrorCode RSQNObjectiveAndGradient(TaoLineSearch ls, Vec X, PetscReal *f, Vec G, void *ptr)
{
  TaoSolver tao = (TaoSolver)ptr;
  TAO_RSQN *rsqnP = (TAO_RSQN*)tao->data;
  PetscReal lmh,hnorm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TaoSolverComputeObjectiveAndGradient(tao,X,f,G); CHKERRQ(ierr);
  ierr = TaoSolverComputeConstraints(tao,X,tao->constraints); CHKERRQ(ierr);
  ierr = TaoSolverComputeJacobianState(tao,X,&tao->jacobian_state,&tao->jacobian_state_pre,&rsqnP->statematflag); CHKERRQ(ierr);
  ierr = TaoSolverComputeJacobianDesign(tao,X,&tao->jacobian_design,&tao->jacobian_design_pre,&rsqnP->designmatflag); CHKERRQ(ierr);
  ierr = VecDot(tao->constraints,rsqnP->LM,&lmh); CHKERRQ(ierr);
  ierr = VecDot(tao->constraints,tao->constraints,&hnorm); CHKERRQ(ierr);
  *f += 0.5*rsqnP->rho * hnorm - lmh;
  
  /* WL = -lm + rho*h */
  ierr = VecCopy(rsqnP->LM, rsqnP->WL); CHKERRQ(ierr);
  ierr = VecScale(rsqnP->WL, -1.0); CHKERRQ(ierr);
  ierr = VecAXPY(rsqnP->WL, rsqnP->rho, tao->constraints); CHKERRQ(ierr);
  
  /* GM = J' WL */
  /*      WU = A' * WL
          WV = B' * WL */
  ierr = MatMultTranspose(tao->jacobian_state,rsqnP->WL,rsqnP->WU); CHKERRQ(ierr);
  ierr = MatMultTranspose(tao->jacobian_design,rsqnP->WL,rsqnP->WV); CHKERRQ(ierr);
  
  ierr = VecScatterBegin(rsqnP->state_scatter, rsqnP->WU, rsqnP->W, INSERT_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
  ierr = VecScatterEnd(rsqnP->state_scatter, rsqnP->WU, rsqnP->W, INSERT_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
  ierr = VecScatterBegin(rsqnP->design_scatter, rsqnP->WV, rsqnP->W, INSERT_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
  ierr = VecScatterEnd(rsqnP->design_scatter, rsqnP->WV, rsqnP->W, INSERT_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
  

  /* G = G + GM */
  ierr = VecAXPY(G,1.0,rsqnP->W); CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}
