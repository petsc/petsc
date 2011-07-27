#include "sqpcon.h"
#include "src/matrix/lmvmmat.h"
#include "src/matrix/approxmat.h"
#include "src/matrix/submatfree.h"
//static PetscErrorCode SQPCONObjectiveAndGradient(TaoLineSearch,Vec,PetscReal*,Vec,void*);


#undef __FUNCT__
#define __FUNCT__ "TaoSolverSQPCONSetStateIS"
PetscErrorCode TaoSolverSQPCONSetStateIS(TaoSolver tao, IS is)
{
  TAO_SQPCON *sqpconP = (TAO_SQPCON*)tao->data;
  const char *type;
  PetscErrorCode ierr;
  PetscBool issqpcon;
  ierr = PetscObjectGetType((PetscObject)tao,&type); CHKERRQ(ierr);
  ierr = PetscStrcmp(type,"tao_sqpcon",&issqpcon); CHKERRQ(ierr);
  if (!issqpcon) {
    ierr = PetscInfo(tao,"Ignored for non-pde_constrained solvers"); CHKERRQ(ierr);
  }
  if (sqpconP->UIS) {
    ierr = PetscObjectDereference((PetscObject)sqpconP->UIS); CHKERRQ(ierr);
  }
  sqpconP->UIS = is;
  if (is) {
    ierr = PetscObjectReference((PetscObject)is); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoSolverDestroy_SQPCON"
static PetscErrorCode TaoSolverDestroy_SQPCON(TaoSolver tao)
{
  TAO_SQPCON *sqpconP = (TAO_SQPCON*)tao->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (tao->setupcalled) {
    ierr = MatDestroy(&sqpconP->Q); CHKERRQ(ierr);
    ierr = MatDestroy(&sqpconP->R); CHKERRQ(ierr);

    ierr = VecDestroy(&sqpconP->LM); CHKERRQ(ierr);
    ierr = VecDestroy(&sqpconP->WL); CHKERRQ(ierr);
    ierr = VecDestroy(&sqpconP->W); CHKERRQ(ierr);
    ierr = VecDestroy(&sqpconP->Xold); CHKERRQ(ierr);
    ierr = VecDestroy(&sqpconP->Gold); CHKERRQ(ierr);
    ierr = VecDestroy(&sqpconP->GL); CHKERRQ(ierr);
    ierr = VecDestroy(&sqpconP->dbar); CHKERRQ(ierr);
    ierr = VecDestroy(&sqpconP->U); CHKERRQ(ierr);
    ierr = VecDestroy(&sqpconP->V); CHKERRQ(ierr);
    ierr = VecDestroy(&sqpconP->GU); CHKERRQ(ierr);
    ierr = VecDestroy(&sqpconP->GV); CHKERRQ(ierr);
    ierr = VecDestroy(&sqpconP->DU); CHKERRQ(ierr);
    ierr = VecDestroy(&sqpconP->DV); CHKERRQ(ierr);
    ierr = VecDestroy(&sqpconP->WU); CHKERRQ(ierr);
    ierr = VecDestroy(&sqpconP->WV); CHKERRQ(ierr);

    ierr = VecDestroy(&sqpconP->Tbar); CHKERRQ(ierr);
    ierr = VecDestroy(&sqpconP->aqwac); CHKERRQ(ierr);

    ierr = ISDestroy(&sqpconP->UIS); CHKERRQ(ierr);
    ierr = ISDestroy(&sqpconP->UID); CHKERRQ(ierr);
    ierr = ISDestroy(&sqpconP->UIM); CHKERRQ(ierr);
  }
  ierr = PetscFree(tao->data);
  tao->data = PETSC_NULL;

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoSolverSetFromOptions_SQPCON"
static PetscErrorCode TaoSolverSetFromOptions_SQPCON(TaoSolver tao)
{
  /*TAO_SQPCON *sqpconP = (TAO_SQPCON*)tao->data;*/
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = TaoLineSearchSetFromOptions(tao->linesearch); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoSolverView_SQPCON"
static PetscErrorCode TaoSolverView_SQPCON(TaoSolver tao, PetscViewer viewer)
{
  /*
  TAO_SQPCON *sqpconP = (TAO_SQPCON*)tao->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscFunctionReturn(0);
  */
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "TaoSolverSetup_SQPCON"
static PetscErrorCode TaoSolverSetup_SQPCON(TaoSolver tao)
{
  TAO_SQPCON *sqpconP = (TAO_SQPCON*)tao->data;
  PetscInt lo, hi, nlocal;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  /* Check for state IS */
  if (!sqpconP->UIS) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONGSTATE,"SQPCON Solver requires an initial state index set -- use TaoSolverSQPCONSetStateIS()");
  }
  ierr = VecDuplicate(tao->solution, &tao->gradient); CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution, &tao->stepdirection); CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution, &sqpconP->W); CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution, &sqpconP->Xold); CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution, &sqpconP->Gold); CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution, &sqpconP->GL); CHKERRQ(ierr);

  ierr = VecDuplicate(tao->constraints, &sqpconP->LM); CHKERRQ(ierr);
  ierr = VecDuplicate(tao->constraints, &sqpconP->DL); CHKERRQ(ierr);
  ierr = VecDuplicate(tao->constraints, &sqpconP->WL); CHKERRQ(ierr);

  ierr = VecSet(sqpconP->LM,0.0); CHKERRQ(ierr);

  ierr = VecGetSize(tao->solution, &sqpconP->n); CHKERRQ(ierr);
  ierr = VecGetSize(tao->constraints, &sqpconP->m); CHKERRQ(ierr);

  ierr = VecGetOwnershipRange(tao->solution,&lo,&hi); CHKERRQ(ierr);
  ierr = ISComplement(sqpconP->UIS,lo,hi,&sqpconP->UID); CHKERRQ(ierr);

  ierr = VecGetOwnershipRange(tao->constraints,&lo,&hi); CHKERRQ(ierr);
  ierr = ISCreateStride(((PetscObject)tao)->comm,hi-lo,lo,1,&sqpconP->UIM);

  IS is_state, is_design;
  ierr = VecCreate(((PetscObject)tao)->comm,&sqpconP->U); CHKERRQ(ierr);
  ierr = VecCreate(((PetscObject)tao)->comm,&sqpconP->V); CHKERRQ(ierr);
  ierr = VecSetSizes(sqpconP->U,PETSC_DECIDE,sqpconP->m); CHKERRQ(ierr);
  ierr = VecSetSizes(sqpconP->V,PETSC_DECIDE,sqpconP->n-sqpconP->m); CHKERRQ(ierr);
  ierr = VecSetType(sqpconP->U,((PetscObject)(tao->solution))->type_name); CHKERRQ(ierr);
  ierr = VecSetType(sqpconP->V,((PetscObject)(tao->solution))->type_name); CHKERRQ(ierr);
  ierr = VecSetFromOptions(sqpconP->U); CHKERRQ(ierr);
  ierr = VecSetFromOptions(sqpconP->V); CHKERRQ(ierr);
  ierr = VecDuplicate(sqpconP->U,&sqpconP->DU); CHKERRQ(ierr);
  ierr = VecDuplicate(sqpconP->U,&sqpconP->GU); CHKERRQ(ierr);
  ierr = VecDuplicate(sqpconP->U,&sqpconP->WU); CHKERRQ(ierr);
  ierr = VecDuplicate(sqpconP->U,&sqpconP->Tbar); CHKERRQ(ierr);
  ierr = VecDuplicate(sqpconP->V,&sqpconP->DV); CHKERRQ(ierr);
  ierr = VecDuplicate(sqpconP->V,&sqpconP->GV); CHKERRQ(ierr);
  ierr = VecDuplicate(sqpconP->V,&sqpconP->dbar); CHKERRQ(ierr);
  ierr = VecDuplicate(sqpconP->V,&sqpconP->WV); CHKERRQ(ierr);
  ierr = VecDuplicate(sqpconP->V,&sqpconP->aqwac); CHKERRQ(ierr);

  /* create scatters for state, design subvecs */
  ierr = VecGetOwnershipRange(sqpconP->U,&lo,&hi); CHKERRQ(ierr);
  ierr = ISCreateStride(((PetscObject)sqpconP->U)->comm,hi-lo,lo,1,&is_state); CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(sqpconP->V,&lo,&hi); CHKERRQ(ierr);
  ierr = ISCreateStride(((PetscObject)sqpconP->V)->comm,hi-lo,lo,1,&is_design); CHKERRQ(ierr);
  ierr = VecScatterCreate(tao->solution,sqpconP->UIS,sqpconP->U,is_state,&sqpconP->state_scatter); CHKERRQ(ierr);
  ierr = VecScatterCreate(tao->solution,sqpconP->UID,sqpconP->V,is_design,&sqpconP->design_scatter); CHKERRQ(ierr);
  ierr = ISDestroy(&is_state); CHKERRQ(ierr);
  ierr = ISDestroy(&is_design); CHKERRQ(ierr);

  ierr = VecGetLocalSize(sqpconP->U,&nlocal); CHKERRQ(ierr);
  ierr = MatCreateAPPROX(((PetscObject)tao)->comm,nlocal,sqpconP->m,&sqpconP->Q); CHKERRQ(ierr);
  ierr = VecGetLocalSize(sqpconP->V,&nlocal); CHKERRQ(ierr);
  ierr = MatCreateLMVM(((PetscObject)tao)->comm,nlocal,sqpconP->n - sqpconP->m,&sqpconP->R); CHKERRQ(ierr);
  ierr = MatApproxAllocateVectors(sqpconP->Q,sqpconP->U); CHKERRQ(ierr);
  ierr = MatLMVMAllocateVectors(sqpconP->R,sqpconP->V); CHKERRQ(ierr);
  sqpconP->rho = 0.1;

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoSolverSolve_SQPCON"
static PetscErrorCode TaoSolverSolve_SQPCON(TaoSolver tao)
{
  TAO_SQPCON *sqpconP = (TAO_SQPCON*)tao->data;
  PetscInt iter=0;
  TaoSolverTerminationReason reason = TAO_CONTINUE_ITERATING;
  TaoLineSearchTerminationReason ls_reason = TAOLINESEARCH_CONTINUE_ITERATING;
  PetscReal step=1.0,f,fm, fold;
  PetscReal cnorm, mnorm;
  PetscBool use_update=PETSC_TRUE; // don't update Q if line search failed
  
  PetscErrorCode ierr;
  PetscFunctionBegin;
  
  
  
  /* Scatter to U,V */
  ierr = VecScatterBegin(sqpconP->state_scatter, tao->solution, sqpconP->U, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(sqpconP->state_scatter, tao->solution, sqpconP->U, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterBegin(sqpconP->design_scatter, tao->solution, sqpconP->V, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(sqpconP->design_scatter, tao->solution, sqpconP->V, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  
  /* Evaluate Function, Gradient, Constraints, and Jacobian */
  ierr = TaoSolverComputeObjectiveAndGradient(tao,tao->solution,&f,tao->gradient); CHKERRQ(ierr);
  ierr = TaoSolverComputeConstraints(tao,tao->solution, tao->constraints); CHKERRQ(ierr);
  ierr = TaoSolverComputeJacobianState(tao,tao->solution, &tao->jacobian_state, &tao->jacobian_state_pre, &sqpconP->statematflag); CHKERRQ(ierr);
  ierr = TaoSolverComputeJacobianDesign(tao,tao->solution, &tao->jacobian_design, &tao->jacobian_design_pre, &sqpconP->statematflag); CHKERRQ(ierr);
  
  /* Scatter gradient to GU,GV */
  ierr = VecScatterBegin(sqpconP->state_scatter, tao->gradient, sqpconP->GU, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(sqpconP->state_scatter, tao->gradient, sqpconP->GU, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterBegin(sqpconP->design_scatter, tao->gradient, sqpconP->GV, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(sqpconP->design_scatter, tao->gradient, sqpconP->GV, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecNorm(tao->gradient, NORM_2, &mnorm); CHKERRQ(ierr);
  
  /* Evaluate constraint norm */
  ierr = VecNorm(tao->constraints, NORM_2, &cnorm); CHKERRQ(ierr);
  
  /* Monitor convergence */
  ierr = TaoSolverMonitor(tao, iter,f,mnorm,cnorm,step,&reason); CHKERRQ(ierr);

  while (reason == TAO_CONTINUE_ITERATING) {


    /* Solve tbar = -A\t (t is constraints vector) */
    ierr = KSPSetOperators(tao->ksp, tao->jacobian_state, tao->jacobian_state_pre, sqpconP->statematflag); CHKERRQ(ierr);
    ierr = KSPSolve(tao->ksp,  tao->constraints, sqpconP->Tbar); CHKERRQ(ierr);
    ierr = VecScale(sqpconP->Tbar, -1.0); CHKERRQ(ierr);

    /* aqwac =  A'\(Q*Tbar + c) */
    if (iter > 0) { 
      ierr = MatMult(sqpconP->Q,sqpconP->Tbar,sqpconP->WV); CHKERRQ(ierr);
    } else {
      ierr = VecCopy(sqpconP->Tbar, sqpconP->WV); CHKERRQ(ierr);
    }
    ierr = VecAXPY(sqpconP->WV,1.0,sqpconP->GU); CHKERRQ(ierr);
    
    ierr = KSPSolve(tao->ksp, sqpconP->WV, sqpconP->aqwac); CHKERRQ(ierr);


    /* Reduced Gradient dbar = d -  B^t * aqwac */
    ierr = MatMultTranspose(tao->jacobian_design,sqpconP->aqwac, sqpconP->dbar); CHKERRQ(ierr);
    ierr = VecScale(sqpconP->dbar, -1.0); CHKERRQ(ierr);
    ierr = VecAXPY(sqpconP->dbar,1.0,sqpconP->GV); CHKERRQ(ierr);

    /* update reduced hessian */
    ierr = MatLMVMUpdate(sqpconP->R, sqpconP->V, sqpconP->dbar); CHKERRQ(ierr);

    /* Solve R*dv = -dbar using approx. hessian */
    ierr = MatLMVMSolve(sqpconP->R, sqpconP->dbar, sqpconP->DV); CHKERRQ(ierr);
    ierr = VecScale(sqpconP->DV, -1.0); CHKERRQ(ierr);

    /* Backsolve for u =  A\(g - B*dv)  = tbar - A\(B*dv)*/
    ierr = MatMult(tao->jacobian_design, sqpconP->DV, sqpconP->WL); CHKERRQ(ierr);
    ierr = KSPSolve(tao->ksp, sqpconP->WL, sqpconP->DU); CHKERRQ(ierr);
    ierr = VecScale(sqpconP->DU, -1.0); CHKERRQ(ierr); 
    ierr = VecAXPY(sqpconP->DU, 1.0, sqpconP->Tbar); CHKERRQ(ierr);


    
    /* Assemble Big D */
    ierr = VecScatterBegin(sqpconP->state_scatter, sqpconP->DU, tao->stepdirection, INSERT_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
    ierr = VecScatterEnd(sqpconP->state_scatter, sqpconP->DU, tao->stepdirection, INSERT_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
    ierr = VecScatterBegin(sqpconP->design_scatter, sqpconP->DV, tao->stepdirection, INSERT_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
    ierr = VecScatterEnd(sqpconP->design_scatter, sqpconP->DV, tao->stepdirection, INSERT_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
    

    /* Perform Line Search */
    ierr = VecCopy(tao->solution, sqpconP->Xold); CHKERRQ(ierr);
    ierr = VecCopy(tao->gradient, sqpconP->Gold); CHKERRQ(ierr);
    fold = f;
    ierr = TaoLineSearchComputeObjectiveAndGradient(tao->linesearch,tao->solution,&fm,sqpconP->GL); CHKERRQ(ierr);
    ierr = TaoLineSearchSetInitialStepLength(tao->linesearch,1.0);
    ierr = TaoLineSearchApply(tao->linesearch, tao->solution, &fm, sqpconP->GL, tao->stepdirection,&step, &ls_reason); CHKERRQ(ierr);
    if (ls_reason < 0) {
      ierr = VecCopy(sqpconP->Xold, tao->solution);
      ierr = VecCopy(sqpconP->Gold, tao->gradient);
      f = fold;
      ierr = VecAXPY(tao->solution, 1.0, tao->stepdirection); CHKERRQ(ierr);
      ierr = PetscInfo(tao,"Line Search Failed, using full step."); CHKERRQ(ierr);
      use_update=PETSC_FALSE;
    } else {
      use_update = PETSC_TRUE;
    }


    /* Scatter X to U,V */
    ierr = VecScatterBegin(sqpconP->state_scatter, tao->solution, sqpconP->U, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
    ierr = VecScatterEnd(sqpconP->state_scatter, tao->solution, sqpconP->U, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
    ierr = VecScatterBegin(sqpconP->design_scatter, tao->solution, sqpconP->V, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
    ierr = VecScatterEnd(sqpconP->design_scatter, tao->solution, sqpconP->V, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);

    
    /* Evaluate Function, Gradient, Constraints, and Jacobian */
    ierr = TaoSolverComputeObjectiveAndGradient(tao,tao->solution,&f,tao->gradient); CHKERRQ(ierr);
    ierr = TaoSolverComputeConstraints(tao,tao->solution, tao->constraints); CHKERRQ(ierr);
    ierr = TaoSolverComputeJacobianState(tao,tao->solution, &tao->jacobian_state, &tao->jacobian_state_pre, &sqpconP->statematflag); CHKERRQ(ierr);
    ierr = TaoSolverComputeJacobianDesign(tao,tao->solution, &tao->jacobian_design, &tao->jacobian_design_pre, &sqpconP->designmatflag); CHKERRQ(ierr);



  
    /* Scatter gradient to GU,GV */
    ierr = VecScatterBegin(sqpconP->state_scatter, tao->gradient, sqpconP->GU, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
    ierr = VecScatterEnd(sqpconP->state_scatter, tao->gradient, sqpconP->GU, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
    ierr = VecScatterBegin(sqpconP->design_scatter, tao->gradient, sqpconP->GV, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
    ierr = VecScatterEnd(sqpconP->design_scatter, tao->gradient, sqpconP->GV, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);

    /* Update approx to hessian of the Lagrangian wrt state (Q) 
          with u_k+1, gu_k+1 */
    if (use_update) {
      ierr = MatApproxUpdate(sqpconP->Q,sqpconP->U,sqpconP->GU); CHKERRQ(ierr);
    }
    
    ierr = VecNorm(sqpconP->GL, NORM_2, &mnorm); CHKERRQ(ierr);

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
#define __FUNCT__ "TaoSolverCreate_SQPCON"
PetscErrorCode TaoSolverCreate_SQPCON(TaoSolver tao)
{
  TAO_SQPCON *sqpconP;
  PetscErrorCode ierr;
  const char *morethuente_type = TAOLINESEARCH_MT;
  PetscFunctionBegin;
  
  tao->ops->setup = TaoSolverSetup_SQPCON;
  tao->ops->solve = TaoSolverSolve_SQPCON;
  tao->ops->view = TaoSolverView_SQPCON;
  tao->ops->setfromoptions = TaoSolverSetFromOptions_SQPCON;
  tao->ops->destroy = TaoSolverDestroy_SQPCON;
  

  ierr = PetscNewLog(tao,TAO_SQPCON,&sqpconP); CHKERRQ(ierr);
  tao->data = (void*)sqpconP;

  tao->max_its=200;
  tao->fatol=1e-4;
  tao->frtol=1e-4;
  tao->gatol=1e-4;
  tao->grtol=1e-4;
  

  //sqpconP->subset_type=SQPCON_SUBSETes_SUBMAT;

  ierr = TaoLineSearchCreate(((PetscObject)tao)->comm, &tao->linesearch); CHKERRQ(ierr);
  ierr = TaoLineSearchSetType(tao->linesearch, morethuente_type); CHKERRQ(ierr);

  ierr = TaoLineSearchUseTaoSolverRoutines(tao->linesearch,tao); CHKERRQ(ierr);

    //SetObjectiveAndGradient(tao->linesearch,SQPCONObjectiveAndGradient, tao); CHKERRQ(ierr);


  ierr = KSPCreate(((PetscObject)tao)->comm,&tao->ksp); CHKERRQ(ierr);

  ierr = KSPSetFromOptions(tao->ksp); CHKERRQ(ierr);
  
  PetscFunctionReturn(0);

}
EXTERN_C_END


/*
#undef __FUNCT__
#define __FUNCT__ "SQPCONObjectiveAndGradient"
static PetscErrorCode SQPCONObjectiveAndGradient(TaoLineSearch ls, Vec X, PetscReal *f, Vec G, void *ptr)
{
  TaoSolver tao = (TaoSolver)ptr;
  TAO_SQPCON *sqpconP = (TAO_SQPCON*)tao->data;
  //PetscReal lmh;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TaoSolverComputeObjectiveAndGradient(tao,X,f,G); CHKERRQ(ierr);
  ierr = TaoSolverComputeConstraints(tao,X,tao->constraints); CHKERRQ(ierr);
  //ierr = TaoSolverComputeJacobianState(tao,X,&tao->jacobian_state,&tao->jacobian_state_pre,&sqpconP->statematflag); CHKERRQ(ierr);
  //ierr = TaoSolverComputeJacobianDesign(tao,X,&tao->jacobian_design,&tao->jacobian_design_pre,&sqpconP->designmatflag); CHKERRQ(ierr);

  //ierr = VecDot(tao->constraints,sqpconP->LM,&lmh); CHKERRQ(ierr);

  // *f -= lmh;
  
  // Gradient of Lagrangian GL = G - J' * lamda 
  //      WU = A' * WL
  //      WV = B' * WL 
  ierr = MatMultTranspose(tao->jacobian_state,sqpconP->LM,sqpconP->WU); CHKERRQ(ierr);
  ierr = MatMultTranspose(tao->jacobian_design,sqpconP->LM,sqpconP->WV); CHKERRQ(ierr);
  
  ierr = VecScatterBegin(sqpconP->state_scatter, sqpconP->WU, sqpconP->W, INSERT_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
  ierr = VecScatterEnd(sqpconP->state_scatter, sqpconP->WU, sqpconP->W, INSERT_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
  ierr = VecScatterBegin(sqpconP->design_scatter, sqpconP->WV, sqpconP->W, INSERT_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
  ierr = VecScatterEnd(sqpconP->design_scatter, sqpconP->WV, sqpconP->W, INSERT_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
  


  ierr = VecAXPY(G,-1.0,sqpconP->W); CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}
*/
