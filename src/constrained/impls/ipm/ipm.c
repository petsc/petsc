#include "taolinesearch.h"
#include "ipm.h" /*I "ipm.h" I*/


/*
% full-space version assumes
% min   d'*x + (1/2)x'*H*x 
% s.t.       AEz x == bE
%            AIz x >= bI
%
*/
/* 
   x,d in R^n
   f in R
   y in R^mi is slack vector (Ax-b) 
   bin in R^mi (tao->constraints_inequality)
   beq in R^me (tao->constraints_equality)
   lamdai in R^mi (ipmP->lamdai)
   lamdae in R^me (ipmP->lamdae)
   Aeq in R^(me x n) (tao->jacobian_equality)
   Ain in R^(mi x n) (tao->jacobian_inequality)
   H in R^(n x n) (tao->hessian)
   min f=(1/2)*x'*H*x + d'*x   
   s.t.  Aeq*x == beq
         Ain*x >= bin
*/

static PetscErrorCode IPMObjectiveAndGradient(TaoLineSearch,Vec,PetscReal*,Vec,void*);
static PetscErrorCode IPMComputeKKT(TaoSolver tao,TAO_IPM *ipmP);

#undef __FUNCT__
#define __FUNCT__ "TaoSolve_IPM"
static PetscErrorCode TaoSolve_IPM(TaoSolver tao)
{
  PetscErrorCode ierr;
  TAO_IPM* ipmP = (TAO_IPM*)tao->data;

  TaoSolverTerminationReason reason = TAO_CONTINUE_ITERATING;
  TaoLineSearchTerminationReason ls_status = TAOLINESEARCH_CONTINUE_ITERATING;
  PetscInt iter = 0;
  PetscReal stepsize=1.0;
  PetscFunctionBegin;
  PetscFunctionReturn(0);
  
}

#undef __FUNCT__
#define __FUNCT__ "TaoSetup_IPM"
static PetscErrorCode TaoSetup_IPM(TaoSolver tao)
{
  TAO_IPM *ipmP = (TAO_IPM*)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ipmP->mi = ipmP->me = 0;
  ierr = VecGetSize(tao->solution,&impP->n); CHKERRQ(ierr);
  if (!tao->gradient) {
    ierr = VecDuplicate(tao->solution, &tao->gradient); CHKERRQ(ierr);
  }
  
  if (tao->constraints_inequality) {
    ierr = VecGetSize(tao->constraints_inequality,&ipmP->mi); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_inequality,&ipmP->yi); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_inequality,&ipmP->lamdai); CHKERRQ(ierr);
  }
  if (tao->constraints_equality) {
    ierr = VecGetSize(tao->constraints_equality,&ipmP->me); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_equality,&ipmP->lamdae); CHKERRQ(ierr);
  }
  
    
  PetscFunctionReturn(0);
  
}

#undef __FUNCT__
#define __FUNCT__ "TaoDestroy_IPM"
static PetscErrorCode TaoDestroy_IPM(TaoSolver tao)
{
}

#undef __FUNCT__
#define __FUNCT__ "TaoSetFromOptions"
static PetscErrorCode TaoSetFromOptions_IPM(TaoSolver tao)
{
}

#undef __FUNCT__
#define __FUNCT__ "TaoView_IPM"
static PetscErrorCode TaoView_IPM(TaoSolver tao, PetscViewer viewer)
{
}

/* IPMObjectiveAndGradient()
   f = d'x + 0.5 * x' * H * x
   rd = H*x + d + Ae'*lame - Ai'*lami
   rpe = Ae*x - be
   rpi = Ai*x - yi - bi
   mu = yi' * lami/mi;
   com = yi.*lami

   phi = ||rd|| + ||rpe|| + ||rpi|| + ||com||
*/
#undef __FUNCT__
#define __FUNCT__ "IPMObjectiveAndGradient"
static PetscErrorCode IPMObjectiveAndGradient(TaoLineSearch ls, Vec X, PetscReal *f, Vec G, void *tptr) 
{
  
}
/*
   f = d'x + 0.5 * x' * H * x
   rd = H*x + d + Ae'*lame - Ai'*lami
   rpe = Ae*x - be
   rpi = Ai*x - yi - bi
   mu = yi' * lami/mi;
   com = yi.*lami

   phi = ||rd|| + ||rpe|| + ||rpi|| + ||com||
*/
#undef __FUNCT__
#define __FUNCT__ "IPMComputeKKT"
static PetscErrorCode IPMComputeKKT(TaoSolver tao,TAO_IPM *ipmP)
{
  PetscScalar norm;
  PetscErrorCode ierr;
  
  ierr = TaoComputeObjectiveAndGradient(tao,tao->solution,&ipmP->kkt_f,ipmP->rd); CHKERRQ(ierr);
  ierr = TaoComputeHessian(tao,tao->solution,&tao->hessian,&tao->hessian_pre,&ipmP->Hflag); CHKERRQ(ierr);
  
  if (ipmP->me > 0) {
    ierr = TaoComputeEqualityConstraints(tao,tao->solution,tao->constraints_equality);
    ierr = TaoComputeJacobianEquality(tao,tao->solution,tao->jacobian_equality,tao->jacobian_equality_pre,&ipmP->Aiflag); CHKERRQ(ierr);

    /* rd = rd + Ae'*lamdae */
    ierr = MatMult(tao->jacobian_equality,ipmP->lamdae,ipmP->work); CHKERRQ(ierr);
    ierr = VecAXPY(ipm->rd, 1.0, ipmP->work); CHKERRQ(ierr);


    /* rpe = Ae*x - be */
    ierr = MatMult(tao->jacobian_equality,tao->solution,ipmP->rpe);CHKERRQ(ierr);
    ierr = VecAXPY(ipmP->rpe, -1.0, tao->constraints_equality); CHKERRQ(ierr);

  }
  if (ipmP->mi > 0) {
    ierr = TaoComputeInequalityConstraints(tao,tao->solution,tao->constraints_inequality);
    ierr = TaoComputeJacobianInequality(tao,tao->solution,tao->jacobian_inequality,tao->jacobian_inequality_pre,&ipmP->Aeflag); CHKERRQ(ierr);

    /* rd = rd + Ai'*lamdai */
    ierr = MatMult(tao->jacobian_inequality,ipmP->lamdae,ipmP->work); CHKERRQ(ierr);
    ierr = VecAXPY(ipm->rd, -1.0, ipmP->work); CHKERRQ(ierr);

    /* rpi = Ai*x - yi - bi */
    ierr = MatMult(tao->jacobian_inequality,tao->solution,ipmP->rpi); CHKERRQ(ierr);
    ierr = VecAXPY(ipmP->rpi, -1.0, tao->constraints_inequality); CHKERRQ(ierr);
    ierr = VecAXPY(ipmP->rpi, -1.0, ipmP->yi); CHKERRQ(ierr);
    
    /* com = yi .* lami */
    ierr = VecPointwiseMult(ipmP->com, ipmP->yi,ipmP->lami); CHKERRQ(ierr);
    
    /* phi = ||rd|| + ||rpe|| + ||rpi|| + ||com|| */
    ierr = VecNorm(impP->rd,NORM_2,&norm); CHKERRQ(ierr);
    ipmP->phi = norm; 
    if (ipmP->me > 0 ) {
      ierr = VecNorm(impP->rpe,NORM_2,&norm); CHKERRQ(ierr);
      ipmP->phi += norm; 
    }
    if (ipmP->mi > 0) {
      ierr = VecNorm(impP->rpi,NORM_2,&norm); CHKERRQ(ierr);
      ipmP->phi += norm; 
      ierr = VecNorm(impP->complementarity,NORM_2,&norm); CHKERRQ(ierr);
      ipmP->phi += norm; 
    }
  }
}


EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "TaoCreate_IPM"
PetscErrorCode TaoCreate_IPM(TaoSolver tao)
{
  TAO_IPM *ipmP;
  const char *morethuente_type = TAOLINESEARCH_MT;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  tao->ops->setup = TaoSetup_IPM;
  tao->ops->solve = TaoSolve_IPM;
  tao->ops->view = TaoView_IPM;
  tao->ops->setfromoptions = TaoSetFromOptions_IPM;
  tao->ops->destroy = TaoDestroy_IPM;
  tao->ops->computedual = TaoComputeDual_IPM;

  ierr = PetscNewLog(tao, TAO_IPM, &blmP); CHKERRQ(ierr);
  tao->data = (void*)ipmP;
  tao->max_it = 2000;
  tao->max_funcs = 4000;
  tao->fatol = 1e-4;
  tao->frtol = 1e-4;

  ierr = VecDestroy(&ipmP->lag); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->rd); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->rpe); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->rpi); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->work); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->lamdae); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->lamdai); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->yi); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->complementarity); CHKERRQ(ierr);

  ierr = TaoLineSearchCreate(((PetscObject)tao)->comm, &tao->linesearch); CHKERRQ(ierr);
  ierr = TaoLineSearchSetType(tao->linesearch, morethuente_type); CHKERRQ(ierr);
  ierr = TaoLineSearchSetObjectiveAndGradientRoutine(tao->linesearch, IPMObjectiveAndGradient, tao); CHKERRQ(ierr);
  PetscFunctionReturn(0);
  

}
EXTERN_C_END
