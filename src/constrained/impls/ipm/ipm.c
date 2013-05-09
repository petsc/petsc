#include "taolinesearch.h"
#include "ipm.h" /*I "ipm.h" I*/
#define DEBUG_K
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

static PetscErrorCode IPMComputeKKT(TaoSolver tao);
static PetscErrorCode IPMPushInitialPoint(TaoSolver tao);
static PetscErrorCode IPMUpdateK(TaoSolver tao);
static PetscErrorCode IPMUpdateAi(TaoSolver tao);
static PetscErrorCode IPMGatherRHS(TaoSolver tao,Vec,Vec,Vec,Vec,Vec);
static PetscErrorCode IPMScatterStep(TaoSolver tao,Vec,Vec,Vec,Vec,Vec);
static PetscErrorCode IPMInitializeBounds(TaoSolver tao);

#undef __FUNCT__
#define __FUNCT__ "TaoSolve_IPM"
static PetscErrorCode TaoSolve_IPM(TaoSolver tao)
{
  PetscErrorCode ierr;
  TAO_IPM* ipmP = (TAO_IPM*)tao->data;

  
  TaoSolverTerminationReason reason = TAO_CONTINUE_ITERATING;
  PetscInt iter = 0,its,i;
  PetscReal stepsize=1.0;
  PetscReal step_y,step_l,alpha,tau,sigma,muaff,phi_target;
  PetscFunctionBegin;

  /* Push initial point away from bounds */
  ierr = IPMInitializeBounds(tao); CHKERRQ(ierr);
  ierr = IPMPushInitialPoint(tao); CHKERRQ(ierr);
  ierr = VecCopy(tao->solution,ipmP->rhs_x); CHKERRQ(ierr);
  ierr = IPMComputeKKT(tao); CHKERRQ(ierr);
  ierr = TaoMonitor(tao,iter++,ipmP->kkt_f,ipmP->phi,0.0,1.0,&reason);


  while (reason == TAO_CONTINUE_ITERATING) {
    ierr = IPMUpdateK(tao); CHKERRQ(ierr);
    /* % compute affine scaling step 
       rhs.lami = -iter.yi.*iter.lami;
       rhs.x = -kkt.rd; 
       rhs.lame = -kkt.rpe; 
       rhs.yi = -kkt.rpi; 
       step = feval(par.compute_step,par,iter,rhs);
       dYaff = step.yi;
       dLaff = step.lami; */
    ierr = VecCopy(tao->solution,ipmP->rhs_x); CHKERRQ(ierr);
    ierr = VecCopy(ipmP->lamdai,ipmP->rhs_lamdai); CHKERRQ(ierr);
    ierr = VecCopy(ipmP->lamdae,ipmP->rhs_lamdae); CHKERRQ(ierr);
    ierr = VecCopy(ipmP->s,ipmP->rhs_s); CHKERRQ(ierr);

    ierr = VecPointwiseMult(ipmP->rhs_lamdai,ipmP->lamdai,ipmP->s); CHKERRQ(ierr);
    ierr = VecScale(ipmP->rhs_lamdai,-1.0); CHKERRQ(ierr);
    ierr = VecCopy(ipmP->rd,ipmP->rhs_x); CHKERRQ(ierr);
    ierr = VecScale(ipmP->rhs_x,-1.0); CHKERRQ(ierr);
    ierr = VecCopy(ipmP->rpe,ipmP->rhs_lamdae); CHKERRQ(ierr);
    ierr = VecScale(ipmP->rhs_lamdae,-1.0); CHKERRQ(ierr);
    ierr = VecCopy(ipmP->rpi,ipmP->rhs_s); CHKERRQ(ierr);
    ierr = VecScale(ipmP->rhs_s,-1.0); CHKERRQ(ierr);

    ierr = IPMGatherRHS(tao,ipmP->bigrhs,ipmP->rhs_x,ipmP->rhs_lamdae,
			ipmP->rhs_s,ipmP->rhs_lamdai); CHKERRQ(ierr);
    ierr = KSPSetOperators(tao->ksp,ipmP->K,ipmP->K,DIFFERENT_NONZERO_PATTERN); CHKERRQ(ierr);
    ierr = KSPSolve(tao->ksp,ipmP->bigrhs,ipmP->bigstep);CHKERRQ(ierr);

    ierr = IPMScatterStep(tao,ipmP->bigstep,tao->stepdirection,ipmP->dlamdae,
			  ipmP->ds,ipmP->dlamdai); CHKERRQ(ierr);
    ierr = KSPGetIterationNumber(tao->ksp,&its); CHKERRQ(ierr);
    tao->ksp_its += its;
    /* check result */
    //printf("solution = \n");
    //ierr = VecView(ipmP->bigstep,0); CHKERRQ(ierr);
    
    ierr = VecCopy(ipmP->ds,ipmP->dYaff); CHKERRQ(ierr);
    ierr = VecCopy(ipmP->dlamdai,ipmP->dLaff); CHKERRQ(ierr);

    /* get max stepsizes
       [yaff,step.alp.y] = max_stepsize(iter.yi,step.yi);
       [laff,step.alp.lam] = max_stepsize(iter.lami,step.lami);
       alp = min(step.alp.y,step.alp.lam);
       yaff = iter.yi + alp*step.yi;
       laff = iter.lami + alp*step.lami;
       muaff = yaff'*laff/mi;
    */
     /* Find distance along step direction to closest bound */
    
    ierr = VecStepBoundInfo(ipmP->s,ipmP->Zero_nb,ipmP->Inf_nb,ipmP->ds,&step_y,PETSC_NULL,PETSC_NULL); CHKERRQ(ierr);
    ierr = VecStepBoundInfo(ipmP->lamdai,ipmP->Zero_nb,ipmP->Inf_nb,ipmP->dlamdai,&step_l,PETSC_NULL,PETSC_NULL); CHKERRQ(ierr);

    alpha = PetscMin(step_y,step_l);
    alpha = PetscMin(alpha,1.0);
    ipmP->alpha1 = alpha;

    ierr = VecCopy(ipmP->s,ipmP->Yaff); CHKERRQ(ierr);
    ierr = VecAXPY(ipmP->Yaff,alpha,ipmP->ds); CHKERRQ(ierr);
    ierr = VecCopy(ipmP->lamdai,ipmP->Laff); CHKERRQ(ierr);
    ierr = VecAXPY(ipmP->Laff,alpha,ipmP->dlamdai); CHKERRQ(ierr);
    ierr = VecDot(ipmP->Yaff,ipmP->Laff,&muaff); CHKERRQ(ierr);
    muaff /= ipmP->nb; CHKERRQ(ierr);

    sigma = (muaff/ipmP->mu);
    sigma *= sigma*sigma;

    /* Compute full step */
    /* rhs.lami = rhs.lami - dYaff.*dLaff + sig*kkt.mu*e); */
    ierr = VecAXPY(ipmP->rhs_lamdai,sigma*ipmP->mu,ipmP->One_nb); CHKERRQ(ierr);
    ierr = VecPointwiseMult(ipmP->dYaff,ipmP->dLaff,ipmP->dYaff); CHKERRQ(ierr);
    ierr = VecAXPY(ipmP->rhs_lamdai, -1.0, ipmP->dYaff); CHKERRQ(ierr);
    /* step = feval(par.compute_step,par,iter,rhs); */
    
    
    ierr = IPMGatherRHS(tao,ipmP->bigrhs,ipmP->rhs_x,ipmP->rhs_lamdae,
			ipmP->rhs_s,ipmP->rhs_lamdai); CHKERRQ(ierr);
    ierr = PetscObjectStateIncrease((PetscObject)ipmP->K);CHKERRQ(ierr);
    
    ierr = KSPSolve(tao->ksp,ipmP->bigrhs,ipmP->bigstep);CHKERRQ(ierr);
    ierr = KSPGetIterationNumber(tao->ksp,&its); CHKERRQ(ierr);
    tao->ksp_its += its;
    ierr = IPMScatterStep(tao,ipmP->bigstep,tao->stepdirection,ipmP->dlamdae,
			  ipmP->ds,ipmP->dlamdai); CHKERRQ(ierr);



    /* % get max stepsizes
       tau = min(1,max(par.taumin,1-sig*kkt.mu));
       [y,step.alp.y] = max_stepsize(iter.yi,step.yi);
       [lam,step.alp.lam] = max_stepsize(iter.lami,step.lami);
       alp = min(step.alp.y,step.alp.lam);
       step.alp.y = alp;
       step.alp.lam = alp;
       % apply frac-to-boundary      
       step.alp.y = tau*step.alp.y;
       step.alp.lam = tau*step.alp.lam;
    */

    tau = PetscMax(ipmP->taumin,1.0-sigma*ipmP->mu);
    tau = PetscMin(tau,1.0);
    ierr = VecStepBoundInfo(ipmP->s,ipmP->Zero_nb,ipmP->Inf_nb,ipmP->ds,&step_y,PETSC_NULL,PETSC_NULL); CHKERRQ(ierr);
    ierr = VecStepBoundInfo(ipmP->lamdai,ipmP->Zero_nb,ipmP->Inf_nb,ipmP->dlamdai,&step_l,PETSC_NULL,PETSC_NULL); CHKERRQ(ierr);

    alpha = PetscMin(step_y,step_l);
    alpha = PetscMin(alpha,1.0);
    alpha *= tau;
    ipmP->alpha2 = alpha;
    

    /*
    % line-search to achieve sufficient decrease
    fr = 1;
    for i = 1:11
      step.alp.y = fr*step.alp.y;
      step.alp.lam = fr*step.alp.lam;
      iter_trial = update_iter(iter,step);
      kkt_trial = feval(par.eval_kkt,par,iter_trial);
      if kkt_trial.phi <= par.dec*kkt.phi
        break
      else
        fr  = fr*(2^(-i));    <-- TODO: Check if this is right
      end
   end
   iter.ls=i-1;
    */

    phi_target = ipmP->dec * ipmP->phi;
    for (i=0; i<11;i++) {
      ierr = VecAXPY(tao->solution,alpha,tao->stepdirection); CHKERRQ(ierr);
      ierr = VecAXPY(ipmP->lamdae,alpha,ipmP->dlamdae); CHKERRQ(ierr);
      ierr = VecAXPY(ipmP->s,alpha,ipmP->ds); CHKERRQ(ierr);
      ierr = VecAXPY(ipmP->lamdai,alpha,ipmP->dlamdai); CHKERRQ(ierr);

      ierr = IPMComputeKKT(tao); CHKERRQ(ierr); CHKERRQ(ierr);
      if (ipmP->phi <= phi_target) break;
      alpha /= 2.0;
    }


    iter++;
    ierr = TaoMonitor(tao,iter,ipmP->kkt_f,ipmP->phi,0.0,stepsize,&reason);
    CHKERRQ(ierr);

  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoSetup_IPM"
static PetscErrorCode TaoSetup_IPM(TaoSolver tao)
{
  TAO_IPM *ipmP = (TAO_IPM*)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ipmP->nb = ipmP->mi = ipmP->me = 0;
  ipmP->K=0;
  ierr = VecGetSize(tao->solution,&ipmP->n); CHKERRQ(ierr);
  if (!tao->gradient) {
    ierr = VecDuplicate(tao->solution, &tao->gradient); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->solution, &tao->stepdirection); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->solution, &ipmP->rd); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->solution, &ipmP->rhs_x); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->solution, &ipmP->work); CHKERRQ(ierr);
  }
  
  if (tao->constraints_equality) {
    ierr = VecGetSize(tao->constraints_equality,&ipmP->me); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_equality,&ipmP->lamdae); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_equality,&ipmP->dlamdae); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_equality,&ipmP->rhs_lamdae); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_equality,&ipmP->rpe); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "IPMInitializeBounds"
static PetscErrorCode IPMInitializeBounds(TaoSolver tao)
{
  TAO_IPM *ipmP = (TAO_IPM*)tao->data;
  Vec xtmp;
  PetscErrorCode ierr;

  MPI_Comm comm;
  PetscFunctionBegin;
  ipmP->mi=0;
  ipmP->nxlb=0;
  ipmP->nxub=0;
  ipmP->niub=0;
  ipmP->nilb=0;
  ipmP->nb=0;
  ipmP->nslack=0;


  /* lamd = PsU'*nusU - PsL'*nusL */
  /*       ?=  lami - corr. ub
               lami + corr. lb */
  
  ierr = VecDuplicate(tao->solution,&xtmp); CHKERRQ(ierr);
  ierr = VecSet(xtmp,TAO_NINFINITY); CHKERRQ(ierr);
  ierr = VecWhichGreaterThan(tao->XL,xtmp,&ipmP->isxl); CHKERRQ(ierr);
  ierr = VecSet(xtmp,TAO_INFINITY); CHKERRQ(ierr);
  ierr = VecWhichLessThan(xtmp,tao->XU,&ipmP->isxu); CHKERRQ(ierr);
  ierr = ISGetSize(ipmP->isxl,&ipmP->nxlb); CHKERRQ(ierr);
  ierr = ISGetSize(ipmP->isxu,&ipmP->nxub); CHKERRQ(ierr);
  ierr = VecDestroy(&xtmp); CHKERRQ(ierr);
  if (tao->constraints_inequality) {
    ierr = VecGetSize(tao->constraints_inequality,&ipmP->mi); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_inequality,&xtmp); CHKERRQ(ierr);
    ierr = VecSet(xtmp,TAO_NINFINITY); CHKERRQ(ierr);
    ierr = VecWhichGreaterThan(tao->IL,xtmp,&ipmP->isil); CHKERRQ(ierr);
    ierr = VecSet(xtmp,TAO_INFINITY); CHKERRQ(ierr);
    ierr = VecWhichLessThan(xtmp,tao->IU,&ipmP->isiu); CHKERRQ(ierr);
    ierr = ISGetSize(ipmP->isil,&ipmP->nilb); CHKERRQ(ierr);
    ierr = ISGetSize(ipmP->isiu,&ipmP->niub); CHKERRQ(ierr);
    ierr = VecDestroy(&xtmp); CHKERRQ(ierr);
  } else {
    ipmP->nilb = ipmP->niub = 0;
  }
  ipmP->nb = ipmP->nxlb + ipmP->nxub + ipmP->nilb + ipmP->niub;
  
  

  if (ipmP->nb > 0) {
    comm = ((PetscObject)(tao->solution))->comm;
    ierr = VecCreate(comm,&ipmP->s); CHKERRQ(ierr);
    ierr = VecSetSizes(ipmP->s,PETSC_DECIDE,ipmP->nb); CHKERRQ(ierr);
    ierr = VecSetFromOptions(ipmP->s); CHKERRQ(ierr);
    ierr = VecDuplicate(ipmP->s,&ipmP->ds); CHKERRQ(ierr);
    ierr = VecDuplicate(ipmP->s,&ipmP->rhs_s); CHKERRQ(ierr);
    ierr = VecDuplicate(ipmP->s,&ipmP->complementarity); CHKERRQ(ierr);

    ierr = VecDuplicate(ipmP->s,&ipmP->lamdai); CHKERRQ(ierr);
    ierr = VecDuplicate(ipmP->s,&ipmP->dlamdai); CHKERRQ(ierr);
    ierr = VecDuplicate(ipmP->s,&ipmP->rhs_lamdai); CHKERRQ(ierr);


    ierr = VecDuplicate(ipmP->s,&ipmP->Yaff); CHKERRQ(ierr);
    ierr = VecDuplicate(ipmP->s,&ipmP->dYaff); CHKERRQ(ierr);
    ierr = VecDuplicate(ipmP->s,&ipmP->Laff); CHKERRQ(ierr);
    ierr = VecDuplicate(ipmP->s,&ipmP->dLaff); CHKERRQ(ierr);
    ierr = VecDuplicate(ipmP->s,&ipmP->rpi); CHKERRQ(ierr);
    ierr = VecDuplicate(ipmP->s,&ipmP->Zero_nb); CHKERRQ(ierr);
    ierr = VecSet(ipmP->Zero_nb,0.0); CHKERRQ(ierr);
    ierr = VecDuplicate(ipmP->s,&ipmP->One_nb); CHKERRQ(ierr);
    ierr = VecSet(ipmP->One_nb,1.0); CHKERRQ(ierr);
    ierr = VecDuplicate(ipmP->s,&ipmP->Inf_nb); CHKERRQ(ierr);
    ierr = VecSet(ipmP->Inf_nb,TAO_INFINITY); CHKERRQ(ierr);
    ierr = VecDuplicate(ipmP->s,&ipmP->worknb); CHKERRQ(ierr);

    /* construct ci from non-infinity bounds 
    ierr = VecDuplicate(ipmP->s,&ipmP->ci); CHKERRQ(ierr);

    r1 = ipmP->nilb;
    r2 = r1 + ipmP->niub;
    r3 = r2 + ipmP->nxlb;
    r4 = r3 + ipmP->nxub;

    ierr = VecGetArray(ipmP->ci,&c); CHKERRQ(ierr);
    ierr = ISGetIndices(ipmP->isil,&indices); CHKERRQ(ierr);
    */    

  }    
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoDestroy_IPM"
static PetscErrorCode TaoDestroy_IPM(TaoSolver tao)
{
  TAO_IPM *ipmP = (TAO_IPM*)tao->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecDestroy(&ipmP->rd); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->rpe); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->rpi); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->work); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->lamdae); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->lamdai); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->s); CHKERRQ(ierr);

  ierr = VecDestroy(&ipmP->rhs_x); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->rhs_lamdae); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->rhs_lamdai); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->rhs_s); CHKERRQ(ierr);

  ierr = VecDestroy(&ipmP->dlamdae); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->dlamdai); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->ds); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->complementarity); CHKERRQ(ierr);

  ierr = VecDestroy(&ipmP->Yaff); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->dYaff); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->Laff); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->dLaff); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->Zero_nb); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->One_nb); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->Inf_nb); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->worknb); CHKERRQ(ierr);

  ierr = MatDestroy(&ipmP->K); CHKERRQ(ierr);
  ierr = MatDestroy(&ipmP->Ai); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->bigrhs); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->bigstep); CHKERRQ(ierr);

  ierr = PetscFree(tao->data); CHKERRQ(ierr);
  tao->data = PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoSetFromOptions_IPM"
static PetscErrorCode TaoSetFromOptions_IPM(TaoSolver tao)
{
  TAO_IPM *ipmP = (TAO_IPM*)tao->data;
  PetscErrorCode ierr;
  PetscBool flg;
  PetscFunctionBegin;
  ierr = PetscOptionsHead("IPM method for constrained optimization"); CHKERRQ(ierr);
  ierr = PetscOptionsBool("-ipm_monitorkkt","monitor kkt status",PETSC_NULL,ipmP->monitorkkt,&ipmP->monitorkkt,&flg); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-ipm_pushs","parameter to push initial slack variables away from bounds",PETSC_NULL,ipmP->pushs,&ipmP->pushs,&flg);
  ierr = PetscOptionsReal("-ipm_pushlam","parameter to push initial dual variables away from bounds",PETSC_NULL,ipmP->pushlam,&ipmP->pushlam,&flg);
  ierr = PetscOptionsTail(); CHKERRQ(ierr);
  ierr =KSPSetFromOptions(tao->ksp); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoView_IPM"
static PetscErrorCode TaoView_IPM(TaoSolver tao, PetscViewer viewer)
{
  return 0;
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
/*
#undef __FUNCT__
#define __FUNCT__ "IPMObjective"
static PetscErrorCode IPMObjective(TaoLineSearch ls, Vec X, PetscReal *f, void *tptr) 
{
  TaoSolver tao = (TaoSolver)tptr;
  TAO_IPM *ipmP = (TAO_IPM*)tao->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = IPMComputeKKT(tao); CHKERRQ(ierr);
  *f = ipmP->phi;
  PetscFunctionReturn(0);
}
*/

/*
   f = d'x + 0.5 * x' * H * x
   rd = H*x + d + Ae'*lame - Ai'*lami
       Ai =   jac_ineq(w/lb)
             -jac_ineq(w/ub)
	       I (w/lb)
	      -I (w/ub) 

   rpe = ceq
   rpi = cin - s;
   com = s.*lami
   mu = yi' * lami/mi;

   phi = ||rd|| + ||rpe|| + ||rpi|| + ||com||
*/
#undef __FUNCT__
#define __FUNCT__ "IPMComputeKKT"
static PetscErrorCode IPMComputeKKT(TaoSolver tao)
{
  TAO_IPM *ipmP = (TAO_IPM *)tao->data;
  PetscScalar norm;
  PetscErrorCode ierr;
  
  
  ierr = TaoComputeObjectiveAndGradient(tao,tao->solution,&ipmP->kkt_f,ipmP->rd); CHKERRQ(ierr);
  ierr = TaoComputeHessian(tao,tao->solution,&tao->hessian,&tao->hessian_pre,&ipmP->Hflag); CHKERRQ(ierr);
  
  if (ipmP->me > 0) {
    ierr = TaoComputeEqualityConstraints(tao,tao->solution,tao->constraints_equality);
    ierr = TaoComputeJacobianEquality(tao,tao->solution,&tao->jacobian_equality,&tao->jacobian_equality_pre,&ipmP->Aiflag); CHKERRQ(ierr);

    /* rd = rd + Ae'*lamdae */
    ierr = MatMultTranspose(tao->jacobian_equality,ipmP->lamdae,ipmP->work); CHKERRQ(ierr);
    ierr = VecAXPY(ipmP->rd, 1.0, ipmP->work); CHKERRQ(ierr);


    /* rpe = ce(x) */
    ierr = VecCopy(tao->constraints_equality,ipmP->rpe); CHKERRQ(ierr);

  }
  if (ipmP->mi > 0) {
    ierr = TaoComputeInequalityConstraints(tao,tao->solution,tao->constraints_inequality);
    ierr = TaoComputeJacobianInequality(tao,tao->solution,&tao->jacobian_inequality,&tao->jacobian_inequality_pre,&ipmP->Aeflag); CHKERRQ(ierr);

    /* rd = rd - Ai'*lamdai */
    /* Ai' =   jac_ineq(w/lb)' | -jac_ineq(w/ub)' | I (w/lb) | -I (w/ub)  */

    ierr = IPMUpdateAi(tao); CHKERRQ(ierr);
    ierr = MatMultTranspose(ipmP->Ai,ipmP->lamdai,ipmP->work); CHKERRQ(ierr);
    ierr = VecAXPY(ipmP->rd, -1.0, ipmP->work); CHKERRQ(ierr);

    /* rpi = cin - s */ /*Aix - s? */
    ierr = MatMult(ipmP->Ai,tao->solution,ipmP->rpi); CHKERRQ(ierr);
    ierr = VecAXPY(ipmP->rpi, -1.0, ipmP->s); CHKERRQ(ierr);
    //  ierr = VecCopy(tao->constraints_inequality,ipmP->rpi); CHKERRQ(ierr);

    /* com = s .* lami */
    ierr = VecPointwiseMult(ipmP->complementarity, ipmP->s,ipmP->lamdai); CHKERRQ(ierr);

    /* mu = s'*lami/nb */
    ierr = VecDot(ipmP->s,ipmP->lamdai,&ipmP->mu); CHKERRQ(ierr);
    ipmP->mu /= ipmP->nb;
  }
  /* phi = ||rd|| + ||rpe|| + ||rpi|| + ||com|| */
  ierr = VecNorm(ipmP->rd,NORM_2,&norm); CHKERRQ(ierr);
  ipmP->phi = norm; 
  if (ipmP->me > 0 ) {
    ierr = VecNorm(ipmP->rpe,NORM_2,&norm); CHKERRQ(ierr);
    ipmP->phi += norm; 
  }
  if (ipmP->mi > 0) {
    ierr = VecNorm(ipmP->rpi,NORM_2,&norm); CHKERRQ(ierr);
    ipmP->phi += norm; 
    ierr = VecNorm(ipmP->complementarity,NORM_2,&norm); CHKERRQ(ierr);
    ipmP->phi += norm; 
  }
  if (ipmP->monitorkkt) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"obj=%G,\tphi = %G,\tmu=%G\talpha1=%G\talpha2=%G\n",ipmP->kkt_f,ipmP->phi,ipmP->mu,ipmP->alpha1,ipmP->alpha2);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "IPMFindInitialPoint"
/* Push initial point away from bounds */
PetscErrorCode IPMPushInitialPoint(TaoSolver tao)
{
  TAO_IPM *ipmP = (TAO_IPM *)tao->data;
  PetscErrorCode ierr;

  ierr = TaoComputeVariableBounds(tao); CHKERRQ(ierr);
  if (tao->XL && tao->XU) {
    ierr = VecMedian(tao->XL, tao->solution, tao->XU, tao->solution); CHKERRQ(ierr);
  }
  ierr = VecSet(ipmP->s,1.0); CHKERRQ(ierr);
  ierr = VecSet(ipmP->lamdai,1.0); CHKERRQ(ierr);
  ierr = VecSet(ipmP->lamdae,1.0); CHKERRQ(ierr);

  ierr = IPMComputeKKT(tao); CHKERRQ(ierr);
  ierr = IPMUpdateK(tao); CHKERRQ(ierr);

  /* Compute affine scaling step */
  ierr = VecCopy(ipmP->rd,ipmP->rhs_x); CHKERRQ(ierr);
  ierr = VecCopy(ipmP->rpe,ipmP->rhs_lamdae); CHKERRQ(ierr);
  ierr = VecCopy(ipmP->rpi,ipmP->rhs_s); CHKERRQ(ierr);
  ierr = VecCopy(ipmP->complementarity,ipmP->rhs_lamdai); CHKERRQ(ierr);
  ierr = IPMGatherRHS(tao,ipmP->bigrhs,ipmP->rhs_x,ipmP->rhs_lamdae,
		      ipmP->rhs_s,ipmP->lamdai); CHKERRQ(ierr);
  
  ierr = KSPSetOperators(tao->ksp,ipmP->K,ipmP->K,DIFFERENT_NONZERO_PATTERN); CHKERRQ(ierr);
  ierr = KSPSolve(tao->ksp,ipmP->bigrhs,ipmP->bigstep);CHKERRQ(ierr);
  ierr = IPMScatterStep(tao,ipmP->bigstep,tao->stepdirection,ipmP->dlamdae,
			  ipmP->ds,ipmP->dlamdai); CHKERRQ(ierr);

  /* Push initial step into feasible region */
  ierr = VecSet(ipmP->rhs_s,ipmP->pushs); CHKERRQ(ierr);
  ierr = VecAXPY(ipmP->s,1.0,ipmP->ds); CHKERRQ(ierr);
  ierr = VecPointwiseMax(ipmP->s,ipmP->rhs_s,ipmP->s); CHKERRQ(ierr);


  ierr = VecSet(ipmP->rhs_lamdai,ipmP->pushlam); CHKERRQ(ierr);
  ierr = VecAXPY(ipmP->lamdai,1.0,ipmP->dlamdai); CHKERRQ(ierr);
  ierr = VecPointwiseMax(ipmP->lamdai,ipmP->rhs_lamdai,ipmP->lamdai); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "IPMUpdateAi"
PetscErrorCode IPMUpdateAi(TaoSolver tao)
{
  /* Ai =     Ji(w/lb)
             -Ji(w/ub)
	      I (w/lb)
	     -I (w/ub) */
  TAO_IPM *ipmP = (TAO_IPM *)tao->data;
  MPI_Comm comm;
  PetscInt i,j;
  PetscScalar newval;
  PetscInt newrow,newcol,ncols;
  const PetscScalar *vals;
  const PetscInt *cols;
  PetscInt *nonzeros;
  const PetscInt *indices;
  PetscInt r1,r2,r3,r4;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;

  r1 = ipmP->nilb;
  r2 = r1 + ipmP->niub;
  r3 = r2 + ipmP->nxlb;
  r4 = r3 + ipmP->nxub;
  
  if (!ipmP->nb) {
    PetscFunctionReturn(0);
  }
  if (!ipmP->Ai) {
    comm = ((PetscObject)(tao->solution))->comm;
    ierr = PetscMalloc(ipmP->nb*sizeof(PetscInt),&nonzeros); CHKERRQ(ierr);
    ierr = ISGetIndices(ipmP->isil,&indices); CHKERRQ(ierr);
    for (i=0;i<r1;i++) {
      ierr = MatGetRow(tao->jacobian_inequality,i,&ncols,PETSC_NULL,PETSC_NULL); CHKERRQ(ierr);
      nonzeros[i] = ncols;
      nonzeros[i+r1] = ncols;
      ierr = MatRestoreRow(tao->jacobian_inequality,i,&ncols,PETSC_NULL,PETSC_NULL); CHKERRQ(ierr);
    }
    for (i=r2;i<r4;i++) {
      nonzeros[i] = 1;
    }
    ierr = ISRestoreIndices(ipmP->isil,&indices); CHKERRQ(ierr);
    ierr = MatCreate(comm,&ipmP->Ai); CHKERRQ(ierr);
    ierr = MatSetSizes(ipmP->Ai,PETSC_DECIDE,PETSC_DECIDE,ipmP->nb,ipmP->n);CHKERRQ(ierr);
    ierr = MatSetType(ipmP->Ai,MATSEQAIJ); CHKERRQ(ierr);
    ierr = MatSetFromOptions(ipmP->Ai); CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(ipmP->Ai,PETSC_NULL,nonzeros);CHKERRQ(ierr);
    ierr = PetscFree(nonzeros);CHKERRQ(ierr);
  }
  ierr = MatZeroEntries(ipmP->Ai); CHKERRQ(ierr);

  /* Ai w/lb */
  ierr = ISGetIndices(ipmP->isil,&indices); CHKERRQ(ierr);
  for (i=0;i<ipmP->nilb;i++) {
    ierr = MatGetRow(tao->jacobian_inequality,indices[i],&ncols,&cols,&vals); CHKERRQ(ierr);
    newrow = i;
    ierr = MatSetValues(ipmP->Ai,1,&newrow,ncols,cols,vals,INSERT_VALUES); CHKERRQ(ierr);
    ierr = MatRestoreRow(tao->jacobian_inequality,indices[i],&ncols,&cols,&vals); CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(ipmP->isil,&indices); CHKERRQ(ierr);

  /* Ai w/ ub */
  ierr = ISGetIndices(ipmP->isiu,&indices); CHKERRQ(ierr);
  for (i=0;i<ipmP->niub;i++) {
    ierr = MatGetRow(tao->jacobian_inequality,indices[i],&ncols,&cols,&vals); CHKERRQ(ierr);
    for (j=0;j<ncols;j++) {
      newrow = i+r1;
      newcol = j;
      newval = -vals[j];
      ierr = MatSetValues(ipmP->Ai,1,&newrow,1,&newcol,&newval,INSERT_VALUES); CHKERRQ(ierr);
    }
    ierr = MatRestoreRow(tao->jacobian_inequality,indices[i],&ncols,&cols,&vals); CHKERRQ(ierr);
    ierr = ISRestoreIndices(ipmP->isiu,&indices); CHKERRQ(ierr);
  }
  /* I w/ xlb */
  ierr = ISGetIndices(ipmP->isxl,&indices); CHKERRQ(ierr);
  for (i=0;i<ipmP->nxlb;i++) {
    newrow = i+r2;
    newcol = i;
    newval = 1.0;
    ierr = MatSetValues(ipmP->Ai,1,&newrow,1,&newcol,&newval,INSERT_VALUES); CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(ipmP->isxl,&indices); CHKERRQ(ierr);

  /* I w/ xub */
  ierr = ISGetIndices(ipmP->isxu,&indices); CHKERRQ(ierr);
  for (i=0;i<ipmP->nxub;i++) {
    newrow = i+r3;
    newcol = i;
    newval = -1.0;
    ierr = MatSetValues(ipmP->Ai,1,&newrow,1,&newcol,&newval,INSERT_VALUES); CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(ipmP->isxu,&indices); CHKERRQ(ierr);

  
  ierr = MatAssemblyBegin(ipmP->Ai,MAT_FINAL_ASSEMBLY);
  ierr = MatAssemblyEnd(ipmP->Ai,MAT_FINAL_ASSEMBLY);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "IPMUpdateK"
/* create K = [ H , 0 , Ae', -Ai']; 
              [Ae , 0,   0  , 0];
              [Ai ,-I,   0 ,  0];  
              [ 0 , S ,  0,   Y ];  */
PetscErrorCode IPMUpdateK(TaoSolver tao)
{
  TAO_IPM *ipmP = (TAO_IPM *)tao->data;
  MPI_Comm comm;
  PetscErrorCode ierr;
  PetscInt i,j;
  PetscInt ncols,newcol,newcols[2],newrow;
  const PetscInt *cols;
  const PetscReal *vals;
  PetscReal *l,*y;
  PetscReal *newvals;
  PetscReal newval;
  PetscInt subsize;
  const PetscInt *indices;
  PetscInt *nonzeros,bigsize;
  PetscInt r1,r2,r3;
  PetscInt c1,c2,c3;
  PetscFunctionBegin;
#if defined DEBUG_K
  printf("H\n");  MatView(tao->hessian,0);
  if (ipmP->nilb+ipmP->niub) {
    printf("Ai\n"); MatView(tao->jacobian_inequality,0);
  }
  if (ipmP->me) {
    printf("Ae\n"); MatView(tao->jacobian_equality,0);
  }
  printf("IL\n"); ISView(ipmP->isil,0);
  printf("IU\n"); ISView(ipmP->isiu,0);
  printf("XL\n"); ISView(ipmP->isxl,0);
  printf("XU\n"); ISView(ipmP->isxu,0);

#endif  
  /* allocate workspace */
  subsize = PetscMax(ipmP->n,ipmP->nb);
  subsize = PetscMax(ipmP->me,subsize);
  subsize = PetscMax(2,subsize);
  ierr = PetscMalloc(sizeof(PetscInt)*subsize,&indices); CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*subsize,&newvals); CHKERRQ(ierr);

  r1 = c1 = ipmP->n;
  r2 = r1 + ipmP->me;  c2 = c1 + ipmP->nb;
  r3 = c3 = r2 + ipmP->nb;
  
  

  if (!ipmP->K) {
    /* TODO get number of nonzeros. This is a hack */
    comm = ((PetscObject)(tao->solution))->comm;
    bigsize = ipmP->n+2*ipmP->nb+ipmP->me;
    ierr = PetscMalloc(bigsize*sizeof(PetscInt),&nonzeros);CHKERRQ(ierr);
    for (i=0;i<ipmP->n;i++) {
      ierr = MatGetRow(tao->hessian,i,&ncols,PETSC_NULL,PETSC_NULL); CHKERRQ(ierr);
      nonzeros[i] = ncols;
      ierr = MatRestoreRow(tao->hessian,i,&ncols,PETSC_NULL,PETSC_NULL); CHKERRQ(ierr);
      nonzeros[i] += ipmP->me+ipmP->nb;
    
    }
    for (i=ipmP->n;i<ipmP->n+ipmP->me;i++) {
      /*TODO get nonzeros in Ae*/
      nonzeros[i] = ipmP->n;
    }
    for (i=ipmP->n+ipmP->me;i<ipmP->n+ipmP->me+ipmP->nb;i++) {
      /* TODO get nonzeros in Ai */
      nonzeros[i] = ipmP->n+1;
    }
    for (i=ipmP->n+ipmP->me+ipmP->nb;i<ipmP->n+ipmP->me+2*ipmP->nb;i++) {
      nonzeros[i] = 2;
    }

    ierr = MatCreate(comm,&ipmP->K); CHKERRQ(ierr);
    ierr = MatSetSizes(ipmP->K,PETSC_DECIDE,PETSC_DECIDE,bigsize,bigsize);CHKERRQ(ierr);
    ierr = MatSetType(ipmP->K,MATSEQAIJ); CHKERRQ(ierr);
    ierr = MatSetFromOptions(ipmP->K); CHKERRQ(ierr);

    ierr = MatSeqAIJSetPreallocation(ipmP->K,PETSC_NULL,nonzeros); CHKERRQ(ierr);

    ierr = VecCreate(comm,&ipmP->bigrhs); CHKERRQ(ierr);
    ierr = VecSetSizes(ipmP->bigrhs,PETSC_DECIDE,bigsize); CHKERRQ(ierr);
    ierr = VecSetType(ipmP->bigrhs,VECSEQ); CHKERRQ(ierr);
    ierr = VecDuplicate(ipmP->bigrhs,&ipmP->bigstep); CHKERRQ(ierr);
    ierr = PetscFree(nonzeros);CHKERRQ(ierr);
  }

 
  ierr = MatZeroEntries(ipmP->K); CHKERRQ(ierr);
  /* Copy H */
  for (i=0;i<r1;i++) {
    ierr = MatGetRow(tao->hessian,i,&ncols,&cols,&vals); CHKERRQ(ierr);
    ierr = MatSetValues(ipmP->K,1,&i,ncols,cols,vals,INSERT_VALUES); CHKERRQ(ierr);
    ierr = MatRestoreRow(tao->hessian,i,&ncols,&cols,&vals); CHKERRQ(ierr);
  }

  /*      Copy Ae and Ae' */
  for (i=r1;i<r2;i++) {
    ierr = MatGetRow(tao->jacobian_equality,i-r1,&ncols,&cols,&vals); CHKERRQ(ierr);
    /*Ae*/
    ierr = MatSetValues(ipmP->K,1,&i,ncols,cols,vals,INSERT_VALUES); CHKERRQ(ierr);
    /*Ae'*/
    for (j=0;j<ncols;j++) {
      newcol = i + c2;
      newrow = j;
      newval = vals[j];
      ierr = MatSetValues(ipmP->K,1,&newrow,1,&newcol,&newval,INSERT_VALUES); CHKERRQ(ierr);
    }
    ierr = MatRestoreRow(tao->jacobian_equality,i-r1,&ncols,&cols,&vals); CHKERRQ(ierr);
  }

  /* Copy Ai,and Ai' */
  for (i=r2;i<r3;i++) {
    ierr = MatGetRow(ipmP->Ai,i-r2,&ncols,&cols,&vals); CHKERRQ(ierr);
    /*Ai*/
    ierr = MatSetValues(ipmP->K,1,&i,ncols,cols,vals,INSERT_VALUES); CHKERRQ(ierr);
    /*-Ai'*/
    for (j=0;j<ncols;j++) {
      newcol = i + c3;
      newrow = j;
      newval = -vals[j];
      ierr = MatSetValues(ipmP->K,1,&newrow,1,&newcol,&newval,INSERT_VALUES); CHKERRQ(ierr);
    }
    ierr = MatRestoreRow(ipmP->Ai,i-r2,&ncols,&cols,&vals); CHKERRQ(ierr);
  }



  /* Ai_new = Ai_user(w/lb)
             -Ai_user(w/ub)
	      I (w/lb)
	      -I (w/ub) */

  /* -Ai_new' = -Ai_user(w/lb) | Ai_user(w/ub) | -I (w/lb) | I (w/ub) */
  ierr = IPMUpdateAi(tao); CHKERRQ(ierr);
  /* for i in isil */
  
  for (i=0;i<ipmP->nilb;i++) {
    /*  Ai_user(w/lb) */ 
    ierr = MatGetRow(tao->jacobian_inequality,indices[i],&ncols,&cols,&vals); CHKERRQ(ierr);
    newrow = i + r2;
    ierr = MatSetValues(ipmP->K,1,&newrow,ncols,cols,vals,INSERT_VALUES); CHKERRQ(ierr);
    
    /*  -Ai_user(w/lb)' */
    for (j=0;i<ncols;j++) {
      newrow = j;
      newcol = r3+i;
      newval = -vals[j];
      ierr = MatSetValues(ipmP->K,1,&newrow,1,&newcol,&newval,INSERT_VALUES); CHKERRQ(ierr);
    }
    ierr = MatRestoreRow(tao->jacobian_inequality,indices[i],&ncols,&cols,&vals); CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(ipmP->isil,&indices); CHKERRQ(ierr);

  
    
  /* -I */
  for (i=0;i<ipmP->nb;i++) {
    newrow = r2+i;
    newcol = c1+i;
    newval = -1.0;
    MatSetValues(ipmP->K,1,&newrow,1,&newcol,&newval,INSERT_VALUES); CHKERRQ(ierr);
  }

  /* Copy L,Y */
  ierr = VecGetArray(ipmP->lamdai,&l); CHKERRQ(ierr);
  ierr = VecGetArray(ipmP->s,&y); CHKERRQ(ierr);

  for (i=0;i<ipmP->nb;i++) {
    newcols[0] = c1+i;
    newcols[1] = c3+i;
    newvals[0] = l[i];
    newvals[1] = y[i];
    newrow = r3+i;
    ierr = MatSetValues(ipmP->K,1,&newrow,2,newcols,newvals,INSERT_VALUES); CHKERRQ(ierr);
  }
      
  ierr = VecRestoreArray(ipmP->lamdai,&l); CHKERRQ(ierr);
  ierr = VecRestoreArray(ipmP->s,&y); CHKERRQ(ierr);
      
  ierr = PetscFree(indices); CHKERRQ(ierr);
  ierr = PetscFree(newvals); CHKERRQ(ierr);
  ierr = MatAssemblyBegin(ipmP->K,MAT_FINAL_ASSEMBLY);
  ierr = MatAssemblyEnd(ipmP->K,MAT_FINAL_ASSEMBLY);
#if defined DEBUG_K  
  printf("K\n");  MatView(ipmP->K,0);
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "IPMGatherRHS"
PetscErrorCode IPMGatherRHS(TaoSolver tao,Vec RHS,Vec X1,Vec X2,Vec X3,Vec X4)
{
  TAO_IPM *ipmP = (TAO_IPM *)tao->data;
  PetscScalar *x,*rhs;
  PetscInt i;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  /* rhs = [x1      (n)
            x2     (me)
	    x3     (nb)
	    x4     (nb)] */
  ierr = VecGetArray(RHS,&rhs); CHKERRQ(ierr);
  ierr = VecGetArray(X1,&x); CHKERRQ(ierr);
  for (i=0;i<ipmP->n;i++) {
    rhs[i] = x[i];
  }
  ierr = VecRestoreArray(X1,&x); CHKERRQ(ierr);


  ierr = VecGetArray(X2,&x); CHKERRQ(ierr);
  for (i=0;i<ipmP->me;i++) {
    rhs[i+ipmP->n] = x[i];
  }
  ierr = VecRestoreArray(X2,&x); CHKERRQ(ierr);

  ierr = VecGetArray(X3,&x); CHKERRQ(ierr);
  for (i=0;i<ipmP->nb;i++) {
    rhs[i+ipmP->n+ipmP->me] = x[i];
  }
  ierr = VecRestoreArray(X3,&x); CHKERRQ(ierr);

  /*lamdai*/
  ierr = VecGetArray(X4,&x); CHKERRQ(ierr);
  for (i=0;i<ipmP->nb;i++) {
    rhs[i+ipmP->n+ipmP->me+ipmP->nb] = x[i];
  }
  ierr = VecRestoreArray(X4,&x); CHKERRQ(ierr);
  ierr = VecRestoreArray(RHS,&rhs); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}



#undef __FUNCT__
#define __FUNCT__ "IPMScatterStep"
PetscErrorCode IPMScatterStep(TaoSolver tao, Vec STEP, Vec X1, Vec X2, Vec X3, Vec X4)
{
  TAO_IPM *ipmP = (TAO_IPM *)tao->data;
  PetscScalar *x,*step;
  PetscInt i;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  /*        [x1    (n)
	     x2    (nb)
 	     x3    (me)
	     x4    (nb) */
  ierr = VecGetArray(STEP,&step); CHKERRQ(ierr);
  /*x*/
  ierr = VecGetArray(X1,&x); CHKERRQ(ierr);
  for (i=0;i<ipmP->n;i++) {
    x[i] = step[i];
  }
  ierr = VecRestoreArray(X1,&x); CHKERRQ(ierr);

  ierr = VecGetArray(X2,&x); CHKERRQ(ierr);
  for (i=0;i<ipmP->nb;i++) {
    x[i] = step[i+ipmP->n];
  }
  ierr = VecRestoreArray(X2,&x); CHKERRQ(ierr);


  ierr = VecGetArray(X3,&x); CHKERRQ(ierr);
  for (i=0;i<ipmP->me;i++) {
    x[i] = step[i+ipmP->n+ipmP->nb];
  }
  ierr = VecRestoreArray(X3,&x); CHKERRQ(ierr);


  ierr = VecGetArray(X4,&x); CHKERRQ(ierr);
  for (i=0;i<ipmP->nb;i++) {
    x[i] = step[i+ipmP->n+ipmP->nb+ipmP->me];
  }
  ierr = VecRestoreArray(X4,&x); CHKERRQ(ierr);
  ierr = VecRestoreArray(STEP,&step); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


EXTERN_C_BEGIN

#undef __FUNCT__
#define __FUNCT__ "TaoCreate_IPM"
PetscErrorCode TaoCreate_IPM(TaoSolver tao)
{
  TAO_IPM *ipmP;
  //  const char *ipmls_type = TAOLINESEARCH_IPM;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  tao->ops->setup = TaoSetup_IPM;
  tao->ops->solve = TaoSolve_IPM;
  tao->ops->view = TaoView_IPM;
  tao->ops->setfromoptions = TaoSetFromOptions_IPM;
  tao->ops->destroy = TaoDestroy_IPM;
  //tao->ops->computedual = TaoComputeDual_IPM;

  ierr = PetscNewLog(tao, TAO_IPM, &ipmP); CHKERRQ(ierr);
  tao->data = (void*)ipmP;
  tao->max_it = 200;
  tao->max_funcs = 500;
  tao->fatol = 1e-4;
  tao->frtol = 1e-4;
  ipmP->dec = 10000; /* line search critera */
  ipmP->taumin = 0.995;
  ipmP->monitorkkt = PETSC_FALSE;
  ipmP->pushs = 1000;
  ipmP->pushlam = 1000;
  /*
  ierr = TaoLineSearchCreate(((PetscObject)tao)->comm, &tao->linesearch); CHKERRQ(ierr);
  ierr = TaoLineSearchSetType(tao->linesearch, ipmls_type); CHKERRQ(ierr);
  ierr = TaoLineSearchSetObjectiveRoutine(tao->linesearch, IPMObjective, tao); CHKERRQ(ierr);
  */
  ierr = KSPCreate(((PetscObject)tao)->comm, &tao->ksp); CHKERRQ(ierr);
  PetscFunctionReturn(0);
  

}
EXTERN_C_END

