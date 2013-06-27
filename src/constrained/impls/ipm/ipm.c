#include "taolinesearch.h"
#include "ipm.h" /*I "ipm.h" I*/
//#define DEBUG_K
//#define DEBUG_KKT
/* 
   x,d in R^n
   f in R
   nb = mi + nlb+nub
   s in R^nb is slack vector CI(x) / x-XL / -x+XU
   bin in R^mi (tao->constraints_inequality)
   beq in R^me (tao->constraints_equality)
   lamdai in R^nb (ipmP->lamdai)
   lamdae in R^me (ipmP->lamdae)
   Jeq in R^(me x n) (tao->jacobian_equality)
   Jin in R^(mi x n) (tao->jacobian_inequality)
   Ai in  R^(nb x n) (ipmP->Ai)
   H in R^(n x n) (tao->hessian)
   min f=(1/2)*x'*H*x + d'*x   
   s.t.  CE(x) == 0
         CI(x) >= 0
	 x >= tao->XL
	 -x >= -tao->XU
*/

static PetscErrorCode IPMComputeKKT(TaoSolver tao);
static PetscErrorCode IPMPushInitialPoint(TaoSolver tao);
static PetscErrorCode IPMEvaluate(TaoSolver tao);
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
  PetscInt iter = 0,its,i,j;
  PetscScalar stepsize=1.0;
  PetscScalar *li,*di;
  PetscScalar step_s,step_l,alpha,tau,sigma,phi_target;
  PetscFunctionBegin;

  /* Push initial point away from bounds */
  ierr = VecSet(tao->solution,0.0); CHKERRQ(ierr);
  ierr = VecNorm(tao->solution,NORM_2,&tau); CHKERRQ(ierr);
  printf("||x0|| = %g\n",tau);
  ierr = IPMInitializeBounds(tao); CHKERRQ(ierr);
  ierr = IPMPushInitialPoint(tao); CHKERRQ(ierr);
  ierr = VecNorm(tao->solution,NORM_2,&tau); CHKERRQ(ierr);
  printf("||x0|| = %g\n",tau);
  ierr = VecCopy(tao->solution,ipmP->rhs_x); CHKERRQ(ierr);
  ierr = IPMEvaluate(tao); CHKERRQ(ierr);
  ierr = IPMComputeKKT(tao); CHKERRQ(ierr);
  ierr = TaoMonitor(tao,iter++,ipmP->kkt_f,ipmP->phi,0.0,1.0,&reason);


  while (reason == TAO_CONTINUE_ITERATING) {
    ierr = IPMUpdateK(tao); CHKERRQ(ierr);
    /* 
       rhs.x    = -rd
       rhs.lame = -rpe
       rhs.lami = -rpi
       rhs.com  = -com 
    */

    ierr = VecCopy(ipmP->rd,ipmP->rhs_x); CHKERRQ(ierr);
    if (ipmP->me > 0) {
      ierr = VecCopy(ipmP->rpe,ipmP->rhs_lamdae); CHKERRQ(ierr);
    }
    if (ipmP->nb > 0) {
	ierr = VecCopy(ipmP->rpi,ipmP->rhs_lamdai); CHKERRQ(ierr);
	ierr = VecCopy(ipmP->complementarity,ipmP->rhs_s); CHKERRQ(ierr);
    }
    ierr = IPMGatherRHS(tao,ipmP->bigrhs,ipmP->rhs_x,ipmP->rhs_lamdae,
			ipmP->rhs_lamdai,ipmP->rhs_s); CHKERRQ(ierr);
    ierr = VecScale(ipmP->bigrhs,-1.0); CHKERRQ(ierr);

    /* solve K * step = rhs */
    ierr = KSPSetOperators(tao->ksp,ipmP->K,ipmP->K,DIFFERENT_NONZERO_PATTERN); CHKERRQ(ierr);
    ierr = KSPSolve(tao->ksp,ipmP->bigrhs,ipmP->bigstep);CHKERRQ(ierr);

    ierr = IPMScatterStep(tao,ipmP->bigstep,tao->stepdirection,ipmP->ds,
			  ipmP->dlamdae,ipmP->dlamdai); CHKERRQ(ierr);
    ierr = KSPGetIterationNumber(tao->ksp,&its); CHKERRQ(ierr);
    tao->ksp_its += its;
#if defined DEBUG_KKT
    printf("first solve.\n");
    printf("rhs_lamdai\n");
    //VecView(ipmP->rhs_lamdai,0);
    //ierr = VecView(ipmP->bigrhs,0);
    //ierr = VecView(ipmP->bigstep,0);
    PetscScalar norm1,norm2;
    ierr = VecNorm(ipmP->bigrhs,NORM_2,&norm1);
    ierr = VecNorm(ipmP->bigstep,NORM_2,&norm2);
    printf("||rhs|| = %g\t ||step|| = %g\n",norm1,norm2);
#endif
     /* Find distance along step direction to closest bound */
    if (ipmP->nb > 0) {
      ierr = VecStepBoundInfo(ipmP->s,ipmP->Zero_nb,ipmP->Inf_nb,ipmP->ds,&step_s,PETSC_NULL,PETSC_NULL); CHKERRQ(ierr);
      ierr = VecStepBoundInfo(ipmP->lamdai,ipmP->Zero_nb,ipmP->Inf_nb,ipmP->dlamdai,&step_l,PETSC_NULL,PETSC_NULL); CHKERRQ(ierr);
      alpha = PetscMin(step_s,step_l);
      alpha = PetscMin(alpha,1.0);
      ipmP->alpha1 = alpha;
    } else {
      ipmP->alpha1 = alpha = 1.0;
    }

    
    /* x_aff = x + alpha*d */
    ierr = VecCopy(tao->solution,ipmP->save_x); CHKERRQ(ierr);
    if (ipmP->me > 0) {
      ierr = VecCopy(ipmP->lamdae,ipmP->save_lamdae); CHKERRQ(ierr);
    }
    if (ipmP->nb > 0) {
      ierr = VecCopy(ipmP->lamdai,ipmP->save_lamdai); CHKERRQ(ierr);
      ierr = VecCopy(ipmP->s,ipmP->save_s); CHKERRQ(ierr);
    }

    ierr = VecAXPY(tao->solution,alpha,tao->stepdirection); CHKERRQ(ierr);
    if (ipmP->me > 0) {
      ierr = VecAXPY(ipmP->lamdae,alpha,ipmP->dlamdae); CHKERRQ(ierr);
    }
    if (ipmP->nb > 0) {
      ierr = VecAXPY(ipmP->lamdai,alpha,ipmP->dlamdai); CHKERRQ(ierr);
      ierr = VecAXPY(ipmP->s,alpha,ipmP->ds); CHKERRQ(ierr);
    }

    
    /* Recompute kkt to find centering parameter sigma = (new_mu/old_mu)^3 */
    if (ipmP->mu == 0.0) {
      sigma = 0.0;
    } else {
      sigma = 1.0/ipmP->mu;
    }
    ierr = IPMComputeKKT(tao); CHKERRQ(ierr);
    sigma *= ipmP->mu;
    sigma*=sigma*sigma;
    
    /* revert kkt info */
    ierr = VecCopy(ipmP->save_x,tao->solution); CHKERRQ(ierr);
    if (ipmP->me > 0) {
      ierr = VecCopy(ipmP->save_lamdae,ipmP->lamdae); CHKERRQ(ierr);
    }
    if (ipmP->nb > 0) {
      ierr = VecCopy(ipmP->save_lamdai,ipmP->lamdai); CHKERRQ(ierr);
      ierr = VecCopy(ipmP->save_s,ipmP->s); CHKERRQ(ierr);
    }
    ierr = IPMComputeKKT(tao); CHKERRQ(ierr);

    /* update rhs with new complementarity vector */
    if (ipmP->nb > 0) {
      ierr = VecCopy(ipmP->complementarity,ipmP->rhs_s); CHKERRQ(ierr);
      ierr = VecScale(ipmP->rhs_s,-1.0); CHKERRQ(ierr);
      ierr = VecShift(ipmP->rhs_s,sigma*ipmP->mu); CHKERRQ(ierr);
    }
    ierr = IPMGatherRHS(tao,ipmP->bigrhs,PETSC_NULL,PETSC_NULL,
		      PETSC_NULL,ipmP->rhs_s); CHKERRQ(ierr);

    /* solve K * step = rhs */
    ierr = KSPSetOperators(tao->ksp,ipmP->K,ipmP->K,DIFFERENT_NONZERO_PATTERN); CHKERRQ(ierr);
    ierr = KSPSolve(tao->ksp,ipmP->bigrhs,ipmP->bigstep);CHKERRQ(ierr);

    ierr = IPMScatterStep(tao,ipmP->bigstep,tao->stepdirection,ipmP->ds,
			  ipmP->dlamdae,ipmP->dlamdai); CHKERRQ(ierr);
    ierr = KSPGetIterationNumber(tao->ksp,&its); CHKERRQ(ierr);
#if defined DEBUG_KKT2
    printf("rhs_lamdai\n");
    VecView(ipmP->rhs_lamdai,0);
    ierr = VecView(ipmP->bigrhs,0);
    ierr = VecView(ipmP->bigstep,0);
#endif
    tao->ksp_its += its;
    

    if (ipmP->nb > 0) {
      /* Get max step size and apply frac-to-boundary */
      tau = PetscMax(ipmP->taumin,1.0-ipmP->mu);
      tau = PetscMin(tau,1.0);
      if (tau != 1.0) {
	ierr = VecScale(ipmP->s,tau); CHKERRQ(ierr);
	ierr = VecScale(ipmP->lamdai,tau); CHKERRQ(ierr);
      }
      ierr = VecStepBoundInfo(ipmP->s,ipmP->Zero_nb,ipmP->Inf_nb,ipmP->ds,&step_s,PETSC_NULL,PETSC_NULL); CHKERRQ(ierr);
      ierr = VecStepBoundInfo(ipmP->lamdai,ipmP->Zero_nb,ipmP->Inf_nb,ipmP->dlamdai,&step_l,PETSC_NULL,PETSC_NULL); CHKERRQ(ierr);
      if (tau != 1.0) {
	ierr = VecCopy(ipmP->save_s,ipmP->s); CHKERRQ(ierr);
	ierr = VecCopy(ipmP->save_lamdai,ipmP->lamdai); CHKERRQ(ierr);
      }
      alpha = PetscMin(step_s,step_l);
      alpha = PetscMin(alpha,1.0);
    } else {
      alpha = 1.0;
    }
    ipmP->alpha2 = alpha;
    /* TODO make phi_target meaningful */
    phi_target = ipmP->dec * ipmP->phi;
    for (i=0; i<11;i++) {
#if defined DEBUG_KKT2      
    printf("alpha2=%g\n",alpha);
      printf("old point:\n");
      VecView(tao->solution,0);
      VecView(ipmP->lamdae,0);
      VecView(ipmP->s,0);
      VecView(ipmP->lamdai,0);
#endif
      ierr = VecAXPY(tao->solution,alpha,tao->stepdirection); CHKERRQ(ierr);
      if (ipmP->nb > 0) {
	ierr = VecAXPY(ipmP->s,alpha,ipmP->ds); CHKERRQ(ierr);
	ierr = VecAXPY(ipmP->lamdai,alpha,ipmP->dlamdai); CHKERRQ(ierr);
      }
      if (ipmP->me > 0) {
	ierr = VecAXPY(ipmP->lamdae,alpha,ipmP->dlamdae); CHKERRQ(ierr);
      }
#if defined DEBUG_KKT
      printf("step direction:\n");
      VecView(tao->stepdirection,0);
      //VecView(ipmP->dlamdae,0);
      //VecView(ipmP->ds,0);
      //VecView(ipmP->dlamdai,0);
	
      //printf("New iterate:\n");
      //VecView(tao->solution,0);
      //VecView(ipmP->lamdae,0);
      //VecView(ipmP->s,0);
      //VecView(ipmP->lamdai,0);
#endif      
      /* update dual variables */
      if (ipmP->me > 0) {
	ierr = VecCopy(ipmP->lamdae,tao->DE); CHKERRQ(ierr);
      }
      if (ipmP->nb > 0) {
	ierr = VecGetArray(ipmP->lamdai,&li); CHKERRQ(ierr);
	ierr = VecGetArray(tao->DI,&di); CHKERRQ(ierr);
	for (j=0;j<ipmP->nilb;j++) {
	  di[j] = li[j];
	}
	ierr = VecRestoreArray(ipmP->lamdai,&li); CHKERRQ(ierr);
	ierr = VecRestoreArray(tao->DI,&di); CHKERRQ(ierr);
      }
      

      ierr = IPMEvaluate(tao); CHKERRQ(ierr);
      ierr = IPMComputeKKT(tao); CHKERRQ(ierr);
      if (ipmP->phi <= phi_target) break; 
      alpha /= 2.0;
    }

    ierr = TaoMonitor(tao,iter,ipmP->kkt_f,ipmP->phi,0.0,stepsize,&reason);
    iter++;
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
    ierr = VecDuplicate(tao->solution, &ipmP->save_x); CHKERRQ(ierr);
  }
  
  if (tao->constraints_equality) {
    ierr = VecGetSize(tao->constraints_equality,&ipmP->me); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_equality,&ipmP->lamdae); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_equality,&ipmP->dlamdae); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_equality,&ipmP->rhs_lamdae); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_equality,&ipmP->save_lamdae); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_equality,&ipmP->rpe); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_equality,&tao->DE); CHKERRQ(ierr);
  }
  if (tao->constraints_inequality) {
    ierr = VecDuplicate(tao->constraints_inequality,&tao->DI); CHKERRQ(ierr);
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


  ierr = VecDuplicate(tao->solution,&xtmp); CHKERRQ(ierr);
  if (!tao->XL && !tao->XU && tao->ops->computebounds) {
    ierr = TaoComputeVariableBounds(tao);
  }
  if (tao->XL) {
    ierr = VecSet(xtmp,TAO_NINFINITY); CHKERRQ(ierr);
    ierr = VecWhichGreaterThan(tao->XL,xtmp,&ipmP->isxl); CHKERRQ(ierr);
    ierr = ISGetSize(ipmP->isxl,&ipmP->nxlb); CHKERRQ(ierr);
  } else {
    ipmP->nxlb=0;
  }
  if (tao->XU) {
    ierr = VecSet(xtmp,TAO_INFINITY); CHKERRQ(ierr);
    ierr = VecWhichLessThan(tao->XU,xtmp,&ipmP->isxu); CHKERRQ(ierr);
    ierr = ISGetSize(ipmP->isxu,&ipmP->nxub); CHKERRQ(ierr);
  } else {
    ipmP->nxub=0;
  }
  ierr = VecDestroy(&xtmp); CHKERRQ(ierr);
  ipmP->niub = 0;
  if (tao->constraints_inequality) {
    ierr = VecGetSize(tao->constraints_inequality,&ipmP->mi); CHKERRQ(ierr);
    ipmP->nilb = ipmP->mi;
  } else {
    ipmP->nilb = ipmP->niub = ipmP->mi = 0;
  }
#if defined DEBUG_K
  printf("isxl:\n");
  if (ipmP->nxlb) {
    ISView(ipmP->isxl,0);  
  }
  printf("isxu:\n");
  if (ipmP->nxub) {
    ISView(ipmP->isxu,0);  
  }
#endif  
  ipmP->nb = ipmP->nxlb + ipmP->nxub + ipmP->nilb + ipmP->niub;
  printf("nvar=%d,ni=%d,nb=%d,ne=%d\n",ipmP->n,ipmP->nilb,ipmP->nb,ipmP->me);
  

  if (ipmP->nb > 0) {
    comm = ((PetscObject)(tao->solution))->comm;
    ierr = VecCreate(comm,&ipmP->s); CHKERRQ(ierr);
    ierr = VecSetSizes(ipmP->s,PETSC_DECIDE,ipmP->nb); CHKERRQ(ierr);
    ierr = VecSetFromOptions(ipmP->s); CHKERRQ(ierr);
    ierr = VecDuplicate(ipmP->s,&ipmP->ds); CHKERRQ(ierr);
    ierr = VecDuplicate(ipmP->s,&ipmP->rhs_s); CHKERRQ(ierr);
    ierr = VecDuplicate(ipmP->s,&ipmP->complementarity); CHKERRQ(ierr);
    ierr = VecDuplicate(ipmP->s,&ipmP->ci); CHKERRQ(ierr);

    ierr = VecDuplicate(ipmP->s,&ipmP->lamdai); CHKERRQ(ierr);
    ierr = VecDuplicate(ipmP->s,&ipmP->dlamdai); CHKERRQ(ierr);
    ierr = VecDuplicate(ipmP->s,&ipmP->rhs_lamdai); CHKERRQ(ierr);
    ierr = VecDuplicate(ipmP->s,&ipmP->save_lamdai); CHKERRQ(ierr);


    ierr = VecDuplicate(ipmP->s,&ipmP->save_s); CHKERRQ(ierr);
    ierr = VecDuplicate(ipmP->s,&ipmP->rpi); CHKERRQ(ierr);
    ierr = VecDuplicate(ipmP->s,&ipmP->Zero_nb); CHKERRQ(ierr);
    ierr = VecSet(ipmP->Zero_nb,0.0); CHKERRQ(ierr);
    ierr = VecDuplicate(ipmP->s,&ipmP->One_nb); CHKERRQ(ierr);
    ierr = VecSet(ipmP->One_nb,1.0); CHKERRQ(ierr);
    ierr = VecDuplicate(ipmP->s,&ipmP->Inf_nb); CHKERRQ(ierr);
    ierr = VecSet(ipmP->Inf_nb,TAO_INFINITY); CHKERRQ(ierr);
    ierr = VecDuplicate(ipmP->s,&ipmP->worknb); CHKERRQ(ierr);


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
  ierr = VecDestroy(&ipmP->ci); CHKERRQ(ierr);

  ierr = VecDestroy(&ipmP->rhs_x); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->rhs_lamdae); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->rhs_lamdai); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->rhs_s); CHKERRQ(ierr);

  ierr = VecDestroy(&ipmP->save_x); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->save_lamdae); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->save_lamdai); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->save_s); CHKERRQ(ierr);

  
  ierr = VecDestroy(&ipmP->dlamdae); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->dlamdai); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->ds); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->complementarity); CHKERRQ(ierr);

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
  ierr = PetscOptionsReal("-ipm_pushnu","parameter to push initial (inequality) dual variables away from bounds",PETSC_NULL,ipmP->pushnu,&ipmP->pushnu,&flg);
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
       Ai =   jac_ineq
	       I (w/lb)
	      -I (w/ub) 

   rpe = ce
   rpi = ci - s;
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
  ierr = VecCopy(tao->gradient,ipmP->rd); CHKERRQ(ierr);

  if (ipmP->me > 0) {
    /* rd = gradient + Ae'*lamdae */
    ierr = MatMultTranspose(tao->jacobian_equality,ipmP->lamdae,ipmP->work); CHKERRQ(ierr);
    ierr = VecAXPY(ipmP->rd, 1.0, ipmP->work); CHKERRQ(ierr);

#if defined DEBUG_KKT
    PetscPrintf(PETSC_COMM_WORLD,"\nAe.lamdae\n");
    ierr = VecView(ipmP->work,0);
#endif
    /* rpe = ce(x) */
    ierr = VecCopy(tao->constraints_equality,ipmP->rpe); CHKERRQ(ierr);

  }
  if (ipmP->nb > 0) {
    /* rd = rd - Ai'*lamdai */
    ierr = MatMultTranspose(ipmP->Ai,ipmP->lamdai,ipmP->work); CHKERRQ(ierr);
    ierr = VecAXPY(ipmP->rd, -1.0, ipmP->work); CHKERRQ(ierr);
#if defined DEBUG_KKT
    PetscPrintf(PETSC_COMM_WORLD,"\nAi\n");
    ierr = MatView(ipmP->Ai,0);
    PetscPrintf(PETSC_COMM_WORLD,"\nAi.lamdai\n");
    ierr = VecView(ipmP->work,0);
#endif
    /* rpi = cin - s */
    ierr = VecCopy(ipmP->ci,ipmP->rpi); CHKERRQ(ierr);
    ierr = VecAXPY(ipmP->rpi, -1.0, ipmP->s); CHKERRQ(ierr);

    /* com = s .* lami */
    ierr = VecPointwiseMult(ipmP->complementarity, ipmP->s,ipmP->lamdai); CHKERRQ(ierr);

  }
  /* phi = ||rd; rpe; rpi; com|| */
  ierr = VecDot(ipmP->rd,ipmP->rd,&norm); CHKERRQ(ierr);
  ipmP->phi = norm; 
  if (ipmP->me > 0 ) {
    ierr = VecDot(ipmP->rpe,ipmP->rpe,&norm); CHKERRQ(ierr);
    ipmP->phi += norm; 
  }
  if (ipmP->nb > 0) {
    ierr = VecDot(ipmP->rpi,ipmP->rpi,&norm); CHKERRQ(ierr);
    ipmP->phi += norm; 
    ierr = VecDot(ipmP->complementarity,ipmP->complementarity,&norm); CHKERRQ(ierr);
    ipmP->phi += norm; 
    /* mu = s'*lami/nb */
    ierr = VecDot(ipmP->s,ipmP->lamdai,&ipmP->mu); CHKERRQ(ierr);
    ipmP->mu /= ipmP->nb;
  } else {
    ipmP->mu = 1.0;
  }

  ipmP->phi = PetscSqrtScalar(ipmP->phi);
  if (ipmP->monitorkkt) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"obj=%G,\tphi = %G,\tmu=%G\talpha1=%G\talpha2=%G\n",ipmP->kkt_f,ipmP->phi,ipmP->mu,ipmP->alpha1,ipmP->alpha2);
  }
  CHKMEMQ;
#if defined DEBUG_KKT
  PetscPrintf(PETSC_COMM_WORLD,"\ngradient\n");
  ierr = VecView(tao->gradient,0);
  PetscPrintf(PETSC_COMM_WORLD,"\nrd\n");
  ierr = VecView(ipmP->rd,0);
  PetscPrintf(PETSC_COMM_WORLD,"\nrpe\n");
  ierr = VecView(ipmP->rpe,0);
  PetscPrintf(PETSC_COMM_WORLD,"\nrpi\n");
  ierr = VecView(ipmP->rpi,0);
  PetscPrintf(PETSC_COMM_WORLD,"\ncomplementarity\n");
  ierr = VecView(ipmP->complementarity,0);
#endif  
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "IPMEvaluate"
/* evaluate user info at current point */
PetscErrorCode IPMEvaluate(TaoSolver tao)
{
  TAO_IPM *ipmP = (TAO_IPM *)tao->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = TaoComputeObjectiveAndGradient(tao,tao->solution,&ipmP->kkt_f,tao->gradient); CHKERRQ(ierr);
  ierr = TaoComputeHessian(tao,tao->solution,&tao->hessian,&tao->hessian_pre,&ipmP->Hflag); CHKERRQ(ierr);
  
  if (ipmP->me > 0) {
    ierr = TaoComputeEqualityConstraints(tao,tao->solution,tao->constraints_equality);
    ierr = TaoComputeJacobianEquality(tao,tao->solution,&tao->jacobian_equality,&tao->jacobian_equality_pre,&ipmP->Aiflag); CHKERRQ(ierr);
  }
  if (ipmP->mi > 0) {
    ierr = TaoComputeInequalityConstraints(tao,tao->solution,tao->constraints_inequality);
    ierr = TaoComputeJacobianInequality(tao,tao->solution,&tao->jacobian_inequality,&tao->jacobian_inequality_pre,&ipmP->Aeflag); CHKERRQ(ierr);

  }
  if (ipmP->nb > 0) {
    /* Ai' =   jac_ineq | I (w/lb) | -I (w/ub)  */
    ierr = IPMUpdateAi(tao); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "IPMPushInitialPoint"
/* Push initial point away from bounds */
PetscErrorCode IPMPushInitialPoint(TaoSolver tao)
{
  TAO_IPM *ipmP = (TAO_IPM *)tao->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = TaoComputeVariableBounds(tao); CHKERRQ(ierr);
  if (tao->XL && tao->XU) {
    ierr = VecMedian(tao->XL, tao->solution, tao->XU, tao->solution); CHKERRQ(ierr);
  }
  if (ipmP->nb > 0) {
    ierr = VecSet(ipmP->s,ipmP->pushs); CHKERRQ(ierr);
    ierr = VecSet(ipmP->lamdai,ipmP->pushnu); CHKERRQ(ierr);
    ierr = VecSet(tao->DI,ipmP->pushnu); CHKERRQ(ierr);
  }
  if (ipmP->me > 0) {
    ierr = VecSet(tao->DE,1.0); CHKERRQ(ierr);
    ierr = VecSet(ipmP->lamdae,1.0); CHKERRQ(ierr);
  }


  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "IPMUpdateAi"
PetscErrorCode IPMUpdateAi(TaoSolver tao)
{
  /* Ai =     Ji
	      I (w/lb)
	     -I (w/ub) */

  /* Ci =    user->ci
             Xi - lb (w/lb)
	     -Xi + ub (w/ub)  */
	     
  TAO_IPM *ipmP = (TAO_IPM *)tao->data;
  MPI_Comm comm;
  PetscInt i;
  PetscScalar newval;
  PetscInt newrow,newcol,ncols;
  PetscScalar *xb,*x;
  const PetscScalar *vals;
  const PetscInt *cols;
  const PetscInt *ind;
  PetscInt *nonzeros;
  PetscInt r2,r3,r4;
  PetscScalar *ci,*userci;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  CHKMEMQ;
  r2 = ipmP->nilb;
  r3 = r2 + ipmP->nxlb;
  r4 = r3 + ipmP->nxub;
  
  if (!ipmP->nb) {
    PetscFunctionReturn(0);
  }
  CHKMEMQ;
  if (!ipmP->Ai) {
    comm = ((PetscObject)(tao->solution))->comm;
    ierr = PetscMalloc(ipmP->nb*sizeof(PetscInt),&nonzeros); CHKERRQ(ierr);
    for (i=0;i<ipmP->nilb;i++) {
      ierr = MatGetRow(tao->jacobian_inequality,i,&ncols,PETSC_NULL,PETSC_NULL); CHKERRQ(ierr);
      nonzeros[i] = ncols;
      ierr = MatRestoreRow(tao->jacobian_inequality,i,&ncols,PETSC_NULL,PETSC_NULL); CHKERRQ(ierr);
    }
    for (i=r2;i<r4;i++) {
      nonzeros[i] = 1;
    }
    CHKMEMQ;
    ierr = MatCreate(comm,&ipmP->Ai); CHKERRQ(ierr);
    ierr = MatSetType(ipmP->Ai,MATSEQAIJ); CHKERRQ(ierr);
    ierr = MatSetSizes(ipmP->Ai,PETSC_DECIDE,PETSC_DECIDE,ipmP->nb,ipmP->n);CHKERRQ(ierr);
    ierr = MatSetFromOptions(ipmP->Ai); CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(ipmP->Ai,PETSC_NULL,nonzeros);CHKERRQ(ierr);
    ierr = PetscFree(nonzeros);CHKERRQ(ierr);
  }
  ierr = MatZeroEntries(ipmP->Ai); CHKERRQ(ierr);

  /* Ai w/lb */
  if (ipmP->nilb) {
    for (i=0;i<ipmP->nilb;i++) {
      ierr = MatGetRow(tao->jacobian_inequality,i,&ncols,&cols,&vals); CHKERRQ(ierr);
      newrow = i;
      ierr = MatSetValues(ipmP->Ai,1,&newrow,ncols,cols,vals,INSERT_VALUES); CHKERRQ(ierr);
      ierr = MatRestoreRow(tao->jacobian_inequality,i,&ncols,&cols,&vals); CHKERRQ(ierr);
    }
  }
  

  /* I w/ xlb */
  if (ipmP->nxlb) {
    for (i=0;i<ipmP->nxlb;i++) {
      newrow = i+r2;
      newcol = i;
      newval = 1.0;
      ierr = MatSetValues(ipmP->Ai,1,&newrow,1,&newcol,&newval,INSERT_VALUES); CHKERRQ(ierr);
    }
  }
  if (ipmP->nxub) {
    /* I w/ xub */
    for (i=0;i<ipmP->nxub;i++) {
      newrow = i+r3;
    newcol = i;
    newval = -1.0;
    ierr = MatSetValues(ipmP->Ai,1,&newrow,1,&newcol,&newval,INSERT_VALUES); CHKERRQ(ierr);
    }
  }

  
  ierr = MatAssemblyBegin(ipmP->Ai,MAT_FINAL_ASSEMBLY);
  ierr = MatAssemblyEnd(ipmP->Ai,MAT_FINAL_ASSEMBLY);
  CHKMEMQ;

  ierr = VecSet(ipmP->ci,0.0); CHKERRQ(ierr);
  ierr = VecGetArray(ipmP->ci,&ci); CHKERRQ(ierr);
  ierr = VecGetArray(tao->solution,&x); CHKERRQ(ierr);

  /* user ci */
  if (ipmP->nilb > 0) {
    ierr = VecGetArray(tao->constraints_inequality,&userci); CHKERRQ(ierr);
    for (i=0;i<ipmP->nilb;i++) {
      ci[i] = userci[i];
    }
    ierr = VecRestoreArray(tao->constraints_inequality,&userci); CHKERRQ(ierr);
  }
  /* lower bounds on variables */
  if (ipmP->nxlb > 0) { 
    ierr = VecGetArray(tao->XL,&xb); CHKERRQ(ierr);
    ierr = ISGetIndices(ipmP->isxl,&ind); CHKERRQ(ierr);
    for (i=0;i<ipmP->nxlb;i++) {
      ci[i+ipmP->nilb] = x[ind[i]]-xb[ind[i]];
    }
    ierr = ISRestoreIndices(ipmP->isxl,&ind); CHKERRQ(ierr);
    ierr = VecRestoreArray(tao->XL,&xb); CHKERRQ(ierr);
  }
  /* upper bounds on variables */
  if (ipmP->nxub > 0) {
    ierr = VecGetArray(tao->XU,&xb); CHKERRQ(ierr);
    ierr = ISGetIndices(ipmP->isxu,&ind); CHKERRQ(ierr);
    for (i=0;i<ipmP->nxub;i++) {
      ci[i+ipmP->nilb+ipmP->nxlb] =- x[ind[i]] + xb[ind[i]];
    }
    ierr = ISRestoreIndices(ipmP->isxu,&ind); CHKERRQ(ierr);
    ierr = VecRestoreArray(tao->XU,&xb); CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(tao->solution,&x); CHKERRQ(ierr);
  ierr = VecRestoreArray(ipmP->ci,&ci); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "IPMUpdateK"
/* create K = [ Hlag , 0 , Ae', -Ai']; 
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


  ierr = IPMUpdateAi(tao); CHKERRQ(ierr);
#if defined DEBUG_K
  printf("H\n");  MatView(tao->hessian,0);
  if (ipmP->nb) {
    printf("Ai\n"); MatView(ipmP->Ai,0);
  }
  if (ipmP->me) {
    printf("Ae\n"); MatView(tao->jacobian_equality,0);
  }

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
    ierr = MatSetType(ipmP->K,MATSEQAIJ); CHKERRQ(ierr);
    ierr = MatSetSizes(ipmP->K,PETSC_DECIDE,PETSC_DECIDE,bigsize,bigsize);CHKERRQ(ierr);
    ierr = MatSetFromOptions(ipmP->K); CHKERRQ(ierr);

    ierr = MatSeqAIJSetPreallocation(ipmP->K,PETSC_NULL,nonzeros); CHKERRQ(ierr);

    ierr = VecCreate(comm,&ipmP->bigrhs); CHKERRQ(ierr);
    ierr = VecSetType(ipmP->bigrhs,VECSEQ); CHKERRQ(ierr);
    ierr = VecSetSizes(ipmP->bigrhs,PETSC_DECIDE,bigsize); CHKERRQ(ierr);
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
      newcol = i -r1 + c2;
      newrow = cols[j];
      newval = vals[j];
      ierr = MatSetValues(ipmP->K,1,&newrow,1,&newcol,&newval,INSERT_VALUES); CHKERRQ(ierr);
    }
    ierr = MatRestoreRow(tao->jacobian_equality,i-r1,&ncols,&cols,&vals); CHKERRQ(ierr);
  }

  if (ipmP->nb > 0) {
    /* Copy Ai,and Ai' */
    for (i=r2;i<r3;i++) {
      ierr = MatGetRow(ipmP->Ai,i-r2,&ncols,&cols,&vals); CHKERRQ(ierr);
      /*Ai*/
      ierr = MatSetValues(ipmP->K,1,&i,ncols,cols,vals,INSERT_VALUES); CHKERRQ(ierr);
      /*-Ai'*/
      for (j=0;j<ncols;j++) {
	newcol = i - r2 + c3;
	newrow = cols[j];
	newval = -vals[j];
	ierr = MatSetValues(ipmP->K,1,&newrow,1,&newcol,&newval,INSERT_VALUES); CHKERRQ(ierr);
      }
      ierr = MatRestoreRow(ipmP->Ai,i-r2,&ncols,&cols,&vals); CHKERRQ(ierr);
    }



  
    
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
  }      

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
  if (X1) {
    ierr = VecGetArray(X1,&x); CHKERRQ(ierr);
    for (i=0;i<ipmP->n;i++) {
      rhs[i] = x[i];
    }
    ierr = VecRestoreArray(X1,&x); CHKERRQ(ierr);
  }


  if (X2) {
    ierr = VecGetArray(X2,&x); CHKERRQ(ierr);
    for (i=0;i<ipmP->me;i++) {
      rhs[i+ipmP->n] = x[i];
    }
    ierr = VecRestoreArray(X2,&x); CHKERRQ(ierr);
  }
  if (X3) {
    ierr = VecGetArray(X3,&x); CHKERRQ(ierr);
    for (i=0;i<ipmP->nb;i++) {
      rhs[i+ipmP->n+ipmP->me] = x[i];
    }
    ierr = VecRestoreArray(X3,&x); CHKERRQ(ierr);
  }
  if (X4) {
    ierr = VecGetArray(X4,&x); CHKERRQ(ierr);
    for (i=0;i<ipmP->nb;i++) {
      rhs[i+ipmP->n+ipmP->me+ipmP->nb] = x[i];
    }
    ierr = VecRestoreArray(X4,&x); CHKERRQ(ierr);
  }
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
  CHKMEMQ;
  /*        [x1    (n)
	     x2    (nb) may be 0
 	     x3    (me) may be 0
	     x4    (nb) may be 0 */
  
  ierr = VecGetArray(STEP,&step); CHKERRQ(ierr);
  /*x*/
  if (X1) {
    ierr = VecGetArray(X1,&x); CHKERRQ(ierr);
    for (i=0;i<ipmP->n;i++) {
      x[i] = step[i];
    }
    ierr = VecRestoreArray(X1,&x); CHKERRQ(ierr);
  }
  
  if (X2) {
    ierr = VecGetArray(X2,&x); CHKERRQ(ierr);
    for (i=0;i<ipmP->nb;i++) {
      x[i] = step[i+ipmP->n];
    }
    ierr = VecRestoreArray(X2,&x); CHKERRQ(ierr);
  }

  if (X3) {
    ierr = VecGetArray(X3,&x); CHKERRQ(ierr);
    for (i=0;i<ipmP->me;i++) {
      x[i] = step[i+ipmP->n+ipmP->nb];
    }
    ierr = VecRestoreArray(X3,&x); CHKERRQ(ierr);
  }

  if (X4) {
    ierr = VecGetArray(X4,&x); CHKERRQ(ierr);
    for (i=0;i<ipmP->nb;i++) {
      x[i] = step[i+ipmP->n+ipmP->nb+ipmP->me];
    }
    ierr = VecRestoreArray(X4,&x); CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(STEP,&step); CHKERRQ(ierr);
  CHKMEMQ;
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
  ipmP->pushs = 100;
  ipmP->pushnu = 100;
  /*
  ierr = TaoLineSearchCreate(((PetscObject)tao)->comm, &tao->linesearch); CHKERRQ(ierr);
  ierr = TaoLineSearchSetType(tao->linesearch, ipmls_type); CHKERRQ(ierr);
  ierr = TaoLineSearchSetObjectiveRoutine(tao->linesearch, IPMObjective, tao); CHKERRQ(ierr);
  */
  ierr = KSPCreate(((PetscObject)tao)->comm, &tao->ksp); CHKERRQ(ierr);
  PetscFunctionReturn(0);
  

}
EXTERN_C_END

