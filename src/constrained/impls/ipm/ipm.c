#include "taolinesearch.h"
#include "ipm.h" /*I "ipm.h" I*/
PetscErrorCode  MatNestSetVecType_Nest(Mat A,const VecType vtype);


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

//static PetscErrorCode IPMObjective(TaoLineSearch,Vec,PetscReal*,void*);
static PetscErrorCode IPMComputeKKT(TaoSolver tao);
static PetscErrorCode IPMFindInitialPoint(TaoSolver tao);
static PetscErrorCode IPMUpdateK(TaoSolver tao);
#undef __FUNCT__
#define __FUNCT__ "TaoSolve_IPM"
static PetscErrorCode TaoSolve_IPM(TaoSolver tao)
{
  PetscErrorCode ierr;
  TAO_IPM* ipmP = (TAO_IPM*)tao->data;

  
  TaoSolverTerminationReason reason = TAO_CONTINUE_ITERATING;
  //  TaoLineSearchTerminationReason ls_reason = TAOLINESEARCH_CONTINUE_ITERATING;
  PetscScalar normk,temp,normrhs;
  PetscInt iter = 0,its,i;
  PetscReal stepsize=1.0;
  PetscReal step_y,step_l,alpha,tau,sigma,muaff,phi_target;
  PetscFunctionBegin;

  //  ierr = TaoComputeVariableBounds(tao); CHKERRQ(ierr);
  //  ierr = VecMedian(tao->XL, tao->solution, tao->XU, tao->solution); CHKERRQ(ierr);
  //  ierr = TaoLineSearchSetVariableBounds(tao->linesearch,tao->XL,tao->XU); CHKERRQ(ierr);
  ierr = IPMFindInitialPoint(tao); CHKERRQ(ierr);
  ierr = VecCopy(tao->solution,ipmP->rhs_x); CHKERRQ(ierr);
  ierr = IPMComputeKKT(tao); CHKERRQ(ierr);

  while (reason == TAO_CONTINUE_ITERATING) {
    ierr = IPMUpdateK(tao); CHKERRQ(ierr);
    ierr = VecCopy(tao->solution,ipmP->Xold); CHKERRQ(ierr);
    ierr = VecCopy(tao->gradient,ipmP->Gold); CHKERRQ(ierr);

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
    ierr = VecCopy(ipmP->yi,ipmP->rhs_yi); CHKERRQ(ierr);

    ierr = VecPointwiseMult(ipmP->rhs_lamdai,ipmP->lamdai,ipmP->yi); CHKERRQ(ierr);
    ierr = VecScale(ipmP->rhs_lamdai,-1.0); CHKERRQ(ierr);
    ierr = VecCopy(ipmP->rd,ipmP->rhs_x); CHKERRQ(ierr);
    ierr = VecScale(ipmP->rhs_x,-1.0); CHKERRQ(ierr);
    ierr = VecCopy(ipmP->rpe,ipmP->rhs_lamdae); CHKERRQ(ierr);
    ierr = VecScale(ipmP->rhs_lamdae,-1.0); CHKERRQ(ierr);
    ierr = VecCopy(ipmP->rpi,ipmP->rhs_yi); CHKERRQ(ierr);
    ierr = VecScale(ipmP->rhs_yi,-1.0); CHKERRQ(ierr);

    //    ierr = VecView(ipmP->bigrhs,0); CHKERRQ(ierr);
    ierr = PetscObjectStateIncrease((PetscObject)ipmP->bigrhs);CHKERRQ(ierr);
    ierr = PetscObjectStateIncrease((PetscObject)ipmP->K);CHKERRQ(ierr);

    ierr = KSPSetOperators(tao->ksp,ipmP->K,ipmP->K,DIFFERENT_NONZERO_PATTERN); CHKERRQ(ierr);
    ierr = KSPSolve(tao->ksp,ipmP->bigrhs,ipmP->bigstep);CHKERRQ(ierr);
    
    ierr = VecNorm(ipmP->rhs_x,NORM_2,&temp); CHKERRQ(ierr);
    printf("||x||=%G\n",temp); CHKERRQ(ierr);
    ierr = VecNorm(ipmP->rhs_lamdae,NORM_2,&temp); CHKERRQ(ierr);
    printf("||lame||=%G\n",temp); CHKERRQ(ierr);
    ierr = VecNorm(ipmP->rhs_yi,NORM_2,&temp); CHKERRQ(ierr);
    printf("||yi||=%G\n",temp); CHKERRQ(ierr);
    ierr = VecNorm(ipmP->rhs_lamdai,NORM_2,&temp); CHKERRQ(ierr);
    printf("||lami||=%G\n",temp); CHKERRQ(ierr);
    ierr = MatNorm(tao->hessian,NORM_FROBENIUS,&normk); CHKERRQ(ierr);
    printf("||H||=%G\n",normk);
    normk*=normk;
    
    ierr = MatNorm(tao->jacobian_equality,NORM_FROBENIUS,&temp); CHKERRQ(ierr); 
    printf("||Ae||=%G\n",temp);
    normk+=2*temp*temp;
    ierr = MatNorm(tao->jacobian_inequality,NORM_FROBENIUS,&temp); CHKERRQ(ierr);
    printf("||Ai||=%G\n",temp);
    normk+=2*temp*temp;
    normk += ipmP->mi;
    ierr = MatNorm(ipmP->Y,NORM_FROBENIUS,&temp); CHKERRQ(ierr);
    normk += temp*temp;
    printf("||Y||=%G\n",temp);
    ierr = MatNorm(ipmP->L,NORM_FROBENIUS,&temp); CHKERRQ(ierr);
    normk += temp*temp;
    printf("||L||=%G\n",temp);
    normk = sqrt(normk);


    ierr = VecNorm(ipmP->bigrhs,NORM_2,&normrhs); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"||K||_fro = %10.6G  ||rhs|| = %10.6G\n",normk,normrhs); CHKERRQ(ierr);
    ierr = KSPGetIterationNumber(tao->ksp,&its); CHKERRQ(ierr);
    tao->ksp_its += its;
    /* check result */
    //printf("solution = \n");
    //ierr = VecView(ipmP->bigstep,0); CHKERRQ(ierr);
    
    ierr = VecCopy(ipmP->dyi,ipmP->dYaff); CHKERRQ(ierr);
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
    ierr = VecStepBoundInfo(ipmP->yi,ipmP->Zero_mi,ipmP->Inf_mi,ipmP->dyi,&step_y,PETSC_NULL,PETSC_NULL); CHKERRQ(ierr);
    
    ierr = VecStepBoundInfo(ipmP->lamdai,ipmP->Zero_mi,ipmP->Inf_mi,ipmP->dlamdai,&step_l,PETSC_NULL,PETSC_NULL); CHKERRQ(ierr);
    alpha = PetscMin(step_y,step_l);
    ierr = VecCopy(ipmP->yi,ipmP->Yaff); CHKERRQ(ierr);
    ierr = VecAXPY(ipmP->Yaff,alpha,ipmP->dyi); CHKERRQ(ierr);
    ierr = VecCopy(ipmP->lamdai,ipmP->Laff); CHKERRQ(ierr);
    ierr = VecAXPY(ipmP->Laff,alpha,ipmP->dlamdai); CHKERRQ(ierr);
    ierr = VecDot(ipmP->Yaff,ipmP->Laff,&muaff); CHKERRQ(ierr);
    muaff /= ipmP->mi; CHKERRQ(ierr);

    sigma = (muaff/ipmP->mu);
    sigma *= sigma*sigma;

    /* Compute full step */
    /* rhs.lami = rhs.lami - dYaff.*dLaff + sig*kkt.mu*e); */
    ierr = VecAXPY(ipmP->rhs_lamdai,sigma*ipmP->mu,ipmP->One_mi); CHKERRQ(ierr);
    ierr = VecPointwiseMult(ipmP->dYaff,ipmP->dLaff,ipmP->dYaff); CHKERRQ(ierr);
    ierr = VecAXPY(ipmP->rhs_lamdai, -1.0, ipmP->dYaff); CHKERRQ(ierr);
    /* step = feval(par.compute_step,par,iter,rhs); */
    
    ierr = VecNorm(ipmP->rhs_x,NORM_2,&temp); CHKERRQ(ierr);
    printf("||x||=%G\n",temp); CHKERRQ(ierr);
    ierr = VecNorm(ipmP->rhs_lamdae,NORM_2,&temp); CHKERRQ(ierr);
    printf("||lame||=%G\n",temp); CHKERRQ(ierr);
    ierr = VecNorm(ipmP->rhs_yi,NORM_2,&temp); CHKERRQ(ierr);
    printf("||yi||=%G\n",temp); CHKERRQ(ierr);
    ierr = VecNorm(ipmP->rhs_lamdai,NORM_2,&temp); CHKERRQ(ierr);
    printf("||lami||=%G\n",temp); CHKERRQ(ierr);
    ierr = MatNorm(tao->hessian,NORM_FROBENIUS,&normk); CHKERRQ(ierr);
    printf("||H||=%G\n",normk);
    normk*=normk;
    
    ierr = MatNorm(tao->jacobian_equality,NORM_FROBENIUS,&temp); CHKERRQ(ierr); 
    printf("||Ae||=%G\n",temp);
    normk+=2*temp*temp;
    ierr = MatNorm(tao->jacobian_inequality,NORM_FROBENIUS,&temp); CHKERRQ(ierr);
    printf("||Ai||=%G\n",temp);
    normk+=2*temp*temp;
    normk += ipmP->mi;
    ierr = MatNorm(ipmP->Y,NORM_FROBENIUS,&temp); CHKERRQ(ierr);
    normk += temp*temp;
    printf("||Y||=%G\n",temp);
    ierr = MatNorm(ipmP->L,NORM_FROBENIUS,&temp); CHKERRQ(ierr);
    normk += temp*temp;
    printf("||L||=%G\n",temp);
    normk = sqrt(normk);

    
    ierr = PetscObjectStateIncrease((PetscObject)ipmP->bigrhs);CHKERRQ(ierr);
    ierr = PetscObjectStateIncrease((PetscObject)ipmP->K);CHKERRQ(ierr);

    ierr = VecNorm(ipmP->bigrhs,NORM_2,&normrhs); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"||K||_fro = %10.6G  ||rhs|| = %10.6G\n",normk,normrhs); CHKERRQ(ierr);
    ierr = KSPSolve(tao->ksp,ipmP->bigrhs,ipmP->bigstep);CHKERRQ(ierr);
    ierr = KSPGetIterationNumber(tao->ksp,&its); CHKERRQ(ierr);
    tao->ksp_its += its;


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
    ierr = VecStepBoundInfo(ipmP->yi,ipmP->Zero_mi,ipmP->Inf_mi,ipmP->dyi,&step_y,PETSC_NULL,PETSC_NULL); CHKERRQ(ierr);
    ierr = VecStepBoundInfo(ipmP->lamdai,ipmP->Zero_mi,ipmP->Inf_mi,ipmP->dlamdai,&step_l,PETSC_NULL,PETSC_NULL); CHKERRQ(ierr);
    alpha = PetscMin(step_y,step_l);
    alpha *= tau;
    

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
      ierr = VecAXPY(ipmP->bigx,alpha,ipmP->bigstep); CHKERRQ(ierr);
      ierr = IPMComputeKKT(tao); CHKERRQ(ierr); CHKERRQ(ierr);
      if (ipmP->phi <= phi_target) break;
      alpha /= 2.0;
    }
    

    /*
    stepsize=1.0;
    ierr = TaoLineSearchApply(tao->linesearch,tao->solution,&f,
			      tao->gradient,tao->stepdirection,&stepsize,
			      &ls_reason); CHKERRQ(ierr);
    if (ls_reason != TAOLINESEARCH_SUCCESS &&
	ls_reason != TAOLINESEARCH_SUCCESS_USER) {
      f = fold;
      ierr = VecCopy(ipmP->Xold, tao->solution); CHKERRQ(ierr);
      ierr = VecCopy(ipmP->Gold, tao->gradient); CHKERRQ(ierr);
      stepsize = 0.0;
      reason = TAO_DIVERGED_LS_FAILURE;
      tao->reason = TAO_DIVERGED_LS_FAILURE;
      break;
    }
    ierr = IPMComputeKKT(tao); CHKERRQ(ierr);
    */
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
  PetscInt localmi, localme,i,lo,hi;
  PetscScalar one = 1.0, mone = -1.0;
  Mat kgrid[16];
  Vec vgrid[4];
  MPI_Comm comm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ipmP->mi = ipmP->me = 0;
  comm = ((PetscObject)(tao->solution))->comm;
  ierr = VecGetSize(tao->solution,&ipmP->n); CHKERRQ(ierr);
  if (!tao->gradient) {
    ierr = VecDuplicate(tao->solution, &tao->gradient); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->solution, &tao->stepdirection); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->solution, &ipmP->rd); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->solution, &ipmP->rhs_x); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->solution, &ipmP->work); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->solution, &ipmP->Xold); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->solution, &ipmP->Gold); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->solution, &ipmP->Ninf_n); CHKERRQ(ierr);
    ierr = VecSet(ipmP->Ninf_n,TAO_NINFINITY); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->solution, &ipmP->Inf_n); CHKERRQ(ierr);
    ierr = VecSet(ipmP->Inf_n,TAO_INFINITY); CHKERRQ(ierr);
  }
  
  if (tao->constraints_inequality) {
    ierr = VecGetSize(tao->constraints_inequality,&ipmP->mi); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_inequality,&ipmP->yi); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_inequality,&ipmP->lamdai); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_inequality,&ipmP->complementarity); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_inequality,&ipmP->dyi); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_inequality,&ipmP->rhs_yi); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_inequality,&ipmP->Yaff); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_inequality,&ipmP->dYaff); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_inequality,&ipmP->dlamdai); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_inequality,&ipmP->rhs_lamdai); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_inequality,&ipmP->Laff); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_inequality,&ipmP->dLaff); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_inequality,&ipmP->rpi); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_inequality,&ipmP->Zero_mi); CHKERRQ(ierr);
    ierr = VecSet(ipmP->Zero_mi,0.0); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_inequality,&ipmP->One_mi); CHKERRQ(ierr);
    ierr = VecSet(ipmP->One_mi,1.0); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_inequality,&ipmP->Inf_mi); CHKERRQ(ierr);
    ierr = VecSet(ipmP->Inf_mi,TAO_INFINITY); CHKERRQ(ierr);

  }
  if (tao->constraints_equality) {
    ierr = VecGetSize(tao->constraints_equality,&ipmP->me); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_equality,&ipmP->lamdae); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_equality,&ipmP->dlamdae); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_equality,&ipmP->rhs_lamdae); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_equality,&ipmP->rpe); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_equality,&ipmP->Zero_me); CHKERRQ(ierr);
    ierr = VecSet(ipmP->Zero_me,0.0); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_equality,&ipmP->Inf_me); CHKERRQ(ierr);
    ierr = VecSet(ipmP->Zero_me,TAO_INFINITY); CHKERRQ(ierr);
  }
  ierr = VecGetLocalSize(ipmP->yi,&localmi); CHKERRQ(ierr);
  ierr = VecGetLocalSize(ipmP->lamdae,&localme); CHKERRQ(ierr);
  if (ipmP->usenest){

    /* create K = [ H , Ae', 0, -Ai']; 
       [Ae , 0,   0  , 0];
       [Ai , 0, -Imi ,  0];  
       [ 0 , 0 ,  L,   Y ];  */

    ierr = MatCreateAIJ(comm,localmi,localmi,ipmP->mi,ipmP->mi,1,PETSC_NULL,
			0,PETSC_NULL,&ipmP->minusI); CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(ipmP->minusI,&lo,&hi); CHKERRQ(ierr);
    for (i=lo;i<hi;i++) {
      MatSetValues(ipmP->minusI,1,&i,1,&i,&mone,INSERT_VALUES); CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(ipmP->minusI,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(ipmP->minusI,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    ierr = MatCreateAIJ(comm,localmi,localmi,ipmP->mi,ipmP->mi,1,PETSC_NULL,
			0,PETSC_NULL,&ipmP->L); CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(ipmP->L,&lo,&hi); CHKERRQ(ierr);
    for (i=lo;i<hi;i++) {
      MatSetValues(ipmP->L,1,&i,1,&i,&one,INSERT_VALUES); CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(ipmP->L,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(ipmP->L,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);


    ierr = MatCreateAIJ(comm,localmi,localmi,ipmP->mi,ipmP->mi,1,PETSC_NULL,
			0,PETSC_NULL,&ipmP->Y); CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(ipmP->Y,&lo,&hi); CHKERRQ(ierr);
    for (i=lo;i<hi;i++) {
      MatSetValues(ipmP->Y,1,&i,1,&i,&one,INSERT_VALUES); CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(ipmP->Y,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(ipmP->Y,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    ierr = MatTranspose(tao->jacobian_inequality,MAT_INITIAL_MATRIX,&ipmP->mAi_T); CHKERRQ(ierr);
    ierr = MatScale(ipmP->mAi_T,-1.0); CHKERRQ(ierr);
    //  ierr = MatCreateTranspose(tao->jacobian_equality,&ipmP->Ae_T); CHKERRQ(ierr);
    ierr = MatTranspose(tao->jacobian_inequality,MAT_INITIAL_MATRIX,&ipmP->Ae_T); CHKERRQ(ierr);


    kgrid[0] = tao->hessian;
    kgrid[1] = ipmP->Ae_T;
    kgrid[2] = PETSC_NULL;
    kgrid[3] = ipmP->mAi_T;

    kgrid[4] = tao->jacobian_equality;
    kgrid[5] = kgrid[6] = kgrid[7] = PETSC_NULL;

    kgrid[8] = tao->jacobian_inequality;
    kgrid[9] = PETSC_NULL;
    kgrid[10] = ipmP->minusI;
    kgrid[11] = kgrid[12] = kgrid[13] = PETSC_NULL;
    kgrid[14] = ipmP->L;
    kgrid[15] = ipmP->Y;

    /*
      printf("Matrix Nesting:\n");
      for (i=0;i<16;i++) {
      printf("(%d,%d):\n",i/4,i%4);
      if (kgrid[i]) {
      MatView(kgrid[i],PETSC_VIEWER_STDOUT_SELF);
      } else {
      printf("ZERO\n");
      }

      }
    */
    /* essentially a matrix-free method. real slow */
    ierr = MatCreateNest(comm,4,PETSC_NULL,4,PETSC_NULL,kgrid,&ipmP->K); CHKERRQ(ierr);
    ierr = MatNestSetVecType_Nest(ipmP->K,VECNEST); CHKERRQ(ierr);

    vgrid[0] = tao->solution;
    vgrid[1] = ipmP->lamdae;
    vgrid[2] = ipmP->yi;
    vgrid[3] = ipmP->lamdai;
    ierr = VecCreateNest(comm,4,PETSC_NULL,vgrid,&ipmP->bigx); CHKERRQ(ierr);
    
    vgrid[0] = ipmP->rhs_x;
    vgrid[1] = ipmP->rhs_lamdae;
    vgrid[2] = ipmP->rhs_yi;
    vgrid[3] = ipmP->rhs_lamdai;
    ierr = VecCreateNest(comm,4,PETSC_NULL,vgrid,&ipmP->bigrhs); CHKERRQ(ierr);


  
    vgrid[0] = tao->stepdirection;
    vgrid[1] = ipmP->dlamdae;
    vgrid[2] = ipmP->dyi;
    vgrid[3] = ipmP->dlamdai;
    ierr = VecCreateNest(comm,4,PETSC_NULL,vgrid,&ipmP->bigstep); CHKERRQ(ierr);
  } else {
    PetscInt bigsize = ipmP->n+2*ipmP->mi+ipmP->me;
    PetscInt *indices;
    
    ierr = PetscMalloc(bigsize*sizeof(PetscInt),&indices);CHKERRQ(ierr);
    for (i=0;i<ipmP->n;i++) {
      indices[i] = ipmP->n+ipmP->me+ipmP->mi;
    }
    for (i=ipmP->n;i<ipmP->n+ipmP->me;i++) {
      indices[i] = ipmP->n;
    }
    for (i=ipmP->n+ipmP->mi;i<ipmP->n+ipmP->mi+ipmP->me;i++) {
      indices[i] = ipmP->n+1;
    }
    for (i=ipmP->n+ipmP->mi+ipmP->me;i<ipmP->n+ipmP->mi+ipmP->me+ipmP->mi;i++) {
      indices[i] = 2;
    }
    ierr = MatCreate(comm,&ipmP->K); CHKERRQ(ierr);
    ierr = MatSetSizes(ipmP->K,PETSC_DECIDE,PETSC_DECIDE,bigsize,bigsize);CHKERRQ(ierr);
    ierr = MatSetType(ipmP->K,MATSEQAIJ); CHKERRQ(ierr);
    ierr = MatSetFromOptions(ipmP->K); CHKERRQ(ierr);

    ierr = MatSeqAIJSetPreallocation(ipmP->K,PETSC_NULL,indices); CHKERRQ(ierr);

    ierr = VecCreate(comm,&ipmP->bigrhs); CHKERRQ(ierr);
    ierr = VecSetSizes(ipmP->bigrhs,PETSC_DECIDE,bigsize); CHKERRQ(ierr);
    ierr = VecSetType(ipmP->bigrhs,VECSEQ); CHKERRQ(ierr);
    ierr = VecDuplicate(ipmP->bigrhs,&ipmP->bigx); CHKERRQ(ierr);
    ierr = VecDuplicate(ipmP->bigrhs,&ipmP->bigstep); CHKERRQ(ierr);
    ierr = PetscFree(indices);CHKERRQ(ierr);
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
  ierr = VecDestroy(&ipmP->Xold); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->Gold); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->rd); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->rpe); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->rpi); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->work); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->lamdae); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->lamdai); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->yi); CHKERRQ(ierr);

  ierr = VecDestroy(&ipmP->rhs_x); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->rhs_lamdae); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->rhs_lamdai); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->rhs_yi); CHKERRQ(ierr);

  ierr = VecDestroy(&ipmP->dlamdae); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->dlamdai); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->dyi); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->complementarity); CHKERRQ(ierr);
  
  ierr = MatDestroy(&ipmP->K); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->bigrhs); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->bigx); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->bigstep); CHKERRQ(ierr);
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
  ierr = PetscOptionsBool("-ipm_nest","use nested matrices/vectors for K",PETSC_NULL,ipmP->usenest,&ipmP->usenest,&flg); CHKERRQ(ierr);
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
   rpe = Ae*x - be
   rpi = Ai*x - yi - bi
   mu = yi' * lami/mi;
   com = yi.*lami

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


    /* rpe = Ae*x - be */
    ierr = MatMult(tao->jacobian_equality,tao->solution,ipmP->rpe);CHKERRQ(ierr);
    ierr = VecAXPY(ipmP->rpe, -1.0, tao->constraints_equality); CHKERRQ(ierr);

  }
  if (ipmP->mi > 0) {
    ierr = TaoComputeInequalityConstraints(tao,tao->solution,tao->constraints_inequality);
    ierr = TaoComputeJacobianInequality(tao,tao->solution,&tao->jacobian_inequality,&tao->jacobian_inequality_pre,&ipmP->Aeflag); CHKERRQ(ierr);

    /* rd = rd + Ai'*lamdai */
    ierr = MatMultTranspose(tao->jacobian_inequality,ipmP->lamdai,ipmP->work); CHKERRQ(ierr);
    ierr = VecAXPY(ipmP->rd, -1.0, ipmP->work); CHKERRQ(ierr);

    /* rpi = Ai*x - yi - bi */
    ierr = MatMult(tao->jacobian_inequality,tao->solution,ipmP->rpi); CHKERRQ(ierr);
    ierr = VecAXPY(ipmP->rpi, -1.0, tao->constraints_inequality); CHKERRQ(ierr);
    ierr = VecAXPY(ipmP->rpi, -1.0, ipmP->yi); CHKERRQ(ierr);

    /* mu = yi'*lami/mi */
    ierr = VecDot(ipmP->yi,ipmP->lamdai,&ipmP->mu); CHKERRQ(ierr);
    ipmP->mu /= ipmP->mi;
    /* com = yi .* lami */
    ierr = VecPointwiseMult(ipmP->complementarity, ipmP->yi,ipmP->lamdai); CHKERRQ(ierr);
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
    ierr = PetscPrintf(PETSC_COMM_WORLD,"obj=%G,\tphi = %G,\tmu=%G\n",ipmP->kkt_f,ipmP->phi,ipmP->mu);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "IPMFindInitialPoint"
PetscErrorCode IPMFindInitialPoint(TaoSolver tao)
{
  TAO_IPM *ipmP = (TAO_IPM *)tao->data;
  PetscErrorCode ierr;
  //  ierr = IPMComputeKKT(tao); CHKERRQ(ierr);
  ierr = VecSet(tao->solution,1.0); CHKERRQ(ierr);
  ierr = VecSet(ipmP->yi,1000.0); CHKERRQ(ierr);
  ierr = VecSet(ipmP->lamdae,1.0); CHKERRQ(ierr);
  ierr = VecSet(ipmP->lamdai,1000.0); CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "IPMUpdateK"
/* create K = [ H , Ae', 0, -Ai']; 
              [Ae , 0,   0  , 0];
              [Ai , 0, -Imi ,  0];  
              [ 0 , 0 ,  L,   Y ];  */
PetscErrorCode IPMUpdateK(TaoSolver tao)
{
  TAO_IPM *ipmP = (TAO_IPM *)tao->data;
  PetscErrorCode ierr;
  PetscInt i,j;
  PetscInt ncols,row,newcols[2];
  const PetscInt *cols;
  const PetscReal *vals;
  PetscReal *l,*y;
  PetscReal *newvals;
  PetscReal minus1=-1.0;
  PetscInt *indices;
  PetscInt subsize;
  PetscFunctionBegin;
  subsize = PetscMax(ipmP->n,ipmP->mi);
  subsize = PetscMax(ipmP->me,subsize);
  subsize = PetscMax(2,subsize);
  if (ipmP->usenest) {
    ierr = MatDiagonalSet(ipmP->L,ipmP->lamdai,INSERT_VALUES); CHKERRQ(ierr);
    ierr = MatDiagonalSet(ipmP->Y,ipmP->yi,INSERT_VALUES); CHKERRQ(ierr);
  } else {
    ierr = MatZeroEntries(ipmP->K); CHKERRQ(ierr);
    ierr = PetscMalloc(sizeof(PetscInt)*subsize,&indices); CHKERRQ(ierr);
    ierr = PetscMalloc(sizeof(PetscReal)*subsize,&newvals); CHKERRQ(ierr);
    /* Copy H */
    for (i=0;i<ipmP->n;i++) {
      ierr = MatGetRow(tao->hessian,i,&ncols,&cols,&vals); CHKERRQ(ierr);
      ierr = MatSetValues(ipmP->K,1,&i,ncols,cols,vals,INSERT_VALUES); CHKERRQ(ierr);
      ierr = MatRestoreRow(tao->hessian,i,&ncols,&cols,&vals); CHKERRQ(ierr);
    }

    /*      Copy Ae and Ae' */
    for (i=ipmP->n;i<ipmP->me+ipmP->n;i++) {
      ierr = MatGetRow(tao->jacobian_equality,i-ipmP->n,&ncols,&cols,&vals); CHKERRQ(ierr);
      for (j=0;j<ncols;j++) {
	indices[j] =cols[j] + ipmP->n;
      }
      ierr = MatSetValues(ipmP->K,1,&i,ncols,indices,vals,INSERT_VALUES); CHKERRQ(ierr);
      ierr = MatSetValues(ipmP->K,ncols,indices,1,&i,vals,INSERT_VALUES); CHKERRQ(ierr);
      ierr = MatRestoreRow(tao->jacobian_equality,i-ipmP->n,&ncols,&cols,&vals); CHKERRQ(ierr);
    }

    /* Copy Ai, -I, and Ai' */
    for (i=ipmP->me+ipmP->n;i<ipmP->me+ipmP->n+ipmP->mi;i++) {
      ierr = MatGetRow(tao->jacobian_inequality,i-ipmP->me-ipmP->n,&ncols,&cols,&vals); CHKERRQ(ierr);
      for (j=0;j<ncols;j++) {
	indices[j] =cols[j] + ipmP->n+ipmP->me;
      }
      ierr = MatSetValues(ipmP->K,1,&i,ncols,indices,vals,INSERT_VALUES); CHKERRQ(ierr);
      for (j=0;j<ncols;j++) {
	indices[j] = indices[j] + ipmP->me;
	newvals[j] = -vals[j];
      }
      ierr = MatSetValues(ipmP->K,ncols,indices,1,&i,newvals,INSERT_VALUES); CHKERRQ(ierr);
      ierr = MatRestoreRow(tao->jacobian_equality,i-ipmP->me-ipmP->n,&ncols,&cols,&vals); CHKERRQ(ierr);
      for (j=ipmP->n+ipmP->mi;j<ipmP->n+ipmP->mi;j++) {
	MatSetValues(ipmP->K,1,&i,1,&j,&minus1,INSERT_VALUES); CHKERRQ(ierr);
      }
    }

    /* Copy L,Y */
    ierr = VecGetArray(ipmP->lamdai,&l); CHKERRQ(ierr);
    ierr = VecGetArray(ipmP->yi,&y); CHKERRQ(ierr);
    for (i=0;i<ipmP->mi;i++) {
      newcols[0] = ipmP->n+ipmP->mi+i;
      newcols[1] = cols[0]+ipmP->me;
      newvals[0] = l[i];
      newvals[1] = y[i];
      row = newcols[1];
      ierr = MatSetValues(ipmP->K,1,&row,2,newcols,newvals,INSERT_VALUES); CHKERRQ(ierr);
    }
      
    ierr = VecRestoreArray(ipmP->lamdai,&l); CHKERRQ(ierr);
    ierr = VecRestoreArray(ipmP->yi,&y); CHKERRQ(ierr);
      
    ierr = PetscFree(indices); CHKERRQ(ierr);
    ierr = PetscFree(newvals); CHKERRQ(ierr);
    ierr = MatAssemblyBegin(ipmP->K,MAT_FINAL_ASSEMBLY);
    ierr = MatAssemblyEnd(ipmP->K,MAT_FINAL_ASSEMBLY);
  }
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
  tao->max_it = 2000;
  tao->max_funcs = 4000;
  tao->fatol = 1e-4;
  tao->frtol = 1e-4;
  ipmP->dec = 0.9; /* line search critera */
  ipmP->taumin = 0.995;
  ipmP->monitorkkt = PETSC_FALSE;
  ipmP->usenest = PETSC_FALSE;
  /*
  ierr = TaoLineSearchCreate(((PetscObject)tao)->comm, &tao->linesearch); CHKERRQ(ierr);
  ierr = TaoLineSearchSetType(tao->linesearch, ipmls_type); CHKERRQ(ierr);
  ierr = TaoLineSearchSetObjectiveRoutine(tao->linesearch, IPMObjective, tao); CHKERRQ(ierr);
  */
  ierr = KSPCreate(((PetscObject)tao)->comm, &tao->ksp); CHKERRQ(ierr);
  PetscFunctionReturn(0);
  

}
EXTERN_C_END

