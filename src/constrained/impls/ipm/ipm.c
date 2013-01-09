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

static PetscErrorCode IPMObjective(TaoLineSearch,Vec,PetscReal*,void*);
static PetscErrorCode IPMComputeKKT(TaoSolver tao);

#undef __FUNCT__
#define __FUNCT__ "TaoSolve_IPM"
static PetscErrorCode TaoSolve_IPM(TaoSolver tao)
{
  PetscErrorCode ierr;
  TAO_IPM* ipmP = (TAO_IPM*)tao->data;


  TaoSolverTerminationReason reason = TAO_CONTINUE_ITERATING;
  TaoLineSearchTerminationReason ls_reason = TAOLINESEARCH_CONTINUE_ITERATING;
  PetscInt iter = 0,its;
  PetscReal stepsize=1.0,f,fold;
  
  PetscFunctionBegin;

  ierr = IPMComputeKKT(tao); CHKERRQ(ierr);
  fold = ipmP->kkt_f;
  while (reason == TAO_CONTINUE_ITERATING) {
    ierr = VecCopy(tao->solution,ipmP->Xold); CHKERRQ(ierr);
    ierr = VecCopy(tao->gradient,ipmP->Gold); CHKERRQ(ierr);

    ierr = KSPSetOperators(tao->ksp,ipmP->K,ipmP->K,DIFFERENT_NONZERO_PATTERN); CHKERRQ(ierr);
    ierr = KSPSolve(tao->ksp,ipmP->bigrhs,ipmP->bigstep);CHKERRQ(ierr);
    ierr = KSPGetIterationNumber(tao->ksp,&its); CHKERRQ(ierr);
    tao->ksp_its += its;
    stepsize=1.0;
    ierr = TaoLineSearchApply(tao->linesearch,tao->solution,&f,
			      tao->gradient,tao->stepdirection,&stepsize,
			      &ls_reason); CHKERRQ(ierr);
    if (ls_reason != TAOLINESEARCH_SUCCESS &&
	ls_reason != TAOLINESEARCH_SUCCESS_USER) {
      /* Failed to find an improving point */
      f = fold;
      ierr = VecCopy(ipmP->Xold, tao->solution); CHKERRQ(ierr);
      ierr = VecCopy(ipmP->Gold, tao->gradient); CHKERRQ(ierr);
      stepsize = 0.0;
      reason = TAO_DIVERGED_LS_FAILURE;
      tao->reason = TAO_DIVERGED_LS_FAILURE;
      break;
    }
    ierr = IPMComputeKKT(tao); CHKERRQ(ierr);
    
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
    ierr = VecDuplicate(tao->solution, &ipmP->work); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->solution, &ipmP->Xold); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->solution, &ipmP->Gold); CHKERRQ(ierr);
  }
  
  if (tao->constraints_inequality) {
    ierr = VecGetSize(tao->constraints_inequality,&ipmP->mi); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_inequality,&ipmP->yi); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_inequality,&ipmP->lamdai); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_inequality,&ipmP->complementarity); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_inequality,&ipmP->dyi); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_inequality,&ipmP->dlamdai); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_inequality,&ipmP->rpi); CHKERRQ(ierr);
  }
  if (tao->constraints_equality) {
    ierr = VecGetSize(tao->constraints_equality,&ipmP->me); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_equality,&ipmP->lamdae); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_equality,&ipmP->dlamdae); CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_equality,&ipmP->rpe); CHKERRQ(ierr);
  }

  /* create K = [ H , 0,   Ae',-Ai']; 
	        [Ae , 0,   0  , 0];
                [Ai ,-Imi, 0 ,  0];  
                [ 0 , L ,  0 ,  Y ];  */
  ierr = VecGetLocalSize(ipmP->yi,&localmi); CHKERRQ(ierr);
  ierr = VecGetLocalSize(ipmP->lamdae,&localme); CHKERRQ(ierr);

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
  ierr = MatCreateTranspose(tao->jacobian_equality,&ipmP->Ae_T); CHKERRQ(ierr);


  kgrid[0] = tao->hessian;
  kgrid[1] = PETSC_NULL;
  kgrid[2] = ipmP->Ae_T;
  kgrid[3] = ipmP->mAi_T;

  kgrid[4] = tao->jacobian_equality;
  kgrid[5] = kgrid[6] = kgrid[7] = PETSC_NULL;

  kgrid[8] = tao->jacobian_inequality;
  kgrid[9] = ipmP->minusI;
  kgrid[10] = kgrid[11] = kgrid[12] = PETSC_NULL;
  kgrid[13] = ipmP->L;
  kgrid[14] = PETSC_NULL;
  kgrid[15] = ipmP->Y;
  
  printf("Matrix Nesting:\n");
  for (i=0;i<16;i++) {
    printf("(%d,%d):\n",i/4,i%4);
    if (kgrid[i]) {
      MatView(kgrid[i],PETSC_VIEWER_STDOUT_SELF);
    } else {
      printf("ZERO\n");
    }

  }
  ierr = MatCreateNest(comm,4,PETSC_NULL,4,PETSC_NULL,kgrid,&ipmP->K); CHKERRQ(ierr);
    
  vgrid[0] = tao->solution;
  vgrid[1] = ipmP->lamdae;
  vgrid[2] = ipmP->yi;
  vgrid[3] = ipmP->lamdai;
  ierr = VecCreateNest(comm,4,PETSC_NULL,vgrid,&ipmP->bigrhs);
  
  vgrid[0] = tao->stepdirection;
  vgrid[1] = ipmP->dyi;
  vgrid[2] = ipmP->dlamdae;
  vgrid[3] = ipmP->dlamdai;
  ierr = VecCreateNest(comm,4,PETSC_NULL,vgrid,&ipmP->bigstep);

  PetscFunctionReturn(0);
  
}

#undef __FUNCT__
#define __FUNCT__ "TaoDestroy_IPM"
static PetscErrorCode TaoDestroy_IPM(TaoSolver tao)
{
  TAO_IPM *ipmP = (TAO_IPM*)tao->data;
  PetscErrorCode ierr;
  ierr = VecDestroy(&ipmP->Xold); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->Gold); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->rd); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->rpe); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->rpi); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->work); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->lamdae); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->lamdai); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->yi); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->dlamdae); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->dlamdai); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->dyi); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->complementarity); CHKERRQ(ierr);
  
  ierr = MatDestroy(&ipmP->K); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->bigrhs); CHKERRQ(ierr);
  ierr = VecDestroy(&ipmP->bigstep); CHKERRQ(ierr);

  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "TaoSetFromOptions"
static PetscErrorCode TaoSetFromOptions_IPM(TaoSolver tao)
{
  return 0;
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
    
    /* com = yi .* lami */
    ierr = VecPointwiseMult(ipmP->complementarity, ipmP->yi,ipmP->lamdai); CHKERRQ(ierr);
    
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
  }
  PetscFunctionReturn(0);
}


EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "TaoCreate_IPM"
PetscErrorCode TaoCreate_IPM(TaoSolver tao)
{
  TAO_IPM *ipmP;
  const char *ipmls_type = TAOLINESEARCH_IPM;
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


  ierr = TaoLineSearchCreate(((PetscObject)tao)->comm, &tao->linesearch); CHKERRQ(ierr);
  ierr = TaoLineSearchSetType(tao->linesearch, ipmls_type); CHKERRQ(ierr);
  ierr = TaoLineSearchSetObjectiveRoutine(tao->linesearch, IPMObjective, tao); CHKERRQ(ierr);

  ierr = KSPCreate(((PetscObject)tao)->comm, &tao->ksp); CHKERRQ(ierr);
  PetscFunctionReturn(0);
  

}
EXTERN_C_END
