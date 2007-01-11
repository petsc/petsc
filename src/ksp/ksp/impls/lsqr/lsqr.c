#define PETSCKSP_DLL

/* lourens.vanzanen@shell.com contributed the standard error estimates of the solution, Jul 25, 2006 */

#define SWAP(a,b,c) { c = a; a = b; b = c; }

#include "include/private/kspimpl.h"

typedef struct {
  PetscInt  nwork_n,nwork_m; 
  Vec       *vwork_m;  /* work vectors of length m, where the system is size m x n */
  Vec       *vwork_n;  /* work vectors of length n */
  Vec       se;        /* Optional standard error vector */
  PetscTruth se_flg;   /* flag for -ksp_lsqr_set_standard_error */
} KSP_LSQR;

extern PetscErrorCode PETSCVEC_DLLEXPORT VecSquare(Vec);

#undef __FUNCT__  
#define __FUNCT__ "KSPSetUp_LSQR"
static PetscErrorCode KSPSetUp_LSQR(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_LSQR       *lsqr = (KSP_LSQR*)ksp->data;

  PetscFunctionBegin;
  if (ksp->pc_side == PC_SYMMETRIC){
    SETERRQ(PETSC_ERR_SUP,"no symmetric preconditioning for KSPLSQR");
  }

  /* Get work vectors */
  lsqr->nwork_m = 2;
  if (lsqr->vwork_m) {
    ierr = VecDestroyVecs(lsqr->vwork_m,lsqr->nwork_m);CHKERRQ(ierr);
  }
  lsqr->nwork_n = 4;
  if (lsqr->vwork_n) {
    ierr = VecDestroyVecs(lsqr->vwork_n,lsqr->nwork_n);CHKERRQ(ierr);
  }
  ierr = KSPGetVecs(ksp,lsqr->nwork_n,&lsqr->vwork_n,lsqr->nwork_m,&lsqr->vwork_m);CHKERRQ(ierr);
  if (lsqr->se_flg && !lsqr->se){
    /* lsqr->se is not set by user, get it from pmat */
    Mat pmat;
    ierr = PCGetOperators(ksp->pc,PETSC_NULL,&pmat,PETSC_NULL);CHKERRQ(ierr);
    ierr = MatGetVecs(pmat,&lsqr->se,PETSC_NULL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSolve_LSQR"
static PetscErrorCode KSPSolve_LSQR(KSP ksp)
{
  PetscErrorCode ierr;
  PetscInt       i,size1,size2;
  PetscScalar    rho,rhobar,phi,phibar,theta,c,s,tmp;
  PetscReal      beta,alpha,rnorm;
  Vec            X,B,V,V1,U,U1,TMP,W,W2,SE;
  Mat            Amat,Pmat;
  MatStructure   pflag;
  KSP_LSQR       *lsqr = (KSP_LSQR*)ksp->data;
  PetscTruth     diagonalscale;

  PetscFunctionBegin;
  ierr    = PCDiagonalScale(ksp->pc,&diagonalscale);CHKERRQ(ierr);
  if (diagonalscale) SETERRQ1(PETSC_ERR_SUP,"Krylov method %s does not support diagonal scaling",ksp->type_name);

  ierr     = PCGetOperators(ksp->pc,&Amat,&Pmat,&pflag);CHKERRQ(ierr);

  /* vectors of length m, where system size is mxn */
  B        = ksp->vec_rhs;
  U        = lsqr->vwork_m[0];
  U1       = lsqr->vwork_m[1];

  /* vectors of length n */
  X        = ksp->vec_sol;
  W        = lsqr->vwork_n[0];
  V        = lsqr->vwork_n[1];
  V1       = lsqr->vwork_n[2];
  W2       = lsqr->vwork_n[3];

  /* standard error vector */
  SE = lsqr->se;
  if (SE){
    ierr = VecGetSize(SE,&size1);CHKERRQ(ierr);
    ierr = VecGetSize(X ,&size2);CHKERRQ(ierr);
    if (size1 != size2) SETERRQ2(PETSC_ERR_ARG_SIZ,"Standard error vector (size %d) does not match solution vector (size %d)",size1,size2);
    ierr = VecSet(SE,0.0);CHKERRQ(ierr); 
  }

  /* Compute initial residual, temporarily use work vector u */
  if (!ksp->guess_zero) {
    ierr = KSP_MatMult(ksp,Amat,X,U);CHKERRQ(ierr);       /*   u <- b - Ax     */
    ierr = VecAYPX(U,-1.0,B);CHKERRQ(ierr);
  } else { 
    ierr = VecCopy(B,U);CHKERRQ(ierr);            /*   u <- b (x is 0) */
  }

  /* Test for nothing to do */
  ierr = VecNorm(U,NORM_2,&rnorm);CHKERRQ(ierr);
  ierr = PetscObjectTakeAccess(ksp);CHKERRQ(ierr);
  ksp->its   = 0;
  ksp->rnorm = rnorm;
  ierr = PetscObjectGrantAccess(ksp);CHKERRQ(ierr);
  KSPLogResidualHistory(ksp,rnorm);
  KSPMonitor(ksp,0,rnorm);
  ierr = (*ksp->converged)(ksp,0,rnorm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
  if (ksp->reason) PetscFunctionReturn(0);

  ierr = VecCopy(B,U);CHKERRQ(ierr);
  ierr = VecNorm(U,NORM_2,&beta);CHKERRQ(ierr);
  ierr = VecScale(U,1.0/beta);CHKERRQ(ierr);
  ierr = KSP_MatMultTranspose(ksp,Amat,U,V);CHKERRQ(ierr);
  ierr = VecNorm(V,NORM_2,&alpha);CHKERRQ(ierr);
  ierr = VecScale(V,1.0/alpha);CHKERRQ(ierr);

  ierr = VecCopy(V,W);CHKERRQ(ierr);
  ierr = VecSet(X,0.0);CHKERRQ(ierr);

  phibar = beta;
  rhobar = alpha;
  i = 0;
  do {

    ierr = KSP_MatMult(ksp,Amat,V,U1);CHKERRQ(ierr);
    ierr = VecAXPY(U1,-alpha,U);CHKERRQ(ierr);
    ierr = VecNorm(U1,NORM_2,&beta);CHKERRQ(ierr);
    ierr = VecScale(U1,1.0/beta);CHKERRQ(ierr);

    ierr = KSP_MatMultTranspose(ksp,Amat,U1,V1);CHKERRQ(ierr);
    ierr = VecAXPY(V1,-beta,V);CHKERRQ(ierr);
    ierr = VecNorm(V1,NORM_2,&alpha);CHKERRQ(ierr);
    ierr = VecScale(V1,1.0/alpha);CHKERRQ(ierr);

    rho    = PetscSqrtScalar(rhobar*rhobar + beta*beta);
    c      = rhobar / rho;
    s      = beta / rho;
    theta  = s * alpha;
    rhobar = - c * alpha;
    phi    = c * phibar;
    phibar = s * phibar;

    ierr = VecAXPY(X,phi/rho,W);CHKERRQ(ierr);  /*    x <- x + (phi/rho) w   */
    if (SE) {
      ierr = VecCopy(W,W2);CHKERRQ(ierr);
      ierr = VecSquare(W2);CHKERRQ(ierr);
      ierr = VecScale(W2,1.0/(rho*rho));CHKERRQ(ierr);
      ierr = VecAXPY(SE, 1.0, W2);CHKERRQ(ierr); /* SE <- SE + (w^2/rho^2) */
    }
    ierr = VecAYPX(W,-theta/rho,V1);CHKERRQ(ierr); /* w <- v - (theta/rho) w */

    rnorm = PetscRealPart(phibar);

    ierr = PetscObjectTakeAccess(ksp);CHKERRQ(ierr);
    ksp->its++;
    ksp->rnorm = rnorm;
    ierr = PetscObjectGrantAccess(ksp);CHKERRQ(ierr);
    KSPLogResidualHistory(ksp,rnorm);
    KSPMonitor(ksp,i+1,rnorm);
    ierr = (*ksp->converged)(ksp,i+1,rnorm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
    if (ksp->reason) break;
    SWAP(U1,U,TMP);
    SWAP(V1,V,TMP);

    i++;
  } while (i<ksp->max_it);
  if (i >= ksp->max_it && !ksp->reason) {
    ksp->reason = KSP_DIVERGED_ITS;
  }

  /* Finish off the standard error estimates */
  if (SE) {
    tmp = 1.0;
    ierr = MatGetSize(Amat,&size1,&size2);CHKERRQ(ierr);
    if ( size1 > size2 ) tmp = size1 - size2;
    tmp = rnorm / PetscSqrtScalar(tmp);
    ierr = VecSqrt(SE);CHKERRQ(ierr);
    ierr = VecScale(SE,tmp);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPDestroy_LSQR" 
PetscErrorCode KSPDestroy_LSQR(KSP ksp)
{
  KSP_LSQR       *lsqr = (KSP_LSQR*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  /* Free work vectors */
  if (lsqr->vwork_n) {
    ierr = VecDestroyVecs(lsqr->vwork_n,lsqr->nwork_n);CHKERRQ(ierr);
  }
  if (lsqr->vwork_m) {
    ierr = VecDestroyVecs(lsqr->vwork_m,lsqr->nwork_m);CHKERRQ(ierr);
  }
  if (lsqr->se_flg && lsqr->se){
    ierr = VecDestroy(lsqr->se);CHKERRQ(ierr);
  }
  ierr = PetscFree(lsqr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPLSQRSetStandardErrorVec"
PetscErrorCode PETSCKSP_DLLEXPORT KSPLSQRSetStandardErrorVec( KSP ksp, Vec se )
{
  KSP_LSQR  *lsqr = (KSP_LSQR*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (lsqr->se) {
    ierr = VecDestroy(lsqr->se);CHKERRQ(ierr);
  }
  lsqr->se     = se;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPLSQRGetStandardErrorVec"
PetscErrorCode PETSCKSP_DLLEXPORT KSPLSQRGetStandardErrorVec( KSP ksp,Vec *se )
{
  KSP_LSQR *lsqr = (KSP_LSQR*)ksp->data;

  PetscFunctionBegin;
  *se = lsqr->se;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSetFromOptions_LSQR"
PetscErrorCode KSPSetFromOptions_LSQR(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_LSQR       *lsqr = (KSP_LSQR*)ksp->data;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("KSP LSQR Options");CHKERRQ(ierr);
  ierr = PetscOptionsName("-ksp_lsqr_set_standard_error","Set Standard Error Estimates of Solution","KSPLSQRSetStandardErrorVec",&lsqr->se_flg);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPView_LSQR" 
PetscErrorCode KSPView_LSQR(KSP ksp,PetscViewer viewer)
{
  KSP_LSQR       *lsqr = (KSP_LSQR*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (lsqr->se) {
    PetscReal rnorm;
    ierr = KSPLSQRGetStandardErrorVec(ksp,&lsqr->se);CHKERRQ(ierr);
    ierr = VecNorm(lsqr->se,NORM_2,&rnorm);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"  Norm of Standard Error %A, Iterations %D\n",rnorm,ksp->its);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*MC
     KSPLSQR - This implements LSQR

   Options Database Keys:
.   see KSPSolve()

   Level: beginner

   Notes:  This algorithm DOES NOT use a preconditioner. It ignores any preconditioner arguments specified.
           Reference: Paige and Saunders, ACM Transactions on Mathematical Software, Vol 8, pp 43-71, 1982

.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP

M*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "KSPCreate_LSQR"
PetscErrorCode PETSCKSP_DLLEXPORT KSPCreate_LSQR(KSP ksp)
{
  KSP_LSQR       *lsqr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNew(KSP_LSQR,&lsqr);CHKERRQ(ierr);
  lsqr->se     = PETSC_NULL;
  lsqr->se_flg = PETSC_FALSE;
  ierr = PCSetType(ksp->pc,PCNONE);CHKERRQ(ierr);
  ksp->data                      = (void*)lsqr;
  ksp->pc_side                   = PC_LEFT;
  ksp->ops->setup                = KSPSetUp_LSQR;
  ksp->ops->solve                = KSPSolve_LSQR;
  ksp->ops->destroy              = KSPDestroy_LSQR;
  ksp->ops->buildsolution        = KSPDefaultBuildSolution;
  ksp->ops->buildresidual        = KSPDefaultBuildResidual;
  ksp->ops->setfromoptions       = KSPSetFromOptions_LSQR;
  ksp->ops->view                 = KSPView_LSQR;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "VecSquare"
PetscErrorCode PETSCVEC_DLLEXPORT VecSquare(Vec v)
{
  PetscErrorCode ierr;
  PetscScalar    *x;
  PetscInt       i, n;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(v, &n);CHKERRQ(ierr);
  ierr = VecGetArray(v, &x);CHKERRQ(ierr);
  for(i = 0; i < n; i++) {
    x[i] *= x[i];
  }
  ierr = VecRestoreArray(v, &x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
