#define PETSCKSP_DLL

#define SWAP(a,b,c) { c = a; a = b; b = c; }

#include "src/ksp/ksp/kspimpl.h"

typedef struct {
  PetscInt  nwork_n,nwork_m; 
  Vec       *vwork_m;  /* work vectors of length m, where the system is size m x n */
  Vec       *vwork_n;  /* work vectors of length m */
} KSP_LSQR;

#undef __FUNCT__  
#define __FUNCT__ "KSPSetUp_LSQR"
static PetscErrorCode KSPSetUp_LSQR(KSP ksp)
{
  PetscErrorCode ierr;
  PetscInt       nw;
  KSP_LSQR       *lsqr = (KSP_LSQR*)ksp->data;

  PetscFunctionBegin;
  if (ksp->pc_side == PC_SYMMETRIC){
    SETERRQ(PETSC_ERR_SUP,"no symmetric preconditioning for KSPLSQR");
  }

  /* Get work vectors */
  lsqr->nwork_m = nw = 2;
  if (lsqr->vwork_m) {
    ierr = VecDestroyVecs(lsqr->vwork_m,lsqr->nwork_m);CHKERRQ(ierr);
  }
  ierr = KSPGetVecs(ksp,nw,&lsqr->vwork_m);CHKERRQ(ierr);
  ierr = PetscLogObjectParents(ksp,nw,lsqr->vwork_m);CHKERRQ(ierr);

  lsqr->nwork_n = nw = 3;
  if (lsqr->vwork_n) {
    ierr = VecDestroyVecs(lsqr->vwork_n,lsqr->nwork_n);CHKERRQ(ierr);
  }
  ierr = KSPGetVecs(ksp,nw,&lsqr->vwork_n);CHKERRQ(ierr);
  ierr = PetscLogObjectParents(ksp,nw,lsqr->vwork_n);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSolve_LSQR"
static PetscErrorCode KSPSolve_LSQR(KSP ksp)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscScalar    rho,rhobar,phi,phibar,theta,c,s,tmp,zero = 0.0,mone=-1.0;
  PetscReal      beta,alpha,rnorm;
  Vec            X,B,V,V1,U,U1,TMP,W;
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

  /* Compute initial residual, temporarily use work vector u */
  if (!ksp->guess_zero) {
    ierr = KSP_MatMult(ksp,Amat,X,U);CHKERRQ(ierr);       /*   u <- b - Ax     */
    ierr = VecAYPX(U,mone,B);CHKERRQ(ierr);
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
  tmp = 1.0/beta; ierr = VecScale(U,tmp);CHKERRQ(ierr);
  ierr = KSP_MatMultTranspose(ksp,Amat,U,V);CHKERRQ(ierr);
  ierr = VecNorm(V,NORM_2,&alpha);CHKERRQ(ierr);
  tmp = 1.0/alpha; ierr = VecScale(V,tmp);CHKERRQ(ierr);

  ierr = VecCopy(V,W);CHKERRQ(ierr);
  ierr = VecSet(X,zero);CHKERRQ(ierr);

  phibar = beta;
  rhobar = alpha;
  i = 0;
  do {

    ierr = KSP_MatMult(ksp,Amat,V,U1);CHKERRQ(ierr);
    tmp  = -alpha; ierr = VecAXPY(U1,tmp,U);CHKERRQ(ierr);
    ierr = VecNorm(U1,NORM_2,&beta);CHKERRQ(ierr);
    tmp  = 1.0/beta; ierr = VecScale(U1,tmp);CHKERRQ(ierr);

    ierr = KSP_MatMultTranspose(ksp,Amat,U1,V1);CHKERRQ(ierr);
    tmp  = -beta; ierr = VecAXPY(V1,tmp,V);CHKERRQ(ierr);
    ierr = VecNorm(V1,NORM_2,&alpha);CHKERRQ(ierr);
    tmp  = 1.0 / alpha; ierr = VecScale(V1,tmp);CHKERRQ(ierr);

    rho    = PetscSqrtScalar(rhobar*rhobar + beta*beta);
    c      = rhobar / rho;
    s      = beta / rho;
    theta  = s * alpha;
    rhobar = - c * alpha;
    phi    = c * phibar;
    phibar = s * phibar;

    tmp  = phi/rho; 
    ierr = VecAXPY(X,tmp,W);CHKERRQ(ierr);  /*    x <- x + (phi/rho) w   */
    tmp  = -theta/rho; 
    ierr = VecAYPX(W,tmp,V1);CHKERRQ(ierr); /*    w <- v - (theta/rho) w */

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

  /* ierr = KSPUnwindPreconditioner(ksp,X,W);CHKERRQ(ierr); */

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
  ierr = PetscFree(lsqr);CHKERRQ(ierr);
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
  ierr = PetscMalloc(sizeof(KSP_LSQR),&lsqr);CHKERRQ(ierr);
  ierr = PetscMemzero(lsqr,sizeof(KSP_LSQR));CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(ksp,sizeof(KSP_LSQR));CHKERRQ(ierr);
  ksp->data                      = (void*)lsqr;
  ksp->pc_side                   = PC_LEFT;
  ksp->ops->setup                = KSPSetUp_LSQR;
  ksp->ops->solve                = KSPSolve_LSQR;
  ksp->ops->destroy              = KSPDestroy_LSQR;
  ksp->ops->buildsolution        = KSPDefaultBuildSolution;
  ksp->ops->buildresidual        = KSPDefaultBuildResidual;
  ksp->ops->setfromoptions       = 0;
  ksp->ops->view                 = 0;
  PetscFunctionReturn(0);
}
EXTERN_C_END
