#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: lsqr.c,v 1.50 1999/03/01 04:55:59 bsmith Exp balay $";
#endif

#define SWAP(a,b,c) { c = a; a = b; b = c; }

/*                       
       This implements LSQR (Paige and Saunders, ACM Transactions on
       Mathematical Software, Vol 8, pp 43-71, 1982).

       This algorithm DOES NOT use a preconditioner. It ignores
       any preconditioner arguments specified.
*/
#include "petsc.h"
#include "src/sles/ksp/kspimpl.h"


typedef struct {
  int  nwork_n, nwork_m; 
  Vec  *vwork_m;  /* work vectors of length m, where the system is size m x n */
  Vec  *vwork_n;  /* work vectors of length m */
} KSP_LSQR;

#undef __FUNC__  
#define __FUNC__ "KSPSetUp_LSQR"
static int KSPSetUp_LSQR(KSP ksp)
{
  int ierr, nw;
  KSP_LSQR *lsqr = (KSP_LSQR *) ksp->data;

  PetscFunctionBegin;
  if (ksp->pc_side == PC_SYMMETRIC){
    SETERRQ(2,0,"no symmetric preconditioning for KSPLSQR");
  }

  /* Get work vectors */
  lsqr->nwork_m = nw = 2;
  if (lsqr->vwork_m) {
    ierr = VecDestroyVecs(lsqr->vwork_m,lsqr->nwork_m);CHKERRQ(ierr);
  }
  ierr = VecDuplicateVecs(ksp->vec_rhs,nw,&lsqr->vwork_m);CHKERRQ(ierr);
  PLogObjectParents(ksp,nw,lsqr->vwork_m);

  lsqr->nwork_n = nw = 3;
  if (lsqr->vwork_n) {
    ierr = VecDestroyVecs(lsqr->vwork_n,lsqr->nwork_n);CHKERRQ(ierr);
  }
  ierr = VecDuplicateVecs(ksp->vec_sol,nw,&lsqr->vwork_n);CHKERRQ(ierr);
  PLogObjectParents(ksp,nw,lsqr->vwork_n);

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPSolve_LSQR"
static int KSPSolve_LSQR(KSP ksp,int *its)
{
  int          i = 0, maxit, cerr = 0, ierr;
  Scalar       rho, rhobar, phi, phibar, theta, c, s,tmp, zero = 0.0,mone=-1.0;
  double       beta, alpha, rnorm;
  Vec          X,B,V,V1,U,U1,TMP,W;
  Mat          Amat, Pmat;
  MatStructure pflag;
  KSP_LSQR     *lsqr = (KSP_LSQR *) ksp->data;

  PetscFunctionBegin;
  ierr     = PCGetOperators(ksp->B,&Amat,&Pmat,&pflag);CHKERRQ(ierr);
  maxit    = ksp->max_it;

  /* vectors of length m, where system size is mxn */
  B        = ksp->vec_rhs;
  U        = lsqr->vwork_m[0];
  U1       = lsqr->vwork_m[1];

  /* vectors of length n */
  X        = ksp->vec_sol;
  W        = lsqr->vwork_n[0];
  V        = lsqr->vwork_n[1];
  V1       = lsqr->vwork_n[2];

  /* BINVF    = lsqr->work[5];*/

  /* Compute initial preconditioned residual */
  /* ierr = KSPResidual(ksp,X,V,U, W,BINVF,B);CHKERRQ(ierr); */

  /* Compute initial residual, temporarily use work vector u */
  if (!ksp->guess_zero) {
    ierr = MatMult(Amat,X,U);CHKERRQ(ierr);       /*   u <- b - Ax     */
    ierr = VecAYPX(&mone,B,U);CHKERRQ(ierr);
  } else { 
    ierr = VecCopy(B,U);CHKERRQ(ierr);            /*   u <- b (x is 0) */
  }

  /* Test for nothing to do */
  ierr = VecNorm(U,NORM_2,&rnorm);CHKERRQ(ierr);
  PetscAMSTakeAccess(ksp);
  ksp->its   = 0;
  ksp->rnorm = rnorm;
  PetscAMSGrantAccess(ksp);
  if ((*ksp->converged)(ksp,0,rnorm,ksp->cnvP)) { *its = 0; PetscFunctionReturn(0);}
  KSPLogResidualHistory(ksp,rnorm);
  KSPMonitor(ksp,0,rnorm);

  ierr = VecCopy(B,U);CHKERRQ(ierr);
  ierr = VecNorm(U,NORM_2,&beta);CHKERRQ(ierr);
  tmp = 1.0/beta; ierr = VecScale(&tmp,U);CHKERRQ(ierr);
  ierr = MatMultTrans(Amat,U,V);CHKERRQ(ierr);
  ierr = VecNorm(V,NORM_2,&alpha);CHKERRQ(ierr);
  tmp = 1.0/alpha; ierr = VecScale(&tmp,V);CHKERRQ(ierr);

  ierr = VecCopy(V,W);CHKERRQ(ierr);
  ierr = VecSet(&zero,X);CHKERRQ(ierr);

  phibar = beta;
  rhobar = alpha;
  for (i=0; i<maxit; i++) {

    ierr = MatMult(Amat,V,U1);CHKERRQ(ierr);
    tmp  = -alpha; ierr = VecAXPY(&tmp,U,U1);CHKERRQ(ierr);
    ierr = VecNorm(U1,NORM_2,&beta);CHKERRQ(ierr);
    tmp  = 1.0/beta; ierr = VecScale(&tmp,U1);CHKERRQ(ierr);

    ierr = MatMultTrans(Amat,U1,V1);CHKERRQ(ierr);
    tmp  = -beta; ierr = VecAXPY(&tmp,V,V1);CHKERRQ(ierr);
    ierr = VecNorm(V1,NORM_2,&alpha);CHKERRQ(ierr);
    tmp  = 1.0 / alpha; ierr = VecScale(&tmp,V1);CHKERRQ(ierr);

    rho    = PetscSqrtScalar(rhobar*rhobar + beta*beta);
    c      = rhobar / rho;
    s      = beta / rho;
    theta  = s * alpha;
    rhobar = - c * alpha;
    phi    = c * phibar;
    phibar = s * phibar;

    tmp  = phi/rho; 
    ierr = VecAXPY(&tmp,W,X);CHKERRQ(ierr);  /*    x <- x + (phi/rho) w   */
    tmp  = -theta/rho; 
    ierr = VecAYPX(&tmp,V1,W);CHKERRQ(ierr); /*    w <- v - (theta/rho) w */

#if defined(USE_PETSC_COMPLEX)
    rnorm = PetscReal(phibar);
#else
    rnorm = phibar;
#endif

    PetscAMSTakeAccess(ksp);
    ksp->its++;
    ksp->rnorm = rnorm;
    PetscAMSGrantAccess(ksp);
    KSPLogResidualHistory(ksp,rnorm);
    KSPMonitor(ksp,i+1,rnorm);
    cerr = (*ksp->converged)(ksp,i+1,rnorm,ksp->cnvP);
    if (cerr) break;
    SWAP( U1, U, TMP );
    SWAP( V1, V, TMP );
  }
  if (i == maxit) i--;

  /* ierr = KSPUnwindPreconditioner(ksp,X,W);CHKERRQ(ierr); */

  if (cerr <= 0) *its = -(i+1);
  else          *its = i + 1;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPDestroy_LSQR" 
int KSPDestroy_LSQR(KSP ksp)
{
  KSP_LSQR *lsqr = (KSP_LSQR *) ksp->data;
  int      ierr;

  PetscFunctionBegin;

  /* Free work vectors */
  if (lsqr->vwork_n) {
    ierr = VecDestroyVecs(lsqr->vwork_n,lsqr->nwork_n);CHKERRQ(ierr);
  }
  if (lsqr->vwork_m) {
    ierr = VecDestroyVecs(lsqr->vwork_m,lsqr->nwork_m);CHKERRQ(ierr);
  }
  PetscFree(lsqr); 
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "KSPCreate_LSQR"
int KSPCreate_LSQR(KSP ksp)
{
  KSP_LSQR *lsqr;
  int      ierr;

  PetscFunctionBegin;
  lsqr = (KSP_LSQR*) PetscMalloc(sizeof(KSP_LSQR));CHKPTRQ(lsqr);
  ierr = PetscMemzero(lsqr,sizeof(KSP_LSQR));CHKERRQ(ierr);
  PLogObjectMemory(ksp,sizeof(KSP_LSQR));
  ksp->data                      = (void *) lsqr;
  ksp->pc_side                   = PC_LEFT;
  ksp->calc_res                  = 1;
  ksp->ops->setup                = KSPSetUp_LSQR;
  ksp->ops->solve                = KSPSolve_LSQR;
  ksp->ops->destroy              = KSPDestroy_LSQR;
  ksp->converged                 = KSPDefaultConverged;
  ksp->ops->buildsolution        = KSPDefaultBuildSolution;
  ksp->ops->buildresidual        = KSPDefaultBuildResidual;
  ksp->ops->view                 = 0;
  PetscFunctionReturn(0);
}
EXTERN_C_END
