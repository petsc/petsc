#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: lsqr.c,v 1.37 1998/06/01 19:32:46 balay Exp balay $";
#endif

#define SWAP(a,b,c) { c = a; a = b; b = c; }

/*                       
       This implements LSQR (Paige and Saunders, ACM Transactions on
       Mathematical Software, Vol 8, pp 43-71, 1982).

       This algorithm does not use a preconditioner. It ignores
       any preconditioner arguments specified.
*/
#include <math.h>
#include "petsc.h"
#include "src/ksp/kspimpl.h"

#undef __FUNC__  
#define __FUNC__ "KSPSetUp_LSQR"
static int KSPSetUp_LSQR(KSP ksp)
{
  int ierr;

  PetscFunctionBegin;
  if (ksp->pc_side == PC_SYMMETRIC){
    SETERRQ(2,0,"no symmetric preconditioning for KSPLSQR");
  }
  ierr = KSPDefaultGetWork( ksp,  6 ); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPSolve_LSQR"
static int KSPSolve_LSQR(KSP ksp,int *its)
{
  int          i = 0, maxit, hist_len, cerr = 0, ierr;
  Scalar       rho, rhobar, phi, phibar, theta, c, s,tmp, zero = 0.0,mone=-1.0;
  double       beta, alpha, rnorm, *history;
  Vec          X,B,V,V1,U,U1,TMP,W,BINVF;
  Mat          Amat, Pmat;
  MatStructure pflag;

  PetscFunctionBegin;
  ksp->its = 0;
  ierr     = PCGetOperators(ksp->B,&Amat,&Pmat,&pflag); CHKERRQ(ierr);
  maxit    = ksp->max_it;
  history  = ksp->residual_history;
  hist_len = ksp->res_hist_size;
  X        = ksp->vec_sol;
  B        = ksp->vec_rhs;
  U        = ksp->work[0];
  U1       = ksp->work[1];
  V        = ksp->work[2];
  V1       = ksp->work[3];
  W        = ksp->work[4];
  BINVF    = ksp->work[5];

  /* Compute initial preconditioned residual */
  /* ierr = KSPResidual(ksp,X,V,U, W,BINVF,B); CHKERRQ(ierr); */

  /* Compute initial residual */
  if (!ksp->guess_zero) {
    ierr = MatMult(Amat,X,W); CHKERRQ(ierr);       /*   w <- b - Ax       */
    ierr = VecAYPX(&mone,B,W); CHKERRQ(ierr);
  } else { 
    ierr = VecCopy(B,W); CHKERRQ(ierr);            /*     w <- b (x is 0) */
  }

  /* Test for nothing to do */
  ierr = VecNorm(W,NORM_2,&rnorm); CHKERRQ(ierr);
  if ((*ksp->converged)(ksp,0,rnorm,ksp->cnvP)) { *its = 0; PetscFunctionReturn(0);}
  KSPMonitor(ksp,0,rnorm);
  ksp->rnorm              = rnorm;
  if (history) history[0] = rnorm;

  ierr = VecCopy(B,U); CHKERRQ(ierr);
  ierr = VecNorm(U,NORM_2,&beta); CHKERRQ(ierr);
  tmp = 1.0/beta; ierr = VecScale(&tmp,U); CHKERRQ(ierr);
  ierr = MatMultTrans(Amat,U,V); CHKERRQ(ierr);
  ierr = VecNorm(V,NORM_2,&alpha); CHKERRQ(ierr);
  tmp = 1.0/alpha; ierr = VecScale(&tmp,V); CHKERRQ(ierr);

  ierr = VecCopy(V,W); CHKERRQ(ierr);
  ierr = VecSet(&zero,X); CHKERRQ(ierr);

  phibar = beta;
  rhobar = alpha;
  for (i=0; i<maxit; i++) {
    ksp->its++;

    ierr = MatMult(Amat,V,U1); CHKERRQ(ierr);
    tmp  = -alpha; ierr = VecAXPY(&tmp,U,U1); CHKERRQ(ierr);
    ierr = VecNorm(U1,NORM_2,&beta); CHKERRQ(ierr);
    tmp  = 1.0/beta; ierr = VecScale(&tmp,U1); CHKERRQ(ierr);

    ierr = MatMultTrans(Amat,U1,V1); CHKERRQ(ierr);
    tmp  = -beta; ierr = VecAXPY(&tmp,V,V1); CHKERRQ(ierr);
    ierr = VecNorm(V1,NORM_2,&alpha); CHKERRQ(ierr);
    tmp  = 1.0 / alpha; ierr = VecScale(&tmp,V1); CHKERRQ(ierr);

    rho    = PetscSqrtScalar(rhobar*rhobar + beta*beta);
    c      = rhobar / rho;
    s      = beta / rho;
    theta  = s * alpha;
    rhobar = - c * alpha;
    phi    = c * phibar;
    phibar = s * phibar;

    tmp  = phi/rho; 
    ierr = VecAXPY(&tmp,W,X); CHKERRQ(ierr);  /*    x <- x + (phi/rho) w   */
    tmp  = -theta/rho; 
    ierr = VecAYPX(&tmp,V1,W); CHKERRQ(ierr); /*    w <- v - (theta/rho) w */

#if defined(USE_PETSC_COMPLEX)
    rnorm = PetscReal(phibar);
#else
    rnorm = phibar;
#endif

    ksp->rnorm = rnorm;
    if (history && hist_len > i + 1) history[i+1] = rnorm;
    KSPMonitor(ksp,i+1,rnorm);
    cerr = (*ksp->converged)(ksp,i+1,rnorm,ksp->cnvP);
    if (cerr) break;
    SWAP( U1, U, TMP );
    SWAP( V1, V, TMP );
  }
  if (i == maxit) i--;
  if (history) ksp->res_act_size = (hist_len < i + 1) ? hist_len : i + 1;

  /* ierr = KSPUnwindPreconditioner(ksp,X,W); CHKERRQ(ierr); */

  if (cerr <= 0) *its = -(i+1);
  else          *its = i + 1;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPCreate_LSQR"
int KSPCreate_LSQR(KSP ksp)
{
  PetscFunctionBegin;
  ksp->data                 = (void *) 0;
  ksp->pc_side              = PC_LEFT;
  ksp->calc_res             = 1;
  ksp->setup                = KSPSetUp_LSQR;
  ksp->solver               = KSPSolve_LSQR;
  ksp->adjustwork           = KSPDefaultAdjustWork;
  ksp->destroy              = KSPDefaultDestroy;
  ksp->converged            = KSPDefaultConverged;
  ksp->buildsolution        = KSPDefaultBuildSolution;
  ksp->buildresidual        = KSPDefaultBuildResidual;
  ksp->view                 = 0;
  PetscFunctionReturn(0);
}
