#ifndef lint
static char vcid[] = "$Id: lsqr.c,v 1.17 1995/11/01 19:09:10 bsmith Exp bsmith $";
#endif

#define SWAP(a,b,c) { c = a; a = b; b = c; }

/*                       
       This implements LSQR (Paige and Saunders, ACM Transactions on
       Mathematical Software, Vol 8, pp 43-71, 1982).
*/
#include <stdio.h>
#include <math.h>
#include "petsc.h"
#include "kspimpl.h"

static int KSPSetUp_LSQR(KSP itP)
{
  int ierr;
  ierr = KSPCheckDef( itP ); CHKERRQ(ierr);
  ierr = KSPiDefaultGetWork( itP,  6 ); CHKERRQ(ierr);
  return 0;
}

static int KSPSolve_LSQR(KSP itP,int *its)
{
  int          i = 0, maxit, hist_len, cerr = 0, ierr;
  Scalar       rho, rhobar, phi, phibar, theta, c, s,tmp, zero = 0.0;
  double       beta, alpha, rnorm, *history;
  Vec          X,B,V,V1,U,U1,TMP,W,BINVF;
  Mat          Amat, Pmat;
  MatStructure pflag;

  ierr     = PCGetOperators(itP->B,&Amat,&Pmat,&pflag); CHKERRQ(ierr);
  maxit    = itP->max_it;
  history  = itP->residual_history;
  hist_len = itP->res_hist_size;
  X        = itP->vec_sol;
  B        = itP->vec_rhs;
  U        = itP->work[0];
  U1       = itP->work[1];
  V        = itP->work[2];
  V1       = itP->work[3];
  W        = itP->work[4];
  BINVF    = itP->work[5];

  /* Compute initial preconditioned residual */
  ierr = KSPResidual(itP,X,V,U, W,BINVF,B); CHKERRQ(ierr);

  /* Test for nothing to do */
  ierr = VecNorm(W,NORM_2,&rnorm); CHKERRQ(ierr);
  if ((*itP->converged)(itP,0,rnorm,itP->cnvP)) { *its = 0; return 0;}
  MONITOR(itP,rnorm,0);
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
    ierr = MatMult(Amat,V,U1); CHKERRQ(ierr);
    tmp  = -alpha; ierr = VecAXPY(&tmp,U,U1); CHKERRQ(ierr);
    ierr = VecNorm(U1,NORM_2,&beta); CHKERRQ(ierr);
    tmp  = 1.0/beta; ierr = VecScale(&tmp,U1); CHKERRQ(ierr);

    ierr = MatMultTrans(Amat,U1,V1); CHKERRQ(ierr);
    tmp  = -beta; ierr = VecAXPY(&tmp,V,V1); CHKERRQ(ierr);
    ierr = VecNorm(V1,NORM_2,&alpha); CHKERRQ(ierr);
    tmp  = 1.0 / alpha; ierr = VecScale(&tmp,V1); CHKERRQ(ierr);

    rho    = sqrt(rhobar*rhobar + beta*beta);
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

#if defined(PETSC_COMPLEX)
    rnorm = real(phibar);
#else
    rnorm = phibar;
#endif

    if (history && hist_len > i + 1) history[i+1] = rnorm;
    MONITOR(itP,rnorm,i+1);
    cerr = (*itP->converged)(itP,i+1,rnorm,itP->cnvP);
    if (cerr) break;
    SWAP( U1, U, TMP );
    SWAP( V1, V, TMP );
  }
  if (i == maxit) i--;
  if (history) itP->res_act_size = (hist_len < i + 1) ? hist_len : i + 1;

  ierr = KSPUnwindPre(itP,X,W); CHKERRQ(ierr);
  if (cerr <= 0) *its = -(i+1);
  else          *its = i + 1;
  return 0;
}

int KSPCreate_LSQR(KSP itP)
{
  itP->data                 = (void *) 0;
  itP->type                 = KSPLSQR;
  itP->right_pre            = 0;
  itP->calc_res             = 1;
  itP->setup                = KSPSetUp_LSQR;
  itP->solver               = KSPSolve_LSQR;
  itP->adjustwork           = KSPiDefaultAdjustWork;
  itP->destroy              = KSPiDefaultDestroy;
  itP->converged            = KSPDefaultConverged;
  itP->buildsolution        = KSPDefaultBuildSolution;
  itP->buildresidual        = KSPDefaultBuildResidual;
  itP->view                 = 0;
  return 0;
}
