#ifndef lint
static char vcid[] = "$Id: lsqr.c,v 1.13 1995/07/17 03:54:05 bsmith Exp curfman $";
#endif

#define SWAP(a,b,c) { c = a; a = b; b = c; }

/*                       
       This implements LSQR (Paige and Saunders, ACM Transactions on
       Mathematical Software, Vol 8, pp 43-71, 1982).

       and add the extern declarations of the create, solve, and setup 
       routines to 

*/
#include <stdio.h>
#include <math.h>
#include "petsc.h"
#include "kspimpl.h"

static int KSPSetUp_LSQR(KSP itP)
{
  int ierr;
  if ((ierr = KSPCheckDef( itP ))) return ierr;
  ierr = KSPiDefaultGetWork( itP,  6 );
  return ierr;
}

static int KSPSolve_LSQR(KSP itP,int *its)
{
int          i = 0, maxit, hist_len, cerr = 0;
Scalar       rho, rhobar, phi, phibar, theta, c, s;
double       beta, alpha, rnorm, *history;
Scalar       tmp, zero = 0.0;
Vec          X,B,V,V1,U,U1,TMP,W,BINVF;
Mat          Amat, Pmat;
MatStructure pflag;

PCGetOperators(itP->B,&Amat,&Pmat,&pflag);
maxit   = itP->max_it;
history = itP->residual_history;
hist_len= itP->res_hist_size;
X       = itP->vec_sol;
B       = itP->vec_rhs;
U       = itP->work[0];
U1      = itP->work[1];
V       = itP->work[2];
V1      = itP->work[3];
W       = itP->work[4];
BINVF   = itP->work[5];

/* Compute initial preconditioned residual */
KSPResidual(itP,X,V,U, W, BINVF, B );

/* Test for nothing to do */
VecNorm(W,&rnorm);
if ((*itP->converged)(itP,0,rnorm,itP->cnvP)) { *its = 0; return 0;}
MONITOR(itP,rnorm,0);
if (history) history[0] = rnorm;

VecCopy(B,U);
VecNorm(U,&beta);
tmp = 1.0/beta; VecScale( &tmp, U );
MatMultTrans(Amat, U, V );
VecNorm(V,&alpha);
tmp = 1.0/alpha; VecScale(&tmp, V );

VecCopy(V,W);
VecSet(&zero,X);

phibar = beta;
rhobar = alpha;
for (i=0; i<maxit; i++) {
    MatMult(Amat,V,U1);
    tmp = -alpha; VecAXPY(&tmp,U,U1);
    VecNorm(U1,&beta);
    tmp = 1.0/beta; VecScale(&tmp, U1 );

    MatMultTrans(Amat,U1,V1);
    tmp = -beta; VecAXPY(&tmp,V,V1);
    VecNorm(V1,&alpha);
    tmp = 1.0 / alpha; VecScale(&tmp , V1 );

    rho   = sqrt(rhobar*rhobar + beta*beta);
    c     = rhobar / rho;
    s     = beta / rho;
    theta = s * alpha;
    rhobar= - c * alpha;
    phi   = c * phibar;
    phibar= s * phibar;

    tmp = phi/rho; VecAXPY(&tmp,W,X);      /*    x <- x + (phi/rho) w   */
    tmp = -theta/rho; VecAYPX(&tmp,V1,W);  /*    w <- v - (theta/rho) w */

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

  KSPUnwindPre(  itP, X, W );
  if (cerr <= 0) *its = -(i+1);
  else          *its = i + 1;
  return 0;
}

int KSPCreate_LSQR(KSP itP)
{
  itP->MethodPrivate        = (void *) 0;
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
