#ifndef lint
static char vcid[] = "$Id: cgs.c,v 1.8 1995/03/25 01:25:49 bsmith Exp bsmith $";
#endif

/*                       
       This implements CGS
*/
#include <stdio.h>
#include <math.h>
#include "petsc.h"
#include "kspimpl.h"

static int KSPSetUp_CGS(KSP itP)
{
  int ierr;
  if ((ierr = KSPCheckDef( itP ))) return ierr;
  if ((ierr = KSPiDefaultGetWork( itP, 8 ))) return ierr;
  return 0;
}


static int  KSPSolve_CGS(KSP itP,int *its)
{
int       i = 0, maxit, hist_len, cerr, ierr;
Scalar    rho, rhoold, a, s, b, tmp, one = 1.0; 
Vec       X,B,V,P,R,RP,T,Q,U, BINVF, AUQ;
double    *history, dp;

maxit   = itP->max_it;
history = itP->residual_history;
hist_len= itP->res_hist_size;
X       = itP->vec_sol;
B       = itP->vec_rhs;
R       = itP->work[0];
RP      = itP->work[1];
V       = itP->work[2];
T       = itP->work[3];
Q       = itP->work[4];
P       = itP->work[5];
BINVF   = itP->work[6];
U       = itP->work[7];
AUQ     = V;

/* Compute initial preconditioned residual */
KSPResidual(itP,X,V,T, R, BINVF, B );

/* Test for nothing to do */
VecNorm(R,&dp);
if (CONVERGED(itP,dp,0)) {*its = 0; return 0;}
MONITOR(itP,dp,0);
if (history) history[0] = dp;

/* Make the initial Rp == R */
if ((ierr = VecCopy(R,RP))) SETERR(ierr,0);

/* Set the initial conditions */
VecDot(RP,R,&rhoold);
if ((ierr = VecCopy(R,U))) SETERR(ierr,0);
if ((ierr = VecCopy(R,P))) SETERR(ierr,0);
PCApplyBAorAB(itP->B,itP->right_pre,P,V,T);

for (i=0; i<maxit; i++) {
    VecDot(RP,V,&s);                        /* s <- rp' v          */
    a = rhoold / s;                        /* a <- rho / s        */
    tmp = -a;VecWAXPY(&tmp,V,U,Q);          /* q <- u - a v        */
    VecWAXPY(&one,U,Q,T);                   /* t <- u + q          */
    if ((ierr = VecAXPY(&a,T,X))) SETERR(ierr,0);  /* x <- x + a (u + q)  */
    PCApplyBAorAB(itP->B,itP->right_pre,T,AUQ,U);
    if ((ierr = VecAXPY(&tmp,AUQ,R))) SETERR(ierr,0);/* r <- r - a K (u + q) */
    VecNorm(R,&dp);

    if (history && hist_len > i + 1) history[i+1] = dp;
    MONITOR(itP,dp,i+1);
    if (CONVERGED(itP,dp,i+1)) break;

    VecDot(RP,R,&rho);                      /* newrho <- rp' r       */
    b = rho / rhoold;                      /* b <- rho / rhoold     */
    VecWAXPY(&b,Q,R,U);                     /* u <- r + b q          */
    if ((ierr = VecAXPY(&b,P,Q))) SETERR(ierr,0);
    VecWAXPY(&b,Q,U,P);                     /* p <- u + b(q + b p)   */
    PCApplyBAorAB(itP->B,itP->right_pre,P,V,Q);    /* v <- K p              */
    rhoold = rho;
    }
if (i == maxit) {i--; itP->nmatop++; itP->nvectors += 4;}
itP->nmatop   += 2 + 2*i;
itP->nvectors +=  8 + 10*i;
if (history) itP->res_act_size = (hist_len < i + 1) ? hist_len : i + 1;

KSPUnwindPre(  itP, X, T );
*its =  RCONV(itP,i+1); return 0;
}

int KSPCreate_CGS(KSP itP)
{
  itP->MethodPrivate        = (void *) 0;
  itP->type                 = KSPCGS;
  itP->right_pre            = 0;
  itP->calc_res             = 1;
  itP->setup                = KSPSetUp_CGS;
  itP->solver               = KSPSolve_CGS;
  itP->adjustwork           = KSPiDefaultAdjustWork;
  itP->destroy              = KSPiDefaultDestroy;
  itP->converged            = KSPDefaultConverged;
  itP->BuildSolution        = KSPDefaultBuildSolution;
  itP->BuildResidual        = KSPDefaultBuildResidual;
  return 0;
}
