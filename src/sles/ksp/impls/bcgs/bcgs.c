#ifndef lint
static char vcid[] = "$Id: bcgs.c,v 1.9 1995/03/25 01:25:45 bsmith Exp bsmith $";
#endif

/*                       
       This implements BiCG Stab
*/
#include <stdio.h>
#include <math.h>
#include "petsc.h"
#include "kspimpl.h"

static int KSPSetUp_BCGS(KSP itP)
{
  int ierr;
  if ((ierr = KSPCheckDef( itP ))) return ierr;
  if (KSPiDefaultGetWork( itP, 7 )) return ierr;
  return 0;;
}

static int  KSPSolve_BCGS(KSP itP,int *its)
{
int       i = 0, maxit, hist_len, cerr;
Scalar    rho, rhoold, alpha, beta, omega, omegaold, d1, d2;
Scalar    zero = 0.0, tmp;
Vec       X,B,V,P,R,RP,T,S, BINVF;
double    dp, *history;

maxit   = itP->max_it;
history = itP->residual_history;
hist_len= itP->res_hist_size;
X       = itP->vec_sol;
B       = itP->vec_rhs;
R       = itP->work[0];
RP      = itP->work[1];
V       = itP->work[2];
T       = itP->work[3];
S       = itP->work[4];
P       = itP->work[5];
BINVF   = itP->work[6];

/* Compute initial preconditioned residual */
KSPResidual(itP,X,V,T, R, BINVF, B );

/* Test for nothing to do */
VecNorm(R,&dp);
if (CONVERGED(itP,dp,0)) {*its = 0; return 0;}
MONITOR(itP,dp,0);
if (history) history[0] = dp;

/* Make the initial Rp == R */
VecCopy(R,RP);

rhoold   = 1.0;
alpha    = 1.0;
omegaold = 1.0;
VecSet(&zero,P);
VecSet(&zero,V);

for (i=0; i<maxit; i++) {
    VecDot(R,RP,&rho);                       /*   rho <- rp' r     */
    if (rho == 0.0) {fprintf(stderr,"Breakdown\n"); *its = -(i+1);return 0;} 
    beta = (rho/rhoold) * (alpha/omegaold);
    tmp = -omegaold; VecAXPY(&tmp,V,P);        /*     p <- p - w v   */
    VecAYPX(&beta,R,P);                      /*     p <- r + p beta */
    PCApplyBAorAB(itP->B,itP->right_pre,P,V,T);  /*     v <- K p       */
    VecDot(RP,V,&d1);
    alpha = rho / d1;                     /*     a <- rho / (rp' v) */
    tmp = -alpha; VecWAXPY(&tmp,V,R,S);       /*     s <- r - a v   */
    PCApplyBAorAB(itP->B,itP->right_pre,S,T,R); /*     t <- K s       */
    VecDot(S,T,&d1);
    VecDot(T,T,&d2);
    if (d2 == 0.0) {
	/* t is 0.  if s is 0, then alpha v == r, and hence alpha p
	   may be our solution.  Give it a try? */
	VecDot(S,S,&d1);
	if (d1 != 0.0) {
	    SETERR(1,"Breakdown in BCGS");
	    }
	VecAXPY(&alpha,P,X);                     /*     x <- x + a p   */
	if (history && hist_len > i+1) history[i+1] = 0.0;
	MONITOR(itP,0.0,i+1);
	break;
	}
    omega = d1 / d2;                      /*     w <- (s't) / (t't) */
    VecAXPY(&alpha,P,X);                     /*     x <- x + a p   */
    VecAXPY(&omega,S,X);                     /*     x <- x + w s   */
    tmp = -omega; VecWAXPY(&tmp,T,S,R);          /*     r <- s - w t   */
    VecNorm(R,&dp);

    rhoold   = rho;
    omegaold = omega;

    if (history && hist_len > i + 1) history[i+1] = dp;
    MONITOR(itP,dp,i+1);
    if (CONVERGED(itP,dp,i+1)) break;
    }
if (i == maxit) i--;
if (history) itP->res_act_size = (hist_len < i + 1) ? hist_len : i + 1;

/* Get floating point work */
itP->nmatop += (i * 2);
itP->nvectors += (i) * 24;

KSPUnwindPre( itP, X, T );
*its = RCONV(itP,i+1); return 0;
}

int KSPCreate_BCGS(KSP itP)
{
  itP->MethodPrivate = (void *) 0;
  itP->type                 = KSPBCGS;
  itP->right_pre            = 0;
  itP->calc_res             = 1;
  itP->setup                = KSPSetUp_BCGS;
  itP->solver               = KSPSolve_BCGS;
  itP->adjustwork           = KSPiDefaultAdjustWork;
  itP->destroy              = KSPiDefaultDestroy;
  itP->converged            = KSPDefaultConverged;
  itP->BuildSolution        = KSPDefaultBuildSolution;
  itP->BuildResidual        = KSPDefaultBuildResidual;
  return 0;
}
