#ifndef lint
static char vcid[] = "$Id: cgs.c,v 1.2 1994/08/21 23:56:49 bsmith Exp $";
#endif

/*                       
       This implements CGS
*/
#include <stdio.h>
#include <math.h>
#include "petsc.h"
#include "kspimpl.h"

static int  KSPiCGSSolve();
static int KSPiCGSSetUp();

int KSPiCGSCreate(itP)
KSP    itP;
{
itP->MethodPrivate        = (void *) 0;
itP->method               = KSPCGS;
itP->right_pre            = 0;
itP->calc_res             = 1;
itP->setup                = KSPiCGSSetUp;
itP->solver               = KSPiCGSSolve;
itP->adjustwork           = KSPiDefaultAdjustWork;
itP->destroy              = KSPiDefaultDestroy;
return 0;
}


static int KSPiCGSSetUp(itP)
KSP itP;
{
  int ierr;
  if (ierr = KSPCheckDef( itP )) return ierr;
  if (ierr = KSPiDefaultGetWork( itP, 8 )) return ierr;
  return 0;
}


static int  KSPiCGSSolve(itP,its)
KSP itP;
int *its;
{
int       i = 0, maxit, res, pres, hist_len, cerr;
double    rho, rhoold, a, s, b, *history, dp, tmp, one = 1.0; 
Vec       X,B,V,P,R,RP,T,Q,U, BINVF, AUQ;

res     = itP->calc_res;
pres    = itP->use_pres;
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
VecCopy(R,RP);

/* Set the initial conditions */
VecDot(RP,R,&rhoold);
VecCopy(R,U);
VecCopy(R,P);
MATOP(itP,P,V,T);

for (i=0; i<maxit; i++) {
    VecDot(RP,V,&s);                        /* s <- rp' v          */
    a = rhoold / s;                        /* a <- rho / s        */
    tmp = -a;VecWAXPY(&tmp,V,U,Q);          /* q <- u - a v        */
    VecWAXPY(&one,U,Q,T);                   /* t <- u + q          */
    VecAXPY(&a,T,X);                        /* x <- x + a (u + q)  */
    MATOP(itP,T,AUQ,U);
    VecAXPY(&tmp,AUQ,R);                    /* r <- r - a K (u + q) */
    VecNorm(R,&dp);

    if (history && hist_len > i + 1) history[i+1] = dp;
    MONITOR(itP,dp,i+1);
    if (CONVERGED(itP,dp,i+1)) break;

    VecDot(RP,R,&rho);                      /* newrho <- rp' r       */
    b = rho / rhoold;                      /* b <- rho / rhoold     */
    VecWAXPY(&b,Q,R,U);                     /* u <- r + b q          */
    VecAXPY(&b,P,Q);
    VecWAXPY(&b,Q,U,P);                     /* p <- u + b(q + b p)   */
    MATOP(itP,P,V,Q);                      /* v <- K p              */
    rhoold = rho;
    }
if (i == maxit) {i--; itP->nmatop++; itP->nvectors += 4;}
itP->nmatop   += 2 + 2*i;
itP->nvectors +=  8 + 10*i;
if (history) itP->res_act_size = (hist_len < i + 1) ? hist_len : i + 1;

KSPUnwindPre(  itP, X, T );
*its =  RCONV(itP,i+1); return 0;
}
