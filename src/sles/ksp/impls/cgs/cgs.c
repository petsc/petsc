#ifndef lint
static char vcid[] = "$Id: cgs.c,v 1.16 1995/08/14 17:06:41 curfman Exp bsmith $";
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
  ierr = KSPCheckDef( itP ); CHKERRQ(ierr);
  return KSPiDefaultGetWork( itP, 8 );
}

static int  KSPSolve_CGS(KSP itP,int *its)
{
  int       i = 0, maxit, hist_len, cerr = 0, ierr;
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
  ierr = KSPResidual(itP,X,V,T, R, BINVF, B ); CHKERRQ(ierr);

  /* Test for nothing to do */
  ierr = VecNorm(R,&dp); CHKERRQ(ierr);
  if ((*itP->converged)(itP,0,dp,itP->cnvP)) {*its = 0; return 0;}
  MONITOR(itP,dp,0);
  if (history) history[0] = dp;

  /* Make the initial Rp == R */
  ierr = VecCopy(R,RP); CHKERRQ(ierr);

  /* Set the initial conditions */
  ierr = VecDot(RP,R,&rhoold); CHKERRQ(ierr);
  ierr = VecCopy(R,U); CHKERRQ(ierr);
  ierr = VecCopy(R,P); CHKERRQ(ierr);
  ierr = PCApplyBAorAB(itP->B,itP->right_pre,P,V,T); CHKERRQ(ierr);

  for (i=0; i<maxit; i++) {
    ierr = VecDot(RP,V,&s); CHKERRQ(ierr);           /* s <- rp' v           */
    a = rhoold / s;                                  /* a <- rho / s         */
    tmp = -a; 
    ierr = VecWAXPY(&tmp,V,U,Q); CHKERRQ(ierr);      /* q <- u - a v         */
    ierr = VecWAXPY(&one,U,Q,T); CHKERRQ(ierr);      /* t <- u + q           */
    ierr = VecAXPY(&a,T,X); CHKERRQ(ierr);           /* x <- x + a (u + q)   */
    ierr = PCApplyBAorAB(itP->B,itP->right_pre,T,AUQ,U); CHKERRQ(ierr);
    ierr = VecAXPY(&tmp,AUQ,R); CHKERRQ(ierr);       /* r <- r - a K (u + q) */
    ierr = VecNorm(R,&dp); CHKERRQ(ierr);

    if (history && hist_len > i + 1) history[i+1] = dp;
    MONITOR(itP,dp,i+1);
    cerr = (*itP->converged)(itP,i+1,dp,itP->cnvP);
    if (cerr) break;

    ierr = VecDot(RP,R,&rho); CHKERRQ(ierr);         /* newrho <- rp' r      */
    b = rho / rhoold;                                /* b <- rho / rhoold    */
    ierr = VecWAXPY(&b,Q,R,U); CHKERRQ(ierr);        /* u <- r + b q         */
    ierr = VecAXPY(&b,P,Q); CHKERRQ(ierr);
    ierr = VecWAXPY(&b,Q,U,P); CHKERRQ(ierr);        /* p <- u + b(q + b p)  */
    ierr = PCApplyBAorAB(itP->B,itP->right_pre,
                         P,V,Q); CHKERRQ(ierr);      /* v <- K p             */
    rhoold = rho;
  }
  if (i == maxit) i--;
  if (history) itP->res_act_size = (hist_len < i + 1) ? hist_len : i + 1;

  ierr = KSPUnwindPre(itP,X,T); CHKERRQ(ierr);
  if (cerr <= 0) *its = -(i+1); 
  else           *its = i+1;
  return 0;
}

int KSPCreate_CGS(KSP itP)
{
  itP->data        = (void *) 0;
  itP->type                 = KSPCGS;
  itP->right_pre            = 0;
  itP->calc_res             = 1;
  itP->setup                = KSPSetUp_CGS;
  itP->solver               = KSPSolve_CGS;
  itP->adjustwork           = KSPiDefaultAdjustWork;
  itP->destroy              = KSPiDefaultDestroy;
  itP->converged            = KSPDefaultConverged;
  itP->buildsolution        = KSPDefaultBuildSolution;
  itP->buildresidual        = KSPDefaultBuildResidual;
  itP->view                 = 0;
  return 0;
}
