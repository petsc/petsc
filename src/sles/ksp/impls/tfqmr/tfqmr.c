#ifndef lint
static char vcid[] = "$Id: tfqmr.c,v 1.12 1995/07/26 01:07:42 curfman Exp curfman $";
#endif

/*                       
       This implements TFQMR
*/
#include <stdio.h>
#include <math.h>
#include "petsc.h"
#include "kspimpl.h"

static int KSPSetUp_TFQMR(KSP itP)
{
  int ierr;
  if ((ierr = KSPCheckDef( itP ))) return ierr;
  if ((ierr = KSPiDefaultGetWork( itP,  10 ))) return ierr;
  return 0;
}

static int  KSPSolve_TFQMR(KSP itP,int *its)
{
  int       i = 0, maxit, m, conv, hist_len, cerr = 0, ierr;
  Scalar    rho, rhoold, a, s, b, eta,
            etaold, psiold,  cf, tmp, one = 1.0, zero = 0.0;
  double    *history,dp,dpold,w,dpest,tau,psi,cm;
  Vec       X,B,V,P,R,RP,T,T1,Q,U, D, BINVF, AUQ;

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
  D       = itP->work[8];
  T1      = itP->work[9];
  AUQ     = V;

  /* Compute initial preconditioned residual */
  ierr = KSPResidual(itP,X,V,T, R, BINVF, B ); CHKERRQ(ierr);

  /* Test for nothing to do */
  ierr = VecNorm(R,&dp); CHKERRQ(ierr);
  if ((*itP->converged)(itP,0,dp,itP->cnvP)) {*its = 0; return 0;}
  MONITOR(itP,dp,0);

  /* Make the initial Rp == R */
  ierr = VecCopy(R,RP); CHKERRQ(ierr);

  /* Set the initial conditions */
  etaold = 0.0;
  psiold = 0.0;
  tau    = dp;
  dpold  = dp;

  ierr = VecDot(RP,R,&rhoold); CHKERRQ(ierr);
  ierr = VecCopy(R,U); CHKERRQ(ierr);
  ierr = VecCopy(R,P); CHKERRQ(ierr);
  ierr = PCApplyBAorAB(itP->B,itP->right_pre,P,V,T); CHKERRQ(ierr);
  ierr = VecSet(&zero,D); CHKERRQ(ierr);

  for (i=0; i<maxit; i++) {
    ierr = VecDot(RP,V,&s); CHKERRQ(ierr);          /* s <- rp' v           */
    a = rhoold / s;                                 /* a <- rho / s         */
    tmp = -a; VecWAXPY(&tmp,V,U,Q); CHKERRQ(ierr);  /* q <- u - a v         */
    ierr = VecWAXPY(&one,U,Q,T); CHKERRQ(ierr);     /* t <- u + q           */
    ierr = PCApplyBAorAB(itP->B,itP->right_pre,T,AUQ,T1); CHKERRQ(ierr);
    ierr = VecAXPY(&tmp,AUQ,R); CHKERRQ(ierr);      /* r <- r - a K (u + q) */
    ierr = VecNorm(R,&dp); CHKERRQ(ierr);
    for (m=0; m<2; m++) {
      if (m == 0)
        w = sqrt(dp*dpold);
      else 
        w = dp;
      psi = w / tau;
      cm  = 1.0 / sqrt( 1.0 + psi * psi );
      tau = tau * psi * cm;
      eta = cm * cm * a;
      cf  = psiold * psiold * etaold / a;
      if (m == 0) {
        ierr = VecAYPX(&cf,U,D); CHKERRQ(ierr);
      }
      else {
	ierr = VecAYPX(&cf,Q,D); CHKERRQ(ierr);
      }
      ierr = VecAXPY(&eta,D,X); CHKERRQ(ierr);

      dpest = sqrt(m + 1.0) * tau;
      if (history && hist_len > i + 1) history[i+1] = dpest;
      MONITOR(itP,dpest,i+1);
      if ((conv = cerr = (*itP->converged)(itP,i+1,dpest,itP->cnvP))) break;

      etaold = eta;
      psiold = psi;
    }
    if (conv) break;

    ierr = VecDot(RP,R,&rho); CHKERRQ(ierr);        /* newrho <- rp' r     */
    b = rho / rhoold;                               /* b <- rho / rhoold   */
    ierr = VecWAXPY(&b,Q,R,U); CHKERRQ(ierr);       /* u <- r + b q        */
    ierr = VecAXPY(&b,P,Q); CHKERRQ(ierr);
    ierr = VecWAXPY(&b,Q,U,P); CHKERRQ(ierr);       /* p <- u + b(q + b p) */
    ierr = PCApplyBAorAB(itP->B,itP->right_pre,
                         P,V,Q); CHKERRQ(ierr);     /* v <- K p            */

    rhoold = rho;
    dpold  = dp;
  }
  if (i == maxit) i--;
  if (history) itP->res_act_size = (hist_len < i + 1) ? hist_len : i + 1;

  ierr = KSPUnwindPre(itP,X,T); CHKERRQ(ierr);
  if (cerr <= 0) *its = -(i+1);
  else          *its = i + 1;
  return 0;
}

int KSPCreate_TFQMR(KSP itP)
{
  itP->MethodPrivate        = (void *) 0;
  itP->type                 = KSPTFQMR;
  itP->right_pre            = 0;
  itP->calc_res             = 1;
  itP->setup                = KSPSetUp_TFQMR;
  itP->solver               = KSPSolve_TFQMR;
  itP->adjustwork           = KSPiDefaultAdjustWork;
  itP->destroy              = KSPiDefaultDestroy;
  itP->converged            = KSPDefaultConverged;
  itP->buildsolution        = KSPDefaultBuildSolution;
  itP->buildresidual        = KSPDefaultBuildResidual;
  itP->view                 = 0;
  return 0;
}
