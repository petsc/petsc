#ifndef lint
static char vcid[] = "$Id: bcgs.c,v 1.25 1996/01/09 03:30:34 curfman Exp curfman $";
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

  if (itP->pc_side == PC_SYMMETRIC)
    {SETERRQ(2,"KSPSetUp_BCGS:no symmetric preconditioning for KSPBCGS");}
  ierr = KSPCheckDef( itP ); CHKERRQ(ierr);
  return KSPiDefaultGetWork( itP, 7 );
}

static int  KSPSolve_BCGS(KSP itP,int *its)
{
  int       i = 0, maxit, hist_len, cerr = 0,ierr;
  Scalar    rho, rhoold, alpha, beta, omega, omegaold, d1, d2,zero = 0.0, tmp;
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
  ierr = KSPResidual(itP,X,V,T,R,BINVF,B); CHKERRQ(ierr);

  /* Test for nothing to do */
  ierr = VecNorm(R,NORM_2,&dp); CHKERRQ(ierr);
  if ((*itP->converged)(itP,0,dp,itP->cnvP)) {*its = 0; return 0;}
  MONITOR(itP,dp,0);
  if (history) history[0] = dp;

  /* Make the initial Rp == R */
  ierr = VecCopy(R,RP); CHKERRQ(ierr);

  rhoold   = 1.0;
  alpha    = 1.0;
  omegaold = 1.0;
  ierr = VecSet(&zero,P); CHKERRQ(ierr);
  ierr = VecSet(&zero,V); CHKERRQ(ierr);

  for (i=0; i<maxit; i++) {
    ierr = VecDot(R,RP,&rho); CHKERRQ(ierr);       /*   rho <- rp' r       */
    if (rho == 0.0) {fprintf(stderr,"Breakdown\n"); *its = -(i+1);return 0;} 
    beta = (rho/rhoold) * (alpha/omegaold);
    tmp = -omegaold; VecAXPY(&tmp,V,P);            /*   p <- p - w v       */
    ierr = VecAYPX(&beta,R,P); CHKERRQ(ierr);      /*   p <- r + p beta    */
    ierr = PCApplyBAorAB(itP->B,itP->pc_side,
                         P,V,T); CHKERRQ(ierr);    /*   v <- K p           */
    ierr = VecDot(RP,V,&d1); CHKERRQ(ierr);
    alpha = rho / d1; tmp = -alpha;                /*   a <- rho / (rp' v) */
    ierr = VecWAXPY(&tmp,V,R,S); CHKERRQ(ierr);    /*   s <- r - a v       */
    ierr = PCApplyBAorAB(itP->B,itP->pc_side,
                         S,T,R); CHKERRQ(ierr);    /*   t <- K s           */
    ierr = VecDot(S,T,&d1); CHKERRQ(ierr);
    ierr = VecDot(T,T,&d2); CHKERRQ(ierr);
    if (d2 == 0.0) {
      /* t is 0.  if s is 0, then alpha v == r, and hence alpha p
	 may be our solution.  Give it a try? */
      ierr = VecDot(S,S,&d1); CHKERRQ(ierr);
      if (d1 != 0.0) {SETERRQ(1,"KSPSolve_BCGS:Breakdown");}
      ierr = VecAXPY(&alpha,P,X); CHKERRQ(ierr);   /*   x <- x + a p       */
      if (history && hist_len > i+1) history[i+1] = 0.0;
      MONITOR(itP,0.0,i+1);
      break;
    }
    omega = d1 / d2;                               /*   w <- (s't) / (t't) */
    ierr = VecAXPY(&alpha,P,X); CHKERRQ(ierr);     /*   x <- x + a p       */
    ierr = VecAXPY(&omega,S,X); CHKERRQ(ierr);     /*   x <- x + w s       */
    tmp = -omega; 
    ierr = VecWAXPY(&tmp,T,S,R); CHKERRQ(ierr);    /*   r <- s - w t       */
    ierr = VecNorm(R,NORM_2,&dp); CHKERRQ(ierr);

    rhoold   = rho;
    omegaold = omega;

    if (history && hist_len > i + 1) history[i+1] = dp;
    MONITOR(itP,dp,i+1);
    cerr = (*itP->converged)(itP,i+1,dp,itP->cnvP);
    if (cerr) break;    
  }
  if (i == maxit) i--;
  if (history) itP->res_act_size = (hist_len < i + 1) ? hist_len : i + 1;

  ierr = KSPUnwindPre(itP,X,T); CHKERRQ(ierr);
  if (cerr <= 0) *its = -(i+1);
  else           *its = i + 1;
  return 0;
}

int KSPCreate_BCGS(KSP itP)
{
  itP->data                 = (void *) 0;
  itP->type                 = KSPBCGS;
  itP->pc_side              = PC_LEFT;
  itP->calc_res             = 1;
  itP->setup                = KSPSetUp_BCGS;
  itP->solver               = KSPSolve_BCGS;
  itP->adjustwork           = KSPiDefaultAdjustWork;
  itP->destroy              = KSPiDefaultDestroy;
  itP->converged            = KSPDefaultConverged;
  itP->buildsolution        = KSPDefaultBuildSolution;
  itP->buildresidual        = KSPDefaultBuildResidual;
  itP->view                 = 0;
  return 0;
}
