#ifndef lint
static char vcid[] = "$Id: cgs.c,v 1.23 1996/03/10 17:27:08 bsmith Exp bsmith $";
#endif

/*                       
    This code implements the CGS (Conjugate Gradient Squared) method. 
    Reference: Sonneveld, 1989.

    Note that for the complex numbers version, the VecDot() arguments
    within the code MUST remain in the order given for correct computation
    of inner products.
*/
#include <stdio.h>
#include <math.h>
#include "petsc.h"
#include "kspimpl.h"

static int KSPSetUp_CGS(KSP ksp)
{
  int ierr;
  if (ksp->pc_side == PC_SYMMETRIC)
    {SETERRQ(2,"KSPSetUp_CGS:no symmetric preconditioning for KSPCGS");}
  ierr = KSPCheckDef( ksp ); CHKERRQ(ierr);
  return KSPiDefaultGetWork( ksp, 8 );
}

static int  KSPSolve_CGS(KSP ksp,int *its)
{
  int       i = 0, maxit, hist_len, cerr = 0, ierr;
  Scalar    rho, rhoold, a, s, b, tmp, one = 1.0; 
  Vec       X,B,V,P,R,RP,T,Q,U, BINVF, AUQ;
  double    *history, dp;

  maxit   = ksp->max_it;
  history = ksp->residual_history;
  hist_len= ksp->res_hist_size;
  X       = ksp->vec_sol;
  B       = ksp->vec_rhs;
  R       = ksp->work[0];
  RP      = ksp->work[1];
  V       = ksp->work[2];
  T       = ksp->work[3];
  Q       = ksp->work[4];
  P       = ksp->work[5];
  BINVF   = ksp->work[6];
  U       = ksp->work[7];
  AUQ     = V;

  /* Compute initial preconditioned residual */
  ierr = KSPResidual(ksp,X,V,T, R, BINVF, B ); CHKERRQ(ierr);

  /* Test for nothing to do */
  ierr = VecNorm(R,NORM_2,&dp); CHKERRQ(ierr);
  if ((*ksp->converged)(ksp,0,dp,ksp->cnvP)) {*its = 0; return 0;}
  MONITOR(ksp,dp,0);
  if (history) history[0] = dp;

  /* Make the initial Rp == R */
  ierr = VecCopy(R,RP); CHKERRQ(ierr);

  /* Set the initial conditions */
  ierr = VecDot(RP,R,&rhoold); CHKERRQ(ierr);
  ierr = VecCopy(R,U); CHKERRQ(ierr);
  ierr = VecCopy(R,P); CHKERRQ(ierr);
  ierr = PCApplyBAorAB(ksp->B,ksp->pc_side,P,V,T); CHKERRQ(ierr);

  for (i=0; i<maxit; i++) {
    ierr = VecDot(RP,V,&s); CHKERRQ(ierr);           /* s <- rp' v           */
    a = rhoold / s;                                  /* a <- rho / s         */
    tmp = -a; 
    ierr = VecWAXPY(&tmp,V,U,Q); CHKERRQ(ierr);      /* q <- u - a v         */
    ierr = VecWAXPY(&one,U,Q,T); CHKERRQ(ierr);      /* t <- u + q           */
    ierr = VecAXPY(&a,T,X); CHKERRQ(ierr);           /* x <- x + a (u + q)   */
    ierr = PCApplyBAorAB(ksp->B,ksp->pc_side,T,AUQ,U); CHKERRQ(ierr);
    ierr = VecAXPY(&tmp,AUQ,R); CHKERRQ(ierr);       /* r <- r - a K (u + q) */
    ierr = VecNorm(R,NORM_2,&dp); CHKERRQ(ierr);

    if (history && hist_len > i + 1) history[i+1] = dp;
    MONITOR(ksp,dp,i+1);
    cerr = (*ksp->converged)(ksp,i+1,dp,ksp->cnvP);
    if (cerr) break;

    ierr = VecDot(RP,R,&rho); CHKERRQ(ierr);         /* newrho <- rp' r      */
    b    = rho / rhoold;                             /* b <- rho / rhoold    */
    ierr = VecWAXPY(&b,Q,R,U); CHKERRQ(ierr);        /* u <- r + b q         */
    ierr = VecAXPY(&b,P,Q); CHKERRQ(ierr);
    ierr = VecWAXPY(&b,Q,U,P); CHKERRQ(ierr);        /* p <- u + b(q + b p)  */
    ierr = PCApplyBAorAB(ksp->B,ksp->pc_side,
                         P,V,Q); CHKERRQ(ierr);      /* v <- K p             */
    rhoold = rho;
  }
  if (i == maxit) i--;
  if (history) ksp->res_act_size = (hist_len < i + 1) ? hist_len : i + 1;

  ierr = KSPUnwindPreconditioner(ksp,X,T); CHKERRQ(ierr);
  if (cerr <= 0) *its = -(i+1); 
  else           *its = i+1;
  return 0;
}

int KSPCreate_CGS(KSP ksp)
{
  ksp->data                 = (void *) 0;
  ksp->type                 = KSPCGS;
  ksp->pc_side              = PC_LEFT;
  ksp->calc_res             = 1;
  ksp->setup                = KSPSetUp_CGS;
  ksp->solver               = KSPSolve_CGS;
  ksp->adjustwork           = KSPiDefaultAdjustWork;
  ksp->destroy              = KSPiDefaultDestroy;
  ksp->converged            = KSPDefaultConverged;
  ksp->buildsolution        = KSPDefaultBuildSolution;
  ksp->buildresidual        = KSPDefaultBuildResidual;
  ksp->view                 = 0;
  return 0;
}
