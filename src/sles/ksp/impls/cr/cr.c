#ifndef lint
static char vcid[] = "$Id: cr.c,v 1.21 1995/11/05 18:52:47 bsmith Exp curfman $";
#endif

/*                       
           This implements Preconditioned Conjugate Residuals.       
*/
#include <stdio.h>
#include <math.h>
#include "petsc.h"
#include "kspimpl.h"

static int KSPSetUp_CR(KSP itP)
{
  int ierr;
  if (itP->pc_side == KSP_RIGHT_PC)
    {SETERRQ(2,"KSPSetUp_CR:no right preconditioning for KSPCR");}
  else if (itP->pc_side == KSP_SYMMETRIC_PC)
    {SETERRQ(2,"KSPSetUp_CR:no symmetric preconditioning for KSPCR");}
  ierr = KSPCheckDef( itP ); CHKERRQ(ierr);
  ierr = KSPiDefaultGetWork( itP, 9  ); CHKERRQ(ierr);
  return ierr;
}

static int  KSPSolve_CR(KSP itP,int *its)
{
  int          i = 0, maxit,pres, hist_len, cerr = 0, ierr;
  MatStructure pflag;
  double       *history, dp;
  Scalar       lambda, alpha0, alpha1; 
  Scalar       btop, bbot, bbotold, tmp, zero = 0.0, mone = -1.0;
  Vec          X,B,R,Pm1,P,Pp1,Sm1,S,Qm1,Q,Qp1,T, Tmp;
  Mat          Amat, Pmat;

  pres    = itP->use_pres;
  maxit   = itP->max_it;
  history = itP->residual_history;
  hist_len= itP->res_hist_size;
  X       = itP->vec_sol;
  B       = itP->vec_rhs;
  R       = itP->work[0];
  Pm1     = itP->work[1];
  P       = itP->work[2];
  Pp1     = itP->work[3];
  Qm1     = itP->work[4];
  Q       = itP->work[5];
  Qp1 = T = itP->work[6];
  Sm1     = itP->work[7];
  S       = itP->work[8];

  ierr = PCGetOperators(itP->B,&Amat,&Pmat,&pflag); CHKERRQ(ierr);
  bbotold = 1.0; /* a hack */
  if (!itP->guess_zero) {
    ierr = MatMult(Amat,X,R); CHKERRQ(ierr);    /*   r <- b - Ax       */
    ierr = VecAYPX(&mone,B,R); CHKERRQ(ierr);
  }
  else { 
    ierr = VecCopy(B,R); CHKERRQ(ierr);         /*    r <- b (x is 0)  */
  }
  ierr = VecSet(&zero,Pm1); CHKERRQ(ierr);      /*    pm1 <- 0         */
  ierr = VecSet(&zero,Sm1); CHKERRQ(ierr);      /*    sm1 <- 0         */
  ierr = VecSet(&zero,Qm1); CHKERRQ(ierr);      /*    Qm1 <- 0         */
  ierr = PCApply(itP->B,R,P); CHKERRQ(ierr);    /*     p <- Br         */
  if (pres) {
    ierr = VecNorm(P,NORM_2,&dp); CHKERRQ(ierr);/*    dp <- z'*z       */
  }
  else {
    ierr = VecNorm(R,NORM_2,&dp); CHKERRQ(ierr);/*    dp <- r'*r       */
  }
  if ((*itP->converged)(itP,0,dp,itP->cnvP)) {*its = 0; return 0;}
  MONITOR(itP,dp,0);
  if (history) history[0] = dp;
  ierr = MatMult(Amat,P,Q); CHKERRQ(ierr);      /*    q <- A p          */

  for ( i=0; i<maxit; i++) {
    ierr   = PCApply(itP->B,Q,S); CHKERRQ(ierr);  /*     s <- Bq          */
    ierr   = VecDot(R,S,&btop); CHKERRQ(ierr);    /*                      */
    ierr   = VecDot(Q,S,&bbot); CHKERRQ(ierr);    /*     lambda =         */
    lambda = btop/bbot;
    ierr   = VecAXPY(&lambda,P,X); CHKERRQ(ierr); /*   x <- x + lambda p  */
    tmp    = -lambda; 
    ierr   = VecAXPY(&tmp,Q,R); CHKERRQ(ierr);     /*   r <- r - lambda q  */
    ierr   = VecNorm(R,NORM_2,&dp); CHKERRQ(ierr); /*   dp <- r'*r         */
    if (history && hist_len > i + 1) history[i+1] = dp;
    MONITOR(itP,dp,i+1);
    cerr   = (*itP->converged)(itP,i+1,dp,itP->cnvP);
    if (cerr) break;
    ierr   = MatMult(Amat,S,T); CHKERRQ(ierr);    /*   T <-   As          */
    ierr   = VecDot(T,S,&btop); CHKERRQ(ierr);
    alpha0 = btop/bbot;
    ierr   = VecDot(T,Sm1,&btop); CHKERRQ(ierr);       
    alpha1 = btop/bbotold; 

    tmp = -alpha0; ierr = VecWAXPY(&tmp,P,S,Pp1); CHKERRQ(ierr);
    tmp = -alpha1; ierr = VecAXPY(&tmp,Pm1,Pp1); CHKERRQ(ierr);
    /* MM(Pp1,Qp1); use 3 term recurrence relation instead */
    tmp = -alpha0; ierr = VecAXPY(&tmp,Q,Qp1); CHKERRQ(ierr);
    tmp = -alpha1; ierr = VecAXPY(&tmp,Qm1,Qp1); CHKERRQ(ierr);
    /* scale the search direction !! Not mentioned in any reference */
    ierr = VecNorm(Pp1,NORM_2,&dp); CHKERRQ(ierr);
    tmp  = 1.0/dp; ierr = VecScale(&tmp,Pp1); CHKERRQ(ierr);
    ierr = VecScale(&tmp,Qp1); CHKERRQ(ierr);
    /* rotate work vectors */
    Tmp = Sm1; Sm1 = S; S = Tmp;
    Tmp = Pm1; Pm1 = P; P = Pp1; Pp1 = Tmp;
    Tmp = Qm1; Qm1 = Q; Q = Qp1; Qp1 = T = Tmp;
    bbotold = bbot; 
  }
  if (i == maxit) i--;
  if (history) itP->res_act_size = (hist_len < i + 1) ? hist_len : i + 1;
  if (cerr <= 0) *its = -(i+1);
  else           *its = i + 1;
  return 0;
}

int KSPCreate_CR(KSP itP)
{
  itP->type                 = KSPCR;
  itP->pc_side              = KSP_LEFT_PC;
  itP->calc_res             = 1;
  itP->setup                = KSPSetUp_CR;
  itP->solver               = KSPSolve_CR;
  itP->adjustwork           = KSPiDefaultAdjustWork;
  itP->destroy              = KSPiDefaultDestroy;
  itP->converged            = KSPDefaultConverged;
  itP->buildsolution        = KSPDefaultBuildSolution;
  itP->buildresidual        = KSPDefaultBuildResidual;
  itP->view                 = 0;
  return 0;
}
