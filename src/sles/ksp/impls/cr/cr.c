#ifndef lint
static char vcid[] = "$Id: cr.c,v 1.13 1995/05/18 22:44:10 bsmith Exp bsmith $";
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
  int    ierr;
  if ( itP->right_pre ) {
      SETERRQ(2,"Right-inverse preconditioning not supported for CR");
  }
  if ((ierr = KSPCheckDef( itP ))) return ierr;
  ierr = KSPiDefaultGetWork( itP, 9  );
  return ierr;
}

static int  KSPSolve_CR(KSP itP,int *its)
{
  int          i = 0, maxit,pres, hist_len, cerr;
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

  PCGetOperators(itP->B,&Amat,&Pmat,&pflag);  
  bbotold = 1.0; /* a hack */
  if (!itP->guess_zero) {
    MatMult(Amat,X,R);                        /*   r <- b - Ax      */
    VecAYPX(&mone,B,R);
  }
  else { 
    VecCopy(B,R);                             /*    r <- b (x is 0)  */
  }
  VecSet(&zero,Pm1);                          /*    pm1 <- 0         */
  VecSet(&zero,Sm1);                          /*    sm1 <- 0         */
  VecSet(&zero,Qm1);                          /*    Qm1 <- 0         */
  PCApply(itP->B,R,P);                        /*     p <- Br         */
  if (pres) {
      VecNorm(P,&dp);                         /*    dp <- z'*z       */
      }
  else {
      VecNorm(R,&dp);                         /*    dp <- r'*r       */       
      }
  if (CONVERGED(itP,dp,0)) {*its = 0; return 0;}
  MONITOR(itP,dp,0);
  if (history) history[0] = dp;
  MatMult(Amat,P,Q);                          /*    q <- A p         */

  for ( i=0; i<maxit; i++) {
     PCApply(itP->B,Q,S);                      /*     s <- Bq        */
     VecDot(R,S,&btop);                        /*                    */
     VecDot(Q,S,&bbot);                        /*     lambda =       */
     lambda = btop/bbot;
     VecAXPY(&lambda,P,X);                     /*     x <- x + lambda p     */
     tmp = -lambda; VecAXPY(&tmp,Q,R);         /*     r <- r - lambda q     */
     VecNorm(R,&dp);                           /*    dp <- r'*r       */       
     if (history && hist_len > i + 1) history[i+1] = dp;
     MONITOR(itP,dp,i+1);
     if (CONVERGED(itP,dp,i+1)) break;
     MatMult(Amat,S,T);                        /* T <-   As           */
     VecDot(T,S,&btop);
     alpha0 = btop/bbot;
     VecDot(T,Sm1,&btop);                    
     alpha1 = btop/bbotold; 

     tmp = -alpha0; VecWAXPY(&tmp,P,S,Pp1);
     tmp = -alpha1; VecAXPY(&tmp,Pm1,Pp1);  
     /* MM(Pp1,Qp1); use 3 term recurrence relation instead */
     tmp = -alpha0; VecAXPY(&tmp,Q,Qp1); 
     tmp = -alpha1; VecAXPY(&tmp,Qm1,Qp1); 
     /* scale the search direction !! Not mentioned in any reference */
     VecNorm(Pp1,&dp); 
     tmp = 1.0/dp; VecScale(&tmp,Pp1); VecScale(&tmp,Qp1);
     /* rotate work vectors */
     Tmp = Sm1; Sm1 = S; S = Tmp;
     Tmp = Pm1; Pm1 = P; P = Pp1; Pp1 = Tmp;
     Tmp = Qm1; Qm1 = Q; Q = Qp1; Qp1 = T = Tmp;
     bbotold = bbot; 
  }
  if (i == maxit) i--;
  if (history) itP->res_act_size = (hist_len < i + 1) ? hist_len : i + 1;
  *its = RCONV(itP,i+1); return 0;
}

int KSPCreate_CR(KSP itP)
{
  itP->type                 = KSPCR;
  itP->right_pre            = 0;
  itP->calc_res             = 1;
  itP->setup                = KSPSetUp_CR;
  itP->solver               = KSPSolve_CR;
  itP->adjustwork           = KSPiDefaultAdjustWork;
  itP->destroy              = KSPiDefaultDestroy;
  itP->converged            = KSPDefaultConverged;
  itP->buildsolution        = KSPDefaultBuildSolution;
  itP->buildresidual        = KSPDefaultBuildResidual;
  return 0;
}
