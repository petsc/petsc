#ifndef lint
static char vcid[] = "$Id: cg.c,v 1.24 1995/07/17 03:53:53 bsmith Exp curfman $";
#endif

/*                       
           This implements Preconditioned Conjugate Gradients.       
*/
#include <stdio.h>
#include <math.h>
#include "kspimpl.h"
#include "cgctx.h"

int KSPSetUp_CG(KSP itP)
{
  KSP_CG *cgP;
  int    maxit,ierr;
  cgP = (KSP_CG *) itP->MethodPrivate;
  maxit = itP->max_it;

  /* check user parameters and functions */
  if ( itP->right_pre ) {
    SETERRQ(2,"KSPSetUp_CG: no right-inverse preconditioning for CG");}
  if ((ierr = KSPCheckDef( itP ))) return ierr;

  /* get work vectors from user code */
  if ((ierr = KSPiDefaultGetWork( itP, 3 ))) return ierr;

  if (itP->calc_eigs) {
    /* get space to store tridiagonal matrix for Lanczo */
    cgP->e = (Scalar *) PETSCMALLOC(4*(maxit+1)*sizeof(Scalar)); CHKPTRQ(cgP->e);
    cgP->d  = cgP->e + maxit + 1; 
    cgP->ee = cgP->d + maxit + 1;
    cgP->dd = cgP->ee + maxit + 1;
  }
  return 0;
}

int  KSPSolve_CG(KSP itP,int *its)
{
  int          ierr, i = 0,maxit,eigs,pres, hist_len, cerr;
  Scalar       dpi, a = 1.0,beta,betaold = 1.0,b,*e = 0,*d = 0, mone = -1.0, ma; 
  double       *history, dp;
  Vec          X,B,Z,R,P;
  KSP_CG       *cgP;
  Mat          Amat, Pmat;
  MatStructure pflag;
  cgP = (KSP_CG *) itP->MethodPrivate;

  eigs    = itP->calc_eigs;
  pres    = itP->use_pres;
  maxit   = itP->max_it;
  history = itP->residual_history;
  hist_len= itP->res_hist_size;
  X       = itP->vec_sol;
  B       = itP->vec_rhs;
  R       = itP->work[0];
  Z       = itP->work[1];
  P       = itP->work[2];

  if (eigs) {e = cgP->e; d = cgP->d; e[0] = 0.0; b = 0.0; }
  PCGetOperators(itP->B,&Amat,&Pmat,&pflag);

  if (!itP->guess_zero) {
    MatMult(Amat,X,R);              /*   r <- b - Ax      */
    ierr = VecAYPX(&mone,B,R); CHKERRQ(ierr);
  }
  else { 
    VecCopy(B,R);                            /*     r <- b (x is 0)*/
  }
  PCApply(itP->B,R,Z);                         /*     z <- Br        */
  if (pres) {
      VecNorm(Z,&dp);                         /*    dp <- z'*z       */
  }
  else {
      VecNorm(R,&dp);                         /*    dp <- r'*r       */       
  }
  cerr = (*itP->converged)(itP,0,dp,itP->cnvP);
  if (cerr) {*its =  0; return 0;}
  MONITOR(itP,dp,0);
  if (history) history[0] = dp;

  for ( i=0; i<maxit; i++) {
     VecDot(R,Z,&beta);                        /*     beta <- r'z    */
     if (i == 0) {
           if (beta == 0.0) break;
           VecCopy(Z,P);                       /*     p <- z         */
     }
     else {
         b = beta/betaold;
#if !defined(PETSC_COMPLEX)
         if (b<0.0) SETERRQ(1,"KSPSolve_CG:Nonsymmetric/bad preconditioner");
#endif
         if (eigs) {
           e[i] = sqrt(b)/a;  
         }
         ierr = VecAYPX(&b,Z,P); CHKERRQ(ierr) /*     p <- z + b* p   */
     }
     betaold = beta;
     MatMult(Amat,P,Z);                        /*     z <- Kp         */
     VecDot(P,Z,&dpi);
     a = beta/dpi;                             /*     a = beta/p'z    */
     if (eigs) {
       d[i] = sqrt(b)*e[i] + 1.0/a;
     }
     VecAXPY(&a,P,X);                          /*     x <- x + ap     */
     ma = -a; VecAXPY(&ma,Z,R);                /*     r <- r - az     */
     if (pres) {
       MatMult(Amat,R,Z);                      /*     z <- Br         */
       VecNorm(Z,&dp);                         /*    dp <- z'*z       */
     }
     else {
       VecNorm(R,&dp);                         /*    dp <- r'*r       */       
     }
     if (history && hist_len > i + 1) history[i+1] = dp;
     MONITOR(itP,dp,i+1);
     cerr = (*itP->converged)(itP,i+1,dp,itP->cnvP);
     if (cerr) break;
     if (!pres) PCApply(itP->B,R,Z);           /*     z <- Br         */
  }
  if (i == maxit) i--;
  if (history) itP->res_act_size = (hist_len < i + 1) ? hist_len : i + 1;
  if (cerr <= 0) *its = -(i+1);
  else          *its = i+1;
  return 0;
}

int KSPDestroy_CG(PetscObject obj)
{
  KSP itP = (KSP) obj;
  KSP_CG *cgP;
  cgP = (KSP_CG *) itP->MethodPrivate;

  /* free space used for eigenvalue calculations */
  if ( itP->calc_eigs ) {
    PETSCFREE(cgP->e);
  }

  KSPiDefaultFreeWork( itP );
  
  /* free the context variables */
  PETSCFREE(cgP); 
  return 0;
}

int KSPCreate_CG(KSP itP)
{
  KSP_CG *cgP;
  cgP = (KSP_CG*) PETSCMALLOC(sizeof(KSP_CG));  CHKPTRQ(cgP);
  itP->MethodPrivate = (void *) cgP;
  itP->type                 = KSPCG;
  itP->right_pre            = 0;
  itP->calc_res             = 1;
  itP->setup                = KSPSetUp_CG;
  itP->solver               = KSPSolve_CG;
  itP->adjustwork           = KSPiDefaultAdjustWork;
  itP->destroy              = KSPDestroy_CG;
  itP->converged            = KSPDefaultConverged;
  itP->buildsolution        = KSPDefaultBuildSolution;
  itP->buildresidual        = KSPDefaultBuildResidual;
  itP->view                 = 0;
  return 0;
}
