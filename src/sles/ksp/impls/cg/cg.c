#ifndef lint
static char vcid[] = "$Id: cg.c,v 1.6 1995/02/28 22:39:27 bsmith Exp bsmith $";
#endif

/*                       
           This implements Preconditioned Conjugate Gradients.       
*/
#include <stdio.h>
#include <math.h>
#include "kspimpl.h"
#include "cgctx.h"

int KSPiCGSetUp(KSP itP)
{
  CGCntx *cgP;
  int    maxit,ierr;
  cgP = (CGCntx *) itP->MethodPrivate;
  maxit = itP->max_it;

  if (itP->method != KSPCG) {
      SETERR(1,"Attempt to use CG Setup on wrong context");}

  /* check user parameters and functions */
  if ( itP->right_pre ) {
      SETERR(2,"Right-inverse preconditioning not supported for CG");}
  if (ierr = KSPCheckDef( itP )) return ierr;

  /* get work vectors from user code */
  if (ierr = KSPiDefaultGetWork( itP, 3 )) return ierr;

  if (itP->calc_eigs) {
    /* get space to store tridiagonal matrix for Lanczo */
    cgP->e = (Scalar *) MALLOC(4*(maxit+1)*sizeof(Scalar)); CHKPTR(cgP->e);
    cgP->d  = cgP->e + maxit + 1; 
    cgP->ee = cgP->d + maxit + 1;
    cgP->dd = cgP->ee + maxit + 1;
  }
  return 0;
}

int  KSPiCGSolve(KSP itP,int *its)
{
  int       ierr, i = 0,maxit,eigs,res,pres, hist_len, cerr;
  Scalar    dpi, a = 1.0,beta,betaold = 1.0,b,*e,*d, mone = -1.0, ma; 
  double   *history, dp;
  Vec       X,B,Z,R,P;
  CGCntx    *cgP;
  cgP = (CGCntx *) itP->MethodPrivate;

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

  if (!itP->guess_zero) {
    MatMult(PCGetMat(itP->B),X,R);              /*   r <- b - Ax      */
    ierr = VecAYPX(&mone,B,R); CHKERR(ierr);
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
  /* Test for nothing to do */
  VecNorm(R,&dp);
  if (CONVERGED(itP,dp,0)) {*its =  RCONV(itP,0); return 0;}
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
         if (eigs) {
           if (b<0.0) SETERR(1,"Nonsymmetric or bad preconditioner");
           e[i] = sqrt(b)/a;  
         }
         ierr = VecAYPX(&b,Z,P); CHKERR(ierr)    /*     p <- z + b* p   */
     }
     betaold = beta;
     MatMult(PCGetMat(itP->B),P,Z);              /*     z <- Kp         */
     VecDot(P,Z,&dpi);
     a = beta/dpi;                             /*     a = beta/p'z    */
     if (eigs) {
       if (b<0.0) SETERR(1,"Nonsymmetric or bad preconditioner");
       d[i] = sqrt(b)*e[i] + 1.0/a;
     }
     VecAXPY(&a,P,X);                           /*     x <- x + ap     */
     ma = -a; VecAXPY(&ma,Z,R);                 /*     r <- r - az     */
     if (pres) {
       MatMult(PCGetMat(itP->B),R,Z);          /*     z <- Br         */
       VecNorm(Z,&dp);                        /*    dp <- z'*z       */
     }
     else {
       VecNorm(R,&dp);                         /*    dp <- r'*r       */       
     }
     if (history && hist_len > i + 1) history[i+1] = dp;
     MONITOR(itP,dp,i+1);
     if (CONVERGED(itP,dp,i+1)) break;
     if (!pres) PCApply(itP->B,R,Z);              /*     z <- Br         */
  }
  if (i == maxit) i--;
  if (history) itP->res_act_size = (hist_len < i + 1) ? hist_len : i + 1;

  /* Update computational work */
  itP->namult += i+1;
  itP->nbinv  += i+1;
  itP->nvectors += (i+1)*10;

  *its = RCONV(itP,i+1); return 0;
}

int KSPiCGDestroy(PetscObject obj)
{
  KSP itP = (KSP) obj;
  CGCntx *cgP;
  cgP = (CGCntx *) itP->MethodPrivate;

  /* free space used for eigenvalue calculations */
  if ( itP->calc_eigs ) {
    FREE(cgP->e);
  }

  KSPiDefaultFreeWork( itP );
  
  /* free the context variables */
  FREE(cgP); FREE(itP);
  return 0;
}

int KSPiCGCreate(KSP itP)
{
  CGCntx *cgP;
  cgP = NEW(CGCntx);  CHKPTR(cgP);
  itP->MethodPrivate = (void *) cgP;
  itP->method               = KSPCG;
  itP->right_pre            = 0;
  itP->calc_res             = 1;
  itP->setup                = KSPiCGSetUp;
  itP->solver               = KSPiCGSolve;
  itP->adjustwork           = KSPiDefaultAdjustWork;
  itP->destroy              = KSPiCGDestroy;
  itP->converged            = KSPDefaultConverged;
  itP->BuildSolution        = KSPDefaultBuildSolution;
  itP->BuildResidual        = KSPDefaultBuildResidual;
  return 0;
}
