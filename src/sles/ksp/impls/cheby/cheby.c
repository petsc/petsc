#ifndef lint
static char vcid[] = "$Id: cheby.c,v 1.23 1995/08/14 17:06:53 curfman Exp bsmith $";
#endif
/*
    This is a first attempt at a Chebychev Routine, it is not 
    necessarily well optimized.
*/
#include <stdio.h>
#include <math.h>
#include "petsc.h"
#include "kspimpl.h"    /*I "ksp.h" I*/
#include "chebctx.h"
#include "pviewer.h"

int KSPSetUp_Chebychev(KSP itP)
{
  int ierr;
  ierr = KSPCheckDef(itP); CHKERRQ(ierr);
  return KSPiDefaultGetWork( itP, 3 );
}
/*@
   KSPChebychevSetEigenvalues - Sets estimates for the extreme eigenvalues
   of the preconditioned problem.

   Input Parameters:
.  itP - the Krylov space context
.  emax, emin - the eigenvalue estimates

.keywords: KSP, Chebyshev, set, eigenvalues
@*/
int KSPChebychevSetEigenvalues(KSP itP,double emax,double emin)
{
  KSP_Chebychev *chebychevP = (KSP_Chebychev *) itP->MethodPrivate;
  PETSCVALIDHEADERSPECIFIC(itP,KSP_COOKIE);
  if (itP->type != KSPCHEBYCHEV) return 0;
  chebychevP->emax = emax;
  chebychevP->emin = emin;
  return 0;
}

int  KSPSolve_Chebychev(KSP itP,int *its)
{
  int              k,kp1,km1,maxit,ktmp,i = 0,pres;
  int              hist_len,cerr,ierr;
  Scalar           alpha,omegaprod;
  Scalar           mu,omega,Gamma,c[3],scale;
  double           rnorm,*history;
  Vec              x,b,p[3],r;
  KSP_Chebychev    *chebychevP = (KSP_Chebychev *) itP->MethodPrivate;
  Scalar           mone = -1.0, tmp;
  Mat              Amat, Pmat;
  MatStructure     pflag;

  ierr = PCGetOperators(itP->B,&Amat,&Pmat,&pflag); CHKERRQ(ierr);
  history = itP->residual_history;
  hist_len= itP->res_hist_size;
  maxit   = itP->max_it;
  pres    = itP->use_pres;
  cerr    = 1;

  /* These three point to the three active solutions, we
     rotate these three at each solution update */
  km1 = 0; k = 1; kp1 = 2;
  x = itP->vec_sol;
  b = itP->vec_rhs;
  p[km1] = x;
  p[k]   = itP->work[0];
  p[kp1] = itP->work[1];
  r      = itP->work[2];

  /* use scale*B as our preconditioner */
  scale = 2.0/( chebychevP->emax + chebychevP->emin );

  /*   -alpha <=  scale*lambda(B^{-1}A) <= alpha   */
  alpha = 1.0 - scale*(chebychevP->emin); ;
  Gamma = 1.0;
  mu = 1.0/alpha; 
  omegaprod = 2.0/alpha;

  c[km1] = 1.0;
  c[k] = mu;

  if (!itP->guess_zero) {
    ierr = MatMult(Amat,x,r); CHKERRQ(ierr);     /*  r = b - Ax     */
    ierr = VecAYPX(&mone,b,r); CHKERRQ(ierr);
  }
  else {ierr = VecCopy(b,r); CHKERRQ(ierr);}
                  
  ierr = PCApply(itP->B,r,p[k]); CHKERRQ(ierr);  /* p[k] = scale B^{-1}r + x */
  ierr = VecAYPX(&scale,x,p[k]); CHKERRQ(ierr);

  for ( i=0; i<maxit; i++) {
    c[kp1] = 2.0*mu*c[k] - c[km1];
    omega = omegaprod*c[k]/c[kp1];

    ierr = MatMult(Amat,p[k],r);                 /*  r = b - Ap[k]    */
    ierr = VecAYPX(&mone,b,r);                       
    ierr = PCApply(itP->B,r,p[kp1]);             /*  p[kp1] = B^{-1}z  */

    /* calculate residual norm if requested */
    if (itP->calc_res) {
      if (!pres) {ierr = VecNorm(r,&rnorm); CHKERRQ(ierr);}
      else {ierr = VecNorm(p[kp1],&rnorm); CHKERRQ(ierr);}
      if (history && hist_len > i) history[i] = rnorm;
      itP->vec_sol = p[k]; 
      MONITOR(itP,rnorm,i);
      cerr = (*itP->converged)(itP,i,rnorm,itP->cnvP);
      if (cerr) break;
    }

    /* y^{k+1} = omega( y^{k} - y^{k-1} + Gamma*r^{k}) + y^{k-1} */
    tmp = omega*Gamma*scale;
    ierr = VecScale(&tmp,p[kp1]); CHKERRQ(ierr);
    tmp = 1.0-omega; VecAXPY(&tmp,p[km1],p[kp1]);
    ierr = VecAXPY(&omega,p[k],p[kp1]); CHKERRQ(ierr);

    ktmp = km1;
    km1  = k;
    k    = kp1;
    kp1  = ktmp;
  }
  if (!cerr && itP->calc_res) {
    ierr = MatMult(Amat,p[k],r); CHKERRQ(ierr);       /*  r = b - Ap[k]    */
    ierr = VecAYPX(&mone,b,r); CHKERRQ(ierr);
    if (!pres) {ierr = VecNorm(r,&rnorm); CHKERRQ(ierr);}
    else {
      ierr = PCApply(itP->B,r,p[kp1]); CHKERRQ(ierr); /* p[kp1] = B^{-1}z */
      ierr = VecNorm(p[kp1],&rnorm); CHKERRQ(ierr);
    }
    if (history && hist_len > i) history[i] = rnorm;
    itP->vec_sol = p[k]; 
    MONITOR(itP,rnorm,i);
  }
  if (history) itP->res_act_size = (hist_len < i) ? hist_len : i;

  /* make sure solution is in vector x */
  itP->vec_sol = x;
  if (k != 0) {
    ierr = VecCopy(p[k],x); CHKERRQ(ierr);
  }
  if (cerr <= 0) *its = -(i+1);
  else           *its = i+1;
  return 0;
}

static int KSPView_Chebychev(PetscObject obj,Viewer viewer)
{
  KSP           itP = (KSP)obj;
  KSP_Chebychev *cheb = (KSP_Chebychev *) itP->MethodPrivate;
  FILE          *fd = ViewerFileGetPointer_Private(viewer);

  MPIU_fprintf(itP->comm,fd,
    "    Chebychev: eigenvalue estimates:  min = %g, max = %g\n",
    cheb->emin,cheb->emax);
  return 0;
}

int KSPCreate_Chebychev(KSP itP)
{
  KSP_Chebychev *chebychevP;

  chebychevP=(KSP_Chebychev*)PETSCMALLOC(sizeof(KSP_Chebychev));CHKPTRQ(chebychevP);
  PLogObjectMemory(itP,sizeof(KSP_Chebychev));
  itP->MethodPrivate = (void *) chebychevP;

  itP->type                 = KSPCHEBYCHEV;
  itP->right_pre            = 0;
  itP->calc_res             = 1;

  chebychevP->emin          = 1.e-2;
  chebychevP->emax          = 1.e+2;

  itP->setup                = KSPSetUp_Chebychev;
  itP->solver               = KSPSolve_Chebychev;
  itP->adjustwork           = KSPiDefaultAdjustWork;
  itP->destroy              = KSPiDefaultDestroy;
  itP->converged            = KSPDefaultConverged;
  itP->buildsolution        = KSPDefaultBuildSolution;
  itP->buildresidual        = KSPDefaultBuildResidual;
  itP->view                 = KSPView_Chebychev;
  return 0;
}
