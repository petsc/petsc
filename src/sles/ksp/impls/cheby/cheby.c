#ifndef lint
static char vcid[] = "$Id: cheby.c,v 1.36 1996/03/19 21:23:58 bsmith Exp bsmith $";
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
#include "pinclude/pviewer.h"

int KSPSetUp_Chebychev(KSP ksp)
{
  int ierr;
  if (ksp->pc_side == PC_SYMMETRIC)
    {SETERRQ(2,"KSPSetUp_Chebychev:no symmetric preconditioning for KSPCHEBYCHEV");}
  ierr = KSPCheckDef(ksp); CHKERRQ(ierr);
  return KSPiDefaultGetWork( ksp, 3 );
}
/*@
   KSPChebychevSetEigenvalues - Sets estimates for the extreme eigenvalues
   of the preconditioned problem.

   Input Parameters:
.  ksp - the Krylov space context
.  emax, emin - the eigenvalue estimates

.keywords: KSP, Chebyshev, set, eigenvalues
@*/
int KSPChebychevSetEigenvalues(KSP ksp,double emax,double emin)
{
  KSP_Chebychev *chebychevP = (KSP_Chebychev *) ksp->data;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  if (ksp->type != KSPCHEBYCHEV) return 0;
  chebychevP->emax = emax;
  chebychevP->emin = emin;
  return 0;
}

int  KSPSolve_Chebychev(KSP ksp,int *its)
{
  int              k,kp1,km1,maxit,ktmp,i = 0,pres,hist_len,cerr,ierr;
  Scalar           alpha,omegaprod,mu,omega,Gamma,c[3],scale,mone = -1.0, tmp;
  double           rnorm,*history;
  Vec              x,b,p[3],r;
  KSP_Chebychev    *chebychevP = (KSP_Chebychev *) ksp->data;
  Mat              Amat, Pmat;
  MatStructure     pflag;

  ierr    = PCGetOperators(ksp->B,&Amat,&Pmat,&pflag); CHKERRQ(ierr);
  history = ksp->residual_history;
  hist_len= ksp->res_hist_size;
  maxit   = ksp->max_it;
  pres    = ksp->use_pres;
  cerr    = 1;

  /* These three point to the three active solutions, we
     rotate these three at each solution update */
  km1    = 0; k = 1; kp1 = 2;
  x      = ksp->vec_sol;
  b      = ksp->vec_rhs;
  p[km1] = x;
  p[k]   = ksp->work[0];
  p[kp1] = ksp->work[1];
  r      = ksp->work[2];

  /* use scale*B as our preconditioner */
  scale  = 2.0/( chebychevP->emax + chebychevP->emin );

  /*   -alpha <=  scale*lambda(B^{-1}A) <= alpha   */
  alpha  = 1.0 - scale*(chebychevP->emin); ;
  Gamma  = 1.0;
  mu     = 1.0/alpha; 
  omegaprod = 2.0/alpha;

  c[km1] = 1.0;
  c[k]   = mu;

  if (!ksp->guess_zero) {
    ierr = MatMult(Amat,x,r); CHKERRQ(ierr);     /*  r = b - Ax     */
    ierr = VecAYPX(&mone,b,r); CHKERRQ(ierr);
  }
  else {ierr = VecCopy(b,r); CHKERRQ(ierr);}
                  
  ierr = PCApply(ksp->B,r,p[k]); CHKERRQ(ierr);  /* p[k] = scale B^{-1}r + x */
  ierr = VecAYPX(&scale,x,p[k]); CHKERRQ(ierr);

  for ( i=0; i<maxit; i++) {
    c[kp1] = 2.0*mu*c[k] - c[km1];
    omega = omegaprod*c[k]/c[kp1];

    ierr = MatMult(Amat,p[k],r);                 /*  r = b - Ap[k]    */
    ierr = VecAYPX(&mone,b,r);                       
    ierr = PCApply(ksp->B,r,p[kp1]);             /*  p[kp1] = B^{-1}z  */

    /* calculate residual norm if requested */
    if (ksp->calc_res) {
      if (!pres) {ierr = VecNorm(r,NORM_2,&rnorm); CHKERRQ(ierr);}
      else {ierr = VecNorm(p[kp1],NORM_2,&rnorm); CHKERRQ(ierr);}
      if (history && hist_len > i) history[i] = rnorm;
      ksp->vec_sol = p[k]; 
      KSPMonitor(ksp,rnorm,i);
      cerr = (*ksp->converged)(ksp,i,rnorm,ksp->cnvP);
      if (cerr) break;
    }

    /* y^{k+1} = omega( y^{k} - y^{k-1} + Gamma*r^{k}) + y^{k-1} */
    tmp  = omega*Gamma*scale;
    ierr = VecScale(&tmp,p[kp1]); CHKERRQ(ierr);
    tmp  = 1.0-omega; VecAXPY(&tmp,p[km1],p[kp1]);
    ierr = VecAXPY(&omega,p[k],p[kp1]); CHKERRQ(ierr);

    ktmp = km1;
    km1  = k;
    k    = kp1;
    kp1  = ktmp;
  }
  if (!cerr && ksp->calc_res) {
    ierr = MatMult(Amat,p[k],r); CHKERRQ(ierr);       /*  r = b - Ap[k]    */
    ierr = VecAYPX(&mone,b,r); CHKERRQ(ierr);
    if (!pres) {ierr = VecNorm(r,NORM_2,&rnorm); CHKERRQ(ierr);}
    else {
      ierr = PCApply(ksp->B,r,p[kp1]); CHKERRQ(ierr); /* p[kp1] = B^{-1}z */
      ierr = VecNorm(p[kp1],NORM_2,&rnorm); CHKERRQ(ierr);
    }
    if (history && hist_len > i) history[i] = rnorm;
    ksp->vec_sol = p[k]; 
    KSPMonitor(ksp,rnorm,i);
  }
  if (history) ksp->res_act_size = (hist_len < i) ? hist_len : i;

  /* make sure solution is in vector x */
  ksp->vec_sol = x;
  if (k != 0) {
    ierr = VecCopy(p[k],x); CHKERRQ(ierr);
  }
  if (cerr <= 0) *its = -(i+1);
  else           *its = i+1;
  return 0;
}

static int KSPView_Chebychev(PetscObject obj,Viewer viewer)
{
  KSP           ksp = (KSP)obj;
  KSP_Chebychev *cheb = (KSP_Chebychev *) ksp->data;
  FILE          *fd;
  int           ierr;
  ViewerType    vtype;
  MPI_Comm      comm = ksp->comm;

  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (vtype == ASCII_FILE_VIEWER || vtype == ASCII_FILES_VIEWER) {
    ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);

    PetscFPrintf(comm,fd,"    Chebychev: eigenvalue estimates:  min = %g, max = %g\n",
               cheb->emin,cheb->emax);
  }
  return 0;
}

int KSPCreate_Chebychev(KSP ksp)
{
  KSP_Chebychev *chebychevP = PetscNew(KSP_Chebychev);CHKPTRQ(chebychevP);
  PLogObjectMemory(ksp,sizeof(KSP_Chebychev));

  ksp->data                 = (void *) chebychevP;
  ksp->type                 = KSPCHEBYCHEV;
  ksp->pc_side              = PC_LEFT;
  ksp->calc_res             = 1;

  chebychevP->emin          = 1.e-2;
  chebychevP->emax          = 1.e+2;

  ksp->setup                = KSPSetUp_Chebychev;
  ksp->solver               = KSPSolve_Chebychev;
  ksp->adjustwork           = KSPiDefaultAdjustWork;
  ksp->destroy              = KSPiDefaultDestroy;
  ksp->converged            = KSPDefaultConverged;
  ksp->buildsolution        = KSPDefaultBuildSolution;
  ksp->buildresidual        = KSPDefaultBuildResidual;
  ksp->view                 = KSPView_Chebychev;
  return 0;
}
