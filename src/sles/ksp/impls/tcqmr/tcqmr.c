/*$Id: tcqmr.c,v 1.47 1999/05/12 03:31:57 bsmith Exp bsmith $*/

/*
    This file contains an implementation of Tony Chan's transpose-free QMR.

    Note: The vector dot products in the code have not been checked for the
    complex numbers version, so most probably some are incorrect.
*/

#include "src/sles/ksp/kspimpl.h"
#include "src/sles/ksp/impls/tcqmr/tcqmrp.h"

#undef __FUNC__  
#define __FUNC__ "KSPSolve_TCQMR"
static int KSPSolve_TCQMR(KSP ksp,int *its )
{
  double      rnorm0, rnorm,dp1,Gamma;
  Scalar      theta, ep, cl1, sl1, cl, sl, sprod, tau_n1, f; 
  Scalar      deltmp, rho, beta, eptmp, ta, s, c, tau_n, delta;
  Scalar      dp11,dp2, rhom1, alpha,tmp, zero = 0.0;
  int         it, cerr, ierr;

  PetscFunctionBegin;
  ksp->its = 0;

  it    = 0;
  ierr  = KSPResidual(ksp,x,u,v,r,v0,b);CHKERRQ(ierr);
  ierr  = VecNorm(r,NORM_2,&rnorm0);CHKERRQ(ierr);         /*  rnorm0 = ||r|| */

  cerr = (*ksp->converged)(ksp,0,rnorm0,ksp->cnvP);
  if (cerr) {*its =  0; PetscFunctionReturn(0);}

  ierr  = VecSet(&zero,um1);CHKERRQ(ierr);
  ierr  = VecCopy(r,u);CHKERRQ(ierr);
  rnorm = rnorm0;
  tmp   = 1.0/rnorm; ierr = VecScale(&tmp,u);CHKERRQ(ierr);
  ierr  = VecSet(&zero,vm1);CHKERRQ(ierr);
  ierr  = VecCopy(u,v);CHKERRQ(ierr);
  ierr  = VecCopy(u,v0);CHKERRQ(ierr);
  ierr  = VecSet(&zero,pvec1);CHKERRQ(ierr);
  ierr  = VecSet(&zero,pvec2);CHKERRQ(ierr);
  ierr  = VecSet(&zero,p);CHKERRQ(ierr);
  theta = 0.0; 
  ep    = 0.0; 
  cl1   = 0.0; 
  sl1   = 0.0; 
  cl    = 0.0; 
  sl    = 0.0;
  sprod = 1.0; 
  tau_n1= rnorm0;
  f     = 1.0; 
  Gamma = 1.0; 
  rhom1 = 1.0;

  /*
   CALCULATE SQUARED LANCZOS  vectors
   */
  while (!(cerr=(*ksp->converged)(ksp,it,rnorm,ksp->cnvP))) {     
    ksp->its++;

    KSPMonitor( ksp, it, rnorm);
    ierr   = PCApplyBAorAB(ksp->B,ksp->pc_side,u,y,vtmp);CHKERRQ(ierr); /* y = A*u */
    ierr   = VecDot(v0,y,&dp11);CHKERRQ(ierr);
    ierr   = VecDot(v0,u,&dp2);CHKERRQ(ierr);
    alpha  = dp11 / dp2;                          /* alpha = v0'*y/v0'*u */
    deltmp = alpha;
    ierr   = VecCopy(y,z);CHKERRQ(ierr);
    tmp    = -alpha; 
    ierr   = VecAXPY(&tmp,u,z);CHKERRQ(ierr); /* z = y - alpha u */
    ierr   = VecDot(v0,u,&rho);CHKERRQ(ierr);
    beta   = rho / (f*rhom1);
    rhom1  = rho;
    ierr   = VecCopy(z,utmp);CHKERRQ(ierr);    /* up1 = (A-alpha*I)*
					         (z-2*beta*p) + f*beta*
					         beta*um1 */
    tmp    = -2.0*beta;VecAXPY(&tmp,p,utmp);
    ierr   = PCApplyBAorAB(ksp->B,ksp->pc_side,utmp,up1,vtmp);CHKERRQ(ierr);
    tmp    = -alpha; ierr = VecAXPY(&tmp,utmp,up1);CHKERRQ(ierr);
    tmp    = f*beta*beta; ierr = VecAXPY(&tmp,um1,up1);CHKERRQ(ierr);
    ierr   = VecNorm(up1,NORM_2,&dp1);CHKERRQ(ierr);
    f      = 1.0 / dp1;
    ierr   = VecScale(&f,up1);CHKERRQ(ierr);
    tmp    = -beta; 
    ierr   = VecAYPX(&tmp,z,p);CHKERRQ(ierr);   /* p = f*(z-beta*p) */
    ierr   = VecScale(&f,p);CHKERRQ(ierr);
    ierr   = VecCopy(u,um1);CHKERRQ(ierr);
    ierr   = VecCopy(up1,u);CHKERRQ(ierr);
    beta   = beta/Gamma;
    eptmp  = beta;
    ierr   = PCApplyBAorAB(ksp->B,ksp->pc_side,v,vp1,vtmp);CHKERRQ(ierr);
    tmp    = -alpha; ierr = VecAXPY(&tmp,v,vp1);CHKERRQ(ierr);
    tmp    = -beta; ierr = VecAXPY(&tmp,vm1,vp1);CHKERRQ(ierr);
    ierr   = VecNorm(vp1,NORM_2,&Gamma);CHKERRQ(ierr);
    tmp    = 1.0/Gamma; ierr = VecScale(&tmp,vp1);CHKERRQ(ierr);
    ierr   = VecCopy(v,vm1);CHKERRQ(ierr);
    ierr   = VecCopy(vp1,v);CHKERRQ(ierr);

  /*
     SOLVE  Ax = b
   */
  /* Apply last two Given's (Gl-1 and Gl) rotations to (beta,alpha,Gamma) */
    if (it > 1) {
      theta =  sl1*beta;
      eptmp = -cl1*beta;
    }
    if (it > 0) {
      ep     = -cl*eptmp + sl*alpha;
      deltmp = -sl*eptmp - cl*alpha;
    }
    if (fabs(Gamma) > PetscAbsScalar(deltmp)) {
      ta = -deltmp / Gamma;
      s  = 1.0 / PetscSqrtScalar(1.0 + ta*ta);
      c  = s*ta;
    } else {
      ta = -Gamma/deltmp;
      c  = 1.0 / PetscSqrtScalar(1.0 + ta*ta);
      s  = c*ta;
    }

    delta  = -c*deltmp + s*Gamma;
    tau_n  = -c*tau_n1; tau_n1 = -s*tau_n1;
    ierr   = VecCopy(vm1,pvec);CHKERRQ(ierr);
    tmp    = -theta; ierr = VecAXPY(&tmp,pvec2,pvec);CHKERRQ(ierr);
    tmp    = -ep; ierr = VecAXPY(&tmp,pvec1,pvec);CHKERRQ(ierr);
    tmp    = 1.0/delta; ierr = VecScale(&tmp,pvec);CHKERRQ(ierr);
    ierr   = VecAXPY(&tau_n,pvec,x);CHKERRQ(ierr);
    cl1    = cl; sl1 = sl; cl = c; sl = s;     

    VecCopy(pvec1,pvec2);
    VecCopy(pvec,pvec1);

    /* Compute the upper bound on the residual norm r (See QMR paper p. 13) */
    sprod = sprod*PetscAbsScalar(s);
#if defined(PETSC_USE_COMPLEX)
    rnorm = rnorm0 * sqrt((double)it+2.0) * PetscReal(sprod);     
#else
    rnorm = rnorm0 * sqrt((double)it+2.0) * sprod;     
#endif
    it++; if (it > ksp->max_it) {break;}
  }

  /* Need to undo preconditioning here  */
  ierr = KSPUnwindPreconditioner(ksp,x,vtmp);CHKERRQ(ierr);

  if (cerr <= 0) *its = -it;
  else           *its = it;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPSetUp_TCQMR"
static int KSPSetUp_TCQMR(KSP ksp)
{
  int ierr;

  PetscFunctionBegin;
  if (ksp->pc_side == PC_SYMMETRIC){
    SETERRQ(2,0,"no symmetric preconditioning for KSPTCQMR");
  }
  ierr = KSPDefaultGetWork(ksp,TCQMR_VECS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "KSPCreate_TCQMR"
int KSPCreate_TCQMR(KSP ksp)
{
  PetscFunctionBegin;
  ksp->data               = (void *) 0;
  ksp->pc_side            = PC_LEFT;
  ksp->converged          = KSPDefaultConverged;
  ksp->ops->buildsolution = KSPDefaultBuildSolution;
  ksp->ops->buildresidual = KSPDefaultBuildResidual;
  ksp->ops->setup         = KSPSetUp_TCQMR;
  ksp->ops->solve         = KSPSolve_TCQMR;
  ksp->ops->destroy       = KSPDefaultDestroy;
  ksp->ops->view          = 0;
  ksp->guess_zero         = 1; 
  PetscFunctionReturn(0);
}
EXTERN_C_END
