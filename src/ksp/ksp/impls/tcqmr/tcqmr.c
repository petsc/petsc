#define PETSCKSP_DLL

/*
    This file contains an implementation of Tony Chan's transpose-free QMR.

    Note: The vector dot products in the code have not been checked for the
    complex numbers version, so most probably some are incorrect.
*/

#include "private/kspimpl.h"
#include "../src/ksp/ksp/impls/tcqmr/tcqmrimpl.h"

#undef __FUNCT__  
#define __FUNCT__ "KSPSolve_TCQMR"
static PetscErrorCode KSPSolve_TCQMR(KSP ksp)
{
  PetscReal      rnorm0,rnorm,dp1,Gamma;
  PetscScalar    theta,ep,cl1,sl1,cl,sl,sprod,tau_n1,f; 
  PetscScalar    deltmp,rho,beta,eptmp,ta,s,c,tau_n,delta;
  PetscScalar    dp11,dp2,rhom1,alpha,tmp;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ksp->its = 0;

  ierr  = KSPInitialResidual(ksp,x,u,v,r,b);CHKERRQ(ierr);
  ierr  = VecNorm(r,NORM_2,&rnorm0);CHKERRQ(ierr);         /*  rnorm0 = ||r|| */

  ierr = (*ksp->converged)(ksp,0,rnorm0,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
  if (ksp->reason) PetscFunctionReturn(0);

  ierr  = VecSet(um1,0.0);CHKERRQ(ierr);
  ierr  = VecCopy(r,u);CHKERRQ(ierr);
  rnorm = rnorm0;
  tmp   = 1.0/rnorm; ierr = VecScale(u,tmp);CHKERRQ(ierr);
  ierr  = VecSet(vm1,0.0);CHKERRQ(ierr);
  ierr  = VecCopy(u,v);CHKERRQ(ierr);
  ierr  = VecCopy(u,v0);CHKERRQ(ierr);
  ierr  = VecSet(pvec1,0.0);CHKERRQ(ierr);
  ierr  = VecSet(pvec2,0.0);CHKERRQ(ierr);
  ierr  = VecSet(p,0.0);CHKERRQ(ierr);
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
  ierr = (*ksp->converged)(ksp,ksp->its,rnorm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
  while (!ksp->reason){
    KSPMonitor(ksp,ksp->its,rnorm);
    ksp->its++;

    ierr   = KSP_PCApplyBAorAB(ksp,u,y,vtmp);CHKERRQ(ierr); /* y = A*u */
    ierr   = VecDot(v0,y,&dp11);CHKERRQ(ierr);
    ierr   = VecDot(v0,u,&dp2);CHKERRQ(ierr);
    alpha  = dp11 / dp2;                          /* alpha = v0'*y/v0'*u */
    deltmp = alpha;
    ierr   = VecCopy(y,z);CHKERRQ(ierr);
    ierr   = VecAXPY(z,-alpha,u);CHKERRQ(ierr); /* z = y - alpha u */
    ierr   = VecDot(v0,u,&rho);CHKERRQ(ierr);
    beta   = rho / (f*rhom1);
    rhom1  = rho;
    ierr   = VecCopy(z,utmp);CHKERRQ(ierr);    /* up1 = (A-alpha*I)*
					         (z-2*beta*p) + f*beta*
					         beta*um1 */
    ierr = VecAXPY(utmp,-2.0*beta,p);CHKERRQ(ierr);
    ierr   = KSP_PCApplyBAorAB(ksp,utmp,up1,vtmp);CHKERRQ(ierr);
    ierr = VecAXPY(up1,-alpha,utmp);CHKERRQ(ierr);
    ierr = VecAXPY(up1,f*beta*beta,um1);CHKERRQ(ierr);
    ierr   = VecNorm(up1,NORM_2,&dp1);CHKERRQ(ierr);
    f      = 1.0 / dp1;
    ierr   = VecScale(up1,f);CHKERRQ(ierr);
    ierr   = VecAYPX(p,-beta,z);CHKERRQ(ierr);   /* p = f*(z-beta*p) */
    ierr   = VecScale(p,f);CHKERRQ(ierr);
    ierr   = VecCopy(u,um1);CHKERRQ(ierr);
    ierr   = VecCopy(up1,u);CHKERRQ(ierr);
    beta   = beta/Gamma;
    eptmp  = beta;
    ierr   = KSP_PCApplyBAorAB(ksp,v,vp1,vtmp);CHKERRQ(ierr);
    ierr = VecAXPY(vp1,-alpha,v);CHKERRQ(ierr);
    ierr = VecAXPY(vp1,-beta,vm1);CHKERRQ(ierr);
    ierr   = VecNorm(vp1,NORM_2,&Gamma);CHKERRQ(ierr);
    ierr = VecScale(vp1,1.0/Gamma);CHKERRQ(ierr);
    ierr   = VecCopy(v,vm1);CHKERRQ(ierr);
    ierr   = VecCopy(vp1,v);CHKERRQ(ierr);

  /*
     SOLVE  Ax = b
   */
  /* Apply last two Given's (Gl-1 and Gl) rotations to (beta,alpha,Gamma) */
    if (ksp->its > 2) {
      theta =  sl1*beta;
      eptmp = -cl1*beta;
    }
    if (ksp->its > 1) {
      ep     = -cl*eptmp + sl*alpha;
      deltmp = -sl*eptmp - cl*alpha;
    }
    if (PetscAbsReal(Gamma) > PetscAbsScalar(deltmp)) {
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
    ierr = VecAXPY(pvec,-theta,pvec2);CHKERRQ(ierr);
    ierr = VecAXPY(pvec,-ep,pvec1);CHKERRQ(ierr);
    ierr = VecScale(pvec,1.0/delta);CHKERRQ(ierr);
    ierr   = VecAXPY(x,tau_n,pvec);CHKERRQ(ierr);
    cl1    = cl; sl1 = sl; cl = c; sl = s;     

    ierr = VecCopy(pvec1,pvec2);CHKERRQ(ierr);
    ierr = VecCopy(pvec,pvec1);CHKERRQ(ierr);

    /* Compute the upper bound on the residual norm r (See QMR paper p. 13) */
    sprod = sprod*PetscAbsScalar(s);
#if defined(PETSC_USE_COMPLEX)
    rnorm = rnorm0 * sqrt((double)ksp->its+2.0) * PetscRealPart(sprod);     
#else
    rnorm = rnorm0 * sqrt((double)ksp->its+2.0) * sprod;     
#endif
    ierr = (*ksp->converged)(ksp,ksp->its,rnorm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
    if (ksp->its >= ksp->max_it) {
      if (!ksp->reason) ksp->reason = KSP_DIVERGED_ITS;
      break;
    }
  }
  KSPMonitor(ksp,ksp->its,rnorm);
  ierr = KSPUnwindPreconditioner(ksp,x,vtmp);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSetUp_TCQMR"
static PetscErrorCode KSPSetUp_TCQMR(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ksp->pc_side == PC_SYMMETRIC){
    SETERRQ(PETSC_ERR_SUP,"no symmetric preconditioning for KSPTCQMR");
  }
  ierr = KSPDefaultGetWork(ksp,TCQMR_VECS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
     KSPRTCQMR - A variant of QMR (quasi minimal residual) developed by Tony Chan

   Options Database Keys:
.   see KSPSolve()

   Level: beginner

  Notes: Supports either left or right preconditioning, but not symmetric

  References:
  Transpose-free formulations of Lanczos-type methods for nonsymmetric linear systems, 
  Tony F. Chan, Lisette de Pillis, and Henk van der Vorst, Numerical Algorithms, 	
  Volume 17, Numbers 1-2 / May, 1998 pp. 51-66.

.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP, KSPTFQMR

M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "KSPCreate_TCQMR"
PetscErrorCode PETSCKSP_DLLEXPORT KSPCreate_TCQMR(KSP ksp)
{
  PetscFunctionBegin;
  ksp->data                = (void*)0;
  ksp->ops->buildsolution  = KSPDefaultBuildSolution;
  ksp->ops->buildresidual  = KSPDefaultBuildResidual;
  ksp->ops->setup          = KSPSetUp_TCQMR;
  ksp->ops->solve          = KSPSolve_TCQMR;
  ksp->ops->destroy        = KSPDefaultDestroy;
  ksp->ops->setfromoptions = 0;
  ksp->ops->view           = 0;
  PetscFunctionReturn(0);
}
EXTERN_C_END
