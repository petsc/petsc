
/*
    This file contains an implementation of Tony Chan's transpose-free QMR.

    Note: The vector dot products in the code have not been checked for the
    complex numbers version, so most probably some are incorrect.
*/

#include <../src/ksp/ksp/impls/tcqmr/tcqmrimpl.h>

static PetscErrorCode KSPSolve_TCQMR(KSP ksp)
{
  PetscReal      rnorm0,rnorm,dp1,Gamma;
  PetscScalar    theta,ep,cl1,sl1,cl,sl,sprod,tau_n1,f;
  PetscScalar    deltmp,rho,beta,eptmp,ta,s,c,tau_n,delta;
  PetscScalar    dp11,dp2,rhom1,alpha,tmp;

  PetscFunctionBegin;
  ksp->its = 0;

  CHKERRQ(KSPInitialResidual(ksp,x,u,v,r,b));
  CHKERRQ(VecNorm(r,NORM_2,&rnorm0));          /*  rnorm0 = ||r|| */
  KSPCheckNorm(ksp,rnorm0);
  if (ksp->normtype != KSP_NORM_NONE) ksp->rnorm = rnorm0;
  else ksp->rnorm = 0;
  CHKERRQ((*ksp->converged)(ksp,0,ksp->rnorm,&ksp->reason,ksp->cnvP));
  if (ksp->reason) PetscFunctionReturn(0);

  CHKERRQ(VecSet(um1,0.0));
  CHKERRQ(VecCopy(r,u));
  rnorm = rnorm0;
  tmp   = 1.0/rnorm; CHKERRQ(VecScale(u,tmp));
  CHKERRQ(VecSet(vm1,0.0));
  CHKERRQ(VecCopy(u,v));
  CHKERRQ(VecCopy(u,v0));
  CHKERRQ(VecSet(pvec1,0.0));
  CHKERRQ(VecSet(pvec2,0.0));
  CHKERRQ(VecSet(p,0.0));
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
  if (ksp->normtype != KSP_NORM_NONE) ksp->rnorm = rnorm;
  else ksp->rnorm = 0;
  CHKERRQ((*ksp->converged)(ksp,ksp->its,ksp->rnorm,&ksp->reason,ksp->cnvP));
  while (!ksp->reason) {
    CHKERRQ(KSPMonitor(ksp,ksp->its,ksp->rnorm));
    ksp->its++;

    CHKERRQ(KSP_PCApplyBAorAB(ksp,u,y,vtmp)); /* y = A*u */
    CHKERRQ(VecDot(y,v0,&dp11));
    KSPCheckDot(ksp,dp11);
    CHKERRQ(VecDot(u,v0,&dp2));
    alpha  = dp11 / dp2;                          /* alpha = v0'*y/v0'*u */
    deltmp = alpha;
    CHKERRQ(VecCopy(y,z));
    CHKERRQ(VecAXPY(z,-alpha,u)); /* z = y - alpha u */
    CHKERRQ(VecDot(u,v0,&rho));
    beta   = rho / (f*rhom1);
    rhom1  = rho;
    CHKERRQ(VecCopy(z,utmp));    /* up1 = (A-alpha*I)*
                                                (z-2*beta*p) + f*beta*
                                                beta*um1 */
    CHKERRQ(VecAXPY(utmp,-2.0*beta,p));
    CHKERRQ(KSP_PCApplyBAorAB(ksp,utmp,up1,vtmp));
    CHKERRQ(VecAXPY(up1,-alpha,utmp));
    CHKERRQ(VecAXPY(up1,f*beta*beta,um1));
    CHKERRQ(VecNorm(up1,NORM_2,&dp1));
    KSPCheckNorm(ksp,dp1);
    f      = 1.0 / dp1;
    CHKERRQ(VecScale(up1,f));
    CHKERRQ(VecAYPX(p,-beta,z));   /* p = f*(z-beta*p) */
    CHKERRQ(VecScale(p,f));
    CHKERRQ(VecCopy(u,um1));
    CHKERRQ(VecCopy(up1,u));
    beta   = beta/Gamma;
    eptmp  = beta;
    CHKERRQ(KSP_PCApplyBAorAB(ksp,v,vp1,vtmp));
    CHKERRQ(VecAXPY(vp1,-alpha,v));
    CHKERRQ(VecAXPY(vp1,-beta,vm1));
    CHKERRQ(VecNorm(vp1,NORM_2,&Gamma));
    KSPCheckNorm(ksp,Gamma);
    CHKERRQ(VecScale(vp1,1.0/Gamma));
    CHKERRQ(VecCopy(v,vm1));
    CHKERRQ(VecCopy(vp1,v));

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

    delta = -c*deltmp + s*Gamma;
    tau_n = -c*tau_n1; tau_n1 = -s*tau_n1;
    CHKERRQ(VecCopy(vm1,pvec));
    CHKERRQ(VecAXPY(pvec,-theta,pvec2));
    CHKERRQ(VecAXPY(pvec,-ep,pvec1));
    CHKERRQ(VecScale(pvec,1.0/delta));
    CHKERRQ(VecAXPY(x,tau_n,pvec));
    cl1   = cl; sl1 = sl; cl = c; sl = s;

    CHKERRQ(VecCopy(pvec1,pvec2));
    CHKERRQ(VecCopy(pvec,pvec1));

    /* Compute the upper bound on the residual norm r (See QMR paper p. 13) */
    sprod = sprod*PetscAbsScalar(s);
    rnorm = rnorm0 * PetscSqrtReal((PetscReal)ksp->its+2.0) * PetscRealPart(sprod);
    if (ksp->normtype != KSP_NORM_NONE) ksp->rnorm = rnorm;
    else ksp->rnorm = 0;
    CHKERRQ((*ksp->converged)(ksp,ksp->its,ksp->rnorm,&ksp->reason,ksp->cnvP));
    if (ksp->its >= ksp->max_it) {
      if (!ksp->reason) ksp->reason = KSP_DIVERGED_ITS;
      break;
    }
  }
  CHKERRQ(KSPMonitor(ksp,ksp->its,ksp->rnorm));
  CHKERRQ(KSPUnwindPreconditioner(ksp,x,vtmp));
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPSetUp_TCQMR(KSP ksp)
{
  PetscFunctionBegin;
  PetscCheckFalse(ksp->pc_side == PC_SYMMETRIC,PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"no symmetric preconditioning for KSPTCQMR");
  CHKERRQ(KSPSetWorkVecs(ksp,TCQMR_VECS));
  PetscFunctionReturn(0);
}

/*MC
     KSPTCQMR - A variant of QMR (quasi minimal residual) developed by Tony Chan

   Options Database Keys:
    see KSPSolve()

   Level: beginner

  Notes:
    Supports either left or right preconditioning, but not symmetric

          The "residual norm" computed in this algorithm is actually just an upper bound on the actual residual norm.
          That is for left preconditioning it is a bound on the preconditioned residual and for right preconditioning
          it is a bound on the true residual.

  References:
. * - Tony F. Chan, Lisette de Pillis, and Henk van der Vorst, Transpose free formulations of Lanczos type methods for nonsymmetric linear systems,
  Numerical Algorithms, Volume 17, 1998.

.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP, KSPTFQMR

M*/

PETSC_EXTERN PetscErrorCode KSPCreate_TCQMR(KSP ksp)
{
  PetscFunctionBegin;
  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,3));
  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_RIGHT,2));
  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_RIGHT,1));

  ksp->data                = (void*)0;
  ksp->ops->buildsolution  = KSPBuildSolutionDefault;
  ksp->ops->buildresidual  = KSPBuildResidualDefault;
  ksp->ops->setup          = KSPSetUp_TCQMR;
  ksp->ops->solve          = KSPSolve_TCQMR;
  ksp->ops->destroy        = KSPDestroyDefault;
  ksp->ops->setfromoptions = NULL;
  ksp->ops->view           = NULL;
  PetscFunctionReturn(0);
}
