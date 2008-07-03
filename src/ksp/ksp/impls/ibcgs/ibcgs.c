#define PETSCKSP_DLL

#include "include/private/kspimpl.h"

#undef __FUNCT__  
#define __FUNCT__ "KSPSetUp_IBCGS"
static PetscErrorCode KSPSetUp_IBCGS(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ksp->pc_side == PC_SYMMETRIC) {
    SETERRQ(PETSC_ERR_SUP,"no symmetric preconditioning for KSPIBCGS");
  }
  ierr = KSPDefaultGetWork(ksp,6);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* 
    The algorithm as copied from the paper (see manual page at bottom of this file)

     r0 = b - A*x0
     u0 = A*r0
     f0 = A'*r0
     q0 = v0 = z0 = 0

     sigma_1 = pi0 = phi0 = tau0

     sigma0 = r0'u0 

     rho0 = alpha0 = omega0 = 1

     do:
        rho1   = ohi0 - w0*sigma_1 + w0*alpha0*pi0
        delta1 = rho1/rho0
        beta1  = delta1/omega0
        tau1   = rho0 + beta1*tau0  - delta1*pi0
        alpha1 = rho1/tau1

        v1 = u0 + beta1*v0 - delta1*q0
        q1 = Av1
        s1 = r0 - alpha1*v1
        t1 = u0 - alpha1*q
        z1 = alpha1*r0 + beta1*z0 - alpha1*delta1*v0

        phi1 = r0's1
        pi1  = r0'q1
        gamma1 = f0's1
        eta1   = f0't1
        theta1 = s1't1
        kappa1 = t1't1
  
        alpha1 = gamma1 - omega1*eta1

        r1 = s1 - omega1*t1
        x1 = x0 + z1 + omega1*s1

        Test for convergence

        u1 = Ar1
     enddo:
   
#undef __FUNCT__  
#define __FUNCT__ "KSPSolve_IBCGS"
static PetscErrorCode  KSPSolve_IBCGS(KSP ksp)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscScalar    rho,rhoold,alpha,beta,omega,omegaold,d1,d2;
  Vec            X,B,V,P,R,RP,T,S;
  PetscReal      dp = 0.0;

  PetscFunctionBegin;

  X       = ksp->vec_sol;
  B       = ksp->vec_rhs;
  R       = ksp->work[0];
  RP      = ksp->work[1];
  V       = ksp->work[2];
  T       = ksp->work[3];
  S       = ksp->work[4];
  P       = ksp->work[5];

  /* Compute initial preconditioned residual */
 ierr = KSPInitialResidual(ksp,X,V,T,R,B);CHKERRQ(ierr);

  /* Test for nothing to do */
  if (ksp->normtype != KSP_NORM_NO) {
    ierr = VecNorm(R,NORM_2,&dp);CHKERRQ(ierr);
  }
  ierr = PetscObjectTakeAccess(ksp);CHKERRQ(ierr);
  ksp->its   = 0;
  ksp->rnorm = dp;
  ierr = PetscObjectGrantAccess(ksp);CHKERRQ(ierr);
  KSPLogResidualHistory(ksp,dp);
  KSPMonitor(ksp,0,dp);
  ierr = (*ksp->converged)(ksp,0,dp,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
  if (ksp->reason) PetscFunctionReturn(0);

  /* Make the initial Rp == R */
  ierr = VecCopy(R,RP);CHKERRQ(ierr);

  rhoold   = 1.0;
  alpha    = 1.0;
  omegaold = 1.0;
  ierr = VecSet(P,0.0);CHKERRQ(ierr);
  ierr = VecSet(V,0.0);CHKERRQ(ierr);

  i=0;
  do {
    ierr = VecDot(R,RP,&rho);CHKERRQ(ierr);       /*   rho <- (r,rp)      */
    beta = (rho/rhoold) * (alpha/omegaold);
    ierr = VecAXPBYPCZ(P,1.0,-omegaold*beta,beta,R,V);CHKERRQ(ierr);  /* p <- r - omega * beta* v + beta * p */
    ierr = KSP_PCApplyBAorAB(ksp,P,V,T);CHKERRQ(ierr);  /*   v <- K p           */
    ierr = VecDot(V,RP,&d1);CHKERRQ(ierr);
    if (d1 == 0.0) SETERRQ(PETSC_ERR_PLIB,"Divide by zero");
    alpha = rho / d1;                 /*   a <- rho / (v,rp)  */
    ierr = VecWAXPY(S,-alpha,V,R);CHKERRQ(ierr);      /*   s <- r - a v       */
    ierr = KSP_PCApplyBAorAB(ksp,S,T,R);CHKERRQ(ierr);/*   t <- K s    */
    ierr = VecDotNorm2(S,T,&d1,&d2);CHKERRQ(ierr);
    if (d2 == 0.0) {
      /* t is 0.  if s is 0, then alpha v == r, and hence alpha p
	 may be our solution.  Give it a try? */
      ierr = VecDot(S,S,&d1);CHKERRQ(ierr);
      if (d1 != 0.0) {
        ksp->reason = KSP_DIVERGED_BREAKDOWN;
        break;
      }
      ierr = VecAXPY(X,alpha,P);CHKERRQ(ierr);   /*   x <- x + a p       */
      ierr = PetscObjectTakeAccess(ksp);CHKERRQ(ierr);
      ksp->its++;
      ksp->rnorm  = 0.0;
      ksp->reason = KSP_CONVERGED_RTOL;
      ierr = PetscObjectGrantAccess(ksp);CHKERRQ(ierr);
      KSPLogResidualHistory(ksp,dp);
      KSPMonitor(ksp,i+1,0.0);
      break;
    }
    omega = d1 / d2;                               /*   w <- (t's) / (t't) */
    ierr = VecAXPBYPCZ(X,alpha,omega,1.0,P,S);CHKERRQ(ierr); /* x <- alpha * p + omega * s + x */
    ierr  = VecWAXPY(R,-omega,T,S);CHKERRQ(ierr);     /*   r <- s - w t       */
    if (ksp->normtype != KSP_NORM_NO && ksp->chknorm < i+2) {
      ierr = VecNorm(R,NORM_2,&dp);CHKERRQ(ierr);
    }

    rhoold   = rho;
    omegaold = omega;

    ierr = PetscObjectTakeAccess(ksp);CHKERRQ(ierr);
    ksp->its++;
    ksp->rnorm = dp;
    ierr = PetscObjectGrantAccess(ksp);CHKERRQ(ierr);
    KSPLogResidualHistory(ksp,dp);
    KSPMonitor(ksp,i+1,dp);
    ierr = (*ksp->converged)(ksp,i+1,dp,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
    if (ksp->reason) break;    
    if (rho == 0.0) {
      ksp->reason = KSP_DIVERGED_BREAKDOWN;
      break;
    }
    i++;
  } while (i<ksp->max_it);

  if (i >= ksp->max_it) {
    ksp->reason = KSP_DIVERGED_ITS;
  }

  ierr = KSPUnwindPreconditioner(ksp,X,T);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
     KSPIBCGS - Implements the IBiCGStab (Improved Stabilized version of BiConjugate Gradient Squared) method
            in an alternative form to have only a single global reduction operation instead of the usual 3 (or 4)

   Options Database Keys:
.   see KSPSolve()

   Level: beginner

   Notes: Reference: The Improved BiCGStab Method for Large and Sparse Unsymmetric Linear Systems on Parallel Distributed Memory
                     Architectures. L. T. Yand and R. Brent, Proceedings of the Fifth International Conference on Algorithms and 
                     Architectures for Parallel Processing, 2002, IEEE.
          See KSPBCGSL for additional stabilization

          Unlike the Bi-CG-stab algorithm, this requires one multiplication be the transpose of the operator
           before the iteration starts.

.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP, KSPBICG, KSPBCGSL, KSPIBCGS
M*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "KSPCreate_IBCGS"
PetscErrorCode PETSCKSP_DLLEXPORT KSPCreate_IBCGS(KSP ksp)
{
  PetscFunctionBegin;
  ksp->data                 = (void*)0;
  ksp->pc_side              = PC_LEFT;
  ksp->ops->setup           = KSPSetUp_IBCGS;
  ksp->ops->solve           = KSPSolve_IBCGS;
  ksp->ops->destroy         = KSPDefaultDestroy;
  ksp->ops->buildsolution   = KSPDefaultBuildSolution;
  ksp->ops->buildresidual   = KSPDefaultBuildResidual;
  ksp->ops->setfromoptions  = 0;
  ksp->ops->view            = 0;
  PetscFunctionReturn(0);
}
EXTERN_C_END
