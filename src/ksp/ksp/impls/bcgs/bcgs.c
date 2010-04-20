#define PETSCKSP_DLL

#include "private/kspimpl.h"

#undef __FUNCT__  
#define __FUNCT__ "KSPSetUp_BCGS"
static PetscErrorCode KSPSetUp_BCGS(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ksp->pc_side == PC_SYMMETRIC) {
    SETERRQ(PETSC_ERR_SUP,"no symmetric preconditioning for KSPBCGS");
  }
  ierr = KSPDefaultGetWork(ksp,6);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSolve_BCGS"
static PetscErrorCode  KSPSolve_BCGS(KSP ksp)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscScalar    rho,rhoold,alpha,beta,omega,omegaold,d1,d2;
  Vec            X,B,V,P,R,RP,T,S;
  PetscReal      dp = 0.0;

  PetscFunctionBegin;
  if (ksp->normtype == KSP_NORM_NATURAL) SETERRQ(PETSC_ERR_SUP,"Cannot use natural residual norm with KSPBCGS");
  if (ksp->normtype == KSP_NORM_PRECONDITIONED && ksp->pc_side != PC_LEFT) SETERRQ(PETSC_ERR_SUP,"Use -ksp_norm_type unpreconditioned for right preconditioning and KSPBCGS");
  if (ksp->normtype == KSP_NORM_UNPRECONDITIONED && ksp->pc_side != PC_RIGHT) SETERRQ(PETSC_ERR_SUP,"Use -ksp_norm_type preconditioned for left preconditioning and KSPBCGS");

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
     KSPBCGS - Implements the BiCGStab (Stabilized version of BiConjugate Gradient Squared) method.

   Options Database Keys:
.   see KSPSolve()

   Level: beginner

   Notes: See KSPBCGSL for additional stabilization
          Supports left and right preconditioning but not symmetric

   References: van der Vorst, SIAM J. Sci. Stat. Comput., 1992.


.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP, KSPBICG, KSPBCGSL, KSPSetPreconditionerSide()
M*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "KSPCreate_BCGS"
PetscErrorCode PETSCKSP_DLLEXPORT KSPCreate_BCGS(KSP ksp)
{
  PetscFunctionBegin;
  ksp->data                 = (void*)0;
  ksp->ops->setup           = KSPSetUp_BCGS;
  ksp->ops->solve           = KSPSolve_BCGS;
  ksp->ops->destroy         = KSPDefaultDestroy;
  ksp->ops->buildsolution   = KSPDefaultBuildSolution;
  ksp->ops->buildresidual   = KSPDefaultBuildResidual;
  ksp->ops->setfromoptions  = 0;
  ksp->ops->view            = 0;
  PetscFunctionReturn(0);
}
EXTERN_C_END
