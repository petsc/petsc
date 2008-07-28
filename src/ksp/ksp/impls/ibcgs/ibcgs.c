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

     sigma_1 = pi0 = phi0 = tau0 = 0

     sigma0 = r0'u0 

     rho0 = alpha0 = omega0 = 1

     do n = 1...:
        rhon   = phin_1 - omegan_1*sigman_2 + omegan_1*alphan_1*pin_1
        deltan = rhon*alphan_1/rhon_1
        betan  = deltan/omegan_1
        taun   = sigman_1 + betan*taun_1  - deltan*pin_1
        alphan = rhon/taun

        vn = un_1 + betan*vn_1 - deltan*qn_1
        qn = A*vn
        sn = rn_1 - alphan*vn
        tn = un_1 - alphan*qn
        zn = alphan*rn_1 + betan*zn_1 - alphan*deltan*vn_1

        phin = r0'sn
        pin  = r0'qn
        gamman = f0'sn
        etan   = f0'tn
        thetan = sn'tn
        kappan = tn'tn

        omegan = thetan/kappan
        sigman = gamman - omegan*etan

        rn = sn - omegan*tn
        xn = xn_1 + zn + omegan*sn

        Test for convergence

        un = A*rn

        Update n-1 locations with n locations
        un_1 = un
        xn_1 = xn
        rn_1 = rn
        sigman_2 = sigman_1
        sigman_1 = sigman
        pin_1    = pin
        phin_1   = phin
        zn_1     = zn
        qn_1     = qn
        vn_1     = vn
        alphan_1 = alphan
        taun_1   = taun
        rhon_1   = rhon


     enddo:

         These previous values are never used so need not be updated

         kappan_1 = kappan, thetan_1 = thetan, etan_1 = etan, gamman_1 = gamman, tn_1 = tn, sn_1 = sn, betan_1  = betan, deltan_1 = deltan
   
*/
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
