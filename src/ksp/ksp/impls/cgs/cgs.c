#define PETSCKSP_DLL

/*                       

    Note that for the complex numbers version, the VecDot() arguments
    within the code MUST remain in the order given for correct computation
    of inner products.
*/
#include "private/kspimpl.h"

#undef __FUNCT__  
#define __FUNCT__ "KSPSetUp_CGS"
static PetscErrorCode KSPSetUp_CGS(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ksp->pc_side == PC_SYMMETRIC) SETERRQ(PETSC_ERR_SUP,"no symmetric preconditioning for KSPCGS");
  ierr = KSPDefaultGetWork(ksp,7);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSolve_CGS"
static PetscErrorCode  KSPSolve_CGS(KSP ksp)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscScalar    rho,rhoold,a,s,b;
  Vec            X,B,V,P,R,RP,T,Q,U,AUQ;
  PetscReal      dp = 0.0;
  PetscTruth     diagonalscale;

  PetscFunctionBegin;
  /* not sure what residual norm it does use, should use for right preconditioning */

  ierr    = PCDiagonalScale(ksp->pc,&diagonalscale);CHKERRQ(ierr);
  if (diagonalscale) SETERRQ1(PETSC_ERR_SUP,"Krylov method %s does not support diagonal scaling",((PetscObject)ksp)->type_name);

  X       = ksp->vec_sol;
  B       = ksp->vec_rhs;
  R       = ksp->work[0];
  RP      = ksp->work[1];
  V       = ksp->work[2];
  T       = ksp->work[3];
  Q       = ksp->work[4];
  P       = ksp->work[5];
  U       = ksp->work[6];
  AUQ     = V;

  /* Compute initial preconditioned residual */
  ierr = KSPInitialResidual(ksp,X,V,T,R,B);CHKERRQ(ierr);

  /* Test for nothing to do */
  ierr = VecNorm(R,NORM_2,&dp);CHKERRQ(ierr);
  if (ksp->normtype == KSP_NORM_NATURAL) {
    dp *= dp;
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
  /*  added for Fidap */
  /* Penalize Startup - Isaac Hasbani Trick for CGS 
     Since most initial conditions result in a mostly 0 residual,
     we change all the 0 values in the vector RP to the maximum.
  */
  if (ksp->normtype == KSP_NORM_NATURAL) {
     PetscReal   vr0max;
     PetscScalar *tmp_RP=0;
     PetscInt    numnp=0, *max_pos=0;
     ierr = VecMax(RP, max_pos, &vr0max);CHKERRQ(ierr);
     ierr = VecGetArray(RP, &tmp_RP);CHKERRQ(ierr);
     ierr = VecGetLocalSize(RP, &numnp);CHKERRQ(ierr);
     for (i=0; i<numnp; i++) {
       if (tmp_RP[i] == 0.0) tmp_RP[i] = vr0max;
     }
     ierr = VecRestoreArray(RP, &tmp_RP);CHKERRQ(ierr);
  }
  /*  end of addition for Fidap */

  /* Set the initial conditions */
  ierr = VecDot(R,RP,&rhoold);CHKERRQ(ierr);        /* rhoold = (r,rp)      */
  ierr = VecCopy(R,U);CHKERRQ(ierr);
  ierr = VecCopy(R,P);CHKERRQ(ierr);
  ierr = KSP_PCApplyBAorAB(ksp,P,V,T);CHKERRQ(ierr);

  i = 0;
  do {

    ierr = VecDot(V,RP,&s);CHKERRQ(ierr);           /* s <- (v,rp)          */
    a = rhoold / s;                                  /* a <- rho / s         */
    ierr = VecWAXPY(Q,-a,V,U);CHKERRQ(ierr);      /* q <- u - a v         */
    ierr = VecWAXPY(T,1.0,U,Q);CHKERRQ(ierr);      /* t <- u + q           */
    ierr = VecAXPY(X,a,T);CHKERRQ(ierr);           /* x <- x + a (u + q)   */
    ierr = KSP_PCApplyBAorAB(ksp,T,AUQ,U);CHKERRQ(ierr);
    ierr = VecAXPY(R,-a,AUQ);CHKERRQ(ierr);       /* r <- r - a K (u + q) */
    ierr = VecDot(R,RP,&rho);CHKERRQ(ierr);         /* rho <- (r,rp)        */
    if (ksp->normtype == KSP_NORM_NATURAL) {
      dp = PetscAbsScalar(rho);
    } else {
      ierr = VecNorm(R,NORM_2,&dp);CHKERRQ(ierr);
    }

    ierr = PetscObjectTakeAccess(ksp);CHKERRQ(ierr);
    ksp->its++;
    ksp->rnorm = dp;
    ierr = PetscObjectGrantAccess(ksp);CHKERRQ(ierr);
    KSPLogResidualHistory(ksp,dp);
    KSPMonitor(ksp,i+1,dp);
    ierr = (*ksp->converged)(ksp,i+1,dp,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
    if (ksp->reason) break;

    b    = rho / rhoold;                             /* b <- rho / rhoold    */
    ierr = VecWAXPY(U,b,Q,R);CHKERRQ(ierr);         /* u <- r + b q         */
    ierr = VecAXPY(Q,b,P);CHKERRQ(ierr);
    ierr = VecWAXPY(P,b,Q,U);CHKERRQ(ierr);         /* p <- u + b(q + b p)  */
    ierr = KSP_PCApplyBAorAB(ksp,P,V,Q);CHKERRQ(ierr);      /* v <- K p    */
    rhoold = rho;
    i++;
  } while (i<ksp->max_it);
  if (i >= ksp->max_it) {
    ksp->reason = KSP_DIVERGED_ITS;
  }

  ierr = KSPUnwindPreconditioner(ksp,X,T);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
     KSPCGS - This code implements the CGS (Conjugate Gradient Squared) method. 

   Options Database Keys:
.   see KSPSolve()

   Level: beginner

   References: Sonneveld, 1989.

   Notes: Does not require a symmetric matrix. Does not apply transpose of the matrix.
        Supports left and right preconditioning, but not symmetric.

   Developer Notes: Has this weird support for doing the convergence test with the natural norm, I assume this works only with 
      no preconditioning and symmetric positive definite operator.

.seealso: KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP, KSPBCGS, KSPSetPreconditionerSide()
M*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "KSPCreate_CGS"
PetscErrorCode PETSCKSP_DLLEXPORT KSPCreate_CGS(KSP ksp)
{
  PetscFunctionBegin;
  ksp->data                      = (void*)0;
  ksp->ops->setup                = KSPSetUp_CGS;
  ksp->ops->solve                = KSPSolve_CGS;
  ksp->ops->destroy              = KSPDefaultDestroy;
  ksp->ops->buildsolution        = KSPDefaultBuildSolution;
  ksp->ops->buildresidual        = KSPDefaultBuildResidual;
  ksp->ops->setfromoptions       = 0;
  ksp->ops->view                 = 0;
  PetscFunctionReturn(0);
}
EXTERN_C_END
