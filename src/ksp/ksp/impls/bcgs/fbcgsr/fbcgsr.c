
/*
    This file implements FBiCGStab-R.
    Only allow right preconditioning.
    FBiCGStab-R is a mathematically equivalent variant of FBiCGStab. Differences are:
      (1) There are fewer MPI_Allreduce calls.
      (2) The convergence occasionally is much faster than that of FBiCGStab.
*/
#include <../src/ksp/ksp/impls/bcgs/bcgsimpl.h>       /*I  "petscksp.h"  I*/
#include <petsc/private/vecimpl.h>

static PetscErrorCode KSPSetUp_FBCGSR(KSP ksp)
{
  PetscFunctionBegin;
  CHKERRQ(KSPSetWorkVecs(ksp,8));
  PetscFunctionReturn(0);
}

static PetscErrorCode  KSPSolve_FBCGSR(KSP ksp)
{
  PetscInt          i,j,N;
  PetscScalar       tau,sigma,alpha,omega,beta;
  PetscReal         rho;
  PetscScalar       xi1,xi2,xi3,xi4;
  Vec               X,B,P,P2,RP,R,V,S,T,S2;
  PetscScalar       *PETSC_RESTRICT rp, *PETSC_RESTRICT r, *PETSC_RESTRICT p;
  PetscScalar       *PETSC_RESTRICT v, *PETSC_RESTRICT s, *PETSC_RESTRICT t, *PETSC_RESTRICT s2;
  PetscScalar       insums[4],outsums[4];
  KSP_BCGS          *bcgs = (KSP_BCGS*)ksp->data;
  PC                pc;
  Mat               mat;

  PetscFunctionBegin;
  PetscCheckFalse(!ksp->vec_rhs->petscnative,PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"Only coded for PETSc vectors");
  CHKERRQ(VecGetLocalSize(ksp->vec_sol,&N));

  X  = ksp->vec_sol;
  B  = ksp->vec_rhs;
  P2 = ksp->work[0];

  /* The followings are involved in modified inner product calculations and vector updates */
  RP = ksp->work[1]; CHKERRQ(VecGetArray(RP,(PetscScalar**)&rp)); CHKERRQ(VecRestoreArray(RP,NULL));
  R  = ksp->work[2]; CHKERRQ(VecGetArray(R,(PetscScalar**)&r));   CHKERRQ(VecRestoreArray(R,NULL));
  P  = ksp->work[3]; CHKERRQ(VecGetArray(P,(PetscScalar**)&p));   CHKERRQ(VecRestoreArray(P,NULL));
  V  = ksp->work[4]; CHKERRQ(VecGetArray(V,(PetscScalar**)&v));   CHKERRQ(VecRestoreArray(V,NULL));
  S  = ksp->work[5]; CHKERRQ(VecGetArray(S,(PetscScalar**)&s));   CHKERRQ(VecRestoreArray(S,NULL));
  T  = ksp->work[6]; CHKERRQ(VecGetArray(T,(PetscScalar**)&t));   CHKERRQ(VecRestoreArray(T,NULL));
  S2 = ksp->work[7]; CHKERRQ(VecGetArray(S2,(PetscScalar**)&s2)); CHKERRQ(VecRestoreArray(S2,NULL));

  /* Only supports right preconditioning */
  PetscCheckFalse(ksp->pc_side != PC_RIGHT,PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"KSP fbcgsr does not support %s",PCSides[ksp->pc_side]);
  if (!ksp->guess_zero) {
    if (!bcgs->guess) {
      CHKERRQ(VecDuplicate(X,&bcgs->guess));
    }
    CHKERRQ(VecCopy(X,bcgs->guess));
  } else {
    CHKERRQ(VecSet(X,0.0));
  }

  /* Compute initial residual */
  CHKERRQ(KSPGetPC(ksp,&pc));
  CHKERRQ(PCSetUp(pc));
  CHKERRQ(PCGetOperators(pc,&mat,NULL));
  if (!ksp->guess_zero) {
    CHKERRQ(KSP_MatMult(ksp,mat,X,P2)); /* P2 is used as temporary storage */
    CHKERRQ(VecCopy(B,R));
    CHKERRQ(VecAXPY(R,-1.0,P2));
  } else {
    CHKERRQ(VecCopy(B,R));
  }

  /* Test for nothing to do */
  CHKERRQ(VecNorm(R,NORM_2,&rho));
  CHKERRQ(PetscObjectSAWsTakeAccess((PetscObject)ksp));
  ksp->its = 0;
  if (ksp->normtype != KSP_NORM_NONE) ksp->rnorm = rho;
  else ksp->rnorm = 0;
  CHKERRQ(PetscObjectSAWsGrantAccess((PetscObject)ksp));
  CHKERRQ(KSPLogResidualHistory(ksp,ksp->rnorm));
  CHKERRQ(KSPMonitor(ksp,0,ksp->rnorm));
  CHKERRQ((*ksp->converged)(ksp,0,ksp->rnorm,&ksp->reason,ksp->cnvP));
  if (ksp->reason) PetscFunctionReturn(0);

  /* Initialize iterates */
  CHKERRQ(VecCopy(R,RP)); /* rp <- r */
  CHKERRQ(VecCopy(R,P)); /* p <- r */

  /* Big loop */
  for (i=0; i<ksp->max_it; i++) {

    /* matmult and pc */
    CHKERRQ(KSP_PCApply(ksp,P,P2)); /* p2 <- K p */
    CHKERRQ(KSP_MatMult(ksp,mat,P2,V)); /* v <- A p2 */

    /* inner prodcuts */
    if (i==0) {
      tau  = rho*rho;
      CHKERRQ(VecDot(V,RP,&sigma)); /* sigma <- (v,rp) */
    } else {
      CHKERRQ(PetscLogEventBegin(VEC_ReduceArithmetic,0,0,0,0));
      tau  = sigma = 0.0;
      for (j=0; j<N; j++) {
        tau   += r[j]*rp[j]; /* tau <- (r,rp) */
        sigma += v[j]*rp[j]; /* sigma <- (v,rp) */
      }
      CHKERRQ(PetscLogFlops(4.0*N));
      CHKERRQ(PetscLogEventEnd(VEC_ReduceArithmetic,0,0,0,0));
      insums[0] = tau;
      insums[1] = sigma;
      CHKERRQ(PetscLogEventBegin(VEC_ReduceCommunication,0,0,0,0));
      CHKERRMPI(MPIU_Allreduce(insums,outsums,2,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)ksp)));
      CHKERRQ(PetscLogEventEnd(VEC_ReduceCommunication,0,0,0,0));
      tau       = outsums[0];
      sigma     = outsums[1];
    }

    /* scalar update */
    alpha = tau / sigma;

    /* vector update */
    CHKERRQ(VecWAXPY(S,-alpha,V,R));  /* s <- r - alpha v */

    /* matmult and pc */
    CHKERRQ(KSP_PCApply(ksp,S,S2)); /* s2 <- K s */
    CHKERRQ(KSP_MatMult(ksp,mat,S2,T)); /* t <- A s2 */

    /* inner prodcuts */
    CHKERRQ(PetscLogEventBegin(VEC_ReduceArithmetic,0,0,0,0));
    xi1  = xi2 = xi3 = xi4 = 0.0;
    for (j=0; j<N; j++) {
      xi1 += s[j]*s[j]; /* xi1 <- (s,s) */
      xi2 += t[j]*s[j]; /* xi2 <- (t,s) */
      xi3 += t[j]*t[j]; /* xi3 <- (t,t) */
      xi4 += t[j]*rp[j]; /* xi4 <- (t,rp) */
    }
    CHKERRQ(PetscLogFlops(8.0*N));
    CHKERRQ(PetscLogEventEnd(VEC_ReduceArithmetic,0,0,0,0));

    insums[0] = xi1;
    insums[1] = xi2;
    insums[2] = xi3;
    insums[3] = xi4;

    CHKERRQ(PetscLogEventBegin(VEC_ReduceCommunication,0,0,0,0));
    CHKERRMPI(MPIU_Allreduce(insums,outsums,4,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)ksp)));
    CHKERRQ(PetscLogEventEnd(VEC_ReduceCommunication,0,0,0,0));
    xi1  = outsums[0];
    xi2  = outsums[1];
    xi3  = outsums[2];
    xi4  = outsums[3];

    /* test denominator */
    if ((xi3 == 0.0) || (sigma == 0.0)) {
      PetscCheckFalse(ksp->errorifnotconverged,PetscObjectComm((PetscObject)ksp),PETSC_ERR_NOT_CONVERGED,"KSPSolve has failed due to zero inner product");
      else ksp->reason = KSP_DIVERGED_BREAKDOWN;
      CHKERRQ(PetscInfo(ksp,"KSPSolve has failed due to zero inner product\n"));
      break;
    }

    /* scalar updates */
    omega = xi2 / xi3;
    beta  = -xi4 / sigma;
    rho   = PetscSqrtReal(PetscAbsScalar(xi1 - omega * xi2)); /* residual norm */

    /* vector updates */
    CHKERRQ(VecAXPBYPCZ(X,alpha,omega,1.0,P2,S2)); /* x <- alpha * p2 + omega * s2 + x */

    /* convergence test */
    CHKERRQ(PetscObjectSAWsTakeAccess((PetscObject)ksp));
    ksp->its++;
    if (ksp->normtype != KSP_NORM_NONE) ksp->rnorm = rho;
    else ksp->rnorm = 0;
    CHKERRQ(PetscObjectSAWsGrantAccess((PetscObject)ksp));
    CHKERRQ(KSPLogResidualHistory(ksp,ksp->rnorm));
    CHKERRQ(KSPMonitor(ksp,i+1,ksp->rnorm));
    CHKERRQ((*ksp->converged)(ksp,i+1,ksp->rnorm,&ksp->reason,ksp->cnvP));
    if (ksp->reason) break;

    /* vector updates */
    CHKERRQ(PetscLogEventBegin(VEC_Ops,0,0,0,0));
    for (j=0; j<N; j++) {
      r[j] = s[j] - omega * t[j]; /* r <- s - omega t */
      p[j] = r[j] + beta * (p[j] - omega * v[j]); /* p <- r + beta * (p - omega v) */
    }
    CHKERRQ(PetscLogFlops(6.0*N));
    CHKERRQ(PetscLogEventEnd(VEC_Ops,0,0,0,0));

  }

  if (i >= ksp->max_it) ksp->reason = KSP_DIVERGED_ITS;
  PetscFunctionReturn(0);
}

/*MC
     KSPFBCGSR - Implements a mathematically equivalent variant of FBiCGSTab.

   Options Database Keys:
    see KSPSolve()

   Level: beginner

   Notes:
    Only allow right preconditioning

.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP, KSPBICG, KSPFBCGSL, KSPSetPCSide()
M*/
PETSC_EXTERN PetscErrorCode KSPCreate_FBCGSR(KSP ksp)
{
  KSP_BCGS       *bcgs;

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(ksp,&bcgs));

  ksp->data                = bcgs;
  ksp->ops->setup          = KSPSetUp_FBCGSR;
  ksp->ops->solve          = KSPSolve_FBCGSR;
  ksp->ops->destroy        = KSPDestroy_BCGS;
  ksp->ops->reset          = KSPReset_BCGS;
  ksp->ops->buildsolution  = KSPBuildSolution_BCGS;
  ksp->ops->buildresidual  = KSPBuildResidualDefault;
  ksp->ops->setfromoptions = KSPSetFromOptions_BCGS;
  ksp->pc_side             = PC_RIGHT; /* set default PC side */

  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,3));
  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_RIGHT,2));
  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_RIGHT,1));
  PetscFunctionReturn(0);
}
