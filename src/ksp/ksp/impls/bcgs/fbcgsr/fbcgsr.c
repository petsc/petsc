
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPSetWorkVecs(ksp,8);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode  KSPSolve_FBCGSR(KSP ksp)
{
  PetscErrorCode    ierr;
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
  if (!ksp->vec_rhs->petscnative) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"Only coded for PETSc vectors");
  ierr = VecGetLocalSize(ksp->vec_sol,&N);CHKERRQ(ierr);

  X  = ksp->vec_sol;
  B  = ksp->vec_rhs;
  P2 = ksp->work[0];

  /* The followings are involved in modified inner product calculations and vector updates */
  RP = ksp->work[1]; ierr = VecGetArray(RP,(PetscScalar**)&rp);CHKERRQ(ierr); ierr = VecRestoreArray(RP,NULL);CHKERRQ(ierr);
  R  = ksp->work[2]; ierr = VecGetArray(R,(PetscScalar**)&r);CHKERRQ(ierr);   ierr = VecRestoreArray(R,NULL);CHKERRQ(ierr);
  P  = ksp->work[3]; ierr = VecGetArray(P,(PetscScalar**)&p);CHKERRQ(ierr);   ierr = VecRestoreArray(P,NULL);CHKERRQ(ierr);
  V  = ksp->work[4]; ierr = VecGetArray(V,(PetscScalar**)&v);CHKERRQ(ierr);   ierr = VecRestoreArray(V,NULL);CHKERRQ(ierr);
  S  = ksp->work[5]; ierr = VecGetArray(S,(PetscScalar**)&s);CHKERRQ(ierr);   ierr = VecRestoreArray(S,NULL);CHKERRQ(ierr);
  T  = ksp->work[6]; ierr = VecGetArray(T,(PetscScalar**)&t);CHKERRQ(ierr);   ierr = VecRestoreArray(T,NULL);CHKERRQ(ierr);
  S2 = ksp->work[7]; ierr = VecGetArray(S2,(PetscScalar**)&s2);CHKERRQ(ierr); ierr = VecRestoreArray(S2,NULL);CHKERRQ(ierr);

  /* Only supports right preconditioning */
  if (ksp->pc_side != PC_RIGHT) SETERRQ1(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"KSP fbcgsr does not support %s",PCSides[ksp->pc_side]);
  if (!ksp->guess_zero) {
    if (!bcgs->guess) {
      ierr = VecDuplicate(X,&bcgs->guess);CHKERRQ(ierr);
    }
    ierr = VecCopy(X,bcgs->guess);CHKERRQ(ierr);
  } else {
    ierr = VecSet(X,0.0);CHKERRQ(ierr);
  }

  /* Compute initial residual */
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetUp(pc);CHKERRQ(ierr);
  ierr = PCGetOperators(pc,&mat,NULL);CHKERRQ(ierr);
  if (!ksp->guess_zero) {
    ierr = KSP_MatMult(ksp,mat,X,P2);CHKERRQ(ierr); /* P2 is used as temporary storage */
    ierr = VecCopy(B,R);CHKERRQ(ierr);
    ierr = VecAXPY(R,-1.0,P2);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(B,R);CHKERRQ(ierr);
  }

  /* Test for nothing to do */
  ierr = VecNorm(R,NORM_2,&rho);CHKERRQ(ierr);
  ierr     = PetscObjectSAWsTakeAccess((PetscObject)ksp);CHKERRQ(ierr);
  ksp->its = 0;
  if (ksp->normtype != KSP_NORM_NONE) ksp->rnorm = rho;
  else ksp->rnorm = 0;
  ierr = PetscObjectSAWsGrantAccess((PetscObject)ksp);CHKERRQ(ierr);
  ierr = KSPLogResidualHistory(ksp,ksp->rnorm);CHKERRQ(ierr);
  ierr = KSPMonitor(ksp,0,ksp->rnorm);CHKERRQ(ierr);
  ierr = (*ksp->converged)(ksp,0,ksp->rnorm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
  if (ksp->reason) PetscFunctionReturn(0);

  /* Initialize iterates */
  ierr = VecCopy(R,RP);CHKERRQ(ierr); /* rp <- r */
  ierr = VecCopy(R,P);CHKERRQ(ierr); /* p <- r */

  /* Big loop */
  for (i=0; i<ksp->max_it; i++) {

    /* matmult and pc */
    ierr = KSP_PCApply(ksp,P,P2);CHKERRQ(ierr); /* p2 <- K p */
    ierr = KSP_MatMult(ksp,mat,P2,V);CHKERRQ(ierr); /* v <- A p2 */

    /* inner prodcuts */
    if (i==0) {
      tau  = rho*rho;
      ierr = VecDot(V,RP,&sigma);CHKERRQ(ierr); /* sigma <- (v,rp) */
    } else {
      ierr = PetscLogEventBegin(VEC_ReduceArithmetic,0,0,0,0);CHKERRQ(ierr);
      tau  = sigma = 0.0;
      for (j=0; j<N; j++) {
        tau   += r[j]*rp[j]; /* tau <- (r,rp) */
        sigma += v[j]*rp[j]; /* sigma <- (v,rp) */
      }
      ierr = PetscLogFlops(4.0*N);CHKERRQ(ierr);
      ierr      = PetscLogEventEnd(VEC_ReduceArithmetic,0,0,0,0);CHKERRQ(ierr);
      insums[0] = tau;
      insums[1] = sigma;
      ierr      = PetscLogEventBegin(VEC_ReduceCommunication,0,0,0,0);CHKERRQ(ierr);
      ierr      = MPIU_Allreduce(insums,outsums,2,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)ksp));CHKERRQ(ierr);
      ierr      = PetscLogEventEnd(VEC_ReduceCommunication,0,0,0,0);CHKERRQ(ierr);
      tau       = outsums[0];
      sigma     = outsums[1];
    }

    /* scalar update */
    alpha = tau / sigma;

    /* vector update */
    ierr = VecWAXPY(S,-alpha,V,R);CHKERRQ(ierr);  /* s <- r - alpha v */

    /* matmult and pc */
    ierr = KSP_PCApply(ksp,S,S2);CHKERRQ(ierr); /* s2 <- K s */
    ierr = KSP_MatMult(ksp,mat,S2,T);CHKERRQ(ierr); /* t <- A s2 */

    /* inner prodcuts */
    ierr = PetscLogEventBegin(VEC_ReduceArithmetic,0,0,0,0);CHKERRQ(ierr);
    xi1  = xi2 = xi3 = xi4 = 0.0;
    for (j=0; j<N; j++) {
      xi1 += s[j]*s[j]; /* xi1 <- (s,s) */
      xi2 += t[j]*s[j]; /* xi2 <- (t,s) */
      xi3 += t[j]*t[j]; /* xi3 <- (t,t) */
      xi4 += t[j]*rp[j]; /* xi4 <- (t,rp) */
    }
    ierr = PetscLogFlops(8.0*N);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(VEC_ReduceArithmetic,0,0,0,0);CHKERRQ(ierr);

    insums[0] = xi1;
    insums[1] = xi2;
    insums[2] = xi3;
    insums[3] = xi4;

    ierr = PetscLogEventBegin(VEC_ReduceCommunication,0,0,0,0);CHKERRQ(ierr);
    ierr = MPIU_Allreduce(insums,outsums,4,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)ksp));CHKERRQ(ierr);
    ierr = PetscLogEventEnd(VEC_ReduceCommunication,0,0,0,0);CHKERRQ(ierr);
    xi1  = outsums[0];
    xi2  = outsums[1];
    xi3  = outsums[2];
    xi4  = outsums[3];

    /* test denominator */
    if (xi3 == 0.0) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_PLIB,"Divide by zero");
    if (sigma == 0.0) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_PLIB,"Divide by zero");

    /* scalar updates */
    omega = xi2 / xi3;
    beta  = -xi4 / sigma;
    rho   = PetscSqrtReal(PetscAbsScalar(xi1 - omega * xi2)); /* residual norm */

    /* vector updates */
    ierr = VecAXPBYPCZ(X,alpha,omega,1.0,P2,S2);CHKERRQ(ierr); /* x <- alpha * p2 + omega * s2 + x */

    /* convergence test */
    ierr = PetscObjectSAWsTakeAccess((PetscObject)ksp);CHKERRQ(ierr);
    ksp->its++;
    if (ksp->normtype != KSP_NORM_NONE) ksp->rnorm = rho;
    else ksp->rnorm = 0;
    ierr = PetscObjectSAWsGrantAccess((PetscObject)ksp);CHKERRQ(ierr);
    ierr = KSPLogResidualHistory(ksp,ksp->rnorm);CHKERRQ(ierr);
    ierr = KSPMonitor(ksp,i+1,ksp->rnorm);CHKERRQ(ierr);
    ierr = (*ksp->converged)(ksp,i+1,ksp->rnorm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
    if (ksp->reason) break;

    /* vector updates */
    ierr = PetscLogEventBegin(VEC_Ops,0,0,0,0);CHKERRQ(ierr);
    for (j=0; j<N; j++) {
      r[j] = s[j] - omega * t[j]; /* r <- s - omega t */
      p[j] = r[j] + beta * (p[j] - omega * v[j]); /* p <- r + beta * (p - omega v) */
    }
    ierr = PetscLogFlops(6.0*N);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(VEC_Ops,0,0,0,0);CHKERRQ(ierr);

  }

  if (i >= ksp->max_it) ksp->reason = KSP_DIVERGED_ITS;
  PetscFunctionReturn(0);
}

/*MC
     KSPFBCGSR - Implements a mathematically equivalent variant of FBiCGSTab.

   Options Database Keys:
.   see KSPSolve()

   Level: beginner

   Notes:
    Only allow right preconditioning

.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP, KSPBICG, KSPFBCGSL, KSPSetPCSide()
M*/
PETSC_EXTERN PetscErrorCode KSPCreate_FBCGSR(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_BCGS       *bcgs;

  PetscFunctionBegin;
  ierr = PetscNewLog(ksp,&bcgs);CHKERRQ(ierr);

  ksp->data                = bcgs;
  ksp->ops->setup          = KSPSetUp_FBCGSR;
  ksp->ops->solve          = KSPSolve_FBCGSR;
  ksp->ops->destroy        = KSPDestroy_BCGS;
  ksp->ops->reset          = KSPReset_BCGS;
  ksp->ops->buildsolution  = KSPBuildSolution_BCGS;
  ksp->ops->buildresidual  = KSPBuildResidualDefault;
  ksp->ops->setfromoptions = KSPSetFromOptions_BCGS;
  ksp->pc_side             = PC_RIGHT; /* set default PC side */

  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,3);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_RIGHT,2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_RIGHT,1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
