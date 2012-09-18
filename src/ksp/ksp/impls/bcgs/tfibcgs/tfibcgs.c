
/*
    This file implements transpose-free improved BiCGStab contributed by Jie Chen.
    It does not require matrix transpose as in ibcgs.c.
    Only right preconditioning is supported.
    This code is only correct for the real case... Need to modify for complex numbers...
*/
#include <../src/ksp/ksp/impls/bcgs/bcgsimpl.h>       /*I  "petscksp.h"  I*/

#undef __FUNCT__
#define __FUNCT__ "KSPSetUp_TFIBCGS"
PetscErrorCode KSPSetUp_TFIBCGS(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPDefaultGetWork(ksp,11);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include <petsc-private/pcimpl.h>            /*I "petscksp.h" I*/
#undef __FUNCT__
#define __FUNCT__ "KSPSolve_TFIBCGS"
PetscErrorCode  KSPSolve_TFIBCGS(KSP ksp)
{
  PetscErrorCode    ierr;
  PetscInt          i,j,N;
  PetscReal         rho;
  PetscScalar       tau,sigma,alpha,omega,beta;
  PetscScalar       xi1,xi2,xi3,xi4,xi5,xi6,xi7;
  Vec               X,B,R,V2,S2,T2,P2,RP,V,S,T,Q,U;
  PetscScalar       *PETSC_RESTRICT rp, *PETSC_RESTRICT r, *PETSC_RESTRICT s, *PETSC_RESTRICT t, *PETSC_RESTRICT q;
  PetscScalar       *PETSC_RESTRICT u, *PETSC_RESTRICT s2, *PETSC_RESTRICT t2, *PETSC_RESTRICT v2, *PETSC_RESTRICT p2;
  PetscScalar       insums[7],outsums[7];
  KSP_BCGS          *bcgs = (KSP_BCGS*)ksp->data;
  PC                pc;

  PetscFunctionBegin;
  if (!ksp->vec_rhs->petscnative) SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_SUP,"Only coded for PETSc vectors");
  ierr = VecGetLocalSize(ksp->vec_sol,&N);CHKERRQ(ierr);

  X   = ksp->vec_sol;
  B   = ksp->vec_rhs;
  V   = ksp->work[0];

  /* The followings are involved in modified inner product calculations and vector updates */
  RP  = ksp->work[1]; ierr = VecGetArray(RP,(PetscScalar**)&rp);CHKERRQ(ierr);   ierr = VecRestoreArray(RP,PETSC_NULL);CHKERRQ(ierr);
  R   = ksp->work[2]; ierr = VecGetArray(R,(PetscScalar**)&r);CHKERRQ(ierr);     ierr = VecRestoreArray(R,PETSC_NULL);CHKERRQ(ierr);
  S   = ksp->work[3]; ierr = VecGetArray(S,(PetscScalar**)&s);CHKERRQ(ierr);     ierr = VecRestoreArray(S,PETSC_NULL);CHKERRQ(ierr);
  T   = ksp->work[4]; ierr = VecGetArray(T,(PetscScalar**)&t);CHKERRQ(ierr);     ierr = VecRestoreArray(T,PETSC_NULL);CHKERRQ(ierr);
  Q   = ksp->work[5]; ierr = VecGetArray(Q,(PetscScalar**)&q);CHKERRQ(ierr);     ierr = VecRestoreArray(Q,PETSC_NULL);CHKERRQ(ierr);
  U   = ksp->work[6]; ierr = VecGetArray(U,(PetscScalar**)&u);CHKERRQ(ierr);     ierr = VecRestoreArray(U,PETSC_NULL);CHKERRQ(ierr);
  S2  = ksp->work[7]; ierr = VecGetArray(S2,(PetscScalar**)&s2);CHKERRQ(ierr);   ierr = VecRestoreArray(S2,PETSC_NULL);CHKERRQ(ierr);
  T2  = ksp->work[8]; ierr = VecGetArray(T2,(PetscScalar**)&t2);CHKERRQ(ierr);   ierr = VecRestoreArray(T2,PETSC_NULL);CHKERRQ(ierr);
  V2  = ksp->work[9]; ierr = VecGetArray(V2,(PetscScalar**)&v2);CHKERRQ(ierr);   ierr = VecRestoreArray(V2,PETSC_NULL);CHKERRQ(ierr);
  P2  = ksp->work[10]; ierr = VecGetArray(P2,(PetscScalar**)&p2);CHKERRQ(ierr);  ierr = VecRestoreArray(P2,PETSC_NULL);CHKERRQ(ierr);

  /* Only supports right preconditioning */
  if (ksp->pc_side != PC_RIGHT)
    SETERRQ1(((PetscObject)ksp)->comm,PETSC_ERR_SUP,"KSP tfibcgs does not support %s",PCSides[ksp->pc_side]);
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
  if (!ksp->guess_zero) {
    ierr = MatMult(pc->mat,X,P2);CHKERRQ(ierr); /* P2 is used as temporary storage */
    ierr = VecCopy(B,R);CHKERRQ(ierr);
    ierr = VecAXPY(R,-1.0,P2);CHKERRQ(ierr);
  }
  else {
    ierr = VecCopy(B,R);CHKERRQ(ierr);
  }

  /* Test for nothing to do */
  if (ksp->normtype != KSP_NORM_NONE) {
    ierr = VecNorm(R,NORM_2,&rho);CHKERRQ(ierr);
  }
  ierr = PetscObjectTakeAccess(ksp);CHKERRQ(ierr);
  ksp->its   = 0;
  ksp->rnorm = rho;
  ierr = PetscObjectGrantAccess(ksp);CHKERRQ(ierr);
  KSPLogResidualHistory(ksp,rho);
  ierr = KSPMonitor(ksp,0,rho);CHKERRQ(ierr);
  ierr = (*ksp->converged)(ksp,0,rho,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
  if (ksp->reason) PetscFunctionReturn(0);

  /* Initialize iterates */
  ierr = VecCopy(R,RP);CHKERRQ(ierr); /* rp <- r */
  ierr = PCApply(pc,R,P2);CHKERRQ(ierr); /* p2 <- K r */
  ierr = MatMult(pc->mat,P2,V);CHKERRQ(ierr); /* v <- A p2 */
  tau = rho*rho; /* tau <- (r,rp) */
  ierr = VecDot(V,RP,&sigma);CHKERRQ(ierr); /* sigma <- (v,rp) */
  /* rho has been computed previously */

  for (i=0; i<ksp->max_it; i++) {

    /* test denominator */
    if (sigma == 0.0) SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_PLIB,"Divide by zero");

    /* scalar update */
    alpha = tau / sigma;

    /* vector update */
    ierr = VecWAXPY(S,-alpha,V,R);CHKERRQ(ierr);  /* s <- r - alpha v */

    /* matmult and pc */
    ierr = PCApply(pc,S,S2);CHKERRQ(ierr); /* s2 <- K s */
    ierr = MatMult(pc->mat,S2,T);CHKERRQ(ierr); /* t <- A s2 */
    ierr = PCApply(pc,T,T2);CHKERRQ(ierr); /* t2 <- K t */
    ierr = MatMult(pc->mat,T2,Q);CHKERRQ(ierr); /* q <- A t2 */
    ierr = PCApply(pc,V,V2);CHKERRQ(ierr); /* v2 <- K v */
    ierr = MatMult(pc->mat,V2,U);CHKERRQ(ierr); /* u <- A v2 */

    /* inner prodcuts */
    ierr = PetscLogEventBegin(VEC_ReduceArithmetic,0,0,0,0);CHKERRQ(ierr);
    xi1 = xi2 = xi3 = xi4 = xi5 = xi6 = xi7 = 0.0;
    for (j=0; j<N; j++) {
      xi1 += s[j]*s[j]; /* xi1 <- (s,s) */
      xi2 += t[j]*s[j]; /* xi2 <- (t,s) */
      xi3 += t[j]*t[j]; /* xi3 <- (t,t) */
      xi4 += t[j]*rp[j]; /* xi4 <- (t,rp) */
      xi5 += s[j]*rp[j]; /* xi5 <- (s,rp) */
      xi6 += q[j]*rp[j]; /* xi6 <- (q,rp) */
      xi7 += u[j]*rp[j]; /* xi7 <- (u,rp) */
    }
    PetscLogFlops(14.0*N);
    ierr = PetscLogEventEnd(VEC_ReduceArithmetic,0,0,0,0);CHKERRQ(ierr);
    insums[0] = xi1;
    insums[1] = xi2;
    insums[2] = xi3;
    insums[3] = xi4;
    insums[4] = xi5;
    insums[5] = xi6;
    insums[6] = xi7;
    ierr = PetscLogEventBarrierBegin(VEC_ReduceBarrier,0,0,0,0,((PetscObject)ksp)->comm);CHKERRQ(ierr);
    ierr = MPI_Allreduce(insums,outsums,7,MPIU_SCALAR,MPIU_SUM,((PetscObject)ksp)->comm);CHKERRQ(ierr);
    ierr = PetscLogEventBarrierEnd(VEC_ReduceBarrier,0,0,0,0,((PetscObject)ksp)->comm);CHKERRQ(ierr);
    xi1 = outsums[0];
    xi2 = outsums[1];
    xi3 = outsums[2];
    xi4 = outsums[3];
    xi5 = outsums[4];
    xi6 = outsums[5];
    xi7 = outsums[6];

    /* test denominator */
    if (xi3 == 0.0) SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_PLIB,"Divide by zero");
    if (sigma == 0.0) SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_PLIB,"Divide by zero");

    /* scalar updates */
    omega = xi2 / xi3;
    beta = - xi4 / sigma;
    rho = PetscSqrtReal(PetscAbsScalar(xi1 - omega * xi2)); /* residual norm */
    tau = xi5 - omega * xi4;
    sigma = - omega * (xi6 + beta * xi7);

    /* vector updates */
    ierr = VecAXPBYPCZ(X,alpha,omega,1.0,P2,S2);CHKERRQ(ierr); /* x <- alpha * p2 + omega * s2 + x */

    /* convergence test */
    ierr = PetscObjectTakeAccess(ksp);CHKERRQ(ierr);
    ksp->its++;
    ksp->rnorm = rho;
    ierr = PetscObjectGrantAccess(ksp);CHKERRQ(ierr);
    KSPLogResidualHistory(ksp,rho);
    ierr = KSPMonitor(ksp,i+1,rho);CHKERRQ(ierr);
    ierr = (*ksp->converged)(ksp,i+1,rho,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
    if (ksp->reason) break;

    /* vector updates */
    ierr = PetscLogEventBegin(VEC_Ops,0,0,0,0);CHKERRQ(ierr);
    for (j=0; j<N; j++) {
      r[j] = s[j] - omega * t[j]; /* r <- s - omega t */
      p2[j] = s2[j] - omega * t2[j] + beta * (p2[j] - omega * v2[j]); /* p2 <- s2 - omega t2 + beta * (p2 - omega v2) */
    }
    PetscLogFlops(8.0*N);
    ierr = PetscLogEventEnd(VEC_Ops,0,0,0,0);CHKERRQ(ierr);

    /* matmult */
    ierr = MatMult(pc->mat,P2,V);CHKERRQ(ierr); /* v <- A p2 */

  }

  if (i >= ksp->max_it) {
    ksp->reason = KSP_DIVERGED_ITS;
  }

  PetscFunctionReturn(0);
}

/*MC
     KSPTFIBCGS - Implements the transpose-free improved BiCGStab (Stabilized version of BiConjugate Gradient Squared) method.

   Options Database Keys:
.   see KSPSolve()

   Level: beginner

   Notes: Only supports right preconditioning

.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP, KSPBICG, KSPFBCGSL, KSPSetPCSide()
M*/
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "KSPCreate_TFIBCGS"
PetscErrorCode  KSPCreate_TFIBCGS(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_BCGS       *bcgs;

  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"TFIBCGS is only correct for the real case... Need to modify for complex numbers...");
#endif
  ierr = PetscNewLog(ksp,KSP_BCGS,&bcgs);CHKERRQ(ierr);
  ksp->data                 = bcgs;
  ksp->ops->setup           = KSPSetUp_TFIBCGS;
  ksp->ops->solve           = KSPSolve_TFIBCGS;
  ksp->ops->destroy         = KSPDestroy_BCGS;
  ksp->ops->reset           = KSPReset_BCGS;
  ksp->ops->buildsolution   = KSPBuildSolution_BCGS;
  ksp->ops->buildresidual   = KSPDefaultBuildResidual;
  ksp->ops->setfromoptions  = KSPSetFromOptions_BCGS;
  ksp->ops->view            = KSPView_BCGS;
  ksp->pc_side              = PC_RIGHT; /* set default PC side */

  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_RIGHT,1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
