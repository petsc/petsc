
/*
    This file implements improved flexible BiCGStab contributed by Jie Chen.
    Only right preconditioning is supported. 
*/
#include <../src/ksp/ksp/impls/bcgs/bcgsimpl.h>       /*I  "petscksp.h"  I*/

#undef __FUNCT__  
#define __FUNCT__ "KSPSetUp_IFBCGS"
PetscErrorCode KSPSetUp_IFBCGS(KSP ksp)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = KSPDefaultGetWork(ksp,11);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}

#include <petsc-private/pcimpl.h>            /*I "petscksp.h" I*/
#undef __FUNCT__  
#define __FUNCT__ "KSPSolve_IFBCGS"
PetscErrorCode  KSPSolve_IFBCGS(KSP ksp)
{
  PetscErrorCode    ierr;
  PetscInt          i,j,N;
  PetscReal         rho;
  PetscScalar       tau,sigma,alpha,omega,beta;
  PetscScalar       xi1,xi2,xi3,xi4,xi5,xi6,xi7,xi8;
  Vec               X,B,R,V2,S2,T2,P2,RP,V,S,T,Q,U;
  PetscScalar       *PETSC_RESTRICT r, *PETSC_RESTRICT v2, *PETSC_RESTRICT s2, *PETSC_RESTRICT t2, *PETSC_RESTRICT p2;
  PetscScalar       *PETSC_RESTRICT rp, *PETSC_RESTRICT v, *PETSC_RESTRICT s, *PETSC_RESTRICT t, *PETSC_RESTRICT q, *PETSC_RESTRICT u;
  PetscScalar       insums[8],outsums[8];
  KSP_BCGS          *bcgs = (KSP_BCGS*)ksp->data;
  PC                pc;

  PetscFunctionBegin;
  if (!ksp->vec_rhs->petscnative) SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_SUP,"Only coded for PETSc vectors");
  ierr = VecGetLocalSize(ksp->vec_sol,&N);CHKERRQ(ierr);

  X   = ksp->vec_sol;
  B   = ksp->vec_rhs;

  /* lowercase variables used in vector updates */
  R   = ksp->work[0]; ierr = VecGetArray(R,(PetscScalar**)&r);CHKERRQ(ierr);
  V2  = ksp->work[1]; ierr = VecGetArray(V2,(PetscScalar**)&v2);CHKERRQ(ierr);
  S2  = ksp->work[2]; ierr = VecGetArray(S2,(PetscScalar**)&s2);CHKERRQ(ierr);
  T2  = ksp->work[3]; ierr = VecGetArray(T2,(PetscScalar**)&t2);CHKERRQ(ierr);
  P2  = ksp->work[4]; ierr = VecGetArray(P2,(PetscScalar**)&p2);CHKERRQ(ierr);

  /* lowercase variables used in inner product calculations */
  RP  = ksp->work[5]; ierr = VecGetArray(RP,(PetscScalar**)&rp);CHKERRQ(ierr);
  V   = ksp->work[6]; ierr = VecGetArray(V,(PetscScalar**)&v);CHKERRQ(ierr);
  S   = ksp->work[7]; ierr = VecGetArray(S,(PetscScalar**)&s);CHKERRQ(ierr);
  T   = ksp->work[8]; ierr = VecGetArray(T,(PetscScalar**)&t);CHKERRQ(ierr);
  Q   = ksp->work[9]; ierr = VecGetArray(Q,(PetscScalar**)&q);CHKERRQ(ierr);
  U   = ksp->work[10]; ierr = VecGetArray(U,(PetscScalar**)&u);CHKERRQ(ierr);

  /* Only supports right preconditioning */
  if (ksp->pc_side != PC_RIGHT) 
    SETERRQ1(((PetscObject)ksp)->comm,PETSC_ERR_SUP,"KSP ifbcgs does not support %s",PCSides[ksp->pc_side]);
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
  if (pc->setupcalled < 2) {
    ierr = PCSetUp(pc);CHKERRQ(ierr);
  }
  ierr = MatMult(pc->mat,X,P2);CHKERRQ(ierr); /* P2 is used as temporary storage */
  ierr = VecCopy(B,R);CHKERRQ(ierr);
  ierr = VecAXPY(R,-1.0,P2);CHKERRQ(ierr);

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
  if (pc->setupcalled < 2) {
    ierr = PCSetUp(pc);CHKERRQ(ierr);
  }
  ierr = PCApply(pc,R,P2);CHKERRQ(ierr); /* p2 <- K r */
  ierr = MatMult(pc->mat,P2,V);CHKERRQ(ierr); /* v <- A p2 */
  tau = rho*rho; /* tau <- (r,rp) */
  ierr = VecDot(V,RP,&sigma);CHKERRQ(ierr); /* sigma <- (v,rp) */
  /* rho has been computed previously */

  for (i=0; i<ksp->max_it; i++) {
    if (sigma == 0.0) SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_PLIB,"Divide by zero");

    /* scalar update */
    alpha = tau / sigma;

    /* vector update */
    ierr = VecWAXPY(S,-alpha,V,R);CHKERRQ(ierr);  /* s <- r - alpha v */

    /* matmult and pc */
    if (pc->setupcalled < 2) {
      ierr = PCSetUp(pc);CHKERRQ(ierr);
    }
    ierr = PCApply(pc,S,S2);CHKERRQ(ierr); /* s2 <- K s */
    ierr = MatMult(pc->mat,S2,T);CHKERRQ(ierr); /* t <- A s2 */
    ierr = PCApply(pc,T,T2);CHKERRQ(ierr); /* t2 <- K t */
    ierr = MatMult(pc->mat,T2,Q);CHKERRQ(ierr); /* q <- A t2 */
    ierr = PCApply(pc,V,V2);CHKERRQ(ierr); /* v2 <- K v */
    ierr = MatMult(pc->mat,V2,U);CHKERRQ(ierr); /* u <- A v2 */

    /* inner prodcuts */
    ierr = PetscLogEventBegin(VEC_ReduceArithmetic,0,0,0,0);CHKERRQ(ierr);
    xi1 = xi2 = xi3 = xi4 = xi5 = xi6 = xi7 = xi8 = 0.0;
    for (j=0; j<N; j++) {
      xi1 += s[j]*s[j]; /* xi1 <- (s,s) */
      xi2 += t[j]*s[j]; /* xi2 <- (t,s) */
      xi3 += t[j]*t[j]; /* xi3 <- (t,t) */
      xi4 += s[j]*rp[j]; /* xi4 <- (s,rp) */
      xi5 += t[j]*rp[j]; /* xi5 <- (t,rp) */
      xi6 += q[j]*rp[j]; /* xi6 <- (q,rp) */
      xi7 += v[j]*rp[j]; /* xi7 <- (v,rp) */
      xi8 += u[j]*rp[j]; /* xi8 <- (u,rp) */
    }
    PetscLogFlops(16.0*N);
    ierr = PetscLogEventEnd(VEC_ReduceArithmetic,0,0,0,0);CHKERRQ(ierr);
    insums[0] = xi1;
    insums[1] = xi2;
    insums[2] = xi3;
    insums[3] = xi4;
    insums[4] = xi5;
    insums[5] = xi6;
    insums[6] = xi7;
    insums[7] = xi8;
    ierr = PetscLogEventBarrierBegin(VEC_ReduceBarrier,0,0,0,0,((PetscObject)ksp)->comm);CHKERRQ(ierr);
    ierr = MPI_Allreduce(insums,outsums,8,MPIU_SCALAR,MPI_SUM,((PetscObject)ksp)->comm);CHKERRQ(ierr);
    ierr = PetscLogEventBarrierEnd(VEC_ReduceBarrier,0,0,0,0,((PetscObject)ksp)->comm);CHKERRQ(ierr);
    xi1 = outsums[0];
    xi2 = outsums[1];
    xi3 = outsums[2];
    xi4 = outsums[3];
    xi5 = outsums[4];
    xi6 = outsums[5];
    xi7 = outsums[6];
    xi8 = outsums[7];

    if (xi3 == 0.0) SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_PLIB,"Divide by zero");
    if (xi7 == 0.0) SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_PLIB,"Divide by zero");

    /* scalar updates */
    omega = xi2 / xi3;
    beta = - xi5 / xi7;
    rho = PetscSqrtReal(PetscAbsScalar(xi1 - omega * xi2)); /* residual norm */
    tau = xi4 - omega * xi5;
    sigma = xi5 - omega * xi6 + beta * (xi7 - omega * xi8);

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

  ierr = VecRestoreArray(R,(PetscScalar**)&r);CHKERRQ(ierr);
  ierr = VecRestoreArray(V2,(PetscScalar**)&v2);CHKERRQ(ierr);
  ierr = VecRestoreArray(S2,(PetscScalar**)&s2);CHKERRQ(ierr);
  ierr = VecRestoreArray(T2,(PetscScalar**)&t2);CHKERRQ(ierr);
  ierr = VecRestoreArray(P2,(PetscScalar**)&p2);CHKERRQ(ierr);

  ierr = VecRestoreArray(RP,(PetscScalar**)&rp);CHKERRQ(ierr);
  ierr = VecRestoreArray(V,(PetscScalar**)&v);CHKERRQ(ierr);
  ierr = VecRestoreArray(S,(PetscScalar**)&s);CHKERRQ(ierr);
  ierr = VecRestoreArray(T,(PetscScalar**)&t);CHKERRQ(ierr);
  ierr = VecRestoreArray(Q,(PetscScalar**)&q);CHKERRQ(ierr);
  ierr = VecRestoreArray(U,(PetscScalar**)&u);CHKERRQ(ierr);

  if (i >= ksp->max_it) {
    ksp->reason = KSP_DIVERGED_ITS;
  }

  PetscFunctionReturn(0);
}

/*MC
     KSPIFBCGS - Implements the improved flexible BiCGStab (Stabilized version of BiConjugate Gradient Squared) method.

   Options Database Keys:
.   see KSPSolve()

   Level: beginner

   Notes: See KSPIFBCGSL for additional stabilization
          Only supports right preconditioning 

.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP, KSPBICG, KSPFBCGSL, KSPSetPCSide()
M*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "KSPCreate_IFBCGS"
PetscErrorCode  KSPCreate_IFBCGS(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_BCGS       *bcgs;

  PetscFunctionBegin;
  ierr = PetscNewLog(ksp,KSP_BCGS,&bcgs);CHKERRQ(ierr);
  ksp->data                 = bcgs;
  ksp->ops->setup           = KSPSetUp_IFBCGS;
  ksp->ops->solve           = KSPSolve_IFBCGS;
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
