
#include <petsc/private/kspimpl.h>
#include <petsc/private/vecimpl.h>

static PetscErrorCode KSPSetUp_IBCGS(KSP ksp)
{
  PetscBool      diagonalscale;

  PetscFunctionBegin;
  PetscCall(PCGetDiagonalScale(ksp->pc,&diagonalscale));
  PetscCheck(!diagonalscale,PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"Krylov method %s does not support diagonal scaling",((PetscObject)ksp)->type_name);
  PetscCall(KSPSetWorkVecs(ksp,9));
  PetscFunctionReturn(0);
}

/*
    The code below "cheats" from PETSc style
       1) VecRestoreArray() is called immediately after VecGetArray() and the array values are still accessed; the reason for the immediate
          restore is that Vec operations are done on some of the vectors during the solve and if we did not restore immediately it would
          generate two VecGetArray() (the second one inside the Vec operation) calls without a restore between them.
       2) The vector operations on done directly on the arrays instead of with VecXXXX() calls

       For clarity in the code we name single VECTORS with two names, for example, Rn_1 and R, but they actually always
     the exact same memory. We do this with macro defines so that compiler won't think they are
     two different variables.

*/
#define Xn_1 Xn
#define xn_1 xn
#define Rn_1 Rn
#define rn_1 rn
#define Un_1 Un
#define un_1 un
#define Vn_1 Vn
#define vn_1 vn
#define Qn_1 Qn
#define qn_1 qn
#define Zn_1 Zn
#define zn_1 zn
static PetscErrorCode  KSPSolve_IBCGS(KSP ksp)
{
  PetscInt       i,N;
  PetscReal      rnorm = 0.0,rnormin = 0.0;
#if defined(PETSC_HAVE_MPI_LONG_DOUBLE) && !defined(PETSC_USE_COMPLEX) && (defined(PETSC_USE_REAL_SINGLE) || defined(PETSC_USE_REAL_DOUBLE))
  /* Because of possible instabilities in the algorithm (as indicated by different residual histories for the same problem
     on the same number of processes  with different runs) we support computing the inner products using Intel's 80 bit arithmetic
     rather than just 64 bit. Thus we copy our double precision values into long doubles (hoping this keeps the 16 extra bits)
     and tell MPI to do its ALlreduces with MPI_LONG_DOUBLE.

     Note for developers that does not effect the code. Intel's long double is implemented by storing the 80 bits of extended double
     precision into a 16 byte space (the rest of the space is ignored)  */
  long double insums[7],outsums[7];
#else
  PetscScalar insums[7],outsums[7];
#endif
  PetscScalar                       sigman_2, sigman_1, sigman, pin_1, pin, phin_1, phin,tmp1,tmp2;
  PetscScalar                       taun_1, taun, rhon, alphan_1, alphan, omegan_1, omegan;
  const PetscScalar *PETSC_RESTRICT r0, *PETSC_RESTRICT f0, *PETSC_RESTRICT qn, *PETSC_RESTRICT b, *PETSC_RESTRICT un;
  PetscScalar *PETSC_RESTRICT       rn, *PETSC_RESTRICT xn, *PETSC_RESTRICT vn, *PETSC_RESTRICT zn;
  /* the rest do not have to keep n_1 values */
  PetscScalar                       kappan, thetan, etan, gamman, betan, deltan;
  const PetscScalar *PETSC_RESTRICT tn;
  PetscScalar *PETSC_RESTRICT       sn;
  Vec                               R0,Rn,Xn,F0,Vn,Zn,Qn,Tn,Sn,B,Un;
  Mat                               A;

  PetscFunctionBegin;
  PetscCheck(ksp->vec_rhs->petscnative,PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"Only coded for PETSc vectors");

 #if defined(PETSC_HAVE_MPI_LONG_DOUBLE) && !defined(PETSC_USE_COMPLEX) && (defined(PETSC_USE_REAL_SINGLE) || defined(PETSC_USE_REAL_DOUBLE))
  /* since 80 bit long doubls do not fill the upper bits, we fill them initially so that
     valgrind won't detect MPI_Allreduce() with uninitialized data */
  PetscCall(PetscMemzero(insums,sizeof(insums)));
  PetscCall(PetscMemzero(insums,sizeof(insums)));
#endif

  PetscCall(PCGetOperators(ksp->pc,&A,NULL));
  PetscCall(VecGetLocalSize(ksp->vec_sol,&N));
  Xn   = ksp->vec_sol; PetscCall(VecGetArray(Xn_1,(PetscScalar**)&xn_1)); PetscCall(VecRestoreArray(Xn_1,NULL));
  B    = ksp->vec_rhs; PetscCall(VecGetArrayRead(B,(const PetscScalar**)&b)); PetscCall(VecRestoreArrayRead(B,NULL));
  R0   = ksp->work[0]; PetscCall(VecGetArrayRead(R0,(const PetscScalar**)&r0)); PetscCall(VecRestoreArrayRead(R0,NULL));
  Rn   = ksp->work[1]; PetscCall(VecGetArray(Rn_1,(PetscScalar**)&rn_1)); PetscCall(VecRestoreArray(Rn_1,NULL));
  Un   = ksp->work[2]; PetscCall(VecGetArrayRead(Un_1,(const PetscScalar**)&un_1)); PetscCall(VecRestoreArrayRead(Un_1,NULL));
  F0   = ksp->work[3]; PetscCall(VecGetArrayRead(F0,(const PetscScalar**)&f0)); PetscCall(VecRestoreArrayRead(F0,NULL));
  Vn   = ksp->work[4]; PetscCall(VecGetArray(Vn_1,(PetscScalar**)&vn_1)); PetscCall(VecRestoreArray(Vn_1,NULL));
  Zn   = ksp->work[5]; PetscCall(VecGetArray(Zn_1,(PetscScalar**)&zn_1)); PetscCall(VecRestoreArray(Zn_1,NULL));
  Qn   = ksp->work[6]; PetscCall(VecGetArrayRead(Qn_1,(const PetscScalar**)&qn_1)); PetscCall(VecRestoreArrayRead(Qn_1,NULL));
  Tn   = ksp->work[7]; PetscCall(VecGetArrayRead(Tn,(const PetscScalar**)&tn)); PetscCall(VecRestoreArrayRead(Tn,NULL));
  Sn   = ksp->work[8]; PetscCall(VecGetArrayRead(Sn,(const PetscScalar**)&sn)); PetscCall(VecRestoreArrayRead(Sn,NULL));

  /* r0 = rn_1 = b - A*xn_1; */
  /* PetscCall(KSP_PCApplyBAorAB(ksp,Xn_1,Rn_1,Tn));
     PetscCall(VecAYPX(Rn_1,-1.0,B)); */
  PetscCall(KSPInitialResidual(ksp,Xn_1,Tn,Sn,Rn_1,B));
  if (ksp->normtype != KSP_NORM_NONE) {
    PetscCall(VecNorm(Rn_1,NORM_2,&rnorm));
    KSPCheckNorm(ksp,rnorm);
  }
  PetscCall(KSPMonitor(ksp,0,rnorm));
  PetscCall((*ksp->converged)(ksp,0,rnorm,&ksp->reason,ksp->cnvP));
  if (ksp->reason) PetscFunctionReturn(0);

  PetscCall(VecCopy(Rn_1,R0));

  /* un_1 = A*rn_1; */
  PetscCall(KSP_PCApplyBAorAB(ksp,Rn_1,Un_1,Tn));

  /* f0   = A'*rn_1; */
  if (ksp->pc_side == PC_RIGHT) { /* B' A' */
    PetscCall(KSP_MatMultTranspose(ksp,A,R0,Tn));
    PetscCall(KSP_PCApplyTranspose(ksp,Tn,F0));
  } else if (ksp->pc_side == PC_LEFT) { /* A' B' */
    PetscCall(KSP_PCApplyTranspose(ksp,R0,Tn));
    PetscCall(KSP_MatMultTranspose(ksp,A,Tn,F0));
  }

  /*qn_1 = vn_1 = zn_1 = 0.0; */
  PetscCall(VecSet(Qn_1,0.0));
  PetscCall(VecSet(Vn_1,0.0));
  PetscCall(VecSet(Zn_1,0.0));

  sigman_2 = pin_1 = taun_1 = 0.0;

  /* the paper says phin_1 should be initialized to zero, it is actually R0'R0 */
  PetscCall(VecDot(R0,R0,&phin_1));
  KSPCheckDot(ksp,phin_1);

  /* sigman_1 = rn_1'un_1  */
  PetscCall(VecDot(R0,Un_1,&sigman_1));

  alphan_1 = omegan_1 = 1.0;

  for (ksp->its = 1; ksp->its<ksp->max_it+1; ksp->its++) {
    rhon = phin_1 - omegan_1*sigman_2 + omegan_1*alphan_1*pin_1;
    if (ksp->its == 1) deltan = rhon;
    else deltan = rhon/taun_1;
    betan = deltan/omegan_1;
    taun  = sigman_1 + betan*taun_1  - deltan*pin_1;
    if (taun == 0.0) {
      PetscCheck(!ksp->errorifnotconverged,PetscObjectComm((PetscObject)ksp),PETSC_ERR_NOT_CONVERGED,"KSPSolve has not converged due to taun is zero, iteration %" PetscInt_FMT,ksp->its);
      ksp->reason = KSP_DIVERGED_NANORINF;
      PetscFunctionReturn(0);
    }
    alphan = rhon/taun;
    PetscCall(PetscLogFlops(15.0));

    /*
        zn = alphan*rn_1 + (alphan/alphan_1)betan*zn_1 - alphan*deltan*vn_1
        vn = un_1 + betan*vn_1 - deltan*qn_1
        sn = rn_1 - alphan*vn

       The algorithm in the paper is missing the alphan/alphan_1 term in the zn update
    */
    PetscCall(PetscLogEventBegin(VEC_Ops,0,0,0,0));
    tmp1 = (alphan/alphan_1)*betan;
    tmp2 = alphan*deltan;
    for (i=0; i<N; i++) {
      zn[i] = alphan*rn_1[i] + tmp1*zn_1[i] - tmp2*vn_1[i];
      vn[i] = un_1[i] + betan*vn_1[i] - deltan*qn_1[i];
      sn[i] = rn_1[i] - alphan*vn[i];
    }
    PetscCall(PetscLogFlops(3.0+11.0*N));
    PetscCall(PetscLogEventEnd(VEC_Ops,0,0,0,0));

    /*
        qn = A*vn
    */
    PetscCall(KSP_PCApplyBAorAB(ksp,Vn,Qn,Tn));

    /*
        tn = un_1 - alphan*qn
    */
    PetscCall(VecWAXPY(Tn,-alphan,Qn,Un_1));

    /*
        phin = r0'sn
        pin  = r0'qn
        gamman = f0'sn
        etan   = f0'tn
        thetan = sn'tn
        kappan = tn'tn
    */
    PetscCall(PetscLogEventBegin(VEC_ReduceArithmetic,0,0,0,0));
    phin = pin = gamman = etan = thetan = kappan = 0.0;
    for (i=0; i<N; i++) {
      phin   += r0[i]*sn[i];
      pin    += r0[i]*qn[i];
      gamman += f0[i]*sn[i];
      etan   += f0[i]*tn[i];
      thetan += sn[i]*tn[i];
      kappan += tn[i]*tn[i];
    }
    PetscCall(PetscLogFlops(12.0*N));
    PetscCall(PetscLogEventEnd(VEC_ReduceArithmetic,0,0,0,0));

    insums[0] = phin;
    insums[1] = pin;
    insums[2] = gamman;
    insums[3] = etan;
    insums[4] = thetan;
    insums[5] = kappan;
    insums[6] = rnormin;

    PetscCall(PetscLogEventBegin(VEC_ReduceCommunication,0,0,0,0));
#if defined(PETSC_HAVE_MPI_LONG_DOUBLE) && !defined(PETSC_USE_COMPLEX) && (defined(PETSC_USE_REAL_SINGLE) || defined(PETSC_USE_REAL_DOUBLE))
    if (ksp->lagnorm && ksp->its > 1) {
      PetscCall(MPIU_Allreduce(insums,outsums,7,MPI_LONG_DOUBLE,MPI_SUM,PetscObjectComm((PetscObject)ksp)));
    } else {
      PetscCall(MPIU_Allreduce(insums,outsums,6,MPI_LONG_DOUBLE,MPI_SUM,PetscObjectComm((PetscObject)ksp)));
    }
#else
    if (ksp->lagnorm && ksp->its > 1 && ksp->normtype != KSP_NORM_NONE) {
      PetscCall(MPIU_Allreduce(insums,outsums,7,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)ksp)));
    } else {
      PetscCall(MPIU_Allreduce(insums,outsums,6,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)ksp)));
    }
#endif
    PetscCall(PetscLogEventEnd(VEC_ReduceCommunication,0,0,0,0));
    phin   = outsums[0];
    pin    = outsums[1];
    gamman = outsums[2];
    etan   = outsums[3];
    thetan = outsums[4];
    kappan = outsums[5];
    if (ksp->lagnorm && ksp->its > 1 && ksp->normtype != KSP_NORM_NONE) rnorm = PetscSqrtReal(PetscRealPart(outsums[6]));

    if (kappan == 0.0) {
      PetscCheck(!ksp->errorifnotconverged,PetscObjectComm((PetscObject)ksp),PETSC_ERR_NOT_CONVERGED,"KSPSolve has not converged due to kappan is zero, iteration %" PetscInt_FMT,ksp->its);
      ksp->reason = KSP_DIVERGED_NANORINF;
      PetscFunctionReturn(0);
    }
    if (thetan == 0.0) {
      PetscCheck(!ksp->errorifnotconverged,PetscObjectComm((PetscObject)ksp),PETSC_ERR_NOT_CONVERGED,"KSPSolve has not converged due to thetan is zero, iteration %" PetscInt_FMT,ksp->its);
      ksp->reason = KSP_DIVERGED_NANORINF;
      PetscFunctionReturn(0);
    }
    omegan = thetan/kappan;
    sigman = gamman - omegan*etan;

    /*
        rn = sn - omegan*tn
        xn = xn_1 + zn + omegan*sn
    */
    PetscCall(PetscLogEventBegin(VEC_Ops,0,0,0,0));
    rnormin = 0.0;
    for (i=0; i<N; i++) {
      rn[i]    = sn[i] - omegan*tn[i];
      rnormin += PetscRealPart(PetscConj(rn[i])*rn[i]);
      xn[i]   += zn[i] + omegan*sn[i];
    }
    PetscCall(PetscObjectStateIncrease((PetscObject)Xn));
    PetscCall(PetscLogFlops(7.0*N));
    PetscCall(PetscLogEventEnd(VEC_Ops,0,0,0,0));

    if (!ksp->lagnorm && ksp->chknorm < ksp->its && ksp->normtype != KSP_NORM_NONE) {
      PetscCall(PetscLogEventBegin(VEC_ReduceCommunication,0,0,0,0));
      PetscCall(MPIU_Allreduce(&rnormin,&rnorm,1,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)ksp)));
      PetscCall(PetscLogEventEnd(VEC_ReduceCommunication,0,0,0,0));
      rnorm = PetscSqrtReal(rnorm);
    }

    /* Test for convergence */
    PetscCall(KSPMonitor(ksp,ksp->its,rnorm));
    PetscCall((*ksp->converged)(ksp,ksp->its,rnorm,&ksp->reason,ksp->cnvP));
    if (ksp->reason) {
      PetscCall(KSPUnwindPreconditioner(ksp,Xn,Tn));
      PetscFunctionReturn(0);
    }

    /* un = A*rn */
    PetscCall(KSP_PCApplyBAorAB(ksp,Rn,Un,Tn));

    /* Update n-1 locations with n locations */
    sigman_2 = sigman_1;
    sigman_1 = sigman;
    pin_1    = pin;
    phin_1   = phin;
    alphan_1 = alphan;
    taun_1   = taun;
    omegan_1 = omegan;
  }
  if (ksp->its >= ksp->max_it) ksp->reason = KSP_DIVERGED_ITS;
  PetscCall(KSPUnwindPreconditioner(ksp,Xn,Tn));
  PetscFunctionReturn(0);
}

/*MC
     KSPIBCGS - Implements the IBiCGStab (Improved Stabilized version of BiConjugate Gradient) method
            in an alternative form to have only a single global reduction operation instead of the usual 3 (or 4)

   Options Database Keys:
    see KSPSolve()

   Level: beginner

   Notes:
    Supports left and right preconditioning

          See KSPBCGSL for additional stabilization

          Unlike the Bi-CG-stab algorithm, this requires one multiplication be the transpose of the operator
           before the iteration starts.

          The paper has two errors in the algorithm presented, they are fixed in the code in KSPSolve_IBCGS()

          For maximum reduction in the number of global reduction operations, this solver should be used with
          KSPSetLagNorm().

          This is not supported for complex numbers.

   Reference: The Improved BiCGStab Method for Large and Sparse Unsymmetric Linear Systems on Parallel Distributed Memory
                     Architectures. L. T. Yang and R. Brent, Proceedings of the Fifth International Conference on Algorithms and
                     Architectures for Parallel Processing, 2002, IEEE.

.seealso: `KSPCreate()`, `KSPSetType()`, `KSPType`, `KSP`, `KSPBICG`, `KSPBCGSL`, `KSPIBCGS`, `KSPSetLagNorm()`
M*/

PETSC_EXTERN PetscErrorCode KSPCreate_IBCGS(KSP ksp)
{
  PetscFunctionBegin;

  PetscCall(KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,3));
  PetscCall(KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_RIGHT,2));
  PetscCall(KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_RIGHT,1));

  ksp->ops->setup          = KSPSetUp_IBCGS;
  ksp->ops->solve          = KSPSolve_IBCGS;
  ksp->ops->destroy        = KSPDestroyDefault;
  ksp->ops->buildsolution  = KSPBuildSolutionDefault;
  ksp->ops->buildresidual  = KSPBuildResidualDefault;
  ksp->ops->setfromoptions = NULL;
  ksp->ops->view           = NULL;
#if defined(PETSC_USE_COMPLEX)
  SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"This is not supported for complex numbers");
#else
  PetscFunctionReturn(0);
#endif
}
