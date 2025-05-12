#define PETSC_SKIP_CXX_COMPLEX_FIX // Kokkos::complex does not need the PetscComplex fix

#include <petsc/private/pcbjkokkosimpl.h>

#include <petsc/private/kspimpl.h>
#include <petscksp.h> /*I "petscksp.h" I*/
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <../src/mat/impls/aij/seq/kokkos/aijkok.hpp>
#include <petscsection.h>
#include <petscdmcomposite.h>

#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/seq/kokkos/aijkok.hpp>

#include <petscdevice_cupm.h>

static PetscErrorCode PCBJKOKKOSCreateKSP_BJKOKKOS(PC pc)
{
  const char    *prefix;
  PC_PCBJKOKKOS *jac = (PC_PCBJKOKKOS *)pc->data;
  DM             dm;

  PetscFunctionBegin;
  PetscCall(KSPCreate(PetscObjectComm((PetscObject)pc), &jac->ksp));
  PetscCall(KSPSetNestLevel(jac->ksp, pc->kspnestlevel));
  PetscCall(KSPSetErrorIfNotConverged(jac->ksp, pc->erroriffailure));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)jac->ksp, (PetscObject)pc, 1));
  PetscCall(PCGetOptionsPrefix(pc, &prefix));
  PetscCall(KSPSetOptionsPrefix(jac->ksp, prefix));
  PetscCall(KSPAppendOptionsPrefix(jac->ksp, "pc_bjkokkos_"));
  PetscCall(PCGetDM(pc, &dm));
  if (dm) {
    PetscCall(KSPSetDM(jac->ksp, dm));
    PetscCall(KSPSetDMActive(jac->ksp, PETSC_FALSE));
  }
  jac->reason       = PETSC_FALSE;
  jac->monitor      = PETSC_FALSE;
  jac->batch_target = 0;
  jac->rank_target  = 0;
  jac->nsolves_team = 1;
  jac->ksp->max_it  = 50; // this is really for GMRES w/o restarts
  PetscFunctionReturn(PETSC_SUCCESS);
}

// y <-- Ax
KOKKOS_INLINE_FUNCTION PetscErrorCode MatMult(const team_member team, const PetscInt *glb_Aai, const PetscInt *glb_Aaj, const PetscScalar *glb_Aaa, const PetscInt *r, const PetscInt *ic, const PetscInt start, const PetscInt end, const PetscScalar *x_loc, PetscScalar *y_loc)
{
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, start, end), [=](const int rowb) {
    int                rowa = ic[rowb];
    int                n    = glb_Aai[rowa + 1] - glb_Aai[rowa];
    const PetscInt    *aj   = glb_Aaj + glb_Aai[rowa]; // global
    const PetscScalar *aa   = glb_Aaa + glb_Aai[rowa];
    PetscScalar        sum;
    Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team, n), [=](const int i, PetscScalar &lsum) { lsum += aa[i] * x_loc[r[aj[i]] - start]; }, sum);
    Kokkos::single(Kokkos::PerThread(team), [=]() { y_loc[rowb - start] = sum; });
  });
  team.team_barrier();
  return PETSC_SUCCESS;
}

// temp buffer per thread with reduction at end?
KOKKOS_INLINE_FUNCTION PetscErrorCode MatMultTranspose(const team_member team, const PetscInt *glb_Aai, const PetscInt *glb_Aaj, const PetscScalar *glb_Aaa, const PetscInt *r, const PetscInt *ic, const PetscInt start, const PetscInt end, const PetscScalar *x_loc, PetscScalar *y_loc)
{
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, end - start), [=](int i) { y_loc[i] = 0; });
  team.team_barrier();
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, start, end), [=](const int rowb) {
    int                rowa = ic[rowb];
    int                n    = glb_Aai[rowa + 1] - glb_Aai[rowa];
    const PetscInt    *aj   = glb_Aaj + glb_Aai[rowa]; // global
    const PetscScalar *aa   = glb_Aaa + glb_Aai[rowa];
    const PetscScalar  xx   = x_loc[rowb - start]; // rowb = ic[rowa] = ic[r[rowb]]
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, n), [=](const int &i) {
      PetscScalar val = aa[i] * xx;
      Kokkos::atomic_fetch_add(&y_loc[r[aj[i]] - start], val);
    });
  });
  team.team_barrier();
  return PETSC_SUCCESS;
}

typedef struct Batch_MetaData_TAG {
  PetscInt           flops;
  PetscInt           its;
  KSPConvergedReason reason;
} Batch_MetaData;

// Solve A(BB^-1)x = y with TFQMR. Right preconditioned to get un-preconditioned residual
static KOKKOS_INLINE_FUNCTION PetscErrorCode BJSolve_TFQMR(const team_member team, const PetscInt *glb_Aai, const PetscInt *glb_Aaj, const PetscScalar *glb_Aaa, const PetscInt *r, const PetscInt *ic, PetscScalar *work_space_global, const int stride_global, const int nShareVec, PetscScalar *work_space_shared, const int stride_shared, PetscReal rtol, PetscReal atol, PetscReal dtol, PetscInt maxit, Batch_MetaData *metad, const PetscInt start, const PetscInt end, const PetscScalar glb_idiag[], const PetscScalar *glb_b, PetscScalar *glb_x, bool monitor)
{
  using Kokkos::parallel_for;
  using Kokkos::parallel_reduce;
  int                Nblk = end - start, it, m, stride = stride_shared, idx = 0;
  PetscReal          dp, dpold, w, dpest, tau, psi, cm, r0;
  const PetscScalar *Diag = &glb_idiag[start];
  PetscScalar       *ptr  = work_space_shared, rho, rhoold, a, s, b, eta, etaold, psiold, cf, dpi;

  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *XX = ptr;
  ptr += stride;
  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *R = ptr;
  ptr += stride;
  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *RP = ptr;
  ptr += stride;
  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *V = ptr;
  ptr += stride;
  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *T = ptr;
  ptr += stride;
  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *Q = ptr;
  ptr += stride;
  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *P = ptr;
  ptr += stride;
  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *U = ptr;
  ptr += stride;
  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *D = ptr;
  ptr += stride;
  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *AUQ = V;

  // init: get b, zero x
  parallel_for(Kokkos::TeamVectorRange(team, start, end), [=](int rowb) {
    int rowa         = ic[rowb];
    R[rowb - start]  = glb_b[rowa];
    XX[rowb - start] = 0;
  });
  team.team_barrier();
  parallel_reduce(Kokkos::TeamVectorRange(team, Nblk), [=](const int idx, PetscScalar &lsum) { lsum += R[idx] * PetscConj(R[idx]); }, dpi);
  team.team_barrier();
  r0 = dp = PetscSqrtReal(PetscRealPart(dpi));
  // diagnostics
#if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_SYCL)
  if (monitor) Kokkos::single(Kokkos::PerTeam(team), [=]() { printf("%3d KSP Residual norm %14.12e\n", 0, (double)dp); });
#endif
  if (dp < atol) {
    metad->reason = KSP_CONVERGED_ATOL_NORMAL_EQUATIONS;
    it            = 0;
    goto done;
  }
  if (0 == maxit) {
    metad->reason = KSP_CONVERGED_ITS;
    it            = 0;
    goto done;
  }

  /* Make the initial Rp = R */
  parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int idx) { RP[idx] = R[idx]; });
  team.team_barrier();
  /* Set the initial conditions */
  etaold = 0.0;
  psiold = 0.0;
  tau    = dp;
  dpold  = dp;

  /* rhoold = (r,rp)     */
  parallel_reduce(Kokkos::TeamVectorRange(team, Nblk), [=](const int idx, PetscScalar &dot) { dot += R[idx] * PetscConj(RP[idx]); }, rhoold);
  team.team_barrier();
  parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int idx) {
    U[idx] = R[idx];
    P[idx] = R[idx];
    T[idx] = Diag[idx] * P[idx];
    D[idx] = 0;
  });
  team.team_barrier();
  static_cast<void>(MatMult(team, glb_Aai, glb_Aaj, glb_Aaa, r, ic, start, end, T, V));

  it = 0;
  do {
    /* s <- (v,rp)          */
    parallel_reduce(Kokkos::TeamVectorRange(team, Nblk), [=](const int idx, PetscScalar &dot) { dot += V[idx] * PetscConj(RP[idx]); }, s);
    team.team_barrier();
    if (s == 0) {
      metad->reason = KSP_CONVERGED_HAPPY_BREAKDOWN;
      goto done;
    }
    a = rhoold / s; /* a <- rho / s         */
    /* q <- u - a v    VecWAXPY(w,alpha,x,y): w = alpha x + y.     */
    /* t <- u + q           */
    parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int idx) {
      Q[idx] = U[idx] - a * V[idx];
      T[idx] = U[idx] + Q[idx];
    });
    team.team_barrier();
    // KSP_PCApplyBAorAB
    parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int idx) { T[idx] = Diag[idx] * T[idx]; });
    team.team_barrier();
    static_cast<void>(MatMult(team, glb_Aai, glb_Aaj, glb_Aaa, r, ic, start, end, T, AUQ));
    /* r <- r - a K (u + q) */
    parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int idx) { R[idx] = R[idx] - a * AUQ[idx]; });
    team.team_barrier();
    parallel_reduce(Kokkos::TeamVectorRange(team, Nblk), [=](const int idx, PetscScalar &lsum) { lsum += R[idx] * PetscConj(R[idx]); }, dpi);
    team.team_barrier();
    dp = PetscSqrtReal(PetscRealPart(dpi));
    for (m = 0; m < 2; m++) {
      if (!m) w = PetscSqrtReal(dp * dpold);
      else w = dp;
      psi = w / tau;
      cm  = 1.0 / PetscSqrtReal(1.0 + psi * psi);
      tau = tau * psi * cm;
      eta = cm * cm * a;
      cf  = psiold * psiold * etaold / a;
      if (!m) {
        /* D = U + cf D */
        parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int idx) { D[idx] = U[idx] + cf * D[idx]; });
      } else {
        /* D = Q + cf D */
        parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int idx) { D[idx] = Q[idx] + cf * D[idx]; });
      }
      team.team_barrier();
      parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int idx) { XX[idx] = XX[idx] + eta * D[idx]; });
      team.team_barrier();
      dpest = PetscSqrtReal(2 * it + m + 2.0) * tau;
#if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_SYCL)
      if (monitor && m == 1) Kokkos::single(Kokkos::PerTeam(team), [=]() { printf("%3d KSP Residual norm %14.12e\n", it + 1, (double)dpest); });
#endif
      if (dpest < atol) {
        metad->reason = KSP_CONVERGED_ATOL_NORMAL_EQUATIONS;
        goto done;
      }
      if (dpest / r0 < rtol) {
        metad->reason = KSP_CONVERGED_RTOL_NORMAL_EQUATIONS;
        goto done;
      }
#if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_SYCL)
      if (dpest / r0 > dtol) {
        metad->reason = KSP_DIVERGED_DTOL;
        Kokkos::single(Kokkos::PerTeam(team), [=]() { printf("ERROR block %d diverged: %d it, res=%e, r_0=%e\n", team.league_rank(), it, dpest, r0); });
        goto done;
      }
#else
      if (dpest / r0 > dtol) {
        metad->reason = KSP_DIVERGED_DTOL;
        goto done;
      }
#endif
      if (it + 1 == maxit) {
        metad->reason = KSP_CONVERGED_ITS;
#if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_SYCL)
        Kokkos::single(Kokkos::PerTeam(team), [=]() { printf("ERROR block %d diverged: TFQMR %d:%d it, res=%e, r_0=%e r_res=%e\n", team.league_rank(), it, m, dpest, r0, dpest / r0); });
#endif
        goto done;
      }
      etaold = eta;
      psiold = psi;
    }

    /* rho <- (r,rp)       */
    parallel_reduce(Kokkos::TeamVectorRange(team, Nblk), [=](const int idx, PetscScalar &dot) { dot += R[idx] * PetscConj(RP[idx]); }, rho);
    team.team_barrier();
    if (rho == 0) {
      metad->reason = KSP_CONVERGED_HAPPY_BREAKDOWN;
      goto done;
    }
    b = rho / rhoold; /* b <- rho / rhoold   */
    /* u <- r + b q        */
    /* p <- u + b(q + b p) */
    parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int idx) {
      U[idx] = R[idx] + b * Q[idx];
      Q[idx] = Q[idx] + b * P[idx];
      P[idx] = U[idx] + b * Q[idx];
    });
    /* v <- K p  */
    team.team_barrier();
    parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int idx) { T[idx] = Diag[idx] * P[idx]; });
    team.team_barrier();
    static_cast<void>(MatMult(team, glb_Aai, glb_Aaj, glb_Aaa, r, ic, start, end, T, V));

    rhoold = rho;
    dpold  = dp;

    it++;
  } while (it < maxit);
done:
  // KSPUnwindPreconditioner
  parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int idx) { XX[idx] = Diag[idx] * XX[idx]; });
  team.team_barrier();
  // put x into Plex order
  parallel_for(Kokkos::TeamVectorRange(team, start, end), [=](int rowb) {
    int rowa    = ic[rowb];
    glb_x[rowa] = XX[rowb - start];
  });
  metad->its = it;
  if (1) {
    int nnz;
    parallel_reduce(Kokkos::TeamVectorRange(team, start, end), [=](const int idx, int &lsum) { lsum += (glb_Aai[idx + 1] - glb_Aai[idx]); }, nnz);
    metad->flops = 2 * (metad->its * (10 * Nblk + 2 * nnz) + 5 * Nblk);
  } else {
    metad->flops = 2 * (metad->its * (10 * Nblk + 2 * 50 * Nblk) + 5 * Nblk); // guess
  }
  return PETSC_SUCCESS;
}

// Solve Ax = y with biCG
static KOKKOS_INLINE_FUNCTION PetscErrorCode BJSolve_BICG(const team_member team, const PetscInt *glb_Aai, const PetscInt *glb_Aaj, const PetscScalar *glb_Aaa, const PetscInt *r, const PetscInt *ic, PetscScalar *work_space_global, const int stride_global, const int nShareVec, PetscScalar *work_space_shared, const int stride_shared, PetscReal rtol, PetscReal atol, PetscReal dtol, PetscInt maxit, Batch_MetaData *metad, const PetscInt start, const PetscInt end, const PetscScalar glb_idiag[], const PetscScalar *glb_b, PetscScalar *glb_x, bool monitor)
{
  using Kokkos::parallel_for;
  using Kokkos::parallel_reduce;
  int                Nblk = end - start, it, stride = stride_shared, idx = 0; // start in shared mem
  PetscReal          dp, r0;
  const PetscScalar *Di  = &glb_idiag[start];
  PetscScalar       *ptr = work_space_shared, dpi, a = 1.0, beta, betaold = 1.0, t1, t2;

  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *XX = ptr;
  ptr += stride;
  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *Rl = ptr;
  ptr += stride;
  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *Zl = ptr;
  ptr += stride;
  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *Pl = ptr;
  ptr += stride;
  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *Rr = ptr;
  ptr += stride;
  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *Zr = ptr;
  ptr += stride;
  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *Pr = ptr;
  ptr += stride;

  /*     r <- b (x is 0) */
  parallel_for(Kokkos::TeamVectorRange(team, start, end), [=](int rowb) {
    int rowa         = ic[rowb];
    Rl[rowb - start] = Rr[rowb - start] = glb_b[rowa];
    XX[rowb - start]                    = 0;
  });
  team.team_barrier();
  /*     z <- Br         */
  parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int idx) {
    Zr[idx] = Di[idx] * Rr[idx];
    Zl[idx] = Di[idx] * Rl[idx];
  });
  team.team_barrier();
  /*    dp <- r'*r       */
  parallel_reduce(Kokkos::TeamVectorRange(team, Nblk), [=](const int idx, PetscScalar &lsum) { lsum += Rr[idx] * PetscConj(Rr[idx]); }, dpi);
  team.team_barrier();
  r0 = dp = PetscSqrtReal(PetscRealPart(dpi));
#if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_SYCL)
  if (monitor) Kokkos::single(Kokkos::PerTeam(team), [=]() { printf("%3d KSP Residual norm %14.12e\n", 0, (double)dp); });
#endif
  if (dp < atol) {
    metad->reason = KSP_CONVERGED_ATOL_NORMAL_EQUATIONS;
    it            = 0;
    goto done;
  }
  if (0 == maxit) {
    metad->reason = KSP_CONVERGED_ITS;
    it            = 0;
    goto done;
  }

  it = 0;
  do {
    /*     beta <- r'z     */
    parallel_reduce(Kokkos::TeamVectorRange(team, Nblk), [=](const int idx, PetscScalar &dot) { dot += Zr[idx] * PetscConj(Rl[idx]); }, beta);
    team.team_barrier();
#if PCBJKOKKOS_VERBOSE_LEVEL >= 6
  #if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_SYCL)
    Kokkos::single(Kokkos::PerTeam(team), [=]() { printf("%7d beta = Z.R = %22.14e \n", i, (double)beta); });
  #endif
#endif
    if (beta == 0.0) {
      metad->reason = KSP_CONVERGED_HAPPY_BREAKDOWN;
      goto done;
    }
    if (it == 0) {
      /*     p <- z          */
      parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int idx) {
        Pr[idx] = Zr[idx];
        Pl[idx] = Zl[idx];
      });
    } else {
      t1 = beta / betaold;
      /*     p <- z + b* p   */
      t2 = PetscConj(t1);
      parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int idx) {
        Pr[idx] = t1 * Pr[idx] + Zr[idx];
        Pl[idx] = t2 * Pl[idx] + Zl[idx];
      });
    }
    team.team_barrier();
    betaold = beta;
    /*     z <- Kp         */
    static_cast<void>(MatMult(team, glb_Aai, glb_Aaj, glb_Aaa, r, ic, start, end, Pr, Zr));
    static_cast<void>(MatMultTranspose(team, glb_Aai, glb_Aaj, glb_Aaa, r, ic, start, end, Pl, Zl));
    /*     dpi <- z'p      */
    parallel_reduce(Kokkos::TeamVectorRange(team, Nblk), [=](const int idx, PetscScalar &lsum) { lsum += Zr[idx] * PetscConj(Pl[idx]); }, dpi);
    team.team_barrier();
    if (dpi == 0) {
      metad->reason = KSP_CONVERGED_HAPPY_BREAKDOWN;
      goto done;
    }
    //
    a  = beta / dpi; /*     a = beta/p'z    */
    t1 = -a;
    t2 = PetscConj(t1);
    /*     x <- x + ap     */
    parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int idx) {
      XX[idx] = XX[idx] + a * Pr[idx];
      Rr[idx] = Rr[idx] + t1 * Zr[idx];
      Rl[idx] = Rl[idx] + t2 * Zl[idx];
    });
    team.team_barrier();
    team.team_barrier();
    /*    dp <- r'*r       */
    parallel_reduce(Kokkos::TeamVectorRange(team, Nblk), [=](const int idx, PetscScalar &lsum) { lsum += Rr[idx] * PetscConj(Rr[idx]); }, dpi);
    team.team_barrier();
    dp = PetscSqrtReal(PetscRealPart(dpi));
#if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_SYCL)
    if (monitor) Kokkos::single(Kokkos::PerTeam(team), [=]() { printf("%3d KSP Residual norm %14.12e\n", it + 1, (double)dp); });
#endif
    if (dp < atol) {
      metad->reason = KSP_CONVERGED_ATOL_NORMAL_EQUATIONS;
      goto done;
    }
    if (dp / r0 < rtol) {
      metad->reason = KSP_CONVERGED_RTOL_NORMAL_EQUATIONS;
      goto done;
    }
#if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_SYCL)
    if (dp / r0 > dtol) {
      metad->reason = KSP_DIVERGED_DTOL;
      Kokkos::single(Kokkos::PerTeam(team), [=]() { printf("ERROR block %d diverged: %d it, res=%e, r_0=%e (BICG does this)\n", team.league_rank(), it, dp, r0); });
      goto done;
    }
#else
    if (dp / r0 > dtol) {
      metad->reason = KSP_DIVERGED_DTOL;
      goto done;
    }
#endif
    if (it + 1 == maxit) {
      metad->reason = KSP_CONVERGED_ITS; // don't worry about hitting max iterations
#if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_SYCL)
      Kokkos::single(Kokkos::PerTeam(team), [=]() { printf("ERROR block %d diverged: BICG %d it, res=%e, r_0=%e r_res=%e\n", team.league_rank(), it, dp, r0, dp / r0); });
#endif
      goto done;
    }
    /* z <- Br  */
    parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int idx) {
      Zr[idx] = Di[idx] * Rr[idx];
      Zl[idx] = Di[idx] * Rl[idx];
    });

    it++;
  } while (it < maxit);
done:
  // put x back into Plex order
  parallel_for(Kokkos::TeamVectorRange(team, start, end), [=](int rowb) {
    int rowa    = ic[rowb];
    glb_x[rowa] = XX[rowb - start];
  });
  metad->its = it;
  if (1) {
    int nnz;
    parallel_reduce(Kokkos::TeamVectorRange(team, start, end), [=](const int idx, int &lsum) { lsum += (glb_Aai[idx + 1] - glb_Aai[idx]); }, nnz);
    metad->flops = 2 * (metad->its * (10 * Nblk + 2 * nnz) + 5 * Nblk);
  } else {
    metad->flops = 2 * (metad->its * (10 * Nblk + 2 * 50 * Nblk) + 5 * Nblk); // guess
  }
  return PETSC_SUCCESS;
}

// KSP solver solve Ax = b; xout is output, bin is input
static PetscErrorCode PCApply_BJKOKKOS(PC pc, Vec bin, Vec xout)
{
  PC_PCBJKOKKOS *jac = (PC_PCBJKOKKOS *)pc->data;
  Mat            A = pc->pmat, Aseq = A;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)A), &rank));
  if (!A->spptr) Aseq = ((Mat_MPIAIJ *)A->data)->A; // MPI
  PetscCall(MatSeqAIJKokkosSyncDevice(Aseq));
  {
    PetscInt           maxit = jac->ksp->max_it;
    const PetscInt     conc = Kokkos::DefaultExecutionSpace().concurrency(), openmp = !!(conc < 1000), team_size = (openmp == 0 && PCBJKOKKOS_VEC_SIZE != 1) ? PCBJKOKKOS_TEAM_SIZE : 1;
    const PetscInt     nwork = jac->nwork, nBlk = jac->nBlocks;
    PetscScalar       *glb_xdata = NULL, *dummy;
    PetscReal          rtol = jac->ksp->rtol, atol = jac->ksp->abstol, dtol = jac->ksp->divtol;
    const PetscScalar *glb_idiag = jac->d_idiag_k->data(), *glb_bdata = NULL;
    const PetscInt    *glb_Aai, *glb_Aaj, *d_bid_eqOffset = jac->d_bid_eqOffset_k->data();
    const PetscScalar *glb_Aaa;
    const PetscInt    *d_isicol = jac->d_isicol_k->data(), *d_isrow = jac->d_isrow_k->data();
    PCFailedReason     pcreason;
    KSPIndex           ksp_type_idx = jac->ksp_type_idx;
    PetscMemType       mtype;
    PetscContainer     container;
    PetscInt           batch_sz;                // the number of repeated DMs, [DM_e_1, DM_e_2, DM_e_batch_sz, DM_i_1, ...]
    VecScatter         plex_batch = NULL;       // not used
    Vec                bvec;                    // a copy of b for scatter (just alias to bin now)
    PetscBool          monitor  = jac->monitor; // captured
    PetscInt           view_bid = jac->batch_target;
    MatInfo            info;

    PetscCall(MatSeqAIJGetCSRAndMemType(Aseq, &glb_Aai, &glb_Aaj, &dummy, &mtype));
    jac->max_nits = 0;
    glb_Aaa       = dummy;
    if (jac->rank_target != rank) view_bid = -1; // turn off all but one process
    PetscCall(MatGetInfo(A, MAT_LOCAL, &info));
    // get field major is to map plex IO to/from block/field major
    PetscCall(PetscObjectQuery((PetscObject)A, "plex_batch_is", (PetscObject *)&container));
    if (container) {
      PetscCall(VecDuplicate(bin, &bvec));
      PetscCall(PetscContainerGetPointer(container, (void **)&plex_batch));
      PetscCall(VecScatterBegin(plex_batch, bin, bvec, INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecScatterEnd(plex_batch, bin, bvec, INSERT_VALUES, SCATTER_FORWARD));
      SETERRQ(PetscObjectComm((PetscObject)A), PETSC_ERR_USER, "No plex_batch_is -- require NO field major ordering for now");
    } else {
      bvec = bin;
    }
    // get x
    PetscCall(VecGetArrayAndMemType(xout, &glb_xdata, &mtype));
#if defined(PETSC_HAVE_CUDA)
    PetscCheck(PetscMemTypeDevice(mtype), PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONG, "No GPU data for x %d != %d", static_cast<int>(mtype), static_cast<int>(PETSC_MEMTYPE_DEVICE));
#endif
    PetscCall(VecGetArrayReadAndMemType(bvec, &glb_bdata, &mtype));
#if defined(PETSC_HAVE_CUDA)
    PetscCheck(PetscMemTypeDevice(mtype), PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONG, "No GPU data for b");
#endif
    // get batch size
    PetscCall(PetscObjectQuery((PetscObject)A, "batch size", (PetscObject *)&container));
    if (container) {
      PetscInt *pNf = NULL;
      PetscCall(PetscContainerGetPointer(container, (void **)&pNf));
      batch_sz = *pNf; // number of times to repeat the DMs
    } else batch_sz = 1;
    PetscCheck(nBlk % batch_sz == 0, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONG, "batch_sz = %" PetscInt_FMT ", nBlk = %" PetscInt_FMT, batch_sz, nBlk);
    if (ksp_type_idx == BATCH_KSP_GMRESKK_IDX) {
      // KK solver - move PETSc data into Kokkos Views, setup solver, solve, move data out of Kokkos, process metadata (convergence tests, etc.)
#if defined(PETSC_HAVE_KOKKOS_KERNELS_BATCH)
      PetscCall(PCApply_BJKOKKOSKERNELS(pc, glb_bdata, glb_xdata, glb_Aai, glb_Aaj, glb_Aaa, team_size, info, batch_sz, &pcreason));
#else
      PetscCheck(ksp_type_idx != BATCH_KSP_GMRESKK_IDX, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONG, "Type: BATCH_KSP_GMRES not supported for complex");
#endif
    } else { // Kokkos Krylov
      using scr_mem_t    = Kokkos::DefaultExecutionSpace::scratch_memory_space;
      using vect2D_scr_t = Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, scr_mem_t>;
      Kokkos::View<Batch_MetaData *, Kokkos::DefaultExecutionSpace> d_metadata("solver meta data", nBlk);
      int                                                           stride_shared, stride_global, global_buff_words;
      d_bid_eqOffset = jac->d_bid_eqOffset_k->data();
      // solve each block independently
      int scr_bytes_team_shared = 0, nShareVec = 0, nGlobBVec = 0;
      if (jac->const_block_size) { // use shared memory for work vectors only if constant block size - TODO: test efficiency loss
        size_t      maximum_shared_mem_size = 64000;
        PetscDevice device;
        PetscCall(PetscDeviceGetDefault_Internal(&device));
        PetscCall(PetscDeviceGetAttribute(device, PETSC_DEVICE_ATTR_SIZE_T_SHARED_MEM_PER_BLOCK, &maximum_shared_mem_size));
        stride_shared = jac->const_block_size;                                                   // captured
        nShareVec     = maximum_shared_mem_size / (jac->const_block_size * sizeof(PetscScalar)); // integer floor, number of vectors that fit in shared
        if (nShareVec > nwork) nShareVec = nwork;
        else nGlobBVec = nwork - nShareVec;
        global_buff_words     = jac->n * nGlobBVec;
        scr_bytes_team_shared = jac->const_block_size * nShareVec * sizeof(PetscScalar);
      } else {
        scr_bytes_team_shared = 0;
        stride_shared         = 0;
        global_buff_words     = jac->n * nwork;
        nGlobBVec             = nwork; // not needed == fix
      }
      stride_global = jac->n; // captured
#if defined(PETSC_HAVE_CUDA)
      nvtxRangePushA("batch-kokkos-solve");
#endif
      Kokkos::View<PetscScalar *, Kokkos::DefaultExecutionSpace> d_work_vecs_k("workvectors", global_buff_words); // global work vectors
#if PCBJKOKKOS_VERBOSE_LEVEL > 1
      PetscCall(PetscInfo(pc, "\tn = %d. %d shared bytes/team, %d global mem bytes, rtol=%e, num blocks %d, team_size=%d, %d vector threads, %d shared vectors, %d global vectors\n", (int)jac->n, scr_bytes_team_shared, global_buff_words, rtol, (int)nBlk, (int)team_size, PCBJKOKKOS_VEC_SIZE, nShareVec, nGlobBVec));
#endif
      PetscScalar *d_work_vecs = d_work_vecs_k.data();
      Kokkos::parallel_for(
        "Solve", Kokkos::TeamPolicy<Kokkos::LaunchBounds<256, 4>>(nBlk, team_size, PCBJKOKKOS_VEC_SIZE).set_scratch_size(PCBJKOKKOS_SHARED_LEVEL, Kokkos::PerTeam(scr_bytes_team_shared)), KOKKOS_LAMBDA(const team_member team) {
          const int    blkID = team.league_rank(), start = d_bid_eqOffset[blkID], end = d_bid_eqOffset[blkID + 1];
          vect2D_scr_t work_vecs_shared(team.team_scratch(PCBJKOKKOS_SHARED_LEVEL), end - start, nShareVec);
          PetscScalar *work_buff_shared = work_vecs_shared.data();
          PetscScalar *work_buff_global = &d_work_vecs[start]; // start inc'ed in
          bool         print            = monitor && (blkID == view_bid);
          switch (ksp_type_idx) {
          case BATCH_KSP_BICG_IDX:
            static_cast<void>(BJSolve_BICG(team, glb_Aai, glb_Aaj, glb_Aaa, d_isrow, d_isicol, work_buff_global, stride_global, nShareVec, work_buff_shared, stride_shared, rtol, atol, dtol, maxit, &d_metadata[blkID], start, end, glb_idiag, glb_bdata, glb_xdata, print));
            break;
          case BATCH_KSP_TFQMR_IDX:
            static_cast<void>(BJSolve_TFQMR(team, glb_Aai, glb_Aaj, glb_Aaa, d_isrow, d_isicol, work_buff_global, stride_global, nShareVec, work_buff_shared, stride_shared, rtol, atol, dtol, maxit, &d_metadata[blkID], start, end, glb_idiag, glb_bdata, glb_xdata, print));
            break;
          default:
#if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_SYCL)
            printf("Unknown KSP type %d\n", ksp_type_idx);
#else
            /* void */;
#endif
          }
        });
      Kokkos::fence();
#if defined(PETSC_HAVE_CUDA)
      nvtxRangePop();
      nvtxRangePushA("Post-solve-metadata");
#endif
      auto h_metadata = Kokkos::create_mirror(Kokkos::HostSpace::memory_space(), d_metadata);
      Kokkos::deep_copy(h_metadata, d_metadata);
      PetscInt max_nnit = -1;
#if PCBJKOKKOS_VERBOSE_LEVEL > 1
      PetscInt mbid = 0;
#endif
      int in[2], out[2];
      if (jac->reason) { // -pc_bjkokkos_ksp_converged_reason
#if PCBJKOKKOS_VERBOSE_LEVEL >= 3
  #if PCBJKOKKOS_VERBOSE_LEVEL >= 4
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Iterations\n"));
  #endif
        // assume species major
  #if PCBJKOKKOS_VERBOSE_LEVEL == 3
        if (batch_sz != 1) PetscCall(PetscPrintf(PetscObjectComm((PetscObject)A), "%s: max iterations per species:", ksp_type_idx == BATCH_KSP_BICG_IDX ? "bicg" : "tfqmr"));
        else PetscCall(PetscPrintf(PetscObjectComm((PetscObject)A), "    Linear solve converged due to %s iterations ", ksp_type_idx == BATCH_KSP_BICG_IDX ? "bicg" : "tfqmr"));
  #endif
        for (PetscInt dmIdx = 0, head = 0, s = 0; dmIdx < jac->num_dms; dmIdx += batch_sz) {
          for (PetscInt f = 0, idx = head; f < jac->dm_Nf[dmIdx]; f++, idx++, s++) {
            for (int bid = 0; bid < batch_sz; bid++) {
  #if PCBJKOKKOS_VERBOSE_LEVEL >= 4
              jac->max_nits += h_metadata[idx + bid * jac->dm_Nf[dmIdx]].its; // report total number of iterations with high verbose
              if (h_metadata[idx + bid * jac->dm_Nf[dmIdx]].its > max_nnit) {
                max_nnit = h_metadata[idx + bid * jac->dm_Nf[dmIdx]].its;
                mbid     = bid;
              }
  #else
              if (h_metadata[idx + bid * jac->dm_Nf[dmIdx]].its > max_nnit) {
                jac->max_nits = max_nnit = h_metadata[idx + bid * jac->dm_Nf[dmIdx]].its;
                mbid                     = bid;
              }
  #endif
            }
  #if PCBJKOKKOS_VERBOSE_LEVEL >= 4
            PetscCall(PetscPrintf(PetscObjectComm((PetscObject)A), "%2" PetscInt_FMT ":", s));
            for (int bid = 0; bid < batch_sz; bid++) PetscCall(PetscPrintf(PetscObjectComm((PetscObject)A), "%3" PetscInt_FMT " ", h_metadata[idx + bid * jac->dm_Nf[dmIdx]].its));
            PetscCall(PetscPrintf(PetscObjectComm((PetscObject)A), "\n"));
  #else // == 3
            PetscCall(PetscPrintf(PetscObjectComm((PetscObject)A), "%3" PetscInt_FMT " ", max_nnit));
  #endif
          }
          head += batch_sz * jac->dm_Nf[dmIdx];
        }
  #if PCBJKOKKOS_VERBOSE_LEVEL == 3
        PetscCall(PetscPrintf(PetscObjectComm((PetscObject)A), "\n"));
  #endif
#endif
        if (max_nnit == -1) { // < 3
          for (int blkID = 0; blkID < nBlk; blkID++) {
            if (h_metadata[blkID].its > max_nnit) {
              jac->max_nits = max_nnit = h_metadata[blkID].its;
#if PCBJKOKKOS_VERBOSE_LEVEL > 1
              mbid = blkID;
#endif
            }
          }
        }
        in[0] = max_nnit;
        in[1] = rank;
        PetscCallMPI(MPIU_Allreduce(in, out, 1, MPI_2INT, MPI_MAXLOC, PetscObjectComm((PetscObject)A)));
#if PCBJKOKKOS_VERBOSE_LEVEL > 1
        if (0 == rank) {
          if (batch_sz != 1)
            PetscCall(PetscPrintf(PETSC_COMM_SELF, "[%d] Linear solve converged due to %s iterations %d (max), on block %" PetscInt_FMT ", species %" PetscInt_FMT " (max)\n", out[1], KSPConvergedReasons[h_metadata[mbid].reason], out[0], mbid % batch_sz, mbid / batch_sz));
          else PetscCall(PetscPrintf(PETSC_COMM_SELF, "[%d] Linear solve converged due to %s iterations %d (max), on block %" PetscInt_FMT "\n", out[1], KSPConvergedReasons[h_metadata[mbid].reason], out[0], mbid));
        }
#endif
      }
      for (int blkID = 0; blkID < nBlk; blkID++) {
        PetscCall(PetscLogGpuFlops((PetscLogDouble)h_metadata[blkID].flops));
        PetscCheck(h_metadata[blkID].reason >= 0 || !jac->ksp->errorifnotconverged, PetscObjectComm((PetscObject)pc), PETSC_ERR_CONV_FAILED, "ERROR reason=%s, its=%" PetscInt_FMT ". species %" PetscInt_FMT ", batch %" PetscInt_FMT,
                   KSPConvergedReasons[h_metadata[blkID].reason], h_metadata[blkID].its, blkID / batch_sz, blkID % batch_sz);
      }
      {
        int errsum;
        Kokkos::parallel_reduce(
          nBlk,
          KOKKOS_LAMBDA(const int idx, int &lsum) {
            if (d_metadata[idx].reason < 0) ++lsum;
          },
          errsum);
        pcreason = errsum ? PC_SUBPC_ERROR : PC_NOERROR;
        if (!errsum && !jac->max_nits) { // set max its to give back to top KSP
          for (int blkID = 0; blkID < nBlk; blkID++) {
            if (h_metadata[blkID].its > jac->max_nits) jac->max_nits = h_metadata[blkID].its;
          }
        } else if (errsum) {
          PetscCall(PetscPrintf(PETSC_COMM_SELF, "[%d] ERROR Kokkos batch solver did not converge in all solves\n", (int)rank));
        }
      }
#if defined(PETSC_HAVE_CUDA)
      nvtxRangePop();
#endif
    } // end of Kokkos (not Kernels) solvers block
    PetscCall(VecRestoreArrayAndMemType(xout, &glb_xdata));
    PetscCall(VecRestoreArrayReadAndMemType(bvec, &glb_bdata));
    PetscCall(PCSetFailedReason(pc, pcreason));
    // map back to Plex space - not used
    if (plex_batch) {
      PetscCall(VecCopy(xout, bvec));
      PetscCall(VecScatterBegin(plex_batch, bvec, xout, INSERT_VALUES, SCATTER_REVERSE));
      PetscCall(VecScatterEnd(plex_batch, bvec, xout, INSERT_VALUES, SCATTER_REVERSE));
      PetscCall(VecDestroy(&bvec));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetUp_BJKOKKOS(PC pc)
{
  PC_PCBJKOKKOS *jac = (PC_PCBJKOKKOS *)pc->data;
  Mat            A = pc->pmat, Aseq = A; // use filtered block matrix, really "P"
  PetscBool      flg;

  PetscFunctionBegin;
  PetscCheck(A, PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONG, "No matrix - A is used above");
  PetscCall(PetscObjectTypeCompareAny((PetscObject)A, &flg, MATSEQAIJKOKKOS, MATMPIAIJKOKKOS, MATAIJKOKKOS, ""));
  PetscCheck(flg, PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONG, "must use '-[dm_]mat_type aijkokkos -[dm_]vec_type kokkos' for -pc_type bjkokkos");
  if (!A->spptr) Aseq = ((Mat_MPIAIJ *)A->data)->A; // MPI
  PetscCall(MatSeqAIJKokkosSyncDevice(Aseq));
  {
    PetscInt    Istart, Iend;
    PetscMPIInt rank;
    PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)A), &rank));
    PetscCall(MatGetOwnershipRange(A, &Istart, &Iend));
    if (!jac->vec_diag) {
      Vec     *subX = NULL;
      DM       pack, *subDM = NULL;
      PetscInt nDMs, n, *block_sizes = NULL;
      IS       isrow, isicol;
      { // Permute the matrix to get a block diagonal system: d_isrow_k, d_isicol_k
        MatOrderingType rtype;
        const PetscInt *rowindices, *icolindices;
        rtype = MATORDERINGRCM;
        // get permutation. And invert. should we convert to local indices?
        PetscCall(MatGetOrdering(Aseq, rtype, &isrow, &isicol)); // only seems to work for seq matrix
        PetscCall(ISDestroy(&isrow));
        PetscCall(ISInvertPermutation(isicol, PETSC_DECIDE, &isrow)); // THIS IS BACKWARD -- isrow is inverse
        // if (rank==1) PetscCall(ISView(isicol, PETSC_VIEWER_STDOUT_SELF));
        if (0) {
          Mat mat_block_order; // debug
          PetscCall(ISShift(isicol, Istart, isicol));
          PetscCall(MatCreateSubMatrix(A, isicol, isicol, MAT_INITIAL_MATRIX, &mat_block_order));
          PetscCall(ISShift(isicol, -Istart, isicol));
          PetscCall(MatViewFromOptions(mat_block_order, NULL, "-ksp_batch_reorder_view"));
          PetscCall(MatDestroy(&mat_block_order));
        }
        PetscCall(ISGetIndices(isrow, &rowindices)); // local idx
        PetscCall(ISGetIndices(isicol, &icolindices));
        const Kokkos::View<PetscInt *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_isrow_k((PetscInt *)rowindices, A->rmap->n);
        const Kokkos::View<PetscInt *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_isicol_k((PetscInt *)icolindices, A->rmap->n);
        jac->d_isrow_k  = new Kokkos::View<PetscInt *>(Kokkos::create_mirror(DefaultMemorySpace(), h_isrow_k));
        jac->d_isicol_k = new Kokkos::View<PetscInt *>(Kokkos::create_mirror(DefaultMemorySpace(), h_isicol_k));
        Kokkos::deep_copy(*jac->d_isrow_k, h_isrow_k);
        Kokkos::deep_copy(*jac->d_isicol_k, h_isicol_k);
        PetscCall(ISRestoreIndices(isrow, &rowindices));
        PetscCall(ISRestoreIndices(isicol, &icolindices));
        // if (rank==1) PetscCall(ISView(isicol, PETSC_VIEWER_STDOUT_SELF));
      }
      // get block sizes & allocate vec_diag
      PetscCall(PCGetDM(pc, &pack));
      if (pack) {
        PetscCall(PetscObjectTypeCompare((PetscObject)pack, DMCOMPOSITE, &flg));
        if (flg) {
          PetscCall(DMCompositeGetNumberDM(pack, &nDMs));
          PetscCall(DMCreateGlobalVector(pack, &jac->vec_diag));
        } else pack = NULL; // flag for no DM
      }
      if (!jac->vec_diag) { // get 'nDMs' and sizes 'block_sizes' w/o DMComposite. TODO: User could provide ISs
        PetscInt        bsrt, bend, ncols, ntot = 0;
        const PetscInt *colsA, nloc = Iend - Istart;
        const PetscInt *rowindices, *icolindices;
        PetscCall(PetscMalloc1(nloc, &block_sizes)); // very inefficient, to big
        PetscCall(ISGetIndices(isrow, &rowindices));
        PetscCall(ISGetIndices(isicol, &icolindices));
        nDMs = 0;
        bsrt = 0;
        bend = 1;
        for (PetscInt row_B = 0; row_B < nloc; row_B++) { // for all rows in block diagonal space
          PetscInt rowA = icolindices[row_B], minj = PETSC_INT_MAX, maxj = 0;
          //PetscCall(PetscPrintf(PETSC_COMM_SELF, "\t[%d] rowA = %d\n",rank,rowA));
          PetscCall(MatGetRow(Aseq, rowA, &ncols, &colsA, NULL)); // not sorted in permutation
          PetscCheck(ncols, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONG, "Empty row not supported: %" PetscInt_FMT, row_B);
          for (PetscInt colj = 0; colj < ncols; colj++) {
            PetscInt colB = rowindices[colsA[colj]]; // use local idx
            //PetscCall(PetscPrintf(PETSC_COMM_SELF, "\t\t[%d] colB = %d\n",rank,colB));
            PetscCheck(colB >= 0 && colB < nloc, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONG, "colB < 0: %" PetscInt_FMT, colB);
            if (colB > maxj) maxj = colB;
            if (colB < minj) minj = colB;
          }
          PetscCall(MatRestoreRow(Aseq, rowA, &ncols, &colsA, NULL));
          if (minj >= bend) { // first column is > max of last block -- new block or last block
            //PetscCall(PetscPrintf(PetscObjectComm((PetscObject)A), "\t\t finish block %d, N loc = %d (%d,%d)\n", nDMs+1, bend - bsrt,bsrt,bend));
            block_sizes[nDMs] = bend - bsrt;
            ntot += block_sizes[nDMs];
            PetscCheck(minj == bend, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "minj != bend: %" PetscInt_FMT " != %" PetscInt_FMT, minj, bend);
            bsrt = bend;
            bend++; // start with size 1 in new block
            nDMs++;
          }
          if (maxj + 1 > bend) bend = maxj + 1;
          PetscCheck(minj >= bsrt || row_B == Iend - 1, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONG, "%" PetscInt_FMT ") minj < bsrt: %" PetscInt_FMT " != %" PetscInt_FMT, rowA, minj, bsrt);
          //PetscCall(PetscPrintf(PETSC_COMM_SELF, "[%d] %d) row %d.%d) cols %d : %d ; bsrt = %d, bend = %d\n",rank,row_B,nDMs,rowA,minj,maxj,bsrt,bend));
        }
        // do last block
        //PetscCall(PetscPrintf(PETSC_COMM_SELF, "\t\t\t [%d] finish block %d, N loc = %d (%d,%d)\n", rank, nDMs+1, bend - bsrt,bsrt,bend));
        block_sizes[nDMs] = bend - bsrt;
        ntot += block_sizes[nDMs];
        nDMs++;
        // cleanup
        PetscCheck(ntot == nloc, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONG, "n total != n local: %" PetscInt_FMT " != %" PetscInt_FMT, ntot, nloc);
        PetscCall(ISRestoreIndices(isrow, &rowindices));
        PetscCall(ISRestoreIndices(isicol, &icolindices));
        PetscCall(PetscRealloc(sizeof(PetscInt) * nDMs, &block_sizes));
        PetscCall(MatCreateVecs(A, &jac->vec_diag, NULL));
        PetscCall(PetscInfo(pc, "Setup Matrix based meta data (not DMComposite not attached to PC) %" PetscInt_FMT " sub domains\n", nDMs));
      }
      PetscCall(ISDestroy(&isrow));
      PetscCall(ISDestroy(&isicol));
      jac->num_dms = nDMs;
      PetscCall(VecGetLocalSize(jac->vec_diag, &n));
      jac->n         = n;
      jac->d_idiag_k = new Kokkos::View<PetscScalar *, Kokkos::LayoutRight>("idiag", n);
      // options
      PetscCall(PCBJKOKKOSCreateKSP_BJKOKKOS(pc));
      PetscCall(KSPSetFromOptions(jac->ksp));
      PetscCall(PetscObjectTypeCompareAny((PetscObject)jac->ksp, &flg, KSPBICG, ""));
      if (flg) {
        jac->ksp_type_idx = BATCH_KSP_BICG_IDX;
        jac->nwork        = 7;
      } else {
        PetscCall(PetscObjectTypeCompareAny((PetscObject)jac->ksp, &flg, KSPTFQMR, ""));
        if (flg) {
          jac->ksp_type_idx = BATCH_KSP_TFQMR_IDX;
          jac->nwork        = 10;
        } else {
#if defined(PETSC_HAVE_KOKKOS_KERNELS_BATCH)
          PetscCall(PetscObjectTypeCompareAny((PetscObject)jac->ksp, &flg, KSPGMRES, ""));
          PetscCheck(flg, PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONG, "Unsupported batch ksp type");
          jac->ksp_type_idx = BATCH_KSP_GMRESKK_IDX;
          jac->nwork        = 0;
#else
          KSPType ksptype;
          PetscCall(KSPGetType(jac->ksp, &ksptype));
          PetscCheck(flg, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONG, "Type: %s not supported in complex", ksptype);
#endif
        }
      }
      PetscOptionsBegin(PetscObjectComm((PetscObject)jac->ksp), ((PetscObject)jac->ksp)->prefix, "Options for Kokkos batch solver", "none");
      PetscCall(PetscOptionsBool("-ksp_converged_reason", "", "bjkokkos.kokkos.cxx.c", jac->reason, &jac->reason, NULL));
      PetscCall(PetscOptionsBool("-ksp_monitor", "", "bjkokkos.kokkos.cxx.c", jac->monitor, &jac->monitor, NULL));
      PetscCall(PetscOptionsInt("-ksp_batch_target", "", "bjkokkos.kokkos.cxx.c", jac->batch_target, &jac->batch_target, NULL));
      PetscCall(PetscOptionsInt("-ksp_rank_target", "", "bjkokkos.kokkos.cxx.c", jac->rank_target, &jac->rank_target, NULL));
      PetscCall(PetscOptionsInt("-ksp_batch_nsolves_team", "", "bjkokkos.kokkos.cxx.c", jac->nsolves_team, &jac->nsolves_team, NULL));
      PetscCheck(jac->batch_target < jac->num_dms, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "-ksp_batch_target (%" PetscInt_FMT ") >= number of DMs (%" PetscInt_FMT ")", jac->batch_target, jac->num_dms);
      PetscOptionsEnd();
      // get blocks - jac->d_bid_eqOffset_k
      if (pack) {
        PetscCall(PetscMalloc(sizeof(*subX) * nDMs, &subX));
        PetscCall(PetscMalloc(sizeof(*subDM) * nDMs, &subDM));
      }
      PetscCall(PetscMalloc(sizeof(*jac->dm_Nf) * nDMs, &jac->dm_Nf));
      PetscCall(PetscInfo(pc, "Have %" PetscInt_FMT " blocks, n=%" PetscInt_FMT " rtol=%g type = %s\n", nDMs, n, (double)jac->ksp->rtol, ((PetscObject)jac->ksp)->type_name));
      if (pack) PetscCall(DMCompositeGetEntriesArray(pack, subDM));
      jac->nBlocks = 0;
      for (PetscInt ii = 0; ii < nDMs; ii++) {
        PetscInt Nf;
        if (subDM) {
          DM           dm = subDM[ii];
          PetscSection section;
          PetscCall(DMGetLocalSection(dm, &section));
          PetscCall(PetscSectionGetNumFields(section, &Nf));
        } else Nf = 1;
        jac->nBlocks += Nf;
#if PCBJKOKKOS_VERBOSE_LEVEL <= 2
        if (ii == 0) PetscCall(PetscInfo(pc, "%" PetscInt_FMT ") %" PetscInt_FMT " blocks (%" PetscInt_FMT " total)\n", ii, Nf, jac->nBlocks));
#else
        PetscCall(PetscInfo(pc, "%" PetscInt_FMT ") %" PetscInt_FMT " blocks (%" PetscInt_FMT " total)\n", ii, Nf, jac->nBlocks));
#endif
        jac->dm_Nf[ii] = Nf;
      }
      { // d_bid_eqOffset_k
        Kokkos::View<PetscInt *, Kokkos::LayoutRight, Kokkos::HostSpace> h_block_offsets("block_offsets", jac->nBlocks + 1);
        if (pack) PetscCall(DMCompositeGetAccessArray(pack, jac->vec_diag, nDMs, NULL, subX));
        h_block_offsets[0]    = 0;
        jac->const_block_size = -1;
        for (PetscInt ii = 0, idx = 0; ii < nDMs; ii++) {
          PetscInt nloc, nblk;
          if (pack) PetscCall(VecGetSize(subX[ii], &nloc));
          else nloc = block_sizes[ii];
          nblk = nloc / jac->dm_Nf[ii];
          PetscCheck(nloc % jac->dm_Nf[ii] == 0, PetscObjectComm((PetscObject)pc), PETSC_ERR_USER, "nloc%%jac->dm_Nf[ii] (%" PetscInt_FMT ") != 0 DMs", nloc % jac->dm_Nf[ii]);
          for (PetscInt jj = 0; jj < jac->dm_Nf[ii]; jj++, idx++) {
            h_block_offsets[idx + 1] = h_block_offsets[idx] + nblk;
#if PCBJKOKKOS_VERBOSE_LEVEL <= 2
            if (idx == 0) PetscCall(PetscInfo(pc, "Add first of %" PetscInt_FMT " blocks with %" PetscInt_FMT " equations\n", jac->nBlocks, nblk));
#else
            PetscCall(PetscInfo(pc, "\t%" PetscInt_FMT ") Add block with %" PetscInt_FMT " equations of %" PetscInt_FMT "\n", idx + 1, nblk, jac->nBlocks));
#endif
            if (jac->const_block_size == -1) jac->const_block_size = nblk;
            else if (jac->const_block_size > 0 && jac->const_block_size != nblk) jac->const_block_size = 0;
          }
        }
        if (pack) {
          PetscCall(DMCompositeRestoreAccessArray(pack, jac->vec_diag, jac->nBlocks, NULL, subX));
          PetscCall(PetscFree(subX));
          PetscCall(PetscFree(subDM));
        }
        jac->d_bid_eqOffset_k = new Kokkos::View<PetscInt *, Kokkos::LayoutRight>(Kokkos::create_mirror(Kokkos::DefaultExecutionSpace::memory_space(), h_block_offsets));
        Kokkos::deep_copy(*jac->d_bid_eqOffset_k, h_block_offsets);
      }
      if (!pack) PetscCall(PetscFree(block_sizes));
    }
    { // get jac->d_idiag_k (PC setup),
      const PetscInt    *d_ai, *d_aj;
      const PetscScalar *d_aa;
      const PetscInt     conc = Kokkos::DefaultExecutionSpace().concurrency(), openmp = !!(conc < 1000), team_size = (openmp == 0 && PCBJKOKKOS_VEC_SIZE != 1) ? PCBJKOKKOS_TEAM_SIZE : 1;
      const PetscInt    *d_bid_eqOffset = jac->d_bid_eqOffset_k->data(), *r = jac->d_isrow_k->data(), *ic = jac->d_isicol_k->data();
      PetscScalar       *d_idiag = jac->d_idiag_k->data(), *dummy;
      PetscMemType       mtype;
      PetscCall(MatSeqAIJGetCSRAndMemType(Aseq, &d_ai, &d_aj, &dummy, &mtype));
      d_aa = dummy;
      Kokkos::parallel_for(
        "Diag", Kokkos::TeamPolicy<>(jac->nBlocks, team_size, PCBJKOKKOS_VEC_SIZE), KOKKOS_LAMBDA(const team_member team) {
          const PetscInt blkID = team.league_rank();
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team, d_bid_eqOffset[blkID], d_bid_eqOffset[blkID + 1]), [=](const int rowb) {
            const PetscInt     rowa = ic[rowb], ai = d_ai[rowa], *aj = d_aj + ai; // grab original data
            const PetscScalar *aa   = d_aa + ai;
            const PetscInt     nrow = d_ai[rowa + 1] - ai;
            int                found;
            Kokkos::parallel_reduce(
              Kokkos::ThreadVectorRange(team, nrow),
              [=](const int &j, int &count) {
                const PetscInt colb = r[aj[j]];
                if (colb == rowb) {
                  d_idiag[rowb] = 1. / aa[j];
                  count++;
                }
              },
              found);
#if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_SYCL)
            if (found != 1) Kokkos::single(Kokkos::PerThread(team), [=]() { printf("ERRORrow %d) found = %d\n", rowb, found); });
#endif
          });
        });
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Default destroy, if it has never been setup */
static PetscErrorCode PCReset_BJKOKKOS(PC pc)
{
  PC_PCBJKOKKOS *jac = (PC_PCBJKOKKOS *)pc->data;

  PetscFunctionBegin;
  PetscCall(KSPDestroy(&jac->ksp));
  PetscCall(VecDestroy(&jac->vec_diag));
  if (jac->d_bid_eqOffset_k) delete jac->d_bid_eqOffset_k;
  if (jac->d_idiag_k) delete jac->d_idiag_k;
  if (jac->d_isrow_k) delete jac->d_isrow_k;
  if (jac->d_isicol_k) delete jac->d_isicol_k;
  jac->d_bid_eqOffset_k = NULL;
  jac->d_idiag_k        = NULL;
  jac->d_isrow_k        = NULL;
  jac->d_isicol_k       = NULL;
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCBJKOKKOSGetKSP_C", NULL)); // not published now (causes configure errors)
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCBJKOKKOSSetKSP_C", NULL));
  PetscCall(PetscFree(jac->dm_Nf));
  jac->dm_Nf = NULL;
  if (jac->rowOffsets) delete jac->rowOffsets;
  if (jac->colIndices) delete jac->colIndices;
  if (jac->batch_b) delete jac->batch_b;
  if (jac->batch_x) delete jac->batch_x;
  if (jac->batch_values) delete jac->batch_values;
  jac->rowOffsets   = NULL;
  jac->colIndices   = NULL;
  jac->batch_b      = NULL;
  jac->batch_x      = NULL;
  jac->batch_values = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCDestroy_BJKOKKOS(PC pc)
{
  PetscFunctionBegin;
  PetscCall(PCReset_BJKOKKOS(pc));
  PetscCall(PetscFree(pc->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCView_BJKOKKOS(PC pc, PetscViewer viewer)
{
  PC_PCBJKOKKOS *jac = (PC_PCBJKOKKOS *)pc->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  if (!jac->ksp) PetscCall(PCBJKOKKOSCreateKSP_BJKOKKOS(pc));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Batched device linear solver: Krylov (KSP) method with Jacobi preconditioning\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "\t\tnwork = %" PetscInt_FMT ", rel tol = %e, abs tol = %e, div tol = %e, max it =%" PetscInt_FMT ", type = %s\n", jac->nwork, jac->ksp->rtol, jac->ksp->abstol, jac->ksp->divtol, jac->ksp->max_it,
                                     ((PetscObject)jac->ksp)->type_name));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetFromOptions_BJKOKKOS(PC pc, PetscOptionItems PetscOptionsObject)
{
  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "PC BJKOKKOS options");
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCBJKOKKOSSetKSP_BJKOKKOS(PC pc, KSP ksp)
{
  PC_PCBJKOKKOS *jac = (PC_PCBJKOKKOS *)pc->data;

  PetscFunctionBegin;
  PetscCall(PetscObjectReference((PetscObject)ksp));
  PetscCall(KSPDestroy(&jac->ksp));
  jac->ksp = ksp;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCBJKOKKOSSetKSP - Sets the `KSP` context for `PCBJKOKKOS`

  Collective

  Input Parameters:
+ pc  - the `PCBJKOKKOS` preconditioner context
- ksp - the `KSP` solver

  Level: advanced

  Notes:
  The `PC` and the `KSP` must have the same communicator

  If the `PC` is not `PCBJKOKKOS` this function returns without doing anything

.seealso: [](ch_ksp), `PCBJKOKKOSGetKSP()`, `PCBJKOKKOS`
@*/
PetscErrorCode PCBJKOKKOSSetKSP(PC pc, KSP ksp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 2);
  PetscCheckSameComm(pc, 1, ksp, 2);
  PetscTryMethod(pc, "PCBJKOKKOSSetKSP_C", (PC, KSP), (pc, ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCBJKOKKOSGetKSP_BJKOKKOS(PC pc, KSP *ksp)
{
  PC_PCBJKOKKOS *jac = (PC_PCBJKOKKOS *)pc->data;

  PetscFunctionBegin;
  if (!jac->ksp) PetscCall(PCBJKOKKOSCreateKSP_BJKOKKOS(pc));
  *ksp = jac->ksp;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCBJKOKKOSGetKSP - Gets the `KSP` context for the `PCBJKOKKOS` preconditioner

  Not Collective but `KSP` returned is parallel if `PC` was parallel

  Input Parameter:
. pc - the preconditioner context

  Output Parameter:
. ksp - the `KSP` solver

  Level: advanced

  Notes:
  You must call `KSPSetUp()` before calling `PCBJKOKKOSGetKSP()`.

  If the `PC` is not a `PCBJKOKKOS` object it raises an error

.seealso: [](ch_ksp), `PCBJKOKKOS`, `PCBJKOKKOSSetKSP()`
@*/
PetscErrorCode PCBJKOKKOSGetKSP(PC pc, KSP *ksp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscAssertPointer(ksp, 2);
  PetscUseMethod(pc, "PCBJKOKKOSGetKSP_C", (PC, KSP *), (pc, ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCPostSolve_BJKOKKOS(PC pc, KSP ksp, Vec b, Vec x)
{
  PC_PCBJKOKKOS *jac = (PC_PCBJKOKKOS *)pc->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  ksp->its = jac->max_nits;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCPreSolve_BJKOKKOS(PC pc, KSP ksp, Vec b, Vec x)
{
  PC_PCBJKOKKOS *jac = (PC_PCBJKOKKOS *)pc->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  jac->ksp->errorifnotconverged = ksp->errorifnotconverged;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
     PCBJKOKKOS - A batched Krylov/block Jacobi solver that runs a solve of each diagaonl block of a block diagonal `MATSEQAIJ` in a Kokkos thread group

   Options Database Key:
.  -pc_bjkokkos_ - options prefix for its `KSP` options

   Level: intermediate

   Note:
   For use with `-ksp_type preonly` to bypass any computation on the CPU

   Developer Notes:
   The entire Krylov (TFQMR or BICG) with diagonal preconditioning for each block of a block diagnaol matrix runs in a Kokkos thread group (eg, one block per SM on NVIDIA). It supports taking a non-block diagonal matrix but this is not tested. One should create an explicit block diagonal matrix and use that as the matrix for constructing the preconditioner in the outer `KSP` solver. Variable block size are supported and tested in src/ts/utils/dmplexlandau/tutorials/ex[1|2].c

.seealso: [](ch_ksp), `PCCreate()`, `PCSetType()`, `PCType`, `PC`, `PCBJACOBI`,
          `PCSHELL`, `PCCOMPOSITE`, `PCSetUseAmat()`, `PCBJKOKKOSGetKSP()`
M*/

PETSC_EXTERN PetscErrorCode PCCreate_BJKOKKOS(PC pc)
{
  PC_PCBJKOKKOS *jac;

  PetscFunctionBegin;
  PetscCall(PetscNew(&jac));
  pc->data = (void *)jac;

  jac->ksp              = NULL;
  jac->vec_diag         = NULL;
  jac->d_bid_eqOffset_k = NULL;
  jac->d_idiag_k        = NULL;
  jac->d_isrow_k        = NULL;
  jac->d_isicol_k       = NULL;
  jac->nBlocks          = 1;
  jac->max_nits         = 0;

  PetscCall(PetscMemzero(pc->ops, sizeof(struct _PCOps)));
  pc->ops->apply          = PCApply_BJKOKKOS;
  pc->ops->applytranspose = NULL;
  pc->ops->setup          = PCSetUp_BJKOKKOS;
  pc->ops->reset          = PCReset_BJKOKKOS;
  pc->ops->destroy        = PCDestroy_BJKOKKOS;
  pc->ops->setfromoptions = PCSetFromOptions_BJKOKKOS;
  pc->ops->view           = PCView_BJKOKKOS;
  pc->ops->postsolve      = PCPostSolve_BJKOKKOS;
  pc->ops->presolve       = PCPreSolve_BJKOKKOS;

  jac->rowOffsets   = NULL;
  jac->colIndices   = NULL;
  jac->batch_b      = NULL;
  jac->batch_x      = NULL;
  jac->batch_values = NULL;

  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCBJKOKKOSGetKSP_C", PCBJKOKKOSGetKSP_BJKOKKOS));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCBJKOKKOSSetKSP_C", PCBJKOKKOSSetKSP_BJKOKKOS));
  PetscFunctionReturn(PETSC_SUCCESS);
}
