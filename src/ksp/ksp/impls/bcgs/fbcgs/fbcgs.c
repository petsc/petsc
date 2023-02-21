
/*
    This file implements flexible BiCGStab (FBiCGStab).
*/
#include <../src/ksp/ksp/impls/bcgs/bcgsimpl.h> /*I  "petscksp.h"  I*/

static PetscErrorCode KSPSetUp_FBCGS(KSP ksp)
{
  PetscFunctionBegin;
  PetscCall(KSPSetWorkVecs(ksp, 8));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Only need a few hacks from KSPSolve_BCGS */

static PetscErrorCode KSPSolve_FBCGS(KSP ksp)
{
  PetscInt    i;
  PetscScalar rho, rhoold, alpha, beta, omega, omegaold, d1;
  Vec         X, B, V, P, R, RP, T, S, P2, S2;
  PetscReal   dp   = 0.0, d2;
  KSP_BCGS   *bcgs = (KSP_BCGS *)ksp->data;
  PC          pc;
  Mat         mat;

  PetscFunctionBegin;
  X  = ksp->vec_sol;
  B  = ksp->vec_rhs;
  R  = ksp->work[0];
  RP = ksp->work[1];
  V  = ksp->work[2];
  T  = ksp->work[3];
  S  = ksp->work[4];
  P  = ksp->work[5];
  S2 = ksp->work[6];
  P2 = ksp->work[7];

  if (!ksp->guess_zero) {
    if (!bcgs->guess) PetscCall(VecDuplicate(X, &bcgs->guess));
    PetscCall(VecCopy(X, bcgs->guess));
  } else {
    PetscCall(VecSet(X, 0.0));
  }

  /* Compute initial residual */
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetUp(pc));
  PetscCall(PCGetOperators(pc, &mat, NULL));
  if (!ksp->guess_zero) {
    PetscCall(KSP_MatMult(ksp, mat, X, S2));
    PetscCall(VecCopy(B, R));
    PetscCall(VecAXPY(R, -1.0, S2));
  } else {
    PetscCall(VecCopy(B, R));
  }

  /* Test for nothing to do */
  if (ksp->normtype != KSP_NORM_NONE) PetscCall(VecNorm(R, NORM_2, &dp));
  PetscCall(PetscObjectSAWsTakeAccess((PetscObject)ksp));
  ksp->its   = 0;
  ksp->rnorm = dp;
  PetscCall(PetscObjectSAWsGrantAccess((PetscObject)ksp));
  PetscCall(KSPLogResidualHistory(ksp, dp));
  PetscCall(KSPMonitor(ksp, 0, dp));
  PetscCall((*ksp->converged)(ksp, 0, dp, &ksp->reason, ksp->cnvP));
  if (ksp->reason) PetscFunctionReturn(PETSC_SUCCESS);

  /* Make the initial Rp == R */
  PetscCall(VecCopy(R, RP));

  rhoold   = 1.0;
  alpha    = 1.0;
  omegaold = 1.0;
  PetscCall(VecSet(P, 0.0));
  PetscCall(VecSet(V, 0.0));

  i = 0;
  do {
    PetscCall(VecDot(R, RP, &rho)); /* rho <- (r,rp) */
    beta = (rho / rhoold) * (alpha / omegaold);
    PetscCall(VecAXPBYPCZ(P, 1.0, -omegaold * beta, beta, R, V)); /* p <- r - omega * beta* v + beta * p */

    PetscCall(KSP_PCApply(ksp, P, P2));      /* p2 <- K p */
    PetscCall(KSP_MatMult(ksp, mat, P2, V)); /* v <- A p2 */

    PetscCall(VecDot(V, RP, &d1));
    if (d1 == 0.0) {
      PetscCheck(!ksp->errorifnotconverged, PetscObjectComm((PetscObject)ksp), PETSC_ERR_NOT_CONVERGED, "KSPSolve breakdown due to zero inner product");
      ksp->reason = KSP_DIVERGED_BREAKDOWN;
      PetscCall(PetscInfo(ksp, "Breakdown due to zero inner product\n"));
      break;
    }
    alpha = rho / d1;                     /* alpha <- rho / (v,rp) */
    PetscCall(VecWAXPY(S, -alpha, V, R)); /* s <- r - alpha v */

    PetscCall(KSP_PCApply(ksp, S, S2));      /* s2 <- K s */
    PetscCall(KSP_MatMult(ksp, mat, S2, T)); /* t <- A s2 */

    PetscCall(VecDotNorm2(S, T, &d1, &d2));
    if (d2 == 0.0) {
      /* t is 0. if s is 0, then alpha v == r, and hence alpha p may be our solution. Give it a try? */
      PetscCall(VecDot(S, S, &d1));
      if (d1 != 0.0) {
        PetscCheck(!ksp->errorifnotconverged, PetscObjectComm((PetscObject)ksp), PETSC_ERR_NOT_CONVERGED, "KSPSolve has failed due to singular preconditioned operator");
        ksp->reason = KSP_DIVERGED_BREAKDOWN;
        PetscCall(PetscInfo(ksp, "Failed due to singular preconditioned operator\n"));
        break;
      }
      PetscCall(VecAXPY(X, alpha, P2)); /* x <- x + alpha p2 */
      PetscCall(PetscObjectSAWsTakeAccess((PetscObject)ksp));
      ksp->its++;
      ksp->rnorm  = 0.0;
      ksp->reason = KSP_CONVERGED_RTOL;
      PetscCall(PetscObjectSAWsGrantAccess((PetscObject)ksp));
      PetscCall(KSPLogResidualHistory(ksp, dp));
      PetscCall(KSPMonitor(ksp, i + 1, 0.0));
      break;
    }
    omega = d1 / d2;                                      /* omega <- (t's) / (t't) */
    PetscCall(VecAXPBYPCZ(X, alpha, omega, 1.0, P2, S2)); /* x <- alpha * p2 + omega * s2 + x */

    PetscCall(VecWAXPY(R, -omega, T, S)); /* r <- s - omega t */
    if (ksp->normtype != KSP_NORM_NONE && ksp->chknorm < i + 2) PetscCall(VecNorm(R, NORM_2, &dp));

    rhoold   = rho;
    omegaold = omega;

    PetscCall(PetscObjectSAWsTakeAccess((PetscObject)ksp));
    ksp->its++;
    ksp->rnorm = dp;
    PetscCall(PetscObjectSAWsGrantAccess((PetscObject)ksp));
    PetscCall(KSPLogResidualHistory(ksp, dp));
    PetscCall(KSPMonitor(ksp, i + 1, dp));
    PetscCall((*ksp->converged)(ksp, i + 1, dp, &ksp->reason, ksp->cnvP));
    if (ksp->reason) break;
    if (rho == 0.0) {
      PetscCheck(!ksp->errorifnotconverged, PetscObjectComm((PetscObject)ksp), PETSC_ERR_NOT_CONVERGED, "KSPSolve breakdown due to zero rho inner product");
      ksp->reason = KSP_DIVERGED_BREAKDOWN;
      PetscCall(PetscInfo(ksp, "Breakdown due to zero rho inner product\n"));
      break;
    }
    i++;
  } while (i < ksp->max_it);

  if (i >= ksp->max_it) ksp->reason = KSP_DIVERGED_ITS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
     KSPFBCGS - Implements the flexible BiCGStab method. [](sec_flexibleksp)

   Level: beginner

   Notes:
   Flexible BiCGStab, unlike most Krylov methods allows the preconditioner to be nonlinear, that is the action of the preconditioner to a vector need not be linear
   in the vector entries.

   `KSPFBCGSR` provides another variant of this algorithm that requires fewer `MPI_Allreduce()` calls and my converge faster

   See `KSPPIPEBCGS` for a pipelined version of the algorithm

   Only supportst right preconditioning

.seealso: [](chapter_ksp),  [](sec_flexibleksp), `KSPFBCGSR`, `KSPPIPEBCGS`, `KSPBCGSL`, `KSPFBCGS`, `KSPCreate()`, `KSPFGMRES`, `KSPSetType()`, `KSPType`, `KSP`, `KSPBICG`, `KSPSetPCSide()`
M*/
PETSC_EXTERN PetscErrorCode KSPCreate_FBCGS(KSP ksp)
{
  KSP_BCGS *bcgs;

  PetscFunctionBegin;
  PetscCall(PetscNew(&bcgs));

  ksp->data                = bcgs;
  ksp->ops->setup          = KSPSetUp_FBCGS;
  ksp->ops->solve          = KSPSolve_FBCGS;
  ksp->ops->destroy        = KSPDestroy_BCGS;
  ksp->ops->reset          = KSPReset_BCGS;
  ksp->ops->buildresidual  = KSPBuildResidualDefault;
  ksp->ops->setfromoptions = KSPSetFromOptions_BCGS;
  ksp->pc_side             = PC_RIGHT;

  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_UNPRECONDITIONED, PC_RIGHT, 2));
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_NONE, PC_RIGHT, 1));
  PetscFunctionReturn(PETSC_SUCCESS);
}
