
#include <petsc/private/kspimpl.h>

static PetscErrorCode KSPSetUp_TFQMR(KSP ksp)
{
  PetscFunctionBegin;
  PetscCheck(ksp->pc_side != PC_SYMMETRIC, PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "no symmetric preconditioning for KSPTFQMR");
  PetscCall(KSPSetWorkVecs(ksp, 9));
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPSolve_TFQMR(KSP ksp)
{
  PetscInt    i, m;
  PetscScalar rho, rhoold, a, s, b, eta, etaold, psiold, cf;
  PetscReal   dp, dpold, w, dpest, tau, psi, cm;
  Vec         X, B, V, P, R, RP, T, T1, Q, U, D, AUQ;

  PetscFunctionBegin;
  X   = ksp->vec_sol;
  B   = ksp->vec_rhs;
  R   = ksp->work[0];
  RP  = ksp->work[1];
  V   = ksp->work[2];
  T   = ksp->work[3];
  Q   = ksp->work[4];
  P   = ksp->work[5];
  U   = ksp->work[6];
  D   = ksp->work[7];
  T1  = ksp->work[8];
  AUQ = V;

  /* Compute initial preconditioned residual */
  PetscCall(KSPInitialResidual(ksp, X, V, T, R, B));

  /* Test for nothing to do */
  PetscCall(VecNorm(R, NORM_2, &dp));
  KSPCheckNorm(ksp, dp);
  PetscCall(PetscObjectSAWsTakeAccess((PetscObject)ksp));
  if (ksp->normtype != KSP_NORM_NONE) ksp->rnorm = dp;
  else ksp->rnorm = 0.0;
  ksp->its = 0;
  PetscCall(PetscObjectSAWsGrantAccess((PetscObject)ksp));
  PetscCall(KSPMonitor(ksp, 0, ksp->rnorm));
  PetscCall((*ksp->converged)(ksp, 0, ksp->rnorm, &ksp->reason, ksp->cnvP));
  if (ksp->reason) PetscFunctionReturn(0);

  /* Make the initial Rp == R */
  PetscCall(VecCopy(R, RP));

  /* Set the initial conditions */
  etaold = 0.0;
  psiold = 0.0;
  tau    = dp;
  dpold  = dp;

  PetscCall(VecDot(R, RP, &rhoold)); /* rhoold = (r,rp)     */
  PetscCall(VecCopy(R, U));
  PetscCall(VecCopy(R, P));
  PetscCall(KSP_PCApplyBAorAB(ksp, P, V, T));
  PetscCall(VecSet(D, 0.0));

  i = 0;
  do {
    PetscCall(PetscObjectSAWsTakeAccess((PetscObject)ksp));
    ksp->its++;
    PetscCall(PetscObjectSAWsGrantAccess((PetscObject)ksp));
    PetscCall(VecDot(V, RP, &s)); /* s <- (v,rp)          */
    KSPCheckDot(ksp, s);
    a = rhoold / s;                    /* a <- rho / s         */
    PetscCall(VecWAXPY(Q, -a, V, U));  /* q <- u - a v         */
    PetscCall(VecWAXPY(T, 1.0, U, Q)); /* t <- u + q           */
    PetscCall(KSP_PCApplyBAorAB(ksp, T, AUQ, T1));
    PetscCall(VecAXPY(R, -a, AUQ)); /* r <- r - a K (u + q) */
    PetscCall(VecNorm(R, NORM_2, &dp));
    KSPCheckNorm(ksp, dp);
    for (m = 0; m < 2; m++) {
      if (!m) w = PetscSqrtReal(dp * dpold);
      else w = dp;
      psi = w / tau;
      cm  = 1.0 / PetscSqrtReal(1.0 + psi * psi);
      tau = tau * psi * cm;
      eta = cm * cm * a;
      cf  = psiold * psiold * etaold / a;
      if (!m) {
        PetscCall(VecAYPX(D, cf, U));
      } else {
        PetscCall(VecAYPX(D, cf, Q));
      }
      PetscCall(VecAXPY(X, eta, D));

      dpest = PetscSqrtReal(2 * i + m + 2.0) * tau;
      PetscCall(PetscObjectSAWsTakeAccess((PetscObject)ksp));
      if (ksp->normtype != KSP_NORM_NONE) ksp->rnorm = dpest;
      else ksp->rnorm = 0.0;
      PetscCall(PetscObjectSAWsGrantAccess((PetscObject)ksp));
      PetscCall(KSPLogResidualHistory(ksp, ksp->rnorm));
      PetscCall(KSPMonitor(ksp, i + 1, ksp->rnorm));
      PetscCall((*ksp->converged)(ksp, i + 1, ksp->rnorm, &ksp->reason, ksp->cnvP));
      if (ksp->reason) break;

      etaold = eta;
      psiold = psi;
    }
    if (ksp->reason) break;

    PetscCall(VecDot(R, RP, &rho));  /* rho <- (r,rp)       */
    b = rho / rhoold;                /* b <- rho / rhoold   */
    PetscCall(VecWAXPY(U, b, Q, R)); /* u <- r + b q        */
    PetscCall(VecAXPY(Q, b, P));
    PetscCall(VecWAXPY(P, b, Q, U));            /* p <- u + b(q + b p) */
    PetscCall(KSP_PCApplyBAorAB(ksp, P, V, Q)); /* v <- K p  */

    rhoold = rho;
    dpold  = dp;

    i++;
  } while (i < ksp->max_it);
  if (i >= ksp->max_it) ksp->reason = KSP_DIVERGED_ITS;

  PetscCall(KSPUnwindPreconditioner(ksp, X, T));
  PetscFunctionReturn(0);
}

/*MC
     KSPTFQMR - A transpose free QMR (quasi minimal residual),

   Level: beginner

   Notes:
   Supports left and right preconditioning, but not symmetric

   The "residual norm" computed in this algorithm is actually just an upper bound on the actual residual norm.
   That is for left preconditioning it is a bound on the preconditioned residual and for right preconditioning
   it is a bound on the true residual.

   References:
.  * - Freund, 1993

.seealso: [](chapter_ksp), `KSPCreate()`, `KSPSetType()`, `KSPType`, `KSP`, `KSPTCQMR`
M*/
PETSC_EXTERN PetscErrorCode KSPCreate_TFQMR(KSP ksp)
{
  PetscFunctionBegin;
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_PRECONDITIONED, PC_LEFT, 3));
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_UNPRECONDITIONED, PC_RIGHT, 2));
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_NONE, PC_LEFT, 1));
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_NONE, PC_RIGHT, 1));

  ksp->data                = (void *)0;
  ksp->ops->setup          = KSPSetUp_TFQMR;
  ksp->ops->solve          = KSPSolve_TFQMR;
  ksp->ops->destroy        = KSPDestroyDefault;
  ksp->ops->buildsolution  = KSPBuildSolutionDefault;
  ksp->ops->buildresidual  = KSPBuildResidualDefault;
  ksp->ops->setfromoptions = NULL;
  ksp->ops->view           = NULL;
  PetscFunctionReturn(0);
}
