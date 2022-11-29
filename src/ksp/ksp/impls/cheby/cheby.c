
#include <../src/ksp/ksp/impls/cheby/chebyshevimpl.h> /*I "petscksp.h" I*/

static PetscErrorCode KSPReset_Chebyshev(KSP ksp)
{
  KSP_Chebyshev *cheb = (KSP_Chebyshev *)ksp->data;

  PetscFunctionBegin;
  if (cheb->kspest) PetscCall(KSPReset(cheb->kspest));
  PetscFunctionReturn(0);
}

/*
 * Must be passed a KSP solver that has "converged", with KSPSetComputeEigenvalues() called before the solve
 */
static PetscErrorCode KSPChebyshevComputeExtremeEigenvalues_Private(KSP kspest, PetscReal *emin, PetscReal *emax)
{
  PetscInt   n, neig;
  PetscReal *re, *im, min, max;

  PetscFunctionBegin;
  PetscCall(KSPGetIterationNumber(kspest, &n));
  PetscCall(PetscMalloc2(n, &re, n, &im));
  PetscCall(KSPComputeEigenvalues(kspest, n, re, im, &neig));
  min = PETSC_MAX_REAL;
  max = PETSC_MIN_REAL;
  for (n = 0; n < neig; n++) {
    min = PetscMin(min, re[n]);
    max = PetscMax(max, re[n]);
  }
  PetscCall(PetscFree2(re, im));
  *emax = max;
  *emin = min;
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPSetUp_Chebyshev(KSP ksp)
{
  KSP_Chebyshev   *cheb = (KSP_Chebyshev *)ksp->data;
  PetscBool        isset, flg;
  Mat              Pmat, Amat;
  PetscObjectId    amatid, pmatid;
  PetscObjectState amatstate, pmatstate;

  PetscFunctionBegin;
  PetscCall(KSPSetWorkVecs(ksp, 3));
  if (cheb->emin == 0. || cheb->emax == 0.) { // User did not specify eigenvalues
    PC pc;
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PetscObjectTypeCompare((PetscObject)pc, PCJACOBI, &flg));
    if (!flg) { // Provided estimates are only relevant for Jacobi
      cheb->emax_provided = 0;
      cheb->emin_provided = 0;
    }
    if (!cheb->kspest) { /* We need to estimate eigenvalues */
      PetscCall(KSPChebyshevEstEigSet(ksp, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE));
    }
  }
  if (cheb->kspest) {
    PetscCall(KSPGetOperators(ksp, &Amat, &Pmat));
    PetscCall(MatIsSPDKnown(Pmat, &isset, &flg));
    if (isset && flg) {
      const char *prefix;
      PetscCall(KSPGetOptionsPrefix(cheb->kspest, &prefix));
      PetscCall(PetscOptionsHasName(NULL, prefix, "-ksp_type", &flg));
      if (!flg) PetscCall(KSPSetType(cheb->kspest, KSPCG));
    }
    PetscCall(PetscObjectGetId((PetscObject)Amat, &amatid));
    PetscCall(PetscObjectGetId((PetscObject)Pmat, &pmatid));
    PetscCall(PetscObjectStateGet((PetscObject)Amat, &amatstate));
    PetscCall(PetscObjectStateGet((PetscObject)Pmat, &pmatstate));
    if (amatid != cheb->amatid || pmatid != cheb->pmatid || amatstate != cheb->amatstate || pmatstate != cheb->pmatstate) {
      PetscReal          max = 0.0, min = 0.0;
      Vec                B;
      KSPConvergedReason reason;
      PetscCall(KSPSetPC(cheb->kspest, ksp->pc));
      if (cheb->usenoisy) {
        B = ksp->work[1];
        PetscCall(KSPSetNoisy_Private(B));
      } else {
        PetscBool change;

        PetscCheck(ksp->vec_rhs, PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "Chebyshev must use a noisy right hand side to estimate the eigenvalues when no right hand side is available");
        PetscCall(PCPreSolveChangeRHS(ksp->pc, &change));
        if (change) {
          B = ksp->work[1];
          PetscCall(VecCopy(ksp->vec_rhs, B));
        } else B = ksp->vec_rhs;
      }
      PetscCall(KSPSolve(cheb->kspest, B, ksp->work[0]));
      PetscCall(KSPGetConvergedReason(cheb->kspest, &reason));
      if (reason == KSP_DIVERGED_ITS) {
        PetscCall(PetscInfo(ksp, "Eigen estimator ran for prescribed number of iterations\n"));
      } else if (reason == KSP_DIVERGED_PC_FAILED) {
        PetscInt       its;
        PCFailedReason pcreason;

        PetscCall(KSPGetIterationNumber(cheb->kspest, &its));
        if (ksp->normtype == KSP_NORM_NONE) {
          PetscInt sendbuf, recvbuf;
          PetscCall(PCGetFailedReasonRank(ksp->pc, &pcreason));
          sendbuf = (PetscInt)pcreason;
          PetscCallMPI(MPI_Allreduce(&sendbuf, &recvbuf, 1, MPIU_INT, MPI_MAX, PetscObjectComm((PetscObject)ksp)));
          PetscCall(PCSetFailedReason(ksp->pc, (PCFailedReason)recvbuf));
        }
        PetscCall(PCGetFailedReason(ksp->pc, &pcreason));
        ksp->reason = KSP_DIVERGED_PC_FAILED;
        PetscCall(PetscInfo(ksp, "Eigen estimator failed: %s %s at iteration %" PetscInt_FMT, KSPConvergedReasons[reason], PCFailedReasons[pcreason], its));
        PetscFunctionReturn(0);
      } else if (reason == KSP_CONVERGED_RTOL || reason == KSP_CONVERGED_ATOL) {
        PetscCall(PetscInfo(ksp, "Eigen estimator converged prematurely. Should not happen except for small or low rank problem\n"));
      } else if (reason < 0) {
        PetscCall(PetscInfo(ksp, "Eigen estimator failed %s, using estimates anyway\n", KSPConvergedReasons[reason]));
      }

      PetscCall(KSPChebyshevComputeExtremeEigenvalues_Private(cheb->kspest, &min, &max));
      PetscCall(KSPSetPC(cheb->kspest, NULL));

      cheb->emin_computed = min;
      cheb->emax_computed = max;

      cheb->amatid    = amatid;
      cheb->pmatid    = pmatid;
      cheb->amatstate = amatstate;
      cheb->pmatstate = pmatstate;
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPChebyshevGetEigenvalues_Chebyshev(KSP ksp, PetscReal *emax, PetscReal *emin)
{
  KSP_Chebyshev *cheb = (KSP_Chebyshev *)ksp->data;

  PetscFunctionBegin;
  *emax = 0;
  *emin = 0;
  if (cheb->emax != 0.) {
    *emax = cheb->emax;
  } else if (cheb->emax_computed != 0.) {
    *emax = cheb->tform[2] * cheb->emin_computed + cheb->tform[3] * cheb->emax_computed;
  } else if (cheb->emax_provided != 0.) {
    *emax = cheb->tform[2] * cheb->emin_provided + cheb->tform[3] * cheb->emax_provided;
  }
  if (cheb->emin != 0.) {
    *emin = cheb->emin;
  } else if (cheb->emin_computed != 0.) {
    *emin = cheb->tform[0] * cheb->emin_computed + cheb->tform[1] * cheb->emax_computed;
  } else if (cheb->emin_provided != 0.) {
    *emin = cheb->tform[0] * cheb->emin_provided + cheb->tform[1] * cheb->emax_provided;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPChebyshevSetEigenvalues_Chebyshev(KSP ksp, PetscReal emax, PetscReal emin)
{
  KSP_Chebyshev *chebyshevP = (KSP_Chebyshev *)ksp->data;

  PetscFunctionBegin;
  PetscCheck(emax > emin, PetscObjectComm((PetscObject)ksp), PETSC_ERR_ARG_INCOMP, "Maximum eigenvalue must be larger than minimum: max %g min %g", (double)emax, (double)emin);
  PetscCheck(emax * emin > 0.0, PetscObjectComm((PetscObject)ksp), PETSC_ERR_ARG_INCOMP, "Both eigenvalues must be of the same sign: max %g min %g", (double)emax, (double)emin);
  chebyshevP->emax = emax;
  chebyshevP->emin = emin;

  PetscCall(KSPChebyshevEstEigSet(ksp, 0., 0., 0., 0.)); /* Destroy any estimation setup */
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPChebyshevEstEigSet_Chebyshev(KSP ksp, PetscReal a, PetscReal b, PetscReal c, PetscReal d)
{
  KSP_Chebyshev *cheb = (KSP_Chebyshev *)ksp->data;

  PetscFunctionBegin;
  if (a != 0.0 || b != 0.0 || c != 0.0 || d != 0.0) {
    if ((cheb->emin_provided == 0. || cheb->emax_provided == 0.) && !cheb->kspest) { /* should this block of code be moved to KSPSetUp_Chebyshev()? */
      PetscCall(KSPCreate(PetscObjectComm((PetscObject)ksp), &cheb->kspest));
      PetscCall(KSPSetErrorIfNotConverged(cheb->kspest, ksp->errorifnotconverged));
      PetscCall(PetscObjectIncrementTabLevel((PetscObject)cheb->kspest, (PetscObject)ksp, 1));
      /* use PetscObjectSet/AppendOptionsPrefix() instead of KSPSet/AppendOptionsPrefix() so that the PC prefix is not changed */
      PetscCall(PetscObjectSetOptionsPrefix((PetscObject)cheb->kspest, ((PetscObject)ksp)->prefix));
      PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)cheb->kspest, "esteig_"));
      PetscCall(KSPSetSkipPCSetFromOptions(cheb->kspest, PETSC_TRUE));

      PetscCall(KSPSetComputeEigenvalues(cheb->kspest, PETSC_TRUE));

      /* We cannot turn off convergence testing because GMRES will break down if you attempt to keep iterating after a zero norm is obtained */
      PetscCall(KSPSetTolerances(cheb->kspest, 1.e-12, PETSC_DEFAULT, PETSC_DEFAULT, cheb->eststeps));
    }
    if (a >= 0) cheb->tform[0] = a;
    if (b >= 0) cheb->tform[1] = b;
    if (c >= 0) cheb->tform[2] = c;
    if (d >= 0) cheb->tform[3] = d;
    cheb->amatid    = 0;
    cheb->pmatid    = 0;
    cheb->amatstate = -1;
    cheb->pmatstate = -1;
  } else {
    PetscCall(KSPDestroy(&cheb->kspest));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPChebyshevEstEigSetUseNoisy_Chebyshev(KSP ksp, PetscBool use)
{
  KSP_Chebyshev *cheb = (KSP_Chebyshev *)ksp->data;

  PetscFunctionBegin;
  cheb->usenoisy = use;
  PetscFunctionReturn(0);
}

/*@
   KSPChebyshevSetEigenvalues - Sets estimates for the extreme eigenvalues
   of the preconditioned problem.

   Logically Collective on ksp

   Input Parameters:
+  ksp - the Krylov space context
-  emax, emin - the eigenvalue estimates

  Options Database Key:
.  -ksp_chebyshev_eigenvalues emin,emax - extreme eigenvalues

   Notes:
   Call `KSPChebyshevEstEigSet()` or use the option -ksp_chebyshev_esteig a,b,c,d to have the KSP
   estimate the eigenvalues and use these estimated values automatically.

   When `KSPCHEBYSHEV` is used as a smoother, one often wants to target a portion of the spectrum rather than the entire
   spectrum. This function takes the range of target eigenvalues for Chebyshev, which will often slightly over-estimate
   the largest eigenvalue of the actual operator (for safety) and greatly overestimate the smallest eigenvalue to
   improve the smoothing properties of Chebyshev iteration on the higher frequencies in the spectrum.

   Level: intermediate

.seealso: [](chapter_ksp), `KSPCHEBYSHEV`, `KSPChebyshevEstEigSet()`,
@*/
PetscErrorCode KSPChebyshevSetEigenvalues(KSP ksp, PetscReal emax, PetscReal emin)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidLogicalCollectiveReal(ksp, emax, 2);
  PetscValidLogicalCollectiveReal(ksp, emin, 3);
  PetscTryMethod(ksp, "KSPChebyshevSetEigenvalues_C", (KSP, PetscReal, PetscReal), (ksp, emax, emin));
  PetscFunctionReturn(0);
}

/*@
   KSPChebyshevEstEigSet - Automatically estimate the eigenvalues to use for Chebyshev

   Logically Collective on ksp

   Input Parameters:
+  ksp - the Krylov space context
.  a - multiple of min eigenvalue estimate to use for min Chebyshev bound (or PETSC_DECIDE)
.  b - multiple of max eigenvalue estimate to use for min Chebyshev bound (or PETSC_DECIDE)
.  c - multiple of min eigenvalue estimate to use for max Chebyshev bound (or PETSC_DECIDE)
-  d - multiple of max eigenvalue estimate to use for max Chebyshev bound (or PETSC_DECIDE)

  Options Database Key:
.  -ksp_chebyshev_esteig a,b,c,d - estimate eigenvalues using a Krylov method, then use this transform for Chebyshev eigenvalue bounds

   Notes:
   The Chebyshev bounds are set using
.vb
   minbound = a*minest + b*maxest
   maxbound = c*minest + d*maxest
.ve
   The default configuration targets the upper part of the spectrum for use as a multigrid smoother, so only the maximum eigenvalue estimate is used.
   The minimum eigenvalue estimate obtained by Krylov iteration is typically not accurate until the method has converged.

   If 0.0 is passed for all transform arguments (a,b,c,d), eigenvalue estimation is disabled.

   The default transform is (0,0.1; 0,1.1) which targets the "upper" part of the spectrum, as desirable for use with multigrid.

   The eigenvalues are estimated using the Lanczo (`KSPCG`) or Arnoldi (`KSPGMRES`) process using a noisy right hand side vector.

   Level: intermediate

.seealso: [](chapter_ksp), `KSPCHEBYSHEV`, `KSPChebyshevEstEigSet()`, `KSPChebyshevEstEigSetUseNoisy()`, `KSPChebyshevEstEigGetKSP()`
@*/
PetscErrorCode KSPChebyshevEstEigSet(KSP ksp, PetscReal a, PetscReal b, PetscReal c, PetscReal d)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidLogicalCollectiveReal(ksp, a, 2);
  PetscValidLogicalCollectiveReal(ksp, b, 3);
  PetscValidLogicalCollectiveReal(ksp, c, 4);
  PetscValidLogicalCollectiveReal(ksp, d, 5);
  PetscTryMethod(ksp, "KSPChebyshevEstEigSet_C", (KSP, PetscReal, PetscReal, PetscReal, PetscReal), (ksp, a, b, c, d));
  PetscFunctionReturn(0);
}

/*@
   KSPChebyshevEstEigSetUseNoisy - use a noisy right hand side in order to do the estimate instead of the given right hand side

   Logically Collective

   Input Parameters:
+  ksp - linear solver context
-  use - `PETSC_TRUE` to use noisy

   Options Database Key:
.  -ksp_chebyshev_esteig_noisy <true,false> - Use noisy right hand side for estimate

   Note:
    This allegedly works better for multigrid smoothers

  Level: intermediate

.seealso: [](chapter_ksp), `KSPCHEBYSHEV`, `KSPChebyshevEstEigSet()`, `KSPChebyshevEstEigGetKSP()`
@*/
PetscErrorCode KSPChebyshevEstEigSetUseNoisy(KSP ksp, PetscBool use)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscTryMethod(ksp, "KSPChebyshevEstEigSetUseNoisy_C", (KSP, PetscBool), (ksp, use));
  PetscFunctionReturn(0);
}

/*@
  KSPChebyshevEstEigGetKSP - Get the Krylov method context used to estimate eigenvalues for the Chebyshev method.  If
  a Krylov method is not being used for this purpose, NULL is returned.  The reference count of the returned `KSP` is
  not incremented: it should not be destroyed by the user.

  Input Parameters:
. ksp - the Krylov space context

  Output Parameters:
. kspest - the eigenvalue estimation Krylov space context

  Level: advanced

.seealso: [](chapter_ksp), `KSPCHEBYSHEV`, `KSPChebyshevEstEigSet()`
@*/
PetscErrorCode KSPChebyshevEstEigGetKSP(KSP ksp, KSP *kspest)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidPointer(kspest, 2);
  *kspest = NULL;
  PetscTryMethod(ksp, "KSPChebyshevEstEigGetKSP_C", (KSP, KSP *), (ksp, kspest));
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPChebyshevEstEigGetKSP_Chebyshev(KSP ksp, KSP *kspest)
{
  KSP_Chebyshev *cheb = (KSP_Chebyshev *)ksp->data;

  PetscFunctionBegin;
  *kspest = cheb->kspest;
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPSetFromOptions_Chebyshev(KSP ksp, PetscOptionItems *PetscOptionsObject)
{
  KSP_Chebyshev *cheb    = (KSP_Chebyshev *)ksp->data;
  PetscInt       neigarg = 2, nestarg = 4;
  PetscReal      eminmax[2] = {0., 0.};
  PetscReal      tform[4]   = {PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE};
  PetscBool      flgeig, flgest;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "KSP Chebyshev Options");
  PetscCall(PetscOptionsInt("-ksp_chebyshev_esteig_steps", "Number of est steps in Chebyshev", "", cheb->eststeps, &cheb->eststeps, NULL));
  PetscCall(PetscOptionsRealArray("-ksp_chebyshev_eigenvalues", "extreme eigenvalues", "KSPChebyshevSetEigenvalues", eminmax, &neigarg, &flgeig));
  if (flgeig) {
    PetscCheck(neigarg == 2, PetscObjectComm((PetscObject)ksp), PETSC_ERR_ARG_INCOMP, "-ksp_chebyshev_eigenvalues: must specify 2 parameters, min and max eigenvalues");
    PetscCall(KSPChebyshevSetEigenvalues(ksp, eminmax[1], eminmax[0]));
  }
  PetscCall(PetscOptionsRealArray("-ksp_chebyshev_esteig", "estimate eigenvalues using a Krylov method, then use this transform for Chebyshev eigenvalue bounds", "KSPChebyshevEstEigSet", tform, &nestarg, &flgest));
  if (flgest) {
    switch (nestarg) {
    case 0:
      PetscCall(KSPChebyshevEstEigSet(ksp, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE));
      break;
    case 2: /* Base everything on the max eigenvalues */
      PetscCall(KSPChebyshevEstEigSet(ksp, PETSC_DECIDE, tform[0], PETSC_DECIDE, tform[1]));
      break;
    case 4: /* Use the full 2x2 linear transformation */
      PetscCall(KSPChebyshevEstEigSet(ksp, tform[0], tform[1], tform[2], tform[3]));
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)ksp), PETSC_ERR_ARG_INCOMP, "Must specify either 0, 2, or 4 parameters for eigenvalue estimation");
    }
  }

  /* We need to estimate eigenvalues; need to set this here so that KSPSetFromOptions() is called on the estimator */
  if ((cheb->emin == 0. || cheb->emax == 0.) && !cheb->kspest) PetscCall(KSPChebyshevEstEigSet(ksp, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE));

  if (cheb->kspest) {
    PetscCall(PetscOptionsBool("-ksp_chebyshev_esteig_noisy", "Use noisy right hand side for estimate", "KSPChebyshevEstEigSetUseNoisy", cheb->usenoisy, &cheb->usenoisy, NULL));
    PetscCall(KSPSetFromOptions(cheb->kspest));
  }
  PetscOptionsHeadEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPSolve_Chebyshev(KSP ksp)
{
  PetscInt    k, kp1, km1, ktmp, i;
  PetscScalar alpha, omegaprod, mu, omega, Gamma, c[3], scale;
  PetscReal   rnorm = 0.0, emax, emin;
  Vec         sol_orig, b, p[3], r;
  Mat         Amat, Pmat;
  PetscBool   diagonalscale;

  PetscFunctionBegin;
  PetscCall(PCGetDiagonalScale(ksp->pc, &diagonalscale));
  PetscCheck(!diagonalscale, PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "Krylov method %s does not support diagonal scaling", ((PetscObject)ksp)->type_name);

  PetscCall(PCGetOperators(ksp->pc, &Amat, &Pmat));
  PetscCall(PetscObjectSAWsTakeAccess((PetscObject)ksp));
  ksp->its = 0;
  PetscCall(PetscObjectSAWsGrantAccess((PetscObject)ksp));
  /* These three point to the three active solutions, we
     rotate these three at each solution update */
  km1      = 0;
  k        = 1;
  kp1      = 2;
  sol_orig = ksp->vec_sol; /* ksp->vec_sol will be assigned to rotating vector p[k], thus save its address */
  b        = ksp->vec_rhs;
  p[km1]   = sol_orig;
  p[k]     = ksp->work[0];
  p[kp1]   = ksp->work[1];
  r        = ksp->work[2];

  PetscCall(KSPChebyshevGetEigenvalues_Chebyshev(ksp, &emax, &emin));
  /* use scale*B as our preconditioner */
  scale = 2.0 / (emax + emin);

  /*   -alpha <=  scale*lambda(B^{-1}A) <= alpha   */
  alpha     = 1.0 - scale * emin;
  Gamma     = 1.0;
  mu        = 1.0 / alpha;
  omegaprod = 2.0 / alpha;

  c[km1] = 1.0;
  c[k]   = mu;

  if (!ksp->guess_zero) {
    PetscCall(KSP_MatMult(ksp, Amat, sol_orig, r)); /*  r = b - A*p[km1] */
    PetscCall(VecAYPX(r, -1.0, b));
  } else {
    PetscCall(VecCopy(b, r));
  }

  /* calculate residual norm if requested, we have done one iteration */
  if (ksp->normtype) {
    switch (ksp->normtype) {
    case KSP_NORM_PRECONDITIONED:
      PetscCall(KSP_PCApply(ksp, r, p[k])); /* p[k] = B^{-1}r */
      PetscCall(VecNorm(p[k], NORM_2, &rnorm));
      break;
    case KSP_NORM_UNPRECONDITIONED:
    case KSP_NORM_NATURAL:
      PetscCall(VecNorm(r, NORM_2, &rnorm));
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "%s", KSPNormTypes[ksp->normtype]);
    }
    PetscCall(PetscObjectSAWsTakeAccess((PetscObject)ksp));
    ksp->rnorm = rnorm;
    PetscCall(PetscObjectSAWsGrantAccess((PetscObject)ksp));
    PetscCall(KSPLogResidualHistory(ksp, rnorm));
    PetscCall(KSPLogErrorHistory(ksp));
    PetscCall(KSPMonitor(ksp, 0, rnorm));
    PetscCall((*ksp->converged)(ksp, 0, rnorm, &ksp->reason, ksp->cnvP));
  } else ksp->reason = KSP_CONVERGED_ITERATING;
  if (ksp->reason || ksp->max_it == 0) {
    if (ksp->max_it == 0) ksp->reason = KSP_DIVERGED_ITS; /* This for a V(0,x) cycle */
    PetscFunctionReturn(0);
  }
  if (ksp->normtype != KSP_NORM_PRECONDITIONED) { PetscCall(KSP_PCApply(ksp, r, p[k])); /* p[k] = B^{-1}r */ }
  PetscCall(VecAYPX(p[k], scale, p[km1])); /* p[k] = scale B^{-1}r + p[km1] */
  PetscCall(PetscObjectSAWsTakeAccess((PetscObject)ksp));
  ksp->its = 1;
  PetscCall(PetscObjectSAWsGrantAccess((PetscObject)ksp));

  for (i = 1; i < ksp->max_it; i++) {
    PetscCall(PetscObjectSAWsTakeAccess((PetscObject)ksp));
    ksp->its++;
    PetscCall(PetscObjectSAWsGrantAccess((PetscObject)ksp));

    PetscCall(KSP_MatMult(ksp, Amat, p[k], r)); /*  r = b - Ap[k]    */
    PetscCall(VecAYPX(r, -1.0, b));
    /* calculate residual norm if requested */
    if (ksp->normtype) {
      switch (ksp->normtype) {
      case KSP_NORM_PRECONDITIONED:
        PetscCall(KSP_PCApply(ksp, r, p[kp1])); /*  p[kp1] = B^{-1}r  */
        PetscCall(VecNorm(p[kp1], NORM_2, &rnorm));
        break;
      case KSP_NORM_UNPRECONDITIONED:
      case KSP_NORM_NATURAL:
        PetscCall(VecNorm(r, NORM_2, &rnorm));
        break;
      default:
        rnorm = 0.0;
        break;
      }
      KSPCheckNorm(ksp, rnorm);
      PetscCall(PetscObjectSAWsTakeAccess((PetscObject)ksp));
      ksp->rnorm = rnorm;
      PetscCall(PetscObjectSAWsGrantAccess((PetscObject)ksp));
      PetscCall(KSPLogResidualHistory(ksp, rnorm));
      PetscCall(KSPMonitor(ksp, i, rnorm));
      PetscCall((*ksp->converged)(ksp, i, rnorm, &ksp->reason, ksp->cnvP));
      if (ksp->reason) break;
      if (ksp->normtype != KSP_NORM_PRECONDITIONED) { PetscCall(KSP_PCApply(ksp, r, p[kp1])); /*  p[kp1] = B^{-1}r  */ }
    } else {
      PetscCall(KSP_PCApply(ksp, r, p[kp1])); /*  p[kp1] = B^{-1}r  */
    }
    ksp->vec_sol = p[k];
    PetscCall(KSPLogErrorHistory(ksp));

    c[kp1] = 2.0 * mu * c[k] - c[km1];
    omega  = omegaprod * c[k] / c[kp1];

    /* y^{k+1} = omega(y^{k} - y^{k-1} + Gamma*r^{k}) + y^{k-1} */
    PetscCall(VecAXPBYPCZ(p[kp1], 1.0 - omega, omega, omega * Gamma * scale, p[km1], p[k]));

    ktmp = km1;
    km1  = k;
    k    = kp1;
    kp1  = ktmp;
  }
  if (!ksp->reason) {
    if (ksp->normtype) {
      PetscCall(KSP_MatMult(ksp, Amat, p[k], r)); /*  r = b - Ap[k]    */
      PetscCall(VecAYPX(r, -1.0, b));
      switch (ksp->normtype) {
      case KSP_NORM_PRECONDITIONED:
        PetscCall(KSP_PCApply(ksp, r, p[kp1])); /* p[kp1] = B^{-1}r */
        PetscCall(VecNorm(p[kp1], NORM_2, &rnorm));
        break;
      case KSP_NORM_UNPRECONDITIONED:
      case KSP_NORM_NATURAL:
        PetscCall(VecNorm(r, NORM_2, &rnorm));
        break;
      default:
        rnorm = 0.0;
        break;
      }
      KSPCheckNorm(ksp, rnorm);
      PetscCall(PetscObjectSAWsTakeAccess((PetscObject)ksp));
      ksp->rnorm = rnorm;
      PetscCall(PetscObjectSAWsGrantAccess((PetscObject)ksp));
      PetscCall(KSPLogResidualHistory(ksp, rnorm));
      PetscCall(KSPMonitor(ksp, i, rnorm));
    }
    if (ksp->its >= ksp->max_it) {
      if (ksp->normtype != KSP_NORM_NONE) {
        PetscCall((*ksp->converged)(ksp, i, rnorm, &ksp->reason, ksp->cnvP));
        if (!ksp->reason) ksp->reason = KSP_DIVERGED_ITS;
      } else ksp->reason = KSP_CONVERGED_ITS;
    }
  }

  /* make sure solution is in vector x */
  ksp->vec_sol = sol_orig;
  if (k) PetscCall(VecCopy(p[k], sol_orig));
  if (ksp->reason == KSP_CONVERGED_ITS) PetscCall(KSPLogErrorHistory(ksp));
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPView_Chebyshev(KSP ksp, PetscViewer viewer)
{
  KSP_Chebyshev *cheb = (KSP_Chebyshev *)ksp->data;
  PetscBool      iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    PetscReal emax, emin;
    PetscCall(KSPChebyshevGetEigenvalues_Chebyshev(ksp, &emax, &emin));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  eigenvalue targets used: min %g, max %g\n", (double)emin, (double)emax));
    if (cheb->kspest) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "  eigenvalues estimated via %s: min %g, max %g\n", ((PetscObject)(cheb->kspest))->type_name, (double)cheb->emin_computed, (double)cheb->emax_computed));
      PetscCall(PetscViewerASCIIPrintf(viewer, "  eigenvalues estimated using %s with transform: [%g %g; %g %g]\n", ((PetscObject)cheb->kspest)->type_name, (double)cheb->tform[0], (double)cheb->tform[1], (double)cheb->tform[2], (double)cheb->tform[3]));
      PetscCall(PetscViewerASCIIPushTab(viewer));
      PetscCall(KSPView(cheb->kspest, viewer));
      PetscCall(PetscViewerASCIIPopTab(viewer));
      if (cheb->usenoisy) PetscCall(PetscViewerASCIIPrintf(viewer, "  estimating eigenvalues using noisy right hand side\n"));
    } else if (cheb->emax_provided != 0.) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "  eigenvalues provided (min %g, max %g) with transform: [%g %g; %g %g]\n", (double)cheb->emin_provided, (double)cheb->emax_provided, (double)cheb->tform[0], (double)cheb->tform[1], (double)cheb->tform[2],
                                       (double)cheb->tform[3]));
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPDestroy_Chebyshev(KSP ksp)
{
  KSP_Chebyshev *cheb = (KSP_Chebyshev *)ksp->data;

  PetscFunctionBegin;
  PetscCall(KSPDestroy(&cheb->kspest));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPChebyshevSetEigenvalues_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPChebyshevEstEigSet_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPChebyshevEstEigSetUseNoisy_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPChebyshevEstEigGetKSP_C", NULL));
  PetscCall(KSPDestroyDefault(ksp));
  PetscFunctionReturn(0);
}

/*MC
     KSPCHEBYSHEV - The preconditioned Chebyshev iterative method

   Options Database Keys:
+   -ksp_chebyshev_eigenvalues <emin,emax> - set approximations to the smallest and largest eigenvalues
                  of the preconditioned operator. If these are accurate you will get much faster convergence.
.   -ksp_chebyshev_esteig <a,b,c,d> - estimate eigenvalues using a Krylov method, then use this
                         transform for Chebyshev eigenvalue bounds (`KSPChebyshevEstEigSet()`)
.   -ksp_chebyshev_esteig_steps - number of estimation steps
-   -ksp_chebyshev_esteig_noisy - use noisy number generator to create right hand side for eigenvalue estimator

   Level: beginner

   Notes:
   The Chebyshev method requires both the matrix and preconditioner to be symmetric positive (semi) definite, but it can work as a smoother in other situations

   Only support for left preconditioning.

   Chebyshev is configured as a smoother by default, targetting the "upper" part of the spectrum.

   The user should call `KSPChebyshevSetEigenvalues()` to get eigenvalue estimates.

.seealso: [](chapter_ksp), `KSPCreate()`, `KSPSetType()`, `KSPType`, `KSP`,
          `KSPChebyshevSetEigenvalues()`, `KSPChebyshevEstEigSet()`, `KSPChebyshevEstEigSetUseNoisy()`
          `KSPRICHARDSON`, `KSPCG`, `PCMG`
M*/

PETSC_EXTERN PetscErrorCode KSPCreate_Chebyshev(KSP ksp)
{
  KSP_Chebyshev *chebyshevP;

  PetscFunctionBegin;
  PetscCall(PetscNew(&chebyshevP));

  ksp->data = (void *)chebyshevP;
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_PRECONDITIONED, PC_LEFT, 3));
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_UNPRECONDITIONED, PC_LEFT, 2));
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_NONE, PC_LEFT, 1));
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_NONE, PC_RIGHT, 1));

  chebyshevP->emin = 0.;
  chebyshevP->emax = 0.;

  chebyshevP->tform[0] = 0.0;
  chebyshevP->tform[1] = 0.1;
  chebyshevP->tform[2] = 0;
  chebyshevP->tform[3] = 1.1;
  chebyshevP->eststeps = 10;
  chebyshevP->usenoisy = PETSC_TRUE;
  ksp->setupnewmatrix  = PETSC_TRUE;

  ksp->ops->setup          = KSPSetUp_Chebyshev;
  ksp->ops->solve          = KSPSolve_Chebyshev;
  ksp->ops->destroy        = KSPDestroy_Chebyshev;
  ksp->ops->buildsolution  = KSPBuildSolutionDefault;
  ksp->ops->buildresidual  = KSPBuildResidualDefault;
  ksp->ops->setfromoptions = KSPSetFromOptions_Chebyshev;
  ksp->ops->view           = KSPView_Chebyshev;
  ksp->ops->reset          = KSPReset_Chebyshev;

  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPChebyshevSetEigenvalues_C", KSPChebyshevSetEigenvalues_Chebyshev));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPChebyshevEstEigSet_C", KSPChebyshevEstEigSet_Chebyshev));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPChebyshevEstEigSetUseNoisy_C", KSPChebyshevEstEigSetUseNoisy_Chebyshev));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPChebyshevEstEigGetKSP_C", KSPChebyshevEstEigGetKSP_Chebyshev));
  PetscFunctionReturn(0);
}
