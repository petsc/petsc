#include <petsc/private/kspimpl.h> /*I "petscksp.h" I*/

typedef struct {
  PetscInt     s;     /* shadow space dimension; default 4 */
  Vec         *GG;    /* s direction vectors G[0..s-1] */
  Vec         *UU;    /* s update vectors   U[0..s-1] */
  Vec         *PP;    /* s shadow vectors   P[0..s-1] (fixed, random orthonormal) */
  Vec          r;     /* current residual */
  Vec          v;     /* work vector */
  Vec          t;     /* work vector (preconditioned operator result) */
  Vec          guess; /* saved initial guess, used with right preconditioning */
  PetscScalar *M;     /* s*s matrix M[j,k] = <G[k],P[j]>, column-major */
  PetscScalar *f;     /* length s: P^T r */
  PetscScalar *c;     /* length s: solution of M c = f */
  PetscReal    cth;   /* omega stabilization threshold (0 = off, default 0.7) */
  PetscRandom  rand;  /* random context to initialize shadow vectors */
} KSP_IDR;

/*
   KSPIDRInitShadowSpace_IDR - Fill shadow space P[0..s-1] with random
   orthonormal vectors (modified Gram-Schmidt). Called from KSPSetUp_IDR().
*/
static PetscErrorCode KSPIDRInitShadowSpace_IDR(KSP ksp)
{
  KSP_IDR    *idr = (KSP_IDR *)ksp->data;
  PetscScalar dot;
  PetscInt    k, j;

  PetscFunctionBegin;
  PetscCall(KSPIDRGetRandom(ksp, &idr->rand));
  for (k = 0; k < idr->s; k++) PetscCall(VecSetRandom(idr->PP[k], idr->rand));
  /* Modified Gram-Schmidt orthonormalization */
  for (k = 0; k < idr->s; k++) {
    PetscCall(VecNormalize(idr->PP[k], NULL));
    for (j = k + 1; j < idr->s; j++) {
      PetscCall(VecDot(idr->PP[j], idr->PP[k], &dot));
      PetscCall(VecAXPY(idr->PP[j], -dot, idr->PP[k]));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   KSPSetUp_IDR - Allocate the (3s+3) work vectors and the s*s+2s scalar
   arrays, then initialize the shadow space P.
*/
static PetscErrorCode KSPSetUp_IDR(KSP ksp)
{
  KSP_IDR *idr = (KSP_IDR *)ksp->data;

  PetscFunctionBegin;
  PetscCall(KSPSetWorkVecs(ksp, 3 + 3 * idr->s));
  idr->r  = ksp->work[0];
  idr->v  = ksp->work[1];
  idr->t  = ksp->work[2];
  idr->GG = ksp->work + 3;
  idr->UU = ksp->work + 3 + idr->s;
  idr->PP = ksp->work + 3 + 2 * idr->s;
  if (idr->M) PetscCall(PetscFree3(idr->M, idr->f, idr->c));
  PetscCall(PetscMalloc3(idr->s * idr->s, &idr->M, idr->s, &idr->f, idr->s, &idr->c));
  PetscCall(KSPIDRInitShadowSpace_IDR(ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   KSPSolve_IDR - IDR(s) biorthogonal solve kernel.

   This implements the biorthogonal IDR(s) recurrence (Algorithm 2 of
   {cite}`gijzen:2011`) applied to the
   preconditioned operator A' (= K^{-1}A for left, AK^{-1} for right
   preconditioning, evaluated by KSP_PCApplyBAorAB()). Working on A'
   keeps x and r consistent through left/right preconditioning and makes
   IDR(1) reduce to BiCGSTAB.

   The s-by-s matrix M (column-major, M[i + j*s] = p_i^H g_j) is kept
   lower triangular: it is initialized to the identity and only its lower
   part is updated, so the small system M[k:s-1,k:s-1] c = f[k:s-1] is a
   forward substitution.
*/
static PetscErrorCode KSPSolve_IDR(KSP ksp)
{
  KSP_IDR     *idr = (KSP_IDR *)ksp->data;
  PetscInt     s   = idr->s, i, j, k;
  PetscScalar *M = idr->M, *f = idr->f, *c = idr->c;
  PetscScalar  alpha, beta, om, tr, sum;
  PetscReal    dp = 0.0, nr, nt, rho;
  Vec          X, B, R, V, T;
  Vec         *G = idr->GG, *U = idr->UU, *P = idr->PP;
  PetscBool    diagonalscale;

  PetscFunctionBegin;
  PetscCall(PCGetDiagonalScale(ksp->pc, &diagonalscale));
  PetscCheck(!diagonalscale, PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "Krylov method %s does not support diagonal scaling", ((PetscObject)ksp)->type_name);

  X  = ksp->vec_sol;
  B  = ksp->vec_rhs;
  R  = idr->r;
  V  = idr->v;
  T  = idr->t;
  nr = 1.0;

  /* Compute initial (preconditioned for left PC) residual R */
  PetscCall(KSPInitialResidual(ksp, X, V, T, R, B));

  if (ksp->pc_side == PC_RIGHT && !ksp->guess_zero) {
    if (!idr->guess) PetscCall(VecDuplicate(X, &idr->guess));
    PetscCall(VecCopy(X, idr->guess));
    PetscCall(VecSet(X, 0.0));
  }

  if (ksp->normtype != KSP_NORM_NONE) {
    PetscCall(VecNorm(R, NORM_2, &dp));
    KSPCheckNorm(ksp, dp);
    nr = dp;
  }
  PetscCall(PetscObjectSAWsTakeAccess((PetscObject)ksp));
  ksp->its   = 0;
  ksp->rnorm = dp;
  PetscCall(PetscObjectSAWsGrantAccess((PetscObject)ksp));
  PetscCall(KSPLogResidualHistory(ksp, dp));
  PetscCall(KSPMonitor(ksp, 0, dp));
  PetscCall((*ksp->converged)(ksp, 0, dp, &ksp->reason, ksp->cnvP));

  if (!ksp->reason) {
    /* Initialize the IDR data: G = U = 0, M = I, omega = 1 */
    for (k = 0; k < s; k++) {
      PetscCall(VecSet(G[k], 0.0));
      PetscCall(VecSet(U[k], 0.0));
    }
    PetscCall(PetscArrayzero(M, s * s));
    for (k = 0; k < s; k++) M[k + k * s] = 1.0;
    om = 1.0;

    while (ksp->its < ksp->max_it && !ksp->reason) {
      /* f = P^H r */
      PetscCall(VecMDot(R, s, P, f));

      for (k = 0; k < s; k++) {
        /* Forward substitution: solve lower-triangular M[k:s-1,k:s-1] c[k:s-1] = f[k:s-1] */
        for (i = k; i < s; i++) {
          sum = f[i];
          for (j = k; j < i; j++) sum -= M[i + j * s] * c[j];
          c[i] = sum / M[i + i * s];
        }

        /* v = r - sum_{j=k}^{s-1} c[j] G[j] */
        PetscCall(VecCopy(R, V));
        for (j = k; j < s; j++) c[j] = -c[j];
        PetscCall(VecMAXPY(V, s - k, c + k, G + k));
        for (j = k; j < s; j++) c[j] = -c[j];

        /* U[k] = omega*v + sum_{j=k}^{s-1} c[j] U[j]  (scale U[k] by c[k] first to avoid aliasing) */
        PetscCall(VecAXPBY(U[k], om, c[k], V));
        if (s - k - 1 > 0) PetscCall(VecMAXPY(U[k], s - k - 1, c + k + 1, U + k + 1));

        /* G[k] = A' U[k] */
        PetscCall(KSP_PCApplyBAorAB(ksp, U[k], G[k], T));

        /* Bi-orthogonalize G[k], U[k] against p_0,...,p_{k-1} */
        for (i = 0; i < k; i++) {
          PetscCall(VecDot(G[k], P[i], &alpha));
          alpha /= M[i + i * s];
          PetscCall(VecAXPY(G[k], -alpha, G[i]));
          PetscCall(VecAXPY(U[k], -alpha, U[i]));
        }

        /* Update column k of M: M[k:s-1][k] = P[k:s-1]^H G[k] */
        PetscCall(VecMDot(G[k], s - k, &P[k], &M[k + k * s]));

        if (PetscAbsScalar(M[k + k * s]) < 10 * PETSC_MACHINE_EPSILON * nr) {
          PetscCheck(!ksp->errorifnotconverged, PetscObjectComm((PetscObject)ksp), PETSC_ERR_NOT_CONVERGED, "KSPSolve breakdown due to zero M[k,k] in IDR(s)");
          ksp->reason = KSP_DIVERGED_BREAKDOWN;
          PetscCall(PetscInfo(ksp, "Breakdown in IDR(s) half-step: M[k,k] = 0\n"));
          break;
        }

        /* Make r orthogonal to p_0,...,p_k:  r -= beta g_k,  x += beta u_k */
        beta = f[k] / M[k + k * s];
        PetscCall(VecAXPY(R, -beta, G[k]));
        PetscCall(VecAXPY(X, beta, U[k]));

        /* With right preconditioning: R doubles as both the residual for x_0 and the
           RHS for the shifted system A K^{-1} y = R iterated from y = 0 */
        if (ksp->normtype != KSP_NORM_NONE) {
          PetscCall(VecNorm(R, NORM_2, &dp));
          KSPCheckNorm(ksp, dp);
        }
        PetscCall(PetscObjectSAWsTakeAccess((PetscObject)ksp));
        ksp->its++;
        ksp->rnorm = dp;
        PetscCall(PetscObjectSAWsGrantAccess((PetscObject)ksp));
        PetscCall(KSPLogResidualHistory(ksp, dp));
        PetscCall(KSPMonitor(ksp, ksp->its, dp));
        PetscCall((*ksp->converged)(ksp, ksp->its, dp, &ksp->reason, ksp->cnvP));
        if (ksp->reason || ksp->its >= ksp->max_it) break;

        /* Update remaining shadow projections: f[j] -= beta M[j][k], j > k */
        for (i = k + 1; i < s; i++) f[i] -= beta * M[i + k * s];
      }
      if (ksp->reason || ksp->its >= ksp->max_it) break;

      /* Minimal-residual (omega) step with angle stabilization.
         Batch ||r||, ||t||, (r,t) into a single MPI collective via Begin/End. */
      PetscCall(KSP_PCApplyBAorAB(ksp, R, T, V));
      PetscCall(VecNormBegin(R, NORM_2, &nr));
      PetscCall(VecNormBegin(T, NORM_2, &nt));
      PetscCall(VecDotBegin(R, T, &tr));
      PetscCall(VecNormEnd(R, NORM_2, &nr));
      PetscCall(VecNormEnd(T, NORM_2, &nt));
      PetscCall(VecDotEnd(R, T, &tr));
      if (nt == 0.0) {
        PetscCheck(!ksp->errorifnotconverged, PetscObjectComm((PetscObject)ksp), PETSC_ERR_NOT_CONVERGED, "KSPSolve breakdown: zero ||A'r|| in IDR(s) omega step");
        ksp->reason = KSP_DIVERGED_BREAKDOWN;
        PetscCall(PetscInfo(ksp, "Breakdown in IDR(s) omega step: ||A'r|| = 0\n"));
        break;
      }
      om = tr / (nt * nt);
      if (idr->cth > 0.0) { /* 0 is a flag */
        rho = PetscAbsScalar(tr) / (nt * nr);
        if (rho < idr->cth) om *= idr->cth / rho;
      }
      PetscCall(VecAXPY(X, om, R));
      PetscCall(VecAXPY(R, -om, T));

      if (ksp->normtype != KSP_NORM_NONE) {
        PetscCall(VecNorm(R, NORM_2, &dp));
        KSPCheckNorm(ksp, dp);
      }
      PetscCall(PetscObjectSAWsTakeAccess((PetscObject)ksp));
      ksp->its++;
      ksp->rnorm = dp;
      PetscCall(PetscObjectSAWsGrantAccess((PetscObject)ksp));
      PetscCall(KSPLogResidualHistory(ksp, dp));
      PetscCall(KSPMonitor(ksp, ksp->its, dp));
      PetscCall((*ksp->converged)(ksp, ksp->its, dp, &ksp->reason, ksp->cnvP));
    }
    if (!ksp->reason && ksp->its >= ksp->max_it) ksp->reason = KSP_DIVERGED_ITS;
  }

  /* Recover the true solution: unwind right preconditioning and add back guess */
  PetscCall(KSPUnwindPreconditioner(ksp, X, T));
  if (ksp->pc_side == PC_RIGHT && !ksp->guess_zero) PetscCall(VecAXPY(X, 1.0, idr->guess));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPReset_IDR(KSP ksp)
{
  KSP_IDR *idr = (KSP_IDR *)ksp->data;

  PetscFunctionBegin;
  PetscCall(VecDestroy(&idr->guess));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPDestroy_IDR(KSP ksp)
{
  KSP_IDR *idr = (KSP_IDR *)ksp->data;

  PetscFunctionBegin;
  PetscCall(KSPReset_IDR(ksp));
  PetscCall(PetscRandomDestroy(&idr->rand));
  PetscCall(PetscFree3(idr->M, idr->f, idr->c));
  PetscCall(KSPDestroyDefault(ksp));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPIDRSetS_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPIDRGetS_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPIDRSetCosine_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPIDRGetCosine_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPIDRSetRandom_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPIDRGetRandom_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPView_IDR(KSP ksp, PetscViewer viewer)
{
  KSP_IDR  *idr = (KSP_IDR *)ksp->data;
  PetscBool isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "  s (shadow space dimension) = %" PetscInt_FMT "\n", idr->s));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  omega stabilization cosine threshold = %g\n", (double)idr->cth));
    if (idr->rand) {
      PetscCall(PetscViewerASCIIPushTab(viewer));
      PetscCall(PetscRandomView(idr->rand, viewer));
      PetscCall(PetscViewerASCIIPopTab(viewer));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPSetFromOptions_IDR(KSP ksp, PetscOptionItems PetscOptionsObject)
{
  KSP_IDR  *idr = (KSP_IDR *)ksp->data;
  PetscReal cth;
  PetscInt  s;
  PetscBool flg;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "KSP IDR(s) options");
  PetscCall(PetscOptionsBoundedInt("-ksp_idr_s", "Shadow space dimension", "KSPIDRSetS", idr->s, &s, &flg, 1));
  if (flg) PetscCall(KSPIDRSetS(ksp, s));
  PetscCall(PetscOptionsRangeReal("-ksp_idr_cosine", "Omega stabilization cosine threshold (0 = off)", "KSPIDRSetCosine", idr->cth, &cth, &flg, 0.0, 1.0));
  if (flg) PetscCall(KSPIDRSetCosine(ksp, cth));
  PetscOptionsHeadEnd();
  PetscCall(KSPIDRGetRandom(ksp, &idr->rand));
  PetscCall(PetscRandomSetFromOptions(idr->rand));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPIDRSetS_IDR(KSP ksp, PetscInt s)
{
  KSP_IDR *idr = (KSP_IDR *)ksp->data;

  PetscFunctionBegin;
  PetscCheck(s >= 1, PetscObjectComm((PetscObject)ksp), PETSC_ERR_ARG_OUTOFRANGE, "Shadow space dimension s must be >= 1, got %" PetscInt_FMT, s);
  if (idr->s != s) {
    idr->s          = s;
    ksp->setupstage = KSP_SETUP_NEW;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPIDRGetS_IDR(KSP ksp, PetscInt *s)
{
  KSP_IDR *idr = (KSP_IDR *)ksp->data;

  PetscFunctionBegin;
  *s = idr->s;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPIDRSetCosine_IDR(KSP ksp, PetscReal cth)
{
  KSP_IDR *idr = (KSP_IDR *)ksp->data;

  PetscFunctionBegin;
  PetscCheck(cth >= 0.0 && cth < 1.0, PetscObjectComm((PetscObject)ksp), PETSC_ERR_ARG_OUTOFRANGE, "Omega stabilization cosine threshold must be in [0,1), got %g", (double)cth);
  idr->cth = cth;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPIDRGetCosine_IDR(KSP ksp, PetscReal *cth)
{
  KSP_IDR *idr = (KSP_IDR *)ksp->data;

  PetscFunctionBegin;
  *cth = idr->cth;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPIDRSetRandom_IDR(KSP ksp, PetscRandom rand)
{
  KSP_IDR *idr = (KSP_IDR *)ksp->data;

  PetscFunctionBegin;
  PetscCall(PetscObjectReference((PetscObject)rand));
  PetscCall(PetscRandomDestroy(&idr->rand));
  idr->rand = rand;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPIDRGetRandom_IDR(KSP ksp, PetscRandom *rand)
{
  KSP_IDR *idr = (KSP_IDR *)ksp->data;

  PetscFunctionBegin;
  if (!idr->rand) {
    PetscCall(PetscRandomCreate(PetscObjectComm((PetscObject)ksp), &idr->rand));
    PetscCall(PetscObjectIncrementTabLevel((PetscObject)idr->rand, (PetscObject)ksp, 1));
    PetscCall(PetscRandomSetOptionsPrefix(idr->rand, ((PetscObject)ksp)->prefix));
    PetscCall(PetscRandomAppendOptionsPrefix(idr->rand, "ksp_idr_"));
    PetscCall(PetscObjectSetOptions((PetscObject)idr->rand, ((PetscObject)ksp)->options));
  }
  *rand = idr->rand;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPIDRSetS - Sets the shadow space dimension s for the `KSPIDR` solver.

  Logically Collective

  Input Parameters:
+ ksp - the Krylov solver context
- s   - shadow space dimension (default 4); must be >= 1

  Options Database Key:
. -ksp_idr_s s - shadow space dimension

  Level: intermediate

  Notes:
  Increasing `s` generally improves convergence but requires `s` additional
  vectors. If `s` is changed after `KSPSetUp()` has been called, the solver
  is reset automatically.

.seealso: [](ch_ksp), `KSPIDR`, `KSPIDRGetS()`, `KSPIDRSetRandom()`
@*/
PetscErrorCode KSPIDRSetS(KSP ksp, PetscInt s)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidLogicalCollectiveInt(ksp, s, 2);
  PetscTryMethod(ksp, "KSPIDRSetS_C", (KSP, PetscInt), (ksp, s));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPIDRGetS - Gets the shadow space dimension s used by the `KSPIDR` solver.

  Not Collective

  Input Parameter:
. ksp - the Krylov solver context

  Output Parameter:
. s - the shadow space dimension

  Level: intermediate

.seealso: [](ch_ksp), `KSPIDR`, `KSPIDRSetS()`
@*/
PetscErrorCode KSPIDRGetS(KSP ksp, PetscInt *s)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscAssertPointer(s, 2);
  PetscUseMethod(ksp, "KSPIDRGetS_C", (KSP, PetscInt *), (ksp, s));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPIDRSetCosine - Sets the omega stabilization cosine threshold for the `KSPIDR` solver.

  Logically Collective

  Input Parameters:
+ ksp - the Krylov solver context
- cth - stabilization cosine threshold in [0,1) (default 0.7, 0 = off)

  Options Database Key:
. -ksp_idr_cosine cth - omega stabilization cosine threshold

  Level: intermediate

  Notes:
  When the cosine of the angle between the residual and the preconditioned
  residual drops below this threshold, omega is scaled to prevent the
  near-orthogonality stalling described in {cite}`sleijpen:1993,sleijpen:1995`.
  Setting `cth` to 0 disables stabilization.

.seealso: [](ch_ksp), `KSPIDR`, `KSPIDRGetCosine()`, `KSPIDRSetS()`, `KSPIDRSetRandom()`
@*/
PetscErrorCode KSPIDRSetCosine(KSP ksp, PetscReal cth)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidLogicalCollectiveReal(ksp, cth, 2);
  PetscTryMethod(ksp, "KSPIDRSetCosine_C", (KSP, PetscReal), (ksp, cth));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPIDRGetCosine - Gets the omega stabilization cosine threshold used by the `KSPIDR` solver.

  Not Collective

  Input Parameter:
. ksp - the Krylov solver context

  Output Parameter:
. cth - the stabilization cosine threshold

  Level: intermediate

.seealso: [](ch_ksp), `KSPIDR`, `KSPIDRSetCosine()`, `KSPIDRGetS()`
@*/
PetscErrorCode KSPIDRGetCosine(KSP ksp, PetscReal *cth)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscAssertPointer(cth, 2);
  PetscUseMethod(ksp, "KSPIDRGetCosine_C", (KSP, PetscReal *), (ksp, cth));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPIDRSetRandom - Sets the `PetscRandom` object used by the `KSPIDR` solver
  to initialize the shadow vectors.

  Collective

  Input Parameters:
+ ksp  - the Krylov solver context
- rand - the random number generator context

  Level: advanced

  Note:
  `KSPIDR` creates its own random number generator internally that can be accessed
  with `KSPIDRGetRandom()` and controlled from the options database with the options
  prefix of the `KSP` object.

.seealso: [](ch_ksp), `KSPIDR`, `KSPIDRGetRandom()`, `PetscRandomCreate()`, `KSPIDRSetS()`, `KSPIDRSetCosine()`
@*/
PetscErrorCode KSPIDRSetRandom(KSP ksp, PetscRandom rand)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidHeaderSpecific(rand, PETSC_RANDOM_CLASSID, 2);
  PetscCheckSameComm(ksp, 1, rand, 2);
  PetscTryMethod(ksp, "KSPIDRSetRandom_C", (KSP, PetscRandom), (ksp, rand));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPIDRGetRandom - Gets the `PetscRandom` object used by the `KSPIDR` solver.

  Collective

  Input Parameter:
. ksp - the Krylov solver context

  Output Parameter:
. rand - the random number generator context

  Level: advanced

.seealso: [](ch_ksp), `KSPIDR`, `KSPIDRSetRandom()`, `KSPIDRSetCosine()`, `KSPIDRGetS()`
@*/
PetscErrorCode KSPIDRGetRandom(KSP ksp, PetscRandom *rand)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscAssertPointer(rand, 2);
  PetscUseMethod(ksp, "KSPIDRGetRandom_C", (KSP, PetscRandom *), (ksp, rand));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  KSPIDR - IDR(s): Induced Dimension Reduction method for general nonsymmetric
  linear systems {cite}`gijzen:2011`.

  Options Database Keys:
+ -ksp_idr_s s        - shadow space dimension (default 4); larger `s` improves convergence at the cost of `s` additional vectors
                        and `s` extra inner products per step, see `KSPIDRSetS()`
- -ksp_idr_cosine cth - omega stabilization cosine threshold (default 0.7, 0 = off); prevents near-orthogonality stalling in the minimal-residual omega step

  Level: intermediate

  Notes:
  IDR(s) is a short-recurrence, non-restarting Krylov method for general
  nonsymmetric linear systems. It requires no growing subspace and avoids
  the restart stagnation of `KSPGMRES`. The parameter `s` controls the
  trade-off between memory and convergence speed\: s=1 is mathematically
  equivalent to `KSPBCGS`; s=4 typically converges as fast as
  GMRES(50); s=8 often outperforms GMRES(100).
  Memory usage is (3s+3) vectors plus an s-by-s dense matrix.
  This implements the biorthogonal variant described in {cite}`gijzen:2011`.

  `KSPIDR` uses a `PetscRandom` which may be obtained with `KSPIDRGetRandom()`
  (see also `KSPIDRSetRandom()`). The `PetscRandom` may be controlled from the
  options database with the options prefix of the `KSP` object.

.seealso: [](ch_ksp), `KSPCreate()`, `KSPSetType()`, `KSPType`, `KSP`,
          `KSPBCGS`, `KSPBCGSL`, `KSPGMRES`, `KSPIDRSetS()`, `KSPIDRGetS()`,
          `KSPIDRSetCosine()`, `KSPIDRGetCosine()`, `KSPIDRSetRandom()`, `KSPIDRGetRandom()`
M*/
PETSC_EXTERN PetscErrorCode KSPCreate_IDR(KSP ksp)
{
  KSP_IDR *idr;

  PetscFunctionBegin;
  PetscCall(PetscNew(&idr));
  idr->s    = 4;
  idr->cth  = 0.7;
  ksp->data = (void *)idr;

  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_PRECONDITIONED, PC_LEFT, 3));
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_UNPRECONDITIONED, PC_RIGHT, 2));
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_NONE, PC_LEFT, 1));
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_NONE, PC_RIGHT, 1));

  ksp->ops->setup          = KSPSetUp_IDR;
  ksp->ops->solve          = KSPSolve_IDR;
  ksp->ops->reset          = KSPReset_IDR;
  ksp->ops->destroy        = KSPDestroy_IDR;
  ksp->ops->view           = KSPView_IDR;
  ksp->ops->setfromoptions = KSPSetFromOptions_IDR;
  ksp->ops->buildsolution  = KSPBuildSolutionDefault;
  ksp->ops->buildresidual  = KSPBuildResidualDefault;

  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPIDRSetS_C", KSPIDRSetS_IDR));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPIDRGetS_C", KSPIDRGetS_IDR));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPIDRSetCosine_C", KSPIDRSetCosine_IDR));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPIDRGetCosine_C", KSPIDRGetCosine_IDR));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPIDRSetRandom_C", KSPIDRSetRandom_IDR));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPIDRGetRandom_C", KSPIDRGetRandom_IDR));
  PetscFunctionReturn(PETSC_SUCCESS);
}
