/*
    This file implements the Extended Krylov Subspace Method (EKSM)
    for multiple shifted linear systems.
 */

#include <../src/ksp/ksp/impls/eksm/eksmimpl.h> /*I "petscksp.h" I*/
#include <petscblaslapack.h>

#define EKSM_DELTA_DIRECTIONS 10

static PetscErrorCode KSPReset_EKSM(KSP);

/*
   Compute the default shift varsigma as the centroid of all provided shifts sigma_i
 */
static PetscErrorCode KSPEKSMSetDefaultShift(KSP ksp, PetscInt nshift, PetscScalar *sigma, PetscBool *cmplx)
{
  KSP_EKSM *eksm = (KSP_EKSM *)ksp->data;

  PetscFunctionBegin;
  eksm->shift = 0.0;
  for (PetscInt i = 0; i < nshift; i++) {
    eksm->shift += sigma[i];
    if (SHIFT_IS_COMPLEX(cmplx, i)) { // Add complex conjugate pair
      eksm->shift += sigma[i];
      i++;
    }
  }
  eksm->shift /= nshift;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPSetUp_EKSM(KSP ksp)
{
  PetscInt       hh, hes, tt, sol, rs, max_k;
  PetscBool      ispcnone;
  KSP_EKSM      *eksm = (KSP_EKSM *)ksp->data;
  Mat            Anest, Pnest, A, P;
  Vec            rhs, b;
  Mat_MultiShift actx, pctx;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)ksp->pc, PCNONE, &ispcnone));
  PetscCheck(ispcnone, PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "In KSPEKSM the PC type must be PCNONE; use KSPEKSMGetKSP() to set an appropriate preconditioner");

  PetscCall(KSPReset_EKSM(ksp));
  PetscCall(PCGetOperators(ksp->pc, &Anest, &Pnest));
  MatCheckMultiShift(Anest, &actx);
  if (Pnest != Anest) MatCheckMultiShift(Pnest, &pctx);

  PetscCall(MatCreateVecs(Anest, NULL, &rhs));
  PetscCall(VecNestGetSubVec(rhs, 0, &b));
  PetscCall(KSPEKSMGetKSP(ksp, &eksm->ksps, actx->M ? &eksm->kspm : NULL));
  if (!eksm->shift_set) PetscCall(KSPEKSMSetDefaultShift(ksp, actx->nshift, actx->sigma, actx->cmplx));

  PetscCall(MatMultiShiftBuildShiftedMatrix_Internal(actx->K, eksm->shift, actx->M, actx->str, PETSC_TRUE, &A));
  P = A;
  if (Pnest != Anest) PetscCall(MatMultiShiftBuildShiftedMatrix_Internal(pctx->K, eksm->shift, pctx->M, pctx->str, PETSC_TRUE, &P));

  PetscCall(KSPSetOperators(eksm->ksps, A, P));
  if (ksp->setfromoptionscalled) PetscCall(KSPSetFromOptions(eksm->ksps));
  PetscCall(KSPSetUp(eksm->ksps));
  PetscCall(MatDestroy(&A));
  if (Pnest != Anest) PetscCall(MatDestroy(&P));
  if (actx->M) {
    if (Pnest != Anest) PetscCall(KSPSetOperators(eksm->kspm, actx->M, pctx->M));
    else PetscCall(KSPSetOperators(eksm->kspm, actx->M, actx->M));
    if (ksp->setfromoptionscalled) PetscCall(KSPSetFromOptions(eksm->kspm));
    PetscCall(KSPSetUp(eksm->kspm));
  }

  eksm->factor = 1;
#if !PetscDefined(USE_COMPLEX)
  // check for complex conjugate pairs of shifts
  for (PetscInt k = 0; k < actx->nshift; k++) {
    if (SHIFT_IS_COMPLEX(actx->cmplx, k)) {
      eksm->factor = 2; // the projected matrix is twice as large
      break;
    }
  }
#endif

  max_k     = 2 * ksp->max_it; /* two vectors per iteration */
  eksm->ldh = max_k + 1;
  eksm->ldk = max_k;
  eksm->ldt = eksm->factor * (max_k + 1);

  hh  = eksm->ldh * max_k;
  hes = eksm->ldk * max_k;
  tt  = eksm->ldt * eksm->factor * max_k;
  sol = eksm->ldk * actx->nshift;
  rs  = eksm->ldt;

  eksm->lwork = eksm->ldt + PetscMax(eksm->ldt, actx->nshift);

  PetscCall(PetscCalloc6(hh, &eksm->hh_origin, hes, &eksm->kk_origin, tt, &eksm->tt_origin, sol, &eksm->yy_origin, rs, &eksm->rs_origin, eksm->lwork, &eksm->work));

  /* Allocate array to hold pointers to basis vectors */
  eksm->vecs_allocated = VEC_OFFSET + 2 + max_k;
  PetscCall(PetscMalloc1(eksm->vecs_allocated, &eksm->vecs));
  PetscCall(PetscMalloc1(VEC_OFFSET + 2 + max_k, &eksm->user_work));
  PetscCall(PetscMalloc1(VEC_OFFSET + 2 + max_k, &eksm->mwork_alloc));
  eksm->vv_allocated = VEC_OFFSET + 2 + PetscMin(2 * eksm->delta_allocate, max_k);
  PetscCall(VecDuplicateVecs(b, eksm->vv_allocated, &eksm->user_work[0]));
  eksm->mwork_alloc[0] = eksm->vv_allocated;
  eksm->nwork_alloc    = 1;
  for (PetscInt k = 0; k < eksm->vv_allocated; k++) eksm->vecs[k] = eksm->user_work[0][k];
  PetscCall(VecDestroy(&rhs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   Allocate more work vectors, starting from VEC_VV(2*it)
 */
static PetscErrorCode KSPEKSMGetNewVectors(KSP ksp, PetscInt it)
{
  KSP_EKSM *eksm  = (KSP_EKSM *)ksp->data;
  PetscInt  nwork = eksm->nwork_alloc, nalloc;

  PetscFunctionBegin;
  nalloc = 2 * eksm->delta_allocate;
  /* Adjust the number to allocate to make sure that we don't exceed the
    number of available slots */
  if (2 * it + VEC_OFFSET + nalloc >= eksm->vecs_allocated) nalloc = eksm->vecs_allocated - 2 * it - VEC_OFFSET;
  if (!nalloc) PetscFunctionReturn(PETSC_SUCCESS);

  eksm->vv_allocated += nalloc;
  PetscCall(VecDuplicateVecs(VEC_VV(0), nalloc, &eksm->user_work[nwork]));
  eksm->mwork_alloc[nwork] = nalloc;
  for (PetscInt k = 0; k < nalloc; k++) eksm->vecs[2 * it + VEC_OFFSET + k] = eksm->user_work[nwork][k];
  eksm->nwork_alloc++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   Solve the projected problem for each shift
 */
static PetscErrorCode KSPEKSMProjectedProblem(KSP ksp, PetscInt m, PetscReal *res, PetscInt nshift, PetscScalar *sigma, PetscBool *cmplx)
{
  KSP_EKSM    *eksm = (KSP_EKSM *)ksp->data;
  PetscScalar  sone = 1.0, szero = 0.0;
  PetscScalar *e;
  PetscBLASInt n, n1, n2, n22, one = 1, ldk, ldt, lwork;

  PetscFunctionBegin;
  e = eksm->rs_origin;
  PetscCall(PetscBLASIntCast(eksm->ldk, &ldk));
  PetscCall(PetscBLASIntCast(eksm->ldt, &ldt));
  PetscCall(PetscBLASIntCast(eksm->lwork, &lwork));
  PetscCall(PetscBLASIntCast(m, &n));
  PetscCall(PetscBLASIntCast(m + 1, &n1));
  PetscCall(PetscBLASIntCast(2 * m, &n2));
  PetscCall(PetscBLASIntCast(2 * m + 2, &n22));
  *res = 0.0;
  for (PetscInt k = 0; k < nshift; k++) {
    PetscCall(PetscArrayzero(eksm->tt_origin, eksm->ldt * eksm->factor * m));
    if (!SHIFT_IS_COMPLEX(cmplx, k)) {
      /*
         Usual case: real shift or complex arithmetic, process just one shift.
         z = argmin || (H + sigma*K) z - e ||,  y = K*z
      */
      PetscCall(PetscArrayzero(e, m + 1));
      e[0] = eksm->v0norm;
      /* T = H + sigma K */
      for (PetscInt i = 0; i < m + 1; i++)
        for (PetscInt j = 0; j < m; j++) *TT(i, j) = *HH(i, j);
      for (PetscInt i = 0; i < m; i++)
        for (PetscInt j = 0; j < m; j++) *TT(i, j) += sigma[k] * *KK(i, j);
      /* z = T\e (least-squares) */
      PetscCallLAPACKInfo("LAPACKgels", LAPACKgels_("N", &n1, &n, &one, TT(0, 0), &ldt, e, &ldt, eksm->work, &lwork, &info));
      *res = PetscMax(*res, PetscAbsScalar(e[m]));
      /* y(:,k) = K z */
      PetscCallBLAS("BLASgemv", BLASgemv_("N", &n, &n, &sone, KK(0, 0), &ldk, e, &one, &szero, YY(0, k), &one));
    } else {
      /*
         Complex conjugate pair of shifts in real arithmetic, process two iterations at once.
         Same least squares problem as before, but coefficient matrix is
            [ Re(H + sigma*K) -Im(H + sigma*K ] = [ H + Re(sigma)*K   -Im(sigma)*K   ]
            [ Im(H + sigma*K)  Re(H + sigma*K ]   [   Im(sigma)*K    H + Re(sigma)*K ]
      */
      PetscCall(PetscArrayzero(e, 2 * m + 2));
      e[0] = eksm->v0norm;
      /* form T */
      for (PetscInt i = 0; i < m + 1; i++)
        for (PetscInt j = 0; j < m; j++) {
          *TT(i, j)             = *HH(i, j);
          *TT(i + m + 1, j + m) = *HH(i, j);
        }
      for (PetscInt i = 0; i < m; i++)
        for (PetscInt j = 0; j < m; j++) {
          *TT(i, j) += sigma[k] * *KK(i, j);
          *TT(i, j + m)     = -sigma[k + 1] * *KK(i, j);
          *TT(i + m + 1, j) = sigma[k + 1] * *KK(i, j);
          *TT(i + m + 1, j + m) += sigma[k] * *KK(i, j);
        }
      /* z = T\e (least-squares) */
      PetscCallLAPACKInfo("LAPACKgels", LAPACKgels_("N", &n22, &n2, &one, TT(0, 0), &ldt, e, &ldt, eksm->work, &lwork, &info));
      *res = PetscMax(*res, PetscHypotReal(PetscAbsScalar(e[2 * m]), PetscAbsScalar(e[2 * m + 1])));
      /* y(:,k) = K z */
      PetscCallBLAS("BLASgemv", BLASgemv_("N", &n, &n, &sone, KK(0, 0), &ldk, e, &one, &szero, YY(0, k), &one));
      PetscCallBLAS("BLASgemv", BLASgemv_("N", &n, &n, &sone, KK(0, 0), &ldk, e + m, &one, &szero, YY(0, k + 1), &one));
      k++; // skip next shift, result for conj(sigma) is conj(y)
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   Build the solution vectors for each shift, i.e., subvectors of vec_sol
 */
static PetscErrorCode KSPEKSMBuildSoln(KSP ksp, Vec vec_sol, PetscInt m, PetscBool *cmplx)
{
  KSP_EKSM *eksm = (KSP_EKSM *)ksp->data;
  Vec       x, xi;
  PetscInt  nshift;

  PetscFunctionBegin;
  PetscCall(VecSet(vec_sol, 0.0));
  PetscCall(VecNestGetSize(vec_sol, &nshift));
  for (PetscInt k = 0; k < nshift; k++) {
    // x = V*y
    PetscCall(VecNestGetSubVec(vec_sol, k, &x));
    PetscCall(VecMAXPY(x, m, YY(0, k), &VEC_VV(0)));
    if (SHIFT_IS_COMPLEX(cmplx, k)) { // imaginary part
      PetscCall(VecNestGetSubVec(vec_sol, k + 1, &xi));
      PetscCall(VecMAXPY(xi, m, YY(0, k + 1), &VEC_VV(0)));
      k++; // skip next shift, result for conj(sigma) is conj(x)
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   Stop the outer iteration if one of the internal linear solves failed
 */
static PetscErrorCode KSPEKSMCheckInnerSolve(KSP ksp, KSP inner)
{
  KSPConvergedReason reason;

  PetscFunctionBegin;
  PetscCall(KSPGetConvergedReason(inner, &reason));
  if (reason < 0) {
    PetscCheck(!ksp->errorifnotconverged, PetscObjectComm((PetscObject)ksp), PETSC_ERR_NOT_CONVERGED, "Internal KSPEKSM solve diverged with %s", KSPConvergedReasons[reason]);
    PetscCall(PetscInfo(ksp, "Internal KSPEKSM solve diverged with %s\n", KSPConvergedReasons[reason]));
    ksp->reason = KSP_DIVERGED_INNER_SOLVE_FAILED;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPSolve_EKSM(KSP ksp)
{
  KSP_EKSM      *eksm   = (KSP_EKSM *)ksp->data;
  PetscBool      hapend = PETSC_FALSE;
  PetscReal      res, nrm;
  PetscInt       max_k, m = 0;
  Mat            Anest;
  Vec            b;
  Mat_MultiShift actx;

  PetscFunctionBegin;
  PetscCheck(ksp->transpose_solve == PETSC_FALSE, PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "This solver does not support transpose solve");
  PetscCheck(ksp->guess_zero, PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "This solver does not support nonzero initial guess");
  PetscCall(VecNestGetSubVec(ksp->vec_rhs, 0, &b));
  PetscCall(VecCopy(b, VEC_VV(0)));
  PetscCall(PCGetOperators(ksp->pc, &Anest, NULL));
  MatCheckMultiShift(Anest, &actx);
  if (actx->M) {
    PetscCall(VecCopy(VEC_VV(0), VEC_TEMP));
    PetscCall(KSPSolve(eksm->kspm, VEC_TEMP, VEC_VV(0)));
    PetscCall(KSPEKSMCheckInnerSolve(ksp, eksm->kspm));
    if (ksp->reason) PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(VecNormalize(VEC_VV(0), &eksm->v0norm));

  /*
    Initial residual norm. The initial guess is zero, so this is the norm of the right-hand side of the
    projected least-squares problem: ||b|| when M is the identity and ||M^{-1} b|| otherwise. Using it
    keeps iteration 0 in the same norm as the projected residual reported at later iterations, so the
    relative convergence test compares like with like.
  */
  res = eksm->v0norm;
  KSPCheckNorm(ksp, res);
  PetscCall(PetscObjectSAWsTakeAccess((PetscObject)ksp));
  ksp->its   = 0;
  ksp->rnorm = res;
  PetscCall(PetscObjectSAWsGrantAccess((PetscObject)ksp));

  max_k = eksm->ldh - 1;
  PetscCheck(max_k == 2 * ksp->max_it, PetscObjectComm((PetscObject)ksp), PETSC_ERR_ARG_WRONGSTATE, "KSPSetTolerances() must be called before KSPSetUp() for KSPEKSM, and not between successive calls to KSPSolve()");

  PetscCall(KSPLogResidualHistory(ksp, res));
  PetscCall(KSPLogErrorHistory(ksp));
  PetscCall(KSPMonitor(ksp, ksp->its, res));
  if (!res) {
    ksp->reason = KSP_CONVERGED_ATOL;
    PetscCall(PetscInfo(ksp, "Converged due to zero residual norm on entry\n"));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* check for the convergence */
  PetscCall((*ksp->converged)(ksp, ksp->its, res, &ksp->reason, ksp->cnvP));

  PetscCall(PetscArrayzero(eksm->hh_origin, eksm->ldh * max_k));

  while (!ksp->reason && ksp->its < ksp->max_it) {
    if (eksm->vv_allocated <= 2 * ksp->its + VEC_OFFSET + 2) PetscCall(KSPEKSMGetNewVectors(ksp, ksp->its + 1));

    if (ksp->its == 0) { /* first iteration */
      if (actx->M) PetscCall(MatMult(actx->M, VEC_VV(0), VEC_TEMP));
      else PetscCall(VecCopy(VEC_VV(0), VEC_TEMP));
      *HH(0, 0) = 1.0;
    } else {
      if (actx->M) PetscCall(MatMult(actx->M, VEC_VV(2 * ksp->its - 1), VEC_TEMP));
      else PetscCall(VecCopy(VEC_VV(2 * ksp->its - 1), VEC_TEMP));
      *HH(2 * ksp->its - 1, 2 * ksp->its) = 1.0;
    }
    PetscCall(KSPSolve(eksm->ksps, VEC_TEMP, VEC_VV(2 * ksp->its + 1)));
    PetscCall(KSPEKSMCheckInnerSolve(ksp, eksm->ksps));
    if (ksp->reason) break;
    /* update Hessenberg matrix and do Gram-Schmidt */
    PetscCall((*ksp->orthog)(ksp, &VEC_VV(0), 2 * ksp->its + 1, NULL, KK(0, 2 * ksp->its)));
    if (ksp->reason) break;

    /* vv(i+1) . vv(i+1) */
    PetscCall(VecNormalize(VEC_VV(2 * ksp->its + 1), &nrm));
    KSPCheckNorm(ksp, nrm);

    /* save the magnitude */
    *KK(2 * ksp->its + 1, 2 * ksp->its) = nrm;
    for (PetscInt i = 0; i < 2 * ksp->its + 2; i++) *HH(i, 2 * ksp->its) -= eksm->shift * *KK(i, 2 * ksp->its);

    /* check for the happy breakdown */
    if (nrm <= eksm->haptol) {
      PetscCall(PetscInfo(ksp, "Detected happy breakdown, nrm = %14.12e\n", (double)nrm));
      hapend = PETSC_TRUE;

      m = 2 * ksp->its + 1;
      PetscCall(KSPEKSMProjectedProblem(ksp, m, &res, actx->nshift, actx->sigma, actx->cmplx));
    } else {
      PetscCall(MatMult(actx->K, VEC_VV(2 * ksp->its), VEC_TEMP));
      if (actx->M) {
        PetscCall(KSPSolve(eksm->kspm, VEC_TEMP, VEC_VV(2 * ksp->its + 2)));
        PetscCall(KSPEKSMCheckInnerSolve(ksp, eksm->kspm));
        if (ksp->reason) break;
      } else PetscCall(VecCopy(VEC_TEMP, VEC_VV(2 * ksp->its + 2)));
      *KK(2 * ksp->its, 2 * ksp->its + 1) = 1.0;

      /* update Hessenberg matrix and do Gram-Schmidt */
      PetscCall((*ksp->orthog)(ksp, &VEC_VV(0), 2 * ksp->its + 2, NULL, HH(0, 2 * ksp->its + 1)));
      if (ksp->reason) break;

      /* vv(i+1) . vv(i+1) */
      PetscCall(VecNormalize(VEC_VV(2 * ksp->its + 2), &nrm));
      KSPCheckNorm(ksp, nrm);

      /* save the magnitude */
      *HH(2 * ksp->its + 2, 2 * ksp->its + 1) = nrm;

      /* check for the happy breakdown */
      if (nrm <= eksm->haptol) {
        PetscCall(PetscInfo(ksp, "Detected happy breakdown, nrm = %14.12e\n", (double)nrm));
        hapend = PETSC_TRUE;
      }

      m = 2 * ksp->its + 2;
      PetscCall(KSPEKSMProjectedProblem(ksp, m, &res, actx->nshift, actx->sigma, actx->cmplx));
    }

    ksp->its++;
    ksp->rnorm = res;
    if (ksp->reason) break;

    PetscCall((*ksp->converged)(ksp, ksp->its, res, &ksp->reason, ksp->cnvP));

    /* Catch error in happy breakdown and signal convergence and break from loop */
    if (hapend) {
      if (ksp->normtype == KSP_NORM_NONE) ksp->reason = KSP_CONVERGED_HAPPY_BREAKDOWN;
      else if (!ksp->reason) {
        PetscCheck(!ksp->errorifnotconverged, PetscObjectComm((PetscObject)ksp), PETSC_ERR_NOT_CONVERGED, "Reached happy breakdown, but convergence was not indicated. Residual norm = %g", (double)res);
        ksp->reason = KSP_DIVERGED_BREAKDOWN;
        break;
      }
    }
    PetscCall(KSPLogResidualHistory(ksp, res));
    PetscCall(KSPLogErrorHistory(ksp));
    PetscCall(KSPMonitor(ksp, ksp->its, res));
  }

  /* Form the solution */
  PetscCall(KSPEKSMBuildSoln(ksp, ksp->vec_sol, m, actx->cmplx));

  if (ksp->reason == KSP_CONVERGED_ITERATING && ksp->its >= ksp->max_it) ksp->reason = KSP_DIVERGED_ITS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPReset_EKSM(KSP ksp)
{
  KSP_EKSM *eksm = (KSP_EKSM *)ksp->data;

  PetscFunctionBegin;
  /* Free the Hessenberg matrices */
  PetscCall(PetscFree6(eksm->hh_origin, eksm->kk_origin, eksm->tt_origin, eksm->yy_origin, eksm->rs_origin, eksm->work));

  /* free work vectors */
  PetscCall(PetscFree(eksm->vecs));
  for (PetscInt i = 0; i < eksm->nwork_alloc; i++) PetscCall(VecDestroyVecs(eksm->mwork_alloc[i], &eksm->user_work[i]));
  eksm->nwork_alloc = 0;

  PetscCall(PetscFree(eksm->user_work));
  PetscCall(PetscFree(eksm->mwork_alloc));

  if (eksm->ksps) PetscCall(KSPReset(eksm->ksps));
  if (eksm->kspm) PetscCall(KSPReset(eksm->kspm));

  eksm->vv_allocated   = 0;
  eksm->vecs_allocated = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPDestroy_EKSM(KSP ksp)
{
  KSP_EKSM *eksm = (KSP_EKSM *)ksp->data;

  PetscFunctionBegin;
  PetscCall(KSPReset_EKSM(ksp));
  PetscCall(KSPDestroy(&eksm->ksps));
  PetscCall(KSPDestroy(&eksm->kspm));
  PetscCall(PetscFree(ksp->data));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPEKSMSetHapTol_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPEKSMGetHapTol_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPEKSMSetKSP_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPEKSMGetKSP_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPEKSMSetShift_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPEKSMGetShift_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPView_EKSM(KSP ksp, PetscViewer viewer)
{
  KSP_EKSM   *eksm = (KSP_EKSM *)ksp->data;
  const char *cstr;
  PetscBool   isascii, isstring;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERSTRING, &isstring));
  if (ksp->orthog == KSPOrthogonalizationClassicalGramSchmidt) {
    switch (ksp->cgstype) {
    case KSP_ORTHOGONALIZATION_CGS_REFINE_NEVER:
      cstr = "classical (unmodified) Gram-Schmidt orthogonalization with no iterative refinement";
      break;
    case KSP_ORTHOGONALIZATION_CGS_REFINE_ALWAYS:
      cstr = "classical (unmodified) Gram-Schmidt orthogonalization with one step of iterative refinement";
      break;
    case KSP_ORTHOGONALIZATION_CGS_REFINE_IFNEEDED:
      cstr = "classical (unmodified) Gram-Schmidt orthogonalization with one step of iterative refinement when needed";
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)ksp), PETSC_ERR_ARG_OUTOFRANGE, "Unknown orthogonalization");
    }
  } else if (ksp->orthog == KSPOrthogonalizationModifiedGramSchmidt) cstr = "modified Gram-Schmidt orthogonalization";
  else cstr = "unknown orthogonalization";
  if (isascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "  using %s\n", cstr));
#if !PetscDefined(USE_COMPLEX)
    PetscCall(PetscViewerASCIIPrintf(viewer, "  shift=%g\n", (double)eksm->shift));
#else
    PetscCall(PetscViewerASCIIPrintf(viewer, "  shift=%g%+gi\n", (double)PetscRealPart(eksm->shift), (double)PetscImaginaryPart(eksm->shift)));
#endif
    PetscCall(PetscViewerASCIIPrintf(viewer, "  happy breakdown tolerance=%g\n", (double)eksm->haptol));
    if (eksm->ksps) {
      PetscCall(PetscViewerASCIIPushTab(viewer));
      PetscCall(KSPView(eksm->ksps, viewer));
      PetscCall(PetscViewerASCIIPopTab(viewer));
    }
    if (eksm->kspm) {
      PetscCall(PetscViewerASCIIPushTab(viewer));
      PetscCall(KSPView(eksm->kspm, viewer));
      PetscCall(PetscViewerASCIIPopTab(viewer));
    }
  } else if (isstring) {
#if !PetscDefined(USE_COMPLEX)
    PetscCall(PetscViewerStringSPrintf(viewer, "%s shift %g", cstr, (double)eksm->shift));
#else
    PetscCall(PetscViewerStringSPrintf(viewer, "%s shift %g%+gi", cstr, (double)PetscRealPart(eksm->shift), (double)PetscImaginaryPart(eksm->shift)));
#endif
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPSetFromOptions_EKSM(KSP ksp, PetscOptionItems PetscOptionsObject)
{
  PetscScalar shift;
  PetscReal   haptol;
  KSP_EKSM   *eksm = (KSP_EKSM *)ksp->data;
  PetscBool   flg;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "KSP EKSM Options");
  PetscCall(PetscOptionsScalar("-ksp_eksm_shift", "Scalar shift for the internal linear solves", "KSPEKSMSetShift", eksm->shift, &shift, &flg));
  if (flg) PetscCall(KSPEKSMSetShift(ksp, shift));
  PetscCall(PetscOptionsReal("-ksp_eksm_haptol", "Tolerance for exact convergence (happy breakdown)", "KSPEKSMSetHapTol", eksm->haptol, &haptol, &flg));
  if (flg) PetscCall(KSPEKSMSetHapTol(ksp, haptol));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPEKSMSetHapTol_EKSM(KSP ksp, PetscReal tol)
{
  KSP_EKSM *eksm = (KSP_EKSM *)ksp->data;

  PetscFunctionBegin;
  PetscCheck(tol >= 0.0, PetscObjectComm((PetscObject)ksp), PETSC_ERR_ARG_OUTOFRANGE, "Tolerance must be non-negative");
  eksm->haptol = tol;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPEKSMSetHapTol - Sets the tolerance for detecting a happy breakdown in EKSM.

  Logically Collective

  Input Parameters:
+ ksp - the Krylov space solver context
- tol - the tolerance for detecting a happy breakdown

  Options Database Key:
. -ksp_eksm_haptol tol - set tolerance for determining happy breakdown

  Level: intermediate

  Notes:
  Happy breakdown is the rare case in `KSPEKSM` where a very near zero matrix entry is generated in the upper Hessenberg matrix indicating
  an 'exact' solution has been obtained.

  The default tolerance value for detecting a happy breakdown with EKSM in PETSc is 1.0e-30.

.seealso: [](ch_ksp), `KSPEKSM`, `KSPSetTolerances()`, `KSPGMRESSetHapTol()`
@*/
PetscErrorCode KSPEKSMSetHapTol(KSP ksp, PetscReal tol)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidLogicalCollectiveReal(ksp, tol, 2);
  PetscTryMethod(ksp, "KSPEKSMSetHapTol_C", (KSP, PetscReal), (ksp, tol));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPEKSMGetHapTol_EKSM(KSP ksp, PetscReal *tol)
{
  KSP_EKSM *eksm = (KSP_EKSM *)ksp->data;

  PetscFunctionBegin;
  *tol = eksm->haptol;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPEKSMGetHapTol - Gets the tolerance for detecting a happy breakdown in EKSM.

  Not Collective

  Input Parameter:
. ksp - the Krylov space solver context

  Output Parameter:
. tol - the tolerance for detecting a happy breakdown

  Level: intermediate

.seealso: [](ch_ksp), `KSPEKSM`, `KSPEKSMSetHapTol()`
@*/
PetscErrorCode KSPEKSMGetHapTol(KSP ksp, PetscReal *tol)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscAssertPointer(tol, 2);
  PetscUseMethod(ksp, "KSPEKSMGetHapTol_C", (KSP, PetscReal *), (ksp, tol));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPEKSMSetKSP_EKSM(KSP ksp, KSP ksps, KSP kspm)
{
  KSP_EKSM *eksm = (KSP_EKSM *)ksp->data;

  PetscFunctionBegin;
  if (ksps) {
    PetscCall(PetscObjectReference((PetscObject)ksps));
    PetscCall(PetscObjectDereference((PetscObject)eksm->ksps));
    eksm->ksps = ksps;
  }
  if (kspm) {
    PetscCall(PetscObjectReference((PetscObject)kspm));
    PetscCall(PetscObjectDereference((PetscObject)eksm->kspm));
    eksm->kspm = kspm;
  }
  if (ksp->setupstage) ksp->setupstage = KSP_SETUP_NEWMATRIX;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPEKSMSetKSP - Sets the internal `KSP` objects to be used by EKSM.

  Not Collective, but all `KSP` objects must live on the same `MPI_Comm`

  Input Parameters:
+ ksp  - the Krylov space solver context
. ksps - the `KSP` context used internally for linear systems with coefficient matrix $K + \varsigma M$, or `NULL` to keep the current one
- kspm - the `KSP` context used internally for linear systems with coefficient matrix $M$, or `NULL` to keep the current one

  Level: intermediate

  Notes:
  This routine is rarely needed. The most common usage is to call `KSPEKSMGetKSP()` to extract the internal
  objects and then configure them.

  Each `KSP` object actually passed in has its reference count increased by one and the internal object it
  replaces has its reference count decreased by one. An argument passed as `NULL` leaves the corresponding
  internal object, and its reference count, unchanged.

.seealso: [](ch_ksp), `KSPEKSM`, `KSPEKSMGetKSP()`
@*/
PetscErrorCode KSPEKSMSetKSP(KSP ksp, KSP ksps, KSP kspm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  if (ksps) {
    PetscValidHeaderSpecific(ksps, KSP_CLASSID, 2);
    PetscCheckSameComm(ksp, 1, ksps, 2);
  }
  if (kspm) {
    PetscValidHeaderSpecific(kspm, KSP_CLASSID, 3);
    PetscCheckSameComm(ksp, 1, kspm, 3);
  }
  PetscTryMethod(ksp, "KSPEKSMSetKSP_C", (KSP, KSP, KSP), (ksp, ksps, kspm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPEKSMGetKSP_EKSM(KSP ksp, KSP *ksps, KSP *kspm)
{
  KSP_EKSM *eksm = (KSP_EKSM *)ksp->data;

  PetscFunctionBegin;
  if (ksps) {
    if (!eksm->ksps) {
      PetscCall(KSPCreate(PetscObjectComm((PetscObject)ksp), &eksm->ksps));
      PetscCall(PetscObjectIncrementTabLevel((PetscObject)eksm->ksps, (PetscObject)ksp, 1));
      PetscCall(PetscObjectSetOptions((PetscObject)eksm->ksps, ((PetscObject)ksp)->options));
      PetscCall(KSPSetOptionsPrefix(eksm->ksps, ((PetscObject)ksp)->prefix));
      PetscCall(KSPAppendOptionsPrefix(eksm->ksps, "eksm_s_"));
      PetscCall(KSPSetTolerances(eksm->ksps, ksp->rtol, ksp->abstol, ksp->divtol, PETSC_DETERMINE));
    }
    *ksps = eksm->ksps;
  }
  if (kspm) {
    if (!eksm->kspm) {
      PetscCall(KSPCreate(PetscObjectComm((PetscObject)ksp), &eksm->kspm));
      PetscCall(PetscObjectIncrementTabLevel((PetscObject)eksm->kspm, (PetscObject)ksp, 1));
      PetscCall(PetscObjectSetOptions((PetscObject)eksm->kspm, ((PetscObject)ksp)->options));
      PetscCall(KSPSetOptionsPrefix(eksm->kspm, ((PetscObject)ksp)->prefix));
      PetscCall(KSPAppendOptionsPrefix(eksm->kspm, "eksm_m_"));
      PetscCall(KSPSetTolerances(eksm->kspm, ksp->rtol, ksp->abstol, ksp->divtol, PETSC_DETERMINE));
    }
    *kspm = eksm->kspm;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPEKSMGetKSP - Returns the internal `KSP` objects used by EKSM.

  Not Collective, but if `ksp` is parallel, then `ksps` and `kspm` are parallel

  Input Parameter:
. ksp - the Krylov space solver context

  Output Parameters:
+ ksps - the `KSP` context used internally for linear systems with coefficient matrix $K + \varsigma M$
- kspm - the `KSP` context used internally for linear systems with coefficient matrix $M$

  Level: intermediate

  Notes:
  The extended Krylov subspace method needs to solve linear systems with matrices $K + \varsigma M$ as well as $M$ (in case $M$ is not the
  identity). Here, $K$ and $M$ are the matrices provided in `MatCreateNestFromMultipleShifts()` and $\varsigma$ is the shift
  set with `KSPEKSMSetShift()`.

  This function is provided to allow the user to configure the internal `KSP` objects that will do most of the work.
  Pass `NULL` to any of the arguments if not needed. In particular, if $M=I$ the `kspm` object is not needed.

  To configure the linear solvers from the command line, use the `-eksm_s_` and `-eksm_m_` prefixes, respectively.
  For instance `-ksp_type eksm -eksm_s_ksp_type bcgs -eksm_s_pc_type bjacobi -eksm_m_pc_type none`.

.seealso: [](ch_ksp), `KSPEKSM`, `MatCreateNestFromMultipleShifts()`, `KSPEKSMSetShift()`, `KSPEKSMSetKSP()`
@*/
PetscErrorCode KSPEKSMGetKSP(KSP ksp, KSP *ksps, KSP *kspm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscUseMethod(ksp, "KSPEKSMGetKSP_C", (KSP, KSP *, KSP *), (ksp, ksps, kspm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPEKSMSetShift_EKSM(KSP ksp, PetscScalar shift)
{
  KSP_EKSM *eksm = (KSP_EKSM *)ksp->data;

  PetscFunctionBegin;
  eksm->shift_set = PETSC_TRUE;
  if (!ksp->setupstage) eksm->shift = shift;
  else if (eksm->shift != shift) {
    eksm->shift     = shift;
    ksp->setupstage = KSP_SETUP_NEW;
    /* free the data structures, then create them again */
    PetscCall(KSPReset_EKSM(ksp));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPEKSMSetShift - Sets the value of the shift $\varsigma$ that will be used for the internal linear solves in the EKSM solver.

  Logically Collective

  Input Parameters:
+ ksp   - the Krylov space solver context
- shift - the value of the shift, $\varsigma$

  Options Database Key:
. -ksp_eksm_shift shift - the value of the shift

  Level: intermediate

  Notes:
  The extended Krylov subspace method needs to solve linear systems with the matrix $K + \varsigma M$ using an auxiliary `KSP` object,
  see `KSPEKSMGetKSP()`. The scalar value $\varsigma$ is provided with this function. If not set, the default
  is to use a value of $\varsigma$ equal to the centroid (or center of mass) of all shifts $\sigma_i$ provided
  by the user in `MatCreateNestFromMultipleShifts()`.

.seealso: [](ch_ksp), `KSPEKSM`, `KSPEKSMGetKSP()`, `KSPEKSMGetShift()`, `MatCreateNestFromMultipleShifts()`
@*/
PetscErrorCode KSPEKSMSetShift(KSP ksp, PetscScalar shift)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidLogicalCollectiveScalar(ksp, shift, 2);
  PetscTryMethod(ksp, "KSPEKSMSetShift_C", (KSP, PetscScalar), (ksp, shift));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPEKSMGetShift_EKSM(KSP ksp, PetscScalar *shift)
{
  KSP_EKSM *eksm = (KSP_EKSM *)ksp->data;

  PetscFunctionBegin;
  *shift = eksm->shift;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPEKSMGetShift - Gets the value of the shift $\varsigma$ used for the internal linear solves.

  Not Collective

  Input Parameter:
. ksp - the Krylov space solver context

  Output Parameter:
. shift - the value of the shift, $\varsigma$

  Level: intermediate

  Note:
  The returned value is the one set with `KSPEKSMSetShift()`. If that routine was not called, the shift is
  computed internally during `KSPSetUp()` as the centroid of the shifts passed to `MatCreateNestFromMultipleShifts()`,
  so this routine returns 0 when called before the solver has been set up.

.seealso: [](ch_ksp), `KSPEKSM`, `KSPEKSMSetShift()`
@*/
PetscErrorCode KSPEKSMGetShift(KSP ksp, PetscScalar *shift)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscAssertPointer(shift, 2);
  PetscUseMethod(ksp, "KSPEKSMGetShift_C", (KSP, PetscScalar *), (ksp, shift));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   KSPEKSM - Implements the Extended Krylov Subspace Method (EKSM) for solving multiple shifted linear systems simultaneously using `KSP`.

   Options Database Keys:
+   -ksp_eksm_shift shift - the scalar shift for the internal linear solves
-   -ksp_eksm_haptol tol  - the tolerance for happy breakdown (exact convergence) of `KSPEKSM`

   Level: beginner

   Note:
   This solver should be used with no preconditioning, `PCNONE`. It is possible to use a preconditioner in the internal
   linear solvers, see `KSPEKSMGetKSP()`. To configure the internal linear solvers from the command line, use the
   `-eksm_s_` and `-eksm_m_` prefixes, respectively.
   For instance `-ksp_type eksm -eksm_s_ksp_type bcgs -eksm_s_pc_type bjacobi -eksm_m_pc_type none`.

   Unlike restarted methods such as `KSPGMRES`, this solver keeps the whole extended Krylov basis, so the maximum
   iteration count set with `KSPSetTolerances()` also fixes the size of the workspace allocated in `KSPSetUp()`\:
   two basis vectors per iteration and $O(\textrm{max\_it}^2)$ scalars of dense storage. Choose it accordingly, and
   set it before `KSPSetUp()`; changing it afterwards makes the next `KSPSolve()` fail.

   When a mass matrix $M$ is given to `MatCreateNestFromMultipleShifts()`, the residual norm used by the convergence
   test and reported by `-ksp_monitor` is $\|M^{-1}(b-(K+\sigma_i M)x_i)\|$ maximized over the shifts, not the
   unpreconditioned residual norm.

.seealso: [](ch_ksp), `KSPCreate()`, `KSPSetType()`, `KSPType`, `KSP`, `KSPEKSM`, `KSPEKSMSetHapTol()`, `KSPEKSMSetKSP()`, `KSPEKSMSetShift()`,
          `MatCreateNestFromMultipleShifts()`, `MatCreateVecNestFromMultipleShifts()`
M*/

PETSC_EXTERN PetscErrorCode KSPCreate_EKSM(KSP ksp)
{
  KSP_EKSM *eksm;
  PC        pc;

  PetscFunctionBegin;
  PetscCall(PetscNew(&eksm));
  ksp->data = (void *)eksm;

  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_PRECONDITIONED, PC_LEFT, 4));
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_UNPRECONDITIONED, PC_RIGHT, 3));
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_PRECONDITIONED, PC_SYMMETRIC, 2));
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_NONE, PC_RIGHT, 1));
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_NONE, PC_LEFT, 1));

  PetscObjectParameterSetDefault(ksp, max_it, 50);

  ksp->setupnewmatrix = PETSC_TRUE;

  ksp->ops->setup          = KSPSetUp_EKSM;
  ksp->ops->solve          = KSPSolve_EKSM;
  ksp->ops->reset          = KSPReset_EKSM;
  ksp->ops->destroy        = KSPDestroy_EKSM;
  ksp->ops->view           = KSPView_EKSM;
  ksp->ops->setfromoptions = KSPSetFromOptions_EKSM;

  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCNONE));

  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPEKSMSetHapTol_C", KSPEKSMSetHapTol_EKSM));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPEKSMGetHapTol_C", KSPEKSMGetHapTol_EKSM));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPEKSMSetKSP_C", KSPEKSMSetKSP_EKSM));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPEKSMGetKSP_C", KSPEKSMGetKSP_EKSM));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPEKSMSetShift_C", KSPEKSMSetShift_EKSM));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPEKSMGetShift_C", KSPEKSMGetShift_EKSM));

  eksm->haptol         = 1.0e-30;
  eksm->delta_allocate = EKSM_DELTA_DIRECTIONS;
  PetscFunctionReturn(PETSC_SUCCESS);
}
