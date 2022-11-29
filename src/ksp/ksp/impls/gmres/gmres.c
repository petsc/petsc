
/*
    This file implements GMRES (a Generalized Minimal Residual) method.
    Reference:  Saad and Schultz, 1986.

    Some comments on left vs. right preconditioning, and restarts.
    Left and right preconditioning.
    If right preconditioning is chosen, then the problem being solved
    by gmres is actually
       My =  AB^-1 y = f
    so the initial residual is
          r = f - Mx
    Note that B^-1 y = x or y = B x, and if x is non-zero, the initial
    residual is
          r = f - A x
    The final solution is then
          x = B^-1 y

    If left preconditioning is chosen, then the problem being solved is
       My = B^-1 A x = B^-1 f,
    and the initial residual is
       r  = B^-1(f - Ax)

    Restarts:  Restarts are basically solves with x0 not equal to zero.
    Note that we can eliminate an extra application of B^-1 between
    restarts as long as we don't require that the solution at the end
    of an unsuccessful gmres iteration always be the solution x.
 */

#include <../src/ksp/ksp/impls/gmres/gmresimpl.h> /*I  "petscksp.h"  I*/
#define GMRES_DELTA_DIRECTIONS 10
#define GMRES_DEFAULT_MAXK     30
static PetscErrorCode KSPGMRESUpdateHessenberg(KSP, PetscInt, PetscBool, PetscReal *);
static PetscErrorCode KSPGMRESBuildSoln(PetscScalar *, Vec, Vec, KSP, PetscInt);

PetscErrorCode KSPSetUp_GMRES(KSP ksp)
{
  PetscInt   hh, hes, rs, cc;
  PetscInt   max_k, k;
  KSP_GMRES *gmres = (KSP_GMRES *)ksp->data;

  PetscFunctionBegin;
  max_k = gmres->max_k; /* restart size */
  hh    = (max_k + 2) * (max_k + 1);
  hes   = (max_k + 1) * (max_k + 1);
  rs    = (max_k + 2);
  cc    = (max_k + 1);

  PetscCall(PetscCalloc5(hh, &gmres->hh_origin, hes, &gmres->hes_origin, rs, &gmres->rs_origin, cc, &gmres->cc_origin, cc, &gmres->ss_origin));

  if (ksp->calc_sings) {
    /* Allocate workspace to hold Hessenberg matrix needed by lapack */
    PetscCall(PetscMalloc1((max_k + 3) * (max_k + 9), &gmres->Rsvd));
    PetscCall(PetscMalloc1(6 * (max_k + 2), &gmres->Dsvd));
  }

  /* Allocate array to hold pointers to user vectors.  Note that we need
   4 + max_k + 1 (since we need it+1 vectors, and it <= max_k) */
  gmres->vecs_allocated = VEC_OFFSET + 2 + max_k + gmres->nextra_vecs;

  PetscCall(PetscMalloc1(gmres->vecs_allocated, &gmres->vecs));
  PetscCall(PetscMalloc1(VEC_OFFSET + 2 + max_k, &gmres->user_work));
  PetscCall(PetscMalloc1(VEC_OFFSET + 2 + max_k, &gmres->mwork_alloc));

  if (gmres->q_preallocate) {
    gmres->vv_allocated = VEC_OFFSET + 2 + max_k;

    PetscCall(KSPCreateVecs(ksp, gmres->vv_allocated, &gmres->user_work[0], 0, NULL));

    gmres->mwork_alloc[0] = gmres->vv_allocated;
    gmres->nwork_alloc    = 1;
    for (k = 0; k < gmres->vv_allocated; k++) gmres->vecs[k] = gmres->user_work[0][k];
  } else {
    gmres->vv_allocated = 5;

    PetscCall(KSPCreateVecs(ksp, 5, &gmres->user_work[0], 0, NULL));

    gmres->mwork_alloc[0] = 5;
    gmres->nwork_alloc    = 1;
    for (k = 0; k < gmres->vv_allocated; k++) gmres->vecs[k] = gmres->user_work[0][k];
  }
  PetscFunctionReturn(0);
}

/*
    Run gmres, possibly with restart.  Return residual history if requested.
    input parameters:

.        gmres  - structure containing parameters and work areas

    output parameters:
.        nres    - residuals (from preconditioned system) at each step.
                  If restarting, consider passing nres+it.  If null,
                  ignored
.        itcount - number of iterations used.  nres[0] to nres[itcount]
                  are defined.  If null, ignored.

    Notes:
    On entry, the value in vector VEC_VV(0) should be the initial residual
    (this allows shortcuts where the initial preconditioned residual is 0).
 */
PetscErrorCode KSPGMRESCycle(PetscInt *itcount, KSP ksp)
{
  KSP_GMRES *gmres = (KSP_GMRES *)(ksp->data);
  PetscReal  res, hapbnd, tt;
  PetscInt   it = 0, max_k = gmres->max_k;
  PetscBool  hapend = PETSC_FALSE;

  PetscFunctionBegin;
  if (itcount) *itcount = 0;
  PetscCall(VecNormalize(VEC_VV(0), &res));
  KSPCheckNorm(ksp, res);

  /* the constant .1 is arbitrary, just some measure at how incorrect the residuals are */
  if ((ksp->rnorm > 0.0) && (PetscAbsReal(res - ksp->rnorm) > gmres->breakdowntol * gmres->rnorm0)) {
    PetscCheck(!ksp->errorifnotconverged, PetscObjectComm((PetscObject)ksp), PETSC_ERR_CONV_FAILED, "Residual norm computed by GMRES recursion formula %g is far from the computed residual norm %g at restart, residual norm at start of cycle %g",
               (double)ksp->rnorm, (double)res, (double)gmres->rnorm0);
    PetscCall(PetscInfo(ksp, "Residual norm computed by GMRES recursion formula %g is far from the computed residual norm %g at restart, residual norm at start of cycle %g", (double)ksp->rnorm, (double)res, (double)gmres->rnorm0));
    ksp->reason = KSP_DIVERGED_BREAKDOWN;
    PetscFunctionReturn(0);
  }
  *GRS(0) = gmres->rnorm0 = res;

  /* check for the convergence */
  PetscCall(PetscObjectSAWsTakeAccess((PetscObject)ksp));
  ksp->rnorm = res;
  PetscCall(PetscObjectSAWsGrantAccess((PetscObject)ksp));
  gmres->it = (it - 1);
  PetscCall(KSPLogResidualHistory(ksp, res));
  PetscCall(KSPLogErrorHistory(ksp));
  PetscCall(KSPMonitor(ksp, ksp->its, res));
  if (!res) {
    ksp->reason = KSP_CONVERGED_ATOL;
    PetscCall(PetscInfo(ksp, "Converged due to zero residual norm on entry\n"));
    PetscFunctionReturn(0);
  }

  PetscCall((*ksp->converged)(ksp, ksp->its, res, &ksp->reason, ksp->cnvP));
  while (!ksp->reason && it < max_k && ksp->its < ksp->max_it) {
    if (it) {
      PetscCall(KSPLogResidualHistory(ksp, res));
      PetscCall(KSPLogErrorHistory(ksp));
      PetscCall(KSPMonitor(ksp, ksp->its, res));
    }
    gmres->it = (it - 1);
    if (gmres->vv_allocated <= it + VEC_OFFSET + 1) PetscCall(KSPGMRESGetNewVectors(ksp, it + 1));
    PetscCall(KSP_PCApplyBAorAB(ksp, VEC_VV(it), VEC_VV(1 + it), VEC_TEMP_MATOP));

    /* update hessenberg matrix and do Gram-Schmidt */
    PetscCall((*gmres->orthog)(ksp, it));
    if (ksp->reason) break;

    /* vv(i+1) . vv(i+1) */
    PetscCall(VecNormalize(VEC_VV(it + 1), &tt));
    KSPCheckNorm(ksp, tt);

    /* save the magnitude */
    *HH(it + 1, it)  = tt;
    *HES(it + 1, it) = tt;

    /* check for the happy breakdown */
    hapbnd = PetscAbsScalar(tt / *GRS(it));
    if (hapbnd > gmres->haptol) hapbnd = gmres->haptol;
    if (tt < hapbnd) {
      PetscCall(PetscInfo(ksp, "Detected happy breakdown, current hapbnd = %14.12e tt = %14.12e\n", (double)hapbnd, (double)tt));
      hapend = PETSC_TRUE;
    }
    PetscCall(KSPGMRESUpdateHessenberg(ksp, it, hapend, &res));

    it++;
    gmres->it = (it - 1); /* For converged */
    ksp->its++;
    ksp->rnorm = res;
    if (ksp->reason) break;

    PetscCall((*ksp->converged)(ksp, ksp->its, res, &ksp->reason, ksp->cnvP));

    /* Catch error in happy breakdown and signal convergence and break from loop */
    if (hapend) {
      if (ksp->normtype == KSP_NORM_NONE) { /* convergence test was skipped in this case */
        ksp->reason = KSP_CONVERGED_HAPPY_BREAKDOWN;
      } else if (!ksp->reason) {
        PetscCheck(!ksp->errorifnotconverged, PetscObjectComm((PetscObject)ksp), PETSC_ERR_NOT_CONVERGED, "You reached the happy break down, but convergence was not indicated. Residual norm = %g", (double)res);
        ksp->reason = KSP_DIVERGED_BREAKDOWN;
        break;
      }
    }
  }

  /* Monitor if we know that we will not return for a restart */
  if (it && (ksp->reason || ksp->its >= ksp->max_it)) {
    PetscCall(KSPLogResidualHistory(ksp, res));
    PetscCall(KSPLogErrorHistory(ksp));
    PetscCall(KSPMonitor(ksp, ksp->its, res));
  }

  if (itcount) *itcount = it;

  /*
    Down here we have to solve for the "best" coefficients of the Krylov
    columns, add the solution values together, and possibly unwind the
    preconditioning from the solution
   */
  /* Form the solution (or the solution so far) */
  PetscCall(KSPGMRESBuildSoln(GRS(0), ksp->vec_sol, ksp->vec_sol, ksp, it - 1));
  PetscFunctionReturn(0);
}

PetscErrorCode KSPSolve_GMRES(KSP ksp)
{
  PetscInt   its, itcount, i;
  KSP_GMRES *gmres      = (KSP_GMRES *)ksp->data;
  PetscBool  guess_zero = ksp->guess_zero;
  PetscInt   N          = gmres->max_k + 1;

  PetscFunctionBegin;
  PetscCheck(!ksp->calc_sings || gmres->Rsvd, PetscObjectComm((PetscObject)ksp), PETSC_ERR_ORDER, "Must call KSPSetComputeSingularValues() before KSPSetUp() is called");

  PetscCall(PetscObjectSAWsTakeAccess((PetscObject)ksp));
  ksp->its = 0;
  PetscCall(PetscObjectSAWsGrantAccess((PetscObject)ksp));

  itcount          = 0;
  gmres->fullcycle = 0;
  ksp->rnorm       = -1.0; /* special marker for KSPGMRESCycle() */
  while (!ksp->reason || (ksp->rnorm == -1 && ksp->reason == KSP_DIVERGED_PC_FAILED)) {
    PetscCall(KSPInitialResidual(ksp, ksp->vec_sol, VEC_TEMP, VEC_TEMP_MATOP, VEC_VV(0), ksp->vec_rhs));
    PetscCall(KSPGMRESCycle(&its, ksp));
    /* Store the Hessenberg matrix and the basis vectors of the Krylov subspace
    if the cycle is complete for the computation of the Ritz pairs */
    if (its == gmres->max_k) {
      gmres->fullcycle++;
      if (ksp->calc_ritz) {
        if (!gmres->hes_ritz) {
          PetscCall(PetscMalloc1(N * N, &gmres->hes_ritz));
          PetscCall(VecDuplicateVecs(VEC_VV(0), N, &gmres->vecb));
        }
        PetscCall(PetscArraycpy(gmres->hes_ritz, gmres->hes_origin, N * N));
        for (i = 0; i < gmres->max_k + 1; i++) PetscCall(VecCopy(VEC_VV(i), gmres->vecb[i]));
      }
    }
    itcount += its;
    if (itcount >= ksp->max_it) {
      if (!ksp->reason) ksp->reason = KSP_DIVERGED_ITS;
      break;
    }
    ksp->guess_zero = PETSC_FALSE; /* every future call to KSPInitialResidual() will have nonzero guess */
  }
  ksp->guess_zero = guess_zero; /* restore if user provided nonzero initial guess */
  PetscFunctionReturn(0);
}

PetscErrorCode KSPReset_GMRES(KSP ksp)
{
  KSP_GMRES *gmres = (KSP_GMRES *)ksp->data;
  PetscInt   i;

  PetscFunctionBegin;
  /* Free the Hessenberg matrices */
  PetscCall(PetscFree5(gmres->hh_origin, gmres->hes_origin, gmres->rs_origin, gmres->cc_origin, gmres->ss_origin));
  PetscCall(PetscFree(gmres->hes_ritz));

  /* free work vectors */
  PetscCall(PetscFree(gmres->vecs));
  for (i = 0; i < gmres->nwork_alloc; i++) PetscCall(VecDestroyVecs(gmres->mwork_alloc[i], &gmres->user_work[i]));
  gmres->nwork_alloc = 0;
  if (gmres->vecb) PetscCall(VecDestroyVecs(gmres->max_k + 1, &gmres->vecb));

  PetscCall(PetscFree(gmres->user_work));
  PetscCall(PetscFree(gmres->mwork_alloc));
  PetscCall(PetscFree(gmres->nrs));
  PetscCall(VecDestroy(&gmres->sol_temp));
  PetscCall(PetscFree(gmres->Rsvd));
  PetscCall(PetscFree(gmres->Dsvd));
  PetscCall(PetscFree(gmres->orthogwork));

  gmres->vv_allocated   = 0;
  gmres->vecs_allocated = 0;
  gmres->sol_temp       = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode KSPDestroy_GMRES(KSP ksp)
{
  PetscFunctionBegin;
  PetscCall(KSPReset_GMRES(ksp));
  PetscCall(PetscFree(ksp->data));
  /* clear composed functions */
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPGMRESSetPreAllocateVectors_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPGMRESSetOrthogonalization_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPGMRESGetOrthogonalization_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPGMRESSetRestart_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPGMRESGetRestart_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPGMRESSetHapTol_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPGMRESSetBreakdownTolerance_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPGMRESSetCGSRefinementType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPGMRESGetCGSRefinementType_C", NULL));
  PetscFunctionReturn(0);
}
/*
    KSPGMRESBuildSoln - create the solution from the starting vector and the
    current iterates.

    Input parameters:
        nrs - work area of size it + 1.
        vs  - index of initial guess
        vdest - index of result.  Note that vs may == vdest (replace
                guess with the solution).

     This is an internal routine that knows about the GMRES internals.
 */
static PetscErrorCode KSPGMRESBuildSoln(PetscScalar *nrs, Vec vs, Vec vdest, KSP ksp, PetscInt it)
{
  PetscScalar tt;
  PetscInt    ii, k, j;
  KSP_GMRES  *gmres = (KSP_GMRES *)(ksp->data);

  PetscFunctionBegin;
  /* Solve for solution vector that minimizes the residual */

  /* If it is < 0, no gmres steps have been performed */
  if (it < 0) {
    PetscCall(VecCopy(vs, vdest)); /* VecCopy() is smart, exists immediately if vguess == vdest */
    PetscFunctionReturn(0);
  }
  if (*HH(it, it) != 0.0) {
    nrs[it] = *GRS(it) / *HH(it, it);
  } else {
    PetscCheck(!ksp->errorifnotconverged, PetscObjectComm((PetscObject)ksp), PETSC_ERR_NOT_CONVERGED, "You reached the break down in GMRES; HH(it,it) = 0");
    ksp->reason = KSP_DIVERGED_BREAKDOWN;

    PetscCall(PetscInfo(ksp, "Likely your matrix or preconditioner is singular. HH(it,it) is identically zero; it = %" PetscInt_FMT " GRS(it) = %g\n", it, (double)PetscAbsScalar(*GRS(it))));
    PetscFunctionReturn(0);
  }
  for (ii = 1; ii <= it; ii++) {
    k  = it - ii;
    tt = *GRS(k);
    for (j = k + 1; j <= it; j++) tt = tt - *HH(k, j) * nrs[j];
    if (*HH(k, k) == 0.0) {
      PetscCheck(!ksp->errorifnotconverged, PetscObjectComm((PetscObject)ksp), PETSC_ERR_NOT_CONVERGED, "Likely your matrix or preconditioner is singular. HH(k,k) is identically zero; k = %" PetscInt_FMT, k);
      ksp->reason = KSP_DIVERGED_BREAKDOWN;
      PetscCall(PetscInfo(ksp, "Likely your matrix or preconditioner is singular. HH(k,k) is identically zero; k = %" PetscInt_FMT "\n", k));
      PetscFunctionReturn(0);
    }
    nrs[k] = tt / *HH(k, k);
  }

  /* Accumulate the correction to the solution of the preconditioned problem in TEMP */
  PetscCall(VecSet(VEC_TEMP, 0.0));
  PetscCall(VecMAXPY(VEC_TEMP, it + 1, nrs, &VEC_VV(0)));

  PetscCall(KSPUnwindPreconditioner(ksp, VEC_TEMP, VEC_TEMP_MATOP));
  /* add solution to previous solution */
  if (vdest != vs) PetscCall(VecCopy(vs, vdest));
  PetscCall(VecAXPY(vdest, 1.0, VEC_TEMP));
  PetscFunctionReturn(0);
}
/*
   Do the scalar work for the orthogonalization.  Return new residual norm.
 */
static PetscErrorCode KSPGMRESUpdateHessenberg(KSP ksp, PetscInt it, PetscBool hapend, PetscReal *res)
{
  PetscScalar *hh, *cc, *ss, tt;
  PetscInt     j;
  KSP_GMRES   *gmres = (KSP_GMRES *)(ksp->data);

  PetscFunctionBegin;
  hh = HH(0, it);
  cc = CC(0);
  ss = SS(0);

  /* Apply all the previously computed plane rotations to the new column
     of the Hessenberg matrix */
  for (j = 1; j <= it; j++) {
    tt  = *hh;
    *hh = PetscConj(*cc) * tt + *ss * *(hh + 1);
    hh++;
    *hh = *cc++ * *hh - (*ss++ * tt);
  }

  /*
    compute the new plane rotation, and apply it to:
     1) the right-hand-side of the Hessenberg system
     2) the new column of the Hessenberg matrix
    thus obtaining the updated value of the residual
  */
  if (!hapend) {
    tt = PetscSqrtScalar(PetscConj(*hh) * *hh + PetscConj(*(hh + 1)) * *(hh + 1));
    if (tt == 0.0) {
      PetscCheck(!ksp->errorifnotconverged, PetscObjectComm((PetscObject)ksp), PETSC_ERR_NOT_CONVERGED, "tt == 0.0");
      ksp->reason = KSP_DIVERGED_NULL;
      PetscFunctionReturn(0);
    }
    *cc          = *hh / tt;
    *ss          = *(hh + 1) / tt;
    *GRS(it + 1) = -(*ss * *GRS(it));
    *GRS(it)     = PetscConj(*cc) * *GRS(it);
    *hh          = PetscConj(*cc) * *hh + *ss * *(hh + 1);
    *res         = PetscAbsScalar(*GRS(it + 1));
  } else {
    /* happy breakdown: HH(it+1, it) = 0, therefore we don't need to apply
            another rotation matrix (so RH doesn't change).  The new residual is
            always the new sine term times the residual from last time (GRS(it)),
            but now the new sine rotation would be zero...so the residual should
            be zero...so we will multiply "zero" by the last residual.  This might
            not be exactly what we want to do here -could just return "zero". */

    *res = 0.0;
  }
  PetscFunctionReturn(0);
}
/*
   This routine allocates more work vectors, starting from VEC_VV(it).
 */
PetscErrorCode KSPGMRESGetNewVectors(KSP ksp, PetscInt it)
{
  KSP_GMRES *gmres = (KSP_GMRES *)ksp->data;
  PetscInt   nwork = gmres->nwork_alloc, k, nalloc;

  PetscFunctionBegin;
  nalloc = PetscMin(ksp->max_it, gmres->delta_allocate);
  /* Adjust the number to allocate to make sure that we don't exceed the
    number of available slots */
  if (it + VEC_OFFSET + nalloc >= gmres->vecs_allocated) nalloc = gmres->vecs_allocated - it - VEC_OFFSET;
  if (!nalloc) PetscFunctionReturn(0);

  gmres->vv_allocated += nalloc;

  PetscCall(KSPCreateVecs(ksp, nalloc, &gmres->user_work[nwork], 0, NULL));

  gmres->mwork_alloc[nwork] = nalloc;
  for (k = 0; k < nalloc; k++) gmres->vecs[it + VEC_OFFSET + k] = gmres->user_work[nwork][k];
  gmres->nwork_alloc++;
  PetscFunctionReturn(0);
}

PetscErrorCode KSPBuildSolution_GMRES(KSP ksp, Vec ptr, Vec *result)
{
  KSP_GMRES *gmres = (KSP_GMRES *)ksp->data;

  PetscFunctionBegin;
  if (!ptr) {
    if (!gmres->sol_temp) { PetscCall(VecDuplicate(ksp->vec_sol, &gmres->sol_temp)); }
    ptr = gmres->sol_temp;
  }
  if (!gmres->nrs) {
    /* allocate the work area */
    PetscCall(PetscMalloc1(gmres->max_k, &gmres->nrs));
  }

  PetscCall(KSPGMRESBuildSoln(gmres->nrs, ksp->vec_sol, ptr, ksp, gmres->it));
  if (result) *result = ptr;
  PetscFunctionReturn(0);
}

PetscErrorCode KSPView_GMRES(KSP ksp, PetscViewer viewer)
{
  KSP_GMRES  *gmres = (KSP_GMRES *)ksp->data;
  const char *cstr;
  PetscBool   iascii, isstring;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERSTRING, &isstring));
  if (gmres->orthog == KSPGMRESClassicalGramSchmidtOrthogonalization) {
    switch (gmres->cgstype) {
    case (KSP_GMRES_CGS_REFINE_NEVER):
      cstr = "Classical (unmodified) Gram-Schmidt Orthogonalization with no iterative refinement";
      break;
    case (KSP_GMRES_CGS_REFINE_ALWAYS):
      cstr = "Classical (unmodified) Gram-Schmidt Orthogonalization with one step of iterative refinement";
      break;
    case (KSP_GMRES_CGS_REFINE_IFNEEDED):
      cstr = "Classical (unmodified) Gram-Schmidt Orthogonalization with one step of iterative refinement when needed";
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)ksp), PETSC_ERR_ARG_OUTOFRANGE, "Unknown orthogonalization");
    }
  } else if (gmres->orthog == KSPGMRESModifiedGramSchmidtOrthogonalization) {
    cstr = "Modified Gram-Schmidt Orthogonalization";
  } else {
    cstr = "unknown orthogonalization";
  }
  if (iascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "  restart=%" PetscInt_FMT ", using %s\n", gmres->max_k, cstr));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  happy breakdown tolerance %g\n", (double)gmres->haptol));
  } else if (isstring) {
    PetscCall(PetscViewerStringSPrintf(viewer, "%s restart %" PetscInt_FMT, cstr, gmres->max_k));
  }
  PetscFunctionReturn(0);
}

/*@C
   KSPGMRESMonitorKrylov - Calls `VecView()` for each new direction in the `KSPGMRES` accumulated Krylov space.

   Collective on ksp

   Input Parameters:
+  ksp - the `KSP` context
.  its - iteration number
.  fgnorm - 2-norm of residual (or gradient)
-  dummy - an collection of viewers created with `KSPViewerCreate()`

   Options Database Key:
.   -ksp_gmres_krylov_monitor <bool> - Plot the Krylov directions

   Level: intermediate

   Note:
    A new `PETSCVIEWERDRAW` is created for each Krylov vector so they can all be simultaneously viewed

.seealso: [](chapter_ksp), `KSPGMRES`, `KSPMonitorSet()`, `KSPMonitorResidual()`, `VecView()`, `KSPViewersCreate()`, `KSPViewersDestroy()`
@*/
PetscErrorCode KSPGMRESMonitorKrylov(KSP ksp, PetscInt its, PetscReal fgnorm, void *dummy)
{
  PetscViewers viewers = (PetscViewers)dummy;
  KSP_GMRES   *gmres   = (KSP_GMRES *)ksp->data;
  Vec          x;
  PetscViewer  viewer;
  PetscBool    flg;

  PetscFunctionBegin;
  PetscCall(PetscViewersGetViewer(viewers, gmres->it + 1, &viewer));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERDRAW, &flg));
  if (!flg) {
    PetscCall(PetscViewerSetType(viewer, PETSCVIEWERDRAW));
    PetscCall(PetscViewerDrawSetInfo(viewer, NULL, "Krylov GMRES Monitor", PETSC_DECIDE, PETSC_DECIDE, 300, 300));
  }
  x = VEC_VV(gmres->it + 1);
  PetscCall(VecView(x, viewer));
  PetscFunctionReturn(0);
}

PetscErrorCode KSPSetFromOptions_GMRES(KSP ksp, PetscOptionItems *PetscOptionsObject)
{
  PetscInt   restart;
  PetscReal  haptol, breakdowntol;
  KSP_GMRES *gmres = (KSP_GMRES *)ksp->data;
  PetscBool  flg;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "KSP GMRES Options");
  PetscCall(PetscOptionsInt("-ksp_gmres_restart", "Number of Krylov search directions", "KSPGMRESSetRestart", gmres->max_k, &restart, &flg));
  if (flg) PetscCall(KSPGMRESSetRestart(ksp, restart));
  PetscCall(PetscOptionsReal("-ksp_gmres_haptol", "Tolerance for exact convergence (happy ending)", "KSPGMRESSetHapTol", gmres->haptol, &haptol, &flg));
  if (flg) PetscCall(KSPGMRESSetHapTol(ksp, haptol));
  PetscCall(PetscOptionsReal("-ksp_gmres_breakdown_tolerance", "Divergence breakdown tolerance during GMRES restart", "KSPGMRESSetBreakdownTolerance", gmres->breakdowntol, &breakdowntol, &flg));
  if (flg) PetscCall(KSPGMRESSetBreakdownTolerance(ksp, breakdowntol));
  flg = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-ksp_gmres_preallocate", "Preallocate Krylov vectors", "KSPGMRESSetPreAllocateVectors", flg, &flg, NULL));
  if (flg) PetscCall(KSPGMRESSetPreAllocateVectors(ksp));
  PetscCall(PetscOptionsBoolGroupBegin("-ksp_gmres_classicalgramschmidt", "Classical (unmodified) Gram-Schmidt (fast)", "KSPGMRESSetOrthogonalization", &flg));
  if (flg) PetscCall(KSPGMRESSetOrthogonalization(ksp, KSPGMRESClassicalGramSchmidtOrthogonalization));
  PetscCall(PetscOptionsBoolGroupEnd("-ksp_gmres_modifiedgramschmidt", "Modified Gram-Schmidt (slow,more stable)", "KSPGMRESSetOrthogonalization", &flg));
  if (flg) PetscCall(KSPGMRESSetOrthogonalization(ksp, KSPGMRESModifiedGramSchmidtOrthogonalization));
  PetscCall(PetscOptionsEnum("-ksp_gmres_cgs_refinement_type", "Type of iterative refinement for classical (unmodified) Gram-Schmidt", "KSPGMRESSetCGSRefinementType", KSPGMRESCGSRefinementTypes, (PetscEnum)gmres->cgstype, (PetscEnum *)&gmres->cgstype, &flg));
  flg = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-ksp_gmres_krylov_monitor", "Plot the Krylov directions", "KSPMonitorSet", flg, &flg, NULL));
  if (flg) {
    PetscViewers viewers;
    PetscCall(PetscViewersCreate(PetscObjectComm((PetscObject)ksp), &viewers));
    PetscCall(KSPMonitorSet(ksp, KSPGMRESMonitorKrylov, viewers, (PetscErrorCode(*)(void **))PetscViewersDestroy));
  }
  PetscOptionsHeadEnd();
  PetscFunctionReturn(0);
}

PetscErrorCode KSPGMRESSetHapTol_GMRES(KSP ksp, PetscReal tol)
{
  KSP_GMRES *gmres = (KSP_GMRES *)ksp->data;

  PetscFunctionBegin;
  PetscCheck(tol >= 0.0, PetscObjectComm((PetscObject)ksp), PETSC_ERR_ARG_OUTOFRANGE, "Tolerance must be non-negative");
  gmres->haptol = tol;
  PetscFunctionReturn(0);
}

PetscErrorCode KSPGMRESSetBreakdownTolerance_GMRES(KSP ksp, PetscReal tol)
{
  KSP_GMRES *gmres = (KSP_GMRES *)ksp->data;

  PetscFunctionBegin;
  if (tol == PETSC_DEFAULT) {
    gmres->breakdowntol = 0.1;
    PetscFunctionReturn(0);
  }
  PetscCheck(tol >= 0.0, PetscObjectComm((PetscObject)ksp), PETSC_ERR_ARG_OUTOFRANGE, "Breakdown tolerance must be non-negative");
  gmres->breakdowntol = tol;
  PetscFunctionReturn(0);
}

PetscErrorCode KSPGMRESGetRestart_GMRES(KSP ksp, PetscInt *max_k)
{
  KSP_GMRES *gmres = (KSP_GMRES *)ksp->data;

  PetscFunctionBegin;
  *max_k = gmres->max_k;
  PetscFunctionReturn(0);
}

PetscErrorCode KSPGMRESSetRestart_GMRES(KSP ksp, PetscInt max_k)
{
  KSP_GMRES *gmres = (KSP_GMRES *)ksp->data;

  PetscFunctionBegin;
  PetscCheck(max_k >= 1, PetscObjectComm((PetscObject)ksp), PETSC_ERR_ARG_OUTOFRANGE, "Restart must be positive");
  if (!ksp->setupstage) {
    gmres->max_k = max_k;
  } else if (gmres->max_k != max_k) {
    gmres->max_k    = max_k;
    ksp->setupstage = KSP_SETUP_NEW;
    /* free the data structures, then create them again */
    PetscCall(KSPReset_GMRES(ksp));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode KSPGMRESSetOrthogonalization_GMRES(KSP ksp, FCN fcn)
{
  PetscFunctionBegin;
  ((KSP_GMRES *)ksp->data)->orthog = fcn;
  PetscFunctionReturn(0);
}

PetscErrorCode KSPGMRESGetOrthogonalization_GMRES(KSP ksp, FCN *fcn)
{
  PetscFunctionBegin;
  *fcn = ((KSP_GMRES *)ksp->data)->orthog;
  PetscFunctionReturn(0);
}

PetscErrorCode KSPGMRESSetPreAllocateVectors_GMRES(KSP ksp)
{
  KSP_GMRES *gmres;

  PetscFunctionBegin;
  gmres                = (KSP_GMRES *)ksp->data;
  gmres->q_preallocate = 1;
  PetscFunctionReturn(0);
}

PetscErrorCode KSPGMRESSetCGSRefinementType_GMRES(KSP ksp, KSPGMRESCGSRefinementType type)
{
  KSP_GMRES *gmres = (KSP_GMRES *)ksp->data;

  PetscFunctionBegin;
  gmres->cgstype = type;
  PetscFunctionReturn(0);
}

PetscErrorCode KSPGMRESGetCGSRefinementType_GMRES(KSP ksp, KSPGMRESCGSRefinementType *type)
{
  KSP_GMRES *gmres = (KSP_GMRES *)ksp->data;

  PetscFunctionBegin;
  *type = gmres->cgstype;
  PetscFunctionReturn(0);
}

/*@
   KSPGMRESSetCGSRefinementType - Sets the type of iterative refinement to use
         in the classical Gram Schmidt orthogonalization.

   Logically Collective on ksp

   Input Parameters:
+  ksp - the Krylov space context
-  type - the type of refinement
.vb
  KSP_GMRES_CGS_REFINE_NEVER
  KSP_GMRES_CGS_REFINE_IFNEEDED
  KSP_GMRES_CGS_REFINE_ALWAYS
.ve

  Options Database Key:
.  -ksp_gmres_cgs_refinement_type <refine_never,refine_ifneeded,refine_always> - refinement type

   Level: intermediate

.seealso: [](chapter_ksp), `KSPGMRES`, `KSPGMRESSetOrthogonalization()`, `KSPGMRESCGSRefinementType`, `KSPGMRESClassicalGramSchmidtOrthogonalization()`, `KSPGMRESGetCGSRefinementType()`,
          `KSPGMRESGetOrthogonalization()`
@*/
PetscErrorCode KSPGMRESSetCGSRefinementType(KSP ksp, KSPGMRESCGSRefinementType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidLogicalCollectiveEnum(ksp, type, 2);
  PetscTryMethod(ksp, "KSPGMRESSetCGSRefinementType_C", (KSP, KSPGMRESCGSRefinementType), (ksp, type));
  PetscFunctionReturn(0);
}

/*@
   KSPGMRESGetCGSRefinementType - Gets the type of iterative refinement to use
         in the classical Gram Schmidt orthogonalization.

   Not Collective

   Input Parameter:
.  ksp - the Krylov space context

   Output Parameter:
.  type - the type of refinement

  Options Database Key:
.  -ksp_gmres_cgs_refinement_type <refine_never,refine_ifneeded,refine_always> - type of refinement

   Level: intermediate

.seealso: [](chapter_ksp), `KSPGMRES`, `KSPGMRESSetOrthogonalization()`, `KSPGMRESCGSRefinementType`, `KSPGMRESClassicalGramSchmidtOrthogonalization()`, `KSPGMRESSetCGSRefinementType()`,
          `KSPGMRESGetOrthogonalization()`
@*/
PetscErrorCode KSPGMRESGetCGSRefinementType(KSP ksp, KSPGMRESCGSRefinementType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscUseMethod(ksp, "KSPGMRESGetCGSRefinementType_C", (KSP, KSPGMRESCGSRefinementType *), (ksp, type));
  PetscFunctionReturn(0);
}

/*@
   KSPGMRESSetRestart - Sets number of iterations at which `KSPGMRES`, `KSPFGMRES` and `KSPLGMRES` restarts.

   Logically Collective on ksp

   Input Parameters:
+  ksp - the Krylov space context
-  restart - integer restart value

  Options Database Key:
.  -ksp_gmres_restart <positive integer> - integer restart value

   Level: intermediate

    Note:
    The default value is 30.

.seealso: [](chapter_ksp), `KSPGMRES`, `KSPSetTolerances()`, `KSPGMRESSetOrthogonalization()`, `KSPGMRESSetPreAllocateVectors()`, `KSPGMRESGetRestart()`
@*/
PetscErrorCode KSPGMRESSetRestart(KSP ksp, PetscInt restart)
{
  PetscFunctionBegin;
  PetscValidLogicalCollectiveInt(ksp, restart, 2);

  PetscTryMethod(ksp, "KSPGMRESSetRestart_C", (KSP, PetscInt), (ksp, restart));
  PetscFunctionReturn(0);
}

/*@
   KSPGMRESGetRestart - Gets number of iterations at which `KSPGMRES`, `KSPFGMRES` and `KSPLGMRES` restarts.

   Not Collective

   Input Parameter:
.  ksp - the Krylov space context

   Output Parameter:
.   restart - integer restart value

   Level: intermediate

.seealso: [](chapter_ksp), `KSPGMRES`, `KSPSetTolerances()`, `KSPGMRESSetOrthogonalization()`, `KSPGMRESSetPreAllocateVectors()`, `KSPGMRESSetRestart()`
@*/
PetscErrorCode KSPGMRESGetRestart(KSP ksp, PetscInt *restart)
{
  PetscFunctionBegin;
  PetscUseMethod(ksp, "KSPGMRESGetRestart_C", (KSP, PetscInt *), (ksp, restart));
  PetscFunctionReturn(0);
}

/*@
   KSPGMRESSetHapTol - Sets tolerance for determining happy breakdown in `KSPGMRES`, `KSPFGMRES` and `KSPLGMRES`

   Logically Collective on ksp

   Input Parameters:
+  ksp - the Krylov space context
-  tol - the tolerance

  Options Database Key:
.  -ksp_gmres_haptol <positive real value> - set tolerance for determining happy breakdown

   Level: intermediate

   Note:
   Happy breakdown is the rare case in `KSPGMRES` where an 'exact' solution is obtained after
   a certain number of iterations. If you attempt more iterations after this point unstable
   things can happen hence very occasionally you may need to set this value to detect this condition

.seealso: [](chapter_ksp), `KSPGMRES`, `KSPSetTolerances()`
@*/
PetscErrorCode KSPGMRESSetHapTol(KSP ksp, PetscReal tol)
{
  PetscFunctionBegin;
  PetscValidLogicalCollectiveReal(ksp, tol, 2);
  PetscTryMethod((ksp), "KSPGMRESSetHapTol_C", (KSP, PetscReal), ((ksp), (tol)));
  PetscFunctionReturn(0);
}

/*@
   KSPGMRESSetBreakdownTolerance - Sets tolerance for determining divergence breakdown in `KSPGMRES`.

   Logically Collective on ksp

   Input Parameters:
+  ksp - the Krylov space context
-  tol - the tolerance

  Options Database Key:
.  -ksp_gmres_breakdown_tolerance <positive real value> - set tolerance for determining divergence breakdown

   Level: intermediate

   Note:
   Divergence breakdown occurs when GMRES residual increases significantly during restart

.seealso: [](chapter_ksp), `KSPGMRES`, `KSPSetTolerances()`, `KSPGMRESSetHapTol()`
@*/
PetscErrorCode KSPGMRESSetBreakdownTolerance(KSP ksp, PetscReal tol)
{
  PetscFunctionBegin;
  PetscValidLogicalCollectiveReal(ksp, tol, 2);
  PetscTryMethod((ksp), "KSPGMRESSetBreakdownTolerance_C", (KSP, PetscReal), (ksp, tol));
  PetscFunctionReturn(0);
}

/*MC
     KSPGMRES - Implements the Generalized Minimal Residual method [1] with restart

   Options Database Keys:
+   -ksp_gmres_restart <restart> - the number of Krylov directions to orthogonalize against
.   -ksp_gmres_haptol <tol> - sets the tolerance for "happy ending" (exact convergence)
.   -ksp_gmres_preallocate - preallocate all the Krylov search directions initially (otherwise groups of
                             vectors are allocated as needed)
.   -ksp_gmres_classicalgramschmidt - use classical (unmodified) Gram-Schmidt to orthogonalize against the Krylov space (fast) (the default)
.   -ksp_gmres_modifiedgramschmidt - use modified Gram-Schmidt in the orthogonalization (more stable, but slower)
.   -ksp_gmres_cgs_refinement_type <refine_never,refine_ifneeded,refine_always> - determine if iterative refinement is used to increase the
                                   stability of the classical Gram-Schmidt  orthogonalization.
-   -ksp_gmres_krylov_monitor - plot the Krylov space generated

   Level: beginner

   Note:
    Left and right preconditioning are supported, but not symmetric preconditioning.

   Reference:
.  [1] - YOUCEF SAAD AND MARTIN H. SCHULTZ, GMRES: A GENERALIZED MINIMAL RESIDUAL ALGORITHM FOR SOLVING NONSYMMETRIC LINEAR SYSTEMS.
          SIAM J. ScI. STAT. COMPUT. Vo|. 7, No. 3, July 1986.

.seealso: [](chapter_ksp), `KSPCreate()`, `KSPSetType()`, `KSPType`, `KSP`, `KSPFGMRES`, `KSPLGMRES`,
          `KSPGMRESSetRestart()`, `KSPGMRESSetHapTol()`, `KSPGMRESSetPreAllocateVectors()`, `KSPGMRESSetOrthogonalization()`, `KSPGMRESGetOrthogonalization()`,
          `KSPGMRESClassicalGramSchmidtOrthogonalization()`, `KSPGMRESModifiedGramSchmidtOrthogonalization()`,
          `KSPGMRESCGSRefinementType`, `KSPGMRESSetCGSRefinementType()`, `KSPGMRESGetCGSRefinementType()`, `KSPGMRESMonitorKrylov()`, `KSPSetPCSide()`
M*/

PETSC_EXTERN PetscErrorCode KSPCreate_GMRES(KSP ksp)
{
  KSP_GMRES *gmres;

  PetscFunctionBegin;
  PetscCall(PetscNew(&gmres));
  ksp->data = (void *)gmres;

  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_PRECONDITIONED, PC_LEFT, 4));
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_UNPRECONDITIONED, PC_RIGHT, 3));
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_PRECONDITIONED, PC_SYMMETRIC, 2));
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_NONE, PC_RIGHT, 1));
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_NONE, PC_LEFT, 1));

  ksp->ops->buildsolution                = KSPBuildSolution_GMRES;
  ksp->ops->setup                        = KSPSetUp_GMRES;
  ksp->ops->solve                        = KSPSolve_GMRES;
  ksp->ops->reset                        = KSPReset_GMRES;
  ksp->ops->destroy                      = KSPDestroy_GMRES;
  ksp->ops->view                         = KSPView_GMRES;
  ksp->ops->setfromoptions               = KSPSetFromOptions_GMRES;
  ksp->ops->computeextremesingularvalues = KSPComputeExtremeSingularValues_GMRES;
  ksp->ops->computeeigenvalues           = KSPComputeEigenvalues_GMRES;
  ksp->ops->computeritz                  = KSPComputeRitz_GMRES;
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPGMRESSetPreAllocateVectors_C", KSPGMRESSetPreAllocateVectors_GMRES));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPGMRESSetOrthogonalization_C", KSPGMRESSetOrthogonalization_GMRES));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPGMRESGetOrthogonalization_C", KSPGMRESGetOrthogonalization_GMRES));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPGMRESSetRestart_C", KSPGMRESSetRestart_GMRES));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPGMRESGetRestart_C", KSPGMRESGetRestart_GMRES));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPGMRESSetHapTol_C", KSPGMRESSetHapTol_GMRES));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPGMRESSetBreakdownTolerance_C", KSPGMRESSetBreakdownTolerance_GMRES));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPGMRESSetCGSRefinementType_C", KSPGMRESSetCGSRefinementType_GMRES));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPGMRESGetCGSRefinementType_C", KSPGMRESGetCGSRefinementType_GMRES));

  gmres->haptol         = 1.0e-30;
  gmres->breakdowntol   = 0.1;
  gmres->q_preallocate  = 0;
  gmres->delta_allocate = GMRES_DELTA_DIRECTIONS;
  gmres->orthog         = KSPGMRESClassicalGramSchmidtOrthogonalization;
  gmres->nrs            = NULL;
  gmres->sol_temp       = NULL;
  gmres->max_k          = GMRES_DEFAULT_MAXK;
  gmres->Rsvd           = NULL;
  gmres->cgstype        = KSP_GMRES_CGS_REFINE_NEVER;
  gmres->orthogwork     = NULL;
  PetscFunctionReturn(0);
}
