#include <petsc/private/dmpleximpl.h> /*I "petscdmplex.h" I*/
#include <petsc/private/snesimpl.h>   /*I "petscsnes.h"   I*/
#include <petscds.h>
#include <petsc/private/petscimpl.h>
#include <petsc/private/petscfeimpl.h>

#ifdef PETSC_HAVE_LIBCEED
  #include <petscdmceed.h>
  #include <petscdmplexceed.h>
#endif

static void pressure_Private(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar p[])
{
  p[0] = u[uOff[1]];
}

/*
  SNESCorrectDiscretePressure_Private - Add a vector in the nullspace to make the continuum integral of the pressure field equal to zero.
  This is normally used only to evaluate convergence rates for the pressure accurately.

  Collective

  Input Parameters:
+ snes      - The `SNES`
. pfield    - The field number for pressure
. nullspace - The pressure nullspace
. u         - The solution vector
- ctx       - An optional user context

  Output Parameter:
. u         - The solution with a continuum pressure integral of zero

  Level: developer

  Note:
  If int(u) = a and int(n) = b, then int(u - a/b n) = a - a/b b = 0. We assume that the nullspace is a single vector given explicitly.

.seealso: [](ch_snes), `SNESConvergedCorrectPressure()`
*/
static PetscErrorCode SNESCorrectDiscretePressure_Private(SNES snes, PetscInt pfield, MatNullSpace nullspace, Vec u, void *ctx)
{
  DM          dm;
  PetscDS     ds;
  const Vec  *nullvecs;
  PetscScalar pintd, *intc, *intn;
  MPI_Comm    comm;
  PetscInt    Nf, Nv;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)snes, &comm));
  PetscCall(SNESGetDM(snes, &dm));
  PetscCheck(dm, comm, PETSC_ERR_ARG_WRONG, "Cannot compute test without a SNES DM");
  PetscCheck(nullspace, comm, PETSC_ERR_ARG_WRONG, "Cannot compute test without a Jacobian nullspace");
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSSetObjective(ds, pfield, pressure_Private));
  PetscCall(MatNullSpaceGetVecs(nullspace, NULL, &Nv, &nullvecs));
  PetscCheck(Nv == 1, comm, PETSC_ERR_ARG_OUTOFRANGE, "Can only handle a single null vector for pressure, not %" PetscInt_FMT, Nv);
  PetscCall(VecDot(nullvecs[0], u, &pintd));
  PetscCheck(PetscAbsScalar(pintd) <= PETSC_SMALL, comm, PETSC_ERR_ARG_WRONG, "Discrete integral of pressure: %g", (double)PetscRealPart(pintd));
  PetscCall(PetscDSGetNumFields(ds, &Nf));
  PetscCall(PetscMalloc2(Nf, &intc, Nf, &intn));
  PetscCall(DMPlexComputeIntegralFEM(dm, nullvecs[0], intn, ctx));
  PetscCall(DMPlexComputeIntegralFEM(dm, u, intc, ctx));
  PetscCall(VecAXPY(u, -intc[pfield] / intn[pfield], nullvecs[0]));
#if defined(PETSC_USE_DEBUG)
  PetscCall(DMPlexComputeIntegralFEM(dm, u, intc, ctx));
  PetscCheck(PetscAbsScalar(intc[pfield]) <= PETSC_SMALL, comm, PETSC_ERR_ARG_WRONG, "Continuum integral of pressure after correction: %g", (double)PetscRealPart(intc[pfield]));
#endif
  PetscCall(PetscFree2(intc, intn));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  SNESConvergedCorrectPressure - The regular `SNES` convergence test that, up on convergence, adds a vector in the nullspace
  to make the continuum integral of the pressure field equal to zero.

  Logically Collective

  Input Parameters:
+ snes  - the `SNES` context
. it    - the iteration (0 indicates before any Newton steps)
. xnorm - 2-norm of current iterate
. gnorm - 2-norm of current step
. f     - 2-norm of function at current iterate
- ctx   - Optional user context

  Output Parameter:
. reason - `SNES_CONVERGED_ITERATING`, `SNES_CONVERGED_ITS`, or `SNES_DIVERGED_FNORM_NAN`

  Options Database Key:
. -snes_convergence_test correct_pressure - see `SNESSetFromOptions()`

  Level: advanced

  Notes:
  In order to use this convergence test, you must set up several PETSc structures. First fields must be added to the `DM`, and a `PetscDS`
  must be created with discretizations of those fields. We currently assume that the pressure field has index 1.
  The pressure field must have a nullspace, likely created using the `DMSetNullSpaceConstructor()` interface.
  Last we must be able to integrate the pressure over the domain, so the `DM` attached to the SNES `must` be a `DMPLEX` at this time.

  Developer Note:
  This is a total misuse of the `SNES` convergence test handling system. It should be removed. Perhaps a `SNESSetPostSolve()` could
  be constructed to handle this process.

.seealso: [](ch_snes), `SNES`, `DM`, `SNESConvergedDefault()`, `SNESSetConvergenceTest()`, `DMSetNullSpaceConstructor()`
@*/
PetscErrorCode SNESConvergedCorrectPressure(SNES snes, PetscInt it, PetscReal xnorm, PetscReal gnorm, PetscReal f, SNESConvergedReason *reason, void *ctx)
{
  PetscBool monitorIntegral = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(SNESConvergedDefault(snes, it, xnorm, gnorm, f, reason, ctx));
  if (monitorIntegral) {
    Mat          J;
    Vec          u;
    MatNullSpace nullspace;
    const Vec   *nullvecs;
    PetscScalar  pintd;

    PetscCall(SNESGetSolution(snes, &u));
    PetscCall(SNESGetJacobian(snes, &J, NULL, NULL, NULL));
    PetscCall(MatGetNullSpace(J, &nullspace));
    PetscCall(MatNullSpaceGetVecs(nullspace, NULL, NULL, &nullvecs));
    PetscCall(VecDot(nullvecs[0], u, &pintd));
    PetscCall(PetscInfo(snes, "SNES: Discrete integral of pressure: %g\n", (double)PetscRealPart(pintd)));
  }
  if (*reason > 0) {
    Mat          J;
    Vec          u;
    MatNullSpace nullspace;
    PetscInt     pfield = 1;

    PetscCall(SNESGetSolution(snes, &u));
    PetscCall(SNESGetJacobian(snes, &J, NULL, NULL, NULL));
    PetscCall(MatGetNullSpace(J, &nullspace));
    PetscCall(SNESCorrectDiscretePressure_Private(snes, pfield, nullspace, u, ctx));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMSNESConvertPlex(DM dm, DM *plex, PetscBool copy)
{
  PetscBool isPlex;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)dm, DMPLEX, &isPlex));
  if (isPlex) {
    *plex = dm;
    PetscCall(PetscObjectReference((PetscObject)dm));
  } else {
    PetscCall(PetscObjectQuery((PetscObject)dm, "dm_plex", (PetscObject *)plex));
    if (!*plex) {
      PetscCall(DMConvert(dm, DMPLEX, plex));
      PetscCall(PetscObjectCompose((PetscObject)dm, "dm_plex", (PetscObject)*plex));
    } else {
      PetscCall(PetscObjectReference((PetscObject)*plex));
    }
    if (copy) {
      PetscCall(DMCopyDMSNES(dm, *plex));
      PetscCall(DMCopyAuxiliaryVec(dm, *plex));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  SNESMonitorFields - Monitors the residual for each field separately

  Collective

  Input Parameters:
+ snes   - the `SNES` context, must have an attached `DM`
. its    - iteration number
. fgnorm - 2-norm of residual
- vf     - `PetscViewerAndFormat` of `PetscViewerType` `PETSCVIEWERASCII`

  Level: intermediate

  Note:
  This routine prints the residual norm at each iteration.

.seealso: [](ch_snes), `SNES`, `SNESMonitorSet()`, `SNESMonitorDefault()`
@*/
PetscErrorCode SNESMonitorFields(SNES snes, PetscInt its, PetscReal fgnorm, PetscViewerAndFormat *vf)
{
  PetscViewer        viewer = vf->viewer;
  Vec                res;
  DM                 dm;
  PetscSection       s;
  const PetscScalar *r;
  PetscReal         *lnorms, *norms;
  PetscInt           numFields, f, pStart, pEnd, p;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 4);
  PetscCall(SNESGetFunction(snes, &res, NULL, NULL));
  PetscCall(SNESGetDM(snes, &dm));
  PetscCall(DMGetLocalSection(dm, &s));
  PetscCall(PetscSectionGetNumFields(s, &numFields));
  PetscCall(PetscSectionGetChart(s, &pStart, &pEnd));
  PetscCall(PetscCalloc2(numFields, &lnorms, numFields, &norms));
  PetscCall(VecGetArrayRead(res, &r));
  for (p = pStart; p < pEnd; ++p) {
    for (f = 0; f < numFields; ++f) {
      PetscInt fdof, foff, d;

      PetscCall(PetscSectionGetFieldDof(s, p, f, &fdof));
      PetscCall(PetscSectionGetFieldOffset(s, p, f, &foff));
      for (d = 0; d < fdof; ++d) lnorms[f] += PetscRealPart(PetscSqr(r[foff + d]));
    }
  }
  PetscCall(VecRestoreArrayRead(res, &r));
  PetscCallMPI(MPIU_Allreduce(lnorms, norms, numFields, MPIU_REAL, MPIU_SUM, PetscObjectComm((PetscObject)dm)));
  PetscCall(PetscViewerPushFormat(viewer, vf->format));
  PetscCall(PetscViewerASCIIAddTab(viewer, ((PetscObject)snes)->tablevel));
  PetscCall(PetscViewerASCIIPrintf(viewer, "%3" PetscInt_FMT " SNES Function norm %14.12e [", its, (double)fgnorm));
  for (f = 0; f < numFields; ++f) {
    if (f > 0) PetscCall(PetscViewerASCIIPrintf(viewer, ", "));
    PetscCall(PetscViewerASCIIPrintf(viewer, "%14.12e", (double)PetscSqrtReal(norms[f])));
  }
  PetscCall(PetscViewerASCIIPrintf(viewer, "]\n"));
  PetscCall(PetscViewerASCIISubtractTab(viewer, ((PetscObject)snes)->tablevel));
  PetscCall(PetscViewerPopFormat(viewer));
  PetscCall(PetscFree2(lnorms, norms));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/********************* SNES callbacks **************************/

/*@
  DMPlexSNESComputeObjectiveFEM - Sums the local objectives from the local input X using pointwise functions specified by the user

  Input Parameters:
+ dm   - The mesh
. X    - Local solution
- user - The user context

  Output Parameter:
. obj - Local objective value

  Level: developer

.seealso: `DM`, `DMPlexSNESComputeResidualFEM()`
@*/
PetscErrorCode DMPlexSNESComputeObjectiveFEM(DM dm, Vec X, PetscReal *obj, void *user)
{
  PetscInt     Nf, cellHeight, cStart, cEnd;
  PetscScalar *cintegral;

  PetscFunctionBegin;
  PetscCall(DMGetNumFields(dm, &Nf));
  PetscCall(DMPlexGetVTKCellHeight(dm, &cellHeight));
  PetscCall(DMPlexGetSimplexOrBoxCells(dm, cellHeight, &cStart, &cEnd));
  PetscCall(PetscCalloc1((cEnd - cStart) * Nf, &cintegral));
  PetscCall(PetscLogEventBegin(DMPLEX_IntegralFEM, dm, 0, 0, 0));
  PetscCall(DMPlexComputeIntegral_Internal(dm, X, cStart, cEnd, cintegral, user));
  /* Sum up values */
  *obj = 0;
  for (PetscInt c = cStart; c < cEnd; ++c)
    for (PetscInt f = 0; f < Nf; ++f) *obj += PetscRealPart(cintegral[(c - cStart) * Nf + f]);
  PetscCall(PetscLogEventBegin(DMPLEX_IntegralFEM, dm, 0, 0, 0));
  PetscCall(PetscFree(cintegral));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexSNESComputeResidualFEM - Sums the local residual into vector `F` from the local input `X` using pointwise functions specified by the user

  Input Parameters:
+ dm   - The mesh
. X    - Local solution
- user - The user context

  Output Parameter:
. F - Local output vector

  Level: developer

  Note:
  The residual is summed into `F`; the caller is responsible for using `VecZeroEntries()` or otherwise ensuring that any data in `F` is intentional.

.seealso: [](ch_snes), `DM`, `DMPLEX`, `DMSNESComputeJacobianAction()`
@*/
PetscErrorCode DMPlexSNESComputeResidualFEM(DM dm, Vec X, Vec F, void *user)
{
  DM       plex;
  IS       allcellIS;
  PetscInt Nds, s;

  PetscFunctionBegin;
  PetscCall(DMSNESConvertPlex(dm, &plex, PETSC_TRUE));
  PetscCall(DMPlexGetAllCells_Internal(plex, &allcellIS));
  PetscCall(DMGetNumDS(dm, &Nds));
  for (s = 0; s < Nds; ++s) {
    PetscDS      ds;
    IS           cellIS;
    PetscFormKey key;

    PetscCall(DMGetRegionNumDS(dm, s, &key.label, NULL, &ds, NULL));
    key.value = 0;
    key.field = 0;
    key.part  = 0;
    if (!key.label) {
      PetscCall(PetscObjectReference((PetscObject)allcellIS));
      cellIS = allcellIS;
    } else {
      IS pointIS;

      key.value = 1;
      PetscCall(DMLabelGetStratumIS(key.label, key.value, &pointIS));
      PetscCall(ISIntersect_Caching_Internal(allcellIS, pointIS, &cellIS));
      PetscCall(ISDestroy(&pointIS));
    }
    PetscCall(DMPlexComputeResidualByKey(plex, key, cellIS, PETSC_MIN_REAL, X, NULL, 0.0, F, user));
    PetscCall(ISDestroy(&cellIS));
  }
  PetscCall(ISDestroy(&allcellIS));
  PetscCall(DMDestroy(&plex));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexSNESComputeResidualDS - Sums the local residual into vector `F` from the local input `X` using all pointwise functions with unique keys in the `PetscDS`

  Input Parameters:
+ dm   - The mesh
. X    - Local solution
- user - The user context

  Output Parameter:
. F - Local output vector

  Level: developer

  Note:
  The residual is summed into `F`; the caller is responsible for using `VecZeroEntries()` or otherwise ensuring that any data in `F` is intentional.

.seealso: [](ch_snes), `DM`, `DMPLEX`, `DMPlexComputeJacobianAction()`
@*/
PetscErrorCode DMPlexSNESComputeResidualDS(DM dm, Vec X, Vec F, void *user)
{
  DM       plex;
  IS       allcellIS;
  PetscInt Nds, s;

  PetscFunctionBegin;
  PetscCall(DMSNESConvertPlex(dm, &plex, PETSC_TRUE));
  PetscCall(DMPlexGetAllCells_Internal(plex, &allcellIS));
  PetscCall(DMGetNumDS(dm, &Nds));
  for (s = 0; s < Nds; ++s) {
    PetscDS ds;
    DMLabel label;
    IS      cellIS;

    PetscCall(DMGetRegionNumDS(dm, s, &label, NULL, &ds, NULL));
    {
      PetscWeakFormKind resmap[2] = {PETSC_WF_F0, PETSC_WF_F1};
      PetscWeakForm     wf;
      PetscInt          Nm = 2, m, Nk = 0, k, kp, off = 0;
      PetscFormKey     *reskeys;

      /* Get unique residual keys */
      for (m = 0; m < Nm; ++m) {
        PetscInt Nkm;
        PetscCall(PetscHMapFormGetSize(ds->wf->form[resmap[m]], &Nkm));
        Nk += Nkm;
      }
      PetscCall(PetscMalloc1(Nk, &reskeys));
      for (m = 0; m < Nm; ++m) PetscCall(PetscHMapFormGetKeys(ds->wf->form[resmap[m]], &off, reskeys));
      PetscCheck(off == Nk, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Number of keys %" PetscInt_FMT " should be %" PetscInt_FMT, off, Nk);
      PetscCall(PetscFormKeySort(Nk, reskeys));
      for (k = 0, kp = 1; kp < Nk; ++kp) {
        if ((reskeys[k].label != reskeys[kp].label) || (reskeys[k].value != reskeys[kp].value)) {
          ++k;
          if (kp != k) reskeys[k] = reskeys[kp];
        }
      }
      Nk = k;

      PetscCall(PetscDSGetWeakForm(ds, &wf));
      for (k = 0; k < Nk; ++k) {
        DMLabel  label = reskeys[k].label;
        PetscInt val   = reskeys[k].value;

        if (!label) {
          PetscCall(PetscObjectReference((PetscObject)allcellIS));
          cellIS = allcellIS;
        } else {
          IS pointIS;

          PetscCall(DMLabelGetStratumIS(label, val, &pointIS));
          PetscCall(ISIntersect_Caching_Internal(allcellIS, pointIS, &cellIS));
          PetscCall(ISDestroy(&pointIS));
        }
        PetscCall(DMPlexComputeResidualByKey(plex, reskeys[k], cellIS, PETSC_MIN_REAL, X, NULL, 0.0, F, user));
        PetscCall(ISDestroy(&cellIS));
      }
      PetscCall(PetscFree(reskeys));
    }
  }
  PetscCall(ISDestroy(&allcellIS));
  PetscCall(DMDestroy(&plex));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexSNESComputeBoundaryFEM - Form the boundary values for the local input `X`

  Input Parameters:
+ dm   - The mesh
- user - The user context

  Output Parameter:
. X - Local solution

  Level: developer

.seealso: [](ch_snes), `DM`, `DMPLEX`, `DMPlexComputeJacobianAction()`
@*/
PetscErrorCode DMPlexSNESComputeBoundaryFEM(DM dm, Vec X, void *user)
{
  DM plex;

  PetscFunctionBegin;
  PetscCall(DMSNESConvertPlex(dm, &plex, PETSC_TRUE));
  PetscCall(DMPlexInsertBoundaryValues(plex, PETSC_TRUE, X, PETSC_MIN_REAL, NULL, NULL, NULL));
  PetscCall(DMDestroy(&plex));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSNESComputeJacobianAction - Compute the action of the Jacobian J(`X`) on `Y`

  Input Parameters:
+ dm   - The `DM`
. X    - Local solution vector
. Y    - Local input vector
- user - The user context

  Output Parameter:
. F - local output vector

  Level: developer

  Note:
  Users will typically use `DMSNESCreateJacobianMF()` followed by `MatMult()` instead of calling this routine directly.

  This only works with `DMPLEX`

  Developer Note:
  This should be called `DMPlexSNESComputeJacobianAction()`

.seealso: [](ch_snes), `DM`, `DMSNESCreateJacobianMF()`, `DMPlexSNESComputeResidualFEM()`
@*/
PetscErrorCode DMSNESComputeJacobianAction(DM dm, Vec X, Vec Y, Vec F, void *user)
{
  DM       plex;
  IS       allcellIS;
  PetscInt Nds, s;

  PetscFunctionBegin;
  PetscCall(DMSNESConvertPlex(dm, &plex, PETSC_TRUE));
  PetscCall(DMPlexGetAllCells_Internal(plex, &allcellIS));
  PetscCall(DMGetNumDS(dm, &Nds));
  for (s = 0; s < Nds; ++s) {
    PetscDS ds;
    DMLabel label;
    IS      cellIS;

    PetscCall(DMGetRegionNumDS(dm, s, &label, NULL, &ds, NULL));
    {
      PetscWeakFormKind jacmap[4] = {PETSC_WF_G0, PETSC_WF_G1, PETSC_WF_G2, PETSC_WF_G3};
      PetscWeakForm     wf;
      PetscInt          Nm = 4, m, Nk = 0, k, kp, off = 0;
      PetscFormKey     *jackeys;

      /* Get unique Jacobian keys */
      for (m = 0; m < Nm; ++m) {
        PetscInt Nkm;
        PetscCall(PetscHMapFormGetSize(ds->wf->form[jacmap[m]], &Nkm));
        Nk += Nkm;
      }
      PetscCall(PetscMalloc1(Nk, &jackeys));
      for (m = 0; m < Nm; ++m) PetscCall(PetscHMapFormGetKeys(ds->wf->form[jacmap[m]], &off, jackeys));
      PetscCheck(off == Nk, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Number of keys %" PetscInt_FMT " should be %" PetscInt_FMT, off, Nk);
      PetscCall(PetscFormKeySort(Nk, jackeys));
      for (k = 0, kp = 1; kp < Nk; ++kp) {
        if ((jackeys[k].label != jackeys[kp].label) || (jackeys[k].value != jackeys[kp].value)) {
          ++k;
          if (kp != k) jackeys[k] = jackeys[kp];
        }
      }
      Nk = k;

      PetscCall(PetscDSGetWeakForm(ds, &wf));
      for (k = 0; k < Nk; ++k) {
        DMLabel  label = jackeys[k].label;
        PetscInt val   = jackeys[k].value;

        if (!label) {
          PetscCall(PetscObjectReference((PetscObject)allcellIS));
          cellIS = allcellIS;
        } else {
          IS pointIS;

          PetscCall(DMLabelGetStratumIS(label, val, &pointIS));
          PetscCall(ISIntersect_Caching_Internal(allcellIS, pointIS, &cellIS));
          PetscCall(ISDestroy(&pointIS));
        }
        PetscCall(DMPlexComputeJacobianActionByKey(plex, jackeys[k], cellIS, 0.0, 0.0, X, NULL, Y, F, user));
        PetscCall(ISDestroy(&cellIS));
      }
      PetscCall(PetscFree(jackeys));
    }
  }
  PetscCall(ISDestroy(&allcellIS));
  PetscCall(DMDestroy(&plex));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexSNESComputeJacobianFEM - Form the local portion of the Jacobian matrix `Jac` at the local solution `X` using pointwise functions specified by the user.

  Input Parameters:
+ dm   - The `DM`
. X    - Local input vector
- user - The user context

  Output Parameters:
+ Jac  - Jacobian matrix
- JacP - approximate Jacobian from which the preconditioner will be built, often `Jac`

  Level: developer

  Note:
  We form the residual one batch of elements at a time. This allows us to offload work onto an accelerator,
  like a GPU, or vectorize on a multicore machine.

.seealso: [](ch_snes), `DMPLEX`, `Mat`
@*/
PetscErrorCode DMPlexSNESComputeJacobianFEM(DM dm, Vec X, Mat Jac, Mat JacP, void *user)
{
  DM        plex;
  IS        allcellIS;
  PetscBool hasJac, hasPrec;
  PetscInt  Nds, s;

  PetscFunctionBegin;
  PetscCall(DMSNESConvertPlex(dm, &plex, PETSC_TRUE));
  PetscCall(DMPlexGetAllCells_Internal(plex, &allcellIS));
  PetscCall(DMGetNumDS(dm, &Nds));
  for (s = 0; s < Nds; ++s) {
    PetscDS      ds;
    IS           cellIS;
    PetscFormKey key;

    PetscCall(DMGetRegionNumDS(dm, s, &key.label, NULL, &ds, NULL));
    key.value = 0;
    key.field = 0;
    key.part  = 0;
    if (!key.label) {
      PetscCall(PetscObjectReference((PetscObject)allcellIS));
      cellIS = allcellIS;
    } else {
      IS pointIS;

      key.value = 1;
      PetscCall(DMLabelGetStratumIS(key.label, key.value, &pointIS));
      PetscCall(ISIntersect_Caching_Internal(allcellIS, pointIS, &cellIS));
      PetscCall(ISDestroy(&pointIS));
    }
    if (!s) {
      PetscCall(PetscDSHasJacobian(ds, &hasJac));
      PetscCall(PetscDSHasJacobianPreconditioner(ds, &hasPrec));
      if (hasJac && hasPrec) PetscCall(MatZeroEntries(Jac));
      PetscCall(MatZeroEntries(JacP));
    }
    PetscCall(DMPlexComputeJacobianByKey(plex, key, cellIS, 0.0, 0.0, X, NULL, Jac, JacP, user));
    PetscCall(ISDestroy(&cellIS));
  }
  PetscCall(ISDestroy(&allcellIS));
  PetscCall(DMDestroy(&plex));
  PetscFunctionReturn(PETSC_SUCCESS);
}

struct _DMSNESJacobianMFCtx {
  DM    dm;
  Vec   X;
  void *ctx;
};

static PetscErrorCode DMSNESJacobianMF_Destroy_Private(Mat A)
{
  struct _DMSNESJacobianMFCtx *ctx;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A, &ctx));
  PetscCall(MatShellSetContext(A, NULL));
  PetscCall(DMDestroy(&ctx->dm));
  PetscCall(VecDestroy(&ctx->X));
  PetscCall(PetscFree(ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMSNESJacobianMF_Mult_Private(Mat A, Vec Y, Vec Z)
{
  struct _DMSNESJacobianMFCtx *ctx;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A, &ctx));
  PetscCall(DMSNESComputeJacobianAction(ctx->dm, ctx->X, Y, Z, ctx->ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSNESCreateJacobianMF - Create a `Mat` which computes the action of the Jacobian matrix-free

  Collective

  Input Parameters:
+ dm   - The `DM`
. X    - The evaluation point for the Jacobian
- user - A user context, or `NULL`

  Output Parameter:
. J - The `Mat`

  Level: advanced

  Notes:
  Vec `X` is kept in `J`, so updating `X` then updates the evaluation point.

  This only works for `DMPLEX`

.seealso: [](ch_snes), `DM`, `SNES`, `DMSNESComputeJacobianAction()`
@*/
PetscErrorCode DMSNESCreateJacobianMF(DM dm, Vec X, void *user, Mat *J)
{
  struct _DMSNESJacobianMFCtx *ctx;
  PetscInt                     n, N;

  PetscFunctionBegin;
  PetscCall(MatCreate(PetscObjectComm((PetscObject)dm), J));
  PetscCall(MatSetType(*J, MATSHELL));
  PetscCall(VecGetLocalSize(X, &n));
  PetscCall(VecGetSize(X, &N));
  PetscCall(MatSetSizes(*J, n, n, N, N));
  PetscCall(PetscObjectReference((PetscObject)dm));
  PetscCall(PetscObjectReference((PetscObject)X));
  PetscCall(PetscMalloc1(1, &ctx));
  ctx->dm  = dm;
  ctx->X   = X;
  ctx->ctx = user;
  PetscCall(MatShellSetContext(*J, ctx));
  PetscCall(MatShellSetOperation(*J, MATOP_DESTROY, (PetscErrorCodeFn *)DMSNESJacobianMF_Destroy_Private));
  PetscCall(MatShellSetOperation(*J, MATOP_MULT, (PetscErrorCodeFn *)DMSNESJacobianMF_Mult_Private));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatComputeNeumannOverlap_Plex(Mat J, PetscReal t, Vec X, Vec X_t, PetscReal s, IS ovl, void *ctx)
{
  SNES   snes;
  Mat    pJ;
  DM     ovldm, origdm;
  DMSNES sdm;
  PetscErrorCode (*bfun)(DM, Vec, void *);
  PetscErrorCode (*jfun)(DM, Vec, Mat, Mat, void *);
  void *bctx, *jctx;

  PetscFunctionBegin;
  PetscCall(PetscObjectQuery((PetscObject)ovl, "_DM_Overlap_HPDDM_MATIS", (PetscObject *)&pJ));
  PetscCheck(pJ, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Missing overlapping Mat");
  PetscCall(PetscObjectQuery((PetscObject)ovl, "_DM_Original_HPDDM", (PetscObject *)&origdm));
  PetscCheck(origdm, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Missing original DM");
  PetscCall(MatGetDM(pJ, &ovldm));
  PetscCall(DMSNESGetBoundaryLocal(origdm, &bfun, &bctx));
  PetscCall(DMSNESSetBoundaryLocal(ovldm, bfun, bctx));
  PetscCall(DMSNESGetJacobianLocal(origdm, &jfun, &jctx));
  PetscCall(DMSNESSetJacobianLocal(ovldm, jfun, jctx));
  PetscCall(PetscObjectQuery((PetscObject)ovl, "_DM_Overlap_HPDDM_SNES", (PetscObject *)&snes));
  if (!snes) {
    PetscCall(SNESCreate(PetscObjectComm((PetscObject)ovl), &snes));
    PetscCall(SNESSetDM(snes, ovldm));
    PetscCall(PetscObjectCompose((PetscObject)ovl, "_DM_Overlap_HPDDM_SNES", (PetscObject)snes));
    PetscCall(PetscObjectDereference((PetscObject)snes));
  }
  PetscCall(DMGetDMSNES(ovldm, &sdm));
  PetscCall(VecLockReadPush(X));
  {
    void *ctx;
    PetscErrorCode (*J)(SNES, Vec, Mat, Mat, void *);
    PetscCall(DMSNESGetJacobian(ovldm, &J, &ctx));
    PetscCallBack("SNES callback Jacobian", (*J)(snes, X, pJ, pJ, ctx));
  }
  PetscCall(VecLockReadPop(X));
  /* this is a no-hop, just in case we decide to change the placeholder for the local Neumann matrix */
  {
    Mat locpJ;

    PetscCall(MatISGetLocalMat(pJ, &locpJ));
    PetscCall(MatCopy(locpJ, J, SAME_NONZERO_PATTERN));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexSetSNESLocalFEM - Use `DMPLEX`'s internal FEM routines to compute `SNES` boundary values, objective, residual, and Jacobian.

  Input Parameters:
+ dm      - The `DM` object
. use_obj - Use the objective function callback
- ctx     - The user context that will be passed to pointwise evaluation routines

  Level: developer

.seealso: [](ch_snes),`DMPLEX`, `SNES`, `PetscDSAddBoundary()`, `PetscDSSetObjective()`, `PetscDSSetResidual()`, `PetscDSSetJacobian()`
@*/
PetscErrorCode DMPlexSetSNESLocalFEM(DM dm, PetscBool use_obj, void *ctx)
{
  PetscBool useCeed;

  PetscFunctionBegin;
  PetscCall(DMPlexGetUseCeed(dm, &useCeed));
  PetscCall(DMSNESSetBoundaryLocal(dm, DMPlexSNESComputeBoundaryFEM, ctx));
  if (use_obj) PetscCall(DMSNESSetObjectiveLocal(dm, DMPlexSNESComputeObjectiveFEM, ctx));
  if (useCeed) {
#ifdef PETSC_HAVE_LIBCEED
    PetscCall(DMSNESSetFunctionLocal(dm, DMPlexSNESComputeResidualCEED, ctx));
#else
    SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Cannot use CEED traversals without LibCEED. Rerun configure with --download-ceed");
#endif
  } else PetscCall(DMSNESSetFunctionLocal(dm, DMPlexSNESComputeResidualFEM, ctx));
  PetscCall(DMSNESSetJacobianLocal(dm, DMPlexSNESComputeJacobianFEM, ctx));
  PetscCall(PetscObjectComposeFunction((PetscObject)dm, "MatComputeNeumannOverlap_C", MatComputeNeumannOverlap_Plex));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSNESCheckDiscretization - Check the discretization error of the exact solution

  Input Parameters:
+ snes - the `SNES` object
. dm   - the `DM`
. t    - the time
. u    - a `DM` vector
- tol  - A tolerance for the check, or -1 to print the results instead

  Output Parameter:
. error - An array which holds the discretization error in each field, or `NULL`

  Level: developer

  Note:
  The user must call `PetscDSSetExactSolution()` beforehand

  Developer Note:
  How is this related to `PetscConvEst`?

.seealso: [](ch_snes), `PetscDSSetExactSolution()`, `DNSNESCheckFromOptions()`, `DMSNESCheckResidual()`, `DMSNESCheckJacobian()`
@*/
PetscErrorCode DMSNESCheckDiscretization(SNES snes, DM dm, PetscReal t, Vec u, PetscReal tol, PetscReal error[])
{
  PetscErrorCode (**exacts)(PetscInt, PetscReal, const PetscReal x[], PetscInt, PetscScalar *u, void *ctx);
  void     **ectxs;
  PetscReal *err;
  MPI_Comm   comm;
  PetscInt   Nf, f;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscValidHeaderSpecific(dm, DM_CLASSID, 2);
  PetscValidHeaderSpecific(u, VEC_CLASSID, 4);
  if (error) PetscAssertPointer(error, 6);

  PetscCall(DMComputeExactSolution(dm, t, u, NULL));
  PetscCall(VecViewFromOptions(u, NULL, "-vec_view"));

  PetscCall(PetscObjectGetComm((PetscObject)snes, &comm));
  PetscCall(DMGetNumFields(dm, &Nf));
  PetscCall(PetscCalloc3(Nf, &exacts, Nf, &ectxs, PetscMax(1, Nf), &err));
  {
    PetscInt Nds, s;

    PetscCall(DMGetNumDS(dm, &Nds));
    for (s = 0; s < Nds; ++s) {
      PetscDS         ds;
      DMLabel         label;
      IS              fieldIS;
      const PetscInt *fields;
      PetscInt        dsNf, f;

      PetscCall(DMGetRegionNumDS(dm, s, &label, &fieldIS, &ds, NULL));
      PetscCall(PetscDSGetNumFields(ds, &dsNf));
      PetscCall(ISGetIndices(fieldIS, &fields));
      for (f = 0; f < dsNf; ++f) {
        const PetscInt field = fields[f];
        PetscCall(PetscDSGetExactSolution(ds, field, &exacts[field], &ectxs[field]));
      }
      PetscCall(ISRestoreIndices(fieldIS, &fields));
    }
  }
  if (Nf > 1) {
    PetscCall(DMComputeL2FieldDiff(dm, t, exacts, ectxs, u, err));
    if (tol >= 0.0) {
      for (f = 0; f < Nf; ++f) PetscCheck(err[f] <= tol, comm, PETSC_ERR_ARG_WRONG, "L_2 Error %g for field %" PetscInt_FMT " exceeds tolerance %g", (double)err[f], f, (double)tol);
    } else if (error) {
      for (f = 0; f < Nf; ++f) error[f] = err[f];
    } else {
      PetscCall(PetscPrintf(comm, "L_2 Error: ["));
      for (f = 0; f < Nf; ++f) {
        if (f) PetscCall(PetscPrintf(comm, ", "));
        PetscCall(PetscPrintf(comm, "%g", (double)err[f]));
      }
      PetscCall(PetscPrintf(comm, "]\n"));
    }
  } else {
    PetscCall(DMComputeL2Diff(dm, t, exacts, ectxs, u, &err[0]));
    if (tol >= 0.0) {
      PetscCheck(err[0] <= tol, comm, PETSC_ERR_ARG_WRONG, "L_2 Error %g exceeds tolerance %g", (double)err[0], (double)tol);
    } else if (error) {
      error[0] = err[0];
    } else {
      PetscCall(PetscPrintf(comm, "L_2 Error: %g\n", (double)err[0]));
    }
  }
  PetscCall(PetscFree3(exacts, ectxs, err));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSNESCheckResidual - Check the residual of the exact solution

  Input Parameters:
+ snes - the `SNES` object
. dm   - the `DM`
. u    - a `DM` vector
- tol  - A tolerance for the check, or -1 to print the results instead

  Output Parameter:
. residual - The residual norm of the exact solution, or `NULL`

  Level: developer

.seealso: [](ch_snes), `DNSNESCheckFromOptions()`, `DMSNESCheckDiscretization()`, `DMSNESCheckJacobian()`
@*/
PetscErrorCode DMSNESCheckResidual(SNES snes, DM dm, Vec u, PetscReal tol, PetscReal *residual)
{
  MPI_Comm  comm;
  Vec       r;
  PetscReal res;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscValidHeaderSpecific(dm, DM_CLASSID, 2);
  PetscValidHeaderSpecific(u, VEC_CLASSID, 3);
  if (residual) PetscAssertPointer(residual, 5);
  PetscCall(PetscObjectGetComm((PetscObject)snes, &comm));
  PetscCall(DMComputeExactSolution(dm, 0.0, u, NULL));
  PetscCall(VecDuplicate(u, &r));
  PetscCall(SNESComputeFunction(snes, u, r));
  PetscCall(VecNorm(r, NORM_2, &res));
  if (tol >= 0.0) {
    PetscCheck(res <= tol, comm, PETSC_ERR_ARG_WRONG, "L_2 Residual %g exceeds tolerance %g", (double)res, (double)tol);
  } else if (residual) {
    *residual = res;
  } else {
    PetscCall(PetscPrintf(comm, "L_2 Residual: %g\n", (double)res));
    PetscCall(VecFilter(r, 1.0e-10));
    PetscCall(PetscObjectSetName((PetscObject)r, "Initial Residual"));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)r, "res_"));
    PetscCall(PetscObjectCompose((PetscObject)r, "__Vec_bc_zero__", (PetscObject)snes));
    PetscCall(VecViewFromOptions(r, NULL, "-vec_view"));
    PetscCall(PetscObjectCompose((PetscObject)r, "__Vec_bc_zero__", NULL));
  }
  PetscCall(VecDestroy(&r));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSNESCheckJacobian - Check the Jacobian of the exact solution against the residual using the Taylor Test

  Input Parameters:
+ snes - the `SNES` object
. dm   - the `DM`
. u    - a `DM` vector
- tol  - A tolerance for the check, or -1 to print the results instead

  Output Parameters:
+ isLinear - Flag indicaing that the function looks linear, or `NULL`
- convRate - The rate of convergence of the linear model, or `NULL`

  Level: developer

.seealso: [](ch_snes), `DNSNESCheckFromOptions()`, `DMSNESCheckDiscretization()`, `DMSNESCheckResidual()`
@*/
PetscErrorCode DMSNESCheckJacobian(SNES snes, DM dm, Vec u, PetscReal tol, PetscBool *isLinear, PetscReal *convRate)
{
  MPI_Comm     comm;
  PetscDS      ds;
  Mat          J, M;
  MatNullSpace nullspace;
  PetscReal    slope, intercept;
  PetscBool    hasJac, hasPrec, isLin = PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  if (dm) PetscValidHeaderSpecific(dm, DM_CLASSID, 2);
  if (u) PetscValidHeaderSpecific(u, VEC_CLASSID, 3);
  if (isLinear) PetscAssertPointer(isLinear, 5);
  if (convRate) PetscAssertPointer(convRate, 6);
  PetscCall(PetscObjectGetComm((PetscObject)snes, &comm));
  if (!dm) PetscCall(SNESGetDM(snes, &dm));
  if (u) PetscCall(DMComputeExactSolution(dm, 0.0, u, NULL));
  else PetscCall(SNESGetSolution(snes, &u));
  /* Create and view matrices */
  PetscCall(DMCreateMatrix(dm, &J));
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSHasJacobian(ds, &hasJac));
  PetscCall(PetscDSHasJacobianPreconditioner(ds, &hasPrec));
  if (hasJac && hasPrec) {
    PetscCall(DMCreateMatrix(dm, &M));
    PetscCall(SNESComputeJacobian(snes, u, J, M));
    PetscCall(PetscObjectSetName((PetscObject)M, "Matrix used to construct preconditioner"));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)M, "jacpre_"));
    PetscCall(MatViewFromOptions(M, NULL, "-mat_view"));
    PetscCall(MatDestroy(&M));
  } else {
    PetscCall(SNESComputeJacobian(snes, u, J, J));
  }
  PetscCall(PetscObjectSetName((PetscObject)J, "Jacobian"));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)J, "jac_"));
  PetscCall(MatViewFromOptions(J, NULL, "-mat_view"));
  /* Check nullspace */
  PetscCall(MatGetNullSpace(J, &nullspace));
  if (nullspace) {
    PetscBool isNull;
    PetscCall(MatNullSpaceTest(nullspace, J, &isNull));
    PetscCheck(isNull, comm, PETSC_ERR_PLIB, "The null space calculated for the system operator is invalid.");
  }
  /* Taylor test */
  {
    PetscRandom rand;
    Vec         du, uhat, r, rhat, df;
    PetscReal   h;
    PetscReal  *es, *hs, *errors;
    PetscReal   hMax = 1.0, hMin = 1e-6, hMult = 0.1;
    PetscInt    Nv, v;

    /* Choose a perturbation direction */
    PetscCall(PetscRandomCreate(comm, &rand));
    PetscCall(VecDuplicate(u, &du));
    PetscCall(VecSetRandom(du, rand));
    PetscCall(PetscRandomDestroy(&rand));
    PetscCall(VecDuplicate(u, &df));
    PetscCall(MatMult(J, du, df));
    /* Evaluate residual at u, F(u), save in vector r */
    PetscCall(VecDuplicate(u, &r));
    PetscCall(SNESComputeFunction(snes, u, r));
    /* Look at the convergence of our Taylor approximation as we approach u */
    for (h = hMax, Nv = 0; h >= hMin; h *= hMult, ++Nv);
    PetscCall(PetscCalloc3(Nv, &es, Nv, &hs, Nv, &errors));
    PetscCall(VecDuplicate(u, &uhat));
    PetscCall(VecDuplicate(u, &rhat));
    for (h = hMax, Nv = 0; h >= hMin; h *= hMult, ++Nv) {
      PetscCall(VecWAXPY(uhat, h, du, u));
      /* F(\hat u) \approx F(u) + J(u) (uhat - u) = F(u) + h * J(u) du */
      PetscCall(SNESComputeFunction(snes, uhat, rhat));
      PetscCall(VecAXPBYPCZ(rhat, -1.0, -h, 1.0, r, df));
      PetscCall(VecNorm(rhat, NORM_2, &errors[Nv]));

      es[Nv] = errors[Nv] == 0 ? -16.0 : PetscLog10Real(errors[Nv]);
      hs[Nv] = PetscLog10Real(h);
    }
    PetscCall(VecDestroy(&uhat));
    PetscCall(VecDestroy(&rhat));
    PetscCall(VecDestroy(&df));
    PetscCall(VecDestroy(&r));
    PetscCall(VecDestroy(&du));
    for (v = 0; v < Nv; ++v) {
      if ((tol >= 0) && (errors[v] > tol)) break;
      else if (errors[v] > PETSC_SMALL) break;
    }
    if (v == Nv) isLin = PETSC_TRUE;
    PetscCall(PetscLinearRegression(Nv, hs, es, &slope, &intercept));
    PetscCall(PetscFree3(es, hs, errors));
    /* Slope should be about 2 */
    if (tol >= 0) {
      PetscCheck(isLin || PetscAbsReal(2 - slope) <= tol, comm, PETSC_ERR_ARG_WRONG, "Taylor approximation convergence rate should be 2, not %0.2f", (double)slope);
    } else if (isLinear || convRate) {
      if (isLinear) *isLinear = isLin;
      if (convRate) *convRate = slope;
    } else {
      if (!isLin) PetscCall(PetscPrintf(comm, "Taylor approximation converging at order %3.2f\n", (double)slope));
      else PetscCall(PetscPrintf(comm, "Function appears to be linear\n"));
    }
  }
  PetscCall(MatDestroy(&J));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMSNESCheck_Internal(SNES snes, DM dm, Vec u)
{
  PetscFunctionBegin;
  PetscCall(DMSNESCheckDiscretization(snes, dm, 0.0, u, -1.0, NULL));
  PetscCall(DMSNESCheckResidual(snes, dm, u, -1.0, NULL));
  PetscCall(DMSNESCheckJacobian(snes, dm, u, -1.0, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSNESCheckFromOptions - Check the residual and Jacobian functions using the exact solution by outputting some diagnostic information

  Input Parameters:
+ snes - the `SNES` object
- u    - representative `SNES` vector

  Level: developer

  Note:
  The user must call `PetscDSSetExactSolution()` before this call

.seealso: [](ch_snes), `SNES`, `DM`
@*/
PetscErrorCode DMSNESCheckFromOptions(SNES snes, Vec u)
{
  DM        dm;
  Vec       sol;
  PetscBool check;

  PetscFunctionBegin;
  PetscCall(PetscOptionsHasName(((PetscObject)snes)->options, ((PetscObject)snes)->prefix, "-dmsnes_check", &check));
  if (!check) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(SNESGetDM(snes, &dm));
  PetscCall(VecDuplicate(u, &sol));
  PetscCall(SNESSetSolution(snes, sol));
  PetscCall(DMSNESCheck_Internal(snes, dm, sol));
  PetscCall(VecDestroy(&sol));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexSetSNESVariableBounds - Compute upper and lower bounds for the solution using pointsie functions from the `PetscDS`

  Collective

  Input Parameters:
+ dm   - The `DM` object
- snes - the `SNES` object

  Level: intermediate

  Notes:
  This calls `SNESVISetVariableBounds()` after generating the bounds vectors, so it only applied to `SNESVI` solves.

  We project the actual bounds into the current finite element space so that they become more accurate with refinement.

.seealso: `SNESVISetVariableBounds()`, `SNESVI`, [](ch_snes), `DM`
@*/
PetscErrorCode DMPlexSetSNESVariableBounds(DM dm, SNES snes)
{
  PetscDS              ds;
  Vec                  lb, ub;
  PetscSimplePointFn **lfuncs, **ufuncs;
  void               **lctxs, **uctxs;
  PetscBool            hasBound, hasLower = PETSC_FALSE, hasUpper = PETSC_FALSE;
  PetscInt             Nf;

  PetscFunctionBegin;
  PetscCall(DMHasBound(dm, &hasBound));
  if (!hasBound) PetscFunctionReturn(PETSC_SUCCESS);
  // TODO Generalize for multiple DSes
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSGetNumFields(ds, &Nf));
  PetscCall(PetscMalloc4(Nf, &lfuncs, Nf, &lctxs, Nf, &ufuncs, Nf, &uctxs));
  for (PetscInt f = 0; f < Nf; ++f) {
    PetscCall(PetscDSGetLowerBound(ds, f, &lfuncs[f], &lctxs[f]));
    PetscCall(PetscDSGetUpperBound(ds, f, &ufuncs[f], &uctxs[f]));
    if (lfuncs[f]) hasLower = PETSC_TRUE;
    if (ufuncs[f]) hasUpper = PETSC_TRUE;
  }
  PetscCall(DMCreateGlobalVector(dm, &lb));
  PetscCall(DMCreateGlobalVector(dm, &ub));
  PetscCall(PetscObjectSetName((PetscObject)lb, "Lower Bound"));
  PetscCall(PetscObjectSetName((PetscObject)ub, "Upper Bound"));
  PetscCall(VecSet(lb, PETSC_NINFINITY));
  PetscCall(VecSet(ub, PETSC_INFINITY));
  if (hasLower) PetscCall(DMProjectFunction(dm, 0., lfuncs, lctxs, INSERT_VALUES, lb));
  if (hasUpper) PetscCall(DMProjectFunction(dm, 0., ufuncs, uctxs, INSERT_VALUES, ub));
  PetscCall(DMPlexInsertBounds(dm, PETSC_TRUE, 0., lb));
  PetscCall(DMPlexInsertBounds(dm, PETSC_FALSE, 0., ub));
  PetscCall(VecViewFromOptions(lb, NULL, "-dm_plex_snes_lb_view"));
  PetscCall(VecViewFromOptions(ub, NULL, "-dm_plex_snes_ub_view"));
  PetscCall(SNESVISetVariableBounds(snes, lb, ub));
  PetscCall(VecDestroy(&lb));
  PetscCall(VecDestroy(&ub));
  PetscCall(PetscFree4(lfuncs, lctxs, ufuncs, uctxs));
  PetscFunctionReturn(PETSC_SUCCESS);
}
