#include <petsc/private/dmpleximpl.h> /*I "petscdmplex.h" I*/
#include <petsc/private/tsimpl.h>     /*I "petscts.h" I*/
#include <petsc/private/snesimpl.h>
#include <petscds.h>
#include <petscfv.h>

static PetscErrorCode DMTSConvertPlex(DM dm, DM *plex, PetscBool copy)
{
  PetscBool      isPlex;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject) dm, DMPLEX, &isPlex));
  if (isPlex) {
    *plex = dm;
    PetscCall(PetscObjectReference((PetscObject) dm));
  } else {
    PetscCall(PetscObjectQuery((PetscObject) dm, "dm_plex", (PetscObject *) plex));
    if (!*plex) {
      PetscCall(DMConvert(dm,DMPLEX,plex));
      PetscCall(PetscObjectCompose((PetscObject) dm, "dm_plex", (PetscObject) *plex));
      if (copy) {
        PetscCall(DMCopyDMTS(dm, *plex));
        PetscCall(DMCopyDMSNES(dm, *plex));
        PetscCall(DMCopyAuxiliaryVec(dm, *plex));
      }
    } else {
      PetscCall(PetscObjectReference((PetscObject) *plex));
    }
  }
  PetscFunctionReturn(0);
}

/*@
  DMPlexTSComputeRHSFunctionFVM - Form the local forcing F from the local input X using pointwise functions specified by the user

  Input Parameters:
+ dm - The mesh
. t - The time
. locX  - Local solution
- user - The user context

  Output Parameter:
. F  - Global output vector

  Level: developer

.seealso: DMPlexComputeJacobianActionFEM()
@*/
PetscErrorCode DMPlexTSComputeRHSFunctionFVM(DM dm, PetscReal time, Vec locX, Vec F, void *user)
{
  Vec          locF;
  IS           cellIS;
  DM           plex;
  PetscInt     depth;
  PetscFormKey key = {NULL, 0, 0, 0};

  PetscFunctionBegin;
  PetscCall(DMTSConvertPlex(dm,&plex,PETSC_TRUE));
  PetscCall(DMPlexGetDepth(plex, &depth));
  PetscCall(DMGetStratumIS(plex, "dim", depth, &cellIS));
  if (!cellIS) PetscCall(DMGetStratumIS(plex, "depth", depth, &cellIS));
  PetscCall(DMGetLocalVector(plex, &locF));
  PetscCall(VecZeroEntries(locF));
  PetscCall(DMPlexComputeResidual_Internal(plex, key, cellIS, time, locX, NULL, time, locF, user));
  PetscCall(DMLocalToGlobalBegin(plex, locF, ADD_VALUES, F));
  PetscCall(DMLocalToGlobalEnd(plex, locF, ADD_VALUES, F));
  PetscCall(DMRestoreLocalVector(plex, &locF));
  PetscCall(ISDestroy(&cellIS));
  PetscCall(DMDestroy(&plex));
  PetscFunctionReturn(0);
}

/*@
  DMPlexTSComputeBoundary - Insert the essential boundary values for the local input X and/or its time derivative X_t using pointwise functions specified by the user

  Input Parameters:
+ dm - The mesh
. t - The time
. locX  - Local solution
. locX_t - Local solution time derivative, or NULL
- user - The user context

  Level: developer

.seealso: DMPlexComputeJacobianActionFEM()
@*/
PetscErrorCode DMPlexTSComputeBoundary(DM dm, PetscReal time, Vec locX, Vec locX_t, void *user)
{
  DM             plex;
  Vec            faceGeometryFVM = NULL;
  PetscInt       Nf, f;

  PetscFunctionBegin;
  PetscCall(DMTSConvertPlex(dm, &plex, PETSC_TRUE));
  PetscCall(DMGetNumFields(plex, &Nf));
  if (!locX_t) {
    /* This is the RHS part */
    for (f = 0; f < Nf; f++) {
      PetscObject  obj;
      PetscClassId id;

      PetscCall(DMGetField(plex, f, NULL, &obj));
      PetscCall(PetscObjectGetClassId(obj, &id));
      if (id == PETSCFV_CLASSID) {
        PetscCall(DMPlexGetGeometryFVM(plex, &faceGeometryFVM, NULL, NULL));
        break;
      }
    }
  }
  PetscCall(DMPlexInsertBoundaryValues(plex, PETSC_TRUE, locX, time, faceGeometryFVM, NULL, NULL));
  PetscCall(DMPlexInsertTimeDerivativeBoundaryValues(plex, PETSC_TRUE, locX_t, time, faceGeometryFVM, NULL, NULL));
  PetscCall(DMDestroy(&plex));
  PetscFunctionReturn(0);
}

/*@
  DMPlexTSComputeIFunctionFEM - Form the local residual F from the local input X using pointwise functions specified by the user

  Input Parameters:
+ dm - The mesh
. t - The time
. locX  - Local solution
. locX_t - Local solution time derivative, or NULL
- user - The user context

  Output Parameter:
. locF  - Local output vector

  Level: developer

.seealso: DMPlexTSComputeIFunctionFEM(), DMPlexTSComputeRHSFunctionFEM()
@*/
PetscErrorCode DMPlexTSComputeIFunctionFEM(DM dm, PetscReal time, Vec locX, Vec locX_t, Vec locF, void *user)
{
  DM             plex;
  IS             allcellIS;
  PetscInt       Nds, s;

  PetscFunctionBegin;
  PetscCall(DMTSConvertPlex(dm, &plex, PETSC_TRUE));
  PetscCall(DMPlexGetAllCells_Internal(plex, &allcellIS));
  PetscCall(DMGetNumDS(dm, &Nds));
  for (s = 0; s < Nds; ++s) {
    PetscDS          ds;
    IS               cellIS;
    PetscFormKey key;

    PetscCall(DMGetRegionNumDS(dm, s, &key.label, NULL, &ds));
    key.value = 0;
    key.field = 0;
    key.part  = 0;
    if (!key.label) {
      PetscCall(PetscObjectReference((PetscObject) allcellIS));
      cellIS = allcellIS;
    } else {
      IS pointIS;

      key.value = 1;
      PetscCall(DMLabelGetStratumIS(key.label, key.value, &pointIS));
      PetscCall(ISIntersect_Caching_Internal(allcellIS, pointIS, &cellIS));
      PetscCall(ISDestroy(&pointIS));
    }
    PetscCall(DMPlexComputeResidual_Internal(plex, key, cellIS, time, locX, locX_t, time, locF, user));
    PetscCall(ISDestroy(&cellIS));
  }
  PetscCall(ISDestroy(&allcellIS));
  PetscCall(DMDestroy(&plex));
  PetscFunctionReturn(0);
}

/*@
  DMPlexTSComputeIJacobianFEM - Form the local Jacobian J from the local input X using pointwise functions specified by the user

  Input Parameters:
+ dm - The mesh
. t - The time
. locX  - Local solution
. locX_t - Local solution time derivative, or NULL
. X_tshift - The multiplicative parameter for dF/du_t
- user - The user context

  Output Parameter:
. locF  - Local output vector

  Level: developer

.seealso: DMPlexTSComputeIFunctionFEM(), DMPlexTSComputeRHSFunctionFEM()
@*/
PetscErrorCode DMPlexTSComputeIJacobianFEM(DM dm, PetscReal time, Vec locX, Vec locX_t, PetscReal X_tShift, Mat Jac, Mat JacP, void *user)
{
  DM             plex;
  IS             allcellIS;
  PetscBool      hasJac, hasPrec;
  PetscInt       Nds, s;

  PetscFunctionBegin;
  PetscCall(DMTSConvertPlex(dm, &plex, PETSC_TRUE));
  PetscCall(DMPlexGetAllCells_Internal(plex, &allcellIS));
  PetscCall(DMGetNumDS(dm, &Nds));
  for (s = 0; s < Nds; ++s) {
    PetscDS          ds;
    IS               cellIS;
    PetscFormKey key;

    PetscCall(DMGetRegionNumDS(dm, s, &key.label, NULL, &ds));
    key.value = 0;
    key.field = 0;
    key.part  = 0;
    if (!key.label) {
      PetscCall(PetscObjectReference((PetscObject) allcellIS));
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
    PetscCall(DMPlexComputeJacobian_Internal(plex, key, cellIS, time, X_tShift, locX, locX_t, Jac, JacP, user));
    PetscCall(ISDestroy(&cellIS));
  }
  PetscCall(ISDestroy(&allcellIS));
  PetscCall(DMDestroy(&plex));
  PetscFunctionReturn(0);
}

/*@
  DMPlexTSComputeRHSFunctionFEM - Form the local residual G from the local input X using pointwise functions specified by the user

  Input Parameters:
+ dm - The mesh
. t - The time
. locX  - Local solution
- user - The user context

  Output Parameter:
. locG  - Local output vector

  Level: developer

.seealso: DMPlexTSComputeIFunctionFEM(), DMPlexTSComputeIJacobianFEM()
@*/
PetscErrorCode DMPlexTSComputeRHSFunctionFEM(DM dm, PetscReal time, Vec locX, Vec locG, void *user)
{
  DM             plex;
  IS             allcellIS;
  PetscInt       Nds, s;

  PetscFunctionBegin;
  PetscCall(DMTSConvertPlex(dm, &plex, PETSC_TRUE));
  PetscCall(DMPlexGetAllCells_Internal(plex, &allcellIS));
  PetscCall(DMGetNumDS(dm, &Nds));
  for (s = 0; s < Nds; ++s) {
    PetscDS          ds;
    IS               cellIS;
    PetscFormKey key;

    PetscCall(DMGetRegionNumDS(dm, s, &key.label, NULL, &ds));
    key.value = 0;
    key.field = 0;
    key.part  = 100;
    if (!key.label) {
      PetscCall(PetscObjectReference((PetscObject) allcellIS));
      cellIS = allcellIS;
    } else {
      IS pointIS;

      key.value = 1;
      PetscCall(DMLabelGetStratumIS(key.label, key.value, &pointIS));
      PetscCall(ISIntersect_Caching_Internal(allcellIS, pointIS, &cellIS));
      PetscCall(ISDestroy(&pointIS));
    }
    PetscCall(DMPlexComputeResidual_Internal(plex, key, cellIS, time, locX, NULL, time, locG, user));
    PetscCall(ISDestroy(&cellIS));
  }
  PetscCall(ISDestroy(&allcellIS));
  PetscCall(DMDestroy(&plex));
  PetscFunctionReturn(0);
}

/*@C
  DMTSCheckResidual - Check the residual of the exact solution

  Input Parameters:
+ ts  - the TS object
. dm  - the DM
. t   - the time
. u   - a DM vector
. u_t - a DM vector
- tol - A tolerance for the check, or -1 to print the results instead

  Output Parameters:
. residual - The residual norm of the exact solution, or NULL

  Level: developer

.seealso: DNTSCheckFromOptions(), DMTSCheckJacobian(), DNSNESCheckFromOptions(), DMSNESCheckDiscretization(), DMSNESCheckJacobian()
@*/
PetscErrorCode DMTSCheckResidual(TS ts, DM dm, PetscReal t, Vec u, Vec u_t, PetscReal tol, PetscReal *residual)
{
  MPI_Comm       comm;
  Vec            r;
  PetscReal      res;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidHeaderSpecific(dm, DM_CLASSID, 2);
  PetscValidHeaderSpecific(u, VEC_CLASSID, 4);
  if (residual) PetscValidRealPointer(residual, 7);
  PetscCall(PetscObjectGetComm((PetscObject) ts, &comm));
  PetscCall(DMComputeExactSolution(dm, t, u, u_t));
  PetscCall(VecDuplicate(u, &r));
  PetscCall(TSComputeIFunction(ts, t, u, u_t, r, PETSC_FALSE));
  PetscCall(VecNorm(r, NORM_2, &res));
  if (tol >= 0.0) {
    PetscCheck(res <= tol,comm, PETSC_ERR_ARG_WRONG, "L_2 Residual %g exceeds tolerance %g", (double) res, (double) tol);
  } else if (residual) {
    *residual = res;
  } else {
    PetscCall(PetscPrintf(comm, "L_2 Residual: %g\n", (double)res));
    PetscCall(VecChop(r, 1.0e-10));
    PetscCall(PetscObjectCompose((PetscObject) r, "__Vec_bc_zero__", (PetscObject) dm));
    PetscCall(PetscObjectSetName((PetscObject) r, "Initial Residual"));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)r,"res_"));
    PetscCall(VecViewFromOptions(r, NULL, "-vec_view"));
    PetscCall(PetscObjectCompose((PetscObject) r, "__Vec_bc_zero__", NULL));
  }
  PetscCall(VecDestroy(&r));
  PetscFunctionReturn(0);
}

/*@C
  DMTSCheckJacobian - Check the Jacobian of the exact solution against the residual using the Taylor Test

  Input Parameters:
+ ts  - the TS object
. dm  - the DM
. t   - the time
. u   - a DM vector
. u_t - a DM vector
- tol - A tolerance for the check, or -1 to print the results instead

  Output Parameters:
+ isLinear - Flag indicaing that the function looks linear, or NULL
- convRate - The rate of convergence of the linear model, or NULL

  Level: developer

.seealso: DNTSCheckFromOptions(), DMTSCheckResidual(), DNSNESCheckFromOptions(), DMSNESCheckDiscretization(), DMSNESCheckResidual()
@*/
PetscErrorCode DMTSCheckJacobian(TS ts, DM dm, PetscReal t, Vec u, Vec u_t, PetscReal tol, PetscBool *isLinear, PetscReal *convRate)
{
  MPI_Comm       comm;
  PetscDS        ds;
  Mat            J, M;
  MatNullSpace   nullspace;
  PetscReal      dt, shift, slope, intercept;
  PetscBool      hasJac, hasPrec, isLin = PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidHeaderSpecific(dm, DM_CLASSID, 2);
  PetscValidHeaderSpecific(u, VEC_CLASSID, 4);
  if (isLinear) PetscValidBoolPointer(isLinear, 7);
  if (convRate) PetscValidRealPointer(convRate, 8);
  PetscCall(PetscObjectGetComm((PetscObject) ts, &comm));
  PetscCall(DMComputeExactSolution(dm, t, u, u_t));
  /* Create and view matrices */
  PetscCall(TSGetTimeStep(ts, &dt));
  shift = 1.0/dt;
  PetscCall(DMCreateMatrix(dm, &J));
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSHasJacobian(ds, &hasJac));
  PetscCall(PetscDSHasJacobianPreconditioner(ds, &hasPrec));
  if (hasJac && hasPrec) {
    PetscCall(DMCreateMatrix(dm, &M));
    PetscCall(TSComputeIJacobian(ts, t, u, u_t, shift, J, M, PETSC_FALSE));
    PetscCall(PetscObjectSetName((PetscObject) M, "Preconditioning Matrix"));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject) M, "jacpre_"));
    PetscCall(MatViewFromOptions(M, NULL, "-mat_view"));
    PetscCall(MatDestroy(&M));
  } else {
    PetscCall(TSComputeIJacobian(ts, t, u, u_t, shift, J, J, PETSC_FALSE));
  }
  PetscCall(PetscObjectSetName((PetscObject) J, "Jacobian"));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject) J, "jac_"));
  PetscCall(MatViewFromOptions(J, NULL, "-mat_view"));
  /* Check nullspace */
  PetscCall(MatGetNullSpace(J, &nullspace));
  if (nullspace) {
    PetscBool isNull;
    PetscCall(MatNullSpaceTest(nullspace, J, &isNull));
    PetscCheck(isNull,comm, PETSC_ERR_PLIB, "The null space calculated for the system operator is invalid.");
  }
  /* Taylor test */
  {
    PetscRandom rand;
    Vec         du, uhat, uhat_t, r, rhat, df;
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
    PetscCall(TSComputeIFunction(ts, t, u, u_t, r, PETSC_FALSE));
    /* Look at the convergence of our Taylor approximation as we approach u */
    for (h = hMax, Nv = 0; h >= hMin; h *= hMult, ++Nv);
    PetscCall(PetscCalloc3(Nv, &es, Nv, &hs, Nv, &errors));
    PetscCall(VecDuplicate(u, &uhat));
    PetscCall(VecDuplicate(u, &uhat_t));
    PetscCall(VecDuplicate(u, &rhat));
    for (h = hMax, Nv = 0; h >= hMin; h *= hMult, ++Nv) {
      PetscCall(VecWAXPY(uhat, h, du, u));
      PetscCall(VecWAXPY(uhat_t, h*shift, du, u_t));
      /* F(\hat u, \hat u_t) \approx F(u, u_t) + J(u, u_t) (uhat - u) + J_t(u, u_t) (uhat_t - u_t) = F(u) + h * J(u) du + h * shift * J_t(u) du = F(u) + h F' du */
      PetscCall(TSComputeIFunction(ts, t, uhat, uhat_t, rhat, PETSC_FALSE));
      PetscCall(VecAXPBYPCZ(rhat, -1.0, -h, 1.0, r, df));
      PetscCall(VecNorm(rhat, NORM_2, &errors[Nv]));

      es[Nv] = PetscLog10Real(errors[Nv]);
      hs[Nv] = PetscLog10Real(h);
    }
    PetscCall(VecDestroy(&uhat));
    PetscCall(VecDestroy(&uhat_t));
    PetscCall(VecDestroy(&rhat));
    PetscCall(VecDestroy(&df));
    PetscCall(VecDestroy(&r));
    PetscCall(VecDestroy(&du));
    for (v = 0; v < Nv; ++v) {
      if ((tol >= 0) && (errors[v] > tol)) break;
      else if (errors[v] > PETSC_SMALL)    break;
    }
    if (v == Nv) isLin = PETSC_TRUE;
    PetscCall(PetscLinearRegression(Nv, hs, es, &slope, &intercept));
    PetscCall(PetscFree3(es, hs, errors));
    /* Slope should be about 2 */
    if (tol >= 0) {
      PetscCheck(isLin || PetscAbsReal(2 - slope) <= tol,comm, PETSC_ERR_ARG_WRONG, "Taylor approximation convergence rate should be 2, not %0.2f", (double) slope);
    } else if (isLinear || convRate) {
      if (isLinear) *isLinear = isLin;
      if (convRate) *convRate = slope;
    } else {
      if (!isLin) PetscCall(PetscPrintf(comm, "Taylor approximation converging at order %3.2f\n", (double) slope));
      else        PetscCall(PetscPrintf(comm, "Function appears to be linear\n"));
    }
  }
  PetscCall(MatDestroy(&J));
  PetscFunctionReturn(0);
}

/*@C
  DMTSCheckFromOptions - Check the residual and Jacobian functions using the exact solution by outputting some diagnostic information

  Input Parameters:
+ ts - the TS object
- u  - representative TS vector

  Note: The user must call PetscDSSetExactSolution() beforehand

  Level: developer
@*/
PetscErrorCode DMTSCheckFromOptions(TS ts, Vec u)
{
  DM             dm;
  SNES           snes;
  Vec            sol, u_t;
  PetscReal      t;
  PetscBool      check;

  PetscFunctionBegin;
  PetscCall(PetscOptionsHasName(((PetscObject)ts)->options,((PetscObject)ts)->prefix, "-dmts_check", &check));
  if (!check) PetscFunctionReturn(0);
  PetscCall(VecDuplicate(u, &sol));
  PetscCall(VecCopy(u, sol));
  PetscCall(TSSetSolution(ts, u));
  PetscCall(TSGetDM(ts, &dm));
  PetscCall(TSSetUp(ts));
  PetscCall(TSGetSNES(ts, &snes));
  PetscCall(SNESSetSolution(snes, u));

  PetscCall(TSGetTime(ts, &t));
  PetscCall(DMSNESCheckDiscretization(snes, dm, t, sol, -1.0, NULL));
  PetscCall(DMGetGlobalVector(dm, &u_t));
  PetscCall(DMTSCheckResidual(ts, dm, t, sol, u_t, -1.0, NULL));
  PetscCall(DMTSCheckJacobian(ts, dm, t, sol, u_t, -1.0, NULL, NULL));
  PetscCall(DMRestoreGlobalVector(dm, &u_t));

  PetscCall(VecDestroy(&sol));
  PetscFunctionReturn(0);
}
