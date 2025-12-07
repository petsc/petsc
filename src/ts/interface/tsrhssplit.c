#include <petsc/private/tsimpl.h> /*I "petscts.h"  I*/
#include <petscdm.h>
static PetscErrorCode TSRHSSplitGetRHSSplit(TS ts, const char splitname[], TS_RHSSplitLink *isplit)
{
  PetscBool found = PETSC_FALSE;

  PetscFunctionBegin;
  *isplit = ts->tsrhssplit;
  /* look up the split */
  while (*isplit) {
    PetscCall(PetscStrcmp((*isplit)->splitname, splitname, &found));
    if (found) break;
    *isplit = (*isplit)->next;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSRHSSplitSetIS - Set the index set for the specified split

  Logically Collective

  Input Parameters:
+ ts        - the `TS` context obtained from `TSCreate()`
. splitname - name of this split, if `NULL` the number of the split is used
- is        - the index set for part of the solution vector

  Level: intermediate

.seealso: [](ch_ts), `TS`, `IS`, `TSRHSSplitGetIS()`, `TSARKIMEXSetFastSlowSplit()`
@*/
PetscErrorCode TSRHSSplitSetIS(TS ts, const char splitname[], IS is)
{
  TS_RHSSplitLink newsplit, next = ts->tsrhssplit;
  char            prefix[128];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidHeaderSpecific(is, IS_CLASSID, 3);

  PetscCall(PetscNew(&newsplit));
  if (splitname) {
    PetscCall(PetscStrallocpy(splitname, &newsplit->splitname));
  } else {
    PetscCall(PetscMalloc1(8, &newsplit->splitname));
    PetscCall(PetscSNPrintf(newsplit->splitname, 7, "%" PetscInt_FMT, ts->num_rhs_splits));
  }
  PetscCall(PetscObjectReference((PetscObject)is));
  newsplit->is = is;
  PetscCall(TSCreate(PetscObjectComm((PetscObject)ts), &newsplit->ts));

  PetscCall(PetscObjectIncrementTabLevel((PetscObject)newsplit->ts, (PetscObject)ts, 1));
  PetscCall(PetscSNPrintf(prefix, sizeof(prefix), "%srhsplit_%s_", ((PetscObject)ts)->prefix ? ((PetscObject)ts)->prefix : "", newsplit->splitname));
  PetscCall(TSSetOptionsPrefix(newsplit->ts, prefix));
  if (!next) ts->tsrhssplit = newsplit;
  else {
    while (next->next) next = next->next;
    next->next = newsplit;
  }
  ts->num_rhs_splits++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSRHSSplitGetIS - Retrieves the elements for a split as an `IS`

  Logically Collective

  Input Parameters:
+ ts        - the `TS` context obtained from `TSCreate()`
- splitname - name of this split

  Output Parameter:
. is - the index set for part of the solution vector

  Level: intermediate

.seealso: [](ch_ts), `TS`, `IS`, `TSRHSSplitSetIS()`
@*/
PetscErrorCode TSRHSSplitGetIS(TS ts, const char splitname[], IS *is)
{
  TS_RHSSplitLink isplit;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  *is = NULL;
  /* look up the split */
  PetscCall(TSRHSSplitGetRHSSplit(ts, splitname, &isplit));
  if (isplit) *is = isplit->is;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TSRHSSplitSetRHSFunction - Set the split right-hand-side functions.

  Logically Collective

  Input Parameters:
+ ts        - the `TS` context obtained from `TSCreate()`
. splitname - name of this split
. r         - vector to hold the residual (or `NULL` to have it created internally)
. rhsfunc   - the RHS function evaluation routine
- ctx       - user-defined context for private data for the split function evaluation routine (may be `NULL`)

  Level: intermediate

.seealso: [](ch_ts), `TS`, `TSRHSFunctionFn`, `IS`, `TSRHSSplitSetIS()`
@*/
PetscErrorCode TSRHSSplitSetRHSFunction(TS ts, const char splitname[], Vec r, TSRHSFunctionFn *rhsfunc, void *ctx)
{
  TS_RHSSplitLink isplit;
  DM              dmc;
  Vec             subvec, ralloc = NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  if (r) PetscValidHeaderSpecific(r, VEC_CLASSID, 3);

  /* look up the split */
  PetscCall(TSRHSSplitGetRHSSplit(ts, splitname, &isplit));
  PetscCheck(isplit, PETSC_COMM_SELF, PETSC_ERR_USER, "The split %s is not created, check the split name or call TSRHSSplitSetIS() to create one", splitname);

  if (!r && ts->vec_sol) {
    PetscCall(VecGetSubVector(ts->vec_sol, isplit->is, &subvec));
    PetscCall(VecDuplicate(subvec, &ralloc));
    r = ralloc;
    PetscCall(VecRestoreSubVector(ts->vec_sol, isplit->is, &subvec));
  }

  if (ts->dm) {
    PetscInt dim;

    PetscCall(DMGetDimension(ts->dm, &dim));
    if (dim != -1) {
      PetscCall(DMClone(ts->dm, &dmc));
      PetscCall(TSSetDM(isplit->ts, dmc));
      PetscCall(DMDestroy(&dmc));
    }
  }

  PetscCall(TSSetRHSFunction(isplit->ts, r, rhsfunc, ctx));
  PetscCall(VecDestroy(&ralloc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TSRHSSplitSetIFunction - Set the split implicit function for `TSARKIMEX`

  Logically Collective

  Input Parameters:
+ ts        - the `TS` context obtained from `TSCreate()`
. splitname - name of this split
. r         - vector to hold the residual (or `NULL` to have it created internally)
. ifunc     - the implicit function evaluation routine
- ctx       - user-defined context for private data for the split function evaluation routine (may be `NULL`)

  Level: intermediate

.seealso: [](ch_ts), `TS`, `TSIFunctionFn`, `IS`, `TSRHSSplitSetIS()`, `TSARKIMEX`, `TSARKIMEXSetFastSlowSplit()`
@*/
PetscErrorCode TSRHSSplitSetIFunction(TS ts, const char splitname[], Vec r, TSIFunctionFn *ifunc, void *ctx)
{
  TS_RHSSplitLink isplit;
  DM              dmc;
  Vec             subvec, ralloc = NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  if (r) PetscValidHeaderSpecific(r, VEC_CLASSID, 3);

  /* look up the split */
  PetscCall(TSRHSSplitGetRHSSplit(ts, splitname, &isplit));
  PetscCheck(isplit, PETSC_COMM_SELF, PETSC_ERR_USER, "The split %s is not created, check the split name or call TSRHSSplitSetIS() to create one", splitname);

  if (!r && ts->vec_sol) {
    PetscCall(VecGetSubVector(ts->vec_sol, isplit->is, &subvec));
    PetscCall(VecDuplicate(subvec, &ralloc));
    r = ralloc;
    PetscCall(VecRestoreSubVector(ts->vec_sol, isplit->is, &subvec));
  }

  if (ts->dm) {
    PetscInt dim;

    PetscCall(DMGetDimension(ts->dm, &dim));
    if (dim != -1) {
      PetscCall(DMClone(ts->dm, &dmc));
      PetscCall(TSSetDM(isplit->ts, dmc));
      PetscCall(DMDestroy(&dmc));
    }
  }

  PetscCall(TSSetIFunction(isplit->ts, r, ifunc, ctx));
  PetscCall(VecDestroy(&ralloc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TSRHSSplitSetIJacobian - Set the Jacobian for the split implicit function with `TSARKIMEX`

  Logically Collective

  Input Parameters:
+ ts        - the `TS` context obtained from `TSCreate()`
. splitname - name of this split
. Amat      - (approximate) matrix to store Jacobian entries computed by `f`
. Pmat      - matrix used to compute preconditioner (usually the same as `Amat`)
. ijac      - the Jacobian evaluation routine
- ctx       - user-defined context for private data for the split function evaluation routine (may be `NULL`)

  Level: intermediate

.seealso: [](ch_ts), `TS`, `TSRHSSplitSetIFunction`, `TSIJacobianFn`, `IS`, `TSRHSSplitSetIS()`, `TSARKIMEXSetFastSlowSplit()`
@*/
PetscErrorCode TSRHSSplitSetIJacobian(TS ts, const char splitname[], Mat Amat, Mat Pmat, TSIJacobianFn *ijac, void *ctx)
{
  TS_RHSSplitLink isplit;
  DM              dmc;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  if (Amat) PetscValidHeaderSpecific(Amat, MAT_CLASSID, 3);
  if (Pmat) PetscValidHeaderSpecific(Pmat, MAT_CLASSID, 4);
  if (Amat) PetscCheckSameComm(ts, 1, Amat, 3);
  if (Pmat) PetscCheckSameComm(ts, 1, Pmat, 4);

  /* look up the split */
  PetscCall(TSRHSSplitGetRHSSplit(ts, splitname, &isplit));
  PetscCheck(isplit, PETSC_COMM_SELF, PETSC_ERR_USER, "The split %s is not created, check the split name or call TSRHSSplitSetIS() to create one", splitname);

  if (ts->dm) {
    PetscInt dim;

    PetscCall(DMGetDimension(ts->dm, &dim));
    if (dim != -1) {
      PetscCall(DMClone(ts->dm, &dmc));
      PetscCall(TSSetDM(isplit->ts, dmc));
      PetscCall(DMDestroy(&dmc));
    }
  }

  PetscCall(TSSetIJacobian(isplit->ts, Amat, Pmat, ijac, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TSRHSSplitGetSubTS - Get the sub-`TS` by split name.

  Logically Collective

  Input Parameter:
. ts - the `TS` context obtained from `TSCreate()`

  Output Parameters:
+ splitname - the number of the split
- subts     - the sub-`TS`

  Level: advanced

.seealso: [](ch_ts), `TS`, `IS`, `TSGetRHSSplitFunction()`
@*/
PetscErrorCode TSRHSSplitGetSubTS(TS ts, const char splitname[], TS *subts)
{
  TS_RHSSplitLink isplit;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscAssertPointer(subts, 3);
  *subts = NULL;
  /* look up the split */
  PetscCall(TSRHSSplitGetRHSSplit(ts, splitname, &isplit));
  if (isplit) *subts = isplit->ts;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TSRHSSplitGetSubTSs - Get an array of all sub-`TS` contexts.

  Logically Collective

  Input Parameter:
. ts - the `TS` context obtained from `TSCreate()`

  Output Parameters:
+ n     - the number of splits
- subts - the array of `TS` contexts

  Level: advanced

  Note:
  After `TSRHSSplitGetSubTS()` the array of `TS`s is to be freed by the user with `PetscFree()`
  (not the `TS` in the array just the array that contains them).

.seealso: [](ch_ts), `TS`, `IS`, `TSGetRHSSplitFunction()`
@*/
PetscErrorCode TSRHSSplitGetSubTSs(TS ts, PetscInt *n, TS *subts[])
{
  TS_RHSSplitLink ilink = ts->tsrhssplit;
  PetscInt        i     = 0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  if (subts) {
    PetscCall(PetscMalloc1(ts->num_rhs_splits, subts));
    while (ilink) {
      (*subts)[i++] = ilink->ts;
      ilink         = ilink->next;
    }
  }
  if (n) *n = ts->num_rhs_splits;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSRHSSplitGetSNES - Returns the `SNES` (nonlinear solver) associated with
  a `TS` (timestepper) context when RHS splits are used.

  Not Collective, but `snes` is parallel if `ts` is parallel

  Input Parameter:
. ts - the `TS` context obtained from `TSCreate()`

  Output Parameter:
. snes - the nonlinear solver context

  Level: intermediate

  Note:
  The returned `SNES` may have a different `DM` with the `TS` `DM`.

.seealso: [](ch_ts), `TS`, `SNES`, `TSCreate()`, `TSRHSSplitSetSNES()`
@*/
PetscErrorCode TSRHSSplitGetSNES(TS ts, SNES *snes)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscAssertPointer(snes, 2);
  if (!ts->snesrhssplit) {
    PetscCall(SNESCreate(PetscObjectComm((PetscObject)ts), &ts->snesrhssplit));
    PetscCall(PetscObjectSetOptions((PetscObject)ts->snesrhssplit, ((PetscObject)ts)->options));
    PetscCall(SNESSetFunction(ts->snesrhssplit, NULL, SNESTSFormFunction, ts));
    PetscCall(PetscObjectIncrementTabLevel((PetscObject)ts->snesrhssplit, (PetscObject)ts, 1));
    if (ts->problem_type == TS_LINEAR) PetscCall(SNESSetType(ts->snesrhssplit, SNESKSPONLY));
  }
  *snes = ts->snesrhssplit;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSRHSSplitSetSNES - Set the `SNES` (nonlinear solver) to be used by the
  timestepping context when RHS splits are used.

  Collective

  Input Parameters:
+ ts   - the `TS` context obtained from `TSCreate()`
- snes - the nonlinear solver context

  Level: intermediate

  Note:
  Most users should have the `TS` created by calling `TSRHSSplitGetSNES()`

.seealso: [](ch_ts), `TS`, `SNES`, `TSCreate()`, `TSRHSSplitGetSNES()`
@*/
PetscErrorCode TSRHSSplitSetSNES(TS ts, SNES snes)
{
  PetscErrorCode (*func)(SNES, Vec, Mat, Mat, void *);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 2);
  PetscCall(PetscObjectReference((PetscObject)snes));
  PetscCall(SNESDestroy(&ts->snesrhssplit));

  ts->snesrhssplit = snes;

  PetscCall(SNESSetFunction(ts->snesrhssplit, NULL, SNESTSFormFunction, ts));
  PetscCall(SNESGetJacobian(ts->snesrhssplit, NULL, NULL, &func, NULL));
  if (func == SNESTSFormJacobian) PetscCall(SNESSetJacobian(ts->snesrhssplit, NULL, NULL, SNESTSFormJacobian, ts));
  PetscFunctionReturn(PETSC_SUCCESS);
}
