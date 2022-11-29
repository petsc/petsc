#include <petsc/private/tshistoryimpl.h> /*I "petscts.h" I*/

typedef struct {
  TSHistory hist;
  PetscBool bw;
} TSAdapt_History;

static PetscErrorCode TSAdaptChoose_History(TSAdapt adapt, TS ts, PetscReal h, PetscInt *next_sc, PetscReal *next_h, PetscBool *accept, PetscReal *wlte, PetscReal *wltea, PetscReal *wlter)
{
  PetscInt         step;
  TSAdapt_History *thadapt = (TSAdapt_History *)adapt->data;

  PetscFunctionBegin;
  PetscCheck(thadapt->hist, PetscObjectComm((PetscObject)adapt), PETSC_ERR_ORDER, "Need to call TSAdaptHistorySetHistory() first");
  PetscCall(TSGetStepNumber(ts, &step));
  PetscCall(TSHistoryGetTimeStep(thadapt->hist, thadapt->bw, step + 1, next_h));
  *accept  = PETSC_TRUE;
  *next_sc = 0;
  *wlte    = -1;
  *wltea   = -1;
  *wlter   = -1;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSAdaptReset_History(TSAdapt adapt)
{
  TSAdapt_History *thadapt = (TSAdapt_History *)adapt->data;

  PetscFunctionBegin;
  PetscCall(TSHistoryDestroy(&thadapt->hist));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSAdaptDestroy_History(TSAdapt adapt)
{
  PetscFunctionBegin;
  PetscCall(TSAdaptReset_History(adapt));
  PetscCall(PetscFree(adapt->data));
  PetscFunctionReturn(0);
}

/* this is not public, as TSHistory is not a public object */
PetscErrorCode TSAdaptHistorySetTSHistory(TSAdapt adapt, TSHistory hist, PetscBool backward)
{
  PetscReal *hist_t;
  PetscInt   n;
  PetscBool  flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt, TSADAPT_CLASSID, 1);
  PetscValidLogicalCollectiveBool(adapt, backward, 3);
  PetscCall(PetscObjectTypeCompare((PetscObject)adapt, TSADAPTHISTORY, &flg));
  if (!flg) PetscFunctionReturn(0);
  PetscCall(TSHistoryGetHistory(hist, &n, (const PetscReal **)&hist_t, NULL, NULL));
  PetscCall(TSAdaptHistorySetHistory(adapt, n, hist_t, backward));
  PetscFunctionReturn(0);
}

/*@
   TSAdaptHistoryGetStep - Gets time and time step for a given step number in the history

   Logically Collective on adapt

   Input Parameters:
+  adapt    - the TSAdapt context
-  step     - the step number

   Output Parameters:
+  t  - the time corresponding to the requested step (can be NULL)
-  dt - the time step to be taken at the requested step (can be NULL)

   Level: advanced

   Note:
   The time history is internally copied, and the user can free the hist array. The user still needs to specify the termination of the solve via `TSSetMaxSteps()`.

.seealso: [](chapter_ts), `TS`, `TSGetAdapt()`, `TSAdaptSetType()`, `TSAdaptHistorySetTrajectory()`, `TSADAPTHISTORY`
@*/
PetscErrorCode TSAdaptHistoryGetStep(TSAdapt adapt, PetscInt step, PetscReal *t, PetscReal *dt)
{
  TSAdapt_History *thadapt;
  PetscBool        flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt, TSADAPT_CLASSID, 1);
  PetscValidLogicalCollectiveInt(adapt, step, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)adapt, TSADAPTHISTORY, &flg));
  PetscCheck(flg, PetscObjectComm((PetscObject)adapt), PETSC_ERR_SUP, "Not for type %s", ((PetscObject)adapt)->type_name);
  thadapt = (TSAdapt_History *)adapt->data;
  PetscCall(TSHistoryGetTimeStep(thadapt->hist, thadapt->bw, step, dt));
  PetscCall(TSHistoryGetTime(thadapt->hist, thadapt->bw, step, t));
  PetscFunctionReturn(0);
}

/*@
   TSAdaptHistorySetHistory - Sets the time history in the adaptor

   Logically Collective on adapt

   Input Parameters:
+  adapt    - the `TSAdapt` context
.  n        - size of the time history
.  hist     - the time history
-  backward - if the time history has to be followed backward

   Level: advanced

   Note:
   The time history is internally copied, and the user can free the hist array. The user still needs to specify the termination of the solve via `TSSetMaxSteps()`.

.seealso: [](chapter_ts), `TSGetAdapt()`, `TSAdaptSetType()`, `TSAdaptHistorySetTrajectory()`, `TSADAPTHISTORY`
@*/
PetscErrorCode TSAdaptHistorySetHistory(TSAdapt adapt, PetscInt n, PetscReal hist[], PetscBool backward)
{
  TSAdapt_History *thadapt;
  PetscBool        flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt, TSADAPT_CLASSID, 1);
  PetscValidLogicalCollectiveInt(adapt, n, 2);
  PetscValidRealPointer(hist, 3);
  PetscValidLogicalCollectiveBool(adapt, backward, 4);
  PetscCall(PetscObjectTypeCompare((PetscObject)adapt, TSADAPTHISTORY, &flg));
  if (!flg) PetscFunctionReturn(0);
  thadapt = (TSAdapt_History *)adapt->data;
  PetscCall(TSHistoryDestroy(&thadapt->hist));
  PetscCall(TSHistoryCreate(PetscObjectComm((PetscObject)adapt), &thadapt->hist));
  PetscCall(TSHistorySetHistory(thadapt->hist, n, hist, NULL, PETSC_FALSE));
  thadapt->bw = backward;
  PetscFunctionReturn(0);
}

/*@
   TSAdaptHistorySetTrajectory - Sets a time history in the adaptor from a given `TSTrajectory`

   Logically Collective on adapt

   Input Parameters:
+  adapt    - the `TSAdapt` context
.  tj       - the `TSTrajectory` context
-  backward - if the time history has to be followed backward

   Level: advanced

   Notes:
   The time history is internally copied, and the user can destroy the `TSTrajectory` if not needed.

   The user still needs to specify the termination of the solve via `TSSetMaxSteps()`.

.seealso: [](chapter_ts), `TSGetAdapt()`, `TSAdaptSetType()`, `TSAdaptHistorySetHistory()`, `TSADAPTHISTORY`, `TSAdapt`
@*/
PetscErrorCode TSAdaptHistorySetTrajectory(TSAdapt adapt, TSTrajectory tj, PetscBool backward)
{
  PetscBool flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt, TSADAPT_CLASSID, 1);
  PetscValidHeaderSpecific(tj, TSTRAJECTORY_CLASSID, 2);
  PetscValidLogicalCollectiveBool(adapt, backward, 3);
  PetscCall(PetscObjectTypeCompare((PetscObject)adapt, TSADAPTHISTORY, &flg));
  if (!flg) PetscFunctionReturn(0);
  PetscCall(TSAdaptHistorySetTSHistory(adapt, tj->tsh, backward));
  PetscFunctionReturn(0);
}

/*MC
   TSADAPTHISTORY - Time stepping controller that follows a given time history, used for Tangent Linear Model simulations

   Level: developer

.seealso: [](chapter_ts), `TS`, `TSAdapt`, `TSGetAdapt()`, `TSAdaptHistorySetHistory()`, `TSAdaptType`
M*/
PETSC_EXTERN PetscErrorCode TSAdaptCreate_History(TSAdapt adapt)
{
  TSAdapt_History *thadapt;

  PetscFunctionBegin;
  PetscCall(PetscNew(&thadapt));
  adapt->matchstepfac[0] = PETSC_SMALL; /* prevent from accumulation errors */
  adapt->matchstepfac[1] = 0.0;         /* we will always match the final step, prevent TSAdaptChoose to mess with it */
  adapt->data            = thadapt;

  adapt->ops->choose  = TSAdaptChoose_History;
  adapt->ops->reset   = TSAdaptReset_History;
  adapt->ops->destroy = TSAdaptDestroy_History;
  PetscFunctionReturn(0);
}
