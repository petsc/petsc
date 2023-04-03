#include <../src/snes/impls/fas/fasimpls.h> /*I  "petscsnes.h"  I*/

/*@
    SNESFASSetType - Sets the update and correction type used for FAS.

   Logically Collective

   Input Parameters:
+  snes  - FAS context
-  fastype  - `SNES_FAS_ADDITIVE`, `SNES_FAS_MULTIPLICATIVE`, `SNES_FAS_FULL`, or `SNES_FAS_KASKADE`

   Level: intermediate

.seealso: `SNESFAS`, `PCMGSetType()`, `SNESFASGetType()`
@*/
PetscErrorCode SNESFASSetType(SNES snes, SNESFASType fastype)
{
  SNES_FAS *fas;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes, SNES_CLASSID, 1, SNESFAS);
  PetscValidLogicalCollectiveEnum(snes, fastype, 2);
  fas          = (SNES_FAS *)snes->data;
  fas->fastype = fastype;
  if (fas->next) PetscCall(SNESFASSetType(fas->next, fastype));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   SNESFASGetType - Gets the update and correction type used for FAS.

  Logically Collective

   Input Parameter:
.  snes - `SNESFAS` context

   Output Parameter:
.  fastype - `SNES_FAS_ADDITIVE` or `SNES_FAS_MULTIPLICATIVE`

   Level: intermediate

.seealso: `SNESFAS`, `PCMGSetType()`, `SNESFASSetType()`
@*/
PetscErrorCode SNESFASGetType(SNES snes, SNESFASType *fastype)
{
  SNES_FAS *fas;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes, SNES_CLASSID, 1, SNESFAS);
  PetscValidPointer(fastype, 2);
  fas      = (SNES_FAS *)snes->data;
  *fastype = fas->fastype;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   SNESFASSetLevels - Sets the number of levels to use with `SNESFAS`.
   Must be called before any other FAS routine.

   Input Parameters:
+  snes   - the snes context
.  levels - the number of levels
-  comms  - optional communicators for each level; this is to allow solving the coarser
            problems on smaller sets of processors.

   Level: intermediate

   Note:
   If the number of levels is one then the multigrid uses the `-fas_levels` prefix
  for setting the level options rather than the `-fas_coarse` prefix.

.seealso: `SNESFAS`, `SNESFASGetLevels()`
@*/
PetscErrorCode SNESFASSetLevels(SNES snes, PetscInt levels, MPI_Comm *comms)
{
  PetscInt    i;
  const char *optionsprefix;
  char        tprefix[128];
  SNES_FAS   *fas;
  SNES        prevsnes;
  MPI_Comm    comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes, SNES_CLASSID, 1, SNESFAS);
  fas = (SNES_FAS *)snes->data;
  PetscCall(PetscObjectGetComm((PetscObject)snes, &comm));
  if (levels == fas->levels) {
    if (!comms) PetscFunctionReturn(PETSC_SUCCESS);
  }
  /* user has changed the number of levels; reset */
  PetscUseTypeMethod(snes, reset);
  /* destroy any coarser levels if necessary */
  PetscCall(SNESDestroy(&fas->next));
  fas->next     = NULL;
  fas->previous = NULL;
  prevsnes      = snes;
  /* setup the finest level */
  PetscCall(SNESGetOptionsPrefix(snes, &optionsprefix));
  PetscCall(PetscObjectComposedDataSetInt((PetscObject)snes, PetscMGLevelId, levels - 1));
  for (i = levels - 1; i >= 0; i--) {
    if (comms) comm = comms[i];
    fas->level  = i;
    fas->levels = levels;
    fas->fine   = snes;
    fas->next   = NULL;
    if (i > 0) {
      PetscCall(SNESCreate(comm, &fas->next));
      PetscCall(SNESGetOptionsPrefix(fas->fine, &optionsprefix));
      PetscCall(PetscSNPrintf(tprefix, sizeof(tprefix), "fas_levels_%d_cycle_", (int)fas->level));
      PetscCall(SNESAppendOptionsPrefix(fas->next, optionsprefix));
      PetscCall(SNESAppendOptionsPrefix(fas->next, tprefix));
      PetscCall(SNESSetType(fas->next, SNESFAS));
      PetscCall(SNESSetTolerances(fas->next, fas->next->abstol, fas->next->rtol, fas->next->stol, fas->n_cycles, fas->next->max_funcs));
      PetscCall(PetscObjectIncrementTabLevel((PetscObject)fas->next, (PetscObject)snes, levels - i));
      PetscCall(PetscObjectComposedDataSetInt((PetscObject)fas->next, PetscMGLevelId, i - 1));

      ((SNES_FAS *)fas->next->data)->previous = prevsnes;

      prevsnes = fas->next;
      fas      = (SNES_FAS *)prevsnes->data;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   SNESFASGetLevels - Gets the number of levels in a `SNESFAS`, including fine and coarse grids

   Input Parameter:
.  snes - the `SNES` nonlinear solver context

   Output parameter:
.  levels - the number of levels

   Level: advanced

.seealso: `SNESFAS`, `SNESFASSetLevels()`, `PCMGGetLevels()`
@*/
PetscErrorCode SNESFASGetLevels(SNES snes, PetscInt *levels)
{
  SNES_FAS *fas;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes, SNES_CLASSID, 1, SNESFAS);
  PetscValidIntPointer(levels, 2);
  fas     = (SNES_FAS *)snes->data;
  *levels = fas->levels;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   SNESFASGetCycleSNES - Gets the `SNES` corresponding to a particular
   level of the `SNESFAS` hierarchy.

   Input Parameters:
+  snes    - the `SNES` nonlinear multigrid context
-  level   - the level to get

   Output Parameter:
.  lsnes   - the `SNES` for the requested level

   Level: advanced

.seealso: `SNESFAS`, `SNESFASSetLevels()`, `SNESFASGetLevels()`
@*/
PetscErrorCode SNESFASGetCycleSNES(SNES snes, PetscInt level, SNES *lsnes)
{
  SNES_FAS *fas;
  PetscInt  i;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes, SNES_CLASSID, 1, SNESFAS);
  PetscValidPointer(lsnes, 3);
  fas = (SNES_FAS *)snes->data;
  PetscCheck(level <= fas->levels - 1, PetscObjectComm((PetscObject)snes), PETSC_ERR_ARG_OUTOFRANGE, "Requested level %" PetscInt_FMT " from SNESFAS containing %" PetscInt_FMT " levels", level, fas->levels);
  PetscCheck(fas->level == fas->levels - 1, PetscObjectComm((PetscObject)snes), PETSC_ERR_ARG_OUTOFRANGE, "SNESFASGetCycleSNES may only be called on the finest-level SNES");

  *lsnes = snes;
  for (i = fas->level; i > level; i--) {
    *lsnes = fas->next;
    fas    = (SNES_FAS *)(*lsnes)->data;
  }
  PetscCheck(fas->level == level, PetscObjectComm((PetscObject)snes), PETSC_ERR_PLIB, "SNESFAS level hierarchy corrupt");
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   SNESFASSetNumberSmoothUp - Sets the number of post-smoothing steps to
   use on all levels.

   Logically Collective

   Input Parameters:
+  snes - the `SNES` nonlinear multigrid context
-  n    - the number of smoothing steps

   Options Database Key:
.  -snes_fas_smoothup <n> - Sets number of pre-smoothing steps

   Level: advanced

.seealso: `SNESFAS`, `SNESFASSetNumberSmoothDown()`
@*/
PetscErrorCode SNESFASSetNumberSmoothUp(SNES snes, PetscInt n)
{
  SNES_FAS *fas;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes, SNES_CLASSID, 1, SNESFAS);
  fas            = (SNES_FAS *)snes->data;
  fas->max_up_it = n;
  if (!fas->smoothu && fas->level != 0) PetscCall(SNESFASCycleCreateSmoother_Private(snes, &fas->smoothu));
  if (fas->smoothu) PetscCall(SNESSetTolerances(fas->smoothu, fas->smoothu->abstol, fas->smoothu->rtol, fas->smoothu->stol, n, fas->smoothu->max_funcs));
  if (fas->next) PetscCall(SNESFASSetNumberSmoothUp(fas->next, n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   SNESFASSetNumberSmoothDown - Sets the number of pre-smoothing steps to
   use on all levels.

   Logically Collective

   Input Parameters:
+  snes - the `SNESFAS` nonlinear multigrid context
-  n    - the number of smoothing steps

   Options Database Key:
.  -snes_fas_smoothdown <n> - Sets number of pre-smoothing steps

   Level: advanced

.seealso: `SNESFAS`, `SNESFASSetNumberSmoothUp()`
@*/
PetscErrorCode SNESFASSetNumberSmoothDown(SNES snes, PetscInt n)
{
  SNES_FAS *fas;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes, SNES_CLASSID, 1, SNESFAS);
  fas = (SNES_FAS *)snes->data;
  if (!fas->smoothd) PetscCall(SNESFASCycleCreateSmoother_Private(snes, &fas->smoothd));
  PetscCall(SNESSetTolerances(fas->smoothd, fas->smoothd->abstol, fas->smoothd->rtol, fas->smoothd->stol, n, fas->smoothd->max_funcs));

  fas->max_down_it = n;
  if (fas->next) PetscCall(SNESFASSetNumberSmoothDown(fas->next, n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   SNESFASSetContinuation - Sets the `SNESFAS` cycle to default to exact Newton solves on the upsweep

   Logically Collective

   Input Parameters:
+  snes - the `SNESFAS` nonlinear multigrid context
-  n    - the number of smoothing steps

   Options Database Key:
.  -snes_fas_continuation - sets continuation to true

   Level: advanced

   Note:
    This sets the prefix on the upsweep smoothers to -fas_continuation

.seealso: `SNESFAS`, `SNESFASSetNumberSmoothUp()`
@*/
PetscErrorCode SNESFASSetContinuation(SNES snes, PetscBool continuation)
{
  const char *optionsprefix;
  char        tprefix[128];
  SNES_FAS   *fas;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes, SNES_CLASSID, 1, SNESFAS);
  fas = (SNES_FAS *)snes->data;
  PetscCall(SNESGetOptionsPrefix(fas->fine, &optionsprefix));
  if (!fas->smoothu) PetscCall(SNESFASCycleCreateSmoother_Private(snes, &fas->smoothu));
  PetscCall(PetscStrncpy(tprefix, "fas_levels_continuation_", sizeof(tprefix)));
  PetscCall(SNESSetOptionsPrefix(fas->smoothu, optionsprefix));
  PetscCall(SNESAppendOptionsPrefix(fas->smoothu, tprefix));
  PetscCall(SNESSetType(fas->smoothu, SNESNEWTONLS));
  PetscCall(SNESSetTolerances(fas->smoothu, fas->fine->abstol, fas->fine->rtol, fas->fine->stol, 50, 100));
  fas->continuation = continuation;
  if (fas->next) PetscCall(SNESFASSetContinuation(fas->next, continuation));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   SNESFASSetCycles - Sets the number of FAS multigrid cycles to use each time a grid is visited.  Use `SNESFASSetCyclesOnLevel()` for more
   complicated cycling.

   Logically Collective

   Input Parameters:
+  snes   - the `SNESFAS` nonlinear multigrid context
-  cycles - the number of cycles -- 1 for V-cycle, 2 for W-cycle

   Options Database Key:
.  -snes_fas_cycles <1,2> - 1 for V-cycle, 2 for W-cycle

   Level: advanced

.seealso: `SNES`, `SNESFAS`, `SNESFASSetCyclesOnLevel()`
@*/
PetscErrorCode SNESFASSetCycles(SNES snes, PetscInt cycles)
{
  SNES_FAS *fas;
  PetscBool isFine;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes, SNES_CLASSID, 1, SNESFAS);
  PetscCall(SNESFASCycleIsFine(snes, &isFine));
  fas           = (SNES_FAS *)snes->data;
  fas->n_cycles = cycles;
  if (!isFine) PetscCall(SNESSetTolerances(snes, snes->abstol, snes->rtol, snes->stol, cycles, snes->max_funcs));
  if (fas->next) PetscCall(SNESFASSetCycles(fas->next, cycles));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   SNESFASSetMonitor - Sets the method-specific cycle monitoring

   Logically Collective

   Input Parameters:
+  snes   - the `SNESFAS` context
.  vf     - viewer and format structure (may be `NULL` if flg is `PETSC_FALSE`)
-  flg    - monitor or not

   Level: advanced

.seealso: `SNESFAS`, `SNESSetMonitor()`, `SNESFASSetCyclesOnLevel()`
@*/
PetscErrorCode SNESFASSetMonitor(SNES snes, PetscViewerAndFormat *vf, PetscBool flg)
{
  SNES_FAS *fas;
  PetscBool isFine;
  PetscInt  i, levels;
  SNES      levelsnes;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes, SNES_CLASSID, 1, SNESFAS);
  PetscCall(SNESFASCycleIsFine(snes, &isFine));
  fas    = (SNES_FAS *)snes->data;
  levels = fas->levels;
  if (isFine) {
    for (i = 0; i < levels; i++) {
      PetscCall(SNESFASGetCycleSNES(snes, i, &levelsnes));
      fas = (SNES_FAS *)levelsnes->data;
      if (flg) {
        /* set the monitors for the upsmoother and downsmoother */
        PetscCall(SNESMonitorCancel(levelsnes));
        /* Only register destroy on finest level */
        PetscCall(SNESMonitorSet(levelsnes, (PetscErrorCode(*)(SNES, PetscInt, PetscReal, void *))SNESMonitorDefault, vf, (!i ? (PetscErrorCode(*)(void **))PetscViewerAndFormatDestroy : NULL)));
      } else if (i != fas->levels - 1) {
        /* unset the monitors on the coarse levels */
        PetscCall(SNESMonitorCancel(levelsnes));
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   SNESFASSetLog - Sets or unsets time logging for various `SNESFAS` stages on all levels

   Logically Collective

   Input Parameters:
+  snes   - the `SNESFAS` context
-  flg    - monitor or not

   Level: advanced

.seealso: `SNESFAS`, `SNESFASSetMonitor()`
@*/
PetscErrorCode SNESFASSetLog(SNES snes, PetscBool flg)
{
  SNES_FAS *fas;
  PetscBool isFine;
  PetscInt  i, levels;
  SNES      levelsnes;
  char      eventname[128];

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes, SNES_CLASSID, 1, SNESFAS);
  PetscCall(SNESFASCycleIsFine(snes, &isFine));
  fas    = (SNES_FAS *)snes->data;
  levels = fas->levels;
  if (isFine) {
    for (i = 0; i < levels; i++) {
      PetscCall(SNESFASGetCycleSNES(snes, i, &levelsnes));
      fas = (SNES_FAS *)levelsnes->data;
      if (flg) {
        PetscCall(PetscSNPrintf(eventname, sizeof(eventname), "FASSetup  %d", (int)i));
        PetscCall(PetscLogEventRegister(eventname, ((PetscObject)snes)->classid, &fas->eventsmoothsetup));
        PetscCall(PetscSNPrintf(eventname, sizeof(eventname), "FASSmooth %d", (int)i));
        PetscCall(PetscLogEventRegister(eventname, ((PetscObject)snes)->classid, &fas->eventsmoothsolve));
        PetscCall(PetscSNPrintf(eventname, sizeof(eventname), "FASResid  %d", (int)i));
        PetscCall(PetscLogEventRegister(eventname, ((PetscObject)snes)->classid, &fas->eventresidual));
        PetscCall(PetscSNPrintf(eventname, sizeof(eventname), "FASInterp %d", (int)i));
        PetscCall(PetscLogEventRegister(eventname, ((PetscObject)snes)->classid, &fas->eventinterprestrict));
      } else {
        fas->eventsmoothsetup    = 0;
        fas->eventsmoothsolve    = 0;
        fas->eventresidual       = 0;
        fas->eventinterprestrict = 0;
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
Creates the default smoother type.

This is SNESNRICHARDSON on each fine level and SNESNEWTONLS on the coarse level.

 */
PetscErrorCode SNESFASCycleCreateSmoother_Private(SNES snes, SNES *smooth)
{
  SNES_FAS   *fas;
  const char *optionsprefix;
  char        tprefix[128];
  SNES        nsmooth;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes, SNES_CLASSID, 1, SNESFAS);
  PetscValidPointer(smooth, 2);
  fas = (SNES_FAS *)snes->data;
  PetscCall(SNESGetOptionsPrefix(fas->fine, &optionsprefix));
  /* create the default smoother */
  PetscCall(SNESCreate(PetscObjectComm((PetscObject)snes), &nsmooth));
  if (fas->level == 0) {
    PetscCall(PetscStrncpy(tprefix, "fas_coarse_", sizeof(tprefix)));
    PetscCall(SNESAppendOptionsPrefix(nsmooth, optionsprefix));
    PetscCall(SNESAppendOptionsPrefix(nsmooth, tprefix));
    PetscCall(SNESSetType(nsmooth, SNESNEWTONLS));
    PetscCall(SNESSetTolerances(nsmooth, nsmooth->abstol, nsmooth->rtol, nsmooth->stol, nsmooth->max_its, nsmooth->max_funcs));
  } else {
    PetscCall(PetscSNPrintf(tprefix, sizeof(tprefix), "fas_levels_%d_", (int)fas->level));
    PetscCall(SNESAppendOptionsPrefix(nsmooth, optionsprefix));
    PetscCall(SNESAppendOptionsPrefix(nsmooth, tprefix));
    PetscCall(SNESSetType(nsmooth, SNESNRICHARDSON));
    PetscCall(SNESSetTolerances(nsmooth, 0.0, 0.0, 0.0, fas->max_down_it, nsmooth->max_funcs));
  }
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)nsmooth, (PetscObject)snes, 1));
  PetscCall(PetscObjectCopyFortranFunctionPointers((PetscObject)snes, (PetscObject)nsmooth));
  PetscCall(PetscObjectComposedDataSetInt((PetscObject)nsmooth, PetscMGLevelId, fas->level));
  *smooth = nsmooth;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ------------- Functions called on a particular level ----------------- */

/*@
   SNESFASCycleSetCycles - Sets the number of cycles on a particular level.

   Logically Collective

   Input Parameters:
+  snes   - the `SNESFAS` nonlinear multigrid context
.  level  - the level to set the number of cycles on
-  cycles - the number of cycles -- 1 for V-cycle, 2 for W-cycle

   Level: advanced

.seealso: `SNESFAS`, `SNESFASSetCycles()`
@*/
PetscErrorCode SNESFASCycleSetCycles(SNES snes, PetscInt cycles)
{
  SNES_FAS *fas;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes, SNES_CLASSID, 1, SNESFAS);
  fas           = (SNES_FAS *)snes->data;
  fas->n_cycles = cycles;
  PetscCall(SNESSetTolerances(snes, snes->abstol, snes->rtol, snes->stol, cycles, snes->max_funcs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   SNESFASCycleGetSmoother - Gets the smoother on a particular cycle level.

   Logically Collective

   Input Parameter:
.  snes   - the `SNESFAS` nonlinear multigrid context

   Output Parameter:
.  smooth - the smoother

   Level: advanced

.seealso: `SNESFAS`, `SNESFASCycleGetSmootherUp()`, `SNESFASCycleGetSmootherDown()`
@*/
PetscErrorCode SNESFASCycleGetSmoother(SNES snes, SNES *smooth)
{
  SNES_FAS *fas;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes, SNES_CLASSID, 1, SNESFAS);
  PetscValidPointer(smooth, 2);
  fas     = (SNES_FAS *)snes->data;
  *smooth = fas->smoothd;
  PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
   SNESFASCycleGetSmootherUp - Gets the up smoother on a particular cycle level.

   Logically Collective

   Input Parameter:
.  snes   - the `SNESFAS` nonlinear multigrid context

   Output Parameter:
.  smoothu - the smoother

   Note:
   Returns the downsmoother if no up smoother is available.  This enables transparent
   default behavior in the process of the solve.

   Level: advanced

.seealso: `SNESFAS`, `SNESFASCycleGetSmoother()`, `SNESFASCycleGetSmootherDown()`
@*/
PetscErrorCode SNESFASCycleGetSmootherUp(SNES snes, SNES *smoothu)
{
  SNES_FAS *fas;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes, SNES_CLASSID, 1, SNESFAS);
  PetscValidPointer(smoothu, 2);
  fas = (SNES_FAS *)snes->data;
  if (!fas->smoothu) *smoothu = fas->smoothd;
  else *smoothu = fas->smoothu;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   SNESFASCycleGetSmootherDown - Gets the down smoother on a particular cycle level.

   Logically Collective

   Input Parameter:
.  snes   - `SNESFAS`, the nonlinear multigrid context

   Output Parameter:
.  smoothd - the smoother

   Level: advanced

.seealso: `SNESFAS`, `SNESFASCycleGetSmootherUp()`, `SNESFASCycleGetSmoother()`
@*/
PetscErrorCode SNESFASCycleGetSmootherDown(SNES snes, SNES *smoothd)
{
  SNES_FAS *fas;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes, SNES_CLASSID, 1, SNESFAS);
  PetscValidPointer(smoothd, 2);
  fas      = (SNES_FAS *)snes->data;
  *smoothd = fas->smoothd;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   SNESFASCycleGetCorrection - Gets the coarse correction FAS context for this level

   Logically Collective

   Input Parameter:
.  snes   - the `SNESFAS` nonlinear multigrid context

   Output Parameter:
.  correction - the coarse correction solve on this level

   Note:
   Returns NULL on the coarsest level.

   Level: advanced

.seealso: `SNESFAS` `SNESFASCycleGetSmootherUp()`, `SNESFASCycleGetSmoother()`
@*/
PetscErrorCode SNESFASCycleGetCorrection(SNES snes, SNES *correction)
{
  SNES_FAS *fas;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes, SNES_CLASSID, 1, SNESFAS);
  PetscValidPointer(correction, 2);
  fas         = (SNES_FAS *)snes->data;
  *correction = fas->next;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   SNESFASCycleGetInterpolation - Gets the interpolation on this level

   Logically Collective

   Input Parameter:
.  snes   - the `SNESFAS` nonlinear multigrid context

   Output Parameter:
.  mat    - the interpolation operator on this level

   Level: advanced

.seealso: `SNESFAS`, `SNESFASCycleGetSmootherUp()`, `SNESFASCycleGetSmoother()`
@*/
PetscErrorCode SNESFASCycleGetInterpolation(SNES snes, Mat *mat)
{
  SNES_FAS *fas;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes, SNES_CLASSID, 1, SNESFAS);
  PetscValidPointer(mat, 2);
  fas  = (SNES_FAS *)snes->data;
  *mat = fas->interpolate;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   SNESFASCycleGetRestriction - Gets the restriction on this level

   Logically Collective

   Input Parameter:
.  snes   - the `SNESFAS` nonlinear multigrid context

   Output Parameter:
.  mat    - the restriction operator on this level

   Level: advanced

.seealso: `SNESFAS`, `SNESFASGetRestriction()`, `SNESFASCycleGetInterpolation()`
@*/
PetscErrorCode SNESFASCycleGetRestriction(SNES snes, Mat *mat)
{
  SNES_FAS *fas;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes, SNES_CLASSID, 1, SNESFAS);
  PetscValidPointer(mat, 2);
  fas  = (SNES_FAS *)snes->data;
  *mat = fas->restrct;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   SNESFASCycleGetInjection - Gets the injection on this level

   Logically Collective

   Input Parameter:
.  snes   - the `SNESFAS` nonlinear multigrid context

   Output Parameter:
.  mat    - the restriction operator on this level

   Level: advanced

.seealso: `SNESFAS`, `SNESFASGetInjection()`, `SNESFASCycleGetRestriction()`
@*/
PetscErrorCode SNESFASCycleGetInjection(SNES snes, Mat *mat)
{
  SNES_FAS *fas;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes, SNES_CLASSID, 1, SNESFAS);
  PetscValidPointer(mat, 2);
  fas  = (SNES_FAS *)snes->data;
  *mat = fas->inject;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   SNESFASCycleGetRScale - Gets the injection on this level

   Logically Collective

   Input Parameter:
.  snes   - the  `SNESFAS` nonlinear multigrid context

   Output Parameter:
.  mat    - the restriction operator on this level

   Level: advanced

.seealso: `SNESFAS`, `SNESFASCycleGetRestriction()`, `SNESFASGetRScale()`
@*/
PetscErrorCode SNESFASCycleGetRScale(SNES snes, Vec *vec)
{
  SNES_FAS *fas;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes, SNES_CLASSID, 1, SNESFAS);
  PetscValidPointer(vec, 2);
  fas  = (SNES_FAS *)snes->data;
  *vec = fas->rscale;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   SNESFASCycleIsFine - Determines if a given cycle is the fine level.

   Logically Collective

   Input Parameter:
.  snes   - the `SNESFAS` `SNES` context

   Output Parameter:
.  flg - indicates if this is the fine level or not

   Level: advanced

.seealso: `SNESFAS`, `SNESFASSetLevels()`
@*/
PetscErrorCode SNESFASCycleIsFine(SNES snes, PetscBool *flg)
{
  SNES_FAS *fas;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes, SNES_CLASSID, 1, SNESFAS);
  PetscValidBoolPointer(flg, 2);
  fas = (SNES_FAS *)snes->data;
  if (fas->level == fas->levels - 1) *flg = PETSC_TRUE;
  else *flg = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*  functions called on the finest level that return level-specific information  */

/*@
   SNESFASSetInterpolation - Sets the `Mat` to be used to apply the
   interpolation from l-1 to the lth level

   Input Parameters:
+  snes      - the `SNESFAS` nonlinear multigrid context
.  mat       - the interpolation operator
-  level     - the level (0 is coarsest) to supply [do not supply 0]

   Level: advanced

   Notes:
          Usually this is the same matrix used also to set the restriction
    for the same level.

          One can pass in the interpolation matrix or its transpose; PETSc figures
    out from the matrix size which one it is.

.seealso: `SNESFAS`, `SNESFASSetInjection()`, `SNESFASSetRestriction()`, `SNESFASSetRScale()`
@*/
PetscErrorCode SNESFASSetInterpolation(SNES snes, PetscInt level, Mat mat)
{
  SNES_FAS *fas;
  SNES      levelsnes;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes, SNES_CLASSID, 1, SNESFAS);
  if (mat) PetscValidHeaderSpecific(mat, MAT_CLASSID, 3);
  PetscCall(SNESFASGetCycleSNES(snes, level, &levelsnes));
  fas = (SNES_FAS *)levelsnes->data;
  PetscCall(PetscObjectReference((PetscObject)mat));
  PetscCall(MatDestroy(&fas->interpolate));
  fas->interpolate = mat;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   SNESFASGetInterpolation - Gets the matrix used to calculate the
   interpolation from l-1 to the lth level

   Input Parameters:
+  snes      - the `SNESFAS` nonlinear multigrid context
-  level     - the level (0 is coarsest) to supply [do not supply 0]

   Output Parameter:
.  mat       - the interpolation operator

   Level: advanced

.seealso: `SNESFAS`, `SNESFASSetInterpolation()`, `SNESFASGetInjection()`, `SNESFASGetRestriction()`, `SNESFASGetRScale()`
@*/
PetscErrorCode SNESFASGetInterpolation(SNES snes, PetscInt level, Mat *mat)
{
  SNES_FAS *fas;
  SNES      levelsnes;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes, SNES_CLASSID, 1, SNESFAS);
  PetscValidPointer(mat, 3);
  PetscCall(SNESFASGetCycleSNES(snes, level, &levelsnes));
  fas  = (SNES_FAS *)levelsnes->data;
  *mat = fas->interpolate;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   SNESFASSetRestriction - Sets the matrix to be used to restrict the defect
   from level l to l-1.

   Input Parameters:
+  snes  - the `SNESFAS` nonlinear multigrid context
.  mat   - the restriction matrix
-  level - the level (0 is coarsest) to supply [Do not supply 0]

   Level: advanced

   Notes:
          Usually this is the same matrix used also to set the interpolation
    for the same level.

          One can pass in the interpolation matrix or its transpose; PETSc figures
    out from the matrix size which one it is.

         If you do not set this, the transpose of the Mat set with SNESFASSetInterpolation()
    is used.

.seealso: `SNESFAS`, `SNESFASSetInterpolation()`, `SNESFASSetInjection()`
@*/
PetscErrorCode SNESFASSetRestriction(SNES snes, PetscInt level, Mat mat)
{
  SNES_FAS *fas;
  SNES      levelsnes;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes, SNES_CLASSID, 1, SNESFAS);
  if (mat) PetscValidHeaderSpecific(mat, MAT_CLASSID, 3);
  PetscCall(SNESFASGetCycleSNES(snes, level, &levelsnes));
  fas = (SNES_FAS *)levelsnes->data;
  PetscCall(PetscObjectReference((PetscObject)mat));
  PetscCall(MatDestroy(&fas->restrct));
  fas->restrct = mat;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   SNESFASGetRestriction - Gets the matrix used to calculate the
   restriction from l to the l-1th level

   Input Parameters:
+  snes      - the `SNESFAS` nonlinear multigrid context
-  level     - the level (0 is coarsest) to supply [do not supply 0]

   Output Parameter:
.  mat       - the interpolation operator

   Level: advanced

.seealso: `SNESFAS`, `SNESFASSetRestriction()`, `SNESFASGetInjection()`, `SNESFASGetInterpolation()`, `SNESFASGetRScale()`
@*/
PetscErrorCode SNESFASGetRestriction(SNES snes, PetscInt level, Mat *mat)
{
  SNES_FAS *fas;
  SNES      levelsnes;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes, SNES_CLASSID, 1, SNESFAS);
  PetscValidPointer(mat, 3);
  PetscCall(SNESFASGetCycleSNES(snes, level, &levelsnes));
  fas  = (SNES_FAS *)levelsnes->data;
  *mat = fas->restrct;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   SNESFASSetInjection - Sets the function to be used to inject the solution
   from level l to l-1.

   Input Parameters:
 +  snes  - the `SNESFAS` nonlinear multigrid context
.  mat   - the restriction matrix
-  level - the level (0 is coarsest) to supply [Do not supply 0]

   Level: advanced

   Note:
         If you do not set this, the restriction and rscale is used to
   project the solution instead.

.seealso: `SNESFAS`, `SNESFASSetInterpolation()`, `SNESFASSetRestriction()`
@*/
PetscErrorCode SNESFASSetInjection(SNES snes, PetscInt level, Mat mat)
{
  SNES_FAS *fas;
  SNES      levelsnes;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes, SNES_CLASSID, 1, SNESFAS);
  if (mat) PetscValidHeaderSpecific(mat, MAT_CLASSID, 3);
  PetscCall(SNESFASGetCycleSNES(snes, level, &levelsnes));
  fas = (SNES_FAS *)levelsnes->data;
  PetscCall(PetscObjectReference((PetscObject)mat));
  PetscCall(MatDestroy(&fas->inject));

  fas->inject = mat;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   SNESFASGetInjection - Gets the matrix used to calculate the
   injection from l-1 to the lth level

   Input Parameters:
+  snes      - the `SNESFAS` nonlinear multigrid context
-  level     - the level (0 is coarsest) to supply [do not supply 0]

   Output Parameter:
.  mat       - the injection operator

   Level: advanced

.seealso: `SNESFAS`, `SNESFASSetInjection()`, `SNESFASGetRestriction()`, `SNESFASGetInterpolation()`, `SNESFASGetRScale()`
@*/
PetscErrorCode SNESFASGetInjection(SNES snes, PetscInt level, Mat *mat)
{
  SNES_FAS *fas;
  SNES      levelsnes;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes, SNES_CLASSID, 1, SNESFAS);
  PetscValidPointer(mat, 3);
  PetscCall(SNESFASGetCycleSNES(snes, level, &levelsnes));
  fas  = (SNES_FAS *)levelsnes->data;
  *mat = fas->inject;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   SNESFASSetRScale - Sets the scaling factor of the restriction
   operator from level l to l-1.

   Input Parameters:
+  snes   - the `SNESFAS` nonlinear multigrid context
.  rscale - the restriction scaling
-  level  - the level (0 is coarsest) to supply [Do not supply 0]

   Level: advanced

   Note:
   This is only used in the case that the injection is not set.

.seealso: `SNESFAS`, `SNESFASSetInjection()`, `SNESFASSetRestriction()`
@*/
PetscErrorCode SNESFASSetRScale(SNES snes, PetscInt level, Vec rscale)
{
  SNES_FAS *fas;
  SNES      levelsnes;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes, SNES_CLASSID, 1, SNESFAS);
  if (rscale) PetscValidHeaderSpecific(rscale, VEC_CLASSID, 3);
  PetscCall(SNESFASGetCycleSNES(snes, level, &levelsnes));
  fas = (SNES_FAS *)levelsnes->data;
  PetscCall(PetscObjectReference((PetscObject)rscale));
  PetscCall(VecDestroy(&fas->rscale));
  fas->rscale = rscale;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   SNESFASGetSmoother - Gets the default smoother on a level.

   Input Parameters:
+  snes   - the `SNESFAS` nonlinear multigrid context
-  level  - the level (0 is coarsest) to supply

   Output Parameter:
   smooth  - the smoother

   Level: advanced

.seealso: `SNESFAS`, `SNESFASSetInjection()`, `SNESFASSetRestriction()`
@*/
PetscErrorCode SNESFASGetSmoother(SNES snes, PetscInt level, SNES *smooth)
{
  SNES_FAS *fas;
  SNES      levelsnes;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes, SNES_CLASSID, 1, SNESFAS);
  PetscValidPointer(smooth, 3);
  PetscCall(SNESFASGetCycleSNES(snes, level, &levelsnes));
  fas = (SNES_FAS *)levelsnes->data;
  if (!fas->smoothd) PetscCall(SNESFASCycleCreateSmoother_Private(levelsnes, &fas->smoothd));
  *smooth = fas->smoothd;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   SNESFASGetSmootherDown - Gets the downsmoother on a level.

   Input Parameters:
+  snes   - the `SNESFAS` nonlinear multigrid context
-  level  - the level (0 is coarsest) to supply

   Output Parameter:
   smooth  - the smoother

   Level: advanced

.seealso: `SNESFAS`, `SNESFASSetInjection()`, `SNESFASSetRestriction()`
@*/
PetscErrorCode SNESFASGetSmootherDown(SNES snes, PetscInt level, SNES *smooth)
{
  SNES_FAS *fas;
  SNES      levelsnes;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes, SNES_CLASSID, 1, SNESFAS);
  PetscValidPointer(smooth, 3);
  PetscCall(SNESFASGetCycleSNES(snes, level, &levelsnes));
  fas = (SNES_FAS *)levelsnes->data;
  /* if the user chooses to differentiate smoothers, create them both at this point */
  if (!fas->smoothd) PetscCall(SNESFASCycleCreateSmoother_Private(levelsnes, &fas->smoothd));
  if (!fas->smoothu) PetscCall(SNESFASCycleCreateSmoother_Private(levelsnes, &fas->smoothu));
  *smooth = fas->smoothd;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   SNESFASGetSmootherUp - Gets the upsmoother on a level.

   Input Parameters:
+  snes   - the `SNESFAS` nonlinear multigrid context
-  level  - the level (0 is coarsest)

   Output Parameter:
   smooth  - the smoother

   Level: advanced

.seealso: `SNESFAS`, `SNESFASSetInjection()`, `SNESFASSetRestriction()`
@*/
PetscErrorCode SNESFASGetSmootherUp(SNES snes, PetscInt level, SNES *smooth)
{
  SNES_FAS *fas;
  SNES      levelsnes;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes, SNES_CLASSID, 1, SNESFAS);
  PetscValidPointer(smooth, 3);
  PetscCall(SNESFASGetCycleSNES(snes, level, &levelsnes));
  fas = (SNES_FAS *)levelsnes->data;
  /* if the user chooses to differentiate smoothers, create them both at this point */
  if (!fas->smoothd) PetscCall(SNESFASCycleCreateSmoother_Private(levelsnes, &fas->smoothd));
  if (!fas->smoothu) PetscCall(SNESFASCycleCreateSmoother_Private(levelsnes, &fas->smoothu));
  *smooth = fas->smoothu;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SNESFASGetCoarseSolve - Gets the coarsest solver.

  Input Parameter:
. snes - the `SNESFAS` nonlinear multigrid context

  Output Parameter:
. coarse - the coarse-level solver

  Level: advanced

.seealso: `SNESFAS`, `SNESFASSetInjection()`, `SNESFASSetRestriction()`
@*/
PetscErrorCode SNESFASGetCoarseSolve(SNES snes, SNES *coarse)
{
  SNES_FAS *fas;
  SNES      levelsnes;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes, SNES_CLASSID, 1, SNESFAS);
  PetscValidPointer(coarse, 2);
  PetscCall(SNESFASGetCycleSNES(snes, 0, &levelsnes));
  fas = (SNES_FAS *)levelsnes->data;
  /* if the user chooses to differentiate smoothers, create them both at this point */
  if (!fas->smoothd) PetscCall(SNESFASCycleCreateSmoother_Private(levelsnes, &fas->smoothd));
  *coarse = fas->smoothd;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   SNESFASFullSetDownSweep - Smooth during the initial downsweep for `SNESFAS`

   Logically Collective

   Input Parameters:
+  snes - the `SNESFAS` nonlinear multigrid context
-  swp - whether to downsweep or not

   Options Database Key:
.  -snes_fas_full_downsweep - Sets number of pre-smoothing steps

   Level: advanced

.seealso: `SNESFAS`, `SNESFASSetNumberSmoothUp()`
@*/
PetscErrorCode SNESFASFullSetDownSweep(SNES snes, PetscBool swp)
{
  SNES_FAS *fas;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes, SNES_CLASSID, 1, SNESFAS);
  fas                 = (SNES_FAS *)snes->data;
  fas->full_downsweep = swp;
  if (fas->next) PetscCall(SNESFASFullSetDownSweep(fas->next, swp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   SNESFASFullSetTotal - Use total residual restriction and total interpolation on the initial down and up sweep of full FAS cycles

   Logically Collective

   Input Parameters:
+  snes - the `SNESFAS`  nonlinear multigrid context
-  total - whether to use total restriction / interpolatiaon or not (the alternative is defect restriction and correction interpolation)

   Options Database Key:
.  -snes_fas_full_total - Use total restriction and interpolation on the initial down and up sweeps for the full FAS cycle

   Level: advanced

   Note:
   This option is only significant if the interpolation of a coarse correction (`MatInterpolate()`) is significantly different from total
   solution interpolation (`DMInterpolateSolution()`).

.seealso: `SNESFAS`, `SNESFASSetNumberSmoothUp()`, `DMInterpolateSolution()`
@*/
PetscErrorCode SNESFASFullSetTotal(SNES snes, PetscBool total)
{
  SNES_FAS *fas;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes, SNES_CLASSID, 1, SNESFAS);
  fas             = (SNES_FAS *)snes->data;
  fas->full_total = total;
  if (fas->next) PetscCall(SNESFASFullSetTotal(fas->next, total));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   SNESFASFullGetTotal - Use total residual restriction and total interpolation on the initial down and up sweep of full FAS cycles

   Logically Collective

   Input Parameter:
.  snes - the `SNESFAS` nonlinear multigrid context

   Output:
.  total - whether to use total restriction / interpolatiaon or not (the alternative is defect restriction and correction interpolation)

   Level: advanced

.seealso: `SNESFAS`, `SNESFASSetNumberSmoothUp()`, `DMInterpolateSolution()`, `SNESFullSetTotal()`
@*/
PetscErrorCode SNESFASFullGetTotal(SNES snes, PetscBool *total)
{
  SNES_FAS *fas;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes, SNES_CLASSID, 1, SNESFAS);
  fas    = (SNES_FAS *)snes->data;
  *total = fas->full_total;
  PetscFunctionReturn(PETSC_SUCCESS);
}
