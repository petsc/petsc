#include <../src/snes/impls/fas/fasimpls.h> /*I  "petscsnes.h"  I*/

/* -------------- functions called on the fine level -------------- */

/*@
    SNESFASSetType - Sets the update and correction type used for FAS.

   Logically Collective

Input Parameters:
+ snes  - FAS context
- fastype  - SNES_FAS_ADDITIVE, SNES_FAS_MULTIPLICATIVE, SNES_FAS_FULL, or SNES_FAS_KASKADE

Level: intermediate

.seealso: PCMGSetType()
@*/
PetscErrorCode  SNESFASSetType(SNES snes,SNESFASType fastype)
{
  SNES_FAS       *fas;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes,SNES_CLASSID,1,SNESFAS);
  PetscValidLogicalCollectiveEnum(snes,fastype,2);
  fas = (SNES_FAS*)snes->data;
  fas->fastype = fastype;
  if (fas->next) {
    CHKERRQ(SNESFASSetType(fas->next, fastype));
  }
  PetscFunctionReturn(0);
}

/*@
SNESFASGetType - Sets the update and correction type used for FAS.

Logically Collective

Input Parameters:
. snes - FAS context

Output Parameters:
. fastype - SNES_FAS_ADDITIVE or SNES_FAS_MULTIPLICATIVE

Level: intermediate

.seealso: PCMGSetType()
@*/
PetscErrorCode  SNESFASGetType(SNES snes,SNESFASType *fastype)
{
  SNES_FAS *fas;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes,SNES_CLASSID,1,SNESFAS);
  PetscValidPointer(fastype, 2);
  fas = (SNES_FAS*)snes->data;
  *fastype = fas->fastype;
  PetscFunctionReturn(0);
}

/*@C
   SNESFASSetLevels - Sets the number of levels to use with FAS.
   Must be called before any other FAS routine.

   Input Parameters:
+  snes   - the snes context
.  levels - the number of levels
-  comms  - optional communicators for each level; this is to allow solving the coarser
            problems on smaller sets of processors.

   Level: intermediate

   Notes:
     If the number of levels is one then the multigrid uses the -fas_levels prefix
  for setting the level options rather than the -fas_coarse prefix.

.seealso: SNESFASGetLevels()
@*/
PetscErrorCode SNESFASSetLevels(SNES snes, PetscInt levels, MPI_Comm *comms)
{
  PetscInt       i;
  const char     *optionsprefix;
  char           tprefix[128];
  SNES_FAS       *fas;
  SNES           prevsnes;
  MPI_Comm       comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes,SNES_CLASSID,1,SNESFAS);
  fas = (SNES_FAS*)snes->data;
  CHKERRQ(PetscObjectGetComm((PetscObject)snes,&comm));
  if (levels == fas->levels) {
    if (!comms) PetscFunctionReturn(0);
  }
  /* user has changed the number of levels; reset */
  CHKERRQ((*snes->ops->reset)(snes));
  /* destroy any coarser levels if necessary */
  CHKERRQ(SNESDestroy(&fas->next));
  fas->next     = NULL;
  fas->previous = NULL;
  prevsnes      = snes;
  /* setup the finest level */
  CHKERRQ(SNESGetOptionsPrefix(snes, &optionsprefix));
  CHKERRQ(PetscObjectComposedDataSetInt((PetscObject) snes, PetscMGLevelId, levels-1));
  for (i = levels - 1; i >= 0; i--) {
    if (comms) comm = comms[i];
    fas->level  = i;
    fas->levels = levels;
    fas->fine   = snes;
    fas->next   = NULL;
    if (i > 0) {
      CHKERRQ(SNESCreate(comm, &fas->next));
      CHKERRQ(SNESGetOptionsPrefix(fas->fine, &optionsprefix));
      CHKERRQ(PetscSNPrintf(tprefix,sizeof(tprefix),"fas_levels_%d_cycle_",(int)fas->level));
      CHKERRQ(SNESAppendOptionsPrefix(fas->next,optionsprefix));
      CHKERRQ(SNESAppendOptionsPrefix(fas->next,tprefix));
      CHKERRQ(SNESSetType(fas->next, SNESFAS));
      CHKERRQ(SNESSetTolerances(fas->next, fas->next->abstol, fas->next->rtol, fas->next->stol, fas->n_cycles, fas->next->max_funcs));
      CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)fas->next, (PetscObject)snes, levels - i));
      CHKERRQ(PetscObjectComposedDataSetInt((PetscObject) fas->next, PetscMGLevelId, i-1));

      ((SNES_FAS*)fas->next->data)->previous = prevsnes;

      prevsnes = fas->next;
      fas      = (SNES_FAS*)prevsnes->data;
    }
  }
  PetscFunctionReturn(0);
}

/*@
   SNESFASGetLevels - Gets the number of levels in a FAS, including fine and coarse grids

   Input Parameter:
.  snes - the nonlinear solver context

   Output parameter:
.  levels - the number of levels

   Level: advanced

.seealso: SNESFASSetLevels(), PCMGGetLevels()
@*/
PetscErrorCode SNESFASGetLevels(SNES snes, PetscInt *levels)
{
  SNES_FAS *fas;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes,SNES_CLASSID,1,SNESFAS);
  PetscValidIntPointer(levels,2);
  fas = (SNES_FAS*)snes->data;
  *levels = fas->levels;
  PetscFunctionReturn(0);
}

/*@
   SNESFASGetCycleSNES - Gets the SNES corresponding to a particular
   level of the FAS hierarchy.

   Input Parameters:
+  snes    - the multigrid context
   level   - the level to get
-  lsnes   - whether to use the nonlinear smoother or not

   Level: advanced

.seealso: SNESFASSetLevels(), SNESFASGetLevels()
@*/
PetscErrorCode SNESFASGetCycleSNES(SNES snes,PetscInt level,SNES *lsnes)
{
  SNES_FAS *fas;
  PetscInt i;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes,SNES_CLASSID,1,SNESFAS);
  PetscValidPointer(lsnes,3);
  fas = (SNES_FAS*)snes->data;
  PetscCheckFalse(level > fas->levels-1,PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_OUTOFRANGE,"Requested level %D from SNESFAS containing %D levels",level,fas->levels);
  PetscCheckFalse(fas->level !=  fas->levels - 1,PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_OUTOFRANGE,"SNESFASGetCycleSNES may only be called on the finest-level SNES.",level,fas->level);

  *lsnes = snes;
  for (i = fas->level; i > level; i--) {
    *lsnes = fas->next;
    fas    = (SNES_FAS*)(*lsnes)->data;
  }
  PetscCheckFalse(fas->level != level,PetscObjectComm((PetscObject)snes),PETSC_ERR_PLIB,"SNESFAS level hierarchy corrupt");
  PetscFunctionReturn(0);
}

/*@
   SNESFASSetNumberSmoothUp - Sets the number of post-smoothing steps to
   use on all levels.

   Logically Collective on SNES

   Input Parameters:
+  snes - the multigrid context
-  n    - the number of smoothing steps

   Options Database Key:
.  -snes_fas_smoothup <n> - Sets number of pre-smoothing steps

   Level: advanced

.seealso: SNESFASSetNumberSmoothDown()
@*/
PetscErrorCode SNESFASSetNumberSmoothUp(SNES snes, PetscInt n)
{
  SNES_FAS       *fas;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes,SNES_CLASSID,1,SNESFAS);
  fas =  (SNES_FAS*)snes->data;
  fas->max_up_it = n;
  if (!fas->smoothu && fas->level != 0) {
    CHKERRQ(SNESFASCycleCreateSmoother_Private(snes, &fas->smoothu));
  }
  if (fas->smoothu) {
    CHKERRQ(SNESSetTolerances(fas->smoothu, fas->smoothu->abstol, fas->smoothu->rtol, fas->smoothu->stol, n, fas->smoothu->max_funcs));
  }
  if (fas->next) {
    CHKERRQ(SNESFASSetNumberSmoothUp(fas->next, n));
  }
  PetscFunctionReturn(0);
}

/*@
   SNESFASSetNumberSmoothDown - Sets the number of pre-smoothing steps to
   use on all levels.

   Logically Collective on SNES

   Input Parameters:
+  snes - the multigrid context
-  n    - the number of smoothing steps

   Options Database Key:
.  -snes_fas_smoothdown <n> - Sets number of pre-smoothing steps

   Level: advanced

.seealso: SNESFASSetNumberSmoothUp()
@*/
PetscErrorCode SNESFASSetNumberSmoothDown(SNES snes, PetscInt n)
{
  SNES_FAS       *fas;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes,SNES_CLASSID,1,SNESFAS);
  fas = (SNES_FAS*)snes->data;
  if (!fas->smoothd) {
    CHKERRQ(SNESFASCycleCreateSmoother_Private(snes, &fas->smoothd));
  }
  CHKERRQ(SNESSetTolerances(fas->smoothd, fas->smoothd->abstol, fas->smoothd->rtol, fas->smoothd->stol, n, fas->smoothd->max_funcs));

  fas->max_down_it = n;
  if (fas->next) {
    CHKERRQ(SNESFASSetNumberSmoothDown(fas->next, n));
  }
  PetscFunctionReturn(0);
}

/*@
   SNESFASSetContinuation - Sets the FAS cycle to default to exact Newton solves on the upsweep

   Logically Collective on SNES

   Input Parameters:
+  snes - the multigrid context
-  n    - the number of smoothing steps

   Options Database Key:
.  -snes_fas_continuation - sets continuation to true

   Level: advanced

   Notes:
    This sets the prefix on the upsweep smoothers to -fas_continuation

.seealso: SNESFAS
@*/
PetscErrorCode SNESFASSetContinuation(SNES snes,PetscBool continuation)
{
  const char     *optionsprefix;
  char           tprefix[128];
  SNES_FAS       *fas;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes,SNES_CLASSID,1,SNESFAS);
  fas  = (SNES_FAS*)snes->data;
  CHKERRQ(SNESGetOptionsPrefix(fas->fine, &optionsprefix));
  if (!fas->smoothu) {
    CHKERRQ(SNESFASCycleCreateSmoother_Private(snes, &fas->smoothu));
  }
  CHKERRQ(PetscStrncpy(tprefix,"fas_levels_continuation_",sizeof(tprefix)));
  CHKERRQ(SNESSetOptionsPrefix(fas->smoothu, optionsprefix));
  CHKERRQ(SNESAppendOptionsPrefix(fas->smoothu, tprefix));
  CHKERRQ(SNESSetType(fas->smoothu,SNESNEWTONLS));
  CHKERRQ(SNESSetTolerances(fas->smoothu,fas->fine->abstol,fas->fine->rtol,fas->fine->stol,50,100));
  fas->continuation = continuation;
  if (fas->next) {
    CHKERRQ(SNESFASSetContinuation(fas->next,continuation));
  }
  PetscFunctionReturn(0);
}

/*@
   SNESFASSetCycles - Sets the number of FAS multigrid cycles to use each time a grid is visited.  Use SNESFASSetCyclesOnLevel() for more
   complicated cycling.

   Logically Collective on SNES

   Input Parameters:
+  snes   - the multigrid context
-  cycles - the number of cycles -- 1 for V-cycle, 2 for W-cycle

   Options Database Key:
.  -snes_fas_cycles <1,2> - 1 for V-cycle, 2 for W-cycle

   Level: advanced

.seealso: SNESFASSetCyclesOnLevel()
@*/
PetscErrorCode SNESFASSetCycles(SNES snes, PetscInt cycles)
{
  SNES_FAS       *fas;
  PetscBool      isFine;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes,SNES_CLASSID,1,SNESFAS);
  CHKERRQ(SNESFASCycleIsFine(snes, &isFine));
  fas = (SNES_FAS*)snes->data;
  fas->n_cycles = cycles;
  if (!isFine) {
    CHKERRQ(SNESSetTolerances(snes, snes->abstol, snes->rtol, snes->stol, cycles, snes->max_funcs));
  }
  if (fas->next) {
    CHKERRQ(SNESFASSetCycles(fas->next, cycles));
  }
  PetscFunctionReturn(0);
}

/*@
   SNESFASSetMonitor - Sets the method-specific cycle monitoring

   Logically Collective on SNES

   Input Parameters:
+  snes   - the FAS context
.  vf     - viewer and format structure (may be NULL if flg is FALSE)
-  flg    - monitor or not

   Level: advanced

.seealso: SNESFASSetCyclesOnLevel()
@*/
PetscErrorCode SNESFASSetMonitor(SNES snes, PetscViewerAndFormat *vf, PetscBool flg)
{
  SNES_FAS       *fas;
  PetscBool      isFine;
  PetscInt       i, levels;
  SNES           levelsnes;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes,SNES_CLASSID,1,SNESFAS);
  CHKERRQ(SNESFASCycleIsFine(snes, &isFine));
  fas = (SNES_FAS*)snes->data;
  levels = fas->levels;
  if (isFine) {
    for (i = 0; i < levels; i++) {
      CHKERRQ(SNESFASGetCycleSNES(snes, i, &levelsnes));
      fas  = (SNES_FAS*)levelsnes->data;
      if (flg) {
        /* set the monitors for the upsmoother and downsmoother */
        CHKERRQ(SNESMonitorCancel(levelsnes));
        /* Only register destroy on finest level */
        CHKERRQ(SNESMonitorSet(levelsnes,(PetscErrorCode (*)(SNES,PetscInt,PetscReal,void*))SNESMonitorDefault,vf,(!i ? (PetscErrorCode (*)(void**))PetscViewerAndFormatDestroy : NULL)));
      } else if (i != fas->levels - 1) {
        /* unset the monitors on the coarse levels */
        CHKERRQ(SNESMonitorCancel(levelsnes));
      }
    }
  }
  PetscFunctionReturn(0);
}

/*@
   SNESFASSetLog - Sets or unsets time logging for various FAS stages on all levels

   Logically Collective on SNES

   Input Parameters:
+  snes   - the FAS context
-  flg    - monitor or not

   Level: advanced

.seealso: SNESFASSetMonitor()
@*/
PetscErrorCode SNESFASSetLog(SNES snes, PetscBool flg)
{
  SNES_FAS       *fas;
  PetscBool      isFine;
  PetscInt       i, levels;
  SNES           levelsnes;
  char           eventname[128];

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes,SNES_CLASSID,1,SNESFAS);
  CHKERRQ(SNESFASCycleIsFine(snes, &isFine));
  fas = (SNES_FAS*)snes->data;
  levels = fas->levels;
  if (isFine) {
    for (i = 0; i < levels; i++) {
      CHKERRQ(SNESFASGetCycleSNES(snes, i, &levelsnes));
      fas  = (SNES_FAS*)levelsnes->data;
      if (flg) {
        CHKERRQ(PetscSNPrintf(eventname,sizeof(eventname),"FASSetup  %d",(int)i));
        CHKERRQ(PetscLogEventRegister(eventname,((PetscObject)snes)->classid,&fas->eventsmoothsetup));
        CHKERRQ(PetscSNPrintf(eventname,sizeof(eventname),"FASSmooth %d",(int)i));
        CHKERRQ(PetscLogEventRegister(eventname,((PetscObject)snes)->classid,&fas->eventsmoothsolve));
        CHKERRQ(PetscSNPrintf(eventname,sizeof(eventname),"FASResid  %d",(int)i));
        CHKERRQ(PetscLogEventRegister(eventname,((PetscObject)snes)->classid,&fas->eventresidual));
        CHKERRQ(PetscSNPrintf(eventname,sizeof(eventname),"FASInterp %d",(int)i));
        CHKERRQ(PetscLogEventRegister(eventname,((PetscObject)snes)->classid,&fas->eventinterprestrict));
      } else {
        fas->eventsmoothsetup    = 0;
        fas->eventsmoothsolve    = 0;
        fas->eventresidual       = 0;
        fas->eventinterprestrict = 0;
      }
    }
  }
  PetscFunctionReturn(0);
}

/*
Creates the default smoother type.

This is SNESNRICHARDSON on each fine level and SNESNEWTONLS on the coarse level.

 */
PetscErrorCode SNESFASCycleCreateSmoother_Private(SNES snes, SNES *smooth)
{
  SNES_FAS       *fas;
  const char     *optionsprefix;
  char           tprefix[128];
  SNES           nsmooth;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes,SNES_CLASSID,1,SNESFAS);
  PetscValidPointer(smooth,2);
  fas  = (SNES_FAS*)snes->data;
  CHKERRQ(SNESGetOptionsPrefix(fas->fine, &optionsprefix));
  /* create the default smoother */
  CHKERRQ(SNESCreate(PetscObjectComm((PetscObject)snes), &nsmooth));
  if (fas->level == 0) {
    CHKERRQ(PetscStrncpy(tprefix,"fas_coarse_",sizeof(tprefix)));
    CHKERRQ(SNESAppendOptionsPrefix(nsmooth, optionsprefix));
    CHKERRQ(SNESAppendOptionsPrefix(nsmooth, tprefix));
    CHKERRQ(SNESSetType(nsmooth, SNESNEWTONLS));
    CHKERRQ(SNESSetTolerances(nsmooth, nsmooth->abstol, nsmooth->rtol, nsmooth->stol, nsmooth->max_its, nsmooth->max_funcs));
  } else {
    CHKERRQ(PetscSNPrintf(tprefix,sizeof(tprefix),"fas_levels_%d_",(int)fas->level));
    CHKERRQ(SNESAppendOptionsPrefix(nsmooth, optionsprefix));
    CHKERRQ(SNESAppendOptionsPrefix(nsmooth, tprefix));
    CHKERRQ(SNESSetType(nsmooth, SNESNRICHARDSON));
    CHKERRQ(SNESSetTolerances(nsmooth, 0.0, 0.0, 0.0, fas->max_down_it, nsmooth->max_funcs));
  }
  CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)nsmooth, (PetscObject)snes, 1));
  CHKERRQ(PetscLogObjectParent((PetscObject)snes,(PetscObject)nsmooth));
  CHKERRQ(PetscObjectCopyFortranFunctionPointers((PetscObject)snes, (PetscObject)nsmooth));
  CHKERRQ(PetscObjectComposedDataSetInt((PetscObject) nsmooth, PetscMGLevelId, fas->level));
  *smooth = nsmooth;
  PetscFunctionReturn(0);
}

/* ------------- Functions called on a particular level ----------------- */

/*@
   SNESFASCycleSetCycles - Sets the number of cycles on a particular level.

   Logically Collective on SNES

   Input Parameters:
+  snes   - the multigrid context
.  level  - the level to set the number of cycles on
-  cycles - the number of cycles -- 1 for V-cycle, 2 for W-cycle

   Level: advanced

.seealso: SNESFASSetCycles()
@*/
PetscErrorCode SNESFASCycleSetCycles(SNES snes, PetscInt cycles)
{
  SNES_FAS       *fas;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes,SNES_CLASSID,1,SNESFAS);
  fas = (SNES_FAS*)snes->data;
  fas->n_cycles = cycles;
  CHKERRQ(SNESSetTolerances(snes, snes->abstol, snes->rtol, snes->stol, cycles, snes->max_funcs));
  PetscFunctionReturn(0);
}

/*@
   SNESFASCycleGetSmoother - Gets the smoother on a particular cycle level.

   Logically Collective on SNES

   Input Parameters:
.  snes   - the multigrid context

   Output Parameters:
.  smooth - the smoother

   Level: advanced

.seealso: SNESFASCycleGetSmootherUp(), SNESFASCycleGetSmootherDown()
@*/
PetscErrorCode SNESFASCycleGetSmoother(SNES snes, SNES *smooth)
{
  SNES_FAS *fas;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes,SNES_CLASSID,1,SNESFAS);
  PetscValidPointer(smooth,2);
  fas     = (SNES_FAS*)snes->data;
  *smooth = fas->smoothd;
  PetscFunctionReturn(0);
}
/*@
   SNESFASCycleGetSmootherUp - Gets the up smoother on a particular cycle level.

   Logically Collective on SNES

   Input Parameters:
.  snes   - the multigrid context

   Output Parameters:
.  smoothu - the smoother

   Notes:
   Returns the downsmoother if no up smoother is available.  This enables transparent
   default behavior in the process of the solve.

   Level: advanced

.seealso: SNESFASCycleGetSmoother(), SNESFASCycleGetSmootherDown()
@*/
PetscErrorCode SNESFASCycleGetSmootherUp(SNES snes, SNES *smoothu)
{
  SNES_FAS *fas;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes,SNES_CLASSID,1,SNESFAS);
  PetscValidPointer(smoothu,2);
  fas = (SNES_FAS*)snes->data;
  if (!fas->smoothu) *smoothu = fas->smoothd;
  else *smoothu = fas->smoothu;
  PetscFunctionReturn(0);
}

/*@
   SNESFASCycleGetSmootherDown - Gets the down smoother on a particular cycle level.

   Logically Collective on SNES

   Input Parameters:
.  snes   - the multigrid context

   Output Parameters:
.  smoothd - the smoother

   Level: advanced

.seealso: SNESFASCycleGetSmootherUp(), SNESFASCycleGetSmoother()
@*/
PetscErrorCode SNESFASCycleGetSmootherDown(SNES snes, SNES *smoothd)
{
  SNES_FAS *fas;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes,SNES_CLASSID,1,SNESFAS);
  PetscValidPointer(smoothd,2);
  fas = (SNES_FAS*)snes->data;
  *smoothd = fas->smoothd;
  PetscFunctionReturn(0);
}

/*@
   SNESFASCycleGetCorrection - Gets the coarse correction FAS context for this level

   Logically Collective on SNES

   Input Parameters:
.  snes   - the multigrid context

   Output Parameters:
.  correction - the coarse correction on this level

   Notes:
   Returns NULL on the coarsest level.

   Level: advanced

.seealso: SNESFASCycleGetSmootherUp(), SNESFASCycleGetSmoother()
@*/
PetscErrorCode SNESFASCycleGetCorrection(SNES snes, SNES *correction)
{
  SNES_FAS *fas;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes,SNES_CLASSID,1,SNESFAS);
  PetscValidPointer(correction,2);
  fas = (SNES_FAS*)snes->data;
  *correction = fas->next;
  PetscFunctionReturn(0);
}

/*@
   SNESFASCycleGetInterpolation - Gets the interpolation on this level

   Logically Collective on SNES

   Input Parameters:
.  snes   - the multigrid context

   Output Parameters:
.  mat    - the interpolation operator on this level

   Level: developer

.seealso: SNESFASCycleGetSmootherUp(), SNESFASCycleGetSmoother()
@*/
PetscErrorCode SNESFASCycleGetInterpolation(SNES snes, Mat *mat)
{
  SNES_FAS *fas;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes,SNES_CLASSID,1,SNESFAS);
  PetscValidPointer(mat,2);
  fas = (SNES_FAS*)snes->data;
  *mat = fas->interpolate;
  PetscFunctionReturn(0);
}

/*@
   SNESFASCycleGetRestriction - Gets the restriction on this level

   Logically Collective on SNES

   Input Parameters:
.  snes   - the multigrid context

   Output Parameters:
.  mat    - the restriction operator on this level

   Level: developer

.seealso: SNESFASGetRestriction(), SNESFASCycleGetInterpolation()
@*/
PetscErrorCode SNESFASCycleGetRestriction(SNES snes, Mat *mat)
{
  SNES_FAS *fas;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes,SNES_CLASSID,1,SNESFAS);
  PetscValidPointer(mat,2);
  fas = (SNES_FAS*)snes->data;
  *mat = fas->restrct;
  PetscFunctionReturn(0);
}

/*@
   SNESFASCycleGetInjection - Gets the injection on this level

   Logically Collective on SNES

   Input Parameters:
.  snes   - the multigrid context

   Output Parameters:
.  mat    - the restriction operator on this level

   Level: developer

.seealso: SNESFASGetInjection(), SNESFASCycleGetRestriction()
@*/
PetscErrorCode SNESFASCycleGetInjection(SNES snes, Mat *mat)
{
  SNES_FAS *fas;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes,SNES_CLASSID,1,SNESFAS);
  PetscValidPointer(mat,2);
  fas = (SNES_FAS*)snes->data;
  *mat = fas->inject;
  PetscFunctionReturn(0);
}

/*@
   SNESFASCycleGetRScale - Gets the injection on this level

   Logically Collective on SNES

   Input Parameters:
.  snes   - the multigrid context

   Output Parameters:
.  mat    - the restriction operator on this level

   Level: developer

.seealso: SNESFASCycleGetRestriction(), SNESFASGetRScale()
@*/
PetscErrorCode SNESFASCycleGetRScale(SNES snes, Vec *vec)
{
  SNES_FAS *fas;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes,SNES_CLASSID,1,SNESFAS);
  PetscValidPointer(vec,2);
  fas  = (SNES_FAS*)snes->data;
  *vec = fas->rscale;
  PetscFunctionReturn(0);
}

/*@
   SNESFASCycleIsFine - Determines if a given cycle is the fine level.

   Logically Collective on SNES

   Input Parameters:
.  snes   - the FAS context

   Output Parameters:
.  flg - indicates if this is the fine level or not

   Level: advanced

.seealso: SNESFASSetLevels()
@*/
PetscErrorCode SNESFASCycleIsFine(SNES snes, PetscBool *flg)
{
  SNES_FAS *fas;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes,SNES_CLASSID,1,SNESFAS);
  PetscValidBoolPointer(flg,2);
  fas = (SNES_FAS*)snes->data;
  if (fas->level == fas->levels - 1) *flg = PETSC_TRUE;
  else *flg = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/* ---------- functions called on the finest level that return level-specific information ---------- */

/*@
   SNESFASSetInterpolation - Sets the function to be used to calculate the
   interpolation from l-1 to the lth level

   Input Parameters:
+  snes      - the multigrid context
.  mat       - the interpolation operator
-  level     - the level (0 is coarsest) to supply [do not supply 0]

   Level: advanced

   Notes:
          Usually this is the same matrix used also to set the restriction
    for the same level.

          One can pass in the interpolation matrix or its transpose; PETSc figures
    out from the matrix size which one it is.

.seealso: SNESFASSetInjection(), SNESFASSetRestriction(), SNESFASSetRScale()
@*/
PetscErrorCode SNESFASSetInterpolation(SNES snes, PetscInt level, Mat mat)
{
  SNES_FAS       *fas;
  SNES           levelsnes;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes,SNES_CLASSID,1,SNESFAS);
  if (mat) PetscValidHeaderSpecific(mat,MAT_CLASSID,3);
  CHKERRQ(SNESFASGetCycleSNES(snes, level, &levelsnes));
  fas  = (SNES_FAS*)levelsnes->data;
  CHKERRQ(PetscObjectReference((PetscObject)mat));
  CHKERRQ(MatDestroy(&fas->interpolate));
  fas->interpolate = mat;
  PetscFunctionReturn(0);
}

/*@
   SNESFASGetInterpolation - Gets the matrix used to calculate the
   interpolation from l-1 to the lth level

   Input Parameters:
+  snes      - the multigrid context
-  level     - the level (0 is coarsest) to supply [do not supply 0]

   Output Parameters:
.  mat       - the interpolation operator

   Level: advanced

.seealso: SNESFASSetInterpolation(), SNESFASGetInjection(), SNESFASGetRestriction(), SNESFASGetRScale()
@*/
PetscErrorCode SNESFASGetInterpolation(SNES snes, PetscInt level, Mat *mat)
{
  SNES_FAS       *fas;
  SNES           levelsnes;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes,SNES_CLASSID,1,SNESFAS);
  PetscValidPointer(mat,3);
  CHKERRQ(SNESFASGetCycleSNES(snes, level, &levelsnes));
  fas  = (SNES_FAS*)levelsnes->data;
  *mat = fas->interpolate;
  PetscFunctionReturn(0);
}

/*@
   SNESFASSetRestriction - Sets the function to be used to restrict the defect
   from level l to l-1.

   Input Parameters:
+  snes  - the multigrid context
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

.seealso: SNESFASSetInterpolation(), SNESFASSetInjection()
@*/
PetscErrorCode SNESFASSetRestriction(SNES snes, PetscInt level, Mat mat)
{
  SNES_FAS       *fas;
  SNES           levelsnes;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes,SNES_CLASSID,1,SNESFAS);
  if (mat) PetscValidHeaderSpecific(mat,MAT_CLASSID,3);
  CHKERRQ(SNESFASGetCycleSNES(snes, level, &levelsnes));
  fas  = (SNES_FAS*)levelsnes->data;
  CHKERRQ(PetscObjectReference((PetscObject)mat));
  CHKERRQ(MatDestroy(&fas->restrct));
  fas->restrct = mat;
  PetscFunctionReturn(0);
}

/*@
   SNESFASGetRestriction - Gets the matrix used to calculate the
   restriction from l to the l-1th level

   Input Parameters:
+  snes      - the multigrid context
-  level     - the level (0 is coarsest) to supply [do not supply 0]

   Output Parameters:
.  mat       - the interpolation operator

   Level: advanced

.seealso: SNESFASSetRestriction(), SNESFASGetInjection(), SNESFASGetInterpolation(), SNESFASGetRScale()
@*/
PetscErrorCode SNESFASGetRestriction(SNES snes, PetscInt level, Mat *mat)
{
  SNES_FAS       *fas;
  SNES           levelsnes;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes,SNES_CLASSID,1,SNESFAS);
  PetscValidPointer(mat,3);
  CHKERRQ(SNESFASGetCycleSNES(snes, level, &levelsnes));
  fas  = (SNES_FAS*)levelsnes->data;
  *mat = fas->restrct;
  PetscFunctionReturn(0);
}

/*@
   SNESFASSetInjection - Sets the function to be used to inject the solution
   from level l to l-1.

   Input Parameters:
 +  snes  - the multigrid context
.  mat   - the restriction matrix
-  level - the level (0 is coarsest) to supply [Do not supply 0]

   Level: advanced

   Notes:
         If you do not set this, the restriction and rscale is used to
   project the solution instead.

.seealso: SNESFASSetInterpolation(), SNESFASSetRestriction()
@*/
PetscErrorCode SNESFASSetInjection(SNES snes, PetscInt level, Mat mat)
{
  SNES_FAS       *fas;
  SNES           levelsnes;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes,SNES_CLASSID,1,SNESFAS);
  if (mat) PetscValidHeaderSpecific(mat,MAT_CLASSID,3);
  CHKERRQ(SNESFASGetCycleSNES(snes, level, &levelsnes));
  fas  = (SNES_FAS*)levelsnes->data;
  CHKERRQ(PetscObjectReference((PetscObject)mat));
  CHKERRQ(MatDestroy(&fas->inject));

  fas->inject = mat;
  PetscFunctionReturn(0);
}

/*@
   SNESFASGetInjection - Gets the matrix used to calculate the
   injection from l-1 to the lth level

   Input Parameters:
+  snes      - the multigrid context
-  level     - the level (0 is coarsest) to supply [do not supply 0]

   Output Parameters:
.  mat       - the injection operator

   Level: advanced

.seealso: SNESFASSetInjection(), SNESFASGetRestriction(), SNESFASGetInterpolation(), SNESFASGetRScale()
@*/
PetscErrorCode SNESFASGetInjection(SNES snes, PetscInt level, Mat *mat)
{
  SNES_FAS       *fas;
  SNES           levelsnes;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes,SNES_CLASSID,1,SNESFAS);
  PetscValidPointer(mat,3);
  CHKERRQ(SNESFASGetCycleSNES(snes, level, &levelsnes));
  fas  = (SNES_FAS*)levelsnes->data;
  *mat = fas->inject;
  PetscFunctionReturn(0);
}

/*@
   SNESFASSetRScale - Sets the scaling factor of the restriction
   operator from level l to l-1.

   Input Parameters:
+  snes   - the multigrid context
.  rscale - the restriction scaling
-  level  - the level (0 is coarsest) to supply [Do not supply 0]

   Level: advanced

   Notes:
         This is only used in the case that the injection is not set.

.seealso: SNESFASSetInjection(), SNESFASSetRestriction()
@*/
PetscErrorCode SNESFASSetRScale(SNES snes, PetscInt level, Vec rscale)
{
  SNES_FAS       *fas;
  SNES           levelsnes;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes,SNES_CLASSID,1,SNESFAS);
  if (rscale) PetscValidHeaderSpecific(rscale,VEC_CLASSID,3);
  CHKERRQ(SNESFASGetCycleSNES(snes, level, &levelsnes));
  fas  = (SNES_FAS*)levelsnes->data;
  CHKERRQ(PetscObjectReference((PetscObject)rscale));
  CHKERRQ(VecDestroy(&fas->rscale));
  fas->rscale = rscale;
  PetscFunctionReturn(0);
}

/*@
   SNESFASGetSmoother - Gets the default smoother on a level.

   Input Parameters:
+  snes   - the multigrid context
-  level  - the level (0 is coarsest) to supply

   Output Parameters:
   smooth  - the smoother

   Level: advanced

.seealso: SNESFASSetInjection(), SNESFASSetRestriction()
@*/
PetscErrorCode SNESFASGetSmoother(SNES snes, PetscInt level, SNES *smooth)
{
  SNES_FAS       *fas;
  SNES           levelsnes;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes,SNES_CLASSID,1,SNESFAS);
  PetscValidPointer(smooth,3);
  CHKERRQ(SNESFASGetCycleSNES(snes, level, &levelsnes));
  fas  = (SNES_FAS*)levelsnes->data;
  if (!fas->smoothd) {
    CHKERRQ(SNESFASCycleCreateSmoother_Private(levelsnes, &fas->smoothd));
  }
  *smooth = fas->smoothd;
  PetscFunctionReturn(0);
}

/*@
   SNESFASGetSmootherDown - Gets the downsmoother on a level.

   Input Parameters:
+  snes   - the multigrid context
-  level  - the level (0 is coarsest) to supply

   Output Parameters:
   smooth  - the smoother

   Level: advanced

.seealso: SNESFASSetInjection(), SNESFASSetRestriction()
@*/
PetscErrorCode SNESFASGetSmootherDown(SNES snes, PetscInt level, SNES *smooth)
{
  SNES_FAS       *fas;
  SNES           levelsnes;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes,SNES_CLASSID,1,SNESFAS);
  PetscValidPointer(smooth,3);
  CHKERRQ(SNESFASGetCycleSNES(snes, level, &levelsnes));
  fas  = (SNES_FAS*)levelsnes->data;
  /* if the user chooses to differentiate smoothers, create them both at this point */
  if (!fas->smoothd) {
    CHKERRQ(SNESFASCycleCreateSmoother_Private(levelsnes, &fas->smoothd));
  }
  if (!fas->smoothu) {
    CHKERRQ(SNESFASCycleCreateSmoother_Private(levelsnes, &fas->smoothu));
  }
  *smooth = fas->smoothd;
  PetscFunctionReturn(0);
}

/*@
   SNESFASGetSmootherUp - Gets the upsmoother on a level.

   Input Parameters:
+  snes   - the multigrid context
-  level  - the level (0 is coarsest)

   Output Parameters:
   smooth  - the smoother

   Level: advanced

.seealso: SNESFASSetInjection(), SNESFASSetRestriction()
@*/
PetscErrorCode SNESFASGetSmootherUp(SNES snes, PetscInt level, SNES *smooth)
{
  SNES_FAS       *fas;
  SNES           levelsnes;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes,SNES_CLASSID,1,SNESFAS);
  PetscValidPointer(smooth,3);
  CHKERRQ(SNESFASGetCycleSNES(snes, level, &levelsnes));
  fas  = (SNES_FAS*)levelsnes->data;
  /* if the user chooses to differentiate smoothers, create them both at this point */
  if (!fas->smoothd) {
    CHKERRQ(SNESFASCycleCreateSmoother_Private(levelsnes, &fas->smoothd));
  }
  if (!fas->smoothu) {
    CHKERRQ(SNESFASCycleCreateSmoother_Private(levelsnes, &fas->smoothu));
  }
  *smooth = fas->smoothu;
  PetscFunctionReturn(0);
}

/*@
  SNESFASGetCoarseSolve - Gets the coarsest solver.

  Input Parameters:
. snes - the multigrid context

  Output Parameters:
. coarse - the coarse-level solver

  Level: advanced

.seealso: SNESFASSetInjection(), SNESFASSetRestriction()
@*/
PetscErrorCode SNESFASGetCoarseSolve(SNES snes, SNES *coarse)
{
  SNES_FAS       *fas;
  SNES           levelsnes;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes,SNES_CLASSID,1,SNESFAS);
  PetscValidPointer(coarse,2);
  CHKERRQ(SNESFASGetCycleSNES(snes, 0, &levelsnes));
  fas  = (SNES_FAS*)levelsnes->data;
  /* if the user chooses to differentiate smoothers, create them both at this point */
  if (!fas->smoothd) {
    CHKERRQ(SNESFASCycleCreateSmoother_Private(levelsnes, &fas->smoothd));
  }
  *coarse = fas->smoothd;
  PetscFunctionReturn(0);
}

/*@
   SNESFASFullSetDownSweep - Smooth during the initial downsweep for SNESFAS

   Logically Collective on SNES

   Input Parameters:
+  snes - the multigrid context
-  swp - whether to downsweep or not

   Options Database Key:
.  -snes_fas_full_downsweep - Sets number of pre-smoothing steps

   Level: advanced

.seealso: SNESFASSetNumberSmoothUp()
@*/
PetscErrorCode SNESFASFullSetDownSweep(SNES snes,PetscBool swp)
{
  SNES_FAS       *fas;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes,SNES_CLASSID,1,SNESFAS);
  fas = (SNES_FAS*)snes->data;
  fas->full_downsweep = swp;
  if (fas->next) {
    CHKERRQ(SNESFASFullSetDownSweep(fas->next,swp));
  }
  PetscFunctionReturn(0);
}

/*@
   SNESFASFullSetTotal - Use total residual restriction and total interpolation on the initial down and up sweep of full FAS cycles

   Logically Collective on SNES

   Input Parameters:
+  snes - the multigrid context
-  total - whether to use total restriction / interpolatiaon or not (the alternative is defect restriction and correction interpolation)

   Options Database Key:
.  -snes_fas_full_total - Use total restriction and interpolation on the initial down and up sweeps for the full FAS cycle

   Level: advanced

   Note: this option is only significant if the interpolation of a coarse correction (MatInterpolate()) is significantly different from total solution interpolation (DMInterpolateSolution()).

.seealso: SNESFASSetNumberSmoothUp(), DMInterpolateSolution()
@*/
PetscErrorCode SNESFASFullSetTotal(SNES snes,PetscBool total)
{
  SNES_FAS       *fas;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes,SNES_CLASSID,1,SNESFAS);
  fas = (SNES_FAS*)snes->data;
  fas->full_total = total;
  if (fas->next) {
    CHKERRQ(SNESFASFullSetTotal(fas->next,total));
  }
  PetscFunctionReturn(0);
}

/*@
   SNESFASFullGetTotal - Use total residual restriction and total interpolation on the initial down and up sweep of full FAS cycles

   Logically Collective on SNES

   Input Parameters:
.  snes - the multigrid context

   Output:
.  total - whether to use total restriction / interpolatiaon or not (the alternative is defect restriction and correction interpolation)

   Level: advanced

.seealso: SNESFASSetNumberSmoothUp(), DMInterpolateSolution(), SNESFullSetTotal()
@*/
PetscErrorCode SNESFASFullGetTotal(SNES snes,PetscBool *total)
{
  SNES_FAS       *fas;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes,SNES_CLASSID,1,SNESFAS);
  fas = (SNES_FAS*)snes->data;
  *total = fas->full_total;
  PetscFunctionReturn(0);
}
