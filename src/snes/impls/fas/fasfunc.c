#include "../src/snes/impls/fas/fasimpls.h" /*I  "petscsnesfas.h"  I*/


extern PetscErrorCode SNESFASCycleCreateSmoother_Private(SNES, SNES *);

/* -------------- functions called on the fine level -------------- */

#undef __FUNCT__
#define __FUNCT__ "SNESFASSetType"
/*@
SNESFASSetType - Sets the update and correction type used for FAS.

Logically Collective

Input Parameters:
+ snes  - FAS context
- fastype  - SNES_FAS_ADDITIVE or SNES_FAS_MULTIPLICATIVE

Level: intermediate

.seealso: PCMGSetType()
@*/
PetscErrorCode  SNESFASSetType(SNES snes,SNESFASType fastype)
{
  SNES_FAS                   *fas = (SNES_FAS*)snes->data;
  PetscErrorCode             ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidLogicalCollectiveEnum(snes,fastype,2);
  fas->fastype = fastype;
  if (fas->next) {ierr = SNESFASSetType(fas->next, fastype);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SNESFASGetType"
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
  SNES_FAS                   *fas = (SNES_FAS*)snes->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidPointer(fastype, 2);
  *fastype = fas->fastype;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESFASSetLevels"
/*@C
   SNESFASSetLevels - Sets the number of levels to use with FAS.
   Must be called before any other FAS routine.

   Input Parameters:
+  snes   - the snes context
.  levels - the number of levels
-  comms  - optional communicators for each level; this is to allow solving the coarser
            problems on smaller sets of processors. Use PETSC_NULL_OBJECT for default in
            Fortran.

   Level: intermediate

   Notes:
     If the number of levels is one then the multigrid uses the -fas_levels prefix
  for setting the level options rather than the -fas_coarse prefix.

.keywords: FAS, MG, set, levels, multigrid

.seealso: SNESFASGetLevels()
@*/
PetscErrorCode SNESFASSetLevels(SNES snes, PetscInt levels, MPI_Comm * comms) {
  PetscErrorCode ierr;
  PetscInt i;
  const char     *optionsprefix;
  char           tprefix[128];
  SNES_FAS * fas = (SNES_FAS *)snes->data;
  SNES prevsnes;
  MPI_Comm comm;
  PetscFunctionBegin;
  comm = ((PetscObject)snes)->comm;
  if (levels == fas->levels) {
    if (!comms)
      PetscFunctionReturn(0);
  }
  /* user has changed the number of levels; reset */
  ierr = SNESReset(snes);CHKERRQ(ierr);
  /* destroy any coarser levels if necessary */
  if (fas->next) SNESDestroy(&fas->next);CHKERRQ(ierr);
  fas->next = PETSC_NULL;
  fas->previous = PETSC_NULL;
  prevsnes = snes;
  /* setup the finest level */
  ierr = SNESGetOptionsPrefix(snes, &optionsprefix);CHKERRQ(ierr);
  for (i = levels - 1; i >= 0; i--) {
    if (comms) comm = comms[i];
    fas->level = i;
    fas->levels = levels;
    fas->fine = snes;
    fas->next = PETSC_NULL;
    if (i > 0) {
      ierr = SNESCreate(comm, &fas->next);CHKERRQ(ierr);
      sprintf(tprefix,"fas_levels_%d_cycle_",(int)fas->level);
      ierr = SNESAppendOptionsPrefix(fas->next,tprefix);CHKERRQ(ierr);
      ierr = SNESAppendOptionsPrefix(fas->next,optionsprefix);CHKERRQ(ierr);
      ierr = SNESSetType(fas->next, SNESFAS);CHKERRQ(ierr);
      ierr = SNESSetTolerances(fas->next, fas->next->abstol, fas->next->rtol, fas->next->stol, fas->n_cycles, fas->next->max_funcs);CHKERRQ(ierr);
      ierr = PetscObjectIncrementTabLevel((PetscObject)fas->next, (PetscObject)snes, levels - i);CHKERRQ(ierr);
      ((SNES_FAS *)fas->next->data)->previous = prevsnes;
      prevsnes = fas->next;
      fas = (SNES_FAS *)prevsnes->data;
    }
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SNESFASGetLevels"
/*@
   SNESFASGetLevels - Gets the number of levels in a FAS, including fine and coarse grids

   Input Parameter:
.  snes - the nonlinear solver context

   Output parameter:
.  levels - the number of levels

   Level: advanced

.keywords: MG, get, levels, multigrid

.seealso: SNESFASSetLevels(), PCMGGetLevels()
@*/
PetscErrorCode SNESFASGetLevels(SNES snes, PetscInt * levels) {
  SNES_FAS * fas = (SNES_FAS *)snes->data;
  PetscFunctionBegin;
  *levels = fas->levels;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SNESFASGetCycleSNES"
/*@
   SNESFASGetCycleSNES - Gets the SNES corresponding to a particular
   level of the FAS hierarchy.

   Input Parameters:
+  snes    - the multigrid context
   level   - the level to get
-  lsnes   - whether to use the nonlinear smoother or not

   Level: advanced

.keywords: FAS, MG, set, cycles, Gauss-Seidel, multigrid

.seealso: SNESFASSetLevels(), SNESFASGetLevels()
@*/
PetscErrorCode SNESFASGetCycleSNES(SNES snes,PetscInt level,SNES *lsnes) {
  SNES_FAS *fas = (SNES_FAS*)snes->data;
  PetscInt i;

  PetscFunctionBegin;
  if (level > fas->levels-1) SETERRQ2(((PetscObject)snes)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Requested level %D from SNESFAS containing %D levels",level,fas->levels);
  if (fas->level !=  fas->levels - 1)
    SETERRQ2(((PetscObject)snes)->comm,PETSC_ERR_ARG_OUTOFRANGE,"SNESFASGetCycleSNES may only be called on the finest-level SNES.",level,fas->level);

  *lsnes = snes;
  for (i = fas->level; i > level; i--) {
    *lsnes = fas->next;
    fas = (SNES_FAS *)(*lsnes)->data;
  }
  if (fas->level != level) SETERRQ(((PetscObject)snes)->comm,PETSC_ERR_PLIB,"SNESFAS level hierarchy corrupt");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESFASSetNumberSmoothUp"
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

.keywords: FAS, MG, smooth, down, pre-smoothing, steps, multigrid

.seealso: SNESFASSetNumberSmoothDown()
@*/
PetscErrorCode SNESFASSetNumberSmoothUp(SNES snes, PetscInt n) {
  SNES_FAS * fas =  (SNES_FAS *)snes->data;
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  fas->max_up_it = n;
  if (!fas->smoothu) {
    ierr = SNESFASCycleCreateSmoother_Private(snes, &fas->smoothu);
  }
  ierr = SNESSetTolerances(fas->smoothu, fas->smoothu->abstol, fas->smoothu->rtol, fas->smoothu->stol, n, fas->smoothu->max_funcs);CHKERRQ(ierr);
  if (fas->next) {
    ierr = SNESFASSetNumberSmoothUp(fas->next, n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESFASSetNumberSmoothDown"
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

.keywords: FAS, MG, smooth, down, pre-smoothing, steps, multigrid

.seealso: SNESFASSetNumberSmoothUp()
@*/
PetscErrorCode SNESFASSetNumberSmoothDown(SNES snes, PetscInt n) {
  SNES_FAS * fas =  (SNES_FAS *)snes->data;
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  if (!fas->smoothu) {
    ierr = SNESFASCycleCreateSmoother_Private(snes, &fas->smoothu);
  }
  if (!fas->smoothd) {
    ierr = SNESFASCycleCreateSmoother_Private(snes, &fas->smoothd);
  }
  ierr = SNESSetTolerances(fas->smoothd, fas->smoothd->abstol, fas->smoothd->rtol, fas->smoothd->stol, n, fas->smoothd->max_funcs);CHKERRQ(ierr);
  fas->max_down_it = n;
  if (fas->next) {
    ierr = SNESFASSetNumberSmoothDown(fas->next, n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SNESFASSetCycles"
/*@
   SNESFASSetCycles - Sets the number of FAS multigrid cycles to use each time a grid is visited.  Use SNESFASSetCyclesOnLevel() for more
   complicated cycling.

   Logically Collective on SNES

   Input Parameters:
+  snes   - the multigrid context
-  cycles - the number of cycles -- 1 for V-cycle, 2 for W-cycle

   Options Database Key:
$  -snes_fas_cycles 1 or 2

   Level: advanced

.keywords: MG, set, cycles, V-cycle, W-cycle, multigrid

.seealso: SNESFASSetCyclesOnLevel()
@*/
PetscErrorCode SNESFASSetCycles(SNES snes, PetscInt cycles) {
  SNES_FAS * fas = (SNES_FAS *)snes->data;
  PetscErrorCode ierr;
  PetscBool isFine;
  PetscFunctionBegin;
  ierr = SNESFASCycleIsFine(snes, &isFine);
  fas->n_cycles = cycles;
  if (!isFine)
    ierr = SNESSetTolerances(snes, snes->abstol, snes->rtol, snes->stol, cycles, snes->max_funcs);CHKERRQ(ierr);
  if (fas->next) {
    ierr = SNESFASSetCycles(fas->next, cycles);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SNESFASSetMonitor"
/*@
   SNESFASSetMonitor - Sets the method-specific cycle monitoring

   Logically Collective on SNES

   Input Parameters:
+  snes   - the FAS context
-  flg    - monitor or not

   Level: advanced

.keywords: FAS, monitor

.seealso: SNESFASSetCyclesOnLevel()
@*/
PetscErrorCode SNESFASSetMonitor(SNES snes, PetscBool flg) {
  SNES_FAS       *fas = (SNES_FAS *)snes->data;
  PetscErrorCode ierr;
  PetscBool      isFine;
  PetscInt       i, levels = fas->levels;
  SNES           levelsnes;
  PetscFunctionBegin;
  ierr = SNESFASCycleIsFine(snes, &isFine);CHKERRQ(ierr);
  if (isFine) {
    for (i = 0; i < levels; i++) {
      ierr = SNESFASGetCycleSNES(snes, i, &levelsnes);
      fas = (SNES_FAS *)levelsnes->data;
      if (flg) {
        fas->monitor = PETSC_VIEWER_STDOUT_(((PetscObject)levelsnes)->comm);CHKERRQ(ierr);
        /* set the monitors for the upsmoother and downsmoother */
        ierr = SNESMonitorCancel(levelsnes);CHKERRQ(ierr);
        ierr = SNESMonitorSet(levelsnes,SNESMonitorDefault,PETSC_NULL,(PetscErrorCode (*)(void**))PetscViewerDestroy);CHKERRQ(ierr);
      } else {
        /* unset the monitors on the coarse levels */
        if (i != fas->levels - 1) {
          ierr = SNESMonitorCancel(levelsnes);CHKERRQ(ierr);
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESFASCycleCreateSmoother_Private"
/*
Creates the default smoother type.

This is SNESNRICHARDSON on each fine level and SNESLS on the coarse level.

 */
PetscErrorCode SNESFASCycleCreateSmoother_Private(SNES snes, SNES *smooth) {
  SNES_FAS       *fas;
  const char     *optionsprefix;
  char           tprefix[128];
  PetscErrorCode ierr;
  SNES           nsmooth;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  fas = (SNES_FAS*)snes->data;
  ierr = SNESGetOptionsPrefix(fas->fine, &optionsprefix);CHKERRQ(ierr);
  /* create the default smoother */
  ierr = SNESCreate(((PetscObject)snes)->comm, &nsmooth);CHKERRQ(ierr);
  if (fas->level == 0) {
    sprintf(tprefix,"fas_coarse_");
    ierr = SNESAppendOptionsPrefix(nsmooth, optionsprefix);CHKERRQ(ierr);
    ierr = SNESAppendOptionsPrefix(nsmooth, tprefix);CHKERRQ(ierr);
    ierr = SNESSetType(nsmooth, SNESLS);CHKERRQ(ierr);
    ierr = SNESSetTolerances(nsmooth, nsmooth->abstol, nsmooth->rtol, nsmooth->stol, nsmooth->max_its, nsmooth->max_funcs);CHKERRQ(ierr);
  } else {
    sprintf(tprefix,"fas_levels_%d_",(int)fas->level);
    ierr = SNESAppendOptionsPrefix(nsmooth, optionsprefix);CHKERRQ(ierr);
    ierr = SNESAppendOptionsPrefix(nsmooth, tprefix);CHKERRQ(ierr);
    ierr = SNESSetType(nsmooth, SNESNRICHARDSON);CHKERRQ(ierr);
    ierr = SNESSetTolerances(nsmooth, 0.0, 0.0, 0.0, fas->max_down_it, nsmooth->max_funcs);CHKERRQ(ierr);
  }
  ierr = PetscObjectIncrementTabLevel((PetscObject)nsmooth, (PetscObject)snes, 1);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(snes,nsmooth);CHKERRQ(ierr);
  ierr = PetscObjectCopyFortranFunctionPointers((PetscObject)snes, (PetscObject)nsmooth);CHKERRQ(ierr);
  *smooth = nsmooth;
  PetscFunctionReturn(0);
}

/* ------------- Functions called on a particular level ----------------- */

#undef __FUNCT__
#define __FUNCT__ "SNESFASCycleSetCycles"
/*@
   SNESFASCycleSetCycles - Sets the number of cycles on a particular level.

   Logically Collective on SNES

   Input Parameters:
+  snes   - the multigrid context
.  level  - the level to set the number of cycles on
-  cycles - the number of cycles -- 1 for V-cycle, 2 for W-cycle

   Level: advanced

.keywords: SNES, FAS, set, cycles, V-cycle, W-cycle, multigrid

.seealso: SNESFASSetCycles()
@*/
PetscErrorCode SNESFASCycleSetCycles(SNES snes, PetscInt cycles) {
  SNES_FAS * fas =  (SNES_FAS *)snes->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  fas->n_cycles = cycles;
  ierr = SNESSetTolerances(snes, snes->abstol, snes->rtol, snes->stol, cycles, snes->max_funcs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SNESFASCycleGetSmoother"
/*@
   SNESFASCycleGetSmoother - Gets the smoother on a particular cycle level.

   Logically Collective on SNES

   Input Parameters:
.  snes   - the multigrid context

   Output Parameters:
.  smooth - the smoother

   Level: advanced

.keywords: SNES, FAS, get, smoother, multigrid

.seealso: SNESFASCycleGetSmootherUp(), SNESFASCycleGetSmootherDown()
@*/
PetscErrorCode SNESFASCycleGetSmoother(SNES snes, SNES *smooth)
{
  SNES_FAS       *fas;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  fas = (SNES_FAS*)snes->data;
  *smooth = fas->smoothd;
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "SNESFASCycleGetSmootherUp"
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

.keywords: SNES, FAS, get, smoother, multigrid

.seealso: SNESFASCycleGetSmoother(), SNESFASCycleGetSmootherDown()
@*/
PetscErrorCode SNESFASCycleGetSmootherUp(SNES snes, SNES *smoothu)
{
  SNES_FAS       *fas;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  fas = (SNES_FAS*)snes->data;
  if (!fas->smoothu) {
    *smoothu = fas->smoothd;
  } else {
    *smoothu = fas->smoothu;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESFASCycleGetSmootherDown"
/*@
   SNESFASCycleGetSmootherDown - Gets the down smoother on a particular cycle level.

   Logically Collective on SNES

   Input Parameters:
.  snes   - the multigrid context

   Output Parameters:
.  smoothd - the smoother

   Level: advanced

.keywords: SNES, FAS, get, smoother, multigrid

.seealso: SNESFASCycleGetSmootherUp(), SNESFASCycleGetSmoother()
@*/
PetscErrorCode SNESFASCycleGetSmootherDown(SNES snes, SNES *smoothd)
{
  SNES_FAS       *fas;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  fas = (SNES_FAS*)snes->data;
  *smoothd = fas->smoothd;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SNESFASCycleGetCorrection"
/*@
   SNESFASCycleGetCorrection - Gets the coarse correction FAS context for this level

   Logically Collective on SNES

   Input Parameters:
.  snes   - the multigrid context

   Output Parameters:
.  correction - the coarse correction on this level

   Notes:
   Returns PETSC_NULL on the coarsest level.

   Level: advanced

.keywords: SNES, FAS, get, smoother, multigrid

.seealso: SNESFASCycleGetSmootherUp(), SNESFASCycleGetSmoother()
@*/
PetscErrorCode SNESFASCycleGetCorrection(SNES snes, SNES *correction)
{
  SNES_FAS       *fas;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  fas = (SNES_FAS*)snes->data;
  *correction = fas->next;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESFASCycleGetInterpolation"
/*@
   SNESFASCycleGetInterpolation - Gets the interpolation on this level

   Logically Collective on SNES

   Input Parameters:
.  snes   - the multigrid context

   Output Parameters:
.  mat    - the interpolation operator on this level

   Level: developer

.keywords: SNES, FAS, get, smoother, multigrid

.seealso: SNESFASCycleGetSmootherUp(), SNESFASCycleGetSmoother()
@*/
PetscErrorCode SNESFASCycleGetInterpolation(SNES snes, Mat *mat)
{
  SNES_FAS       *fas;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  fas = (SNES_FAS*)snes->data;
  *mat = fas->interpolate;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SNESFASCycleGetRestriction"
/*@
   SNESFASCycleGetRestriction - Gets the restriction on this level

   Logically Collective on SNES

   Input Parameters:
.  snes   - the multigrid context

   Output Parameters:
.  mat    - the restriction operator on this level

   Level: developer

.keywords: SNES, FAS, get, smoother, multigrid

.seealso: SNESFASGetRestriction(), SNESFASCycleGetInterpolation()
@*/
PetscErrorCode SNESFASCycleGetRestriction(SNES snes, Mat *mat)
{
  SNES_FAS       *fas;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  fas = (SNES_FAS*)snes->data;
  *mat = fas->restrct;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SNESFASCycleGetInjection"
/*@
   SNESFASCycleGetInjection - Gets the injection on this level

   Logically Collective on SNES

   Input Parameters:
.  snes   - the multigrid context

   Output Parameters:
.  mat    - the restriction operator on this level

   Level: developer

.keywords: SNES, FAS, get, smoother, multigrid

.seealso: SNESFASGetInjection(), SNESFASCycleGetRestriction()
@*/
PetscErrorCode SNESFASCycleGetInjection(SNES snes, Mat *mat)
{
  SNES_FAS       *fas;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  fas = (SNES_FAS*)snes->data;
  *mat = fas->inject;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESFASCycleGetRScale"
/*@
   SNESFASCycleGetRScale - Gets the injection on this level

   Logically Collective on SNES

   Input Parameters:
.  snes   - the multigrid context

   Output Parameters:
.  mat    - the restriction operator on this level

   Level: developer

.keywords: SNES, FAS, get, smoother, multigrid

.seealso: SNESFASCycleGetRestriction(), SNESFASGetRScale()
@*/
PetscErrorCode SNESFASCycleGetRScale(SNES snes, Vec *vec)
{
  SNES_FAS       *fas;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  fas = (SNES_FAS*)snes->data;
  *vec = fas->rscale;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESFASCycleIsFine"
/*@
   SNESFASCycleIsFine - Determines if a given cycle is the fine level.

   Logically Collective on SNES

   Input Parameters:
.  snes   - the FAS context

   Output Parameters:
.  flg - indicates if this is the fine level or not

   Level: advanced

.keywords: SNES, FAS

.seealso: SNESFASSetLevels()
@*/
PetscErrorCode SNESFASCycleIsFine(SNES snes, PetscBool *flg)
{
  SNES_FAS       *fas;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  fas = (SNES_FAS*)snes->data;
  if (fas->level == fas->levels - 1) {
    *flg = PETSC_TRUE;
  } else {
    *flg = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

/* ---------- functions called on the finest level that return level-specific information ---------- */

#undef __FUNCT__
#define __FUNCT__ "SNESFASSetInterpolation"
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

.keywords:  FAS, multigrid, set, interpolate, level

.seealso: SNESFASSetInjection(), SNESFASSetRestriction(), SNESFASSetRScale()
@*/
PetscErrorCode SNESFASSetInterpolation(SNES snes, PetscInt level, Mat mat) {
  SNES_FAS       *fas =  (SNES_FAS *)snes->data;
  PetscErrorCode ierr;
  SNES           levelsnes;
  PetscFunctionBegin;
  ierr = SNESFASGetCycleSNES(snes, level, &levelsnes);CHKERRQ(ierr);
  fas = (SNES_FAS *)levelsnes->data;
  ierr = PetscObjectReference((PetscObject)mat);CHKERRQ(ierr);
  ierr = MatDestroy(&fas->interpolate);CHKERRQ(ierr);
  fas->interpolate = mat;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESFASGetInterpolation"
/*@
   SNESFASGetInterpolation - Gets the matrix used to calculate the
   interpolation from l-1 to the lth level

   Input Parameters:
+  snes      - the multigrid context
-  level     - the level (0 is coarsest) to supply [do not supply 0]

   Output Parameters:
.  mat       - the interpolation operator

   Level: advanced

.keywords:  FAS, multigrid, get, interpolate, level

.seealso: SNESFASSetInterpolation(), SNESFASGetInjection(), SNESFASGetRestriction(), SNESFASGetRScale()
@*/
PetscErrorCode SNESFASGetInterpolation(SNES snes, PetscInt level, Mat *mat) {
  SNES_FAS       *fas =  (SNES_FAS *)snes->data;
  PetscErrorCode ierr;
  SNES           levelsnes;
  PetscFunctionBegin;
  ierr = SNESFASGetCycleSNES(snes, level, &levelsnes);CHKERRQ(ierr);
  fas = (SNES_FAS *)levelsnes->data;
  *mat = fas->interpolate;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESFASSetRestriction"
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

.keywords: FAS, MG, set, multigrid, restriction, level

.seealso: SNESFASSetInterpolation(), SNESFASSetInjection()
@*/
PetscErrorCode SNESFASSetRestriction(SNES snes, PetscInt level, Mat mat) {
  SNES_FAS * fas =  (SNES_FAS *)snes->data;
  PetscErrorCode ierr;
  SNES levelsnes;
  PetscFunctionBegin;
  ierr = SNESFASGetCycleSNES(snes, level, &levelsnes);CHKERRQ(ierr);
  fas = (SNES_FAS *)levelsnes->data;
  ierr = PetscObjectReference((PetscObject)mat);CHKERRQ(ierr);
  ierr = MatDestroy(&fas->restrct);CHKERRQ(ierr);
  fas->restrct = mat;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESFASGetRestriction"
/*@
   SNESFASGetRestriction - Gets the matrix used to calculate the
   restriction from l to the l-1th level

   Input Parameters:
+  snes      - the multigrid context
-  level     - the level (0 is coarsest) to supply [do not supply 0]

   Output Parameters:
.  mat       - the interpolation operator

   Level: advanced

.keywords:  FAS, multigrid, get, restrict, level

.seealso: SNESFASSetRestriction(), SNESFASGetInjection(), SNESFASGetInterpolation(), SNESFASGetRScale()
@*/
PetscErrorCode SNESFASGetRestriction(SNES snes, PetscInt level, Mat *mat) {
  SNES_FAS       *fas =  (SNES_FAS *)snes->data;
  PetscErrorCode ierr;
  SNES           levelsnes;
  PetscFunctionBegin;
  ierr = SNESFASGetCycleSNES(snes, level, &levelsnes);CHKERRQ(ierr);
  fas = (SNES_FAS *)levelsnes->data;
  *mat = fas->restrct;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SNESFASSetInjection"
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

.keywords: FAS, MG, set, multigrid, restriction, level

.seealso: SNESFASSetInterpolation(), SNESFASSetRestriction()
@*/
PetscErrorCode SNESFASSetInjection(SNES snes, PetscInt level, Mat mat) {
  SNES_FAS       *fas =  (SNES_FAS *)snes->data;
  PetscErrorCode ierr;
  SNES           levelsnes;
  PetscFunctionBegin;
  ierr = SNESFASGetCycleSNES(snes, level, &levelsnes);CHKERRQ(ierr);
  fas = (SNES_FAS *)levelsnes->data;
  ierr = PetscObjectReference((PetscObject)mat);CHKERRQ(ierr);
  ierr = MatDestroy(&fas->inject);CHKERRQ(ierr);
  fas->inject = mat;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SNESFASGetInjection"
/*@
   SNESFASGetInjection - Gets the matrix used to calculate the
   injection from l-1 to the lth level

   Input Parameters:
+  snes      - the multigrid context
-  level     - the level (0 is coarsest) to supply [do not supply 0]

   Output Parameters:
.  mat       - the injection operator

   Level: advanced

.keywords:  FAS, multigrid, get, restrict, level

.seealso: SNESFASSetInjection(), SNESFASGetRestriction(), SNESFASGetInterpolation(), SNESFASGetRScale()
@*/
PetscErrorCode SNESFASGetInjection(SNES snes, PetscInt level, Mat *mat) {
  SNES_FAS       *fas =  (SNES_FAS *)snes->data;
  PetscErrorCode ierr;
  SNES           levelsnes;
  PetscFunctionBegin;
  ierr = SNESFASGetCycleSNES(snes, level, &levelsnes);CHKERRQ(ierr);
  fas = (SNES_FAS *)levelsnes->data;
  *mat = fas->inject;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESFASSetRScale"
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

.keywords: FAS, MG, set, multigrid, restriction, level

.seealso: SNESFASSetInjection(), SNESFASSetRestriction()
@*/
PetscErrorCode SNESFASSetRScale(SNES snes, PetscInt level, Vec rscale) {
  SNES_FAS * fas;
  PetscErrorCode ierr;
  SNES levelsnes;
  PetscFunctionBegin;
  ierr = SNESFASGetCycleSNES(snes, level, &levelsnes);CHKERRQ(ierr);
  fas = (SNES_FAS *)levelsnes->data;
  ierr = PetscObjectReference((PetscObject)rscale);CHKERRQ(ierr);
  ierr = VecDestroy(&fas->rscale);CHKERRQ(ierr);
  fas->rscale = rscale;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESFASGetSmoother"
/*@
   SNESFASGetSmoother - Gets the default smoother on a level.

   Input Parameters:
+  snes   - the multigrid context
-  level  - the level (0 is coarsest) to supply

   Output Parameters:
   smooth  - the smoother

   Level: advanced

.keywords: FAS, MG, get, multigrid, smoother, level

.seealso: SNESFASSetInjection(), SNESFASSetRestriction()
@*/
PetscErrorCode SNESFASGetSmoother(SNES snes, PetscInt level, SNES *smooth) {
  SNES_FAS * fas;
  PetscErrorCode ierr;
  SNES levelsnes;
  PetscFunctionBegin;
  ierr = SNESFASGetCycleSNES(snes, level, &levelsnes);CHKERRQ(ierr);
  fas = (SNES_FAS *)levelsnes->data;
  if (!fas->smoothd) {
    ierr = SNESFASCycleCreateSmoother_Private(snes, &fas->smoothd);CHKERRQ(ierr);
  }
  *smooth = fas->smoothd;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESFASGetSmootherDown"
/*@
   SNESFASGetSmootherDown - Gets the downsmoother on a level.

   Input Parameters:
+  snes   - the multigrid context
-  level  - the level (0 is coarsest) to supply

   Output Parameters:
   smooth  - the smoother

   Level: advanced

.keywords: FAS, MG, get, multigrid, smoother, level

.seealso: SNESFASSetInjection(), SNESFASSetRestriction()
@*/
PetscErrorCode SNESFASGetSmootherDown(SNES snes, PetscInt level, SNES *smooth) {
  SNES_FAS * fas;
  PetscErrorCode ierr;
  SNES levelsnes;
  PetscFunctionBegin;
  ierr = SNESFASGetCycleSNES(snes, level, &levelsnes);CHKERRQ(ierr);
  fas = (SNES_FAS *)levelsnes->data;
  /* if the user chooses to differentiate smoothers, create them both at this point */
  if (!fas->smoothd) {
    ierr = SNESFASCycleCreateSmoother_Private(snes, &fas->smoothd);CHKERRQ(ierr);
  }
  if (!fas->smoothu) {
    ierr = SNESFASCycleCreateSmoother_Private(snes, &fas->smoothu);CHKERRQ(ierr);
  }
  *smooth = fas->smoothd;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESFASGetSmootherUp"
/*@
   SNESFASGetSmootherUp - Gets the upsmoother on a level.

   Input Parameters:
+  snes   - the multigrid context
-  level  - the level (0 is coarsest)

   Output Parameters:
   smooth  - the smoother

   Level: advanced

.keywords: FAS, MG, get, multigrid, smoother, level

.seealso: SNESFASSetInjection(), SNESFASSetRestriction()
@*/
PetscErrorCode SNESFASGetSmootherUp(SNES snes, PetscInt level, SNES *smooth) {
  SNES_FAS * fas;
  PetscErrorCode ierr;
  SNES levelsnes;
  PetscFunctionBegin;
  ierr = SNESFASGetCycleSNES(snes, level, &levelsnes);CHKERRQ(ierr);
  fas = (SNES_FAS *)levelsnes->data;
  /* if the user chooses to differentiate smoothers, create them both at this point */
  if (!fas->smoothd) {
    ierr = SNESFASCycleCreateSmoother_Private(snes, &fas->smoothd);CHKERRQ(ierr);
  }
  if (!fas->smoothu) {
    ierr = SNESFASCycleCreateSmoother_Private(snes, &fas->smoothu);CHKERRQ(ierr);
  }
  *smooth = fas->smoothu;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESFASGetCoarseSolve"
/*@
   SNESFASGetCoarseSolve - Gets the coarsest solver.

   Input Parameters:
+  snes   - the multigrid context

   Output Parameters:
   solve  - the coarse-level solver

   Level: advanced

.keywords: FAS, MG, get, multigrid, solver, coarse

.seealso: SNESFASSetInjection(), SNESFASSetRestriction()
@*/
PetscErrorCode SNESFASGetCoarseSolve(SNES snes, SNES *smooth) {
  SNES_FAS * fas;
  PetscErrorCode ierr;
  SNES levelsnes;
  PetscFunctionBegin;
  ierr = SNESFASGetCycleSNES(snes, 0, &levelsnes);CHKERRQ(ierr);
  fas = (SNES_FAS *)levelsnes->data;
  /* if the user chooses to differentiate smoothers, create them both at this point */
  if (!fas->smoothd) {
    ierr = SNESFASCycleCreateSmoother_Private(snes, &fas->smoothd);CHKERRQ(ierr);
  }
  *smooth = fas->smoothd;
  PetscFunctionReturn(0);
}
