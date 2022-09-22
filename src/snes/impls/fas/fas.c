/* Defines the basic SNES object */
#include <../src/snes/impls/fas/fasimpls.h> /*I  "petscsnes.h"  I*/

const char *const SNESFASTypes[] = {"MULTIPLICATIVE", "ADDITIVE", "FULL", "KASKADE", "SNESFASType", "SNES_FAS", NULL};

static PetscErrorCode SNESReset_FAS(SNES snes)
{
  SNES_FAS *fas = (SNES_FAS *)snes->data;

  PetscFunctionBegin;
  PetscCall(SNESDestroy(&fas->smoothu));
  PetscCall(SNESDestroy(&fas->smoothd));
  PetscCall(MatDestroy(&fas->inject));
  PetscCall(MatDestroy(&fas->interpolate));
  PetscCall(MatDestroy(&fas->restrct));
  PetscCall(VecDestroy(&fas->rscale));
  PetscCall(VecDestroy(&fas->Xg));
  PetscCall(VecDestroy(&fas->Fg));
  if (fas->next) PetscCall(SNESReset(fas->next));
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESDestroy_FAS(SNES snes)
{
  SNES_FAS *fas = (SNES_FAS *)snes->data;

  PetscFunctionBegin;
  /* recursively resets and then destroys */
  PetscCall(SNESReset_FAS(snes));
  PetscCall(SNESDestroy(&fas->next));
  PetscCall(PetscFree(fas));
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESFASSetUpLineSearch_Private(SNES snes, SNES smooth)
{
  SNESLineSearch linesearch;
  SNESLineSearch slinesearch;
  void          *lsprectx, *lspostctx;
  PetscErrorCode (*precheck)(SNESLineSearch, Vec, Vec, PetscBool *, void *);
  PetscErrorCode (*postcheck)(SNESLineSearch, Vec, Vec, Vec, PetscBool *, PetscBool *, void *);

  PetscFunctionBegin;
  if (!snes->linesearch) PetscFunctionReturn(0);
  PetscCall(SNESGetLineSearch(snes, &linesearch));
  PetscCall(SNESGetLineSearch(smooth, &slinesearch));
  PetscCall(SNESLineSearchGetPreCheck(linesearch, &precheck, &lsprectx));
  PetscCall(SNESLineSearchGetPostCheck(linesearch, &postcheck, &lspostctx));
  PetscCall(SNESLineSearchSetPreCheck(slinesearch, precheck, lsprectx));
  PetscCall(SNESLineSearchSetPostCheck(slinesearch, postcheck, lspostctx));
  PetscCall(PetscObjectCopyFortranFunctionPointers((PetscObject)linesearch, (PetscObject)slinesearch));
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESFASCycleSetUpSmoother_Private(SNES snes, SNES smooth)
{
  SNES_FAS *fas = (SNES_FAS *)snes->data;

  PetscFunctionBegin;
  PetscCall(PetscObjectCopyFortranFunctionPointers((PetscObject)snes, (PetscObject)smooth));
  PetscCall(SNESSetFromOptions(smooth));
  PetscCall(SNESFASSetUpLineSearch_Private(snes, smooth));

  PetscCall(PetscObjectReference((PetscObject)snes->vec_sol));
  PetscCall(PetscObjectReference((PetscObject)snes->vec_sol_update));
  PetscCall(PetscObjectReference((PetscObject)snes->vec_func));
  smooth->vec_sol        = snes->vec_sol;
  smooth->vec_sol_update = snes->vec_sol_update;
  smooth->vec_func       = snes->vec_func;

  if (fas->eventsmoothsetup) PetscCall(PetscLogEventBegin(fas->eventsmoothsetup, smooth, 0, 0, 0));
  PetscCall(SNESSetUp(smooth));
  if (fas->eventsmoothsetup) PetscCall(PetscLogEventEnd(fas->eventsmoothsetup, smooth, 0, 0, 0));
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESSetUp_FAS(SNES snes)
{
  SNES_FAS *fas = (SNES_FAS *)snes->data;
  PetscInt  dm_levels;
  SNES      next;
  PetscBool isFine, hasCreateRestriction, hasCreateInjection;

  PetscFunctionBegin;
  PetscCall(SNESFASCycleIsFine(snes, &isFine));
  if (fas->usedmfornumberoflevels && isFine) {
    PetscCall(DMGetRefineLevel(snes->dm, &dm_levels));
    dm_levels++;
    if (dm_levels > fas->levels) {
      /* reset the number of levels */
      PetscCall(SNESFASSetLevels(snes, dm_levels, NULL));
      PetscCall(SNESSetFromOptions(snes));
    }
  }
  PetscCall(SNESFASCycleGetCorrection(snes, &next));
  if (!isFine) snes->gridsequence = 0; /* no grid sequencing inside the multigrid hierarchy! */

  PetscCall(SNESSetWorkVecs(snes, 2)); /* work vectors used for intergrid transfers */

  /* set up the smoothers if they haven't already been set up */
  if (!fas->smoothd) PetscCall(SNESFASCycleCreateSmoother_Private(snes, &fas->smoothd));

  if (snes->dm) {
    /* set the smoother DMs properly */
    if (fas->smoothu) PetscCall(SNESSetDM(fas->smoothu, snes->dm));
    PetscCall(SNESSetDM(fas->smoothd, snes->dm));
    /* construct EVERYTHING from the DM -- including the progressive set of smoothers */
    if (next) {
      /* for now -- assume the DM and the evaluation functions have been set externally */
      if (!next->dm) {
        PetscCall(DMCoarsen(snes->dm, PetscObjectComm((PetscObject)next), &next->dm));
        PetscCall(SNESSetDM(next, next->dm));
      }
      /* set the interpolation and restriction from the DM */
      if (!fas->interpolate) {
        PetscCall(DMCreateInterpolation(next->dm, snes->dm, &fas->interpolate, &fas->rscale));
        if (!fas->restrct) {
          PetscCall(DMHasCreateRestriction(next->dm, &hasCreateRestriction));
          /* DM can create restrictions, use that */
          if (hasCreateRestriction) {
            PetscCall(DMCreateRestriction(next->dm, snes->dm, &fas->restrct));
          } else {
            PetscCall(PetscObjectReference((PetscObject)fas->interpolate));
            fas->restrct = fas->interpolate;
          }
        }
      }
      /* set the injection from the DM */
      if (!fas->inject) {
        PetscCall(DMHasCreateInjection(next->dm, &hasCreateInjection));
        if (hasCreateInjection) PetscCall(DMCreateInjection(next->dm, snes->dm, &fas->inject));
      }
    }
  }

  /*pass the smoother, function, and jacobian up to the next level if it's not user set already */
  if (fas->galerkin) {
    if (next) PetscCall(SNESSetFunction(next, NULL, SNESFASGalerkinFunctionDefault, next));
    if (fas->smoothd && fas->level != fas->levels - 1) PetscCall(SNESSetFunction(fas->smoothd, NULL, SNESFASGalerkinFunctionDefault, snes));
    if (fas->smoothu && fas->level != fas->levels - 1) PetscCall(SNESSetFunction(fas->smoothu, NULL, SNESFASGalerkinFunctionDefault, snes));
  }

  /* sets the down (pre) smoother's default norm and sets it from options */
  if (fas->smoothd) {
    if (fas->level == 0 && fas->levels != 1) {
      PetscCall(SNESSetNormSchedule(fas->smoothd, SNES_NORM_NONE));
    } else {
      PetscCall(SNESSetNormSchedule(fas->smoothd, SNES_NORM_FINAL_ONLY));
    }
    PetscCall(SNESFASCycleSetUpSmoother_Private(snes, fas->smoothd));
  }

  /* sets the up (post) smoother's default norm and sets it from options */
  if (fas->smoothu) {
    if (fas->level != fas->levels - 1) {
      PetscCall(SNESSetNormSchedule(fas->smoothu, SNES_NORM_NONE));
    } else {
      PetscCall(SNESSetNormSchedule(fas->smoothu, SNES_NORM_FINAL_ONLY));
    }
    PetscCall(SNESFASCycleSetUpSmoother_Private(snes, fas->smoothu));
  }

  if (next) {
    /* gotta set up the solution vector for this to work */
    if (!next->vec_sol) PetscCall(SNESFASCreateCoarseVec(snes, &next->vec_sol));
    if (!next->vec_rhs) PetscCall(SNESFASCreateCoarseVec(snes, &next->vec_rhs));
    PetscCall(PetscObjectCopyFortranFunctionPointers((PetscObject)snes, (PetscObject)next));
    PetscCall(SNESFASSetUpLineSearch_Private(snes, next));
    PetscCall(SNESSetUp(next));
  }

  /* setup FAS work vectors */
  if (fas->galerkin) {
    PetscCall(VecDuplicate(snes->vec_sol, &fas->Xg));
    PetscCall(VecDuplicate(snes->vec_sol, &fas->Fg));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESSetFromOptions_FAS(SNES snes, PetscOptionItems *PetscOptionsObject)
{
  SNES_FAS      *fas    = (SNES_FAS *)snes->data;
  PetscInt       levels = 1;
  PetscBool      flg = PETSC_FALSE, upflg = PETSC_FALSE, downflg = PETSC_FALSE, monflg = PETSC_FALSE, galerkinflg = PETSC_FALSE, continuationflg = PETSC_FALSE;
  SNESFASType    fastype;
  const char    *optionsprefix;
  SNESLineSearch linesearch;
  PetscInt       m, n_up, n_down;
  SNES           next;
  PetscBool      isFine;

  PetscFunctionBegin;
  PetscCall(SNESFASCycleIsFine(snes, &isFine));
  PetscOptionsHeadBegin(PetscOptionsObject, "SNESFAS Options-----------------------------------");

  /* number of levels -- only process most options on the finest level */
  if (isFine) {
    PetscCall(PetscOptionsInt("-snes_fas_levels", "Number of Levels", "SNESFASSetLevels", levels, &levels, &flg));
    if (!flg && snes->dm) {
      PetscCall(DMGetRefineLevel(snes->dm, &levels));
      levels++;
      fas->usedmfornumberoflevels = PETSC_TRUE;
    }
    PetscCall(SNESFASSetLevels(snes, levels, NULL));
    fastype = fas->fastype;
    PetscCall(PetscOptionsEnum("-snes_fas_type", "FAS correction type", "SNESFASSetType", SNESFASTypes, (PetscEnum)fastype, (PetscEnum *)&fastype, &flg));
    if (flg) PetscCall(SNESFASSetType(snes, fastype));

    PetscCall(SNESGetOptionsPrefix(snes, &optionsprefix));
    PetscCall(PetscOptionsInt("-snes_fas_cycles", "Number of cycles", "SNESFASSetCycles", fas->n_cycles, &m, &flg));
    if (flg) PetscCall(SNESFASSetCycles(snes, m));
    PetscCall(PetscOptionsBool("-snes_fas_continuation", "Corrected grid-sequence continuation", "SNESFASSetContinuation", fas->continuation, &continuationflg, &flg));
    if (flg) PetscCall(SNESFASSetContinuation(snes, continuationflg));

    PetscCall(PetscOptionsBool("-snes_fas_galerkin", "Form coarse problems with Galerkin", "SNESFASSetGalerkin", fas->galerkin, &galerkinflg, &flg));
    if (flg) PetscCall(SNESFASSetGalerkin(snes, galerkinflg));

    if (fas->fastype == SNES_FAS_FULL) {
      PetscCall(PetscOptionsBool("-snes_fas_full_downsweep", "Smooth on the initial down sweep for full FAS cycles", "SNESFASFullSetDownSweep", fas->full_downsweep, &fas->full_downsweep, &flg));
      if (flg) PetscCall(SNESFASFullSetDownSweep(snes, fas->full_downsweep));
      PetscCall(PetscOptionsBool("-snes_fas_full_total", "Use total restriction and interpolaton on the indial down and up sweeps for the full FAS cycle", "SNESFASFullSetUseTotal", fas->full_total, &fas->full_total, &flg));
      if (flg) PetscCall(SNESFASFullSetTotal(snes, fas->full_total));
    }

    PetscCall(PetscOptionsInt("-snes_fas_smoothup", "Number of post-smoothing steps", "SNESFASSetNumberSmoothUp", fas->max_up_it, &n_up, &upflg));

    PetscCall(PetscOptionsInt("-snes_fas_smoothdown", "Number of pre-smoothing steps", "SNESFASSetNumberSmoothDown", fas->max_down_it, &n_down, &downflg));

    {
      PetscViewer       viewer;
      PetscViewerFormat format;
      PetscCall(PetscOptionsGetViewer(PetscObjectComm((PetscObject)snes), ((PetscObject)snes)->options, ((PetscObject)snes)->prefix, "-snes_fas_monitor", &viewer, &format, &monflg));
      if (monflg) {
        PetscViewerAndFormat *vf;
        PetscCall(PetscViewerAndFormatCreate(viewer, format, &vf));
        PetscCall(PetscObjectDereference((PetscObject)viewer));
        PetscCall(SNESFASSetMonitor(snes, vf, PETSC_TRUE));
      }
    }
    flg    = PETSC_FALSE;
    monflg = PETSC_TRUE;
    PetscCall(PetscOptionsBool("-snes_fas_log", "Log times for each FAS level", "SNESFASSetLog", monflg, &monflg, &flg));
    if (flg) PetscCall(SNESFASSetLog(snes, monflg));
  }

  PetscOptionsHeadEnd();

  /* setup from the determined types if there is no pointwise procedure or smoother defined */
  if (upflg) PetscCall(SNESFASSetNumberSmoothUp(snes, n_up));
  if (downflg) PetscCall(SNESFASSetNumberSmoothDown(snes, n_down));

  /* set up the default line search for coarse grid corrections */
  if (fas->fastype == SNES_FAS_ADDITIVE) {
    if (!snes->linesearch) {
      PetscCall(SNESGetLineSearch(snes, &linesearch));
      PetscCall(SNESLineSearchSetType(linesearch, SNESLINESEARCHL2));
    }
  }

  /* recursive option setting for the smoothers */
  PetscCall(SNESFASCycleGetCorrection(snes, &next));
  if (next) PetscCall(SNESSetFromOptions(next));
  PetscFunctionReturn(0);
}

#include <petscdraw.h>
static PetscErrorCode SNESView_FAS(SNES snes, PetscViewer viewer)
{
  SNES_FAS *fas = (SNES_FAS *)snes->data;
  PetscBool isFine, iascii, isdraw;
  PetscInt  i;
  SNES      smoothu, smoothd, levelsnes;

  PetscFunctionBegin;
  PetscCall(SNESFASCycleIsFine(snes, &isFine));
  if (isFine) {
    PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
    PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERDRAW, &isdraw));
    if (iascii) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "  type is %s, levels=%" PetscInt_FMT ", cycles=%" PetscInt_FMT "\n", SNESFASTypes[fas->fastype], fas->levels, fas->n_cycles));
      if (fas->galerkin) {
        PetscCall(PetscViewerASCIIPrintf(viewer, "  Using Galerkin computed coarse grid function evaluation\n"));
      } else {
        PetscCall(PetscViewerASCIIPrintf(viewer, "  Not using Galerkin computed coarse grid function evaluation\n"));
      }
      for (i = 0; i < fas->levels; i++) {
        PetscCall(SNESFASGetCycleSNES(snes, i, &levelsnes));
        PetscCall(SNESFASCycleGetSmootherUp(levelsnes, &smoothu));
        PetscCall(SNESFASCycleGetSmootherDown(levelsnes, &smoothd));
        if (!i) {
          PetscCall(PetscViewerASCIIPrintf(viewer, "  Coarse grid solver -- level %" PetscInt_FMT " -------------------------------\n", i));
        } else {
          PetscCall(PetscViewerASCIIPrintf(viewer, "  Down solver (pre-smoother) on level %" PetscInt_FMT " -------------------------------\n", i));
        }
        PetscCall(PetscViewerASCIIPushTab(viewer));
        if (smoothd) {
          PetscCall(SNESView(smoothd, viewer));
        } else {
          PetscCall(PetscViewerASCIIPrintf(viewer, "Not yet available\n"));
        }
        PetscCall(PetscViewerASCIIPopTab(viewer));
        if (i && (smoothd == smoothu)) {
          PetscCall(PetscViewerASCIIPrintf(viewer, "  Up solver (post-smoother) same as down solver (pre-smoother)\n"));
        } else if (i) {
          PetscCall(PetscViewerASCIIPrintf(viewer, "  Up solver (post-smoother) on level %" PetscInt_FMT " -------------------------------\n", i));
          PetscCall(PetscViewerASCIIPushTab(viewer));
          if (smoothu) {
            PetscCall(SNESView(smoothu, viewer));
          } else {
            PetscCall(PetscViewerASCIIPrintf(viewer, "Not yet available\n"));
          }
          PetscCall(PetscViewerASCIIPopTab(viewer));
        }
      }
    } else if (isdraw) {
      PetscDraw draw;
      PetscReal x, w, y, bottom, th, wth;
      SNES_FAS *curfas = fas;
      PetscCall(PetscViewerDrawGetDraw(viewer, 0, &draw));
      PetscCall(PetscDrawGetCurrentPoint(draw, &x, &y));
      PetscCall(PetscDrawStringGetSize(draw, &wth, &th));
      bottom = y - th;
      while (curfas) {
        if (!curfas->smoothu) {
          PetscCall(PetscDrawPushCurrentPoint(draw, x, bottom));
          if (curfas->smoothd) PetscCall(SNESView(curfas->smoothd, viewer));
          PetscCall(PetscDrawPopCurrentPoint(draw));
        } else {
          w = 0.5 * PetscMin(1.0 - x, x);
          PetscCall(PetscDrawPushCurrentPoint(draw, x - w, bottom));
          if (curfas->smoothd) PetscCall(SNESView(curfas->smoothd, viewer));
          PetscCall(PetscDrawPopCurrentPoint(draw));
          PetscCall(PetscDrawPushCurrentPoint(draw, x + w, bottom));
          if (curfas->smoothu) PetscCall(SNESView(curfas->smoothu, viewer));
          PetscCall(PetscDrawPopCurrentPoint(draw));
        }
        /* this is totally bogus but we have no way of knowing how low the previous one was draw to */
        bottom -= 5 * th;
        if (curfas->next) curfas = (SNES_FAS *)curfas->next->data;
        else curfas = NULL;
      }
    }
  }
  PetscFunctionReturn(0);
}

/*
Defines the action of the downsmoother
 */
static PetscErrorCode SNESFASDownSmooth_Private(SNES snes, Vec B, Vec X, Vec F, PetscReal *fnorm)
{
  SNESConvergedReason reason;
  Vec                 FPC;
  SNES                smoothd;
  PetscBool           flg;
  SNES_FAS           *fas = (SNES_FAS *)snes->data;

  PetscFunctionBegin;
  PetscCall(SNESFASCycleGetSmootherDown(snes, &smoothd));
  PetscCall(SNESSetInitialFunction(smoothd, F));
  if (fas->eventsmoothsolve) PetscCall(PetscLogEventBegin(fas->eventsmoothsolve, smoothd, B, X, 0));
  PetscCall(SNESSolve(smoothd, B, X));
  if (fas->eventsmoothsolve) PetscCall(PetscLogEventEnd(fas->eventsmoothsolve, smoothd, B, X, 0));
  /* check convergence reason for the smoother */
  PetscCall(SNESGetConvergedReason(smoothd, &reason));
  if (reason < 0 && !(reason == SNES_DIVERGED_MAX_IT || reason == SNES_DIVERGED_LOCAL_MIN || reason == SNES_DIVERGED_LINE_SEARCH)) {
    snes->reason = SNES_DIVERGED_INNER;
    PetscFunctionReturn(0);
  }

  PetscCall(SNESGetFunction(smoothd, &FPC, NULL, NULL));
  PetscCall(SNESGetAlwaysComputesFinalResidual(smoothd, &flg));
  if (!flg) PetscCall(SNESComputeFunction(smoothd, X, FPC));
  PetscCall(VecCopy(FPC, F));
  if (fnorm) PetscCall(VecNorm(F, NORM_2, fnorm));
  PetscFunctionReturn(0);
}

/*
Defines the action of the upsmoother
 */
static PetscErrorCode SNESFASUpSmooth_Private(SNES snes, Vec B, Vec X, Vec F, PetscReal *fnorm)
{
  SNESConvergedReason reason;
  Vec                 FPC;
  SNES                smoothu;
  PetscBool           flg;
  SNES_FAS           *fas = (SNES_FAS *)snes->data;

  PetscFunctionBegin;
  PetscCall(SNESFASCycleGetSmootherUp(snes, &smoothu));
  if (fas->eventsmoothsolve) PetscCall(PetscLogEventBegin(fas->eventsmoothsolve, smoothu, 0, 0, 0));
  PetscCall(SNESSolve(smoothu, B, X));
  if (fas->eventsmoothsolve) PetscCall(PetscLogEventEnd(fas->eventsmoothsolve, smoothu, 0, 0, 0));
  /* check convergence reason for the smoother */
  PetscCall(SNESGetConvergedReason(smoothu, &reason));
  if (reason < 0 && !(reason == SNES_DIVERGED_MAX_IT || reason == SNES_DIVERGED_LOCAL_MIN || reason == SNES_DIVERGED_LINE_SEARCH)) {
    snes->reason = SNES_DIVERGED_INNER;
    PetscFunctionReturn(0);
  }
  PetscCall(SNESGetFunction(smoothu, &FPC, NULL, NULL));
  PetscCall(SNESGetAlwaysComputesFinalResidual(smoothu, &flg));
  if (!flg) PetscCall(SNESComputeFunction(smoothu, X, FPC));
  PetscCall(VecCopy(FPC, F));
  if (fnorm) PetscCall(VecNorm(F, NORM_2, fnorm));
  PetscFunctionReturn(0);
}

/*@
   SNESFASCreateCoarseVec - create `Vec` corresponding to a state vector on one level coarser than current level

   Collective

   Input Parameter:
.  snes - `SNESFAS` object

   Output Parameter:
.  Xcoarse - vector on level one coarser than snes

   Level: developer

.seealso: `SNESFASSetRestriction()`, `SNESFASRestrict()`
@*/
PetscErrorCode SNESFASCreateCoarseVec(SNES snes, Vec *Xcoarse)
{
  SNES_FAS *fas;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(snes, SNES_CLASSID, 1, SNESFAS);
  PetscValidPointer(Xcoarse, 2);
  fas = (SNES_FAS *)snes->data;
  if (fas->rscale) {
    PetscCall(VecDuplicate(fas->rscale, Xcoarse));
  } else if (fas->inject) {
    PetscCall(MatCreateVecs(fas->inject, Xcoarse, NULL));
  } else SETERRQ(PetscObjectComm((PetscObject)snes), PETSC_ERR_ARG_WRONGSTATE, "Must set restriction or injection");
  PetscFunctionReturn(0);
}

/*@
   SNESFASRestrict - restrict a `Vec` to the next coarser level

   Collective

   Input Parameters:
+  fine - `SNES` from which to restrict
-  Xfine - vector to restrict

   Output Parameter:
.  Xcoarse - result of restriction

   Level: developer

.seealso: `SNES`, `SNESFAS`, `SNESFASSetRestriction()`, `SNESFASSetInjection()`
@*/
PetscErrorCode SNESFASRestrict(SNES fine, Vec Xfine, Vec Xcoarse)
{
  SNES_FAS *fas;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(fine, SNES_CLASSID, 1, SNESFAS);
  PetscValidHeaderSpecific(Xfine, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(Xcoarse, VEC_CLASSID, 3);
  fas = (SNES_FAS *)fine->data;
  if (fas->inject) {
    PetscCall(MatRestrict(fas->inject, Xfine, Xcoarse));
  } else {
    PetscCall(MatRestrict(fas->restrct, Xfine, Xcoarse));
    PetscCall(VecPointwiseMult(Xcoarse, fas->rscale, Xcoarse));
  }
  PetscFunctionReturn(0);
}

/*

Performs a variant of FAS using the interpolated total coarse solution

fine problem:   F(x) = b
coarse problem: F^c(x^c) = Rb, Initial guess Rx
interpolated solution: x^f = I x^c (total solution interpolation

 */
static PetscErrorCode SNESFASInterpolatedCoarseSolution(SNES snes, Vec X, Vec X_new)
{
  Vec                 X_c, B_c;
  SNESConvergedReason reason;
  SNES                next;
  Mat                 restrct, interpolate;
  SNES_FAS           *fasc;

  PetscFunctionBegin;
  PetscCall(SNESFASCycleGetCorrection(snes, &next));
  if (next) {
    fasc = (SNES_FAS *)next->data;

    PetscCall(SNESFASCycleGetRestriction(snes, &restrct));
    PetscCall(SNESFASCycleGetInterpolation(snes, &interpolate));

    X_c = next->vec_sol;

    if (fasc->eventinterprestrict) PetscCall(PetscLogEventBegin(fasc->eventinterprestrict, snes, 0, 0, 0));
    /* restrict the total solution: Rb */
    PetscCall(SNESFASRestrict(snes, X, X_c));
    B_c = next->vec_rhs;
    if (snes->vec_rhs) {
      /* restrict the total rhs defect: Rb */
      PetscCall(MatRestrict(restrct, snes->vec_rhs, B_c));
    } else {
      PetscCall(VecSet(B_c, 0.));
    }
    if (fasc->eventinterprestrict) PetscCall(PetscLogEventEnd(fasc->eventinterprestrict, snes, 0, 0, 0));

    PetscCall(SNESSolve(next, B_c, X_c));
    PetscCall(SNESGetConvergedReason(next, &reason));
    if (reason < 0 && reason != SNES_DIVERGED_MAX_IT) {
      snes->reason = SNES_DIVERGED_INNER;
      PetscFunctionReturn(0);
    }
    /* x^f <- Ix^c*/
    DM dmc, dmf;

    PetscCall(SNESGetDM(next, &dmc));
    PetscCall(SNESGetDM(snes, &dmf));
    if (fasc->eventinterprestrict) PetscCall(PetscLogEventBegin(fasc->eventinterprestrict, snes, 0, 0, 0));
    PetscCall(DMInterpolateSolution(dmc, dmf, interpolate, X_c, X_new));
    if (fasc->eventinterprestrict) PetscCall(PetscLogEventEnd(fasc->eventinterprestrict, snes, 0, 0, 0));
    PetscCall(PetscObjectSetName((PetscObject)X_c, "Coarse solution"));
    PetscCall(VecViewFromOptions(X_c, NULL, "-fas_coarse_solution_view"));
    PetscCall(PetscObjectSetName((PetscObject)X_new, "Updated Fine solution"));
    PetscCall(VecViewFromOptions(X_new, NULL, "-fas_levels_1_solution_view"));
  }
  PetscFunctionReturn(0);
}

/*

Performs the FAS coarse correction as:

fine problem:   F(x) = b
coarse problem: F^c(x^c) = b^c

b^c = F^c(Rx) - R(F(x) - b)

 */
PetscErrorCode SNESFASCoarseCorrection(SNES snes, Vec X, Vec F, Vec X_new)
{
  Vec                 X_c, Xo_c, F_c, B_c;
  SNESConvergedReason reason;
  SNES                next;
  Mat                 restrct, interpolate;
  SNES_FAS           *fasc;

  PetscFunctionBegin;
  PetscCall(SNESFASCycleGetCorrection(snes, &next));
  if (next) {
    fasc = (SNES_FAS *)next->data;

    PetscCall(SNESFASCycleGetRestriction(snes, &restrct));
    PetscCall(SNESFASCycleGetInterpolation(snes, &interpolate));

    X_c  = next->vec_sol;
    Xo_c = next->work[0];
    F_c  = next->vec_func;
    B_c  = next->vec_rhs;

    if (fasc->eventinterprestrict) PetscCall(PetscLogEventBegin(fasc->eventinterprestrict, snes, 0, 0, 0));
    PetscCall(SNESFASRestrict(snes, X, Xo_c));
    /* restrict the defect: R(F(x) - b) */
    PetscCall(MatRestrict(restrct, F, B_c));
    if (fasc->eventinterprestrict) PetscCall(PetscLogEventEnd(fasc->eventinterprestrict, snes, 0, 0, 0));

    if (fasc->eventresidual) PetscCall(PetscLogEventBegin(fasc->eventresidual, next, 0, 0, 0));
    /* F_c = F^c(Rx) - R(F(x) - b) since the second term was sitting in next->vec_rhs */
    PetscCall(SNESComputeFunction(next, Xo_c, F_c));
    if (fasc->eventresidual) PetscCall(PetscLogEventEnd(fasc->eventresidual, next, 0, 0, 0));

    /* solve the coarse problem corresponding to F^c(x^c) = b^c = F^c(Rx) - R(F(x) - b) */
    PetscCall(VecCopy(B_c, X_c));
    PetscCall(VecCopy(F_c, B_c));
    PetscCall(VecCopy(X_c, F_c));
    /* set initial guess of the coarse problem to the projected fine solution */
    PetscCall(VecCopy(Xo_c, X_c));

    /* recurse to the next level */
    PetscCall(SNESSetInitialFunction(next, F_c));
    PetscCall(SNESSolve(next, B_c, X_c));
    PetscCall(SNESGetConvergedReason(next, &reason));
    if (reason < 0 && reason != SNES_DIVERGED_MAX_IT) {
      snes->reason = SNES_DIVERGED_INNER;
      PetscFunctionReturn(0);
    }
    /* correct as x <- x + I(x^c - Rx)*/
    PetscCall(VecAXPY(X_c, -1.0, Xo_c));

    if (fasc->eventinterprestrict) PetscCall(PetscLogEventBegin(fasc->eventinterprestrict, snes, 0, 0, 0));
    PetscCall(MatInterpolateAdd(interpolate, X_c, X, X_new));
    if (fasc->eventinterprestrict) PetscCall(PetscLogEventEnd(fasc->eventinterprestrict, snes, 0, 0, 0));
    PetscCall(PetscObjectSetName((PetscObject)X_c, "Coarse correction"));
    PetscCall(VecViewFromOptions(X_c, NULL, "-fas_coarse_solution_view"));
    PetscCall(PetscObjectSetName((PetscObject)X_new, "Updated Fine solution"));
    PetscCall(VecViewFromOptions(X_new, NULL, "-fas_levels_1_solution_view"));
  }
  PetscFunctionReturn(0);
}

/*

The additive cycle looks like:

xhat = x
xhat = dS(x, b)
x = coarsecorrection(xhat, b_d)
x = x + nu*(xhat - x);
(optional) x = uS(x, b)

With the coarse RHS (defect correction) as below.

 */
static PetscErrorCode SNESFASCycle_Additive(SNES snes, Vec X)
{
  Vec                  F, B, Xhat;
  Vec                  X_c, Xo_c, F_c, B_c;
  SNESConvergedReason  reason;
  PetscReal            xnorm, fnorm, ynorm;
  SNESLineSearchReason lsresult;
  SNES                 next;
  Mat                  restrct, interpolate;
  SNES_FAS            *fas = (SNES_FAS *)snes->data, *fasc;

  PetscFunctionBegin;
  PetscCall(SNESFASCycleGetCorrection(snes, &next));
  F    = snes->vec_func;
  B    = snes->vec_rhs;
  Xhat = snes->work[1];
  PetscCall(VecCopy(X, Xhat));
  /* recurse first */
  if (next) {
    fasc = (SNES_FAS *)next->data;
    PetscCall(SNESFASCycleGetRestriction(snes, &restrct));
    PetscCall(SNESFASCycleGetInterpolation(snes, &interpolate));
    if (fas->eventresidual) PetscCall(PetscLogEventBegin(fas->eventresidual, snes, 0, 0, 0));
    PetscCall(SNESComputeFunction(snes, Xhat, F));
    if (fas->eventresidual) PetscCall(PetscLogEventEnd(fas->eventresidual, snes, 0, 0, 0));
    PetscCall(VecNorm(F, NORM_2, &fnorm));
    X_c  = next->vec_sol;
    Xo_c = next->work[0];
    F_c  = next->vec_func;
    B_c  = next->vec_rhs;

    PetscCall(SNESFASRestrict(snes, Xhat, Xo_c));
    /* restrict the defect */
    PetscCall(MatRestrict(restrct, F, B_c));

    /* solve the coarse problem corresponding to F^c(x^c) = b^c = Rb + F^c(Rx) - RF(x) */
    if (fasc->eventresidual) PetscCall(PetscLogEventBegin(fasc->eventresidual, next, 0, 0, 0));
    PetscCall(SNESComputeFunction(next, Xo_c, F_c));
    if (fasc->eventresidual) PetscCall(PetscLogEventEnd(fasc->eventresidual, next, 0, 0, 0));
    PetscCall(VecCopy(B_c, X_c));
    PetscCall(VecCopy(F_c, B_c));
    PetscCall(VecCopy(X_c, F_c));
    /* set initial guess of the coarse problem to the projected fine solution */
    PetscCall(VecCopy(Xo_c, X_c));

    /* recurse */
    PetscCall(SNESSetInitialFunction(next, F_c));
    PetscCall(SNESSolve(next, B_c, X_c));

    /* smooth on this level */
    PetscCall(SNESFASDownSmooth_Private(snes, B, X, F, &fnorm));

    PetscCall(SNESGetConvergedReason(next, &reason));
    if (reason < 0 && reason != SNES_DIVERGED_MAX_IT) {
      snes->reason = SNES_DIVERGED_INNER;
      PetscFunctionReturn(0);
    }

    /* correct as x <- x + I(x^c - Rx)*/
    PetscCall(VecAYPX(X_c, -1.0, Xo_c));
    PetscCall(MatInterpolate(interpolate, X_c, Xhat));

    /* additive correction of the coarse direction*/
    PetscCall(SNESLineSearchApply(snes->linesearch, X, F, &fnorm, Xhat));
    PetscCall(SNESLineSearchGetReason(snes->linesearch, &lsresult));
    PetscCall(SNESLineSearchGetNorms(snes->linesearch, &xnorm, &snes->norm, &ynorm));
    if (lsresult) {
      if (++snes->numFailures >= snes->maxFailures) {
        snes->reason = SNES_DIVERGED_LINE_SEARCH;
        PetscFunctionReturn(0);
      }
    }
  } else {
    PetscCall(SNESFASDownSmooth_Private(snes, B, X, F, &snes->norm));
  }
  PetscFunctionReturn(0);
}

/*

Defines the FAS cycle as:

fine problem: F(x) = b
coarse problem: F^c(x) = b^c

b^c = F^c(Rx) - R(F(x) - b)

correction:

x = x + I(x^c - Rx)

 */
static PetscErrorCode SNESFASCycle_Multiplicative(SNES snes, Vec X)
{
  Vec  F, B;
  SNES next;

  PetscFunctionBegin;
  F = snes->vec_func;
  B = snes->vec_rhs;
  /* pre-smooth -- just update using the pre-smoother */
  PetscCall(SNESFASCycleGetCorrection(snes, &next));
  PetscCall(SNESFASDownSmooth_Private(snes, B, X, F, &snes->norm));
  if (next) {
    PetscCall(SNESFASCoarseCorrection(snes, X, F, X));
    PetscCall(SNESFASUpSmooth_Private(snes, B, X, F, &snes->norm));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESFASCycleSetupPhase_Full(SNES snes)
{
  SNES      next;
  SNES_FAS *fas = (SNES_FAS *)snes->data;
  PetscBool isFine;

  PetscFunctionBegin;
  /* pre-smooth -- just update using the pre-smoother */
  PetscCall(SNESFASCycleIsFine(snes, &isFine));
  PetscCall(SNESFASCycleGetCorrection(snes, &next));
  fas->full_stage = 0;
  if (next) PetscCall(SNESFASCycleSetupPhase_Full(next));
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESFASCycle_Full(SNES snes, Vec X)
{
  Vec       F, B;
  SNES_FAS *fas = (SNES_FAS *)snes->data;
  PetscBool isFine;
  SNES      next;

  PetscFunctionBegin;
  F = snes->vec_func;
  B = snes->vec_rhs;
  PetscCall(SNESFASCycleIsFine(snes, &isFine));
  PetscCall(SNESFASCycleGetCorrection(snes, &next));

  if (isFine) PetscCall(SNESFASCycleSetupPhase_Full(snes));

  if (fas->full_stage == 0) {
    /* downsweep */
    if (next) {
      if (fas->level != 1) next->max_its += 1;
      if (fas->full_downsweep) PetscCall(SNESFASDownSmooth_Private(snes, B, X, F, &snes->norm));
      fas->full_downsweep = PETSC_TRUE;
      if (fas->full_total) PetscCall(SNESFASInterpolatedCoarseSolution(snes, X, X));
      else PetscCall(SNESFASCoarseCorrection(snes, X, F, X));
      fas->full_total = PETSC_FALSE;
      PetscCall(SNESFASUpSmooth_Private(snes, B, X, F, &snes->norm));
      if (fas->level != 1) next->max_its -= 1;
    } else {
      /* The smoother on the coarse level is the coarse solver */
      PetscCall(SNESFASDownSmooth_Private(snes, B, X, F, &snes->norm));
    }
    fas->full_stage = 1;
  } else if (fas->full_stage == 1) {
    if (snes->iter == 0) PetscCall(SNESFASDownSmooth_Private(snes, B, X, F, &snes->norm));
    if (next) {
      PetscCall(SNESFASCoarseCorrection(snes, X, F, X));
      PetscCall(SNESFASUpSmooth_Private(snes, B, X, F, &snes->norm));
    }
  }
  /* final v-cycle */
  if (isFine) {
    if (next) {
      PetscCall(SNESFASCoarseCorrection(snes, X, F, X));
      PetscCall(SNESFASUpSmooth_Private(snes, B, X, F, &snes->norm));
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESFASCycle_Kaskade(SNES snes, Vec X)
{
  Vec  F, B;
  SNES next;

  PetscFunctionBegin;
  F = snes->vec_func;
  B = snes->vec_rhs;
  PetscCall(SNESFASCycleGetCorrection(snes, &next));
  if (next) {
    PetscCall(SNESFASCoarseCorrection(snes, X, F, X));
    PetscCall(SNESFASUpSmooth_Private(snes, B, X, F, &snes->norm));
  } else {
    PetscCall(SNESFASDownSmooth_Private(snes, B, X, F, &snes->norm));
  }
  PetscFunctionReturn(0);
}

PetscBool  SNEScite       = PETSC_FALSE;
const char SNESCitation[] = "@techreport{pbmkbsxt2012,\n"
                            "  title = {Composing Scalable Nonlinear Algebraic Solvers},\n"
                            "  author = {Peter Brune and Mathew Knepley and Barry Smith and Xuemin Tu},\n"
                            "  year = 2013,\n"
                            "  type = Preprint,\n"
                            "  number = {ANL/MCS-P2010-0112},\n"
                            "  institution = {Argonne National Laboratory}\n}\n";

static PetscErrorCode SNESSolve_FAS(SNES snes)
{
  PetscInt  i;
  Vec       X, F;
  PetscReal fnorm;
  SNES_FAS *fas = (SNES_FAS *)snes->data, *ffas;
  DM        dm;
  PetscBool isFine;

  PetscFunctionBegin;
  PetscCheck(!snes->xl && !snes->xu && !snes->ops->computevariablebounds, PetscObjectComm((PetscObject)snes), PETSC_ERR_ARG_WRONGSTATE, "SNES solver %s does not support bounds", ((PetscObject)snes)->type_name);

  PetscCall(PetscCitationsRegister(SNESCitation, &SNEScite));
  snes->reason = SNES_CONVERGED_ITERATING;
  X            = snes->vec_sol;
  F            = snes->vec_func;

  PetscCall(SNESFASCycleIsFine(snes, &isFine));
  /* norm setup */
  PetscCall(PetscObjectSAWsTakeAccess((PetscObject)snes));
  snes->iter = 0;
  snes->norm = 0;
  PetscCall(PetscObjectSAWsGrantAccess((PetscObject)snes));
  if (!snes->vec_func_init_set) {
    if (fas->eventresidual) PetscCall(PetscLogEventBegin(fas->eventresidual, snes, 0, 0, 0));
    PetscCall(SNESComputeFunction(snes, X, F));
    if (fas->eventresidual) PetscCall(PetscLogEventEnd(fas->eventresidual, snes, 0, 0, 0));
  } else snes->vec_func_init_set = PETSC_FALSE;

  PetscCall(VecNorm(F, NORM_2, &fnorm)); /* fnorm <- ||F||  */
  SNESCheckFunctionNorm(snes, fnorm);
  PetscCall(PetscObjectSAWsTakeAccess((PetscObject)snes));
  snes->norm = fnorm;
  PetscCall(PetscObjectSAWsGrantAccess((PetscObject)snes));
  PetscCall(SNESLogConvergenceHistory(snes, fnorm, 0));
  PetscCall(SNESMonitor(snes, snes->iter, fnorm));

  /* test convergence */
  PetscUseTypeMethod(snes, converged, 0, 0.0, 0.0, fnorm, &snes->reason, snes->cnvP);
  if (snes->reason) PetscFunctionReturn(0);

  if (isFine) {
    /* propagate scale-dependent data up the hierarchy */
    PetscCall(SNESGetDM(snes, &dm));
    for (ffas = fas; ffas->next; ffas = (SNES_FAS *)ffas->next->data) {
      DM dmcoarse;
      PetscCall(SNESGetDM(ffas->next, &dmcoarse));
      PetscCall(DMRestrict(dm, ffas->restrct, ffas->rscale, ffas->inject, dmcoarse));
      dm = dmcoarse;
    }
  }

  for (i = 0; i < snes->max_its; i++) {
    /* Call general purpose update function */
    PetscTryTypeMethod(snes, update, snes->iter);

    if (fas->fastype == SNES_FAS_MULTIPLICATIVE) {
      PetscCall(SNESFASCycle_Multiplicative(snes, X));
    } else if (fas->fastype == SNES_FAS_ADDITIVE) {
      PetscCall(SNESFASCycle_Additive(snes, X));
    } else if (fas->fastype == SNES_FAS_FULL) {
      PetscCall(SNESFASCycle_Full(snes, X));
    } else if (fas->fastype == SNES_FAS_KASKADE) {
      PetscCall(SNESFASCycle_Kaskade(snes, X));
    } else SETERRQ(PetscObjectComm((PetscObject)snes), PETSC_ERR_ARG_WRONGSTATE, "Unsupported FAS type");

    /* check for FAS cycle divergence */
    if (snes->reason != SNES_CONVERGED_ITERATING) PetscFunctionReturn(0);

    /* Monitor convergence */
    PetscCall(PetscObjectSAWsTakeAccess((PetscObject)snes));
    snes->iter = i + 1;
    PetscCall(PetscObjectSAWsGrantAccess((PetscObject)snes));
    PetscCall(SNESLogConvergenceHistory(snes, snes->norm, 0));
    PetscCall(SNESMonitor(snes, snes->iter, snes->norm));
    /* Test for convergence */
    if (isFine) {
      PetscUseTypeMethod(snes, converged, snes->iter, 0.0, 0.0, snes->norm, &snes->reason, snes->cnvP);
      if (snes->reason) break;
    }
  }
  if (i == snes->max_its) {
    PetscCall(PetscInfo(snes, "Maximum number of iterations has been reached: %" PetscInt_FMT "\n", i));
    if (!snes->reason) snes->reason = SNES_DIVERGED_MAX_IT;
  }
  PetscFunctionReturn(0);
}

/*MC

SNESFAS - Full Approximation Scheme nonlinear multigrid solver.

   The nonlinear problem is solved by correction using coarse versions
   of the nonlinear problem.  This problem is perturbed so that a projected
   solution of the fine problem elicits no correction from the coarse problem.

   Options Database Keys and Prefixes:
+   -snes_fas_levels -  The number of levels
.   -snes_fas_cycles<1> -  The number of cycles -- 1 for V, 2 for W
.   -snes_fas_type<additive,multiplicative,full,kaskade>  -  Additive or multiplicative cycle
.   -snes_fas_galerkin<`PETSC_FALSE`> -  Form coarse problems by projection back upon the fine problem
.   -snes_fas_smoothup<1> -  The number of iterations of the post-smoother
.   -snes_fas_smoothdown<1> -  The number of iterations of the pre-smoother
.   -snes_fas_monitor -  Monitor progress of all of the levels
.   -snes_fas_full_downsweep<`PETSC_FALSE`> - call the downsmooth on the initial downsweep of full FAS
.   -fas_levels_snes_ -  `SNES` options for all smoothers
.   -fas_levels_cycle_snes_ -  `SNES` options for all cycles
.   -fas_levels_i_snes_ -  `SNES` options for the smoothers on level i
.   -fas_levels_i_cycle_snes_ - `SNES` options for the cycle on level i
-   -fas_coarse_snes_ -  `SNES` options for the coarsest smoother

   Note:
   The organization of the FAS solver is slightly different from the organization of `PCMG`
   As each level has smoother `SNES` instances(down and potentially up) and a cycle `SNES` instance.
   The cycle `SNES` instance may be used for monitoring convergence on a particular level.

   Level: beginner

   References:
.  * - Peter R. Brune, Matthew G. Knepley, Barry F. Smith, and Xuemin Tu, "Composing Scalable Nonlinear Algebraic Solvers",
   SIAM Review, 57(4), 2015

.seealso: `PCMG`, `SNESCreate()`, `SNES`, `SNESSetType()`, `SNESType`, `SNESFASSetRestriction()`, `SNESFASSetInjection()`,
          `SNESFASFullGetTotal()`
M*/

PETSC_EXTERN PetscErrorCode SNESCreate_FAS(SNES snes)
{
  SNES_FAS *fas;

  PetscFunctionBegin;
  snes->ops->destroy        = SNESDestroy_FAS;
  snes->ops->setup          = SNESSetUp_FAS;
  snes->ops->setfromoptions = SNESSetFromOptions_FAS;
  snes->ops->view           = SNESView_FAS;
  snes->ops->solve          = SNESSolve_FAS;
  snes->ops->reset          = SNESReset_FAS;

  snes->usesksp = PETSC_FALSE;
  snes->usesnpc = PETSC_FALSE;

  if (!snes->tolerancesset) {
    snes->max_funcs = 30000;
    snes->max_its   = 10000;
  }

  snes->alwayscomputesfinalresidual = PETSC_TRUE;

  PetscCall(PetscNew(&fas));

  snes->data                  = (void *)fas;
  fas->level                  = 0;
  fas->levels                 = 1;
  fas->n_cycles               = 1;
  fas->max_up_it              = 1;
  fas->max_down_it            = 1;
  fas->smoothu                = NULL;
  fas->smoothd                = NULL;
  fas->next                   = NULL;
  fas->previous               = NULL;
  fas->fine                   = snes;
  fas->interpolate            = NULL;
  fas->restrct                = NULL;
  fas->inject                 = NULL;
  fas->usedmfornumberoflevels = PETSC_FALSE;
  fas->fastype                = SNES_FAS_MULTIPLICATIVE;
  fas->full_downsweep         = PETSC_FALSE;
  fas->full_total             = PETSC_FALSE;

  fas->eventsmoothsetup    = 0;
  fas->eventsmoothsolve    = 0;
  fas->eventresidual       = 0;
  fas->eventinterprestrict = 0;
  PetscFunctionReturn(0);
}
